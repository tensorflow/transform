# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An implementation of tf.Transform using Beam.

The beam implementation takes a user defined preprocessing function (see
examples for how to define a preprocessing function) and implements it as a
Beam PTransform.

The AnalyzeDataset takes the user's preprocessing function and converts into
a TensorFlow function that can be applied to each row of a dataset.  For
example if the user's preprocessing function describes normalizing a column by
subtracting its mean, the tensorflow function will contain the mean of the
column as a constant, and will subtract this value from each value of the
column.  We refer to the result of AnalyzeDataset as a "transform function".

Since AnalyzeDataset is implemented with beam, it accepts a PCollection that
represents the dataset (see below for the exact format) and returns a singleton
PCollection containing the transform function (as a serialized TF graph).

The TransformDataset PTransform takes a dataset and a transform function, and
returns the transformed dataset where the transform function is applied to each
row of the original dataset.

There is also an AnalyzeAndTransformDataset PTransform that applies
AnalyzeDataset and TransformDataset to the same input dataset, possibly with
optimizations.
"""

import copy
import datetime
import os

import apache_beam as beam

from apache_beam.runners.portability import fn_api_runner
from apache_beam.typehints import Any
from apache_beam.typehints import Dict
from apache_beam.typehints import Iterable
from apache_beam.typehints import List
from apache_beam.typehints import Set
from apache_beam.typehints import Tuple
from apache_beam.typehints import Union
from apache_beam.utils import shared

import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_transform import annotators
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import graph_context
from tensorflow_transform import graph_tools
from tensorflow_transform import impl_helper
from tensorflow_transform import nodes
from tensorflow_transform import schema_inference
from tensorflow_transform.beam import analysis_graph_builder
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform.beam import beam_nodes
from tensorflow_transform.beam import common as beam_common
from tensorflow_transform.beam import context
from tensorflow_transform.beam import deep_copy
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.saved import saved_transform_io_v2
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.telemetry import collection as telemetry
from tfx_bsl.tfxio import tensor_representation_util
from tfx_bsl.tfxio import tensor_to_arrow
from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio.tensor_adapter import TensorAdapter
from tfx_bsl.tfxio.tensor_adapter import TensorAdapterConfig
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple

from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2

# TODO(b/123325923): Fix the key type here to agree with the actual keys.
_DatasetElementType = Dict[Any,  # Any -> str?
                           Union[beam_common.PRIMITIVE_TYPE,
                                 # Arbitrarily-nested lists are allowed.
                                 List[Any], np.generic, np.ndarray]]
_TransformFnPathType = str

Context = context.Context

_CREATE_SAVED_MODEL_COUNTER_NAME = 'saved_models_created'

# For some runners, we rely on Beam to manage concurrency, i.e. we expect it to
# run one session per CPU--so we don't want to proliferate TF threads.
# Nonetheless we provide 4 threads per session for TF ops, 2 inter-
# and 2 intra-thread.  In many cases only 2 of these will be runnable
# at any given time.  This approach oversubscribes a bit to make sure
# the CPUs are really saturated.
_FIXED_PARALLELISM_TF_CONFIG = tf.compat.v1.ConfigProto(
    # TODO(b/36091595): use_per_session_threads is deprecated, but the
    # replacement session_inter_op_thread_pool is experimental; using
    # the former for now.
    use_per_session_threads=True,
    inter_op_parallelism_threads=2,
    intra_op_parallelism_threads=2)

_DEFAULT_TENSORFLOW_CONFIG_BY_BEAM_RUNNER_TYPE = {
    # TODO(katsiapis): Perhaps remove this entry once b/69922446 and b/30837990
    # are resolved.
    beam.runners.DataflowRunner: _FIXED_PARALLELISM_TF_CONFIG,

    beam.runners.DirectRunner: _FIXED_PARALLELISM_TF_CONFIG,
    fn_api_runner.FnApiRunner: _FIXED_PARALLELISM_TF_CONFIG,
}

# TODO(b/68154497): pylint: disable=no-value-for-parameter


# TODO(b/64956765): Remove this once either the keepalive issue (b/30837990), or
# the mentioned bug above is resolved.
# TODO(zoyahav): Make this a PTransform.
def _clear_shared_state_after_barrier(pipeline, input_barrier):
  """Clears any shared state from within a pipeline context.

  This will only be cleared once input_barrier becomes available.

  Args:
    pipeline: A `beam.Pipeline` object.
    input_barrier: A `PCollection` which the pipeline should wait for.

  Returns:
    An empty `PCollection`.
  """
  empty_pcoll = input_barrier | 'MakeCheapBarrier' >> beam.FlatMap(
      lambda x: None)
  return (pipeline
          | 'PrepareToClearSharedKeepAlives' >> beam.Create([None])
          | 'WaitAndClearSharedKeepAlives' >> beam.Map(
              lambda x, empty_side_input: shared.Shared().acquire(lambda: None),
              beam.pvalue.AsIter(empty_pcoll)))


# TODO(b/36223892): Verify that these type hints work and make needed fixes.
@beam.typehints.with_input_types(Union[List[_DatasetElementType],
                                       pa.RecordBatch], _TransformFnPathType)
@beam.typehints.with_output_types(
    Dict[str, Union[np.ndarray, tf.compat.v1.SparseTensorValue]])
class _RunMetaGraphDoFn(beam.DoFn):
  """Maps a PCollection of dicts to a PCollection of dicts via a TF graph.

  The TF graph may contain more inputs than the schema provided. In that case,
  a subset of the inputs will be fed, which may cause an error if the excluded
  inputs are required to produce the included outputs.
  """

  class _GraphStateCommon:
    """A container for a shared graph state."""

    def __init__(self, saved_model_dir, input_tensor_keys, output_tensor_keys,
                 callable_get_outputs):
      self.saved_model_dir = saved_model_dir
      self.inputs_tensor_keys = input_tensor_keys
      self.outputs_tensor_keys = output_tensor_keys
      self.callable_get_outputs = callable_get_outputs

  # Thread-safe.
  class _GraphStateCompatV1(_GraphStateCommon):
    """A container for a shared TF1 graph state."""

    def __init__(self, saved_model_dir, input_tensor_names, exclude_outputs,
                 tf_config):
      with tf.compat.v1.Graph().as_default() as graph:
        self._session = tf.compat.v1.Session(graph=graph, config=tf_config)
        with self._session.as_default():
          inputs, outputs = (
              saved_transform_io.partially_apply_saved_transform_internal(
                  saved_model_dir, {}))
        self._session.run(tf.compat.v1.global_variables_initializer())
        self._session.run(tf.compat.v1.tables_initializer())
        graph.finalize()

        if set(input_tensor_names).difference(inputs.keys()):
          raise ValueError(
              'Input tensor names contained tensors not in graph: %s' %
              input_tensor_names)
        if set(exclude_outputs).difference(outputs.keys()):
          raise ValueError('Excluded outputs contained keys not in graph: %s' %
                           exclude_outputs)
        non_excluded_output_keys = sorted(
            set(outputs.keys()).difference(exclude_outputs))
        fetches = [outputs[key] for key in non_excluded_output_keys]
        tensor_inputs = graph_tools.get_dependent_inputs(graph, inputs, fetches)
        inputs_tensor_keys = sorted(tensor_inputs.keys())
        outputs_tensor_keys = non_excluded_output_keys

        tensor_inputs_list = [tensor_inputs[key] for key in inputs_tensor_keys]
        callable_get_outputs = self._session.make_callable(
            fetches, feed_list=tensor_inputs_list)
        super().__init__(saved_model_dir, inputs_tensor_keys,
                         outputs_tensor_keys, callable_get_outputs)

  # Thread-safe.
  class _GraphStateV2(_GraphStateCommon):
    """A container for a shared TF2 graph state."""

    def __init__(self, saved_model_dir, input_tensor_names, exclude_outputs):
      saved_model_loader = saved_transform_io_v2.SavedModelLoader(
          saved_model_dir)
      callable_get_outputs = saved_model_loader.apply_transform_model
      outputs_tensor_keys = set(
          saved_model_loader.structured_outputs.keys()).difference(
              exclude_outputs)
      saved_model_loader.finalize(input_tensor_names, outputs_tensor_keys)
      super().__init__(saved_model_dir, input_tensor_names, outputs_tensor_keys,
                       callable_get_outputs)

  def __init__(self,
               tf_config,
               shared_graph_state_handle,
               passthrough_keys,
               use_tf_compat_v1,
               input_tensor_adapter_config,
               exclude_outputs=None,
               convert_passthrough_data=True):
    """Initialize.

    Args:
      tf_config: A tf.ConfigProto to use in sessions. None implies use
        Tensorflow defaults.
      shared_graph_state_handle: an instance of shared.Shared() that allows us
        to load the graph once and share it across multiple threads in the
        current process.
      passthrough_keys: A set of strings that are keys to instances that should
        pass through the pipeline and be hidden from the preprocessing_fn.
      use_tf_compat_v1: Boolean to indicate whether TFT APIs should use TF in
        compat.v1 mode.
      input_tensor_adapter_config: Tensor Adapter config.
      exclude_outputs: (Optional) A list of names of outputs to exclude.
      convert_passthrough_data: (Optional) A boolean indicating whether
        passthrough data needs to be converted to a Python list. If `False`,
        the passthrough data will be kept in input format.
    """
    super().__init__()
    self._use_tf_compat_v1 = use_tf_compat_v1
    self._input_tensor_adapter_config = input_tensor_adapter_config
    self._exclude_outputs = (
        exclude_outputs if exclude_outputs is not None else [])
    self._tf_config = tf_config
    passthrough_keys = set(passthrough_keys)
    schema_keys = self._get_input_tensor_names()
    if passthrough_keys - schema_keys != passthrough_keys:
      raise ValueError(
          'passthrough_keys overlap with schema keys: {}, {}'.format(
              passthrough_keys, schema_keys))
    self._passthrough_keys = sorted(passthrough_keys)

    # The shared graph state handle allows us to load the graph once and share
    # it across multiple threads in the current process.
    self._shared_graph_state_handle = shared_graph_state_handle

    # Initialized in process().
    self._graph_state = None
    # Initialized in setup().
    self._tensor_adapter = None
    # i-th element in this list contains the index of the column corresponding
    # to self._passthrough_keys[i].
    self._passthrough_column_indices = None
    self._convert_passthrough_data = convert_passthrough_data

    # Metrics.
    self._graph_load_seconds_distribution = beam.metrics.Metrics.distribution(
        beam_common.METRICS_NAMESPACE, 'graph_load_seconds')
    self._batch_size_distribution = beam.metrics.Metrics.distribution(
        beam_common.METRICS_NAMESPACE, 'batch_size')
    self._num_instances = beam.metrics.Metrics.counter(
        beam_common.METRICS_NAMESPACE, 'num_instances')

  def _get_input_tensor_names(self):
    return set(self._input_tensor_adapter_config.tensor_representations.keys())

  def _update_metrics(self, batch):
    self._batch_size_distribution.update(batch.num_rows)
    self._num_instances.inc(batch.num_rows)

  def _make_feed_dict(self, batch):
    # If self._use_tf_compat_v1 is True, do not produce eager tensors.
    produce_eager_tensors = not self._use_tf_compat_v1
    return self._tensor_adapter.ToBatchTensors(
        batch, produce_eager_tensors=produce_eager_tensors)

  def _get_passthrough_data_from_recordbatch(
      self, batch: pa.RecordBatch
  ) -> Dict[str, Union[List[beam_common.PRIMITIVE_TYPE], pa.Array]]:
    result = {}
    for passthrough_key, column_index in zip(self._passthrough_keys,
                                             self._passthrough_column_indices):
      if column_index >= 0:
        # The key is present in the input batch.
        passthrough_data_column = batch.column(column_index)
        # The passthrough column should be of list<primitive> type with each
        # sub-list being either null or of length 1.
        assert (pa.types.is_list(passthrough_data_column.type) or
                pa.types.is_large_list(passthrough_data_column.type))
        if self._convert_passthrough_data:
          result[passthrough_key] = [
              None if elem is None else elem[0]
              for elem in passthrough_data_column.to_pylist()
          ]
        else:
          result[passthrough_key] = passthrough_data_column
    return result

  def _handle_batch(self, batch):
    self._update_metrics(batch)
    # No need to remove (and cannot remove) the passthrough columns here:
    # 1) The TensorAdapter expects the RecordBatch to be of the same schema as
    # statically determined by the TFXIO implementation the yields the
    # TensorAdapter.
    # 2) It's not possible to leak passthrough columns through TensorAdapter
    # because they are not going to be converted to Tensors.

    feed_dict = self._make_feed_dict(batch)
    try:
      if self._use_tf_compat_v1:
        # Use self._graph_state.inputs_tensor_keys and not the dictionary keys
        # to maintain order of the feed list.
        feed_list = [
            feed_dict[name] for name in self._graph_state.inputs_tensor_keys
        ]
        outputs_list = self._graph_state.callable_get_outputs(*feed_list)
        assert len(self._graph_state.outputs_tensor_keys) == len(outputs_list)
        result = {
            key: value for key, value in zip(
                self._graph_state.outputs_tensor_keys, outputs_list)
        }
      else:
        result = self._graph_state.callable_get_outputs(feed_dict)
        assert len(self._graph_state.outputs_tensor_keys) == len(result)
    except Exception as e:
      raise ValueError(
          """An error occured while trying to apply the transformation: "{}".
          Batch instances: {},
          Fetching the values for the following Tensor keys: {}.""".format(
              str(e), batch, self._graph_state.outputs_tensor_keys))

    result.update(self._get_passthrough_data_from_recordbatch(batch))

    return result

  def _make_graph_state(self, saved_model_dir):
    start = datetime.datetime.now()
    if self._use_tf_compat_v1:
      result = self._GraphStateCompatV1(saved_model_dir,
                                        self._get_input_tensor_names(),
                                        self._exclude_outputs, self._tf_config)
    else:
      result = self._GraphStateV2(saved_model_dir,
                                  self._get_input_tensor_names(),
                                  self._exclude_outputs)
    self._graph_load_seconds_distribution.update(
        int((datetime.datetime.now() - start).total_seconds()))
    return result

  def setup(self):
    if self._input_tensor_adapter_config is not None:
      self._tensor_adapter = TensorAdapter(self._input_tensor_adapter_config)
      arrow_schema = self._input_tensor_adapter_config.arrow_schema
      self._passthrough_column_indices = [
          arrow_schema.get_field_index(k) for k in self._passthrough_keys
      ]

  def process(self, batch, saved_model_dir):
    """Runs the given graph to realize the output `Tensor` or `SparseTensor`s.

    Runs the graph in a TF session for computing the output values of the
    `Tensor` or `SparseTensor`s, given an input row of data (input `Tensor` or
    `SparseTensor`s).

    Args:
      batch: the batch of elements being processed by the DoFn
      saved_model_dir: Directory containing saved model.

    Yields:
      A representation of output features as a dict mapping keys (logical column
      names) to values.
    """
    if self._graph_state is None:
      # If available, acquire will return a cached _GraphStateCommon, since
      # calling _make_graph_state is expensive.
      self._graph_state = self._shared_graph_state_handle.acquire(
          lambda: self._make_graph_state(saved_model_dir))

    # This should remain true throughout the lifetime of this DoFn, regardless
    # of whether or not self._graph_state was cached.
    assert self._graph_state.saved_model_dir == saved_model_dir

    yield self._handle_batch(batch)


def _assert_tensorflow_version(use_tf_compat_v1):
  """Check that we're using a compatible TF version."""
  # Fail with a clear error in case we are not using a compatible TF version.
  major, minor, _ = tf.version.VERSION.split('.')
  if (int(major) not in (1, 2)) or (int(major == 1 and int(minor) < 15)):
    raise RuntimeError(
        'Tensorflow version >= 1.15, < 3 is required. Found (%s). Please '
        'install the latest 1.x or 2.x version from '
        'https://github.com/tensorflow/tensorflow. ' % tf.version.VERSION)
  if int(major) == 2 and use_tf_compat_v1:
    tf.compat.v1.logging.warning(
        'Tensorflow version (%s) found. However Tensorflow Transform '
        'is running in tf.compat.v1 mode. This could be either because TF2 '
        'was disabled or `Context.force_tf_compat_v1=True`. Features such as '
        'tf.function may not work as intended. ' % tf.version.VERSION)


def _convert_and_unbatch_to_instance_dicts(batch_dict, schema,
                                           passthrough_keys):
  """Convert batches of ndarrays to unbatched instance dicts."""

  # Making a copy of batch_dict because mutating PCollection elements is not
  # allowed.
  if passthrough_keys:
    batch_dict = copy.copy(batch_dict)
  passthrough_data = {
      key: batch_dict.pop(key) for key in passthrough_keys if key in batch_dict
  }

  result = impl_helper.to_instance_dicts(schema, batch_dict)

  for key, data in passthrough_data.items():
    data_set = set(data)
    if len(data_set) == 1:
      # Relaxing ValueError below to only trigger in case pass-through data
      # has more than one value.
      data = (data_set.pop(),) * len(result)
    if len(data) != len(result):
      raise ValueError(
          'Cannot pass-through data when input and output batch sizes '
          'are different ({} vs. {})'.format(len(data), len(result)))
    for instance, instance_data in zip(result, data):
      instance[key] = instance_data

  return result


def _convert_to_record_batch(
    batch_dict: Dict[str, Union[common_types.TensorValueType, pa.Array]],
    schema: schema_pb2.Schema,
    converter: tensor_to_arrow.TensorsToRecordBatchConverter,
    passthrough_keys: Set[str],
    input_metadata: Union[TensorAdapterConfig, dataset_metadata.DatasetMetadata]
) -> Tuple[pa.RecordBatch, Dict[str, pa.Array]]:
  """Convert batches of ndarrays to pyarrow.RecordBatch."""

  # Making a copy of batch_dict because mutating PCollection elements is not
  # allowed.
  if passthrough_keys:
    batch_dict = copy.copy(batch_dict)
  passthrough_data = {
      key: batch_dict.pop(key) for key in passthrough_keys if key in batch_dict
  }

  arrow_columns, arrow_schema = impl_helper.convert_to_arrow(
      schema, converter, batch_dict)

  batch_size = len(arrow_columns[0])
  # This dict will contain pass-through data with batch size of 1 if it doesn't
  # match batch size of the transformed data.
  unary_passthrough_features = {}
  for key, data in passthrough_data.items():
    # Only raising a ValueError in case pass-through data has more than one
    # distinct value. If it has one value and batch_size>1 then it will have to
    # be handled by the user.
    # TODO(b/38376110): Restrict to matching batch dimensions and clean this up
    # once the internal feature key is deprecated.
    if len(data) not in (batch_size, 1):
      # The passthrough column should be of list<primitive> type with each
      # sub-list being either null or of length 1.
      data_set = set(
          None if elem is None else elem[0] for elem in data.to_pylist())
      if len(data_set) == 1:
        elem = data_set.pop()
        data = pa.array([None if elem is None else [elem]], type=data.type)
      else:
        raise ValueError(
            'Cannot pass-through data when input and output batch sizes '
            'are different ({} vs. {})'.format(len(data), batch_size))
    if len(data) == batch_size:
      arrow_schema = arrow_schema.append(input_metadata.arrow_schema.field(key))
      arrow_columns.append(data)
    else:
      unary_passthrough_features[key] = data

  return pa.RecordBatch.from_arrays(
      arrow_columns, schema=arrow_schema), unary_passthrough_features


_TensorBinding = tfx_namedtuple.namedtuple(
    '_TensorBinding',
    ['value', 'tensor_name', 'dtype_enum', 'is_asset_filepath'])


@beam_common.register_ptransform(beam_nodes.CreateTensorBinding)
@beam.typehints.with_input_types(Union[np.generic, np.ndarray,
                                       Iterable[beam_common.PRIMITIVE_TYPE]])
@beam.typehints.with_output_types(_TensorBinding)
class _CreateTensorBindingsImpl(beam.PTransform):
  """Maps a PCollection of data to a PCollection of `_TensorBinding`s."""

  def __init__(self, operation, extra_args):
    del extra_args
    self._dtype_enum = operation.dtype_enum
    self._tensor_name = operation.tensor_name
    self._is_asset_file = operation.is_asset_filepath

  def expand(self, inputs):
    pcoll, = inputs
    return pcoll | 'ToTensorBinding' >> beam.Map(
        _TensorBinding, self._tensor_name, self._dtype_enum,
        self._is_asset_file)


def _get_tensor_replacement_map(graph, *tensor_bindings):
  """Get Tensor replacement map."""
  tensor_replacement_map = {}

  is_graph_mode = not ops.executing_eagerly_outside_functions()
  for tensor_binding in tensor_bindings:
    assert isinstance(tensor_binding, _TensorBinding), tensor_binding
    value = tensor_binding.value
    # TODO(b/160294509): tf.constant doesn't accept List[np.ndarray] in TF 1.15
    # graph mode. Remove this condition.
    if (is_graph_mode and isinstance(value, list) and
        any(isinstance(x, np.ndarray) for x in value)):
      value = np.asarray(tensor_binding.value)
    replacement_tensor = tf.constant(
        value, tf.dtypes.as_dtype(tensor_binding.dtype_enum))
    if graph is not None and tensor_binding.is_asset_filepath:
      graph.add_to_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS,
                              replacement_tensor)
    tensor_replacement_map[tensor_binding.tensor_name] = replacement_tensor
  return tensor_replacement_map


def _replace_tensors_with_constant_values(saved_model_dir, base_temp_dir,
                                          *tensor_bindings):
  """Replaces specified `Tensor`s with constant values.

  Constants are accepted as Python values; these are automatically
  wrapped in `tf.constant()`.

  This method creates its own temp dir, and is therefore idempotent
  since any retry will use a different temp dir.

  Args:
    saved_model_dir: A SavedModel directory providing a transform
      graph.  The MetaGraphDef and signature are selected from the
      SavedModel using keys defined in `../constants.py` ('transform'
      and 'transform_signature', respectively).
    base_temp_dir: Base temp dir for storage of new model.
    *tensor_bindings: An iterable of `_TensorBinding`s.

  Returns:
    The directory name containing the updated SavedModel.

    Raises:
      RuntimeError: if there is no default graph available to which to
        apply the transform.
  """
  with tf.compat.v1.Graph().as_default() as graph:
    tensor_replacement_map = (
        _get_tensor_replacement_map(graph, *tensor_bindings))

    with tf.compat.v1.Session(graph=graph) as session:
      temp_dir = beam_common.get_unique_temp_path(base_temp_dir)
      input_tensors, output_tensors = (
          saved_transform_io.partially_apply_saved_transform_internal(
              saved_model_dir, {}, tensor_replacement_map))
      session.run(tf.compat.v1.global_variables_initializer())
      saved_transform_io.write_saved_transform_from_session(
          session, input_tensors, output_tensors, temp_dir)
    return temp_dir


@beam_common.register_ptransform(
    beam_nodes.CreateSavedModel,
    tags={beam_common.EnvironmentTags.TF_COMPAT_V1})
@beam.typehints.with_input_types(_TensorBinding)
@beam.typehints.with_output_types(_TransformFnPathType)
class _CreateSavedModelImpl(beam.PTransform):
  """Create a SavedModel from a TF Graph."""

  def __init__(self, operation, extra_args):
    self._base_temp_dir = extra_args.base_temp_dir
    self._graph = extra_args.graph
    self._input_signature = extra_args.input_signature
    self._table_initializers = operation.table_initializers
    self._output_signature = operation.output_signature

  def expand(self, inputs):
    unbound_saved_model_dir = beam_common.get_unique_temp_path(
        self._base_temp_dir)
    with self._graph.as_default():
      with tf.compat.v1.Session(graph=self._graph) as session:
        table_initializers_ref = tf.compat.v1.get_collection_ref(
            tf.compat.v1.GraphKeys.TABLE_INITIALIZERS)
        original_table_initializers = list(table_initializers_ref)
        del table_initializers_ref[:]
        table_initializers_ref.extend(self._table_initializers)
        # Initialize all variables so they can be saved.
        session.run(tf.compat.v1.global_variables_initializer())
        saved_transform_io.write_saved_transform_from_session(
            session, self._input_signature, self._output_signature,
            unbound_saved_model_dir)
        del table_initializers_ref[:]
        table_initializers_ref.extend(original_table_initializers)
    return (inputs
            | 'BindTensors' >> _BindTensors(self._base_temp_dir,
                                            unbound_saved_model_dir)
            | 'Count' >>
            beam_common.IncrementCounter(_CREATE_SAVED_MODEL_COUNTER_NAME))


def _create_v2_saved_model(tensor_replacement_map, base_temp_dir,
                           preprocessing_fn, input_signature,
                           baseline_analyzers_fingerprint,
                           output_keys_to_name_map):
  """Writes out a SavedModelV2 with preprocessing_fn traced using tf.function.

  The SavedModel written contains a method called `transform_fn` that
  represents the traced `preprocessing_fn`. Additionally, if this is the final
  SavedModel being written out, it will contain a method called `metadata_fn`
  that provides deferred schema annotations.

  Args:
    tensor_replacement_map: A map from placeholder tensor names to their
      evaluated replacement tensors.
    base_temp_dir: Base path to write SavedModel and temporary artifacts to.
    preprocessing_fn: A user defined python function to be traced.
    input_signature: TypeSpecs describing the inputs to the `preprocessing_fn`.
    baseline_analyzers_fingerprint: A mapping from analyzer name to a set of
      paths that define its fingerprint.
    output_keys_to_name_map: A map from output dictionary keys to the names of
      the tensors that they represent.

  Returns:
    Path to which SavedModel was written.
  """
  saved_model_dir = beam_common.get_unique_temp_path(base_temp_dir)
  impl_helper.trace_and_write_v2_saved_model(saved_model_dir, preprocessing_fn,
                                             input_signature, base_temp_dir,
                                             baseline_analyzers_fingerprint,
                                             tensor_replacement_map,
                                             output_keys_to_name_map)
  return saved_model_dir


@beam_common.register_ptransform(
    beam_nodes.CreateSavedModel, tags={beam_common.EnvironmentTags.TF_V2_ONLY})
@beam.typehints.with_input_types(_TensorBinding)
@beam.typehints.with_output_types(str)
class _CreateSavedModelImplV2(beam.PTransform):
  """Create a SavedModel from a TF Graph."""

  def __init__(self, operation, extra_args):
    self._base_temp_dir = extra_args.base_temp_dir
    self._preprocessing_fn = extra_args.preprocessing_fn
    self._input_signature = extra_args.input_specs
    self._output_signature = operation.output_signature
    self._analyzers_fingerprint = extra_args.analyzers_fingerprint

  def _maybe_get_output_tensor_names_dict(self):
    # output_signature will contain CompositeTensors only if this is the final
    # SavedModel export. In this scenario, we do not need the output_signature
    # anymore as we will output everything that the preprocessing_fn returns.
    if all(isinstance(v, tf.Tensor) for v in self._output_signature.values()):
      return {k: v.name for k, v in self._output_signature.items()}
    else:
      return {}

  def expand(self, inputs):
    pipeline = (inputs[0] if isinstance(inputs, tuple) else inputs).pipeline

    input_pcoll = pipeline | 'CreateSole' >> beam.Create([None])
    if not isinstance(inputs, beam.pvalue.PBegin):
      input_pcoll |= ('ReplaceWithConstants' >> beam.Map(
          lambda _, *args: _get_tensor_replacement_map(None, *args),
          *[beam.pvalue.AsSingleton(pcoll) for pcoll in inputs]))

    return (
        input_pcoll
        | 'CreateSavedModel' >> beam.Map(
            _create_v2_saved_model, self._base_temp_dir, self._preprocessing_fn,
            self._input_signature, self._analyzers_fingerprint,
            self._maybe_get_output_tensor_names_dict())
        | 'Count' >>
        beam_common.IncrementCounter(_CREATE_SAVED_MODEL_COUNTER_NAME))


class _BindTensors(beam.PTransform):
  """PTransform to bind tensor in a SavedModel."""

  def __init__(self, base_temp_dir, unbound_saved_model_dir):
    self._base_temp_dir = base_temp_dir
    self._unbound_saved_model_dir = unbound_saved_model_dir

  def expand(self, inputs):
    pipeline = (inputs[0] if isinstance(inputs, tuple) else inputs).pipeline
    saved_model_dir_pcoll = pipeline | 'CreateSavedModel' >> beam.Create(
        [self._unbound_saved_model_dir])

    if isinstance(inputs, beam.pvalue.PBegin):
      return saved_model_dir_pcoll

    return saved_model_dir_pcoll | 'ReplaceWithConstants' >> beam.Map(
        _replace_tensors_with_constant_values, self._base_temp_dir,
        *[beam.pvalue.AsSingleton(pcoll) for pcoll in inputs])


@beam_common.register_ptransform(beam_nodes.ExtractInputForSavedModel)
class _ExtractInputForSavedModelImpl(beam.PTransform):
  """Returns a PCollection for analysis based on the specified dataset_key."""

  def __init__(self, operation, extra_args):
    self._dataset_key = operation.dataset_key
    self._flat_pcollection = extra_args.flat_pcollection
    self._pcollection_dict = extra_args.pcollection_dict

  def expand(self, pbegin):
    # TODO(b/151921205): we have to do an identity map for unmodified
    # PCollections below because otherwise we get an error from beam.
    identity_map = 'Identity' >> beam.Map(lambda x: x)
    if self._dataset_key.is_flattened_dataset_key():
      if self._flat_pcollection:
        return self._flat_pcollection | identity_map
      else:
        return (
            list(self._pcollection_dict.values())
            | 'FlattenAnalysisInputs' >> beam.Flatten(pipeline=pbegin.pipeline))
    else:
      return self._pcollection_dict[self._dataset_key] | identity_map


@beam_common.register_ptransform(beam_nodes.ApplySavedModel)
class _ApplySavedModelImpl(beam.PTransform):
  """PTransform to apply a SavedModel to data."""

  def __init__(self, operation, extra_args):
    self._use_tf_compat_v1 = extra_args.use_tf_compat_v1
    self._input_tensor_adapter_config = extra_args.input_tensor_adapter_config
    self._tf_config = extra_args.tf_config
    self._phase = operation.phase

  def expand(self, inputs):
    saved_model_dir_pcol, input_values_pcol = inputs

    # We don't deep_copy pcollections used for the first phase, or when
    # the user defined `Context` disables it.
    if self._phase > 0 and Context.get_use_deep_copy_optimization():
      # Obviates unnecessary data materialization when the input data source is
      # safe to read more than once.
      tf.compat.v1.logging.info('Deep copying inputs for phase: %d',
                                self._phase)
      input_values_pcol = deep_copy.deep_copy(input_values_pcol)

    def _convert_to_numpy(input_dict):
      """Converts eager tensors to numpy arrays."""
      return {
          k: np.asarray(v) if isinstance(v, tf.Tensor) else v
          for k, v in input_dict.items()
      }

    result = (
        input_values_pcol | 'ApplySavedModel' >> beam.ParDo(
            _RunMetaGraphDoFn(
                self._tf_config,
                use_tf_compat_v1=self._use_tf_compat_v1,
                input_tensor_adapter_config=self._input_tensor_adapter_config,
                shared_graph_state_handle=shared.Shared(),
                passthrough_keys=Context.get_passthrough_keys()),
            saved_model_dir=beam.pvalue.AsSingleton(saved_model_dir_pcol)))
    if not self._use_tf_compat_v1:
      result |= 'ConvertToNumpy' >> beam.Map(_convert_to_numpy)
    return result


@beam_common.register_ptransform(beam_nodes.ExtractFromDict)
@beam.typehints.with_input_types(Dict[str,
                                      Union[np.ndarray,
                                            tf.compat.v1.SparseTensorValue]])
class _ExtractFromDictImpl(beam.PTransform):
  """Implements ExtractFromDict by extracting the configured keys."""

  def __init__(self, operation, extra_args):
    del extra_args
    self._keys = operation.keys

  def expand(self, inputs):
    pcoll, = inputs

    def extract_keys(input_dict, keys):
      return (tuple(input_dict[k] for k in keys)
              if isinstance(keys, tuple) else input_dict[keys])

    if isinstance(self._keys, tuple):
      output_type = Tuple[(np.ndarray,) * len(self._keys)]
    else:
      output_type = np.ndarray
    return pcoll | 'ExtractKeys' >> beam.Map(
        extract_keys, keys=self._keys).with_output_types(output_type)


@beam_common.register_ptransform(beam_nodes.Flatten)
class _Flatten(beam.PTransform):
  """PTransform to flatten PCollections."""

  def __init__(self, operation, extra_args):
    del operation, extra_args  # unused

  def expand(self, inputs):
    return inputs | beam.Flatten()


def _infer_metadata_from_saved_model(
    saved_model_dir: str,
    use_tf_compat_v1: bool) -> dataset_metadata.DatasetMetadata:
  """Infers a DatasetMetadata for outputs of a SavedModel."""
  if use_tf_compat_v1:
    return _infer_metadata_from_saved_model_v1(saved_model_dir)
  else:
    return _infer_metadata_from_saved_model_v2(saved_model_dir)


def _infer_metadata_from_saved_model_v1(
    saved_model_dir: str) -> dataset_metadata.DatasetMetadata:
  """Infers a DatasetMetadata for outputs of a TF1 SavedModel."""
  with tf.compat.v1.Graph().as_default() as graph:
    with tf.compat.v1.Session(graph=graph) as session:
      _, outputs = (
          saved_transform_io.partially_apply_saved_transform_internal(
              saved_model_dir, {}))

      session.run(tf.compat.v1.global_variables_initializer())
      session.run(tf.compat.v1.tables_initializer())
      return dataset_metadata.DatasetMetadata(
          schema=schema_inference.infer_feature_schema(outputs, graph, session))


def _infer_metadata_from_saved_model_v2(
    saved_model_dir: str) -> dataset_metadata.DatasetMetadata:
  """Infers a DatasetMetadata for outputs of a TF2 SavedModel."""

  metadata_path = os.path.join(saved_model_dir, impl_helper.METADATA_DIR_NAME)
  return metadata_io.read_metadata(metadata_path)


class _InstrumentAPI(beam.PTransform):
  """PTransform that adds metrics for API usage."""

  def __init__(self, tf_graph, force_tf_compat_v1, use_tf_compat_v1):

    def _get_counter_from_graph_collection(collection_name):
      collection = tf_graph.get_collection(collection_name)
      if len(collection) > 1:
        raise ValueError(
            "Expected TF graph collection '{}' to contain at most one element. "
            'Encountered {}.'.format(collection_name, len(collection)))
      return collection[0] if collection else {}

    self._analyzer_use_counter = _get_counter_from_graph_collection(
        common.ANALYZER_COLLECTION)
    self._mapper_use_counter = _get_counter_from_graph_collection(
        common.MAPPER_COLLECTION)
    self._force_tf_compat_v1 = force_tf_compat_v1
    self._use_tf_compat_v1 = use_tf_compat_v1

  def expand(self, pipeline):

    def _make_and_increment_counters(unused_element, analyzer_counter,
                                     mapper_counter, force_tf_compat_v1,
                                     use_tf_compat_v1):
      del unused_element
      beam.metrics.Metrics.counter(beam_common.METRICS_NAMESPACE,
                                   'requested_tf_compat_v1').inc(
                                       int(force_tf_compat_v1))
      beam.metrics.Metrics.counter(beam_common.METRICS_NAMESPACE,
                                   'running_tf_compat_v1').inc(
                                       int(use_tf_compat_v1))
      for counter_prefix, counter in (('tft_analyzer_{}', analyzer_counter),
                                      ('tft_mapper_{}', mapper_counter)):
        for name, count in counter.items():
          beam.metrics.Metrics.counter(beam_common.METRICS_NAMESPACE,
                                       counter_prefix.format(name)).inc(count)

    _ = (
        pipeline
        | 'CreateSoleAPIUse' >> beam.Create([None])
        | 'CountAPIUse' >>
        beam.Map(_make_and_increment_counters, self._analyzer_use_counter,
                 self._mapper_use_counter, self._force_tf_compat_v1,
                 self._use_tf_compat_v1))


@beam.typehints.with_input_types(_DatasetElementType)
@beam.typehints.with_output_types(pa.RecordBatch)
class _InstanceDictInputToTFXIOInput(beam.PTransform):
  """PTransform that turns instance dicts into RecordBatches."""

  def __init__(self, schema, desired_batch_size):
    self._schema = schema
    # Infer all features in the schema as TensorRepresentations. This is needed
    # because TFXIO will ignore Features if TensorRepresentations are present in
    # the schema.
    # TODO(b/202791319) Investigate whether this should be done at TFXIO level.
    tensor_representations = (
        tensor_representation_util.InferTensorRepresentationsFromMixedSchema(
            schema))
    extended_schema = schema_pb2.Schema()
    extended_schema.CopyFrom(schema)
    tensor_representation_util.SetTensorRepresentationsInSchema(
        extended_schema, tensor_representations)
    self._tfxio = tf_example_record.TFExampleBeamRecord(
        physical_format='inmem',
        telemetry_descriptors=['StandaloneTFTransform'],
        schema=extended_schema)
    self._desired_batch_size = desired_batch_size

  def tensor_adapter_config(self):
    return self._tfxio.TensorAdapterConfig()

  def expand(self, instance_dict_pcoll):
    return (
        instance_dict_pcoll
        | 'EncodeInstanceDictsAsTfExample' >> beam.Map(
            example_proto_coder.ExampleProtoCoder(self._schema).encode)
        | 'TfExampleToRecordBatch' >> self._tfxio.BeamSource(
            batch_size=self._desired_batch_size))


class _AnalyzeDatasetCommon(beam.PTransform):
  """Common implementation for AnalyzeDataset, with or without cache."""

  def __init__(self, preprocessing_fn, pipeline=None):
    """Init method.

    Args:
      preprocessing_fn: A function that accepts and returns a dictionary from
        strings to `Tensor` or `SparseTensor`s.
      pipeline: (Optional) a beam Pipeline.
    """
    self._preprocessing_fn = preprocessing_fn
    self.pipeline = pipeline
    self._use_tf_compat_v1 = Context.get_use_tf_compat_v1()
    _assert_tensorflow_version(self._use_tf_compat_v1)

  def _extract_input_pvalues(self, dataset):
    # This method returns all nested pvalues to inform beam of nested pvalues.
    flat_data, data_dict, dataset_cache_dict, metadata = dataset
    pvalues = []
    # flat_data should be None when performing analysis with cache.
    if flat_data is not None:
      pvalues.append(flat_data)
    for value in data_dict.values():
      # Dataset PCollections can be None if it's fully covered by cache and so
      # there's no need in reading it.
      if value is not None:
        pvalues.append(value)
    if dataset_cache_dict is not None:
      for cache_dict in dataset_cache_dict.values():
        for cache_pcoll in cache_dict.values():
          pvalues.append(cache_pcoll)
    if isinstance(metadata, beam_metadata_io.BeamDatasetMetadata):
      pvalues.append(metadata.deferred_metadata)
    assert (self.pipeline is not None or
            pvalues), 'If there is no data, a pipeline must be provided'
    return dataset, pvalues

  def expand(self, dataset):
    """Analyze the dataset.

    Args:
      dataset: A dataset.

    Returns:
      A TransformFn containing the deferred transform function.

    Raises:
      ValueError: If preprocessing_fn has no outputs.
    """
    (flattened_pcoll, input_values_pcoll_dict, dataset_cache_dict,
     input_metadata) = dataset
    input_values_pcoll_dict = input_values_pcoll_dict or dict()

    if isinstance(input_metadata, dataset_metadata.DatasetMetadata):
      if Context.get_passthrough_keys():
        raise ValueError('passthrough_keys is set to {} but it is not supported'
                         'with instance dicts + DatasetMetadata input. Follow '
                         'the guide to switch to the TFXIO format.'.format(
                             Context.get_passthrough_keys()))
      tf.compat.v1.logging.warning(
          'You are passing instance dicts and DatasetMetadata to TFT which '
          'will not provide optimal performance. Consider following the TFT '
          'guide to upgrade to the TFXIO format (Apache Arrow RecordBatch).')
      to_tfxio_ptransform = _InstanceDictInputToTFXIOInput(
          input_metadata.schema, Context.get_desired_batch_size())
      input_tensor_adapter_config = to_tfxio_ptransform.tensor_adapter_config()
      if flattened_pcoll is not None:
        flattened_pcoll |= 'InstanceDictToRecordBatch' >> to_tfxio_ptransform
      for key in input_values_pcoll_dict.keys():
        if input_values_pcoll_dict[key] is not None:
          input_values_pcoll_dict[key] |= (
              'InstanceDictToRecordBatch[{}]'.format(key) >>
              to_tfxio_ptransform)
    else:
      input_tensor_adapter_config = input_metadata

    specs = TensorAdapter(input_tensor_adapter_config).OriginalTypeSpecs()

    if not specs:
      raise ValueError('The input metadata is empty.')

    base_temp_dir = Context.create_base_temp_dir()
    # TODO(b/149997088): Do not pass base_temp_dir here as this graph does not
    # need to be serialized to SavedModel.
    graph, structured_inputs, structured_outputs = (
        impl_helper.trace_preprocessing_function(self._preprocessing_fn, specs,
                                                 self._use_tf_compat_v1,
                                                 base_temp_dir))

    # At this point we check that the preprocessing_fn has at least one
    # output. This is because if we allowed the output of preprocessing_fn to
    # be empty, we wouldn't be able to determine how many instances to
    # "unbatch" the output into.
    if not structured_outputs:
      raise ValueError('The preprocessing function returned an empty dict')

    if graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES):
      raise ValueError(
          'The preprocessing function contained trainable variables '
          '{}'.format(
              graph.get_collection_ref(
                  tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)))

    pipeline = self.pipeline or (flattened_pcoll or next(
        v for v in input_values_pcoll_dict.values() if v is not None)).pipeline

    # Add a stage that inspects graph collections for API use counts and logs
    # them as a beam metric.
    _ = (pipeline | 'InstrumentAPI' >> _InstrumentAPI(
        graph, Context._get_force_tf_compat_v1(), self._use_tf_compat_v1))  # pylint: disable=protected-access

    if flattened_pcoll is not None:
      _ = (
          flattened_pcoll
          | 'InstrumentInputBytes[AnalysisFlattenedPColl]' >>
          telemetry.TrackRecordBatchBytes(beam_common.METRICS_NAMESPACE,
                                          'analysis_input_bytes'))
    else:
      for key in input_values_pcoll_dict.keys():
        if input_values_pcoll_dict[key] is not None:
          _ = (
              input_values_pcoll_dict[key]
              | f'InstrumentInputBytes[AnalysisPCollDict][{key}]' >>
              telemetry.TrackRecordBatchBytes(
                  beam_common.METRICS_NAMESPACE, 'analysis_input_bytes'))

    asset_map = annotators.get_asset_annotations(graph)
    # TF.HUB can error when unapproved collections are present. So we explicitly
    # clear out the collections in the graph.
    annotators.clear_asset_annotations(graph)

    analyzers_fingerprint = graph_tools.get_analyzers_fingerprint(
        graph, structured_inputs) if not self._use_tf_compat_v1 else None

    tf_config = _DEFAULT_TENSORFLOW_CONFIG_BY_BEAM_RUNNER_TYPE.get(
        type(pipeline.runner))
    extra_args = beam_common.ConstructBeamPipelineVisitor.ExtraArgs(
        base_temp_dir=base_temp_dir,
        tf_config=tf_config,
        pipeline=pipeline,
        flat_pcollection=flattened_pcoll,
        pcollection_dict=input_values_pcoll_dict,
        graph=graph,
        input_signature=structured_inputs,
        input_specs=specs,
        input_tensor_adapter_config=input_tensor_adapter_config,
        use_tf_compat_v1=self._use_tf_compat_v1,
        cache_pcoll_dict=dataset_cache_dict,
        preprocessing_fn=self._preprocessing_fn,
        analyzers_fingerprint=analyzers_fingerprint)

    transform_fn_future, cache_value_nodes = analysis_graph_builder.build(
        graph,
        structured_inputs,
        structured_outputs,
        input_values_pcoll_dict.keys(),
        cache_dict=dataset_cache_dict)
    traverser = nodes.Traverser(
        beam_common.ConstructBeamPipelineVisitor(extra_args))
    transform_fn_pcoll = traverser.visit_value_node(transform_fn_future)

    if cache_value_nodes is not None:
      output_cache_pcoll_dict = {}
      for (dataset_key, cache_key), value_node in cache_value_nodes.items():
        if dataset_key not in output_cache_pcoll_dict:
          output_cache_pcoll_dict[dataset_key] = {}
        output_cache_pcoll_dict[dataset_key][cache_key] = (
            traverser.visit_value_node(value_node))
    else:
      output_cache_pcoll_dict = None

    # Infer metadata.  We take the inferred metadata and apply overrides that
    # refer to values of tensors in the graph.  The override tensors must
    # be "constant" in that they don't depend on input data.  The tensors can
    # depend on analyzer outputs though.  This allows us to set metadata that
    # depends on analyzer outputs. _infer_metadata_from_saved_model will use the
    # analyzer outputs stored in `transform_fn` to compute the metadata in a
    # deferred manner, once the analyzer outputs are known.
    if self._use_tf_compat_v1:
      schema = schema_inference.infer_feature_schema(structured_outputs, graph)
    else:
      # Use metadata_fn here as func_graph outputs may be wrapped in an identity
      # op and hence may not return the same tensors that were annotated.
      tf_graph_context = graph_context.TFGraphContext(
          module_to_export=tf.Module(),
          temp_dir=base_temp_dir,
          evaluated_replacements={})
      concrete_metadata_fn = schema_inference.get_traced_metadata_fn(
          preprocessing_fn=self._preprocessing_fn,
          structured_inputs=structured_inputs,
          tf_graph_context=tf_graph_context,
          evaluate_schema_overrides=False)
      schema = schema_inference.infer_feature_schema_v2(
          structured_outputs,
          concrete_metadata_fn,
          evaluate_schema_overrides=False)
    deferred_metadata = (
        transform_fn_pcoll
        | 'ComputeDeferredMetadata[compat_v1={}]'.format(self._use_tf_compat_v1)
        >> beam.Map(_infer_metadata_from_saved_model, self._use_tf_compat_v1))

    full_metadata = beam_metadata_io.BeamDatasetMetadata(
        dataset_metadata.DatasetMetadata(schema=schema), deferred_metadata,
        asset_map)

    _clear_shared_state_after_barrier(pipeline, transform_fn_pcoll)

    return (transform_fn_pcoll, full_metadata), output_cache_pcoll_dict


class AnalyzeDatasetWithCache(_AnalyzeDatasetCommon):
  r"""Takes a preprocessing_fn and computes the relevant statistics.

  WARNING: This is experimental.

  Operates similarly to AnalyzeDataset, by computing the required statistics
  except this will not re-compute statistics when they are already cached, and
  will write out cache for statistics that it does compute whenever possible.

  Example use:

  >>> span_0_key = tft_beam.analyzer_cache.DatasetKey('span-0')
  >>> cache_dir = tempfile.mkdtemp()
  >>> output_path = os.path.join(tempfile.mkdtemp(), 'result')
  >>> def preprocessing_fn(inputs):
  ...   x = inputs['x']
  ...   return {'x_mean': tft.mean(x, name='x') + tf.zeros_like(x)}
  >>> feature_spec = {'x': tf.io.FixedLenFeature([], tf.float32)}
  >>> input_metadata = dataset_metadata.DatasetMetadata(
  ...   schema_utils.schema_from_feature_spec(feature_spec))
  >>> input_data_dict_0 = {span_0_key: [{'x': x} for x in range(6)]}
  >>> input_data_dict_1 = {span_0_key: [{'x': x} for x in range(6, 11)]}
  >>> empty_input_cache = {}
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   with beam.Pipeline() as p:
  ...     # Iteration #0:
  ...     transform_fn, output_cache = (
  ...         (input_data_dict_0, empty_input_cache, input_metadata)
  ...         | tft_beam.AnalyzeDatasetWithCache(preprocessing_fn))
  ...     output_cache | tft_beam.analyzer_cache.WriteAnalysisCacheToFS(
  ...         p, cache_dir)
  ...
  ...     # Iteration #1:
  ...     input_cache = p | tft_beam.analyzer_cache.ReadAnalysisCacheFromFS(
  ...          cache_dir, [span_0_key])
  ...     transform_fn, output_cache = (
  ...         (input_data_dict_1, input_cache, input_metadata)
  ...         | tft_beam.AnalyzeDatasetWithCache(preprocessing_fn))
  ...     output_cache | tft_beam.analyzer_cache.WriteAnalysisCacheToFS(
  ...         p, cache_dir)
  ...
  ...     # Applying the accumulated transformation:
  ...     transform_data = p | beam.Create(input_data_dict_0[span_0_key])
  ...     transformed_dataset = (
  ...         ((transform_data, input_metadata), transform_fn)
  ...         | tft_beam.TransformDataset())
  ...     transformed_data, transformed_metadata = transformed_dataset
  ...     (transformed_data
  ...         | beam.combiners.Sample.FixedSizeGlobally(1)
  ...         | beam.io.WriteToText(output_path, shard_name_template=''))
  >>> with open(output_path) as f:
  ...   f.read()

  "[{'x_mean': 5.0}]\n"
  """

  def _make_parent_dataset(self, dataset):
    if len(dataset) > 3:
      raise ValueError('This API no longer requires flattened_pcoll')
    return (None,) + dataset

  def _extract_input_pvalues(self, dataset):
    # This method returns all nested pvalues to inform beam of nested pvalues.
    super_dataset = self._make_parent_dataset(dataset)
    _, pvalues = super()._extract_input_pvalues(super_dataset)
    return dataset, pvalues

  def expand(self, dataset):
    input_values_pcoll_dict = dataset[1] or dict()
    analyzer_cache.validate_dataset_keys(input_values_pcoll_dict.keys())
    return super().expand(self._make_parent_dataset(dataset))


class AnalyzeDataset(_AnalyzeDatasetCommon):
  """Takes a preprocessing_fn and computes the relevant statistics.

  AnalyzeDataset accepts a preprocessing_fn in its constructor.  When its
  `expand` method is called on a dataset, it computes all the relevant
  statistics required to run the transformation described by the
  preprocessing_fn, and returns a TransformFn representing the application of
  the preprocessing_fn.
  """

  def _extract_input_pvalues(self, dataset):
    # This method returns all nested pvalues to inform beam of nested pvalues.
    data, metadata = dataset
    pvalues = [data]
    if isinstance(metadata, beam_metadata_io.BeamDatasetMetadata):
      pvalues.append(metadata.deferred_metadata)
    return dataset, pvalues

  def expand(self, dataset):
    input_values, input_metadata = dataset
    result, cache = super().expand((input_values, None, None, input_metadata))
    assert not cache
    return result


@beam.typehints.with_input_types(Union[_DatasetElementType, pa.RecordBatch])
# This PTransfrom outputs multiple PCollections and the output typehint is
# checked against each of them. That is why it needs to represent elements of
# all PCollections at the same time.
# TODO(b/182935989) Change union to multiple output types once supported.
@beam.typehints.with_output_types(Union[Union[Union[Tuple[pa.RecordBatch,
                                                          Dict[str, pa.Array]],
                                                    _DatasetElementType],
                                              dataset_metadata.DatasetMetadata],
                                        _TransformFnPathType])
class AnalyzeAndTransformDataset(beam.PTransform):
  """Combination of AnalyzeDataset and TransformDataset.

  ```python
  transformed, transform_fn = AnalyzeAndTransformDataset(
      preprocessing_fn).expand(dataset)
  ```

  should be equivalent to

  ```python
  transform_fn = AnalyzeDataset(preprocessing_fn).expand(dataset)
  transformed = TransformDataset().expand((dataset, transform_fn))
  ```

  but may be more efficient since it avoids multiple passes over the data.
  """

  def __init__(self, preprocessing_fn, output_record_batches=False):
    """Init method.

    Args:
      preprocessing_fn: A function that accepts and returns a dictionary from
          strings to `Tensor` or `SparseTensor`s.
      output_record_batches: (Optional) A bool. If `True`,
          `AnalyzeAndTransformDataset` outputs `pyarrow.RecordBatch`es;
          otherwise, outputs instance dicts.
    """
    self._preprocessing_fn = preprocessing_fn
    self._output_record_batches = output_record_batches

  def _extract_input_pvalues(self, dataset):
    # This method returns all nested pvalues to inform beam of nested pvalues.
    data, metadata = dataset
    pvalues = [data]
    if isinstance(metadata, beam_metadata_io.BeamDatasetMetadata):
      pvalues.append(metadata.deferred_metadata)
    return dataset, pvalues

  def expand(self, dataset):
    """Transform the dataset by applying the preprocessing_fn.

    Args:
      dataset: A dataset.

    Returns:
      A (Dataset, TransformFn) pair containing the preprocessed dataset and
      the graph that maps the input to the output data.
    """
    # Expand is currently implemented by composing AnalyzeDataset and
    # TransformDataset.  Future versions however could do somthing more optimal,
    # e.g. caching the values of expensive computations done in AnalyzeDataset.
    transform_fn = (
        dataset | 'AnalyzeDataset' >> AnalyzeDataset(self._preprocessing_fn))

    if Context.get_use_deep_copy_optimization():
      data, metadata = dataset

      # obviates unnecessary data materialization when the input data source is
      # safe to read more than once.
      tf.compat.v1.logging.info(
          'Deep copying the dataset before applying transformation')
      dataset = (deep_copy.deep_copy(data), metadata)

    transformed_dataset = (
        (dataset, transform_fn)
        | 'TransformDataset' >>
        TransformDataset(output_record_batches=self._output_record_batches))
    return transformed_dataset, transform_fn


def _remove_columns_from_metadata(metadata, excluded_columns):
  """Remove columns from metadata without mutating original metadata."""
  feature_spec, domains = schema_utils.schema_as_feature_spec(metadata.schema)
  new_feature_spec = {name: spec for name, spec in feature_spec.items()
                      if name not in excluded_columns}
  new_domains = {name: spec for name, spec in domains.items()
                 if name not in excluded_columns}
  return dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec(new_feature_spec, new_domains))


@beam.typehints.with_input_types(Union[_DatasetElementType, pa.RecordBatch],
                                 Union[dataset_metadata.DatasetMetadata,
                                       TensorAdapterConfig,
                                       _TransformFnPathType])
# This PTransfrom outputs multiple PCollections and the output typehint is
# checked against each of them. That is why it needs to represent elements of
# all PCollections at the same time.
# TODO(b/182935989) Change union to multiple output types once supported.
@beam.typehints.with_output_types(Union[Union[Tuple[pa.RecordBatch,
                                                    Dict[str, pa.Array]],
                                              _DatasetElementType],
                                        dataset_metadata.DatasetMetadata])
class TransformDataset(beam.PTransform):
  """Applies the transformation computed by transforming a Dataset.

  TransformDataset's `expand` method is called on a (dataset, transform_fn)
  pair. It applies the transform_fn to each row of the input dataset and
  returns the resulting dataset.

  args:
    exclude_outputs: (Optional) Output features that should not be produced.
    output_record_batches: (Optional) A bool. If `True`, `TransformDataset`
        outputs `pyarrow.RecordBatch`es; otherwise, outputs instance dicts.
  """

  def __init__(self, exclude_outputs=None, output_record_batches=False):
    self._exclude_outputs = exclude_outputs
    self._output_record_batches = output_record_batches
    self._use_tf_compat_v1 = Context.get_use_tf_compat_v1()
    _assert_tensorflow_version(self._use_tf_compat_v1)

  def _extract_input_pvalues(self, dataset_and_transform_fn):
    # This method returns all nested pvalues to inform beam of nested pvalues.
    (data, input_metadata), (transform_fn, output_metadata) = (
        dataset_and_transform_fn)
    pvalues = [data, transform_fn]
    if isinstance(input_metadata, beam_metadata_io.BeamDatasetMetadata):
      pvalues.append(input_metadata.deferred_metadata)
    if isinstance(output_metadata, beam_metadata_io.BeamDatasetMetadata):
      pvalues.append(output_metadata.deferred_metadata)
    return dataset_and_transform_fn, pvalues

  def expand(self, dataset_and_transform_fn):
    """Transforms the dataset using the transform_fn.

    Args:
      dataset_and_transform_fn: A tuple of dataset and preprocessing
      function.

    Returns:
      A dataset transformed according to the transform_fn.
    """
    (input_values, input_metadata), (transform_fn, output_metadata) = (
        dataset_and_transform_fn)
    if isinstance(input_metadata, dataset_metadata.DatasetMetadata):
      if Context.get_passthrough_keys():
        raise ValueError('passthrough_keys is set to {} but it is not '
                         'supported with instance dicts + DatasetMetadata '
                         'input. Follow the guide to switch to the TFXIO '
                         'format.'.format(Context.get_passthrough_keys()))
      tf.compat.v1.logging.warning(
          'You are passing instance dicts and DatasetMetadata to TFT which '
          'will not provide optimal performance. Consider following the TFT '
          'guide to upgrade to the TFXIO format (Apache Arrow RecordBatch).')
      to_tfxio_ptransform = _InstanceDictInputToTFXIOInput(
          input_metadata.schema, Context.get_desired_batch_size())
      input_tensor_adapter_config = to_tfxio_ptransform.tensor_adapter_config()
      input_values |= 'InstanceDictToRecordBatch' >> to_tfxio_ptransform
    else:
      input_tensor_adapter_config = input_metadata

    # If exclude_outputs is set, update the output metadata.
    if self._exclude_outputs is not None:
      if isinstance(output_metadata, beam_metadata_io.BeamDatasetMetadata):
        new_metadata = _remove_columns_from_metadata(
            output_metadata.dataset_metadata, self._exclude_outputs)
        new_deferred_metadata = (
            output_metadata.deferred_metadata
            | 'RemoveColumms' >> beam.Map(_remove_columns_from_metadata,
                                          self._exclude_outputs))
        output_metadata = beam_metadata_io.BeamDatasetMetadata(
            new_metadata, new_deferred_metadata, output_metadata.asset_map)
      else:
        output_metadata = _remove_columns_from_metadata(
            output_metadata, self._exclude_outputs)

    if isinstance(output_metadata, beam_metadata_io.BeamDatasetMetadata):
      deferred_schema = (
          output_metadata.deferred_metadata
          | 'GetDeferredSchema' >> beam.Map(lambda m: m.schema))
    else:
      deferred_schema = (
          self.pipeline
          | 'CreateDeferredSchema' >> beam.Create([output_metadata.schema]))

    _ = (
        input_values
        | 'InstrumentInputBytes[Transform]' >> telemetry.TrackRecordBatchBytes(
            beam_common.METRICS_NAMESPACE, 'transform_input_bytes'))

    tf_config = _DEFAULT_TENSORFLOW_CONFIG_BY_BEAM_RUNNER_TYPE.get(
        type(self.pipeline.runner))
    output_batches = (
        input_values
        | 'Transform' >> beam.ParDo(
            _RunMetaGraphDoFn(
                tf_config,
                input_tensor_adapter_config=input_tensor_adapter_config,
                use_tf_compat_v1=self._use_tf_compat_v1,
                shared_graph_state_handle=shared.Shared(),
                passthrough_keys=Context.get_passthrough_keys(),
                exclude_outputs=self._exclude_outputs,
                convert_passthrough_data=not self._output_record_batches),
            saved_model_dir=beam.pvalue.AsSingleton(transform_fn)))
    if self._output_record_batches:
      # Since we are using a deferred schema, obtain a pcollection containing
      # the converter that will be created from it.
      converter_pcol = (
          deferred_schema | 'MakeTensorToArrowConverter' >> beam.Map(
              impl_helper.make_tensor_to_arrow_converter))
      output_data = (
          output_batches | 'ConvertToRecordBatch' >> beam.Map(
              _convert_to_record_batch,
              schema=beam.pvalue.AsSingleton(deferred_schema),
              converter=beam.pvalue.AsSingleton(converter_pcol),
              passthrough_keys=Context.get_passthrough_keys(),
              input_metadata=input_metadata))
    else:
      output_data = (
          output_batches | 'ConvertAndUnbatchToInstanceDicts' >> beam.FlatMap(
              _convert_and_unbatch_to_instance_dicts,
              schema=beam.pvalue.AsSingleton(deferred_schema),
              passthrough_keys=Context.get_passthrough_keys()))

    _clear_shared_state_after_barrier(self.pipeline, output_data)

    return (output_data, output_metadata)
