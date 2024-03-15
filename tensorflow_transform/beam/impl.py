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

import collections
import copy
import dataclasses
import datetime
import os

from absl import logging
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
from apache_beam.typehints import Any
from apache_beam.typehints import Dict
from apache_beam.typehints import Iterable
from apache_beam.typehints import List
from apache_beam.typehints import Optional
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
from tfx_bsl import beam as tfx_bsl_beam
from tfx_bsl.coders import example_coder
from tfx_bsl.telemetry import collection as telemetry
from tfx_bsl.telemetry import util as telemetry_util
from tfx_bsl.tfxio import tensor_representation_util
from tfx_bsl.tfxio import tensor_to_arrow
from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio.tensor_adapter import TensorAdapter
from tfx_bsl.tfxio.tensor_adapter import TensorAdapterConfig

from tensorflow_metadata.proto.v0 import schema_pb2


tfx_bsl_beam.fix_code_type_pickling()

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

# Batches larger than this will be sliced into smaller chunks. This size
# constraint must be at least as strict as the following constraints:
#   1. Number of elements in each individual array of the batch must be less
#      than or equal to 2^31 - 1. Beam's `pa.RecordBatch` PCoder does not
#      support larger sizes (even though the produced containers such as
#      LargeListArray and LargeBinaryArray support them).
#   2. Serialized size of the batch must be less than 2GB. Beam's  shuffle
#      stage will wrap the serialized batches into a proto for materialization.
#      2GB is the proto size limit.
# We set a much stricter limit than the above to additionaly improve the outputs
# handling by making the size distributed over larger number of (still
# reasonably big) batches.
_MAX_TRANSFORMED_BATCH_BYTES_SIZE = 200 << 10 << 10  # 200MB

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
@beam.typehints.with_input_types(
    Union[List[common_types.InstanceDictType], pa.RecordBatch],
    _TransformFnPathType,
)
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

  # Initialized in process().
  _graph_state: _GraphStateCommon
  # Initialized in setup().
  _tensor_adapter: TensorAdapter
  # i-th element in this list contains the index of the column corresponding
  # to self._passthrough_keys[i].
  _passthrough_column_indices: List[int]

  def __init__(
      self,
      tf_config,
      shared_graph_state_handle,
      passthrough_keys,
      use_tf_compat_v1,
      input_tensor_adapter_config,
      exclude_outputs=None,
  ):
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
  ) -> Dict[str, pa.Array]:
    result = {}
    for passthrough_key, column_index in zip(
        self._passthrough_keys, self._passthrough_column_indices
    ):
      if column_index >= 0:
        # The key is present in the input batch.
        passthrough_data_column = batch.column(column_index)
        # The passthrough column should be of (large_)list<primitive> type with
        # each sub-list being either null or of length 1.
        assert (pa.types.is_list(passthrough_data_column.type) or
                pa.types.is_large_list(passthrough_data_column.type))
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
          """An error occurred while trying to apply the transformation: "{}".
          Batch instances: {},
          Fetching the values for the following Tensor keys: {}.""".format(
              str(e), batch, self._graph_state.outputs_tensor_keys)) from e

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
    assert self._input_tensor_adapter_config is not None
    self._tensor_adapter = TensorAdapter(self._input_tensor_adapter_config)
    arrow_schema = self._input_tensor_adapter_config.arrow_schema
    self._passthrough_column_indices = [
        arrow_schema.get_field_index(k) for k in self._passthrough_keys
    ]

  def process(self, batch, saved_model_dir):
    """Runs the given graph to realize the outputs.

    Runs the graph in a TF session for computing the output values of the
    `Tensor`s, `SparseTensor`s, or `RaggedTensor`s, given an input row of data
    (input `Tensor`s, `SparseTensor`s, or `RaggedTensor`s).

    Args:
      batch: the batch of elements being processed by the DoFn
      saved_model_dir: Directory containing saved model.

    Yields:
      A representation of output features as a dict mapping keys (logical column
      names) to values.
    """
    if not hasattr(self, '_graph_state'):
      # If available, acquire will return a cached _GraphStateCommon, since
      # calling _make_graph_state is expensive.
      self._graph_state = self._shared_graph_state_handle.acquire(
          lambda: self._make_graph_state(saved_model_dir))

    # This should remain true throughout the lifetime of this DoFn, regardless
    # of whether or not self._graph_state was cached.
    assert self._graph_state.saved_model_dir == saved_model_dir

    yield self._handle_batch(batch)


def _warn_about_tf_compat_v1():
  """Warns about using tf.compat.v1."""
  logging.warning(
      'Tensorflow Transform is running in tf.compat.v1 mode. This could be '
      'either because TF2 was disabled or `Context.force_tf_compat_v1=True`. '
      'Features such as tf.function may not work as intended.')


def _maybe_slice_large_record_batch(
    record_batch: pa.RecordBatch,
) -> Iterable[pa.RecordBatch]:
  """Slices large batches into smaller chunks."""
  if record_batch.nbytes > _MAX_TRANSFORMED_BATCH_BYTES_SIZE:
    if record_batch.num_rows < 2:
      logging.warning(
          'Transformed data row may be too large: %d bytes. '
          'Consider reshaping outputs to distribute elements over a larger '
          'number of rows to allow automatic slicing.',
          record_batch.nbytes,
      )
      yield record_batch
      return
    # Note that slicing is a zero-copy operation, so the produced batches will
    # still share memory with the original one up to the materialization
    # boundary.
    mid_point = record_batch.num_rows // 2
    yield from _maybe_slice_large_record_batch(
        record_batch.slice(offset=0, length=mid_point)
    )
    yield from _maybe_slice_large_record_batch(
        record_batch.slice(offset=mid_point)
    )
  else:
    yield record_batch


def _convert_to_record_batch(
    batch_dict: Dict[str, Union[common_types.TensorValueType, pa.Array]],
    converter: tensor_to_arrow.TensorsToRecordBatchConverter,
    passthrough_keys: Set[str],
    input_metadata: Union[
        TensorAdapterConfig, dataset_metadata.DatasetMetadata
    ],
    validate_varlen_sparse_values: bool = False,
) -> Iterable[Tuple[pa.RecordBatch, Dict[str, pa.Array]]]:
  """Convert batch of ndarrays to pyarrow.RecordBatches."""

  # Making a copy of batch_dict because mutating PCollection elements is not
  # allowed.
  if passthrough_keys:
    batch_dict = copy.copy(batch_dict)
  passthrough_data = {
      key: batch_dict.pop(key) for key in passthrough_keys if key in batch_dict
  }

  if validate_varlen_sparse_values:
    for name, representation in converter.tensor_representations().items():
      if representation.WhichOneof('kind') == 'varlen_sparse_tensor':
        impl_helper.validate_varlen_sparse_value(name, batch_dict[name])

  record_batch = converter.convert(batch_dict)
  arrow_columns, arrow_schema = record_batch.columns, record_batch.schema

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
  for reccord_batch in _maybe_slice_large_record_batch(
      pa.RecordBatch.from_arrays(arrow_columns, schema=arrow_schema)
  ):
    yield reccord_batch, unary_passthrough_features


def _transformed_batch_to_instance_dicts(
    transformed_batch: Tuple[pa.RecordBatch, Dict[str, pa.Array]],
    schema: schema_pb2.Schema,
):
  """Converts batch of transformed data to unbatched instance dicts."""
  record_batch, unary_passthrough_features = transformed_batch
  result = impl_helper.record_batch_to_instance_dicts(record_batch, schema)
  # Convert unary passthrough data to Python primitives.
  for key, value in unary_passthrough_features.items():
    value = value.to_pylist()
    value = None if value[0] is None else value[0]
    for instance in result:
      instance[key] = value
  return result


@dataclasses.dataclass(frozen=True)
class _TensorBinding:
  value: Any
  tensor_name: str
  dtype_enum: int
  is_asset_filepath: bool


@beam_common.register_ptransform(beam_nodes.CreateTensorBinding)
@beam.typehints.with_input_types(common_types.InstanceValueType)
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

  for tensor_binding in tensor_bindings:
    assert isinstance(tensor_binding, _TensorBinding), tensor_binding
    replacement_tensor = tf.constant(
        tensor_binding.value, tf.dtypes.as_dtype(tensor_binding.dtype_enum))
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
                           output_keys_to_name_map, save_options):
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
    save_options: The tf.saved_model.SaveOptions to save the model with.

  Returns:
    Path to which SavedModel was written.
  """
  saved_model_dir = beam_common.get_unique_temp_path(base_temp_dir)
  impl_helper.trace_and_write_v2_saved_model(saved_model_dir, preprocessing_fn,
                                             input_signature, base_temp_dir,
                                             baseline_analyzers_fingerprint,
                                             tensor_replacement_map,
                                             output_keys_to_name_map,
                                             save_options)
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
    self._save_options = extra_args.save_options

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
            self._maybe_get_output_tensor_names_dict(), self._save_options)
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
      logging.info('Deep copying inputs for phase: %d', self._phase)
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
      output_type = Tuple[(Any,) * len(self._keys)]
    else:
      output_type = Any
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


@beam.typehints.with_input_types(common_types.InstanceDictType)
@beam.typehints.with_output_types(pa.RecordBatch)
class _InstanceDictInputToTFXIOInput(beam.PTransform):
  """PTransform that turns instance dicts into RecordBatches."""

  def __init__(self, schema, desired_batch_size):
    self._schema = schema
    self._tfxio = tf_example_record.TFExampleBeamRecord(
        physical_format='inmem',
        telemetry_descriptors=['StandaloneTFTransform'],
        schema=schema)
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


def _make_output_cache(
    cache_value_nodes: Optional[analysis_graph_builder.AnalysisCache],
    traverser: nodes.Traverser,
    dataset_metrics: Dict[
        analyzer_cache.DatasetKey, analyzer_cache.DatasetCacheMetadata
    ],
) -> Optional[analyzer_cache.BeamAnalysisCache]:
  """Triggers dataset cache encoding and composes analysis cache output."""
  if cache_value_nodes is None:
    return None
  cache_dict = collections.defaultdict(dict)
  for dataset_key, dataset_cache in cache_value_nodes.items():
    for cache_key, value_node in dataset_cache.items():
      cache_dict[dataset_key][cache_key] = traverser.visit_value_node(
          value_node
      )
  return {
      dataset_key: analyzer_cache.DatasetCache(cache,
                                               dataset_metrics[dataset_key])
      for dataset_key, cache in cache_dict.items()
  }


class _AnalyzeDatasetCommon(beam.PTransform):
  """Common implementation for AnalyzeDataset, with or without cache."""

  def __init__(self, preprocessing_fn, pipeline=None):
    """Init method.

    Args:
      preprocessing_fn: A function that accepts and returns a dictionary from
        strings to `Tensor`s, `SparseTensor`s, or `RaggedTensor`s.
      pipeline: (Optional) a beam Pipeline.
    """
    self._preprocessing_fn = preprocessing_fn
    self.pipeline = pipeline
    self._save_options = Context.get_save_options()
    self._use_tf_compat_v1 = Context.get_use_tf_compat_v1()
    if self._use_tf_compat_v1:
      _warn_about_tf_compat_v1()

  def _extract_input_pvalues(self, dataset):
    # This method returns all nested pvalues to inform beam of nested pvalues.
    flat_data, data_dict, dataset_cache_dict, metadata = dataset
    pvalues = []
    # flat_data should be None when performing analysis with cache.
    if flat_data is not None:
      pvalues.append(flat_data)
    if data_dict:
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
      logging.warning(
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
    assert input_tensor_adapter_config is not None

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
    if not isinstance(structured_outputs, dict):
      raise ValueError(
          'A `preprocessing_fn` must return a '
          'Dict[str, Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]]. '
          f'Got: {structured_outputs}')
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

    dataset_metrics = {}
    if flattened_pcoll is not None:
      _ = (
          flattened_pcoll
          | 'InstrumentInputBytes[AnalysisFlattenedPColl]' >>
          telemetry.TrackRecordBatchBytes(beam_common.METRICS_NAMESPACE,
                                          'analysis_input_bytes'))
    else:
      for idx, key in enumerate(sorted(input_values_pcoll_dict.keys())):
        infix = f'AnalysisIndex{idx}'
        input_value = input_values_pcoll_dict[key]
        if input_value is not None:
          dataset_metrics[key] = (
              input_value
              | f'GetRecordBatchSize[{infix}]' >> beam.Map(lambda rb: rb.nbytes)
              | f'SumTotalBytes[{infix}]' >> beam.CombineGlobally(sum)
              | f'ConstructMetadata[{infix}]' >> beam.Map(
                  analyzer_cache.DatasetCacheMetadata))
          _ = (
              input_value
              | f'InstrumentInputBytes[{infix}]'
              >> telemetry.TrackRecordBatchBytes(
                  beam_common.METRICS_NAMESPACE, 'analysis_input_bytes'))

    # Gather telemetry on types of input features.
    _ = (
        pipeline
        | 'CreateAnalyzeInputTensorRepresentations'
        >> beam.Create([input_tensor_adapter_config.tensor_representations])
        | 'InstrumentAnalyzeInputTensors'
        >> telemetry.TrackTensorRepresentations(
            telemetry_util.AppendToNamespace(
                beam_common.METRICS_NAMESPACE, ['analyze_input_tensors']
            )
        )
    )

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
        analyzers_fingerprint=analyzers_fingerprint,
        save_options=self._save_options)

    (transform_fn_future, cache_value_nodes,
     detached_sideeffect_leafs) = analysis_graph_builder.build(
         graph,
         structured_inputs,
         structured_outputs,
         input_values_pcoll_dict.keys(),
         cache_dict=dataset_cache_dict)
    traverser = nodes.Traverser(
        beam_common.ConstructBeamPipelineVisitor(extra_args))
    transform_fn_pcoll = traverser.visit_value_node(transform_fn_future)

    # Cause side-effect nodes to get executed.
    for node in detached_sideeffect_leafs:
      traverser.visit_value_node(node)

    output_cache_pcoll_dict = _make_output_cache(cache_value_nodes, traverser,
                                                 dataset_metrics)

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
  >>> input_metadata = tft.DatasetMetadata.from_feature_spec(feature_spec)
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


@beam.typehints.with_input_types(
    Union[common_types.InstanceDictType, pa.RecordBatch]
)
# This PTransfrom outputs multiple PCollections and the output typehint is
# checked against each of them. That is why it needs to represent elements of
# all PCollections at the same time.
# TODO(b/182935989) Change union to multiple output types once supported.
@beam.typehints.with_output_types(
    Union[
        Union[
            Union[
                Tuple[pa.RecordBatch, Dict[str, pa.Array]],
                common_types.InstanceDictType,
            ],
            dataset_metadata.DatasetMetadata,
        ],
        _TransformFnPathType,
    ]
)
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
          strings to `Tensor`s, `SparseTensor`s, or `RaggedTensor`s.
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
      logging.info(
          'Deep copying the dataset before applying transformation')
      dataset = (deep_copy.deep_copy(data), metadata)

    transformed_dataset = (
        (dataset, transform_fn)
        | 'TransformDataset' >>
        TransformDataset(output_record_batches=self._output_record_batches))
    return transformed_dataset, transform_fn


def _remove_columns_from_metadata(metadata, excluded_columns):
  """Remove columns from metadata without mutating original metadata."""
  generated = schema_utils.schema_as_feature_spec(metadata.schema)
  new_feature_spec = {
      name: spec
      for name, spec in generated.feature_spec.items()
      if name not in excluded_columns
  }
  new_domains = {
      name: spec
      for name, spec in generated.domains.items()
      if name not in excluded_columns
  }
  return dataset_metadata.DatasetMetadata.from_feature_spec(
      new_feature_spec, new_domains)


class _MaybeInferTensorRepresentationsDoFn(beam.DoFn):
  """Tries to infer TensorRepresentations from a Schema."""

  def process(
      self, schema: schema_pb2.Schema
  ) -> Iterable[Dict[str, schema_pb2.TensorRepresentation]]:
    try:
      yield (tensor_representation_util
             .InferTensorRepresentationsFromMixedSchema(schema))
    except ValueError:
      # Ignore any inference errors since the output is only used for metrics.
      yield {}


@beam.typehints.with_input_types(
    Union[common_types.InstanceDictType, pa.RecordBatch],
    Union[
        dataset_metadata.DatasetMetadata,
        TensorAdapterConfig,
        _TransformFnPathType,
    ],
)
# This PTransfrom outputs multiple PCollections and the output typehint is
# checked against each of them. That is why it needs to represent elements of
# all PCollections at the same time.
# TODO(b/182935989) Change union to multiple output types once supported.
@beam.typehints.with_output_types(
    Union[
        Union[
            Tuple[pa.RecordBatch, Dict[str, pa.Array]],
            common_types.InstanceDictType,
        ],
        dataset_metadata.DatasetMetadata,
    ]
)
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
    if self._use_tf_compat_v1:
      _warn_about_tf_compat_v1()

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
      logging.warning(
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
            | 'RemoveColumns'
            >> beam.Map(_remove_columns_from_metadata, self._exclude_outputs)
        )
        output_metadata = beam_metadata_io.BeamDatasetMetadata(
            new_metadata, new_deferred_metadata, output_metadata.asset_map)
      else:
        output_metadata = _remove_columns_from_metadata(
            output_metadata, self._exclude_outputs)

    if isinstance(output_metadata, beam_metadata_io.BeamDatasetMetadata):
      deferred_schema = (
          output_metadata.deferred_metadata
          | 'GetDeferredSchema' >> beam.Map(lambda m: m.schema))
      output_dataset_metadata = output_metadata.dataset_metadata
    else:
      deferred_schema = (
          self.pipeline
          | 'CreateDeferredSchema' >> beam.Create([output_metadata.schema]))
      output_dataset_metadata = output_metadata
    output_dataset_metadata._output_record_batches = self._output_record_batches  # pylint: disable=protected-access

    # Increment input metrics.
    _ = (
        input_values
        | 'InstrumentInputBytes[Transform]' >> telemetry.TrackRecordBatchBytes(
            beam_common.METRICS_NAMESPACE, 'transform_input_bytes'))

    _ = (
        self.pipeline | 'CreateTransformInputTensorRepresentations' >>
        beam.Create([input_tensor_adapter_config.tensor_representations])
        | 'InstrumentTransformInputTensors' >>
        telemetry.TrackTensorRepresentations(
            telemetry_util.AppendToNamespace(beam_common.METRICS_NAMESPACE,
                                             ['transform_input_tensors'])))

    tf_config = _DEFAULT_TENSORFLOW_CONFIG_BY_BEAM_RUNNER_TYPE.get(
        type(self.pipeline.runner))
    output_batches = input_values | 'Transform' >> beam.ParDo(
        _RunMetaGraphDoFn(
            tf_config,
            input_tensor_adapter_config=input_tensor_adapter_config,
            use_tf_compat_v1=self._use_tf_compat_v1,
            shared_graph_state_handle=shared.Shared(),
            passthrough_keys=Context.get_passthrough_keys(),
            exclude_outputs=self._exclude_outputs,
        ),
        saved_model_dir=beam.pvalue.AsSingleton(transform_fn),
    )

    # Since we are using a deferred schema, obtain a pcollection containing
    # the converter that will be created from it.
    converter_pcol = deferred_schema | 'MakeTensorToArrowConverter' >> beam.Map(
        impl_helper.make_tensor_to_arrow_converter
    )

    # Increment output data metrics.
    _ = (
        converter_pcol
        | 'MapToTensorRepresentations'
        >> beam.Map(lambda converter: converter.tensor_representations())
        | 'InstrumentTransformOutputTensors'
        >> telemetry.TrackTensorRepresentations(
            telemetry_util.AppendToNamespace(
                beam_common.METRICS_NAMESPACE, ['transform_output_tensors']
            )
        )
    )

    output_data = output_batches | 'ConvertToRecordBatch' >> beam.FlatMap(
        _convert_to_record_batch,
        converter=beam.pvalue.AsSingleton(converter_pcol),
        passthrough_keys=Context.get_passthrough_keys(),
        input_metadata=input_metadata,
        # TODO(b/254822532): Consider always doing the validation.
        validate_varlen_sparse_values=not self._output_record_batches,
    )

    if not self._output_record_batches:
      logging.warning(
          'You are outputting instance dicts from `TransformDataset` which '
          'will not provide optimal performance. Consider setting  '
          '`output_record_batches=True` to upgrade to the TFXIO format (Apache '
          'Arrow RecordBatch). Encoding functionality in this module works '
          'with both formats.'
      )
      output_data |= 'ConvertAndUnbatchToInstanceDicts' >> beam.FlatMap(
          _transformed_batch_to_instance_dicts,
          schema=beam.pvalue.AsSingleton(deferred_schema),
      )

    _clear_shared_state_after_barrier(self.pipeline, output_data)

    return (output_data, output_metadata)


class EncodeTransformedDataset(beam.PTransform):
  """Encodes transformed data into serialized tf.Examples.

  Should operate on the output of `TransformDataset`, this can operate on either
  record batch or instance dict data.
  The expected input is a (transformed_data, transformed_metadata) tuple.

  Example use:

  >>> def preprocessing_fn(inputs):
  ...   return {'x_scaled': tft.scale_to_z_score(inputs['x'], name='x')}
  >>> raw_data = [dict(x=1), dict(x=2), dict(x=3)]
  >>> feature_spec = dict(x=tf.io.FixedLenFeature([], tf.int64))
  >>> raw_data_metadata = tft.DatasetMetadata.from_feature_spec(feature_spec)
  >>> output_path = os.path.join(tempfile.mkdtemp(), 'result')
  >>> with beam.Pipeline() as p:
  ...   with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...     data_pcoll = p | beam.Create(raw_data)
  ...     transformed_dataset, transform_fn = (
  ...         (data_pcoll, raw_data_metadata)
  ...         | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  ...     _ = (
  ...       transformed_dataset
  ...       | tft_beam.EncodeTransformedDataset()
  ...       | beam.io.WriteToTFRecord(output_path, shard_name_template=''))
  >>> result_feature_spec ={'x_scaled': tf.io.FixedLenFeature([], tf.float32)}
  >>> list(tf.data.TFRecordDataset([output_path])
  ...      .map(lambda x: tf.io.parse_example(x, result_feature_spec))
  ...      .as_numpy_iterator())
  [{'x_scaled': -1.2247448}, {'x_scaled': 0.0}, {'x_scaled': 1.2247448}]
  """

  def _extract_input_pvalues(self, transformed_data_and_metadata):
    # This method lets beam know that metadata is not a pvalue.
    return transformed_data_and_metadata, [transformed_data_and_metadata[0]]

  def expand(self, transformed_data_and_metadata):

    transformed_data, transformed_metadata = transformed_data_and_metadata

    deferred_schema = (
        transformed_metadata.deferred_metadata
        | 'GetDeferredSchema' >> beam.Map(lambda m: m.schema))

    if transformed_metadata.dataset_metadata._output_record_batches:  # pylint: disable=protected-access
      transformed_data_coder_pcol = (
          deferred_schema | 'RecordBatchToExamplesEncoder' >> beam.Map(
              example_coder.RecordBatchToExamplesEncoder))
      encode_ptransform = 'EncodeRecordBatches' >> beam.FlatMap(
          # Dropping passthrough features.
          lambda elem, coder: coder.encode(elem[0]),
          coder=beam.pvalue.AsSingleton(transformed_data_coder_pcol))
    else:
      transformed_data_coder_pcol = (
          deferred_schema
          | 'ExampleProtoCoder' >> beam.Map(
              example_proto_coder.ExampleProtoCoder))
      encode_ptransform = 'EncodeInstances' >> beam.Map(
          lambda data, data_coder: data_coder.encode(data),
          data_coder=beam.pvalue.AsSingleton(transformed_data_coder_pcol))

    return transformed_data | encode_ptransform
