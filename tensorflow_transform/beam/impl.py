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
../api.py for how to defined a preprocessing function) and implements it as a
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

Typical usage of these functions is shown below.

def preprocessing_fn(inputs):
  ...

with beam.Pipeline(...) as p:
  with beam_impl.Context(temp_dir=my_temp_dir):
    input = p | beam_impl.read_examples(..., schema)
    transformed, transform_fn = ((input, schema)
        | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))
    transformed | beam_impl.write_examples_and_metadata(
        examples_path, metadata_path)
    transform_fn | beam_impl.write_transform_fn(transform_fn_path)

Implementation note: TensorFlow code (including our code) makes frequent use of
the default graph.  We want to avoid adding to the default graph, or including
the default graph in our own SavedModel's.  This means that wherever we call
TensorFlow code (or our code that uses the default graph) we should create a
graph and mark it as the default.  This is achieved by identifying the
entrypoints into our code where this happens and creating a
"with ... .as_default()" block.  There are four places this happens.

1) In AnalyzeDatset.expand() which is typically called from the main thread
2) In _GraphState.__init__ which is called from the worker running
   _RunMetaGraphDoFn
3) In _replace_tensors_with_constant_values, which is called in a beam.Map.
4) In extract_scalar_constants, which is called in a beam.Map.
"""
# TODO(KesterTong): Document data format.
# TODO(KesterTong): Refactor and rename now that "TransformFn" is the path to a
# SavedModel, not an in-memory object.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import datetime
import os
import threading

# GOOGLE-INITIALIZATION

import apache_beam as beam

from apache_beam.transforms import util
from apache_beam.typehints import Any
from apache_beam.typehints import Dict
from apache_beam.typehints import List
from apache_beam.typehints import Union
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types

import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import analyzer_cache
from tensorflow_transform import impl_helper
from tensorflow_transform import nodes
from tensorflow_transform import schema_inference
from tensorflow_transform.beam import analysis_graph_builder
from tensorflow_transform.beam import beam_nodes
from tensorflow_transform.beam import common
from tensorflow_transform.beam import deep_copy
from tensorflow_transform.beam import shared
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

# TODO(b/123325923): Fix the key type here to agree with the actual keys.
_DATASET_ELEMENT_TYPE = Dict[Any,  # Any -> six.text_type?
                             Union[common.PRIMITIVE_TYPE,
                                   # Arbitrarily-nested lists are allowed.
                                   List[Any], np.generic, np.ndarray]]

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


class Context(object):
  """Context manager for tensorflow-transform.

  All the attributes in this context are kept on a thread local state.

  Args:
    temp_dir: (Optional) The temporary directory used within in this block.
    desired_batch_size: (Optional) A batch size to batch elements by. If not
        provided, a batch size will be computed automatically.
    passthrough_keys: (Optional) A set of strings that are keys to
        instances that should pass through the pipeline and be hidden from
        the preprocessing_fn. This should only be used in cases where additional
        information should be attached to instances in the pipeline which should
        not be part of the transformation graph, instance keys is one such
        example.

  Note that the temp dir should be accessible to worker jobs, e.g. if running
  with the Cloud Dataflow runner, the temp dir should be on GCS and should have
  permissions that allow both launcher and workers to access it.
  """

  class _State(
      collections.namedtuple('_State', [
          'temp_dir',
          'desired_batch_size',
          'passthrough_keys',
          'use_deep_copy_optimization',
      ])):
    pass

  class _StateStack(object):
    """Stack of states for this context manager (found in thread-local storage).
    """

    def __init__(self):
      self.frames = []

  # TODO(b/36359436) Ensure tf.Transform code only uses consistent filesystem
  # operations on Cloud.
  _TEMP_SUBDIR = 'tftransform_tmp'

  _thread_local = threading.local()

  def __init__(self,
               temp_dir=None,
               desired_batch_size=None,
               passthrough_keys=None,
               use_deep_copy_optimization=None):
    state = getattr(self._thread_local, 'state', None)
    if not state:
      self._thread_local.state = self._StateStack()
      self._thread_local.state.frames.append(
          self._State(*(None,) * len(self._State._fields)))

    self._temp_dir = temp_dir
    self._desired_batch_size = desired_batch_size
    self._passthrough_keys = passthrough_keys
    self._use_deep_copy_optimization = use_deep_copy_optimization

  def __enter__(self):
    # Previous State's properties are inherited if not explicitly specified.
    last_frame = self._get_topmost_state_frame()
    self._thread_local.state.frames.append(
        self._State(
            temp_dir=self._temp_dir
            if self._temp_dir is not None else last_frame.temp_dir,
            desired_batch_size=self._desired_batch_size
            if self._desired_batch_size is not None else
            last_frame.desired_batch_size,
            passthrough_keys=self._passthrough_keys if
            self._passthrough_keys is not None else last_frame.passthrough_keys,
            use_deep_copy_optimization=self._use_deep_copy_optimization
            if self._use_deep_copy_optimization is not None else
            last_frame.use_deep_copy_optimization,
        ))

  def __exit__(self, *exn_info):
    self._thread_local.state.frames.pop()

  @classmethod
  def _get_topmost_state_frame(cls):
    if cls._thread_local.state.frames:
      return cls._thread_local.state.frames[-1]
    return None

  @classmethod
  def create_base_temp_dir(cls):
    """Generate a temporary location."""
    state = cls._get_topmost_state_frame()
    if state is None or not state.temp_dir:
      raise ValueError(
          'A tf.Transform function that required a temp dir was called but no '
          'temp dir was set.  To set a temp dir use the impl.Context context '
          'manager.')
    base_temp_dir = os.path.join(state.temp_dir, cls._TEMP_SUBDIR)

    # TODO(b/35363519): Perhaps use Beam IO eventually?
    tf.gfile.MakeDirs(base_temp_dir)
    return base_temp_dir

  @classmethod
  def get_desired_batch_size(cls):
    """Retrieves a user set fixed batch size, None if not set."""
    state = cls._get_topmost_state_frame()
    if state is not None and state.desired_batch_size is not None:
      tf.logging.info('Using fixed batch size: %d', state.desired_batch_size)
      return state.desired_batch_size
    return None

  @classmethod
  def get_passthrough_keys(cls):
    """Retrieves a user set passthrough_keys, None if not set."""
    state = cls._get_topmost_state_frame()
    if state is not None and state.passthrough_keys is not None:
      return state.passthrough_keys
    return set()

  @classmethod
  def get_use_deep_copy_optimization(cls):
    """Retrieves a user set use_deep_copy_optimization, None if not set."""
    state = cls._get_topmost_state_frame()
    if state is not None and state.use_deep_copy_optimization is not None:
      return state.use_deep_copy_optimization
    return False


@beam.ptransform_fn
@with_input_types(_DATASET_ELEMENT_TYPE)
@with_output_types(List[_DATASET_ELEMENT_TYPE])
def _BatchElements(pcoll):  # pylint: disable=invalid-name
  """Batches elements either automatically or to the given batch_size."""
  desired_batch_size = Context.get_desired_batch_size()
  kwargs = dict(
      min_batch_size=desired_batch_size, max_batch_size=desired_batch_size
  ) if desired_batch_size is not None else {}
  return pcoll | 'BatchElements' >> util.BatchElements(**kwargs)


# TODO(b/36223892): Verify that these type hints work and make needed fixes.
@with_input_types(List[_DATASET_ELEMENT_TYPE], str)
@with_output_types(Dict[str, Union[np.ndarray, tf.SparseTensorValue]])
class _RunMetaGraphDoFn(beam.DoFn):
  """Maps a PCollection of dicts to a PCollection of dicts via a TF graph.

  The TF graph may contain more inputs than the schema provided. In that case,
  a subset of the inputs will be fed, which may cause an error if the excluded
  inputs are required to produce the included outputs.

  Args:
    input_schema: A `Schema` representing the inputs of this transform phase.
    serialized_tf_config: A serialized tf.ConfigProto to use in sessions. None
      implies use Tensorflow defaults.
    shared_graph_state_handle: an instance of shared.Shared() that allows us to
      load the graph once and share it across multiple threads in the current
      process.
    passthrough_keys: A set of strings that are keys to instances that
      should pass through the pipeline and be hidden from the preprocessing_fn.
    exclude_outputs: (Optional) A list of names of outputs to exclude.
  """

  # Thread-safe.
  class _GraphState(object):
    """A container for a shared graph state."""

    def __init__(self, saved_model_dir, input_schema, exclude_outputs,
                 tf_config):
      self.saved_model_dir = saved_model_dir
      graph = tf.Graph()
      self._session = tf.Session(graph=graph, config=tf_config)
      with graph.as_default():
        with self._session.as_default():
          inputs, outputs = (
              saved_transform_io.partially_apply_saved_transform_internal(
                  saved_model_dir, {}))
        self._session.run(tf.global_variables_initializer())
        self._session.run(tf.tables_initializer())
        graph.finalize()

        input_schema_keys = sorted(input_schema.as_feature_spec().keys())
        extra_input_keys = set(input_schema_keys).difference(inputs.keys())
        if extra_input_keys:
          raise ValueError('Input schema contained keys not in graph: %s' %
                           input_schema_keys)
        extra_output_keys = set(exclude_outputs).difference(outputs.keys())
        if extra_output_keys:
          raise ValueError('Excluded outputs contained keys not in graph: %s' %
                           exclude_outputs)
        non_excluded_output_keys = sorted(
            set(outputs.keys()).difference(exclude_outputs))
        fetches = [outputs[key] for key in non_excluded_output_keys]
        tensor_inputs = [inputs[key] for key in input_schema_keys]

        self.callable_get_outputs = self._session.make_callable(
            fetches, feed_list=tensor_inputs)

        self.inputs_tensor_keys = input_schema_keys
        self.outputs_tensor_keys = non_excluded_output_keys

  def __init__(self,
               input_schema,
               serialized_tf_config,
               shared_graph_state_handle,
               passthrough_keys,
               exclude_outputs=None):
    super(_RunMetaGraphDoFn, self).__init__()
    self._input_schema = input_schema
    self._exclude_outputs = (
        exclude_outputs if exclude_outputs is not None else [])
    self._serialized_tf_config = serialized_tf_config
    self._passthrough_keys = set(passthrough_keys)
    schema_keys = set(input_schema.as_feature_spec().keys())
    if self._passthrough_keys - schema_keys != self._passthrough_keys:
      raise ValueError(
          'passthrough_keys overlap with schema keys: {}, {}'.format(
              self._passthrough_keys, schema_keys))

    # The shared graph state handle allows us to load the graph once and share
    # it across multiple threads in the current process.
    self._shared_graph_state_handle = shared_graph_state_handle
    self._graph_state = None

    # Metrics.
    self._graph_load_seconds_distribution = beam.metrics.Metrics.distribution(
        common.METRICS_NAMESPACE, 'graph_load_seconds')
    self._batch_size_distribution = beam.metrics.Metrics.distribution(
        common.METRICS_NAMESPACE, 'batch_size')
    self._num_instances = beam.metrics.Metrics.counter(
        common.METRICS_NAMESPACE, 'num_instances')

  def _handle_batch(self, batch):
    self._batch_size_distribution.update(len(batch))
    self._num_instances.inc(len(batch))

    # Making a copy of batch because mutating PCollection elements is not
    # allowed.
    if self._passthrough_keys:
      batch = [copy.copy(x) for x in batch]
    # Extract passthrough data.
    passthrough_data = {
        key: [instance.pop(key) for instance in batch
             ] for key in self._passthrough_keys
    }

    feed_list = impl_helper.make_feed_list(self._graph_state.inputs_tensor_keys,
                                           self._input_schema, batch)

    try:
      outputs_list = self._graph_state.callable_get_outputs(*feed_list)
    except Exception as e:
      tf.logging.error('%s while applying transform function for tensors %s',
                       e, self._graph_state.outputs_tensor_keys)
      raise ValueError('bad inputs: {}'.format(feed_list))

    assert len(self._graph_state.outputs_tensor_keys) == len(outputs_list)
    result = {
        key: value for key, value in zip(self._graph_state.outputs_tensor_keys,
                                         outputs_list)
    }

    for key, value in six.iteritems(passthrough_data):
      result[key] = value

    return result

  def _make_graph_state(self, saved_model_dir):
    start = datetime.datetime.now()
    tf_config = common._maybe_deserialize_tf_config(  # pylint: disable=protected-access
        self._serialized_tf_config)
    result = self._GraphState(saved_model_dir, self._input_schema,
                              self._exclude_outputs, tf_config)
    self._graph_load_seconds_distribution.update(
        int((datetime.datetime.now() - start).total_seconds()))
    return result

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
      # If available, acquire will return a cached _GraphState, since calling
      # _make_graph_state is expensive.
      self._graph_state = self._shared_graph_state_handle.acquire(
          lambda: self._make_graph_state(saved_model_dir))

    # This should remain true throughout the lifetime of this DoFn, regardless
    # of whether or not self._graph_state was cached.
    assert self._graph_state.saved_model_dir == saved_model_dir

    yield self._handle_batch(batch)


def _assert_tensorflow_version():
  # Fail with a clear error in case we are not using a compatible TF version.
  major, minor, _ = tf.__version__.split('.')
  if int(major) != 1 or int(minor) < 12:
    raise RuntimeError(
        'TensorFlow version >= 1.12, < 2 is required. Found (%s). Please '
        'install the latest 1.x version from '
        'https://github.com/tensorflow/tensorflow. ' % tf.__version__)


def _convert_and_unbatch_to_instance_dicts(batch_dict, schema,
                                           passthrough_keys):
  """Convert batches of ndarrays to unbatched instance dicts."""

  # Making a copy of batch_dict because mutating PCollection elements is not
  # allowed.
  if passthrough_keys:
    batch_dict = copy.copy(batch_dict)
  passthrough_data = {key: batch_dict.pop(key) for key in passthrough_keys}

  result = impl_helper.to_instance_dicts(schema, batch_dict)

  for key, data in six.iteritems(passthrough_data):
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


_TensorBinding = collections.namedtuple(
    '_TensorBinding', ['value', 'tensor_name', 'is_asset_filepath'])


@common.register_ptransform(beam_nodes.CreateTensorBinding)
def _create_tensor_bindings_impl(inputs, operation, extra_args):
  del extra_args  # unused
  return inputs[0] | operation.label >> beam.Map(
      _TensorBinding, operation.tensor, operation.is_asset_filepath)


def _replace_tensors_with_constant_values(saved_model_dir, tensor_bindings,
                                          base_temp_dir):
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
    tensor_bindings: An iterable of `_TensorBinding`s.
    base_temp_dir: Base temp dir for storage of new model.

  Returns:
    The directory name containing the updated SavedModel.

    Raises:
      RuntimeError: if there is no default graph available to which to
        apply the transform.
  """
  with tf.Graph().as_default() as graph:
    tensor_replacement_map = {}
    for value, tensor_name, is_asset_filepath in tensor_bindings:
      replacement_tensor = tf.constant(value)
      if is_asset_filepath:
        graph.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS,
                                replacement_tensor)
      tensor_replacement_map[tensor_name] = replacement_tensor

    with tf.Session(graph=graph) as session:
      temp_dir = common.get_unique_temp_path(base_temp_dir)
      input_tensors, output_tensors = (
          saved_transform_io.partially_apply_saved_transform_internal(
              saved_model_dir, {}, tensor_replacement_map))
      session.run(tf.global_variables_initializer())
      saved_transform_io.write_saved_transform_from_session(
          session, input_tensors, output_tensors, temp_dir)
    return temp_dir


@common.register_ptransform(beam_nodes.CreateSavedModel)
def _create_saved_model_impl(inputs, operation, extra_args):
  """Create a SavedModel from a TF Graph."""
  unbound_saved_model_dir = common.get_unique_temp_path(
      extra_args.base_temp_dir)
  with extra_args.graph.as_default():
    with tf.Session(graph=extra_args.graph) as session:
      table_initializers_ref = tf.get_collection_ref(
          tf.GraphKeys.TABLE_INITIALIZERS)
      original_table_initializers = list(table_initializers_ref)
      del table_initializers_ref[:]
      table_initializers_ref.extend(operation.table_initializers)
      # Initialize all variables so they can be saved.
      session.run(tf.global_variables_initializer())
      saved_transform_io.write_saved_transform_from_session(
          session, extra_args.input_signature, operation.output_signature,
          unbound_saved_model_dir)
      del table_initializers_ref[:]
      table_initializers_ref.extend(original_table_initializers)
  return inputs | operation.label >> _BindTensors(
      extra_args.base_temp_dir, unbound_saved_model_dir, extra_args.pipeline)


class _BindTensors(beam.PTransform):
  """PTransform to bind tensor in a SavedModel."""

  def __init__(self, base_temp_dir, unbound_saved_model_dir, pipeline):
    self._base_temp_dir = base_temp_dir
    self._unbound_saved_model_dir = unbound_saved_model_dir
    self.pipeline = pipeline

  def expand(self, inputs):
    saved_model_dir_pcoll = self.pipeline | 'CreateSavedModel' >> beam.Create(
        [self._unbound_saved_model_dir])

    if not inputs:
      return saved_model_dir_pcoll

    flattened_tensor_bindings = (
        inputs | 'Flatten' >> beam.Flatten(pipeline=self.pipeline))
    return saved_model_dir_pcoll | 'BindTensors' >> beam.Map(
        _replace_tensors_with_constant_values,
        tensor_bindings=beam.pvalue.AsIter(flattened_tensor_bindings),
        base_temp_dir=self._base_temp_dir)


@common.register_ptransform(beam_nodes.ApplySavedModel)
class _ApplySavedModelImpl(beam.PTransform):
  """PTransform to apply a SavedModel to data."""

  def __init__(self, operation, extra_args):
    self._input_schema = extra_args.input_schema
    self._serialized_tf_config = extra_args.serialized_tf_config
    self._phase = operation.phase
    if operation.dataset_key is None:
      self._input_values_pcoll = extra_args.flat_pcollection
    else:
      self._input_values_pcoll = extra_args.pcollection_dict[
          operation.dataset_key]

  def expand(self, inputs):

    # We don't deep_copy pcollections used for the first phase, or when
    # the user defined `Context` disables it.
    if self._phase > 0 and Context.get_use_deep_copy_optimization():
      # Obviates unnecessary data materialization when the input data source is
      # safe to read more than once.
      tf.logging.info('Deep copying inputs for phase: %d', self._phase)
      input_values = deep_copy.deep_copy(self._input_values_pcoll)
    else:
      input_values = self._input_values_pcoll

    return (
        input_values
        | 'BatchInputs' >> _BatchElements()
        | 'ApplySavedModel' >> beam.ParDo(
            _RunMetaGraphDoFn(
                self._input_schema,
                self._serialized_tf_config,
                shared_graph_state_handle=shared.Shared(),
                passthrough_keys=Context.get_passthrough_keys()),
            saved_model_dir=beam.pvalue.AsSingleton(inputs[0])))


@common.register_ptransform(beam_nodes.ExtractFromDict)
def _extract_from_dict_impl(inputs, operation, extra_args):
  del extra_args  # unused
  return inputs[0] | operation.label >> beam.Map(
      lambda d, keys=operation.keys: tuple(d[key] for key in keys))


@common.register_ptransform(beam_nodes.Flatten)
class _Flatten(beam.PTransform):
  """PTransform to flatten PCollections."""

  def __init__(self, operation, extra_args):
    del extra_args  # unused
    self._label = operation.label

  def default_label(self):
    return self._label

  def expand(self, inputs):
    return inputs | beam.Flatten()


def _infer_metadata_from_saved_model(saved_model_dir):
  """Infers a DatasetMetadata for outputs of a SavedModel."""
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      _, outputs = (
          saved_transform_io.partially_apply_saved_transform_internal(
              saved_model_dir, {}))

      session.run(tf.global_variables_initializer())
      session.run(tf.tables_initializer())
      return dataset_metadata.DatasetMetadata(
          schema=schema_inference.infer_feature_schema(outputs, graph, session))


class CacheLocation(
    collections.namedtuple('CacheLocation',
                           ['input_cache_dir', 'output_cache_dir'])):
  pass


class _AnalyzeDatasetCommon(beam.PTransform):
  """Common implementation for AnalyzeDataset, with or without cache."""

  def __init__(self, preprocessing_fn, cache_location=None):
    self._preprocessing_fn = preprocessing_fn
    self._cache_location = cache_location
    _assert_tensorflow_version()

  def _extract_input_pvalues(self, dataset):
    # This method returns all nested pvalues to inform beam of nested pvalues.
    flat_data, data_dict, metadata = dataset
    pvalues = [flat_data] + [data_dict[k] for k in data_dict]
    if isinstance(metadata, beam_metadata_io.BeamDatasetMetadata):
      pvalues.append(metadata.deferred_metadata)
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
    flattened_pcoll, input_values_pcoll_dict, input_metadata = dataset
    input_schema = input_metadata.schema

    input_values_pcoll_dict = input_values_pcoll_dict or dict()

    analyzer_cache.validate_dataset_keys(input_values_pcoll_dict.keys())

    with tf.Graph().as_default() as graph:

      with tf.name_scope('inputs'):
        feature_spec = input_schema.as_feature_spec()
        input_signature = impl_helper.feature_spec_as_batched_placeholders(
            feature_spec)
        # In order to avoid a bug where import_graph_def fails when the
        # input_map and return_elements of an imported graph are the same
        # (b/34288791), we avoid using the placeholder of an input column as an
        # output of a graph. We do this by applying tf.identity to all inputs of
        # the preprocessing_fn.  Note this applies at the level of raw tensors.
        # TODO(b/34288791): Remove this workaround and use a shallow copy of
        # inputs instead.  A shallow copy is needed in case
        # self._preprocessing_fn mutates its input.
        copied_inputs = impl_helper.copy_tensors(input_signature)

      output_signature = self._preprocessing_fn(copied_inputs)

    # At this point we check that the preprocessing_fn has at least one
    # output. This is because if we allowed the output of preprocessing_fn to
    # be empty, we wouldn't be able to determine how many instances to
    # "unbatch" the output into.
    if not output_signature:
      raise ValueError('The preprocessing function returned an empty dict')

    if graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
      raise ValueError(
          'The preprocessing function contained trainable variables '
          '{}'.format(
              graph.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)))

    pipeline = flattened_pcoll.pipeline
    serialized_tf_config = common._DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER.get(  # pylint: disable=protected-access
        pipeline.runner)
    extra_args = common.ConstructBeamPipelineVisitor.ExtraArgs(
        base_temp_dir=Context.create_base_temp_dir(),
        serialized_tf_config=serialized_tf_config,
        pipeline=pipeline,
        flat_pcollection=flattened_pcoll,
        pcollection_dict=input_values_pcoll_dict,
        graph=graph,
        input_signature=input_signature,
        input_schema=input_schema,
        cache_location=self._cache_location)

    transform_fn_future = analysis_graph_builder.build(
        graph, input_signature, output_signature,
        input_values_pcoll_dict.keys(), self._cache_location)

    transform_fn_pcoll = nodes.Traverser(
        common.ConstructBeamPipelineVisitor(extra_args)).visit_value_node(
            transform_fn_future)

    # Infer metadata.  We take the inferred metadata and apply overrides that
    # refer to values of tensors in the graph.  The override tensors must
    # be "constant" in that they don't depend on input data.  The tensors can
    # depend on analyzer outputs though.  This allows us to set metadata that
    # depends on analyzer outputs. _augment_metadata will use the analyzer
    # outputs stored in `transform_fn` to compute the metadata in a
    # deferred manner, once the analyzer outputs are known.
    metadata = dataset_metadata.DatasetMetadata(
        schema=schema_inference.infer_feature_schema(output_signature, graph))

    deferred_metadata = (
        transform_fn_pcoll
        |
        'ComputeDeferredMetadata' >> beam.Map(_infer_metadata_from_saved_model))

    full_metadata = beam_metadata_io.BeamDatasetMetadata(
        metadata, deferred_metadata)

    _clear_shared_state_after_barrier(pipeline, transform_fn_pcoll)

    return transform_fn_pcoll, full_metadata


class AnalyzeDatasetWithCache(_AnalyzeDatasetCommon):
  """Takes a preprocessing_fn and computes the relevant statistics.

  WARNING: This is experimental.

  Operates similarly to AnalyzeDataset, by computing the required statistics
  except this will not re-compute statistics when they are already cached, and
  will write out cache for statistics that it does compute whenever possible.

  Args:
    preprocessing_fn: A function that accepts and returns a dictionary from
      strings to `Tensor` or `SparseTensor`s.
    cache_location: A `CacheLocation` used to determine where cache should
      written to and read from.
  """

  def __init__(self, preprocessing_fn, cache_location):
    super(AnalyzeDatasetWithCache, self).__init__(preprocessing_fn,
                                                  cache_location)


class AnalyzeDataset(_AnalyzeDatasetCommon):
  """Takes a preprocessing_fn and computes the relevant statistics.

  AnalyzeDataset accepts a preprocessing_fn in its constructor.  When its
  `expand` method is called on a dataset, it computes all the relevant
  statistics required to run the transformation described by the
  preprocessing_fn, and returns a TransformFn representing the application of
  the preprocessing_fn.

  Args:
    preprocessing_fn: A function that accepts and returns a dictionary from
      strings to `Tensor` or `SparseTensor`s.
  """

  def __init__(self, preprocessing_fn):
    super(AnalyzeDataset, self).__init__(preprocessing_fn)

  def _extract_input_pvalues(self, dataset):
    # This method returns all nested pvalues to inform beam of nested pvalues.
    data, metadata = dataset
    pvalues = [data]
    if isinstance(metadata, beam_metadata_io.BeamDatasetMetadata):
      pvalues.append(metadata.deferred_metadata)
    return dataset, pvalues

  def expand(self, dataset):
    input_values, input_metadata = dataset
    return super(AnalyzeDataset, self).expand((input_values, None,
                                               input_metadata))


class AnalyzeAndTransformDataset(beam.PTransform):
  """Combination of AnalyzeDataset and TransformDataset.

  transformed, transform_fn = AnalyzeAndTransformDataset(
      preprocessing_fn).expand(dataset)

  should be equivalent to

  transform_fn = AnalyzeDataset(preprocessing_fn).expand(dataset)
  transformed = TransformDataset().expand((dataset, transform_fn))

  but may be more efficient since it avoids multiple passes over the data.

  Args:
    preprocessing_fn: A function that accepts and returns a dictionary from
        strings to `Tensor` or `SparseTensor`s.
  """

  def __init__(self, preprocessing_fn):
    self._preprocessing_fn = preprocessing_fn
    _assert_tensorflow_version()

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
      tf.logging.info('Deep copying the dataset before applying transformation')
      dataset = (deep_copy.deep_copy(data), metadata)

    transformed_dataset = ((dataset, transform_fn)
                           | 'TransformDataset' >> TransformDataset())
    return transformed_dataset, transform_fn


def _remove_columns_from_metadata(metadata, excluded_columns):
  """Remove columns from metadata without mutating original metadata."""
  feature_spec = metadata.schema.as_feature_spec()
  domains = metadata.schema.domains()
  new_feature_spec = {name: spec for name, spec in feature_spec.items()
                      if name not in excluded_columns}
  new_domains = {name: spec for name, spec in domains.items()
                 if name not in excluded_columns}
  return dataset_metadata.DatasetMetadata(
      dataset_schema.from_feature_spec(new_feature_spec, new_domains))


class TransformDataset(beam.PTransform):
  """Applies the transformation computed by transforming a Dataset.

  TransformDataset's `expand` method is called on a (dataset, transform_fn)
  pair. It applies the transform_fn to each row of the input dataset and
  returns the resulting dataset.

  args:
    exclude_outputs: (Optional) Output features that should not be produced.
  """

  def __init__(self, exclude_outputs=None):
    self._exclude_outputs = exclude_outputs
    _assert_tensorflow_version()

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
            new_metadata, new_deferred_metadata)
      else:
        output_metadata = _remove_columns_from_metadata(
            output_metadata, self._exclude_outputs)

    serialized_tf_config = (
        common._DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER.get(  # pylint: disable=protected-access
            self.pipeline.runner))

    output_instances = (
        input_values
        | 'Batch' >> _BatchElements()
        | 'Transform' >> beam.ParDo(
            _RunMetaGraphDoFn(
                input_metadata.schema,
                serialized_tf_config,
                shared_graph_state_handle=shared.Shared(),
                passthrough_keys=Context.get_passthrough_keys(),
                exclude_outputs=self._exclude_outputs),
            saved_model_dir=beam.pvalue.AsSingleton(transform_fn))
        | 'ConvertAndUnbatch' >> beam.FlatMap(
            _convert_and_unbatch_to_instance_dicts,
            schema=output_metadata.schema,
            passthrough_keys=Context.get_passthrough_keys()))

    _clear_shared_state_after_barrier(self.pipeline, output_instances)

    return (output_instances, output_metadata)
