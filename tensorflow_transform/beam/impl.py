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

"""

import datetime
import os
import threading
import uuid


import apache_beam as beam

from apache_beam.transforms import window
from apache_beam.typehints import Any
from apache_beam.typehints import Dict
from apache_beam.typehints import List
from apache_beam.typehints import Union
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types
from apache_beam.utils.windowed_value import WindowedValue

import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import analyzers as tft_analyzers
from tensorflow_transform import api as tft_api
from tensorflow_transform import impl_helper
from tensorflow_transform.beam import analyzer_impls
from tensorflow_transform.beam import common
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

_DEFAULT_DESIRED_BATCH_SIZE = 1000

_DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER = {
    # We rely on Beam to manage concurrency, i.e. we expect it to run one
    # session per CPU--so we don't want to proliferate TF threads.
    # Nonetheless we provide 4 threads per session for TF ops, 2 inter-
    # and 2 intra-thread.  In many cases only 2 of these will be runnable
    # at any given time.  This approach oversubscribes a bit to make sure
    # the CPUs are really saturated.
    #
    beam.runners.DataflowRunner:
        tf.ConfigProto(
            use_per_session_threads=True,
            inter_op_parallelism_threads=2,
            intra_op_parallelism_threads=2).SerializeToString(),

}


def _maybe_deserialize_tf_config(serialized_tf_config):
  if serialized_tf_config is None:
    return None

  result = tf.ConfigProto()
  result.ParseFromString(serialized_tf_config)
  return result


class Context(object):
  """Context manager for tensorflow-transform.

  All the attributes in this context are kept on a thread local state.

  Args:
    teamp_dir: the temporary directory used within in this block.

  Note that the temp dir should be accessible to worker jobs, e.g. if running
  with the Cloud Dataflow runner, the temp dir should be on GCS and should have
  permissions that allow both launcher and workers to access it.

  When running on Cloud Dataflow, the temp dir should also be in a regional
  bucket, as only regional buckets provide the consistency guarantees required
  by tf.Transform.  This requirement will be removed in later versions.
  """

  class _State(object):
    """State for this context manager (found in thread-local storage).

    Attributes:
      temp_dirs: A stack for storing the nested-context temporary directories.
    """

    def __init__(self):
      self.temp_dirs = []

  _TEMP_SUBDIR = 'tftransform_tmp'

  _thread_local = threading.local()

  def __init__(self, temp_dir):
    state = getattr(self._thread_local, 'state', None)
    if not state:
      self._thread_local.state = self._State()
    self._temp_dir = temp_dir

  def __enter__(self):
    self._thread_local.state.temp_dirs.append(self._temp_dir)

  def __exit__(self, *exn_info):
    self._thread_local.state.temp_dirs.pop()

  @classmethod
  def create_base_temp_dir(cls):
    """Generate a temporary location."""
    if cls._thread_local.state.temp_dirs:
      base_temp_dir = os.path.join(cls._thread_local.state.temp_dirs[-1],
                                   cls._TEMP_SUBDIR)
    else:
      raise ValueError(
          'A tf.Transform function that required a temp dir was called but no '
          'temp dir was set.  To set a temp dir use the impl.Context context '
          'manager.')
    tf.gfile.MakeDirs(base_temp_dir)
    return base_temp_dir


@with_input_types(Dict[str,
                       Union[common.PRIMITIVE_TYPE,
                             List[Any],  # Arbitrarily-nested lists are allowed.
                             np.generic,
                             np.ndarray]],
                  str)
@with_output_types(Dict[str, Union[np.ndarray, tf.SparseTensorValue]])
class _RunMetaGraphDoFn(beam.DoFn):
  """Maps a PCollection of dicts to a PCollection of dicts via a TF graph.

  The TF graph may contain more inputs and outputs than the schema provided.
  In that case, a subset of the graph will be run, which may cause an error if
  the excluded inputs are required to produce the included outputs.

  Args:
    input_schema: A `Schema` representing the inputs of this transform phase.
    output_schema: A `Schema` representing the outputs of this transform phase.
    exclude_outputs: A list of names of outputs to exclude.
    desired_batch_size: The desired number of instances to convert into a batch
      before feeding to Tensorflow.
    serialized_tf_config: A serialized tf.ConfigProto to use in sessions. None
      implies use Tensorflow defaults.
  """

  class _GraphState(object):

    def __init__(self, saved_model_dir, input_schema, output_schema,
                 tf_config):
      self.saved_model_dir = saved_model_dir
      self.graph = tf.Graph()
      self.session = tf.Session(graph=self.graph, config=tf_config)
      with self.graph.as_default():
        with tf.Session(config=tf_config):
          inputs, outputs = saved_transform_io.partially_apply_saved_transform(
              saved_model_dir, {})
        self.session.run(tf.tables_initializer())

        input_schema_keys = input_schema.column_schemas.keys()
        output_schema_keys = output_schema.column_schemas.keys()
        extra_input_keys = set(input_schema_keys).difference(inputs.keys())
        if extra_input_keys:
          raise ValueError('Input schema contained keys not in graph: %s' %
                           input_schema_keys)
        extra_output_keys = set(output_schema_keys).difference(outputs.keys())
        if extra_output_keys:
          raise ValueError('Output schema contained keys not in graph: %s' %
                           extra_output_keys)
        self.inputs = {key: inputs[key] for key in input_schema_keys}
        self.outputs = {key: outputs[key] for key in output_schema_keys}

  _thread_local = threading.local()

  def __init__(self,
               input_schema,
               output_schema,
               serialized_tf_config,
               exclude_outputs=None,
               desired_batch_size=_DEFAULT_DESIRED_BATCH_SIZE):
    super(_RunMetaGraphDoFn, self).__init__()
    self._input_schema = input_schema
    self._output_schema = output_schema
    self._serialized_tf_config = serialized_tf_config
    self._exclude_outputs = exclude_outputs
    self._desired_batch_size = desired_batch_size

    self._batch = []
    self._graph_state = None

    # Metrics.
    self._graph_load_seconds_distribution = beam.metrics.Metrics.distribution(
        self.__class__, 'graph_load_seconds')
    self._batch_size_distribution = beam.metrics.Metrics.distribution(
        self.__class__, 'batch_size')
    self._num_instances = beam.metrics.Metrics.counter(self.__class__,
                                                       'num_instances')

  def _flush_batch(self):
    self._batch_size_distribution.update(len(self._batch))
    self._num_instances.inc(len(self._batch))

    feed_dict = impl_helper.make_feed_dict(
        self._graph_state.inputs, self._input_schema, self._batch)
    del self._batch[:]

    try:
      return self._graph_state.session.run(
          self._graph_state.outputs, feed_dict=feed_dict)
    except Exception as e:
      tf.logging.error('%s while applying transform function for tensors %s' %
                       (e, self._graph_state.outputs))
      raise

  def process(self, element, saved_model_dir):
    """Runs the given graph to realize the output `Tensor` or `SparseTensor`s.

    Runs the graph in a TF session for computing the output values of the
    `Tensor` or `SparseTensor`s, given an input row of data (input `Tensor` or
    `SparseTensor`s).

    Args:
      element: the element being processed by the DoFn
      saved_model_dir: Directory containing saved model.

    Yields:
      A representation of output features as a dict mapping keys (logical column
      names) to values.
    """
    if self._graph_state is None:
      if (getattr(self._thread_local, 'graph_state', None) is None or
          self._thread_local.graph_state.saved_model_dir != saved_model_dir):
        start = datetime.datetime.now()
        tf_config = _maybe_deserialize_tf_config(self._serialized_tf_config)
        self._thread_local.graph_state = self._GraphState(
            saved_model_dir, self._input_schema, self._output_schema, tf_config)
        self._graph_load_seconds_distribution.update(
            int((datetime.datetime.now() - start).total_seconds()))
      self._graph_state = self._thread_local.graph_state
    else:
      assert self._graph_state.saved_model_dir == saved_model_dir

    self._batch.append(element)
    if len(self._batch) >= self._desired_batch_size:
      yield self._flush_batch()

  def finish_bundle(self):
    if self._batch:
      yield WindowedValue(self._flush_batch(), -1, [window.GlobalWindow()])


def _assert_tensorflow_version():
  try:
    _ = tf.SparseFeature
    _ = tf.tables_initializer
  except AttributeError:
    raise RuntimeError(
        'Tensorflow version 1.0 is required. Please install the latest version '
        'from https://github.com/tensorflow/tensorflow.')


def _make_unique_temp_dir(base_temp_dir):
  """Create path to a unique temp dir from given base temp dir."""
  return os.path.join(base_temp_dir, uuid.uuid4().hex)


def _write_saved_transform(graph, inputs, outputs, saved_model_dir):
  """Write the given function as a saved transform."""
  with tf.Session(graph=graph) as session:
    # Remove collections that can't be serialized, as these produce annoying
    # warnings.
    collections_blacklist = [
        tft_api.FUNCTION_APPLICATION_COLLECTION,
        tft_analyzers.ANALYZER_COLLECTION
    ]
    removed_collections = []
    for collection_name in collections_blacklist:
      removed_collections.append(
          (collection_name, graph.get_collection(collection_name)))
      graph.clear_collection(collection_name)
    saved_transform_io.write_saved_transform_from_session(
        session, inputs, outputs, saved_model_dir)
    for collection_name, collection in removed_collections:
      graph.get_collection(collection_name).extend(collection)


class AnalyzeDataset(beam.PTransform):
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
    self._preprocessing_fn = preprocessing_fn
    _assert_tensorflow_version()

  def _extract_input_pvalues(self, dataset):
    data, _ = dataset
    return dataset, [data]

  def expand(self, dataset):
    """Analyze the dataset.

    Args:
      dataset: A dataset.

    Returns:
      A TransformFn containing the deferred transform function.
    """

    input_values, input_metadata = dataset
    input_schema = input_metadata.schema

    base_temp_dir = Context.create_base_temp_dir()

    class _ReplaceTensorsWithConstants(beam.PTransform):
      """Bind statistics in a deferred manner.

      This transform fills in analyzer outputs with their actual computed
      values.

      Args:
        saved_model_dir: The directory containing the SavedModel.
      """

      def __init__(self, saved_model_dir):
        # Generally the pipeline is inferred from its inputs, however we need
        # to know the pipeline for beam.Create.
        self.pipeline = input_values.pipeline
        self._saved_model_dir = saved_model_dir

      def expand(self, tensor_pcoll_mapping):
        """Converts a dict of statistics to a transform function.

        Args:
          tensor_pcoll_mapping: A dictionary mapping `Tensor`s to singleton
              `PCollection`s.

        Returns:
          A single-element PCollection containing the directory name with the
              SavedModel.
        """
        transform_fn = (
            self.pipeline | 'CreateTransformFn' >> beam.Create(
                [self._saved_model_dir]))

        if not tensor_pcoll_mapping:
          return transform_fn

        # Convert tensor_value_mapping into a DictPCollectionView so it can be
        # passed as a side input to the beam Map below.
        tensor_value_pairs = []
        for name, pcoll in six.iteritems(tensor_pcoll_mapping):
          tensor_value_pairs.append(
              pcoll
              | 'AddName[%s]' % name
              >> beam.Map(lambda x, name=name: (name, x)))
        tensor_value_mapping = beam.pvalue.AsDict(
            tensor_value_pairs | 'MergeTensorValuePairs' >> beam.Flatten())

        # Run a mapper that inserts statistic values into the graph.  We wrap
        # replace_tensors_with_constant_values in a wrapper that also creates
        # a temp dir.  This makes the wrapper idempotent since any retry will
        # use a different temp dir.
        def replace_tensors_with_constant_values(
            saved_model_dir, tensor_value_mapping, serialized_tf_config):

          tf_config = _maybe_deserialize_tf_config(serialized_tf_config)
          with tf.Session(config=tf_config) as session:
            temp_dir = _make_unique_temp_dir(base_temp_dir)
            input_tensors, output_tensors = (
                saved_transform_io.partially_apply_saved_transform(
                    saved_model_dir, {}, tensor_value_mapping))
            saved_transform_io.write_saved_transform_from_session(
                session, input_tensors, output_tensors, temp_dir)
          return temp_dir

        serialized_tf_config = _DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER.get(
            self.pipeline.runner)
        return (transform_fn |
                'ReplaceTensorsWithConstantValues' >> beam.Map(
                    replace_tensors_with_constant_values,
                    tensor_value_mapping=tensor_value_mapping,
                    serialized_tf_config=serialized_tf_config))

    class _ComputeTensorPcollMappingUpdate(beam.PTransform):
      """Create a mapping from `Tensor`s to PCollections.

      Creates a mapping from `Tensor`s to PCollections for the outputs of the
      new analyzers.  An existing mapping will be provided as the argument
      to the extend() method.

      Args:
        phase: The Phase to run
      """

      def __init__(self, saved_model_dir, analyzer_inputs_schema, analyzers):
        self._saved_model_dir = saved_model_dir
        self._analyzer_inputs_schema = analyzer_inputs_schema
        self._analyzers = analyzers

      def expand(self, input_values_and_tensor_pcoll_mapping):
        input_values, tensor_pcoll_mapping = (
            input_values_and_tensor_pcoll_mapping)

        # Create a transform_fn to produce inputs to new analyzers.
        transform_fn = (
            tensor_pcoll_mapping
            | 'ReplaceTensorsWithConstants'
            >> _ReplaceTensorsWithConstants(self._saved_model_dir))

        # Run the transform_fn.
        serialized_tf_config = _DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER.get(
            self.pipeline.runner)
        analyzer_input_values = (
            input_values | 'ComputeAnalyzerInputs' >> beam.ParDo(
                _RunMetaGraphDoFn(input_schema, self._analyzer_inputs_schema,
                                  serialized_tf_config),
                saved_model_dir=beam.pvalue.AsSingleton(transform_fn)))

        # For each analyzer output, look up its input values (by tensor name)
        # and run the analyzer on these values.
        #
        tensor_pcoll_mapping_update = {}
        for idx, analyzer in enumerate(self._analyzers):
          analyzer_impl = analyzer_impls._impl_for_analyzer(analyzer.spec)
          # pylint: enable=protected-access

          assert len(analyzer.inputs) == 1
          output_pcolls = (
              analyzer_input_values
              | 'Extract_%d' % idx >> beam.Map(
                  lambda batch, key: batch[key],
                  key=analyzer.inputs[0].name)
              | 'Analyze_%d' % idx >> analyzer_impl)
          assert len(analyzer.outputs) == len(output_pcolls)
          for tensor, pcoll in zip(analyzer.outputs, output_pcolls):
            tensor_pcoll_mapping_update[tensor.name] = pcoll
        return tensor_pcoll_mapping_update

    # NOTE: it's important that create_phases is called directly after
    # run_preprocessing_fn, because we later mutate the graph's
    # TABLE_INITIALIZERS collection which would break the logic in
    # create_phases.
    graph, inputs, outputs = impl_helper.run_preprocessing_fn(
        self._preprocessing_fn, input_schema)
    phases = impl_helper.create_phases(graph)

    # Iterate through levels, generating PCollections for columns that are the
    # outputs of `Operations` that are not `MapOperation`s.
    tensor_pcoll_mapping = {}
    table_initializers = graph.get_collection_ref(
        tf.GraphKeys.TABLE_INITIALIZERS)
    original_table_initializers = list(table_initializers)
    del table_initializers[:]

    for level, phase in enumerate(phases):
      analyzer_inputs = {}
      for analyzer in phase.analyzers:
        for input_tensor in analyzer.inputs:
          analyzer_inputs[input_tensor.name] = input_tensor
      analyzer_inputs_schema = impl_helper.infer_feature_schema(
          analyzer_inputs)
      table_initializers.extend(phase.table_initializers)
      saved_model_dir = _make_unique_temp_dir(base_temp_dir)
      _write_saved_transform(graph, inputs, analyzer_inputs, saved_model_dir)

      tensor_pcoll_mapping_update = (
          (input_values, tensor_pcoll_mapping)
          | 'ComputeTensorPcollMappingUpdate_%d' % level
          >> _ComputeTensorPcollMappingUpdate(
              saved_model_dir, analyzer_inputs_schema, phase.analyzers))
      tensor_pcoll_mapping.update(tensor_pcoll_mapping_update)

    output_metadata = dataset_metadata.DatasetMetadata(
        schema=impl_helper.infer_feature_schema(outputs))
    del table_initializers[:]
    table_initializers.extend(original_table_initializers)
    saved_model_dir = _make_unique_temp_dir(base_temp_dir)
    _write_saved_transform(graph, inputs, outputs, saved_model_dir)
    transform_fn = (
        tensor_pcoll_mapping
        | 'ReplaceTensorsWithConstants'
        >> _ReplaceTensorsWithConstants(saved_model_dir))

    return transform_fn, output_metadata


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
        strings to `Tensor` or `SparseTensor`s
  """

  def __init__(self, preprocessing_fn):
    self._preprocessing_fn = preprocessing_fn
    _assert_tensorflow_version()

  def _extract_input_pvalues(self, dataset):
    data, _ = dataset
    return dataset, [data]

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
    transform_fn = dataset | AnalyzeDataset(self._preprocessing_fn)
    transformed_dataset = (dataset, transform_fn) | TransformDataset()
    return transformed_dataset, transform_fn


class TransformDataset(beam.PTransform):
  """Applies the transformation computed by transforming a Dataset.

  TransformDataset's `expand` method is called on a (dataset, transform_fn)
  pair. It applies the transform_fn to each row of the input dataset and
  returns the resulting dataset.

  args:
    exclude_outputs: Output features that should not be produced.
  """

  def __init__(self, exclude_outputs=None):
    _assert_tensorflow_version()
    self._exclude_outputs = exclude_outputs

  def _extract_input_pvalues(self, dataset):
    (data, _), (transform_fn, _) = dataset
    return dataset, [data, transform_fn]

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

    # If exclude_outputs is set, update the output metadata, which will also
    # cause _RunMetaGraphDoFn not to create the excluded outputs.
    if self._exclude_outputs is not None:
      schema = output_metadata.schema
      output_metadata = dataset_metadata.DatasetMetadata(
          schema=dataset_schema.Schema(
              {key: column_schema
               for key, column_schema in six.iteritems(schema.column_schemas)
               if key not in self._exclude_outputs}))

    def convert_and_unbatch(batch_dict):
      return impl_helper.to_instance_dicts(
          impl_helper.make_output_dict(output_metadata.schema, batch_dict))

    serialized_tf_config = _DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER.get(
        self.pipeline.runner)
    output_instances = (
        input_values
        | 'Transform' >> beam.ParDo(
            _RunMetaGraphDoFn(
                input_metadata.schema,
                output_metadata.schema,
                serialized_tf_config,
                exclude_outputs=self._exclude_outputs),
            saved_model_dir=beam.pvalue.AsSingleton(transform_fn))
        | 'ConvertAndUnbatch' >> beam.FlatMap(convert_and_unbatch))
    return (output_instances, output_metadata)
