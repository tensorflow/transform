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
  input = p | beam_impl.read_examples(..., schema)
  transformed, transform_fn = ((input, schema)
      | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))
  transformed | beam_impl.write_examples_and_metadata(
      examples_path, metadata_path)
  transform_fn | beam_impl.write_transform_fn(transform_fn_path)

"""

import collections
import os
import threading


import apache_beam as beam

from apache_beam.typehints import Dict
from apache_beam.typehints import List
from apache_beam.typehints import Union
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types

import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import impl_helper
import tensorflow_transform.api as api
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

_DEFAULT_DESIRED_BATCH_SIZE = 1000

_PRIMITIVE_TYPE = Union[float, int, long, str, unicode]


class _BatchDoFn(beam.DoFn):
  """Concatenates individual instances into batches.

  DoFn that takes instances and creates batches of size up to
  `desired_batch_size`.

  Args:
    desired_batch_size: The desired number of instances to convert into a batch.

  Yields:
    A list of size up to `desired_batch_size` representing a batch of instances.
  """

  def __init__(self, desired_batch_size=_DEFAULT_DESIRED_BATCH_SIZE):
    self._desired_batch_size = desired_batch_size
    self._batch = []

    # Metrics.
    self._num_batches = beam.metrics.Metrics.counter(self.__class__,
                                                     'num_batches')
    self._batch_size_distribution = beam.metrics.Metrics.distribution(
        self.__class__, 'batch_size_distribution')
    self._num_instances = beam.metrics.Metrics.counter(self.__class__,
                                                       'num_instances')

  def _flush_batch(self):
    self._num_batches.inc(1)
    self._batch_size_distribution.update(len(self._batch))
    self._num_instances.inc(len(self._batch))
    result = self._batch
    self._batch = []
    return result

  def process(self, element):
    self._batch.append(element)
    if len(self._batch) >= self._desired_batch_size:
      yield self._flush_batch()

  def finish_bundle(self):
    if self._batch:
      yield self._flush_batch()


@with_input_types(Dict[str,
                       Union[_PRIMITIVE_TYPE,
                             List[_PRIMITIVE_TYPE],
                             np.generic,
                             np.ndarray]],
                  str)
@with_output_types(List[Dict[str, Union[np.generic, np.ndarray]]])
class _RunMetaGraphDoFn(beam.DoFn):
  """Maps a PCollection of dicts to a PCollection of dicts via a TF graph.

  The TF graph may contain more inputs and outputs than the schema provided.
  In that case, a subset of the graph will be run, which may cause an error if
  the excluded inputs are required to produce the included outputs.

  Args:
    input_schema: A `Schema` representing the inputs of this transform phase.
    output_schema: A `Schema` representing the outputs of this transform phase.
  """

  class _GraphState(object):

    def __init__(self, saved_model_dir, input_schema, output_schema):
      self.saved_model_dir = saved_model_dir
      self.graph = tf.Graph()
      self.session = tf.Session(graph=self.graph)
      with self.graph.as_default():
        inputs, outputs = impl_helper.load_transform_fn_def(saved_model_dir)
        # Run the op that initializes all tables in the graph.
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

  def __init__(self, input_schema, output_schema, exclude_outputs=None):
    super(_RunMetaGraphDoFn, self).__init__()
    self._input_schema = input_schema
    self._output_schema = output_schema
    self._exclude_outputs = exclude_outputs

    # Metrics.
    self._num_graph_loads = beam.metrics.Metrics.counter(self.__class__,
                                                         'num_graph_loads')

  def process(self, element, saved_model_dir):
    """Runs the given graph to realize the output tensors (i.e. features).

    Runs the graph in a TF session for computing the output values of the
    tensors, given an input row of data (input tensors). Due to the record-by
    record nature of beam we are operating sess.run() on individual record
    tensors vs batched tensors.

    Args:
      element: the element being processed by the DoFn
      saved_model_dir: Directory containing saved model.

    Yields:
      A representation of output features as a dict mapping keys (logical column
      names) to values.
    """
    if (not hasattr(self._thread_local, 'graph_state') or
        self._thread_local.graph_state.saved_model_dir != saved_model_dir):
      self._num_graph_loads.inc(1)
      self._thread_local.graph_state = self._GraphState(
          saved_model_dir, self._input_schema, self._output_schema)

    feed_dict = impl_helper.make_feed_dict(
        self._thread_local.graph_state.inputs, self._input_schema, element)
    fetched_dict = self._thread_local.graph_state.session.run(
        self._thread_local.graph_state.outputs, feed_dict=feed_dict)
    yield impl_helper.make_output_dict(self._output_schema, fetched_dict)


def _assert_tensorflow_version():
  try:
    _ = tf.SparseFeature
    _ = tf.tables_initializer
  except AttributeError:
    raise RuntimeError(
        'Tensorflow version 1.0 is required. Please install the latest version '
        'from https://github.com/tensorflow/tensorflow.')


class AnalyzeDataset(beam.PTransform):
  """Takes a preprocessing_fn and computes the relevant statistics.

  AnalyzeDataset accepts a preprocessing_fn in its constructor.  When its
  `expand` method is called on a dataset, it computes all the relevant
  statistics required to run the transformation described by the
  preprocessing_fn, and returns a TransformFn representing the application of
  the preprocessing_fn.

  Args:
    preprocessing_fn: A function that accepts and returns a dictionary from
      strings to `Column`s or `Statistic`s.
    output_dir: Location of the SavedModel that will represent the transform
      function.
  """

  def __init__(self, preprocessing_fn, output_dir):
    self._preprocessing_fn = preprocessing_fn
    self._output_dir = output_dir
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
    input_batches = input_values | 'BatchInstances' >> beam.ParDo(_BatchDoFn())

    class _CreateTransformFn(beam.PTransform):
      """Create a TransformFnDef, binding statistics in a deferred manner.

      This function constructs a tensorflow graph eagerly and then (in a
      deferred manner) fills in analyzer outputs with their actual computed
      values. We construct the tensorflow graph up front because that implies
      serializing MetaGraphDef protos rather than pickling the user-defined TITO
      functions. The graph contains placeholders for `_AnalyzerOutput`s which
      are then replaced with their actual values (as constant tensors) in a
      deferred manner.

      Args:
        input_columns: A map from column names to `Column`s.
        output_columns: A map from column names to `Column`s.
        temp_dir: Temp dir to store `SavedModel`s.
      """

      def __init__(self, input_columns, output_columns, temp_dir):
        # Generally the pipeline is inferred from its inputs, however we need
        # to know the pipeline for beam.Create.
        self.pipeline = input_values.pipeline
        self._input_columns = input_columns
        self._output_columns = output_columns
        self._temp_dir = temp_dir

      def expand(self, analyzer_outputs_to_pcoll):
        """Converts a dict of statistics to a transform function.

        Args:
          analyzer_outputs_to_pcoll: A dictionary mapping `_AnalyzerOutput`s
              to the values of these statistics as a PCollection.

        Returns:
          A single-element PCollection containing the directory name with the
              SavedModel.
        """
        # Create a transform_fn with unbound values.

        unbound_transform_fn_dir = os.path.join(self._temp_dir,
                                                'unbound_transform_fn')
        input_columns_to_statistics = impl_helper.make_transform_fn_def(
            input_schema, self._input_columns, self._output_columns,
            unbound_transform_fn_dir)

        transform_fn = (
            self.pipeline | 'CreateTransformFn' >> beam.Create(
                [unbound_transform_fn_dir]))

        if not analyzer_outputs_to_pcoll:
          return transform_fn

        # Convert the statistics dict into a DictPCollectionView so it can be
        # passed as a side input to the beam Map below.
        tagged_statistics = []
        for tag, statistic in input_columns_to_statistics.items():
          pcoll = analyzer_outputs_to_pcoll[statistic]
          tagged_statistics.append(
              pcoll
              | 'AddTag[%s]' % tag >> beam.Map(lambda x, tag=tag: (tag, x)))

        statistics_side_input = beam.pvalue.AsDict(
            tagged_statistics | 'MergeStatistics' >> beam.Flatten())

        # Run a mapper that inserts statistic values into the graph.
        return (transform_fn |
                'ReplaceTensorsWithConstantValues' >> beam.Map(
                    impl_helper.replace_tensors_with_constant_values,
                    bound_saved_model_dir=os.path.join(self._temp_dir,
                                                       'transform_fn'),
                    input_value_mapping=statistics_side_input))

    inputs, outputs = impl_helper.run_preprocessing_fn(
        self._preprocessing_fn, input_schema)

    # Get a list of lists, containing analyzers (i.e. _AnalyzerOutput objects)
    # by level in the DAG of Columns/Statistics. Analyzers at level n are ready
    # to run once all analyzers at level n - 1 are complete.
    analyzers_by_level = self._analyzers_by_level(outputs)

    # Iterate through levels, keeping track of analyzer outputs (i.e.
    # statistics) via a mapping of `_AnalyzerOutput` -> single element
    # PCollection.
    analyzer_outputs_to_pcoll = {}
    for level, analyzer_outputs in enumerate(analyzers_by_level):
      # Create a TransformFnDef representing the graph needed to generate
      # all the inputs required by the analyzer_outputs at this level.  We
      # assign arbitrary names to the outputs of this TransformFnDef.
      analyzer_input_columns = {}
      for idx, analyzer_output in enumerate(analyzer_outputs):
        if len(analyzer_output.inputs) != 1:
          raise NotImplementedError('Analyzers must have exactly one input')
        analyzer_input_key = 'analyzer_%d_input' % idx
        analyzer_input_columns[analyzer_input_key] = analyzer_output.inputs[0]

      transform_fn = (
          analyzer_outputs_to_pcoll
          | 'CreateTransformFn_%d' % level >> _CreateTransformFn(
              inputs, analyzer_input_columns,
              os.path.join(self._output_dir, 'tmp', 'level_%s' % level)))
      analyzer_input_schema = impl_helper.infer_feature_schema(
          analyzer_input_columns)

      # Run the TransformFnDef in a mapper.
      analysis_inputs = (
          input_batches | 'ComputeAnalyzerInputs_%d' % level >> beam.ParDo(
              _RunMetaGraphDoFn(input_schema, analyzer_input_schema),
              saved_model_dir=beam.pvalue.AsSingleton(transform_fn)))

      # For each analyzer output, look up its input values (by tensor name)
      # and run the analyzer in these values.
      for idx, analyzer_output in enumerate(analyzer_outputs):
        analyzer_input_key = 'analyzer_%d_input' % idx
        analyzer_outputs_to_pcoll[analyzer_output] = (
            analysis_inputs
            | 'Extract_%d_%d' % (level, idx) >> beam.Map(
                # pylint: disable=cell-var-from-loop
                # This lint warning is prone to false positives, and it's not
                # clear why the warning is required here.
                lambda x, key=analyzer_input_key: [inst[key] for inst in x])
            | 'Analyze_%d_%d' % (level, idx) >> self._Analyze(analyzer_output))

    output_metadata = dataset_metadata.DatasetMetadata(
        schema=impl_helper.infer_feature_schema(outputs))
    transform_fn = (
        analyzer_outputs_to_pcoll
        | 'CreateTransformFn' >> _CreateTransformFn(
            inputs, outputs, self._output_dir))

    return transform_fn, output_metadata

  def _analyzers_by_level(self, outputs):
    """Returns a list of lists, containing analyzers by level.

    We need to run analyzers in order so that when running the TF graph to get
    the inputs of an analyzer, we only rely on analyzers that have already been
    computed.

    This is acheived by running analyzers in order of their `level` where the
    level of an analyzer is defined as its depth in the dependency graph of
    analyzers.

    Instead of computing the dependency graph of analyzers explicitly, we
    compute the level of each analyzer by recursively walking the tree of
    Column/Statistics.  For each column/statistic we define its level to be the
    greatest level of an analyzer that it depends on.  From this definition we
    get the following rules for computing the level of a column:
      - An `_AnalyzerOutput` has level one greater than the max of its inputs.
      - A `_TransformedColumn` or `_TransformedStatistic` has level equal to the
        max of its inputs.
      - An `_InputColumn` has level -1 so that the first analyzer ready to run
        has level 0.

    Args:
      outputs: All output columns.

    Returns:
      a list of lists of `_AnalyzerOutput`s ordered by level.
    """
    memoized_column_levels = {}
    analyzers_by_level = collections.defaultdict(list)
    def column_level(column):
      """Adds all analyzers above this column to analyzers_by_level.

      Visits all parents of this column or statistic in the `Column`/`Statistic`
      graph, and adds each _AnalyzerOutput to analyzers_by_level according to
      its level.

      Args:
        column: A Column or Statistic.

      Returns:
        The level of this column.

      Raises:
        ValueError: if the passed column argument is not a `Column`.
      """
      if column in memoized_column_levels:
        return memoized_column_levels[column]

      # pylint: disable=protected-access
      if isinstance(column, api._AnalyzerOutput):
        level = max(
            [column_level(input_column) for input_column in column.inputs]) + 1
        analyzers_by_level[level].append(column)
      elif isinstance(column,
                      (api._TransformedColumn, api._TransformedStatistic)):
        level = max(
            [column_level(input_column) for input_column in column.inputs])
      elif isinstance(column, api._InputColumn):
        level = -1
      else:
        raise ValueError('Not a Column: {}'.format(column))
      # pylint: enable=protected-access

      memoized_column_levels[column] = level
      return level

    # Call column_level for all outputs, which has the side effect of populating
    # analyzers_by_level.
    for column in outputs.values():
      column_level(column)

    # Turn the defaultdict analyzers_by_level into a list.  We know that by
    # construction the set of keys will be of the form 0,1,...,k.
    return [analyzers_by_level[level]
            for level in sorted(analyzers_by_level.keys())]

  class _Analyze(beam.PTransform):

    def __init__(self, analyzer_output):
      self._analyzer_name = analyzer_output.analyzer_name
      self._args_dict = analyzer_output.args_dict
      self._tensor = analyzer_output.tensor

    def expand(self, pcoll):

      def combine_by_batch(fn):
        """Reduces a PCollection of batches according to the given function."""
        return (pcoll
                | 'FlattenValue'  # Flatten N-d values into 1-d.
                >> beam.Map(lambda x: np.array(x).ravel())
                | 'CombineWithinBatch' >> beam.Map(fn)
                | 'CombineGlobally'
                >> beam.CombineGlobally(fn).without_defaults())

      analysis_result = None
      if self._analyzer_name == api.CanonicalAnalyzers.MIN:
        assert not self._args_dict
        analysis_result = combine_by_batch(min)

      elif self._analyzer_name == api.CanonicalAnalyzers.MAX:
        assert not self._args_dict
        analysis_result = combine_by_batch(max)

      elif self._analyzer_name == api.CanonicalAnalyzers.SUM:
        assert not self._args_dict
        analysis_result = combine_by_batch(sum)

      elif self._analyzer_name == api.CanonicalAnalyzers.UNIQUES:
        # Creates a PCollection of (count, element) pairs, then iterates over
        # this to create a single element PCollection containing this list of
        # pairs in sorted order by decreasing counts (and by values for equal
        # counts).

        top_k = self._args_dict['top_k']
        assert top_k is None or top_k >= 0

        frequency_threshold = self._args_dict['frequency_threshold']
        assert frequency_threshold is None or frequency_threshold >= 0

        def to_iterable(instance_value):
          if isinstance(instance_value, (six.string_types, float, int, long)):
            return [instance_value]
          else:
            # Value is a list or ndarray and so is already an iterable.
            return instance_value

        counts = (pcoll
                  | 'Unbatch' >> beam.FlatMap(lambda batch: batch)
                  | 'ExtractElements' >> beam.FlatMap(to_iterable)
                  | 'CountPerElement'
                  >> beam.transforms.combiners.Count.PerElement()
                  | 'SwapElementsAndCounts' >> beam.KvSwap())

        # Filtration is cheaper than TopK computation and the two commute, so do
        # filtration first.
        if frequency_threshold is not None:
          counts |= ('FilterByFrequencyThreshold(%s)' % frequency_threshold >>
                     beam.Filter(lambda kv: kv[0] >= frequency_threshold))

        if top_k is not None:
          counts = (counts
                    | 'Top(%s)' % top_k
                    >> beam.transforms.combiners.Top.Largest(top_k)
                    | 'FlattenList' >> beam.FlatMap(lambda lst: lst))

        # Performance optimization to obviate reading from finely sharded files
        # via AsIter. By forcing all data into a single group we end up reading
        # from a single file.
        #
        @beam.ptransform_fn
        def Reshard(pcoll):  # pylint: disable=invalid-name
          return (
              pcoll
              | 'PairWithNone' >> beam.Map(lambda x: (None, x))
              | 'GroupByNone' >> beam.GroupByKey()
              | 'ExtractValues' >> beam.FlatMap(lambda x: x[1]))
        counts |= 'ReshardToOneGroup' >> Reshard()  # pylint: disable=no-value-for-parameter

        # Using AsIter instead of AsList below in order to reduce max memory
        # usage (due to AsList caching).
        def order_by_decreasing_counts(_, counts_iter):
          counts = list(counts_iter)
          counts.sort(reverse=True)  # Largest first.
          return [element for _, element in counts]

        analysis_result = (pcoll.pipeline
                           | 'Prepare' >> beam.Create([None])
                           | 'OrderByDecreasingCounts' >> beam.Map(
                               order_by_decreasing_counts,
                               counts_iter=beam.pvalue.AsIter(counts)))
      else:
        raise NotImplementedError(self._analyzer_name)

      # Note we pass in dtype as string and shape as a tuple, to avoid pickling
      # issues (b/35133536)
      return (analysis_result
              | 'ConstantTensorValue' >> beam.Map(
                  impl_helper.ConstantTensorValue,
                  dtype=self._tensor.dtype.name,
                  shape=tuple(dim.value for dim in self._tensor.get_shape())))


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
        strings to `Column`s or `Statistic`s
    output_dir: Location of the SavedModel that will represent the transform
      function.
  """

  def __init__(self, preprocessing_fn, output_dir):
    self._preprocessing_fn = preprocessing_fn
    self._output_dir = output_dir
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
    transform_fn = dataset | AnalyzeDataset(self._preprocessing_fn,
                                            self._output_dir)
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
               for key, column_schema in schema.column_schemas.items()
               if key not in self._exclude_outputs}))

    output_instances = (
        input_values
        | 'BatchInstances' >> beam.ParDo(_BatchDoFn())
        | 'TransformBatches' >> beam.ParDo(
            _RunMetaGraphDoFn(input_metadata.schema,
                              output_metadata.schema,
                              self._exclude_outputs),
            saved_model_dir=beam.pvalue.AsSingleton(transform_fn))
        | 'Unbatch' >> beam.FlatMap(lambda batch: batch))
    return (output_instances, output_metadata)
