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
"""An implementation of go/tf-transform using Beam.
"""
import collections


import apache_beam as beam

from apache_beam.typehints import Dict
from apache_beam.typehints import List
from apache_beam.typehints import Union
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types

import tensorflow as tf
from tensorflow_transform import impl_helper
import tensorflow_transform.api as api


# TODO(kestert): Update these types,
@with_input_types(Dict[str, Union[float, int, long, str]],
                  impl_helper.TransformFnDef)
@with_output_types(List[Dict[str, Union[float, int, long, str]]])
class _RunMetaGraphDoFn(beam.DoFn):
  """Maps a PCollection of dicts to a PCollection of dicts via a TF graph.

  Args:
    input_schema: A map from column names to `FixedLenFeature`, `VarLenFeature`
      or `SparseFeature` objects representing the inputs of this transform
      phase.
    output_schema: A map from column names to `FixedLenFeature`, `VarLenFeature`
      or `SparseFeature` objects representing the outputs of this transform
      phase.
  """

  def _serialize(self, schema):
    """Serialize an input or output schema dictionary so it can be pickled.

    Args:
      schema: A map from column names to `FixedLenFeature`, `VarLenFeature`
      or `SparseFeature`.
    Returns:
      A map from column names to a dictionary representing the feature spec.
    """
    def encode(feature_spec):
      result = feature_spec._asdict()
      result['__type__'] = type(feature_spec).__name__
      result['dtype'] = result['dtype'].name
      return result

    return {name: encode(feature_spec)
            for name, feature_spec in schema.items()}

  def _deserialize(self, schema):
    """Deserialize into an input or output schema dictionary.

    Args:
      schema: A map from column names to a dictionary representing the feature
        spec.
    Returns:
      A map from column names to `FixedLenFeature`, `VarLenFeature` or
        `SparseFeature`.

    """
    def decode(feature_spec_dict):
      feature_spec_name = feature_spec_dict.pop('__type__')
      feature_spec_dict['dtype'] = tf.as_dtype(feature_spec_dict['dtype'])
      feature_spec_class = getattr(tf, feature_spec_name)
      return feature_spec_class(**feature_spec_dict)

    return {name: decode(feature_spec)
            for name, feature_spec in schema.items()}

  def __init__(self, input_schema, output_schema):
    super(_RunMetaGraphDoFn, self).__init__()
    self._transform_fn_def = None
    self._graph = None
    self._session = None
    self._inputs = None
    self._outputs = None
    # We need to serialize the input and output schema here since we cannot use
    # a coder directly due to b/34628545.
    self._serialized_input_schema = self._serialize(input_schema)
    self._serialized_output_schema = self._serialize(output_schema)

  def _initialize_graph(self, transform_fn_def):
    self._transform_fn_def = transform_fn_def
    if self._session is not None:
      self._session.close()
    self._graph = tf.Graph()
    self._session = tf.Session(graph=self._graph)
    with self._graph.as_default():
      inputs, outputs = impl_helper.load_transform_fn_def(
          transform_fn_def)
    self._inputs = inputs
    self._outputs = outputs
    self._input_schema = self._deserialize(self._serialized_input_schema)
    self._output_schema = self._deserialize(self._serialized_output_schema)

  def process(self, context, transform_fn_def):
    """Runs the given graph to realize the output tensors (i.e. features).

    Runs the graph in a TF session for computing the output values of the
    tensors, given an input row of data (input tensors). Due to the record-by
    record nature of beam we are operating sess.run() on individual record
    tensors vs batched tensors.

    Args:
      context: a DoFnContext object
      transform_fn_def: A TransformFnDef containing a description of the
          graph to be run.

    Yields:
      A representation of output features as a dict mapping keys (logical column
      names) to values.
    """
    if transform_fn_def != self._transform_fn_def:
      self._initialize_graph(transform_fn_def)
    feed_dict = impl_helper.make_feed_dict(self._inputs, self._input_schema,
                                           context.element)
    fetched_dict = self._session.run(self._outputs, feed_dict=feed_dict)
    yield impl_helper.make_output_dict(self._output_schema, fetched_dict)

  def finish_bundle(self, context):
    self._transform_fn_def = None
    if self._session is not None:
      self._session.close()


def _assert_tensorflow_version():
  try:
    _ = tf.SparseFeature
  except AttributeError:
    raise RuntimeError(
        'Tensorflow version 1.0 is required. Please install the latest version '
        'from https://github.com/tensorflow/tensorflow.')


class AnalyzeDataset(api.AnalyzeDataset, beam.PTransform):
  """Takes a preprocessing_fn and computes the relevant statistics."""

  def __init__(self, preprocessing_fn):
    super(AnalyzeDataset, self).__init__(preprocessing_fn)
    _assert_tensorflow_version()

  def _extract_input_pvalues(self, dataset):
    data, _ = dataset
    return dataset, [data]

  def __ror__(self, dataset, label=None):
    return beam.PTransform.__ror__(self, dataset, label)

  def expand(self, dataset):
    # TODO(rajivpb) Perform some validation of keys subject to b/33456712.

    input_values, input_schema = dataset

    class _CreateTransformFn(beam.PTransform):
      """Create a TransformFnDef, binding statistics in a deferred manner.

      Args:
        graph: The tensorflow graph representing the transform function.
        input_tensors: A map from input column names to tensors in the graph.
        output_tensors: A map from output column names to tensors in the graph.
      """

      def __init__(self, graph, input_tensors, output_tensors):
        # Generally the pipeline is inferred from its inputs, however we need
        # to know the pipeline for beam.Create.
        self.pipeline = input_values.pipeline
        self._graph = graph
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors

      def expand(self, statistic_placeholders_to_pcoll):
        """Converts a dict of statistics to a transform function.

        Args:
          statistic_placeholders_to_pcoll: A dictionary mapping the names of
              placeholder tensors for statistics, to the values of these
              statistics as a PCollection.

        Returns:
          A single-element PCollection wrapping a TransformFnDef that
              represents the transform function with these statistics bound
              to constants.
        """
        # Create a transform_fn with unbound values.
        # TODO(kestert): Ensure stage names follow beam conventions.
        transform_fn = (
            self.pipeline | 'CreateTransformFn' >> beam.Create([
                impl_helper.make_transform_fn_def(
                    self._graph, self._input_tensors, self._output_tensors)
            ]))

        if not statistic_placeholders_to_pcoll:
          return transform_fn

        # Convert the statistics dict into a DictPCollectionView so it can be
        # passed as a side input to the beam Map below.
        tagged_statistics = [
            pc | 'AddTag[%s]' % tag >> beam.Map(lambda x, tag=tag: (tag, x))
            for tag, pc in statistic_placeholders_to_pcoll.items()
        ]
        statistics_side_input = beam.pvalue.AsDict(
            tagged_statistics | beam.Flatten())

        # Run a mapper that inserts statistic values into the graph.
        return transform_fn | beam.Map(
            impl_helper.replace_tensors_with_constant_values,
            tensor_value_mapping=statistics_side_input)

    graph = tf.Graph()

    inputs, outputs = impl_helper.run_preprocessing_fn(
        self._preprocessing_fn, input_schema, graph)

    input_tensors = {key: col.placeholder for (key, col) in inputs.items()}
    output_tensors = {key: col.tensor for (key, col) in outputs.items()}

    # Get a list of lists, containing analyzers (i.e. _AnalyzerOutput objects)
    # by level in the DAG of Columns/Statistics. Analyzers at level n are ready
    # to run once all analyzers at level n - 1 are complete.
    analyzers_by_level = self._analyzers_by_level(outputs)

    # Iterate through levels, keeping track of analyzer outputs (i.e.
    # statistics) via a mapping of placeholder tensor name ->
    # single element PCollection.
    statistic_placeholders_to_pcoll = {}
    for level, analyzer_outputs in enumerate(analyzers_by_level):
      # Create a TransformFnDef representing the graph needed to generate
      # all the inputs required by the analyzer_outputs at this level.
      analyzer_input_tensors = {}
      for analyzer_output in analyzer_outputs:
        for input_column in analyzer_output.inputs:
          analyzer_input_tensors[input_column.tensor.name] = input_column.tensor

      transform_fn = (
          statistic_placeholders_to_pcoll
          | 'CreateTransformFn_%d' % level >> _CreateTransformFn(
              graph, input_tensors, analyzer_input_tensors))
      analyzer_input_schema = impl_helper.infer_feature_schema(
          analyzer_input_tensors)

      # Run the TransformFnDef in a mapper.
      analysis_inputs = (
          input_values | 'Compute_Analyzer_Inputs_%d' % level >> beam.ParDo(
              _RunMetaGraphDoFn(input_schema, analyzer_input_schema),
              transform_fn_def=beam.pvalue.AsSingleton(transform_fn)))

      # For each analyzer output, look up its input values (by tensor name)
      # and run the analyzer in these values.
      for ix, analyzer_output in enumerate(analyzer_outputs):
        # TODO(rajivpb): Support multi-column analysis. Just one for now.
        if len(analyzer_output.inputs) != 1:
          raise NotImplementedError('Analyzers must have exactly one input')
        in_tensor = analyzer_output.inputs[0].tensor
        analyzer_ptransform = self._create_analyzer_ptransform(
            analyzer_output.analyzer_name, analyzer_output.args_dict)
        statistic_placeholders_to_pcoll[analyzer_output.tensor.name] = (
            analysis_inputs
            | 'Extract_%d_%d' % (level, ix) >> beam.Map(
                # pylint: disable=cell-var-from-loop
                # This lint warning is prone to false positives, and it's not
                # clear why the warning is required here.
                lambda x, key=in_tensor.name: x[key])
            | 'Analyze_%d_%d' % (level, ix) >> analyzer_ptransform)

    output_schema = impl_helper.infer_feature_schema(output_tensors)
    transform_fn = (
        statistic_placeholders_to_pcoll
        | 'CreateTransformFn' >> _CreateTransformFn(
            graph, input_tensors, output_tensors))

    # TODO(kestert): Ensure metadata attached to columns is in the "schema".
    return transform_fn, output_schema

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
      - A `_TransformedColumn` has level equal to the max of its inputs.
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
      """
      if column in memoized_column_levels:
        return memoized_column_levels[column]

      # pylint: disable=protected-access
      if isinstance(column, api._AnalyzerOutput):
        level = max(
            [column_level(input_column) for input_column in column.inputs]) + 1
        analyzers_by_level[level].append(column)
      elif isinstance(column, api._TransformedColumn):
        level = max(
            [column_level(input_column) for input_column in column.inputs])
      elif isinstance(column, api._InputColumn):
        level = -1
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

  # TODO(rajivpb): This method is a stop-gap for a specific set of analyzers
  # until we support user-defined/arbitrary analyzers (exact implementation
  # details of which is TBD).
  def _create_analyzer_ptransform(self, analyzer_name, args_dict):
    if analyzer_name == api.CanonicalAnalyzers.MIN:
      assert not args_dict
      return beam.CombineGlobally(min).without_defaults()
    elif analyzer_name == api.CanonicalAnalyzers.MAX:
      assert not args_dict
      return beam.CombineGlobally(max).without_defaults()
    raise NotImplementedError(analyzer_name)


class AnalyzeAndTransformDataset(api.AnalyzeAndTransformDataset,
                                 beam.PTransform):
  """Maps data to features via a preprocessing_fn."""

  def __init__(self, preprocessing_fn):
    super(AnalyzeAndTransformDataset, self).__init__(preprocessing_fn)
    _assert_tensorflow_version()

  def _extract_input_pvalues(self, dataset):
    data, _ = dataset
    return dataset, [data]

  def __ror__(self, dataset, label=None):
    return beam.PTransform.__ror__(self, dataset, label)

  def expand(self, dataset):
    # Expand is currently implemented by composing AnalyzeDataset and
    # TransformDataset.  Future versions however could do somthing more optimal,
    # e.g. caching the values of expensive computations done in AnalyzeDataset.
    transform_fn = dataset | AnalyzeDataset(self.preprocessing_fn)
    transformed_dataset = (dataset, transform_fn) | TransformDataset()
    return transformed_dataset, transform_fn


class TransformDataset(api.TransformDataset, beam.PTransform):
  """Maps data to features via a transform_fn."""

  def __init__(self):
    super(TransformDataset, self).__init__()
    _assert_tensorflow_version()

  def _extract_input_pvalues(self, dataset):
    (data, _), (transform_fn, _) = dataset
    return dataset, [data, transform_fn]

  def __ror__(self, dataset, label=None):
    return beam.PTransform.__ror__(self, dataset, label)

  def expand(self, dataset_and_transform_fn):
    (input_values, input_schema), (transform_fn, output_schema) = (
        dataset_and_transform_fn)
    output_values = input_values | 'MapInstances' >> beam.ParDo(
        _RunMetaGraphDoFn(input_schema, output_schema),
        transform_fn_def=beam.pvalue.AsSingleton(transform_fn))
    return (output_values, output_schema)
