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
"""Beam implementations of tf.Transform canonical analyzers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import apache_beam as beam

from apache_beam.typehints import List
from apache_beam.typehints import with_input_types

import numpy as np
from tensorflow_transform import analyzers
from tensorflow_transform.beam import common


@with_input_types(List[np.ndarray])
class _AnalyzerImpl(beam.PTransform):
  """PTransform that implements a given analyzer.

  _AnalyzerImpl accepts a PCollection where each element is a list of
  `ndarray`s. Each element in this list contains a batch of values for the
  corresponding input tensor of the analyzer. _AnalyzerImpl returns a tuple of
  `PCollection`s each containing a single element which is an `ndarray`.

  _AnalyzerImpl dispatches to an implementation transform, with the same
  signature as _AnalyzerImpl.
  """

  def __init__(self, spec, temp_assets_dir):
    self._spec = spec
    self._temp_assets_dir = temp_assets_dir

  def expand(self, pcoll):
    # pylint: disable=protected-access
    if isinstance(self._spec, analyzers._VocabularySpec):
      return pcoll | _VocabularyAnalyzerImpl(self._spec, self._temp_assets_dir)
    elif isinstance(self._spec, analyzers.CombinerSpec):
      return pcoll | _CombinerAnalyzerImpl(self._spec)
    elif isinstance(self._spec, analyzers._CombinePerKeySpec):
      return pcoll | _CombinePerKeyAnalyzerImpl(self._spec)
    else:
      raise NotImplementedError(self._spec.__class__)


class _OrderElementsFn(beam.DoFn):
  """Sort the vocabulary by descending frequency count."""

  def __init__(self, store_frequency):
    self._store_frequency = store_frequency

    # Metrics.
    self._vocab_size_distribution = beam.metrics.Metrics.distribution(
        common.METRICS_NAMESPACE, 'vocabulary_size')

  def process(self, element, counts_iter):
    del element
    counts = list(counts_iter)
    self._vocab_size_distribution.update(len(counts))

    if not counts:
      counts = [(1, '49d0cd50-04bb-48c0-bc6f-5b575dce351a')]

    counts.sort(reverse=True)  # Largest first.
    for count, entry in counts:
      if self._store_frequency:
        yield '{} {}'.format(count, entry)
      else:
        yield entry


@with_input_types(List[np.ndarray])
class _VocabularyAnalyzerImpl(beam.PTransform):
  """Saves the unique elements in a PCollection of batches."""

  def __init__(self, spec, temp_assets_dir):
    assert isinstance(spec, analyzers._VocabularySpec)  # pylint: disable=protected-access
    self._spec = spec
    self._temp_assets_dir = temp_assets_dir

  def expand(self, pcoll):
    top_k = self._spec.top_k
    frequency_threshold = self._spec.frequency_threshold
    assert top_k is None or top_k >= 0
    assert frequency_threshold is None or frequency_threshold >= 0

    # Create a PCollection of (count, element) pairs, then iterates over
    # this to create a single element PCollection containing this list of
    # pairs in sorted order by decreasing counts (and by values for equal
    # counts).

    def flatten_value_and_weights_to_list_of_tuples(batch_values):
      """Converts a batch of vocabulary and weights to a list of KV tuples."""
      # Ravel for flattening and tolist so that we go to native Python types
      # for more efficient followup processing.
      #
      batch_value, weights = batch_values
      batch_value = batch_value.ravel().tolist()
      weights = weights.ravel().tolist()
      if len(batch_value) != len(weights):
        raise ValueError(
            'Values and weights contained different number of values ({} vs {})'
            .format(len(batch_value), len(weights)))
      return zip(batch_value, weights)

    def flatten_value_to_list(batch_values):
      """Converts an N-D dense or sparse batch to a 1-D list."""
      # Ravel for flattening and tolist so that we go to native Python types
      # for more efficient followup processing.
      #
      batch_value, = batch_values
      return batch_value.ravel().tolist()

    if self._spec.has_weights:
      flatten_map_fn = flatten_value_and_weights_to_list_of_tuples
      combine_transform = beam.CombinePerKey(sum)
    else:
      flatten_map_fn = flatten_value_to_list
      combine_transform = beam.combiners.Count.PerElement()

    def is_problematic_string(kv):
      string, _ = kv  # Ignore counts.
      return string and '\n' not in string and '\r' not in string

    counts = (
        pcoll
        | 'FlattenStringsAndMaybeWeights' >> beam.FlatMap(flatten_map_fn)
        | 'CountPerString' >> combine_transform
        | 'FilterProblematicStrings' >> beam.Filter(is_problematic_string)
        | 'SwapStringsAndCounts' >> beam.KvSwap())

    # Filter is cheaper than TopK computation and the two commute, so
    # filter first.
    if frequency_threshold is not None:
      counts |= ('FilterByFrequencyThreshold(%s)' % frequency_threshold >>
                 beam.Filter(lambda kv: kv[0] >= frequency_threshold))

    if top_k is None:
      # Performance optimization to obviate reading from finely sharded files
      # via AsIter in order_elements below. By breaking fusion, we allow sharded
      # files' sizes to be automatically computed (when possible), so we end up
      # reading from fewer and larger files. This is not needed when top_k is
      # provided since that already induces a single-sharded output (due to the
      # CombineGlobaly).
      counts |= 'Reshard' >> beam.transforms.Reshuffle()  # pylint: disable=no-value-for-parameter
    else:
      counts = (counts
                | 'Top(%s)' % top_k
                # Using without_defaults() below since it obviates unnecessary
                # materializations. This is worth doing because:
                # a) Some vocabs could be really large and allthough they do
                #    fit in memory they might go over per-record
                #    materialization limits (TopCombineFn is producing
                #    single-record with the entire vocabulary as a list).
                # b) More fusion leads to increased performance in general.
                >> beam.CombineGlobally(
                    beam.combiners.TopCombineFn(top_k)).without_defaults()
                | 'FlattenList' >> beam.FlatMap(lambda lst: lst))

    vocabulary_file = os.path.join(self._temp_assets_dir,
                                   self._spec.vocab_filename)
    vocab_is_written = (
        pcoll.pipeline
        | 'Prepare' >> beam.Create([None])
        | 'OrderElements' >> beam.ParDo(
            _OrderElementsFn(self._spec.store_frequency),
            # Using AsIter instead of AsList at the callsite below in order to
            # reduce max memory usage.
            counts_iter=beam.pvalue.AsIter(counts))
        | 'WriteToFile' >> beam.io.WriteToText(vocabulary_file,
                                               shard_name_template=''))
    # Return the vocabulary path.
    wait_for_vocabulary_transform = (
        pcoll.pipeline
        | 'CreatePath' >> beam.Create([np.array(vocabulary_file)])
        # Ensure that the analysis returns only after the file is written.
        | 'WaitForVocabularyFile' >> beam.Map(
            lambda x, y: x, y=beam.pvalue.AsIter(vocab_is_written)))
    return (wait_for_vocabulary_transform,)


@with_input_types(List[np.ndarray])
class _CombineFnWrapper(beam.CombineFn):
  """Class to wrap a analyzers._CombinerSpec as a beam.CombineFn."""

  def __init__(self, spec, serialized_tf_config):
    if isinstance(spec, analyzers._QuantilesCombinerSpec):  # pylint: disable=protected-access
      spec.initialize_local_state(
          common._maybe_deserialize_tf_config(serialized_tf_config))  # pylint: disable=protected-access
    self._spec = spec
    self._serialized_tf_config = serialized_tf_config

  def __reduce__(self):
    return _CombineFnWrapper, (self._spec, self._serialized_tf_config)

  def create_accumulator(self):
    return self._spec.create_accumulator()

  def add_input(self, accumulator, next_input):
    return self._spec.add_input(accumulator, next_input)

  def merge_accumulators(self, accumulators):
    return self._spec.merge_accumulators(accumulators)

  def extract_output(self, accumulator):
    return self._spec.extract_output(accumulator)


def _split_inputs_by_key(batch_values):
  """Takes inputs where first input is a key, and returns (key, value) pairs.

  Takes inputs of the form (key, arg0, ..., arg{N-1}) where `key` is a vector
  and arg0, ..., arg{N-1} have dimension >1 with size in the first dimension
  matching `key`.

  It yields pairs of the form

  (key[i], [arg0[i], ..., arg{N-1}[i]])

  for 0 < i < len(key).

  Args:
    batch_values: A list of ndarrays representing the input from a batch.

  Yields:
    (key, args) pairs where key is a string and args is a list of ndarrays.

  Raises:
    ValueError: if inputs do not have correct sizes.
  """
  keys = batch_values[0]
  if keys.ndim != 1:
    raise ValueError(
        'keys for CombinePerKey should have rank 1, got shape {}'.format(
            keys.shape))
  for arg_index, arg_values in enumerate(batch_values[1:]):
    if arg_values.ndim < 1:
      raise ValueError(
          'Argument {} for CombinePerKey should have rank >=1, '
          'got shape {}'.format(arg_index, arg_values.shape))
    if arg_values.shape[0] != keys.shape[0]:
      raise ValueError(
          'Argument {} had shape {} whose first dimension was not equal to the '
          'size of the keys vector ({})'.format(
              arg_index, arg_values.shape, keys.shape[0]))

  for instance_index, key in enumerate(keys):
    instance_args = [arg_values[instance_index]
                     for arg_values in batch_values[1:]]
    yield (key, instance_args)


def _merge_outputs_by_key(keys_and_outputs, num_outputs):
  """Merge outputs of analyzers per key into a single output.

  Takes a list of elements of the form (key, [output0, ..., output{N-1}]) and
  returns a list of ndarrays of the form [keys, outputs0, ..., outputs[{N-1}]]
  where keys is formed by stacking the values of `key` from the list and
  similarly outputs{k} is formed by stacking the individual elements of
  output{k} from the list.

  For each k, output{k} must be an ndarray whose size is the same for each
  element of the list.

  Args:
    keys_and_outputs: a list of elements of the form
      (key, [output0, ..., output{N-1}])
    num_outputs: The number of expected outputs.

  Yields:
    The `TaggedOutput`s: keys, outputs0, ..., outputs[{N-1}]

  Raises:
    ValueError: If the number is outputs doesn't match num_outputs.
  """
  # Sort keys_and_outputs by keys.
  keys_and_outputs.sort(key=lambda x: x[0])
  # Convert from a list of pairs of the form (key, outputs_for_key) to a list of
  # keys and a list of outputs (where the outer dimension is the number of
  # outputs not the number of keys).
  key, outputs = zip(*keys_and_outputs)
  outputs = zip(*outputs)
  # key is a list of scalars so we use np.stack to convert a single array.
  yield beam.pvalue.TaggedOutput('key', np.stack(key, axis=0))
  if len(outputs) != num_outputs:
    raise ValueError(
        'Analyzer has {} outputs but its implementation produced {} '
        'values'.format(num_outputs, len(outputs)))
  for i, output in enumerate(outputs):
    yield beam.pvalue.TaggedOutput(str(i), np.stack(output, axis=0))


@with_input_types(List[np.ndarray])
class _CombinerAnalyzerImpl(beam.PTransform):
  """Implement an analyzer based on a CombinerSpec."""

  def __init__(self, spec):
    self._spec = spec

  def expand(self, pcoll):
    serialized_tf_config = None
    # NOTE: Currently, all combiner specs except _QuantilesCombinerSpec
    # require .with_defaults(False) to be set.
    has_defaults = False

    if isinstance(self._spec, analyzers._QuantilesCombinerSpec):  # pylint: disable=protected-access
      serialized_tf_config = common._DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER.get(  # pylint: disable=protected-access
          pcoll.pipeline.runner)
      has_defaults = True

    def extract_outputs(outputs, num_outputs):
      if len(outputs) != num_outputs:
        raise ValueError(
            'Analyzer has {} outputs but its implementation produced {} '
            'values'.format(num_outputs, len(outputs)))
      for i, output in enumerate(outputs):
        yield beam.pvalue.TaggedOutput(str(i), output)

    output_keys = [str(i) for i in range(self._spec.num_outputs())]
    outputs_tuple = (
        pcoll
        | 'CombineGlobally' >> beam.CombineGlobally(_CombineFnWrapper(
            self._spec, serialized_tf_config)).with_defaults(has_defaults)
        | 'ExtractOutputs'
        >> beam.FlatMap(extract_outputs, self._spec.num_outputs()).with_outputs(
            *output_keys))
    return tuple(outputs_tuple[key] for key in output_keys)


@with_input_types(List[np.ndarray])
class _CombinePerKeyAnalyzerImpl(beam.PTransform):
  """Implement an analyzer based on a _CombinePerKeySpec."""

  def __init__(self, spec):
    self._spec = spec.combiner_spec

  def expand(self, pcoll):
    serialized_tf_config = None

    if isinstance(self._spec, analyzers._QuantilesCombinerSpec):  # pylint: disable=protected-access
      serialized_tf_config = common._DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER.get(  # pylint: disable=protected-access
          pcoll.pipeline.runner)

    output_keys = ['key'] + [str(i) for i in range(self._spec.num_outputs())]
    outputs_tuple = (
        pcoll
        | 'SplitByKey' >> beam.FlatMap(_split_inputs_by_key)
        | 'CombinePerKey' >> beam.CombinePerKey(_CombineFnWrapper(
            self._spec, serialized_tf_config))
        | 'ToList' >> beam.combiners.ToList()
        | 'MergeByKey' >> beam.FlatMap(
            _merge_outputs_by_key,
            self._spec.num_outputs()).with_outputs(*output_keys))
    return tuple(outputs_tuple[key] for key in output_keys)


