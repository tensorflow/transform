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

import collections
import os


import apache_beam as beam

from apache_beam.transforms.ptransform import ptransform_fn
from apache_beam.typehints import KV
from apache_beam.typehints import Tuple
from apache_beam.typehints import with_input_types

import numpy as np
from tensorflow_transform import analyzers
from tensorflow_transform import attributes_classes
from tensorflow_transform import tf_utils
from tensorflow_transform.beam import common


class _OrderElementsFn(beam.DoFn):
  """Sort the vocabulary by descending frequency count."""

  def __init__(self, store_frequency):
    self._store_frequency = store_frequency

    # Metrics.
    self._vocab_size = beam.metrics.Metrics.distribution(
        common.METRICS_NAMESPACE, 'vocabulary_size')

  def process(self, element, counts_iter):
    del element
    counts = list(counts_iter)
    self._vocab_size.update(len(counts))

    if not counts:
      counts = [(1, '49d0cd50-04bb-48c0-bc6f-5b575dce351a')]

    counts.sort(reverse=True)  # Largest first.
    for count, entry in counts:
      if self._store_frequency:
        yield '{} {}'.format(count, entry)
      else:
        yield entry


@ptransform_fn
@beam.typehints.with_input_types(KV[float, str])
@beam.typehints.with_output_types(KV[float, str])
def _ApplyFrequencyThresholdAndTopK(counts,  # pylint: disable=invalid-name
                                    frequency_threshold,
                                    top_k
                                   ):
  """Applies `frequency_threshold` and `top_k` to (count, value) pairs."""
  # Filter is cheaper than TopK computation and the two commute, so filter
  # first.
  if frequency_threshold is not None:
    counts |= ('FilterByFrequencyThreshold(%s)' % frequency_threshold >>
               beam.Filter(lambda kv: kv[0] >= frequency_threshold))

  if top_k is None:
    # Performance optimization to obviate reading from finely sharded files via
    # AsIter in order_elements below. By breaking fusion, we allow sharded
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
              # a) Some vocabs could be really large and allthough they do fit
              #    in memory they might go over per-record materialization
              #    limits (TopCombineFn is producing single-record with the
              #    entire vocabulary as a list).
              # b) More fusion leads to increased performance in general.
              >> beam.CombineGlobally(
                  beam.combiners.TopCombineFn(top_k)).without_defaults()
              | 'FlattenList' >> beam.FlatMap(lambda lst: lst))
  return counts


@ptransform_fn
@beam.typehints.with_input_types(KV[float, str])
@beam.typehints.with_output_types(str)
def _WriteVocabFile(counts, temp_assets_dir, vocab_filename, store_frequency):  # pylint: disable=invalid-name
  """Writes a vocab file from a pcoll of (weighted_count, value) pairs."""
  vocabulary_file = os.path.join(temp_assets_dir, vocab_filename)
  vocab_is_written = (
      counts.pipeline
      | 'Prepare' >> beam.Create([None])

      # Using AsIter instead of AsList at the callsite below in order to reduce
      # max memory usage.
      | 'OrderElements' >> beam.ParDo(_OrderElementsFn(store_frequency),
                                      counts_iter=beam.pvalue.AsIter(counts))
      | 'WriteToFile' >> beam.io.WriteToText(vocabulary_file,
                                             shard_name_template=''))
  # Return the vocabulary path.
  wait_for_vocabulary_transform = (
      counts.pipeline
      | 'CreatePath' >> beam.Create([np.array(vocabulary_file)])
      # Ensure that the analysis returns only after the file is written.
      | 'WaitForVocabularyFile' >> beam.Map(
          lambda x, y: x, y=beam.pvalue.AsIter(vocab_is_written)))
  return (wait_for_vocabulary_transform,)


@common.register_ptransform(attributes_classes.Vocabulary)
class VocabularyImpl(beam.PTransform):
  """Saves the unique elements in a PCollection of batches."""

  def __init__(self, num_outputs, attributes, base_temp_dir, **kwargs):
    if num_outputs != 1:
      raise ValueError('num_outputs for VocabularyImpl should be 1, got {}`.'.
                       format(num_outputs))
    self._top_k = attributes.top_k
    self._frequency_threshold = attributes.frequency_threshold
    self._store_frequency = attributes.store_frequency
    self._vocab_filename = attributes.vocab_filename
    self._vocab_ordering_type = attributes.vocab_ordering_type
    self._name = attributes.name
    self._base_temp_dir = base_temp_dir

  def default_label(self):
    return 'VocabularyImpl[{}]'.format(self._name)

  def expand(self, inputs):
    pcoll, = inputs
    if self._top_k is not None and self._top_k < 0:
      raise ValueError('top_k for VocabularyImpl should be >= 0 or None, got '
                       '{}.'.format(self._top_k))
    if self._frequency_threshold is not None and self._frequency_threshold < 0:
      raise ValueError(
          'frequency_threshold for VocabularyImpl should be >= 0 or None, '
          'got {}.'.format(self._frequency_threshold))

    # Create a PCollection of (count, element) pairs, then iterates over
    # this to create a single element PCollection containing this list of
    # pairs in sorted order by decreasing counts (and by values for equal
    # counts).

    def is_problematic_string(kv):
      string, _ = kv  # Ignore counts.
      return string and '\n' not in string and '\r' not in string

    if (self._vocab_ordering_type ==
        tf_utils.VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION):
      flatten_map_fn = (
          _flatten_positive_label_weights_total_weights_and_counts)

      # count_and_means is a pcollection that contains a
      # _CountAndWeightsMeansAccumulator where:
      #   `weighted_mean` is the weighted mean of positive labels
      #       for all features.
      #   `count` is the count for all features.
      #   `weights_mean` is the mean of the weights for all features.
      count_and_means = (
          pcoll
          | 'SumBatchCountAndWeightsMeans' >> beam.Map(_count_and_means)
          | 'ComputeCountAndWeightsMeansGlobally' >> beam.CombineGlobally(
              CountAndWeightsMeansCombineFn()))

      # CountAndWeightsMeansCombineFn returns a tuple of the form:
      # (feature,_CountAndWeightsMeansAccumulator) where:
      #   `feature` is a single string, which is the word in the vocabulary
      #       whose mutual information with the label is being computed.
      #   `weighted_mean` is the weighted mean of y positive given x.
      #   `count` is the count of weights for a feature.
      #   `weights_mean` is the mean of the weights for a feature.
      combine_transform = (
          'ComputeCountAndWeightsMeansPerUniqueWord' >> beam.CombinePerKey(
              CountAndWeightsMeansCombineFn())
          | 'CalculateMutualInformationPerUniqueWord' >> beam.Map(
              _calculate_mutual_information,
              global_accumulator=beam.pvalue.AsSingleton(count_and_means)))
    elif (self._vocab_ordering_type ==
          tf_utils.VocabOrderingType.WEIGHTED_FREQUENCY):
      flatten_map_fn = _flatten_value_and_weights_to_list_of_tuples
      combine_transform = beam.CombinePerKey(sum)
    else:
      flatten_map_fn = _flatten_value_to_list
      combine_transform = beam.combiners.Count.PerElement()

    raw_counts = (
        pcoll
        | 'FlattenStringsAndMaybeWeightsLabels' >> beam.FlatMap(flatten_map_fn)
        | 'CountPerString' >> combine_transform
        | 'FilterProblematicStrings' >> beam.Filter(is_problematic_string)
        | 'SwapStringsAndCounts' >> beam.KvSwap())

    counts = (
        raw_counts | 'ApplyFrequencyThresholdAndTopK' >> (
            _ApplyFrequencyThresholdAndTopK(  # pylint: disable=no-value-for-parameter
                self._frequency_threshold,
                self._top_k
                )))

    return counts | 'WriteVocabFile' >> (
        _WriteVocabFile(  # pylint: disable=no-value-for-parameter
            self._base_temp_dir, self._vocab_filename, self._store_frequency))


def _flatten_value_to_list(batch_values):
  """Converts an N-D dense or sparse batch to a 1-D list."""
  batch_value, = batch_values

  return batch_value.tolist()


def _flatten_value_and_weights_to_list_of_tuples(batch_values):
  """Converts a batch of vocabulary and weights to a list of KV tuples."""
  batch_value, weights = batch_values

  batch_value = batch_value.tolist()
  weights = weights.tolist()
  return zip(batch_value, weights)


def _flatten_positive_label_weights_total_weights_and_counts(batch_values):
  """Converts a batch of vocab weights and counts to a list of KV tuples."""
  batch_value, total_weights, positive_label_weights, counts = batch_values
  batch_value = batch_value.tolist()
  positive_label_weights = positive_label_weights.tolist()
  total_weights = total_weights.tolist()
  counts = counts.tolist()
  return zip(batch_value, zip(positive_label_weights, total_weights, counts))


def _count_and_means(batch_values):
  _, total_weights, positive_label_weights, counts = batch_values
  return [sum(positive_label_weights), sum(total_weights), sum(counts)]


def _clip_probability(p):
  epsilon = 1e-6
  p = np.clip(p, epsilon, 1 - epsilon)
  return p, 1 - p


def _calculate_mutual_information(feature_and_accumulator, global_accumulator):
  """Calculates the mutual information of a feature.

  H(x, y) = (feature_sum_weights *
             [(P(y|x)*log2(P(y|x)/P(y))) + (P(~y|x)*log2(P(~y|x)/P(~y)))])
  x is feature and y is label. feature_sum_weights instead of p_x is used,
  as this makes the "mutual_information" more interpretable.
  If we don't divide by global_sum_weights, it can be though of as an "adjusted"
  weighted count.

  Args:
    feature_and_accumulator: A tuple of the form:
    (feature, (_CountAndWeightsMeansAccumulator)) where: `feature` is a single
      string, which is the word in the vocabulary whose mutual information with
      the label is beingcomputed. `weighted_mean` is the weighted mean positive
      given x. `count` is the count of weights for a feature. `weights_mean`is
      the mean of the weights for a feature.
    global_accumulator: A _CountAndWeightsMeansAccumulator where:
      `weighted_mean` is the weighted mean of positive labels for all features.
      `count` is the count for all features. `mean` is the mean of the weights
      for all features.

  Returns:
    The feature and its mutual information.
  """
  feature, current_accumulator = feature_and_accumulator
  feature_sum_weights = (
      current_accumulator.count * current_accumulator.weights_mean)
  global_sum_weights = (
      global_accumulator.count * global_accumulator.weights_mean)
  if global_sum_weights == 0:
    return float('NaN')
  weighted_mean_positive_given_x, weighted_mean_negative_given_x = (
      _clip_probability(current_accumulator.weighted_mean))
  weighted_mean_positive, weighted_mean_negative = _clip_probability(
      global_accumulator.weighted_mean)
  mutual_information = feature_sum_weights * (
      (weighted_mean_positive_given_x * np.log2(
          weighted_mean_positive_given_x / weighted_mean_positive)) +
      (weighted_mean_negative_given_x * np.log2(
          weighted_mean_negative_given_x / weighted_mean_negative)))
  return (feature, mutual_information)


class _CountAndWeightsMeansAccumulator(
    collections.namedtuple('CountAndWeightsMeansAccumulator',
                           ['weighted_mean', 'count', 'weights_mean'])):
  """Container for CountAndWeightsMeansCombiner intermediate values."""

  @classmethod
  def make_nan_to_num(cls, weighted_means, counts, weights_mean):
    return cls(
        np.nan_to_num(weighted_means), counts, np.nan_to_num(weights_mean))

  def __reduce__(self):
    return self.__class__, tuple(self)


class CountAndWeightsMeansCombineFn(beam.CombineFn):
  """CountAndWeightsMeansCombineFn calculates total count and weighted means.

  """

  def create_accumulator(self):
    """Create an accumulator with all zero entries."""
    return _CountAndWeightsMeansAccumulator(
        weighted_mean=0., count=0, weights_mean=0.)

  def add_input(self, accumulator, batch_values):
    """Composes an accumulator from batch_values and calls merge_accumulators.

    Args:
      accumulator: The `_CountAndWeightsMeansAccumulator` computed so far.
      batch_values: A `_CountAndWeightsMeansAccumulator` for the current batch.

    Returns:
      A `_CountAndWeightsMeansAccumulator` which is accumulator and batch_values
      combined.
    """
    (element_sum_positive, element_weights_sum_total,
     element_count) = batch_values
    new_accumulator = _CountAndWeightsMeansAccumulator(
        weighted_mean=(element_sum_positive / element_weights_sum_total),
        count=element_count,
        weights_mean=(element_weights_sum_total / element_count))
    return self._combine_counts_and_means_accumulators(accumulator,
                                                       new_accumulator)

  def merge_accumulators(self, accumulators):
    """Merges several `_CountAndWeightsMeansAccumulator`s.

    Args:
      accumulators: A list of `_CountAndWeightsMeansAccumulator`s and/or Nones.

    Returns:
      The sole merged `_CountAndWeightsMeansAccumulator`.
    """
    non_empty_accumulators = [
        accumulator for accumulator in accumulators if accumulator is not None
    ]
    if not non_empty_accumulators:
      return self.create_accumulator()

    result = non_empty_accumulators[0]

    for accumulator in non_empty_accumulators[1:]:
      result = self._combine_counts_and_means_accumulators(result, accumulator)

    return result

  def extract_output(self, accumulator):
    """Returns the accumulator as the output.

    Args:
      accumulator: the final `_CountAndWeightsMeansAccumulator` value.

    Returns:
     The accumulator which could be None.
    """
    return accumulator

  # Mean update formulas which are more numerically stable when a and b vary in
  # magnitude.
  def _compute_running_weighted_mean(self, total_count, total_weights_mean,
                                     previous_weighted_mean, new_weighted_mean,
                                     new_weights_mean):
    return (
        previous_weighted_mean + (new_weighted_mean - previous_weighted_mean) *
        (new_weights_mean / (total_count * total_weights_mean)))

  def _combine_counts_and_means_accumulators(self, a, b):
    # NaNs get preserved through division by a.count + b.count.
    a = _CountAndWeightsMeansAccumulator.make_nan_to_num(*a)
    b = _CountAndWeightsMeansAccumulator.make_nan_to_num(*b)

    # a.count >= b.count following this logic.
    if np.sum(a.count) < np.sum(b.count):
      a, b = b, a

    if np.sum(a.count) == 0:
      return b

    combined_count = a.count + b.count

    # We use the mean of the weights because it is more numerically stable than
    # summing all of the weights.
    combined_weights_mean = self._compute_running_weighted_mean(
        total_count=combined_count,
        total_weights_mean=1.,
        previous_weighted_mean=a.weights_mean,
        new_weighted_mean=b.weights_mean,
        new_weights_mean=b.count)
    combined_weighted_mean = self._compute_running_weighted_mean(
        total_count=combined_count,
        total_weights_mean=combined_weights_mean,
        previous_weighted_mean=a.weighted_mean,
        new_weighted_mean=b.weighted_mean,
        new_weights_mean=(b.count * b.weights_mean))
    return _CountAndWeightsMeansAccumulator(
        weighted_mean=combined_weighted_mean,
        count=combined_count,
        weights_mean=combined_weights_mean)


@with_input_types(Tuple[np.ndarray, ...])
class _CombinerWrapper(beam.CombineFn):
  """Class to wrap a attributes_classes.Combiner as a beam.CombineFn."""

  def __init__(self, combiner, serialized_tf_config):
    if isinstance(combiner, analyzers.QuantilesCombiner):
      tf_config = common._maybe_deserialize_tf_config(  # pylint: disable=protected-access
          serialized_tf_config)
      combiner.initialize_local_state(tf_config)
    self._combiner = combiner
    self._serialized_tf_config = serialized_tf_config

  def __reduce__(self):
    return _CombinerWrapper, (self._combiner, self._serialized_tf_config)

  def create_accumulator(self):
    return self._combiner.create_accumulator()

  def add_input(self, accumulator, next_input):
    return self._combiner.add_input(accumulator, next_input)

  def merge_accumulators(self, accumulators):
    return self._combiner.merge_accumulators(accumulators)

  def extract_output(self, accumulator):
    return self._combiner.extract_output(accumulator)


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


@common.register_ptransform(attributes_classes.Combine)
class CombineImpl(beam.PTransform):
  """Implement an analyzer based on a Combine."""

  def __init__(self, num_outputs, attributes, serialized_tf_config, **kwargs):
    self._combiner = attributes.combiner
    self._name = attributes.name
    self._serialized_tf_config = serialized_tf_config
    self._num_outputs = num_outputs

  def default_label(self):
    return 'CombineImpl[{}]'.format(self._name)

  def expand(self, inputs):
    pcoll, = inputs
    # NOTE: Currently, all combiners except QuantilesCombiner
    # require .with_defaults(False) to be set.
    has_defaults = isinstance(self._combiner, analyzers.QuantilesCombiner)

    def extract_outputs(outputs, num_outputs):
      if len(outputs) != num_outputs:
        raise ValueError(
            'Analyzer has {} outputs but its implementation produced {} '
            'values'.format(num_outputs, len(outputs)))
      for i, output in enumerate(outputs):
        yield beam.pvalue.TaggedOutput(str(i), output)

    output_keys = [str(i) for i in range(self._num_outputs)]
    outputs_tuple = (
        pcoll
        | 'CombineGlobally' >> beam.CombineGlobally(
            _CombinerWrapper(self._combiner, self._serialized_tf_config))
        .with_defaults(has_defaults)
        | 'ExtractOutputs' >> beam.FlatMap(
            extract_outputs, self._num_outputs).with_outputs(*output_keys))
    return tuple(outputs_tuple[key] for key in output_keys)


@common.register_ptransform(attributes_classes.CombinePerKey)
class _CombinePerKeyImpl(beam.PTransform):
  """Implement an analyzer based on a CombinePerKey."""

  def __init__(self, num_outputs, attributes, serialized_tf_config, **kwargs):
    if num_outputs < 1:
      raise ValueError('num_outputs for _ComvinePerKeyImpl should be >= 1, got '
                       '{}'.format(num_outputs))
    self._combiner = attributes.combiner
    self._name = attributes.name
    self._serialized_tf_config = serialized_tf_config
    # Note we define self._num_outputs to be the number of outputs of the
    # CombineFn, to which will be added the key vocabulary after merging by
    # key.
    self._num_outputs = num_outputs - 1

  def default_label(self):
    return 'CombinePerKeyImpl[{}]'.format(self._name)

  def expand(self, inputs):
    pcoll, = inputs
    output_keys = ['key'] + [str(i) for i in range(self._num_outputs)]
    outputs_tuple = (
        pcoll
        | 'SplitByKey' >> beam.FlatMap(_split_inputs_by_key)
        | 'CombinePerKey' >> beam.CombinePerKey(
            _CombinerWrapper(self._combiner, self._serialized_tf_config))
        | 'ToList' >> beam.combiners.ToList()
        | 'MergeByKey' >> beam.FlatMap(
            _merge_outputs_by_key,
            self._num_outputs).with_outputs(*output_keys))
    return tuple(outputs_tuple[key] for key in output_keys)
