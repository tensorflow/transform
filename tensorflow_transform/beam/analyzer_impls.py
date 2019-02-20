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
import math
import os

# GOOGLE-INITIALIZATION

import apache_beam as beam

from apache_beam.transforms.ptransform import ptransform_fn
from apache_beam.typehints import Any
from apache_beam.typehints import KV
from apache_beam.typehints import Tuple
from apache_beam.typehints import Union
from apache_beam.typehints import with_input_types

import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import analyzers
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
      # TODO(b/62272023) remove this workaround if/when fixed on tensorflow.
      # If the vocabulary is empty add a dummy value with count one so
      # the tensorflow index operations don't fail to initialize with empty
      # tensors downstream.
      counts = [(1, '49d0cd50-04bb-48c0-bc6f-5b575dce351a')]

    counts.sort(reverse=True)  # Largest first.
    for count, entry in counts:
      if self._store_frequency:
        # Converts bytes to unicode for PY3, otherwise the result will look like
        # "b'real_string'". We convert everything to bytes afterwards.
        if six.PY2:
          yield '{} {}'.format(count, entry)
        else:
          yield tf.compat.as_bytes('{} {}'.format(count,
                                                  tf.compat.as_text(entry)))
      else:
        yield entry


@ptransform_fn
@beam.typehints.with_input_types(KV[float, str])
@beam.typehints.with_output_types(KV[float, str])
def _ApplyFrequencyThresholdAndTopK(  # pylint: disable=invalid-name
    counts, frequency_threshold, top_k, key_fn):
  """Applies `frequency_threshold` and `top_k` to (count, value) pairs."""
  # TODO(b/117796748): Filter frequency per-key when key feature input enabled.
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
    # provided since that already induces a single-sharded output.
    # TODO(b/26245647): Remove this "Reshard".
    counts |= 'Reshard' >> beam.transforms.Reshuffle()  # pylint: disable=no-value-for-parameter
  else:
    # TODO(katsiapis): Perhaps enhance Beam's Top to accept an N that can
    # signify "unlimited" and then we can simplify a lot of our code (though
    # that might come at a performance penalty).
    if key_fn:
      def map_key_to_count_and_term(kv, key_fn):
        """Parses key from term with `key_fn` and maps it to count and term."""
        count, term = kv
        key = key_fn(term)
        return key, (count, term)

      return (
          counts
          | 'MapKeyToCountAndTerm' >> beam.Map(
              lambda x: map_key_to_count_and_term(x, key_fn))
          | 'CoverageTop(%s)' % top_k >> beam.combiners.Top.LargestPerKey(top_k)
          | 'FlattenCoverageTerms' >> beam.FlatMap(lambda kv: kv[1]))
    counts = (counts
              | 'Top(%s)' % top_k >> beam.combiners.Top.Of(top_k)
              | 'FlattenList' >> beam.FlatMap(lambda lst: lst))
  return counts


@common.register_ptransform(analyzer_nodes.VocabularyAccumulate)
@beam.typehints.with_input_types(Tuple[np.ndarray, ...])
# TODO(b/123325923): Constrain the key type here to the right string type.
@beam.typehints.with_output_types(KV[Any, Union[int, float]])  # Any -> np.str?
class VocabularyAccumulateImpl(beam.PTransform):
  """Accumulates the unique elements in a PCollection of batches."""

  def __init__(self, operation, extra_args):
    self._vocab_ordering_type = operation.vocab_ordering_type

  def expand(self, inputs):
    pcoll, = inputs

    # Create a PCollection of (count, element) pairs, then iterates over
    # this to create a single element PCollection containing this list of
    # pairs in sorted order by decreasing counts (and by values for equal
    # counts).

    # TODO(b/62379925) Filter empty strings or strings containing the \n or \r
    # tokens since index_table_from_file doesn't allow empty rows.
    def is_problematic_string(kv):
      string, _ = kv  # Ignore counts.
      return string and b'\n' not in string and b'\r' not in string

    # TODO(b/112916494): Unify the graph in both cases once possible.
    if (self._vocab_ordering_type ==
        tf_utils.VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION):
      flatten_map_fn = _flatten_to_key_and_means_accumulator_list
      combine_transform = _MutualInformationTransformAccumulate()  # pylint: disable=no-value-for-parameter
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
        | 'FilterProblematicStrings' >> beam.Filter(is_problematic_string))

    return raw_counts


@common.register_ptransform(analyzer_nodes.VocabularyMerge)
@beam.typehints.with_input_types(KV[np.str, Union[int, float]])
# TODO(b/123325923): Constrain the value type here to the right string type.
@beam.typehints.with_output_types(KV[Union[int, float], Any])  # Any -> np.str?
class VocabularyMergeImpl(beam.PTransform):
  """Merges vocabulary accumulators of (word, num) pairs."""

  def __init__(self, operation, extra_args):
    self._vocab_ordering_type = operation.vocab_ordering_type
    self._use_adjusted_mutual_info = operation.use_adjusted_mutual_info
    self._min_diff_from_avg = operation.min_diff_from_avg

  def expand(self, inputs):
    if (self._vocab_ordering_type ==
        tf_utils.VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION):
      combine_transform = _MutualInformationTransformMerge(  # pylint: disable=no-value-for-parameter
          self._use_adjusted_mutual_info, self._min_diff_from_avg)
    else:
      combine_transform = beam.CombinePerKey(sum)

    pcoll, = inputs

    raw_counts = (
        pcoll
        | 'CountPerString' >> combine_transform
        | 'SwapStringsAndCounts' >> beam.KvSwap())

    return raw_counts


@common.register_ptransform(analyzer_nodes.VocabularyOrderAndFilter)
@beam.typehints.with_input_types(KV[Union[int, float], np.str])
# TODO(b/123325923): Constrain the value type here to the right string type.
@beam.typehints.with_output_types(KV[Union[int, float], Any])  # Any -> np.str?
class VocabularyOrderAndFilterImpl(beam.PTransform):
  """Order, filters and writes the computed vocabulary file."""

  def __init__(self, operation, extra_args):
    self._top_k = operation.top_k
    self._frequency_threshold = operation.frequency_threshold
    self._coverage_top_k = operation.coverage_top_k
    self._coverage_frequency_threshold = operation.coverage_frequency_threshold
    self._key_fn = operation.key_fn

  def expand(self, inputs):
    if self._top_k is not None and self._top_k < 0:
      raise ValueError('top_k for VocabularyImpl should be >= 0 or None, got '
                       '{}.'.format(self._top_k))
    if self._frequency_threshold is not None and self._frequency_threshold < 0:
      raise ValueError(
          'frequency_threshold for VocabularyImpl should be >= 0 or None, '
          'got {}.'.format(self._frequency_threshold))
    if self._coverage_top_k is not None and self._coverage_top_k < 0:
      raise ValueError('coverage_top_k for VocabularyImpl should be >= 0 or '
                       'None, got {}.'.format(self._coverage_top_k))
    if (self._coverage_frequency_threshold is not None and
        self._coverage_frequency_threshold < 0):
      raise ValueError(
          'coverage_frequency_threshold for VocabularyImpl should be >= 0 or '
          'None, got {}.'.format(self._coverage_frequency_threshold))
    pcoll, = inputs

    counts = (
        pcoll | 'ApplyFrequencyThresholdAndTopK' >> (
            _ApplyFrequencyThresholdAndTopK(  # pylint: disable=no-value-for-parameter
                self._frequency_threshold, self._top_k, None)))

    if self._key_fn:
      coverage_counts = (
          pcoll | 'ApplyCoverageFrequencyThresholdAndTopK' >> (
              _ApplyFrequencyThresholdAndTopK(  # pylint: disable=no-value-for-parameter
                  self._coverage_frequency_threshold, self._coverage_top_k,
                  self._key_fn)))

      counts = (
          (counts, coverage_counts)
          | 'MergeStandardAndCoverageArms' >> beam.Flatten()
          | 'RemoveDuplicates' >> beam.RemoveDuplicates())

    return counts


@common.register_ptransform(analyzer_nodes.VocabularyWrite)
@beam.typehints.with_input_types(KV[Union[int, float], np.str])
@beam.typehints.with_output_types(np.ndarray)
class VocabularyWriteImpl(beam.PTransform):
  """Writes the computed vocabulary file."""

  def __init__(self, operation, extra_args):
    self._base_temp_dir = extra_args.base_temp_dir
    self._store_frequency = operation.store_frequency
    self._vocab_filename = operation.vocab_filename

  def expand(self, inputs):
    counts, = inputs
    vocabulary_file = os.path.join(self._base_temp_dir, self._vocab_filename)
    vocab_is_written = (
        counts.pipeline
        | 'Prepare' >> beam.Create([None])

        # Using AsIter instead of AsList at the callsite below in order to
        # reduce max memory usage.
        | 'OrderElements' >> beam.ParDo(_OrderElementsFn(self._store_frequency),
                                        counts_iter=beam.pvalue.AsIter(counts))
        # TODO(b/62379925) For now force a single file. Should
        # `InitializeTableFromTextFile` operate on a @N set of files?
        # TODO(b/67863471) Here we are relying on fusion (an implementation
        # detail) for the ordering to be maintained when the results are written
        # to disk. Perform the write within the body of `OrderElements` maybe
        # `OrderElementsAndWrite`. This would mean using TF IO instead of Beam
        # IO so it's perhaps not great.
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


def _flatten_value_to_list(batch_values):
  """Converts an N-D dense or sparse batch to a 1-D list."""
  batch_value, = batch_values

  # TODO(b/36603294): Perhaps obviate the tolist(). It is currently used so
  # that we go to native Python types for more efficient followup
  # processing.
  return batch_value.tolist()


def _flatten_value_and_weights_to_list_of_tuples(batch_values):
  """Converts a batch of vocabulary and weights to a list of KV tuples."""
  batch_value, weights = batch_values

  # TODO(b/36603294): Perhaps obviate the tolist(). It is currently used so
  # that we go to native Python types for more efficient followup
  # processing.
  batch_value = batch_value.tolist()
  weights = weights.tolist()
  return zip(batch_value, weights)


def _make_count_and_weights_means_accumulator(sum_positive, weights_sum_total,
                                              count):
  return _CountAndWeightsMeansAccumulator(
      weighted_mean=(sum_positive / weights_sum_total),
      count=count,
      weights_mean=(weights_sum_total / count))


def _flatten_to_key_and_means_accumulator_list(batch_values):
  """Converts a batch of keys, weights, and counts to a list of KV pairs."""
  keys, total_weights, positive_label_weights, counts = batch_values

  # TODO(b/36603294): Perhaps obviate the tolist(). It is currently used so
  # that we go to native Python types for more efficient followup
  # processing.
  keys = keys.tolist()
  positive_label_weights = positive_label_weights.tolist()
  total_weights = total_weights.tolist()
  counts = counts.tolist()

  return zip(keys, [
      _make_count_and_weights_means_accumulator(*batch)
      for batch in zip(positive_label_weights, total_weights, counts)
  ])


def _clip_probability(p):
  epsilon = 1e-6
  p = np.clip(p, epsilon, 1 - epsilon)
  return p, 1 - p


def _calculate_mutual_information(feature_and_accumulator, global_accumulator,
                                  use_adjusted_mutual_info, min_diff_from_avg):
  """Calculates the (adjusted) mutual information of a feature.

  Mutual information H(x, y) = (sum(weights) *
             [(P(y|x)*log2(P(y|x)/P(y))) + (P(~y|x)*log2(P(~y|x)/P(~y)))])
  Where x is feature and y is label.
  We use sum(weights) instead of P(x), as this makes the mutual information more
  interpretable.
  If we don't divide by sum(weights), it can be thought of as an adjusted
  weighted count.

  Adjusted mutual information AMI(x, y) = MI(x, y) - EMI(x, y)
  x is the feature and y is label. It's calculated by subtracting the expected
  mutual information (EMI) from mutual information. The calculation is based on
  the following paper:

  Vinh, N. X.; Epps, J.; Bailey, J. (2009). "Information theoretic measures for
  clusterings comparison". Proceedings of the 26th Annual International Confere
  nce on Machine Learning - ICML '09. p. 1.
  doi:10.1145/1553374.1553511. ISBN 9781605585161.

  Short summary can be found in the Wikipedia link:
  https://en.wikipedia.org/wiki/Adjusted_mutual_information

  Args:
    feature_and_accumulator: A tuple of the form:
      (feature, _CountAndWeightsMeansAccumulator) where:
        `feature` is a single string, which is the word in the vocabulary whose
          mutual information with the label is being computed.
        `weighted_mean` is the weighted mean positive given x.
        `count` is the count of weights for a feature.
        `weights_mean` is the mean of the weights for a feature.
    global_accumulator: A _CountAndWeightsMeansAccumulator where:
      `weighted_mean` is the weighted mean of positive labels for all features.
      `count` is the count for all features. `mean` is the mean of the weights
      for all features.
    use_adjusted_mutual_info: If set to True, use adjusted mutual information.
    min_diff_from_avg: Mutual information of a feature will be adjusted to zero
      whenever the absolute difference between count of the feature with any
      label and its expected count is lower than min_diff_from_average.

  Returns:
    The feature and its mutual information.
  """
  feature, current_accumulator = feature_and_accumulator
  x = (current_accumulator.count * current_accumulator.weights_mean)
  n = (global_accumulator.count * global_accumulator.weights_mean)
  if n == 0:
    return (feature, float('NaN'))

  n_1, n_0 = [
      weighted_mean * current_accumulator.weights_mean *
      current_accumulator.count
      for weighted_mean in _clip_probability(current_accumulator.weighted_mean)
  ]
  y_1, y_0 = [
      weighted_mean * global_accumulator.weights_mean * global_accumulator.count
      for weighted_mean in _clip_probability(global_accumulator.weighted_mean)
  ]

  diff_from_avg = x * y_1 / n - n_1
  if abs(diff_from_avg) < min_diff_from_avg:
    return (feature, 0)
  mutual_information = (
      n_1 * (np.log2(n_1) + np.log2(n) - np.log2(y_1) - np.log2(x)) +
      n_0 * (np.log2(n_0) + np.log2(n) - np.log2(y_0) - np.log2(x)))

  if use_adjusted_mutual_info:
    expected_mutual_information = (
        _calculate_expected_mutual_information_per_label(n, x, y_1) +
        _calculate_expected_mutual_information_per_label(n, x, y_0))

    return (feature, mutual_information - expected_mutual_information)
  else:
    return (feature, mutual_information)


def _calculate_expected_mutual_information_per_label(n, x, y_j):
  """Calculates the expected mutual information of a feature and a label.

    EMI(x, y) = sum_{n_ij = max(0, x_i + y_j - n) to min(x_i, y_j)} (
      n_ij / n * log2((n * n_ij / (x_i * y_j))
      * ((x_i! * y_j! * (n - x_i)! * (n - y_j)!) /
      (n! * n_ij! * (x_i - n_ij)! * (y_j - n_ij)! * (n - x_i - y_j + n_ij)!)))
    where n_ij is the joint count of feature and label, x_i is the count for
    feature x, y_j is the count for label y, and n represents total count.

    Note: In the paper, expected mutual information is calculated by summing
    over both i and j, but here we don't count the consitrbution of the case i=0
    (where the feature is not present), and this is consistent with how mutual
    information is computed.

  Args:
    n: The sum of weights for all features.
    x: The sum of weights for the feature whose expected mutual information is
      computed.
    y_j: The sum of weights for positive (or negative) labels for all features.

  Returns:
    Calculated expected mutual information.
  """
  coefficient = (-np.log2(x) - np.log2(y_j) + np.log2(n))
  sum_probability = 0.0
  partial_result = 0.0
  for n_j, p_j in _hypergeometric_pmf(n, x, y_j):
    if n_j != 0:
      partial_result += n_j * (coefficient + np.log2(n_j)) * p_j
    sum_probability += p_j
  # With approximate calculations for log2(x) and exp2(x) with large x, need a
  # correction to the probablity approximation.
  return partial_result / sum_probability


def _hypergeometric_pmf(n, x, y_j):
  """Probablity for expectation computation under hypergeometric distribution.

  Args:
    n: The sum of weights for all features.
    x: The sum of weights for the feature whose expected mutual information is
      computed.
    y_j: The sum of weights for positive (or negative) labels for all features.

  Yields:
    Calculated coefficient, numerator and denominator for hypergeometric
    distribution.
  """
  start = int(max(0, n - (n - x) - (n - y_j)))
  end = int(min(x, y_j))
  numerator = (
      _logfactorial(x) + _logfactorial(y_j) + _logfactorial(n - x) +
      _logfactorial(n - y_j))
  denominator = (
      _logfactorial(n) + _logfactorial(start) + _logfactorial(x - start) +
      _logfactorial(y_j - start) + _logfactorial(n - x - y_j + start))
  for n_j in range(start, end + 1):
    p_j = np.exp(numerator - denominator)
    denominator += (
        np.log(n_j + 1) - np.log(x - n_j) - np.log(y_j - n_j) +
        np.log(n - x - y_j + n_j + 1))
    yield n_j, p_j


def _logfactorial(n):
  """Calculate natural logarithm of n!."""
  return math.lgamma(n + 1)


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


@ptransform_fn
@beam.typehints.with_input_types(KV[str, _CountAndWeightsMeansAccumulator])
@beam.typehints.with_output_types(KV[str, _CountAndWeightsMeansAccumulator])
def _MutualInformationTransformAccumulate(pcol):  # pylint: disable=invalid-name
  """Accumulates information needed for mutual information computation."""
  return (pcol | 'VocabCountPerLabelPerStringAccumulate' >> beam.CombinePerKey(
      _CountAndWeightsMeansCombineFn()))


@ptransform_fn
@beam.typehints.with_input_types(KV[str, _CountAndWeightsMeansAccumulator])
@beam.typehints.with_output_types(KV[str, float])
def _MutualInformationTransformMerge(  # pylint: disable=invalid-name
    pcol, use_adjusted_mutual_info, min_diff_from_avg):
  """Computes mutual information for each key using the given accumulators."""
  feature_accumulator_pcol = (
      pcol | 'VocabCountPerLabelPerStringMerge' >> beam.CombinePerKey(
          _CountAndWeightsMeansCombineFn()))

  global_accumulator = (
      feature_accumulator_pcol
      | 'DropKeys' >> beam.Values()
      | 'VocabCountPerLabelGlobally' >> beam.CombineGlobally(
          _CountAndWeightsMeansCombineFn()))

  return (feature_accumulator_pcol
          | 'CalculateMutualInformationPerString' >> beam.Map(
              _calculate_mutual_information,
              beam.pvalue.AsSingleton(global_accumulator),
              use_adjusted_mutual_info=use_adjusted_mutual_info,
              min_diff_from_avg=min_diff_from_avg))


# TODO(b/116698987): Share logic with MeanAndVarCombiner.
class _CountAndWeightsMeansCombineFn(beam.CombineFn):
  """_CountAndWeightsMeansCombineFn calculates total count and weighted means.

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
    return self._combine_counts_and_means_accumulators(accumulator,
                                                       batch_values)

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
  """Class to wrap a analyzer_nodes.Combiner as a beam.CombineFn."""

  def __init__(self,
               combiner,
               serialized_tf_config,
               is_combining_accumulators,
               should_extract_output=None):
    """Init method for _CombinerWrapper.

    Args:
      combiner: A `analyzer_nodes.Combiner` object used to combine.
      serialized_tf_config: A str which is a serialized form of
        `tf.ConfigProto`.
      is_combining_accumulators: A bool which indicates whether this is
        combining single or batched inputs, or already accumulated objects.
      should_extract_output: A bool which indicates whether this should call the
        combiner's extract_output method in extract_output. If not specified, we
        assume it's the same value as `should_extract_output`.
    """
    # TODO(b/69566045): Move initialization to start_bundle(), removing the need
    # for initialize_local_state to be called here.
    if isinstance(combiner, analyzers.QuantilesCombiner):
      tf_config = common._maybe_deserialize_tf_config(  # pylint: disable=protected-access
          serialized_tf_config)
      combiner.initialize_local_state(tf_config)
    self._combiner = combiner
    self._serialized_tf_config = serialized_tf_config
    self._is_combining_accumulators = is_combining_accumulators
    if should_extract_output is None:
      should_extract_output = is_combining_accumulators
    self._should_extract_output = should_extract_output

  def __reduce__(self):
    return _CombinerWrapper, (self._combiner, self._serialized_tf_config,
                              self._is_combining_accumulators,
                              self._should_extract_output)

  def create_accumulator(self):
    return self._combiner.create_accumulator()

  def add_input(self, accumulator, next_input):
    if self._is_combining_accumulators:
      # First accumulator can be None.
      accumulators = []
      if accumulator is not None:
        accumulators.append(accumulator)
      if next_input is not None:
        accumulators.append(next_input)
      return self.merge_accumulators(accumulators)
    return self._combiner.add_input(accumulator, next_input)

  def merge_accumulators(self, accumulators):
    return self._combiner.merge_accumulators(accumulators)

  def extract_output(self, accumulator):
    if self._should_extract_output:
      return self._combiner.extract_output(accumulator)
    return accumulator


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
  # TODO(b/77873002): Raise these errors in the graph where more informative
  # errors can be generated.  Keep these as a fallback for user-defined
  # `Combiner`s.
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


def _merge_outputs_by_key(keys_and_outputs, outputs_dtype):
  """Merge outputs of analyzers per key into a single output.

  Takes a list of elements of the form (key, [output0, ..., output{N-1}]) and
  returns a list of ndarrays of the form [keys, outputs0, ..., outputs[{N-1}]]
  where keys is formed by stacking the values of `key` from the list and
  similarly outputs{k} is formed by stacking the individual elements of
  output{k} from the list.

  For each k, output{k} must be an ndarray whose size is the same for each
  element of the list.

  Args:
    keys_and_outputs: A list of elements of the form
      (key, [output0, ..., output{N-1}])
    outputs_dtype: A list of tf.DType. Each element corresponds to an output.

  Yields:
    The `TaggedOutput`s: keys, outputs0, ..., outputs[{N-1}]

  Raises:
    ValueError: If the number is outputs doesn't match num_outputs.
  """
  num_outputs = len(outputs_dtype)

  # Sort a copy of keys_and_outputs by keys.
  sorted_keys_and_outputs = sorted(keys_and_outputs, key=lambda x: x[0])

  # Convert from a list of pairs of the form (key, outputs_for_key) to a list of
  # keys and a list of outputs (where the outer dimension is the number of
  # outputs not the number of keys).
  key = []
  outputs = []
  for k, o in sorted_keys_and_outputs:
    key.append(k)
    outputs.append(o)
  if not outputs:
    outputs = [[]] * num_outputs
  else:
    outputs = list(zip(*outputs))
  yield beam.pvalue.TaggedOutput('key',
                                 np.array(key, dtype=tf.string.as_numpy_dtype))
  if len(outputs) != num_outputs:
    raise ValueError(
        'Analyzer has {} outputs but its implementation produced {} '
        'values'.format(num_outputs, len(outputs)))
  for i, (output, dtype) in enumerate(zip(outputs, outputs_dtype)):
    yield beam.pvalue.TaggedOutput(str(i), np.array(output,
                                                    dtype=dtype.as_numpy_dtype))


@common.register_ptransform(analyzer_nodes.CacheableCombineAccumulate)
class _IntermediateAccumulateCombineImpl(beam.PTransform):
  """Implement an analyzer based on a Combine."""

  def __init__(self, operation, extra_args):
    self._combiner = operation.combiner
    self._serialized_tf_config = extra_args.serialized_tf_config
    self._num_outputs = operation.num_outputs
    self._name = operation.label

  def expand(self, inputs):
    pcoll, = inputs
    # NOTE: Currently, all combiners except QuantilesCombiner
    # require .with_defaults(False) to be set.
    # TODO(b/34792459): Don't set with_defaults.
    has_defaults = isinstance(self._combiner, analyzers.QuantilesCombiner)

    return (
        pcoll
        | 'InitialCombineGlobally' >> beam.CombineGlobally(
            _CombinerWrapper(
                self._combiner,
                self._serialized_tf_config,
                is_combining_accumulators=False)).with_defaults(has_defaults))


@common.register_ptransform(analyzer_nodes.CacheableCombineMerge)
class _MergeAccumulatorsCombineImpl(beam.PTransform):
  """Implement an analyzer based on a Combine."""

  def __init__(self, operation, extra_args):
    self._combiner = operation.combiner
    self._serialized_tf_config = extra_args.serialized_tf_config
    self._num_outputs = operation.num_outputs

    self._name = operation.label

  def expand(self, inputs):
    pcoll, = inputs
    # NOTE: Currently, all combiners except QuantilesCombiner
    # require .with_defaults(False) to be set.
    # TODO(b/34792459): Don't set with_defaults.
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
        | 'MergeCombinesGlobally' >> beam.CombineGlobally(
            _CombinerWrapper(
                self._combiner,
                self._serialized_tf_config,
                is_combining_accumulators=True)).with_defaults(has_defaults)
        | 'ExtractOutputs' >> beam.FlatMap(
            extract_outputs, self._num_outputs).with_outputs(*output_keys))
    return tuple(outputs_tuple[key] for key in output_keys)


@common.register_ptransform(analyzer_nodes.CacheableCombinePerKeyAccumulate)
class _IntermediateAccumulateCombinePerKeyImpl(beam.PTransform):
  """Implement an analyzer based on a CombinePerKey."""

  def __init__(self, operation, extra_args):
    self._combiner = operation.combiner
    self._serialized_tf_config = extra_args.serialized_tf_config

  def expand(self, inputs):
    pcoll, = inputs
    return (pcoll
            | 'SplitByKey' >> beam.FlatMap(_split_inputs_by_key)
            | 'CombinePerKey' >> beam.CombinePerKey(
                _CombinerWrapper(
                    self._combiner,
                    self._serialized_tf_config,
                    is_combining_accumulators=False)))


@common.register_ptransform(analyzer_nodes.CacheableCombinePerKeyMerge)
class _MergeAccumulatorsCombinePerKeyImpl(beam.PTransform):
  """Implement an analyzer based on a CombinePerKey."""

  def __init__(self, operation, extra_args):
    self._combiner = operation.combiner
    self._serialized_tf_config = extra_args.serialized_tf_config

  def expand(self, inputs):
    pcoll, = inputs
    output_keys = (
        ['key'
        ] + [str(i) for i in range(len(self._combiner.output_tensor_infos()))])
    outputs_tuple = (
        pcoll
        | 'MergeCombinePerKey' >> beam.CombinePerKey(
            _CombinerWrapper(
                self._combiner,
                self._serialized_tf_config,
                is_combining_accumulators=True))
        | 'ToList' >> beam.combiners.ToList()
        | 'MergeByKey' >> beam.FlatMap(_merge_outputs_by_key, [
            info.dtype for info in self._combiner.output_tensor_infos()
        ]).with_outputs(*output_keys))
    return tuple(outputs_tuple[key] for key in output_keys)


@common.register_ptransform(analyzer_nodes.PTransform)
def _ptransform_impl(inputs, operation, extra_args):
  del extra_args  # unused
  pcoll, = inputs
  return pcoll | operation.label >> operation.ptransform


@common.register_ptransform(analyzer_nodes.WriteCache)
class _WriteCacheImpl(beam.PTransform):
  """A PTransform that writes a cache object and returns it."""

  def __init__(self, operation, extra_args):
    self._path = os.path.join(extra_args.cache_location.output_cache_dir,
                              operation.path)
    self._coder = operation.coder
    self._label = operation.label

  def expand(self, inputs):
    pcoll, = inputs

    cache_is_written = (
        pcoll
        | 'EncodeCache[%s][%s]' %
        (self._label, self._path) >> beam.Map(self._coder.encode_cache)
        | 'WriteCache[%s][%s]' % (self._label, self._path) >>
        beam.io.WriteToTFRecord(self._path, file_name_suffix='.gz'))

    result = (
        pcoll
        | 'WaitForCacheFile' >>
        beam.Map(lambda x, y: x, y=beam.pvalue.AsIter(cache_is_written)))

    return (result,)


@common.register_ptransform(analyzer_nodes.ReadCache)
def _read_cache_impl(inputs, operation, extra_args):
  """A PTransform-like method that reads and decodes a cache object."""
  # This is implemented as a PTransform-like function because it has no
  # PCollection inputs.
  assert not inputs

  absolute_path = os.path.join(extra_args.cache_location.input_cache_dir,
                               operation.path)
  pattern = '{}-*-of-*.gz'.format(absolute_path)

  cache = (
      extra_args.pipeline
      | 'ReadCache[%s][%s]' % (operation.label, operation.path) >>
      beam.io.ReadFromTFRecord(pattern, validate=False)
      | 'DecodeCache[%s][%s]' % (operation.label, operation.path) >> beam.Map(
          operation.coder.decode_cache))

  return (cache,)
