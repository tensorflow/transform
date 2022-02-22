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

import functools
import hashlib
import itertools
import math
import os

from absl import logging
import apache_beam as beam

from apache_beam.transforms.ptransform import ptransform_fn
from apache_beam.typehints import Any
from apache_beam.typehints import Dict
from apache_beam.typehints import Iterable
from apache_beam.typehints import KV
from apache_beam.typehints import List
from apache_beam.typehints import Tuple
from apache_beam.typehints import TypeVariable
from apache_beam.typehints import Union

import numpy as np
import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import analyzers
from tensorflow_transform import common_types
from tensorflow_transform import info_theory
from tensorflow_transform import tf_utils
from tensorflow_transform.beam import common
from tensorflow_transform.beam import experimental

# TODO(b/199789764): Enable beam type checks (and remove this) after the
# violations are fixed.
_ENABLE_BEAM_TYPE_CHECKS = False

_VocabOrderingType = analyzers._VocabOrderingType  # pylint: disable=protected-access
_VocabTokenType = Union[bytes, int]
_VocabAccumulatedIndicatorType = Union[int, Tuple[
    float, float], analyzers.WeightedMeanAndVarCombiner.accumulator_class]
_VocabIndicatorType = Union[float, int, bytes, Tuple[float, float]]

# TODO(b/140645408, b/31727404, b/160207487): Remove this manual fanout when
# it is no longer needed due to framework improvments.
_DEFAULT_COMBINE_GLOBALLY_FANOUT = 10

# Size of a batch for pre-sorting in `VocabularyOrderAndWrite`. It was chosen to
# approximately minimize the stage's wall time for a vocabulary with 100M
# entries.
_PRESORT_BATCH_SIZE = 150000


@ptransform_fn
@beam.typehints.with_input_types(KV[_VocabIndicatorType, _VocabTokenType])
@beam.typehints.with_output_types(Iterable[KV[_VocabIndicatorType,
                                              _VocabTokenType]])
def _BatchAndPreSort(counts, sort_kwargs):  # pylint: disable=invalid-name
  """Batches vocabulary and pre-sorts the batches."""
  # This PTransform aims to partially parallelize vocabulary sorting in
  # `VocabularyOrderAndWrite`. Pre-sorting of batches in a parallel mode with
  # `beam.Map` allows to significantly speed up final single-threaded sorting of
  # a union in `_OrderElementsFn`. This is because `list.sort()` uses an
  # adaptive merge sort that identifies pre-existing order in the union. See
  # https://en.wikipedia.org/wiki/Timsort for more details.
  return (counts
          | 'BatchVocabulary' >> beam.BatchElements(
              min_batch_size=_PRESORT_BATCH_SIZE,
              max_batch_size=_PRESORT_BATCH_SIZE)
          | 'SortBatches' >> beam.Map(lambda b: sorted(b, **sort_kwargs)))  # pylint: disable=unnecessary-lambda


def maybe_add_empty_vocabulary_dummy(
    counts: List[KV[_VocabIndicatorType,
                    _VocabTokenType]], dtype: Union[tf.dtypes.DType, str]
) -> List[KV[_VocabIndicatorType, _VocabTokenType]]:
  """Returns a list with a dummy token if counts list is empty."""
  if not counts:
    return [analyzers.get_empy_vocabulary_dummy_value(dtype)]
  else:
    return counts


def _count_and_token_to_bytes(count: _VocabIndicatorType,
                              token: _VocabTokenType) -> bytes:
  # Converts `token` (bytes) to unicode first as otherwise the result will
  # look like b"1 b'real_string'" in PY3. We convert everything to bytes
  # afterwards to get b'1 real_string'.
  return tf.compat.as_bytes('{} {}'.format(count, tf.compat.as_str_any(token)))


class _OrderElementsFn(beam.DoFn):
  """Sort the vocabulary by either descending frequency count or hash order."""

  def __init__(self, store_frequency, sort_kwargs, input_dtype):
    self._store_frequency = store_frequency
    self._sort_kwargs = sort_kwargs
    self._input_dtype = input_dtype

    # Metrics.
    self._vocab_size = beam.metrics.Metrics.distribution(
        common.METRICS_NAMESPACE, 'vocabulary_size')

  def process(self, element, batched_counts_iter):
    del element
    counts = []
    for c in batched_counts_iter:
      counts.extend(c)
    self._vocab_size.update(len(counts))
    counts = maybe_add_empty_vocabulary_dummy(counts, self._input_dtype)

    counts.sort(**self._sort_kwargs)

    for count, entry in counts:
      if self._store_frequency:
        yield _count_and_token_to_bytes(count, entry)
      else:
        yield entry


@ptransform_fn
@beam.typehints.with_input_types(KV[_VocabIndicatorType, _VocabTokenType])
@beam.typehints.with_output_types(KV[_VocabIndicatorType, _VocabTokenType])
def _ApplyThresholdsAndTopK(  # pylint: disable=invalid-name
    counts,
    frequency_threshold,
    top_k,
    input_dtype,
    info_threshold=float('-inf'),
    key_fn=None):
  """Applies `frequency_threshold` and `top_k` to (count, value) pairs."""
  # TODO(b/117796748): Filter frequency per-key when key feature input enabled.
  # Filter is cheaper than TopK computation and the two commute, so filter
  # first.
  if frequency_threshold > 0 or info_threshold > float('-inf'):

    def filter_by_thresholds(values):
      """Returns True if values are greater than specified thresholds."""
      values, _ = values
      # The values can be a single number (the frequency) or a tuple of the
      # informativeness and the frequency.
      if isinstance(values, tuple):
        informativeness, freq = values
      else:
        informativeness = float('inf')
        freq = values
      if freq >= frequency_threshold and informativeness >= info_threshold:
        return True
      return False

    counts |= ('FilterByThresholds(%s)' % frequency_threshold >>
               beam.Filter(filter_by_thresholds))
  # If a tuple of multiple metrics, flatten to only the first. This is needed
  # for the case the accumulator has tracked informativeness and frequency.
  def flatten_to_single_metric(values):
    value, term = values
    value = value[0] if isinstance(value, tuple) else value
    return value, term

  counts |= 'FlattenToSingleMetric' >> beam.Map(flatten_to_single_metric)

  if input_dtype != tf.string.name:
    counts |= 'EncodeNumericalTerms' >> beam.MapTuple(
        lambda k, v: (k, tf.compat.as_bytes(tf.compat.as_str_any(v))))

  if top_k is not None:
    # TODO(katsiapis): Perhaps enhance Beam's Top to accept an N that can
    # signify "unlimited" and then we can simplify a lot of our code (though
    # that might come at a performance penalty).
    if key_fn:
      def map_key_to_count_and_term(kv, key_fn):
        """Parses key from term with `key_fn` and maps it to count and term."""
        count, term = kv
        # TODO(b/184196242): Ideally we wouldn't be producing numpy.float64
        # counts in the first place, as opposed to casting to float here. See
        # also b/79751861.
        count = float(count) if isinstance(count, np.float64) else count
        key = key_fn(term)
        return key, (count, term)

      counts = (
          counts
          | 'MapKeyToCountAndTerm' >> beam.Map(
              lambda x: map_key_to_count_and_term(x, key_fn))
          | 'CoverageTop(%s)' % top_k >> beam.combiners.Top.LargestPerKey(top_k)
          | 'FlattenCoverageTerms' >> beam.FlatMap(lambda kv: kv[1]))
    else:
      # LINT.IfChange(top_k_impl)
      # Stages that follow this block rely on the sorted order of `Top.Of`'s
      # output and fusion with the `FlattenList`. If changing this part of
      # implementation, either make sure that these hold true or adjust the
      # appropriate arg of `VocabularyOrderAndWrite` node.
      counts = (
          counts
          | 'Top(%s)' % top_k >> beam.combiners.Top.Of(top_k)
          | 'MaybeAddDummy' >> beam.Map(
              maybe_add_empty_vocabulary_dummy, dtype=input_dtype)
          | 'FlattenList' >> beam.FlatMap(lambda lst: lst))
      # LINT.ThenChange(../analyzers.py:input_is_sorted)

  return counts


# Experimental
def sum_labeled_weights(
    accs: List[Tuple[float, List[float]]]) -> Tuple[float, List[float]]:
  """Sums up a collection of labeled-weight tables.

  Args:
    accs: a list of (w, lw) tuples, where w is the total weight (floating point)
      and lw is a list of weights for each label.

  Returns:
    component-wise sum of the inputs in the same format.
  """
  total_weight, labeled_weights = 0., []
  for acc in accs:
    total_weight = total_weight + acc[0]
    accumulator_labeled_weights = acc[1]
    if len(accumulator_labeled_weights) > len(labeled_weights):
      labeled_weights.extend(
          [0.] * (len(accumulator_labeled_weights) - len(labeled_weights)))
    for i in range(len(accumulator_labeled_weights)):
      labeled_weights[i] = labeled_weights[i] + accumulator_labeled_weights[i]
  return (total_weight, labeled_weights)


@common.register_ptransform(analyzer_nodes.VocabularyAccumulate)
@beam.typehints.with_input_types(Tuple[np.ndarray, ...])
@beam.typehints.with_output_types(KV[_VocabTokenType,
                                     _VocabAccumulatedIndicatorType])
class _VocabularyAccumulateImpl(beam.PTransform):
  """Accumulates the unique elements in a PCollection of batches."""

  def __init__(self, operation, extra_args):
    self._vocab_ordering_type = operation.vocab_ordering_type

  def expand(self, inputs):
    pcoll, = inputs

    # Create a PCollection of (count, element) pairs, then iterates over
    # this to create a single element PCollection containing this list of
    # pairs in sorted order by decreasing counts (and by values for equal
    # counts).

    # TODO(b/112916494): Unify the graph in both cases once possible.
    if (self._vocab_ordering_type ==
        _VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION):
      flatten_map_fn = functools.partial(
          _flatten_to_key_and_means_accumulator_list, compute_weighted=True)
      combine_transform = _MutualInformationTransformAccumulate(  # pylint: disable=no-value-for-parameter
          compute_weighted=True)
    elif self._vocab_ordering_type == _VocabOrderingType.MUTUAL_INFORMATION:
      flatten_map_fn = functools.partial(
          _flatten_to_key_and_means_accumulator_list, compute_weighted=False)
      combine_transform = _MutualInformationTransformAccumulate(  # pylint: disable=no-value-for-parameter
          compute_weighted=False)
    elif self._vocab_ordering_type == _VocabOrderingType.WEIGHTED_FREQUENCY:
      flatten_map_fn = _flatten_value_and_weights_to_list_of_tuples
      combine_transform = beam.CombinePerKey(sum)
    elif self._vocab_ordering_type == _VocabOrderingType.WEIGHTED_LABELS:
      flatten_map_fn = _flatten_value_and_labeled_weights_to_list_of_tuples
      # TODO(b/199789764) This returns a type outside of
      # _VocabAccumulatedIndicatorType union -
      # it is only supported by the specific contrib transform (vocab_map).
      # This probably should be moved to vocab_map itself in order to not make
      # every transform to support this type.
      combine_transform = beam.CombinePerKey(sum_labeled_weights)
    else:
      flatten_map_fn = _flatten_value_to_list
      combine_transform = beam.combiners.Count.PerElement()

    if not _ENABLE_BEAM_TYPE_CHECKS:
      combine_transform = combine_transform.with_input_types(Any)
      combine_transform = combine_transform.with_output_types(Any)

    result = (
        pcoll
        | 'FlattenTokensAndMaybeWeightsLabels' >> beam.FlatMap(flatten_map_fn)
        | 'CountPerToken' >> combine_transform)

    return result


@common.register_ptransform(analyzer_nodes.VocabularyCount)
@beam.typehints.with_input_types(KV[_VocabIndicatorType, _VocabTokenType])
@beam.typehints.with_output_types(np.int64)
class _VocabularyCountImpl(beam.PTransform):
  """Counts the total number of tokens in the vocabulary."""

  def __init__(self, operation, extra_args):
    super().__init__()

  def expand(self, inputs):
    pcoll, = inputs

    return (pcoll
            | 'TotalVocabSize' >> beam.combiners.Count.Globally()
            | 'ToInt64' >> beam.Map(np.int64))


@common.register_ptransform(analyzer_nodes.VocabularyMerge)
@beam.typehints.with_input_types(KV[_VocabTokenType,
                                    _VocabAccumulatedIndicatorType])
@beam.typehints.with_output_types(KV[_VocabIndicatorType, _VocabTokenType])
class _VocabularyMergeImpl(beam.PTransform):
  """Merges vocabulary accumulators of (token, num) pairs."""

  def __init__(self, operation, extra_args):
    self._vocab_ordering_type = operation.vocab_ordering_type
    self._use_adjusted_mutual_info = operation.use_adjusted_mutual_info
    self._min_diff_from_avg = operation.min_diff_from_avg

  def expand(self, inputs):
    if (self._vocab_ordering_type ==
        _VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION):
      combine_transform = _MutualInformationTransformMerge(  # pylint: disable=no-value-for-parameter
          self._use_adjusted_mutual_info,
          self._min_diff_from_avg,
          compute_weighted=True)
    elif self._vocab_ordering_type == _VocabOrderingType.MUTUAL_INFORMATION:
      combine_transform = _MutualInformationTransformMerge(  # pylint: disable=no-value-for-parameter
          self._use_adjusted_mutual_info,
          self._min_diff_from_avg,
          compute_weighted=False)
    elif self._vocab_ordering_type == _VocabOrderingType.WEIGHTED_LABELS:
      # TODO(b/199789764) This returns a type outside of
      # _VocabAccumulatedIndicatorType union -
      # it is only supported by the specific contrib transform (vocab_map).
      # This probably should be moved to vocab_map itself in order to not make
      # every transform to support this type.
      combine_transform = beam.CombinePerKey(sum_labeled_weights)
    else:
      combine_transform = beam.CombinePerKey(sum)

    pcoll, = inputs
    if not _ENABLE_BEAM_TYPE_CHECKS:
      combine_transform = combine_transform.with_input_types(Any)
      combine_transform = combine_transform.with_output_types(Any)

    return (pcoll
            | 'MergeCountPerToken' >> combine_transform
            | 'SwapTokensAndCounts' >> beam.KvSwap())


@common.register_ptransform(analyzer_nodes.VocabularyPrune)
@beam.typehints.with_input_types(KV[_VocabIndicatorType, _VocabTokenType])
@beam.typehints.with_output_types(KV[_VocabIndicatorType, _VocabTokenType])
class _VocabularyPruneImpl(beam.PTransform):
  """Order, filters and writes the computed vocabulary file."""

  def __init__(self, operation, extra_args):
    self._top_k = operation.top_k
    self._frequency_threshold = operation.frequency_threshold
    self._informativeness_threshold = operation.informativeness_threshold
    self._coverage_top_k = operation.coverage_top_k
    self._coverage_frequency_threshold = operation.coverage_frequency_threshold
    self._coverage_informativeness_threshold = (
        operation.coverage_informativeness_threshold)
    self._key_fn = operation.key_fn
    self._input_dtype = operation.input_dtype

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

    result = (
        pcoll
        | 'ApplyThresholdsAndTopK' >> (
            _ApplyThresholdsAndTopK(  # pylint: disable=no-value-for-parameter
                self._frequency_threshold, self._top_k, self._input_dtype,
                self._informativeness_threshold, None)))

    if self._key_fn:
      # Note: current APIs do not allow for specifying a coverage
      # informativeness threshold.
      coverage_counts = (
          pcoll | 'ApplyCoverageThresholdAndTopK' >> (
              _ApplyThresholdsAndTopK(  # pylint: disable=no-value-for-parameter
                  self._coverage_frequency_threshold, self._coverage_top_k,
                  self._input_dtype, self._coverage_informativeness_threshold,
                  self._key_fn)))

      result = ((result, coverage_counts)
                | 'MergeStandardAndCoverageArms' >> beam.Flatten()
                | 'RemoveDuplicates' >> beam.Distinct())

    return result


@common.register_ptransform(analyzer_nodes.VocabularyOrderAndWrite)
@beam.typehints.with_input_types(KV[_VocabIndicatorType, _VocabTokenType])
@beam.typehints.with_output_types(np.ndarray)
class _VocabularyOrderAndWriteImpl(beam.PTransform):
  """Writes the computed vocabulary file."""

  def __init__(self, operation, extra_args):
    self._base_temp_dir = extra_args.base_temp_dir
    self._store_frequency = operation.store_frequency
    self._vocab_filename = operation.vocab_filename
    self._fingerprint_shuffle = operation.fingerprint_shuffle
    self._input_dtype = operation.input_dtype
    self._file_format: common_types.VocabularyFileFormatType = (
        operation.file_format)
    self._input_is_sorted = operation.input_is_sorted

  def expand(self, inputs):
    counts, = inputs
    vocabulary_file = os.path.join(self._base_temp_dir, self._vocab_filename)

    def fingerprint_sort_fn(kv):
      # hashlib.sha1 expects bytes
      return hashlib.sha1(kv[1]).digest()

    # TODO(b/62379925) For now force a single file. We can write a sharded
    # file instead.
    # TODO(b/190580668) Here we are relying on fusion (an implementation
    # detail) for the ordering to be maintained when the results are written
    # to disk. This includes fusion of `_OrderElementsFn` and writing PTransform
    # when `_input_is_sorted` is false and fusion of the last stage in
    # `_ApplyThresholdsAndTopK` and writing PTransform when `_input_is_sorted`
    # is true.
    # Perform the write within the body of `OrderElements` maybe
    # `OrderElementsAndWrite`. This would mean using TF IO instead of Beam
    # IO so it's perhaps not great.
    # Alternatively, we could verify the proper ordering after vocabulary is
    # written during `TransformDataset` stage.
    if self._file_format == 'text':
      write_ptransform = 'WriteToText' >> beam.io.WriteToText(
          vocabulary_file, shard_name_template='')
    elif self._file_format == 'tfrecord_gzip':
      # Setting the suffix as .gz ensures that the vocabulary will be written
      # with GZIP compression.
      vocabulary_file = '{}.tfrecord.gz'.format(vocabulary_file)
      write_ptransform = 'WriteToTFRecord' >> beam.io.WriteToTFRecord(
          vocabulary_file, shard_name_template='')

    if self._input_is_sorted:
      assert not self._fingerprint_shuffle
      if self._store_frequency:
        formatted_vocabulary = (
            counts | 'ToBytes' >> beam.MapTuple(_count_and_token_to_bytes))
      else:
        formatted_vocabulary = counts | 'ExtractTokens' >> beam.Values()
    else:
      if self._fingerprint_shuffle:
        sort_kwargs = dict(key=fingerprint_sort_fn)
      else:
        sort_kwargs = dict(reverse=True)  # Largest first.
      batched_counts = counts | 'BatchAndPreSort' >> _BatchAndPreSort(  # pylint: disable=no-value-for-parameter
          sort_kwargs=sort_kwargs)

      formatted_vocabulary = (
          batched_counts.pipeline
          | 'Prepare' >> beam.Create([None])
          | 'OrderElements' >> beam.ParDo(
              _OrderElementsFn(self._store_frequency, sort_kwargs,
                               self._input_dtype),
              batched_counts_iter=beam.pvalue.AsIter(batched_counts)))
    vocab_is_written = formatted_vocabulary | write_ptransform

    # Return the vocabulary path.
    wait_for_vocabulary_transform = (
        counts.pipeline
        | 'CreatePath' >> beam.Create([np.array(vocabulary_file)])
        # Ensure that the analysis returns only after the file is written.
        | 'WaitForVocabularyFile' >> beam.Map(
            lambda x, y: x, y=beam.pvalue.AsIter(vocab_is_written)))
    return (wait_for_vocabulary_transform,)


def _flatten_value_to_list(
    batch_values: Tuple[np.ndarray, ...]) -> Iterable[Any]:
  """Converts an N-D dense or sparse batch to a 1-D list."""
  batch_value, = batch_values

  # TODO(b/36603294): Perhaps obviate the tolist(). It is currently used so
  # that we go to native Python types for more efficient followup
  # processing.
  return batch_value.tolist()


def _flatten_value_and_weights_to_list_of_tuples(
    batch_values: Tuple[np.ndarray, ...]) -> Iterable[Any]:
  """Converts a batch of vocabulary and weights to a list of KV tuples."""
  batch_value, weights = batch_values

  # TODO(b/36603294): Perhaps obviate the tolist(). It is currently used so
  # that we go to native Python types for more efficient followup
  # processing.
  batch_value = batch_value.tolist()
  weights = weights.tolist()
  return zip(batch_value, weights)


# Experimental
def _flatten_value_and_labeled_weights_to_list_of_tuples(
    batch_values: Tuple[np.ndarray, ...]) -> Iterable[Any]:
  """Converts a batch of vocabulary and labeled weights to a list of KV tuples.

  Args:
    batch_values: A row in the batch consists of a value, a (total) weight, and
      a list of weights for each label.
  """
  batch_value, weights, labeled_weights = batch_values

  # TODO(b/36603294): Perhaps obviate the tolist(). It is currently used so
  # that we go to native Python types for more efficient followup
  # processing.
  batch_value = batch_value.tolist()
  weights = weights.tolist()
  labeled_weights = labeled_weights.tolist()
  return zip(batch_value, zip(weights, labeled_weights))


def _make_count_and_weights_means_accumulator(sum_positive, weights_sum_total,
                                              count):
  """Create a WeightedMeanAndVarCombiner according to the parameter values."""
  # TODO(b/165003832): We're going back and forth from lists to numpy here.
  # Can we remove this overhead?
  if weights_sum_total is None:
    mean = np.array(sum_positive) / count
    weight = None
  else:
    mean = np.array(sum_positive) / weights_sum_total
    weight = weights_sum_total / count

  return analyzers.WeightedMeanAndVarCombiner.accumulator_class(
      count=np.array(count),
      mean=mean,
      variance=np.array(0.),  # Variance is not used for vocabularies.
      weight=weight)


def _flatten_to_key_and_means_accumulator_list(batch_values,
                                               compute_weighted=True):
  """Converts a batch of keys, weights, and counts to a list of KV pairs."""
  if compute_weighted:
    keys, total_weights, positive_label_weights, counts = batch_values
    total_weights = total_weights.tolist()
  else:
    keys, positive_label_weights, counts = batch_values
    total_weights = []

  # TODO(b/36603294): Perhaps obviate the tolist(). It is currently used so
  # that we go to native Python types for more efficient followup
  # processing.
  keys = keys.tolist()
  positive_label_weights = positive_label_weights.tolist()
  counts = counts.tolist()

  return zip(keys, [
      _make_count_and_weights_means_accumulator(*batch) for batch in
      itertools.zip_longest(positive_label_weights, total_weights, counts)
  ])


def _clip_probability(p, epsilon=1e-6):
  return np.clip(p, epsilon, 1 - epsilon)


def _calculate_mutual_information_for_feature_value(feature_and_accumulator,
                                                    global_accumulator,
                                                    use_adjusted_mutual_info,
                                                    min_diff_from_avg):
  """Calculates the (possibly adjusted) mutual information of a feature value.

  Used as a measure of relatedness between a single feature value and a label.

  Mutual information is calculated as:
  H(x, y) = (sum(weights) *
             [(P(y|x)*log2(P(y|x)/P(y))) + (P(~y|x)*log2(P(~y|x)/P(~y)))])
  where x is feature and y is label. We use sum(weights) instead of P(x), as
  this makes the mutual information more interpretable.
  If we don't divide by sum(weights), it can be thought of as an adjusted
  weighted count.

  If use_adjusted_mutual_info is True, we use Adjusted Mutual Information (AMI)
  which accounts for relatedness due to chance. AMI is generally calculated as:
  AMI(x, y) = MI(x, y) - EMI(x, y) / (max(H(x), H(y)) - EMI(x, y))
  where x is the feature and y is label. Here, we leave off the normalization
  and only subtract expected mutual information (EMI) from mutual information.
  The calculation is based on the following paper:

  Vinh, N. X.; Epps, J.; Bailey, J. (2009). "Information theoretic measures for
  clusterings comparison". Proceedings of the 26th Annual International Confere
  nce on Machine Learning - ICML '09. p. 1.
  doi:10.1145/1553374.1553511. ISBN 9781605585161.

  Short summary can be found in the Wikipedia link:
  https://en.wikipedia.org/wiki/Adjusted_mutual_information

  Args:
    feature_and_accumulator: A tuple of the form:
      (feature, WeightedMeanAndVarCombiner.accumulator_class) where: `feature`
        is the single token in the vocabulary for which (possibly adjusted)
        mutual information with the label is being computed. `mean` is the
        weighted mean positive for each label value given x. `count` is the
        count of weights for a feature. `weight` is the mean of the weights for
        a feature.
    global_accumulator: A WeightedMeanAndVarCombiner.accumulator_class where:
      `mean` is the weighted mean positive for each label value for all
      features. `count` is the count for all features. `weight` is the mean of
      the weights for all features.
    use_adjusted_mutual_info: If set to True, use adjusted mutual information.
    min_diff_from_avg: A regularization parameter that pushes low MI/AMI towards
      zero. The Mutual information of a feature x label pair will be adjusted to
      zero whenever the absolute difference the weight and the expected
      (average) weight is lower than min_diff_from_average.

  Returns:
    A tuple of:
      The feature value
      The mutual information with the label. If use_adjusted_mutual_info, this
        is the mutual information - the expected mutual information, otherwise
        it is the raw mutual information.
      The expected mutual information (EMI) if use_adjusted_mutual_info is
        True, otherwise NaN.
      The total weighted sum for the feature value.
  """
  # Compute the frequency of each label value.
  global_label_counts = (
      global_accumulator.mean * global_accumulator.weight *
      global_accumulator.count)
  feature_value, current_accumulator = feature_and_accumulator
  total_label_counts = sum(global_label_counts)
  n = global_accumulator.count * global_accumulator.weight
  # TODO(b/168469757): Consider raising here once b/168469757 is resolved.
  if round(total_label_counts) != round(n):
    logging.warn(
        'Weighted label sum (%s) != total weighted count (%s), label means=%s',
        total_label_counts, n, global_accumulator.mean)
  if n == 0:
    return (feature_value, (float('NaN'), float('NaN'), 0))

  mutual_information = 0
  expected_mutual_information = 0 if use_adjusted_mutual_info else None
  x_i = (current_accumulator.count * current_accumulator.weight)
  # If x_i == n, the feature is a constant and thus has no information.
  if round(x_i) == round(n):
    return feature_value, (0, 0, x_i)
  if round(x_i) > round(n):
    raise ValueError(
        'Frequency of token {} higher than number of records {} > {}'.format(
            feature_value, x_i, n) +
        ' This likely means you have provided tft.vocabulary with input that'
        ' has repeated tokens per row, rather than a set representation.')
  for label_ix in range(len(global_label_counts)):
    y_i = global_label_counts[label_ix]
    if y_i == 0:
      continue
    local_mean = 0
    if label_ix < len(current_accumulator.mean):
      local_mean = current_accumulator.mean[label_ix]
    n_i = (
        _clip_probability(local_mean) * current_accumulator.weight *
        current_accumulator.count)
    diff_from_avg = (x_i * y_i / n) - n_i
    if abs(diff_from_avg) < min_diff_from_avg:
      continue
    mutual_information += (
        info_theory.calculate_partial_mutual_information(n_i, x_i, y_i, n))
    if use_adjusted_mutual_info:
      expected_mutual_information += (
          info_theory.calculate_partial_expected_mutual_information(
              n, x_i, y_i))

  if use_adjusted_mutual_info:
    # TODO(b/127366670): Consider implementing the normalization step as per
    # AMI(x, y) = MI(x, y) - EMI(x, y) / (max(H(x), H(y)) - EMI(x, y))
    return (feature_value, (mutual_information - expected_mutual_information,
                            expected_mutual_information, x_i))
  else:
    return (feature_value, (mutual_information, float('NaN'), x_i))


@ptransform_fn
@beam.typehints.with_input_types(
    KV[_VocabTokenType, analyzers.WeightedMeanAndVarCombiner.accumulator_class])
@beam.typehints.with_output_types(KV[_VocabTokenType,
                                     _VocabAccumulatedIndicatorType])
def _MutualInformationTransformAccumulate(pcol, compute_weighted=True):  # pylint: disable=invalid-name
  """Accumulates information needed for mutual information computation."""
  return (pcol | 'VocabCountPerLabelPerTokenAccumulate' >> beam.CombinePerKey(
      _WeightedMeanCombineFn(
          output_shape=(None,), compute_weighted=compute_weighted)))


def _extract_sentinels(kv):
  """Separate out label sentinel accumulators from vocab accumulators.

  To keep track of the frequencies of label values, we store global label
  frequencies associated with a special sentinel value. These are accumulated
  just like other vocabulary tokens, but must be separated out before computing
  mutual information.

  Args:
    kv: tuple of key, accumulator

  Yields:
    A Beam TaggedOutout separating the sentinel and regular tokens.
  """
  token, _ = kv
  if (token == tf_utils.GLOBAL_Y_COUNT_SENTINEL_STRING or
      token == tf_utils.GLOBAL_Y_COUNT_SENTINEL_INT):
    # Throw away the sentinel token, since it's not needed.
    yield beam.pvalue.TaggedOutput('global', kv[1])
  else:
    yield beam.pvalue.TaggedOutput('feature', kv)


@ptransform_fn
@beam.typehints.with_input_types(KV[_VocabTokenType,
                                    _VocabAccumulatedIndicatorType])
@beam.typehints.with_output_types(KV[_VocabTokenType, Tuple[float, float]])
def _MutualInformationTransformMerge(  # pylint: disable=invalid-name
    pcol, use_adjusted_mutual_info, min_diff_from_avg, compute_weighted):
  """Computes mutual information for each key using the given accumulators."""
  feature_accumulator_pcol = (
      pcol | 'VocabCountPerLabelPerTokenMerge' >> beam.CombinePerKey(
          _WeightedMeanCombineFn(
              output_shape=(None,), compute_weighted=compute_weighted)))

  accumulators_by_feature, global_accumulator = (
      feature_accumulator_pcol
      | 'ExtractSentinels' >> beam.FlatMap(_extract_sentinels).with_outputs(
          'feature', 'global'))
  if min_diff_from_avg is None:
    min_diff_from_avg = (
        global_accumulator | 'AutoMinDiffFromAvg' >>
        beam.Map(lambda acc: analyzers.calculate_recommended_min_diff_from_avg(  # pylint: disable=g-long-lambda
            acc.count * acc.weight)))
    min_diff_from_avg = beam.pvalue.AsSingleton(min_diff_from_avg)

  def _extract_merged_values(term, results):
    """Returns the key and tuple of (mutual information, frequency)."""
    # Ignore the second value, which is the Expected Mutual Info.
    (mi, _, frequency) = results
    return term, (mi, frequency)

  return (accumulators_by_feature
          | 'CalculateMutualInformationPerToken' >> beam.Map(
              _calculate_mutual_information_for_feature_value,
              beam.pvalue.AsSingleton(global_accumulator),
              use_adjusted_mutual_info=use_adjusted_mutual_info,
              min_diff_from_avg=min_diff_from_avg)
          | beam.MapTuple(_extract_merged_values))


class _WeightedMeanCombineFn(beam.CombineFn):
  """_WeightedMeanCombineFn calculates total count and weighted means."""

  def __init__(self, output_shape, compute_weighted=True):
    self._combiner = analyzers.WeightedMeanAndVarCombiner(
        np.float32,
        output_shape=output_shape,
        compute_variance=False,
        compute_weighted=compute_weighted)

  def create_accumulator(self):
    """Create an accumulator with all zero entries."""
    return self._combiner.create_accumulator()

  def add_input(self, accumulator, batch_values):
    """Composes an accumulator from batch_values and calls merge_accumulators.

    Args:
      accumulator: The `WeightedMeanAndVarCombiner.accumulator_class` computed
        so far.
      batch_values: A `WeightedMeanAndVarCombiner.accumulator_class` for the
        current batch.

    Returns:
      A `WeightedMeanAndVarCombiner.accumulator_class` which is accumulator and
      batch_values
      combined.
    """
    return self._combiner.add_input(accumulator, batch_values)

  def merge_accumulators(self, accumulators):
    """Merges several `WeightedMeanAndVarCombiner.accumulator_class`s.

    Args:
      accumulators: A list of `WeightedMeanAndVarCombiner.accumulator_class`s
        and/or Nones.

    Returns:
      The sole merged `WeightedMeanAndVarCombiner.accumulator_class`.
    """
    return self._combiner.merge_accumulators(accumulators)

  def extract_output(self, accumulator):
    """Returns the accumulator as the output.

    Args:
      accumulator: the final `WeightedMeanAndVarCombiner.accumulator_class`
        value.

    Returns:
     The accumulator which could be None.
    """
    return self._combiner.extract_output(accumulator)


class _CombinerWrapper(beam.CombineFn):
  """Class to wrap a analyzer_nodes.Combiner as a beam.CombineFn."""

  def __init__(self,
               combiner,
               is_combining_accumulators):
    """Init method for _CombinerWrapper.

    Args:
      combiner: A `analyzer_nodes.Combiner` object used to combine.
      is_combining_accumulators: A bool which indicates whether this is
        combining single or batched inputs, or already accumulated objects. In
        the former case, output of the CombineFn is an accumulator, whereas
        in the latter case, output is extracted from the combined accumulators
        using the combiner's extract_output.
    """
    self._combiner = combiner
    self._is_combining_accumulators = is_combining_accumulators

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

  def compact(self, accumulator):
    return self._combiner.compact(accumulator)

  def extract_output(self, accumulator):
    if self._is_combining_accumulators:
      return self._combiner.extract_output(accumulator)
    return accumulator


@beam.typehints.with_input_types(Union[Dict[str, Any], Tuple[str, Any]])
@beam.typehints.with_output_types(Dict[str, Any])
class _PackedCombinerWrapper(beam.combiners.TupleCombineFn):
  """Class to wrap a analyzer_nodes.Combiner as a beam.CombineFn.

  PackedCombineWrapper is used for combining input batches as well as
  accumulators. When combining input batches, the input is a PCollection of
  Dicts from feature keys to numpy arrays. When combining accumulators, the
  input is a PCollection of tuples (key, accumulator), where the key represents
  the individual combine label that is being packed.
  """

  def __init__(self,
               combiner_ops,
               is_combining_accumulators):
    """Init method for _PackedCombinerWrapper.

    Args:
      combiner_ops: A List `analysis_graph_builder._CombinerOpWrapper` objects.
      is_combining_accumulators: A bool which indicates whether this is
        combining single or batched inputs, or already accumulated objects.
    """
    super().__init__(*[
        _CombinerWrapper(c.combiner, is_combining_accumulators)
        for c in combiner_ops
    ])
    self._is_combining_accumulators = is_combining_accumulators
    if self._is_combining_accumulators:
      # When combining accumulators, we expect to have only a single key which
      # represents the label of the individual combine.
      for op in combiner_ops:
        assert len(op.keys) == 1
      self._combiner_label_to_index = {
          op.keys[0]: index for index, op in enumerate(combiner_ops)}
    else:
      self._combiner_keys = [c.keys for c in combiner_ops]
    self._combiner_labels = [c.label for c in combiner_ops]

  def add_input(self, accumulator, element):
    if self._is_combining_accumulators:
      key, value = element
      index = self._combiner_label_to_index[key]
      accumulator[index] = self._combiners[index].add_input(
          accumulator[index], value)
      return accumulator
    else:
      return super().add_input(
          accumulator,
          [tuple(element[key] for key in keys) for keys in self._combiner_keys])

  def extract_output(self, accumulator):
    outputs = super().extract_output(accumulator)
    return {
        combiner_label: output
        for combiner_label, output in zip(self._combiner_labels, outputs)
    }


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
    (key, args) pairs where args is a list of ndarrays.

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


def _make_strictly_increasing_boundaries_rows(
    boundary_matrix: np.ndarray) -> np.ndarray:
  """Converts a 2-d array of increasing rows to strictly increasing rows.

  Args:
    boundary_matrix: A 2-d np.array where each row is increasing.

  Returns:
    A 2-d np.array of the same size as `boundary_matrix` where each row is
    strictly increasing.
  """
  epsilon = (1e-6 *
             np.expand_dims(boundary_matrix[:, -1] - boundary_matrix[:, 0], 1))

  # Make sure every value in epsilon is positive.
  epsilon[epsilon <= 0] = 1e-6

  deltas = np.diff(boundary_matrix, axis=1)
  corrected_deltas = np.maximum(deltas, epsilon)

  # Reconstruct the matrix with corrected deltas without the 1st column.
  corrected_boundaries = (
      np.cumsum(corrected_deltas, axis=1) +
      np.expand_dims(boundary_matrix[:, 0], 1))

  # Reinsert the 1st column.
  return np.insert(corrected_boundaries, 0, boundary_matrix[:, 0], axis=1)


def _join_boundary_rows(
    boundary_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Joins boundaries per key, by scaling and shifting them.

  This returns a new list of boundaries which is composed from the given 2-d
  array. For each row we compute a scale factor, and a shift value which are
  used to compute the transformed boundaries, and should be used to transform
  a value before its bucket is computed.

  Neighboring key bucket boundaries have their adjacent boundaries merged into
  one.

  Args:
    boundary_matrix: A 2-d np.array where each row is a list of boundaries for a
      certain key.

  Returns:
    A 4-tuple of (boundaries, scale, shift, num_buckets).
    The returned boundaries is a 1-d np.array of size:
    ((num_buckets - 2) * num_keys) + 1
  """
  boundary_matrix = _make_strictly_increasing_boundaries_rows(boundary_matrix)

  num_buckets = np.array(boundary_matrix.shape[1] + 1, dtype=np.int64)

  # Min boundary for each row.
  min_boundary = np.min(boundary_matrix, axis=1)

  # Max boundary for each row.
  max_boundary = np.max(boundary_matrix, axis=1)

  boundary_difference = max_boundary - min_boundary
  scale = np.divide(
      1.0,
      boundary_difference,
      out=np.ones_like(boundary_difference),
      where=boundary_difference != 0)

  # Shifts what would shift values so that when applied to min[key_id] we
  # get: min[key_id] * scale[key_id] + shift[key_id] = key_id
  # Therefore shift is defined as:
  # shift[key_id] = key_id -  min[key_id] * scale[key_id]
  shift = np.arange(scale.size, dtype=np.float32) - min_boundary * scale

  scaled_buckets = (
      boundary_matrix[:, 1:] * np.expand_dims(scale, axis=1) +
      np.expand_dims(shift, axis=1))
  boundaries = np.insert(scaled_buckets.flatten(), 0, 0.)

  return boundaries, scale, shift, num_buckets


@common.register_ptransform(
    analyzer_nodes.ScaleAndFlattenPerKeyBucketBouandaries)
class _ScaleAndFlattenPerKeyBucketBouandariesImpl(beam.PTransform):
  """Combines boundaries per-key to a single list of boundaries."""

  _OUTPUT_TAGS = ('boundaries', 'scale_factor_per_key', 'shift_per_key',
                  'num_buckets')

  def __init__(self, operation, extra_args):
    self._dtype = operation.output_tensor_dtype
    self._name = operation.label

  def _transform_boundaries(self, boundary_matrix):
    results = _join_boundary_rows(boundary_matrix)
    assert len(self._OUTPUT_TAGS) == len(results)
    return [
        beam.pvalue.TaggedOutput(tag, value)
        for tag, value in zip(self._OUTPUT_TAGS, results)
    ]

  def expand(self, inputs):
    pcoll, = inputs
    output_dict = pcoll | beam.FlatMap(
        self._transform_boundaries).with_outputs(*self._OUTPUT_TAGS)
    return tuple(output_dict[key] for key in self._OUTPUT_TAGS)


@common.register_ptransform(analyzer_nodes.PackedCombineAccumulate)
@beam.typehints.with_input_types(Dict[str, Any])
@beam.typehints.with_output_types(Dict[str, Any])
class _InitialAccumulatePackedCombineImpl(beam.PTransform):
  """Implement an packed analyzer accumulate based on a Combine."""

  def __init__(self, operation, extra_args):
    self._combiners = operation.combiners

  def expand(self, inputs):
    pcoll, = inputs
    # We specify a fanout so that the packed combiner doesn't exhibit stragglers
    # during the 'reduce' phase when we have a lot of combine analyzers packed.
    fanout = int(math.ceil(math.sqrt(len(self._combiners))))
    fanout = max(_DEFAULT_COMBINE_GLOBALLY_FANOUT, fanout)
    return (
        pcoll
        | 'InitialPackedCombineGlobally' >> beam.CombineGlobally(
            _PackedCombinerWrapper(
                self._combiners,
                is_combining_accumulators=False
            )
        ).with_fanout(fanout)
        | 'Count' >>
        common.IncrementCounter('num_packed_accumulate_combiners'))


@common.register_ptransform(analyzer_nodes.PackedCombineMerge)
@beam.typehints.with_input_types(Tuple[str, Any])
@beam.typehints.with_output_types(Dict[str, Any])
class _MergeAccumulatorsPackedCombineImpl(beam.PTransform):
  """Implement an packed analyzer merge based on a Combine."""

  def __init__(self, operation, extra_args):
    self._combiners = operation.combiners

  def expand(self, inputs):
    pcoll, = inputs

    return (
        pcoll
        | 'MergePackedCombinesGlobally' >> beam.CombineGlobally(
            _PackedCombinerWrapper(
                self._combiners,
                is_combining_accumulators=True))
        | 'Count' >>
        common.IncrementCounter('num_packed_merge_combiners'))


# TODO(zoyahav): Share logic with _InitialAccumulatePackedCombineImpl.
@common.register_ptransform(analyzer_nodes.CacheableCombineAccumulate)
@beam.typehints.with_input_types(Tuple[np.ndarray, ...])
class _InitialAccumulateCombineImpl(beam.PTransform):
  """Implement an analyzer based on a Combine."""

  def __init__(self, operation, extra_args):
    self._combiner = operation.combiner
    self._num_outputs = operation.num_outputs
    self._name = operation.label

  def expand(self, inputs):
    pcoll, = inputs

    return (pcoll
            | 'InitialCombineGlobally' >> beam.CombineGlobally(
                _CombinerWrapper(
                    self._combiner,
                    is_combining_accumulators=False)).with_fanout(
                        _DEFAULT_COMBINE_GLOBALLY_FANOUT))


@common.register_ptransform(analyzer_nodes.CacheableCombineMerge)
class _MergeAccumulatorsCombineImpl(beam.PTransform):
  """Implement an analyzer based on a Combine."""

  def __init__(self, operation, extra_args):
    self._combiner = operation.combiner
    self._name = operation.label

  def expand(self, inputs):
    pcoll, = inputs

    return (
        pcoll
        | 'MergeCombinesGlobally' >> beam.CombineGlobally(
            _CombinerWrapper(
                self._combiner,
                is_combining_accumulators=True)))


@common.register_ptransform(analyzer_nodes.CacheableCombinePerKeyAccumulate)
class _InitialAccumulateCombinePerKeyImpl(beam.PTransform):
  """Implement an analyzer based on a CombinePerKey."""

  def __init__(self, operation, extra_args):
    self._combiner = operation.combiner

  def expand(self, inputs):
    pcoll, = inputs
    return (pcoll
            | 'SplitByKey' >> beam.FlatMap(_split_inputs_by_key)
            | 'CombinePerKey' >> beam.CombinePerKey(
                _CombinerWrapper(
                    self._combiner,
                    is_combining_accumulators=False)))


@common.register_ptransform(analyzer_nodes.CacheableCombinePerKeyMerge)
class _MergeAccumulatorsCombinePerKeyImpl(beam.PTransform):
  """Implement an analyzer based on a CombinePerKey."""

  def __init__(self, operation, extra_args):
    self._combiner = operation.combiner

  def expand(self, inputs):
    pcoll, = inputs
    return (
        pcoll
        | 'MergeCombinePerKey' >> beam.CombinePerKey(
            _CombinerWrapper(
                self._combiner,
                is_combining_accumulators=True)))


@common.register_ptransform(analyzer_nodes.CacheableCombinePerKeyFormatKeys)
class _CombinePerKeyFormatKeysImpl(beam.PTransform):
  """An analyzer that formats output for the non-stored per-key case."""

  def __init__(self, operation, extra_args):
    self._combiner = operation.combiner

  def expand(self, inputs):
    pcoll, = inputs
    output_keys = (
        ['key'
        ] + [str(i) for i in range(len(self._combiner.output_tensor_infos()))])
    outputs_tuple = (
        pcoll
        | 'ToList' >> beam.combiners.ToList()
        | 'MergeByKey' >> beam.FlatMap(_merge_outputs_by_key, [
            info.dtype for info in self._combiner.output_tensor_infos()
        ]).with_outputs(*output_keys))
    return tuple(outputs_tuple[key] for key in output_keys)


@common.register_ptransform(analyzer_nodes.CacheableCombinePerKeyFormatLarge)
class _CombinePerKeyFormatLargeImpl(beam.PTransform):
  """An analyzer that formats output before writing to file for per-key case."""

  def __init__(self, operation, extra_args):
    super().__init__()

  def expand(self, inputs):
    to_str = tf.compat.as_str_any
    pcoll, = inputs
    return (
        pcoll
        | 'EncodeValueAndSwapWithKey' >> beam.MapTuple(
            lambda k, v: (to_str(','.join(map(to_str, v))), k)))


@common.register_ptransform(analyzer_nodes.PTransform)
class _PTransformImpl(beam.PTransform):
  """Implements a registered PTransform node by passing through the inputs."""

  def __init__(self, operation, extra_args):
    self._ptransform = operation.ptransform
    if isinstance(self._ptransform, experimental.PTransformAnalyzer):
      self._ptransform.base_temp_dir = common.get_unique_temp_path(
          extra_args.base_temp_dir)

  def expand(self, inputs):
    pcoll, = inputs
    return pcoll | self._ptransform


@common.register_ptransform(analyzer_nodes.EncodeCache)
@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(bytes)
class _EncodeCacheImpl(beam.PTransform):
  """A PTransform that encodes cache entries."""

  def __init__(self, operation, extra_args):
    self._coder = operation.coder

  def expand(self, inputs):
    pcoll, = inputs

    return (pcoll
            | 'Encode' >> beam.Map(self._coder.encode_cache)
            | 'Count' >> common.IncrementCounter('cache_entries_encoded'))


@common.register_ptransform(analyzer_nodes.DecodeCache)
@beam.typehints.with_input_types(beam.pvalue.PBegin)
@beam.typehints.with_output_types(Any)
class _DecodeCacheImpl(beam.PTransform):
  """A PTransform method that extracts and decodes a cache object."""

  def __init__(self, operation, extra_args):
    self._cache_pcoll = (
        extra_args.cache_pcoll_dict[operation.dataset_key][operation.cache_key])
    self._coder = operation.coder

  def expand(self, pbegin):
    del pbegin  # unused

    return (self._cache_pcoll
            | 'Decode' >> beam.Map(self._coder.decode_cache)
            | 'Count' >> common.IncrementCounter('cache_entries_decoded'))


@common.register_ptransform(analyzer_nodes.AddKey)
@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(Tuple[str, Any])
class _AddKeyImpl(beam.PTransform):
  """Implements AddKey."""

  def __init__(self, operation, extra_args):
    del extra_args  # unused
    self._key = operation.key

  def expand(self, inputs):
    pcoll, = inputs
    return pcoll | 'AddKey' >> beam.Map(lambda value: (self._key, value))


_FlattenListsItemType = TypeVariable('_FlattenListsItemType')


@common.register_ptransform(analyzer_nodes.FlattenLists)
@beam.typehints.with_input_types(List[_FlattenListsItemType])
@beam.typehints.with_output_types(_FlattenListsItemType)
class _FlattenListsImpl(beam.PTransform):
  """PTransform to flatten a PCollection of lists."""

  def __init__(self, operation, extra_args):
    del operation, extra_args  # unused

  def expand(self, inputs):
    pcoll, = inputs
    return pcoll | 'FlattenLists' >> beam.FlatMap(lambda x: x)


@common.register_ptransform(analyzer_nodes.ExtractCombineMergeOutputs)
@common.register_ptransform(analyzer_nodes.ExtractPackedCombineMergeOutputs)
class _ExtractOutputImpl(beam.PTransform):
  """Implements ExtractOutputs."""

  def __init__(self, operation, extra_args):
    del extra_args  # unused
    self._num_outputs = operation.num_outputs

  def expand(self, inputs):
    pcoll, = inputs
    def extract_outputs(outputs, num_outputs):
      if len(outputs) != num_outputs:
        raise ValueError(
            'Analyzer has {} outputs but its implementation produced {} '
            'values'.format(num_outputs, len(outputs)))
      for i, output in enumerate(outputs):
        yield beam.pvalue.TaggedOutput(str(i), output)

    output_keys = [str(i) for i in range(self._num_outputs)]
    outputs_tuple = (
        pcoll |
        'ExtractOutputs' >> beam.FlatMap(
            extract_outputs, self._num_outputs).with_outputs(*output_keys))
    return tuple(outputs_tuple[key] for key in output_keys)
