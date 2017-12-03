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

from apache_beam.typehints import Any
from apache_beam.typehints import KV
from apache_beam.typehints import List
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types

import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform.beam import common
from tensorflow.contrib.boosted_trees.python.ops import quantile_ops
from tensorflow.python.ops import resources

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


@with_input_types(np.ndarray)
@with_output_types(List[Any])
class _AnalyzerImpl(beam.PTransform):
  """PTransform that implements a given analyzer.

  _AnalyzerImpl accepts a PCollection where each element is a batch of values,
  and returns a PCollection containing a single element representing which is a
  list of values for each output, where each value can be converted to an
  ndarray via np.asarray.

  _AnalyzerImpl dispatches to an implementation transform, with the same
  signature as _AnalyzerImpl.
  """

  def __init__(self, spec, temp_assets_dir):
    self._spec = spec
    self._temp_assets_dir = temp_assets_dir

  def expand(self, pcoll):
    # pylint: disable=protected-access
    if isinstance(self._spec, analyzers._UniquesSpec):
      return pcoll | _UniquesAnalyzerImpl(self._spec, self._temp_assets_dir)
    elif isinstance(self._spec, analyzers._QuantilesSpec):
      return pcoll | _QuantilesAnalyzerImpl(self._spec)
    elif isinstance(self._spec, analyzers.CombinerSpec):
      return pcoll | beam.CombineGlobally(
          _CombineFnWrapper(self._spec)).without_defaults()
    else:
      raise NotImplementedError(self._spec.__class__)


def _flatten_value_to_list(batch):
  """Converts an N-D dense or sparse batch to a 1-D list."""
  # Ravel for flattening and tolist so that we go to native Python types
  # for more efficient followup processing.
  #
  return batch.ravel().tolist()


@with_input_types(np.ndarray)
@with_output_types(List[Any])
class _UniquesAnalyzerImpl(beam.PTransform):
  """Saves the unique elements in a PCollection of batches."""

  def __init__(self, spec, temp_assets_dir):
    assert isinstance(spec, analyzers._UniquesSpec)  # pylint: disable=protected-access
    self._spec = spec
    self._temp_assets_dir = temp_assets_dir

  def expand(self, pcoll):
    top_k = self._spec.top_k
    frequency_threshold = self._spec.frequency_threshold
    assert top_k is None or top_k >= 0
    assert frequency_threshold is None or frequency_threshold >= 0

    # Creates a PCollection of (count, element) pairs, then iterates over
    # this to create a single element PCollection containing this list of
    # pairs in sorted order by decreasing counts (and by values for equal
    # counts).
    counts = (
        pcoll
        | 'FlattenValueToList' >> beam.Map(_flatten_value_to_list)
        | 'CountWithinList' >>
        # Specification of with_output_types allows for combiner optimizations.
        (beam.FlatMap(lambda lst: six.iteritems(collections.Counter(lst))).
         with_output_types(KV[common.PRIMITIVE_TYPE, int]))
        | 'CountGlobally' >> beam.CombinePerKey(sum))

    counts = (
        counts
        | 'FilterProblematicStrings' >> beam.Filter(
            lambda kv: kv[0] and '\n' not in kv[0] and '\r' not in kv[0])
        | 'SwapElementsAndCounts' >> beam.KvSwap())

    # Filter is cheaper than TopK computation and the two commute, so
    # filter first.
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
    def order_by_decreasing_counts(ignored, counts_iter, store_frequency):
      """Sort the vocabulary by frequency count."""
      del ignored
      counts = list(counts_iter)
      if not counts:
        counts = [(1, '49d0cd50-04bb-48c0-bc6f-5b575dce351a')]
      counts.sort(reverse=True)  # Largest first.
      if store_frequency:
        # Returns ['count1 element1', ... ]
        return ['{} {}'.format(count, element) for count, element in counts]
      else:
        return [element for _, element in counts]

    vocabulary_file = os.path.join(self._temp_assets_dir,
                                   self._spec.vocab_filename)
    vocab_is_written = (
        pcoll.pipeline
        | 'Prepare' >> beam.Create([None])
        | 'OrderByDecreasingCounts' >> beam.FlatMap(
            order_by_decreasing_counts,
            counts_iter=beam.pvalue.AsIter(counts),
            store_frequency=self._spec.store_frequency)
        | 'WriteToFile' >> beam.io.WriteToText(vocabulary_file,
                                               shard_name_template=''))
    # Return the vocabulary path.
    wait_for_vocabulary_transform = (
        pcoll.pipeline
        | 'CreatePath' >> beam.Create([[vocabulary_file]])
        # Ensure that the analysis returns only after the file is written.
        | 'WaitForVocabularyFile' >> beam.Map(
            lambda x, y: x, y=beam.pvalue.AsIter(vocab_is_written)))
    return wait_for_vocabulary_transform


@with_input_types(np.ndarray)
@with_output_types(List[Any])
class _ComputeQuantiles(beam.CombineFn):
  """Computes quantiles on the PCollection.

  This implementation is based on go/squawd.
  For additional details on the algorithm, such as streaming and summary,
  see also http://web.cs.ucla.edu/~weiwang/paper/SSDBM07_2.pdf
  """

  def __init__(self, num_quantiles, epsilon, serialized_tf_config=None):
    self._num_quantiles = num_quantiles
    self._epsilon = epsilon
    self._serialized_tf_config = serialized_tf_config

    # _stamp_token is used to commit the state of the qaccumulator. In
    # this case, the qaccumulator state is completely returned and stored
    # as part of quantile_state/summary in the combiner fn (i.e the summary is
    # extracted and stored outside the qaccumulator). So we don't use
    # the timestamp mechanism to signify progress in the qaccumulator state.
    self._stamp_token = 0
    # Represents an empty summary. This could be changed to a tf.constant
    # implemented by the quantile ops library.
    self._empty_summary = None

    # Create a new session with a new graph for quantile ops.
    self._session = tf.Session(
        graph=tf.Graph(),
        config=_maybe_deserialize_tf_config(serialized_tf_config))
    with self._session.graph.as_default():
      with self._session.as_default():
        self._qaccumulator = quantile_ops.QuantileAccumulator(
            init_stamp_token=self._stamp_token,
            num_quantiles=self._num_quantiles,
            epsilon=self._epsilon,
            name='qaccumulator')
        resources.initialize_resources(resources.shared_resources()).run()

  def __reduce__(self):
    return _ComputeQuantiles, (self._num_quantiles,
                               self._epsilon, self._serialized_tf_config)

  def create_accumulator(self):
    return self._empty_summary

  def add_input(self, summary, next_input):
    next_input = _flatten_value_to_list(next_input)
    with self._session.graph.as_default():
      update = self._qaccumulator.add_summary(
          stamp_token=self._stamp_token,
          column=[next_input],
          # All weights are equal, and the weight vector is the
          # same length as the input.
          example_weights=([[1] * len(next_input)]))

      if summary is not self._empty_summary:
        self._session.run(
            self._qaccumulator.add_prebuilt_summary(
                stamp_token=self._stamp_token,
                summary=tf.constant(summary)))

      self._session.run(update)

      # After the flush_summary, qaccumulator will not contain any
      # uncommitted information that represents the input. Instead all the
      # digested information is returned as 'summary'. Many such summaries
      # will be combined by merge_accumulators().
      return self._session.run(
          self._qaccumulator.flush_summary(
              stamp_token=self._stamp_token,
              next_stamp_token=self._stamp_token))

  def merge_accumulators(self, summaries):
    if summaries is self._empty_summary:
      return self._empty_summary

    with self._session.graph.as_default():
      summary_placeholder = tf.placeholder(tf.string)
      add_summary = self._qaccumulator.add_prebuilt_summary(
          stamp_token=self._stamp_token,
          summary=summary_placeholder)
      for summary in summaries:
        self._session.run(add_summary, {summary_placeholder: summary})

      # Compute new summary.
      # All relevant state about the input is captured by 'summary'
      # (see comment at the end of add_input()).
      return self._session.run(
          self._qaccumulator.flush_summary(
              stamp_token=self._stamp_token,
              next_stamp_token=self._stamp_token))

  def extract_output(self, summary):
    if summary is self._empty_summary:
      return [[[]]]

    # All relevant state about the input is captured by 'summary'
    # (see comment in add_input() and merge_accumulators()).
    with self._session.graph.as_default():
      self._session.run(
          self._qaccumulator.add_prebuilt_summary(
              stamp_token=self._stamp_token, summary=tf.constant(summary)))
      self._session.run(
          self._qaccumulator.flush(
              stamp_token=self._stamp_token,
              next_stamp_token=self._stamp_token))
      are_ready_flush, buckets = (
          self._qaccumulator.get_buckets(stamp_token=self._stamp_token))
      buckets, _ = self._session.run([buckets, are_ready_flush])

    return [[buckets]]


@with_input_types(np.ndarray)
@with_output_types(List[Any])
class _QuantilesAnalyzerImpl(beam.PTransform):
  """Computes the quantile buckets in a PCollection of batches."""

  def __init__(self, spec):
    assert isinstance(spec, analyzers._QuantilesSpec)  # pylint: disable=protected-access
    self._spec = spec

  def expand(self, pcoll):
    serialized_tf_config = _DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER.get(
        pcoll.pipeline.runner)
    return (pcoll
            | 'ComputeQuantiles' >> beam.CombineGlobally(
                _ComputeQuantiles(
                    num_quantiles=self._spec.num_buckets,
                    epsilon=self._spec.epsilon,
                    serialized_tf_config=serialized_tf_config)))


@with_input_types(np.ndarray)
@with_output_types(List[Any])
class _CombineFnWrapper(beam.CombineFn):
  """Class to wrap a analyzers._CombinerSpec as a beam.CombineFn."""

  def __init__(self, spec):
    self._spec = spec

  def create_accumulator(self):
    return self._spec.create_accumulator()

  def add_input(self, accumulator, next_input):
    return self._spec.add_input(accumulator, next_input)

  def merge_accumulators(self, accumulators):
    return self._spec.merge_accumulators(accumulators)

  def extract_output(self, accumulator):
    return self._spec.extract_output(accumulator)
