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

from apache_beam.typehints import KV
from apache_beam.typehints import List
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types

import numpy as np
import six
from tensorflow_transform import analyzers
from tensorflow_transform.beam import common


@with_input_types(List[np.ndarray])
@with_output_types(List[np.ndarray])
class _AnalyzerImpl(beam.PTransform):
  """PTransform that implements a given analyzer.

  _AnalyzerImpl accepts a PCollection where each element is a list of ndarrays.
  Each element in this list contains a batch of values for the corresponding
  input tensor of the analyzer. _AnalyzerImpl returns a PCollection containing a
  single element which is a list of `ndarray`s.

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
    elif isinstance(self._spec, analyzers.CombinerSpec):
      return pcoll | _CombinerAnalyzerImpl(self._spec)
    elif isinstance(self._spec, analyzers._CombinePerKeySpec):
      return pcoll | _CombinePerKeyAnalyzerImpl(self._spec)
    else:
      raise NotImplementedError(self._spec.__class__)


def _flatten_value_to_list(batch_values):
  """Converts an N-D dense or sparse batch to a 1-D list."""
  # Ravel for flattening and tolist so that we go to native Python types
  # for more efficient followup processing.
  #
  batch_value, = batch_values
  return batch_value.ravel().tolist()


@with_input_types(List[np.ndarray])
@with_output_types(List[np.ndarray])
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
    # via AsIter. By breaking fusion, we allow sharded files' sizes to be
    # automatically computed (when possible), so we end up reading from fewer
    # and larger files.
    counts |= 'Reshard' >> beam.transforms.Reshuffle()  # pylint: disable=no-value-for-parameter

    # Using AsIter instead of AsList below in order to reduce max memory
    # usage (due to AsList caching).
    def order_by_decreasing_counts(ignored, counts_iter, store_frequency):
      """Sort the vocabulary by frequency count."""
      del ignored
      counts = list(counts_iter)
      if not counts:
        counts = [(1, '49d0cd50-04bb-48c0-bc6f-5b575dce351a')]
      counts.sort(reverse=True)  # Largest first.

      # Log vocabulary size to metrics.  Note we can call
      # beam.metrics.Metrics.distribution here because this function only gets
      # called once, so there is no need to amortize the cost of calling the
      # constructor by putting in a DoFn initializer.
      vocab_size_distribution = beam.metrics.Metrics.distribution(
          common.METRICS_NAMESPACE, 'vocabulary_size')
      vocab_size_distribution.update(len(counts))

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
        | 'CreatePath' >> beam.Create([[np.array(vocabulary_file)]])
        # Ensure that the analysis returns only after the file is written.
        | 'WaitForVocabularyFile' >> beam.Map(
            lambda x, y: x, y=beam.pvalue.AsIter(vocab_is_written)))
    return wait_for_vocabulary_transform


@with_input_types(List[np.ndarray])
@with_output_types(List[np.ndarray])
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


def _merge_outputs_by_key(keys_and_outputs):
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

  Returns:
    A list of ndarrays of the form [keys, outputs0, ..., outputs[{N-1}]]
  """
  # Sort keys_and_outputs by keys.
  keys_and_outputs.sort(key=lambda x: x[0])
  # Convert from a list of pairs of the form (key, outputs_for_key) to a list of
  # keys and a list of outputs (where the outer dimension is the number of
  # outputs not the number of keys).
  key, outputs = zip(*keys_and_outputs)
  outputs = zip(*outputs)
  # key is a list of scalars so we use np.stack to convert a single array.
  return ([np.stack(key, axis=0)] +
          [np.stack(output, axis=0) for output in outputs])


@with_input_types(List[np.ndarray])
@with_output_types(List[np.ndarray])
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

    return (
        pcoll
        | 'CombineGlobally' >> beam.CombineGlobally(_CombineFnWrapper(
            self._spec, serialized_tf_config)).with_defaults(has_defaults))


@with_input_types(List[np.ndarray])
@with_output_types(List[np.ndarray])
class _CombinePerKeyAnalyzerImpl(beam.PTransform):
  """Implement an analyzer based on a _CombinePerKeySpec."""

  def __init__(self, spec):
    self._spec = spec.combiner_spec

  def expand(self, pcoll):
    serialized_tf_config = None

    if isinstance(self._spec, analyzers._QuantilesCombinerSpec):  # pylint: disable=protected-access
      serialized_tf_config = common._DEFAULT_TENSORFLOW_CONFIG_BY_RUNNER.get(  # pylint: disable=protected-access
          pcoll.pipeline.runner)

    return (
        pcoll
        | 'SplitByKey' >> beam.FlatMap(_split_inputs_by_key)
        | 'CombinePerKey' >> beam.CombinePerKey(_CombineFnWrapper(
            self._spec, serialized_tf_config))
        | 'ToList' >> beam.combiners.ToList()
        | 'MergeByKey' >> beam.Map(_merge_outputs_by_key))


