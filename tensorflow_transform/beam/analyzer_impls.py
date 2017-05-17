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

from collections import Counter


import apache_beam as beam

from apache_beam.typehints import KV
from apache_beam.typehints import List
from apache_beam.typehints import Union
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types

import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform.beam import common


def _impl_for_analyzer(spec):
  if isinstance(spec, analyzers.NumericCombineSpec):
    return _NumericCombineAnalyzerImpl(spec)
  elif isinstance(spec, analyzers.UniquesSpec):
    return _UniquesAnalyzerImpl(spec)
  else:
    raise NotImplementedError(spec.__class__)


def _flatten_value_to_list(batch):
  """Converts an N-D dense or sparse batch to a 1-D list."""
  if isinstance(batch, tf.SparseTensorValue):
    dense_values = batch.values
  else:
    dense_values = batch
  # Ravel for flattening and tolist so that we go to native Python types
  # for more efficient followup processing.
  #
  return dense_values.ravel().tolist()


_BUILTIN_COMBINERS_BY_OPERATION = {
    analyzers.NumericCombineSpec.MIN: min,
    analyzers.NumericCombineSpec.MAX: max,
    analyzers.NumericCombineSpec.SUM: sum
}


_NUMPY_COMBINERS_BY_OPERATION = {
    analyzers.NumericCombineSpec.MIN: np.min,
    analyzers.NumericCombineSpec.MAX: np.max,
    analyzers.NumericCombineSpec.SUM: np.sum
}


@beam.ptransform_fn
def WrapAsNDArray(x, dtype):  # pylint: disable=invalid-name
  return x | beam.Map(
      lambda v, np_dtype=dtype.as_numpy_dtype: np.asarray(v, np_dtype))


@with_input_types(Union[np.ndarray, tf.SparseTensorValue])
@with_output_types(List[np.ndarray])
class _NumericCombineAnalyzerImpl(beam.PTransform):
  """Reduces a PCollection of batches according to the given function."""

  class _CombineOnBatchDim(beam.CombineFn):
    """Combines the PCollection only on the 0th dimension using nparray."""

    def __init__(self, fn):
      self._fn = fn

    def create_accumulator(self):
      return []

    def add_input(self, accumulator, next_input):
      batch = self._fn(next_input, axis=0)
      if any(accumulator):
        return self._fn((accumulator, batch), axis=0)
      else:
        return batch

    def merge_accumulators(self, accumulators):
      # numpy's sum, min, max, etc functions operate on array-like objects, but
      # not arbitrary iterables. Convert the provided accumulators into a list
      return self._fn(list(accumulators), axis=0)

    def extract_output(self, accumulator):
      return accumulator

  def __init__(self, spec):
    assert isinstance(spec, analyzers.NumericCombineSpec)
    self._spec = spec

  def expand(self, pcoll):
    if self._spec.reduce_instance_dims:
      fn = _BUILTIN_COMBINERS_BY_OPERATION[self._spec.combiner_type]
      output = (pcoll
                | 'FlattenValueToList' >> beam.Map(_flatten_value_to_list)
                | 'CombineWithinList' >> beam.Map(fn)
                | 'CombineGlobally'
                >> beam.CombineGlobally(fn).without_defaults())
    else:
      fn = _NUMPY_COMBINERS_BY_OPERATION[self._spec.combiner_type]
      output = (pcoll | 'CombineOnBatchDim'
                >> beam.CombineGlobally(self._CombineOnBatchDim(fn)))

    # pylint: disable=no-value-for-parameter
    output |= 'WrapAsNDArray' >> WrapAsNDArray(self._spec.dtype)
    return [output]


@with_input_types(Union[np.ndarray, tf.SparseTensorValue])
@with_output_types(List[np.ndarray])
class _UniquesAnalyzerImpl(beam.PTransform):
  """Returns the unique elements in a PCollection of batches."""

  def __init__(self, spec):
    assert isinstance(spec, analyzers.UniquesSpec)
    self._spec = spec

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
        beam.FlatMap(lambda lst: six.iteritems(Counter(lst))).with_output_types(
            KV[common.PRIMITIVE_TYPE, int])
        | 'CountGlobally' >> beam.CombinePerKey(sum)
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
    def order_by_decreasing_counts(_, counts_iter):  # pylint: disable=invalid-name
      counts = list(counts_iter)
      counts.sort(reverse=True)  # Largest first.
      return [element for _, element in counts]

    # pylint: disable=no-value-for-parameter
    output = (pcoll.pipeline
              | 'Prepare' >> beam.Create([None])
              | 'OrderByDecreasingCounts' >> beam.Map(
                  order_by_decreasing_counts,
                  counts_iter=beam.pvalue.AsIter(counts))
              | 'WrapAsNDArray' >> WrapAsNDArray(self._spec.dtype))
    return [output]

