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
"""Tests for tensorflow_transform.beam.analyzer_impls."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

import numpy as np
from tensorflow_transform import analyzers
from tensorflow_transform.beam import analyzer_impls as impl
from tensorflow_transform.beam import impl as beam_impl


import unittest
from tensorflow.python.framework import test_util


class AnalyzerImplsTest(test_util.TensorFlowTestCase):

  def assertCombine(self, combine_fn, shards, expected, check_np_type=False):
    """Tests the provided combiner.

    Args:
      combine_fn: A beam.ComineFn to exercise.
      shards: A list of next_inputs to add via the combiner.
      expected: The expected output from extract_output.
      check_np_type: check strict equivalence of output numpy type.

    Exercises create_accumulator, add_input, merge_accumulators,
    and extract_output.
    """
    accumulators = [
        combine_fn.add_input(combine_fn.create_accumulator(), shard)
        for shard in shards]
    final_accumulator = combine_fn.merge_accumulators(accumulators)
    extracted = combine_fn.extract_output(final_accumulator)
    # Extract output 0 since all analyzers have a single output
    extracted = extracted[0]
    if check_np_type:
      # This is currently applicable only for quantile buckets, which conains a
      # single element list of numpy array; the numpy array contains the bucket
      # boundaries.
      self.assertEqual(len(expected), 1)
      self.assertEqual(len(extracted), 1)
      self.assertEqual(expected[0].dtype, extracted[0].dtype)
    self.assertAllEqual(expected, extracted)

  def testCombineOnBatchSimple(self):
    lst_1 = [np.ones(6), np.ones(6)]
    lst_2 = [np.ones(6)]
    out = [3 for _ in range(6)]
    analyzer = impl._CombineFnWrapper(
        analyzers._NumPyCombinerSpec(np.sum, reduce_instance_dims=False))
    self.assertCombine(analyzer, [lst_1, lst_2], out)

  def testCombineOnBatchAllEmptyRow(self):
    analyzer = impl._CombineFnWrapper(
        analyzers._NumPyCombinerSpec(np.sum, reduce_instance_dims=False))
    self.assertCombine(analyzer, [[[]], [[]], [[]]], [])

  def testCombineOnBatchLotsOfData(self):
    shards = [[np.ones(3)] for _ in range(
        beam_impl._DEFAULT_DESIRED_BATCH_SIZE * 2)]
    out = [1 for _ in range(3)]
    analyzer = impl._CombineFnWrapper(
        analyzers._NumPyCombinerSpec(np.min, reduce_instance_dims=False))
    self.assertCombine(analyzer, shards, out)

  def _test_compute_quantiles_single_batch_helper(self, nptype):
    lst_1 = np.linspace(1, 100, 100, nptype).tolist()
    analyzer = impl._ComputeQuantiles(num_quantiles=3, epsilon=0.00001)
    out = [np.array([1, 35, 68, 100], dtype=np.float32)]
    self.assertCombine(analyzer, np.array(lst_1), out, check_np_type=True)

  def testComputeQuantilesSingleBatch(self):
    self._test_compute_quantiles_single_batch_helper(np.double)
    self._test_compute_quantiles_single_batch_helper(np.float32)
    self._test_compute_quantiles_single_batch_helper(np.float64)
    self._test_compute_quantiles_single_batch_helper(np.int32)
    self._test_compute_quantiles_single_batch_helper(np.int64)

  def _test_compute_quantiles_multipe_batch_helper(self, nptype):
    lst_1 = np.linspace(1, 100, 100, dtype=nptype).tolist()
    lst_2 = np.linspace(101, 200, 100, dtype=nptype).tolist()
    lst_3 = np.linspace(201, 300, 100, dtype=nptype).tolist()
    analyzer = impl._ComputeQuantiles(num_quantiles=5, epsilon=0.00001)
    out = [np.array([1, 61, 121, 181, 241, 300], dtype=np.float32)]
    self.assertCombine(
        analyzer, np.array([lst_1, lst_2, lst_3]), out, check_np_type=True)

  def testComputeQuantilesMultipleBatch(self):
    self._test_compute_quantiles_multipe_batch_helper(np.double)
    self._test_compute_quantiles_multipe_batch_helper(np.float32)
    self._test_compute_quantiles_multipe_batch_helper(np.float64)
    self._test_compute_quantiles_multipe_batch_helper(np.int32)
    self._test_compute_quantiles_multipe_batch_helper(np.int64)


if __name__ == '__main__':
  unittest.main()
