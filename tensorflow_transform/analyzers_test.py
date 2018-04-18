# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tests for tensorflow_transform.analyzers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

from tensorflow_transform import analyzers

import unittest
from tensorflow.python.framework import test_util


class AnalyzersTest(test_util.TensorFlowTestCase):

  def _test_combiner_spec_helper(self, combiner_spec, batches,
                                 expected_outputs):
    """Tests the provided combiner.

    Args:
      combiner_spec: A CominerSpec.
      batches: A list of lists of ndarrays.  The outer list is a list of batches
        and the inner list is a inputs where each input is an ndarray
        representing the values of an input tensor of the analyzer over a single
        batch.
      expected_outputs: The expected outputs from extract_output.

    Exercises create_accumulator, add_input, merge_accumulators,
    and extract_output.
    """
    if isinstance(combiner_spec, analyzers._QuantilesCombinerSpec):
      combiner_spec.initialize_local_state()
    # Note `accumulators` is a generator, not list.  We do this to ensure that
    # add_input is not relying on its input being a list.
    accumulators = (
        combiner_spec.add_input(combiner_spec.create_accumulator(), batch)
        for batch in batches)
    final_accumulator = combiner_spec.merge_accumulators(accumulators)
    outputs = combiner_spec.extract_output(final_accumulator)
    # Extract output 0 since all analyzers have a single output
    self.assertEqual(len(outputs), len(expected_outputs))
    for output, expected_output in zip(outputs, expected_outputs):
      self.assertEqual(output.dtype, expected_output.dtype)
      self.assertAllEqual(output, expected_output)

  def testSum(self):
    self._test_combiner_spec_helper(
        combiner_spec=analyzers._NumPyCombinerSpec(
            np.sum, reduce_instance_dims=False, output_dtypes=[np.int64]),
        batches=[[np.ones((2, 6))], [np.ones((1, 6))]],
        expected_outputs=[np.ones((6,), np.int64) * 3])

  def testSumOfSizeZeroTensors(self):
    self._test_combiner_spec_helper(
        combiner_spec=analyzers._NumPyCombinerSpec(
            np.sum, reduce_instance_dims=False, output_dtypes=[np.int64]),
        batches=[[np.ones((2, 0))], [np.ones((1, 0))]],
        expected_outputs=[np.ones((0,), np.int64) * 3])

  def _test_compute_quantiles_single_batch_helper(self, nptype):
    self._test_combiner_spec_helper(
        combiner_spec=analyzers._QuantilesCombinerSpec(
            num_quantiles=3, epsilon=0.00001, bucket_numpy_dtype=np.float32),
        batches=[[np.linspace(1, 100, 100, nptype)]],
        expected_outputs=[np.array([35, 68], dtype=np.float32)])

  def testComputeQuantilesSingleBatch(self):
    self._test_compute_quantiles_single_batch_helper(np.double)
    self._test_compute_quantiles_single_batch_helper(np.float32)
    self._test_compute_quantiles_single_batch_helper(np.float64)
    self._test_compute_quantiles_single_batch_helper(np.int32)
    self._test_compute_quantiles_single_batch_helper(np.int64)

  def _test_compute_quantiles_multipe_batch_helper(self, nptype):
    self._test_combiner_spec_helper(
        combiner_spec=analyzers._QuantilesCombinerSpec(
            num_quantiles=5, epsilon=0.00001, bucket_numpy_dtype=np.float32),
        batches=[
            [np.linspace(1, 100, 100, dtype=nptype)],
            [np.linspace(101, 200, 100, dtype=nptype)],
            [np.linspace(201, 300, 100, dtype=nptype)]
        ],
        expected_outputs=[np.array([61, 121, 181, 241], dtype=np.float32)])

  def testComputeQuantilesMultipleBatch(self):
    self._test_compute_quantiles_multipe_batch_helper(np.double)
    self._test_compute_quantiles_multipe_batch_helper(np.float32)
    self._test_compute_quantiles_multipe_batch_helper(np.float64)
    self._test_compute_quantiles_multipe_batch_helper(np.int32)
    self._test_compute_quantiles_multipe_batch_helper(np.int64)

  def testCovarianceSizeZeroTensors(self):
    self._test_combiner_spec_helper(
        combiner_spec=analyzers._CovarianceCombinerSpec(numpy_dtype=np.float64),
        batches=[[np.empty((1, 0))], [np.empty((2, 0))]],
        expected_outputs=[np.empty((0, 0), dtype=np.float64)])

  def testCovarianceWithDegenerateCovarianceMatrix(self):
    self._test_combiner_spec_helper(
        combiner_spec=analyzers._CovarianceCombinerSpec(numpy_dtype=np.float64),
        batches=[
            [np.array([[0, 0, 1]])],
            [np.array([[4, 0, 1], [2, -1, 1]])],
            [np.array([[2, 1, 1]])]
        ],
        expected_outputs=[
            np.array([[2, 0, 0], [0, 0.5, 0], [0, 0, 0]], dtype=np.float64)
        ])

  def testCovarianceWithLargeNumbers(self):
    self._test_combiner_spec_helper(
        combiner_spec=analyzers._CovarianceCombinerSpec(numpy_dtype=np.float64),
        batches=[
            [np.array([[2e15, 0], [1e15, 0]])],
            [np.array([[-2e15, 0], [-1e15, 0]])]
        ],
        expected_outputs=[np.array([[2.5e30, 0], [0, 0]], dtype=np.float64)])

  def testPCAWithDegenerateCovarianceMatrix(self):
    self._test_combiner_spec_helper(
        combiner_spec=analyzers._PCACombinerSpec(numpy_dtype=np.float64),
        batches=[
            [np.array([[0, 0, 1]])],
            [np.array([[4, 0, 1], [2, -1, 1]])],
            [np.array([[2, 1, 1]])]
        ],
        expected_outputs=[
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        ])


if __name__ == '__main__':
  unittest.main()
