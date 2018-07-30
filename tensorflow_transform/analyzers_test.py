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
from tensorflow_transform import test_case

import unittest

_NP_TYPES = (np.float32, np.float64, np.int32, np.int64)

_SUM_TEST = dict(
    testcase_name='Sum',
    combiner_spec=analyzers._NumPyCombinerSpec(
        np.sum, reduce_instance_dims=False, output_dtypes=[np.int64]),
    batches=[
        [np.ones((2, 6))],
        [np.ones((1, 6))],
    ],
    expected_outputs=[np.ones((6,), np.int64) * 3],
)

_SUM_OF_SIZE_ZERO_TENSORS_TEST = dict(
    testcase_name='SumOfSizeZeroTensors',
    combiner_spec=analyzers._NumPyCombinerSpec(
        np.sum, reduce_instance_dims=False, output_dtypes=[np.int64]),
    batches=[
        [np.ones((2, 0))],
        [np.ones((1, 0))],
    ],
    expected_outputs=[np.ones((0,), np.int64) * 3],
)

_COVARIANCE_SIZE_ZERO_TENSORS_TEST = dict(
    testcase_name='CovarianceSizeZeroTensors',
    combiner_spec=analyzers._CovarianceCombinerSpec(numpy_dtype=np.float64),
    batches=[
        [np.empty((1, 0))],
        [np.empty((2, 0))],
    ],
    expected_outputs=[np.empty((0, 0), dtype=np.float64)],
)

_COVARIANCE_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST = dict(
    testcase_name='CovarianceWithDegenerateCovarianceMatrix',
    combiner_spec=analyzers._CovarianceCombinerSpec(numpy_dtype=np.float64),
    batches=[
        [np.array([[0, 0, 1]])],
        [np.array([[4, 0, 1], [2, -1, 1]])],
        [np.array([[2, 1, 1]])],
    ],
    expected_outputs=[
        np.array([[2, 0, 0], [0, 0.5, 0], [0, 0, 0]], dtype=np.float64)
    ],
)

_COVARIANCE_WITH_LARGE_NUMBERS_TEST = dict(
    testcase_name='CovarianceWithLargeNumbers',
    combiner_spec=analyzers._CovarianceCombinerSpec(numpy_dtype=np.float64),
    batches=[
        [np.array([[2e15, 0], [1e15, 0]])],
        [np.array([[-2e15, 0], [-1e15, 0]])],
    ],
    expected_outputs=[np.array([[2.5e30, 0], [0, 0]], dtype=np.float64)],
)

_PCA_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST = dict(
    testcase_name='PCAWithDegenerateCovarianceMatrix',
    combiner_spec=analyzers._PCACombinerSpec(numpy_dtype=np.float64),
    batches=[
        [np.array([[0, 0, 1]])],
        [np.array([[4, 0, 1], [2, -1, 1]])],
        [np.array([[2, 1, 1]])],
    ],
    expected_outputs=[
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    ],
)

_QUANTILES_SINGLE_BATCH_TESTS = [
    dict(
        testcase_name='ComputeQuantilesSingleBatch-{}'.format(np_type),
        combiner_spec=analyzers._QuantilesCombinerSpec(
            num_quantiles=5, epsilon=0.00001, bucket_numpy_dtype=np.float32,
            always_return_num_quantiles=False),
        batches=[
            [np.linspace(1, 100, 100, dtype=np_type)],
            [np.linspace(101, 200, 100, dtype=np_type)],
            [np.linspace(201, 300, 100, dtype=np_type)],
        ],
        expected_outputs=[np.array([61, 121, 181, 241], dtype=np.float32)],
    ) for np_type in _NP_TYPES
]

_QUANTILES_MULTIPLE_BATCH_TESTS = [
    dict(
        testcase_name='ComputeQuantilesMultipleBatch-{}'.format(np_type),
        combiner_spec=analyzers._QuantilesCombinerSpec(
            num_quantiles=3, epsilon=0.00001, bucket_numpy_dtype=np.float32,
            always_return_num_quantiles=False),
        batches=[
            [np.linspace(1, 100, 100, np_type)],
        ],
        expected_outputs=[np.array([35, 68], dtype=np.float32)],
    ) for np_type in _NP_TYPES
]

_EXACT_NUM_QUANTILES_TESTS = [
    dict(
        testcase_name='ComputeExactNumQuantiles-{}'.format(np_type),
        combiner_spec=analyzers._QuantilesCombinerSpec(
            num_quantiles=4, epsilon=0.00001, bucket_numpy_dtype=np.float32,
            always_return_num_quantiles=True),
        batches=[
            [np.array([1, 1])],
        ],
        expected_outputs=[np.array([1, 1, 1], dtype=np.float32)],
    ) for np_type in _NP_TYPES
]


class AnalyzersTest(test_case.TransformTestCase):

  @test_case.named_parameters(*[
      _SUM_TEST,
      _SUM_OF_SIZE_ZERO_TENSORS_TEST,
      _COVARIANCE_SIZE_ZERO_TENSORS_TEST,
      _COVARIANCE_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST,
      _COVARIANCE_WITH_LARGE_NUMBERS_TEST,
      _PCA_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST
  ] + _QUANTILES_SINGLE_BATCH_TESTS + _QUANTILES_MULTIPLE_BATCH_TESTS +
                              _EXACT_NUM_QUANTILES_TESTS)
  def testCombinerSpec(self, combiner_spec, batches, expected_outputs):
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
    self.assertEqual(len(outputs), len(expected_outputs))
    for output, expected_output in zip(outputs, expected_outputs):
      self.assertEqual(output.dtype, expected_output.dtype)
      self.assertAllEqual(output, expected_output)


if __name__ == '__main__':
  unittest.main()
