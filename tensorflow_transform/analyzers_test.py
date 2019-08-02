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

import pickle

# GOOGLE-INITIALIZATION

import numpy as np
import tensorflow as tf

from tensorflow_transform import analyzers
from tensorflow_transform import test_case

_NP_TYPES = (np.float32, np.float64, np.int32, np.int64)

_SUM_TEST = dict(
    testcase_name='Sum',
    combiner=analyzers.NumPyCombiner(
        np.sum, output_dtypes=[np.int64], output_shapes=[(None,)]),
    batches=[
        (np.array([1, 2, 3, 4, 5, 6]),),
        (np.array([1, 2, 3, 4, 5, 6]),),
    ],
    expected_outputs=[np.array([2, 4, 6, 8, 10, 12])],
)

_SUM_SCALAR_TEST = dict(
    testcase_name='SumScalar',
    combiner=analyzers.NumPyCombiner(
        np.sum, output_dtypes=[np.int64], output_shapes=[(None,)]),
    batches=[
        (np.array(1),),
        (np.array(2),),
    ],
    expected_outputs=[np.array(3)],
)

_SUM_OF_SIZE_ZERO_TENSORS_TEST = dict(
    testcase_name='SumOfSizeZeroTensors',
    combiner=analyzers.NumPyCombiner(
        np.sum, output_dtypes=[np.int64], output_shapes=[(None, None)]),
    batches=[
        (np.array([]),),
        (np.array([]),),
    ],
    expected_outputs=[np.array([], np.int64) * 2],
)

_COVARIANCE_SIZE_ZERO_TENSORS_TEST = dict(
    testcase_name='CovarianceSizeZeroTensors',
    combiner=analyzers.CovarianceCombiner(numpy_dtype=np.float64),
    batches=[
        (np.empty((1, 0)),),
        (np.empty((2, 0)),),
    ],
    expected_outputs=[np.empty((0, 0), dtype=np.float64)],
)

_COVARIANCE_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST = dict(
    testcase_name='CovarianceWithDegenerateCovarianceMatrix',
    combiner=analyzers.CovarianceCombiner(numpy_dtype=np.float64),
    batches=[
        (np.array([[0, 0, 1]]),),
        (np.array([[4, 0, 1], [2, -1, 1]]),),
        (np.array([[2, 1, 1]]),),
    ],
    expected_outputs=[
        np.array([[2, 0, 0], [0, 0.5, 0], [0, 0, 0]], dtype=np.float64)
    ],
)

_COVARIANCE_WITH_LARGE_NUMBERS_TEST = dict(
    testcase_name='CovarianceWithLargeNumbers',
    combiner=analyzers.CovarianceCombiner(numpy_dtype=np.float64),
    batches=[
        (np.array([[2e15, 0], [1e15, 0]]),),
        (np.array([[-2e15, 0], [-1e15, 0]]),),
    ],
    expected_outputs=[np.array([[2.5e30, 0], [0, 0]], dtype=np.float64)],
)

_PCA_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST = dict(
    testcase_name='PCAWithDegenerateCovarianceMatrix',
    combiner=analyzers.PCACombiner(numpy_dtype=np.float64),
    batches=[
        (np.array([[0, 0, 1]]),),
        (np.array([[4, 0, 1], [2, -1, 1]]),),
        (np.array([[2, 1, 1]]),),
    ],
    expected_outputs=[
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    ],
)


def _make_mean_and_var_accumulator_from_instance(instance, axis=None):
  return analyzers._WeightedMeanAndVarAccumulator(
      count=np.sum(np.ones_like(instance), axis=axis),
      mean=np.mean(instance, axis=axis),
      weight=np.sum(np.ones_like(instance), axis=axis),
      variance=np.var(instance, axis=axis))


_MEAN_AND_VAR_TEST = dict(
    testcase_name='WeightedMeanAndVar',
    combiner=analyzers.WeightedMeanAndVarCombiner(np.float32, output_shape=()),
    batches=[
        _make_mean_and_var_accumulator_from_instance([[1, 2, 3, 4, 5, 6, 7]]),
        # Count is 5*0xFFFF=327675 for this accumulator.
        _make_mean_and_var_accumulator_from_instance([[8, 9, 10, 11, 12]] *
                                                     0xFFFF),
        _make_mean_and_var_accumulator_from_instance([[100, 200, 3000]]),
    ],
    expected_outputs=[
        np.float32(10.00985092390558),
        np.float32(29.418185761379473),
    ],
)

_MEAN_AND_VAR_BIG_TEST = dict(
    testcase_name='WeightedMeanAndVarBig',
    combiner=analyzers.WeightedMeanAndVarCombiner(np.float32, output_shape=()),
    batches=[
        _make_mean_and_var_accumulator_from_instance([[1, 2, 3, 4, 5, 6, 7]]),
        _make_mean_and_var_accumulator_from_instance([[1e15, 2e15, 3000]]),
        _make_mean_and_var_accumulator_from_instance([[100, 200]]),
    ],
    expected_outputs=[
        np.float32(2.50e+14),
        np.float32(3.541666666665e+29),
    ],
)

_MEAN_AND_VAR_VECTORS_TEST = dict(
    testcase_name='WeightedMeanAndVarForVectors',
    combiner=analyzers.WeightedMeanAndVarCombiner(
        np.float32, output_shape=(None,)),
    batches=[
        _make_mean_and_var_accumulator_from_instance([[1, 2, 3, 4, 5, 6]],
                                                     axis=0),
        _make_mean_and_var_accumulator_from_instance([[7, 8, 9, 10, 11, 12]],
                                                     axis=0),
        _make_mean_and_var_accumulator_from_instance(
            [[100, 200, 3000, 17, 27, 53]], axis=0),
    ],
    expected_outputs=[
        np.float32([36., 70., 1004., 10.33333333, 14.33333333, 23.66666667]),
        np.float32(
            [2054., 8456., 1992014., 28.22222222, 86.22222222, 436.22222222]),
    ],
)

_MEAN_AND_VAR_ND_TEST = dict(
    testcase_name='WeightedMeanAndVarForNDVectors',
    combiner=analyzers.WeightedMeanAndVarCombiner(
        np.float32, output_shape=(None, None)),
    batches=[
        _make_mean_and_var_accumulator_from_instance([[[1], [1], [2]]], axis=2),
        _make_mean_and_var_accumulator_from_instance([[[1], [2], [2]]], axis=2),
        _make_mean_and_var_accumulator_from_instance([[[2], [2], [2]]], axis=2),
    ],
    expected_outputs=[
        np.float32([[1.333333333, 1.666666666, 2]]),
        np.float32([[.222222222, .222222222, 0]]),
    ],
)

_QUANTILES_NO_ELEMENTS_TEST = dict(
    testcase_name='ComputeQuantilesNoElements',
    combiner=analyzers.QuantilesCombiner(
        num_quantiles=5,
        epsilon=0.00001,
        bucket_numpy_dtype=np.float32,
        always_return_num_quantiles=False),
    batches=[
        (np.empty((0, 1), dtype=np.float32),),
    ],
    expected_outputs=[np.zeros((0,), dtype=np.float32)],
)

_QUANTILES_EXACT_NO_ELEMENTS_TEST = dict(
    testcase_name='ComputeExactQuantilesNoElements',
    combiner=analyzers.QuantilesCombiner(
        num_quantiles=5,
        epsilon=0.00001,
        bucket_numpy_dtype=np.float32,
        always_return_num_quantiles=True),
    batches=[
        (np.empty((0, 1), dtype=np.float32),),
    ],
    expected_outputs=[np.zeros((4,), dtype=np.float32)],
)

_QUANTILES_NO_TRIM_TEST = dict(
    testcase_name='NoTrimQuantilesTest',
    combiner=analyzers.QuantilesCombiner(
        num_quantiles=4,
        epsilon=0.00001,
        bucket_numpy_dtype=np.float32,
        always_return_num_quantiles=True,
        include_max_and_min=True),
    batches=[
        (np.array([1, 1]),),
    ],
    expected_outputs=[np.array([1, 1, 1, 1, 1], dtype=np.float32)],
)

# pylint: disable=g-complex-comprehension
_QUANTILES_SINGLE_BATCH_TESTS = [
    dict(
        testcase_name='ComputeQuantilesSingleBatch-{}'.format(np_type),
        combiner=analyzers.QuantilesCombiner(
            num_quantiles=5,
            epsilon=0.00001,
            bucket_numpy_dtype=np.float32,
            always_return_num_quantiles=False),
        batches=[
            (np.linspace(1, 100, 100, dtype=np_type),),
            (np.linspace(101, 200, 100, dtype=np_type),),
            (np.linspace(201, 300, 100, dtype=np_type),),
            (np.empty((0, 3)),),
        ],
        expected_outputs=[np.array([61, 121, 181, 241], dtype=np.float32)],
    ) for np_type in _NP_TYPES
]

_QUANTILES_ELEMENTWISE_TESTS = [
    dict(
        testcase_name='ComputeQuantilesElementwise-{}'.format(np_type),
        combiner=analyzers.QuantilesCombiner(
            num_quantiles=5,
            epsilon=0.00001,
            bucket_numpy_dtype=np.float32,
            always_return_num_quantiles=True,
            feature_shape=[3]),
        batches=[
            (np.vstack([np.linspace(1, 100, 100, dtype=np_type),
                        np.linspace(101, 200, 100, dtype=np_type),
                        np.linspace(201, 300, 100, dtype=np_type)]).T,),
            (np.empty((0, 3)),),
        ],
        expected_outputs=[np.array([[21, 41, 61, 81],
                                    [121, 141, 161, 181],
                                    [221, 241, 261, 281]], dtype=np.float32)],
    ) for np_type in _NP_TYPES
]

_QUANTILES_MULTIPLE_BATCH_TESTS = [
    dict(
        testcase_name='ComputeQuantilesMultipleBatch-{}'.format(np_type),
        combiner=analyzers.QuantilesCombiner(
            num_quantiles=3,
            epsilon=0.00001,
            bucket_numpy_dtype=np.float32,
            always_return_num_quantiles=False),
        batches=[
            (np.linspace(1, 100, 100, np_type),),
        ],
        expected_outputs=[np.array([35, 68], dtype=np.float32)],
    ) for np_type in _NP_TYPES
]

_EXACT_NUM_QUANTILES_TESTS = [
    dict(
        testcase_name='ComputeExactNumQuantiles-{}'.format(np_type),
        combiner=analyzers.QuantilesCombiner(
            num_quantiles=4,
            epsilon=0.00001,
            bucket_numpy_dtype=np.float32,
            always_return_num_quantiles=True),
        batches=[
            (np.array([1, 1]),),
        ],
        expected_outputs=[np.array([1, 1, 1], dtype=np.float32)],
    ) for np_type in _NP_TYPES
]
# pylint: enable=g-complex-comprehension


class AnalyzersTest(test_case.TransformTestCase):

  @test_case.named_parameters(*[
      _SUM_TEST,
      _SUM_SCALAR_TEST,
      _SUM_OF_SIZE_ZERO_TENSORS_TEST,
      _COVARIANCE_SIZE_ZERO_TENSORS_TEST,
      _COVARIANCE_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST,
      _COVARIANCE_WITH_LARGE_NUMBERS_TEST,
      _PCA_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST,
      _MEAN_AND_VAR_TEST,
      _MEAN_AND_VAR_BIG_TEST,
      _MEAN_AND_VAR_VECTORS_TEST,
      _MEAN_AND_VAR_ND_TEST,
      _QUANTILES_NO_ELEMENTS_TEST,
      _QUANTILES_NO_TRIM_TEST,
      _QUANTILES_EXACT_NO_ELEMENTS_TEST,
  ] + _QUANTILES_SINGLE_BATCH_TESTS + _QUANTILES_MULTIPLE_BATCH_TESTS +
                              _QUANTILES_ELEMENTWISE_TESTS +
                              _EXACT_NUM_QUANTILES_TESTS)
  def testCombiner(self, combiner, batches, expected_outputs):
    """Tests the provided combiner.

    Args:
      combiner: An object implementing the Combiner interface.
      batches: A list of batches, each is a tuples of ndarrays.  each ndarray
        represents the values of an input tensor of the analyzer over a single
        batch.
      expected_outputs: The expected outputs from extract_output.

    Exercises create_accumulator, add_input, merge_accumulators,
    and extract_output.
    """
    # Test serialization faithfully reproduces the object. If tests
    # mysteriously break, it could be because __reduce__ is missing something.
    combiner = pickle.loads(pickle.dumps(combiner))

    if isinstance(combiner, analyzers.QuantilesCombiner):
      combiner.initialize_local_state()

    # Note `accumulators` is a generator, not list.  We do this to ensure that
    # add_input is not relying on its input being a list.
    accumulators = (
        combiner.add_input(combiner.create_accumulator(), batch)
        for batch in batches)

    final_accumulator = combiner.merge_accumulators(accumulators)
    outputs = combiner.extract_output(final_accumulator)
    tensor_infos = combiner.output_tensor_infos()
    self.assertEqual(len(outputs), len(expected_outputs))
    self.assertEqual(len(outputs), len(tensor_infos))
    for output, expected_output, tensor_info in zip(
        outputs, expected_outputs, tensor_infos):
      self.assertEqual(output.dtype, expected_output.dtype)
      self.assertEqual(tensor_info.dtype,
                       tf.as_dtype(expected_output.dtype))
      self.assertAllEqual(output, expected_output)

  @test_case.named_parameters(
      {
          'testcase_name': '1d',
          'a': np.array([1]),
          'b': np.array([1, 1]),
          'expected_a': np.array([1, 0]),
          'expected_b': np.array([1, 1]),
      },
      {
          'testcase_name': '2d_1different',
          'a': np.array([[1], [1]]),
          'b': np.array([[1], [1], [2]]),
          'expected_a': np.array([[1], [1], [0]]),
          'expected_b': np.array([[1], [1], [2]]),
      },
      {
          'testcase_name': '2d_2different',
          'a': np.array([[1, 3], [1, 3]]),
          'b': np.array([[1], [1], [2]]),
          'expected_a': np.array([[1, 3], [1, 3], [0, 0]]),
          'expected_b': np.array([[1, 0], [1, 0], [2, 0]]),
      },
      {
          'testcase_name': '3d_1different',
          'a': np.array([[[1], [1]], [[1], [1]]]),
          'b': np.array([[[1], [1]]]),
          'expected_a': np.array([[[1], [1]], [[1], [1]]]),
          'expected_b': np.array([[[1], [1]], [[0], [0]]]),
      },
      {
          'testcase_name': '3d_2different',
          'a': np.array([[[1], [1]], [[1], [1]]]),
          'b': np.array([[[1, 1], [1, 1]]]),
          'expected_a': np.array([[[1, 0], [1, 0]], [[1, 0], [1, 0]]]),
          'expected_b': np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]]),
      },
  )
  def test_pad_arrays_to_match(self, a, b, expected_a, expected_b):
    a2, b2 = analyzers._pad_arrays_to_match(a, b)
    self.assertAllClose(a2, expected_a)
    self.assertAllClose(b2, expected_b)

  def testMinDiffFromAvg(self):
    # Small dataset gets the minimum of 2
    self.assertEqual(
        analyzers.calculate_recommended_min_diff_from_avg(10000), 2)
    self.assertEqual(
        analyzers.calculate_recommended_min_diff_from_avg(100000), 4)
    self.assertEqual(
        analyzers.calculate_recommended_min_diff_from_avg(500000), 13)
    # Large dataset gets the maximum of 25
    self.assertEqual(
        analyzers.calculate_recommended_min_diff_from_avg(100000000), 25)


if __name__ == '__main__':
  test_case.main()
