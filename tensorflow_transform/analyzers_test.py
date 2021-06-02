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

import pickle

import numpy as np
import tensorflow as tf

from tensorflow_transform import analyzers
from tensorflow_transform import test_case

_NP_TYPES = (np.float32, np.float64, np.int32, np.int64)

_SUM_TEST = dict(
    testcase_name='Sum',
    combiner=analyzers.NumPyCombiner(
        fn=np.sum,
        default_accumulator_value=0,
        output_dtypes=[np.int64],
        output_shapes=[None]),
    batches=[
        (np.array([1, 2, 3, 4, 5, 6]),),
        (np.array([1, 2, 3, 4, 5, 6]),),
    ],
    expected_outputs=[np.array([2, 4, 6, 8, 10, 12])],
)

_SUM_SCALAR_TEST = dict(
    testcase_name='SumScalar',
    combiner=analyzers.NumPyCombiner(
        fn=np.sum,
        default_accumulator_value=0,
        output_dtypes=[np.int64],
        output_shapes=[None]),
    batches=[
        (np.array(1),),
        (np.array(2),),
    ],
    expected_outputs=[np.array(3)],
)

_SUM_OF_SIZE_ZERO_TENSORS_TEST = dict(
    testcase_name='SumOfSizeZeroTensors',
    combiner=analyzers.NumPyCombiner(
        fn=np.sum,
        default_accumulator_value=0,
        output_dtypes=[np.int64],
        output_shapes=[None]),
    batches=[
        (np.array([]),),
        (np.array([]),),
    ],
    expected_outputs=[np.array([], np.int64) * 2],
)

_COVARIANCE_SIZE_ZERO_TENSORS_TEST = dict(
    testcase_name='CovarianceSizeZeroTensors',
    combiner=analyzers.CovarianceCombiner(output_shape=(0, 0),
                                          numpy_dtype=np.float64),
    batches=[
        (np.empty((1, 0)),),
        (np.empty((2, 0)),),
    ],
    expected_outputs=[np.empty((0, 0), dtype=np.float64)],
)

_COVARIANCE_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST = dict(
    testcase_name='CovarianceWithDegenerateCovarianceMatrix',
    combiner=analyzers.CovarianceCombiner(output_shape=(3, 3),
                                          numpy_dtype=np.float64),
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
    combiner=analyzers.CovarianceCombiner(output_shape=(2, 2),
                                          numpy_dtype=np.float64),
    batches=[
        (np.array([[2e15, 0], [1e15, 0]]),),
        (np.array([[-2e15, 0], [-1e15, 0]]),),
    ],
    expected_outputs=[np.array([[2.5e30, 0], [0, 0]], dtype=np.float64)],
)

_PCA_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST = dict(
    testcase_name='PCAWithDegenerateCovarianceMatrix',
    combiner=analyzers.PCACombiner(output_shape=(3, 3),
                                   numpy_dtype=np.float64),
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

_MEAN_AND_VAR_SIMPLE_TEST = dict(
    testcase_name='WeightedMeanAndVarSimple',
    combiner=analyzers.WeightedMeanAndVarCombiner(
        np.float32,
        output_shape=(),
        compute_variance=False,
        compute_weighted=False),
    batches=[
        _make_mean_and_var_accumulator_from_instance([[1, 2, 3, 4, 5, 6, 7]]),
        # Count is 5*0xFFFF=327675 for this accumulator.
        _make_mean_and_var_accumulator_from_instance([[8, 9, 10, 11, 12]] *
                                                     0xFFFF),
        _make_mean_and_var_accumulator_from_instance([[100, 200, 3000]]),
    ],
    expected_outputs=analyzers._WeightedMeanAndVarAccumulator(
        count=np.array(327685),
        mean=np.float32(10.00985092390558),
        weight=np.float32(1.0),
        variance=np.float32(0.0)))

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

_L_MOMENTS_TESTS = [dict(
    testcase_name='LMoments_one_batch',
    combiner=analyzers._LMomentsCombiner(np.float32, output_shape=()),
    batches=[
        # Accumulator for the sequence:
        # np.concatenate((np.power(2.0, np.arange(0, 10, 0.01)),
        #                 -np.power(1.9, np.arange(0, 10, 0.01)))
        analyzers._LMomentsAccumulator(
            count_l1=np.float32(2000.),
            count_l2=np.float32(1999000.),
            count_l3=np.float32(1.331334e+09),
            count_l4=np.float32(6.6466854e+11),
            l1=np.float32(26.00855),
            l2=np.float32(103.25489),
            l3=np.float32(17.549286),
            l4=np.float32(47.41136))
    ],
    expected_outputs=[
        np.float32(5.769684),
        np.float32(81.381424),
        np.float32(0.39079103),
        np.float32(0.55846965)
    ],
), dict(
    testcase_name='LMoments_small_batch',
    combiner=analyzers._LMomentsCombiner(np.float32, output_shape=()),
    batches=[
        # Accumulator for the sequence: [1., 1., 2., 2.].
        analyzers._LMomentsAccumulator(
            count_l1=np.float32(4.),
            count_l2=np.float32(6.),
            count_l3=np.float32(4.),
            count_l4=np.float32(1.),
            l1=np.float32(1.5),
            l2=np.float32(0.33333334),
            l3=np.float32(0.),
            l4=np.float32(-0.5))
    ],
    expected_outputs=[
        np.float32(1.5),
        np.float32(np.sqrt(np.pi) / 3.0),
        np.float32(0.0),
        np.float32(0.0)
    ],
), dict(
    testcase_name='LMoments_one_sample',
    combiner=analyzers._LMomentsCombiner(np.float32, output_shape=()),
    batches=[
        # Accumulator for the sequence: [1.].
        analyzers._LMomentsAccumulator(
            count_l1=np.float32(1.),
            count_l2=np.float32(0.),
            count_l3=np.float32(-0.),
            count_l4=np.float32(0.),
            l1=np.float32(1.),
            l2=np.float32(0.),
            l3=np.float32(-0.),
            l4=np.float32(0.))
    ],
    expected_outputs=[
        np.float32(1.0),
        np.float32(1.0),
        np.float32(0.0),
        np.float32(0.0)
    ],
), dict(
    testcase_name='LMoments_two_samples',
    combiner=analyzers._LMomentsCombiner(np.float32, output_shape=()),
    batches=[
        # Accumulator for the sequence: [1., 1.].
        analyzers._LMomentsAccumulator(
            count_l1=np.float32(2.),
            count_l2=np.float32(1.),
            count_l3=np.float32(0.),
            count_l4=np.float32(-0.),
            l1=np.float32(1.),
            l2=np.float32(0.),
            l3=np.float32(0.),
            l4=np.float32(0.))
    ],
    expected_outputs=[
        np.float32(1.0),
        np.float32(1.0),
        np.float32(0.0),
        np.float32(0.0)
    ],
), dict(
    testcase_name='LMoments_multiple_batches',
    combiner=analyzers._LMomentsCombiner(np.float32, output_shape=()),
    batches=[
        # Accumulator for the sequence:
        # np.concatenate((np.power(2.0, np.arange(0, 10, 0.02)),
        #                 -np.power(1.9, np.arange(0, 10, 0.02)))
        analyzers._LMomentsAccumulator(
            count_l1=np.float32(1000.),
            count_l2=np.float32(499500.),
            count_l3=np.float32(1.66167e+08),
            count_l4=np.float32(4.1417126e+10),
            l1=np.float32(25.90623),
            l2=np.float32(102.958664),
            l3=np.float32(17.50719),
            l4=np.float32(47.393063)),
        # Accumulator for the sequence:
        # np.concatenate((np.power(2.0, np.arange(0.01, 10, 0.02)),
        #                 -np.power(1.9, np.arange(0.01, 10, 0.02)))
        analyzers._LMomentsAccumulator(
            count_l1=np.float32(1000.),
            count_l2=np.float32(499500.),
            count_l3=np.float32(1.66167e+08),
            count_l4=np.float32(4.1417126e+10),
            l1=np.float32(26.110888),
            l2=np.float32(103.65407),
            l3=np.float32(17.64386),
            l4=np.float32(47.71353)),
    ],
    expected_outputs=[
        np.float32(5.751478),
        np.float32(81.16352),
        np.float32(0.3923474),
        np.float32(0.55972165)
    ],
)]

_L_MOMENTS_ND_TESTS = [dict(
    testcase_name='LMomentsOneBatchForNDVectors',
    combiner=analyzers._LMomentsCombiner(np.float32, output_shape=(None, None)),
    batches=[
        # Accumulator for the sequence:
        # np.concatenate((
        #     np.concatenate((
        #         np.power(2.0, np.arange(0, 10, 0.01)),
        #         -np.power(1.9, np.arange(0, 10, 0.01)))).reshape(
        #             [-1, 1, 1]),
        #     np.concatenate((
        #         np.power(1.9, np.arange(0, 10, 0.01)),
        #         -np.power(2.0, np.arange(0, 10, 0.01)))).reshape(
        #             [-1, 1, 1])), axis=2),
        # axis=0),
        analyzers._LMomentsAccumulator(
            count_l1=np.array([[2000., 2000.]], dtype=np.float32),
            count_l2=np.array([[1999000., 1999000.]], dtype=np.float32),
            count_l3=np.array([[1.331334e+09, 1.331334e+09]], dtype=np.float32),
            count_l4=np.array(
                [[6.6466854e+11, 6.6466854e+11]], dtype=np.float32),
            l1=np.array([[26.00855, -26.008562]], dtype=np.float32),
            l2=np.array([[103.25489, 103.25489]], dtype=np.float32),
            l3=np.array([[17.549286, -17.549274]], dtype=np.float32),
            l4=np.array([[47.41136, 47.41136]], dtype=np.float32))
    ],
    expected_outputs=[
        np.array([[5.7696896, -5.7697697]], dtype=np.float32),
        np.array([[81.38142, 81.381386]], dtype=np.float32),
        np.array([[0.39079103, 0.55846965]], dtype=np.float32),
        np.array([[0.55846965, 0.39079177]], dtype=np.float32)
    ],
), dict(
    testcase_name='LMomentsMultipleBatchesForNDVectors',
    combiner=analyzers._LMomentsCombiner(np.float32, output_shape=(None, None)),
    batches=[
        # Accumulator for the sequence:
        # np.concatenate((
        #     np.concatenate((
        #         np.power(2.0, np.arange(0, 10, 0.02)),
        #         -np.power(1.9, np.arange(0., 10, 0.02)))).reshape(
        #             [-1, 1, 1]),
        #     np.concatenate((
        #         np.power(1.9, np.arange(0, 10, 0.02)),
        #         -np.power(2.0, np.arange(0., 10, 0.02)))).reshape(
        #             [-1, 1, 1])), axis=2),
        # axis=0)
        analyzers._LMomentsAccumulator(
            count_l1=np.array([[1000., 1000.]], dtype=np.float32),
            count_l2=np.array([[499500., 499500.]], dtype=np.float32),
            count_l3=np.array([[1.66167e+08, 1.66167e+08]], dtype=np.float32),
            count_l4=np.array(
                [[4.1417126e+10, 4.1417126e+10]], dtype=np.float32),
            l1=np.array([[25.90623, -25.90623]], dtype=np.float32),
            l2=np.array([[102.958664, 102.958664]], dtype=np.float32),
            l3=np.array([[17.50719, -17.507195]], dtype=np.float32),
            l4=np.array([[47.393063, 47.393066]], dtype=np.float32)),
        # Accumulator for the sequence:
        # np.concatenate((
        #     np.concatenate((
        #         np.power(2.0, np.arange(0.01, 10, 0.02)),
        #         -np.power(1.9, np.arange(0.01, 10, 0.02)))).reshape(
        #             [-1, 1, 1]),
        #     np.concatenate((
        #         np.power(1.9, np.arange(0.01, 10, 0.02)),
        #         -np.power(2.0, np.arange(0.01, 10, 0.02)))).reshape(
        #             [-1, 1, 1])), axis=2),
        # axis=0)
        analyzers._LMomentsAccumulator(
            count_l1=np.array([[1000., 1000.]], dtype=np.float32),
            count_l2=np.array([[499500., 499500.]], dtype=np.float32),
            count_l3=np.array([[1.66167e+08, 1.66167e+08]], dtype=np.float32),
            count_l4=np.array(
                [[4.1417126e+10, 4.1417126e+10]], dtype=np.float32),
            l1=np.array([[26.110888, -26.110888]], dtype=np.float32),
            l2=np.array([[103.65407, 103.654076]], dtype=np.float32),
            l3=np.array([[17.64386, -17.643852]], dtype=np.float32),
            l4=np.array([[47.71353, 47.71353]], dtype=np.float32))
    ],
    expected_outputs=[
        np.array([[5.751478, -5.751478]], dtype=np.float32),
        np.array([[81.16352, 81.16352]], dtype=np.float32),
        np.array([[0.3923474, 0.55972165]], dtype=np.float32),
        np.array([[0.55972165, 0.3923474]], dtype=np.float32)
    ],
)]

_QUANTILES_NO_ELEMENTS_TEST = dict(
    testcase_name='ComputeQuantilesNoElements',
    combiner=analyzers.QuantilesCombiner(
        num_quantiles=5, epsilon=0.00001, bucket_numpy_dtype=np.float32),
    batches=[
        (np.empty((0, 1), dtype=np.float32),),
    ],
    expected_outputs=[np.zeros((4,), dtype=np.float32)],
)

_QUANTILES_EXACT_NO_ELEMENTS_TEST = dict(
    testcase_name='ComputeExactQuantilesNoElements',
    combiner=analyzers.QuantilesCombiner(
        num_quantiles=5, epsilon=0.00001, bucket_numpy_dtype=np.float32),
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
            num_quantiles=5, epsilon=0.00001, bucket_numpy_dtype=np.float32),
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
            num_quantiles=3, epsilon=0.00001, bucket_numpy_dtype=np.float32),
        batches=[
            (np.linspace(1, 100, 100, np_type),),
        ],
        expected_outputs=[np.array([34, 67], dtype=np.float32)],
    ) for np_type in _NP_TYPES
]

_EXACT_NUM_QUANTILES_TESTS = [
    dict(
        testcase_name='ComputeExactNumQuantiles-{}'.format(np_type),
        combiner=analyzers.QuantilesCombiner(
            num_quantiles=4, epsilon=0.00001, bucket_numpy_dtype=np.float32),
        batches=[
            (np.array([1, 1]),),
        ],
        expected_outputs=[np.array([1, 1, 1], dtype=np.float32)],
    ) for np_type in _NP_TYPES
]
# pylint: enable=g-complex-comprehension


class AnalyzersTest(test_case.TransformTestCase):

  @test_case.named_parameters(
      *[
          _SUM_TEST,
          _SUM_SCALAR_TEST,
          _SUM_OF_SIZE_ZERO_TENSORS_TEST,
          _COVARIANCE_SIZE_ZERO_TENSORS_TEST,
          _COVARIANCE_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST,
          _COVARIANCE_WITH_LARGE_NUMBERS_TEST,
          _PCA_WITH_DEGENERATE_COVARIANCE_MATRIX_TEST,
          _MEAN_AND_VAR_TEST,
          _MEAN_AND_VAR_SIMPLE_TEST,
          _MEAN_AND_VAR_BIG_TEST,
          _MEAN_AND_VAR_VECTORS_TEST,
          _MEAN_AND_VAR_ND_TEST,
          _QUANTILES_NO_ELEMENTS_TEST,
          _QUANTILES_NO_TRIM_TEST,
          _QUANTILES_EXACT_NO_ELEMENTS_TEST,
      ] + _L_MOMENTS_TESTS + _L_MOMENTS_ND_TESTS +
      _QUANTILES_SINGLE_BATCH_TESTS + _QUANTILES_MULTIPLE_BATCH_TESTS +
      _QUANTILES_ELEMENTWISE_TESTS + _EXACT_NUM_QUANTILES_TESTS)
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
      self.assertEqual(tensor_info.dtype, tf.as_dtype(expected_output.dtype))

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
