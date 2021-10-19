# Copyright 2020 Google Inc. All Rights Reserved.
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
"""Tests for tft.tukey_* calls (Tukey HH parameters)."""

import itertools

import apache_beam as beam
import numpy as np

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import impl_test  # Use attributes, but no tests.
from tensorflow_transform.beam import tft_unit


# The input_data in _SCALE_TO_Z_SCORE_TEST_CASES (this is defined in impl_tests
# to test tft.scale_to_z_score) do not have long tails;
# therefore, gaussianization produces the same result of z_score.

_SCALE_TO_GAUSSIAN_TEST_CASES = impl_test._SCALE_TO_Z_SCORE_TEST_CASES + [
    dict(testcase_name='gaussianization_int32',
         input_data=np.array(
             [516, -871, 737, 415, 584, 583, 152, 479, 576, 409,
              591, 844, -16, 508, 669, 617, 502, 532, 517, 479],
             dtype=np.int32),
         output_data=np.array(
             [-0.09304726, -2.24682532, 1.56900163, -0.78244931, 0.48285998,
              0.47461339, -1.50929952, -0.39008015, 0.41659823, -0.81174337,
              0.54027596, 2.11624695, -1.72816411, -0.16046759, 1.13320023,
              0.74814557, -0.21014091, 0.04373742, -0.08454805, -0.39008015],
             dtype=np.float32),
         elementwise=False),
    dict(testcase_name='gaussianization_float32',
         input_data=np.array(
             [516., -871., 737., 415., 584., 583., 152., 479., 576., 409.,
              591., 844., -16., 508., 669., 617., 502., 532., 517., 479.],
             dtype=np.float32),
         output_data=np.array(
             [-0.09304726, -2.24682532, 1.56900163, -0.78244931, 0.48285998,
              0.47461339, -1.50929952, -0.39008015, 0.41659823, -0.81174337,
              0.54027596, 2.11624695, -1.72816411, -0.16046759, 1.13320023,
              0.74814557, -0.21014091, 0.04373742, -0.08454805, -0.39008015],
             dtype=np.float32),
         elementwise=False),
    dict(testcase_name='gaussianization_vector',
         input_data=np.array(
             [[516., -871.], [737., 415.], [584., 583.], [152., 479.],
              [576., 409.], [591., 844.], [-16., 508.], [669., 617.],
              [502., 532.], [517., 479.]],
             dtype=np.float32),
         output_data=np.array(
             [[-0.09304726, -2.24682532], [1.56900163, -0.78244931],
              [0.48285998, 0.47461339], [-1.50929952, -0.39008015],
              [0.41659823, -0.81174337], [0.54027596, 2.11624695],
              [-1.72816411, -0.16046759], [1.13320023, 0.74814557],
              [-0.21014091, 0.04373742], [-0.08454805, -0.39008015]],
             dtype=np.float32),
         elementwise=False),
    dict(testcase_name='gaussianization_vector_elementwise',
         input_data=np.array(
             [[516., -479.], [-871., -517.], [737., -532.], [415., -502.],
              [584., -617.], [583., -669.], [152., -508.], [479., 16.],
              [576., -844.], [409., -591.], [591., -409.], [844., -576.],
              [-16., -479.], [508., -152.], [669., -583.], [617., -584.],
              [502., -415.], [532., -737.], [517., 871.], [479., -516.]],
             dtype=np.float32),
         output_data=np.array(
             [[-0.09304726, 0.39008015], [-2.24682532, 0.08454805],
              [1.56900163, -0.04373742], [-0.78244931, 0.21014091],
              [0.48285998, -0.74814557], [0.47461339, -1.13320023],
              [-1.50929952, 0.16046759], [-0.39008015, 1.72816411],
              [0.41659823, -2.11624695], [-0.81174337, -0.54027596],
              [0.54027596, 0.81174337], [2.11624695, -0.41659823],
              [-1.72816411, 0.39008015], [-0.16046759, 1.50929952],
              [1.13320023, -0.47461339], [0.74814557, -0.48285998],
              [-0.21014091, 0.78244931], [0.04373742, -1.56900163],
              [-0.08454805, 2.24682532], [-0.39008015, 0.09304726]],
             dtype=np.float32),
         elementwise=True),
]


class TukeyHHParamsIntegrationTest(tft_unit.TransformTestCase):

  def setUp(self):
    self._context = beam_impl.Context(use_deep_copy_optimization=True)
    self._context.__enter__()
    super().setUp()

  def tearDown(self):
    self._context.__exit__()
    super().tearDown()

  @tft_unit.named_parameters(*_SCALE_TO_GAUSSIAN_TEST_CASES)
  def testGaussianize(self, input_data, output_data, elementwise):

    def preprocessing_fn(inputs):
      x = inputs['x']
      x_cast = tf.cast(x, tf.as_dtype(input_data.dtype))
      x_gaussianized = tft.scale_to_gaussian(x_cast, elementwise=elementwise)
      self.assertEqual(x_gaussianized.dtype, tf.as_dtype(output_data.dtype))
      return {'x_gaussianized': tf.cast(x_gaussianized, tf.float32)}

    input_data_dicts = [{'x': x} for x in input_data]
    expected_data_dicts = [
        {'x_gaussianized': x_gaussianized} for x_gaussianized in output_data]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x':
            tf.io.FixedLenFeature(
                input_data.shape[1:],
                tft_unit.canonical_numeric_dtype(tf.as_dtype(
                    input_data.dtype))),
    })
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_gaussianized': tf.io.FixedLenFeature(
            output_data.shape[1:], tf.float32),
    })
    self.assertAnalyzeAndTransformResults(
        input_data_dicts, input_metadata, preprocessing_fn, expected_data_dicts,
        expected_metadata, desired_batch_size=20, beam_pipeline=beam.Pipeline())

  @tft_unit.parameters(*itertools.product([
      tf.int16,
      tf.int32,
      tf.int64,
      tf.float32,
      tf.float64,
  ], (True, False)))
  def testGaussianizeSparse(self, input_dtype, elementwise):

    def preprocessing_fn(inputs):
      x_gaussianized = tft.scale_to_gaussian(
          tf.cast(inputs['x'], input_dtype), elementwise=elementwise)
      self.assertEqual(x_gaussianized.dtype,
                       impl_test._mean_output_dtype(input_dtype))
      return {
          'x_gaussianized': tf.cast(x_gaussianized, tf.float32)
      }

    input_data_values = [516, -871, 737, 415, 584, 583, 152, 479, 576, 409, 591,
                         844, -16, 508, 669, 617, 502, 532, 517, 479]
    input_data = []
    for idx, v in enumerate(input_data_values):
      input_data.append({
          'idx0': [1, 1],
          'idx1': [0, 1],
          'val': [v, -input_data_values[-1 - idx]]
      })
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x':
            tf.io.SparseFeature(['idx0', 'idx1'], 'val',
                                tft_unit.canonical_numeric_dtype(input_dtype),
                                (4, 5))
    })
    if elementwise:
      expected_data_values = [
          -0.09304726, -2.24682532, 1.56900163, -0.78244931, 0.48285998,
          0.47461339, -1.50929952, -0.39008015, 0.41659823, -0.81174337,
          0.54027596, 2.11624695, -1.72816411, -0.16046759, 1.13320023,
          0.74814557, -0.21014091, 0.04373742, -0.08454805, -0.39008015]
    else:
      expected_data_values = [
          0.91555131, -1.54543642, 1.30767697, 0.73634456, 1.03620536,
          1.03443104, 0.26969729, 0.84990131, 1.02201077, 0.72569862,
          1.04862563, 1.49752966, -0.02838919, 0.90135672, 1.18702292,
          1.09475806, 0.89071077, 0.9439405, 0.91732564, 0.84990131]
    expected_data = []
    for idx, v in enumerate(expected_data_values):
      expected_data.append({
          'x_gaussianized$sparse_values': ([v,
                                            -expected_data_values[-1 - idx]]),
          'x_gaussianized$sparse_indices_0': [1, 1],
          'x_gaussianized$sparse_indices_1': [0, 1],
      })

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        desired_batch_size=20,
        beam_pipeline=beam.Pipeline())

  @tft_unit.parameters(
      (tf.int16,),
      (tf.int32,),
      (tf.int64,),
      (tf.float32,),
      (tf.float64,),
  )
  def testGaussianizeRagged(self, input_dtype):
    tft_unit.skip_if_not_tf2('RaggedFeature is not available in TF 1.x.')

    def preprocessing_fn(inputs):
      x_gaussianized = tft.scale_to_gaussian(tf.cast(inputs['x'], input_dtype))
      self.assertEqual(x_gaussianized.dtype,
                       impl_test._mean_output_dtype(input_dtype))
      return {'x_gaussianized': tf.cast(x_gaussianized, tf.float32)}

    input_data_values = [
        516, -871, 737, 415, 584, 583, 152, 479, 576, 409, 591, 844, -16, 508,
        669, 617, 502, 532, 517, 479
    ]
    input_data = []
    for idx, v in enumerate(input_data_values):
      input_data.append({
          'val': [v, -input_data_values[-1 - idx]],
          'row_lengths_1': [2, 1, 0],
          'row_lengths_2': [1, 0, 1],
      })
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x':
            tf.io.RaggedFeature(
                tft_unit.canonical_numeric_dtype(input_dtype),
                value_key='val',
                partitions=[
                    tf.io.RaggedFeature.RowLengths('row_lengths_1'),  # pytype: disable=attribute-error
                    tf.io.RaggedFeature.RowLengths('row_lengths_2')  # pytype: disable=attribute-error
                ]),
    })
    expected_data_values = [
        0.91555131, -1.54543642, 1.30767697, 0.73634456, 1.03620536, 1.03443104,
        0.26969729, 0.84990131, 1.02201077, 0.72569862, 1.04862563, 1.49752966,
        -0.02838919, 0.90135672, 1.18702292, 1.09475806, 0.89071077, 0.9439405,
        0.91732564, 0.84990131
    ]
    expected_data = []
    for idx, v in enumerate(expected_data_values):
      expected_data.append({
          'x_gaussianized$ragged_values': ([v,
                                            -expected_data_values[-1 - idx]]),
          'x_gaussianized$row_lengths_1': [2, 1, 0],
          'x_gaussianized$row_lengths_2': [1, 0, 1]
      })

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        desired_batch_size=20,
        # Runs the test deterministically on the whole batch.
        beam_pipeline=beam.Pipeline())

  @tft_unit.named_parameters(
      dict(
          testcase_name='tukey_int64in',
          input_dtype=tf.int64,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          }),
      dict(
          testcase_name='tukey_int32in',
          input_dtype=tf.int32,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          }),
      dict(
          testcase_name='tukey_int16in',
          input_dtype=tf.int16,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          }),
      dict(
          testcase_name='tukey_float64in',
          input_dtype=tf.float64,
          output_dtypes={
              'tukey_location': tf.float64,
              'tukey_scale': tf.float64,
              'tukey_hl': tf.float64,
              'tukey_hr': tf.float64
          }),
      dict(
          testcase_name='tukey_float32in',
          input_dtype=tf.float32,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          },
          elementwise=True),
      dict(
          testcase_name='tukey_float32in_reduce',
          input_dtype=tf.float32,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          },
          elementwise=False),
  )
  def testTukeyHHAnalyzersWithDenseInputs(
      self, input_dtype, output_dtypes, elementwise=True):

    def analyzer_fn(inputs):
      a = tf.cast(inputs['a'], input_dtype)

      def assert_and_cast_dtype(tensor, out_dtype):
        self.assertEqual(tensor.dtype, out_dtype)
        return tf.cast(tensor, tft_unit.canonical_numeric_dtype(out_dtype))

      return {
          'tukey_location': assert_and_cast_dtype(
              tft.tukey_location(a, reduce_instance_dims=not elementwise),
              output_dtypes['tukey_location']),
          'tukey_scale': assert_and_cast_dtype(
              tft.tukey_scale(a, reduce_instance_dims=not elementwise),
              output_dtypes['tukey_scale']),
          'tukey_hl': assert_and_cast_dtype(
              tft.tukey_h_params(a, reduce_instance_dims=not elementwise)[0],
              output_dtypes['tukey_hl']),
          'tukey_hr': assert_and_cast_dtype(
              tft.tukey_h_params(a, reduce_instance_dims=not elementwise)[1],
              output_dtypes['tukey_hr']),
      }

    input_data_values = [516, -871, 737, 415, 584, 583, 152, 479, 576, 409, 591,
                         844, -16, 508, 669, 617, 502, 532, 517, 479]
    input_data = []
    for idx, v in enumerate(input_data_values):
      input_data.append({'a': [v, -input_data_values[-1 - idx]]})
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature(
            [2], tft_unit.canonical_numeric_dtype(input_dtype))
    })
    expected_outputs = {
        'tukey_location':
            np.array(
                [526.89355, -526.89355] if elementwise else 0.0,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_location']).as_numpy_dtype),
        'tukey_scale':
            np.array(
                [116.73997, 116.73997] if elementwise else 572.277649,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_scale']).as_numpy_dtype),
        'tukey_hl':
            np.array(
                [0.6629082, 0.11148566] if elementwise else 0.0,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_hl']).as_numpy_dtype),
        'tukey_hr':
            np.array(
                [0.11148566, 0.6629082] if elementwise else 0.0,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_hr']).as_numpy_dtype),
    }

    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=20,
        # Runs the test deterministically on the whole batch.
        beam_pipeline=beam.Pipeline())

  def testTukeyHHAnalyzersWithNDDenseInputs(self):

    def analyzer_fn(inputs):
      a = inputs['a']

      return {
          'tukey_location': tft.tukey_location(a, reduce_instance_dims=False),
          'tukey_scale': tft.tukey_scale(a, reduce_instance_dims=False),
          'tukey_hl': tft.tukey_h_params(a, reduce_instance_dims=False)[0],
          'tukey_hr': tft.tukey_h_params(a, reduce_instance_dims=False)[1],
      }

    input_data_values = [516, -871, 737, 415, 584, 583, 152, 479, 576, 409, 591,
                         844, -16, 508, 669, 617, 502, 532, 517, 479]
    input_data = []
    for idx, v in enumerate(input_data_values):
      input_data.append({'a': [
          [v, -input_data_values[-1 - idx]],
          [2 * v, -2 * input_data_values[-1 - idx]]]})
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([2, 2], tf.float32)
    })
    expected_outputs = {
        'tukey_location':
            np.array(
                [[526.89355, -526.89355], [2. * 526.89355, -2. * 526.89355]],
                np.float32),
        'tukey_scale':
            np.array([[116.73997, 116.73997], [2. * 116.73997, 2. * 116.73997]],
                     np.float32),
        'tukey_hl':
            np.array(
                [[0.6629082, 0.11148566], [0.6629082, 0.11148566]], np.float32),
        'tukey_hr':
            np.array(
                [[0.11148566, 0.6629082], [0.11148566, 0.6629082]], np.float32)
    }

    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=20,
        # Runs the test deterministically on the whole batch.
        beam_pipeline=beam.Pipeline())

  @tft_unit.named_parameters(
      dict(
          testcase_name='_int64in',
          input_dtype=tf.int64,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          }),
      dict(
          testcase_name='_int32in',
          input_dtype=tf.int32,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          }),
      dict(
          testcase_name='_int16in',
          input_dtype=tf.int16,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          }),
      dict(
          testcase_name='_float64in',
          input_dtype=tf.float64,
          output_dtypes={
              'tukey_location': tf.float64,
              'tukey_scale': tf.float64,
              'tukey_hl': tf.float64,
              'tukey_hr': tf.float64
          }),
      dict(
          testcase_name='_float32in',
          input_dtype=tf.float32,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          },
          elementwise=True
      ),
      dict(
          testcase_name='_float32in_reduce',
          input_dtype=tf.float32,
          output_dtypes={
              'tukey_location': tf.float32,
              'tukey_scale': tf.float32,
              'tukey_hl': tf.float32,
              'tukey_hr': tf.float32
          },
          elementwise=False
      ),
  )
  def testTukeyHHAnalyzersWithSparseInputs(
      self, input_dtype, output_dtypes, elementwise=True):

    def analyzer_fn(inputs):
      a = tf.cast(inputs['a'], input_dtype)

      def assert_and_cast_dtype(tensor, out_dtype):
        self.assertEqual(tensor.dtype, out_dtype)
        return tf.cast(tensor, tft_unit.canonical_numeric_dtype(out_dtype))

      return {
          'tukey_location': assert_and_cast_dtype(
              tft.tukey_location(a, reduce_instance_dims=not elementwise),
              output_dtypes['tukey_location']),
          'tukey_scale': assert_and_cast_dtype(
              tft.tukey_scale(a, reduce_instance_dims=not elementwise),
              output_dtypes['tukey_scale']),
          'tukey_hl': assert_and_cast_dtype(
              tft.tukey_h_params(a, reduce_instance_dims=not elementwise)[0],
              output_dtypes['tukey_hl']),
          'tukey_hr': assert_and_cast_dtype(
              tft.tukey_h_params(a, reduce_instance_dims=not elementwise)[1],
              output_dtypes['tukey_hr']),
      }

    input_data_values = [516, -871, 737, 415, 584, 583, 152, 479, 576, 409, 591,
                         844, -16, 508, 669, 617, 502, 532, 517, 479]
    input_data = []
    for idx, v in enumerate(input_data_values):
      input_data.append({
          'idx0': [0, 0],
          'idx1': [0, 1],
          'val': [v, -input_data_values[-1 - idx]]
      })
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a':
            tf.io.SparseFeature(['idx0', 'idx1'], 'val',
                                tft_unit.canonical_numeric_dtype(input_dtype),
                                (2, 2))
    })

    expected_outputs = {
        'tukey_location':
            np.array(
                [[526.89355, -526.89355], [0., 0.]] if elementwise else 0.0,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_location']).as_numpy_dtype),
        'tukey_scale':
            np.array(
                [[116.73997, 116.73997], [1., 1.]] if elementwise else 572.2776,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_scale']).as_numpy_dtype),
        'tukey_hl':
            np.array(
                [[0.6629082, 0.11148566], [0., 0.]] if elementwise else 0.0,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_hl']).as_numpy_dtype),
        'tukey_hr':
            np.array(
                [[0.11148566, 0.6629082], [0., 0.]] if elementwise else 0.0,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_hr']).as_numpy_dtype),
    }

    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=20,
        # Runs the test deterministically on the whole batch.
        beam_pipeline=beam.Pipeline())

  @tft_unit.parameters(
      (tf.int16,),
      (tf.int32,),
      (tf.int64,),
      (tf.float32,),
      (tf.float64,),
  )
  def testTukeyHHAnalyzersWithRaggedInputs(self, input_dtype):
    tft_unit.skip_if_not_tf2('RaggedFeature is not available in TF 1.x.')

    output_dtype = impl_test._mean_output_dtype(input_dtype)
    canonical_output_dtype = tft_unit.canonical_numeric_dtype(output_dtype)

    def analyzer_fn(inputs):
      a = tf.cast(inputs['a'], input_dtype)

      def assert_and_cast_dtype(tensor):
        self.assertEqual(tensor.dtype, output_dtype)
        return tf.cast(tensor, canonical_output_dtype)

      return {
          'tukey_location': assert_and_cast_dtype(tft.tukey_location(a)),
          'tukey_scale': assert_and_cast_dtype(tft.tukey_scale(a)),
          'tukey_hl': assert_and_cast_dtype(tft.tukey_h_params(a)[0]),
          'tukey_hr': assert_and_cast_dtype(tft.tukey_h_params(a)[1]),
      }

    input_data_values = [
        516, -871, 737, 415, 584, 583, 152, 479, 576, 409, 591, 844, -16, 508,
        669, 617, 502, 532, 517, 479
    ]
    input_data = []
    for idx, v in enumerate(input_data_values):
      input_data.append({
          'val': [v, -input_data_values[-1 - idx]],
          'row_lengths_1': [2, 0, 1],
          'row_lengths_2': [0, 1, 1]
      })
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a':
            tf.io.RaggedFeature(
                tft_unit.canonical_numeric_dtype(input_dtype),
                value_key='val',
                partitions=[
                    tf.io.RaggedFeature.RowLengths('row_lengths_1'),  # pytype: disable=attribute-error
                    tf.io.RaggedFeature.RowLengths('row_lengths_2')  # pytype: disable=attribute-error
                ]),
    })

    expected_outputs = {
        'tukey_location':
            np.array(0.0, canonical_output_dtype.as_numpy_dtype),
        'tukey_scale':
            np.array(572.2776, canonical_output_dtype.as_numpy_dtype),
        'tukey_hl':
            np.array(0.0, canonical_output_dtype.as_numpy_dtype),
        'tukey_hr':
            np.array(0.0, canonical_output_dtype.as_numpy_dtype),
    }

    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=20,
        # Runs the test deterministically on the whole batch.
        beam_pipeline=beam.Pipeline())

if __name__ == '__main__':
  tft_unit.main()
