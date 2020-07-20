# Lint as: python3
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

# GOOGLE-INITIALIZATION

import apache_beam as beam
import numpy as np

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import tft_unit


class TukeyHHParamsIntegrationTest(tft_unit.TransformTestCase):

  def setUp(self):
    self._context = beam_impl.Context(use_deep_copy_optimization=True)
    self._context.__enter__()
    super(TukeyHHParamsIntegrationTest, self).setUp()

  def tearDown(self):
    self._context.__exit__()
    super(TukeyHHParamsIntegrationTest, self).tearDown()

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
      input_data.append({'a': [v] + [-input_data_values[-1 - idx]]})
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
    for i, v in enumerate(input_data_values):
      input_data.append({'a': [
          [v, -input_data_values[-1 - i]],
          [2 * v, -2 * input_data_values[-1 - i]]]})
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
    for i, v in enumerate(input_data_values):
      input_data.append({'idx': [0, 1],
                         'val': [v] + [-input_data_values[-1 - i]]})
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a':
        tf.io.SparseFeature('idx', 'val',
                            tft_unit.canonical_numeric_dtype(input_dtype),
                            4)
    })

    expected_outputs = {
        'tukey_location':
            np.array(
                [526.89355, -526.89355, 0., 0.] if elementwise else 0.0,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_location']).as_numpy_dtype),
        'tukey_scale':
            np.array(
                [116.73997, 116.73997, 1., 1.] if elementwise else 572.277649,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_scale']).as_numpy_dtype),
        'tukey_hl':
            np.array(
                [0.6629082, 0.11148566, 0., 0.] if elementwise else 0.0,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['tukey_hl']).as_numpy_dtype),
        'tukey_hr':
            np.array(
                [0.11148566, 0.6629082, 0., 0.] if elementwise else 0.0,
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

if __name__ == '__main__':
  tft_unit.main()
