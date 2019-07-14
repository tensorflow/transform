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
"""Tests for inspect_preprocessing_fn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# GOOGLE-INITIALIZATION
import six
import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import inspect_preprocessing_fn
from tensorflow_transform import mappers
from tensorflow_transform import test_case

_FEATURE_SPEC = {
    'x': tf.io.FixedLenFeature([], tf.float32),
    'y': tf.io.VarLenFeature(tf.int64),
    's': tf.io.FixedLenFeature([], tf.string),
}


def _identity_preprocessing_fn(inputs):
  return inputs.copy()


def _side_affect_preprocessing_fn(inputs):
  _ = analyzers.vocabulary(inputs['s'])
  return {}


def _non_identity_ops_preprocessing_fn(inputs):
  outputs = inputs.copy()
  outputs['new_feature'] = tf.constant(1)
  return outputs


def _renaming_preprocessing_fn(inputs):
  return {'id_{}'.format(key): value for key, value in six.iteritems(inputs)}


def _one_phase_preprocessing_fn(inputs):
  x_centered = inputs['x'] - analyzers.mean(inputs['y'])
  _ = analyzers.vocabulary(inputs['s'])
  return {'x_centered': x_centered}


def _two_phases_preprocessing_fn(inputs):
  x = inputs['x']
  x_mean = analyzers.mean(x)
  x_square_deviations = tf.square(x - x_mean)
  x_var = analyzers.mean(x_square_deviations + analyzers.mean(inputs['y']))
  x_normalized = (x - x_mean) / tf.sqrt(x_var)
  return {
      'x_normalized': x_normalized,
      's_id': mappers.compute_and_apply_vocabulary(inputs['s'])
  }


class InspectPreprocessingFnTest(test_case.TransformTestCase):

  @test_case.named_parameters(
      ('identity', _identity_preprocessing_fn, [], ['x', 'y', 's']),
      ('side_affect', _side_affect_preprocessing_fn, ['s'], []),
      ('non_identity_ops', _non_identity_ops_preprocessing_fn, [],
       ['x', 'y', 's']),
      ('feature_renaming', _renaming_preprocessing_fn, [], ['x', 'y', 's']),
      ('one_phase', _one_phase_preprocessing_fn, ['y', 's'], ['x']),
      ('two_phases', _two_phases_preprocessing_fn, ['x', 'y', 's'], ['x', 's']),
  )
  def test_column_inference(self, preprocessing_fn,
                            expected_anazlye_input_columns,
                            expected_transform_input_columns):
    analyze_input_columns = (
        inspect_preprocessing_fn.get_analyze_input_columns(
            preprocessing_fn, _FEATURE_SPEC))
    transform_input_columns = (
        inspect_preprocessing_fn.get_transform_input_columns(
            preprocessing_fn, _FEATURE_SPEC))
    self.assertCountEqual(analyze_input_columns, expected_anazlye_input_columns)
    self.assertCountEqual(transform_input_columns,
                          expected_transform_input_columns)


if __name__ == '__main__':
  test_case.main()
