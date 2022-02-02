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

_TYPE_SPEC = {
    'x': tf.TensorSpec([None], tf.float32),
    'y': tf.SparseTensorSpec(shape=[None, None], dtype=tf.int64),
    's': tf.TensorSpec([None], tf.string),
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
  return {'id_{}'.format(key): value for key, value in inputs.items()}


@tf.function
def _plus_one(x):
  return x + 1


def _one_phase_preprocessing_fn(inputs):
  x_plus_one = _plus_one(inputs['x'])
  subtracted = tf.sparse.add(
      tf.cast(inputs['y'], tf.float32), -analyzers.mean(x_plus_one))
  _ = analyzers.vocabulary(inputs['s'])
  return {'subtracted': subtracted}


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


def _preprocessing_fn_with_control_dependency(inputs):
  with tf.init_scope():
    initializer = tf.lookup.KeyValueTensorInitializer(['foo', 'bar'], [0, 1])
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
  # The table created here will add an automatic control dependency.
  s_int = table.lookup(inputs['s']) + 1

  # Perform some TF Ops to ensure x is part of the graph of dependencies for the
  # outputs.
  x_abs = tf.math.abs(inputs['x'])
  y_centered = (
      tf.sparse.add(
          tf.cast(inputs['y'], tf.float32), -analyzers.mean(inputs['y'])))
  return {'s_int': s_int, 'x_abs': x_abs, 'y_centered': y_centered}


class InspectPreprocessingFnTest(test_case.TransformTestCase):

  @test_case.named_parameters(
      *test_case.cross_named_parameters([
          dict(
              testcase_name='identity',
              preprocessing_fn=_identity_preprocessing_fn,
              expected_analyze_input_columns=[],
              expected_transform_input_columns=['x', 'y', 's']),
          dict(
              testcase_name='side_affect',
              preprocessing_fn=_side_affect_preprocessing_fn,
              expected_analyze_input_columns=['s'],
              expected_transform_input_columns=[]),
          dict(
              testcase_name='non_identity_ops',
              preprocessing_fn=_non_identity_ops_preprocessing_fn,
              expected_analyze_input_columns=[],
              expected_transform_input_columns=['x', 'y', 's']),
          dict(
              testcase_name='feature_renaming',
              preprocessing_fn=_renaming_preprocessing_fn,
              expected_analyze_input_columns=[],
              expected_transform_input_columns=['x', 'y', 's']),
          dict(
              testcase_name='one_phase',
              preprocessing_fn=_one_phase_preprocessing_fn,
              expected_analyze_input_columns=['x', 's'],
              expected_transform_input_columns=['y']),
          dict(
              testcase_name='two_phases',
              preprocessing_fn=_two_phases_preprocessing_fn,
              expected_analyze_input_columns=['x', 'y', 's'],
              expected_transform_input_columns=['x', 's'])
      ], [
          dict(testcase_name='tf_compat_v1', force_tf_compat_v1=True),
          dict(testcase_name='tf2', force_tf_compat_v1=False)
      ]),
      *test_case.cross_named_parameters([
          dict(
              testcase_name='control_dependencies',
              preprocessing_fn=_preprocessing_fn_with_control_dependency,
              expected_transform_input_columns=['x', 'y', 's'])
      ], [
          dict(
              testcase_name='tf_compat_v1',
              force_tf_compat_v1=True,
              expected_analyze_input_columns=['y']),
          dict(
              testcase_name='tf2',
              force_tf_compat_v1=False,
              expected_analyze_input_columns=['s', 'y'])
      ]))
  def test_column_inference(self, preprocessing_fn,
                            expected_analyze_input_columns,
                            expected_transform_input_columns,
                            force_tf_compat_v1):
    if not force_tf_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')
      specs = _TYPE_SPEC
    else:
      specs = _FEATURE_SPEC

    analyze_input_columns = (
        inspect_preprocessing_fn.get_analyze_input_columns(
            preprocessing_fn, specs, force_tf_compat_v1))
    transform_input_columns = (
        inspect_preprocessing_fn.get_transform_input_columns(
            preprocessing_fn, specs, force_tf_compat_v1))
    self.assertCountEqual(analyze_input_columns, expected_analyze_input_columns)
    self.assertCountEqual(transform_input_columns,
                          expected_transform_input_columns)


if __name__ == '__main__':
  test_case.main()
