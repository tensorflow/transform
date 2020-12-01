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
"""Tests for tensorflow_transform.impl_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

# GOOGLE-INITIALIZATION

import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import impl_helper
from tensorflow_transform import test_case
from tensorflow_transform.output_wrapper import TFTransformOutput
from tensorflow_transform.tf_metadata import schema_utils


_FEATURE_SPEC = {
    'a':
        tf.io.FixedLenFeature([], tf.int64),
    'b':
        tf.io.FixedLenFeature([], tf.float32),
    'c':
        tf.io.FixedLenFeature([1], tf.float32),
    'd':
        tf.io.FixedLenFeature([2, 2], tf.float32),
    'e':
        tf.io.VarLenFeature(tf.string),
    'f':
        tf.io.SparseFeature('idx', 'val', tf.float32, 10),
    'g':
        tf.io.SparseFeature(['g_idx0', 'g_idx1'], 'g_val', tf.float32, [2, 10]),
}
_FEED_DICT = {
    'a':
        np.array([100, 100]),
    'b':
        np.array([1.0, 2.0]),
    'c':
        np.array([[2.0], [4.0]]),
    'd':
        np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    'e':
        tf.compat.v1.SparseTensorValue(
            indices=np.array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
            values=np.array([b'doe', b'a', b'deer', b'a', b'female', b'deer']),
            dense_shape=(2, 3)),
    'f':
        tf.compat.v1.SparseTensorValue(
            indices=np.array([(0, 2), (0, 4), (0, 8)]),
            values=np.array([10.0, 20.0, 30.0]),
            dense_shape=(2, 10)),
    'g':
        tf.compat.v1.SparseTensorValue(
            indices=np.array([(0, 0, 3), (0, 1, 5), (0, 1, 9)]),
            values=np.array([110.0, 210.0, 310.0]),
            dense_shape=(2, 2, 10)),
}

_ROUNDTRIP_CASES = [
    dict(
        testcase_name='multiple_features',
        feature_spec=_FEATURE_SPEC,
        instances=[{
            'a': 100,
            'b': 1.0,
            'c': [2.0],
            'd': [[1.0, 2.0], [3.0, 4.0]],
            'e': [b'doe', b'a', b'deer'],
            'idx': [2, 4, 8],
            'val': [10.0, 20.0, 30.0],
            'g_idx0': [0, 1, 1],
            'g_idx1': [3, 5, 9],
            'g_val': [110.0, 210.0, 310.0],
        }, {
            'a': 100,
            'b': 2.0,
            'c': [4.0],
            'd': [[5.0, 6.0], [7.0, 8.0]],
            'e': [b'a', b'female', b'deer'],
            'idx': [],
            'val': [],
            'g_idx0': [],
            'g_idx1': [],
            'g_val': [],
        }],
        feed_dict=_FEED_DICT),
    dict(
        testcase_name='multiple_features_ndarrays',
        feature_spec=_FEATURE_SPEC,
        instances=[{
            'a': np.int64(100),
            'b': np.array(1.0, np.float32),
            'c': np.array([2.0], np.float32),
            'd': np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
            'e': [b'doe', b'a', b'deer'],
            'idx': np.array([2, 4, 8]),
            'val': np.array([10.0, 20.0, 30.0]),
            'g_idx0': np.array([0, 1, 1]),
            'g_idx1': np.array([3, 5, 9]),
            'g_val': np.array([110.0, 210.0, 310.0]),
        }, {
            'a': np.int64(100),
            'b': np.array(2.0, np.float32),
            'c': np.array([4.0], np.float32),
            'd': np.array([[5.0, 6.0], [7.0, 8.0]], np.float32),
            'e': [b'a', b'female', b'deer'],
            'idx': np.array([], np.int32),
            'val': np.array([], np.float32),
            'g_idx0': np.array([], np.float32),
            'g_idx1': np.array([], np.float32),
            'g_val': np.array([], np.float32),
        }],
        feed_dict=_FEED_DICT),
    dict(
        testcase_name='empty_var_len_feature',
        feature_spec={'varlen': tf.io.VarLenFeature(tf.string)},
        instances=[{
            'varlen': []
        }],
        feed_dict={
            'varlen':
                tf.compat.v1.SparseTensorValue(
                    indices=np.empty([0, 2]),
                    values=np.array([]),
                    dense_shape=[1, 0])
        }),
    # Mainly to test the empty-ndarray optimization though this is also
    # exercised by empty_var_len_feature
    dict(
        testcase_name='some_empty_int_var_len_feature',
        feature_spec={'varlen': tf.io.VarLenFeature(tf.int64)},
        instances=[{
            'varlen': [0]
        }, {
            'varlen': []
        }, {
            'varlen': [1]
        }, {
            'varlen': []
        }],
        feed_dict={
            'varlen':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([(0, 0), (2, 0)]),
                    values=np.array([0, 1], np.int64),
                    dense_shape=(4, 1)),
        }),
    dict(
        testcase_name='some_empty_float_var_len_feature',
        feature_spec={'varlen': tf.io.VarLenFeature(tf.float32)},
        instances=[{
            'varlen': [0.5]
        }, {
            'varlen': []
        }, {
            'varlen': [1.5]
        }, {
            'varlen': []
        }],
        feed_dict={
            'varlen':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([(0, 0), (2, 0)]),
                    values=np.array([0.5, 1.5], np.float32),
                    dense_shape=(4, 1)),
        }),
    dict(
        testcase_name='some_empty_string_var_len_feature',
        feature_spec={'varlen': tf.io.VarLenFeature(tf.string)},
        instances=[{
            'varlen': [b'a']
        }, {
            'varlen': []
        }, {
            'varlen': [b'b']
        }, {
            'varlen': []
        }],
        feed_dict={
            'varlen':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([(0, 0), (2, 0)]),
                    values=np.array([b'a', b'b'], np.object),
                    dense_shape=(4, 1)),
        }),
    dict(
        testcase_name='empty_sparse_feature',
        feature_spec={
            'sparse': tf.io.SparseFeature('idx', 'val', tf.float32, 10)
        },
        instances=[{
            'idx': [],
            'val': []
        }],
        feed_dict={
            'sparse':
                tf.compat.v1.SparseTensorValue(
                    indices=np.empty([0, 2]),
                    values=np.array([]),
                    dense_shape=[1, 10])
        }),
    dict(
        testcase_name='non_ragged_sparse_feature',
        feature_spec={
            'sparse': tf.io.SparseFeature('idx', 'val', tf.float32, 10)
        },
        instances=[{
            'idx': [],
            'val': []
        }, {
            'idx': [9],
            'val': [0.3]
        }],
        feed_dict={
            'sparse':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([[1, 9]]),
                    values=np.array([0.3]),
                    dense_shape=[2, 10])
        }),
]

# Non-canonical inputs that will not be the output of to_instance_dicts but
# are valid inputs to make_feed_dict.
_MAKE_FEED_DICT_CASES = [
    dict(
        testcase_name='none_feature',
        feature_spec={
            'varlen_feature': tf.io.VarLenFeature(tf.int64),
        },
        instances=[{
            'varlen_feature': []
        }, {
            'varlen_feature': None
        }, {
            'varlen_feature': [1, 2]
        }],
        feed_dict={
            'varlen_feature':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([(2, 0), (2, 1)]),
                    values=np.array([1, 2]),
                    dense_shape=[3, 2])
        }),
]

_TO_INSTANCE_DICT_ERROR_CASES = [
    dict(
        testcase_name='var_len_with_non_consecutive_indices',
        feature_spec={'a': tf.io.VarLenFeature(tf.float32)},
        feed_dict={
            'a':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([(0, 2), (0, 4), (0, 8)]),
                    values=np.array([10.0, 20.0, 30.0]),
                    dense_shape=(1, 20))
        },
        error_msg='cannot be decoded by ListColumnRepresentation'),
    dict(
        testcase_name='var_len_with_rank_not_2',
        feature_spec={'a': tf.io.VarLenFeature(tf.float32)},
        feed_dict={
            'a':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([(0, 0, 1), (0, 0, 2), (0, 0, 3)]),
                    values=np.array([10.0, 20.0, 30.0]),
                    dense_shape=(1, 10, 10))
        },
        error_msg='cannot be decoded by ListColumnRepresentation'),
    dict(
        testcase_name='var_len_with_out_of_order_indices',
        feature_spec={'a': tf.io.VarLenFeature(tf.float32)},
        feed_dict={
            'a':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([(0, 2), (2, 4), (1, 8)]),
                    values=np.array([10.0, 20.0, 30.0]),
                    dense_shape=(3, 20))
        },
        error_msg='Encountered out-of-order sparse index'),
    dict(
        testcase_name='var_len_with_different_batch_dim_sizes',
        feature_spec={
            'a': tf.io.VarLenFeature(tf.float32),
            'b': tf.io.VarLenFeature(tf.float32),
        },
        feed_dict={
            'a':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([(0, 0)]),
                    values=np.array([10.0]),
                    dense_shape=(1, 20)),
            'b':
                tf.compat.v1.SparseTensorValue(
                    indices=np.array([(0, 0)]),
                    values=np.array([10.0]),
                    dense_shape=(2, 20)),
        },
        error_msg=(r'Inconsistent batch sizes: "\w" had batch dimension \d, '
                   r'"\w" had batch dimension \d')),
    dict(
        testcase_name='fixed_len_with_different_batch_dim_sizes',
        feature_spec={
            'a': tf.io.FixedLenFeature([], tf.float32),
            'b': tf.io.FixedLenFeature([], tf.float32),
        },
        feed_dict={
            'a': np.array([1]),
            'b': np.array([1, 2])
        },
        error_msg=(r'Inconsistent batch sizes: "\w" had batch dimension \d, '
                   r'"\w" had batch dimension \d')),
]


def _get_value_from_eager_tensors(eager_tensors):
  """Given a list of eager tensors, get their values in a TF1 compat format."""
  result = []
  for tensor in eager_tensors:
    if isinstance(tensor, tf.Tensor):
      result.append(tensor.numpy())
    elif isinstance(tensor, tf.sparse.SparseTensor):
      result.append(
          tf.compat.v1.SparseTensorValue(
              indices=tensor.indices.numpy(),
              values=tensor.values.numpy(),
              dense_shape=tensor.dense_shape.numpy()))
    else:
      raise ValueError('Expected tf.Tensor or tf.SparseTensor')
  return result


class ImplHelperTest(test_case.TransformTestCase):

  def test_batched_placeholders_from_feature_spec(self):
    feature_spec = {
        'fixed_len_float':
            tf.io.FixedLenFeature([2, 3], tf.float32),
        'fixed_len_string':
            tf.io.FixedLenFeature([], tf.string),
        '_var_len_underscored':
            tf.io.VarLenFeature(tf.string),
        'var_len_int':
            tf.io.VarLenFeature(tf.int64),
        'sparse_1d':
            tf.io.SparseFeature('1d_idx', '1d_value', tf.int64, 7),
        'sparse_2d':
            tf.io.SparseFeature(['2d_idx0', '2d_idx1'], '2d_value', tf.int64,
                                [2, 17]),
    }
    with tf.compat.v1.Graph().as_default():
      features = impl_helper.batched_placeholders_from_specs(feature_spec)
    self.assertCountEqual(features.keys(), [
        'fixed_len_float',
        'fixed_len_string',
        'var_len_int',
        '_var_len_underscored',
        'sparse_1d',
        'sparse_2d',
    ])
    self.assertEqual(type(features['fixed_len_float']), tf.Tensor)
    self.assertEqual(features['fixed_len_float'].get_shape().as_list(),
                     [None, 2, 3])
    self.assertEqual(type(features['fixed_len_string']), tf.Tensor)
    self.assertEqual(features['fixed_len_string'].get_shape().as_list(),
                     [None])
    self.assertEqual(type(features['var_len_int']), tf.SparseTensor)
    self.assertEqual(features['var_len_int'].get_shape().as_list(),
                     [None, None])
    self.assertEqual(type(features['_var_len_underscored']), tf.SparseTensor)
    self.assertEqual(features['_var_len_underscored'].get_shape().as_list(),
                     [None, None])
    self.assertEqual(type(features['sparse_1d']), tf.SparseTensor)
    self.assertEqual(type(features['sparse_2d']), tf.SparseTensor)
    if tf.__version__ >= '2':
      self.assertEqual(features['sparse_1d'].get_shape().as_list(), [None, 7])
      self.assertEqual(features['sparse_2d'].get_shape().as_list(),
                       [None, 2, 17])
    else:
      self.assertEqual(features['sparse_1d'].get_shape().as_list(),
                       [None, None])
      self.assertEqual(features['sparse_2d'].get_shape().as_list(),
                       [None, None, None])

  def test_batched_placeholders_from_typespecs(self):
    typespecs = {
        'dense_float':
            tf.TensorSpec(dtype=tf.float32, shape=[None, 2, 3]),
        'dense_string':
            tf.TensorSpec(shape=[None], dtype=tf.string),
        '_sparse_underscored':
            tf.SparseTensorSpec(dtype=tf.string, shape=[None, None, 17]),
        'ragged_string':
            tf.RaggedTensorSpec(
                dtype=tf.string, ragged_rank=1, shape=[None, None]),
        'ragged_multi_dimension':
            tf.RaggedTensorSpec(
                dtype=tf.int64,
                ragged_rank=3,
                shape=[None, None, None, None, 5]),
    }
    with tf.compat.v1.Graph().as_default():
      features = impl_helper.batched_placeholders_from_specs(typespecs)
    self.assertCountEqual(features.keys(), [
        'dense_float',
        'dense_string',
        '_sparse_underscored',
        'ragged_string',
        'ragged_multi_dimension',
    ])
    self.assertEqual(type(features['dense_float']), tf.Tensor)
    self.assertEqual(features['dense_float'].get_shape().as_list(),
                     [None, 2, 3])
    self.assertEqual(features['dense_float'].dtype, tf.float32)

    self.assertEqual(type(features['dense_string']), tf.Tensor)
    self.assertEqual(features['dense_string'].get_shape().as_list(), [None])
    self.assertEqual(features['dense_string'].dtype, tf.string)

    self.assertEqual(type(features['_sparse_underscored']), tf.SparseTensor)
    # TODO(zoyahav): Change last dimension size to 17 once SparseTensors propogate
    # static dense_shape from typespec correctly.
    self.assertEqual(features['_sparse_underscored'].get_shape().as_list(),
                     [None, None, None])
    self.assertEqual(features['_sparse_underscored'].dtype, tf.string)

    self.assertEqual(type(features['ragged_string']), tf.RaggedTensor)
    self.assertEqual(features['ragged_string'].shape.as_list(), [None, None])
    self.assertEqual(features['ragged_string'].ragged_rank, 1)
    self.assertEqual(features['ragged_string'].dtype, tf.string)

    self.assertEqual(type(features['ragged_multi_dimension']), tf.RaggedTensor)
    self.assertEqual(features['ragged_multi_dimension'].shape.as_list(),
                     [None, None, None, None, 5])
    self.assertEqual(features['ragged_multi_dimension'].ragged_rank, 3)
    self.assertEqual(features['ragged_multi_dimension'].dtype, tf.int64)

  def test_batched_placeholders_from_specs_invalid_dtype(self):
    with self.assertRaisesRegexp(ValueError, 'had invalid dtype'):
      impl_helper.batched_placeholders_from_specs(
          {'f': tf.TensorSpec(dtype=tf.int32, shape=[None])})
    with self.assertRaisesRegexp(ValueError, 'had invalid dtype'):
      impl_helper.batched_placeholders_from_specs(
          {'f': tf.io.FixedLenFeature(dtype=tf.int32, shape=[None])})

  def test_batched_placeholders_from_specs_invalid_mixing(self):
    with self.assertRaisesRegexp(TypeError, 'Specs must be all'):
      impl_helper.batched_placeholders_from_specs({
          'f1': tf.TensorSpec(dtype=tf.int64, shape=[None]),
          'f2': tf.io.FixedLenFeature(dtype=tf.int64, shape=[None]),
      })

  @test_case.named_parameters(
      *test_case.cross_named_parameters(_ROUNDTRIP_CASES, [
          dict(testcase_name='eager_tensors', feed_eager_tensors=True),
          dict(testcase_name='session_run_values', feed_eager_tensors=False)
      ]))
  def test_to_instance_dicts(self, feature_spec, instances, feed_dict,
                             feed_eager_tensors):
    if feed_eager_tensors:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')
    schema = schema_utils.schema_from_feature_spec(feature_spec)
    feed_dict_local = copy.copy(feed_dict)
    if feed_eager_tensors:
      for key, value in six.iteritems(feed_dict_local):
        if isinstance(value, tf.compat.v1.SparseTensorValue):
          feed_dict_local[key] = tf.sparse.SparseTensor.from_value(value)
        else:
          feed_dict_local[key] = tf.constant(value)
    result = impl_helper.to_instance_dicts(schema, feed_dict_local)
    np.testing.assert_equal(instances, result)

  @test_case.named_parameters(*_TO_INSTANCE_DICT_ERROR_CASES)
  def test_to_instance_dicts_error(self, feature_spec, feed_dict, error_msg,
                                   error_type=ValueError):
    schema = schema_utils.schema_from_feature_spec(feature_spec)
    with self.assertRaisesRegexp(error_type, error_msg):
      impl_helper.to_instance_dicts(schema, feed_dict)

  @test_case.named_parameters(
      dict(testcase_name='tf_compat_v1', force_tf_compat_v1=True),
      dict(testcase_name='native_tf2', force_tf_compat_v1=False))
  def test_analyze_in_place(self, force_tf_compat_v1):
    if not force_tf_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')

    def preprocessing_fn(inputs):
      return {'x_add_1': inputs['x'] + 1}

    feature_spec = {'x': tf.io.FixedLenFeature([], tf.int64)}
    type_spec = {
        'x': tf.TensorSpec(dtype=tf.int64, shape=[
            None,
        ])
    }
    output_path = os.path.join(self.get_temp_dir(), self._testMethodName)
    impl_helper.analyze_in_place(preprocessing_fn, force_tf_compat_v1,
                                 feature_spec, type_spec, output_path)

    tft_output = TFTransformOutput(output_path)
    expected_value = np.array([2], dtype=np.int64)
    if force_tf_compat_v1:
      with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session(graph=graph).as_default():
          transformed_features = tft_output.transform_raw_features(
              {'x': tf.constant([1], dtype=tf.int64)})
          transformed_value = transformed_features['x_add_1'].eval()
    else:
      transformed_features = tft_output.transform_raw_features(
          {'x': tf.constant([1], dtype=tf.int64)})
      transformed_value = transformed_features['x_add_1'].numpy()
    self.assertEqual(transformed_value, expected_value)

    transformed_feature_spec = tft_output.transformed_feature_spec()
    expected_feature_spec = feature_spec = {
        'x_add_1': tf.io.FixedLenFeature([], tf.int64)
    }
    self.assertEqual(transformed_feature_spec, expected_feature_spec)

  @test_case.named_parameters(
      dict(testcase_name='tf_compat_v1', force_tf_compat_v1=True),
      dict(testcase_name='native_tf2', force_tf_compat_v1=False))
  def test_analyze_in_place_with_analyzers_raises_error(self,
                                                        force_tf_compat_v1):
    if not force_tf_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')

    def preprocessing_fn(inputs):
      return {'x_add_1': analyzers.mean(inputs['x'])}

    feature_spec = {'x': tf.io.FixedLenFeature([], tf.int64)}
    type_spec = {
        'x': tf.TensorSpec(dtype=tf.int64, shape=[
            None,
        ])
    }
    output_path = os.path.join(self.get_temp_dir(), self._testMethodName)
    with self.assertRaisesRegexp(RuntimeError, 'analyzers found when tracing'):
      impl_helper.analyze_in_place(preprocessing_fn, force_tf_compat_v1,
                                   feature_spec, type_spec, output_path)


def _subtract_ten_with_tf_while(x):
  """Subtracts 10 from x using control flow ops.

  This function is equivalent to "x - 10" but uses a tf.while_loop, in order
  to test the use of functions that involve control flow ops.

  Args:
    x: A tensor of integral type.

  Returns:
    A tensor representing x - 10.
  """
  def stop_condition(counter, x_minus_counter):
    del x_minus_counter  # unused
    return tf.less(counter, 10)
  def iteration(counter, x_minus_counter):
    return tf.add(counter, 1), tf.add(x_minus_counter, -1)
  initial_values = [tf.constant(0), x]
  return tf.while_loop(
      cond=stop_condition, body=iteration, loop_vars=initial_values)[1]


if __name__ == '__main__':
  test_case.main()
