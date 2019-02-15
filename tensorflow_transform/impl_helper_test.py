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

# GOOGLE-INITIALIZATION

import numpy as np
import tensorflow as tf
from tensorflow_transform import impl_helper
from tensorflow_transform import test_case
from tensorflow_transform.tf_metadata import dataset_schema

_FEATURE_SPEC = {
    'a': tf.FixedLenFeature([], tf.int64),
    'b': tf.FixedLenFeature([], tf.float32),
    'c': tf.FixedLenFeature([1], tf.float32),
    'd': tf.FixedLenFeature([2, 2], tf.float32),
    'e': tf.VarLenFeature(tf.string),
    'f': tf.SparseFeature('idx', 'val', tf.float32, 10),
}
_FEED_DICT = {
    'a': np.array([100, 100]),
    'b': np.array([1.0, 2.0]),
    'c': np.array([[2.0], [4.0]]),
    'd': np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    'e': tf.SparseTensorValue(
        indices=np.array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
        values=np.array(['doe', 'a', 'deer', 'a', 'female', 'deer']),
        dense_shape=(2, 3)),
    'f': tf.SparseTensorValue(
        indices=np.array([(0, 2), (0, 4), (0, 8)]),
        values=np.array([10.0, 20.0, 30.0]),
        dense_shape=(2, 10)),
}

_ROUNDTRIP_CASES = [
    dict(testcase_name='multiple_features',
         feature_spec=_FEATURE_SPEC,
         instances=[{
             'a': 100,
             'b': 1.0,
             'c': [2.0],
             'd': [[1.0, 2.0], [3.0, 4.0]],
             'e': ['doe', 'a', 'deer'],
             'f': ([2, 4, 8], [10.0, 20.0, 30.0]),
         }, {
             'a': 100,
             'b': 2.0,
             'c': [4.0],
             'd': [[5.0, 6.0], [7.0, 8.0]],
             'e': ['a', 'female', 'deer'],
             'f': ([], []),
         }],
         feed_dict=_FEED_DICT),
    dict(testcase_name='multiple_features_ndarrays',
         feature_spec=_FEATURE_SPEC,
         instances=[{
             'a': np.int64(100),
             'b': np.array(1.0, np.float32),
             'c': np.array([2.0], np.float32),
             'd': np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
             'e': ['doe', 'a', 'deer'],
             'f': (np.array([2, 4, 8]), np.array([10.0, 20.0, 30.0])),
         }, {
             'a': np.int64(100),
             'b': np.array(2.0, np.float32),
             'c': np.array([4.0], np.float32),
             'd': np.array([[5.0, 6.0], [7.0, 8.0]], np.float32),
             'e': ['a', 'female', 'deer'],
             'f': (np.array([], np.int32), np.array([], np.float32))
         }],
         feed_dict=_FEED_DICT),
    dict(testcase_name='empty_var_len_feature',
         feature_spec={'varlen': tf.VarLenFeature(tf.string)},
         instances=[{'varlen': []}],
         feed_dict={
             'varlen': tf.SparseTensorValue(
                 indices=np.empty([0, 2]),
                 values=np.array([]),
                 dense_shape=[1, 0])
         }),
    # Mainly to test the empty-ndarray optimization though this is also
    # exercised by empty_var_len_feature
    dict(testcase_name='some_empty_int_var_len_feature',
         feature_spec={'varlen': tf.VarLenFeature(tf.int64)},
         instances=[
             {'varlen': [0]},
             {'varlen': []},
             {'varlen': [1]},
             {'varlen': []}
         ],
         feed_dict={
             'varlen': tf.SparseTensorValue(
                 indices=np.array([(0, 0), (2, 0)]),
                 values=np.array([0, 1], np.int64),
                 dense_shape=(4, 1)),
         }),
    dict(testcase_name='some_empty_float_var_len_feature',
         feature_spec={'varlen': tf.VarLenFeature(tf.float32)},
         instances=[
             {'varlen': [0.5]},
             {'varlen': []},
             {'varlen': [1.5]},
             {'varlen': []}
         ],
         feed_dict={
             'varlen': tf.SparseTensorValue(
                 indices=np.array([(0, 0), (2, 0)]),
                 values=np.array([0.5, 1.5], np.float32),
                 dense_shape=(4, 1)),
         }),
    dict(testcase_name='some_empty_string_var_len_feature',
         feature_spec={'varlen': tf.VarLenFeature(tf.string)},
         instances=[
             {'varlen': ['a']},
             {'varlen': []},
             {'varlen': ['b']},
             {'varlen': []}
         ],
         feed_dict={
             'varlen': tf.SparseTensorValue(
                 indices=np.array([(0, 0), (2, 0)]),
                 values=np.array(['a', 'b'], np.object),
                 dense_shape=(4, 1)),
         }),
    dict(testcase_name='empty_sparse_feature',
         feature_spec={
             'sparse': tf.SparseFeature('idx', 'val', tf.float32, 10)
         },
         instances=[{'sparse': ([], [])}],
         feed_dict={
             'sparse': tf.SparseTensorValue(
                 indices=np.empty([0, 2]),
                 values=np.array([]),
                 dense_shape=[1, 10])
         }),
]

# Non-canonical inputs that will not be the output of to_instance_dicts but
# are valid inputs to make_feed_dict.
_MAKE_FEED_DICT_CASES = [
    dict(testcase_name='none_feature',
         feature_spec={
             'varlen_feature': tf.VarLenFeature(tf.int64),
         },
         instances=[
             {'varlen_feature': []},
             {'varlen_feature': None},
             {'varlen_feature': [1, 2]}
         ],
         feed_dict={
             'varlen_feature': tf.SparseTensorValue(
                 indices=np.array([(2, 0), (2, 1)]),
                 values=np.array([1, 2]),
                 dense_shape=[3, 2])
         }),
]

_MAKE_FEED_LIST_ERROR_CASES = [
    dict(testcase_name='missing_feature',
         feature_spec={
             'a': tf.FixedLenFeature([1], tf.int64),
             'b': tf.FixedLenFeature([1], tf.int64),
         },
         instances=[{'a': 100}],
         error_msg='b',
         error_type=KeyError),
    dict(testcase_name='sparse_feature_index_negative',
         feature_spec={
             'a': tf.SparseFeature('idx', 'val', tf.float32, 10)
         },
         instances=[{'a': ([-1, 2], [1.0, 2.0])}],
         error_msg='has index .* out of range'),
    dict(testcase_name='sparse_feature_index_too_high',
         feature_spec={
             'a': tf.SparseFeature('idx', 'val', tf.float32, 10)
         },
         instances=[{'a': ([11, 2], [1.0, 2.0])}],
         error_msg='has index .* out of range'),
    dict(testcase_name='sparse_feature_indices_and_values_different_lengths',
         feature_spec={
             'a': tf.SparseFeature('idx', 'val', tf.float32, 10)
         },
         instances=[{'a': ([1, 2], [1])}],
         error_msg='indices and values of different lengths'),
    dict(testcase_name='sparse_feature_not_a_pair',
         feature_spec={
             'a': tf.SparseFeature('idx', 'val', tf.float32, 10)
         },
         instances=[{'a': ([1], [2], [3])}],
         error_msg='too many values to unpack'),
]

_TO_INSTANCE_DICT_ERROR_CASES = [
    dict(testcase_name='var_len_with_non_consecutive_indices',
         feature_spec={
             'a': tf.VarLenFeature(tf.float32)
         },
         feed_dict={
             'a': tf.SparseTensorValue(
                 indices=np.array([(0, 2), (0, 4), (0, 8)]),
                 values=np.array([10.0, 20.0, 30.0]),
                 dense_shape=(1, 20))
         },
         error_msg='cannot be decoded by ListColumnRepresentation'),
    dict(testcase_name='var_len_with_rank_not_2',
         feature_spec={
             'a': tf.VarLenFeature(tf.float32)
         },
         feed_dict={
             'a': tf.SparseTensorValue(
                 indices=np.array([(0, 0, 1), (0, 0, 2), (0, 0, 3)]),
                 values=np.array([10.0, 20.0, 30.0]),
                 dense_shape=(1, 10, 10))
         },
         error_msg='cannot be decoded by ListColumnRepresentation'),
    dict(testcase_name='var_len_with_out_of_order_indices',
         feature_spec={
             'a': tf.VarLenFeature(tf.float32)
         },
         feed_dict={
             'a': tf.SparseTensorValue(
                 indices=np.array([(0, 2), (2, 4), (1, 8)]),
                 values=np.array([10.0, 20.0, 30.0]),
                 dense_shape=(3, 20))
         },
         error_msg='Encountered out-of-order sparse index'),
    dict(testcase_name='var_len_with_different_batch_dim_sizes',
         feature_spec={
             'a': tf.VarLenFeature(tf.float32),
             'b': tf.VarLenFeature(tf.float32),
         },
         feed_dict={
             'a': tf.SparseTensorValue(
                 indices=np.array([(0, 0)]),
                 values=np.array([10.0]),
                 dense_shape=(1, 20)),
             'b': tf.SparseTensorValue(
                 indices=np.array([(0, 0)]),
                 values=np.array([10.0]),
                 dense_shape=(2, 20)),
         },
         error_msg=(r'Inconsistent batch sizes: "\w" had batch dimension \d, '
                    r'"\w" had batch dimension \d')),
    dict(testcase_name='fixed_len_with_different_batch_dim_sizes',
         feature_spec={
             'a': tf.FixedLenFeature([], tf.float32),
             'b': tf.FixedLenFeature([], tf.float32),
         },
         feed_dict={
             'a': np.array([1]),
             'b': np.array([1, 2])
         },
         error_msg=(r'Inconsistent batch sizes: "\w" had batch dimension \d, '
                    r'"\w" had batch dimension \d')),
]


class ImplHelperTest(test_case.TransformTestCase):

  def test_feature_spec_as_batched_placeholders(self):
    feature_spec = {
        'fixed_len_float': tf.FixedLenFeature([2, 3], tf.float32),
        'fixed_len_string': tf.FixedLenFeature([], tf.string),
        'var_len_int': tf.VarLenFeature(tf.int64)
    }
    with tf.Graph().as_default():
      features = impl_helper.feature_spec_as_batched_placeholders(feature_spec)
    self.assertCountEqual(
        features.keys(), ['fixed_len_float', 'fixed_len_string', 'var_len_int'])
    self.assertEqual(type(features['fixed_len_float']), tf.Tensor)
    self.assertEqual(features['fixed_len_float'].get_shape().as_list(),
                     [None, 2, 3])
    self.assertEqual(type(features['fixed_len_string']), tf.Tensor)
    self.assertEqual(features['fixed_len_string'].get_shape().as_list(),
                     [None])
    self.assertEqual(type(features['var_len_int']), tf.SparseTensor)
    self.assertEqual(features['var_len_int'].get_shape().as_list(),
                     [None, None])

  @test_case.named_parameters(*(_ROUNDTRIP_CASES + _MAKE_FEED_DICT_CASES))
  def test_make_feed_list(self, feature_spec, instances, feed_dict):
    schema = dataset_schema.from_feature_spec(feature_spec)
    feature_names = list(feature_spec.keys())
    expected_feed_list = [feed_dict[key] for key in feature_names]
    np.testing.assert_equal(
        impl_helper.make_feed_list(feature_names, schema, instances),
        expected_feed_list)

  @test_case.named_parameters(*_MAKE_FEED_LIST_ERROR_CASES)
  def test_make_feed_list_error(self,
                                feature_spec,
                                instances,
                                error_msg,
                                error_type=ValueError):
    tensors = tf.parse_example(tf.placeholder(tf.string, [None]), feature_spec)
    schema = dataset_schema.from_feature_spec(feature_spec)
    with self.assertRaisesRegexp(error_type, error_msg):
      impl_helper.make_feed_list(tensors, schema, instances)

  @test_case.named_parameters(*_ROUNDTRIP_CASES)
  def test_to_instance_dicts(self, feature_spec, instances, feed_dict):
    schema = dataset_schema.from_feature_spec(feature_spec)
    np.testing.assert_equal(
        instances,
        impl_helper.to_instance_dicts(schema, feed_dict))

  @test_case.named_parameters(*_TO_INSTANCE_DICT_ERROR_CASES)
  def test_to_instance_dicts_error(self, feature_spec, feed_dict, error_msg,
                                   error_type=ValueError):
    schema = dataset_schema.from_feature_spec(feature_spec)
    with self.assertRaisesRegexp(error_type, error_msg):
      impl_helper.to_instance_dicts(schema, feed_dict)

  def test_copy_tensors_produces_different_tensors(self):
    tensors = {
        'dense': tf.placeholder(tf.int64, (None,), name='my_dense_input'),
        'sparse': tf.sparse_placeholder(tf.int64, name='my_sparse_input')
    }
    copied_tensors = impl_helper.copy_tensors(tensors)

    self.assertNotEqual(tensors['dense'],
                        copied_tensors['dense'])
    self.assertNotEqual(tensors['sparse'].indices,
                        copied_tensors['sparse'].indices)
    self.assertNotEqual(tensors['sparse'].values,
                        copied_tensors['sparse'].values)
    self.assertNotEqual(tensors['sparse'].dense_shape,
                        copied_tensors['sparse'].dense_shape)

  def test_copy_tensors_produces_equivalent_tensors(self):
    tensors = {
        'dense': tf.placeholder(tf.int64, (None,), name='my_dense_input'),
        'sparse': tf.sparse_placeholder(tf.int64, name='my_sparse_input')
    }
    copied_tensors = impl_helper.copy_tensors(tensors)

    with tf.Session() as session:
      dense_value = [1, 2]
      sparse_value = tf.SparseTensorValue(
          indices=[[0, 0], [0, 2], [1, 1]],
          values=[3, 4, 5],
          dense_shape=[2, 3])
      sample_tensors = session.run(copied_tensors, feed_dict={
          tensors['dense']: dense_value,
          tensors['sparse']: sparse_value
      })
      self.assertAllEqual(sample_tensors['dense'], dense_value)
      self.assertAllEqual(sample_tensors['sparse'].indices,
                          sparse_value.indices)
      self.assertAllEqual(sample_tensors['sparse'].values,
                          sparse_value.values)
      self.assertAllEqual(sample_tensors['sparse'].dense_shape,
                          sparse_value.dense_shape)


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
  return tf.while_loop(stop_condition, iteration, initial_values)[1]


if __name__ == '__main__':
  test_case.main()
