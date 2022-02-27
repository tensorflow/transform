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
"""Tests for tensorflow_transform.tf_utils."""

import os

import numpy as np
import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import annotators
from tensorflow_transform import tf_utils
from tensorflow_transform import test_case

import unittest
from tensorflow.python.framework import composite_tensor  # pylint: disable=g-direct-tensorflow-import

_CONSTRUCT_TABLE_PARAMETERS = [
    dict(testcase_name='_string', asset_path_input_fn=lambda x: x),
    dict(testcase_name='_string_tensor', asset_path_input_fn=tf.constant),
]


def _construct_table(asset_file_path,
                     key_dtype=tf.string,
                     key_index=0,
                     value_dtype=tf.int64,
                     value_index=1,
                     default_value=-1):
  initializer = tf.lookup.TextFileInitializer(
      asset_file_path,
      key_dtype=key_dtype,
      key_index=key_index,
      value_dtype=value_dtype,
      value_index=value_index)
  return tf.lookup.StaticHashTable(initializer, default_value=default_value)


def _value_to_tensor(value):
  if isinstance(value, tf.compat.v1.SparseTensorValue):
    return tf.compat.v1.convert_to_tensor_or_sparse_tensor(value)
  elif isinstance(value, tf.compat.v1.ragged.RaggedTensorValue):
    return tf.ragged.constant(value.to_list())
  else:
    return tf.constant(value)


class _SparseTensorSpec:

  def __init__(self, shape, dtype):
    self._shape = shape
    self._dtype = dtype

if not hasattr(tf, 'SparseTensorSpec'):
  tf.SparseTensorSpec = _SparseTensorSpec


class TFUtilsTest(test_case.TransformTestCase):

  def _assertCompositeRefEqual(self, left, right):
    """Asserts that a two `tf_util._CompositeTensorRef`s are equal."""
    self.assertEqual(left.type_spec, right.type_spec)
    self.assertAllEqual(left.list_of_refs, right.list_of_refs)

  def test_copy_tensors_produces_different_tensors(self):
    with tf.compat.v1.Graph().as_default():
      tensors = {
          'dense':
              tf.compat.v1.placeholder(
                  tf.int64, (None,), name='my_dense_input'),
          'sparse':
              tf.compat.v1.sparse_placeholder(tf.int64, name='my_sparse_input'),
          'ragged':
              tf.compat.v1.ragged.placeholder(
                  tf.int64, ragged_rank=2, name='my_ragged_input')
      }
      copied_tensors = tf_utils.copy_tensors(tensors)

      self.assertNotEqual(tensors['dense'], copied_tensors['dense'])
      self.assertNotEqual(tensors['sparse'].indices,
                          copied_tensors['sparse'].indices)
      self.assertNotEqual(tensors['sparse'].values,
                          copied_tensors['sparse'].values)
      self.assertNotEqual(tensors['sparse'].dense_shape,
                          copied_tensors['sparse'].dense_shape)
      self.assertNotEqual(tensors['ragged'].values,
                          copied_tensors['ragged'].values)
      self.assertNotEqual(tensors['ragged'].row_splits,
                          copied_tensors['ragged'].row_splits)

  def test_copy_tensors_produces_equivalent_tensors(self):
    with tf.compat.v1.Graph().as_default():
      tensors = {
          'dense':
              tf.compat.v1.placeholder(
                  tf.int64, (None,), name='my_dense_input'),
          'sparse':
              tf.compat.v1.sparse_placeholder(tf.int64, name='my_sparse_input'),
          'ragged':
              tf.compat.v1.ragged.placeholder(
                  tf.int64, ragged_rank=1, name='my_ragged_input')
      }
      copied_tensors = tf_utils.copy_tensors(tensors)

      with tf.compat.v1.Session() as session:
        dense_value = [1, 2]
        sparse_value = tf.compat.v1.SparseTensorValue(
            indices=[[0, 0], [0, 2], [1, 1]],
            values=[3, 4, 5],
            dense_shape=[2, 3])
        ragged_value = tf.compat.v1.ragged.RaggedTensorValue(
            values=np.array([3, 4, 5], dtype=np.int64),
            row_splits=np.array([0, 2, 3], dtype=np.int64))
        sample_tensors = session.run(
            copied_tensors,
            feed_dict={
                tensors['dense']: dense_value,
                tensors['sparse']: sparse_value,
                tensors['ragged']: ragged_value
            })
        self.assertAllEqual(sample_tensors['dense'], dense_value)
        self.assertAllEqual(sample_tensors['sparse'].indices,
                            sparse_value.indices)
        self.assertAllEqual(sample_tensors['sparse'].values,
                            sparse_value.values)
        self.assertAllEqual(sample_tensors['sparse'].dense_shape,
                            sparse_value.dense_shape)
        self.assertAllEqual(sample_tensors['ragged'].values,
                            ragged_value.values)
        self.assertAllEqual(sample_tensors['ragged'].row_splits,
                            ragged_value.row_splits)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='2d',
              tensor=tf.compat.v1.ragged.RaggedTensorValue(
                  values=np.array([1.2, 1., 1.2, 1.]),
                  row_splits=np.array([0, 2, 4])),
              rowids=[0, 0, 1, 1],
              tensor_spec=tf.RaggedTensorSpec([None, None], tf.float32)),
          dict(
              testcase_name='3d',
              tensor=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=np.array([1.2, 1., 1.2, 1.]),
                      row_splits=np.array([0, 3, 4])),
                  row_splits=np.array([0, 1, 1, 2])),
              rowids=[0, 0, 0, 2],
              tensor_spec=tf.RaggedTensorSpec([None, None, None], tf.float32)),
      ]))
  def test_get_ragged_batch_value_rowids(self, tensor, rowids, tensor_spec,
                                         function_handler):

    @function_handler(input_signature=[tensor_spec])
    def get_ragged_batch_value_rowids(tensor):
      return tf_utils._get_ragged_batch_value_rowids(tensor)

    self.assertAllEqual(get_ragged_batch_value_rowids(tensor), rowids)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='rank1',
              x=['a', 'b', 'a'],
              x_spec=tf.TensorSpec(None, tf.string),
              weights=[1, 1, 2],
              filter_regex=None,
              expected_unique_x=[b'a', b'b'],
              expected_summed_weights_per_x=[3, 1]),
          dict(
              testcase_name='rank2',
              x=[['a', 'b\n', 'a'], ['b\n', 'a', 'b\n']],
              x_spec=tf.TensorSpec(None, tf.string),
              weights=[[1, 2, 1], [1, 2, 2]],
              filter_regex=None,
              expected_unique_x=[b'a', b'b\n'],
              expected_summed_weights_per_x=[4, 5]),
          dict(
              testcase_name='rank3',
              x=[[['a', 'b', 'a'], ['b', 'a', 'b']],
                 [['a', 'b', 'a'], ['b', 'a', 'b']]],
              x_spec=tf.TensorSpec(None, tf.string),
              weights=[[[1, 1, 2], [1, 2, 1]], [[1, 2, 1], [1, 2, 1]]],
              filter_regex=None,
              expected_unique_x=[b'a', b'b'],
              expected_summed_weights_per_x=[9, 7]),
          dict(
              testcase_name='sparse',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 1], [2, 1]],
                  values=['a', 'a', 'b'],
                  dense_shape=[4, 2]),
              x_spec=tf.SparseTensorSpec([4, 2], tf.string),
              weights=[2, 3, 4],
              filter_regex=None,
              expected_unique_x=[b'a', b'b'],
              expected_summed_weights_per_x=[5, 4]),
          dict(
              testcase_name='ragged',
              x=tf.compat.v1.ragged.RaggedTensorValue(  # pylint: disable=g-long-lambda
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=np.array(['a', 'b', 'b', 'a']),
                      row_splits=np.array([0, 2, 4])),
                  row_splits=np.array([0, 2])),
              x_spec=tf.RaggedTensorSpec([None, None, None], tf.string),
              weights=[2, 3, 4, 6],
              filter_regex=None,
              expected_unique_x=[b'a', b'b'],
              expected_summed_weights_per_x=[8, 7]),
          dict(
              testcase_name='regex_filtering',
              x=[['a\n', '', '\n\r'], ['\r', 'a', 'b']],
              x_spec=tf.TensorSpec(None, tf.string),
              weights=[[1, 2, 1], [1, 2, 2]],
              filter_regex=analyzers._EMPTY_STRING_OR_NEWLINE_CHARS_REGEX,
              expected_unique_x=[b'a', b'b'],
              expected_summed_weights_per_x=[2, 2]),
          dict(
              testcase_name='regex_filtering_invalid_utf8',
              x=[[b'\xe1\n', b'\xa9', b'\n\xb8\r'],
                 [b'\xe8\r', b'\xc6', b'\n\xb3']],
              x_spec=tf.TensorSpec(None, tf.string),
              weights=[[1, 3, 1], [1, 4, 2]],
              filter_regex=analyzers._EMPTY_STRING_OR_NEWLINE_CHARS_REGEX,
              expected_unique_x=[b'\xa9', b'\xc6'],
              expected_summed_weights_per_x=[3, 4]),
      ]))
  def test_reduce_batch_weighted_counts(self, x, x_spec, weights, filter_regex,
                                        expected_unique_x,
                                        expected_summed_weights_per_x,
                                        function_handler):
    input_signature = [x_spec, tf.TensorSpec(None, tf.float32)]
    @function_handler(input_signature=input_signature)
    def _reduce_batch_weighted_counts(x, weights):
      (unique_x, summed_weights_per_x, summed_positive_per_x_and_y,
       counts_per_x) = tf_utils.reduce_batch_weighted_counts(
           x, weights, filter_regex=filter_regex)
      self.assertIsNone(summed_positive_per_x_and_y)
      self.assertIsNone(counts_per_x)
      return unique_x, summed_weights_per_x

    unique_x, summed_weights_per_x = _reduce_batch_weighted_counts(x, weights)

    self.assertAllEqual(unique_x,
                        expected_unique_x)
    self.assertAllEqual(summed_weights_per_x,
                        expected_summed_weights_per_x)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='rank1',
              x=['a', 'b', 'a'],
              filter_regex=None,
              expected_result=[b'a', b'b', b'a'],
          ),
          dict(
              testcase_name='rank2',
              x=[['a', 'b\r', 'a'], ['b\r', 'a', 'b\r']],
              filter_regex=None,
              expected_result=[b'a', b'b\r', b'a', b'b\r', b'a', b'b\r'],
          ),
          dict(
              testcase_name='rank3',
              x=[[['a', 'b', 'a'], ['b', 'a', 'b']],
                 [['a', 'b', 'a'], ['b', 'a', 'b']]],
              filter_regex=None,
              expected_result=[
                  b'a', b'b', b'a', b'b', b'a', b'b', b'a', b'b', b'a', b'b',
                  b'a', b'b'
              ],
          ),
          dict(
              testcase_name='regex_filtering_empty_result',
              x=['a\n\r', 'b\n', 'a\r', '', 'a\rsd', ' \r', '\nas'],
              filter_regex=analyzers._EMPTY_STRING_OR_NEWLINE_CHARS_REGEX,
              expected_result=[],
          ),
      ]))
  def test_reduce_batch_weighted_counts_weights_none(self, x, filter_regex,
                                                     expected_result,
                                                     function_handler):
    input_signature = [tf.TensorSpec(None, tf.string)]

    @function_handler(input_signature=input_signature)
    def _reduce_batch_weighted_counts(x):
      (unique_x, summed_weights_per_x, summed_positive_per_x_and_y,
       counts_per_x) = tf_utils.reduce_batch_weighted_counts(
           x, force=False, filter_regex=filter_regex)
      self.assertIsNone(summed_weights_per_x)
      self.assertIsNone(summed_positive_per_x_and_y)
      self.assertIsNone(counts_per_x)
      return unique_x

    unique_x = _reduce_batch_weighted_counts(x)
    self.assertAllEqual(unique_x, expected_result)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='rank1',
              x=['a', 'b', 'a'],
              filter_regex=None,
              expected_result=([b'a', b'b'], [2, 1]),
          ),
          dict(
              testcase_name='rank3',
              x=[[['a', 'b', 'a'], ['b', 'a', 'b']],
                 [['a', 'b', 'a'], ['b', 'a', 'b']]],
              filter_regex=None,
              expected_result=([b'a', b'b'], [6, 6]),
          ),
          dict(
              testcase_name='regex_filtering',
              x=['a\n\r', 'b\n', 'a\r', '', 'asd', ' ', '\nas'],
              filter_regex=analyzers._EMPTY_STRING_OR_NEWLINE_CHARS_REGEX,
              expected_result=([b'asd', b' '], [1, 1]),
          ),
          dict(
              testcase_name='regex_filtering_empty_result',
              x=['a\n\r', 'b\n', 'a\r', '', 'a\rsd', ' \r', '\nas'],
              filter_regex=analyzers._EMPTY_STRING_OR_NEWLINE_CHARS_REGEX,
              expected_result=([], []),
          ),
      ]))
  def test_reduce_batch_weighted_counts_weights_none_force(
      self, x, filter_regex, expected_result, function_handler):
    input_signature = [tf.TensorSpec(None, tf.string)]

    @function_handler(input_signature=input_signature)
    def _reduce_batch_weighted_counts(x):
      (unique_x, summed_weights_per_x, summed_positive_per_x_and_y,
       counts_per_x) = tf_utils.reduce_batch_weighted_counts(
           x, force=True, filter_regex=filter_regex)
      self.assertIsNone(summed_weights_per_x)
      self.assertIsNone(summed_positive_per_x_and_y)
      return unique_x, counts_per_x

    expected_unique_x, expected_counts_per_x = expected_result
    unique_x, counts_per_x = _reduce_batch_weighted_counts(x)
    self.assertAllEqual(unique_x, expected_unique_x)
    self.assertAllEqual(counts_per_x, expected_counts_per_x)

  @test_case.named_parameters([
      dict(testcase_name='constant', get_value_fn=lambda: tf.constant([1.618])),
      dict(testcase_name='op', get_value_fn=lambda: tf.identity),
      dict(testcase_name='int', get_value_fn=lambda: 4),
      dict(testcase_name='object', get_value_fn=object),
      dict(
          testcase_name='sparse',
          get_value_fn=lambda: tf.SparseTensor(  # pylint: disable=g-long-lambda
              indices=[[0, 0], [2, 1]],
              values=['a', 'b'],
              dense_shape=[4, 2])),
      dict(
          testcase_name='ragged',
          get_value_fn=lambda: tf.RaggedTensor.from_row_splits(  # pylint: disable=g-long-lambda
              values=['a', 'b'],
              row_splits=[0, 1, 2])),
      dict(
          testcase_name='ragged_multi_dimension',
          get_value_fn=lambda: tf.RaggedTensor.from_row_splits(  # pylint: disable=g-long-lambda
              values=tf.RaggedTensor.from_row_splits(
                  values=[[0, 1], [2, 3]], row_splits=[0, 1, 2]),
              row_splits=[0, 2])),
  ])
  def test_hashable_tensor_or_op(self, get_value_fn):
    with tf.compat.v1.Graph().as_default():
      input_value = get_value_fn()
      input_ref = tf_utils.hashable_tensor_or_op(input_value)
      input_dict = {input_ref: input_value}
      input_deref = tf_utils.deref_tensor_or_op(input_ref)
      if isinstance(input_value, composite_tensor.CompositeTensor):
        self._assertCompositeRefEqual(
            input_ref, tf_utils.hashable_tensor_or_op(input_deref))
      else:
        self.assertAllEqual(input_ref,
                            tf_utils.hashable_tensor_or_op(input_deref))

      if isinstance(input_value, tf.SparseTensor):
        input_deref = input_deref.values
        input_dict[input_ref] = input_dict[input_ref].values
        input_value = input_value.values

      self.assertAllEqual(input_value, input_deref)
      self.assertAllEqual(input_value, input_dict[input_ref])

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='rank1_with_weights_and_binary_y',
              x=['a', 'b', 'a'],
              weights=[1, 1, 2],
              y=[0, 1, 1],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [3, 1, 4],
                  [[1, 2], [0, 1], [1, 3]], [2, 1, 3]),
              filter_regex=None,
          ),
          dict(
              testcase_name='rank1_with_weights_and_multi_class_y',
              x=['a', 'b\n', 'a', 'a'],
              weights=[1, 1, 2, 2],
              y=[0, 2, 1, 1],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b\n', b'global_y_count_sentinel'], [5, 1, 6],
                  [[1, 4, 0], [0, 0, 1], [1, 4, 1]], [3, 1, 4]),
              filter_regex=None,
          ),
          dict(
              testcase_name='rank1_with_weights_and_missing_y_values',
              x=['a', 'b', 'a', 'a'],
              weights=[1, 1, 2, 2],
              y=[3, 5, 6, 6],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [5, 1, 6],
                  [[0, 0, 0, 1, 0, 0, 4], [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 1, 4]], [3, 1, 4]),
              filter_regex=None,
          ),
          dict(
              testcase_name='rank2_with_weights_and_binary_y',
              x=[['a', 'b', 'a'], ['b', 'a', 'b']],
              weights=[[1, 2, 1], [1, 2, 2]],
              y=[[1, 0, 1], [1, 0, 0]],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [4, 5, 9],
                  [[2, 2], [4, 1], [6, 3]], [3, 3, 6]),
              filter_regex=None,
          ),
          dict(
              testcase_name='rank3_with_weights_and_binary_y',
              x=[[['a', 'b', 'a'], ['b', 'a', 'b']],
                 [['a', 'b', 'a'], ['b', 'a', 'b']]],
              weights=[[[1, 1, 2], [1, 2, 1]], [[1, 2, 1], [1, 2, 1]]],
              y=[[[1, 1, 0], [1, 0, 1]], [[1, 0, 1], [1, 0, 1]]],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [9, 7, 16],
                  [[6, 3], [2, 5], [8, 8]], [6, 6, 12]),
              filter_regex=None,
          ),
          dict(
              testcase_name='rank1_with_weights_multi_class_y_and_filtering',
              x=['\na\r', '', '\na\r', 'a', ''],
              weights=[1, 1, 2, 2, 3],
              y=[0, 2, 1, 1, 2],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'global_y_count_sentinel'], [2, 9],
                  [[0, 2, 0], [1, 4, 4]], [1, 5]),
              filter_regex=analyzers._EMPTY_STRING_OR_NEWLINE_CHARS_REGEX,
          ),
          dict(
              testcase_name='rank1_with_weights_filtering_empty_result',
              x=['\na\r', '', '\na\r', '\ra', ''],
              weights=[1, 1, 2, 2, 3],
              y=[0, 2, 1, 1, 2],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'global_y_count_sentinel'], [9], [[1, 4, 4]], [5]),
              filter_regex=analyzers._EMPTY_STRING_OR_NEWLINE_CHARS_REGEX,
          ),
      ]))
  def test_reduce_batch_coocurrences(self, x, weights, y, expected_result,
                                     filter_regex, function_handler):
    input_signature = [tf.TensorSpec(None, tf.string),
                       tf.TensorSpec(None, tf.int64),
                       tf.TensorSpec(None, tf.int64)]

    @function_handler(input_signature=input_signature)
    def _reduce_batch_weighted_cooccurrences(x, y, weights):
      return tf_utils.reduce_batch_weighted_cooccurrences(
          x, y, weights, filter_regex=filter_regex)

    result = _reduce_batch_weighted_cooccurrences(x, y, weights)

    self.assertAllEqual(result.unique_x,
                        expected_result.unique_x)
    self.assertAllEqual(result.summed_weights_per_x,
                        expected_result.summed_weights_per_x)
    self.assertAllEqual(result.summed_positive_per_x_and_y,
                        expected_result.summed_positive_per_x_and_y)
    self.assertAllEqual(result.counts_per_x,
                        expected_result.counts_per_x)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='rank1_with_binary_y',
              x=['a', 'b', 'a'],
              y=[0, 1, 1],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [2, 1, 3],
                  [[1, 1], [0, 1], [1, 2]], [2, 1, 3]),
              input_signature=[
                  tf.TensorSpec(None, tf.string),
                  tf.TensorSpec(None, tf.int64)
              ],
              filter_regex=None),
          dict(
              testcase_name='rank1_with_multi_class_y',
              x=['yes', 'no', 'yes', 'may\rbe', 'yes'],
              y=[1, 1, 0, 2, 3],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'yes', b'no', b'may\rbe', b'global_y_count_sentinel'],
                  [3, 1, 1, 5],
                  [[1, 1, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 2, 1, 1]],
                  [3, 1, 1, 5]),
              input_signature=[
                  tf.TensorSpec(None, tf.string),
                  tf.TensorSpec(None, tf.int64)
              ],
              filter_regex=None),
          dict(
              testcase_name='rank2_with_binary_y',
              x=[['a', 'b', 'a'], ['b', 'a', 'b']],
              y=[[1, 0, 1], [1, 0, 0]],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [3, 3, 6],
                  [[1, 2], [2, 1], [3, 3]], [3, 3, 6]),
              input_signature=[
                  tf.TensorSpec(None, tf.string),
                  tf.TensorSpec(None, tf.int64)
              ],
              filter_regex=None),
          dict(
              testcase_name='rank2_with_missing_y_values',
              x=[['a', 'b', 'a'], ['b', 'a', 'b']],
              y=[[2, 0, 2], [2, 0, 0]],
              # The label 1 isn't in the batch but it will have a position (with
              # weights of 0) in the resulting array.
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [3, 3, 6],
                  [[1, 0, 2], [2, 0, 1], [3, 0, 3]], [3, 3, 6]),
              input_signature=[
                  tf.TensorSpec(None, tf.string),
                  tf.TensorSpec(None, tf.int64)
              ],
              filter_regex=None),
          dict(
              testcase_name='rank2_with_multi_class_y',
              x=[['a', 'b', 'a'], ['b', 'a', 'b']],
              y=[[1, 0, 1], [1, 0, 2]],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [3, 3, 6],
                  [[1, 2, 0], [1, 1, 1], [2, 3, 1]], [3, 3, 6]),
              input_signature=[
                  tf.TensorSpec(None, tf.string),
                  tf.TensorSpec(None, tf.int64)
              ],
              filter_regex=None),
          dict(
              testcase_name='rank3_with_binary_y',
              x=[[['a', 'b', 'a'], ['b', 'a', 'b']],
                 [['a', 'b', 'a'], ['b', 'a', 'b']]],
              y=[[[1, 1, 0], [1, 0, 1]], [[1, 0, 1], [1, 0, 1]]],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [6, 6, 12],
                  [[3, 3], [1, 5], [4, 8]], [6, 6, 12]),
              input_signature=[
                  tf.TensorSpec(None, tf.string),
                  tf.TensorSpec(None, tf.int64)
              ],
              filter_regex=None),
          dict(
              testcase_name='sparse',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [2, 1]],
                  values=['a', 'b'],
                  dense_shape=[4, 2]),
              y=[0, 1, 0, 0],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [1, 1, 4],
                  [[1, 0], [1, 0], [3, 1]], [1, 1, 4]),
              input_signature=[
                  tf.SparseTensorSpec([None, 2], tf.string),
                  tf.TensorSpec([None], tf.int64)
              ],
              filter_regex=None),
          dict(
              testcase_name='empty_sparse',
              x=tf.compat.v1.SparseTensorValue(
                  indices=np.empty([0, 2]), values=[], dense_shape=[4, 2]),
              y=[1, 0, 1, 1],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'global_y_count_sentinel'], [4], [[1, 3]], [4]),
              input_signature=[
                  tf.SparseTensorSpec([None, 2], tf.string),
                  tf.TensorSpec([None], tf.int64)
              ],
              filter_regex=None),
          dict(
              testcase_name='ragged',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array(['a', 'b', 'a', 'b', 'b']),
                          row_splits=np.array([0, 2, 3, 4, 5])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              y=[1, 0],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'a', b'b', b'global_y_count_sentinel'], [2, 3, 2],
                  [[0, 2], [1, 2], [1, 1]], [2, 3, 2]),
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.string),
                  tf.TensorSpec([None], tf.int64)
              ],
              filter_regex=None),
          dict(
              testcase_name='rank1_with_filtering',
              x=['yes\n', 'no', 'yes\n', '', 'yes\n'],
              y=[1, 1, 0, 2, 3],
              expected_result=tf_utils.ReducedBatchWeightedCounts(
                  [b'no', b'global_y_count_sentinel'], [1, 5],
                  [[0, 1, 0, 0], [1, 2, 1, 1]], [1, 5]),
              input_signature=[
                  tf.TensorSpec(None, tf.string),
                  tf.TensorSpec(None, tf.int64)
              ],
              filter_regex=analyzers._EMPTY_STRING_OR_NEWLINE_CHARS_REGEX),
      ]))
  def test_reduce_batch_coocurrences_no_weights(self, x, y, expected_result,
                                                input_signature, filter_regex,
                                                function_handler):
    @function_handler(input_signature=input_signature)
    def _reduce_batch_weighted_cooccurrences_no_weights(x, y):
      return tf_utils.reduce_batch_weighted_cooccurrences(
          x, y, filter_regex=filter_regex)

    result = _reduce_batch_weighted_cooccurrences_no_weights(x, y)

    self.assertAllEqual(result.unique_x,
                        expected_result.unique_x)
    self.assertAllEqual(result.summed_weights_per_x,
                        expected_result.summed_weights_per_x)
    self.assertAllEqual(result.summed_positive_per_x_and_y,
                        expected_result.summed_positive_per_x_and_y)
    self.assertAllEqual(result.counts_per_x,
                        expected_result.counts_per_x)

  @test_case.parameters(
      ([[1], [2]], [[1], [2], [3]], None, None, tf.errors.InvalidArgumentError,
       'Condition x == y did not hold element-wise:'),
      ([[1], [2], [3]], [[1], [2], [3]], [None, None], [None], ValueError,
       r'Shapes \(None, None\) and \(None,\) are incompatible'),
  )
  def test_same_shape_exceptions(self, x_input, y_input, x_shape, y_shape,
                                 exception_cls, error_string):

    with tf.compat.v1.Graph().as_default():
      x = tf.compat.v1.placeholder(tf.int32, x_shape)
      y = tf.compat.v1.placeholder(tf.int32, y_shape)
      with tf.compat.v1.Session() as sess:
        with self.assertRaisesRegexp(exception_cls, error_string):
          sess.run(tf_utils.assert_same_shape(x, y), {x: x_input, y: y_input})

  @test_case.named_parameters(test_case.FUNCTION_HANDLERS)
  def test_same_shape(self, function_handler):
    input_signature = [tf.TensorSpec(None, tf.int64),
                       tf.TensorSpec(None, tf.int64)]

    @function_handler(input_signature=input_signature)
    def _assert_shape(x, y):
      x_return, _ = tf_utils.assert_same_shape(x, y)
      return x_return

    input_list = [[1], [2], [3]]
    x_return = _assert_shape(input_list, input_list)
    self.assertAllEqual(x_return, input_list)

  @test_case.named_parameters([
      dict(
          testcase_name='_all_keys_in_vocab',
          query_list=['a', 'a', 'b', 'a', 'b'],
          key_vocab_list=['a', 'b'],
          query_shape=[None],
          expected_output=[0, 0, 1, 0, 1]),
      dict(
          testcase_name='_missing_keys_in_vocab',
          query_list=['a', 'c', 'b', 'a', 'b'],
          key_vocab_list=['a', 'b'],
          query_shape=[None],
          expected_output=[0, -1, 1, 0, 1]),
      dict(
          testcase_name='_nd_keys',
          query_list=[['a', 'c', 'b'], ['a', 'b', 'a']],
          key_vocab_list=['a', 'b'],
          query_shape=[None, None],
          expected_output=[[0, -1, 1], [0, 1, 0]]),
      dict(
          testcase_name='_empty_vocab',
          query_list=['a', 'c', 'b', 'a', 'b'],
          key_vocab_list=[],
          query_shape=[None],
          expected_output=[-1, -1, -1, -1, -1]),
      dict(
          testcase_name='_empty_query',
          query_list=[],
          key_vocab_list=['a'],
          query_shape=[None],
          expected_output=[]),
  ])
  def test_lookup_key(self, query_list, key_vocab_list, query_shape,
                      expected_output):
    with tf.compat.v1.Graph().as_default():
      query_ph = tf.compat.v1.placeholder(
          dtype=tf.string, shape=query_shape, name='query')
      key_vocab_ph = tf.compat.v1.placeholder(
          dtype=tf.string, shape=[None], name='key_vocab')
      key_indices = tf_utils.lookup_key(query_ph, key_vocab_ph)
      with tf.compat.v1.Session().as_default() as sess:
        output = sess.run(
            key_indices,
            feed_dict={
                query_ph.name: query_list,
                key_vocab_ph.name: key_vocab_list
            })
        self.assertAllEqual(expected_output, output)

  @test_case.named_parameters([
      dict(
          testcase_name='_with_default',
          with_default_value=True,
          input_keys=['a', 'b', 'c', 'd', 'e']),
      dict(
          testcase_name='_wihout_default',
          with_default_value=False,
          input_keys=['a', 'b', 'c', 'd', 'e']),
      dict(
          testcase_name='_single_oov_key',
          with_default_value=False,
          input_keys=['e'])
  ])
  def test_apply_per_key_vocab(self, with_default_value, input_keys):
    default_value = '-7,-5' if with_default_value else None
    vocab_data = [('0,0', 'a'), ('1,-1', 'b'), ('-1,1', 'c'), ('-2,2', 'd')]
    expected_missing_key_result = [-7, -5] if default_value else [0, 0]
    expected_lookup_results = {
        'a': [0, 0],
        'b': [1, -1],
        'c': [-1, 1],
        'd': [-2, 2],
    }

    with tf.compat.v1.Graph().as_default():
      input_tensor = _value_to_tensor(input_keys)
      vocab_filename = os.path.join(self.get_temp_dir(), 'test.txt')
      encoded_vocab = '\n'.join([' '.join(pair) for pair in vocab_data])
      with tf.io.gfile.GFile(vocab_filename, 'w') as f:
        f.write(encoded_vocab)

      output_tensor = tf_utils.apply_per_key_vocabulary(
          tf.constant(vocab_filename),
          input_tensor,
          default_value=default_value)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.tables_initializer())
        output = output_tensor.eval()

      expected_data = [
          expected_lookup_results.get(key, expected_missing_key_result)
          for key in input_keys
      ]
      self.assertAllEqual(output, expected_data)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='dense',
              x=[[[1], [2]], [[1], [2]]],
              expected_result=4,
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec(None, tf.int64)]),
          dict(
              testcase_name='dense_with_nans',
              x=[[[1], [np.nan]], [[1], [2]]],
              expected_result=3,
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='dense_elementwise',
              x=[[[1], [2]], [[1], [2]]],
              expected_result=[[2], [2]],
              reduce_instance_dims=False,
              input_signature=[tf.TensorSpec(None, tf.int64)]),
          dict(
              testcase_name='dense_elementwise_with_nans',
              x=[[[1], [2]], [[1], [np.nan]]],
              expected_result=[[2], [1]],
              reduce_instance_dims=False,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='sparse',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0]],
                  values=[1., 2., 3., 4.],
                  dense_shape=[2, 4, 1]),
              expected_result=4,
              reduce_instance_dims=True,
              input_signature=[tf.SparseTensorSpec([None, 4, 1], tf.float32)]),
          dict(
              testcase_name='sparse_with_nans',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0],
                           [1, 3, 0]],
                  values=[1., 2., 3., 4., np.nan],
                  dense_shape=[2, 4, 1]),
              expected_result=4,
              reduce_instance_dims=True,
              input_signature=[tf.SparseTensorSpec([None, 4, 1], tf.float32)]),
          dict(
              testcase_name='sparse_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0]],
                  values=[1., 2., 3., 4.],
                  dense_shape=[2, 4, 1]),
              expected_result=[[1], [1], [2], [0]],
              reduce_instance_dims=False,
              input_signature=[tf.SparseTensorSpec([None, 4, 1], tf.float32)]),
          dict(
              testcase_name='sparse_elementwise_with_nans',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0],
                           [1, 3, 0]],
                  values=[1., 2., 3., 4., np.nan],
                  dense_shape=[2, 4, 1]),
              expected_result=[[1], [1], [2], [0]],
              reduce_instance_dims=False,
              input_signature=[tf.SparseTensorSpec([None, 4, 1], tf.float32)]),
          dict(
              testcase_name='ragged',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5.], np.float32),
                          row_splits=np.array([0, 2, 3, 4, 5])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_result=5,
              reduce_instance_dims=True,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32)
              ]),
          dict(
              testcase_name='ragged_with_nans',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5., np.nan],
                                          np.float32),
                          row_splits=np.array([0, 2, 3, 4, 6])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_result=5,
              reduce_instance_dims=True,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32)
              ]),
          dict(
              testcase_name='ragged_elementwise',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5.], np.float32),
                          row_splits=np.array([0, 2, 2, 4, 5])),
                      row_splits=np.array([0, 3, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_result=[[[2, 1], [0., 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 0]]],
              reduce_instance_dims=False,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32)
              ]),
          dict(
              testcase_name='ragged_elementwise_with_nans',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5., np.nan],
                                          np.float32),
                          row_splits=np.array([0, 2, 2, 4, 6])),
                      row_splits=np.array([0, 3, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_result=[[[2, 1], [0., 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 0]]],
              reduce_instance_dims=False,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32)
              ]),
      ]))
  def test_reduce_batch_count(self, x, input_signature, expected_result,
                              reduce_instance_dims, function_handler):

    @function_handler(input_signature=input_signature)
    def _reduce_batch_count(x):
      result = tf_utils.reduce_batch_count(
          x, reduce_instance_dims=reduce_instance_dims)
      # Verify that the output shape is maintained.
      # TODO(b/178189903): This will fail if _dense_shape_default isn't set in
      # reduce_batch_count.
      if (not isinstance(x, tf.RaggedTensor) and not reduce_instance_dims and
          x.get_shape().ndims):
        self.assertEqual(x.get_shape()[1:].as_list(),
                         result.get_shape().as_list())
      return result

    result = _reduce_batch_count(x)
    self.assertAllEqual(result, expected_result)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='dense',
              x=[[[1], [2]], [[3], [4]]],
              expected_count=4,
              expected_mean=2.5,
              expected_var=1.25,
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='dense_with_nans',
              x=[[[1], [2]], [[3], [np.nan]], [[np.nan], [4]]],
              expected_count=4,
              expected_mean=2.5,
              expected_var=1.25,
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='dense_elementwise',
              x=[[[1], [2]], [[3], [4]]],
              expected_count=[[2.], [2.]],
              expected_mean=[[2.], [3.]],
              expected_var=[[1.], [1.]],
              reduce_instance_dims=False,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='dense_elementwise_with_nans',
              x=[[[1], [2]], [[3], [np.nan]], [[np.nan], [4]]],
              expected_count=[[2.], [2.]],
              expected_mean=[[2.], [3.]],
              expected_var=[[1.], [1.]],
              reduce_instance_dims=False,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='sparse',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 2], [1, 1], [1, 2]],
                  values=[1., 2., 3., 4.],
                  dense_shape=[2, 4]),
              expected_count=4,
              expected_mean=2.5,
              expected_var=1.25,
              reduce_instance_dims=True,
              input_signature=[tf.SparseTensorSpec([None, 4], tf.float32)]),
          dict(
              testcase_name='sparse_with_nans',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 2], [1, 1], [1, 2], [1, 3]],
                  values=[1., 2., 3., 4., np.nan],
                  dense_shape=[2, 4]),
              expected_count=4,
              expected_mean=2.5,
              expected_var=1.25,
              reduce_instance_dims=True,
              input_signature=[tf.SparseTensorSpec([None, 4], tf.float32)]),
          dict(
              testcase_name='sparse_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 3], [1, 1], [1, 3]],
                  values=[1., 2., 3., 4.],
                  dense_shape=[2, 5]),
              expected_count=[1.0, 1.0, 0.0, 2.0, 0.0],
              expected_mean=[1.0, 3.0, 0.0, 3.0, 0.0],
              expected_var=[0.0, 0.0, 0.0, 1.0, 0.0],
              reduce_instance_dims=False,
              input_signature=[tf.SparseTensorSpec([None, 5], tf.float32)]),
          dict(
              testcase_name='sparse_elementwise_with_nans',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 3], [1, 1], [1, 2], [1, 3]],
                  values=[1., 2., 3., np.nan, 4.],
                  dense_shape=[2, 5]),
              expected_count=[1.0, 1.0, 0.0, 2.0, 0.0],
              expected_mean=[1.0, 3.0, 0.0, 3.0, 0.0],
              expected_var=[0.0, 0.0, 0.0, 1.0, 0.0],
              reduce_instance_dims=False,
              input_signature=[tf.SparseTensorSpec([None, 5], tf.float32)]),
          dict(
              testcase_name='sparse_3d_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0, 3], [0, 1, 0], [0, 1, 3], [1, 1, 1],
                           [1, 1, 3]],
                  values=[-10., 1., 2., 3., 4.],
                  dense_shape=[2, 3, 5]),
              expected_count=[[0, 0, 0, 1, 0], [1, 1, 0, 2, 0], [0] * 5],
              expected_mean=[[0, 0, 0, -10, 0], [1, 3, 0, 3, 0], [0] * 5],
              expected_var=[[0] * 5, [0, 0, 0, 1, 0], [0] * 5],
              reduce_instance_dims=False,
              input_signature=[tf.SparseTensorSpec([None, 3, 5], tf.float32)]),
          dict(
              testcase_name='ragged',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5.], np.float32),
                          row_splits=np.array([0, 2, 3, 4, 5])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_count=5,
              expected_mean=3,
              expected_var=2,
              reduce_instance_dims=True,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32)
              ]),
          dict(
              testcase_name='ragged_with_nans',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5., np.nan],
                                          np.float32),
                          row_splits=np.array([0, 2, 3, 4, 6])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_count=5,
              expected_mean=3,
              expected_var=2,
              reduce_instance_dims=True,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32)
              ]),
          dict(
              testcase_name='ragged_elementwise',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5.], np.float32),
                          row_splits=np.array([0, 2, 2, 4, 5])),
                      row_splits=np.array([0, 3, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_count=[[[2., 1.], [0., 0.], [1., 1.]],
                              [[0., 0.], [0., 0.], [0., 0.]]],
              expected_mean=[[[3., 2.], [0., 0.], [3., 4.]],
                             [[0., 0.], [0., 0.], [0., 0.]]],
              expected_var=[[[4., 0.], [0., 0.], [0., 0.]],
                            [[0., 0.], [0., 0.], [0., 0.]]],
              reduce_instance_dims=False,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32)
              ]),
          dict(
              testcase_name='ragged_elementwise_with_nans',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5., np.nan],
                                          np.float32),
                          row_splits=np.array([0, 2, 2, 4, 6])),
                      row_splits=np.array([0, 3, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_count=[[[2., 1.], [0., 0.], [1., 1.]],
                              [[0., 0.], [0., 0.], [0., 0.]]],
              expected_mean=[[[3., 2.], [0., 0.], [3., 4.]],
                             [[0., 0.], [0., 0.], [0., 0.]]],
              expected_var=[[[4., 0.], [0., 0.], [0., 0.]],
                            [[0., 0.], [0., 0.], [0., 0.]]],
              reduce_instance_dims=False,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32)
              ]),
      ]))
  def test_reduce_batch_count_mean_and_var(
      self, x, input_signature, expected_count, expected_mean, expected_var,
      reduce_instance_dims, function_handler):

    @function_handler(input_signature=input_signature)
    def _reduce_batch_count_mean_and_var(x):
      result = tf_utils.reduce_batch_count_mean_and_var(
          x, reduce_instance_dims=reduce_instance_dims)
      # Verify that the output shapes are maintained.
      # TODO(b/178189903): This will fail if _dense_shape_default isn't set in
      # reduce_batch_count.
      if (not isinstance(x, tf.RaggedTensor) and not reduce_instance_dims and
          x.get_shape().ndims):
        for tensor in result:
          self.assertEqual(x.get_shape()[1:].as_list(),
                           tensor.get_shape().as_list())
      return result

    count, mean, var = _reduce_batch_count_mean_and_var(x)
    self.assertAllEqual(expected_count, count)
    self.assertAllEqual(expected_mean, mean)
    self.assertAllEqual(expected_var, var)

  @test_case.named_parameters([
      dict(
          testcase_name='num_samples_1',
          num_samples=1,
          dtype=tf.float32,
          expected_counts=np.array([1, 0, 0, 0], np.float32),
          expected_factors=np.array([[1.0], [0.0], [0.0], [0.0]], np.float32)),
      dict(
          testcase_name='num_samples_2',
          num_samples=2,
          dtype=tf.float32,
          expected_counts=np.array([2, 1, 0, 0], np.float32),
          expected_factors=np.array(
              [[1. / 2., 1. / 2.], [-1. / 2., 1. / 2.], [0., 0.], [0., 0.]],
              np.float32)),
      dict(
          testcase_name='num_samples_3',
          num_samples=3,
          dtype=tf.float32,
          expected_counts=np.array([3, 3, 1, 0], np.float32),
          expected_factors=np.array(
              [[1. / 3., 1. / 3., 1. / 3.], [-1. / 3., 0., 1. / 3.],
               [1. / 3., -2. / 3., 1. / 3.], [0., 0., 0.]], np.float32)),
      dict(
          testcase_name='num_samples_4',
          num_samples=4,
          dtype=tf.float32,
          expected_counts=np.array([4, 6, 4, 1], np.float32),
          expected_factors=np.array(
              [[1. / 4., 1. / 4., 1. / 4., 1. / 4.],
               [-3. / 12., -1. / 12., 1. / 12., 3. / 12.],
               [1. / 4., -1. / 4., -1. / 4., 1. / 4.],
               [-1. / 4., 3. / 4., -3. / 4., 1. / 4.]], np.float32))
  ])
  def test_num_terms_and_factors(
      self, num_samples, dtype, expected_counts, expected_factors):
    results = tf_utils._num_terms_and_factors(num_samples, dtype)
    counts = results[0:4]
    assert len(expected_counts) == len(counts), (expected_counts, counts)
    for result, expected_count in zip(counts, expected_counts):
      self.assertEqual(result.dtype, dtype)
      self.assertAllClose(result, expected_count)

    factors = results[4:]
    assert len(expected_factors) == len(factors), (expected_factors, factors)
    for result, expected_factor in zip(factors, expected_factors):
      self.assertEqual(result.dtype, dtype)
      self.assertAllClose(result, expected_factor)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='dense',
              x=[[[1], [2]], [[3], [4]]],
              expected_counts=np.array([4., 6., 4., 1.], np.float32),
              expected_moments=np.array([2.5, 10.0 / 12.0, 0.0, 0.0],
                                        np.float32),
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='dense_large',
              x=[2.0, 3.0, 4.0, 2.4, 5.5, 1.2, 5.4, 2.2, 7.1, 1.3, 1.5],
              expected_counts=np.array(
                  [11, 11 * 10 // 2, 11 * 10 * 9 // 6, 11 * 10 * 9 * 8 // 24],
                  np.float32),
              expected_moments=np.array([
                  3.2363636363636363, 1.141818181818182, 0.31272727272727263,
                  0.026666666666666616
              ], np.float32),
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='dense_very_large',
              x=-np.log(1.0 - np.arange(0, 1, 1e-6, dtype=np.float32)),
              expected_counts=np.array([
                  1000000, 499999500000.0, 1.66666166667e+17,
                  4.1666416667125e+22
              ], np.float32),
              expected_moments=np.array([
                  0.99999217330, 0.4999936732947, 0.166660839941,
                  0.0833278399134
              ], np.float32),
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='dense_elementwise',
              x=[[[1], [2]], [[3], [4]]],
              expected_counts=np.array(
                  [[[2], [2]], [[1], [1]], [[0], [0]], [[0], [0]]], np.float32),
              expected_moments=np.array([[[2.0], [3.0]], [[1.0], [1.0]],
                                         [[0.0], [0.0]], [[0.0], [0.0]]],
                                        np.float32),
              reduce_instance_dims=False,
              input_signature=[tf.TensorSpec(None, tf.float32)]),
          dict(
              testcase_name='sparse',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 2], [2, 0], [2, 2]],
                  values=[1., 2., 3., 4.],
                  dense_shape=[3, 4]),
              expected_counts=np.array([4, 6, 4, 1], np.float32),
              expected_moments=np.array([2.5, 10.0 / 12.0, 0.0, 0.0],
                                        np.float32),
              reduce_instance_dims=True,
              input_signature=[tf.SparseTensorSpec([None, 4], tf.float32)]),
          dict(
              testcase_name='sparse_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0],
                           [3, 3, 0]],
                  values=[1., 2., 3., 4., 5.],
                  dense_shape=[3, 5, 1]),
              expected_counts=np.array(
                  [[[2], [0], [2], [1], [0]], [[1], [0], [1], [0], [0]],
                   [[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]]],
                  np.float32),
              expected_moments=np.array([[[2.0], [0.0], [3.0], [5.0], [0.0]],
                                         [[1.0], [0.0], [1.0], [0.0], [0.0]],
                                         [[0.0], [0.0], [0.0], [0.0], [0.0]],
                                         [[0.0], [0.0], [0.0], [0.0], [0.0]]],
                                        np.float32),
              reduce_instance_dims=False,
              input_signature=[tf.SparseTensorSpec([None, 5, 1], tf.float32)]),
          dict(
              testcase_name='ragged',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5.], np.float32),
                          row_splits=np.array([0, 2, 3, 4, 5])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_counts=np.array([5., 10., 10., 5.], np.float32),
              expected_moments=np.array([3., 1., 0., 0.], np.float32),
              reduce_instance_dims=True,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32)
              ]),
      ]))
  def test_reduce_batch_count_l_moments(
      self, x, input_signature, expected_counts, expected_moments,
      reduce_instance_dims, function_handler):

    @function_handler(input_signature=input_signature)
    def _reduce_batch_count_l_moments(x):
      result = tf_utils.reduce_batch_count_l_moments(
          x, reduce_instance_dims=reduce_instance_dims)
      for tensor in result:
        if not reduce_instance_dims and x.get_shape().ndims:
          self.assertEqual(x.get_shape()[1:].as_list(),
                           tensor.get_shape().as_list())
      return result

    count_and_moments = _reduce_batch_count_l_moments(x)
    counts = count_and_moments[0::2]
    moments = count_and_moments[1::2]
    for i in range(0, 4):
      self.assertEqual(counts[i].dtype, expected_counts[i].dtype)
      self.assertAllClose(counts[i], expected_counts[i], rtol=1e-8)
      self.assertEqual(moments[i].dtype, expected_moments[i].dtype)
      self.assertAllClose(moments[i], expected_moments[i], rtol=1e-8)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='dense',
              x=[[1], [2], [3], [4], [4]],
              key=['a', 'a', 'a', 'b', 'a'],
              expected_key_vocab=[b'a', b'b'],
              expected_count=[4., 1.],
              expected_mean=[2.5, 4.],
              expected_var=[1.25, 0.],
              reduce_instance_dims=True,
              input_signature=[
                  tf.TensorSpec([None, 1], tf.float32),
                  tf.TensorSpec([None], tf.string)
              ]),
          dict(
              testcase_name='dense_with_nans',
              x=[[1], [2], [3], [4], [4], [np.nan], [np.nan]],
              key=['a', 'a', 'a', 'b', 'a', 'a', 'b'],
              expected_key_vocab=[b'a', b'b'],
              expected_count=[4., 1.],
              expected_mean=[2.5, 4.],
              expected_var=[1.25, 0.],
              reduce_instance_dims=True,
              input_signature=[
                  tf.TensorSpec([None, 1], tf.float32),
                  tf.TensorSpec([None], tf.string)
              ]),
          dict(
              testcase_name='dense_elementwise',
              x=[[1, 2], [3, 4], [1, 2]],
              key=['a', 'a', 'b'],
              expected_key_vocab=[b'a', b'b'],
              expected_count=[[2., 2.], [1., 1.]],
              expected_mean=[[2., 3.], [1., 2.]],
              expected_var=[[1., 1.], [0., 0.]],
              reduce_instance_dims=False,
              input_signature=[
                  tf.TensorSpec([None, 2], tf.float32),
                  tf.TensorSpec([None], tf.string)
              ]),
          dict(
              testcase_name='dense_elementwise_with_nans',
              x=[[1, 2], [3, 4], [1, 2], [np.nan, np.nan]],
              key=['a', 'a', 'b', 'a'],
              expected_key_vocab=[b'a', b'b'],
              expected_count=[[2., 2.], [1., 1.]],
              expected_mean=[[2., 3.], [1., 2.]],
              expected_var=[[1., 1.], [0., 0.]],
              reduce_instance_dims=False,
              input_signature=[
                  tf.TensorSpec([None, 2], tf.float32),
                  tf.TensorSpec([None], tf.string)
              ]),
          dict(
              testcase_name='sparse',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 2], [1, 1], [1, 2], [2, 3]],
                  values=[1., 2., 3., 4., 4.],
                  dense_shape=[3, 4]),
              key=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 2], [1, 1], [1, 2], [2, 3]],
                  values=['a', 'a', 'a', 'a', 'b'],
                  dense_shape=[3, 4]),
              expected_key_vocab=[b'a', b'b'],
              expected_count=[4, 1],
              expected_mean=[2.5, 4],
              expected_var=[1.25, 0],
              reduce_instance_dims=True,
              input_signature=[
                  tf.SparseTensorSpec([None, 4], tf.float32),
                  tf.SparseTensorSpec([None, 4], tf.string)
              ]),
          dict(
              testcase_name='sparse_with_nans',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 2], [1, 1], [1, 2], [2, 2], [2, 3]],
                  values=[1., 2., 3., 4., np.nan, 4.],
                  dense_shape=[3, 4]),
              key=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 2], [1, 1], [1, 2], [2, 2], [2, 3]],
                  values=['a', 'a', 'a', 'a', 'a', 'b'],
                  dense_shape=[3, 4]),
              expected_key_vocab=[b'a', b'b'],
              expected_count=[4, 1],
              expected_mean=[2.5, 4],
              expected_var=[1.25, 0],
              reduce_instance_dims=True,
              input_signature=[
                  tf.SparseTensorSpec([None, 4], tf.float32),
                  tf.SparseTensorSpec([None, 4], tf.string)
              ]),
          dict(
              testcase_name='sparse_x_dense_key',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 2], [1, 1], [1, 2], [2, 3]],
                  values=[1., 2., 3., 4., 4.],
                  dense_shape=[3, 4]),
              key=['a', 'a', 'b'],
              expected_key_vocab=[b'a', b'b'],
              expected_count=[4, 1],
              expected_mean=[2.5, 4],
              expected_var=[1.25, 0],
              reduce_instance_dims=True,
              input_signature=[
                  tf.SparseTensorSpec([None, 4], tf.float32),
                  tf.TensorSpec([None], tf.string)
              ]),
          dict(
              testcase_name='ragged',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([3., 2., 3., 4., 5.], np.float32),
                          row_splits=np.array([0, 2, 3, 4, 5])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              key=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array(['a', 'a', 'b', 'a', 'b']),
                          row_splits=np.array([0, 2, 3, 4, 5])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_key_vocab=[b'a', b'b'],
              expected_count=[3, 2],
              expected_mean=[3, 4],
              expected_var=[np.float32(0.666667), 1.],
              reduce_instance_dims=True,
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32),
                  tf.RaggedTensorSpec([None, None, None, None], tf.string)
              ]),
          dict(
              testcase_name='ragged_x_dense_key',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([3., 2., 3., 4., 5.], np.float32),
                          row_splits=np.array([0, 2, 3, 4, 5])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              key=['a', 'b'],
              expected_key_vocab=[b'a', b'b'],
              expected_count=[4, 1],
              expected_mean=[3, 5],
              expected_var=[.5, 0.],
              reduce_instance_dims=True,
              input_signature=[
                  tf.RaggedTensorSpec([2, None, None, None], tf.float32),
                  tf.TensorSpec([2], tf.string)
              ]),
          dict(
              testcase_name='ragged_with_nans',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([3., 2., 3., 4., 5., np.nan],
                                          np.float32),
                          row_splits=np.array([0, 2, 3, 4, 6])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              key=['a', 'b'],
              expected_key_vocab=[b'a', b'b'],
              expected_count=[4, 1],
              expected_mean=[3, 5],
              expected_var=[.5, 0.],
              reduce_instance_dims=True,
              input_signature=[
                  tf.RaggedTensorSpec([2, None, None, None], tf.float32),
                  tf.TensorSpec([2], tf.string)
              ]),
      ]))
  def test_reduce_batch_count_mean_and_var_per_key(
      self, x, key, input_signature, expected_key_vocab, expected_count,
      expected_mean, expected_var, reduce_instance_dims, function_handler):

    @function_handler(input_signature=input_signature)
    def _reduce_batch_count_mean_and_var_per_key(x, key):
      return tf_utils.reduce_batch_count_mean_and_var_per_key(
          x, key, reduce_instance_dims=reduce_instance_dims)

    key_vocab, count, mean, var = _reduce_batch_count_mean_and_var_per_key(
        x, key)

    self.assertAllEqual(key_vocab, expected_key_vocab)
    self.assertAllEqual(count, expected_count)
    self.assertAllEqual(mean, expected_mean)
    self.assertAllEqual(var, expected_var)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='sparse',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 1], [0, 2]],
                  values=[3, 2, -1],
                  dense_shape=[1, 5]),
              expected_x_minus_min=1,
              expected_x_max=3,
              reduce_instance_dims=True,
              input_signature=[tf.SparseTensorSpec([None, None], tf.int64)]),
          dict(
              testcase_name='float',
              x=[[1, 5, 2]],
              expected_x_minus_min=-1,
              expected_x_max=5,
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec([None, None], tf.float32)]),
          dict(
              testcase_name='sparse_float_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 1], [1, 0]],
                  values=[3, 2, -1],
                  dense_shape=[2, 3]),
              expected_x_minus_min=[1, -2, np.nan],
              expected_x_max=[3, 2, np.nan],
              reduce_instance_dims=False,
              input_signature=[tf.SparseTensorSpec([None, None], tf.float32)]),
          dict(
              testcase_name='float_elementwise',
              x=[[1, 5, 2], [2, 3, 4]],
              reduce_instance_dims=False,
              expected_x_minus_min=[-1, -3, -2],
              expected_x_max=[2, 5, 4],
              input_signature=[tf.TensorSpec([None, None], tf.float32)]),
          dict(
              testcase_name='sparse_int64_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 1], [1, 0]],
                  values=[3, 2, -1],
                  dense_shape=[2, 3]),
              reduce_instance_dims=False,
              expected_x_minus_min=[1, -2, tf.int64.min + 1],
              expected_x_max=[3, 2, tf.int64.min + 1],
              input_signature=[tf.SparseTensorSpec([None, None], tf.int64)]),
          dict(
              testcase_name='sparse_int32_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 1], [1, 0]],
                  values=[3, 2, -1],
                  dense_shape=[2, 3]),
              reduce_instance_dims=False,
              expected_x_minus_min=[1, -2, tf.int32.min + 1],
              expected_x_max=[3, 2, tf.int32.min + 1],
              input_signature=[tf.SparseTensorSpec([None, None], tf.int32)]),
          dict(
              testcase_name='sparse_float64_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 1], [1, 0]],
                  values=[3, 2, -1],
                  dense_shape=[2, 3]),
              reduce_instance_dims=False,
              expected_x_minus_min=[1, -2, np.nan],
              expected_x_max=[3, 2, np.nan],
              input_signature=[tf.SparseTensorSpec([None, None], tf.float64)]),
          dict(
              testcase_name='sparse_float32_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [0, 1], [1, 0]],
                  values=[3, 2, -1],
                  dense_shape=[2, 3]),
              reduce_instance_dims=False,
              expected_x_minus_min=[1, -2, np.nan],
              expected_x_max=[3, 2, np.nan],
              input_signature=[tf.SparseTensorSpec([None, None], tf.float32)]),
          dict(
              testcase_name='sparse_3d_elementwise',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0, 0], [0, 0, 1], [1, 0, 1]],
                  values=[3, 2, -1],
                  dense_shape=[2, 3, 3]),
              reduce_instance_dims=False,
              expected_x_minus_min=[[-3, 1, np.nan], [np.nan] * 3,
                                    [np.nan] * 3],
              expected_x_max=[[3, 2, np.nan], [np.nan] * 3, [np.nan] * 3],
              input_signature=[
                  tf.SparseTensorSpec([None, None, None], tf.float32)
              ]),
          dict(
              testcase_name='ragged',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=np.array([1., 2., 3., 4., 5.], np.float32),
                      row_splits=np.array([0, 2, 3, 5])),
                  row_splits=np.array([0, 2, 3])),
              reduce_instance_dims=True,
              expected_x_minus_min=-1.,
              expected_x_max=5.,
              input_signature=[
                  tf.RaggedTensorSpec([2, None, None], tf.float32)
              ]),
          dict(
              testcase_name='ragged_elementwise',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([1., 2., 3., 4., 5.], np.float32),
                          row_splits=np.array([0, 2, 2, 4, 5])),
                      row_splits=np.array([0, 3, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              reduce_instance_dims=False,
              expected_x_minus_min=[[[-1.0, -2.0], [np.nan, np.nan],
                                     [-3.0, -4.0]],
                                    [[np.nan, np.nan], [np.nan, np.nan],
                                     [np.nan, np.nan]]],
              expected_x_max=[[[5.0, 2.0], [np.nan, np.nan], [3.0, 4.0]],
                              [[np.nan, np.nan], [np.nan, np.nan],
                               [np.nan, np.nan]]],
              input_signature=[
                  tf.RaggedTensorSpec([2, None, None, None], tf.float32)
              ]),
          dict(
              testcase_name='all_nans',
              x=[[np.nan, np.nan, np.nan]],
              # Output of `tf.reduce_max` if all inputs are NaNs for older
              # versions of TF is -inf.
              expected_x_minus_min=(-np.inf
                                    if tf.__version__ < '2.4' else np.nan),
              expected_x_max=-np.inf if tf.__version__ < '2.4' else np.nan,
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec([None, None], tf.float32)]),
          dict(
              testcase_name='empty_batch',
              x=[[]],
              expected_x_minus_min=-np.inf,
              expected_x_max=-np.inf,
              reduce_instance_dims=True,
              input_signature=[tf.TensorSpec([None, None], tf.float32)]),
      ]))
  def test_reduce_batch_minus_min_and_max(
      self, x, expected_x_minus_min, expected_x_max, reduce_instance_dims,
      input_signature, function_handler):

    @function_handler(input_signature=input_signature)
    def _reduce_batch_minus_min_and_max(x):
      result = tf_utils.reduce_batch_minus_min_and_max(
          x, reduce_instance_dims=reduce_instance_dims)
      # Verify that the output shapes are maintained.
      if (not reduce_instance_dims and not isinstance(x, tf.RaggedTensor)):
        for tensor in result:
          self.assertEqual(x.get_shape()[1:].as_list(),
                           tensor.get_shape().as_list())
      return result

    x_minus_min, x_max = _reduce_batch_minus_min_and_max(x)

    self.assertAllEqual(x_minus_min, expected_x_minus_min)
    self.assertAllEqual(x_max, expected_x_max)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='sparse',
              x=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [1, 1], [2, 2], [3, 1]],
                  values=[3, 2, -1, 3],
                  dense_shape=[4, 5]),
              key=['a', 'a', 'a', 'b'],
              expected_key_vocab=[b'a', b'b'],
              expected_x_minus_min=[1, -3],
              expected_x_max=[3, 3],
              input_signature=[
                  tf.SparseTensorSpec([None, None], tf.int64),
                  tf.TensorSpec([None], tf.string)
              ]),
          dict(
              testcase_name='float',
              x=[[1], [5], [2], [3]],
              key=['a', 'a', 'a', 'b'],
              expected_key_vocab=[b'a', b'b'],
              expected_x_minus_min=[-1, -3],
              expected_x_max=[5, 3],
              input_signature=[
                  tf.TensorSpec([None, None], tf.float32),
                  tf.TensorSpec([None], tf.string)
              ]),
          dict(
              testcase_name='float3dims',
              x=[[[1, 5], [1, 1]], [[5, 1], [5, 5]], [[2, 2], [2, 5]],
                 [[3, -3], [3, 3]]],
              key=['a', 'a', 'a', 'b'],
              expected_key_vocab=[b'a', b'b'],
              expected_x_minus_min=[-1, 3],
              expected_x_max=[5, 3],
              input_signature=[
                  tf.TensorSpec([None, None, None], tf.float32),
                  tf.TensorSpec([None], tf.string)
              ]),
          dict(
              testcase_name='ragged',
              x=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array([3., 2., 3., 4., 5.], np.float32),
                          row_splits=np.array([0, 2, 3, 4, 5])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              key=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=tf.compat.v1.ragged.RaggedTensorValue(
                          values=np.array(['a', 'a', 'b', 'a', 'b']),
                          row_splits=np.array([0, 2, 3, 4, 5])),
                      row_splits=np.array([0, 2, 3, 4])),
                  row_splits=np.array([0, 2, 3])),
              expected_key_vocab=[b'a', b'b'],
              expected_x_minus_min=[-2., -3.],
              expected_x_max=[4., 5.],
              input_signature=[
                  tf.RaggedTensorSpec([None, None, None, None], tf.float32),
                  tf.RaggedTensorSpec([None, None, None, None], tf.string)
              ]),
      ]))
  def test_reduce_batch_minus_min_and_max_per_key(
      self, x, key, expected_key_vocab, expected_x_minus_min, expected_x_max,
      input_signature, function_handler):

    @function_handler(input_signature=input_signature)
    def _reduce_batch_minus_min_and_max_per_key(x, key):
      return tf_utils.reduce_batch_minus_min_and_max_per_key(x, key)

    key_vocab, x_minus_min, x_max = _reduce_batch_minus_min_and_max_per_key(
        x, key)

    self.assertAllEqual(key_vocab, expected_key_vocab)
    self.assertAllEqual(x_minus_min, expected_x_minus_min)
    self.assertAllEqual(x_max, expected_x_max)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='dense',
              key=['a', 'a', 'a', 'b'],
              spec=tf.TensorSpec([None], tf.string),
              expected_key_vocab=[b'a', b'b'],
              expected_count=[3, 1]),
          dict(
              testcase_name='sparse',
              key=tf.compat.v1.SparseTensorValue(
                  indices=[[0, 0], [1, 1], [2, 2], [3, 1]],
                  values=[3, 2, -1, 3],
                  dense_shape=[4, 5]),
              spec=tf.SparseTensorSpec([4, 5], tf.int64),
              expected_key_vocab=[b'3', b'2', b'-1'],
              expected_count=[2, 1, 1]),
          dict(
              testcase_name='ragged',
              key=tf.compat.v1.ragged.RaggedTensorValue(
                  values=tf.compat.v1.ragged.RaggedTensorValue(
                      values=np.array([1.2, 1., 1.2, 1.]),
                      row_splits=np.array([0, 2, 4])),
                  row_splits=np.array([0, 2])),
              spec=tf.RaggedTensorSpec([1, None, None], tf.float32),
              expected_key_vocab=[b'1.200000', b'1.000000'],
              expected_count=[2, 2]),
      ]))
  def test_reduce_batch_count_per_key(self, key, spec, expected_key_vocab,
                                      expected_count, function_handler):

    @function_handler(input_signature=[spec])
    def _reduce_batch_count_per_key(key):
      return tf_utils.reduce_batch_count_per_key(key)

    key_vocab, key_counts = _reduce_batch_count_per_key(key)

    self.assertAllEqual(key_vocab, expected_key_vocab)
    self.assertAllEqual(key_counts, expected_count)

  @test_case.named_parameters(test_case.cross_with_function_handlers([
      dict(
          testcase_name='full',
          bucket_vocab=['1', '2', '0'],
          counts=[3, 1, 4],
          boundary_size=3,
          expected_counts=[4, 3, 1]),
      dict(
          testcase_name='missing',
          bucket_vocab=['1', '3', '0'],
          counts=[3, 1, 4],
          boundary_size=5,
          expected_counts=[4, 3, 0, 1, 0]),
  ]))
  def test_reorder_histogram(
      self, bucket_vocab, counts, boundary_size,
      expected_counts, function_handler):
    input_signature = [tf.TensorSpec([None], tf.string),
                       tf.TensorSpec([None], tf.int64),
                       tf.TensorSpec([], tf.int32)]
    @function_handler(input_signature=input_signature)
    def _reorder_histogram(bucket_vocab, counts, boundary_size):
      return tf_utils.reorder_histogram(bucket_vocab, counts, boundary_size)

    counts = _reorder_histogram(bucket_vocab, counts, boundary_size)
    self.assertAllEqual(counts, expected_counts)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers([
          dict(
              testcase_name='simple',
              x=[0.0, 2.0, 3.5, 4.0],
              x_spec=tf.TensorSpec([None], tf.float32),
              boundaries=[[1.0, 2.0, 3.0, 3.9]],
              boundaries_spec=tf.TensorSpec([1, None], tf.float32),
              side=tf_utils.Side.LEFT,
              expected_buckets=[0, 1, 3, 3]),
          dict(
              testcase_name='simple_right',
              x=[0.0, 2.0, 3.5, 4.0],
              x_spec=tf.TensorSpec([None], tf.float32),
              boundaries=[1.0, 2.0, 3.0, 3.9],
              boundaries_spec=tf.TensorSpec([None], tf.float32),
              side=tf_utils.Side.RIGHT,
              expected_buckets=[0, 2, 3, 4]),
          dict(
              testcase_name='2dim',
              x=[[0.0, 4.0, 3.5, 2.0, 1.7]],
              x_spec=tf.TensorSpec([1, None], tf.float32),
              boundaries=[[1.0, 2.0, 3.0, 5.0]],
              boundaries_spec=tf.TensorSpec([1, None], tf.float32),
              side=tf_utils.Side.LEFT,
              expected_buckets=[[0, 3, 3, 1, 1]]),
          dict(
              testcase_name='large_buckets',
              x=[[50_000_000]],
              x_spec=tf.TensorSpec([1, None], tf.int64),
              boundaries=[0, 50_000_001, 100_000_001],
              boundaries_spec=tf.TensorSpec([None], tf.int64),
              side=tf_utils.Side.RIGHT,
              expected_buckets=[[1]]),
      ]))
  def test_assign_buckets(self, x, x_spec, boundaries, boundaries_spec, side,
                          expected_buckets, function_handler):

    @function_handler(input_signature=[x_spec, boundaries_spec])
    def _assign_buckets(x, boundaries):
      return tf_utils.assign_buckets(x, boundaries, side)

    buckets = _assign_buckets(x, boundaries)
    self.assertAllEqual(buckets, expected_buckets)

  def test_sparse_indices(self):
    exception_cls = tf.errors.InvalidArgumentError
    error_string = 'Condition x == y did not hold element-wise:'
    value = tf.compat.v1.SparseTensorValue(
        indices=[[0, 0], [1, 1], [2, 2], [3, 1]],
        values=[3, 2, -1, 3],
        dense_shape=[4, 5])
    key_value = tf.compat.v1.SparseTensorValue(
        indices=[[0, 0], [1, 2], [2, 2], [3, 1]],
        values=['a', 'a', 'a', 'b'],
        dense_shape=[4, 5])
    with tf.compat.v1.Graph().as_default():
      x = tf.compat.v1.sparse_placeholder(tf.int64, shape=[None, None])
      key = tf.compat.v1.sparse_placeholder(tf.string, shape=[None, None])
      with tf.compat.v1.Session() as sess:
        with self.assertRaisesRegexp(exception_cls, error_string):
          sess.run(tf_utils.reduce_batch_minus_min_and_max_per_key(x, key),
                   feed_dict={x: value, key: key_value})

  def test_convert_sparse_indices(self):
    exception_cls = tf.errors.InvalidArgumentError
    error_string = 'Condition x == y did not hold element-wise:'
    sparse = tf.SparseTensor(
        indices=[[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 1]],
        values=[3, 2, -1, 3],
        dense_shape=[4, 2, 5])
    dense = tf.constant(['a', 'b', 'c', 'd'])
    x, key = tf_utils._validate_and_get_dense_value_key_inputs(sparse, sparse)
    self.assertAllEqual(self.evaluate(x), sparse.values)
    self.assertAllEqual(self.evaluate(key), sparse.values)

    x, key = tf_utils._validate_and_get_dense_value_key_inputs(sparse, dense)
    self.assertAllEqual(self.evaluate(x), sparse.values)
    self.assertAllEqual(self.evaluate(key), dense)

    with tf.compat.v1.Graph().as_default():
      sparse1 = tf.compat.v1.sparse_placeholder(
          tf.int64, shape=[None, None, None])
      sparse2 = tf.compat.v1.sparse_placeholder(
          tf.int64, shape=[None, None, None])
      sparse_value1 = tf.compat.v1.SparseTensorValue(
          indices=[[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 1]],
          values=[3, 2, -1, 3],
          dense_shape=[4, 2, 5])
      sparse_value2 = tf.compat.v1.SparseTensorValue(
          indices=[[0, 0, 0], [1, 0, 2], [2, 0, 2], [3, 0, 1]],
          values=[3, 2, -1, 3],
          dense_shape=[4, 2, 5])

      with tf.compat.v1.Session() as sess:
        with self.assertRaisesRegexp(exception_cls, error_string):
          sess.run(tf_utils._validate_and_get_dense_value_key_inputs(sparse1,
                                                                     sparse2),
                   feed_dict={sparse1: sparse_value1, sparse2: sparse_value2})

  def test_convert_ragged_indices(self):
    exception_cls = tf.errors.InvalidArgumentError
    error_string = 'Condition x == y did not hold element-wise:'
    ragged = tf.RaggedTensor.from_row_splits(
        values=tf.RaggedTensor.from_row_splits(
            values=np.array([1.2, 1., 1.2, 1.]), row_splits=np.array([0, 2,
                                                                      4])),
        row_splits=np.array([0, 1, 2]))
    dense = tf.constant(['a', 'b'])
    dense_result = tf.constant(['a', 'a', 'b', 'b'])
    x, key = tf_utils._validate_and_get_dense_value_key_inputs(ragged, ragged)
    self.assertAllEqual(self.evaluate(x), ragged.flat_values)
    self.assertAllEqual(self.evaluate(key), ragged.flat_values)

    x, key = tf_utils._validate_and_get_dense_value_key_inputs(ragged, dense)
    self.assertAllEqual(self.evaluate(x), ragged.flat_values)
    self.assertAllEqual(self.evaluate(key), dense_result)

    with tf.compat.v1.Graph().as_default():
      ragged1 = tf.compat.v1.ragged.placeholder(tf.float32, 2)
      ragged2 = tf.compat.v1.ragged.placeholder(tf.float32, 2)
      ragged_value1 = tf.compat.v1.ragged.RaggedTensorValue(
          values=tf.compat.v1.ragged.RaggedTensorValue(
              values=np.array([1.2, 1., 1.2, 1.]),
              row_splits=np.array([0, 2, 4])),
          row_splits=np.array([0, 2]))
      ragged_value2 = tf.compat.v1.ragged.RaggedTensorValue(
          values=tf.compat.v1.ragged.RaggedTensorValue(
              values=np.array([1.2, 1., 1.2, 1.]),
              row_splits=np.array([0, 3, 4])),
          row_splits=np.array([0, 2]))

      with tf.compat.v1.Session() as sess:
        with self.assertRaisesRegex(exception_cls, error_string):
          sess.run(
              tf_utils._validate_and_get_dense_value_key_inputs(
                  ragged1, ragged2),
              feed_dict={
                  ragged1: ragged_value1,
                  ragged2: ragged_value2
              })

  @test_case.named_parameters(
      dict(
          testcase_name='dense_tensor',
          key=['b', 'a', 'b'],
          key_vocab=['a', 'b'],
          reductions=([1, 2], [3, 4]),
          x=[5, 6, 7],
          expected_results=([2, 1, 2], [4, 3, 4])),
      dict(
          testcase_name='sparse_tensor_dense_key',
          key=['b', 'a', 'b'],
          key_vocab=['a', 'b'],
          reductions=([1, 2], [3, 4]),
          x=tf.compat.v1.SparseTensorValue(
              indices=[[0, 0], [1, 2], [2, 2], [2, 3]],
              values=[3, 2, -1, 3],
              dense_shape=[3, 5]),
          expected_results=([2, 1, 2, 2], [4, 3, 4, 4])),
      dict(
          testcase_name='sparse_tensor_sparse_key',
          key=tf.compat.v1.SparseTensorValue(
              indices=[[0, 0], [1, 2], [2, 2], [2, 3]],
              values=['b', 'a', 'b', 'b'],
              dense_shape=[3, 5]),
          key_vocab=['a', 'b'],
          reductions=([1, 2], [3, 4]),
          x=tf.compat.v1.SparseTensorValue(
              indices=[[0, 0], [1, 2], [2, 2], [2, 3]],
              values=[3, 2, -1, 3],
              dense_shape=[3, 5]),
          expected_results=([2, 1, 2, 2], [4, 3, 4, 4])),
      dict(
          testcase_name='ragged_tensor_dense_key',
          key=['a', 'b', 'a'],
          key_vocab=['a', 'b'],
          reductions=([1, 2], [3, 4]),
          x=tf.compat.v1.ragged.RaggedTensorValue(
              values=tf.compat.v1.ragged.RaggedTensorValue(
                  values=np.array([1.2, 1., 1.2, 1.]),
                  row_splits=np.array([0, 2, 4])),
              row_splits=np.array([0, 1, 2, 2])),
          expected_results=([1, 1, 2, 2], [3, 3, 4, 4])),
      dict(
          testcase_name='ragged_tensor_ragged_key',
          key=tf.compat.v1.ragged.RaggedTensorValue(
              values=tf.compat.v1.ragged.RaggedTensorValue(
                  values=np.array(['a', 'b', 'b', 'a']),
                  row_splits=np.array([0, 2, 4])),
              row_splits=np.array([0, 2])),
          key_vocab=['a', 'b'],
          reductions=([1, 2], [3, 4]),
          x=tf.compat.v1.ragged.RaggedTensorValue(
              values=tf.compat.v1.ragged.RaggedTensorValue(
                  values=np.array([1.2, 1., 1.2, 1.]),
                  row_splits=np.array([0, 2, 4])),
              row_splits=np.array([0, 2])),
          expected_results=([1, 2, 2, 1], [3, 4, 4, 3])),
      dict(
          testcase_name='missing_key',
          key=['b', 'a', 'c'],
          key_vocab=['z', 'a', 'b'],
          reductions=([-77, 1, 2], [-99, 3, 4]),
          x=[5, 6, 7],
          expected_results=([2, 1, 0], [4, 3, 0])),
  )
  def test_map_per_key_reductions(
      self, key, key_vocab, reductions, x, expected_results):
    with tf.compat.v1.Graph().as_default():
      key = _value_to_tensor(key)
      key_vocab = tf.constant(key_vocab)
      reductions = tuple([tf.constant(t) for t in reductions])
      x = _value_to_tensor(x)
      expected_results = tuple(tf.constant(t) for t in expected_results)
      results = tf_utils.map_per_key_reductions(reductions, key, key_vocab, x)
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.tables_initializer())
        output = sess.run(results)
        for result, expected_result in zip(output, expected_results):
          self.assertAllEqual(result, expected_result)

  @test_case.named_parameters(test_case.cross_with_function_handlers([
      dict(
          testcase_name='sparse_tensor',
          feature=tf.compat.v1.SparseTensorValue(
              indices=[[0, 0], [0, 1], [0, 2], [1, 0]],
              values=[1., 2., 3., 4.],
              dense_shape=[2, 5]),
          input_signature=[tf.SparseTensorSpec([None, 5], tf.float32)],
          ascii_protos=[
              'float_list { value: [1.0, 2.0, 3.0] }',
              'float_list { value: [4.0] }',
          ]),
      dict(
          testcase_name='dense_scalar_int',
          feature=[0, 1, 2],
          input_signature=[tf.TensorSpec([None], tf.int64)],
          ascii_protos=[
              'int64_list { value: [0] }',
              'int64_list { value: [1] }',
              'int64_list { value: [2] }',
          ]),
      dict(
          testcase_name='dense_scalar_float',
          feature=[0.5, 1.5, 2.5],
          input_signature=[tf.TensorSpec([None], tf.float32)],
          ascii_protos=[
              'float_list { value: [0.5] }',
              'float_list { value: [1.5] }',
              'float_list { value: [2.5] }',
          ]),
      dict(
          testcase_name='dense_scalar_string',
          feature=['hello', 'world'],
          input_signature=[tf.TensorSpec([None], tf.string)],
          ascii_protos=[
              'bytes_list { value: "hello" }',
              'bytes_list { value: "world" }',
          ]),
      dict(
          testcase_name='dense_vector_int',
          feature=[[0, 1], [2, 3]],
          input_signature=[tf.TensorSpec([None, 2], tf.int64)],
          ascii_protos=[
              'int64_list { value: [0, 1] }',
              'int64_list { value: [2, 3] }',
          ]),
      dict(
          testcase_name='dense_matrix_int',
          feature=[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
          input_signature=[tf.TensorSpec([None, 2, 2], tf.int64)],
          ascii_protos=[
              'int64_list { value: [0, 1, 2, 3] }',
              'int64_list { value: [4, 5, 6, 7] }',
          ]),
  ]))
  def test_serialize_feature(
      self, feature, input_signature, ascii_protos, function_handler):

    @function_handler(input_signature=input_signature)
    def _serialize_feature(feature):
      return tf_utils._serialize_feature(feature)

    serialized_features = _serialize_feature(feature)

    self.assertEqual(len(ascii_protos), len(serialized_features))
    for ascii_proto, serialized_feature in zip(ascii_protos,
                                               serialized_features):
      feature_proto = tf.train.Feature()
      feature_proto.ParseFromString(serialized_feature)
      self.assertProtoEquals(ascii_proto, feature_proto)

  @test_case.named_parameters(
      dict(
          testcase_name='multiple_features',
          examples={
              'my_value':
                  tf.compat.v1.SparseTensorValue(
                      indices=[[0, 0], [0, 1], [0, 2], [1, 0]],
                      values=[1., 2., 3., 4.],
                      dense_shape=[2, 5]),
              'my_other_value':
                  np.array([1, 2], np.int64),
          },
          ascii_protos=[
              """
               features {
                 feature {
                   key: "my_value"
                   value: { float_list { value: [1, 2, 3] } }
                 }
                 feature {
                   key: "my_other_value"
                    value: { int64_list { value: [1] } }
                 }
               }
               """, """
               features {
                 feature {
                   key: "my_value"
                   value: { float_list { value: [4] } }
                 }
                 feature {
                   key: "my_other_value"
                    value: { int64_list { value: [2] } }
                 }
               }
               """
          ]))
  def test_serialize_example(self, examples, ascii_protos):
    with tf.compat.v1.Graph().as_default():
      serialized_examples_tensor = tf_utils.serialize_example(examples)
      with tf.compat.v1.Session():
        serialized_examples = serialized_examples_tensor.eval()
        example_proto = tf.train.Example()
    self.assertEqual(len(serialized_examples), len(ascii_protos))
    for ascii_proto, serialized_example in zip(ascii_protos,
                                               serialized_examples):
      example_proto.ParseFromString(serialized_example)
      self.assertProtoEquals(ascii_proto, example_proto)

  def test_extend_reduced_batch_with_y_counts(self):
    initial_reduction = tf_utils.ReducedBatchWeightedCounts(
        unique_x=tf.constant(['foo', 'bar']),
        summed_weights_per_x=tf.constant([2.0, 4.0]),
        summed_positive_per_x_and_y=tf.constant([[1.0, 3.0], [1.0, 1.0]]),
        counts_per_x=tf.constant([2, 4], tf.int64))
    y = tf.constant([0, 1, 1, 1, 0, 1, 1], tf.int64)
    extended_batch = tf_utils.extend_reduced_batch_with_y_counts(
        initial_reduction, y)
    self.assertAllEqual(self.evaluate(extended_batch.unique_x),
                        np.array([b'foo', b'bar', b'global_y_count_sentinel']))
    self.assertAllClose(self.evaluate(extended_batch.summed_weights_per_x),
                        np.array([2.0, 4.0, 7.0]))
    self.assertAllClose(
        self.evaluate(extended_batch.summed_positive_per_x_and_y),
        np.array([[1.0, 3.0], [1.0, 1.0], [2.0, 5.0]]))
    self.assertAllClose(self.evaluate(extended_batch.counts_per_x),
                        np.array([2.0, 4.0, 7.0]))


class VocabTFUtilsTest(test_case.TransformTestCase):

  def setUp(self):
    if (not tf_utils.is_vocabulary_tfrecord_supported() and
        test_case.is_external_environment()):
      raise unittest.SkipTest('Test requires DatasetInitializer')
    super().setUp()

  def _write_tfrecords(self, path, bytes_records):
    with tf.io.TFRecordWriter(path, 'GZIP') as writer:
      for record in bytes_records:
        writer.write(record)

  def test_split_vocabulary_entries(self):
    x = tf.constant([b'1  a b ', b'2 c', b'3      . . .   '])
    keys, values = tf_utils._split_vocabulary_entries(x)
    expected_keys = [b' a b ', b'c', b'     . . .   ']
    expected_values = [b'1', b'2', b'3']
    self.assertAllEqual(self.evaluate(keys), np.array(expected_keys))
    self.assertAllEqual(self.evaluate(values), np.array(expected_values))

  @unittest.skipIf(
      test_case.is_tf_api_version_1(),
      'TFRecord vocabulary dataset tests require TF API version>1')
  def test_read_tfrecord_vocabulary_dataset(self):
    vocab_file = os.path.join(self.get_temp_dir(), 'vocab.tfrecord.gz')
    contents = [b'a', b'b', b'c']
    self._write_tfrecords(vocab_file, contents)
    self.AssertVocabularyContents(vocab_file, contents)

    ds = tf.data.TFRecordDataset(vocab_file, compression_type='GZIP')
    self.assertAllEqual(np.array(contents), list(ds.as_numpy_iterator()))

  @test_case.named_parameters([
      dict(
          testcase_name='_common',
          contents=[b'a', b'b', b' c '],
          expected=[(b'a', 0), (b'b', 1), (b' c ', 2)],
          key_dtype=tf.string,
          value_dtype=tf.int64,
          return_indicator_as_value=False,
          has_indicator=False),
      dict(
          testcase_name='_dtypes',
          contents=[b'17', b'42'],
          expected=[(17, 0.), (42, 1.)],
          key_dtype=tf.int64,
          value_dtype=tf.float32,
          return_indicator_as_value=False,
          has_indicator=False),
      dict(
          testcase_name='_drop_indicator',
          contents=[b'17 a', b'42 b'],
          expected=[(b'a', 0), (b'b', 1)],
          key_dtype=tf.string,
          value_dtype=tf.int64,
          return_indicator_as_value=False,
          has_indicator=True),
      dict(
          testcase_name='_indicator_value',
          contents=[b'17 a', b'42 b '],
          expected=[(b'a', 17), (b'b ', 42)],
          key_dtype=tf.string,
          value_dtype=tf.int64,
          return_indicator_as_value=True,
          has_indicator=True),
      dict(
          testcase_name='_indicator_value_dtype',
          contents=[b'17 a', b'42 b'],
          expected=[(b'a', 17.), (b'b', 42.)],
          key_dtype=tf.string,
          value_dtype=tf.float32,
          return_indicator_as_value=True,
          has_indicator=True),
  ])
  @unittest.skipIf(
      test_case.is_tf_api_version_1(),
      'TFRecord vocabulary dataset tests require TF API version>1')
  def test_make_tfrecord_vocabulary_dataset(self, contents, expected, key_dtype,
                                            value_dtype,
                                            return_indicator_as_value,
                                            has_indicator):
    vocab_file = os.path.join(self.get_temp_dir(), 'vocab.tfrecord.gz')
    self._write_tfrecords(vocab_file, contents)

    ds = tf_utils._make_tfrecord_vocabulary_dataset(
        vocab_file,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        return_indicator_as_value=return_indicator_as_value,
        has_indicator=has_indicator)

    def validate_dtypes(key, value):
      self.assertEqual(key.dtype, key_dtype)
      self.assertEqual(value.dtype, value_dtype)
      return key, value

    ds = ds.map(validate_dtypes)

    vocabulary = list(ds.as_numpy_iterator())
    self.assertAllEqual(expected, vocabulary)

  @test_case.named_parameters(test_case.FUNCTION_HANDLERS)
  def test_make_tfrecord_vocabulary_lookup_initializer(self, function_handler):
    vocab_file = os.path.join(self.get_temp_dir(), 'vocab.tfrecord.gz')
    contents = [b'%i' % idx for idx in range(1000)]
    self._write_tfrecords(vocab_file, contents)

    input_signature = [tf.TensorSpec(None, tf.string)]

    @function_handler(input_signature=input_signature)
    def lookup(x):
      initializer = tf_utils.make_tfrecord_vocabulary_lookup_initializer(
          vocab_file)
      table = tf.lookup.StaticHashTable(initializer, -1)
      return table.lookup(x)

    # make_tfrecord_vocabulary_lookup_initializer calls annotators.track_object
    # which expects to be invoked inside an object_tracker_scope.
    with annotators.object_tracker_scope(annotators.ObjectTracker()):
      self.assertEqual(lookup('5'), 5)
      self.assertEqual(lookup('1000'), -1)

  @test_case.named_parameters(
      test_case.cross_with_function_handlers(_CONSTRUCT_TABLE_PARAMETERS))
  def test_construct_and_lookup_table(self, asset_path_input_fn,
                                      function_handler):
    vocab_filename = os.path.join(self.get_temp_dir(), 'test.txt')
    vocab_data = [('a', '0'), ('b', '1'), ('c', '1'), ('d', '2'), ()]
    encoded_vocab = '\n'.join(['\t'.join(pair) for pair in vocab_data])
    with tf.io.gfile.GFile(vocab_filename, 'w') as writer:
      writer.write(encoded_vocab)

    @function_handler(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def foo(input_tensor):
      output_tensor, unused_table_size = tf_utils.construct_and_lookup_table(
          _construct_table, asset_path_input_fn(vocab_filename), input_tensor)
      return output_tensor

    expected_data = [0, 1, 1, 2, -1]
    output_tensor = foo(['a', 'b', 'c', 'd', 'e'])
    self.assertAllEqual(output_tensor, expected_data)


if __name__ == '__main__':
  # TODO(b/160294509): Remove this once this is enabled by default in all
  # supported TF versions.
  tf.compat.v1.enable_v2_tensorshape()
  test_case.main()
