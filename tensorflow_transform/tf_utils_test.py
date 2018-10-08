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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_transform import test_case
from tensorflow_transform import tf_utils


class AnalyzersTest(test_case.TransformTestCase):

  @test_case.named_parameters(
      dict(
          testcase_name='rank1_without_weights_or_labels',
          x=['a', 'b', 'a'],
          weights=None,
          labels=None,
          expected_results=[[b'a', b'b', b'a']]),
      dict(
          testcase_name='rank1_with_weights',
          x=['a', 'b', 'a'],
          weights=[1, 1, 2],
          labels=None,
          expected_results=[[b'a', b'b'], [3, 1]]),
      dict(
          testcase_name='rank1_with_labels',
          x=['a', 'b', 'a'],
          weights=None,
          labels=[0, 1, 1],
          expected_results=[[b'a', b'b'], [2, 1], [1, 1], [2, 1]]),
      dict(
          testcase_name='rank1_with_weights_and_labels',
          x=['a', 'b', 'a'],
          weights=[1, 1, 2],
          labels=[0, 1, 1],
          expected_results=[[b'a', b'b'], [3, 1], [2, 1], [2, 1]]),
      dict(
          testcase_name='rank2_without_weights_or_labels',
          x=[['a', 'b', 'a'], ['b', 'a', 'b']],
          weights=None,
          labels=None,
          expected_results=[[b'a', b'b', b'a', b'b', b'a', b'b']]),
      dict(
          testcase_name='rank2_with_weights',
          x=[['a', 'b', 'a'], ['b', 'a', 'b']],
          weights=[[1, 2, 1], [1, 2, 2]],
          labels=None,
          expected_results=[[b'a', b'b'], [4, 5]]),
      dict(
          testcase_name='rank2_with_labels',
          x=[['a', 'b', 'a'], ['b', 'a', 'b']],
          weights=None,
          labels=[[1, 0, 1], [1, 0, 0]],
          expected_results=[[b'a', b'b'], [3, 3], [2, 1], [3, 3]]),
      dict(
          testcase_name='rank2_with_weights_and_labels',
          x=[['a', 'b', 'a'], ['b', 'a', 'b']],
          weights=[[1, 2, 1], [1, 2, 2]],
          labels=[[1, 0, 1], [1, 0, 0]],
          expected_results=[[b'a', b'b'], [4, 5], [2, 1], [3, 3]]),
      dict(
          testcase_name='rank3_without_weights_or_labels',
          x=[[['a', 'b', 'a'], ['b', 'a', 'b']],
             [['a', 'b', 'a'], ['b', 'a', 'b']]],
          weights=None,
          labels=None,
          expected_results=[[
              b'a', b'b', b'a', b'b', b'a', b'b', b'a', b'b', b'a', b'b', b'a',
              b'b'
          ]]),
      dict(
          testcase_name='rank3_with_weights',
          x=[[['a', 'b', 'a'], ['b', 'a', 'b']],
             [['a', 'b', 'a'], ['b', 'a', 'b']]],
          weights=[[[1, 1, 2], [1, 2, 1]], [[1, 2, 1], [1, 2, 1]]],
          labels=None,
          expected_results=[[b'a', b'b'], [9, 7]]),
      dict(
          testcase_name='rank3_with_labels',
          x=[[['a', 'b', 'a'], ['b', 'a', 'b']],
             [['a', 'b', 'a'], ['b', 'a', 'b']]],
          weights=None,
          labels=[[[1, 1, 0], [1, 0, 1]], [[1, 0, 1], [1, 0, 1]]],
          expected_results=[[b'a', b'b'], [6, 6], [3, 5], [6, 6]]),
      dict(
          testcase_name='rank3_with_weights_and_labels',
          x=[[['a', 'b', 'a'], ['b', 'a', 'b']],
             [['a', 'b', 'a'], ['b', 'a', 'b']]],
          weights=[[[1, 1, 2], [1, 2, 1]], [[1, 2, 1], [1, 2, 1]]],
          labels=[[[1, 1, 0], [1, 0, 1]], [[1, 0, 1], [1, 0, 1]]],
          expected_results=[[b'a', b'b'], [9, 7], [3, 5], [6, 6]]),
  )
  def test_reduce_batch_vocabulary(self, x, weights, labels, expected_results):
    x = tf.constant(x)
    vocab_ordering_type = (
        tf_utils.VocabOrderingType.FREQUENCY)
    if weights is not None:
      weights = tf.constant(weights)
      vocab_ordering_type = (
          tf_utils.VocabOrderingType.WEIGHTED_FREQUENCY)
    if labels is not None:
      labels = tf.constant(labels, dtype=tf.int64)
      vocab_ordering_type = (
          tf_utils.VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION)

    uniques, sum_weights, sum_positive, counts = (
        tf_utils.reduce_batch_vocabulary(x, vocab_ordering_type, weights,
                                         labels))
    with tf.Session() as sess:
      results = sess.run([
          a for a in [uniques, sum_weights, sum_positive, counts]
          if a is not None
      ])
      for result, expected in zip(results, expected_results):
        self.assertAllEqual(result, np.array(expected))

  def test_reduce_vocabulary_inputs(self):
    with tf.Session() as sess:
      x = tf.constant(['yes', 'no', 'yes'])
      weights = tf.constant([1, 2, 3])
      labels = tf.constant([1, 1, 0], tf.int64)
      results = sess.run(
          tf_utils._reduce_vocabulary_inputs(x, weights, labels))
      expected_results = [[b'yes', b'no'], [4, 2], [1, 2], [2, 1]]
      for result, expected in zip(results, expected_results):
        self.assertAllEqual(result, np.array(expected))

  @test_case.parameters(
      ([[1], [2]], [[1], [2], [3]], None, None, tf.errors.InvalidArgumentError,
       'Condition x == y did not hold element-wise:'),
      ([[1], [2], [3]], [[1], [2], [3]], [None, None], [None], ValueError,
       r'Shapes \(\?, \?\) and \(\?,\) are incompatible'),
  )
  def test_same_shape_exceptions(self, x_input, y_input, x_shape, y_shape,
                                 exception_cls, error_string):
    x = tf.placeholder(tf.int32, shape=x_shape)
    y = tf.placeholder(tf.int32, shape=y_shape)
    with tf.Session() as sess:
      with self.assertRaisesRegexp(exception_cls, error_string):
        sess.run(tf_utils.assert_same_shape(x, y), {x: x_input, y: y_input})

  def test_same_shape(self):
    with tf.Session() as sess:
      input_list = [[1], [2], [3]]
      x = tf.placeholder(tf.int32, shape=None)
      y = tf.placeholder(tf.int32, shape=None)
      x_return = sess.run(
          tf_utils.assert_same_shape(x, y), {
              x: input_list,
              y: input_list
          })
      self.assertAllEqual(x_return, input_list)

  def test_reduce_batch_count(self):
    x = tf.constant([[[1], [2]], [[1], [2]]])
    with tf.Session():
      self.assertAllEqual(
          tf_utils.reduce_batch_count(x, reduce_instance_dims=True).eval(), 4)

  def test_reduce_batch_count_elementwise(self):
    x = tf.constant([[[1], [2]], [[1], [2]]])
    with tf.Session():
      self.assertAllEqual(
          tf_utils.reduce_batch_count(x, reduce_instance_dims=False).eval(),
          [[2], [2]])

  def test_reduce_batch_count_sparse(self):
    x = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0]],
        values=[1., 2., 3., 4.],
        dense_shape=[2, 4, 1])
    with tf.Session():
      self.assertAllEqual(
          tf_utils.reduce_batch_count(x, reduce_instance_dims=True).eval(), 4)

  def test_reduce_batch_count_sparse_elementwise(self):
    x = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0]],
        values=[1., 2., 3., 4.],
        dense_shape=[2, 4, 1])
    with tf.Session():
      self.assertAllEqual(
          tf_utils.reduce_batch_count(x, reduce_instance_dims=False).eval(),
          [[1], [1], [2], [0]])

  def test_reduce_batch_count_mean_and_var(self):
    x = tf.constant([[[1], [2]], [[3], [4]]], dtype=tf.float32)
    count, mean, var = tf_utils.reduce_batch_count_mean_and_var(
        x, reduce_instance_dims=True)
    with tf.Session():
      self.assertAllEqual(count.eval(), 4)
      self.assertAllEqual(mean.eval(), 2.5)
      self.assertAllEqual(var.eval(), 1.25)

  def test_reduce_batch_count_mean_and_var_elementwise(self):
    x = tf.constant([[[1], [2]], [[3], [4]]], dtype=tf.float32)
    count, mean, var = tf_utils.reduce_batch_count_mean_and_var(
        x, reduce_instance_dims=False)
    with tf.Session():
      self.assertAllEqual(count.eval(), [[2.], [2.]])
      self.assertAllEqual(mean.eval(), [[2.], [3.]])
      self.assertAllEqual(var.eval(), [[1.], [1.]])

  def test_reduce_batch_count_mean_and_var_sparse(self):
    x = tf.SparseTensor(
        indices=[[0, 0], [0, 2], [1, 1], [1, 2]],
        values=[1., 2., 3., 4.],
        dense_shape=[2, 4])
    count, mean, var = tf_utils.reduce_batch_count_mean_and_var(
        x, reduce_instance_dims=True)
    with tf.Session():
      self.assertAllEqual(count.eval(), 4)
      self.assertAllEqual(mean.eval(), 2.5)
      self.assertAllEqual(var.eval(), 1.25)

  def test_reduce_batch_count_mean_and_var_sparse_elementwise(self):
    x = tf.SparseTensor(
        indices=[[0, 0], [0, 2], [1, 1], [1, 2]],
        values=[1., 2., 3., 4.],
        dense_shape=[2, 5])
    count, mean, var = tf_utils.reduce_batch_count_mean_and_var(
        x, reduce_instance_dims=False)
    nan = float('nan')
    inf = float('inf')
    with tf.Session():
      self.assertAllEqual(count.eval(), [1.0, 1.0, 2.0, 0.0, 0.0])
      self.assertAllEqual(mean.eval(), [1.0, 3.0, 3.0, nan, nan])
      self.assertAllEqual(var.eval(), [2.0, 2.0, 1.0, inf, inf])


if __name__ == '__main__':
  test_case.main()
