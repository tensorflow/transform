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


import tensorflow as tf

from tensorflow_transform import test_case
from tensorflow_transform import tf_utils


class AnalyzersTest(test_case.TransformTestCase):

  def test_reduce_batch_count(self):
    x = tf.constant([[[1], [2]], [[1], [2]]])
    with tf.Session():
      self.assertAllEqual(
          tf_utils.reduce_batch_count(
              x, reduce_instance_dims=True).eval(), 4)

  def test_reduce_batch_count_elementwise(self):
    x = tf.constant([[[1], [2]], [[1], [2]]])
    with tf.Session():
      self.assertAllEqual(
          tf_utils.reduce_batch_count(
              x, reduce_instance_dims=False).eval(), [[2], [2]])

  def test_reduce_batch_count_sparse(self):
    x = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0]],
        values=[1., 2., 3., 4.],
        dense_shape=[2, 4, 1])
    with tf.Session():
      self.assertAllEqual(
          tf_utils.reduce_batch_count(
              x, reduce_instance_dims=True).eval(), 4)

  def test_reduce_batch_count_sparse_elementwise(self):
    x = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0]],
        values=[1., 2., 3., 4.],
        dense_shape=[2, 4, 1])
    with tf.Session():
      self.assertAllEqual(
          tf_utils.reduce_batch_count(
              x, reduce_instance_dims=False).eval(), [[1], [1], [2], [0]])

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
