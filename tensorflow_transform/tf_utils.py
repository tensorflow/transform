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
"""TF utils for computing information over given data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def assert_same_shape(x, y):
  """Asserts two tensors have the same dynamic and static shape.

  Args:
    x: A `Tensor`.
    y: A `Tensor`

  Returns:
    The element `x`, the result must be used in order to ensure that the dynamic
    check is executed.
  """
  x.shape.assert_is_compatible_with(y.shape)
  assert_eq = tf.assert_equal(tf.shape(x), tf.shape(y))
  with tf.control_dependencies([assert_eq]):
    return tf.identity(x)


def reduce_batch_count(x, reduce_instance_dims):
  """Counts elements in the given tensor.

  Args:
    x: A `Tensor` or `SparseTensor`.
    reduce_instance_dims: A bool, if True - collapses the batch and instance
        dimensions to arrive at a single scalar output. Otherwise, only
        collapses the batch dimension and outputs a `Tensor` of the same shape
        as the input.

  Returns:
    The element count of `x`. The result is either a scalar if
    reduce_instance_dims is True, otherwise a `Tensor` of the same shape as `x`.
  """
  if isinstance(x, tf.SparseTensor):
    if reduce_instance_dims:
      x = x.values
    else:
      ones_like = tf.SparseTensor(
          indices=x.indices,
          values=tf.ones_like(x.values, tf.int64),
          dense_shape=x.dense_shape)
      return tf.sparse_reduce_sum(ones_like, axis=0)

  if reduce_instance_dims:
    return tf.size(x)

  # Fill a tensor shaped like x except batch_size=1 with batch_size.
  x_shape = tf.shape(x)
  return tf.fill(x_shape[1:], x_shape[0])


def reduce_batch_count_mean_and_var(x, reduce_instance_dims):
  """Computes element count, mean and var for the given tensor.

  Args:
    x: A `Tensor` or `SparseTensor`.
    reduce_instance_dims: A bool, if True - collapses the batch and instance
        dimensions to arrive at a single scalar output. Otherwise, only
        collapses the batch dimension and outputs a `Tensor` of the same shape
        as the input.

  Returns:
    A 3-tuple containing the `Tensor`s (count, mean, var).
  """
  if isinstance(x, tf.SparseTensor) and reduce_instance_dims:
    x = x.values

  x_count = tf.cast(reduce_batch_count(x, reduce_instance_dims), x.dtype)
  if not reduce_instance_dims:
    # Remove the batch dimension.
    # x_count = tf.squeeze(x_count, axis=0)
    pass

  reduce_sum_fn = (
      tf.sparse_reduce_sum if isinstance(x, tf.SparseTensor) else tf.reduce_sum)
  axis = None if reduce_instance_dims else 0

  x_mean = reduce_sum_fn(x, axis=axis) / x_count

  if isinstance(x, tf.SparseTensor):
    # This means reduce_instance_dims=False.
    if x.get_shape().ndims != 2:
      raise NotImplementedError(
          'Mean and var only support SparseTensors with rank 2')

    mean_values = tf.gather(x_mean, x.indices[:, 1])
    x_minus_mean = x.values - mean_values
  else:
    x_minus_mean = x - x_mean
  x_variance = tf.reduce_sum(tf.square(x_minus_mean), axis=axis) / x_count

  return (x_count, x_mean, x_variance)
