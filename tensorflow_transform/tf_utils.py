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


class VocabOrderingType(object):
  FREQUENCY = 1
  WEIGHTED_FREQUENCY = 2
  WEIGHTED_MUTUAL_INFORMATION = 3


def reduce_batch_vocabulary(x, vocab_ordering_type,
                            weights=None, labels=None):
  """Performs batch-wise reduction of vocabulary.

  Args:
    x: Input `Tensor` to compute a vocabulary over.
    vocab_ordering_type: VocabOrderingType enum.
    weights: (Optional) Weights input `Tensor`.
    labels: (Optional) Binary labels input `Tensor`.


  Returns:
    A tuple of 3 `Tensor`s:
      * unique values
      * total weights sum for unique values when labels and or weights is
        provided, otherwise, None.
      * sum of positive weights for unique values when labels is provided,
        otherwise, None.
  """
  if vocab_ordering_type == VocabOrderingType.FREQUENCY:
    x = tf.reshape(x, [-1])
    return (x, None, None, None)

  if vocab_ordering_type == VocabOrderingType.WEIGHTED_MUTUAL_INFORMATION:
    tf.assert_type(labels, tf.int64)
    x = assert_same_shape(x, labels)
    if weights is None:
      weights = tf.ones_like(labels)
    labels = tf.reshape(labels, [-1])
  x = assert_same_shape(x, weights)
  weights = tf.reshape(weights, [-1])
  x = tf.reshape(x, [-1])
  return _reduce_vocabulary_inputs(x, weights, labels)


def _reduce_vocabulary_inputs(x, weights, labels=None):
  """Reduces vocabulary inputs.

  Args:
    x: Input `Tensor` for vocabulary analyzer.
    weights: Weights `Tensor` for vocabulary analyzer.
    labels: (optional) Binary Labels `Tensor` for vocabulary analyzer.

  Returns:
    A tuple of 3 `Tensor`s:
      * unique values
      * total weights sum for unique values
      * sum of positive weights for unique values when labels is provided,
        otherwise, None.
  """
  unique = tf.unique_with_counts(x, out_idx=tf.int64)

  summed_weights = tf.unsorted_segment_sum(weights, unique.idx,
                                           tf.size(unique.y))
  if labels is None:
    summed_positive_weights = None
    counts = None
  else:
    less_assert = tf.Assert(tf.less_equal(tf.reduce_max(labels), 1), [labels])
    greater_assert = tf.Assert(tf.greater_equal(
        tf.reduce_min(labels), 0), [labels])
    with tf.control_dependencies([less_assert, greater_assert]):
      labels = tf.identity(labels)
    positive_weights = (
        tf.cast(labels, tf.float32) * tf.cast(weights, tf.float32))
    summed_positive_weights = tf.unsorted_segment_sum(
        positive_weights, unique.idx, tf.size(unique.y))
    counts = unique.count

  return (unique.y, summed_weights, summed_positive_weights, counts)


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
