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

import collections

# GOOGLE-INITIALIZATION
import tensorflow as tf

from tensorflow.contrib.proto.python.ops import encode_proto_op

_FLOATING_NAN = float('nan')
# Global sentinels used to keep track of the total counts of y
GLOBAL_Y_COUNT_SENTINEL_STRING = b'global_y_count_sentinel'
GLOBAL_Y_COUNT_SENTINEL_INT = tf.int64.limits[0]

ReducedBatchWeightedCounts = collections.namedtuple('ReducedBatchCounts', [
    'unique_x', 'summed_weights_per_x', 'summed_positive_per_x_and_y',
    'counts_per_x'
])


def reduce_batch_weighted_counts(x, weights=None):
  """Performs batch-wise reduction to produce (possibly weighted) counts.

  Args:
    x: Input `Tensor`.
    weights: (Optional) Weights input `Tensor`.

  Returns:
    a named tuple of...
      The unique values in x
      The sum of the weights for each unique value in x if weights are provided,
        else None
  """
  if isinstance(x, tf.SparseTensor):
    x = x.values
  if weights is None:
    # TODO(b/112916494): Always do batch wise reduction once possible.

    return ReducedBatchWeightedCounts(tf.reshape(x, [-1]), None, None, None)
  # TODO(b/134075780): Revisit expected weights shape when input is sparse.
  x = assert_same_shape(x, weights)
  weights = tf.reshape(weights, [-1])
  x = tf.reshape(x, [-1])
  unique_x_values, unique_idx, _ = tf.unique_with_counts(x, out_idx=tf.int64)
  summed_weights_per_x = tf.math.unsorted_segment_sum(
      weights, unique_idx, tf.size(input=unique_x_values))
  return ReducedBatchWeightedCounts(unique_x_values, summed_weights_per_x, None,
                                    None)


def reduce_batch_weighted_cooccurrences(x_input,
                                        y_input,
                                        weights_input=None,
                                        extend_with_sentinel_counts=True):
  """Performs batch-wise reduction to produce weighted co-occurrences.

  Somputes the weighted co-occurrence of each feature value in x, for each value
  in the range [0, max(y)). If extend_with_sentinel_counts is true, the return
  value will include an additional sentinel token (not in the true vocabulary)
  that is used to accumulate the global distribution of y values.

  Args:
    x_input: Input `Tensor` or `SparseTensor`.
    y_input: Integer `Tensor` or `SparseTensor` with which to compute the
      co-occurrence with x_input.
    weights_input: (Optional) Weights input `Tensor`.
    extend_with_sentinel_counts: If True, the reduced batch will be extended
      a sentinel value that accumlate the total distribution of y values. Should
      be True except when called recursively with the sentinel value as input.

  Returns:
    a namedtuple of...
    unique_x_values: the unique values in x
    summed_weights_per_x: sum of the weights for each unique value in x
    summed_positive_per_x_and_y: If tensor y is provided, the sum of
      positive weights for each unique y value, for each unique value in x.
      If y tensor is not provided, value is None.
    counts_per_x: if y is provided, counts of each of the unique values in x,
      otherwise, None.
  """
  tf.compat.v1.assert_type(y_input, tf.int64)
  # TODO(b/134075780): Revisit expected weights shape when input is sparse.
  if isinstance(x_input, tf.SparseTensor):
    batch_indices = x_input.indices[:, 0]
    y = tf.gather(y_input, batch_indices)
    x = x_input.values
  else:
    y = y_input
    x = x_input
  if weights_input is None:
    weights = tf.ones_like(x, dtype=tf.float32)
  else:
    x = assert_same_shape(x, weights_input)
    weights = weights_input
  y = _broadcast_to_x_shape(x, y)
  x = assert_same_shape(x, y)
  x = tf.reshape(x, [-1])
  y = tf.reshape(y, [-1])
  weights = tf.reshape(weights, [-1])

  unique_x_values, unique_idx, unique_count = tf.unique_with_counts(
      x, out_idx=tf.int64)

  summed_weights_per_x = tf.math.unsorted_segment_sum(
      weights, unique_idx, tf.size(input=unique_x_values))
  # For each feature value in x, computed the weighted sum positive for each
  # unique value in y.

  max_y_value = tf.cast(tf.reduce_max(input_tensor=y), tf.int64)
  max_x_idx = tf.cast(tf.size(unique_x_values), tf.int64)
  dummy_index = (max_y_value + 1) * unique_idx + y
  summed_positive_per_x_and_y = tf.cast(
      tf.math.unsorted_segment_sum(weights, dummy_index,
                                   max_x_idx * (max_y_value + 1)),
      dtype=tf.float32)
  summed_positive_per_x_and_y = tf.reshape(summed_positive_per_x_and_y,
                                           [max_x_idx, max_y_value + 1])

  reduced_batch = ReducedBatchWeightedCounts(
      unique_x=unique_x_values,
      summed_weights_per_x=summed_weights_per_x,
      summed_positive_per_x_and_y=summed_positive_per_x_and_y,
      counts_per_x=unique_count)
  # Add a sentinel token tracking the full distribution of y values.
  if extend_with_sentinel_counts:
    reduced_batch = extend_reduced_batch_with_y_counts(reduced_batch, y_input,
                                                       weights_input)
  return reduced_batch


def extend_reduced_batch_with_y_counts(reduced_batch, y, weights=None):
  """Extend the ReducedBatchWeightedCounts with global counts for y.

  This is used to maintain an accurate count of global frequencies of each value
  in y. When x is multivalent, the sum over the summed_positive_per_x_and_y
  will over-count the occurrence of y. To keep track of the true distribution
  of y values, we add a sentinel value that tracks the global counts of each
  distinct value in y. This is useful for computing the mutual information
  between values in x and y.

  Args:
    reduced_batch: A ReducedBatchWeightedCounts instance.
    y: A `Tensor` representing a batch of y values.
    weights: Optional `Tensor` representing a batch of weight values.

  Returns:
    A new ReducedBatchWeightedCounts instance with sentinel values appended.
  """
  # Create a dummy sentinel token that is present in every record.
  if reduced_batch.unique_x.dtype.is_integer:
    sentinel_values = tf.cast(
        tf.fill(tf.shape(y), GLOBAL_Y_COUNT_SENTINEL_INT), tf.int64)
  else:
    sentinel_values = tf.fill(tf.shape(y), GLOBAL_Y_COUNT_SENTINEL_STRING)
  # Computing the batch reduction over this sentinel token will reduce to a
  # single sentinel value in sentinel_batch.unique_x, with the
  # summed_positive_per_x_and_y thus capturing the total summed positive per
  # value in y.
  sentinel_batch = reduce_batch_weighted_cooccurrences(
      sentinel_values, y, weights, extend_with_sentinel_counts=False)

  # Concatenate the sentinel counts with the existing reduced batch.
  return ReducedBatchWeightedCounts(
      unique_x=tf.concat([reduced_batch.unique_x, sentinel_batch.unique_x],
                         axis=0),
      summed_weights_per_x=tf.concat([
          reduced_batch.summed_weights_per_x,
          sentinel_batch.summed_weights_per_x
      ],
                                     axis=0),
      summed_positive_per_x_and_y=tf.concat([
          reduced_batch.summed_positive_per_x_and_y,
          sentinel_batch.summed_positive_per_x_and_y
      ],
                                            axis=0),
      counts_per_x=tf.concat(
          [reduced_batch.counts_per_x, sentinel_batch.counts_per_x], axis=0))


def _broadcast_to_x_shape(x, y):
  """Broadcasts y to same shape as x as needed.

  Args:
    x: An input feature.
    y: A feature that is either the same shape as x or has the same outer
      dimensions as x. If the latter, y is broadcast to the same shape as x.
  """
  # The batch dimension of x and y must be the same, and y must be 1D.
  x_shape = tf.shape(input=x)
  y_shape = tf.shape(input=y)
  assert_eq = tf.compat.v1.assert_equal(x_shape[0], y_shape[0])
  with tf.control_dependencies([assert_eq]):
    y = tf.identity(y)
  rank_delta = tf.rank(x) - tf.rank(y)
  target_shape = tf.concat(
      [tf.shape(y), tf.ones(rank_delta, dtype=tf.int32)], axis=0)
  matched_rank = tf.reshape(y, target_shape)
  return tf.broadcast_to(matched_rank, x_shape)


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
  assert_eq = tf.compat.v1.assert_equal(tf.shape(input=x), tf.shape(input=y))
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
      return tf.sparse.reduce_sum(ones_like, axis=0)

  if reduce_instance_dims:
    return tf.size(input=x)

  # Fill a tensor shaped like x except batch_size=1 with batch_size.
  x_shape = tf.shape(input=x)
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
      tf.sparse.reduce_sum if isinstance(x, tf.SparseTensor) else tf.reduce_sum)
  axis = None if reduce_instance_dims else 0

  x_mean = reduce_sum_fn(x, axis=axis) / x_count

  if isinstance(x, tf.SparseTensor):
    # This means reduce_instance_dims=False.
    # TODO(b/112656428): Support SparseTensors with rank other than 2.
    if x.get_shape().ndims != 2:
      raise NotImplementedError(
          'Mean and var only support SparseTensors with rank 2')

    mean_values = tf.gather(x_mean, x.indices[:, 1])
    x_minus_mean = x.values - mean_values
  else:
    x_minus_mean = x - x_mean
  x_variance = tf.reduce_sum(
      input_tensor=tf.square(x_minus_mean), axis=axis) / x_count

  return (x_count, x_mean, x_variance)


def reduce_batch_count_mean_and_var_per_key(x, key, reduce_instance_dims):
  """Computes per-key element count, mean and var for the given tensor.

  Args:
    x: A `Tensor` or `SparseTensor`.
    key: A `Tensor` or `SparseTensor`, must be either sparse and correspond
        one-to-one with x, or both x and key are dense.
    reduce_instance_dims: A bool, if True - collapses the batch and instance
        dimensions to arrive at a single scalar output. Otherwise, only
        collapses the batch dimension and outputs a `Tensor` of the same shape
        as the input. Not supported for `SparseTensor`s.

  Returns:
    A 4-tuple containing the `Tensor`s (key_vocab, count, mean, var).
  """

  if isinstance(x, tf.SparseTensor):
    if not reduce_instance_dims:
      raise NotImplementedError(
          'Mean and var per key only support reduced dims for SparseTensors')
    if not isinstance(key, tf.SparseTensor):
      # TODO(b/132071166): Allow for sparse values, dense keys.
      raise NotImplementedError(
          'Mean and var per key require sparse key if x is sparse')

    # TODO(b/131920907): Validate that x and key have the same indices.
    key = key.values
    x = x.values

  unique = tf.unique_with_counts(key, out_idx=tf.int64)
  x_count = unique.count
  x_count = tf.cast(x_count, x.dtype)
  if not reduce_instance_dims:
    x_count = tf.tile(tf.expand_dims(x_count, axis=-1), [1, x.shape[1]])

  if reduce_instance_dims:
    sums = tf.reduce_sum(x, axis=1) if x.get_shape().ndims != 1 else x
    sums = tf.math.unsorted_segment_sum(sums, unique.idx, tf.size(unique.y))
  else:
    sums = tf.math.unsorted_segment_sum(x, unique.idx, tf.size(unique.y))

  means = tf.cast(sums, x.dtype) / x_count
  sum_sqs = tf.math.unsorted_segment_sum(tf.square(x),
                                         unique.idx,
                                         tf.size(input=unique.y))
  if sum_sqs.get_shape().ndims != 1 and reduce_instance_dims:
    sum_sqs = tf.reduce_sum(sum_sqs, axis=1)

  variances = sum_sqs / x_count - tf.square(means)

  return unique.y, x_count, means, variances


# Code for serializing and example proto


_DEFAULT_VALUE_BY_DTYPE = {
    tf.string: '',
    tf.float32: 0,
    tf.int64: 0
}


def _encode_proto(values_dict, message_type):
  """A wrapper around encode_proto_op.encode_proto."""
  field_names = []
  sizes = []
  values = []
  for field_name, value in sorted(values_dict.items(), key=lambda x: x[0]):
    if isinstance(value, tf.SparseTensor):
      size = tf.sparse.reduce_sum(
          tf.SparseTensor(value.indices,
                          tf.ones_like(value.values, dtype=tf.int32),
                          value.dense_shape),
          axis=1)
      value = tf.sparse.to_dense(value, _DEFAULT_VALUE_BY_DTYPE[value.dtype])
    else:
      value = tf.reshape(value, [tf.shape(input=value)[0], -1])
      size = tf.fill((tf.shape(input=value)[0],), tf.shape(input=value)[1])
    field_names.append(field_name)
    values.append(value)
    sizes.append(size)

  sizes = tf.stack(sizes, axis=1)
  return encode_proto_op.encode_proto(sizes, values, field_names, message_type)


def _serialize_feature(values):
  """Serialize a Tensor or SparseTensor as `Feature` protos.

  `values` should be a Tensor of rank >=1 or SparseTensor of rank 2.  We will
  refer to the size of the first dimension as batch_size.

  This function encodes each row of the `Tensor` as a list of values (flattening
  the other dimensions) and each row of the `SparseTensor` as a list of values,
  where the indices within each row are ignored and assumed to be 0, 1, ....

  Args:
    values: A `Tensor` or `SparseTensor`.

  Returns:
    A tensor of shape (batch_size,) and type `tf.string` where each element is
        a serialized `Feature` proto.

  Raises:
    ValueError: If the dtype is of `values` is not `tf.string`, `tf.float32`
        or `tf.int64`.
  """
  values = tf.compat.v1.convert_to_tensor_or_sparse_tensor(values)
  if values.dtype == tf.string:
    values_dict = {
        'bytes_list': _encode_proto({'value': values}, 'tensorflow.BytesList')
    }
  elif values.dtype == tf.float32:
    values_dict = {
        'float_list': _encode_proto({'value': values}, 'tensorflow.FloatList')
    }
  elif values.dtype == tf.int64:
    values_dict = {
        'int64_list': _encode_proto({'value': values}, 'tensorflow.Int64List')
    }
  else:
    raise ValueError('Cannot encode values of dtype {}'.format(values.dtype))
  return _encode_proto(values_dict, 'tensorflow.Feature')


def serialize_example(features):
  """Serialized a dict of `Tensor` or `SparseTensor`s as example protos.

  `features` should be a dict where each value is a Tensor of rank >=1 or
  SparseTensor of rank 2.  The sizes of the first dimension of each value should
  be the same, and we refer to this size as batch_size.

  Args:
    features: A dictionary whose values are `Tensor`s or `SparseTensor`s.

  Returns:
    A tensor of shape (batch_size,) and type `tf.string` where each element is
        a serialized `Example` proto.
  """
  features_dict = []
  for key, value in sorted(features.items(), key=lambda x: x[0]):
    serialized_value = _serialize_feature(value)
    features_dict.append(
        _encode_proto({
            'key': tf.fill((tf.shape(input=serialized_value)[0],), key),
            'value': serialized_value,
        }, 'tensorflow.Features.FeatureEntry'))
  features_dict = tf.stack(features_dict, axis=1)
  features = _encode_proto({'feature': features_dict}, 'tensorflow.Features')
  return _encode_proto({'features': features}, 'tensorflow.Example')


def _sparse_minus_reduce_min_and_reduce_max(x):
  """Computes the -min and max of a SparseTensor x.

  It differs from sparse_reduce_max in that sparse_reduce_max returns 0 when all
  elements are missing along axis 0.
  We replace the 0 with NaN when x's dtype is float and dtype.min+1 when it's
  int.

  Args:
    x: A `SparseTensor`.

  Returns:
    Two `Tensors' which are the -min and max.

  Raises:
    TypeError: If the type of `x` is not supported.
  """
  if not isinstance(x, tf.SparseTensor):
    raise TypeError('Expected a SparseTensor, but got %r' % x)
  minus_x = tf.SparseTensor(
      indices=x.indices, values=0 - x.values, dense_shape=x.dense_shape)
  x_count = reduce_batch_count(x, reduce_instance_dims=False)
  batch_has_no_values = tf.equal(x_count, tf.constant(0, dtype=tf.int64))
  x_batch_max = tf.sparse.reduce_max(sp_input=x, axis=0)
  x_batch_minus_min = tf.sparse.reduce_max(sp_input=minus_x, axis=0)

  if x.dtype.is_floating:
    missing_value = tf.constant(_FLOATING_NAN, x.dtype)
  else:
    missing_value = tf.constant(x.dtype.min + 1, x.dtype)

  x_batch_max = tf.where(batch_has_no_values,
                         tf.fill(tf.shape(input=x_batch_max), missing_value),
                         x_batch_max)
  x_batch_minus_min = tf.where(
      batch_has_no_values,
      tf.fill(tf.shape(input=x_batch_minus_min), missing_value),
      x_batch_minus_min)
  return x_batch_minus_min, x_batch_max


def _inf_to_nan(tensor, output_dtype):
  if tensor.dtype.is_floating:
    nan = tf.constant(_FLOATING_NAN, output_dtype)
    return tf.where(tf.math.is_inf(tensor), tensor + nan, tensor)
  return tensor


def reduce_batch_minus_min_and_max(x, reduce_instance_dims):
  """Computes the -min and max of a tensor x.

  Args:
    x: A `tf.Tensor`.
    reduce_instance_dims: A bool indicating whether this should collapse the
      batch and instance dimensions to arrive at a single scalar output, or only
      collapse the batch dimension and outputs a vector of the same shape as the
      input.

  Returns:
    The computed `tf.Tensor`s (batch -min, batch max) pair.
  """
  output_dtype = x.dtype

  if x.dtype == tf.uint8 or x.dtype == tf.uint16:
    x = tf.cast(x, tf.int32)

  elif x.dtype == tf.uint32 or x.dtype == tf.uint64:
    raise TypeError('Tensor type %r is not supported' % x.dtype)

  if reduce_instance_dims:
    if isinstance(x, tf.SparseTensor):
      x = x.values

    x_batch_max = tf.reduce_max(input_tensor=x)
    x_batch_minus_min = tf.reduce_max(input_tensor=tf.zeros_like(x) - x)
    x_batch_minus_min = assert_same_shape(x_batch_minus_min, x_batch_max)
  elif isinstance(x, tf.SparseTensor):
    x_batch_minus_min, x_batch_max = (
        _sparse_minus_reduce_min_and_reduce_max(x))
  else:
    x_batch_max = tf.reduce_max(input_tensor=x, axis=0)
    x_batch_minus_min = tf.reduce_max(input_tensor=0 - x, axis=0)

  # TODO(b/112309021): Remove workaround once tf.reduce_max of a tensor of all
  # NaNs produces -inf.
  return (_inf_to_nan(x_batch_minus_min, output_dtype),
          _inf_to_nan(x_batch_max, output_dtype))


def reduce_batch_minus_min_and_max_per_key(x, key):
  """Computes the -min and max of a tensor x.

  Args:
    x: A `tf.Tensor`.
    key: A `Tensor` of rank 1.

  Returns:
    A 3-tuple containing the `Tensor`s (key_vocab, min_per_key, max_per_key).
  """
  output_dtype = x.dtype

  if x.dtype == tf.uint8 or x.dtype == tf.uint16:
    x = tf.cast(x, tf.int32)

  elif x.dtype == tf.uint32 or x.dtype == tf.uint64:
    raise TypeError('Tensor type %r is not supported' % x.dtype)

  if isinstance(x, tf.SparseTensor):
    key = tf.gather(key, x.indices[:, 0])
    x = x.values

  def get_batch_max_per_key(tensor, key_uniques, dtype):  # pylint: disable=missing-docstring
    if tensor.get_shape().ndims < 2:
      row_maxes = tensor
    else:
      row_maxes = tf.reduce_max(
          tensor, axis=tf.range(1, tensor.get_shape().ndims))
    batch_max = tf.math.unsorted_segment_max(
        row_maxes, key_uniques.idx, tf.size(input=key_uniques.y))

    # TODO(b/112309021): Remove workaround once tf.reduce_max of a tensor of all
    # NaNs produces -inf.
    return _inf_to_nan(batch_max, dtype)

  unique = tf.unique_with_counts(key, out_idx=tf.int64)
  x_batch_maxes = get_batch_max_per_key(x, unique, output_dtype)
  x_batch_minus_mins = get_batch_max_per_key(-x, unique, output_dtype)

  x_batch_minus_mins = assert_same_shape(x_batch_minus_mins, x_batch_maxes)

  return (unique.y, x_batch_minus_mins, x_batch_maxes)
