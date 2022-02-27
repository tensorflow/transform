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

import contextlib
import enum
from typing import Callable, Optional, Tuple, Union

import tensorflow as tf
from tensorflow_transform import annotators
from tensorflow_transform import common_types
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.util import object_identity
# pylint: enable=g-direct-tensorflow-import

_AssetFileType = Union[tf.Tensor, str]

_FLOATING_NAN = float('nan')
# Global sentinels used to keep track of the total counts of y
GLOBAL_Y_COUNT_SENTINEL_STRING = b'global_y_count_sentinel'
GLOBAL_Y_COUNT_SENTINEL_INT = tf.int64.limits[1]

# Key for graph collection containing tuple of a key to the eager tensor
# representing asset path and the graph tensor tracking the analyzer in
# `analyzer_nodes.TENSOR_REPLACEMENTS`.
_ASSET_REPLACEMENTS = 'tft_asset_replacements'

ReducedBatchWeightedCounts = tfx_namedtuple.namedtuple('ReducedBatchCounts', [
    'unique_x', 'summed_weights_per_x', 'summed_positive_per_x_and_y',
    'counts_per_x'
])

_CompositeTensorRef = tfx_namedtuple.namedtuple('_CompositeTensorRef',
                                                ['type_spec', 'list_of_refs'])


def get_values(x: common_types.TensorType) -> tf.Tensor:
  """Extracts values if the given tensor is composite."""
  if isinstance(x, tf.SparseTensor):
    return x.values
  elif isinstance(x, tf.RaggedTensor):
    return x.flat_values
  else:
    return x


def copy_tensors(tensors):
  """Makes deep copies of a dict of tensors.

  Makes deep copies (using tf.identity or its equivalent for `CompositeTensor`s)
  of the values of `tensors`.

  Args:
    tensors: A a dict whose keys are strings and values are `Tensors`s or
      `CompositeTensor`s.

  Returns:
    A copy of `tensors` with values replaced by tf.identity applied to the
        value, or the equivalent for `CompositeTensor`s.
  """
  return {
      name: _copy_tensor_or_composite_tensor(tensor)
      for name, tensor in tensors.items()
  }


def _copy_tensor(tensor):
  return tf.identity(tensor, name='{}_copy'.format(tensor.op.name))


def _copy_tensor_or_composite_tensor(tensor):
  if isinstance(tensor, composite_tensor.CompositeTensor):
    return tf.nest.map_structure(_copy_tensor, tensor, expand_composites=True)
  return _copy_tensor(tensor)


def _get_ragged_batch_value_rowids(tensor: tf.RaggedTensor) -> tf.Tensor:
  nested_value_rowids = tensor.nested_value_rowids()
  result = nested_value_rowids[-1]
  for value_rowids in reversed(nested_value_rowids[:-1]):
    result = tf.gather(value_rowids, result)
  return result


def _make_regex_filter_fn(
    x: tf.Tensor,
    filter_regex: Optional[str]) -> Callable[[tf.Tensor], tf.Tensor]:
  """Returns a filter function that applies `x`'s mask."""
  if filter_regex is None:
    return lambda values: values
  else:
    if x.dtype != tf.string:
      raise ValueError('Regex filtering is only possible with string input, '
                       f'got {x.dtype}')
    filter_mask = tf.logical_not(tf.strings.regex_full_match(x, filter_regex))
    return lambda values: tf.boolean_mask(values, filter_mask)


def reduce_batch_weighted_counts(
    x: common_types.TensorType,
    weights: Optional[tf.Tensor] = None,
    force: bool = False,
    filter_regex: Optional[str] = None) -> ReducedBatchWeightedCounts:
  """Performs batch-wise reduction to produce (possibly weighted) counts.

  Args:
    x: Input `Tensor` or `CompositeTensor`.
    weights: (Optional) Input weights.
    force: If True, reduces input tensor without weights to unique elements and
      counts.
    filter_regex: (Optional) Regex that matches tokens that have to be filtered
      out. May only be specified if `x` has string dtype.

  Returns:
    a named tuple of...
      The unique values in x
      The sum of the weights for each unique value in x if weights are provided,
        else None
  """
  if isinstance(x, tf.SparseTensor):
    x = x.values
  elif isinstance(x, tf.RaggedTensor):
    x = x.flat_values
  flat_x = tf.reshape(x, [-1])
  filter_fn = _make_regex_filter_fn(flat_x, filter_regex)
  flat_x = filter_fn(flat_x)
  if weights is None:
    if force:
      unique, _, counts = tf.unique_with_counts(flat_x)
      return ReducedBatchWeightedCounts(unique, None, None, counts)
    else:
      # TODO(b/112916494): Always do batch wise reduction once possible.
      return ReducedBatchWeightedCounts(flat_x, None, None, None)
  # TODO(b/134075780): Revisit expected weights shape when input is composite.
  x, weights = assert_same_shape(x, weights)
  weights = filter_fn(tf.reshape(weights, [-1]))
  unique_x_values, unique_idx, _ = tf.unique_with_counts(
      flat_x, out_idx=tf.int64)
  summed_weights_per_x = tf.math.unsorted_segment_sum(
      weights, unique_idx, tf.size(input=unique_x_values))
  return ReducedBatchWeightedCounts(unique_x_values, summed_weights_per_x, None,
                                    None)


def reduce_batch_weighted_cooccurrences(
    x_input: common_types.TensorType,
    y_input: tf.Tensor,
    weights_input: Optional[tf.Tensor] = None,
    extend_with_sentinel_counts: bool = True,
    filter_regex: Optional[str] = None) -> ReducedBatchWeightedCounts:
  """Performs batch-wise reduction to produce weighted co-occurrences.

  Computes the weighted co-occurrence of each feature value in x, for each value
  in the range [0, max(y)). If extend_with_sentinel_counts is true, the return
  value will include an additional sentinel token (not in the true vocabulary)
  that is used to accumulate the global distribution of y values.

  Args:
    x_input: Input `Tensor` or `CompositeTensor`.
    y_input: Integer `Tensor` with which to compute the co-occurrence with
      x_input.
    weights_input: (Optional) Weights input `Tensor`.
    extend_with_sentinel_counts: If True, the reduced batch will be extended
      a sentinel value that accumlate the total distribution of y values. Should
      be True except when called recursively with the sentinel value as input.
    filter_regex: (Optional) Regex that matches tokens that have to be filtered
      out. Can only be specified if `x_input` has string dtype.

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
    # y and densified x should have the same batch dimension.
    assert_eq = tf.compat.v1.assert_equal(
        tf.shape(y_input)[0], tf.cast(x_input.dense_shape[0], tf.int32))
    with tf.control_dependencies([assert_eq]):
      y = tf.gather(y_input, batch_indices)
    x = x_input.values
  elif isinstance(x_input, tf.RaggedTensor):
    # Each batch instance in x corresponds to a single value in y.
    x_row_indices = _get_ragged_batch_value_rowids(x_input)
    assert_compatible = tf.debugging.assert_greater_equal(
        tf.shape(y_input, out_type=tf.int64)[0], x_input.bounding_shape(axis=0))
    with tf.control_dependencies([assert_compatible]):
      x = tf.ensure_shape(x_input.flat_values, [None])
      y = tf.gather(y_input, x_row_indices)
  else:
    y = y_input
    x = x_input
  if weights_input is None:
    weights = tf.ones_like(x, dtype=tf.float32)
  else:
    x, weights_input = assert_same_shape(x, weights_input)
    weights = weights_input
  y = _broadcast_to_x_shape(x, y)
  x, y = assert_same_shape(x, y)
  x = tf.reshape(x, [-1])
  filter_fn = _make_regex_filter_fn(x, filter_regex)
  x = filter_fn(x)
  y = filter_fn(tf.reshape(y, [-1]))
  weights = filter_fn(tf.reshape(weights, [-1]))

  unique_x_values, unique_idx, unique_count = tf.unique_with_counts(
      x, out_idx=tf.int64)

  summed_weights_per_x = tf.math.unsorted_segment_sum(
      weights, unique_idx, tf.size(input=unique_x_values))
  # For each feature value in x, computed the weighted sum positive for each
  # unique value in y.

  max_y_value = tf.cast(tf.reduce_max(input_tensor=y_input), tf.int64)
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


def hashable_tensor_or_op(tensor_or_op):
  """Returns a hashable reference to a Tensor if given a Tensor/CompositeTensor.

  Use deref_tensor_or_op on the result to get the Tensor (or SparseTensor).

  Args:
    tensor_or_op: A `tf.Tensor`, `tf.CompositeTensor`, or other type.

  Returns:
    A hashable representation for the Tensor or CompositeTensor, or the original
    value for other types.
  """
  if isinstance(tensor_or_op, tf.Tensor):
    return tensor_or_op.experimental_ref()
  if isinstance(tensor_or_op, composite_tensor.CompositeTensor):
    # TODO(b/156759471): Use tf.type_spec_from_value here.
    return _CompositeTensorRef(
        type_spec=tensor_or_op._type_spec,  # pylint: disable=protected-access
        list_of_refs=tuple(
            hashable_tensor_or_op(component) for component in tf.nest.flatten(
                tensor_or_op, expand_composites=True)
        ))
  return tensor_or_op


def deref_tensor_or_op(tensor_or_op):
  """Returns a Tensor or CompositeTensor if given a reference, otherwise input.

  Args:
    tensor_or_op: An output of `hashable_tensor_or_op`.

  Returns:
    A Tensor, CompositeTensor, or the given tensor_or_op.
  """
  if isinstance(tensor_or_op, object_identity.Reference):
    return tensor_or_op.deref()
  if isinstance(tensor_or_op, _CompositeTensorRef):
    return tf.nest.pack_sequence_as(
        structure=tensor_or_op.type_spec,
        flat_sequence=[
            deref_tensor_or_op(component)
            for component in tensor_or_op.list_of_refs
        ],
        expand_composites=True)
  return tensor_or_op


def _broadcast_to_x_shape(x, y):
  """Broadcasts y to same shape as x as needed.

  Args:
    x: An input feature.
    y: A feature that is either the same shape as x or has the same outer
      dimensions as x. If the latter, y is broadcast to the same shape as x.

  Returns:
    A Tensor that contains the broadcasted feature, y.
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
    The elements `x` and `y`, the results must be used in order to ensure that
    the dynamic check is executed.
  """
  x.shape.assert_is_compatible_with(y.shape)
  assert_eq = tf.compat.v1.assert_equal(tf.shape(input=x), tf.shape(input=y))
  with tf.control_dependencies([assert_eq]):
    return tf.identity(x), tf.identity(y)


# TODO(b/178189903): This is needed because tf.sparse.reduce_* produces a dense
# tensor which loses its original shape information.
def _sparse_reduce_batch_keep_shape(
    sparse_reduce_fn: Callable, sparse_tensor: tf.SparseTensor) -> tf.Tensor:  # pylint: disable=g-bare-generic
  """Applies a tf.sparse.reduce_* method on the given sparse_tensor."""
  result = sparse_reduce_fn(sparse_tensor, axis=0)
  result.set_shape(sparse_tensor.get_shape()[1:])
  return result


def reduce_batch_count(x: common_types.TensorType,
                       reduce_instance_dims: bool) -> tf.Tensor:
  """Counts elements in the given tensor.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    reduce_instance_dims: A bool, if True - collapses the batch and instance
      dimensions to arrive at a single scalar output. Otherwise, only collapses
      the batch dimension and outputs a `Tensor` of the same shape as the input.

  Returns:
    The element count of `x`. The result is either a scalar if
    reduce_instance_dims is True, otherwise a `Tensor` having shape of `x`
    without the first (batch) dimension. NaNs and infinite input values are
    ignored.
  """
  if isinstance(x, tf.SparseTensor):
    if reduce_instance_dims:
      x = x.values
    else:
      ones_like = tf.SparseTensor(
          indices=x.indices,
          values=tf.cast(_is_finite(x.values), tf.int64),
          dense_shape=x.dense_shape)
      # TODO(b/178189903): Remove this once we no longer lose static shape
      # information.
      # TODO(b/160294509): Remove the hasattr contition once TFT no longer
      # supports TF<2.
      if hasattr(x, '_dense_shape_default'):
        ones_like._dense_shape_default = x._dense_shape_default  # pylint: disable=protected-access
      return _sparse_reduce_batch_keep_shape(tf.sparse.reduce_sum, ones_like)
  elif isinstance(x, tf.RaggedTensor):
    if reduce_instance_dims:
      x = x.flat_values
    else:
      finite_mask = tf.cast(_is_finite(x), tf.int64)
      return tf.math.reduce_sum(finite_mask, axis=0).to_tensor()

  # Exlude NaNs and infinite elements from size calculation. They can only occur
  # in tensors with floating data types.
  if x.dtype.is_floating:
    finite_mask = tf.cast(tf.math.is_finite(x), tf.int64)
    return tf.reduce_sum(finite_mask, axis=None if reduce_instance_dims else 0)

  if reduce_instance_dims:
    return tf.size(input=x)

  # Fill a tensor shaped like x except batch_size=1 with batch_size.
  x_shape = tf.shape(input=x)
  return tf.fill(x_shape[1:], x_shape[0])


def _to_string(x: common_types.TensorType) -> common_types.TensorType:
  """Converts values in the given `Tensor` or `CompositeTensor` to strings."""
  if x.dtype is tf.string:
    return x
  elif isinstance(x, tf.SparseTensor):
    return tf.SparseTensor(
        values=tf.strings.as_string(x.values),
        indices=x.indices,
        dense_shape=x.dense_shape)
  elif isinstance(x, tf.RaggedTensor):
    return tf.RaggedTensor.from_row_splits(
        values=_to_string(x.values), row_splits=x.row_splits)
  else:
    return tf.strings.as_string(x)


def reduce_batch_count_per_key(
    key: common_types.TensorType) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes per-key counts in the given tensor.

  Args:
    key: A `Tensor` or `CompositeTensor`.

  Returns:
    A 2-tuple containing the tensor's (key_vocab, count_per_key).
  """
  key = _to_string(key)

  if isinstance(key, tf.SparseTensor):
    key = key.values
  elif isinstance(key, tf.RaggedTensor):
    key = key.flat_values
  key.set_shape([None])
  unique = tf.unique_with_counts(key, out_idx=tf.int64)

  return unique.y, unique.count


def reorder_histogram(bucket_vocab: tf.Tensor, counts: tf.Tensor,
                      boundary_size: int) -> tf.Tensor:
  """Return the histogram counts in indexed order, and zero out missing values.

  The count_elements analyzer returns counts in alphanumeric order, only for the
  values that are present. To construct a well-formed histogram, we need to
  rearrange them in numerical order, and fill in the missing values.

  Ex: The data contains values in the following form: [0, 1, 0, 1, 0, 3, 0, 1]
  bucket_indices happen to be the same as these values, and
  count_elements(tf.strings.as_string(bucket_indices)) returns:
    bucket_vocab=['1', '3', '0'],
    counts=[3, 1, 4]

  If boundaries=[0, 1, 2, 3, 4], we expect counts=[4, 3, 0, 1, 0],
  which this function will return.

  Args:
    bucket_vocab: A `Tensor` that names the buckets corresponding to the count
      information returned.
    counts: A `Tensor` that matches the bucket_vocab.
    boundary_size: A scalar that provides information about how big the returned
      counts should be.

  Returns:
    counts: A `Tensor` of size boundary_size corresponding to counts of all
        available buckets.
  """
  if bucket_vocab.dtype == tf.string:
    bucket_vocab = tf.strings.to_number(bucket_vocab, tf.int32)
  # counts/bucket_vocab may be out of order and missing values (empty buckets).
  ordering = tf.argsort(
      tf.concat([bucket_vocab,
                 tf.sets.difference([tf.range(boundary_size)],
                                    [bucket_vocab]).values], axis=-1))
  counts = tf.pad(counts, [[0, boundary_size - tf.size(counts)]])
  return tf.gather(counts, ordering)


# TODO(b/62379925): Remove this once all supported TF versions have
# tf.data.experimental.DatasetInitializer.
def is_vocabulary_tfrecord_supported() -> bool:
  if isinstance(ops.get_default_graph(), func_graph.FuncGraph):
    return False
  return ((hasattr(tf.data.experimental, 'DatasetInitializer') or
           hasattr(tf.lookup.experimental, 'DatasetInitializer')) and
          tf.version.VERSION >= '2.4')


# Used to decide which bucket boundary index to assign to a value.
class Side(enum.Enum):
  RIGHT = 'right'
  LEFT = 'left'


def assign_buckets(x: tf.Tensor,
                   bucket_boundaries: tf.Tensor,
                   side: Side = Side.LEFT) -> tf.Tensor:
  """Assigns every value in x to a bucket index defined by bucket_boundaries.

  Note that `x` and `bucket_boundaries` will be cast to a common type that can
  hold the largest of values.

  Args:
    x: a `Tensor` of values to be bucketized.
    bucket_boundaries:  The bucket boundaries `Tensor`. Note that the boundaries
      are going to be flattened.
    side: Controlls index of a bucket that is being assigned: LEFT means that
      a value is going to be assigned index of the rightmost boundary such that
      boundary <= value; RIGHT means that a value is assigned index of the
      leftmost boundary such that value < boundary.

  Returns:
    A `Tensor` of dtype int64 with the same shape as `x`, and each element in
    the returned tensor representing the bucketized value. Bucketized value is
    in the range [0, len(bucket_boundaries)].
  """
  with tf.compat.v1.name_scope(None, 'assign_buckets'):
    flat_x = tf.reshape(x, [-1])
    flat_boundaries = tf.reshape(bucket_boundaries, [-1])

    # Cast values or boundaries to the "largest" dtype to avoid truncating
    # larger values and avoid casting if dtypes are the same.
    if flat_x.dtype.max > flat_boundaries.dtype.max:
      flat_boundaries = tf.cast(flat_boundaries, flat_x.dtype)
    else:
      flat_x = tf.cast(flat_x, flat_boundaries.dtype)

    if side == Side.LEFT:
      # Ignore the last boundary to replicate behavior of the previously used
      # `BoostedTreesBucketize` for backwards compatibility.
      flat_boundaries = flat_boundaries[:-1]

    buckets = tf.searchsorted(
        flat_boundaries, flat_x, side=side.value, out_type=tf.int64)
    return tf.reshape(buckets, tf.shape(x))


# TODO(b/62379925): Remove this once all supported TF versions have
# tf.data.experimental.DatasetInitializer.
class _DatasetInitializerCompat(
    getattr(tf.data.experimental, 'DatasetInitializer',
            getattr(tf.lookup.experimental, 'DatasetInitializer', object))):
  """Extends DatasetInitializer when possible and registers the init_op."""

  def __init__(self, *args, **kwargs):
    if self.__class__.mro()[1] == object:
      raise NotImplementedError(
          'Cannot create a DatasetInitializer with this version of TF: {}'
          .format(tf.__version__))
    super().__init__(*args, **kwargs)

  def initialize(self, table):
    init_op = super().initialize(table)
    collection_ref = tf.compat.v1.get_collection_ref(
        tf.compat.v1.GraphKeys.TABLE_INITIALIZERS)
    if init_op not in collection_ref:
      collection_ref.append(init_op)
    return init_op


def _make_vocab_entry_to_dtype_fn(dtype):

  def vocab_entry_to_dtype(key):
    return key if dtype is tf.string else tf.strings.to_number(
        key, out_type=dtype)

  return vocab_entry_to_dtype


def _make_tfrecord_vocabulary_dataset(vocab_path,
                                      key_dtype=tf.string,
                                      value_dtype=tf.int64,
                                      return_indicator_as_value=False,
                                      has_indicator=False):
  """Makes a (key, value) dataset from a compressed tfrecord file."""
  if not (value_dtype.is_floating or value_dtype.is_integer):
    raise ValueError('value_dtype must be numeric. Got: %s' % value_dtype)
  dataset = tf.data.TFRecordDataset(vocab_path, compression_type='GZIP')
  key_dtype_fn = _make_vocab_entry_to_dtype_fn(key_dtype)
  value_dtype_fn = _make_vocab_entry_to_dtype_fn(value_dtype)

  if return_indicator_as_value:
    assert has_indicator

    def convert_dtype(k, v):
      return key_dtype_fn(k), value_dtype_fn(v)

    return dataset.map(
        _split_vocabulary_entries,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).map(convert_dtype)

  else:
    if has_indicator:
      drop_indicator = lambda k, v: k
      dataset = dataset.map(
          _split_vocabulary_entries,
          num_parallel_calls=tf.data.experimental.AUTOTUNE).map(drop_indicator)

    def convert_dtype_and_swap(v, k):
      return key_dtype_fn(k), tf.cast(v, value_dtype)

    return dataset.enumerate().map(convert_dtype_and_swap)


def make_tfrecord_vocabulary_lookup_initializer(filename_tensor,
                                                key_dtype=tf.string,
                                                value_dtype=tf.int64,
                                                return_indicator_as_value=False,
                                                has_indicator=False):
  """Makes a lookup table initializer from a compressed tfrecord file."""
  graph = ops.get_default_graph()
  with contextlib.ExitStack() as stack:
    # TODO(b/165884902): Use tf.inside_function after dropping TF 2.3 support.
    # If filename_tensor is a graph tensor (e.g. temporary analyzer output), the
    # following operation cannot be lifted to init scope. Hence, check it is an
    # eager tensor or a string constant.
    if isinstance(graph, func_graph.FuncGraph) and isinstance(
        filename_tensor, (ops.EagerTensor, str)):
      # Lift the dataset creation out of graph construction to avoid
      # repeated initialization in TF2.
      stack.enter_context(tf.init_scope())

    dataset = _make_tfrecord_vocabulary_dataset(filename_tensor, key_dtype,
                                                value_dtype,
                                                return_indicator_as_value,
                                                has_indicator)
    # TODO(b/165884902): Use tf.inside_function after dropping TF 2.3 support.
    if isinstance(graph, func_graph.FuncGraph):
      annotators.track_object(dataset, name=None)
    return _DatasetInitializerCompat(dataset)


def _split_vocabulary_entries(batched_vocab_lines):
  """Splits vocabulary entries separated by a single space.

  Vocabulary entries that include indicators are formatted as:
  "<indicator><single space><key>"

  Args:
    batched_vocab_lines: A possible batched string tensor.

  Returns:
    A pair of (indicator, key) tensors.
  """
  # Setting maxsplit=1 allows the vocabulary entries to include space
  # characters.
  split = tf.strings.split(batched_vocab_lines, sep=' ', maxsplit=1)
  if isinstance(split, tf.RaggedTensor):
    split_tensor = split.to_tensor()
    return split_tensor[:, 1], split_tensor[:, 0]
  # TODO(b/160294509): Remove this condition when TFT no longer supports TF<2.
  elif isinstance(split, tf.SparseTensor):
    split_tensor = tf.sparse.to_dense(split)
    return split_tensor[:, 1], split_tensor[:, 0]
  else:
    return split[1], split[0]


def apply_per_key_vocabulary(per_key_filename: tf.Tensor,
                             key: tf.Tensor,
                             default_value: Optional[str] = None,
                             target_ndims: Optional[int] = None) -> tf.Tensor:
  """Apply a stored key-value mapping to a set of keys.

  We expect the values stored in per_key_filename to be two comma-delimited
  numbers, such that it has the following form:
  a 1,3
  b 2,4
  if a and b are the keys corresponding to each row.

  Args:
    per_key_filename:  The file name for the per-key vocabulary file.
    key: A `Tensor` of dtype tf.string, which will determine which values are
      returned.
    default_value: (Optional) A string that determines the default output for
      keys that are not found.
    target_ndims: (Optional) The requested rank of each returned value (wrapped
      in a single Tensor).

  Returns:
    A `Tensor` representing the mapped values of shape [None, 2, ...], where
    extra dimensions are added according to `target_dims`.
    If no default value is given, maps oov keys to [0, 0].
  """
  if default_value is None:
    default_value = '0,0'

  def _construct_table(asset_filepath):
    initializer = tf.lookup.TextFileInitializer(
        asset_filepath,
        key_dtype=tf.string,
        key_index=1,
        value_dtype=tf.string,
        value_index=0,
        delimiter=' ')
    return tf.lookup.StaticHashTable(initializer, default_value=default_value)

  table_lookup, unused_table_size = construct_and_lookup_table(
      _construct_table, per_key_filename, key)

  sparse_result = tf.compat.v1.strings.split(table_lookup, sep=',')
  dense_result = tf.sparse.to_dense(sparse_result, '0')
  # Add 0s where dense_result has empty strings.
  number_strings = tf.where(
      tf.strings.length(dense_result) > 0, dense_result,
      tf.fill(tf.shape(dense_result), '0'))
  numbers = tf.strings.to_number(number_strings)
  # We add 1 to represent the dimension of the multiple associated values found
  # in the vocabulary file (the d values present for every key).
  return numbers if not target_ndims else _align_dims(numbers, target_ndims + 1)


def _is_finite(x: common_types.TensorType) -> common_types.TensorType:
  """Extension of `tf.math.is_finite` that works with all dtypes."""
  if x.dtype.is_floating:
    return tf.math.is_finite(x)
  return tf.ones_like(x, dtype=tf.bool)


def _reduce_batch_count_mean_and_var_sparse(
    x: tf.SparseTensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes elementwise count, mean and var for the given sparse tensor."""
  x_count = tf.cast(reduce_batch_count(x, reduce_instance_dims=False), x.dtype)
  finite_x = tf.SparseTensor(
      indices=x.indices,
      values=tf.where(_is_finite(x.values), x.values, tf.zeros_like(x.values)),
      dense_shape=x.dense_shape)
  x_sum = _sparse_reduce_batch_keep_shape(tf.sparse.reduce_sum, finite_x)
  x_mean = tf.math.divide_no_nan(x_sum, x_count)
  x_minus_mean = tf.sparse.add(finite_x, -tf.broadcast_to(x_mean, tf.shape(x)))
  x_minus_mean_sparse = tf.SparseTensor(x.indices,
                                        tf.gather_nd(x_minus_mean, x.indices),
                                        x.dense_shape)
  sum_of_squares = tf.math.reduce_sum(
      tf.square(tf.sparse.to_dense(x_minus_mean_sparse)), axis=0)
  x_variance = tf.math.divide_no_nan(sum_of_squares, x_count)
  return (x_count, x_mean, x_variance)


def _reduce_batch_count_mean_and_var_ragged(
    x: tf.RaggedTensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes elementwise count, mean and var for the given ragged tensor."""
  zeros_like_x = tf.zeros_like(x)
  x_is_finite = _is_finite(x)
  x_sum = tf.reduce_sum(tf.where(x_is_finite, x, zeros_like_x), axis=0)
  dense_x_count = tf.cast(
      reduce_batch_count(x, reduce_instance_dims=False), x.dtype)
  x_count = tf.RaggedTensor.from_tensor(
      dense_x_count, lengths=x_sum.nested_row_lengths())
  x_mean = tf.math.divide_no_nan(x_sum, x_count).to_tensor()
  dense_x = x.to_tensor()
  dense_x_is_finite = _is_finite(dense_x)
  x_minus_mean = tf.where(dense_x_is_finite, dense_x - x_mean,
                          tf.zeros_like(dense_x))
  x_minus_mean = tf.RaggedTensor.from_tensor(
      x_minus_mean, lengths=x.nested_row_lengths())
  sum_of_squares = tf.reduce_sum(input_tensor=tf.square(x_minus_mean), axis=0)
  x_variance = tf.math.divide_no_nan(sum_of_squares, x_count)
  return (dense_x_count, x_mean, x_variance.to_tensor())


def _reduce_batch_count_mean_and_var_dense(
    x: tf.Tensor,
    reduce_instance_dims: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes count, mean and var for the given dense tensor."""
  axis = None if reduce_instance_dims else 0
  x_count = tf.cast(reduce_batch_count(x, reduce_instance_dims), x.dtype)
  zeros_like_x = tf.zeros_like(x)
  x_is_finite = _is_finite(x)
  x_sum = tf.reduce_sum(tf.where(x_is_finite, x, zeros_like_x), axis=axis)
  x_mean = tf.math.divide_no_nan(x_sum, x_count)
  x_minus_mean = tf.where(x_is_finite, x - x_mean, zeros_like_x)
  sum_of_squares = tf.reduce_sum(
      input_tensor=tf.square(x_minus_mean), axis=axis)
  x_variance = tf.math.divide_no_nan(sum_of_squares, x_count)
  return (x_count, x_mean, x_variance)


def reduce_batch_count_mean_and_var(
    x: common_types.TensorType,
    reduce_instance_dims: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes element count, mean and var for the given tensor.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    reduce_instance_dims: A bool, if True - collapses the batch and instance
        dimensions to arrive at a single scalar output. Otherwise, only
        collapses the batch dimension and outputs a `Tensor` of the same shape
        as the input.

  Returns:
    A 3-tuple containing the tensor's (count, mean, var). NaNs and infinite
    input values are ignored.
  """
  if isinstance(x, tf.SparseTensor):
    if reduce_instance_dims:
      return _reduce_batch_count_mean_and_var_dense(
          x.values, reduce_instance_dims=True)
    else:
      return _reduce_batch_count_mean_and_var_sparse(x)
  elif isinstance(x, tf.RaggedTensor):
    if reduce_instance_dims:
      return _reduce_batch_count_mean_and_var_dense(
          x.flat_values, reduce_instance_dims=True)
    else:
      return _reduce_batch_count_mean_and_var_ragged(x)
  else:
    return _reduce_batch_count_mean_and_var_dense(x, reduce_instance_dims)


def _num_terms_and_factors(num_samples, dtype):
  """Computes counts and sample multipliers for the given number of samples.

  Args:
    num_samples: An integral type scalar `Tensor` containing the number of
    samples used to compute the L-moments. This must be non-negative.
    dtype: The dtype of the samples to process. This determines the output
    `Tensor`s dtype.

  Returns:
    The tuple (current_samples, current_pairs, current_triplets,
    current_quadruplets, l1_factors, l2_factors, l3_factors, l4_factors).
    Entries are `Tensor`s with the given dtype containing counters for each
    moment and the factors to use to compute the moments.
  """
  has_pairs = tf.math.greater(num_samples, 1)
  has_triplets = tf.math.greater(num_samples, 2)
  has_quadruplets = tf.math.greater(num_samples, 3)

  current_samples = tf.cast(num_samples, dtype=dtype)
  current_pairs = tf.cast(
      current_samples * (current_samples - 1.0) / 2.0, dtype=dtype)
  current_triplets = tf.cast(
      current_pairs * (current_samples - 2.0) / 3.0, dtype=dtype)
  current_quadruplets = tf.cast(
      current_triplets * (current_samples - 3.0) / 4.0, dtype=dtype)

  term_up = tf.range(0, current_samples, 1, dtype=dtype)
  term_up_delay_1 = tf.range(-1, current_samples - 1, 1, dtype=dtype)
  term_up_delay_2 = tf.range(-2, current_samples - 2, 1, dtype=dtype)
  term_down = tf.range(current_samples - 1, -1, -1, dtype=dtype)
  term_down_delay_1 = tf.range(current_samples - 2, -2, -1, dtype=dtype)
  term_down_delay_2 = tf.range(current_samples - 3, -3, -1, dtype=dtype)

  l1_denominator = tf.cond(tf.math.greater(num_samples, 0),
                           lambda: current_samples,
                           lambda: tf.constant(1, dtype))
  l1_factors = tf.ones([num_samples], dtype=dtype) / l1_denominator
  l2_denominator = tf.cond(has_pairs,
                           lambda: tf.cast(current_pairs * 2.0, dtype=dtype),
                           lambda: tf.constant(1, dtype))
  l2_factors = (term_up - term_down) / l2_denominator
  l3_denominator = tf.cond(has_triplets,
                           lambda: tf.cast(current_triplets * 6, dtype=dtype),
                           lambda: tf.constant(1, dtype))
  l3_factors = ((term_up * term_up_delay_1 - 4.0 * term_up * term_down +
                 term_down * term_down_delay_1) / l3_denominator)
  l4_denominator = tf.cond(
      has_quadruplets,
      lambda: tf.cast(current_quadruplets * 24, dtype=dtype),
      lambda: tf.constant(1, dtype))
  l4_factors = ((term_up * term_up_delay_1 * term_up_delay_2 -
                 9.0 * term_up * term_up_delay_1 * term_down +
                 9.0 * term_up * term_down * term_down_delay_1 -
                 term_down * term_down_delay_1 * term_down_delay_2) /
                l4_denominator)
  return (current_samples, current_pairs, current_triplets, current_quadruplets,
          l1_factors, l2_factors, l3_factors, l4_factors)


@tf.function
def _condition_l_moments_sparse(
    current_index, unused_l1_sum, unused_l2_sum, unused_l3_sum, unused_l4_sum,
    unused_count_samples, unused_count_pairs, unused_count_triplets,
    unused_count_quadruplets, x_rank_2):
  """Condition for the loop that computes L-moments for a `SparseTensor`."""
  return tf.less(current_index, x_rank_2.dense_shape[1])


@tf.function
def _iteration_l_moments_sparse(
    current_index, l1_sum, l2_sum, l3_sum, l4_sum, count_samples,
    count_pairs, count_triplets, count_quadruplets, x_rank_2):
  """Process one column of a `SparseTensor` and updates L-moments variables."""
  current_x = tf.boolean_mask(
      x_rank_2.values,
      tf.math.equal(x_rank_2.indices[:, 1], [current_index]))
  sorted_x = tf.sort(current_x, axis=0)
  num_samples = tf.shape(current_x)[0]
  (current_samples, current_pairs, current_triplets, current_quadruplets,
   l1_factors, l2_factors, l3_factors,
   l4_factors) = _num_terms_and_factors(num_samples, x_rank_2.values.dtype)

  dim_1 = x_rank_2.dense_shape[1]
  new_l1_sum = l1_sum + tf.scatter_nd(
      [[current_index]],
      [tf.reduce_sum(tf.multiply(sorted_x, l1_factors), axis=0)], [dim_1])
  new_l2_sum = l2_sum + tf.scatter_nd(
      [[current_index]],
      [tf.reduce_sum(tf.multiply(sorted_x, l2_factors), axis=0)], [dim_1])
  new_l3_sum = l3_sum + tf.scatter_nd(
      [[current_index]],
      [tf.reduce_sum(tf.multiply(sorted_x, l3_factors), axis=0)], [dim_1])
  new_l4_sum = l4_sum + tf.scatter_nd(
      [[current_index]],
      [tf.reduce_sum(tf.multiply(sorted_x, l4_factors), axis=0)], [dim_1])

  new_count_samples = count_samples + tf.scatter_nd(
      [[current_index]], [current_samples], [dim_1])
  new_count_pairs = count_pairs + tf.scatter_nd(
      [[current_index]], [current_pairs], [dim_1])
  new_count_triplets = count_triplets + tf.scatter_nd(
      [[current_index]], [current_triplets], [dim_1])
  new_count_quadruplets = count_quadruplets + tf.scatter_nd(
      [[current_index]], [current_quadruplets], [dim_1])

  return (tf.add(current_index, 1),
          new_l1_sum, new_l2_sum, new_l3_sum, new_l4_sum,
          new_count_samples, new_count_pairs, new_count_triplets,
          new_count_quadruplets, x_rank_2)


@tf.function
def _condition_l_moments_dense(
    current_index, unused_l1_sum, unused_l2_sum, unused_l3_sum, unused_l4_sum,
    unused_l1_factors, unused_l2_factors, unused_l3_factors, unused_l4_factors,
    x_rank_2):
  """Condition for the loop that computes L-moments for a `Tensor`."""
  return tf.less(current_index, tf.shape(x_rank_2)[1])


@tf.function
def _iteration_l_moments_dense(
    current_index, l1_sum, l2_sum, l3_sum, l4_sum, l1_factors, l2_factors,
    l3_factors, l4_factors, x_rank_2):
  """Process one column of a `Tensor` and updates L-moments variables."""
  current_x = x_rank_2[:, current_index]
  sorted_x = tf.sort(current_x)

  dim_1 = tf.shape(x_rank_2)[1]
  new_l1_sum = l1_sum + tf.scatter_nd(
      [[current_index]],
      [tf.reduce_sum(tf.multiply(sorted_x, l1_factors), axis=0)], [dim_1])
  new_l2_sum = l2_sum + tf.scatter_nd(
      [[current_index]],
      [tf.reduce_sum(tf.multiply(sorted_x, l2_factors), axis=0)], [dim_1])
  new_l3_sum = l3_sum + tf.scatter_nd(
      [[current_index]],
      [tf.reduce_sum(tf.multiply(sorted_x, l3_factors), axis=0)], [dim_1])
  new_l4_sum = l4_sum + tf.scatter_nd(
      [[current_index]],
      [tf.reduce_sum(tf.multiply(sorted_x, l4_factors), axis=0)], [dim_1])
  return (tf.add(current_index, 1),
          new_l1_sum, new_l2_sum, new_l3_sum, new_l4_sum, l1_factors,
          l2_factors, l3_factors, l4_factors, x_rank_2)


def reduce_batch_count_l_moments(
    x: common_types.TensorType, reduce_instance_dims: bool
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
           tf.Tensor, tf.Tensor]:
  """Computes element first 4 L-moments and the corresponding counts.

  Computes the first 4 L-moments (https://en.wikipedia.org/wiki/L-moment) and
  the number of samples, pairs, etc. used to compute them.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    reduce_instance_dims: A bool, if True - collapses the batch and instance
        dimensions to arrive at a single scalar output. Otherwise, only
        collapses the batch dimension and outputs a `Tensor` of the same shape
        as the input.

  Returns:
    The tuple (count_samples, l1, count_pairs, l2, count_triplets, l3,
    count_quadruplets, l4). Each entry is a `Tensor` with the same dtype as x.
    If reduce_instance_dims is True, the tensors are scalars; otherwise the
    shape is x.shape[1:], i.e. the batch dimension is removed.
  """
  if isinstance(x, tf.SparseTensor) and reduce_instance_dims:
    x = x.values
  elif isinstance(x, tf.RaggedTensor):
    if reduce_instance_dims:
      x = x.flat_values
    else:
      raise NotImplementedError(
          'L-moments only support reduced dims for RaggedTensors')

  if isinstance(x, tf.SparseTensor):
    batch_size = x.dense_shape[0]
    x_rank_2 = tf.sparse.reshape(x, [batch_size, -1])
    dim_1 = x_rank_2.dense_shape[1]
    initial_values = tf.zeros([dim_1], dtype=x.dtype)
    (unused_current_index, l1_sum, l2_sum, l3_sum, l4_sum,
     count_samples, count_pairs, count_triplets,
     count_quadruplets, unused_x_rank_2) = tf.while_loop(
         _condition_l_moments_sparse,
         _iteration_l_moments_sparse,
         [tf.constant(0, dim_1.dtype)] + [initial_values] * 8 + [x_rank_2])
    if reduce_instance_dims:
      final_shape = ()
    elif x.get_shape().ndims and x.get_shape()[1:].is_fully_defined():
      final_shape = x.get_shape()[1:]
    else:
      final_shape = tf.shape(x)[1:]
    l1 = tf.reshape(l1_sum, final_shape)
    l2 = tf.reshape(l2_sum, final_shape)
    l3 = tf.reshape(l3_sum, final_shape)
    l4 = tf.reshape(l4_sum, final_shape)
    count_l1 = tf.reshape(count_samples, final_shape)
    count_l2 = tf.reshape(count_pairs, final_shape)
    count_l3 = tf.reshape(count_triplets, final_shape)
    count_l4 = tf.reshape(count_quadruplets, final_shape)

  else:
    num_samples = tf.size(x) if reduce_instance_dims else tf.shape(x)[0]
    (count_samples, count_pairs, count_triplets, count_quadruplets,
     l1_factors, l2_factors, l3_factors, l4_factors) = _num_terms_and_factors(
         num_samples, x.dtype)
    x_rank_2 = tf.reshape(x, [num_samples, -1])
    dim_1 = tf.shape(x_rank_2)[1]
    initial_moment_values = tf.zeros([dim_1], dtype=x.dtype)
    (unused_current_index, l1_sum, l2_sum, l3_sum, l4_sum, unused_l1_factors,
     unused_l2_factors, unused_l3_factors, unused_l4_factors,
     unused_x_rank_2) = tf.while_loop(
         _condition_l_moments_dense,
         _iteration_l_moments_dense,
         [tf.constant(0, dim_1.dtype)] + [initial_moment_values] * 4 +
         [l1_factors, l2_factors, l3_factors, l4_factors, x_rank_2])
    final_shape = (() if reduce_instance_dims else tf.shape(x)[1:])
    l1 = tf.reshape(l1_sum, final_shape)
    l2 = tf.reshape(l2_sum, final_shape)
    l3 = tf.reshape(l3_sum, final_shape)
    l4 = tf.reshape(l4_sum, final_shape)
    count_l1 = tf.fill(final_shape, count_samples)
    count_l2 = tf.fill(final_shape, count_pairs)
    count_l3 = tf.fill(final_shape, count_triplets)
    count_l4 = tf.fill(final_shape, count_quadruplets)

  return  (count_l1, l1, count_l2, l2, count_l3, l3, count_l4, l4)


def _validate_and_get_dense_value_key_inputs(
    x: common_types.TensorType,
    key: common_types.TensorType) -> Tuple[tf.Tensor, tf.Tensor]:
  """Validate x and key and returns dense representations if feasible.

  Check if sparse x and sparse key have identical indices, map key if dense.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    key: A `Tensor` or `CompositeTensor`. Must be `Tensor` if x is `Tensor`.

  Returns:
    The values of x and key if both are composite, the values of x and a mapped
    key if only x is composite, or the original x and key if both are dense.
  """

  if isinstance(x, tf.Tensor) and isinstance(key, tf.Tensor):
    return x, key
  elif isinstance(x, tf.Tensor):
    raise ValueError('A dense key is required if x is dense')

  elif isinstance(x, tf.SparseTensor) and isinstance(key, tf.SparseTensor):
    assert_shape = tf.debugging.assert_equal(x.dense_shape, key.dense_shape)
    assert_eq = tf.debugging.assert_equal(x.indices, key.indices)
    with tf.control_dependencies([assert_eq, assert_shape]):
      return tf.identity(x.values), tf.identity(key.values)
  elif isinstance(x, tf.SparseTensor) and isinstance(key, tf.Tensor):
    # In this case, the row of x corresponds to the key at that row.
    x_row_indices = x.indices[:, 0]
    assert_compatible = tf.debugging.assert_greater_equal(
        tf.shape(key, out_type=tf.int64)[0], x.dense_shape[0])
    with tf.control_dependencies([assert_compatible]):
      return x.values, tf.gather(key, x_row_indices)
  elif isinstance(x, tf.SparseTensor):
    raise ValueError('A sparse or dense key is required if x is sparse')

  elif isinstance(x, tf.RaggedTensor) and isinstance(key, tf.RaggedTensor):
    x.shape.assert_is_compatible_with(key.shape)
    assert_ops = [
        tf.debugging.assert_equal(x_split, key_split) for x_split, key_split in
        zip(x.nested_row_splits, key.nested_row_splits)
    ]
    with tf.control_dependencies(assert_ops):
      return (tf.ensure_shape(tf.identity(x.flat_values), [None]),
              tf.ensure_shape(tf.identity(key.flat_values), [None]))
  elif isinstance(x, tf.RaggedTensor) and isinstance(key, tf.Tensor):
    # Each batch instance in x corresponds to a single element in key.
    x_row_indices = _get_ragged_batch_value_rowids(x)
    assert_compatible = tf.debugging.assert_greater_equal(
        tf.shape(key, out_type=tf.int64)[0], x.bounding_shape(axis=0))
    with tf.control_dependencies([assert_compatible]):
      return (tf.ensure_shape(x.flat_values,
                              [None]), tf.gather(key, x_row_indices))
  else:
    raise ValueError('A ragged or dense key is required if x is ragged')


def lookup_key(query: tf.Tensor, key_vocab: tf.Tensor) -> tf.Tensor:
  """Look up the index of each element in query in key_vocab.

  Args:
    query: A `Tensor`.
    key_vocab: A 1-D `Tensor` of unique keys.

  Returns:
    The indices of the keys in query, determined by position in key_vocab.
  """

  def _lookup_key():
    # Obtain 0-indexed int64 positions for the keys in key_vocab.
    indices = tf.cast(tf.range(tf.size(key_vocab)), tf.int64)

    expanded_vocab_size = tf.expand_dims(tf.size(key_vocab), axis=0)
    matrix_shape = tf.concat([expanded_vocab_size, tf.shape(query)], axis=0)
    # Expand dims of key_vocab to rank of query.
    vocab_shape = tf.concat(
        [expanded_vocab_size,
         tf.ones(tf.rank(query), dtype=tf.int32)], axis=0)
    # Make copies of key_vocab to fill matrix_shape.
    expand_vocab = tf.broadcast_to(
        tf.reshape(key_vocab, vocab_shape), matrix_shape)
    # Make copies of indices to fill matrix_shape.
    expand_indices = tf.broadcast_to(
        tf.reshape(indices, vocab_shape), matrix_shape)
    # Make copies of query to fill matrix_shape.
    expand_query = tf.broadcast_to(query, matrix_shape)

    # Indices where expand_query equals expand_vocab is set to the key's
    # index. All the other indices are -1.
    expand_result = tf.where(
        tf.math.equal(expand_query, expand_vocab), expand_indices,
        tf.cast(tf.fill(matrix_shape, -1), tf.int64))
    # Reduce matrix above to desired 1-D shape.
    result = tf.math.reduce_max(expand_result, axis=0)
    result.set_shape(query.shape)
    return result

  def _check_vocab_size_and_lookup_key():
    return tf.cond(
        tf.math.equal(tf.size(key_vocab), 0),
        lambda: tf.cast(tf.fill(tf.shape(query), -1), tf.int64), _lookup_key)

  def _check_input_size_and_lookup_key():
    return tf.cond(
        tf.math.equal(tf.size(query),
                      0), lambda: tf.constant([], dtype=tf.int64),
        _check_vocab_size_and_lookup_key)

  return _check_input_size_and_lookup_key()


def _align_dims(tensor: tf.Tensor, target_ndims: int) -> tf.Tensor:
  """Expand the rank of input tensor until it matches the target rank.

  Non-elementwise per-key reduce returns a tensor with rank 1 (batch).
  The dimension count needs to match with x to finish the final mapping, because
  we want to broadcast each reduction with x. To do so we need to add singleton
  dimensions, otherwise TF will try to broadcast along the wrong dimensions.

  Args:
    tensor: A `Tensor`.
    target_ndims: The count of dims we want the output to meet or exceed.

  Returns:
    The original input, with dimension count >= target_ndims.
  """
  if target_ndims is None or target_ndims <= tensor.get_shape().ndims:
    return tensor
  for _ in range(target_ndims - tensor.get_shape().ndims):
    tensor = tf.expand_dims(tensor, -1)
  return tensor


def map_per_key_reductions(
    tensors_to_map: Tuple[tf.Tensor, ...], key: common_types.TensorType,
    key_vocab: tf.Tensor,
    original_input: common_types.TensorType) -> Tuple[tf.Tensor, ...]:
  """Rearrange the reduced per-key result to correspond to the original keys.

  Args:
    tensors_to_map: A tuple of 1-D `Tensor`s that are same shape as key_vocab,
        to be mapped to respective key.
    key: A `Tensor` or `CompositeTensor`.
    key_vocab: A 1-D `Tensor`.
    original_input: A `Tensor` or `CompositeTensor`.

  Returns:
    A tuple same length as tensors_to_map, of `Tensor`s the same dimension as
    original_input. We are mapping using the key for each original_input,
    but output rank needs to match original_input in the dense case.
    For the sparse case, it is enough for output to match original_input.values.
    Any missing key would result in a mapping to 0.
  """

  _, key = _validate_and_get_dense_value_key_inputs(original_input, key)
  key_indices = lookup_key(key, key_vocab)

  ndims = (None if isinstance(original_input,
                              (tf.SparseTensor, tf.RaggedTensor)) else
           original_input.get_shape().ndims)

  # Append a 0 to allow mapping OOVs to it.
  tensors_to_map = [tf.concat([t, [0]], axis=0) for t in tensors_to_map]

  # Replace `-1`s due to OOV with size of key_vocab.
  adjusted_indices = tf.where(
      key_indices >= 0, key_indices,
      tf.cast(
          tf.fill(tf.shape(key_indices), tf.size(key_vocab)), dtype=tf.int64))

  mapped_result = [_align_dims(tf.gather(t, adjusted_indices, axis=-1), ndims)
                   for t in tensors_to_map]

  return tuple(mapped_result)


def reduce_batch_count_mean_and_var_per_key(
    x: common_types.TensorType, key: common_types.TensorType,
    reduce_instance_dims: bool
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes per-key element count, mean and var for the given tensor.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    key: A `Tensor` or `CompositeTensor` (cannot be None).
        Must meet one of the following conditions:
        1. Both x and key are dense,
        2. Both x and key are composite and `key` must exactly match `x` in
        everything except values,
        3. The axis=1 index of each element of sparse x matches its index of
        dense key.
    reduce_instance_dims: A bool, if True - collapses the batch and instance
        dimensions to arrive at a single scalar output. Otherwise, only
        collapses the batch dimension and outputs a `Tensor` of the same shape
        as the input. Not supported for `CompositeTensor`s.

  Returns:
    A 4-tuple containing the `Tensor`s (key_vocab, count, mean, var). NaNs and
    infinite input values are ignored.
  """

  if isinstance(x, (tf.SparseTensor, tf.RaggedTensor)):
    if not reduce_instance_dims:
      raise NotImplementedError(
          'Mean and var per key only support reduced dims for CompositeTensors')

  x, key = _validate_and_get_dense_value_key_inputs(x, key)

  unique = tf.unique(key, out_idx=tf.int64)
  x_is_finite = _is_finite(x)

  finite_x = tf.where(x_is_finite, x, tf.zeros_like(x))
  if reduce_instance_dims:
    x_count = tf.cast(x_is_finite, x.dtype)
    if x.get_shape().ndims != 1:
      x_count = tf.reduce_sum(x_count, axis=1)
    x_count = tf.math.unsorted_segment_sum(x_count, unique.idx,
                                           tf.size(unique.y))
    sums = (
        tf.reduce_sum(finite_x, axis=1)
        if x.get_shape().ndims != 1 else finite_x)
    sums = tf.math.unsorted_segment_sum(sums, unique.idx, tf.size(unique.y))
  else:
    sums = tf.math.unsorted_segment_sum(finite_x, unique.idx, tf.size(unique.y))
    x_count = tf.math.unsorted_segment_sum(
        tf.cast(x_is_finite, tf.float32), unique.idx, tf.size(unique.y))

  means = tf.math.divide_no_nan(tf.cast(sums, x.dtype), x_count)
  sum_sqs = tf.math.unsorted_segment_sum(
      tf.square(finite_x), unique.idx, tf.size(input=unique.y))
  if sum_sqs.get_shape().ndims != 1 and reduce_instance_dims:
    sum_sqs = tf.reduce_sum(sum_sqs, axis=1)

  variances = tf.math.divide_no_nan(sum_sqs, x_count) - tf.square(means)

  return unique.y, tf.cast(x_count, tf.int64), means, variances


# Code for serializing and example proto


_DEFAULT_VALUE_BY_DTYPE = {
    tf.string: '',
    tf.float32: 0,
    tf.int64: 0
}


def _encode_proto(values_dict, message_type, descriptor_source=''):
  """A wrapper around tf.raw_ops.EncodeProto."""
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
  return tf.raw_ops.EncodeProto(
      sizes=sizes,
      values=values,
      field_names=field_names,
      message_type=message_type,
      descriptor_source=descriptor_source)


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


def _get_missing_value(dtype: tf.DType) -> tf.Tensor:
  if dtype.is_floating:
    return tf.constant(_FLOATING_NAN, dtype)
  else:
    return tf.constant(dtype.min + 1, dtype)


def _sparse_minus_reduce_min_and_reduce_max(
    x: tf.SparseTensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
  minus_x = tf.SparseTensor(
      indices=x.indices, values=0 - x.values, dense_shape=x.dense_shape)
  x_count = reduce_batch_count(x, reduce_instance_dims=False)
  batch_has_no_values = tf.equal(x_count, tf.constant(0, dtype=tf.int64))
  x_batch_max = _sparse_reduce_batch_keep_shape(tf.sparse.reduce_max, x)
  x_batch_minus_min = _sparse_reduce_batch_keep_shape(tf.sparse.reduce_max,
                                                      minus_x)
  missing_value = _get_missing_value(x.dtype)
  x_batch_max = tf.where(batch_has_no_values,
                         tf.fill(tf.shape(input=x_batch_max), missing_value),
                         x_batch_max)
  x_batch_minus_min = tf.where(
      batch_has_no_values,
      tf.fill(tf.shape(input=x_batch_minus_min), missing_value),
      x_batch_minus_min)
  return x_batch_minus_min, x_batch_max


def reduce_batch_minus_min_and_max(
    x: common_types.TensorType,
    reduce_instance_dims: bool) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes the -min and max of a tensor x.

  NOTE: For TF versions < 2.4, if all feature values are NaNs, the -min and max
  will both be -inf (consistent with`tf.reduce_max`).

  Args:
    x: A `Tensor` or `CompositeTensor`.
    reduce_instance_dims: A bool indicating whether this should collapse the
      batch and instance dimensions to arrive at a single scalar output, or only
      collapse the batch dimension and outputs a vector of the same shape as the
      input.

  Returns:
    The computed tensor's (batch -min, batch max) pair.
  """
  # In TF < 2.3, neg(x) would throw an exception, if x was tf.int16. Hence, cast
  # to tf.int32.
  if x.dtype in (tf.uint8, tf.uint16, tf.int16):
    x = tf.cast(x, tf.int32)

  elif x.dtype == tf.uint32 or x.dtype == tf.uint64:
    raise TypeError('Tensor type %r is not supported' % x.dtype)

  if reduce_instance_dims:
    if isinstance(x, tf.SparseTensor):
      x = x.values
    elif isinstance(x, tf.RaggedTensor):
      x = x.flat_values

    x_batch_max = tf.reduce_max(input_tensor=x)
    x_batch_minus_min = tf.reduce_max(input_tensor=tf.zeros_like(x) - x)
    return assert_same_shape(x_batch_minus_min, x_batch_max)

  elif isinstance(x, tf.SparseTensor):
    return _sparse_minus_reduce_min_and_reduce_max(x)

  x_batch_max = tf.reduce_max(input_tensor=x, axis=0)
  if isinstance(x, tf.RaggedTensor):
    x_batch_minus_min = tf.reduce_max(input_tensor=tf.math.negative(x), axis=0)
    missing_value = _get_missing_value(x.dtype)
    return (x_batch_minus_min.to_tensor(default_value=missing_value),
            x_batch_max.to_tensor(default_value=missing_value))
  else:
    # TODO(iindyk): switch to `tf.math.negative` when analyzer cache will get
    # invalidated next time.
    return (tf.reduce_max(input_tensor=0 - x, axis=0), x_batch_max)


def reduce_batch_minus_min_and_max_per_key(
    x: common_types.TensorType,
    key: common_types.TensorType) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the -min and max of a tensor x.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    key: A `Tensor` or `CompositeTensor`.
        Must meet one of the following conditions:
        1. Both x and key are dense,
        2. Both x and key are composite and `key` must exactly match `x` in
        everything except values,
        3. The axis=1 index of each element of sparse x matches its index of
        dense key.
  Returns:
    A 3-tuple containing the `Tensor`s (key_vocab, min_per_key, max_per_key).
  """
  if x.dtype == tf.uint8 or x.dtype == tf.uint16:
    x = tf.cast(x, tf.int32)

  elif x.dtype == tf.uint32 or x.dtype == tf.uint64:
    raise TypeError('Tensor type %r is not supported' % x.dtype)

  x, key = _validate_and_get_dense_value_key_inputs(x, key)

  def get_batch_max_per_key(tensor, key_uniques):  # pylint: disable=missing-docstring
    if tensor.get_shape().ndims < 2:
      row_maxes = tensor
    else:
      row_maxes = tf.reduce_max(
          tensor, axis=tf.range(1, tensor.get_shape().ndims))
    return tf.math.unsorted_segment_max(row_maxes, key_uniques.idx,
                                        tf.size(input=key_uniques.y))

  unique = tf.unique_with_counts(key, out_idx=tf.int64)
  x_batch_maxes = get_batch_max_per_key(x, unique)
  x_batch_minus_mins = get_batch_max_per_key(-x, unique)

  x_batch_minus_mins, x_batch_maxes = assert_same_shape(x_batch_minus_mins,
                                                        x_batch_maxes)

  return (unique.y, x_batch_minus_mins, x_batch_maxes)


def track_asset_analyzer_output(eager_asset_path: ops.EagerTensor,
                                graph_tensor: tf.Tensor):
  """Track `graph_tensor` representing analyzer output written to `eager_asset_path`."""
  graph = ops.get_default_graph()
  graph.add_to_collection(
      _ASSET_REPLACEMENTS,
      (hashable_tensor_or_op(graph_tensor), eager_asset_path))


def _get_asset_analyzer_output_and_control_dependency(
    asset_filepath: _AssetFileType
) -> Tuple[_AssetFileType, Optional[tf.Tensor]]:
  """Returns a tuple of (asset filepath, control dependency)."""
  control_dependency = None
  asset_replacements_coll = ops.get_default_graph().get_collection(
      _ASSET_REPLACEMENTS)
  if not asset_replacements_coll:
    return asset_filepath, control_dependency

  if not isinstance(asset_filepath, tf.Tensor):
    raise ValueError('Expected asset_filepath ({}) to be a tf.Tensor.'.format(
        asset_filepath))
  eager_asset_filepath = dict(asset_replacements_coll).get(
      hashable_tensor_or_op(asset_filepath), None)
  if eager_asset_filepath:
    control_dependency = asset_filepath
    asset_filepath = eager_asset_filepath
  return asset_filepath, control_dependency


def _lookup_table(table: lookup_ops.LookupInterface, x: tf.Tensor,
                  control_dependency: Optional[tf.Tensor]) -> tf.Tensor:
  """Look up x in table with an optional depndency on control_dependency."""
  with contextlib.ExitStack() as stack:
    # tf.control_dependencies([tensor]) adds a dependency to tensor.op. Wrap the
    # tensor in an identity op to ensure that walking the graph from `result`
    # encounters the control_dependency tensor.
    if control_dependency is not None:
      stack.enter_context(
          tf.control_dependencies([tf.identity(control_dependency)]))
    result = table.lookup(x)
  return result


def construct_and_lookup_table(construct_table_callable: Callable[
    [_AssetFileType], lookup_ops.LookupInterface],
                               asset_filepath: _AssetFileType,
                               x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Construct a table and look x up in it.

  Args:
    construct_table_callable: A Callable that takes a path to an asset file and
      constructs a lookup table.
    asset_filepath: Path to an asset used to construct the table. Can be a
      python string, a `tf.Tensor`, a `tf.Placeholder`.
    x: A categorical `Tensor` of type tf.string or tf.int[8|16|32|64] to which
      the table lookup should be applied.

  Returns:
    A tuple of the result from looking x up in a table and the table's size.

  """
  graph = ops.get_default_graph()
  # If table is lifted into an initialization scope, add a control dependency
  # on the graph tensor used to track this analyzer in
  # `analyzer_nodes.TENSOR_REPLACEMENTS`.
  asset_filepath, control_dependency = (
      _get_asset_analyzer_output_and_control_dependency(asset_filepath))
  with contextlib.ExitStack() as stack:
    # TODO(b/165884902): Use tf.inside_function after dropping TF 2.3 support.
    if isinstance(graph, func_graph.FuncGraph) and isinstance(
        asset_filepath, (ops.EagerTensor, str)):
      # Lift the table initialization out of graph construction to avoid
      # repeated initialization in TF2.
      stack.enter_context(tf.init_scope())

    table = construct_table_callable(asset_filepath)
    table_size = table.size()
  return _lookup_table(table, x, control_dependency), table_size


def lookup_table(lookup_fn: Callable[[common_types.TensorType, tf.Tensor],
                                     Tuple[tf.Tensor, tf.Tensor]],
                 asset_filepath: _AssetFileType, x: common_types.TensorType):
  """Takes a `lookup_fn` and invokes it on `x` and `asset_filepath`.

  If an eager tensor is being tracked by `asset_filepath`, `lookup_fn` is
  invoked on it instead.

  Args:
    lookup_fn: A Callable that should take a tensor and a deferred vocab
      filename as an input and return a lookup `op` along with the table size.
    asset_filepath: Path to an asset used to construct the table. Can be a
      python string, a `tf.Tensor`, a `tf.Placeholder`.
    x: A categorical `Tensor` or `SparseTensor` of type tf.string or
      tf.int[8|16|32|64] to which the table lookup should be applied.

  Returns:
    A tuple of the result from looking x up and the table size.
  """
  # If table is lifted into an initialization scope, add a control dependency
  # on the graph tensor used to track this analyzer in
  # `analyzer_nodes.TENSOR_REPLACEMENTS`.
  asset_filepath, control_dependency = (
      _get_asset_analyzer_output_and_control_dependency(asset_filepath))
  lookup_result, table_size = lookup_fn(x, asset_filepath)
  with contextlib.ExitStack() as stack:
    # tf.control_dependencies([tensor]) adds a dependency to tensor.op. Wrap the
    # `lookup_result` in an identity op to ensure that walking the graph from
    # it encounters the `control_dependency` tensor. The table size should not
    # have the `control_dependency` tensor as its parent, hence it is returned
    # as is.
    if control_dependency is not None:
      stack.enter_context(
          tf.control_dependencies([tf.identity(control_dependency)]))
    return tf.identity(lookup_result), table_size
