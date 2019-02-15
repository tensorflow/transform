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
"""Helper/utility functions that a tf-transform implementation would find handy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# GOOGLE-INITIALIZATION

import numpy as np
import six
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

_CACHED_EMPTY_ARRAY_BY_DTYPE = {}


def _get_empty_array(dtype):
  if dtype not in _CACHED_EMPTY_ARRAY_BY_DTYPE:
    empty_array = np.array([], dtype)
    empty_array.setflags(write=False)
    _CACHED_EMPTY_ARRAY_BY_DTYPE[dtype] = empty_array
  return _CACHED_EMPTY_ARRAY_BY_DTYPE[dtype]


def feature_spec_as_batched_placeholders(feature_spec):
  """Returns placeholders for the given feature spec.

  Returns a dictionary of placeholders with the same type and shape as calling
  tf.parse_example with the given feature spec.

  Args:
    feature_spec: A TensorFlow feature spec.

  Returns:
    A dictionary from strings to `Tensor` or `SparseTensor`s.

  Raises:
    ValueError: If the feature spec contains feature types not supported.
  """
  result = {}
  for name, spec in six.iteritems(feature_spec):
    if spec.dtype not in (tf.int64, tf.float32, tf.string):
      raise ValueError('{} had invalid dtype'.format(spec))
    if isinstance(spec, tf.FixedLenFeature):
      result[name] = tf.placeholder(spec.dtype, [None] + spec.shape, name=name)
    elif isinstance(spec, tf.VarLenFeature):
      result[name] = tf.sparse_placeholder(spec.dtype, [None, None], name=name)
    elif isinstance(spec, tf.SparseFeature):
      result[name] = tf.sparse_placeholder(spec.dtype, [None, spec.size],
                                           name=name)
    else:
      raise TypeError('Feature spec {} of type {} is not supported'.format(
          spec, type(spec)))
  return result


def make_feed_list(column_names, schema, instances):
  """Creates a feed list for passing data to the graph.

  Converts a list of instances in the in-memory representation to a batch
  suitable for passing to `tf.Session.run`.

  Args:
    column_names: A list of column names.
    schema: A `Schema` object.
    instances: A list of instances, each of which is a map from column name to a
      python primitive, list, or ndarray.

  Returns:
    A list of batches in the format required by a tf `Callable`.

  Raises:
    ValueError: If `schema` is invalid.
  """
  def make_batch_indices(instance_indices):
    """Converts a list of instance indices to the corresponding batch indices.

    Given a list of iterables representing the indices of N sparse tensors,
    creates a single list of indices representing the result of concatenating
    the sparse tensors along the 0'th dimension into a batch of size N.

    Args:
      instance_indices: A list of N iterables, each containing the sparse tensor
        indices for an instance.

    Returns:
      A list of indices with a batch dimension prepended.
    """
    batch_indices = list(itertools.chain.from_iterable([
        [(row_number, index) for index in indices]
        for row_number, indices in enumerate(instance_indices)
    ]))
    # Indices must have shape (?, 2). Therefore if we encounter an empty
    # batch, we return an empty ndarray with shape (0, 2).
    return batch_indices if batch_indices else np.empty([0, 2], dtype=np.int64)

  def make_sparse_batch(instance_indices, instance_values, max_index):
    """Converts a list of sparse instances into a sparse batch.

    Takes lists representing the indices and values of N sparse instances and
    concatenates them along the 0'th dimension into a sparse batch of size N.

    Args:
      instance_indices: A list of N iterables, each containing the sparse tensor
        indices for an instance.
      instance_values: A list of N iterables, each containing the sparse tensor
        values for an instance.
      max_index: An int representing the maximum index in `instance_indices`.

    Returns:
      A `SparseTensorValue` representing a batch of N sparse instances.
    """
    batch_indices = make_batch_indices(instance_indices)
    batch_values = list(itertools.chain.from_iterable(instance_values))
    batch_shape = (len(instance_indices), max_index)
    return tf.SparseTensorValue(batch_indices, batch_values, batch_shape)

  result = []
  feature_spec = schema.as_feature_spec()
  for name in column_names:
    spec = feature_spec[name]
    # TODO(abrao): Validate dtypes, shapes etc.
    if isinstance(spec, tf.FixedLenFeature):
      feed_value = [instance[name] for instance in instances]

    elif isinstance(spec, tf.VarLenFeature):
      values = [[] if instance[name] is None else instance[name]
                for instance in instances]
      indices = [range(len(value)) for value in values]
      max_index = max([len(value) for value in values])
      feed_value = make_sparse_batch(indices, values, max_index)

    elif isinstance(spec, tf.SparseFeature):
      # TODO(abrao): Add support for N-d SparseFeatures.
      max_index = spec.size
      indices, values = [], []
      for instance in instances:
        instance_indices, instance_values = instance[name]
        check_valid_sparse_tensor(
            instance_indices, instance_values, max_index, name)
        indices.append(instance_indices)
        values.append(instance_values)
      feed_value = make_sparse_batch(indices, values, max_index)

    else:
      raise ValueError('Invalid feature spec {}.'.format(spec))
    result.append(feed_value)

  return result


def to_instance_dicts(schema, fetches):
  """Maps the values fetched by `tf.Session.run` to the internal batch format.

  Args:
    schema: A `Schema` object.
    fetches: A dict representing a batch of data, as returned by `Session.run`.

  Returns:
    A list of dicts where each dict is an in-memory representation of an
        instance.

  Raises:
    ValueError: If `schema` is invalid.
  """

  def decompose_sparse_batch(sparse_value):
    """Decomposes a sparse batch into a list of sparse instances.

    Args:
      sparse_value: A `SparseTensorValue` representing a batch of N sparse
        instances. The indices of the SparseTensorValue are expected to be
        sorted by row order.

    Returns:
      A tuple (instance_indices, instance_values) where the elements are lists
      of N lists representing the indices and values, respectively, of the
      instances in the batch.

    Raises:
      ValueError: If `sparse_value` contains out-of-order indices.
    """
    batch_indices, batch_values, batch_shape = sparse_value
    # Preallocate lists of length batch_size, initialized to empty ndarrays,
    # representing the indices and values of instances. We can reuse the return
    # value of _get_empty_array here because it is immutable.
    instance_indices = [_get_empty_array(batch_indices.dtype)] * batch_shape[0]
    instance_values = [_get_empty_array(batch_values.dtype)] * batch_shape[0]
    instance_rank = len(batch_shape[1:])

    # Iterate over the rows in the batch. At each row, consume all the elements
    # that belong to that row.
    current_offset = 0
    for current_row in range(batch_shape[0]):
      start_offset = current_offset

      # Scan forward until we reach an element that does not belong to the
      # current row.
      while current_offset < len(batch_indices):
        row = batch_indices[current_offset][0]
        if row == current_row:
          # This element belongs to the current row.
          current_offset += 1
        elif row > current_row:
          # We've reached the end of the current row.
          break
        else:
          raise ValueError('Encountered out-of-order sparse index: {}.'.format(
              batch_indices[current_offset]))

      if current_offset == start_offset:
        # If the current row is empty, leave the default value, which is an
        # empty array.
        pass
      else:
        instance_indices[current_row] = batch_indices[
            start_offset:current_offset, 1:]
        if instance_rank == 1:
          # In this case indices will have length 1, so for convenience we
          # reshape from [-1, 1] to [-1].
          instance_indices[current_row] = (
              instance_indices[current_row].reshape([-1]))
        instance_values[current_row] = batch_values[start_offset:current_offset]

    return instance_indices, instance_values

  batch_dict = {}
  batch_sizes = {}
  feature_spec = schema.as_feature_spec()
  for name, value in six.iteritems(fetches):
    spec = feature_spec[name]
    if isinstance(spec, tf.FixedLenFeature):
      batch_dict[name] = [value[i] for i in range(value.shape[0])]
      batch_sizes[name] = value.shape[0]

    elif isinstance(spec, tf.VarLenFeature):
      if not isinstance(value, tf.SparseTensorValue):
        raise ValueError(
            'Expected a SparseTensorValue, but got {}'.format(value))
      instance_indices, instance_values = decompose_sparse_batch(value)
      for indices in instance_indices:
        if len(indices.shape) > 1 or np.any(indices != np.arange(len(indices))):
          raise ValueError('Encountered a SparseTensorValue that cannot be '
                           'decoded by ListColumnRepresentation.')
      batch_dict[name] = instance_values
      batch_sizes[name] = len(instance_values)

    elif isinstance(spec, tf.SparseFeature):
      if not isinstance(value, tf.SparseTensorValue):
        raise ValueError(
            'Expected a SparseTensorValue, but got {}'.format(value))
      # TODO(abrao): Add support for N-d SparseFeatures.
      instance_indices, instance_values = decompose_sparse_batch(value)
      batch_dict[name] = zip(instance_indices, instance_values)
      batch_sizes[name] = len(instance_values)

    else:
      raise ValueError('Invalid feature spec {}.'.format(spec))

  # Check batch size is the same for each output.  Note this assumes that
  # fetches is not empty.
  batch_size = next(six.itervalues(batch_sizes))
  for name, batch_size_for_name in six.iteritems(batch_sizes):
    if batch_size_for_name != batch_size:
      raise ValueError(
          'Inconsistent batch sizes: "{}" had batch dimension {}, "{}" had'
          ' batch dimension {}'.format(name, batch_size_for_name,
                                       next(six.iterkeys(batch_sizes)),
                                       batch_size))

  # The following is the simplest way to convert batch_dict from a dict of
  # iterables to a list of dicts.  It does this by first extracting the values
  # of batch_dict, and reversing the order of iteration, then recombining with
  # the keys of batch_dict to create a dict.
  return [dict(zip(six.iterkeys(batch_dict), instance_values))
          for instance_values in zip(*six.itervalues(batch_dict))]


# TODO(b/36040669): Consider moving this to where it can be shared with coders.
def check_valid_sparse_tensor(indices, values, size, name):
  # Check that all indices are in range.
  if len(indices):  # pylint: disable=g-explicit-length-test
    i_min, i_max = min(indices), max(indices)
    if i_min < 0 or i_max >= size:
      i_bad = i_min if i_min < 0 else i_max
      raise ValueError(
          'Sparse column {} has index {} out of range [0, {})'.format(
              name, i_bad, size))

  if len(indices) != len(values):
    raise ValueError(
        'Sparse column {} has indices and values of different lengths: '
        'values: {}, indices: {}'.format(name, values, indices))


def copy_tensors(tensors):
  """Makes deep copies of a dict of tensors.

  Makes deep copies (using tf.identity or its equivalent for `SparseTensor`s) of
  the values of `tensors`.

  Args:
    tensors: A a dict whose keys are strings and values are `Tensors`s or
        `SparseTensor`s.

  Returns:
    A copy of `tensors` with values replaced by tf.identity applied to the
        value, or the equivalent for `SparseTensor`s.
  """
  return {name: _copy_tensor_or_sparse_tensor(tensor)
          for name, tensor in six.iteritems(tensors)}


def _copy_tensor(tensor):
  return tf.identity(tensor, name='{}_copy'.format(tensor.op.name))


def _copy_tensor_or_sparse_tensor(tensor):
  if isinstance(tensor, tf.SparseTensor):
    indices = _copy_tensor(tensor.indices)
    values = _copy_tensor(tensor.values)
    dense_shape = _copy_tensor(tensor.dense_shape)
    return tf.SparseTensor(indices, values, dense_shape)
  return _copy_tensor(tensor)
