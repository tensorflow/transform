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

import collections
import itertools


import numpy as np
import six
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_transform import api
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_schema


def infer_feature_schema(columns):
  """Given a dict of columns, creates a `Schema`.

  Infers a schema, in the format of a tf.Transform `Schema`, for the given
  dictionary of columns.

  Args:
    columns: A dict mapping column names to `Column`s. The tensors represented
      by these columns should have a 0'th dimension interpreted as the batch
      dimension. In order to pass a tensor representing a single instance, it
      must be wrapped in a batch of size 1.

  Returns:
    A `Schema` object.
  """
  # If the column already has a schema attached, use that. Otherwise infer the
  # schema from the underlying tensor.
  return dataset_schema.Schema({
      name: (column.schema if column.schema
             else dataset_schema.infer_column_schema_from_tensor(column.tensor))
      for name, column in six.iteritems(columns)
  })


def make_feed_dict(input_tensors, schema, instances):
  """Creates a feed dict for passing data to the graph.

  Converts a list of instances in the in-memory representation to a batch
  suitable for passing to `tf.Session.run`.

  Args:
    input_tensors: A map from column names to `Tensor`s or `SparseTensor`s.
    schema: A `Schema` object.
    instances: A list of instances, each of which is a map from column name to a
      python primitive, list, or ndarray.

  Returns:
    A map from `Tensor`s or `SparseTensor`s to batches in the format required by
    `tf.Session.run`.

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

  result = {}
  for key, input_tensor in six.iteritems(input_tensors):
    representation = schema.column_schemas[key].representation
    if isinstance(representation, dataset_schema.FixedColumnRepresentation):
      feed_value = [instance[key] for instance in instances]

    elif isinstance(representation, dataset_schema.ListColumnRepresentation):
      values = [instance[key] for instance in instances]
      indices = [range(len(instance[key])) for instance in instances]
      max_index = max([len(instance[key]) for instance in instances])
      feed_value = make_sparse_batch(indices, values, max_index)

    elif isinstance(representation, dataset_schema.SparseColumnRepresentation):
      max_index = schema.column_schemas[key].axes[0].size
      indices, values = [], []
      for instance in instances:
        instance_indices, instance_values = instance[key]
        check_valid_sparse_tensor(
            instance_indices, instance_values, max_index, key)
        indices.append(instance_indices)
        values.append(instance_values)
      feed_value = make_sparse_batch(indices, values, max_index)

    else:
      raise ValueError('Invalid column %r.' % schema.column_schemas[key])
    result[input_tensor] = feed_value

  return result


def make_output_dict(schema, fetches):
  """Maps the values fetched by `tf.Session.run` to the internal batch format.

  Args:
    schema: A `Schema` object.
    fetches: A dict representing a batch of data, as returned by `Session.run`.

  Returns:
    A dict from keys to a list or 2-tuple of lists.

  Raises:
    ValueError: If `schema` is invalid.
  """
  def decompose_sparse_batch(sparse_value):
    """Decomposes a sparse batch into a list of sparse instances.

    Args:
      sparse_value: A `SparseTensorValue` representing a batch of N sparse
        instances.

    Returns:
      A tuple (instance_indices, instance_values) where the elements are lists
      of N lists representing the indices and values, respectively, of the
      instances in the batch.
    """
    batch_indices, batch_values, batch_shape = sparse_value
    # Preallocate a list (of lists) of length batch_size representing instances.
    instance_indices = [[] for _ in range(batch_shape[0])]
    instance_values = [[] for _ in range(batch_shape[0])]
    # Iterate over the flat list of batch indices and values and assign them to
    # the appropriate instance.
    current_row = -1
    for index, value in zip(batch_indices, batch_values):
      row = index[0]
      column = index[1] if len(index) == 2 else tuple(index[1:])
      while row > current_row:
        current_row += 1
      instance_indices[current_row].append(column)
      instance_values[current_row].append(value)
    return instance_indices, instance_values

  # Make a dict where the values are lists with one element per instance.
  result = {}
  for key, value in six.iteritems(fetches):
    representation = schema.column_schemas[key].representation
    if isinstance(representation, dataset_schema.FixedColumnRepresentation):
      result[key] = [value[i] for i in range(value.shape[0])]

    elif isinstance(representation, dataset_schema.ListColumnRepresentation):
      if not isinstance(value, tf.SparseTensorValue):
        raise ValueError('Expected a SparseTensorValue, but got %r' % value)
      instance_indices, instance_values = decompose_sparse_batch(value)
      if any(indices != list(range(len(indices)))
             for indices in instance_indices):
        raise ValueError('Encountered a SparseTensorValue that cannot be '
                         'decoded by ListColumnRepresentation.')
      result[key] = instance_values

    elif isinstance(representation, dataset_schema.SparseColumnRepresentation):
      if not isinstance(value, tf.SparseTensorValue):
        raise ValueError('Expected a SparseTensorValue, but got %r' % value)
      result[key] = decompose_sparse_batch(value)

    else:
      raise ValueError('Unhandled column representation: %r.' % representation)

  return result


def to_instance_dicts(batch_dict):
  """Converts from the internal batch format to a list of instances.

  Args:
    batch_dict: A dict in the in-memory batch format, as returned by
      `make_output_dict`.

  Returns:
    A list of dicts in the in-memory instance format.
  """
  def get_instance_values(batch_dict):
    # SparseFeatures are represented as a 2-tuple of list of lists, so
    # in that case we convert to a list of 2-tuples of lists.
    columns = (column if not isinstance(column, tuple) else zip(*column)
               for column in six.itervalues(batch_dict))
    return itertools.izip(*columns)

  return [dict(zip(six.iterkeys(batch_dict), instance_values))
          for instance_values in get_instance_values(batch_dict)]


def check_valid_sparse_tensor(indices, values, size, name):
  # Check that all indices are in range.
  if len(indices):  # pylint: disable=g-explicit-length-test
    i_min, i_max = min(indices), max(indices)
    if i_min < 0 or i_max >= size:
      i_bad = i_min if i_min < 0 else i_max
      raise ValueError('Sparse column %r has index %d out of range [0, %d)'
                       % (name, i_bad, size))

  if len(indices) != len(values):
    raise ValueError(
        'Sparse column %r has indices and values of different lengths: '
        'values: %r, indices: %r' % (name, values, indices))


def _make_input_columns(schema):
  """Create input columns based on a `Schema`."""
  placeholders = schema.as_batched_placeholders()
  return {
      # pylint: disable=protected-access
      key: api._InputColumn(placeholders[key], column_schema)
      for key, column_schema in six.iteritems(schema.column_schemas)
  }


# Arguments to the constructor of tf.Constant.
ConstantTensorValue = collections.namedtuple(
    'ConstantTensorValue', ['value', 'dtype', 'shape'])


def replace_tensors_with_constant_values(
    saved_model_dir, bound_saved_model_dir, input_value_mapping):
  """Takes a SavedModel and replaces some inputs with constant values.

  Replaces some inputs from the SavedModel with constant tensors constructed
  based on `tensor_value_mapping`.

  Args:
    saved_model_dir: The directory of a SavedModel.
    bound_saved_model_dir: The directory to which to write the SavedModel with
       some inputs bound to constants.
    input_value_mapping: A map from inputs to `ConstantTensorValue`s.
  """
  with tf.Graph().as_default():
    # Create constant tensors representing bound inputs.
    bound_input_tensors = {
        key: tf.constant(value.value, value.dtype)
        for key, value in six.iteritems(input_value_mapping)
    }
    with tf.Session() as session:
      input_tensors, output_tensors = (
          saved_transform_io.partially_apply_saved_transform(
              saved_model_dir, bound_input_tensors))
      saved_transform_io.write_saved_transform_from_session(
          session, input_tensors, output_tensors, bound_saved_model_dir)


def _copy_placeholder(placeholder):
  """Copies a placeholder to a new graph."""
  if isinstance(placeholder, tf.SparseTensor):
    # NOTE: We don't use sparse_placeholder because we want to ensure that the
    # placeholder we produce is identical to the original tensor.
    return tf.SparseTensor(
        indices=_copy_placeholder(placeholder.indices),
        values=_copy_placeholder(placeholder.values),
        dense_shape=_copy_placeholder(placeholder.dense_shape))
  else:
    if placeholder.op.type != 'Placeholder':
      raise ValueError(
          'Attempted to copy a tensor that was not a placeholder: %s'
          % placeholder.op.type)
    return tf.placeholder(placeholder.dtype, shape=placeholder.get_shape())


def make_transform_fn_def(schema, inputs, outputs, saved_model_dir):
  """Loads the graph defined by a partial preprocesssing function.

  Creates a SavedModel on disk representing the transform function.  The given
  input and output columns implicitly define a transformation DAG; this is the
  function that is written.  The resulting SavedModel requires additional inputs
  providing analyzer results.  The mapping from these input names to the
  `_AnalyzerOutput`s will be returned.

  Args:
    schema: A `Schema` object.
    inputs: A dict from strings to `Column`s.
    outputs: A dict from strings to `Column`s.
    saved_model_dir: The directory where the SavedModel should be stored.

  Returns:
    A dict from input names in saved model to statistics (`_AnalyzerOutput`s).

  Raises:
    ValueError: If `schema` and `inputs` do not have the same keys, or if output
      columns cannot be derived from input columns.
  """
  # Construct the graph, keeping track of tensors for input columns, output
  # columns, and statistic placeholders.  Note that while each column already
  # has a tensor, these are only for validation.  We ignore these and construct
  # a new graph here, because it's easier to construct the subgraph we are
  # interested in, than to extract it from the graph we already have.
  input_tensors = {}
  column_names_to_statistics = {}
  if (sorted(six.iterkeys(schema.as_feature_spec())) !=
      sorted(six.iterkeys(inputs))):
    raise ValueError('Schema and input columns had different keys (%s vs %s).'
                     % (sorted(six.iterkeys(schema.as_feature_spec())),
                        sorted(six.iterkeys(inputs))))

  def get_new_input_column_name():
    analyzer_idx = 0
    while True:
      name = 'analyzer_placeholder_input_column_%d' % analyzer_idx
      analyzer_idx += 1
      if name not in input_tensors:
        return name

  cached_column_to_tensor = {}
  def column_to_tensor(column):
    """Returns the tensor that represents the given column."""
    if column in cached_column_to_tensor:
      return cached_column_to_tensor[column]

    # pylint: disable=protected-access
    if isinstance(column, api._AnalyzerOutput):
      # For analyzer outputs, copy over the placeholder tensor and add the
      # placeholder to the dict that keeps track of the map between tensors and
      # analyzer output placeholders.
      tensor = _copy_placeholder(column.tensor)
      name = get_new_input_column_name()
      input_tensors[name] = tensor
      column_names_to_statistics[name] = column
    elif isinstance(column,
                    (api._TransformedColumn, api._TransformedStatistic)):
      # For transformed columns or statistics, apply the transformation.
      tensor = column.fn(*[column_to_tensor(input_column)
                           for input_column in column.inputs])
    elif isinstance(column, api._InputColumn):
      raise ValueError('Reached input column that wasn\'t in input dict')
    # pylint: enable=protected-access

    cached_column_to_tensor[column] = tensor
    return tensor

  graph = tf.Graph()
  with graph.as_default():
    # Input columns form the roots of the graph, and so we need the create them
    # again from scratch in this new graph.
    new_input_columns = _make_input_columns(schema)

    # Compute placeholder for input columns.
    input_tensors.update({
        key: column.placeholder
        for key, column in six.iteritems(new_input_columns)
    })

    # Initialize cache of column tensors with the input columns.
    cached_column_to_tensor.update({
        inputs[key]: new_input_columns[key].tensor
        for key in six.iterkeys(inputs)
    })

    # Compute tensors representing output columns.  As a side effect this will
    # populate column_names_to_statistics with all placeholders for
    # `_AnalyzerOutputs` that are parents of outputs, and also augment
    # input_tensors
    output_tensors = {key: column_to_tensor(column)
                      for key, column in six.iteritems(outputs)}

    with tf.Session() as session:
      saved_transform_io.write_saved_transform_from_session(
          session, input_tensors, output_tensors, saved_model_dir)
  return column_names_to_statistics


def load_transform_fn_def(saved_model_dir):
  """Loads a TransformFnDef into a graph.

  Similar to apply_transform_fn_def except it loads input placeholders and
  returns a column to tensor mapping for inputs.

  Args:
    saved_model_dir: The location of the SavedModel.

  Returns:
    A pair of dicts, for inputs and outputs, whose keys are column names and
    whose values are `Tensor`s or `SparseTensor`s representing these columns.
  """
  with tf.Session():
    return saved_transform_io.partially_apply_saved_transform(
        saved_model_dir, {})


def run_preprocessing_fn(preprocessing_fn, schema):
  """Runs the user-defined preprocessing function, returning a DAG of columns.

  Args:
    preprocessing_fn: A function that takes a dict of `Column`s as input and
      returns a dict of `Column`s as output.
    schema: A dict mapping column names to `tf.FixedLenFeature`,
      `tf.VarLenFeature` or `tf.SparseFeature` objects.

  Returns:
    A tuple of input columns and output columns.

  Raises:
    ValueError: If `schema` contains unsupported feature types.
  """
  # Run the preprocessing function, which will construct a TF graph for the
  # purpose of validation.  The graphs used for computation will be built from
  # the DAG of columns in make_transform_fn_def.
  graph = tf.Graph()
  with graph.as_default():
    inputs = _make_input_columns(schema)

    # Construct the deferred preprocessing graph by calling preprocessing_fn on
    # the inputs.
    outputs = preprocessing_fn(inputs)

  return inputs, outputs
