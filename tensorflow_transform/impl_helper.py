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
from tensorflow_transform import analyzers
from tensorflow_transform import api
from tensorflow_transform.tf_metadata import dataset_schema

_EMPTY_ARRAY = np.array([])
_EMPTY_ARRAY.setflags(write=False)


def infer_feature_schema(graph, tensors):
  """Given a dict of tensors, creates a `Schema`.

  Infers a schema, in the format of a tf.Transform `Schema`, for the given
  dictionary of tensors.  If a tensor has a ColumnSchema set using
  api.set_column_schema then this schema will be used instead of inferring a
  schema.

  Args:
    graph: The graph that tensors belong to.
    tensors: A dict mapping column names to tensors. The tensors should have a
      0'th dimension interpreted as the batch dimension.

  Returns:
    A `Schema` object.
  """
  schema_overrides = api.get_column_schemas(graph)

  # If the tensor already has a schema attached, use that. Otherwise infer the
  # schema from the underlying tensor.
  return dataset_schema.Schema({
      name: schema_overrides.get(
          tensor, dataset_schema.infer_column_schema_from_tensor(tensor))
      for name, tensor in six.iteritems(tensors)
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
    # representing the indices and values of instances. We can reuse
    # _EMPTY_ARRAY here because it is immutable.
    instance_indices = [_EMPTY_ARRAY] * batch_shape[0]
    instance_values = [_EMPTY_ARRAY] * batch_shape[0]
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
          raise ValueError('Encountered out-of-order sparse index: %r.' %
                           batch_indices[current_offset])

      if current_offset == start_offset:
        # If the current row is empty, leave the default value, _EMPTY_ARRAY.
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
      for indices in instance_indices:
        if len(indices.shape) > 1 or np.any(indices != np.arange(len(indices))):
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


Phase = collections.namedtuple(
    'Phase', ['analyzers', 'table_initializers'])


def create_phases(graph):
  """Returns a list of `Phase`s describing how to execute the pipeline.

  The `graph` is assumed to contain some `Analyzer`s which must be executed by
  doing a full pass over the dataset, and passing the inputs for that analyzer
  into some implementation, then taking the results and replacing the
  `Analyzer`s outputs with constants in the graph containing these results.

  The execution plan is described by a list of `Phase`s.  Each phase contains
  a list of `Analyzer`s, which are the `Analyzer`s which are ready to run in
  that phase, together with a list of ops, which are the table initializers that
  are ready to run in that phase.

  An `Analyzer` or op is ready to run when all its dependencies in the graph
  have been computed.  Thus if the graph is constructed by

  def preprocessing_fn(input)
    x = inputs['x']
    scaled_0 = x - tft.min(x)
    scaled_0_1 = scaled_0 / tft.max(scaled_0)

  Then the first phase will contain the analyzer corresponding to the call to
  `min`, because `x` is an input and so is ready to compute in the first phase,
  while the second phase will contain the analyzer corresponding to the call to
  `max` since `scaled_1` depends on the result of the call to `tft.min` which
  is computed in the first phase.

  More generally, we define a level for each op and each `Analyzer` by walking
  the graph, assigning to each operation the max level of its inputs, to each
  `Tensor` the level of its operation, unless it's the output of an `Analyzer`
  in which case we assign the level of its `Analyzer` plus one.

  The above description omits the role of `FunctionApplication`s.  A
  `FunctionApplication` is a hint to create_phases about the control flow of the
  graph.  Because control flow ops can introduce circular dependencies (and
  other circumstances such as mutable reference introduce similar problems) we
  allow users to construct a `FunctionApplication` which is a hint that the
  outputs `Tensor`s depend only on the input `Tensor`s.  `FunctionApplication`s
  are also needed to collect table initializers to determine which phase a table
  initializer is ready to run in.

  Args:
    graph: A `Graph`.

  Returns:
    A list of `Phase`s.

  Raises:
    ValueError: if the graph cannot be analyzed.
  """
  # Trace through the graph to determine the order in which analyzers must be
  # run.
  with graph.as_default():
    all_analyzers = tf.get_collection(analyzers.ANALYZER_COLLECTION)
    all_maps = tf.get_collection(api.FUNCTION_APPLICATION_COLLECTION)

    analyzer_outputs = {}
    for analyzer in all_analyzers:
      for output_tensor in analyzer.outputs:
        analyzer_outputs[output_tensor] = analyzer

    map_outputs = {}
    for m in all_maps:
      for output_tensor in m.outputs:
        map_outputs[output_tensor] = m

    def _tensor_level(tensor):
      if tensor in analyzer_outputs:
        return _generalized_op_level(analyzer_outputs[tensor]) + 1
      elif tensor in map_outputs:
        return _generalized_op_level(map_outputs[tensor])
      else:
        return _generalized_op_level(tensor.op)

    memoized_levels = {}
    stack = []
    def _generalized_op_level(op):
      """Get the level of a tf.Operation, FunctionApplication or Analyzer."""
      if op not in memoized_levels:
        if op in stack:
          # Append op to stack so cycle appears in error message.
          stack.append(op)
          raise ValueError(
              'Cycle detected: %r.  Cycles may arise by failing to call '
              'apply_function when calling a function that internally uses '
              'tables or control flow ops.' % (stack,))
        stack.append(op)
        inputs = list(op.inputs) + list(getattr(op, 'control_flow_inputs', []))
        memoized_levels[op] = max(
            [_tensor_level(input_tensor) for input_tensor in inputs] + [0])
        assert op == stack.pop()
      return memoized_levels[op]

    analyzers_by_level = collections.defaultdict(list)
    for analyzer in all_analyzers:
      analyzers_by_level[_generalized_op_level(analyzer)].append(analyzer)

    table_initializers_by_level = collections.defaultdict(list)
    all_table_initializers = set()
    for m in all_maps:
      table_initializers_by_level[_generalized_op_level(m)].extend(
          m.table_initializers)
      all_table_initializers.update(m.table_initializers)
    expected_table_initializers = set(
        tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
    if expected_table_initializers - all_table_initializers:
      raise ValueError(
          'Found table initializers (%r) that were not associated with any '
          'FunctionApplication.  Use tft.apply_function to wrap any code '
          'that generates tables.'
          % (expected_table_initializers - all_table_initializers))
    if all_table_initializers - expected_table_initializers:
      raise ValueError(
          'The operations (%r) were registered as table initializers during '
          'a call to apply_function, but were not in the TABLE_INITIALIZERS '
          'collection.  This may be a bug in tf.Transform, or you may have '
          'cleared or altered this collection'
          % (all_table_initializers - expected_table_initializers))

    assert len(table_initializers_by_level) <= len(analyzers_by_level) + 1
    return [
        Phase(analyzers_by_level[level], table_initializers_by_level[level])
        for level in sorted(six.iterkeys(analyzers_by_level))]


def run_preprocessing_fn(preprocessing_fn, schema):
  """Runs the user-defined preprocessing function.

  Args:
    preprocessing_fn: A function that takes a dict of `Tensor` or
        `SparseTensor`s as input and returns a dict of `Tensor` or
        `SparseTensor`s as output.
    schema: A `tf_metadata.Schema`.

  Returns:
    A tuple of a graph, and dicts from logical names to `Tensor` or
        `SparseTensor`s, for inputs and outputs respectively.

  Raises:
    ValueError: If `schema` contains unsupported feature types.
  """
  # Run the preprocessing function, which will construct a TF graph for the
  # purpose of validation.  The graphs used for computation will be built from
  # the DAG of columns in make_transform_fn_def.
  graph = tf.Graph()
  with graph.as_default():
    inputs = {}
    input_copies = {}

    with tf.name_scope('inputs'):
      for key, column_schema in six.iteritems(schema.column_schemas):
        with tf.name_scope(key):
          tensor = column_schema.as_batched_placeholder()
          # In order to avoid a bug where import_graph_def fails when the
          # input_map and return_elements of an imported graph are the same
          # (b/34288791), we avoid using the placeholder of an input column as
          # an output of a graph. We do this by applying tf.identity to the
          # placeholder and using the output of tf.identity as the tensor
          # representing the output of this column, thus preventing the
          # placeholder from being used as both an input and an output.
          if isinstance(tensor, tf.SparseTensor):
            copied_tensor = tf.SparseTensor(
                indices=tf.identity(tensor.indices),
                values=tf.identity(tensor.values),
                dense_shape=tf.identity(tensor.dense_shape))
          else:
            copied_tensor = tf.identity(tensor)

          inputs[key] = tensor
          input_copies[key] = copied_tensor

    # Construct the deferred preprocessing graph by calling preprocessing_fn on
    # the inputs.
    outputs = preprocessing_fn(input_copies)

  return graph, inputs, outputs
