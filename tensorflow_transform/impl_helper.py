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
from tensorflow_transform import graph_tools

_EMPTY_ARRAY = np.array([])
_EMPTY_ARRAY.setflags(write=False)


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
  feature_spec = schema.as_feature_spec()
  for name, input_tensor in six.iteritems(input_tensors):
    spec = feature_spec[name]
    if isinstance(spec, tf.FixedLenFeature):
      feed_value = [instance[name] for instance in instances]

    elif isinstance(spec, tf.VarLenFeature):
      values = [instance[name] for instance in instances]
      indices = [range(len(instance[name])) for instance in instances]
      max_index = max([len(instance[name]) for instance in instances])
      feed_value = make_sparse_batch(indices, values, max_index)

    elif isinstance(spec, tf.SparseFeature):
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
    result[input_tensor] = feed_value

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
          raise ValueError('Encountered out-of-order sparse index: {}.'.format(
              batch_indices[current_offset]))

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
      instance_indices, instance_values = decompose_sparse_batch(value)
      batch_dict[name] = zip(instance_indices, instance_values)
      batch_sizes[name] = len(instance_values)

    else:
      raise ValueError('Invalid feature spec {}.'.format(spec))

  # Check batch size is the same for each output.  Note this assumes that
  # fetches is not empty.
  batch_size = batch_sizes.values()[0]
  for name, batch_size_for_name in six.iteritems(batch_sizes):
    if batch_size_for_name != batch_size:
      raise ValueError(
          'Inconsistent batch sizes: "{}" had batch dimension {}, "{}" had'
          ' batch dimension {}'.format(
              name, batch_size_for_name, batch_sizes.keys()[0], batch_size))

  # The following is the simplest way to convert batch_dict from a dict of
  # iterables to a list of dicts.  It does this by first extracting the values
  # of batch_dict, and reversing the order of iteration, then recombining with
  # the keys of batch_dict to create a dict.
  return [dict(zip(six.iterkeys(batch_dict), instance_values))
          for instance_values in zip(*six.itervalues(batch_dict))]


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


# Named tuple with details for each output of an Analyzer.
AnalyzerOutputInfo = collections.namedtuple(
    'AnalyzerOutputInfo', ['name', 'is_asset'])


AnalyzerInfo = collections.namedtuple(
    'AnalyzerInfo',
    ['name', 'input_tensor_names', 'spec', 'output_infos'])


Phase = collections.namedtuple(
    'Phase', ['analyzer_infos', 'table_initializers'])


def create_phases(inputs):
  """Returns a list of `Phase`s describing how to execute the pipeline.

  The default graph is assumed to contain some `Analyzer`s which must be
  executed by doing a full pass over the dataset, and passing the inputs for
  that analyzer into some implementation, then taking the results and replacing
  the `Analyzer`s outputs with constants in the graph containing these results.

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
    inputs: A dict whose keys are strings and values are `Tensor` or
        `SparseTensor`s.

  Returns:
    A list of `Phase`s.

  Raises:
    ValueError: if the graph cannot be analyzed.
  """
  feed_tensors = inputs.values()

  remaining_analyzers = tf.get_collection(analyzers.ANALYZER_COLLECTION)
  analyzer_output_ready = {}
  for analyzer in remaining_analyzers:
    for tensor in analyzer.outputs:
      analyzer_output_ready[tensor] = False

  # Construct `AnalyzerInfo`s, removing any tensors that are analyzer outputs
  # from the ASSET_FILEPATHS collection.  These tensors will be replaced and
  # the replacements will be added to the ASSET_FILEPATHS.  Setting
  # AnalyzerOutputInfo.is_asset instructs the implementation to do this.
  asset_filepaths_collection = tf.get_collection_ref(
      tf.GraphKeys.ASSET_FILEPATHS)
  asset_filepaths = collections.OrderedDict(
      (tensor, True)
      for tensor in tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))

  phases = []
  while remaining_analyzers:
    analyzer_inputs = []
    for analyzer in remaining_analyzers:
      analyzer_inputs.extend(analyzer.inputs)
    ready_init_ops, ready_analyzer_inputs = (
        graph_tools.determine_ready_tensors_and_table_initializers(
            tf.get_default_graph(), analyzer_inputs, feed_tensors,
            analyzer_output_ready))
    ready_analyzer_inputs = set(ready_analyzer_inputs)

    new_remaining_analyzers = []
    analyzer_infos = []
    for analyzer in remaining_analyzers:
      if all(tensor in ready_analyzer_inputs for tensor in analyzer.inputs):
        input_tensor_names = [tensor.name for tensor in analyzer.inputs]
        output_infos = [
            AnalyzerOutputInfo(tensor.name, asset_filepaths.pop(tensor, False))
            for tensor in analyzer.outputs]
        analyzer_infos.append(AnalyzerInfo(
            analyzer.name, input_tensor_names, analyzer.spec, output_infos))

        for tensor in analyzer.outputs:
          analyzer_output_ready[tensor] = True
      else:
        new_remaining_analyzers.append(analyzer)
    phases.append(Phase(analyzer_infos, ready_init_ops))

    assert len(new_remaining_analyzers) < len(remaining_analyzers)
    remaining_analyzers = new_remaining_analyzers

  del asset_filepaths_collection[:]
  asset_filepaths_collection.extend(six.iterkeys(asset_filepaths))

  return phases


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
