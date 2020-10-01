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

import copy
import itertools
import re

# GOOGLE-INITIALIZATION

import numpy as np
import six
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import graph_context
from tensorflow_transform.tf_metadata import schema_utils
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
# pylint: enable=g-direct-tensorflow-import

_CACHED_EMPTY_ARRAY_BY_DTYPE = {}
_VALID_SCOPE_REGEX = re.compile('^[A-Za-z0-9]*$')
_INVALID_SCOPE_CHAR = re.compile('[^A-Za-z0-9_.\\-/>]')


def _get_empty_array(dtype):
  if dtype not in _CACHED_EMPTY_ARRAY_BY_DTYPE:
    empty_array = np.array([], dtype)
    empty_array.setflags(write=False)
    _CACHED_EMPTY_ARRAY_BY_DTYPE[dtype] = empty_array
  return _CACHED_EMPTY_ARRAY_BY_DTYPE[dtype]


def batched_placeholders_from_specs(specs):
  """Returns placeholders for the given tf.TypeSpecs or feature specs.

  Args:
    specs: a Dict[Text, Union[tf.TypeSpec, FeatureSpec]]. Note that the values
      in this dict must be of the same type. Mixing is not allowed.

  Returns:
    A dictionary from strings to `Tensor`, `SparseTensor`s, or `RaggedTensor`s.

  Raises:
    ValueError: when the TypeSpec or feature spec has an unsupported dtype.
  """
  if not (all([_is_feature_spec(s) for s in six.itervalues(specs)]) or
          all([isinstance(s, tf.TypeSpec) for s in six.itervalues(specs)])):
    raise TypeError('Specs must be all tf.TypeSpecs or feature specs. '
                    'Mixing is not allowed. Got: {}'.format(specs))

  result = {}
  for name, spec in six.iteritems(specs):
    if isinstance(spec, tf.RaggedTensorSpec):
      # TODO(b/159717195): clean up protected-access
      spec_dtype = spec._dtype  # pylint: disable=protected-access
    else:
      spec_dtype = spec.dtype
    if spec_dtype not in (tf.int64, tf.float32, tf.string):
      raise ValueError('Feature {} ({}, {}) had invalid dtype'
                       .format(name, spec, type(spec)))
    if isinstance(spec, tf.TypeSpec):
      result[name] = _batched_placeholder_from_typespec(name, spec)
    else:
      result[name] = _batched_placeholder_from_feature_spec(name, spec)

  return result


# Older TFX versions may refer to this function instead.
# TODO(b/150721482): remove once TFX 0.21.1 is released.
feature_spec_as_batched_placeholders = batched_placeholders_from_specs


def _is_feature_spec(spec):
  return isinstance(spec, (
      tf.io.VarLenFeature, tf.io.SparseFeature, tf.io.FixedLenFeature))


def _sanitize_scope_name(name):
  scope_name = _INVALID_SCOPE_CHAR.sub('_', name)
  if not _VALID_SCOPE_REGEX.match(scope_name):
    scope_name = 'F_{}'.format(scope_name)
  return scope_name


def _batched_placeholder_from_typespec(name, typespec):
  """Creates a batched placeholder from a tf.TypeSpec."""
  if isinstance(typespec,
                (tf.TensorSpec, tf.SparseTensorSpec, tf.RaggedTensorSpec)):
    with tf.name_scope(_sanitize_scope_name(name)):
      return tf.nest.map_structure(
          lambda tspec: tf.compat.v1.placeholder(tspec.dtype, tspec.shape),
          typespec,
          expand_composites=True)

  raise ValueError('Unsupported typespec: {}({}) for feature {}'.format(
      typespec, type(typespec), name))


def _batched_placeholder_from_feature_spec(name, feature_spec):
  """Creates a batched placeholder from a feature spec."""
  scope_name = _sanitize_scope_name(name)
  if isinstance(feature_spec, tf.io.FixedLenFeature):
    return tf.compat.v1.placeholder(
        feature_spec.dtype, [None] + feature_spec.shape, name=scope_name)
  elif isinstance(feature_spec, tf.io.VarLenFeature):
    return tf.compat.v1.sparse_placeholder(
        feature_spec.dtype, [None, None], name=scope_name)
  elif isinstance(feature_spec, tf.io.SparseFeature):
    return tf.compat.v1.sparse_placeholder(
        feature_spec.dtype, [None, feature_spec.size], name=scope_name)

  raise ValueError('Unsupported feature spec: {}({}) for feature {}'
                   .format(feature_spec, type(feature_spec), name))


def make_feed_list(column_names,
                   schema,
                   instances,
                   produce_eager_tensors=False):
  """Creates a feed list for passing data to the graph.

  This converts a list of instances in the in-memory representation to:
  * If `produce_eager_tensors` is `False`: a batch suitable for passing to
    `tf.Session.run`.
  * If `produce_eager_tensors` is `True`: a batch of eager tensors for passing
    as arguments to a `tf.function`.

  Args:
    column_names: A list of column names.
    schema: A `Schema` proto.
    instances: A list of instances, each of which is a map from column name to a
      python primitive, list, or ndarray.
    produce_eager_tensors: (Optional) Boolean indicating whether eager tensors
      should be returned. Default is `False`.

  Returns:
    A list of batches in the format required by a tf `Callable`.

  Raises:
    ValueError: If `schema` is invalid.
    RuntimeError: If `produce_eager_tensors` is True, but eager mode is
      disabled.
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

  def make_sparse_batch(instance_indices, instance_values, max_index, dtype):
    """Converts a list of sparse instances into a sparse batch.

    Takes lists representing the indices and values of N sparse instances and
    concatenates them along the 0'th dimension into a sparse batch of size N.

    Args:
      instance_indices: A list of N iterables, each containing the sparse tensor
        indices for an instance.
      instance_values: A list of N iterables, each containing the sparse tensor
        values for an instance.
      max_index: An int representing the maximum index in `instance_indices`.
      dtype: dtype of the sparse tensor values.

    Returns:
      A `SparseTensorValue` representing a batch of N sparse instances.
    """
    batch_indices = make_batch_indices(instance_indices)
    batch_values = list(itertools.chain.from_iterable(instance_values))
    if produce_eager_tensors:
      batch_values = tf.constant(batch_values, dtype=dtype)
    batch_shape = (len(instance_indices), max_index)
    return tf.compat.v1.SparseTensorValue(
        indices=batch_indices, values=batch_values, dense_shape=batch_shape)

  if produce_eager_tensors and not tf.executing_eagerly():
    raise RuntimeError(
        'Eager Tensors were requested but eager mode was not enabled.')
  result = []
  feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
  for name in column_names:
    spec = feature_spec[name]
    # TODO(abrao): Validate dtypes, shapes etc.
    if isinstance(spec, tf.io.FixedLenFeature):
      feed_value = [instance[name] for instance in instances]

    elif isinstance(spec, tf.io.VarLenFeature):
      values = [[] if instance[name] is None else instance[name]
                for instance in instances]
      indices = [range(len(value)) for value in values]
      max_index = max([len(value) for value in values])
      feed_value = make_sparse_batch(indices, values, max_index, spec.dtype)

    elif isinstance(spec, tf.io.SparseFeature):
      # TODO(KesterTong): Add support for N-d SparseFeatures.
      max_index = spec.size
      indices, values = [], []
      for instance in instances:
        instance_indices = instance[spec.index_key]
        instance_values = instance[spec.value_key]
        check_valid_sparse_tensor(
            instance_indices, instance_values, max_index, name)
        indices.append(instance_indices)
        values.append(instance_values)
      feed_value = make_sparse_batch(indices, values, max_index, spec.dtype)

    else:
      raise ValueError('Invalid feature spec {}.'.format(spec))
    if produce_eager_tensors:
      if isinstance(feed_value, tf.compat.v1.SparseTensorValue):
        feed_value = tf.sparse.SparseTensor.from_value(feed_value)
      else:
        feed_value = tf.constant(feed_value, dtype=spec.dtype)
    result.append(feed_value)

  return result


def to_instance_dicts(schema, fetches):
  """Converts fetches to the internal batch format.

  Maps the values fetched by `tf.Session.run` or returned by a tf.function to
  the internal batch format.

  Args:
    schema: A `Schema` proto.
    fetches: A dict representing a batch of data, either as returned by
      `Session.run` or eager tensors.

  Returns:
    A list of dicts where each dict is an in-memory representation of an
        instance.

  Raises:
    ValueError: If `schema` is invalid.
  """

  def decompose_sparse_batch(sparse_value):
    """Decomposes a sparse batch into a list of sparse instances.

    Args:
      sparse_value: A `SparseTensor` or `SparseTensorValue` representing a batch
        of N sparse instances. The indices of the SparseTensorValue are expected
        to be sorted by row order.

    Returns:
      A tuple (instance_indices, instance_values) where the elements are lists
      of N lists representing the indices and values, respectively, of the
      instances in the batch.

    Raises:
      ValueError: If `sparse_value` is neither `SparseTensor` nor
        `SparseTensorValue`.
      ValueError: If `sparse_value` contains out-of-order indices.
    """
    if isinstance(sparse_value, tf.sparse.SparseTensor):
      batch_indices, batch_values, batch_shape = (
          sparse_value.indices.numpy(), sparse_value.values.numpy(),
          sparse_value.dense_shape.numpy())
    elif isinstance(sparse_value, tf.compat.v1.SparseTensorValue):
      batch_indices, batch_values, batch_shape = sparse_value
    else:
      raise ValueError(
          'Expected SparseTensor or SparseTensorValue , but got {}'.format(
              sparse_value))

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
  feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
  for name, tensor_or_value in six.iteritems(fetches):
    spec = feature_spec[name]
    if isinstance(spec, tf.io.FixedLenFeature):
      value = tensor_or_value.numpy() if isinstance(
          tensor_or_value, tf.Tensor) else tensor_or_value
      batch_dict[name] = [value[i] for i in range(value.shape[0])]
      batch_sizes[name] = value.shape[0]

    elif isinstance(spec, tf.io.VarLenFeature):
      instance_indices, instance_values = decompose_sparse_batch(
          tensor_or_value)
      for indices in instance_indices:
        if len(indices.shape) > 1 or np.any(indices != np.arange(len(indices))):
          raise ValueError('Encountered a SparseTensorValue that cannot be '
                           'decoded by ListColumnRepresentation.\n'
                           '"{}" : {}'.format(name, tensor_or_value))
      batch_dict[name] = instance_values
      batch_sizes[name] = len(instance_values)

    elif isinstance(spec, tf.io.SparseFeature):
      # TODO(abrao): Add support for N-d SparseFeatures.
      instance_indices, instance_values = decompose_sparse_batch(
          tensor_or_value)
      batch_dict[spec.index_key] = instance_indices
      batch_dict[spec.value_key] = instance_values
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
      for name, tensor in six.iteritems(tensors)
  }


def _copy_tensor(tensor):
  return tf.identity(tensor, name='{}_copy'.format(tensor.op.name))


def _copy_tensor_or_composite_tensor(tensor):
  if isinstance(tensor, composite_tensor.CompositeTensor):
    return tf.nest.map_structure(_copy_tensor, tensor, expand_composites=True)
  return _copy_tensor(tensor)


# TODO(b/149997088): Split into two APIs one that will just trace the
# `preprocessing_fn` using tf.function as is and another that will return
# specific outputs requested for.
def get_traced_transform_fn(preprocessing_fn,
                            input_signature,
                            base_temp_dir,
                            tensor_replacement_map=None,
                            output_keys_to_name_map=None):
  """Get preprocessing_fn traced using tf.function.

  Args:
    preprocessing_fn: A user defined python function to be traced.
    input_signature: `tf.TypeSpec`s describing the inputs to the
      `preprocessing_fn`.
    base_temp_dir: Base path to write any dummy assets to during tracing.
    tensor_replacement_map: (Optional) A map from placeholder tensor names to
      their evaluated replacement tensors.
    output_keys_to_name_map: (Optional) A map from output dictionary keys to the
      names of the tensors that they represent.

  Returns:
    A tf.function object representing a function with the same input signature
    as `preprocessing_fn`.
    If `output_keys_to_name_map` is None or there are no more TFT analyzers to
    evaluate in the `preprocessing_fn`, the output signature of this
    tf.function
    is the same as the `preprocessing_fn`.
    Otherwise, its output signature contains the keys in
    `output_keys_to_name_map` and the tensor represented by the corresponding
    dictionary values.
  """

  assert all(
      [isinstance(s, tf.TypeSpec) for s in six.itervalues(input_signature)])

  @tf.function(input_signature=[input_signature])
  def transform_fn(inputs):
    graph = ops.get_default_graph()
    # If any analyzers have already been evaluated, pass them using the
    # `graph_context.TFGraphContext`. This will be used in place of the analyzer
    # nodes.
    with graph_context.TFGraphContext(
        temp_dir=base_temp_dir, evaluated_replacements=tensor_replacement_map):
      transformed_features = preprocessing_fn(inputs)
    # An empty `TENSOR_REPLACEMENTS` collection symbolizes that there is no
    # analyzer left for Transform to evaluate. Either if this collection is
    # empty or if no specific outputs have been requested, return
    # the same output as `preprocessing_fn` (i.e, transformed_features).
    if (output_keys_to_name_map is None or
        not graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS)):
      return transformed_features
    else:
      return {
          key: graph.get_tensor_by_name(value)
          for key, value in six.iteritems(output_keys_to_name_map)
      }

  return transform_fn


def _trace_preprocessing_fn_v1(preprocessing_fn, specs):
  """Trace TF1 graph for `preprocessing_fn`."""
  with tf.compat.v1.Graph().as_default() as graph:
    with tf.compat.v1.name_scope('inputs'):
      structured_inputs = batched_placeholders_from_specs(specs)
      # In order to avoid a bug where import_graph_def fails when the
      # input_map and return_elements of an imported graph are the same
      # (b/34288791), we avoid using the placeholder of an input column as an
      # output of a graph. We do this by applying tf.identity to all inputs of
      # the preprocessing_fn.  Note this applies at the level of raw tensors.
      # TODO(b/34288791): Remove this workaround and use a shallow copy of
      # inputs instead.  A shallow copy is needed in case
      # self._preprocessing_fn mutates its input.
      copied_inputs = copy_tensors(structured_inputs)

    structured_outputs = preprocessing_fn(copied_inputs)
  return graph, structured_inputs, structured_outputs


def _trace_preprocessing_fn_v2(preprocessing_fn, specs, base_temp_dir):
  """Trace TF2 graph for `preprocessing_fn`."""
  concrete_fn = get_traced_transform_fn(preprocessing_fn, specs,
                                        base_temp_dir).get_concrete_function()
  graph = concrete_fn.graph
  num_captures = len(graph.internal_captures)
  graph_inputs = copy.copy(graph.inputs)
  # Only consider user provided inputs.
  if num_captures > 0:
    graph_inputs = graph_inputs[:-num_captures]
  # Pack tensors in `graph_inputs` into the structure specified by `specs`.
  structured_inputs = tf.nest.pack_sequence_as(
      structure=specs, flat_sequence=graph_inputs, expand_composites=True)
  return graph, structured_inputs, concrete_fn.structured_outputs


def trace_preprocessing_function(preprocessing_fn,
                                 input_specs,
                                 use_tf_compat_v1,
                                 base_temp_dir=None):
  """Trace graph for `preprocessing_fn`.

  Args:
    preprocessing_fn: A user defined python function to be traced.
    input_specs: A dictionary from input feature name to its FeatureSpec or
      TypeSpec. If use_tf_compat_v1 is `False`, input_specs must be a dictionary
      of TypeSpecs.
    use_tf_compat_v1: (Optional) If `True`, the `preprocessing_fn` is traced as
      a TF 1.x graph. Else, it is traced using tf.function.
    base_temp_dir: (Optional) Base path to write any dummy assets to during
      tracing. Required when `use_tf_compat_v1` is `False`.

  Returns:
    A tuple of:

      0. the graph representing the traced `preprocessing_fn`
      1. the graph's structured inputs
      2. the graph's structured outputs

  """
  if use_tf_compat_v1:
    return _trace_preprocessing_fn_v1(preprocessing_fn, input_specs)
  else:
    return _trace_preprocessing_fn_v2(preprocessing_fn, input_specs,
                                      base_temp_dir)
