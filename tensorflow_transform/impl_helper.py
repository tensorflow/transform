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

import os
import re
from typing import Dict, List, Tuple, Union

# GOOGLE-INITIALIZATION

import numpy as np
import six
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import common_types
from tensorflow_transform import graph_context
from tensorflow_transform import schema_inference
from tensorflow_transform import tf2_utils
from tensorflow_transform import tf_utils
from tensorflow_transform.output_wrapper import TFTransformOutput
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.tracking import tracking
# pylint: enable=g-direct-tensorflow-import

_SparseTensorValueType = Union[tf.SparseTensor, tf.compat.v1.SparseTensorValue]
_SparseComponentType = List[np.ndarray]

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
    sanitized_name = _sanitize_scope_name(name)
    with tf.name_scope(sanitized_name):
      return tf.nest.map_structure(
          lambda tspec: tf.raw_ops.Placeholder(  # pylint: disable=g-long-lambda
              dtype=tspec.dtype,
              shape=tspec.shape,
              name=sanitized_name),
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
    shape = [None] + feature_spec.size if isinstance(
        feature_spec.size, list) else [None, feature_spec.size]
    return tf.compat.v1.sparse_placeholder(
        feature_spec.dtype, shape, name=scope_name)

  raise ValueError('Unsupported feature spec: {}({}) for feature {}'
                   .format(feature_spec, type(feature_spec), name))


def _extract_sparse_components(
    sparse_value: _SparseTensorValueType
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  if isinstance(sparse_value, tf.SparseTensor):
    return (sparse_value.indices.numpy(), sparse_value.values.numpy(),
            sparse_value.dense_shape.numpy())
  elif isinstance(sparse_value, tf.compat.v1.SparseTensorValue):
    return sparse_value
  else:
    raise ValueError(
        'Expected SparseTensor or SparseTensorValue , but got {}'.format(
            sparse_value))


def _decompose_sparse_or_varlen_batch(
    sparse_value: _SparseTensorValueType, should_decode_as_varlen: bool
) -> Tuple[Union[_SparseComponentType, List[_SparseComponentType]],
           _SparseComponentType]:
  """Decomposes a sparse batch into a list of sparse/varlen instances.

  Args:
    sparse_value: A `SparseTensor` or `SparseTensorValue` representing a batch
      of N sparse instances. The indices of the SparseTensorValue are expected
      to be sorted by row order.
    should_decode_as_varlen: A bool indicating if the `sparse_value` should be
      decomposed as a varlen batch instead of sparse.

  Returns:
    A tuple (instance_indices, instance_values) where the elements are lists
    of N ndarrays representing the indices and values, respectively, of the
    instances in the batch. If the batch is decoded as sparse, the
    `instance_indices` will include an ndarray per dimension.

  Raises:
    ValueError: If `sparse_value` is neither `SparseTensor` nor
      `SparseTensorValue`.
    ValueError: If `sparse_value` contains out-of-order indices.
  """
  batch_indices, batch_values, batch_shape = _extract_sparse_components(
      sparse_value)
  batch_size = batch_shape[0]
  instance_rank = len(batch_shape) - 1

  # Preallocate lists of length batch_size, initialized to empty ndarrays,
  # representing the indices and values of instances. We can reuse the return
  # value of _get_empty_array here because it is immutable.
  instance_values = [_get_empty_array(batch_values.dtype)] * batch_size
  if should_decode_as_varlen:
    instance_indices = [_get_empty_array(batch_indices.dtype)] * batch_size
  else:
    instance_indices = [[_get_empty_array(batch_indices.dtype)] * instance_rank
                        for _ in range(batch_size)]

  # Iterate over the rows in the batch. At each row, consume all the elements
  # that belong to that row.
  current_offset = 0
  for current_row in range(batch_size):
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
    elif should_decode_as_varlen:
      instance_indices[current_row] = batch_indices[start_offset:current_offset,
                                                    1:]
      if instance_rank == 1:
        # In this case indices will have length 1, so for convenience we
        # reshape from [-1, 1] to [-1].
        current_row_indices = instance_indices[current_row]  # type: np.ndarray
        instance_indices[current_row] = current_row_indices.reshape([-1])
      instance_values[current_row] = batch_values[start_offset:current_offset]
    else:
      instance_values[current_row] = batch_values[start_offset:current_offset]
      num_values = len(instance_values[current_row])

      for dimension in range(instance_rank):
        # We use dimension + 1 because we're ignoring the batch dimension.
        current_indices = batch_indices[current_row:current_row + num_values,
                                        dimension + 1]
        instance_indices[current_row][dimension] = current_indices

  return instance_indices, instance_values


def _handle_varlen_batch(tensor_or_value: _SparseTensorValueType,
                         name: str) -> _SparseComponentType:
  """Decomposes a varlen tensor value into sparse tensor components."""
  instance_indices, instance_values = _decompose_sparse_or_varlen_batch(
      tensor_or_value, should_decode_as_varlen=True)
  for indices in instance_indices:  # type: np.ndarray
    if len(indices.shape) > 1 or np.any(indices != np.arange(len(indices))):
      raise ValueError('Encountered a SparseTensorValue that cannot be '
                       'decoded by ListColumnRepresentation.\n'
                       '"{}" : {}'.format(name, tensor_or_value))
  return instance_values


def _handle_sparse_batch(
    tensor_or_value: _SparseTensorValueType, spec: common_types.FeatureSpecType,
    name: str
) -> Dict[str, Union[List[_SparseComponentType], _SparseComponentType]]:
  """Decomposes a sparse tensor value into sparse tensor components."""
  if len(spec.index_key) == 1:
    index_keys = spec.index_key[0]
    instance_indices, instance_values = _decompose_sparse_or_varlen_batch(
        tensor_or_value, should_decode_as_varlen=True)
  else:
    index_keys = spec.index_key
    instance_indices, instance_values = _decompose_sparse_or_varlen_batch(
        tensor_or_value, should_decode_as_varlen=False)
  result = {}
  if isinstance(index_keys, list):
    assert isinstance(instance_indices, list)
    for key, indices in zip(index_keys, zip(*instance_indices)):
      result[key] = indices
  else:
    result[index_keys] = instance_indices
  result[spec.value_key] = instance_values
  _check_valid_sparse_tensor(instance_indices, instance_values, spec.size, name)
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
      instance_values = _handle_varlen_batch(tensor_or_value, name)
      batch_dict[name] = instance_values
      batch_sizes[name] = len(instance_values)

    elif isinstance(spec, tf.io.SparseFeature):
      batch_dict_update = _handle_sparse_batch(tensor_or_value, spec, name)
      batch_dict.update(batch_dict_update)
      batch_sizes[name] = len(batch_dict_update[spec.value_key])

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
def _check_valid_sparse_tensor(indices: Union[_SparseComponentType,
                                              List[_SparseComponentType]],
                               values: _SparseComponentType,
                               size: Union[int, List[int]], name: str):
  """Validates sparse tensor components."""
  # Check that all indices are in range.
  for current_indices in indices:
    if isinstance(current_indices, np.ndarray):
      current_indices = [current_indices]
    for dim, indices_array in enumerate(current_indices):
      if indices_array.size and size[dim] >= 0:
        i_min, i_max = min(indices_array), max(indices_array)
        if i_min < 0 or i_max >= size[dim]:
          i_bad = i_min if i_min < 0 else i_max
          raise ValueError(
              'Sparse column {} has index {} out of range [0, {})'.format(
                  name, i_bad, size[dim]))

  if len(indices) != len(values):
    raise ValueError(
        'Sparse column {} has indices and values of different lengths: '
        'values: {}, indices: {}'.format(name, values, indices))


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
    # The user defined `preprocessing_fn` may directly modify its inputs which
    # is not allowed in a tf.function. Hence, we make a copy here.
    inputs_copy = tf_utils.copy_tensors(inputs)
    with graph_context.TFGraphContext(
        temp_dir=base_temp_dir, evaluated_replacements=tensor_replacement_map):
      transformed_features = preprocessing_fn(inputs_copy)
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
      copied_inputs = tf_utils.copy_tensors(structured_inputs)

    structured_outputs = preprocessing_fn(copied_inputs)
  return graph, structured_inputs, structured_outputs


def _trace_preprocessing_fn_v2(preprocessing_fn, specs, base_temp_dir):
  """Trace TF2 graph for `preprocessing_fn`."""
  concrete_fn = get_traced_transform_fn(preprocessing_fn, specs,
                                        base_temp_dir).get_concrete_function()
  return (concrete_fn.graph,
          tf2_utils.get_structured_inputs_from_func_graph(concrete_fn.graph),
          concrete_fn.structured_outputs)


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


# TODO(b/160550490): Move to saved.saved_transform_io_v2.
def trace_and_write_v2_saved_model(saved_model_dir, preprocessing_fn,
                                   input_signature, base_temp_dir,
                                   tensor_replacement_map,
                                   output_keys_to_name_map):
  """Writes out a SavedModelV2 with preprocessing_fn traced using tf.function.

  The SavedModel written contains a method called `transform_fn` that
  represents the traced `preprocessing_fn`. Additionally, if this is the final
  SavedModel being written out, it will contain a method called `metadata_fn`
  that provides deferred schema annotations.

  Args:
    saved_model_dir: Path to write SavedModel to.
    preprocessing_fn: A user defined python function to be traced.
    input_signature: TypeSpecs describing the inputs to the `preprocessing_fn`.
    base_temp_dir: Base path to write temporary artifacts to.
    tensor_replacement_map: A map from placeholder tensor names to their
      evaluated replacement tensors.
    output_keys_to_name_map: A map from output dictionary keys to the names of
      the tensors that they represent.

  Returns:
    A tuple containing a pair of `tf.ConcreteFunction`s:
      1. The traced preprocessing_fn.
      2. A metadata_fn that returns a dictionary containing the deferred
      annotations added to the graph when invoked with any valid input.
  """

  module = tf.Module()
  transform_fn = get_traced_transform_fn(
      preprocessing_fn,
      input_signature,
      base_temp_dir,
      tensor_replacement_map=tensor_replacement_map,
      output_keys_to_name_map=output_keys_to_name_map)
  metadata_fn = None

  resource_tracker = tracking.ResourceTracker()
  created_variables = []

  def _variable_creator(next_creator, **kwargs):
    var = next_creator(**kwargs)
    created_variables.append(var)
    return var

  # TODO(b/164921571): Handle generic Trackable objects.
  # Trace the `transform_fn` and `metadata_fn` to gather any resources in it
  # using the resource_tracker. These are then assigned to `module.resources`
  # and tracked before exporting to SavedModel.
  with tracking.resource_tracker_scope(
      resource_tracker), tf.variable_creator_scope(_variable_creator):
    concrete_transform_fn = transform_fn.get_concrete_function()
    concrete_metadata_fn = None
    # If the `TENSOR_REPLACEMENTS` graph collection is empty, all TFT analyzers
    # in the `preprocessing_fn` have already been evaluated.
    if not concrete_transform_fn.graph.get_collection(
        analyzer_nodes.TENSOR_REPLACEMENTS):
      metadata_fn = schema_inference.get_traced_metadata_fn(
          tensor_replacement_map,
          preprocessing_fn,
          input_signature,
          base_temp_dir,
          evaluate_schema_overrides=True)
      concrete_metadata_fn = metadata_fn.get_concrete_function()

  # Save ConcreteFunction when possible since the above workaround won't work if
  # the tf.function is retraced.
  if tf.compat.forward_compatible(2020, 10, 8):
    module.transform_fn = concrete_transform_fn
    module.metadata_fn = concrete_metadata_fn
  else:
    module.transform_fn = transform_fn
    module.metadata_fn = metadata_fn

  # Any variables created need to be explicitly tracked.
  module.created_variables = created_variables
  # Resources need to be explicitly tracked.
  module.resources = resource_tracker.resources
  # TODO(b/158011374) - Stop explicitly tracking initializers. Tracking the
  # table should be sufficient.
  initializers = []
  for resource in module.resources:
    if isinstance(resource, lookup_ops.InitializableLookupTableBase):
      initializers.append(resource._initializer)  # pylint: disable=protected-access
  module.initializers = initializers
  module.assets = [
      temporary_analyzer_info.asset
      for temporary_analyzer_info in concrete_transform_fn.graph.get_collection(
          analyzer_nodes.ASSET_REPLACEMENTS)
  ]
  tf.saved_model.save(module, saved_model_dir)
  return concrete_transform_fn, concrete_metadata_fn


def _assert_no_analyzers_in_graph(graph):
  if graph.get_collection(analyzer_nodes.TENSOR_REPLACEMENTS):
    raise RuntimeError('TFT analyzers found when tracing the given '
                       '`preprocessing_fn`. Please use '
                       '`tft.beam.AnalyzeDataset` to analyze this function.')


def analyze_in_place(preprocessing_fn, force_tf_compat_v1, feature_specs,
                     type_specs, transform_output_path):
  """Analyzes the `preprocessing_fn` in-place without looking at the data.

  This should only be used if the `preprocessing_fn` contains no TFT
  analyzers or TFT mappers that use analyzers.

  Writes out a transform function and transformed metadata to subdirs under
  `transform_output_path`.

  Args:
    preprocessing_fn: The tf.Transform preprocessing_fn.
    force_tf_compat_v1: If True, call Transform's API to use Tensorflow in
      tf.compat.v1 mode.
    feature_specs: a Dict from input feature key to its feature spec.
    type_specs: a Dict from input feature key to its type spec.
    transform_output_path: An absolute path to write the output to.

  Raises:
    RuntimeError if `preprocessing_fn` contains TFT analyzers.
  """
  use_tf_compat_v1 = tf2_utils.use_tf_compat_v1(force_tf_compat_v1)
  transform_fn_path = os.path.join(transform_output_path,
                                   TFTransformOutput.TRANSFORM_FN_DIR)
  if use_tf_compat_v1:
    graph, structured_inputs, structured_outputs = (
        trace_preprocessing_function(
            preprocessing_fn, feature_specs, use_tf_compat_v1=use_tf_compat_v1))
    _assert_no_analyzers_in_graph(graph)
    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.tables_initializer())
      saved_transform_io.write_saved_transform_from_session(
          sess, structured_inputs, structured_outputs, transform_fn_path)

      transformed_metadata = dataset_metadata.DatasetMetadata(
          schema=schema_inference.infer_feature_schema(structured_outputs,
                                                       graph, sess))
  else:
    concrete_transform_fn, concrete_metadata_fn = (
        trace_and_write_v2_saved_model(
            saved_model_dir=transform_fn_path,
            preprocessing_fn=preprocessing_fn,
            input_signature=type_specs,
            base_temp_dir=None,
            tensor_replacement_map=None,
            output_keys_to_name_map=None))
    _assert_no_analyzers_in_graph(concrete_transform_fn.graph)
    # This should be a no-op as if concrete_metadata_fn is None,
    # `_assert_no_analyzers_in_graph` should have raised an error.
    assert concrete_metadata_fn
    structured_outputs = tf.nest.pack_sequence_as(
        structure=concrete_transform_fn.structured_outputs,
        flat_sequence=concrete_transform_fn.outputs,
        expand_composites=True)
    transformed_metadata = dataset_metadata.DatasetMetadata(
        schema=schema_inference.infer_feature_schema_v2(
            structured_outputs,
            concrete_metadata_fn,
            evaluate_schema_overrides=True))
  transformed_metadata_dir = os.path.join(
      transform_output_path, TFTransformOutput.TRANSFORMED_METADATA_DIR)
  metadata_io.write_metadata(transformed_metadata, transformed_metadata_dir)
