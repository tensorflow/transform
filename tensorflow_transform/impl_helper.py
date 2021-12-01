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
"""Helper/utility functions that a tf-transform implementation would find handy."""

import os
import re
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

from absl import logging
import numpy as np
import pyarrow as pa

import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import annotators
from tensorflow_transform import common_types
from tensorflow_transform import graph_context
from tensorflow_transform import graph_tools
from tensorflow_transform import schema_inference
from tensorflow_transform import tf2_utils
from tensorflow_transform import tf_utils
from tensorflow_transform.output_wrapper import TFTransformOutput
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.saved import saved_transform_io_v2
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.tfxio import tensor_to_arrow
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import function
from tensorflow.python.framework import ops
# pylint: enable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2

_CompositeInstanceComponentType = np.ndarray
_CompositeComponentType = List[_CompositeInstanceComponentType]

_CACHED_EMPTY_ARRAY_BY_DTYPE = {}
_VALID_SCOPE_REGEX = re.compile('^[A-Za-z0-9]*$')
_INVALID_SCOPE_CHAR = re.compile('[^A-Za-z0-9_.\\-/>]')

METADATA_DIR_NAME = '.tft_metadata'


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
  if not (all([_is_feature_spec(s) for s in specs.values()]) or
          all([isinstance(s, tf.TypeSpec) for s in specs.values()])):
    raise TypeError('Specs must be all tf.TypeSpecs or feature specs. '
                    'Mixing is not allowed. Got: {}'.format(specs))

  result = {}
  for name, spec in specs.items():
    if isinstance(spec, tf.RaggedTensorSpec):
      # TODO(b/159717195): clean up protected-access
      spec_dtype = spec._dtype  # pylint: disable=protected-access
    else:
      spec_dtype = spec.dtype
    if spec_dtype not in (tf.int64, tf.float32, tf.string):
      raise ValueError('Feature {} ({}, {}) had invalid dtype'.format(
          name, spec, type(spec)))
    if isinstance(spec, tf.TypeSpec):
      result[name] = _batched_placeholder_from_typespec(name, spec)
    else:
      result[name] = _batched_placeholder_from_feature_spec(name, spec)

  return result


def _is_feature_spec(spec):
  if isinstance(
      spec, (tf.io.VarLenFeature, tf.io.SparseFeature, tf.io.FixedLenFeature)):
    return True
  return common_types.is_ragged_feature(spec)


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

  raise ValueError('Unsupported feature spec: {}({}) for feature {}'.format(
      feature_spec, type(feature_spec), name))


def _extract_sparse_components(
    sparse_value: common_types.SparseTensorValueType
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  if isinstance(sparse_value, tf.SparseTensor):
    return (np.asarray(sparse_value.indices), np.asarray(sparse_value.values),
            np.asarray(sparse_value.dense_shape))
  elif isinstance(sparse_value, tf.compat.v1.SparseTensorValue):
    return sparse_value
  else:
    raise ValueError(
        'Expected SparseTensor or SparseTensorValue , but got {}'.format(
            sparse_value))


def _get_num_values_per_instance_in_sparse_batch(batch_indices: np.ndarray,
                                                 batch_size: int) -> List[int]:
  """Computes the number of values per instance of the batch."""
  result = [0] * batch_size
  for arr in batch_indices:
    result[arr[0]] += 1
  return result


def _decompose_sparse_batch(
    sparse_value: common_types.SparseTensorValueType
) -> Tuple[List[_CompositeComponentType], _CompositeComponentType]:
  """Decomposes a sparse batch into a list of sparse instances.

  Args:
    sparse_value: A `SparseTensor` or `SparseTensorValue` representing a batch
      of N sparse instances. The indices of the SparseTensorValue are expected
      to be sorted by row order.

  Returns:
    A tuple (instance_indices, instance_values) where the elements are lists
    of N ndarrays representing the indices and values, respectively, of the
    instances in the batch. The `instance_indices` include an ndarray per
    dimension.
  """
  batch_indices, batch_values, batch_shape = _extract_sparse_components(
      sparse_value)
  batch_size = batch_shape[0]
  instance_rank = len(batch_shape) - 1

  # Preallocate lists of length batch_size, initialized to empty ndarrays,
  # representing the indices and values of instances. We can reuse the return
  # value of _get_empty_array here because it is immutable.
  instance_values = [_get_empty_array(batch_values.dtype)] * batch_size
  instance_indices = [[_get_empty_array(batch_indices.dtype)] * instance_rank
                      for idx in range(batch_size)]

  values_per_instance = _get_num_values_per_instance_in_sparse_batch(
      batch_indices, batch_size)

  offset = 0
  for idx, num_values in enumerate(values_per_instance):
    if num_values < 1:
      continue
    instance_values[idx] = batch_values[offset:offset + num_values]
    for dim in range(instance_rank):
      # Skipping the first dimension since that is the batch dimension.
      instance_indices[idx][dim] = batch_indices[offset:offset + num_values,
                                                 dim + 1]
    offset += num_values
  return instance_indices, instance_values


def _decompose_varlen_batch(
    sparse_value: common_types.SparseTensorValueType
) -> Tuple[_CompositeComponentType, _CompositeComponentType]:
  """Decomposes a sparse batch into a list of sparse/varlen instances.

  Args:
    sparse_value: A `SparseTensor` or `SparseTensorValue` representing a batch
      of N sparse instances. The indices of the SparseTensorValue are expected
      to be sorted by row order.

  Returns:
    A tuple (instance_indices, instance_values) where the elements are lists
    of N ndarrays representing the indices and values, respectively, of the
    instances in the batch.

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
  instance_indices = [_get_empty_array(batch_indices.dtype)] * batch_size

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
    else:
      instance_indices[current_row] = batch_indices[start_offset:current_offset,
                                                    1:]
      if instance_rank == 1:
        # In this case indices will have length 1, so for convenience we
        # reshape from [-1, 1] to [-1].
        current_row_indices = instance_indices[current_row]  # type: np.ndarray
        instance_indices[current_row] = current_row_indices.reshape([-1])
      instance_values[current_row] = batch_values[start_offset:current_offset]
  return instance_indices, instance_values


def _handle_varlen_batch(tensor_or_value: common_types.SparseTensorValueType,
                         name: str) -> _CompositeComponentType:
  """Decomposes a varlen tensor value into sparse tensor components."""
  instance_indices, instance_values = _decompose_varlen_batch(tensor_or_value)
  for indices in instance_indices:  # type: np.ndarray
    if len(indices.shape) > 1 or np.any(indices != np.arange(len(indices))):
      raise ValueError('Encountered a SparseTensorValue that cannot be '
                       'decoded by ListColumnRepresentation.\n'
                       '"{}" : {}'.format(name, tensor_or_value))
  return instance_values


def _handle_sparse_batch(
    tensor_or_value: common_types.SparseTensorValueType,
    spec: common_types.FeatureSpecType, name: str
) -> Dict[str, Union[List[_CompositeComponentType], _CompositeComponentType]]:
  """Decomposes a sparse tensor value into sparse tensor components."""
  if len(spec.index_key) == 1:
    index_keys = spec.index_key[0]
    instance_indices, instance_values = _decompose_varlen_batch(tensor_or_value)
  else:
    index_keys = spec.index_key
    instance_indices, instance_values = _decompose_sparse_batch(tensor_or_value)
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


def _get_ragged_instance_component(
    component_batch: _CompositeInstanceComponentType, batch_splits: np.ndarray,
    instance_idx: int) -> _CompositeInstanceComponentType:
  """Extracts an instance component from a flat batch with given splits."""
  instance_begin = batch_splits[instance_idx]
  instance_end = batch_splits[instance_idx + 1]
  return component_batch[instance_begin:instance_end]


def _get_num_inner_uniform_elements(spec: common_types.RaggedFeature,
                                    name: str) -> int:
  """Extracts the number of elements in inner dimensions of a ragged feature.

  Also validates partitions in the feature spec.

  Args:
    spec: A ragged feature spec.
    name: Name of the feature that the spec belongs to.

  Returns:
    A number of elements in the inner uniform dimensions per one innermost
    ragged element. It is calculated as a product of inner uniform dimension
    sizes.

  Raises:
    NotImplementedError: if the spec contains partitions other than `RowLengths`
      and `UniformRowLengths` or there is a `RowLengths` partition that follows
      a `UniformRowLengths` partition.
  """
  result = 1
  uniform_partition = False
  for partition in spec.partitions:
    if isinstance(partition, tf.io.RaggedFeature.RowLengths):  # pytype: disable=attribute-error
      if uniform_partition:
        # Upstream and downstream logic only supports inner uniform dimensions.
        # This function will have to be extended to support other cases if
        # needed as well, so we fail with an explicit error for now.
        raise NotImplementedError(
            'Only inner partitions are allowed to be uniform, unsupported '
            'partition sequence for feature "{}" with spec {}'.format(
                name, spec))
    elif isinstance(partition, tf.io.RaggedFeature.UniformRowLength):  # pytype: disable=attribute-error
      uniform_partition = True
      result *= partition.length
    else:
      raise ValueError(
          'Only `RowLengths` and `UniformRowLengths` partitions of ragged '
          'features are supported, got {} for ragged feature "{}" with spec {}'
          .format(partition, name, spec))
  return result


def _handle_ragged_batch(tensor_or_value: common_types.RaggedTensorValueType,
                         spec: common_types.FeatureSpecType,
                         name: str) -> Dict[str, _CompositeComponentType]:
  """Decomposes a ragged tensor or value into ragged tensor components."""
  if isinstance(tensor_or_value, tf.RaggedTensor):
    nested_row_splits = tuple(
        x.numpy() for x in tensor_or_value.nested_row_splits)
    flat_values = np.ravel(tensor_or_value.flat_values.numpy())
  elif isinstance(tensor_or_value, tf.compat.v1.ragged.RaggedTensorValue):
    nested_row_splits = tensor_or_value.nested_row_splits
    flat_values = np.ravel(tensor_or_value.flat_values)
  else:
    raise ValueError('Expected RaggedTensor or RaggedTensorValue , but '
                     'got {}'.format(tensor_or_value))

  result = {}
  # The outermost row split represents batch dimension.
  batch_splits = nested_row_splits[0]
  batch_size = len(batch_splits) - 1
  inner_uniform_elements = _get_num_inner_uniform_elements(spec, name)

  # Iterate over all but batch dimension splits. Note that
  # `nested_row_splits[1:]` may be shorter than the list of partitions in the
  # presense of inner uniform partitions. Partition types and sequence is
  # validated in `_get_num_inner_uniform_elements`.
  for row_splits, partition in zip(nested_row_splits[1:], spec.partitions):
    assert isinstance(partition, tf.io.RaggedFeature.RowLengths), partition  # pytype: disable=attribute-error
    row_lengths = (row_splits[1:] - row_splits[:-1]) * inner_uniform_elements
    result[partition.key] = [
        _get_ragged_instance_component(row_lengths, batch_splits, idx)
        for idx in range(batch_size)
    ]

    # Translate batch split indices for the current dimension to the
    # next dimension.
    batch_splits = row_splits[batch_splits]

  # Split flat values according to the innermost dimension batch splits.
  result[spec.value_key] = [
      _get_ragged_instance_component(flat_values,
                                     batch_splits * inner_uniform_elements, idx)
      for idx in range(batch_size)
  ]
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
  for name, tensor_or_value in fetches.items():
    spec = feature_spec[name]
    if isinstance(spec, tf.io.FixedLenFeature):
      value = np.asarray(tensor_or_value)
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

    elif common_types.is_ragged_feature(spec):
      batch_dict_update = _handle_ragged_batch(tensor_or_value, spec, name)
      batch_dict.update(batch_dict_update)
      batch_sizes[name] = len(batch_dict_update[spec.value_key])

    else:
      raise ValueError('Invalid feature spec {}.'.format(spec))

  # Check batch size is the same for each output.  Note this assumes that
  # fetches is not empty.
  batch_size = next(iter(batch_sizes.values()))
  for name, batch_size_for_name in batch_sizes.items():
    if batch_size_for_name != batch_size:
      raise ValueError(
          'Inconsistent batch sizes: "{}" had batch dimension {}, "{}" had'
          ' batch dimension {}'.format(name, batch_size_for_name,
                                       next(iter(batch_sizes.keys())),
                                       batch_size))

  # The following is the simplest way to convert batch_dict from a dict of
  # iterables to a list of dicts.  It does this by first extracting the values
  # of batch_dict, and reversing the order of iteration, then recombining with
  # the keys of batch_dict to create a dict.
  return [
      dict(zip(batch_dict, instance_values))
      for instance_values in zip(*batch_dict.values())
  ]


def _tf_dtype_to_arrow_type(dtype: tf.DType) -> pa.DataType:
  """Maps a tf data type to a pyarrow data type."""
  if dtype == tf.string:
    return pa.large_binary()
  elif dtype == tf.int64:
    return pa.int64()
  elif dtype == tf.float32:
    return pa.float32()
  else:
    raise TypeError('Unable to handle data type {}'.format(dtype))


def get_type_specs_from_feature_specs(
    feature_specs: Dict[str, common_types.FeatureSpecType]
) -> Dict[str, tf.TypeSpec]:
  """Returns `tf.TypeSpec`s for the given feature specs.

  Returns a dictionary of type_spec with the same type and shape as defined by
  `feature_specs`.

  Args:
    feature_specs: A TensorFlow feature spec.

  Returns:
    A dictionary from strings to `tf.TensorSpec`, `tf.SparseTensorSpec` or
    `tf.RaggedTensorSpec`s.

  Raises:
    ValueError: If the feature spec contains feature types not supported.
  """
  result = {}
  for name, feature_spec in feature_specs.items():
    if isinstance(feature_spec, tf.io.FixedLenFeature):
      result[name] = tf.TensorSpec([None] + list(feature_spec.shape),
                                   feature_spec.dtype)
    elif isinstance(feature_spec, tf.io.VarLenFeature):
      result[name] = tf.SparseTensorSpec([None, None], feature_spec.dtype)
    elif isinstance(feature_spec, tf.io.SparseFeature):
      # `TensorsToRecordBatchConverter` ignores `SparseFeature`s since arbitrary
      # `SparseTensor`s are not yet supported. They are handled in
      # `convert_to_arrow`.
      # TODO(b/181868576): Handle `SparseFeature`s by the converter once the
      # support is implemented.
      pass
    elif common_types.is_ragged_feature(feature_spec):
      # Number of dimensions is number of partitions + 1 + 1 batch dimension.
      shape = [None, None]
      ragged_rank = 1
      for partition in feature_spec.partitions:
        if isinstance(partition, tf.io.RaggedFeature.UniformRowLength):  # pytype: disable=attribute-error
          shape.append(partition.length)
        else:
          shape.append(None)
          ragged_rank += 1
      result[name] = tf.RaggedTensorSpec(
          shape=shape,
          dtype=feature_spec.dtype,
          ragged_rank=ragged_rank,
          row_splits_dtype=feature_spec.row_splits_dtype)
    else:
      raise ValueError('Invalid feature spec {}.'.format(feature_spec))
  return result


def make_tensor_to_arrow_converter(
    schema: schema_pb2.Schema) -> tensor_to_arrow.TensorsToRecordBatchConverter:
  """Constructs a `tf.Tensor` to `pa.RecordBatch` converter."""
  feature_specs = schema_utils.schema_as_feature_spec(schema).feature_spec
  type_specs = get_type_specs_from_feature_specs(feature_specs)
  return tensor_to_arrow.TensorsToRecordBatchConverter(type_specs)


def convert_to_arrow(
    schema: schema_pb2.Schema,
    converter: tensor_to_arrow.TensorsToRecordBatchConverter,
    fetches: Dict[str, common_types.TensorValueType]
) -> Tuple[List[pa.Array], pa.Schema]:
  """Converts fetches to a list of pyarrow arrays and schema.

  Maps the values fetched by `tf.Session.run` or returned by a tf.function to
  pyarrow format.

  Args:
    schema: A `Schema` proto.
    converter: A `tf.Tensor` to `pa.RecordBatch` converter that contains
      `tf.TypeSpec`s of `FixedLen` and `VarLen` features. Note that the
      converter doesn't support general `SparseFeature`s, they are handled here.
    fetches: A dict representing a batch of data, either as returned by
      `Session.run` or eager tensors.

  Returns:
    A tuple of a list of pyarrow arrays and schema representing fetches.

  Raises:
    ValueError: If batch sizes are inconsistent.
  """

  tensors = {}
  sparse_arrays = []
  sparse_fields = []
  feature_specs = schema_utils.schema_as_feature_spec(schema).feature_spec
  for name, tensor_or_value in fetches.items():
    feature_spec = feature_specs[name]
    if isinstance(feature_spec, tf.io.SparseFeature):
      sparse_components = _handle_sparse_batch(tensor_or_value, feature_spec,
                                               name)
      values_type = pa.large_list(_tf_dtype_to_arrow_type(feature_spec.dtype))
      indices_type = pa.large_list(pa.int64())
      sparse_arrays.append(
          pa.array(
              sparse_components.pop(feature_spec.value_key), type=values_type))
      sparse_fields.append(pa.field(feature_spec.value_key, values_type))
      for indices_key, instance_indices in sparse_components.items():
        flat_indices = [np.ravel(indices) for indices in instance_indices]
        sparse_arrays.append(pa.array(flat_indices, type=indices_type))
        sparse_fields.append(pa.field(indices_key, indices_type))
    else:
      tensors[name] = tensor_or_value
  record_batch = converter.convert(tensors)
  arrow_schema = record_batch.schema
  for field in sparse_fields:
    arrow_schema = arrow_schema.append(field)

  return record_batch.columns + sparse_arrays, arrow_schema


# TODO(b/36040669): Consider moving this to where it can be shared with coders.
def _check_valid_sparse_tensor(indices: Union[_CompositeComponentType,
                                              List[_CompositeComponentType]],
                               values: _CompositeComponentType,
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
def get_traced_transform_fn(
    preprocessing_fn: Callable[[Mapping[str, common_types.TensorType]],
                               Mapping[str, common_types.TensorType]],
    input_signature: Mapping[str, tf.TypeSpec],
    tf_graph_context: graph_context.TFGraphContext,
    output_keys_to_name_map: Optional[Dict[str,
                                           str]] = None) -> function.Function:
  """Get preprocessing_fn traced using tf.function.

  Args:
    preprocessing_fn: A user defined python function to be traced.
    input_signature: `tf.TypeSpec`s describing the inputs to the
      `preprocessing_fn`.
    tf_graph_context: A `TFGraphContext` context manager to invoke the
      `preprocessing_fn` in.
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

  assert all([isinstance(s, tf.TypeSpec) for s in input_signature.values()])

  # TODO(b/177672051): Investigate performance impact of enabling autograph.
  @tf.function(input_signature=[input_signature], autograph=False)
  def transform_fn(inputs):
    graph = ops.get_default_graph()
    # If any analyzers have already been evaluated, pass them using the
    # `graph_context.TFGraphContext`. This will be used in place of the analyzer
    # nodes.
    # The user defined `preprocessing_fn` may directly modify its inputs which
    # is not allowed in a tf.function. Hence, we make a copy here.
    inputs_copy = tf_utils.copy_tensors(inputs)
    with tf_graph_context:
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
          for key, value in output_keys_to_name_map.items()
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
  tf_graph_context = graph_context.TFGraphContext(
      module_to_export=tf.Module(),
      temp_dir=base_temp_dir,
      evaluated_replacements=None)
  with annotators.object_tracker_scope(annotators.ObjectTracker()):
    concrete_fn = get_traced_transform_fn(
        preprocessing_fn, specs, tf_graph_context).get_concrete_function()
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


def _trace_and_write_transform_fn(
    saved_model_dir: str,
    preprocessing_fn: Callable[[Mapping[str, common_types.TensorType]],
                               Mapping[str, common_types.TensorType]],
    input_signature: Mapping[str, tf.TypeSpec], base_temp_dir: Optional[str],
    tensor_replacement_map: Optional[Dict[str, tf.Tensor]],
    output_keys_to_name_map: Optional[Dict[str,
                                           str]]) -> function.ConcreteFunction:
  """Trace `preprocessing_fn` and serialize to a SavedModel."""
  tf_graph_context = graph_context.TFGraphContext(
      module_to_export=tf.Module(),
      temp_dir=base_temp_dir,
      evaluated_replacements=tensor_replacement_map)
  transform_fn = get_traced_transform_fn(
      preprocessing_fn,
      input_signature,
      tf_graph_context,
      output_keys_to_name_map=output_keys_to_name_map)
  return saved_transform_io_v2.write_v2_saved_model(
      tf_graph_context.module_to_export, transform_fn, 'transform_fn',
      saved_model_dir)


def _trace_and_get_metadata(
    concrete_transform_fn: function.ConcreteFunction,
    structured_inputs: Mapping[str, common_types.TensorType],
    preprocessing_fn: Callable[[Mapping[str, common_types.TensorType]],
                               Mapping[str, common_types.TensorType]],
    base_temp_dir: Optional[str],
    tensor_replacement_map: Optional[Dict[str, tf.Tensor]]
) -> dataset_metadata.DatasetMetadata:
  """Compute and return metadata for the outputs of `concrete_transform_fn`."""
  tf_graph_context = graph_context.TFGraphContext(
      module_to_export=tf.Module(),
      temp_dir=base_temp_dir,
      evaluated_replacements=tensor_replacement_map)
  concrete_metadata_fn = schema_inference.get_traced_metadata_fn(
      preprocessing_fn,
      structured_inputs,
      tf_graph_context,
      evaluate_schema_overrides=True)
  return dataset_metadata.DatasetMetadata(
      schema=schema_inference.infer_feature_schema_v2(
          concrete_transform_fn.structured_outputs,
          concrete_metadata_fn,
          evaluate_schema_overrides=True))


def _validate_analyzers_fingerprint(
    baseline_analyzers_fingerprint: Mapping[str,
                                            graph_tools.AnalyzersFingerprint],
    graph: tf.Graph, structured_inputs: Mapping[str, common_types.TensorType]):
  """Validates analyzers fingerprint in `graph` is same as baseline."""
  analyzers_fingerprint = graph_tools.get_analyzers_fingerprint(
      graph, structured_inputs)
  error_msg = (
      'The order of analyzers in your `preprocessing_fn` appears to be '
      'non-deterministic. This can be fixed either by changing your '
      '`preprocessing_fn` such that tf.Transform analyzers are encountered '
      'in a deterministic order or by passing a unique name to each '
      'analyzer API call.')
  for analyzer in analyzers_fingerprint:
    if analyzer not in baseline_analyzers_fingerprint:
      prefix_msg = (f'Analyzer node ({analyzer}) not found in '
                    f'{baseline_analyzers_fingerprint.keys()}. ')
      raise RuntimeError(prefix_msg + error_msg)
    if (baseline_analyzers_fingerprint[analyzer].source_keys !=
        analyzers_fingerprint[analyzer].source_keys):
      raise RuntimeError(error_msg)

    if (baseline_analyzers_fingerprint[analyzer].unique_path_hash !=
        analyzers_fingerprint[analyzer].unique_path_hash):
      logging.warning(
          'Analyzer (%s) node\'s cache key varies on repeated tracing.'
          ' This warning is safe to ignore if you either specify `name` for all'
          ' analyzers or if the order in which they are invoked is'
          ' deterministic. If not, please file a bug with details.', analyzer)


def trace_and_write_v2_saved_model(
    saved_model_dir: str,
    preprocessing_fn: Callable[[Mapping[str, common_types.TensorType]],
                               Mapping[str, common_types.TensorType]],
    input_signature: Mapping[str, tf.TypeSpec], base_temp_dir: Optional[str],
    baseline_analyzers_fingerprint: Mapping[str,
                                            graph_tools.AnalyzersFingerprint],
    tensor_replacement_map: Optional[Dict[str, tf.Tensor]],
    output_keys_to_name_map: Optional[Dict[str, str]]):
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
    baseline_analyzers_fingerprint: A mapping from analyzer name to a set of
      paths that define its fingerprint.
    tensor_replacement_map: A map from placeholder tensor names to their
      evaluated replacement tensors.
    output_keys_to_name_map: A map from output dictionary keys to the names of
      the tensors that they represent.

  Returns:
    A tuple containing a pair of `tf.ConcreteFunction`s:
      1. The traced preprocessing_fn.
      2. A metadata_fn that returns a dictionary containing the deferred
      annotations added to the graph when invoked with any valid input.

  Raises:
    RuntimeError: if analyzers in `preprocessing_fn` are encountered in a
    non-deterministic order.
  """
  concrete_transform_fn = _trace_and_write_transform_fn(
      saved_model_dir, preprocessing_fn, input_signature, base_temp_dir,
      tensor_replacement_map, output_keys_to_name_map)
  structured_inputs = tf2_utils.get_structured_inputs_from_func_graph(
      concrete_transform_fn.graph)
  _validate_analyzers_fingerprint(baseline_analyzers_fingerprint,
                                  concrete_transform_fn.graph,
                                  structured_inputs)

  # If the `TENSOR_REPLACEMENTS` graph collection is empty, all TFT analyzers
  # in the `preprocessing_fn` have already been evaluated.
  if not concrete_transform_fn.graph.get_collection(
      analyzer_nodes.TENSOR_REPLACEMENTS):
    metadata = _trace_and_get_metadata(concrete_transform_fn, structured_inputs,
                                       preprocessing_fn, base_temp_dir,
                                       tensor_replacement_map)
    metadata_io.write_metadata(metadata,
                               os.path.join(saved_model_dir, METADATA_DIR_NAME))


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
    concrete_transform_fn = _trace_and_write_transform_fn(
        saved_model_dir=transform_fn_path,
        preprocessing_fn=preprocessing_fn,
        input_signature=type_specs,
        base_temp_dir=None,
        tensor_replacement_map=None,
        output_keys_to_name_map=None)
    _assert_no_analyzers_in_graph(concrete_transform_fn.graph)
    structured_inputs = tf2_utils.get_structured_inputs_from_func_graph(
        concrete_transform_fn.graph)
    transformed_metadata = _trace_and_get_metadata(
        concrete_transform_fn=concrete_transform_fn,
        structured_inputs=structured_inputs,
        preprocessing_fn=preprocessing_fn,
        base_temp_dir=None,
        tensor_replacement_map=None)
  transformed_metadata_dir = os.path.join(
      transform_output_path, TFTransformOutput.TRANSFORMED_METADATA_DIR)
  metadata_io.write_metadata(transformed_metadata, transformed_metadata_dir)
