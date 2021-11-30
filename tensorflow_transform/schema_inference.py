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
"""Logic associated with schema inference and propagation.

This module contains functionality to set the schema assciated with a Tensor,
and to infer the schema for a tensor, including any information that has been
set.  This module will also contain any schema propagation logic, i.e. deducing
the schema of a tensor from its parents in the graph.
"""

import collections
from typing import Callable, List, Mapping, Optional, Tuple, Union

from absl import logging
import tensorflow as tf
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import graph_context
from tensorflow_transform import tf2_utils
from tensorflow_transform import tf_utils
from tensorflow_transform.saved import saved_transform_io_v2
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.tfxio import tensor_representation_util

from google.protobuf import any_pb2
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import function
from tensorflow.python.framework import ops
# pylint: enable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


def _ragged_feature_spec_from_batched_tensor(
    name: str, tensor: tf.RaggedTensor
) -> Union[tf.io.VarLenFeature, common_types.RaggedFeature]:
  """Infer `tf.io.RaggedFeature` from a batched `tf.RaggedTensor`."""
  if common_types.is_ragged_feature_available():
    logging.warn(
        'Feature %s is a RaggedTensor, its support is currently '
        'experimental', name)

    partitions = []
    row_lengths_partition_idx = 1
    # Ignore batch dimension.
    for dim in tensor.values.shape[1:]:
      if dim or dim == 0:
        partitions.append(
            tf.io.RaggedFeature.UniformRowLength(  # pytype: disable=attribute-error
                length=dim))
      else:
        partitions.append(
            tf.io.RaggedFeature.RowLengths(  # pytype: disable=attribute-error
                key='{}$row_lengths_{}'.format(name,
                                               row_lengths_partition_idx)))
        row_lengths_partition_idx += 1

    return tf.io.RaggedFeature(
        dtype=tensor.dtype,
        value_key='{}$ragged_values'.format(name),
        partitions=partitions,
        row_splits_dtype=tensor.row_splits.dtype)
  else:
    logging.warn(
        'Feature %s was a RaggedTensor.  A Schema will be generated but the '
        'Schema cannot be used with a coder (e.g. to materialize output '
        'data) or to generated a feature spec.', name)
    # Arbitrarily select VarLenFeature.
    return tf.io.VarLenFeature(tensor.dtype)


def _feature_spec_from_batched_tensors(
    tensors: Mapping[str, common_types.TensorType],
    is_evaluation_complete: bool) -> Mapping[str, common_types.FeatureSpecType]:
  """Infer a feature spec from a dict of tensors.

  Args:
    tensors: A dict whose keys are strings and values are `Tensor`,
      `SparseTensor`, or `RaggedTensor`s.
    is_evaluation_complete: A boolean indicating whether all analyzers have been
      evaluated or not.

  Returns:
    A feature spec inferred from the types and shapes of the tensors.

  Raises:
    ValueError: If the feature spec cannot be inferred.
    TypeError: If any of the values of `tensors` are not a `Tensor`,
        `SparseTensor`, or `RaggedTensor`.
  """
  feature_spec = {}
  for name, tensor in tensors.items():
    if tensor.dtype not in (tf.string, tf.int64, tf.float32):
      raise ValueError(
          'Feature {} ({}) had invalid dtype {} for feature spec'.format(
              name, tensor, tensor.dtype))
    if isinstance(tensor, tf.SparseTensor):
      shape = tensor.get_shape()
      if shape.ndims > 2:
        feature_spec[name] = tf.io.SparseFeature(
            index_key=[
                '{}$sparse_indices_{}'.format(name, idx)
                for idx in range(shape.ndims - 1)
            ],
            value_key='{}$sparse_values'.format(name),
            dtype=tensor.dtype,
            size=shape[1:],
            already_sorted=True)
      else:
        feature_spec[name] = tf.io.VarLenFeature(tensor.dtype)
    elif isinstance(tensor, tf.Tensor):
      shape = tensor.get_shape()
      if shape.ndims in [None, 0]:
        raise ValueError(
            'Feature {} ({}) had invalid shape {} for FixedLenFeature: must '
            'have rank at least 1'.format(name, tensor, shape))
      if is_evaluation_complete and any(
          dim is None for dim in shape.as_list()[1:]):
        raise ValueError(
            'Feature {} ({}) had invalid shape {} for FixedLenFeature: apart '
            'from the batch dimension, all dimensions must have known size'
            .format(name, tensor, shape))
      feature_spec[name] = tf.io.FixedLenFeature(shape.as_list()[1:],
                                                 tensor.dtype)
    elif isinstance(tensor, tf.RaggedTensor):
      feature_spec[name] = (
          _ragged_feature_spec_from_batched_tensor(name, tensor))
    else:
      raise TypeError(
          'Expected a Tensor, SparseTensor, or RaggedTensor got {} of type {} '
          'for feature {}'.format(tensor, type(tensor), name))

  return feature_spec


def infer_feature_schema(
    features: Mapping[str, common_types.TensorType],
    graph: tf.compat.v1.Graph,
    session: Optional[tf.compat.v1.Session] = None) -> schema_pb2.Schema:
  """Given a dict of tensors, creates a `Schema`.

  Infers a schema, in the format of a tf.Transform `Schema`, for the given
  dictionary of tensors.

  If there is an override specified, we override the inferred schema for the
  given feature's tensor.  An override has the meaning that we should set
  is_categorical=True.  If session is not provided then we just set
  is_categorical=True, and if the session is provided then was also compute
  values of the tensors representing the min and max values and set them in the
  schema.

  If annotations have been specified, they are added to the output schema.

  Args:
    features: A dict mapping column names to `Tensor`, `SparseTensor` or
      `RaggedTensor`. The `Tensor`, `SparseTensor` or `RaggedTensor` should have
      a 0'th dimension which is interpreted as the batch dimension.
    graph: A `tf.Graph` used to determine schema overrides.
    session: (optional) A `tf.Session` used to compute schema overrides.  If
      None, schema overrides will not be computed. It is assumed that if all
      analyzers have been evaluated, `session` is passed to this API.

  Returns:
    A `Schema` proto.
  """
  tensor_ranges = _get_tensor_ranges(graph)
  if session is None:
    tensor_ranges = {hashable: (None, None) for hashable in tensor_ranges}
    tensor_annotations = {}
    global_annotations = []
  else:
    tensor_ranges = session.run(tensor_ranges)
    tensor_annotations, global_annotations = _get_schema_annotations(
        graph, session)
  modified_tensor_ranges = {}
  feature_annotations = {}
  for name, tensor in features.items():
    if isinstance(tensor, tf.SparseTensor):
      values = tensor.values
    elif isinstance(tensor, tf.RaggedTensor):
      values = tensor.flat_values
    else:
      values = tensor
    hashable_values = tf_utils.hashable_tensor_or_op(values)
    if hashable_values in tensor_ranges:
      assert values.dtype == tf.int64
      modified_tensor_ranges[name] = tensor_ranges[hashable_values]
    # tensor_annotations is a defaultdict(list) so always returns a list.
    feature_annotations[name] = tensor_annotations.get(hashable_values, [])

  return _infer_feature_schema_common(
      features,
      modified_tensor_ranges,
      feature_annotations,
      global_annotations,
      # A session is passed to compute schema overrides which exist only if all
      # analyzers have been evaluated. Hence, if `session` was provided to this
      # API assume TFT's evaluation is complete.
      is_evaluation_complete=session is not None)


def infer_feature_schema_v2(
    features: Mapping[str, common_types.TensorType],
    concrete_metadata_fn: function.ConcreteFunction,
    evaluate_schema_overrides: bool) -> schema_pb2.Schema:
  """Given a dict of tensors, creates a `Schema`.

  Infers a schema, in the format of a tf.Transform `Schema`, for the given
  dictionary of tensors.

  If there is an override specified, we override the inferred schema for the
  given feature's tensor.  An override has the meaning that we should set
  is_categorical=True.  If evaluate_schema_overrides is False then we just set
  is_categorical=True, and if evaluate_schema_overrides is True then we also
  compute values of the tensors representing the min and max values and set them
  in the schema.

  If annotations have been specified, they are added to the output schema.

  Args:
    features: A dict mapping column names to `Tensor`, `SparseTensor` or
      `RaggedTensor`. The `Tensor`, `SparseTensor` or `RaggedTensor` should have
      a 0'th dimension which is interpreted as the batch dimension.
    concrete_metadata_fn: A `tf.ConcreteFunction` that returns a dictionary
      containing the deferred annotations added to the graph when invoked with
      any valid input.
    evaluate_schema_overrides: A Boolean used to compute schema overrides. If
      `False`, schema overrides will not be computed.

  Returns:
    A `Schema` proto.
  """
  metadata = collections.defaultdict(list, concrete_metadata_fn())

  if not evaluate_schema_overrides:
    tensor_ranges = {
        tensor.numpy().decode(): (None, None)
        for tensor in metadata[_TF_METADATA_TENSOR_COLLECTION]
    }
    tensor_annotations = {}
    global_annotations = []
  else:
    tensor_ranges = _get_tensor_ranges_v2(metadata)
    tensor_annotations, global_annotations = _get_schema_annotations_v2(
        metadata)
  return _infer_feature_schema_common(
      features,
      tensor_ranges,
      tensor_annotations,
      global_annotations,
      is_evaluation_complete=evaluate_schema_overrides)


def _infer_feature_schema_common(
    features: Mapping[str, common_types.TensorType],
    tensor_ranges: Mapping[str, Tuple[int, int]],
    feature_annotations: Mapping[str, List[any_pb2.Any]],
    global_annotations: List[any_pb2.Any],
    is_evaluation_complete: bool) -> schema_pb2.Schema:
  """Given a dict of tensors, creates a `Schema`.

  Args:
    features: A dict mapping column names to `Tensor`, `SparseTensor` or
      `RaggedTensor`. The `Tensor`, `SparseTensor` or `RaggedTensor` should have
      a 0'th dimension which is interpreted as the batch dimension.
    tensor_ranges: A dict mapping a tensor to a tuple containing its min and max
      value.
    feature_annotations: dictionary from feature name to list of any_pb2.Any
      protos to be added as an annotation for that feature in the schema.
    global_annotations: list of any_pb2.Any protos to be added at the global
      schema level.
    is_evaluation_complete: A boolean indicating whether all analyzers have been
      evaluated or not.

  Returns:
    A `Schema` proto.
  """
  domains = {}
  feature_tags = collections.defaultdict(list)
  for name, tensor in features.items():
    if (isinstance(tensor, tf.RaggedTensor) and
        not common_types.is_ragged_feature_available()):
      # Add the 'ragged_tensor' tag which will cause coder and
      # schema_as_feature_spec to raise an error, as there is no feature spec
      # for ragged tensors in TF 1.x.
      feature_tags[name].append(schema_utils.RAGGED_TENSOR_TAG)
    if name in tensor_ranges:
      min_value, max_value = tensor_ranges[name]
      domains[name] = schema_pb2.IntDomain(
          min=min_value, max=max_value, is_categorical=True)
  feature_spec = _feature_spec_from_batched_tensors(features,
                                                    is_evaluation_complete)

  schema_proto = schema_utils.schema_from_feature_spec(feature_spec, domains)

  # Add the annotations to the schema.
  for annotation in global_annotations:
    schema_proto.annotation.extra_metadata.add().CopyFrom(annotation)
  # Build a map from logical feature names to Feature protos
  feature_protos_by_name = {}
  for feature in schema_proto.feature:
    feature_protos_by_name[feature.name] = feature
  for sparse_feature in schema_proto.sparse_feature:
    for index_feature in sparse_feature.index_feature:
      feature_protos_by_name.pop(index_feature.name)
    value_feature = feature_protos_by_name.pop(
        sparse_feature.value_feature.name)
    feature_protos_by_name[sparse_feature.name] = value_feature

  # Handle ragged tensor representations.
  tensor_representations = (
      tensor_representation_util.GetTensorRepresentationsFromSchema(
          schema_proto, schema_utils.TENSOR_REPRESENTATION_GROUP))
  if tensor_representations is not None:
    for name, tensor_representation in tensor_representations.items():
      feature_protos_by_name[name] = schema_utils.pop_ragged_source_columns(
          name, tensor_representation, feature_protos_by_name)

  # Update annotations
  for feature_name, annotations in feature_annotations.items():
    feature_proto = feature_protos_by_name[feature_name]
    for annotation in annotations:
      feature_proto.annotation.extra_metadata.add().CopyFrom(annotation)
  for feature_name, tags in feature_tags.items():
    feature_proto = feature_protos_by_name[feature_name]
    for tag in tags:
      feature_proto.annotation.tag.append(tag)
  return schema_proto


# Names of collections, which should all be the same length and contain tensors.
# Each tensor in the first collection should have its min/max described by the
# tensors in the other two collections.
_TF_METADATA_TENSOR_COLLECTION = 'tft_schema_override_tensor'
_TF_METADATA_TENSOR_MIN_COLLECTION = 'tft_schema_override_min'
_TF_METADATA_TENSOR_MAX_COLLECTION = 'tft_schema_override_max'
# Collections for adding to annotation.extra_metadata on the schema. Each
# tensor in the first collection should have a proto type and proto message in
# the other two collections
_TF_METADATA_EXTRA_ANNOTATION = 'tft_schema_override_annotation_tensor'
_TF_METADATA_EXTRA_ANNOTATION_TYPE_URL = 'tft_schema_override_annotation_type'
_TF_METADATA_EXTRA_ANNOTATION_PROTO = 'tft_schema_override_annotation_proto'
# Used to indicate that an annotation should be applied at the schema level.
_TF_METADATA_EXTRA_ANNOTATION_GLOBAL = 'tft_schema_override_global_sentinel'


def set_tensor_schema_override(tensor, min_value, max_value):
  """Override parts of the schema of a `Tensor`.

  Args:
    tensor: The `Tensor` whose range is being set.  Must have dtype int64.
    min_value: A `Tensor` representing the min value of `tensor`.
    max_value: A `Tensor` representing the max value of `tensor`.

  Raises:
    ValueError: If any arguments are invalid.
  """
  if not isinstance(tensor, tf.Tensor):
    raise ValueError('tensor {} was not a Tensor'.format(tensor))
  if tensor.dtype != tf.int64:
    raise ValueError(
        'Range can only be set for feature of type tf.int64, got {}'.format(
            tensor.dtype))
  if not isinstance(min_value, tf.Tensor):
    raise ValueError('min_value {} was not a Tensor'.format(min_value))
  if not isinstance(max_value, tf.Tensor):
    raise ValueError('max_value {} was not a Tensor'.format(max_value))
  tf.compat.v1.add_to_collection(_TF_METADATA_TENSOR_COLLECTION, tensor)
  tf.compat.v1.add_to_collection(_TF_METADATA_TENSOR_MIN_COLLECTION, min_value)
  tf.compat.v1.add_to_collection(_TF_METADATA_TENSOR_MAX_COLLECTION, max_value)


def _get_tensor_ranges(graph):
  """Lookup overrides for `Tensor`s  or `SparseTensor`s."""
  tensors = graph.get_collection(_TF_METADATA_TENSOR_COLLECTION)
  min_values = graph.get_collection(_TF_METADATA_TENSOR_MIN_COLLECTION)
  max_values = graph.get_collection(_TF_METADATA_TENSOR_MAX_COLLECTION)
  assert len(tensors) == len(min_values), '{} != {}'.format(tensors, min_values)
  assert len(tensors) == len(max_values), '{} != {}'.format(tensors, max_values)
  return dict(
      zip(
          map(tf_utils.hashable_tensor_or_op, tensors),
          zip(min_values, max_values)))


def _get_tensor_ranges_v2(metadata):
  """Lookup overrides for `Tensor`s  or `SparseTensor`s."""
  tensors = metadata[_TF_METADATA_TENSOR_COLLECTION]
  min_values = metadata[_TF_METADATA_TENSOR_MIN_COLLECTION]
  max_values = metadata[_TF_METADATA_TENSOR_MAX_COLLECTION]
  assert len(tensors) == len(min_values), '{} != {}'.format(tensors, min_values)
  assert len(tensors) == len(max_values), '{} != {}'.format(tensors, max_values)
  return {
      tensor.numpy().decode(): (min_value.numpy(), max_value.numpy())
      for (tensor, min_value, max_value) in zip(tensors, min_values, max_values)
  }


def get_tensor_schema_override(
    tensor: common_types.TensorType) -> Tuple[tf.Tensor, tf.Tensor]:
  """Lookup schema overrides for a `Tensor`  or `CompositeTensor`."""
  if isinstance(tensor, tf.SparseTensor):
    tensor = tensor.values
  elif isinstance(tensor, tf.RaggedTensor):
    tensor = tensor.flat_values
  overrides = _get_tensor_ranges(tensor.graph)
  min_max = overrides.get(tf_utils.hashable_tensor_or_op(tensor), None)
  if min_max is None:
    raise ValueError('Requested tensor does not have recorded min/max values')
  return min_max


def annotate(type_url, proto_message, tensor=None):
  """Adds a deferred annotation to the schema.

  Experimental: This API is subject to change.

  This function allows analyzers or end users to annotate the post-transform
  schema with additional information based on analyzer output. These annotations
  are stored in the annotation.extra_metadata field of the tf.metadata schema:
  https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L193

  Args:
    type_url: A string or string `Tensor` containing the type url which uniquely
      identifies the type of the serialized proto message. See
      https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/any.proto#L151
    proto_message: A deferred string tensor containing the serialized proto to
      write to the feature schema.
    tensor: (optional) If provided, the annotation will be written to the
      Feature proto that is created for this tensor in the schema. If None,
      the annotation is assumed to be global. Note: if the tensor is not present
        in the output signature of `preprocessing_fn`, this will be a no-op.
  """
  if tensor is None:
    tensor = tf.constant('unused', name=_TF_METADATA_EXTRA_ANNOTATION_GLOBAL)

  if not isinstance(tensor, (tf.Tensor, tf.SparseTensor)):
    raise ValueError('tensor {} was not a Tensor'.format(tensor))
  if not isinstance(proto_message, tf.Tensor):
    raise ValueError('proto_message {} was not a Tensor'.format(proto_message))

  # If the type_url is passed as a plain string, create a string tensor.
  if not isinstance(type_url, tf.Tensor):
    type_url = tf.constant(type_url, dtype=tf.string)
  # Note: The tensors, types, and messages are stored in separate collections
  # because SavedModel only supports primitive types in collections.
  tf.compat.v1.add_to_collection(_TF_METADATA_EXTRA_ANNOTATION, tensor)
  tf.compat.v1.add_to_collection(_TF_METADATA_EXTRA_ANNOTATION_TYPE_URL,
                                 type_url)
  tf.compat.v1.add_to_collection(_TF_METADATA_EXTRA_ANNOTATION_PROTO,
                                 proto_message)


def _get_schema_annotations(graph, session):
  """Fetch extra_metadata annotations to be applied to the schema.

  Extracts any deferred annotations that have been added to the graph and
  evaluates them to obtain any_pb2.Any proto messages.

  Args:
    graph: A `tf.Graph` used to determine schema overrides.
    session: (optional) A `tf.Session` used to compute schema annotations.  If
      None, schema annotations will not be computed.

  Returns:
    tensor_annotations: dictionary from tensor to list of any_pb2.Any protos to
      be added as an annotation for that tensor's feature in the schema.
    global_annotations: list of any_pb2.Any protos to be added at the global
      schema level.
  """
  tensors = graph.get_collection(_TF_METADATA_EXTRA_ANNOTATION)
  type_urls = session.run(
      graph.get_collection(_TF_METADATA_EXTRA_ANNOTATION_TYPE_URL))
  proto_values = session.run(
      graph.get_collection(_TF_METADATA_EXTRA_ANNOTATION_PROTO))
  tensor_annotation_keys = []
  for tensor in tensors:
    # Entries meant for the global schema annotation will have names like
    # tft_schema_override_global_sentinel:0 or
    # transform/tft_schema_override_global_sentinel_1:0
    tensor_name = tensor.name.split('/')[-1]
    if tensor_name.startswith(_TF_METADATA_EXTRA_ANNOTATION_GLOBAL):
      tensor_annotation_keys.append(_TF_METADATA_EXTRA_ANNOTATION_GLOBAL)
    else:
      tensor_annotation_keys.append(tf_utils.hashable_tensor_or_op(tensor))
  return _get_schema_annotations_common(tensor_annotation_keys, type_urls,
                                        proto_values)


def _get_schema_annotations_v2(metadata):
  """Fetch extra_metadata annotations to be applied to the schema.

  Extracts any deferred annotations that have been added to the graph and
  evaluates them to obtain any_pb2.Any proto messages.

  Args:
    metadata: A dictionary containing the deferred annotations added to the
      graph.

  Returns:
    tensor_annotations: dictionary from tensor to list of any_pb2.Any protos to
      be added as an annotation for that tensor's feature in the schema.
    global_annotations: list of any_pb2.Any protos to be added at the global
      schema level.
  """
  type_urls = [
      type_url.numpy()
      for type_url in metadata[_TF_METADATA_EXTRA_ANNOTATION_TYPE_URL]
  ]
  proto_values = [
      proto_value.numpy()
      for proto_value in metadata[_TF_METADATA_EXTRA_ANNOTATION_PROTO]
  ]
  tensor_annotation_keys = [
      tensor.numpy().decode()
      for tensor in metadata[_TF_METADATA_EXTRA_ANNOTATION]
  ]
  return _get_schema_annotations_common(tensor_annotation_keys, type_urls,
                                        proto_values)


def _get_schema_annotations_common(tensor_annotation_keys, type_urls,
                                   proto_values):
  """Fetch extra_metadata annotations to be applied to the schema.

  Args:
    tensor_annotation_keys: A list containing either
      `_TF_METADATA_EXTRA_ANNOTATION_GLOBAL` or a hashed tensor representation
      corresponding to each entry in `proto_values`. If an entry
      is`_TF_METADATA_EXTRA_ANNOTATION_GLOBAL`, the corresponding any_pb2.Any
      proto in `proto_values` is returned in `global_annotations`. Otherwise, it
      is returned in `feature_annotations`.
    type_urls: A list of type urls corresponding to the serialized protos in
      `proto_values`.
    proto_values: A list of serialized any_pb2.Any protos.

  Returns:
    A tuple of:
    tensor_annotations: dictionary from tensor to list of any_pb2.Any protos to
      be added as an annotation for that tensor's feature in the schema.
    global_annotations: list of any_pb2.Any protos to be added at the global
      schema level.
  """
  tensor_annotations = collections.defaultdict(list)
  global_annotations = []
  if not common.IS_ANNOTATIONS_PB_AVAILABLE:
    return tensor_annotations, global_annotations
  assert len(tensor_annotation_keys) == len(type_urls) == len(proto_values)
  for (tensor_annotation_key, type_url,
       proto_value) in zip(tensor_annotation_keys, type_urls, proto_values):
    annotation = any_pb2.Any(type_url=type_url, value=proto_value)
    if (isinstance(_TF_METADATA_EXTRA_ANNOTATION_GLOBAL,
                   type(tensor_annotation_key)) and
        tensor_annotation_key == _TF_METADATA_EXTRA_ANNOTATION_GLOBAL):
      global_annotations.append(annotation)
    else:
      tensor_annotations[tensor_annotation_key].append(annotation)
  return tensor_annotations, global_annotations


def _get_tensor_value_to_key_map(features_dict):
  """Get reverse map from name of tensor values to key in `features_dict`."""
  result = {}
  for key, tensor in features_dict.items():
    if isinstance(tensor, tf.SparseTensor):
      values = tensor.values
    elif isinstance(tensor, tf.RaggedTensor):
      values = tensor.flat_values
    else:
      values = tensor
    result[values.name] = key
  return result


def _get_schema_overrides(graph,
                          tensor_name_to_key_map,
                          tensor_collection_key,
                          overrides_keys,
                          default_tensor_name=None):
  """Obtain schema overrides from graph collections.

  For every tensor in the `tensor_collection_key` collection, the corresponding
  feature name is in `tensor_name_to_key_map` and its schema overrides are in
  the graph collections defined by keys in `overrides_keys`.
  If a tensor does not exist in `tensor_name_to_key_map` but its name starts
  with `default_tensor_name` (if provided), the overrides are returned with this
  key.

  Args:
    graph: A `FuncGraph`.
    tensor_name_to_key_map: A dictionary from tensor name to output feature key.
    tensor_collection_key: Key for the graph collection that contains list of
      tensors to annotate.
    overrides_keys: A list of graph collection keys that contain schema
      overrides/annotations.
    default_tensor_name: (Optional) A String. If provided, use as feature key if
      a tensor in the graph collections is not in `tensor_name_to_key_map`.

  Returns:
    A dictionary from graph collection keys to lists of features and their
    schema overrides/annotations.

  """
  tensors = graph.get_collection(tensor_collection_key)
  overrides_list = [graph.get_collection(k) for k in overrides_keys]

  result = collections.defaultdict(list)
  assert (len(tensors) == len(overrides_list[0]) and
          all(len(lst) == len(overrides_list[0]) for lst in overrides_list))
  for tensor, overrides_tuple in zip(tensors, zip(*overrides_list)):
    if tensor.name in tensor_name_to_key_map:
      result[tensor_collection_key].append(tensor_name_to_key_map[tensor.name])
    else:
      if default_tensor_name is None:
        continue
      tensor_name = tensor.name.split('/')[-1]
      if tensor.dtype == tf.string and tensor_name.startswith(
          default_tensor_name):
        result[tensor_collection_key].append(default_tensor_name)
      else:
        continue

    # If a feature name was added to the result list for tensor_collection_key,
    # add its annotations as well.
    assert len(overrides_keys) == len(overrides_tuple)
    for overrides_key, override in zip(overrides_keys, overrides_tuple):
      result[overrides_key].append(override)
  return result


def get_traced_metadata_fn(
    preprocessing_fn: Callable[[Mapping[str, common_types.TensorType]],
                               Mapping[str, common_types.TensorType]],
    structured_inputs: Mapping[str, common_types.TensorType],
    tf_graph_context: graph_context.TFGraphContext,
    evaluate_schema_overrides: bool) -> function.Function:
  """Get a tf.function that returns a dictionary of annotations.

  Annotations are added to graph collections keyed by graph tensor names when
  `preprocessing_fn` is being traced. The metadata fn defined by this method
  converts the graph tensor names to output feature keys.

  If `evaluate_schema_overrides` is True, tracing the `preprocessing_fn` will
  add overrides for feature ranges (min/max) and/or feature protos to the graph
  collection, if applicable. These overrides are returned when the function
  returned by this method is invoked.

  Args:
    preprocessing_fn: A user defined python function to be traced.
    structured_inputs: A dictionary of placeholder inputs to `preprocessing_fn`.
    tf_graph_context: A `TFGraphContext` context manager to invoke the
      `preprocessing_fn` in.
    evaluate_schema_overrides: If `False`, the returned dictionary contains a
      single key `_TF_METADATA_TENSOR_COLLECTION` as all other annotations are
      deferred. Else, the returned dictionary contains several deferred
      annotations.

  Returns:
    A tf.function which when invoked returns a dictionary whose keys represent
    the types of annotations and the values are collections of feature
    keys/annotations.
  """
  # Since this is a TFT-internal function with constant outputs, autograph will
  # not affect its behavior. It will only increase tracing time, if enabled.
  # Hence, trace with `autograph=False` here.
  @tf.function(input_signature=[], autograph=False)
  def metadata_fn():
    graph = ops.get_default_graph()
    inputs = tf2_utils.supply_missing_inputs(structured_inputs, batch_size=1)
    with tf_graph_context:
      transformed_features = preprocessing_fn(inputs)

    # Get a map from tensor value names to feature keys.
    reversed_features = _get_tensor_value_to_key_map(transformed_features)

    result = collections.defaultdict(list)
    if not evaluate_schema_overrides:
      schema_override_tensors = graph.get_collection(
          _TF_METADATA_TENSOR_COLLECTION)
      for tensor in schema_override_tensors:
        if tensor.name in reversed_features:
          result[_TF_METADATA_TENSOR_COLLECTION].append(
              reversed_features[tensor.name])
    else:
      # Obtain schema overrides for feature tensor ranges.
      result.update(
          _get_schema_overrides(graph, reversed_features,
                                _TF_METADATA_TENSOR_COLLECTION, [
                                    _TF_METADATA_TENSOR_MIN_COLLECTION,
                                    _TF_METADATA_TENSOR_MAX_COLLECTION
                                ]))
      # Obtain schema overrides for feature protos. If no feature tensor is in
      # the `_TF_METADATA_EXTRA_ANNOTATION` collection for a specified
      # annotation, `_TF_METADATA_EXTRA_ANNOTATION_GLOBAL` is used as the
      # feature name to indicate that this annotation should be added to the
      # global schema.
      result.update(
          _get_schema_overrides(graph, reversed_features,
                                _TF_METADATA_EXTRA_ANNOTATION, [
                                    _TF_METADATA_EXTRA_ANNOTATION_TYPE_URL,
                                    _TF_METADATA_EXTRA_ANNOTATION_PROTO
                                ], _TF_METADATA_EXTRA_ANNOTATION_GLOBAL))
    return result

  # Though this tf.function is not serialized to SavedModel, if it is capturing
  # any resources, they need to be tracked to ensure they are not garbage
  # collected.
  # We strip control dependencies from this function as we only want to evaluate
  # annotations which are constants available at graph construction time and
  # have no dependency on inputs.
  # TODO(b/149352022): Re-factor metadata computation when it is possible to
  # evaluate non output tensors from a func graph.
  module = tf_graph_context.module_to_export
  saved_transform_io_v2.trace_and_update_module(
      module, metadata_fn, 'metadata_fn', strip_control_dependencies=True)
  return module.metadata_fn
