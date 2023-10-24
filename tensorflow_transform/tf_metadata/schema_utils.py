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
"""Utilities for using the tf.Metadata Schema within TensorFlow."""

import dataclasses
import typing
from typing import Dict, List, Mapping, Optional, Tuple, Union

import tensorflow as tf
from tensorflow_transform import common_types
from tensorflow_transform.tf_metadata import schema_utils_legacy
from tfx_bsl.tfxio import tensor_representation_util

from tensorflow_metadata.proto.v0 import path_pb2
from tensorflow_metadata.proto.v0 import schema_pb2

# We use an empty name for the default tensor representation group in the output
# schema. It contains all ragged output tensor representations.
TENSOR_REPRESENTATION_GROUP = ''


def schema_from_feature_spec(
    feature_spec: Mapping[str, common_types.FeatureSpecType],
    domains: Optional[Mapping[str, common_types.DomainType]] = None
) -> schema_pb2.Schema:
  """Convert a feature spec to a Schema proto.

  Args:
    feature_spec: A TensorFlow feature spec
    domains: (optional) a dict whose keys are feature names and values are one
      of schema_pb2.IntDomain, schema_pb2.StringDomain or
      schema_pb2.FloatDomain.

  Returns:
    A Schema proto

  Raises:
    ValueError: If the feature spec cannot be converted to a Schema proto.
  """
  if domains is None:
    domains = {}

  result = schema_pb2.Schema()

  # Some feature specs can only be represented with the legacy schema, in
  # particular feature specs where any FixedLenFeature has default_value set.
  # We represent these (and only these) using a schema with
  # generate_legacy_feature_spec=True.  Note the generate_legacy_feature_spec
  # field is not part of the open source codebase.
  if schema_utils_legacy.should_set_generate_legacy_feature_spec(feature_spec):
    return _legacy_schema_from_feature_spec(feature_spec, domains)

  schema_utils_legacy.set_generate_legacy_feature_spec(result, False)

  # Add the features to the schema.
  for name, spec in sorted(feature_spec.items()):
    if isinstance(spec, tf.io.SparseFeature):
      (index_feature, value_feature, sparse_feature) = (
          _sparse_feature_from_feature_spec(spec, name, domains))
      for f in index_feature:
        result.feature.add().CopyFrom(f)
      result.feature.add().CopyFrom(value_feature)
      result.sparse_feature.add().CopyFrom(sparse_feature)

    elif isinstance(spec, tf.io.RaggedFeature):
      (value_feature, partitions_features, ragged_tensor_representation) = (
          _ragged_tensor_representation_from_feature_spec(spec, name, domains))
      result.feature.add().CopyFrom(value_feature)
      for f in partitions_features:
        result.feature.add().CopyFrom(f)
      tensor_representation_map = result.tensor_representation_group[
          TENSOR_REPRESENTATION_GROUP].tensor_representation
      tensor_representation_map[name].CopyFrom(ragged_tensor_representation)

    else:
      result.feature.add().CopyFrom(
          _feature_from_feature_spec(spec, name, domains))
  return result


def _ragged_tensor_representation_from_feature_spec(
    spec: tf.io.RaggedFeature, name: str, domains: Dict[str,
                                                        common_types.DomainType]
) -> Tuple[schema_pb2.Feature, List[schema_pb2.Feature],
           schema_pb2.TensorRepresentation]:
  """Returns representation of a RaggedTensor from a feature spec.

  Args:
    spec: A tf.io.RaggedFeature feature spec.
    name: Feature name.
    domains: A dict whose keys are feature names and values are one of
      schema_pb2.IntDomain, schema_pb2.StringDomain or schema_pb2.FloatDomain.

  Returns:
    A tuple (value_feature, partitions_features, ragged_tensor_rep),
      where value_feature represents RaggedTensor values, partitions_features
      represent row lengths partitions and ragged_tensor_rep - ragged
      TensorRepresentation.

  Raises:
    ValueError: If the feature spec contains partition types different from
      UniformRowLength and RowLengths.
  """
  value_feature = schema_pb2.Feature(name=spec.value_key or name)
  _set_type(name, value_feature, spec.dtype)
  _set_domain(name, value_feature, domains.get(name))

  ragged_tensor = schema_pb2.TensorRepresentation.RaggedTensor(
      feature_path=path_pb2.Path(step=[spec.value_key or name]))

  partitions_features = []
  for partition in spec.partitions:
    if isinstance(partition, tf.io.RaggedFeature.UniformRowLength):  # pytype: disable=attribute-error
      ragged_tensor.partition.append(
          schema_pb2.TensorRepresentation.RaggedTensor.Partition(
              uniform_row_length=partition.length))
    elif isinstance(partition, tf.io.RaggedFeature.RowLengths):  # pytype: disable=attribute-error
      ragged_tensor.partition.append(
          schema_pb2.TensorRepresentation.RaggedTensor.Partition(
              row_length=partition.key))
      partitions_features.append(
          schema_pb2.Feature(name=partition.key, type=schema_pb2.INT))
    else:
      raise ValueError(
          'RaggedFeature can only be created with UniformRowLength and '
          'RowLengths partitions.')

  return value_feature, partitions_features, schema_pb2.TensorRepresentation(
      ragged_tensor=ragged_tensor)


def _sparse_feature_from_feature_spec(spec, name, domains):
  """Returns a representation of a SparseFeature from a feature spec."""
  if isinstance(spec.index_key, list):
    assert isinstance(spec.size, (list, tuple, tf.TensorShape)), type(spec.size)
    assert len(spec.index_key) == len(spec.size), (spec.index_key, spec.size)
    spec_size = [
        s.value if isinstance(s, tf.compat.v1.Dimension) else s
        for s in spec.size
    ]
    spec_size = [s if s != -1 else None for s in spec_size]
    int_domains = [
        schema_pb2.IntDomain(min=0, max=size - 1) if size is not None else None
        for size in spec_size
    ]
    index_feature = [
        schema_pb2.Feature(
            name=key, type=schema_pb2.INT, int_domain=int_domain)
        for (key, int_domain) in zip(spec.index_key, int_domains)
    ]
    index_feature_ref = [
        schema_pb2.SparseFeature.IndexFeature(name=key)
        for key in spec.index_key
    ]
  else:
    # Create a index feature.
    index_feature = [
        schema_pb2.Feature(
            name=spec.index_key,
            type=schema_pb2.INT,
            int_domain=schema_pb2.IntDomain(min=0, max=spec.size - 1))
    ]
    index_feature_ref = [
        schema_pb2.SparseFeature.IndexFeature(name=spec.index_key)
    ]

  # Create a value feature.
  value_feature = schema_pb2.Feature(name=spec.value_key)
  _set_type(name, value_feature, spec.dtype)
  _set_domain(name, value_feature, domains.get(name))

  # Create a sparse feature which refers to the index and value features.
  value_feature_ref = schema_pb2.SparseFeature.ValueFeature(name=spec.value_key)
  sparse_feature = schema_pb2.SparseFeature(
      name=name,
      is_sorted=True if spec.already_sorted else None,
      index_feature=index_feature_ref,
      value_feature=value_feature_ref)

  return (index_feature, value_feature, sparse_feature)


def _feature_from_feature_spec(spec, name, domains):
  """Returns a representation of a Feature from a feature spec."""
  if isinstance(spec, tf.io.FixedLenFeature):
    if spec.default_value is not None:
      raise ValueError(
          'feature "{}" had default_value {}, but FixedLenFeature must have '
          'default_value=None'.format(name, spec.default_value))
    dims = [schema_pb2.FixedShape.Dim(size=size) for size in spec.shape]
    feature = schema_pb2.Feature(
        name=name,
        presence=schema_pb2.FeaturePresence(min_fraction=1.0),
        shape=schema_pb2.FixedShape(dim=dims))
  elif isinstance(spec, tf.io.VarLenFeature):
    feature = schema_pb2.Feature(name=name)
  else:
    raise TypeError('Spec for feature "{}" was {} of type {}, expected a '
                    'FixedLenFeature, VarLenFeature or SparseFeature'.format(
                        name, spec, type(spec)))

  _set_type(name, feature, spec.dtype)
  _set_domain(name, feature, domains.get(name))
  return feature


def _set_type(name, feature, dtype):
  """Set the type of a Feature proto."""
  if dtype == tf.int64:
    feature.type = schema_pb2.INT
  elif dtype == tf.float32:
    feature.type = schema_pb2.FLOAT
  elif dtype == tf.string:
    feature.type = schema_pb2.BYTES
  else:
    raise ValueError('Feature "{}" has invalid dtype {}'.format(name, dtype))


def _set_domain(name, feature, domain):
  """Set the domain of a Feature proto."""
  if domain is None:
    return

  if isinstance(domain, schema_pb2.IntDomain):
    feature.int_domain.CopyFrom(domain)
  elif isinstance(domain, schema_pb2.StringDomain):
    feature.string_domain.CopyFrom(domain)
  elif isinstance(domain, schema_pb2.FloatDomain):
    feature.float_domain.CopyFrom(domain)
  else:
    raise ValueError('Feature "{}" has invalid domain {}'.format(name, domain))


@dataclasses.dataclass(frozen=True)
class SchemaAsFeatureSpecResult:
  feature_spec: Dict[str, common_types.FeatureSpecType]
  domains: Dict[str, common_types.DomainType]

  # This is needed because many users unpack this with:
  # `feature_spec, domains = schema_utils.schema_as_feature_spec()`.
  def __iter__(self):
    return (getattr(self, field.name) for field in dataclasses.fields(self))


def _standardize_default_value(
    spec: tf.io.FixedLenFeature) -> tf.io.FixedLenFeature:
  """Converts bytes to strings and unwraps lists with a single element."""
  if spec.default_value is None:
    return spec
  default_value = spec.default_value
  assert isinstance(default_value, list), spec.default_value
  # Convert bytes to string
  if spec.dtype == tf.string:

    # Handle bytes string by trying to decode them (for legacy backwards
    # compatibility) and if failed, keep the default value as bytes.
    def try_decode(value: bytes) -> Union[str, bytes]:
      try:
        return value.decode('utf-8')
      except UnicodeError:
        return value

    default_value = [try_decode(value) for value in default_value]
  # Unwrap a list with a single element.
  if len(default_value) == 1:
    default_value = default_value[0]
  return tf.io.FixedLenFeature(
      shape=spec.shape, dtype=spec.dtype, default_value=default_value)


def schema_as_feature_spec(
    schema_proto: schema_pb2.Schema) -> SchemaAsFeatureSpecResult:
  """Generates a feature spec from a Schema proto.

  For a Feature with a FixedShape we generate a FixedLenFeature with no default.
  For a Feature without a FixedShape we generate a VarLenFeature.  For a
  SparseFeature we generate a SparseFeature.
  If schema contains struct feature, then it must also contain
  TensorRepresentations and is assumed to describe SequenceExample data. The
  result in such case is union of context and sequence feature specs.

  Args:
    schema_proto: A Schema proto.

  Returns:
    A pair (feature spec, domains) where feature spec is a dict whose keys are
        feature names and values are instances of FixedLenFeature,
        VarLenFeature, SparseFeature or RaggedFeature, and `domains` is a dict
        whose keys are feature names and values are one of the `domain_info`
        oneof, e.g. IntDomain.

  Raises:
    ValueError: If the schema proto is invalid.
  """

  # Presence of a struct means that data's physical format is tf.SequenceExample
  # and the struct contains sequence features.
  if any(feature.type == schema_pb2.STRUCT for feature in schema_proto.feature):
    return _sequence_schema_as_feature_spec(schema_proto)

  tensor_representations = (
      tensor_representation_util.InferTensorRepresentationsFromMixedSchema(
          schema_proto))

  feature_spec = {}
  # Will hold the domain_info (IntDomain, FloatDomain etc.) of the feature.  For
  # sparse features, will hold the domain_info of the values feature.  Features
  # that do not have a domain set will not be present in `domains`.
  domains = {}
  string_domains = _get_string_domains(schema_proto)
  feature_by_name = {feature.name: feature for feature in schema_proto.feature}
  for name, tensor_representation in tensor_representations.items():
    value_feature = str(
        tensor_representation_util.GetSourceValueColumnFromTensorRepresentation(
            tensor_representation))
    spec = (
        tensor_representation_util.CreateTfExampleParserConfig(
            tensor_representation, feature_by_name[value_feature].type))
    if isinstance(spec, tf.io.FixedLenFeature):
      spec = _standardize_default_value(spec)
    feature_spec[name] = spec
    domain = _get_domain(feature_by_name[value_feature], string_domains)
    if domain is not None:
      domains[name] = domain
  return SchemaAsFeatureSpecResult(feature_spec, domains)


def _sequence_schema_as_feature_spec(
    schema: schema_pb2.Schema) -> SchemaAsFeatureSpecResult:
  """Generates a feature spec from a Schema describing tf.SequenceExample data.

  See `tensor_representation_util.CreateTfSequenceExampleParserConfig`s
  docstring for feature spec generation rules.
  We mix context and sequence feature specs to replicate how preprocessing_fn
  sees input features -- as top-level values of a single `inputs` dict. Note
  that this makes the feature spec generation irreversible without additional
  input since it's no longer possible to distinguish context and sequence
  features to produce the original schema.

  Args:
    schema: A TFMD Schema proto.

  Returns:
    A pair (feature spec, domains) where feature spec is a dict whose keys are
        feature names and values are instances of FixedLenFeature,
        VarLenFeature, SparseFeature or RaggedFeature, and `domains` is a dict
        whose keys are feature names and values are one of the `domain_info`
        oneof, e.g. IntDomain.

  Raises:
    ValueError: If `TensorRepresentation`s in the schema result in feature specs
        that are not supported.
  """
  (context_feature_spec, sequence_feature_spec
  ) = tensor_representation_util.CreateTfSequenceExampleParserConfig(schema)
  feature_spec = {**context_feature_spec, **sequence_feature_spec}
  string_domains = _get_string_domains(schema)
  domain_by_feature_name = _get_source_feature_domains(schema, string_domains)
  domains = {}
  for name, spec in feature_spec.items():
    if isinstance(spec, (tf.io.FixedLenFeature, tf.io.VarLenFeature)):
      source_feature_name = name
    elif isinstance(spec, (tf.io.SparseFeature, tf.io.RaggedFeature)):
      source_feature_name = spec.value_key
    else:
      raise ValueError('spec is not recognized')
    if source_feature_name in domain_by_feature_name:
      domains[name] = domain_by_feature_name[source_feature_name]
  return SchemaAsFeatureSpecResult(feature_spec, domains)


def _get_source_feature_domains(
    schema_or_domain: Union[schema_pb2.Schema, schema_pb2.StructDomain],
    string_domains: Dict[str, schema_pb2.StringDomain]
) -> Dict[str, common_types.DomainType]:
  """Recursively extracts domains of all source features in the schema."""
  result = {}
  for feature in schema_or_domain.feature:
    domain_info = feature.WhichOneof('domain_info')
    if domain_info == 'struct_domain':
      result.update(
          _get_source_feature_domains(feature.struct_domain, string_domains))
    else:
      domain = _get_domain(feature, string_domains)
      if domain is not None:
        result[feature.name] = domain
  return result


def _get_string_domains(
    schema: schema_pb2.Schema) -> Dict[str, schema_pb2.StringDomain]:
  return {domain.name: domain for domain in schema.string_domain}


def _get_domain(feature, string_domains):
  """Get the domain of a feature, possibly looking up a schema-level domain."""
  domain_info = feature.WhichOneof('domain_info')
  if domain_info is None:
    return None
  if domain_info == 'domain':
    try:
      return string_domains[feature.domain]
    except KeyError:
      tf.compat.v1.logging.warn(
          'Feature "%s" referred to string domain "%s" which did not exist',
          feature.name, feature.domain)
      return None
  return getattr(feature, domain_info)


def pop_ragged_source_columns(
    name: str, tensor_representation: schema_pb2.TensorRepresentation,
    feature_by_name: Dict[str, schema_pb2.Feature]) -> schema_pb2.Feature:
  """Removes source columns of a ragged tensor from the given features dict.

  Args:
    name: Name of the ragged tensor.
    tensor_representation: Ragged TensorRepresentation.
    feature_by_name: Dict of features that contains source columns of the ragged
      TensorRepresentation.

  Returns:
    Value feature of the ragged tensor.

  Raises:
    ValueError: If any of the source columns are missing in the features dict.
  """
  source_columns = (
      tensor_representation_util.GetSourceColumnsFromTensorRepresentation(
          tensor_representation))
  missing_column_error_format = (
      'Ragged feature "{}" referred to value feature "{}" which did not exist '
      'in the schema or was referred to as an index or value multiple times.')

  assert source_columns
  assert len(source_columns[0].steps()) == 1, (name, source_columns[0].steps())
  try:
    value_feature = feature_by_name.pop(source_columns[0].steps()[0])
  except KeyError:
    raise ValueError(
        missing_column_error_format.format(name, source_columns[0].steps()[0]))
  for column_path in source_columns[1:]:
    assert len(column_path.steps()) == 1, (name, column_path.steps())
    try:
      row_length_feature = feature_by_name.pop(column_path.steps()[0])
    except KeyError:
      raise ValueError(
          missing_column_error_format.format(name,
                                             column_path.steps()[0]))
    if row_length_feature.type != schema_pb2.FeatureType.INT:
      raise ValueError(
          'Row length feature "{}" is not an integer feature.'.format(
              row_length_feature.name))
  return value_feature


def _ragged_tensor_representation_as_feature_spec(
    name: str, tensor_representation: schema_pb2.TensorRepresentation,
    feature_by_name: Dict[str, schema_pb2.Feature],
    string_domains: Dict[str, common_types.DomainType]
) -> Tuple[tf.io.RaggedFeature, Optional[common_types.DomainType]]:
  """Returns a representation of a RaggedTensor as a feature spec."""
  value_feature = pop_ragged_source_columns(name, tensor_representation,
                                            feature_by_name)
  spec = tensor_representation_util.CreateTfExampleParserConfig(
      tensor_representation, value_feature.type)
  domain = _get_domain(value_feature, string_domains)
  return typing.cast(tf.io.RaggedFeature, spec), domain


def _legacy_schema_from_feature_spec(feature_spec, domains=None):
  """Infer a Schema from a feature spec, using the legacy feature spec logic.

  Infers a Schema proto that with generate_legacy_feature_spec set to true,
  which will result in the given feature spec and domains when
  schema_as_feature_spec is called.  This is used to represent feature specs
  that can only be represented when generate_legacy_feature_spec is true.  In
  particular, feature specs with a default value set.

  Args:
    feature_spec: A TensorFlow feature spec
    domains: A dict from key names to `IntDomain`s

  Returns:
    A Schema proto.

  Raises:
    ValueError: If a default value is invalid.
    TypeError: If an unknown type of feature spec is encountered.
  """
  result = schema_pb2.Schema()
  result.generate_legacy_feature_spec = True
  for name, spec in sorted(feature_spec.items()):
    if isinstance(spec, tf.io.FixedLenFeature):
      # Validate shape first as shape governs which default values are valid.
      if len(spec.shape) == 0:  # pylint: disable=g-explicit-length-test
        size = 1
        expected_default_value = '' if spec.dtype == tf.string else -1
      elif len(spec.shape) == 1 and spec.shape[0] > 1:
        size = spec.shape[0]
        expected_default_value = ['' if spec.dtype == tf.string else -1] * size
      else:
        raise ValueError(
            'When inferring legacy schema from feature spec, feature "{}" had '
            'shape {}, but FixedLenFeature must have shape [] or [k] where '
            'k > 1.'.format(name, spec.shape))

      if spec.default_value is None:
        min_fraction = 1
      elif spec.default_value == expected_default_value:
        min_fraction = 0
      else:
        raise ValueError(
            'When inferring legacy schema from feature spec, feature "{}" had '
            'default_value {}, but FixedLenFeature must have '
            'default_value=None or {}'.format(name, spec.default_value,
                                              expected_default_value))

      feature = result.feature.add(
          name=name,
          presence=schema_pb2.FeaturePresence(min_fraction=min_fraction),
          value_count=schema_pb2.ValueCount(min=size, max=size))
    elif isinstance(spec, tf.io.VarLenFeature):
      feature = result.feature.add(name=name)
    else:
      raise TypeError(
          'When inferring legacy schema from feature spec, spec for feature '
          '"{}" was {} of type {}, expected a FixedLenFeature or '
          'VarLenFeature '.format(name, spec, type(spec)))

    _set_type(name, feature, spec.dtype)
    _set_domain(name, feature, domains.get(name))

  return result
