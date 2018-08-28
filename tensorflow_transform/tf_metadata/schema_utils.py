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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


import tensorflow as tf

from tensorflow_metadata.proto.v0 import schema_pb2


def schema_from_feature_spec(feature_spec, domains=None):
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

  # Add the features to the schema.
  for name, spec in sorted(feature_spec.items()):
    if isinstance(spec, tf.SparseFeature):
      (index_feature, value_feature, sparse_feature) = (
          _sparse_feature_from_feature_spec(spec, name, domains))
      result.feature.add().CopyFrom(index_feature)
      result.feature.add().CopyFrom(value_feature)
      result.sparse_feature.add().CopyFrom(sparse_feature)
    else:
      result.feature.add().CopyFrom(
          _feature_from_feature_spec(spec, name, domains))
  return result


def _sparse_feature_from_feature_spec(spec, name, domains):
  """Returns a representation of a SparseFeature from a feature spec."""
  if isinstance(spec.index_key, list):
    raise ValueError(
        'SparseFeature "{}" had index_key {}, but size and index_key '
        'fields should be single values'.format(name, spec.index_key))
  if isinstance(spec.size, list):
    raise ValueError(
        'SparseFeature "{}" had size {}, but size and index_key fields '
        'should be single values'.format(name, spec.size))

  # Create a index feature.
  index_feature = schema_pb2.Feature(
      name=spec.index_key, type=schema_pb2.INT,
      int_domain=schema_pb2.IntDomain(min=0, max=spec.size - 1))

  # Create a value feature.
  value_feature = schema_pb2.Feature(name=spec.value_key)
  _set_type(name, value_feature, spec.dtype)
  _set_domain(name, value_feature, domains.get(name))

  # Create a sparse feature which refers to the index and value features.
  index_feature_ref = schema_pb2.SparseFeature.IndexFeature(
      name=spec.index_key)
  value_feature_ref = schema_pb2.SparseFeature.ValueFeature(
      name=spec.value_key)
  sparse_feature = schema_pb2.SparseFeature(
      name=name, is_sorted=True if spec.already_sorted else None,
      index_feature=[index_feature_ref], value_feature=value_feature_ref)

  return (index_feature, value_feature, sparse_feature)


def _feature_from_feature_spec(spec, name, domains):
  """Returns a representation of a Feature from a feature spec."""
  if isinstance(spec, tf.FixedLenFeature):
    if spec.default_value is not None:
      raise ValueError(
          'feature "{}" had default_value {}, but FixedLenFeature must have '
          'default_value=None'.format(name, spec.default_value))
    dims = [schema_pb2.FixedShape.Dim(size=size) for size in spec.shape]
    feature = schema_pb2.Feature(
        name=name,
        presence=schema_pb2.FeaturePresence(min_fraction=1.0),
        shape=schema_pb2.FixedShape(dim=dims))
  elif isinstance(spec, tf.VarLenFeature):
    feature = schema_pb2.Feature(name=name)
  else:
    raise TypeError(
        'Spec for feature "{}" was {} of type {}, expected a '
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
    raise ValueError(
        'Feature "{}" has invalid dtype {}'.format(name, dtype))


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
    raise ValueError(
        'Feature "{}" has invalid domain {}'.format(name, domain))


SchemaAsFeatureSpecResult = collections.namedtuple(
    'SchemaAsFeatureSpecResult', ['feature_spec', 'domains'])


def schema_as_feature_spec(schema_proto):
  """Generates a feature spec from a Schema proto.

  For a Feature with a FixedShape we generate a FixedLenFeature with no default.
  For a Feature without a FixedShape we generate a VarLenFeature.  For a
  SparseFeature we generate a SparseFeature.

  Args:
    schema_proto: A Schema proto.

  Returns:
    A pair (feature spec, domains) where feature spec is a dict whose keys are
        feature names and values are instances of FixedLenFeature, VarLenFeature
        or SparseFeature, and `domains` is a dict whose keys are feature names
        and values are one of the `domain_info` oneof, e.g. IntDomain.

  Raises:
    ValueError: If the schema proto is invalid.
  """
  feature_spec = {}
  # Will hold the domain_info (IntDomain, FloatDomain etc.) of the feature.  For
  # sparse features, will hold the domain_info of the values feature.  Features
  # that do not have a domain set will not be present in `domains`.
  domains = {}
  feature_by_name = {feature.name: feature for feature in schema_proto.feature}
  string_domains = _get_string_domains(schema_proto)

  # Generate a `tf.SparseFeature` for each element of
  # `schema_proto.sparse_feature`.  This also removed the features from
  # feature_by_name.
  for feature in schema_proto.sparse_feature:
    if _include_in_parsing_spec(feature):
      feature_spec[feature.name], domains[feature.name] = (
          _sparse_feature_as_feature_spec(
              feature, feature_by_name, string_domains))

  # Generate a `tf.FixedLenFeature` or `tf.VarLenFeature` for each element of
  # `schema_proto.feature` that was not referenced by a `SparseFeature`.
  for name, feature in feature_by_name.items():
    if _include_in_parsing_spec(feature):
      feature_spec[name], domains[name] = _feature_as_feature_spec(
          feature, string_domains)

  domains = {name: domain for name, domain in domains.items()
             if domain is not None}
  return SchemaAsFeatureSpecResult(feature_spec, domains)


def _get_string_domains(schema):
  return {domain.name: domain for domain in schema.string_domain}


def _get_domain(feature, string_domains):
  domain_info = feature.WhichOneof('domain_info')
  if domain_info is None:
    return None
  if domain_info == 'domain':
    return string_domains[feature.domain]
  return getattr(feature, domain_info)


def _sparse_feature_as_feature_spec(feature, feature_by_name, string_domains):
  """Returns a representation of a SparseFeature as a feature spec."""
  index_keys = [index_feature.name for index_feature in feature.index_feature]
  index_features = []
  for index_key in index_keys:
    try:
      index_features.append(feature_by_name.pop(index_key))
    except KeyError:
      raise ValueError(
          'sparse_feature "{}" referred to index feature "{}" which did not '
          'exist in the schema'.format(feature.name, index_key))

  if len(index_features) != 1:
    raise ValueError(
        'sparse_feature "{}" had rank {} but currently only rank 1'
        ' sparse features are supported'.format(
            feature.name, len(index_features)))

  value_key = feature.value_feature.name
  try:
    value_feature = feature_by_name.pop(value_key)
  except KeyError:
    raise ValueError(
        'sparse_feature "{}" referred to value feature "{}" which did not '
        'exist in the schema or was referred to as an index or value multiple '
        'times.'.format(feature.name, value_key))

  if index_features[0].HasField('int_domain'):
    # Currently we only handle O-based INT index features whose minimum
    # domain value must be zero.
    if not index_features[0].int_domain.HasField('min'):
      raise ValueError('Cannot determine dense shape of sparse feature '
                       '"{}". The minimum domain value of index feature "{}"'
                       ' is not set.'.format(feature.name, index_keys[0]))
    if index_features[0].int_domain.min != 0:
      raise ValueError('Only 0-based index features are supported. Sparse '
                       'feature "{}" has index feature "{}" whose minimum '
                       'domain value is {}.'.format(
                           feature.name, index_keys[0],
                           index_features[0].int_domain.min))

    if not index_features[0].int_domain.HasField('max'):
      raise ValueError('Cannot determine dense shape of sparse feature '
                       '"{}". The maximum domain value of index feature "{}"'
                       ' is not set.'.format(feature.name, index_keys[0]))
    shape = [index_features[0].int_domain.max + 1]
  else:
    raise ValueError('Cannot determine dense shape of sparse feature "{}".'
                     ' The index feature "{}" had no int_domain set.'.format(
                         feature.name, index_keys[0]))

  dtype = _feature_dtype(value_feature)
  if len(index_keys) != len(shape):
    raise ValueError(
        'sparse_feature "{}" had rank {} (shape {}) but {} index keys were'
        ' given'.format(feature.name, len(shape), shape, len(index_keys)))
  spec = tf.SparseFeature(
      index_keys[0], value_key, dtype, shape[0], feature.is_sorted)
  domain = _get_domain(value_feature, string_domains)
  return spec, domain


def _feature_as_feature_spec(feature, string_domains):
  """Returns a representation of a Feature as a feature spec."""
  dtype = _feature_dtype(feature)
  if feature.HasField('shape'):
    if feature.presence.min_fraction != 1:
      raise ValueError(
          'Feature "{}" had shape {} set but min_fraction {} != 1.  Use'
          ' value_count not shape field when min_fraction != 1.'.format(
              feature.name, feature.shape, feature.presence.min_fraction))
    spec = tf.FixedLenFeature(
        _fixed_shape_as_tf_shape(feature.shape), dtype, default_value=None)
  else:
    spec = tf.VarLenFeature(dtype)
  domain = _get_domain(feature, string_domains)
  return spec, domain


def _feature_dtype(feature):
  """Returns a representation of a Feature's type as a tensorflow dtype."""
  if feature.type == schema_pb2.BYTES:
    return tf.string
  elif feature.type == schema_pb2.INT:
    return tf.int64
  elif feature.type == schema_pb2.FLOAT:
    return tf.float32
  else:
    raise ValueError('Feature "{}" had invalid type {}'.format(
        feature.name, schema_pb2.FeatureType.Name(feature.type)))


def _fixed_shape_as_tf_shape(fixed_shape):
  """Returns a representation of a FixedShape as a tensorflow shape."""
  return [dim.size for dim in fixed_shape.dim]


_DEPRECATED_LIFECYCLE_STAGES = [
    schema_pb2.DEPRECATED,
    schema_pb2.PLANNED,
    schema_pb2.ALPHA,
    schema_pb2.DEBUG_ONLY
]


def _include_in_parsing_spec(feature):
  return feature.lifecycle_stage not in _DEPRECATED_LIFECYCLE_STAGES

