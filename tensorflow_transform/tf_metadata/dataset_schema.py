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
"""In-memory representation of the schema of a dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import copyreg
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils

from tensorflow.python.util import deprecation
from tensorflow_metadata.proto.v0 import schema_pb2


# pylint: disable=invalid-name
def serialize_schema(schema):
  return (deserialize_schema, (schema.SerializeToString(),))


def deserialize_schema(serialized):
  result = schema_pb2.Schema()
  result.MergeFromString(serialized)
  return result


# Override default pickling.
copyreg.pickle(schema_pb2.Schema, serialize_schema)


@deprecation.deprecated(
    None,
    'Schema is a deprecated, use schema_utils.schema_from_feature_spec '
    'to create a `Schema`')
def Schema(column_schemas):
  """Legacy constructor for a tensorflow_metadata.proto.v0.schema_pb2.Schema.

  Use schema_utils.schema_from_feature_spec instead.

  Args:
    column_schemas: (optional) A dict from logical column names to
        `ColumnSchema`s.

  Returns:
    A Schema proto.
  """
  feature_spec = {name: spec
                  for name, (_, spec) in column_schemas.items()}
  domains = {name: domain
             for name, (domain, _) in column_schemas.items()
             if domain is not None}
  return schema_utils.schema_from_feature_spec(feature_spec, domains)


@deprecation.deprecated(
    None,
    'ColumnSchema is a deprecated, use from_feature_spec to create a `Schema`')
def ColumnSchema(domain, axes, representation):
  """Legacy constructor for a column schema."""
  if isinstance(domain, tf.DType):
    dtype = domain
    int_domain = None
  elif isinstance(domain, schema_pb2.IntDomain):
    dtype = tf.int64
    int_domain = domain
  else:
    raise TypeError('Invalid domain: {}'.format(domain))

  if isinstance(representation, FixedColumnRepresentation):
    spec = tf.io.FixedLenFeature(axes, dtype, representation.default_value)
  elif isinstance(representation, ListColumnRepresentation):
    spec = tf.io.VarLenFeature(dtype)
  else:
    raise TypeError('Invalid representation: {}'.format(representation))

  return int_domain, spec


def IntDomain(dtype, min_value=None, max_value=None, is_categorical=None):
  """Legacy constructor for an IntDomain."""
  if dtype != tf.int64:
    raise ValueError('IntDomain must be called with dtype=tf.int64')
  return schema_pb2.IntDomain(min=min_value, max=max_value,
                              is_categorical=is_categorical)


class FixedColumnRepresentation(collections.namedtuple(
    'FixedColumnRepresentation', ['default_value'])):

  def __new__(cls, default_value=None):
    return super(FixedColumnRepresentation, cls).__new__(cls, default_value)


ListColumnRepresentation = collections.namedtuple(
    'ListColumnRepresentation', [])


@deprecation.deprecated(
    None,
    'from_feature_spec is a deprecated, use '
    'schema_utils.schema_from_feature_spec')
def from_feature_spec(feature_spec, domains=None):
  """Convert a feature_spec to a Schema.

  Args:
    feature_spec: a features specification in the format expected by
        tf.parse_example(), i.e.
        `{name: FixedLenFeature(...), name: VarLenFeature(...), ...'
    domains: a dictionary whose keys are a subset of the keys of `feature_spec`
        and values are an schema_pb2.IntDomain, schema_pb2.StringDomain or
        schema_pb2.FloatDomain.

  Returns:
    A Schema representing the provided set of columns.
  """
  return schema_utils.schema_from_feature_spec(feature_spec, domains)
