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

import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils

from tensorflow.python.util import deprecation
from tensorflow_metadata.proto.v0 import schema_pb2


class Schema(object):
  """The schema of a dataset.

  This is an in-memory representation that may be serialized and deserialized to
  and from a variety of disk representations.

  Args:
    column_schemas: (optional) A dict from logical column names to
        `ColumnSchema`s.
  """

  def __init__(self, column_schemas):
    feature_spec = {name: spec
                    for name, (_, spec) in column_schemas.items()}
    domains = {name: domain
               for name, (domain, _) in column_schemas.items()
               if domain is not None}
    self._schema_proto = schema_utils.schema_from_feature_spec(
        feature_spec, domains)

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self._schema_proto == other._schema_proto  # pylint: disable=protected-access
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, repr(self._schema_proto))


  def as_feature_spec(self):
    """Returns a representation of this Schema as a feature spec.

    A feature spec (for a whole dataset) is a dictionary from logical feature
    names to one of `FixedLenFeature`, `SparseFeature` or `VarLenFeature`.

    Returns:
      A representation of this Schema as a feature spec.
    """
    return schema_utils.schema_as_feature_spec(
        self._schema_proto)[0]

  def domains(self):
    """Returns the domains for this feature spec."""
    return schema_utils.schema_as_feature_spec(
        self._schema_proto)[1]

  # Implement reduce so that the proto is serialized using proto serialization
  # instead of the default pickling.
  def __getstate__(self):
    return self._schema_proto.SerializeToString()

  def __setstate__(self, state):
    self._schema_proto = schema_pb2.Schema()
    self._schema_proto.MergeFromString(state)


@deprecation.deprecated(
    None,
    'ColumnSchema is a deprecated, use from_feature_spec to create a `Schema`')
def ColumnSchema(domain, axes, representation):  # pylint: disable=invalid-name
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
    spec = tf.FixedLenFeature(axes, dtype, representation.default_value)
  elif isinstance(representation, ListColumnRepresentation):
    spec = tf.VarLenFeature(dtype)
  else:
    raise TypeError('Invalid representation: {}'.format(representation))

  return int_domain, spec


def IntDomain(dtype, min_value=None, max_value=None, is_categorical=None):  # pylint: disable=invalid-name
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
  if domains is None:
    domains = {}
  column_schemas = {name: (domains.get(name), spec)
                    for name, spec in feature_spec.items()}
  return Schema(column_schemas)
