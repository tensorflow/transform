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
"""Coder classes for encoding/decoding Example proto into tf.Transform datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import apache_beam as beam
import numpy as np

import tensorflow as tf


def _make_feature_value_fn(feature_spec):
  """Return a function to extract the typed value from the feature.

  For performance reasons this is preferred to having the ifs run on the
  handlers at execution time.

  Args:
    feature_spec: A one of the TF feature specs from the schema.
  Returns:
    A function to extract the value field from the feature depending on the
    schema dtype.
  """
  if feature_spec.dtype.is_integer:
    return lambda tf_feature: tf_feature.int64_list.value
  elif feature_spec.dtype.is_floating:
    return lambda tf_feature: tf_feature.float_list.value
  else:
    return lambda tf_feature: tf_feature.bytes_list.value


class _FixedLenFeatureHandler(object):
  """Handler for `FixedLenFeature` values.

  `FixedLenFeature` values will be parsed to a list of the corresponding
  dtype.
  """

  def __init__(self, name, feature_spec):
    self._name = name
    self._feature_value_fn = _make_feature_value_fn(feature_spec)
    self._shape = len(feature_spec.shape)
    self._numpy_dtype = feature_spec.dtype.as_numpy_dtype

  @property
  def name(self):
    return self._name

  def parse_value(self, feature_map):
    tf_feature = feature_map[self._name]
    # TODO(elmerg) replace this with if self._shape which right now is a
    # TensorShape and evaluates to False.
    if self._shape:
      return np.array(list(self._feature_value_fn(tf_feature)),
                      dtype=self._numpy_dtype)
    else:
      # Encode the values as a scalar if shape == []
      return np.array(self._feature_value_fn(tf_feature)[0],
                      dtype=self._numpy_dtype)

  def encode_value(self, feature_map, values):
    tf_feature = feature_map[self._name]
    if self._shape:
      self._feature_value_fn(tf_feature).extend(values)
    else:
      # Encode the values as a scalar if shape == []
      self._feature_value_fn(tf_feature).append(values.tolist())


class _VarLenFeatureHandler(object):
  """Handler for `VarLenFeature` values.

  `VarLenFeature` values will be parsed as an array of the corresponding dtype.
  """

  def __init__(self, name, feature_spec):
    self._name = name
    self._feature_value_fn = _make_feature_value_fn(feature_spec)
    self._numpy_dtype = feature_spec.dtype.as_numpy_dtype

  @property
  def name(self):
    return self._name

  def parse_value(self, feature_map):
    tf_feature = feature_map[self._name]
    return np.array(list(self._feature_value_fn(tf_feature)),
                    dtype=self._numpy_dtype)

  def encode_value(self, feature_map, values):
    tf_feature = feature_map[self._name]
    self._feature_value_fn(tf_feature).extend(values)


class _SparseFeatureHandler(object):
  """Handler for `SparseFeature` values.

  `SparseFeature` values will be parsed as a tuple of 1-D arrays where the first
  array corresponds to the values and the second to their indexes.
  """

  def __init__(self, name, feature_spec):
    self._name = name
    self._value_key = feature_spec.value_key
    self._index_key = feature_spec.index_key
    self._feature_value_fn = _make_feature_value_fn(feature_spec)
    self._numpy_dtype = feature_spec.dtype.as_numpy_dtype

  @property
  def name(self):
    return self._name

  def parse_value(self, feature_map):
    tf_feature = feature_map[self._value_key]
    values = np.array(list(self._feature_value_fn(tf_feature)),
                      dtype=self._numpy_dtype)
    indices = np.array(list(feature_map[self._index_key].int64_list.value),
                       dtype=np.int64)
    return (values, indices)

  def encode_value(self, feature_map, sparse_value):
    values, indices = sparse_value
    self._feature_value_fn(feature_map[self._value_key]).extend(values)
    feature_map[self._index_key].int64_list.value.extend(indices)


class ExampleProtoCoder(beam.coders.Coder):
  """A coder between serialized Example.proto and tf.Transform datasets."""

  def __init__(self, schema):
    """Build an ExampleProtoCoder.

    Args:
      schema: A tf-transform schema, right now is a feature spec.
    Raises:
      ValueError: If the schema contains a non-supported feature_spec.
    """
    self._feature_handlers = []
    for name, feature_spec in schema.iteritems():
      if isinstance(feature_spec, tf.FixedLenFeature):
        self._feature_handlers.append(
            _FixedLenFeatureHandler(name, feature_spec))
      elif isinstance(feature_spec, tf.VarLenFeature):
        self._feature_handlers.append(
            _VarLenFeatureHandler(name, feature_spec))
      elif isinstance(feature_spec, tf.SparseFeature):
        self._feature_handlers.append(
            _SparseFeatureHandler(name, feature_spec))
      else:
        raise ValueError('feature_spec should be one of tf.FixedLenFeature, '
                         'tf.VarLenFeature or tf.SparseFeature: %s was %s' %
                         (name, type(feature_spec)))

  def encode(self, instance):
    example_proto = tf.train.Example()
    feature_map = example_proto.features.feature
    for feature_handler in self._feature_handlers:
      value = instance[feature_handler.name]
      feature_handler.encode_value(feature_map, value)
    return example_proto.SerializeToString()

  def decode(self, serialized_example_proto):
    example_proto = tf.train.Example()
    example_proto.ParseFromString(serialized_example_proto)
    feature_map = example_proto.features.feature

    return {feature_handler.name: feature_handler.parse_value(feature_map)
            for feature_handler in self._feature_handlers}
