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
"""Coder classes for encoding/decoding TF Examples into tf.Transform datasets.
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


def _make_cast_fn(np_dtype):
  """Return a function to extract the typed value from the feature.

  For performance reasons it is preferred to have the cast fn
  constructed once (for each handler).

  Args:
    np_dtype: The numpy type of the Tensorflow feature.
  Returns:
    A function to extract the value field from a string depending on dtype.
  """

  def _float_cast_fn(x):
    if isinstance(x, (np.generic, np.ndarray)):
      # This works for both np.generic and np.array (of any shape).
      return x.tolist()
    elif isinstance(x, list):
      # This works for python list which might contain np.generic values.
      return map(float, x)
    else:
      # This works for python scalars, which require no casting.
      return x

  if issubclass(np_dtype, np.floating):
    # Attempt to append to a FloatList. Success indicates that the proto library
    # installed can handle the numpy conversions properly, while a TypeError
    # indicates we need more work on our part. Interestingly non-float types
    # seem to convert properly even for old versions of the proto library.
    try:
      float_list = tf.train.FloatList()
      float_list.value.append(np.float32(0.1))  # Any dummy value will do.
      return lambda x: x
    except TypeError:
      return _float_cast_fn
  else:
    return lambda x: x


def _make_feature_value_fn(dtype):
  """Return a function to extract the typed value from the feature.

  For performance reasons it is preferred to have the feature value fn
  constructed once (for each handler).

  Args:
    dtype: The type of the Tensorflow feature.
  Returns:
    A function to extract the value field from the feature depending on dtype.
  """
  if dtype.is_integer:
    return lambda feature: feature.int64_list.value
  elif dtype.is_floating:
    return lambda feature: feature.float_list.value
  else:
    return lambda feature: feature.bytes_list.value


class _FixedLenFeatureHandler(object):
  """Handler for `FixedLenFeature` values.

  `FixedLenFeature` values will be parsed to a list of the corresponding
  dtype.
  """

  def __init__(self, name, feature_spec):
    self._name = name
    self._value_fn = _make_feature_value_fn(feature_spec.dtype)
    self._shape = len(feature_spec.shape)
    self._np_dtype = feature_spec.dtype.as_numpy_dtype

  @property
  def name(self):
    return self._name

  def initialize_encode_cache(self, example):
    """Initialize fields (performance caches) that point to example's state."""
    self._value = self._value_fn(example.features.feature[self._name])
    self._cast_fn = _make_cast_fn(self._np_dtype)

  def parse_value(self, feature_map):
    feature = feature_map[self._name]
    if self._shape:
      return list(self._value_fn(feature))
    else:
      # Encode the values as a scalar if shape == []
      return self._value_fn(feature)[0]

  def encode_value(self, values):
    del self._value[:]
    if self._shape:
      self._value.extend(self._cast_fn(values))
    else:
      self._value.append(self._cast_fn(values))


class _VarLenFeatureHandler(object):
  """Handler for `VarLenFeature` values.

  `VarLenFeature` values will be parsed as an array of the corresponding dtype.
  """

  def __init__(self, name, feature_spec):
    self._name = name
    self._value_fn = _make_feature_value_fn(feature_spec.dtype)
    self._np_dtype = feature_spec.dtype.as_numpy_dtype

  @property
  def name(self):
    return self._name

  def initialize_encode_cache(self, example):
    """Initialize fields (performance caches) that point to example's state."""
    self._value = self._value_fn(example.features.feature[self._name])
    self._cast_fn = _make_cast_fn(self._np_dtype)

  def parse_value(self, feature_map):
    feature = feature_map[self._name]
    return list(self._value_fn(feature))

  def encode_value(self, values):
    del self._value[:]
    self._value.extend(self._cast_fn(values))


class _SparseFeatureHandler(object):
  """Handler for `SparseFeature` values.

  `SparseFeature` values will be parsed as a tuple of 1-D arrays where the first
  array corresponds to the values and the second to their indexes.
  """

  def __init__(self, name, feature_spec):
    self._name = name
    self._value_key = feature_spec.value_key
    self._index_key = feature_spec.index_key
    self._value_value_fn = _make_feature_value_fn(feature_spec.dtype)
    self._index_value_fn = _make_feature_value_fn(tf.int64)
    self._np_dtype = feature_spec.dtype.as_numpy_dtype

  @property
  def name(self):
    return self._name

  def initialize_encode_cache(self, example):
    """Initialize fields (performance caches) that point to example's state."""
    self._value_value = self._value_value_fn(example.features.feature[
        self._value_key])
    self._index_value = self._index_value_fn(example.features.feature[
        self._index_key])
    self._cast_fn = _make_cast_fn(self._np_dtype)

  def parse_value(self, feature_map):
    value_feature = feature_map[self._value_key]
    index_feature = feature_map[self._index_key]
    values = list(self._value_value_fn(value_feature))
    indices = list(self._index_value_fn(index_feature))
    return (values, indices)

  def encode_value(self, sparse_value):
    del self._value_value[:]
    del self._index_value[:]
    values, indices = sparse_value
    self._value_value.extend(self._cast_fn(values))
    self._index_value.extend(indices)


class ExampleProtoCoder(object):
  """A coder between serialized TF Examples and tf.Transform datasets."""

  def __init__(self, schema):
    """Build an ExampleProtoCoder.

    Args:
      schema: A `Schema` object.
    Raises:
      ValueError: If `schema` is invalid.
    """
    self._schema = schema

    # Using pre-allocated tf.train.Example objects for performance reasons.
    #
    # The _encode_example_cache is used solely by "encode" paths while the
    # the _decode_example_cache is used solely be "decode" paths, since the
    # caching strategies are incompatible with each other (due to proto
    # parsing/merging implementation).
    #
    # Since the output of both "encode" and "decode" are deep as opposed to
    # shallow transformations, and since the schema always fully defines the
    # Example's FeatureMap (ie all fields are always cleared/assigned or
    # copied), the optimizations and implementation are correct and
    # thread-compatible.
    #
    # Due to pickling issues actual initialization of this will happen lazily
    # in encode or decode respectively.
    self._encode_example_cache = None
    self._decode_example_cache = None

    self._feature_handlers = []
    for name, feature_spec in schema.as_feature_spec().iteritems():
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

  def __reduce__(self):
    return ExampleProtoCoder, (self._schema,)

  @property
  def name(self):
    return 'example_proto'

  def encode(self, instance):
    """Encode a tf.transform encoded dict as serialized tf.Example."""
    if self._encode_example_cache is None:
      # Initialize the encode Example cache (used by this and all subsequent
      # calls).
      example = tf.train.Example()
      for feature_handler in self._feature_handlers:
        feature_handler.initialize_encode_cache(example)
      self._encode_example_cache = example

    # Encode and serialize using the Example cache.
    for feature_handler in self._feature_handlers:
      value = instance[feature_handler.name]
      feature_handler.encode_value(value)
    return self._encode_example_cache.SerializeToString()

  def decode(self, serialized_example_proto):
    """Decode serialized tf.Example as a tf.transform encoded dict."""
    if self._decode_example_cache is None:
      # Initialize the decode Example cache (used by this and all subsequent
      # calls).
      self._decode_example_cache = tf.train.Example()

    example = self._decode_example_cache
    example.ParseFromString(serialized_example_proto)
    feature_map = example.features.feature
    return {feature_handler.name: feature_handler.parse_value(feature_map)
            for feature_handler in self._feature_handlers}
