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
"""Coder classes for encoding TF Examples into tf.Transform datasets."""

# TODO(b/33688275): Rename ExampleProto to just Example, for all aspects of this
# API (eg Classes, Files, Benchmarks etc).

import numpy as np
import tensorflow as tf
from tensorflow_transform import common_types
from tensorflow_transform.tf_metadata import schema_utils


# This function needs to be called at pipeline execution time as it depends on
# the protocol buffer library installed in the workers (which might be different
# from the one installed in the pipeline constructor).
def _make_cast_fn(np_dtype):
  """Return a function to extract the typed value from the feature.

  For performance reasons it is preferred to have the cast fn
  constructed once (for each handler).

  Args:
    np_dtype: The numpy type of the Tensorflow feature.

  Returns:
    A function to extract the value field from a string depending on dtype.
  """

  def identity(x):
    return x

  # This is in agreement with Tensorflow conversions for Unicode values for both
  # Python 2 and 3 (and also works for non-Unicode objects). It is also in
  # agreement with the testWithUnicode of the Beam impl.
  def utf8(s):
    return s if isinstance(s, bytes) else s.encode('utf-8')

  vectorize = np.vectorize(utf8)

  def string_cast(x):
    if isinstance(x, list) or isinstance(x, np.ndarray) and x.ndim > 0:
      return map(utf8, x)
    elif isinstance(x, np.ndarray):
      return vectorize(x).tolist()
    return utf8(x)

  if issubclass(np_dtype, np.floating) or issubclass(np_dtype, np.integer):
    return identity

  return string_cast


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

  if dtype.is_floating:
    return lambda feature: feature.float_list.value

  return lambda feature: feature.bytes_list.value


class _FixedLenFeatureHandler:
  """Handler for `FixedLenFeature` values.

  `FixedLenFeature` values will be parsed to a list of the corresponding
  dtype.
  """

  def __init__(self, name, feature_spec):
    self._name = name
    self._np_dtype = feature_spec.dtype.as_numpy_dtype
    self._value_fn = _make_feature_value_fn(feature_spec.dtype)
    self._rank = len(feature_spec.shape)
    self._size = 1
    for dim in feature_spec.shape:
      self._size *= dim

  @property
  def name(self):
    """The name of the feature."""
    return self._name

  def initialize_encode_cache(self, example):
    """Initialize fields (performance caches) that point to example's state."""
    self._cast_fn = _make_cast_fn(self._np_dtype)
    self._value = self._value_fn(example.features.feature[self._name])

  def encode_value(self, values):
    """Encodes a feature into its Example proto representation."""
    del self._value[:]
    if self._rank == 0:
      scalar_value = values if not isinstance(values,
                                              np.ndarray) else values.item()
      self._value.append(self._cast_fn(scalar_value))
    else:
      flattened_values = (
          values if self._rank == 1 else np.asarray(
              values, dtype=self._np_dtype).reshape(-1))
      if len(flattened_values) != self._size:
        raise ValueError('FixedLenFeature %r got wrong number of values. '
                         'Expected %d but got %d' %
                         (self._name, self._size, len(flattened_values)))
      self._value.extend(self._cast_fn(flattened_values))


class _VarLenFeatureHandler:
  """Handler for `VarLenFeature` values.

  `VarLenFeature` values will be parsed as an array of the corresponding dtype.
  """

  def __init__(self, name, dtype):
    self._name = name
    self._np_dtype = dtype.as_numpy_dtype
    self._value_fn = _make_feature_value_fn(dtype)

  @property
  def name(self):
    """The name of the feature."""
    return self._name

  def initialize_encode_cache(self, example):
    """Initialize fields (performance caches) that point to example's state."""
    self._cast_fn = _make_cast_fn(self._np_dtype)
    self._feature = example.features.feature[self._name]
    self._value = self._value_fn(self._feature)

  def encode_value(self, values):
    """Encode values as tf.train.Feature."""
    if values is None:
      self._feature.Clear()
      # Note after Clear(), self._value no longer points to a submessage of
      # self._feature so we need to reset it.
      self._value = self._value_fn(self._feature)
    else:
      del self._value[:]

      # Scalar must be length 1 array.
      values = values if isinstance(values, (list, np.ndarray)) else [values]
      casted = self._cast_fn(values)
      self._value.extend(casted)


class ExampleProtoCoder:
  """A coder between maybe-serialized TF Examples and tf.Transform datasets."""

  def __init__(self, schema, serialized=True):
    """Build an ExampleProtoCoder.

    Args:
      schema: A `Schema` proto.
      serialized: Whether to encode serialized Example protos (as opposed to
        in-memory Example protos).

    Raises:
      ValueError: If `schema` is invalid.
    """
    self._schema = schema
    self._serialized = serialized

    # Using pre-allocated tf.train.Example and FeatureHandler objects for
    # performance reasons.
    #
    # Since the output of "encode" is deep as opposed to shallow
    # transformations, and since the schema always fully defines the Example's
    # FeatureMap (ie all fields are always cleared/assigned or copied), the
    # optimization and implementation are correct and thread-compatible.
    self._encode_example_cache = tf.train.Example()
    self._feature_handlers = []
    for name, feature_spec in schema_utils.schema_as_feature_spec(
        schema).feature_spec.items():
      if isinstance(feature_spec, tf.io.FixedLenFeature):
        self._feature_handlers.append(
            _FixedLenFeatureHandler(name, feature_spec))
      elif isinstance(feature_spec, tf.io.VarLenFeature):
        self._feature_handlers.append(
            _VarLenFeatureHandler(name, feature_spec.dtype))
      elif isinstance(feature_spec, tf.io.SparseFeature):
        index_keys = (
            feature_spec.index_key if isinstance(feature_spec.index_key, list)
            else [feature_spec.index_key])
        for index_key in index_keys:
          self._feature_handlers.append(
              _VarLenFeatureHandler(index_key, tf.int64))
        self._feature_handlers.append(
            _VarLenFeatureHandler(feature_spec.value_key, feature_spec.dtype))
      elif common_types.is_ragged_feature(feature_spec):
        uniform_partition = False
        for partition in feature_spec.partitions:
          if isinstance(partition, tf.io.RaggedFeature.RowLengths):
            if uniform_partition:
              raise ValueError(
                  'Encountered ragged dimension after uniform for feature '
                  '"{}": only inner dimensions can be uniform. Feature spec '
                  'is {}'.format(name, feature_spec))
            self._feature_handlers.append(
                _VarLenFeatureHandler(partition.key, tf.int64))
          elif isinstance(partition, tf.io.RaggedFeature.UniformRowLength):
            # We don't encode uniform partitions since they can be recovered
            # from the shape information.
            uniform_partition = True
          else:
            raise ValueError(
                'Only `RowLengths` and `UniformRowLength` partitions of ragged '
                'features are supported, got {}'.format(type(partition)))
        self._feature_handlers.append(
            _VarLenFeatureHandler(feature_spec.value_key, feature_spec.dtype))
      else:
        raise ValueError('feature_spec should be one of tf.io.FixedLenFeature, '
                         'tf.io.VarLenFeature, tf.io.SparseFeature or '
                         'tf.io.RaggedFeature: "{}" was {}'.format(
                             name, type(feature_spec)))

    for feature_handler in self._feature_handlers:
      feature_handler.initialize_encode_cache(self._encode_example_cache)

  def __reduce__(self):
    return self.__class__, (self._schema, self._serialized)

  def encode(self, instance):
    """Encode a tf.transform encoded dict as tf.Example."""
    # The feature handles encode using the self._encode_example_cache.
    for feature_handler in self._feature_handlers:
      value = instance[feature_handler.name]
      try:
        feature_handler.encode_value(value)
      except TypeError as e:
        raise TypeError('%s while encoding feature "%s"' %
                        (e, feature_handler.name))

    if self._serialized:
      return self._encode_example_cache.SerializeToString()

    result = tf.train.Example()
    result.CopyFrom(self._encode_example_cache)
    return result
