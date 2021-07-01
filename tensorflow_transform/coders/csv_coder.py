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
"""Coder classes for encoding CSV into tf.Transform datasets."""
import csv
import io

import numpy as np
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils


def _to_string(x):
  """Converts x to string.

  This will return Unicode for Py3. This is needed as a pre-processing step
  before calling csv reader/writer since it only supports Unicode for Py3.

  Args:
    x: The data to be converted.

  Returns:
    Unicode representation for Py3.

  """
  return tf.compat.as_str_any(x)


class _FixedLenFeatureHandler:
  """Handler for `FixedLenFeature` values.

  `FixedLenFeature` values will be parsed as a scalar or an array of the
  corresponding dtype. In case the value is missing the default_value will
  be returned. If the default value is not present a ValueError will be raised.
  """

  def __init__(self, name, feature_spec, index, encoder=None):
    self._name = name
    self._default_value = feature_spec.default_value
    self._index = index
    self._encoder = encoder
    self._np_dtype = feature_spec.dtype.as_numpy_dtype
    self._shape = feature_spec.shape
    self._rank = len(feature_spec.shape)
    self._size = 1
    for dim in feature_spec.shape:
      self._size *= dim

  @property
  def name(self):
    return self._name

  def encode_value(self, string_list, values):
    """Encode the value of this feature into the CSV line."""

    if self._rank == 0:
      flattened_values = [values]
    elif self._rank == 1:
      # Short-circuit the reshaping logic needed for rank > 1.
      flattened_values = values
    else:
      flattened_values = np.asarray(values, dtype=self._np_dtype).reshape(-1)

    if len(flattened_values) != self._size:
      raise ValueError(
          'FixedLenFeature "{}" got wrong number of values. Expected {} but '
          'got {}'.format(self._name, self._size, len(flattened_values)))

    if self._encoder:
      string_list[self._index] = self._encoder.encode_record(flattened_values)
    else:
      string_list[self._index] = _to_string(flattened_values[0])


class _VarLenFeatureHandler:
  """Handler for `VarLenFeature` values.

  `VarLenFeature` values will be parsed as an array of values of the
  corresponding dtype. In case the value is missing an empty array
  will be returned.
  """

  def __init__(self, name, dtype, index, encoder=None):
    self._name = name
    self._np_dtype = dtype.as_numpy_dtype
    self._index = index
    self._encoder = encoder

  @property
  def name(self):
    return self._name

  def encode_value(self, string_list, values):
    """Encode the value of this feature into the CSV line."""
    if self._encoder:
      string_list[self._index] = self._encoder.encode_record(values)
    else:
      string_list[self._index] = _to_string(values[0]) if values else ''


class CsvCoder:
  """A coder to encode CSV formatted data."""

  class _WriterWrapper:
    """A wrapper for csv.writer to make it picklable."""

    def __init__(self, delimiter):
      """Initializes the writer wrapper.

      Args:
        delimiter: A one-character string used to separate fields.
      """
      self._state = (delimiter)
      self._buffer = io.StringIO()

      # Since we use self._writer to encode individual rows, we set
      # lineterminator='' so that self._writer doesn't add a newline.
      self._writer = csv.writer(
          self._buffer, lineterminator='', delimiter=delimiter)

    def encode_record(self, record):
      """Converts the record to bytes.

      Since csv writer only supports Unicode for PY3, we need to convert them
      conditionally before calling csv writer. We always return result in bytes
      format to be consistent with current behavior.

      Args:
        record: The data to be converted.

      Returns:
        Bytes representation input.
      """
      self._writer.writerow([_to_string(x) for x in record])
      result = tf.compat.as_bytes(self._buffer.getvalue())
      # Reset the buffer.
      self._buffer.seek(0)
      self._buffer.truncate(0)
      return result

    def __getstate__(self):
      return self._state

    def __setstate__(self, state):
      self.__init__(*state)

  def __init__(self,
               column_names,
               schema,
               delimiter=',',
               secondary_delimiter=None,
               multivalent_columns=None):
    """Initializes CsvCoder.

    Args:
      column_names: Tuple of strings. Order must match the order in the file.
      schema: A `Schema` proto.
      delimiter: A one-character string used to separate fields.
      secondary_delimiter: A one-character string used to separate values within
        the same field.
      multivalent_columns: A list of names for multivalent columns that need to
        be split based on secondary delimiter.

    Raises:
      ValueError: If `schema` is invalid.
    """
    self._column_names = column_names
    self._schema = schema
    self._delimiter = delimiter
    self._secondary_delimiter = secondary_delimiter
    self._encoder = self._WriterWrapper(delimiter)

    if multivalent_columns is None:
      multivalent_columns = []
    self._multivalent_columns = multivalent_columns

    if secondary_delimiter:
      secondary_encoder = self._WriterWrapper(secondary_delimiter)
    elif multivalent_columns:
      raise ValueError(
          'secondary_delimiter unspecified for multivalent columns "{}"'.format(
              multivalent_columns))
    secondary_encoder_by_name = {
        name: secondary_encoder for name in multivalent_columns
    }
    indices_by_name = {
        name: index for index, name in enumerate(self._column_names)
    }

    def index(name):
      index = indices_by_name.get(name)
      if index is None:
        raise ValueError('Column not found: "{}"'.format(name))
      else:
        return index

    self._feature_handlers = []
    for name, feature_spec in schema_utils.schema_as_feature_spec(
        schema).feature_spec.items():
      if isinstance(feature_spec, tf.io.FixedLenFeature):
        self._feature_handlers.append(
            _FixedLenFeatureHandler(name, feature_spec, index(name),
                                    secondary_encoder_by_name.get(name)))
      elif isinstance(feature_spec, tf.io.VarLenFeature):
        self._feature_handlers.append(
            _VarLenFeatureHandler(name, feature_spec.dtype, index(name),
                                  secondary_encoder_by_name.get(name)))
      elif isinstance(feature_spec, tf.io.SparseFeature):
        index_keys = (
            feature_spec.index_key if isinstance(feature_spec.index_key, list)
            else [feature_spec.index_key])
        for key in index_keys:
          self._feature_handlers.append(
              _VarLenFeatureHandler(key, tf.int64, index(key),
                                    secondary_encoder_by_name.get(name)))
        self._feature_handlers.append(
            _VarLenFeatureHandler(feature_spec.value_key, feature_spec.dtype,
                                  index(feature_spec.value_key),
                                  secondary_encoder_by_name.get(name)))
      else:
        raise ValueError(
            'feature_spec should be one of tf.FixedLenFeature, '
            'tf.VarLenFeature or tf.SparseFeature: {!r} was {!r}'.format(
                name, type(feature_spec)))

  def __reduce__(self):
    return self.__class__, (self._column_names, self._schema, self._delimiter,
                            self._secondary_delimiter,
                            self._multivalent_columns)

  def encode(self, instance):
    """Encode a tf.transform encoded dict to a csv-formatted string.

    Args:
      instance: A python dictionary where the keys are the column names and the
        values are fixed len or var len encoded features.

    Returns:
      A csv-formatted string. The order of the columns is given by column_names.
    """
    string_list = [None] * len(self._column_names)
    for feature_handler in self._feature_handlers:
      try:
        feature_handler.encode_value(string_list,
                                     instance[feature_handler.name])
      except TypeError as e:
        raise TypeError('{} while encoding feature "{}"'.format(
            e, feature_handler.name))
    return self._encoder.encode_record(string_list)
