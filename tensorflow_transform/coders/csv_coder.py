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
"""Coder classes for encoding/decoding CSV into tf.Transform datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
# GOOGLE-INITIALIZATION

import numpy as np
import six
from six import moves
import tensorflow as tf


# This is in agreement with Tensorflow conversions for Unicode values for both
# Python 2 and 3 (and also works for non-Unicode objects).
# TODO(b/123241312): Remove this fn since we will only support bytes input.
def _to_bytes(x):
  """Converts x to bytes."""
  return tf.compat.as_bytes(x)


def _to_string(x):
  """Converts x to string.

  This will return bytes for Py2 and Unicode for Py3. This is needed as a
  pre-processing step before calling csv reader/writer since it only supports
  bytes for Py2 and Unicode for Py3.

  Args:
    x: The data to be converted.

  Returns:
    Bytes representation of x for Py2 and Unicode representation for Py3.

  """
  return tf.compat.as_str_any(x)


def _elements_to_bytes(x):
  if isinstance(x, (list, np.ndarray)):
    return list(map(_to_bytes, x))
  return _to_bytes(x)


# TODO(b/119621361): Consider harmonizing _make_cast_fn() for all coders.
def _make_cast_fn(dtype):
  """Return a function to extract the typed value from the feature.

  For performance reasons it is preferred to have the cast fn
  constructed once (for each handler).

  For boolean values the function will only accept "True" or "False" as input.

  Args:
    dtype: The type of the Tensorflow feature.

  Returns:
    A function to extract the value field from a string depending on dtype.
  """

  def to_boolean(value):
    if value == 'True':
      return True
    elif value == 'False':
      return False
    else:
      raise ValueError('expected "True" or "False" as inputs.')

  if dtype.is_integer:
    # In Python 2, if the value is too large to fit into an int, int(..) returns
    # a long, but ints are cheaper to use when possible.
    return int
  elif dtype.is_floating:
    return float
  elif dtype.is_bool:
    return to_boolean
  else:
    return _elements_to_bytes


def _decode_with_reader(value, reader):
  """Parse the input value into a list of strings.

  Args:
    value: A string to be decoded.
    reader: A optional reader for splitting the input string.

  Returns:
    A list of strings.
  Raises:
    DecodeError: An error occurred when parsing the value with reader.
  """
  try:
    result = reader.read_record(value)
  except Exception as e:  # pylint: disable=broad-except
    raise DecodeError('{}: {}'.format(e, value))
  return result


class _FixedLenFeatureHandler(object):
  """Handler for `FixedLenFeature` values.

  `FixedLenFeature` values will be parsed as a scalar or an array of the
  corresponding dtype. In case the value is missing the default_value will
  be returned. If the default value is not present a ValueError will be raised.
  """

  def __init__(self, name, feature_spec, index, reader=None, encoder=None):
    self._name = name
    self._cast_fn = _make_cast_fn(feature_spec.dtype)
    self._default_value = feature_spec.default_value
    self._index = index
    self._reader = reader
    self._encoder = encoder
    self._np_dtype = feature_spec.dtype.as_numpy_dtype
    self._shape = feature_spec.shape
    self._rank = len(feature_spec.shape)
    self._size = 1
    for dim in feature_spec.shape:
      self._size *= dim
    # Check that the size of the feature matches the valency.
    if self._size != 1 and not self._reader:
      raise ValueError(
          'FixedLenFeature "{}" was not multivalent (see CsvCoder constructor) '
          'but had shape {} whose size was not 1'.format(
              name, feature_spec.shape))

  @property
  def name(self):
    return self._name

  def parse_value(self, string_list):
    """Parse the value of this feature from string list split from CSV line."""
    value_str = string_list[self._index]
    if value_str and self._reader:
      # NOTE: The default value is ignored when self._reader is set.
      values = list(
          map(self._cast_fn, _decode_with_reader(value_str, self._reader)))
    elif value_str:
      values = [self._cast_fn(value_str)]
    elif self._default_value is not None:
      values = [self._default_value]
    else:
      values = []

    if len(values) != self._size:
      if self._reader:
        raise ValueError(
            'FixedLenFeature "{}" got wrong number of values. Expected'
            ' {} but got {}'.format(self._name, self._size, len(values)))
      else:
        # If there is no reader and size of values doesn't match, then this
        # must be because the value was missing.
        raise ValueError('expected a value on column "{}"'.format(self._name))

    if self._rank == 0:
      # Encode the values as a scalar if shape == [].
      return values[0]
    elif self._rank == 1:
      # Short-circuit the reshaping logic needed for rank > 1.
      return np.asarray(values, dtype=self._np_dtype)
    else:
      return np.asarray(values, dtype=self._np_dtype).reshape(self._shape)

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


class _VarLenFeatureHandler(object):
  """Handler for `VarLenFeature` values.

  `VarLenFeature` values will be parsed as an array of values of the
  corresponding dtype. In case the value is missing an empty array
  will be returned.
  """

  def __init__(self, name, feature_spec, index, reader=None, encoder=None):
    self._name = name
    self._cast_fn = _make_cast_fn(feature_spec.dtype)
    self._np_dtype = feature_spec.dtype.as_numpy_dtype
    self._index = index
    self._reader = reader
    self._encoder = encoder

  @property
  def name(self):
    return self._name

  def parse_value(self, string_list):
    """Parse the value of this feature from string list split from CSV line."""
    value_str = string_list[self._index]
    if value_str and self._reader:
      return list(
          map(self._cast_fn, _decode_with_reader(value_str, self._reader)))
    elif value_str:
      return [self._cast_fn(value_str)]
    else:
      return []

  def encode_value(self, string_list, values):
    """Encode the value of this feature into the CSV line."""
    if self._encoder:
      string_list[self._index] = self._encoder.encode_record(values)
    else:
      string_list[self._index] = _to_string(values[0]) if values else None


class _SparseFeatureHandler(object):
  """Handler for `SparseFeature` values.

  `SparseFeature` values will be parsed as a tuple of 1-D arrays where the first
  array corresponds to their indices and the second to the values.
  """

  def __init__(self,
               name,
               feature_spec,
               value_index,
               index_index,
               reader=None,
               encoder=None):
    self._name = name
    self._cast_fn = _make_cast_fn(feature_spec.dtype)
    self._np_dtype = feature_spec.dtype.as_numpy_dtype
    self._value_index = value_index
    self._value_name = feature_spec.value_key
    self._index_index = index_index
    self._index_name = feature_spec.index_key
    self._size = feature_spec.size
    self._reader = reader
    self._encoder = encoder

  @property
  def name(self):
    return self._name

  def parse_value(self, string_list):
    """Parse the value of this feature from string list split from CSV line."""
    value_str = string_list[self._value_index]
    index_str = string_list[self._index_index]

    if value_str and self._reader:
      values = list(
          map(self._cast_fn, _decode_with_reader(value_str, self._reader)))
    elif value_str:
      values = [self._cast_fn(value_str)]
    else:
      values = []

    # In Python 2, if the value is too large to fit into an int, int(..) returns
    # a long, but ints are cheaper to use when possible.
    if index_str and self._reader:
      indices = list(map(int, _decode_with_reader(index_str, self._reader)))
    elif index_str:
      indices = [int(index_str)]
    else:
      indices = []

    # Check that all indices are in range.
    # TODO(b/36040669): Move this validation so it can be shared by all coders.
    if indices:
      i_min, i_max = min(indices), max(indices)
      if i_min < 0 or i_max >= self._size:
        i_bad = i_min if i_min < 0 else i_max
        raise ValueError(
            'SparseFeature "{}" has index {} out of range [0, {})'.format(
                self._name, i_bad, self._size))

    if len(values) != len(indices):
      raise ValueError(
          'SparseFeature "{}" has indices and values of different lengths: '
          'values: {}, indices: {}'.format(self._name, values, indices))

    return (np.asarray(indices, dtype=np.int64),
            np.asarray(values, dtype=self._np_dtype))

  def encode_value(self, string_list, sparse_value):
    """Encode the value of this feature into the CSV line."""
    index, value = sparse_value
    if len(value) == len(index):
      if self._encoder:
        string_list[self._value_index] = self._encoder.encode_record(value)
        string_list[self._index_index] = self._encoder.encode_record(index)
      else:
        string_list[self._value_index] = _to_string(value[0]) if value else ''
        string_list[self._index_index] = _to_string(index[0]) if index else ''
    else:
      raise ValueError(
          'SparseFeature {!r} has value and index unaligned {!r} vs {!r}.'
          .format(self._name, value, index))


class DecodeError(Exception):
  """Base decode error."""
  pass


class EncodeError(Exception):
  """Base encode error."""
  pass


# TODO(b/32491265) Revisit using cStringIO for design compatibility with
# coders.CsvCoder.
class _LineGenerator(object):
  """A csv line generator that allows feeding lines to a csv.DictReader."""

  def __init__(self):
    self._lines = []

  def push_line(self, line):
    # This API currently supports only one line at a time.
    assert not self._lines
    self._lines.append(line)

  def __iter__(self):
    return self

  def __next__(self):
    # This API currently supports only one line at a time.
    # If this ever supports more than one row be aware that DictReader might
    # attempt to read more than one record if one of the records is empty line
    line_length = len(self._lines)
    if line_length == 0:
      raise DecodeError(
          'Columns do not match specified csv headers: empty line was found')
    assert line_length == 1, 'Unexpected number of lines %d' % line_length
    # This doesn't maintain insertion order to the list, which is fine
    # because the list has only 1 element. If there were more and we wanted
    # to maintain order and timecomplexity we would switch to deque.popleft.
    return self._lines.pop()

  next = __next__


class CsvCoder(object):
  """A coder to encode and decode CSV formatted data."""

  class _ReaderWrapper(object):
    """A wrapper for csv.reader to make it picklable."""

    def __init__(self, delimiter):
      self._state = (delimiter)
      self._line_generator = _LineGenerator()
      self._reader = csv.reader(
          self._line_generator, delimiter=_to_string(delimiter))

    def read_record(self, x):
      """Reads out bytes for PY2 and Unicode for PY3."""
      if six.PY2:
        line = _to_bytes(x)
      else:
        line = _to_string(x)
      self._line_generator.push_line(line)
      return next(self._reader)

    def __getstate__(self):
      return self._state

    def __setstate__(self, state):
      self.__init__(*state)

  class _WriterWrapper(object):
    """A wrapper for csv.writer to make it picklable."""

    def __init__(self, delimiter):
      """Initializes the writer wrapper.

      Args:
        delimiter: A one-character string used to separate fields.
      """
      self._state = (delimiter)
      self._buffer = moves.cStringIO()

      # Since we use self._writer to encode individual rows, we set
      # lineterminator='' so that self._writer doesn't add a newline.
      self._writer = csv.writer(
          self._buffer, lineterminator='', delimiter=delimiter)

    def encode_record(self, record):
      """Converts the record to bytes.

      Since csv writer only supports bytes for PY2 and Unicode for PY3, we need
      to convert them conditionally before calling csv writer. We always return
      result in bytes format to be consistent with current behavior.

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
      schema: A `Schema` object.
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
    self._reader = self._ReaderWrapper(delimiter)
    self._encoder = self._WriterWrapper(delimiter)

    if multivalent_columns is None:
      multivalent_columns = []
    self._multivalent_columns = multivalent_columns

    if secondary_delimiter:
      secondary_reader = self._ReaderWrapper(secondary_delimiter)
      secondary_encoder = self._WriterWrapper(secondary_delimiter)
    elif multivalent_columns:
      raise ValueError(
          'secondary_delimiter unspecified for multivalent columns "{}"'.format(
              multivalent_columns))
    secondary_reader_by_name = {
        name: secondary_reader for name in multivalent_columns
    }
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
    for name, feature_spec in six.iteritems(schema.as_feature_spec()):
      if isinstance(feature_spec, tf.FixedLenFeature):
        self._feature_handlers.append(
            _FixedLenFeatureHandler(name, feature_spec, index(name),
                                    secondary_reader_by_name.get(name),
                                    secondary_encoder_by_name.get(name)))
      elif isinstance(feature_spec, tf.VarLenFeature):
        self._feature_handlers.append(
            _VarLenFeatureHandler(name, feature_spec, index(name),
                                  secondary_reader_by_name.get(name),
                                  secondary_encoder_by_name.get(name)))
      elif isinstance(feature_spec, tf.SparseFeature):
        self._feature_handlers.append(
            _SparseFeatureHandler(name, feature_spec,
                                  index(feature_spec.value_key),
                                  index(feature_spec.index_key),
                                  secondary_reader_by_name.get(name),
                                  secondary_encoder_by_name.get(name)))
      else:
        raise ValueError(
            'feature_spec should be one of tf.FixedLenFeature, '
            'tf.VarLenFeature or tf.SparseFeature: {!r} was {!r}'.format(
                name, type(feature_spec)))

  def __reduce__(self):
    return CsvCoder, (self._column_names, self._schema, self._delimiter,
                      self._secondary_delimiter, self._multivalent_columns)

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

  # Please run tensorflow_transform/coders/benchmark_coders_test.py
  # if you make any changes on these methods.
  def decode(self, csv_string):
    """Decodes the given string record according to the schema.

    Missing value handling is as follows:

    1. For FixedLenFeature:
        1. If FixedLenFeature and has a default value, use that value for
        missing entries.
        2. If FixedLenFeature and doesn't have default value throw an Exception
        on missing entries.

    2. For VarLenFeature return an empty array.

    3. For SparseFeature throw an Exception if only one of the indices or values
       has a missing entry. If both indices and values are missing, return
       a tuple of 2 empty arrays.

    For the case of multivalent columns a ValueError will occur if
    FixedLenFeature gets the wrong number of values, or a SparseFeature gets
    different length indices and values.

    Args:
      csv_string: String to be decoded.

    Returns:
      Dictionary of column name to value.

    Raises:
      DecodeError: If columns do not match specified csv headers.
      ValueError: If some numeric column has non-numeric data, if a
          SparseFeature has missing indices but not values or vice versa or
          multivalent data has the wrong length.
    """
    try:
      raw_values = self._reader.read_record(csv_string)
    except Exception as e:  # pylint: disable=broad-except
      raise DecodeError('%s: %s' % (e, csv_string))

    # An empty string when we expect a single column is potentially valid.  This
    # is probably more permissive than the csv standard but is useful for
    # testing so that we can test single column CSV lines.
    if not raw_values and len(self._column_names) == 1:
      raw_values = ['']

    # Check record length mismatches.
    if len(raw_values) != len(self._column_names):
      raise DecodeError(
          'Columns do not match specified csv headers: {} -> {}'.format(
              self._column_names, raw_values))

    return {
        feature_handler.name: feature_handler.parse_value(raw_values)
        for feature_handler in self._feature_handlers
    }
