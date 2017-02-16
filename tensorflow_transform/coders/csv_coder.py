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
"""Coder classes for encoding/decoding CSV into tf.Transform datasets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv


from six.moves import cStringIO

import tensorflow as tf


def _make_cast_fn(dtype):
  """Return a function to extract the typed value from the feature.

  For performance reasons it is preferred to have the cast fn
  constructed once (for each handler).

  Args:
    dtype: The type of the Tensorflow feature.
  Returns:
    A function to extract the value field from a string depending on dtype.
  """
  if dtype.is_integer:
    return long
  elif dtype.is_floating:
    return float
  else:
    return lambda x: x


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
    raise DecodeError('%s: %s' % (e, value))
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

  @property
  def name(self):
    return self._name

  def parse_value(self, string_list):
    value = string_list[self._index]
    if value:
      if self._reader:
        return map(self._cast_fn, _decode_with_reader(value, self._reader))
      else:
        return self._cast_fn(value)
    elif self._default_value is None:
      raise ValueError('FixedLenFeature expected a value on column "%s".' %
                       self._name)
    else:
      return self._default_value

  def encode_value(self, string_list, values):
    if self._encoder:
      string_list[self._index] = self._encoder.encode_record(map(str, values))
    else:
      string_list[self._index] = str(values)


class _VarLenFeatureHandler(object):
  """Handler for `VarLenFeature` values.

  `VarLenFeature` values will be parsed as an array of values of the
  corresponding dtype. In case the value is missing an empty array
  will be returned.
  """

  def __init__(self, name, feature_spec, index, reader=None, encoder=None):
    self._name = name
    self._cast_fn = _make_cast_fn(feature_spec.dtype)
    self._index = index
    self._reader = reader
    self._encoder = encoder

  @property
  def name(self):
    return self._name

  def parse_value(self, string_list):
    value = string_list[self._index]
    if value:
      if self._reader:
        return map(self._cast_fn, _decode_with_reader(value, self._reader))
      else:
        return [self._cast_fn(value)]
    else:
      return []

  def encode_value(self, string_list, values):
    if self._encoder:
      string_list[self._index] = self._encoder.encode_record(map(str, values))
    else:
      string_list[self._index] = str(values[0]) if values else ''


class _SparseFeatureHandler(object):
  """Handler for `SparseFeature` values.

  `SparseFeature` values will be parsed as a tuple of 1-D arrays where the first
  array corresponds to the values and the second to their indexes.
  """

  def __init__(self, name, feature_spec, value_index, index_index,
               reader=None, encoder=None):
    self._name = name
    self._cast_fn = _make_cast_fn(feature_spec.dtype)
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
    value = string_list[self._value_index]
    index = string_list[self._index_index]
    if value and index:
      if self._reader:
        values = map(self._cast_fn, _decode_with_reader(value, self._reader))
        indices = map(long, _decode_with_reader(index, self._reader))
      else:
        values = [self._cast_fn(value)]
        indices = [long(index)]
      i_min, i_max = min(indices), max(indices)
      if i_min < 0 or i_max >= self._size:
        i_bad = i_min if i_min < 0 else i_max
        raise ValueError('SparseFeature "%s" has index %s out of range [0, %s)'
                         % (self._name, i_bad, self._size))
      return (values, indices)
    elif value and not index:
      raise ValueError(
          'SparseFeature "%s" expected an index in column "%s".' %
          (self._name, self._index_name))
    elif not value and index:
      raise ValueError(
          'SparseFeature "%s" expected a value in column "%s".' %
          (self._name, self._value_name))
    else:
      return ([], [])

  def encode_value(self, string_list, sparse_value):
    value, index = sparse_value
    if len(value) == len(index):
      if self._encoder:
        string_list[self._value_index] = self._encoder.encode_record(
            map(str, value))
        string_list[self._index_index] = self._encoder.encode_record(
            map(str, index))
      else:
        string_list[self._value_index] = str(value[0]) if value else ''
        string_list[self._index_index] = str(index[0]) if index else ''
    else:
      raise ValueError(
          'SparseFeature "%s" has value and index unaligned "%s" vs "%s".' %
          (self._name, value, index))


class DecodeError(Exception):
  """Base decode error."""
  pass


class EncodeError(Exception):
  """Base encode error."""
  pass


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

  def next(self):
    # This API currently supports only one line at a time.
    # If this ever supports more than one row be aware that DictReader might
    # attempt to read more than one record if one of the records is empty line
    line_length = len(self._lines)
    if line_length == 0:
      raise DecodeError(
          'Columns do not match specified csv headers: empty line was found')
    assert line_length == 1, 'Unexpected number of lines %s' % line_length
    # This doesn't maintain insertion order to the list, which is fine
    # because the list has only 1 element. If there were more and we wanted
    # to maintain order and timecomplexity we would switch to deque.popleft.
    return self._lines.pop()


class CsvCoder(object):
  """A coder to encode and decode CSV formatted data."""

  class _ReaderWrapper(object):
    """A wrapper for csv.reader to make it picklable."""

    def __init__(self, delimiter):
      self._state = (delimiter)
      self._line_generator = _LineGenerator()
      self._reader = csv.reader(self._line_generator, delimiter=str(delimiter))

    def read_record(self, x):
      self._line_generator.push_line(x)
      return self._reader.next()

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
      self._buffer = cStringIO()

      # Since we use self._writer to encode individual rows, we set
      # lineterminator='' so that self._writer doesn't add a newline.
      self._writer = csv.writer(
          self._buffer,
          lineterminator='',
          delimiter=delimiter)

    def encode_record(self, record):
      self._writer.writerow(record)
      result = self._buffer.getvalue()
      # Reset the buffer.
      self._buffer.seek(0)
      self._buffer.truncate(0)
      return result

    def __getstate__(self):
      return self._state

    def __setstate__(self, state):
      self.__init__(*state)

  def __init__(self, column_names, schema, delimiter=',',
               secondary_delimiter=None, multivalent_columns=None):
    """Initializes CsvCoder.

    Args:
      column_names: Tuple of strings. Order must match the order in the file.
      schema: A `Schema` object.
      delimiter: A one-character string used to separate fields.
      secondary_delimiter: A one-character string used to separate values within
        the same field.
      multivalent_columns: A list of names for multivalent columns that need
          to be split based on secondary delimiter.
    Raises:
      ValueError: If `schema` is invalid.
    """
    self._column_names = column_names
    self._schema = schema
    self._delimiter = delimiter
    self._secondary_delimiter = secondary_delimiter
    self._multivalent_columns = multivalent_columns
    self._reader = self._ReaderWrapper(delimiter)
    self._encoder = self._WriterWrapper(delimiter)

    if multivalent_columns is None:
      multivalent_columns = []
    if secondary_delimiter:
      secondary_reader = self._ReaderWrapper(secondary_delimiter)
      secondary_encoder = self._WriterWrapper(secondary_delimiter)
    elif multivalent_columns:
      raise ValueError('secondary_delimiter unspecified for multivalent'
                       'columns %s' % multivalent_columns)
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
        raise ValueError('Column not found: %s' % name)
      else:
        return index

    self._feature_handlers = []
    for name, feature_spec in schema.as_feature_spec().items():
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
        raise ValueError('feature_spec should be one of tf.FixedLenFeature, '
                         'tf.VarLenFeature or tf.SparseFeature: %s was %s' %
                         (name, type(feature_spec)))

  def __reduce__(self):
    return CsvCoder, (self._column_names,
                      self._schema,
                      self._delimiter,
                      self._secondary_delimiter,
                      self._multivalent_columns)

  @property
  def name(self):
    return 'csv'

  def encode(self, instance):
    """Encode a tf.transform encoded dict to a csv-formatted string.

    Args:
      instance: A python dictionary where the keys are the column names and
        the values are fixed len or var len encoded features.

    Returns:
      A csv-formatted string. The order of the columns is given by column_names.
    """
    string_list = [None] * len(self._column_names)
    for feature_handler in self._feature_handlers:
      feature_handler.encode_value(string_list, instance[feature_handler.name])
    return self._encoder.encode_record(string_list)

  # Please run tensorflow_transform/coders/benchmark_coders_test.py
  # if you make any changes on these methods.
  def decode(self, csv_string):
    """Decodes the given string record according to the schema.

    Missing value handling is as follows:

    1.a) If FixedLenFeature and has a default value, use that value for missing
         entries.
    1.b) If FixedLenFeature and doesn't has default value throw an Exception on
         missing entries.

    2) For VarLenFeature return an empty array.

    3) For SparseFeature throw an Exception if only one of the indices or values
       has a missing entry. If both indices and values are missing, return
       a tuple of 2 empty arrays.

    Args:
      csv_string: String to be decoded.

    Returns:
      Dictionary of column name to value.

    Raises:
      DecodeError: If columns do not match specified csv headers.
      ValueError: If some numeric column has non-numeric data.
    """
    try:
      raw_values = self._reader.read_record(csv_string)
    except Exception as e:  # pylint: disable=broad-except
      raise DecodeError('%s: %s' % (e, csv_string))

    # Check record length mismatches.
    if len(raw_values) != len(self._column_names):
      raise DecodeError(
          'Columns do not match specified csv headers: %s -> %s' % (
              self._column_names, raw_values))

    return {feature_handler.name: feature_handler.parse_value(raw_values)
            for feature_handler in self._feature_handlers}
