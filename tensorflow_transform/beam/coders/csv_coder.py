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


import apache_beam as beam
import numpy as np
from six.moves import cStringIO

import tensorflow as tf


class DecodeError(Exception):
  """Base decode error."""
  pass


class _FixedLenFeatureHandler(object):
  """Handler for `FixedLenFeature` values.

  `FixedLenFeature` values will be parsed as a scalar of the corresponding
  dtype. In case the value is missing the default_value will be returned
  If the default value is not present it will rais a ValueError.
  """

  def __init__(self, name, feature_spec):
    self._name = name
    self._default_value = feature_spec.default_value
    self._numpy_dtype = feature_spec.dtype.as_numpy_dtype

  @property
  def name(self):
    return self._name

  def parse_value(self, value):
    return np.array(value).astype(self._numpy_dtype).item(0)

  def encode_value(self, value):
    return str(value)

  def handle_missing_value(self):
    if self._default_value is None:
      raise ValueError('FixedLenFeature expected a value on column "%s".' %
                       self.name)
    else:
      return self._default_value


class _VarLenFeatureHandler(object):
  """Handler for `VarLenFeature` values.

  `VarLenFeature` values will be parsed as an array with a single element
  corresponding to the value. In case the value is missing an empty array
  will be returned.
  """

  def __init__(self, name, feature_spec):
    self._name = name
    self._numpy_dtype = feature_spec.dtype.as_numpy_dtype

  @property
  def name(self):
    return self._name

  def parse_value(self, value):
    return np.array(value).astype(self._numpy_dtype)

  def encode_value(self, value):
    if value:
      return str(value.item(0))
    else:
      return ''

  def handle_missing_value(self):
    return []


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


class CsvCoder(beam.coders.Coder):
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

  def __init__(self, column_names, schema, delimiter=','):
    """Initializes CsvCoder.

    Args:
      column_names: Tuple of strings. Order must match the order in the file.
      schema: A tf-transform schema, right now is a feature spec.
      delimiter: A one-character string used to separate fields.
    Raises:
      ValueError: If the schema specifies `SparseFeature` as an input.
    """
    self._column_names = column_names

    self._feature_handlers = []
    for name in self._column_names:
      feature_spec = schema[name]
      if isinstance(feature_spec, tf.FixedLenFeature):
        self._feature_handlers.append(
            _FixedLenFeatureHandler(name, feature_spec))
      elif isinstance(feature_spec, tf.VarLenFeature):
        self._feature_handlers.append(
            _VarLenFeatureHandler(name, feature_spec))
      elif isinstance(feature_spec, tf.SparseFeature):
        raise ValueError('SparseFeatures are not supported for CSV data "%s".' %
                         name)
      else:
        raise ValueError('feature_spec should be one of tf.FixedLenFeature, '
                         'tf.VarLenFeature or tf.SparseFeature: %s was %s' %
                         (name, type(feature_spec)))

    self._reader = self._ReaderWrapper(delimiter)
    self._encoder = self._WriterWrapper(delimiter)

  # Please run tensorflow_transform/beam/benchmark_coders_test.py
  # if you make any changes on these methods.
  def decode(self, csv_string):
    """Decodes the given string record according to the schema.

    There are some limitations though. This decoder will only convert integer
    and floating values if tf.DType.is_integer and tf.DType.is_floating
    respectively.

    Missing value handling is as follows:

    1.a) If FixedLenFeature and has a default value, use that value for missing
         entries.
    1.b) If FixedLenFeature and doesn't has default value throw an Exception on
         missing entries.

    2) For VarLengthFeature return an empty array.

    Args:
      csv_string: String to be decoded.

    Returns:
      Dictionary of column name to value.

    Raises:
      DecodeError: If columns do not match specified csv headers.
      ValueError: If some numeric column has non-numeric data.
    """
    try:
      record = self._reader.read_record(csv_string)
    except Exception as e:  # pylint: disable=broad-except
      raise DecodeError('%s: %s' % (e, csv_string))

    # Check record length mismatches.
    if len(record) != len(self._column_names):
      raise DecodeError(
          'Columns do not match specified csv headers: %s -> %s' % (
              self._column_names, record))
    result = {}
    for index, feature_handler in enumerate(self._feature_handlers):
      value = record[index]
      # If the string value is completely empty handle appropriately.
      if value:
        result[feature_handler.name] = feature_handler.parse_value(value)
      else:
        result[feature_handler.name] = feature_handler.handle_missing_value()
    return result

  def encode(self, python_data):
    """Encode a tf.transform encoded dict to a csv-formatted string.

    Args:
      python_data: A python dictionary where the keys are the column names and
        the values are fixed len or var len encoded features.

    Returns:
      A csv-formatted string. The order of the columns is given by column_names.
    """
    record = []
    for feature_handler in self._feature_handlers:
      record.append(feature_handler.encode_value(
          python_data[feature_handler.name]))
    return self._encoder.encode_record(record)
