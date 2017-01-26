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
"""Tensorflow-transform CSVCoder tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_transform.beam.coders import csv_coder

import unittest


class TestCSVCoder(unittest.TestCase):

  _COLUMNS = ['numeric1', 'text1', 'category1', 'numeric2']
  # The following input schema has no default values, so any invocations to
  # decode with missing values should raise an error. CsvDecoderTest adds good
  # coverage for missing value handling.
  _INPUT_SCHEMA = {
      'numeric1': tf.FixedLenFeature(shape=[], dtype=tf.int32),
      'numeric2': tf.VarLenFeature(dtype=tf.float32),
      'text1': tf.FixedLenFeature(shape=[], dtype=tf.string),
      'category1': tf.VarLenFeature(dtype=tf.string)
  }

  def _assert_encode_decode(self, coder, data, expected_decoded):
    decoded = coder.decode(data)
    self.assertEqual(decoded, expected_decoded)

    encoded = coder.encode(decoded)
    self.assertEqual(encoded, data)

    decoded_again = coder.decode(encoded)
    self.assertEqual(decoded_again, expected_decoded)

  def _assert_encode_not_equal_decode(self, coder, data, expected_decoded):
    decoded = coder.decode(data)
    self.assertEqual(decoded, expected_decoded)

    # Encoder turns the original numbers-strings into real numbers.
    encoded = coder.encode(decoded)
    # Hence assert not equal.
    self.assertNotEqual(encoded, data)

    decoded_again = coder.decode(encoded)
    self.assertEqual(decoded_again, expected_decoded)

  def test_csv_coder(self):
    data = '12,"this is a ,text",female,89.0'

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)
    expected_decoded = {'category1': 'female',
                        'numeric1': 12,
                        'text1': 'this is a ,text',
                        'numeric2': 89.0}
    self._assert_encode_decode(coder, data, expected_decoded)

  def test_coder_with_tab(self):
    data = '12\t"this is a \ttext"\tfemale\t89.0'

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA,
                               delimiter='\t')
    expected_decoded = {'category1': 'female',
                        'numeric1': 12,
                        'text1': 'this is a \ttext',
                        'numeric2': 89.0}
    self._assert_encode_decode(coder, data, expected_decoded)

  def test_data_types(self):
    # The numbers are strings.
    data = '"12","this is a ,text",female,"89.0"'

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)
    expected_decoded = {'category1': 'female',
                        'numeric1': 12,
                        'text1': 'this is a ,text',
                        'numeric2': 89.0}
    self._assert_encode_not_equal_decode(coder, data, expected_decoded)

  def test_missing_data(self):
    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    data = '12,,female,89.0'
    with self.assertRaisesRegexp(ValueError,
                                 'expected a value on column "text1"'):
      coder.decode(data)

  def test_missing_numeric_data(self):
    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # The numbers are missing.
    data = ',"this is a ,text",female,89.0'
    with self.assertRaisesRegexp(ValueError,
                                 'expected a value on column "numeric1"'):
      coder.decode(data)

  def test_bad_row(self):
    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # The data has a wrong number of columns.
    data = '12,"this is a ,text",female,89.0,"oh no, I\'m an error"'
    with self.assertRaisesRegexp(Exception,
                                 'Columns do not match specified csv headers'):
      coder.decode(data)

    # The data has less columns.
    data = '12,"this is a ,text",female"'
    with self.assertRaisesRegexp(Exception,
                                 'Columns do not match specified csv headers'):
      coder.decode(data)


class CsvCoderDecodeTest(unittest.TestCase):

  def test_all_values_present(self):
    columns = ['a', 'b', 'c']
    input_schema = {
        'b': tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'a': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'c': tf.VarLenFeature(dtype=tf.string),
    }
    decoder = csv_coder.CsvCoder(
        column_names=columns, schema=input_schema, delimiter=',')
    self.assertEqual(
        decoder.decode('a_value,1.0,0'),
        # Column 'c' is specified as a string so the value is not casted.
        {'a': 'a_value', 'b': 1.0, 'c': ['0']})

  def test_fixed_length_missing_values(self):
    input_schema = {
        'b': tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=-1),
        'a': tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
    }
    decoder = csv_coder.CsvCoder(
        column_names=['a', 'b'], schema=input_schema, delimiter=',')
    self.assertEqual(decoder.decode('a_value,'), {'a': 'a_value', 'b': -1.0})
    self.assertEqual(decoder.decode(',1.0'), {'a': '', 'b': 1.0})
    self.assertEqual(decoder.decode(','), {'a': '', 'b': -1.0})

  def test_fixed_length_missing_values_no_default(self):
    input_schema = {
        'b': tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'a': tf.FixedLenFeature(shape=[], dtype=tf.string),
    }
    decoder = csv_coder.CsvCoder(
        column_names=['a', 'b'], schema=input_schema, delimiter=',')

    with self.assertRaisesRegexp(ValueError, 'expected a value on column "b"'):
      decoder.decode('a_value,')

  def test_sparse_feature_not_supported(self):
    input_schema = {
        'a': tf.SparseFeature('idx', 'value', tf.float32, 10),
    }
    with self.assertRaisesRegexp(ValueError,
                                 'SparseFeatures are not supported'):
      csv_coder.CsvCoder(
          column_names=['a'], schema=input_schema, delimiter=',')

  def test_var_length_missing_values(self):
    input_schema = {
        'b': tf.VarLenFeature(dtype=tf.float32),
        'a': tf.VarLenFeature(dtype=tf.string),
    }
    decoder = csv_coder.CsvCoder(
        column_names=['a', 'b'], schema=input_schema, delimiter=',')

    self.assertEqual(decoder.decode('a_value,'), {'a': ['a_value'], 'b': []})
    self.assertEqual(decoder.decode(',1.0'), {'a': [], 'b': [1.0]})
    self.assertEqual(decoder.decode(','), {'a': [], 'b': []})

  def test_decode_errors(self):
    input_schema = {
        'b': tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'a': tf.FixedLenFeature(shape=[], dtype=tf.string),
    }
    decoder = csv_coder.CsvCoder(
        column_names=['a', 'b'], schema=input_schema, delimiter=',')

    # Test non-numerical column value.
    with self.assertRaisesRegexp(ValueError,
                                 'could not convert string to float: b_value'):
      decoder.decode('a_value, b_value')

    # Test bad csv.
    with self.assertRaisesRegexp(csv_coder.DecodeError,
                                 'string or Unicode object, int found'):
      decoder.decode(123)

    # Test extra column.
    with self.assertRaisesRegexp(csv_coder.DecodeError,
                                 'Columns do not match specified csv headers'):
      decoder.decode('1,2,')

    # Test missing column.
    with self.assertRaisesRegexp(csv_coder.DecodeError,
                                 'Columns do not match specified csv headers'):
      decoder.decode('a_value')

    # Test empty row.
    with self.assertRaisesRegexp(csv_coder.DecodeError,
                                 'Columns do not match specified csv headers'):
      decoder.decode('')


if __name__ == '__main__':
  unittest.main()
