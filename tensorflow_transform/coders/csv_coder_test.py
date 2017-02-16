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
"""Tensorflow-transform CsvCoder tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import numpy as np
import tensorflow as tf
from tensorflow_transform.coders import csv_coder
from tensorflow_transform.tf_metadata import dataset_schema

import unittest


class TestCSVCoder(unittest.TestCase):


  _COLUMNS = ['numeric1', 'text1', 'category1', 'idx', 'numeric2', 'value']
  # The following input schema has no default values, so any invocations to
  # decode with missing values should raise an error. CsvCoderDecodeTest adds
  # good coverage for missing value handling.
  _INPUT_SCHEMA = dataset_schema.from_feature_spec({
      'numeric1': tf.FixedLenFeature(shape=[], dtype=tf.int32),
      'numeric2': tf.VarLenFeature(dtype=tf.float32),
      'text1': tf.FixedLenFeature(shape=[], dtype=tf.string),
      'category1': tf.VarLenFeature(dtype=tf.string),
      'y': tf.SparseFeature('idx', 'value', tf.float32, 10),
  })

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
    data = '12,"this is a ,text",female,1,89.0,12.0'

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # Python types.
    expected_decoded = {'category1': ['female'],
                        'numeric1': 12,
                        'numeric2': [89.0],
                        'text1': 'this is a ,text',
                        'y': ([12.0], [1])}
    self._assert_encode_decode(coder, data, expected_decoded)

    # Numpy types.
    expected_decoded = {'category1': np.array(['female']),
                        'numeric1': np.array(12),
                        'numeric2': np.array([89.0]),
                        'text1': np.array(['this is a ,text']),
                        'y': (np.array([12.0]), np.array(1))}
    self._assert_encode_decode(coder, data, expected_decoded)

  def test_tsv_coder(self):
    data = '12\t"this is a \ttext"\tfemale\t1\t89.0\t12.0'

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA,
                               delimiter='\t')
    expected_decoded = {'category1': ['female'],
                        'numeric1': 12,
                        'numeric2': [89.0],
                        'text1': 'this is a \ttext',
                        'y': ([12.0], [1])}
    self._assert_encode_decode(coder, data, expected_decoded)

  def test_valency(self):
    data = '11|12,"this is a ,text",female|male,1|3,89.0|91.0,12.0|15.0'
    feature_spec = self._INPUT_SCHEMA.as_feature_spec().copy()
    feature_spec['numeric1'] = tf.FixedLenFeature(shape=[2], dtype=tf.int32)
    schema = dataset_schema.from_feature_spec(feature_spec)
    multivalent_columns = ['numeric1', 'numeric2', 'y']
    coder = csv_coder.CsvCoder(self._COLUMNS, schema,
                               delimiter=',', secondary_delimiter='|',
                               multivalent_columns=multivalent_columns)
    expected_decoded = {'category1': ['female|male'],
                        'numeric1': [11, 12],
                        'numeric2': [89.0, 91.0],
                        'text1': 'this is a ,text',
                        'y': ([12.0, 15.0], [1, 3])}
    self._assert_encode_decode(coder, data, expected_decoded)

  def test_data_types(self):
    # The numbers are strings.
    data = '"12","this is a ,text",female,"1","89.0","12.0"'

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)
    expected_decoded = {'category1': ['female'],
                        'numeric1': 12,
                        'numeric2': [89.0],
                        'text1': 'this is a ,text',
                        'y': ([12.0], [1])}
    self._assert_encode_not_equal_decode(coder, data, expected_decoded)

  def test_missing_data(self):
    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    data = '12,,female,1,89.0,12.0'
    with self.assertRaisesRegexp(ValueError,
                                 'expected a value on column "text1"'):
      coder.decode(data)

  def test_missing_numeric_data(self):
    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # The numbers are missing.
    data = ',"this is a ,text",female,1,89.0,12.0'
    with self.assertRaisesRegexp(ValueError,
                                 'expected a value on column "numeric1"'):
      coder.decode(data)

  def test_bad_row(self):
    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # The data has a more columns than expected.
    data = '12,"this is a ,text",female,1,89.0,12.0,"oh no, I\'m an error"'
    with self.assertRaisesRegexp(Exception,
                                 'Columns do not match specified csv headers'):
      coder.decode(data)

    # The data has a fewer columns than expected.
    data = '12,"this is a ,text",female"'
    with self.assertRaisesRegexp(Exception,
                                 'Columns do not match specified csv headers'):
      coder.decode(data)

  def test_column_not_found(self):
    with self.assertRaisesRegexp(
        ValueError, 'Column not found: '):
      csv_coder.CsvCoder([], self._INPUT_SCHEMA)

  def test_picklable(self):
    encoded_data = '12,"this is a ,text",female,1,89.0,12.0'

    expected_decoded = {'category1': ['female'],
                        'numeric1': 12,
                        'numeric2': [89.0],
                        'text1': 'this is a ,text',
                        'y': ([12.0], [1])}

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # Ensure we can pickle right away.
    coder = pickle.loads(pickle.dumps(coder))
    self._assert_encode_decode(coder, encoded_data, expected_decoded)

    #  And after use.
    coder = pickle.loads(pickle.dumps(coder))
    self._assert_encode_decode(coder, encoded_data, expected_decoded)


class CsvCoderDecodeTest(unittest.TestCase):

  def test_all_values_present(self):
    columns = ['a', 'b', 'c', 'd', 'e']
    input_schema = dataset_schema.from_feature_spec({
        'b': tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'a': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'c': tf.VarLenFeature(dtype=tf.string),
        'y': tf.SparseFeature('d', 'e', tf.float32, 10),
    })
    coder = csv_coder.CsvCoder(column_names=columns, schema=input_schema)
    self.assertEqual(
        coder.decode('a_value,1.0,0,1,12.0'),
        # Column 'c' is specified as a string so the value is not casted.
        {'a': 'a_value', 'b': 1.0, 'c': ['0'], 'y': ([12.0], [1])})

  def test_fixed_length_missing_values(self):
    input_schema = dataset_schema.from_feature_spec({
        'b': tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=-1),
        'a': tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
    })
    coder = csv_coder.CsvCoder(column_names=['a', 'b'], schema=input_schema)
    self.assertEqual(coder.decode('a_value,'), {'a': 'a_value', 'b': -1.0})
    self.assertEqual(coder.decode(',1.0'), {'a': '', 'b': 1.0})
    self.assertEqual(coder.decode(','), {'a': '', 'b': -1.0})

  def test_fixed_length_missing_values_no_default(self):
    input_schema = dataset_schema.from_feature_spec({
        'b': tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'a': tf.FixedLenFeature(shape=[], dtype=tf.string),
    })
    coder = csv_coder.CsvCoder(column_names=['a', 'b'], schema=input_schema)

    with self.assertRaisesRegexp(ValueError, 'expected a value on column "b"'):
      coder.decode('a_value,')

  def test_var_length_missing_values(self):
    input_schema = dataset_schema.from_feature_spec({
        'b': tf.VarLenFeature(dtype=tf.float32),
        'a': tf.VarLenFeature(dtype=tf.string),
    })
    coder = csv_coder.CsvCoder(column_names=['a', 'b'], schema=input_schema)

    self.assertEqual(coder.decode('a_value,'), {'a': ['a_value'], 'b': []})
    self.assertEqual(coder.decode(',0'), {'a': [], 'b': [0.0]})
    self.assertEqual(coder.decode(',1.0'), {'a': [], 'b': [1.0]})
    self.assertEqual(coder.decode(','), {'a': [], 'b': []})

  def test_sparse_feature_missing_values(self):
    input_schema = dataset_schema.from_feature_spec({
        'a': tf.SparseFeature('idx', 'value', tf.float32, 10),})
    coder = csv_coder.CsvCoder(
        column_names=['idx', 'value'], schema=input_schema)

    # Missing both value and index (which is allowed).
    self.assertEqual(coder.decode(','), {'a': ([], [])})

    # Missing index only (not allowed).
    with self.assertRaisesRegexp(
        ValueError, 'expected an index in column "idx"'):
      coder.decode(',12.0')

    # Missing value only (not allowed).
    with self.assertRaisesRegexp(
        ValueError, 'expected a value in column "value"'):
      coder.decode('1,')

  def test_sparse_feature_incorrect_values(self):
    input_schema = dataset_schema.from_feature_spec({
        'a': tf.SparseFeature('idx', 'value', tf.float32, 10),
    })
    coder = csv_coder.CsvCoder(
        column_names=['idx', 'value'], schema=input_schema)

    # Index negative.
    with self.assertRaisesRegexp(ValueError,
                                 'has index -1 out of range'):
      coder.decode('-1,12.0')

    # Index equal to size.
    with self.assertRaisesRegexp(ValueError,
                                 'has index 10 out of range'):
      coder.decode('10,12.0')

    # Index greater than size.
    with self.assertRaisesRegexp(ValueError,
                                 'has index 11 out of range'):
      coder.decode('11,12.0')

  def test_decode_errors(self):
    input_schema = dataset_schema.from_feature_spec({
        'b': tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'a': tf.FixedLenFeature(shape=[], dtype=tf.string),
    })
    coder = csv_coder.CsvCoder(column_names=['a', 'b'], schema=input_schema)

    # Test non-numerical column value.
    with self.assertRaisesRegexp(ValueError,
                                 'could not convert string to float: b_value'):
      coder.decode('a_value, b_value')

    # Test bad csv.
    with self.assertRaisesRegexp(csv_coder.DecodeError,
                                 'string or Unicode object, int found'):
      coder.decode(123)

    # Test extra column.
    with self.assertRaisesRegexp(csv_coder.DecodeError,
                                 'Columns do not match specified csv headers'):
      coder.decode('1,2,')

    # Test missing column.
    with self.assertRaisesRegexp(csv_coder.DecodeError,
                                 'Columns do not match specified csv headers'):
      coder.decode('a_value')

    # Test empty row.
    with self.assertRaisesRegexp(csv_coder.DecodeError,
                                 'Columns do not match specified csv headers'):
      coder.decode('')


if __name__ == '__main__':
  unittest.main()
