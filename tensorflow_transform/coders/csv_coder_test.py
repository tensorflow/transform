# coding=utf-8
#
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


  _COLUMNS = ['numeric1', 'text1', 'category1', 'idx', 'numeric2', 'value',
              'boolean1']
  # The following input schema has no default values, so any invocations to
  # decode with missing values should raise an error. CsvCoderDecodeTest adds
  # good coverage for missing value handling.
  _INPUT_SCHEMA = dataset_schema.from_feature_spec({
      'numeric1': tf.FixedLenFeature(shape=[], dtype=tf.int64),
      'numeric2': tf.VarLenFeature(dtype=tf.float32),
      'boolean1': tf.FixedLenFeature(shape=[1], dtype=tf.bool),
      'text1': tf.FixedLenFeature(shape=[], dtype=tf.string),
      'category1': tf.VarLenFeature(dtype=tf.string),
      'y': tf.SparseFeature('idx', 'value', tf.float32, 10),
  })

  _ENCODE_DECODE_CASES = [
      # FixedLenFeature scalar int.
      ('12', 12, False,
       tf.FixedLenFeature(shape=[], dtype=tf.int64)),
      # FixedLenFeature scalar float without decimal point.
      ('12', 12, False,
       tf.FixedLenFeature(shape=[], dtype=tf.float32)),
      # FixedLenFeature scalar boolean.
      ('True', True, False,
       tf.FixedLenFeature(shape=[], dtype=tf.bool)),
      # FixedLenFeature scalar boolean.
      ('False', False, False,
       tf.FixedLenFeature(shape=[], dtype=tf.bool)),
      # FixedLenFeature length 1 vector int.
      ('12', [12], False,
       tf.FixedLenFeature(shape=[1], dtype=tf.int64)),
      # FixedLenFeature size 1 matrix int.
      ('12', [[12]], False,
       tf.FixedLenFeature(shape=[1, 1], dtype=tf.int64)),
      # FixedLenFeature unquoted text.
      ('this is unquoted text', 'this is unquoted text', False,
       tf.FixedLenFeature(shape=[], dtype=tf.string)),
      # FixedLenFeature quoted text.
      ('"this is a ,text"', 'this is a ,text', False,
       tf.FixedLenFeature(shape=[], dtype=tf.string)),
      # FixedLenFeature scalar numeric with default value.
      ('4', 4, False,
       tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1)),
      # FixedLenFeature scalar text with default value set.
      ('a test', 'a test', False,
       tf.FixedLenFeature(shape=[], dtype=tf.string, default_value='d')),
      # VarLenFeature text.
      ('a test', ['a test'], False,
       tf.VarLenFeature(dtype=tf.string)),
      # SparseFeature float one value.
      ('5,2.0', ([5], [2.0]), False,
       tf.SparseFeature('idx', 'value', tf.float32, 10)),
      # SparseFeature float no values.
      (',', ([], []), False,
       tf.SparseFeature('idx', 'value', tf.float32, 10)),
      # FixedLenFeature scalar int, multivalent.
      ('12', 12, True,
       tf.FixedLenFeature(shape=[], dtype=tf.int64)),
      # FixedLenFeature length 1 vector int, multivalent.
      ('12', [12], True,
       tf.FixedLenFeature(shape=[1], dtype=tf.int64)),
      # FixedLenFeature length 2 vector int, multivalent.
      ('12|14', [12, 14], True,
       tf.FixedLenFeature(shape=[2], dtype=tf.int64)),
      # FixedLenFeature size 1 matrix int.
      ('12', [[12]], True,
       tf.FixedLenFeature(shape=[1, 1], dtype=tf.int64)),
      # FixedLenFeature size (2, 2) matrix int.
      ('12|13|14|15', [[12, 13], [14, 15]], True,
       tf.FixedLenFeature(shape=[2, 2], dtype=tf.int64)),
  ]

  _DECODE_ERROR_CASES = [
      # FixedLenFeature scalar numeric missing value.
      ('', ValueError, r'expected a value on column \'x\'', False,
       tf.FixedLenFeature(shape=[], dtype=tf.int64)),
      # FixedLenFeature length 1 vector numeric missing value.
      ('', ValueError, r'expected a value on column \'x\'', False,
       tf.FixedLenFeature(shape=[1], dtype=tf.int64)),
      # FixedLenFeature length >1 vector.
      ('1', ValueError,
       r'FixedLenFeature \'x\' was not multivalent', False,
       tf.FixedLenFeature(shape=[2], dtype=tf.int64)),
      # FixedLenFeature scalar text missing value.
      ('', ValueError, r'expected a value on column \'x\'', False,
       tf.FixedLenFeature(shape=[], dtype=tf.string)),
      # SparseFeature with missing value but present index.
      ('5,', ValueError,
       r'SparseFeature \'x\' has indices and values of different lengths',
       False,
       tf.SparseFeature('idx', 'value', tf.float32, 10)),
      # SparseFeature with missing index but present value.
      (',2.0', ValueError,
       r'SparseFeature \'x\' has indices and values of different lengths',
       False,
       tf.SparseFeature('idx', 'value', tf.float32, 10)),
      # SparseFeature with negative index.
      ('-1,2.0', ValueError, r'has index -1 out of range', False,
       tf.SparseFeature('idx', 'value', tf.float32, 10)),
      # SparseFeature with index equal to size.
      ('10,2.0', ValueError, r'has index 10 out of range', False,
       tf.SparseFeature('idx', 'value', tf.float32, 10)),
      # SparseFeature with index greater than size.
      ('11,2.0', ValueError, r'has index 11 out of range', False,
       tf.SparseFeature('idx', 'value', tf.float32, 10)),
      # FixedLenFeature with text missing value.
      ('test', ValueError, r'could not convert string to float: test', False,
       tf.FixedLenFeature(shape=[], dtype=tf.float32)),
      # FixedLenFeature scalar int, multivalent, too many values.
      ('1|2', ValueError,
       r'FixedLenFeature \'x\' got wrong number of values', True,
       tf.FixedLenFeature(shape=[], dtype=tf.float32)),
      # FixedLenFeature length 1 int, multivalent, too many values.
      ('1|2', ValueError,
       r'FixedLenFeature \'x\' got wrong number of values', True,
       tf.FixedLenFeature(shape=[1], dtype=tf.float32)),
      # FixedLenFeature length 2 int, multivalent, too few values.
      ('1', ValueError,
       r'FixedLenFeature \'x\' got wrong number of values', True,
       tf.FixedLenFeature(shape=[2], dtype=tf.float32)),
  ]

  _ENCODE_ERROR_CASES = [
      # FixedLenFeature length 2 vector, multivalent with wrong number of
      # values.
      ([1, 2, 3], ValueError,
       r'FixedLenFeature \'x\' got wrong number of values', True,
       tf.FixedLenFeature(shape=[2], dtype=tf.string))
  ]

  _DECODE_ONLY_CASES = [
      # FixedLenFeature scalar float with decimal point.
      ('12.0', 12, False,
       tf.FixedLenFeature(shape=[], dtype=tf.float32)),
      # FixedLenFeature scalar float with quoted value.
      ('"12.0"', 12, False,
       tf.FixedLenFeature(shape=[], dtype=tf.float32)),
      # FixedLenFeature scalar numeric with missing value and default value set.
      ('', -1, False,
       tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1)),
      # FixedLenFeature scalar text with missing value and default value set.
      ('', 'd', False,
       tf.FixedLenFeature(shape=[], dtype=tf.string, default_value='d')),
      # FixedLenFeature scalar numeric with missing value and default value set,
      # where default value is falsy.
      ('', 0, False,
       tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)),
      # FixedLenFeature scalar text with missing value and default value set,
      # where default value is falsy.
      ('', '', False,
       tf.FixedLenFeature(shape=[], dtype=tf.string, default_value='')),
      # VarLenFeature text with missing value.
      ('', [], False,
       tf.VarLenFeature(dtype=tf.string)),
      # FixedLenFeature scalar text with default value set.
      ('', True, False,
       tf.FixedLenFeature(shape=[], dtype=tf.bool, default_value=True)),
  ]

  longMessage = True

  def _msg_for_decode_case(self, csv_line, feature_spec):
    return 'While decoding "{csv_line}" with FeatureSpec {feature_spec}'.format(
        csv_line=csv_line, feature_spec=feature_spec)

  def _msg_for_encode_case(self, value, feature_spec):
    return 'While encoding {value} with FeatureSpec {feature_spec}'.format(
        value=value, feature_spec=feature_spec)

  def _assert_encode_decode(self, coder, data, expected_decoded):
    decoded = coder.decode(data)
    np.testing.assert_equal(decoded, expected_decoded)

    encoded = coder.encode(decoded)
    np.testing.assert_equal(encoded, data.encode('utf-8'))

    decoded_again = coder.decode(encoded)
    np.testing.assert_equal(decoded_again, expected_decoded)

  def test_csv_coder(self):
    data = '12,"this is a ,text",categorical_value,1,89.0,12.0,False'

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # Python types.
    expected_decoded = {'category1': ['categorical_value'],
                        'numeric1': 12,
                        'numeric2': [89.0],
                        'boolean1': [False],
                        'text1': 'this is a ,text',
                        'y': ([1], [12.0])}
    self._assert_encode_decode(coder, data, expected_decoded)

    # Numpy types.
    expected_decoded = {'category1': np.array(['categorical_value']),
                        'numeric1': np.array(12),
                        'numeric2': np.array([89.0]),
                        'boolean1': np.array([False]),
                        'text1': np.array(['this is a ,text']),
                        'y': (np.array(1), np.array([12.0]))}
    self._assert_encode_decode(coder, data, expected_decoded)

  def test_csv_coder_with_unicode(self):
    data = u'12,"this is a ,text",שקרכלשהו,1,89.0,12.0,False'

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # Python types.
    expected_decoded = {
        'category1': [u'שקרכלשהו'.encode('utf-8')],
        'numeric1': 12,
        'numeric2': [89.0],
        'boolean1': [False],
        'text1': 'this is a ,text',
        'y': ([1], [12.0])
    }
    self._assert_encode_decode(coder, data, expected_decoded)

    # Numpy types.
    expected_decoded = {
        'category1': np.array([u'שקרכלשהו'.encode('utf-8')]),
        'numeric1': np.array(12),
        'numeric2': np.array([89.0]),
        'boolean1': np.array([False]),
        'text1': np.array(['this is a ,text']),
        'y': (np.array(1), np.array([12.0]))
    }
    self._assert_encode_decode(coder, data, expected_decoded)

  def test_tsv_coder(self):
    data = '12\t"this is a \ttext"\tcategorical_value\t1\t89.0\t12.0\tTrue'

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA,
                               delimiter='\t')
    expected_decoded = {'category1': ['categorical_value'],
                        'numeric1': 12,
                        'numeric2': [89.0],
                        'boolean1': [True],
                        'text1': 'this is a \ttext',
                        'y': ([1], [12.0])}
    self._assert_encode_decode(coder, data, expected_decoded)

  def test_valency(self):
    data = ('11|12,"this is a ,text",categorical_value|other_value,1|3,89.0|'
            '91.0,12.0|15.0,False')
    feature_spec = self._INPUT_SCHEMA.as_feature_spec().copy()
    feature_spec['numeric1'] = tf.FixedLenFeature(shape=[2], dtype=tf.int64)
    schema = dataset_schema.from_feature_spec(feature_spec)
    multivalent_columns = ['numeric1', 'numeric2', 'y']
    coder = csv_coder.CsvCoder(self._COLUMNS, schema,
                               delimiter=',', secondary_delimiter='|',
                               multivalent_columns=multivalent_columns)
    expected_decoded = {'category1': ['categorical_value|other_value'],
                        'numeric1': [11, 12],
                        'numeric2': [89.0, 91.0],
                        'boolean1': [False],
                        'text1': 'this is a ,text',
                        'y': ([1, 3], [12.0, 15.0])}
    self._assert_encode_decode(coder, data, expected_decoded)

  # Test successful decoding with a single column.
  def testDecode(self):
    for csv_line, value, multivalent, feature_spec in (
        self._ENCODE_DECODE_CASES + self._DECODE_ONLY_CASES):
      schema = dataset_schema.from_feature_spec({'x': feature_spec})
      if isinstance(feature_spec, tf.SparseFeature):
        columns = [feature_spec.index_key, feature_spec.value_key]
      else:
        columns = 'x'

      if multivalent:
        coder = csv_coder.CsvCoder(columns, schema, secondary_delimiter='|',
                                   multivalent_columns=columns)
      else:
        coder = csv_coder.CsvCoder(columns, schema)

      np.testing.assert_equal(coder.decode(csv_line), {'x': value},
                              self._msg_for_decode_case(csv_line, feature_spec))

  # Test decode errors with a single column.
  def testDecodeErrors(self):
    for csv_line, error_type, error_msg, multivalent, feature_spec in (
        self._DECODE_ERROR_CASES):
      schema = dataset_schema.from_feature_spec({'x': feature_spec})
      if isinstance(feature_spec, tf.SparseFeature):
        columns = [feature_spec.index_key, feature_spec.value_key]
      else:
        columns = 'x'

      with self.assertRaisesRegexp(
          error_type, error_msg,
          msg=self._msg_for_decode_case(csv_line, feature_spec)):
        # We don't distinguish between errors in the coder constructor and in
        # the decode method.
        if multivalent:
          coder = csv_coder.CsvCoder(columns, schema, secondary_delimiter='|',
                                     multivalent_columns=columns)
        else:
          coder = csv_coder.CsvCoder(columns, schema)
        coder.decode(csv_line)

  # Test successful encoding with a single column.
  def testEncode(self):
    for csv_line, value, multivalent, feature_spec in self._ENCODE_DECODE_CASES:
      schema = dataset_schema.from_feature_spec({'x': feature_spec})
      if isinstance(feature_spec, tf.SparseFeature):
        columns = [feature_spec.index_key, feature_spec.value_key]
      else:
        columns = 'x'

      if multivalent:
        coder = csv_coder.CsvCoder(columns, schema, secondary_delimiter='|',
                                   multivalent_columns=columns)
      else:
        coder = csv_coder.CsvCoder(columns, schema)

      self.assertEqual(coder.encode({'x': value}), csv_line,
                       msg=self._msg_for_encode_case(value, feature_spec))

  # Test successful encoding with a single column.
  def testEncodeErrors(self):
    for value, error_type, error_msg, multivalent, feature_spec in (
        self._ENCODE_ERROR_CASES):
      schema = dataset_schema.from_feature_spec({'x': feature_spec})
      if isinstance(feature_spec, tf.SparseFeature):
        columns = [feature_spec.index_key, feature_spec.value_key]
      else:
        columns = 'x'

      with self.assertRaisesRegexp(
          error_type, error_msg,
          msg=self._msg_for_encode_case(value, feature_spec)):
        if multivalent:
          coder = csv_coder.CsvCoder(columns, schema, secondary_delimiter='|',
                                     multivalent_columns=columns)
        else:
          coder = csv_coder.CsvCoder(columns, schema)

        coder.encode({'x': value})

  def test_missing_data(self):
    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    data = '12,,categorical_value,1,89.0,12.0,True'
    with self.assertRaisesRegexp(ValueError,
                                 'expected a value on column \'text1\''):
      coder.decode(data)

  def test_bad_boolean_data(self):
    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    data = '12,text value,categorical_value,1,89.0,12.0,0'
    with self.assertRaisesRegexp(ValueError,
                                 'expected "True" or "False" as inputs'):
      coder.decode(data)

  def test_bad_row(self):
    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # The data has a more columns than expected.
    data = ('12,"this is a ,text",categorical_value,1,89.0,12.0,'
            '"oh no, I\'m an error",14')
    with self.assertRaisesRegexp(Exception,
                                 'Columns do not match specified csv headers'):
      coder.decode(data)

    # The data has a fewer columns than expected.
    data = '12,"this is a ,text",categorical_value"'
    with self.assertRaisesRegexp(Exception,
                                 'Columns do not match specified csv headers'):
      coder.decode(data)

  def test_column_not_found(self):
    with self.assertRaisesRegexp(
        ValueError, 'Column not found: '):
      csv_coder.CsvCoder([], self._INPUT_SCHEMA)

  def test_picklable(self):
    encoded_data = '12,"this is a ,text",categorical_value,1,89.0,12.0,False'

    expected_decoded = {'category1': ['categorical_value'],
                        'numeric1': 12,
                        'numeric2': [89.0],
                        'boolean1': [False],
                        'text1': 'this is a ,text',
                        'y': ([1], [12.0])}

    coder = csv_coder.CsvCoder(self._COLUMNS, self._INPUT_SCHEMA)

    # Ensure we can pickle right away.
    coder = pickle.loads(pickle.dumps(coder))
    self._assert_encode_decode(coder, encoded_data, expected_decoded)

    #  And after use.
    coder = pickle.loads(pickle.dumps(coder))
    self._assert_encode_decode(coder, encoded_data, expected_decoded)

  def test_decode_errors(self):
    input_schema = dataset_schema.from_feature_spec({
        'b': tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'a': tf.FixedLenFeature(shape=[], dtype=tf.string),
    })
    coder = csv_coder.CsvCoder(column_names=['a', 'b'], schema=input_schema)

    # Test bad csv.
    with self.assertRaisesRegexp(
        csv_coder.DecodeError,
        '\'int\' object has no attribute \'encode\': 123'):
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
