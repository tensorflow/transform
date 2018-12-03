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
from tensorflow_transform import test_case
from tensorflow_transform.coders import csv_coder
from tensorflow_transform.tf_metadata import dataset_schema

_COLUMNS = [
    'numeric1', 'text1', 'category1', 'idx', 'numeric2', 'value', 'numeric3'
]

_FEATURE_SPEC = {
    'numeric1': tf.FixedLenFeature([], tf.int64),
    'numeric2': tf.VarLenFeature(tf.float32),
    'numeric3': tf.FixedLenFeature([1], tf.int64),
    'text1': tf.FixedLenFeature([], tf.string),
    'category1': tf.VarLenFeature(tf.string),
    'y': tf.SparseFeature('idx', 'value', tf.float32, 10),
}

_ENCODE_DECODE_CASES = [
    dict(
        testcase_name='multiple_columns',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line='12,"this is a ,text",categorical_value,1,89.0,12.0,20',
        instance={
            'category1': [b'categorical_value'],
            'numeric1': 12,
            'numeric2': [89.0],
            'numeric3': [20],
            'text1': b'this is a ,text',
            'y': ([1], [12.0])
        }),
    dict(
        testcase_name='multiple_columns_unicode',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line=u'12,"this is a ,text",Hello κόσμε,1,89.0,12.0,20',
        instance={
            'category1': [u'Hello κόσμε'.encode('utf-8')],
            'numeric1': 12,
            'numeric2': [89.0],
            'numeric3': [20],
            'text1': b'this is a ,text',
            'y': ([1], [12.0])
        }),
    dict(
        testcase_name='multiple_columns_tab_separated',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line=(
            '12\t"this is a \ttext"\tcategorical_value\t1\t89.0\t12.0\t20'),
        instance={
            'category1': [b'categorical_value'],
            'numeric1': 12,
            'numeric2': [89.0],
            'numeric3': [20],
            'text1': b'this is a \ttext',
            'y': ([1], [12.0])
        },
        delimiter='\t'),
    dict(
        testcase_name='multiple_columns_multivalent',
        columns=[
            'numeric1', 'category1', 'idx', 'numeric2', 'value', 'numeric3'
        ],
        feature_spec={
            'numeric1': tf.FixedLenFeature([2], tf.int64),
            'numeric2': tf.VarLenFeature(tf.float32),
            'numeric3': tf.FixedLenFeature([1], tf.int64),
            'category1': tf.VarLenFeature(tf.string),
            'y': tf.SparseFeature('idx', 'value', tf.float32, 10),
        },
        csv_line=('11|12,categorical_value|other_value,1|3,89.0|91.0,'
                  '12.0|15.0,20'),
        instance={
            'category1': [b'categorical_value|other_value'],
            'numeric1': [11, 12],
            'numeric2': [89.0, 91.0],
            'numeric3': [20],
            'y': ([1, 3], [12.0, 15.0])
        },
        secondary_delimiter='|',
        multivalent_columns=['numeric1', 'numeric2', 'y']),
    dict(
        testcase_name='scalar_int',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.int64)},
        csv_line='12',
        instance={'x': 12}),
    dict(
        testcase_name='scalar_float',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.float32)},
        csv_line='12',
        instance={'x': 12}),
    dict(
        testcase_name='size_1_vector_int',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([1], tf.int64)},
        csv_line='12',
        instance={'x': [12]}),
    dict(
        testcase_name='1x1_matrix_int',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([1, 1], tf.int64)},
        csv_line='12',
        instance={'x': [[12]]}),
    dict(
        testcase_name='unquoted_text',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.string)},
        csv_line='this is unquoted text',
        instance={'x': b'this is unquoted text'}),
    dict(
        testcase_name='quoted_text',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.string)},
        csv_line='"this is a ,text"',
        instance={'x': b'this is a ,text'}),
    dict(
        testcase_name='var_len_text',
        columns=['x'],
        feature_spec={'x': tf.VarLenFeature(tf.string)},
        csv_line='a test',
        instance={'x': [b'a test']}),
    dict(
        testcase_name='sparse_float_one_value',
        columns=['idx', 'value'],
        feature_spec={'x': tf.SparseFeature('idx', 'value', tf.float32, 10)},
        csv_line='5,2.0',
        instance={'x': ([5], [2.0])}),
    dict(
        testcase_name='sparse_float_no_values',
        columns=['idx', 'value'],
        feature_spec={'x': tf.SparseFeature('idx', 'value', tf.float32, 10)},
        csv_line=',',
        instance={'x': ([], [])}),
    dict(
        testcase_name='size_2_vector_int_multivalent',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([2], tf.int64)},
        csv_line='12|14',
        instance={'x': [12, 14]},
        secondary_delimiter='|',
        multivalent_columns=['x']),
    dict(
        testcase_name='2x2_matrix_int_multivalent',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([2, 2], tf.int64)},
        csv_line='12|13|14|15',
        instance={'x': [[12, 13], [14, 15]]},
        secondary_delimiter='|',
        multivalent_columns=['x']),
]

_DECODE_ONLY_CASES = [
    dict(
        testcase_name='scalar_float_with_decimal_point',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.float32)},
        csv_line='12.0',
        instance={'x': 12}),
    dict(
        testcase_name='scalar_float_with_quoted_value',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.float32)},
        csv_line='"12.0"',
        instance={'x': 12}),
    dict(
        testcase_name='var_len_string_with_missing_value',
        columns=['x'],
        feature_spec={'x': tf.VarLenFeature(tf.string)},
        csv_line='',
        instance={'x': []}),
    dict(
        testcase_name='multiple_columns_numpy',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line='12,"this is a ,text",categorical_value,1,89.0,12.0,20',
        instance={
            'category1': np.array([b'categorical_value']),
            'numeric1': np.array(12),
            'numeric2': np.array([89.0]),
            'numeric3': np.array([20]),
            'text1': np.array([b'this is a ,text']),
            'y': (np.array(1), np.array([12.0]))
        }),
    dict(
        testcase_name='multiple_columns_unicode_numpy',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line=u'12,"this is a ,text",Hello κόσμε,1,89.0,12.0,20',
        instance={
            'category1': np.array([u'Hello κόσμε'.encode('utf-8')]),
            'numeric1': np.array(12),
            'numeric2': np.array([89.0]),
            'numeric3': np.array([20]),
            'text1': np.array([b'this is a ,text']),
            'y': (np.array(1), np.array([12.0]))
        }),
    dict(
        testcase_name='multiple_missing_var_len_features',
        columns=['a', 'b', 'c'],
        feature_spec={
            'a': tf.VarLenFeature(tf.float32),
            'b': tf.VarLenFeature(tf.int64),
            'c': tf.VarLenFeature(tf.string),
        },
        csv_line=',,',
        instance={
            'a': [],
            'b': [],
            'c': []
        }),
]

_CONSTRUCTOR_ERROR_CASES = [
    dict(
        testcase_name='size_1_vector_with_missing_value',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([2], tf.int64)},
        error_msg=r'FixedLenFeature \"x\" was not multivalent'),
    dict(
        testcase_name='missing_column',
        columns=[],
        feature_spec={'x': tf.FixedLenFeature([], tf.int64)},
        error_msg='Column not found: '),
]

_DECODE_ERROR_CASES = [
    dict(
        testcase_name='scalar_with_missing_value',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.int64)},
        csv_line='',
        error_msg=r'expected a value on column \"x\"'),
    dict(
        testcase_name='size_1_vector_with_missing_value',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([1], tf.int64)},
        csv_line='',
        error_msg=r'expected a value on column \"x\"'),
    dict(
        testcase_name='scalar_string_missing_value',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.string)},
        csv_line='',
        error_msg=r'expected a value on column \"x\"'),
    dict(
        testcase_name='sparse_feature_with_missing_value_but_present_index',
        columns=['idx', 'value'],
        feature_spec={'x': tf.SparseFeature('idx', 'value', tf.float32, 10)},
        csv_line='5,',
        error_msg=(r'SparseFeature \"x\" has indices and values of different '
                   r'lengths')),
    dict(
        testcase_name='sparse_feature_with_missing_index_but_present_value',
        columns=['idx', 'value'],
        feature_spec={'x': tf.SparseFeature('idx', 'value', tf.float32, 10)},
        csv_line=',2.0',
        error_msg=(r'SparseFeature \"x\" has indices and values of different '
                   r'lengths')),
    dict(
        testcase_name='sparse_feature_with_negative_index',
        columns=['idx', 'value'],
        feature_spec={'x': tf.SparseFeature('idx', 'value', tf.float32, 10)},
        csv_line='-1,2.0',
        error_msg=r'has index -1 out of range'),
    dict(
        testcase_name='sparse_feature_with_index_equal_to_size',
        columns=['idx', 'value'],
        feature_spec={'x': tf.SparseFeature('idx', 'value', tf.float32, 10)},
        csv_line='10,2.0',
        error_msg=r'has index 10 out of range'),
    dict(
        testcase_name='sparse_feature_with_index_greater_than_size',
        columns=['idx', 'value'],
        feature_spec={'x': tf.SparseFeature('idx', 'value', tf.float32, 10)},
        csv_line='11,2.0',
        error_msg=r'has index 11 out of range'),
    dict(
        testcase_name='scalar_float_with_non_float_value',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.float32)},
        csv_line='test',
        error_msg=r'could not convert string to float*'),
    dict(
        testcase_name='multivalent_scalar_float_too_many_values',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([], tf.float32)},
        csv_line='1|2',
        error_msg=r'FixedLenFeature \"x\" got wrong number of values',
        secondary_delimiter='|',
        multivalent_columns=['x']),
    dict(
        testcase_name='multivalent_size_1_vector_float_too_many_values',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([1], tf.float32)},
        csv_line='1|2',
        error_msg=r'FixedLenFeature \"x\" got wrong number of values',
        secondary_delimiter='|',
        multivalent_columns=['x']),
    dict(
        testcase_name='multivalent_size_2_vector_float_too_many_values',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([2], tf.float32)},
        csv_line='1',
        error_msg=r'FixedLenFeature \"x\" got wrong number of values',
        secondary_delimiter='|',
        multivalent_columns=['x']),
    dict(
        testcase_name='row_has_more_columns_than_expected',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line=('12,"this is a ,text",categorical_value,1,89.0,12.0,'
                  '"oh no, I\'m an error",14'),
        error_msg='Columns do not match specified csv headers',
        error_type=csv_coder.DecodeError),
    dict(
        testcase_name='row_has_fewer_columns_than_expected',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line='12,"this is a ,text",categorical_value"',
        error_msg='Columns do not match specified csv headers',
        error_type=csv_coder.DecodeError),
    dict(
        testcase_name='row_is_empty',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line='',
        error_msg='Columns do not match specified csv headers',
        error_type=csv_coder.DecodeError),
    dict(
        testcase_name='csv_line_not_a_string',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line=123,
        error_msg=r'.*',
        error_type=csv_coder.DecodeError),
]

_ENCODE_ERROR_CASES = [
    dict(
        testcase_name='multivalent_size_2_vector_3_values',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([2], tf.string)},
        instance={'x': [1, 2, 3]},
        error_msg=r'FixedLenFeature \"x\" got wrong number of values',
        secondary_delimiter='|',
        multivalent_columns=['x']),
    dict(
        testcase_name='multivalent_size_2_vector_1_value',
        columns=['x'],
        feature_spec={'x': tf.FixedLenFeature([2], tf.string)},
        instance={'x': [1]},
        error_msg=r'FixedLenFeature \"x\" got wrong number of values',
        secondary_delimiter='|',
        multivalent_columns=['x']),
]


class TestCSVCoder(test_case.TransformTestCase):

  @test_case.named_parameters(*(_ENCODE_DECODE_CASES + _DECODE_ONLY_CASES))
  def test_decode(self, columns, feature_spec, csv_line, instance, **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = csv_coder.CsvCoder(columns, schema, **kwargs)
    np.testing.assert_equal(coder.decode(csv_line), instance)

  @test_case.named_parameters(*_ENCODE_DECODE_CASES)
  def test_encode(self, columns, feature_spec, csv_line, instance, **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = csv_coder.CsvCoder(columns, schema, **kwargs)
    self.assertEqual(coder.encode(instance), csv_line.encode('utf-8'))

  @test_case.named_parameters(*_CONSTRUCTOR_ERROR_CASES)
  def test_constructor_error(self,
                             columns,
                             feature_spec,
                             error_msg,
                             error_type=ValueError,
                             **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    with self.assertRaisesRegexp(error_type, error_msg):
      csv_coder.CsvCoder(columns, schema, **kwargs)

  @test_case.named_parameters(*_DECODE_ERROR_CASES)
  def test_decode_error(self,
                        columns,
                        feature_spec,
                        csv_line,
                        error_msg,
                        error_type=ValueError,
                        **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = csv_coder.CsvCoder(columns, schema, **kwargs)
    with self.assertRaisesRegexp(error_type, error_msg):
      coder.decode(csv_line)

  @test_case.named_parameters(*_ENCODE_ERROR_CASES)
  def test_encode_error(self,
                        columns,
                        feature_spec,
                        instance,
                        error_msg,
                        error_type=ValueError,
                        **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = csv_coder.CsvCoder(columns, schema, **kwargs)
    with self.assertRaisesRegexp(error_type, error_msg):
      coder.encode(instance)

  def test_picklable(self):
    csv_line = '12,"this is a ,text",categorical_value,1,89.0,12.0,20'
    instance = {
        'category1': [b'categorical_value'],
        'numeric1': 12,
        'numeric2': [89.0],
        'numeric3': [20],
        'text1': b'this is a ,text',
        'y': ([1], [12.0])
    }
    schema = dataset_schema.from_feature_spec(_FEATURE_SPEC)
    coder = csv_coder.CsvCoder(_COLUMNS, schema)
    # Repeat twice to ensure the act of encoding/decoding doesn't break
    # pickling.
    for _ in range(2):
      coder = pickle.loads(pickle.dumps(coder))
      self.assertEqual(coder.decode(csv_line), instance)
      self.assertEqual(coder.encode(instance), csv_line.encode('utf-8'))


if __name__ == '__main__':
  test_case.main()
