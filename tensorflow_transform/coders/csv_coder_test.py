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
import pickle

import tensorflow as tf
from tensorflow_transform.coders import csv_coder
from tensorflow_transform import test_case
from tensorflow_transform.tf_metadata import schema_utils

_COLUMNS = [
    'numeric1',
    'text1',
    'category1',
    'idx',
    'numeric2',
    'value',
    'numeric3',
    '2d_idx0',
    '2d_idx1',
    '2d_val',
]

_FEATURE_SPEC = {
    'numeric1':
        tf.io.FixedLenFeature([], tf.int64),
    'numeric2':
        tf.io.VarLenFeature(tf.float32),
    'numeric3':
        tf.io.FixedLenFeature([1], tf.int64),
    'text1':
        tf.io.FixedLenFeature([], tf.string),
    'category1':
        tf.io.VarLenFeature(tf.string),
    'y':
        tf.io.SparseFeature('idx', 'value', tf.float32, 10),
    '2dsparse':
        tf.io.SparseFeature(['2d_idx0', '2d_idx1'], '2d_val', tf.float32,
                            [2, 10]),
}

_ENCODE_CASES = [
    dict(
        testcase_name='multiple_columns',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line='12,"this is a ,text",categorical_value,1,89.0,12.0,20,1,7,17.0',
        instance={
            'category1': [b'categorical_value'],
            'numeric1': 12,
            'numeric2': [89.0],
            'numeric3': [20],
            'text1': b'this is a ,text',
            'idx': [1],
            'value': [12.0],
            '2d_idx0': [1],
            '2d_idx1': [7],
            '2d_val': [17.0],
        }),
    dict(
        testcase_name='multiple_columns_unicode',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line=u'12,"this is a ,text",Hello κόσμε,1,89.0,12.0,20,1,7,17.0',
        instance={
            'category1': [u'Hello κόσμε'.encode('utf-8')],
            'numeric1': 12,
            'numeric2': [89.0],
            'numeric3': [20],
            'text1': b'this is a ,text',
            'idx': [1],
            'value': [12.0],
            '2d_idx0': [1],
            '2d_idx1': [7],
            '2d_val': [17.0],
        }),
    dict(
        testcase_name='multiple_columns_tab_separated',
        columns=_COLUMNS,
        feature_spec=_FEATURE_SPEC,
        csv_line=(
            '12\t"this is a \ttext"\tcategorical_value\t1\t89.0\t12.0\t20\t1\t7\t17.0'
        ),
        instance={
            'category1': [b'categorical_value'],
            'numeric1': 12,
            'numeric2': [89.0],
            'numeric3': [20],
            'text1': b'this is a \ttext',
            'idx': [1],
            'value': [12.0],
            '2d_idx0': [1],
            '2d_idx1': [7],
            '2d_val': [17.0],
        },
        delimiter='\t'),
    dict(
        testcase_name='multiple_columns_multivalent',
        columns=[
            'numeric1', 'category1', 'idx', 'numeric2', 'value', 'numeric3'
        ],
        feature_spec={
            'numeric1': tf.io.FixedLenFeature([2], tf.int64),
            'numeric2': tf.io.VarLenFeature(tf.float32),
            'numeric3': tf.io.FixedLenFeature([1], tf.int64),
            'category1': tf.io.VarLenFeature(tf.string),
            'y': tf.io.SparseFeature('idx', 'value', tf.float32, 10),
        },
        csv_line=('11|12,categorical_value|other_value,1|3,89.0|91.0,'
                  '12.0|15.0,20'),
        instance={
            'category1': [b'categorical_value|other_value'],
            'numeric1': [11, 12],
            'numeric2': [89.0, 91.0],
            'numeric3': [20],
            'idx': [1, 3],
            'value': [12.0, 15.0],
        },
        secondary_delimiter='|',
        multivalent_columns=['numeric1', 'numeric2', 'y']),
    dict(
        testcase_name='scalar_int',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([], tf.int64)},
        csv_line='12',
        instance={'x': 12}),
    dict(
        testcase_name='scalar_float',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([], tf.float32)},
        csv_line='12',
        instance={'x': 12}),
    dict(
        testcase_name='size_1_vector_int',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([1], tf.int64)},
        csv_line='12',
        instance={'x': [12]}),
    dict(
        testcase_name='1x1_matrix_int',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([1, 1], tf.int64)},
        csv_line='12',
        instance={'x': [[12]]}),
    dict(
        testcase_name='unquoted_text',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([], tf.string)},
        csv_line='this is unquoted text',
        instance={'x': b'this is unquoted text'}),
    dict(
        testcase_name='quoted_text',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([], tf.string)},
        csv_line='"this is a ,text"',
        instance={'x': b'this is a ,text'}),
    dict(
        testcase_name='var_len_text',
        columns=['x'],
        feature_spec={'x': tf.io.VarLenFeature(tf.string)},
        csv_line='a test',
        instance={'x': [b'a test']}),
    dict(
        testcase_name='sparse_float_one_value',
        columns=['idx', 'value'],
        feature_spec={'x': tf.io.SparseFeature('idx', 'value', tf.float32, 10)},
        csv_line='5,2.0',
        instance={
            'idx': [5],
            'value': [2.0]
        }),
    dict(
        testcase_name='sparse_float_no_values',
        columns=['idx', 'value'],
        feature_spec={'x': tf.io.SparseFeature('idx', 'value', tf.float32, 10)},
        csv_line=',',
        instance={
            'idx': [],
            'value': []
        }),
    dict(
        testcase_name='size_2_vector_int_multivalent',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([2], tf.int64)},
        csv_line='12|14',
        instance={'x': [12, 14]},
        secondary_delimiter='|',
        multivalent_columns=['x']),
    dict(
        testcase_name='2x2_matrix_int_multivalent',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([2, 2], tf.int64)},
        csv_line='12|13|14|15',
        instance={'x': [[12, 13], [14, 15]]},
        secondary_delimiter='|',
        multivalent_columns=['x']),
]

_CONSTRUCTOR_ERROR_CASES = [
    dict(
        testcase_name='missing_column',
        columns=[],
        feature_spec={'x': tf.io.FixedLenFeature([], tf.int64)},
        error_msg='Column not found: '),
]

_ENCODE_ERROR_CASES = [
    dict(
        testcase_name='multivalent_size_2_vector_3_values',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([2], tf.string)},
        instance={'x': [1, 2, 3]},
        error_msg=r'FixedLenFeature \"x\" got wrong number of values',
        secondary_delimiter='|',
        multivalent_columns=['x']),
    dict(
        testcase_name='multivalent_size_2_vector_1_value',
        columns=['x'],
        feature_spec={'x': tf.io.FixedLenFeature([2], tf.string)},
        instance={'x': [1]},
        error_msg=r'FixedLenFeature \"x\" got wrong number of values',
        secondary_delimiter='|',
        multivalent_columns=['x']),
]


class TestCSVCoder(test_case.TransformTestCase):

  @test_case.named_parameters(*_ENCODE_CASES)
  def test_encode(self, columns, feature_spec, csv_line, instance, **kwargs):
    schema = schema_utils.schema_from_feature_spec(feature_spec)
    coder = csv_coder.CsvCoder(columns, schema, **kwargs)
    self.assertEqual(coder.encode(instance), csv_line.encode('utf-8'))

  @test_case.named_parameters(*_CONSTRUCTOR_ERROR_CASES)
  def test_constructor_error(self,
                             columns,
                             feature_spec,
                             error_msg,
                             error_type=ValueError,
                             **kwargs):
    schema = schema_utils.schema_from_feature_spec(feature_spec)
    with self.assertRaisesRegexp(error_type, error_msg):
      csv_coder.CsvCoder(columns, schema, **kwargs)

  @test_case.named_parameters(*_ENCODE_ERROR_CASES)
  def test_encode_error(self,
                        columns,
                        feature_spec,
                        instance,
                        error_msg,
                        error_type=ValueError,
                        **kwargs):
    schema = schema_utils.schema_from_feature_spec(feature_spec)
    coder = csv_coder.CsvCoder(columns, schema, **kwargs)
    with self.assertRaisesRegexp(error_type, error_msg):
      coder.encode(instance)

  def test_picklable(self):
    csv_line = '12,"this is a ,text",categorical_value,1,89.0,12.0,20,1,7,17.0'
    instance = {
        'category1': [b'categorical_value'],
        'numeric1': 12,
        'numeric2': [89.0],
        'numeric3': [20],
        'text1': b'this is a ,text',
        'idx': [1],
        'value': [12.0],
        '2d_idx0': [1],
        '2d_idx1': [7],
        '2d_val': [17.0],
    }
    schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)
    coder = csv_coder.CsvCoder(_COLUMNS, schema)
    # Repeat twice to ensure the act of encoding/decoding doesn't break
    # pickling.
    for _ in range(2):
      coder = pickle.loads(pickle.dumps(coder))
      self.assertEqual(coder.encode(instance), csv_line.encode('utf-8'))


if __name__ == '__main__':
  test_case.main()
