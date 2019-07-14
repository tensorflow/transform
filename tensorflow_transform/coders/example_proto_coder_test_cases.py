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
"""Test cases for example_proto_coder_test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


FEATURE_SPEC = {
    'scalar_feature_1': tf.io.FixedLenFeature([], tf.int64),
    'scalar_feature_2': tf.io.FixedLenFeature([], tf.int64),
    'scalar_feature_3': tf.io.FixedLenFeature([], tf.float32),
    'varlen_feature_1': tf.io.VarLenFeature(tf.float32),
    'varlen_feature_2': tf.io.VarLenFeature(tf.string),
    '1d_vector_feature': tf.io.FixedLenFeature([1], tf.string),
    '2d_vector_feature': tf.io.FixedLenFeature([2, 2], tf.float32),
    'sparse_feature': tf.io.SparseFeature('idx', 'value', tf.float32, 10),
}

ENCODE_DECODE_CASES = [
    dict(
        testcase_name='multiple_columns',
        feature_spec=FEATURE_SPEC,
        ascii_proto="""\
features {
  feature { key: "scalar_feature_1" value { int64_list { value: [ 12 ] } } }
  feature { key: "varlen_feature_1"
            value { float_list { value: [ 89.0 ] } } }
  feature { key: "scalar_feature_2" value { int64_list { value: [ 12 ] } } }
  feature { key: "scalar_feature_3"
            value { float_list { value: [ 1.0 ] } } }
  feature { key: "1d_vector_feature"
            value { bytes_list { value: [ 'this is a ,text' ] } } }
  feature { key: "2d_vector_feature"
            value { float_list { value: [ 1.0, 2.0, 3.0, 4.0 ] } } }
  feature { key: "varlen_feature_2"
            value { bytes_list { value: [ 'female' ] } } }
  feature { key: "value" value { float_list { value: [ 12.0, 20.0 ] } } }
feature { key: "idx" value { int64_list { value: [ 1, 4 ] } } }
}""",
        instance={
            'scalar_feature_1': 12,
            'scalar_feature_2': 12,
            'scalar_feature_3': 1.0,
            'varlen_feature_1': [89.0],
            '1d_vector_feature': [b'this is a ,text'],
            '2d_vector_feature': [[1.0, 2.0], [3.0, 4.0]],
            'varlen_feature_2': [b'female'],
            'idx': [1, 4],
            'value': [12.0, 20.0],
        }),
    dict(
        testcase_name='multiple_columns_ndarray',
        feature_spec=FEATURE_SPEC,
        ascii_proto="""\
features {
  feature { key: "scalar_feature_1" value { int64_list { value: [ 13 ] } } }
  feature { key: "varlen_feature_1" value { float_list { } } }
  feature { key: "scalar_feature_2"
            value { int64_list { value: [ 214 ] } } }
  feature { key: "scalar_feature_3"
            value { float_list { value: [ 2.0 ] } } }
  feature { key: "1d_vector_feature"
            value { bytes_list { value: [ 'this is another ,text' ] } } }
  feature { key: "2d_vector_feature"
            value { float_list { value: [ 9.0, 8.0, 7.0, 6.0 ] } } }
  feature { key: "varlen_feature_2"
            value { bytes_list { value: [ 'male' ] } } }
  feature { key: "value" value { float_list { value: [ 13.0, 21.0 ] } } }
  feature { key: "idx" value { int64_list { value: [ 2, 5 ] } } }
}""",
        instance={
            'scalar_feature_1': np.array(13),
            'scalar_feature_2': np.int32(214),
            'scalar_feature_3': np.array(2.0),
            'varlen_feature_1': np.array([]),
            '1d_vector_feature': np.array([b'this is another ,text']),
            '2d_vector_feature': np.array([[9.0, 8.0], [7.0, 6.0]]),
            'varlen_feature_2': np.array([b'male']),
            'idx': np.array([2, 5]),
            'value': np.array([13.0, 21.0]),
        }),
    dict(
        testcase_name='multiple_columns_with_missing',
        feature_spec={'varlen_feature': tf.io.VarLenFeature(tf.string)},
        ascii_proto="""\
features { feature { key: "varlen_feature" value {} } }""",
        instance={'varlen_feature': None}),
]

ENCODE_ONLY_CASES = [
    dict(
        testcase_name='unicode',
        feature_spec={'unicode_feature': tf.io.FixedLenFeature([], tf.string)},
        ascii_proto="""\
features {
  feature { key: "unicode_feature" value { bytes_list { value: [ "Hello κόσμε" ] } } }
}""",
        instance={'unicode_feature': u'Hello κόσμε'}),
]

DECODE_ONLY_CASES = []

DECODE_ERROR_CASES = [
    dict(
        testcase_name='to_few_values',
        feature_spec={
            '2d_vector_feature': tf.io.FixedLenFeature([2, 2], tf.int64),
        },
        ascii_proto="""\
features {
  feature {
    key: "2d_vector_feature"
    value { int64_list { value: [ 1, 2, 3 ] } }
  }
}""",
        error_msg='got wrong number of values'),
]

ENCODE_ERROR_CASES = [
    dict(
        testcase_name='to_few_values',
        feature_spec={
            '2d_vector_feature': tf.io.FixedLenFeature([2, 2], tf.int64),
        },
        instance={'2d_vector_feature': [1, 2, 3]},
        error_msg='got wrong number of values'),
]
