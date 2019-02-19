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
"""Tensorflow-transform ExampleProtoCoder tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys

# Note that this needs to happen before any non-python imports, so we do it
# pretty early on.
if any(arg == '--proto_implementation_type=python' for arg in sys.argv):
  os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
elif any(arg == '--proto_implementation_type=cpp' for arg in sys.argv):
  os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'cpp'
elif any(arg.startswith('--proto_implementation_type') for arg in sys.argv):
  raise ValueError('Unexpected value for --proto_implementation_type')

import numpy as np
import tensorflow as tf
from tensorflow_transform import test_case
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_schema

from google.protobuf.internal import api_implementation
from google.protobuf import text_format


tf.flags.DEFINE_string(
    'proto_implementation_type', 'cpp',
    'The implementation type of python proto to use when exercising this test')


_FEATURE_SPEC = {
    'scalar_feature_1': tf.FixedLenFeature([], tf.int64),
    'scalar_feature_2': tf.FixedLenFeature([], tf.int64),
    'scalar_feature_3': tf.FixedLenFeature([], tf.float32),
    'varlen_feature_1': tf.VarLenFeature(tf.float32),
    'varlen_feature_2': tf.VarLenFeature(tf.string),
    '1d_vector_feature': tf.FixedLenFeature([1], tf.string),
    '2d_vector_feature': tf.FixedLenFeature([2, 2], tf.float32),
    'sparse_feature': tf.SparseFeature('idx', 'value', tf.float32, 10),
}

_ENCODE_DECODE_CASES = [
    dict(
        testcase_name='multiple_columns',
        feature_spec=_FEATURE_SPEC,
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
            'sparse_feature': ([1, 4], [12.0, 20.0])
        }),
    dict(
        testcase_name='multiple_columns_ndarray',
        feature_spec=_FEATURE_SPEC,
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
            'sparse_feature': (np.array([2, 5]), np.array([13.0, 21.0]))
        }),
    dict(
        testcase_name='multiple_columns_with_missing',
        feature_spec={'varlen_feature': tf.VarLenFeature(tf.string)},
        ascii_proto="""\
features { feature { key: "varlen_feature" value {} } }""",
        instance={'varlen_feature': None}),
]

_ENCODE_ONLY_CASES = [
    dict(
        testcase_name='unicode',
        feature_spec={'unicode_feature': tf.FixedLenFeature([], tf.string)},
        ascii_proto="""\
features {
  feature { key: "unicode_feature" value { bytes_list { value: [ "Hello κόσμε" ] } } }
}""",
        instance={'unicode_feature': u'Hello κόσμε'}),
]

_DECODE_ONLY_CASES = [
]

_DECODE_ERROR_CASES = [
    dict(
        testcase_name='to_few_values',
        feature_spec={
            '2d_vector_feature': tf.FixedLenFeature([2, 2], tf.int64),
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

_ENCODE_ERROR_CASES = [
    dict(
        testcase_name='to_few_values',
        feature_spec={
            '2d_vector_feature': tf.FixedLenFeature([2, 2], tf.int64),
        },
        instance={'2d_vector_feature': [1, 2, 3]},
        error_msg='got wrong number of values'),
]


def _ascii_to_example(ascii_proto):
  return text_format.Merge(ascii_proto, tf.train.Example())


def _ascii_to_binary(ascii_proto):
  return _ascii_to_example(ascii_proto).SerializeToString()


def _binary_to_example(serialized_proto):
  return tf.train.Example.FromString(serialized_proto)


class ExampleProtoCoderTest(test_case.TransformTestCase):

  def setUp(self):
    # Verify that the implementation we requested via the Flag is honoured.
    assert api_implementation.Type() == tf.flags.FLAGS.proto_implementation_type

  def assertSerializedProtosEqual(self, a, b):
    np.testing.assert_equal(_binary_to_example(a), _binary_to_example(b))

  @test_case.named_parameters(*(_ENCODE_DECODE_CASES + _DECODE_ONLY_CASES))
  def test_decode(self, feature_spec, ascii_proto, instance, **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = example_proto_coder.ExampleProtoCoder(schema, **kwargs)
    serialized_proto = _ascii_to_binary(ascii_proto)
    np.testing.assert_equal(coder.decode(serialized_proto), instance)

  @test_case.named_parameters(*(_ENCODE_DECODE_CASES + _DECODE_ONLY_CASES))
  def test_decode_non_serialized(self, feature_spec, ascii_proto, instance,
                                 **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = example_proto_coder.ExampleProtoCoder(
        schema, serialized=False, **kwargs)
    proto = _ascii_to_example(ascii_proto)
    np.testing.assert_equal(coder.decode(proto), instance)

  @test_case.named_parameters(*(_ENCODE_DECODE_CASES + _ENCODE_ONLY_CASES))
  def test_encode(self, feature_spec, ascii_proto, instance, **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = example_proto_coder.ExampleProtoCoder(schema, **kwargs)
    serialized_proto = _ascii_to_binary(ascii_proto)
    self.assertSerializedProtosEqual(coder.encode(instance), serialized_proto)

  @test_case.named_parameters(*(_ENCODE_DECODE_CASES + _ENCODE_ONLY_CASES))
  def test_encode_non_serialized(self, feature_spec, ascii_proto, instance,
                                 **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = example_proto_coder.ExampleProtoCoder(
        schema, serialized=False, **kwargs)
    proto = _ascii_to_example(ascii_proto)
    np.testing.assert_equal(coder.encode(instance), proto)

  @test_case.named_parameters(*_DECODE_ERROR_CASES)
  def test_decode_error(self,
                        feature_spec,
                        ascii_proto,
                        error_msg,
                        error_type=ValueError,
                        **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = example_proto_coder.ExampleProtoCoder(schema, **kwargs)
    serialized_proto = _ascii_to_binary(ascii_proto)
    with self.assertRaisesRegexp(error_type, error_msg):
      coder.decode(serialized_proto)

  @test_case.named_parameters(*_ENCODE_ERROR_CASES)
  def test_encode_error(self,
                        feature_spec,
                        instance,
                        error_msg,
                        error_type=ValueError,
                        **kwargs):
    schema = dataset_schema.from_feature_spec(feature_spec)
    coder = example_proto_coder.ExampleProtoCoder(schema, **kwargs)
    with self.assertRaisesRegexp(error_type, error_msg):
      coder.encode(instance)

  def test_example_proto_coder_picklable(self):
    schema = dataset_schema.from_feature_spec(_FEATURE_SPEC)
    coder = example_proto_coder.ExampleProtoCoder(schema)
    ascii_proto = """
    features {
      feature { key: "scalar_feature_1" value { int64_list { value: [ 12 ] } } }
      feature { key: "varlen_feature_1"
                value { float_list { value: [ 89.0 ] } } }
      feature { key: "scalar_feature_2" value { int64_list { value: [ 12 ] } } }
      feature { key: "scalar_feature_3"
                value { float_list { value: [ 2.0 ] } } }
      feature { key: "1d_vector_feature"
                value { bytes_list { value: [ 'this is a ,text' ] } } }
      feature { key: "2d_vector_feature"
                value { float_list { value: [ 1.0, 2.0, 3.0, 4.0 ] } } }
      feature { key: "varlen_feature_2"
                value { bytes_list { value: [ 'female' ] } } }
      feature { key: "value" value { float_list { value: [ 12.0, 20.0 ] } } }
      feature { key: "idx" value { int64_list { value: [ 1, 4 ] } } }
    }
    """
    instance = {
        'scalar_feature_1': 12,
        'scalar_feature_2': 12,
        'scalar_feature_3': 2.0,
        'varlen_feature_1': [89.0],
        '1d_vector_feature': [b'this is a ,text'],
        '2d_vector_feature': [[1.0, 2.0], [3.0, 4.0]],
        'varlen_feature_2': [b'female'],
        'sparse_feature': ([1, 4], [12.0, 20.0])
    }
    serialized_proto = _ascii_to_binary(ascii_proto)
    for _ in range(2):
      coder = pickle.loads(pickle.dumps(coder))
      np.testing.assert_equal(coder.decode(serialized_proto), instance)
      self.assertSerializedProtosEqual(coder.encode(instance), serialized_proto)

  def test_example_proto_coder_cache(self):
    """Test that the cache remains valid after reading/writing None."""
    schema = dataset_schema.from_feature_spec({
        'varlen': tf.VarLenFeature(tf.int64),
    })
    coder = example_proto_coder.ExampleProtoCoder(schema)
    ascii_protos = [
        'features {feature {key: "varlen" value {int64_list {value: [5] }}}}',
        'features {feature {key: "varlen" value {}}}',
        'features {feature {key: "varlen" value {int64_list {value: [6] }}}}',
    ]
    instances = [{'varlen': [5]}, {'varlen': None}, {'varlen': [6]}]
    serialized_protos = map(_ascii_to_binary, ascii_protos)
    for instance, serialized_proto in zip(instances, serialized_protos):
      np.testing.assert_equal(coder.decode(serialized_proto), instance)
      self.assertSerializedProtosEqual(coder.encode(instance), serialized_proto)


if __name__ == '__main__':
  test_case.main()
