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

import copy
import os
import pickle
import sys

from absl import flags

# Note that this needs to happen before any non-python imports, so we do it
# pretty early on.
if any(arg == '--proto_implementation_type=python' for arg in sys.argv):
  os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
elif any(arg == '--proto_implementation_type=cpp' for arg in sys.argv):
  os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'cpp'
elif any(arg.startswith('--proto_implementation_type') for arg in sys.argv):
  raise ValueError('Unexpected value for --proto_implementation_type')

# pylint: disable=g-import-not-at-top
import numpy as np
import tensorflow as tf
from tensorflow_transform import common_types
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform import test_case
from tensorflow_transform.tf_metadata import schema_utils

from google.protobuf.internal import api_implementation
from google.protobuf import text_format
# pylint: enable=g-import-not-at-top

flags.DEFINE_string(
    'proto_implementation_type', 'cpp',
    'The implementation type of python proto to use when exercising this test')

_FEATURE_SPEC = {
    'scalar_feature_1':
        tf.io.FixedLenFeature([], tf.int64),
    'scalar_feature_2':
        tf.io.FixedLenFeature([], tf.int64),
    'scalar_feature_3':
        tf.io.FixedLenFeature([], tf.float32),
    'varlen_feature_1':
        tf.io.VarLenFeature(tf.float32),
    'varlen_feature_2':
        tf.io.VarLenFeature(tf.string),
    '1d_vector_feature':
        tf.io.FixedLenFeature([1], tf.string),
    '2d_vector_feature':
        tf.io.FixedLenFeature([2, 2], tf.float32),
    'sparse_feature':
        tf.io.SparseFeature('sparse_idx', 'sparse_val', tf.float32, 10),
    '2d_sparse_feature':
        tf.io.SparseFeature(['2d_sparse_idx0', '2d_sparse_idx1'],
                            '2d_sparse_val', tf.float32, [2, 10]),
}

_ENCODE_CASES = {
    'unicode':
        dict(
            testcase_name='unicode',
            feature_spec={
                'unicode_feature': tf.io.FixedLenFeature([], tf.string)
            },
            ascii_proto="""\
features {
  feature { key: "unicode_feature"
            value { bytes_list { value: [ "Hello κόσμε" ] } } }
}""",
            instance={'unicode_feature': u'Hello κόσμε'}),
    'scalar_string_to_varlen':
        dict(
            testcase_name='scalar_string_to_varlen',
            feature_spec={'varlen_string': tf.io.VarLenFeature(tf.string)},
            ascii_proto="""\
features {
  feature { key: "varlen_string" value { bytes_list { value: [ "foo" ] } } }
}""",
            instance={'varlen_string': 'foo'}),
    'scalar_int_to_varlen':
        dict(
            testcase_name='scalar_int_to_varlen',
            feature_spec={'varlen_int': tf.io.VarLenFeature(tf.int64)},
            ascii_proto="""\
features {
  feature { key: "varlen_int" value { int64_list { value: [ 123 ] } } }
}""",
            instance={'varlen_int': 123}),
    'multiple_columns':
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
  feature { key: "sparse_val" value { float_list { value: [ 12.0, 20.0 ] } } }
  feature { key: "sparse_idx" value { int64_list { value: [ 1, 4 ] } } }
  feature { key: "2d_sparse_idx0" value { int64_list { value: [ 1, 1 ]} } }
  feature { key: "2d_sparse_idx1" value { int64_list { value: [ 3, 7 ]} } }
  feature { key: "2d_sparse_val"
            value { float_list { value: [ 13.0, 23.0 ] } } }
}""",
            ragged_ascii_proto="""
  feature { key: "ragged_val"
            value { float_list { value: [ 7.0, 13.0, 21.0 ] } } }
  feature { key: "ragged_row_lengths1"
            value { int64_list { value: [ 1, 2 ] } } }
  feature { key: "2d_ragged_val"
            value { bytes_list { value: [ "aa a", "abc", "hi" ] } } }
  feature { key: "2d_ragged_row_lengths1"
            value { int64_list { value: [ 0, 3 ] } } }
  feature { key: "2d_ragged_row_lengths2"
            value { int64_list { value: [ 1, 0, 2 ] } } }
  feature { key: "ragged_uniform_val"
            value { int64_list { value: [ 1, -1, 2, 1, -1, 2] } } }
  feature { key: "2d_ragged_uniform_val"
            value { int64_list { value: [ 1, -1, 2, 1, -1, 2] } } }
  feature { key: "2d_ragged_uniform_row_lengths1"
            value { int64_list { value: [ 1, 0, 2 ] } } }
}
  """,
            instance={
                'scalar_feature_1': 12,
                'scalar_feature_2': 12,
                'scalar_feature_3': 1.0,
                'varlen_feature_1': [89.0],
                '1d_vector_feature': [b'this is a ,text'],
                '2d_vector_feature': [[1.0, 2.0], [3.0, 4.0]],
                'varlen_feature_2': [b'female'],
                'sparse_idx': [1, 4],
                'sparse_val': [12.0, 20.0],
                '2d_sparse_idx0': [1, 1],
                '2d_sparse_idx1': [3, 7],
                '2d_sparse_val': [13.0, 23.0],
            },
            ragged_instance={
                'ragged_val': [7.0, 13.0, 21.0],
                'ragged_row_lengths1': [1, 2],
                '2d_ragged_val': [b'aa a', b'abc', b'hi'],
                '2d_ragged_row_lengths1': [0, 3],
                '2d_ragged_row_lengths2': [1, 0, 2],
                'ragged_uniform_val': [1, -1, 2, 1, -1, 2],
                '2d_ragged_uniform_val': [1, -1, 2, 1, -1, 2],
                '2d_ragged_uniform_row_lengths1': [1, 0, 2],
            }),
    'multiple_columns_ndarray':
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
  feature { key: "sparse_val" value { float_list { value: [ 13.0, 21.0 ] } } }
  feature { key: "sparse_idx" value { int64_list { value: [ 2, 5 ] } } }
  feature { key: "2d_sparse_idx0" value { int64_list { value: [ 1, 1 ]} } }
  feature { key: "2d_sparse_idx1" value { int64_list { value: [ 3, 7 ]} } }
  feature { key: "2d_sparse_val"
            value { float_list { value: [ 13.0, 23.0 ] } } }
}""",
            ragged_ascii_proto="""
  feature { key: "ragged_val"
            value { float_list { value: [ 22.0, 22.0, 21.0 ] } } }
  feature { key: "ragged_row_lengths1"
            value { int64_list { value: [ 0, 2, 1 ] } } }
  feature { key: "2d_ragged_val"
            value { bytes_list { value: [ "oh", "hello ", "" ] } } }
  feature { key: "2d_ragged_row_lengths1"
            value { int64_list { value: [ 1, 2 ] } } }
  feature { key: "2d_ragged_row_lengths2"
            value { int64_list { value: [ 0, 0, 3 ] } } }
  feature { key: "ragged_uniform_val"
            value { int64_list { value: [ 12, -11, 2, 1, -1, 12] } } }
  feature { key: "2d_ragged_uniform_val"
            value { int64_list { value: [ 1, -1, 23, 1, -1, 32] } } }
  feature { key: "2d_ragged_uniform_row_lengths1"
            value { int64_list { value: [ 1, 0, 2 ] } } }
}
  """,
            instance={
                'scalar_feature_1': np.array(13),
                'scalar_feature_2': np.int32(214),
                'scalar_feature_3': np.array(2.0),
                'varlen_feature_1': np.array([]),
                '1d_vector_feature': np.array([b'this is another ,text']),
                '2d_vector_feature': np.array([[9.0, 8.0], [7.0, 6.0]]),
                'varlen_feature_2': np.array([b'male']),
                'sparse_idx': np.array([2, 5]),
                'sparse_val': np.array([13.0, 21.0]),
                '2d_sparse_idx0': np.array([1, 1]),
                '2d_sparse_idx1': np.array([3, 7]),
                '2d_sparse_val': np.array([13.0, 23.0]),
            },
            ragged_instance={
                'ragged_val': np.array([22.0, 22.0, 21.0]),
                'ragged_row_lengths1': np.array([0, 2, 1]),
                '2d_ragged_val': np.array([b'oh', b'hello ', b'']),
                '2d_ragged_row_lengths1': np.array([1, 2]),
                '2d_ragged_row_lengths2': np.array([0, 0, 3]),
                'ragged_uniform_val': np.array([12, -11, 2, 1, -1, 12]),
                '2d_ragged_uniform_val': np.array([1, -1, 23, 1, -1, 32]),
                '2d_ragged_uniform_row_lengths1': np.array([1, 0, 2]),
            }),
    'multiple_columns_with_missing':
        dict(
            testcase_name='multiple_columns_with_missing',
            feature_spec={'varlen_feature': tf.io.VarLenFeature(tf.string)},
            ascii_proto="""\
features { feature { key: "varlen_feature" value {} } }""",
            instance={'varlen_feature': None}),
    'multivariate_string_to_varlen':
        dict(
            testcase_name='multivariate_string_to_varlen',
            feature_spec={'varlen_string': tf.io.VarLenFeature(tf.string)},
            ascii_proto="""\
features {
  feature { key: "varlen_string" value { bytes_list { value: [ "foo", "bar" ] } } }
}""",
            instance={'varlen_string': [b'foo', b'bar']}),
}

_ENCODE_ERROR_CASES = [
    dict(
        testcase_name='to_few_values',
        feature_spec={
            '2d_vector_feature': tf.io.FixedLenFeature([2, 2], tf.int64),
        },
        instance={'2d_vector_feature': [1, 2, 3]},
        error_msg='got wrong number of values'),
]

# TODO(b/160294509): Move these to the initial definition once TF 1.x support is
# dropped.
if common_types.is_ragged_feature_available():
  _FEATURE_SPEC.update({
      'ragged_feature':
          tf.io.RaggedFeature(
              tf.float32,
              value_key='ragged_val',
              partitions=[
                  tf.io.RaggedFeature.RowLengths('ragged_row_lengths1')
              ]),
      '2d_ragged_feature':
          tf.io.RaggedFeature(
              tf.string,
              value_key='2d_ragged_val',
              partitions=[
                  tf.io.RaggedFeature.RowLengths('2d_ragged_row_lengths1'),
                  tf.io.RaggedFeature.RowLengths('2d_ragged_row_lengths2')
              ]),
      'ragged_uniform_feature':
          tf.io.RaggedFeature(
              tf.int64,
              value_key='ragged_uniform_val',
              partitions=[tf.io.RaggedFeature.UniformRowLength(2)]),
      '2d_ragged_uniform_feature':
          tf.io.RaggedFeature(
              tf.int64,
              value_key='2d_ragged_uniform_val',
              partitions=[
                  tf.io.RaggedFeature.RowLengths(
                      '2d_ragged_uniform_row_lengths1'),
                  tf.io.RaggedFeature.UniformRowLength(2)
              ]),
  })

  _ENCODE_ERROR_CASES.append(
      dict(
          testcase_name='unsupported_ragged_partition_sequence',
          feature_spec={
              '2d_ragged_feature':
                  tf.io.RaggedFeature(
                      tf.string,
                      value_key='2d_ragged_val',
                      partitions=[
                          tf.io.RaggedFeature.UniformRowLength(4),
                          tf.io.RaggedFeature.RowLengths(
                              '2d_ragged_row_lengths1')
                      ]),
          },
          instance={'2d_ragged_val': [b'not', b'necessary']},
          error_msg='Encountered ragged dimension after uniform',
      ))


def _maybe_extend_encode_case_with_ragged(encode_case):
  result = copy.deepcopy(encode_case)
  ragged_ascii_proto = result.pop('ragged_ascii_proto', '}')
  ragged_instance = result.pop('ragged_instance', {})
  if common_types.is_ragged_feature_available():
    result['ascii_proto'] = (
        encode_case['ascii_proto'][:-1] + ragged_ascii_proto)
    result['instance'].update(ragged_instance)
  return result


def _maybe_extend_encode_cases_with_ragged(encode_cases):
  for case in encode_cases.values():
    yield _maybe_extend_encode_case_with_ragged(case)


def _ascii_to_example(ascii_proto):
  return text_format.Merge(ascii_proto, tf.train.Example())


def _ascii_to_binary(ascii_proto):
  return _ascii_to_example(ascii_proto).SerializeToString()


def _binary_to_example(serialized_proto):
  return tf.train.Example.FromString(serialized_proto)


class ExampleProtoCoderTest(test_case.TransformTestCase):

  def setUp(self):
    super().setUp()
    # Verify that the implementation we requested via the Flag is honoured.
    assert api_implementation.Type() == flags.FLAGS.proto_implementation_type, (
        'Expected proto implementation type '
        f'"{flags.FLAGS.proto_implementation_type}", got: '
        f'"{api_implementation.Type()}"')

  def assertSerializedProtosEqual(self, a, b):
    np.testing.assert_equal(_binary_to_example(a), _binary_to_example(b))

  @test_case.named_parameters(
      *_maybe_extend_encode_cases_with_ragged(_ENCODE_CASES))
  def test_encode(self, feature_spec, ascii_proto, instance, **kwargs):
    schema = schema_utils.schema_from_feature_spec(feature_spec)
    coder = example_proto_coder.ExampleProtoCoder(schema, **kwargs)
    serialized_proto = _ascii_to_binary(ascii_proto)
    self.assertSerializedProtosEqual(coder.encode(instance), serialized_proto)

  @test_case.named_parameters(
      *_maybe_extend_encode_cases_with_ragged(_ENCODE_CASES))
  def test_encode_non_serialized(self, feature_spec, ascii_proto, instance,
                                 **kwargs):
    schema = schema_utils.schema_from_feature_spec(feature_spec)
    coder = example_proto_coder.ExampleProtoCoder(
        schema, serialized=False, **kwargs)
    proto = _ascii_to_example(ascii_proto)
    self.assertProtoEquals(coder.encode(instance), proto)

  @test_case.named_parameters(*_ENCODE_ERROR_CASES)
  def test_encode_error(self,
                        feature_spec,
                        instance,
                        error_msg,
                        error_type=ValueError,
                        **kwargs):
    schema = schema_utils.schema_from_feature_spec(feature_spec)
    with self.assertRaisesRegexp(error_type, error_msg):
      coder = example_proto_coder.ExampleProtoCoder(schema, **kwargs)
      coder.encode(instance)

  def test_example_proto_coder_picklable(self):
    encode_case = _maybe_extend_encode_case_with_ragged(
        _ENCODE_CASES['multiple_columns'])
    schema = schema_utils.schema_from_feature_spec(encode_case['feature_spec'])
    coder = example_proto_coder.ExampleProtoCoder(schema)
    ascii_proto = encode_case['ascii_proto']
    instance = encode_case['instance']
    serialized_proto = _ascii_to_binary(ascii_proto)
    for _ in range(2):
      coder = pickle.loads(pickle.dumps(coder))
      self.assertSerializedProtosEqual(coder.encode(instance), serialized_proto)

  def test_example_proto_coder_cache(self):
    """Test that the cache remains valid after reading/writing None."""
    schema = schema_utils.schema_from_feature_spec({
        'varlen': tf.io.VarLenFeature(tf.int64),
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
      self.assertSerializedProtosEqual(coder.encode(instance), serialized_proto)


if __name__ == '__main__':
  test_case.main()
