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
"""Tests for tensorflow_transform.tf_metadata.schema_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils

from google.protobuf import text_format
import unittest
from tensorflow_metadata.proto.v0 import schema_pb2


EQUIVALENT_FEATURE_SPEC_AND_SCHEMAS = [
    # Test different dtypes
    {
        'testcase_name': 'int',
        'ascii_proto': '''feature: {name: "x" type: INT}''',
        'feature_spec': {'x': tf.VarLenFeature(tf.int64)}
    },
    {
        'testcase_name': 'string',
        'ascii_proto': '''feature: {name: "x" type: BYTES}''',
        'feature_spec': {'x': tf.VarLenFeature(tf.string)}
    },
    {
        'testcase_name': 'float',
        'ascii_proto': '''feature: {name: "x" type: FLOAT}''',
        'feature_spec': {'x': tf.VarLenFeature(tf.float32)}
    },
    # Test different shapes
    {
        'testcase_name': 'fixed_len_vector',
        'ascii_proto': '''
          feature: {
            name: "x" type: INT shape: {dim {size: 1}}
            presence: {min_fraction: 1}
          }
        ''',
        'feature_spec': {'x': tf.FixedLenFeature([1], tf.int64, None)}
    },
    {
        'testcase_name': 'fixed_len_matrix',
        'ascii_proto': '''
          feature: {
            name: "x" type: INT shape: {dim {size: 2} dim {size: 2}}
            presence: {min_fraction: 1}
          }
        ''',
        'feature_spec': {'x': tf.FixedLenFeature([2, 2], tf.int64, None)}
    },
    {
        'testcase_name': 'var_len',
        'ascii_proto': '''feature: {name: "x" type: INT}''',
        'feature_spec': {'x': tf.VarLenFeature(tf.int64)}
    },
    {
        'testcase_name': 'sparse',
        'ascii_proto': '''
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 max: 9 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
        ''',
        'feature_spec': {
            'x': tf.SparseFeature('index_key', 'value_key', tf.int64, 10, False)
        }
    },
    {
        'testcase_name': 'sparse_sorted',
        'ascii_proto': '''
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 max: 9 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            is_sorted: true
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
        ''',
        'feature_spec': {
            'x': tf.SparseFeature('index_key', 'value_key', tf.int64, 10, True)
        }
    },
    # Test domains
    {
        'testcase_name': 'int_domain',
        'ascii_proto': '''
          feature: {
            name: "x" type: INT
            int_domain {min: 0 max: 5 is_categorical: true}
          }
        ''',
        'feature_spec': {'x': tf.VarLenFeature(tf.int64)},
        'domains': {
            'x': schema_pb2.IntDomain(min=0, max=5, is_categorical=True)
        }
    },
    {
        'testcase_name': 'string_domain',
        'ascii_proto': '''
          feature: {
            name: "x" type: BYTES
            string_domain {value: "a" value: "b"}
          }
        ''',
        'feature_spec': {'x': tf.VarLenFeature(tf.string)},
        'domains': {
            'x': schema_pb2.StringDomain(value=['a', 'b'])
        }
    },
    {
        'testcase_name': 'float_domain',
        'ascii_proto': '''
          feature: {
            name: "x" type: FLOAT
            float_domain {min: 0.0 max: 0.5}
          }
        ''',
        'feature_spec': {'x': tf.VarLenFeature(tf.float32)},
        'domains': {
            'x': schema_pb2.FloatDomain(min=0.0, max=0.5)
        }
    },
]

NON_ROUNDTRIP_SCHEMAS = [
    {
        'testcase_name': 'deprecated_feature',
        'ascii_proto': '''
          feature: {name: "x" type: INT lifecycle_stage: DEPRECATED}
        ''',
        'feature_spec': {}
    },
    {
        'testcase_name': 'schema_level_string_domain',
        'ascii_proto': '''
          feature: {name: "x" type: BYTES domain: "my_domain"}
          string_domain {name: "my_domain" value: "a" value: "b"}
        ''',
        'feature_spec': {'x': tf.VarLenFeature(tf.string)},
        'domains': {
            'x': schema_pb2.StringDomain(name='my_domain', value=['a', 'b'])
        }
    },
]

INVALID_SCHEMA_PROTOS = [
    {
        'testcase_name': 'no_type',
        'ascii_proto': '''
          feature: {name: "x"}
          ''',
        'error_msg': 'Feature "x" had invalid type TYPE_UNKNOWN'
    },
    {
        'testcase_name': 'feature_has_shape_but_not_always_present',
        'ascii_proto': '''
          feature: {name: "x" type: INT shape: {}}
        ''',
        'error_msg': r'Feature "x" had shape  set but min_fraction 0.0 != 1.  '
                     r'Use value_count not shape field when min_fraction != 1.'
    },
    {
        'testcase_name': 'sparse_feature_no_index_int_domain',
        'ascii_proto': '''
          feature {
            name: "index_key"
            type: INT
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
          ''',
        'error_msg': r'Cannot determine dense shape of sparse feature "x"'
    },
    {
        'testcase_name': 'sparse_feature_no_index_int_domain_min',
        'ascii_proto': '''
          feature {
            name: "index_key"
            type: INT
            int_domain { max: 9 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
          ''',
        'error_msg': r'Cannot determine dense shape of sparse feature "x". '
                     r'The minimum domain value of index feature "index_key"'
                     r' is not set.'
    },
    {
        'testcase_name': 'sparse_feature_non_zero_index_int_domain_min',
        'ascii_proto': '''
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 1 max: 9 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
          ''',
        'error_msg': r'Only 0-based index features are supported. Sparse '
                     r'feature "x" has index feature "index_key" whose '
                     r'minimum domain value is 1'
    },
    {
        'testcase_name': 'sparse_feature_no_index_int_domain_max',
        'ascii_proto': '''
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
          ''',
        'error_msg': r'Cannot determine dense shape of sparse feature "x". '
                     r'The maximum domain value of index feature "index_key"'
                     r' is not set.'
    },
    {
        'testcase_name': 'sparse_feature_rank_0',
        'ascii_proto': '''
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            value_feature {name: "value_key"}
          }
        ''',
        'error_msg': r'sparse_feature "x" had rank 0 but currently only'
                     r' rank 1 sparse features are supported'
    },
    {
        'testcase_name': 'sparse_feature_rank_2',
        'ascii_proto': '''
          feature {
            name: "index_key_1"
            type: INT
          }
          feature {
            name: "index_key_2"
            type: INT
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            is_sorted: true
            index_feature {name: "index_key_1"}
            index_feature {name: "index_key_2"}
            value_feature {name: "value_key"}
          }
        ''',
        'error_msg': r'sparse_feature "x" had rank 2 but currently only '
                     r'rank 1 sparse features are supported'
    },
    {
        'testcase_name': 'sparse_feature_missing_index_key',
        'ascii_proto': '''
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            is_sorted: true
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
        ''',
        'error_msg': r'sparse_feature "x" referred to index feature '
                     r'"index_key" which did not exist in the schema'
    },
    {
        'testcase_name': 'sparse_feature_missing_value_key',
        'ascii_proto': '''
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 max: 9 }
          }
          sparse_feature {
            name: "x"
            is_sorted: true
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
        ''',
        'error_msg': r'sparse_feature "x" referred to value feature '
                     r'"value_key" which did not exist in the schema'
    },
]

INVALID_FEATURE_SPECS = [
    {
        'testcase_name': 'bad_type',
        'feature_spec': {'x': tf.FixedLenFeature([], tf.bool)},
        'error_msg': 'Feature "x" has invalid dtype'
    },
    {
        'testcase_name': 'bad_index_key',
        'feature_spec': {
            'x': tf.SparseFeature(['index_key'], 'value_key', tf.int64, 10,
                                  False)
        },
        'error_msg': r'SparseFeature "x" had index_key \[\'index_key\'\], but '
                     r'size and index_key fields should be single values'
    },
    {
        'testcase_name': 'bad_size',
        'feature_spec': {
            'x': tf.SparseFeature('index_key', 'value_key', tf.int64, [10],
                                  False)
        },
        'error_msg': r'SparseFeature "x" had size \[10\], but '
                     r'size and index_key fields should be single values'
    },
    {
        'testcase_name': 'unsupported_type',
        'feature_spec': {
            'x': tf.FixedLenSequenceFeature([], tf.int64)
        },
        'error_msg': r'Spec for feature "x" was .* of type .*, expected a '
                     r'FixedLenFeature, VarLenFeature or SparseFeature',
        'error_class': TypeError
    },
]


def _parse_schema_ascii_proto(ascii_proto):
  schema = text_format.Parse(ascii_proto, schema_pb2.Schema())
  return schema


class SchemaUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(*EQUIVALENT_FEATURE_SPEC_AND_SCHEMAS)
  def test_schema_from_feature_spec(self, ascii_proto, feature_spec,
                                    domains=None):
    expected_schema_proto = _parse_schema_ascii_proto(ascii_proto)
    self.assertEqual(
        schema_utils.schema_from_feature_spec(feature_spec, domains),
        expected_schema_proto)

  @parameterized.named_parameters(
      *(EQUIVALENT_FEATURE_SPEC_AND_SCHEMAS + NON_ROUNDTRIP_SCHEMAS))
  def test_schema_as_feature_spec(self, ascii_proto, feature_spec,
                                  domains=None):
    schema_proto = _parse_schema_ascii_proto(ascii_proto)
    self.assertEqual(
        schema_utils.schema_as_feature_spec(schema_proto),
        (feature_spec, domains or {}))

  @parameterized.named_parameters(*INVALID_SCHEMA_PROTOS)
  def test_schema_as_feature_spec_fails(
      self, ascii_proto, error_msg, error_class=ValueError):
    schema_proto = _parse_schema_ascii_proto(ascii_proto)
    with self.assertRaisesRegexp(error_class, error_msg):
      schema_utils.schema_as_feature_spec(schema_proto)

  @parameterized.named_parameters(*INVALID_FEATURE_SPECS)
  def test_schema_from_feature_spec_fails(
      self, feature_spec, error_msg, domain=None, error_class=ValueError):
    with self.assertRaisesRegexp(error_class, error_msg):
      schema_utils.schema_from_feature_spec(feature_spec, domain)


if __name__ == '__main__':
  unittest.main()
