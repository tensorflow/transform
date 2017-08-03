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
"""Tests for dataset_metadata.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import tensorflow as tf

from tensorflow_transform.tf_metadata import dataset_schema as sch
from tensorflow_transform.tf_metadata import futures
from tensorflow_transform.tf_metadata import test_common
import unittest


class DatasetSchemaTest(unittest.TestCase):

  def test_feature_spec_roundtrip(self):
    schema = sch.from_feature_spec(test_common.test_feature_spec)
    generated_feature_spec = schema.as_feature_spec()
    self.assertEqual(test_common.test_feature_spec, generated_feature_spec)

  def test_feature_spec_unsupported_dtype(self):
    schema = sch.Schema()
    schema.column_schemas['fixed_float_with_default'] = (
        sch.ColumnSchema(tf.float64, [1], sch.FixedColumnRepresentation(0.0)))

    with self.assertRaisesRegexp(ValueError,
                                 'tf.Example parser supports only types '
                                 r'\[tf.string, tf.int64, tf.float32, tf.bool\]'
                                 ', so it is invalid to generate a feature_spec'
                                 ' with type tf.float64.'):
      schema.as_feature_spec()


  def test_sequence_feature_not_supported(self):
    feature_spec = {
        # FixedLenSequenceFeatures
        'fixed_seq_bool':
            tf.FixedLenSequenceFeature(shape=[10], dtype=tf.bool),
        'fixed_seq_bool_allow_missing':
            tf.FixedLenSequenceFeature(
                shape=[5], dtype=tf.bool, allow_missing=True),
        'fixed_seq_int':
            tf.FixedLenSequenceFeature(shape=[5], dtype=tf.int64),
        'fixed_seq_float':
            tf.FixedLenSequenceFeature(shape=[5], dtype=tf.float32),
        'fixed_seq_string':
            tf.FixedLenSequenceFeature(shape=[5], dtype=tf.string),
    }

    with self.assertRaisesRegexp(ValueError,
                                 'DatasetSchema does not support '
                                 'FixedLenSequenceFeature yet.'):
      sch.from_feature_spec(feature_spec)

  def test_manually_create_schema(self):
    schema = test_common.get_manually_created_schema()
    generated_feature_spec = schema.as_feature_spec()
    self.assertEqual(test_common.test_feature_spec, generated_feature_spec)

  def test_domain_picklable(self):
    domain = sch._dtype_to_domain(tf.float32)
    domain_new = pickle.loads(pickle.dumps(domain))

    self.assertEqual(type(domain), type(domain_new))
    self.assertEqual(domain.dtype, domain_new.dtype)

  def test_infer_column_schema_from_tensor(self):
    dense = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32, shape=[2, 2])
    column_schema = sch.infer_column_schema_from_tensor(dense)
    expected_column_schema = sch.ColumnSchema(
        tf.float32, [2], sch.FixedColumnRepresentation())
    self.assertEqual(expected_column_schema, column_schema)

    varlen = tf.sparse_placeholder(tf.string)
    column_schema = sch.infer_column_schema_from_tensor(varlen)
    expected_column_schema = sch.ColumnSchema(
        tf.string, [None], sch.ListColumnRepresentation())
    self.assertEqual(expected_column_schema, column_schema)

  def test_schema_equality(self):
    schema1 = sch.Schema(column_schemas={
        'fixed_bool_with_default': sch.ColumnSchema(
            tf.bool, [1], sch.FixedColumnRepresentation(False)),
        'var_float': sch.ColumnSchema(
            tf.float32, None, sch.ListColumnRepresentation())
    })
    schema2 = sch.Schema(column_schemas={
        'fixed_bool_with_default': sch.ColumnSchema(
            tf.bool, [1], sch.FixedColumnRepresentation(False)),
        'var_float': sch.ColumnSchema(
            tf.float32, None, sch.ListColumnRepresentation())
    })
    schema3 = sch.Schema(column_schemas={
        'fixed_bool_with_default': sch.ColumnSchema(
            tf.bool, [1], sch.FixedColumnRepresentation(False)),
        'var_float': sch.ColumnSchema(
            tf.float64, None, sch.ListColumnRepresentation())
    })
    schema4 = sch.Schema(column_schemas={
        'fixed_bool_with_default': sch.ColumnSchema(
            tf.bool, [1], sch.FixedColumnRepresentation(False))
    })

    self.assertEqual(schema1, schema2)
    self.assertNotEqual(schema1, schema3)
    self.assertNotEqual(schema1, schema4)

  def test_column_schema_equality(self):
    c1 = sch.ColumnSchema(
        tf.bool, [1], sch.FixedColumnRepresentation(False))
    c2 = sch.ColumnSchema(
        tf.bool, [1], sch.FixedColumnRepresentation(False))
    c3 = sch.ColumnSchema(
        tf.bool, [1], sch.FixedColumnRepresentation())
    c4 = sch.ColumnSchema(
        tf.bool, [2], sch.FixedColumnRepresentation())

    self.assertEqual(c1, c2)
    self.assertNotEqual(c1, c3)
    self.assertNotEqual(c3, c4)

  def test_domain_equality(self):
    d1 = sch._dtype_to_domain(tf.int64)
    d2 = sch._dtype_to_domain(tf.int64)
    d3 = sch._dtype_to_domain(tf.int32)
    d4 = sch._dtype_to_domain(tf.bool)

    self.assertEqual(d1, d2)
    self.assertNotEqual(d1, d3)
    self.assertNotEqual(d3, d4)

  def test_int_domain_defaults(self):
    self.assertFalse(sch.IntDomain(tf.int64).is_categorical)
    self.assertTrue(sch.IntDomain(tf.int64, is_categorical=True).is_categorical)
    self.assertEqual(tf.int64.min, sch.IntDomain(tf.int64).min_value)
    self.assertEqual(-3,
                     sch.IntDomain(tf.int64, min_value=-3).min_value)
    self.assertEqual(tf.int64.max, sch.IntDomain(tf.int64).max_value)
    self.assertEqual(3, sch.IntDomain(tf.int64, max_value=3).max_value)

  def test_axis_equality(self):
    a1 = sch.Axis(0)
    a2 = sch.Axis(0)
    a3 = sch.Axis(None)

    self.assertEqual(a1, a2)
    self.assertNotEqual(a1, a3)

  def test_column_representation_equality(self):
    fixed1 = sch.FixedColumnRepresentation(1.1)
    fixed2 = sch.FixedColumnRepresentation(1.1)
    fixed3 = sch.FixedColumnRepresentation()

    list1 = sch.ListColumnRepresentation()
    list2 = sch.ListColumnRepresentation()

    sparse1 = sch.SparseColumnRepresentation(
        'val', [sch.SparseIndexField('idx1', False),
                sch.SparseIndexField('idx2', True)])
    sparse2 = sch.SparseColumnRepresentation(
        'val', [sch.SparseIndexField('idx1', False),
                sch.SparseIndexField('idx2', True)])
    sparse3 = sch.SparseColumnRepresentation(
        'val', [sch.SparseIndexField('idx1', False),
                sch.SparseIndexField('idx2', False)])

    self.assertEqual(fixed1, fixed2)
    self.assertNotEqual(fixed1, fixed3)
    self.assertNotEqual(fixed1, list1)
    self.assertNotEqual(fixed1, sparse1)

    self.assertEqual(list1, list2)
    self.assertNotEqual(list1, sparse1)

    self.assertEqual(sparse1, sparse2)
    self.assertNotEqual(sparse1, sparse3)

  def test_sparse_index_field_equality(self):
    f1 = sch.SparseIndexField('foo', False)
    f2 = sch.SparseIndexField('foo', False)
    f3 = sch.SparseIndexField('bar', False)

    self.assertEqual(f1, f2)
    self.assertNotEqual(f2, f3)

  def test_schema_with_futures(self):
    schema = sch.Schema()

    schema.column_schemas['fixed_bool_without_default'] = (
        sch.ColumnSchema(
            tf.bool,
            [5, futures.Future('foo_dim_1'), 7, futures.Future('foo_dim_3')],
            sch.FixedColumnRepresentation()))

    schema.column_schemas['fixed_int_with_default'] = (
        sch.ColumnSchema(tf.int64, [1], sch.FixedColumnRepresentation(
            default_value=futures.Future('bar_int_default'))))

    schema.column_schemas['fixed_categorical_int_with_range'] = (
        sch.ColumnSchema(sch.IntDomain(tf.int64,
                                       futures.Future('baz_int_min'),
                                       futures.Future('baz_int_max'),
                                       is_categorical=True),
                         [1],
                         sch.FixedColumnRepresentation(default_value=0)))

    self.assertFalse(schema.all_futures_resolved())
    schema.substitute_futures({'foo_dim_1': 6, 'foo_dim_3': 8,
                               'bar_int_default': 12,
                               'baz_int_min': 3, 'baz_int_max': 4})
    self.assertTrue(schema.all_futures_resolved())

    expected_schema = sch.Schema()

    expected_schema.column_schemas['fixed_bool_without_default'] = (
        sch.ColumnSchema(tf.bool, [5, 6, 7, 8],
                         sch.FixedColumnRepresentation()))

    expected_schema.column_schemas['fixed_int_with_default'] = (
        sch.ColumnSchema(tf.int64, [1],
                         sch.FixedColumnRepresentation(default_value=12)))

    expected_schema.column_schemas['fixed_categorical_int_with_range'] = (
        sch.ColumnSchema(
            sch.IntDomain(tf.int64, 3, 4, is_categorical=True), [1],
            sch.FixedColumnRepresentation(default_value=0)))

    self.assertEqual(expected_schema, schema)


if __name__ == '__main__':
  unittest.main()
