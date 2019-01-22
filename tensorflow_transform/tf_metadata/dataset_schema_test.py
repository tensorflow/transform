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

import tensorflow as tf

from tensorflow_transform.tf_metadata import dataset_schema as sch
from tensorflow_transform.tf_metadata import test_common
import unittest


class DatasetSchemaTest(unittest.TestCase):

  def test_feature_spec_roundtrip(self):
    schema = sch.from_feature_spec(test_common.test_feature_spec)
    generated_feature_spec = schema.as_feature_spec()
    self.assertEqual(test_common.test_feature_spec, generated_feature_spec)

  def test_feature_spec_unsupported_dtype(self):
    with self.assertRaisesRegexp(ValueError, 'invalid dtype'):
      sch.Schema({
          'fixed_float': sch.ColumnSchema(
              tf.float64, [], sch.FixedColumnRepresentation())
      })

  def test_manually_create_schema(self):
    schema = test_common.get_manually_created_schema()
    generated_feature_spec = schema.as_feature_spec()
    self.assertEqual(test_common.test_feature_spec, generated_feature_spec)

  def test_schema_equality(self):
    schema1 = sch.Schema(column_schemas={
        'fixed_int': sch.ColumnSchema(
            tf.int64, [2], sch.FixedColumnRepresentation()),
        'var_float': sch.ColumnSchema(
            tf.float32, None, sch.ListColumnRepresentation())
    })
    schema2 = sch.Schema(column_schemas={
        'fixed_int': sch.ColumnSchema(
            tf.int64, [2], sch.FixedColumnRepresentation()),
        'var_float': sch.ColumnSchema(
            tf.float32, None, sch.ListColumnRepresentation())
    })
    schema3 = sch.Schema(column_schemas={
        'fixed_int': sch.ColumnSchema(
            tf.int64, [2], sch.FixedColumnRepresentation()),
        'var_float': sch.ColumnSchema(
            tf.string, None, sch.ListColumnRepresentation())
    })
    schema4 = sch.Schema(column_schemas={
        'fixed_int': sch.ColumnSchema(
            tf.int64, [2], sch.FixedColumnRepresentation())
    })

    self.assertEqual(schema1, schema2)
    self.assertNotEqual(schema1, schema3)
    self.assertNotEqual(schema1, schema4)


if __name__ == '__main__':
  unittest.main()
