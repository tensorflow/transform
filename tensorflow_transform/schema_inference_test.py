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
"""Tests for tensorflow_transform.internal.schema_inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow_transform import schema_inference
from tensorflow_transform.tf_metadata import dataset_schema

import unittest


class SchemaInferenceTest(unittest.TestCase):

  def testInferFeatureSchemaWithoutSession(self):
    with tf.Graph().as_default() as graph:
      tensors = {
          'a': tf.placeholder(tf.float32, (None,)),
          'b': tf.placeholder(tf.string, (1, 2, 3)),
          'c': tf.placeholder(tf.int64, None)
      }
      schema_inference.set_tensor_schema_override(
          tensors['c'], tf.constant(5), tf.constant(6))
    schema = schema_inference.infer_feature_schema(tensors, graph)
    expected_schema = dataset_schema.Schema(column_schemas={
        'a': dataset_schema.ColumnSchema(
            tf.float32, [], dataset_schema.FixedColumnRepresentation()),
        'b': dataset_schema.ColumnSchema(
            tf.string, [2, 3], dataset_schema.FixedColumnRepresentation()),
        'c': dataset_schema.ColumnSchema(
            dataset_schema.IntDomain(tf.int64, is_categorical=True),
            None, dataset_schema.FixedColumnRepresentation())
    })
    self.assertEqual(schema, expected_schema)

  def testInferFeatureSchemaBadRank(self):
    with tf.Graph().as_default() as graph:
      tensors = {
          'a': tf.placeholder(tf.float32, ()),
      }
    with self.assertRaises(ValueError):
      schema_inference.infer_feature_schema(tensors, graph)

  def testInferFeatureSchemaWithSession(self):
    with tf.Graph().as_default() as graph:
      tensors = {
          'a': tf.placeholder(tf.float32, (None,)),
          'b': tf.placeholder(tf.string, (1, 2, 3)),
          'c': tf.placeholder(tf.int64, None)
      }
      schema_inference.set_tensor_schema_override(
          tensors['c'], tf.constant(5), tf.constant(6))
      with tf.Session(graph=graph) as session:
        schema = schema_inference.infer_feature_schema(tensors, graph, session)

    expected_schema = dataset_schema.Schema(column_schemas={
        'a': dataset_schema.ColumnSchema(
            tf.float32, [], dataset_schema.FixedColumnRepresentation()),
        'b': dataset_schema.ColumnSchema(
            tf.string, [2, 3], dataset_schema.FixedColumnRepresentation()),
        'c': dataset_schema.ColumnSchema(
            dataset_schema.IntDomain(tf.int64, 5, 6, is_categorical=True),
            None, dataset_schema.FixedColumnRepresentation())
    })
    self.assertEqual(schema, expected_schema)


if __name__ == '__main__':
  unittest.main()
