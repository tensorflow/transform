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

# GOOGLE-INITIALIZATION

import tensorflow as tf
from tensorflow_transform import schema_inference
from tensorflow_transform import test_case
from tensorflow_transform.tf_metadata import dataset_schema

import unittest
from tensorflow_metadata.proto.v0 import schema_pb2


def _make_tensors_with_override():
  x = tf.placeholder(tf.int64, (None,))
  schema_inference.set_tensor_schema_override(x, tf.constant(5), tf.constant(6))
  return {'x': x}


class SchemaInferenceTest(test_case.TransformTestCase):

  @test_case.named_parameters(
      dict(testcase_name='fixed_len_int',
           make_tensors_fn=lambda: {'x': tf.placeholder(tf.int64, (None,))},
           feature_spec={'x': tf.FixedLenFeature([], tf.int64)}),
      dict(testcase_name='fixed_len_string',
           make_tensors_fn=lambda: {'x': tf.placeholder(tf.string, (None,))},
           feature_spec={'x': tf.FixedLenFeature([], tf.string)}),
      dict(testcase_name='fixed_len_float',
           make_tensors_fn=lambda: {'x': tf.placeholder(tf.float32, (None,))},
           feature_spec={'x': tf.FixedLenFeature([], tf.float32)}),
      dict(testcase_name='override',
           make_tensors_fn=_make_tensors_with_override,
           feature_spec={'x': tf.FixedLenFeature([], tf.int64)},
           domains={'x': schema_pb2.IntDomain(is_categorical=True)}),
      dict(testcase_name='override_with_session',
           make_tensors_fn=_make_tensors_with_override,
           feature_spec={'x': tf.FixedLenFeature([], tf.int64)},
           domains={'x': schema_pb2.IntDomain(
               min=5, max=6, is_categorical=True)},
           create_session=True)
  )
  def test_infer_feature_schema(self, make_tensors_fn, feature_spec,
                                domains=None, create_session=False):
    with tf.Graph().as_default() as graph:
      tensors = make_tensors_fn()

    if create_session:
      with tf.Session(graph=graph) as session:
        schema = schema_inference.infer_feature_schema(tensors, graph, session)
    else:
      schema = schema_inference.infer_feature_schema(tensors, graph)

    expected_schema = dataset_schema.from_feature_spec(feature_spec, domains)
    self.assertEqual(schema, expected_schema)

  def test_infer_feature_schema_bad_rank(self):
    with tf.Graph().as_default() as graph:
      tensors = {
          'a': tf.placeholder(tf.float32, ()),
      }
    with self.assertRaises(ValueError):
      schema_inference.infer_feature_schema(tensors, graph)


if __name__ == '__main__':
  unittest.main()
