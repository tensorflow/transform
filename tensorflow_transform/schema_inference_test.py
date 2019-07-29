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

import os
# GOOGLE-INITIALIZATION

import tensorflow as tf

from tensorflow_transform import mappers
from tensorflow_transform import schema_inference
from tensorflow_transform import test_case
from tensorflow_transform.tf_metadata import schema_utils_legacy
from tensorflow_transform.tf_metadata import schema_utils

from google.protobuf import text_format
import unittest
from tensorflow_metadata.proto.v0 import schema_pb2


def _make_tensors_with_override():
  x = tf.compat.v1.placeholder(tf.int64, (None,))
  schema_inference.set_tensor_schema_override(x, tf.constant(5), tf.constant(6))
  return {'x': x}


class SchemaInferenceTest(test_case.TransformTestCase):

  # pylint: disable=g-long-lambda
  @test_case.named_parameters(
      dict(
          testcase_name='fixed_len_int',
          make_tensors_fn=lambda:
          {'x': tf.compat.v1.placeholder(tf.int64, (None,))},
          feature_spec={'x': tf.io.FixedLenFeature([], tf.int64)}),
      dict(
          testcase_name='fixed_len_string',
          make_tensors_fn=lambda:
          {'x': tf.compat.v1.placeholder(tf.string, (None,))},
          feature_spec={'x': tf.io.FixedLenFeature([], tf.string)}),
      dict(
          testcase_name='fixed_len_float',
          make_tensors_fn=lambda:
          {'x': tf.compat.v1.placeholder(tf.float32, (None,))},
          feature_spec={'x': tf.io.FixedLenFeature([], tf.float32)}),
      dict(
          testcase_name='override',
          make_tensors_fn=_make_tensors_with_override,
          feature_spec={'x': tf.io.FixedLenFeature([], tf.int64)},
          domains={'x': schema_pb2.IntDomain(is_categorical=True)}),
      dict(
          testcase_name='override_with_session',
          make_tensors_fn=_make_tensors_with_override,
          feature_spec={'x': tf.io.FixedLenFeature([], tf.int64)},
          domains={
              'x': schema_pb2.IntDomain(min=5, max=6, is_categorical=True)
          },
          create_session=True))
  # pylint: enable=g-long-lambda
  def test_infer_feature_schema(self,
                                make_tensors_fn,
                                feature_spec,
                                domains=None,
                                create_session=False):
    with tf.Graph().as_default() as graph:
      tensors = make_tensors_fn()

    if create_session:
      with tf.compat.v1.Session(graph=graph) as session:
        schema = schema_inference.infer_feature_schema(tensors, graph, session)
    else:
      schema = schema_inference.infer_feature_schema(tensors, graph)

    expected_schema = schema_utils.schema_from_feature_spec(
        feature_spec, domains)
    self.assertEqual(schema, expected_schema)

  def test_infer_feature_schema_bad_rank(self):
    with tf.Graph().as_default() as graph:
      tensors = {
          'a': tf.compat.v1.placeholder(tf.float32, ()),
      }
    with self.assertRaises(ValueError):
      schema_inference.infer_feature_schema(tensors, graph)

  def test_bucketization_annotation(self):
    # TODO(b/132098015): Schema annotations aren't yet supported in OSS builds.
    # pylint: disable=g-import-not-at-top
    try:
      from tensorflow_transform import annotations_pb2
    except ImportError:
      return
    # pylint: enable=g-import-not-at-top
    with tf.Graph().as_default() as graph:
      inputs = {
          'foo': tf.convert_to_tensor([0, 1, 2, 3]),
          'bar': tf.convert_to_tensor([0, 2, 0, 2]),
      }
      boundaries_foo = tf.expand_dims(tf.convert_to_tensor([.5, 1.5]), axis=0)
      boundaries_bar = tf.expand_dims(tf.convert_to_tensor([.1, .2]), axis=0)
      outputs = {}

      # tft.apply_buckets will annotate the feature in the output schema to
      # indicate the bucket boundaries that were applied.
      outputs['Bucketized_foo'] = mappers.apply_buckets(inputs['foo'],
                                                        boundaries_foo)
      outputs['Bucketized_bar'] = mappers.apply_buckets(inputs['bar'],
                                                        boundaries_bar)
      # Create a session to actually evaluate the annotations and extract the
      # the output schema with annotations applied.
      with tf.compat.v1.Session(graph=graph) as session:
        schema = schema_inference.infer_feature_schema(outputs, graph, session)
        self.assertLen(schema.feature, 2)
        for feature in schema.feature:
          self.assertLen(feature.annotation.extra_metadata, 1)
          for annotation in feature.annotation.extra_metadata:

            # Extract the annotated message and validate its contents
            message = annotations_pb2.BucketBoundaries()
            annotation.Unpack(message)
            if feature.name == 'Bucketized_foo':
              self.assertAllClose(list(message.boundaries), [.5, 1.5])
            elif feature.name == 'Bucketized_bar':
              self.assertAllClose(list(message.boundaries), [.1, .2])
            else:
              raise RuntimeError('Unexpected features in schema')

  def test_global_annotation(self):
    # TODO(b/132098015): Schema annotations aren't yet supported in OSS builds.
    # pylint: disable=g-import-not-at-top
    try:
      from tensorflow_transform import annotations_pb2
    except ImportError:
      return
    # pylint: enable=g-import-not-at-top
    with tf.Graph().as_default() as graph:
      outputs = {
          'foo': tf.convert_to_tensor([0, 1, 2, 3], dtype=tf.int64),
          'bar': tf.convert_to_tensor([0, 2, 0, 2], dtype=tf.int64),
      }

      # Annotate an arbitrary proto at the schema level (not sure what global
      # schema boundaries would mean, but hey I'm just a test).
      boundaries = tf.constant([[1.0]])
      message_type = annotations_pb2.BucketBoundaries.DESCRIPTOR.full_name
      sizes = tf.expand_dims([tf.size(boundaries)], axis=0)
      message_proto = tf.raw_ops.EncodeProto(
          sizes=sizes, values=[tf.cast(boundaries, tf.float32)],
          field_names=['boundaries'], message_type=message_type)[0]
      type_url = os.path.join('type.googleapis.com', message_type)
      schema_inference.annotate(type_url, message_proto)

      with tf.compat.v1.Session(graph=graph) as session:
        schema = schema_inference.infer_feature_schema(outputs, graph, session)
        self.assertLen(schema.annotation.extra_metadata, 1)
        for annotation in schema.annotation.extra_metadata:
          # Extract the annotated message and validate its contents
          message = annotations_pb2.BucketBoundaries()
          annotation.Unpack(message)
          self.assertAllClose(list(message.boundaries), [1])

  def test_infer_feature_schema_with_ragged_tensor(self):
    with tf.Graph().as_default() as graph:
      outputs = {
          'foo': tf.RaggedTensor.from_row_splits(
              values=tf.constant([3, 1, 4, 1, 5, 9, 2, 6], tf.int64),
              row_splits=[0, 4, 4, 7, 8, 8]),
      }
      with tf.compat.v1.Session(graph=graph) as session:
        schema = schema_inference.infer_feature_schema(outputs, graph, session)
        expected_schema_ascii = """feature {
  name: "foo"
  type: INT
  annotation {
    tag: "ragged_tensor"
  }
}
"""
        expected_schema = text_format.Parse(expected_schema_ascii,
                                            schema_pb2.Schema())
        schema_utils_legacy.set_generate_legacy_feature_spec(expected_schema,
                                                             False)
        self.assertProtoEquals(expected_schema, schema)
        with self.assertRaisesRegexp(ValueError,
                                     'Feature "foo" had tag "ragged_tensor"'):
          schema_utils.schema_as_feature_spec(schema)


if __name__ == '__main__':
  unittest.main()
