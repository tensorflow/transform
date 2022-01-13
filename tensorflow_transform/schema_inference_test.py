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

import functools
import os

import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import graph_context
from tensorflow_transform import mappers
from tensorflow_transform import schema_inference
from tensorflow_transform import tf2_utils
from tensorflow_transform import test_case
from tensorflow_transform.tf_metadata import schema_utils_legacy
from tensorflow_transform.tf_metadata import schema_utils

from google.protobuf import text_format
import unittest
from tensorflow_metadata.proto.v0 import schema_pb2


if common.IS_ANNOTATIONS_PB_AVAILABLE:
  from tensorflow_transform import annotations_pb2  # pylint: disable=g-import-not-at-top


def _make_tensors(inputs):
  return {'x': tf.identity(inputs['x'])}


def _make_tensors_with_override(inputs):
  x = tf.identity(inputs['x'])
  schema_inference.set_tensor_schema_override(x, tf.constant(5), tf.constant(6))
  return {'x': x}


def _make_tensors_with_depth(inputs, depth=None):
  if depth is None:
    depth = tf.raw_ops.Placeholder(dtype=tf.int32, shape=[])
  else:
    depth = tf.constant(depth, dtype=tf.int32)
  return {'x': tf.one_hot(inputs['x'], depth=depth, dtype=inputs['x'].dtype)}


class SchemaInferenceTest(test_case.TransformTestCase):

  def _get_schema(self,
                  preprocessing_fn,
                  use_compat_v1,
                  inputs=None,
                  input_signature=None,
                  create_session=False):
    if inputs is None:
      inputs = {}
    if input_signature is None:
      input_signature = {}
    if use_compat_v1:
      with tf.compat.v1.Graph().as_default() as graph:
        # Convert eager tensors to graph tensors.
        inputs_copy = {
            k: tf.constant(v, input_signature[k].dtype)
            for k, v in inputs.items()
        }
        tensors = preprocessing_fn(inputs_copy)
        if create_session:
          # Create a session to actually evaluate the annotations and extract
          # the output schema with annotations applied.
          with tf.compat.v1.Session(graph=graph) as session:
            schema = schema_inference.infer_feature_schema(
                tensors, graph, session)
        else:
          schema = schema_inference.infer_feature_schema(tensors, graph)
    else:
      tf_func = tf.function(
          preprocessing_fn,
          input_signature=[input_signature]).get_concrete_function()
      tensors = tf.nest.pack_sequence_as(
          structure=tf_func.structured_outputs,
          flat_sequence=tf_func.outputs,
          expand_composites=True)
      structured_inputs = tf2_utils.get_structured_inputs_from_func_graph(
          tf_func.graph)
      tf_graph_context = graph_context.TFGraphContext(
          module_to_export=tf.Module(),
          temp_dir=os.path.join(self.get_temp_dir(), self._testMethodName),
          evaluated_replacements={})
      concrete_metadata_fn = schema_inference.get_traced_metadata_fn(
          preprocessing_fn=preprocessing_fn,
          structured_inputs=structured_inputs,
          tf_graph_context=tf_graph_context,
          evaluate_schema_overrides=create_session)
      schema = schema_inference.infer_feature_schema_v2(
          tensors,
          concrete_metadata_fn,
          evaluate_schema_overrides=create_session)
    return schema

  # pylint: disable=g-long-lambda
  @test_case.named_parameters(*test_case.cross_named_parameters([
      dict(
          testcase_name='fixed_len_int',
          make_tensors_fn=_make_tensors,
          feature_spec={'x': tf.io.FixedLenFeature([], tf.int64)}),
      dict(
          testcase_name='fixed_len_string',
          make_tensors_fn=_make_tensors,
          feature_spec={'x': tf.io.FixedLenFeature([], tf.string)}),
      dict(
          testcase_name='fixed_len_float',
          make_tensors_fn=_make_tensors,
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
          create_session=True),
      dict(
          testcase_name='unknown_output_non_batch_dim',
          make_tensors_fn=_make_tensors_with_depth,
          feature_spec={'x': tf.io.FixedLenFeature([None], tf.int64)}),
      dict(
          testcase_name='known_output_non_batch_dim',
          make_tensors_fn=functools.partial(_make_tensors_with_depth, depth=10),
          feature_spec={'x': tf.io.FixedLenFeature([10], tf.int64)},
          create_session=True)
  ], [
      dict(testcase_name='compat_v1', use_compat_v1=True),
      dict(testcase_name='v2', use_compat_v1=False)
  ]))
  # pylint: enable=g-long-lambda
  def test_infer_feature_schema(self,
                                make_tensors_fn,
                                feature_spec,
                                use_compat_v1,
                                domains=None,
                                create_session=False):
    if not use_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')
    x_val = '0' if feature_spec['x'].dtype == tf.string else 0
    inputs = {'x': [x_val]}
    input_signature = {
        'x': tf.TensorSpec([None], dtype=feature_spec['x'].dtype)
    }
    schema = self._get_schema(
        make_tensors_fn,
        use_compat_v1,
        inputs=inputs,
        input_signature=input_signature,
        create_session=create_session)
    expected_schema = schema_utils.schema_from_feature_spec(
        feature_spec, domains)
    self.assertEqual(schema, expected_schema)

  @test_case.named_parameters(
      dict(testcase_name='compat_v1', use_compat_v1=True),
      dict(testcase_name='v2', use_compat_v1=False))
  def test_infer_feature_schema_bad_rank(self, use_compat_v1):
    if not use_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')
    inputs = {'x': 0}
    input_signature = {'x': tf.TensorSpec([], dtype=tf.float32)}
    with self.assertRaises(ValueError):
      self._get_schema(
          _make_tensors,
          use_compat_v1,
          inputs=inputs,
          input_signature=input_signature)

  @unittest.skipIf(not common.IS_ANNOTATIONS_PB_AVAILABLE,
                     'Schema annotations are not available')
  @test_case.named_parameters(
      dict(testcase_name='compat_v1', use_compat_v1=True),
      dict(testcase_name='v2', use_compat_v1=False))
  def test_vocab_annotation(self, use_compat_v1):
    if not use_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')

    def preprocessing_fn(_):
      analyzers._maybe_annotate_vocab_metadata('file1',
                                               tf.constant(100, dtype=tf.int64),
                                               tf.constant(75, dtype=tf.int64))
      analyzers._maybe_annotate_vocab_metadata('file2',
                                               tf.constant(200, dtype=tf.int64),
                                               tf.constant(175, dtype=tf.int64))
      return {
          'foo': tf.convert_to_tensor([0, 1, 2, 3], dtype=tf.int64),
      }

    schema = self._get_schema(
        preprocessing_fn, use_compat_v1, create_session=True)
    self.assertLen(schema.annotation.extra_metadata, 2)
    unfiltered_sizes = {}
    filtered_sizes = {}
    for annotation in schema.annotation.extra_metadata:
      message = annotations_pb2.VocabularyMetadata()
      annotation.Unpack(message)
      unfiltered_sizes[message.file_name] = message.unfiltered_vocabulary_size
      filtered_sizes[message.file_name] = message.filtered_vocabulary_size
    self.assertDictEqual(unfiltered_sizes, {'file1': 100, 'file2': 200})
    self.assertDictEqual(filtered_sizes, {'file1': 75, 'file2': 175})

  @unittest.skipIf(not common.IS_ANNOTATIONS_PB_AVAILABLE,
                     'Schema annotations are not available')
  @test_case.named_parameters(
      dict(testcase_name='compat_v1', use_compat_v1=True),
      dict(testcase_name='v2', use_compat_v1=False))
  def test_bucketization_annotation(self, use_compat_v1):
    if not use_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')

    def preprocessing_fn(_):
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
      return outputs

    schema = self._get_schema(
        preprocessing_fn, use_compat_v1, create_session=True)
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

  @unittest.skipIf(not common.IS_ANNOTATIONS_PB_AVAILABLE,
                     'Schema annotations are not available')
  @test_case.named_parameters(
      dict(testcase_name='compat_v1', use_compat_v1=True),
      dict(testcase_name='v2', use_compat_v1=False))
  def test_global_annotation(self, use_compat_v1):
    # pylint: enable=g-import-not-at-top
    if not use_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')

    def preprocessing_fn(_):
      # Annotate an arbitrary proto at the schema level (not sure what global
      # schema boundaries would mean, but hey I'm just a test).
      boundaries = tf.constant([[1.0]])
      message_type = annotations_pb2.BucketBoundaries.DESCRIPTOR.full_name
      sizes = tf.expand_dims([tf.size(boundaries)], axis=0)
      message_proto = tf.raw_ops.EncodeProto(
          sizes=sizes,
          values=[tf.cast(boundaries, tf.float32)],
          field_names=['boundaries'],
          message_type=message_type)[0]
      type_url = os.path.join('type.googleapis.com', message_type)
      schema_inference.annotate(type_url, message_proto)
      return {
          'foo': tf.convert_to_tensor([0, 1, 2, 3], dtype=tf.int64),
          'bar': tf.convert_to_tensor([0, 2, 0, 2], dtype=tf.int64),
      }

    schema = self._get_schema(
        preprocessing_fn, use_compat_v1, create_session=True)
    self.assertLen(schema.annotation.extra_metadata, 1)
    for annotation in schema.annotation.extra_metadata:
      # Extract the annotated message and validate its contents
      message = annotations_pb2.BucketBoundaries()
      annotation.Unpack(message)
      self.assertAllClose(list(message.boundaries), [1])

  @test_case.named_parameters(
      dict(testcase_name='compat_v1', use_compat_v1=True),
      dict(testcase_name='v2', use_compat_v1=False))
  def test_infer_feature_schema_with_ragged_tensor(self, use_compat_v1):
    if not use_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')

    def preprocessing_fn(_):
      return {
          'foo':
              tf.RaggedTensor.from_row_splits(
                  values=tf.constant([3, 1, 4, 1, 5, 9, 2, 6], tf.int64),
                  row_splits=[0, 4, 4, 7, 8, 8]),
          'bar':
              tf.RaggedTensor.from_row_splits(
                  values=tf.RaggedTensor.from_row_splits(
                      values=tf.ones([5], tf.float32), row_splits=[0, 2, 3, 5]),
                  row_splits=[0, 0, 0, 2, 2, 4]),
          'baz':
              tf.RaggedTensor.from_row_splits(
                  values=tf.ones([5, 3], tf.float32), row_splits=[0, 2, 3, 5]),
          'qux':
              tf.RaggedTensor.from_row_splits(
                  values=tf.RaggedTensor.from_row_splits(
                      values=tf.ones([5, 7], tf.float32),
                      row_splits=[0, 2, 3, 5]),
                  row_splits=[0, 0, 0, 2, 2, 4]),
      }

    schema = self._get_schema(
        preprocessing_fn, use_compat_v1, create_session=True)
    if common_types.is_ragged_feature_available():
      expected_schema_ascii = """
        feature {
          name: "bar$ragged_values"
          type: FLOAT
        }
        feature {
          name: "bar$row_lengths_1"
          type: INT
        }
        feature {
          name: "baz$ragged_values"
          type: FLOAT
        }
        feature {
          name: "foo$ragged_values"
          type: INT
        }
        feature {
          name: "qux$ragged_values"
          type: FLOAT
        }
        feature {
          name: "qux$row_lengths_1"
          type: INT
        }
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "foo"
              value {
                ragged_tensor {
                  feature_path { step: "foo$ragged_values" }
                }
              }
            }
            tensor_representation {
              key: "bar"
              value {
                ragged_tensor {
                  feature_path { step: "bar$ragged_values" }
                  partition { row_length: "bar$row_lengths_1"}
                }
              }
            }
            tensor_representation {
              key: "baz"
              value {
                ragged_tensor {
                  feature_path { step: "baz$ragged_values" }
                  partition { uniform_row_length: 3}
                }
              }
            }
            tensor_representation {
              key: "qux"
              value {
                ragged_tensor {
                  feature_path { step: "qux$ragged_values" }
                  partition { row_length: "qux$row_lengths_1"}
                  partition { uniform_row_length: 7}
                }
              }
            }
          }
        }
        """
    else:
      expected_schema_ascii = """
        feature {
          name: "bar"
          type: FLOAT
          annotation {
            tag: "ragged_tensor"
          }
        }
        feature {
          name: "baz"
          type: FLOAT
          annotation {
            tag: "ragged_tensor"
          }
        }
        feature {
          name: "foo"
          type: INT
          annotation {
            tag: "ragged_tensor"
          }
        }
        feature {
          name: "qux"
          type: FLOAT
          annotation {
            tag: "ragged_tensor"
          }
        }
        """
    expected_schema = text_format.Parse(expected_schema_ascii,
                                        schema_pb2.Schema())
    schema_utils_legacy.set_generate_legacy_feature_spec(expected_schema, False)
    self.assertProtoEquals(expected_schema, schema)
    if not common_types.is_ragged_feature_available():
      with self.assertRaisesRegexp(ValueError,
                                   'Feature "bar" had tag "ragged_tensor"'):
        schema_utils.schema_as_feature_spec(schema)


if __name__ == '__main__':
  unittest.main()
