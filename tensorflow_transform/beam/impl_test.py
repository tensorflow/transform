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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import math
import os
import random
import shutil


from absl.testing import parameterized

import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np
import six
from six.moves import range

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import analyzers
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.saved import constants
from tensorflow_transform.saved import saved_model_loader
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema as sch
from tensorflow_transform.tf_metadata import metadata_io

from google.protobuf import text_format
import unittest
from tensorflow.core.example import example_pb2


def _construct_test_bucketization_parameters():
  args_without_dtype = (
      (range(1, 10), [4, 7], False, None, False, False),
      (range(1, 100), [26, 51, 76], False, None, False, False),

      # The following is similar to range(1, 100) test above, except that
      # only odd numbers are in the input; so boundaries differ (26 -> 27 and
      # 76 -> 77).
      (range(1, 100, 2), [27, 51, 77], False, None, False, False),

      # Test some inversely sorted inputs, and with different strides, and
      # boundaries/buckets.
      (range(9, 0, -1), [4, 7], False, None, False, False),
      (range(19, 0, -1), [11], False, None, False, False),
      (range(99, 0, -1), [51], False, None, False, False),
      (range(99, 0, -1), [34, 67], False, None, False, False),
      (range(99, 0, -2), [34, 68], False, None, False, False),
      (range(99, 0, -1), range(11, 100, 10), False, None, False, False),

      # These tests do a random shuffle of the inputs, which must not affect the
      # boundaries (or the computed buckets).
      (range(99, 0, -1), range(11, 100, 10), True, None, False, False),
      (range(1, 100), range(11, 100, 10), True, None, False, False),

      # The following test is with multiple batches (3 batches with default
      # batch of 1000).
      (range(1, 3000), [1503], False, None, False, False),
      (range(1, 3000), [1001, 2001], False, None, False, False),

      # Test with specific error for bucket boundaries. This is same as the test
      # above with 3 batches and a single boundary, but with a stricter error
      # tolerance (0.001) than the default error (0.01). The result is that the
      # computed boundary in the test below is closer to the middle (1501) than
      # that computed by the boundary of 1503 above.
      (range(1, 3000), [1501], False, 0.001, False, False),

      # Test with specific error for bucket boundaries, with more relaxed error
      # tolerance (0.1) than the default (0.01). Now the boundary diverges
      # further to 1519 (compared to boundary of 1501 with error 0.001, and
      # boundary of 1503 with error 0.01).
      (range(1, 3000), [1519], False, 0.1, False, False),

      # Tests for tft.apply_buckets.
      (range(1, 100), [26, 51, 76], False, 0.00001, True, False),
  )
  dtypes = (tf.int32, tf.int64, tf.float16, tf.float32, tf.float64, tf.double)
  return (x + (dtype,) for x in args_without_dtype for dtype in dtypes)


class BeamImplTest(tft_unit.TransformTestCase):

  def assertMetadataEqual(self, a, b):
    # Use extra assertEqual for schemas, since full metadata assertEqual error
    # message is not conducive to debugging.
    self.assertEqual(a.schema.column_schemas, b.schema.column_schemas)
    self.assertEqual(a, b)

  def assertAnalyzerOutputs(self,
                            input_data,
                            input_metadata,
                            analyzer_fn,
                            expected_outputs,
                            desired_batch_size=None):
    """Assert that input data and metadata is transformed as expected.

    This methods asserts transformed data and transformed metadata match
    with expected_data and expected_metadata.

    Args:
      input_data: A sequence of dicts whose values are
          either strings, lists of strings, numeric types or a pair of those.
          Must have at least one key so that we can infer the batch size.
      input_metadata: DatasetMetadata describing input_data.
      analyzer_fn: A function taking a dict of tensors and returning
          a dict of tensors.  Unlike a preprocessing_fn, this should emit
          the results of a call to an analyzer, while a preprocessing_fn must
          typically add a batch dimension and broadcast across this batch
          dimension.
      expected_outputs: A dict whose keys are the same as those of the output
          of `analyzer_fn` and whose values are convertible to an ndarrays.
      desired_batch_size: (Optional) A batch size to batch elements by. If not
          provided, a batch size will be computed automatically.
    Raises:
      AssertionError: If the expected output does not match the results of
          the analyzer_fn.
    """
    def preprocessing_fn(inputs):
      # Get tensors representing the outputs of the analyzers
      analyzer_outputs = analyzer_fn(inputs)

      # Check that keys of analyzer_outputs match expected_output.
      six.assertCountEqual(self, analyzer_outputs.keys(),
                           expected_outputs.keys())

      # Get batch size from any input tensor.
      an_input = inputs.values()[0]
      batch_size = tf.shape(an_input)[0]
      result = {}

      # Add a batch dimension and broadcast the analyzer outputs.
      result = {}
      for key, output_tensor in six.iteritems(analyzer_outputs):
        # Get the expected shape, and set it.
        output_shape = list(expected_outputs[key].shape)
        output_tensor.set_shape(output_shape)
        # Add a batch dimension
        output_tensor = tf.expand_dims(output_tensor, 0)
        # Broadcast along the batch dimension
        result[key] = tf.tile(
            output_tensor, multiples=[batch_size] + [1] * len(output_shape))

      return result

    # Create test dataset by repeating the first instance a number of times.
    num_test_instances = 3
    test_data = [input_data[0]] * num_test_instances
    expected_data = [expected_outputs] * num_test_instances
    expected_metadata = dataset_metadata.DatasetMetadata({
        key: sch.ColumnSchema(tf.as_dtype(value.dtype), value.shape,
                              sch.FixedColumnRepresentation())
        for key, value in six.iteritems(expected_outputs)})

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        test_data=test_data,
        desired_batch_size=desired_batch_size)

  def testApplySavedModelSingleInput(self):
    def save_model_with_single_input(instance, export_dir):
      builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
      with instance.test_session(graph=tf.Graph()) as sess:
        input1 = tf.placeholder(dtype=tf.int64, shape=[3], name='myinput1')
        initializer = tf.constant_initializer([1, 2, 3])
        with tf.variable_scope('Model', reuse=None, initializer=initializer):
          v1 = tf.get_variable('v1', [3], dtype=tf.int64)
        output1 = tf.add(v1, input1, name='myadd1')
        inputs = {'single_input': input1}
        outputs = {'single_output': output1}
        signature_def_map = {
            'serving_default':
                tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs, outputs)
        }
        sess.run(tf.global_variables_initializer())
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map)
        builder.save(False)

    export_dir = os.path.join(self.get_temp_dir(), 'saved_model_single')

    def preprocessing_fn(inputs):
      x = inputs['x']
      output_col = tft.apply_saved_model(
          export_dir, x, tags=[tf.saved_model.tag_constants.SERVING])
      return {'out': output_col}

    save_model_with_single_input(self, export_dir)
    input_data = [
        {'x': [1, 2, 3]},
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.int64, [3], sch.FixedColumnRepresentation()),
    })
    # [1, 2, 3] + [1, 2, 3] = [2, 4, 6]
    expected_data = [
        {'out': [2, 4, 6]}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'out': sch.ColumnSchema(tf.int64, [3], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testApplySavedModelWithHashTable(self):
    def save_model_with_hash_table(instance, export_dir):
      builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
      with instance.test_session(graph=tf.Graph()) as sess:
        key = tf.constant('test_key', shape=[1])
        value = tf.constant('test_value', shape=[1])
        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(key, value),
            '__MISSING__')

        input1 = tf.placeholder(dtype=tf.string, shape=[1], name='myinput')
        output1 = tf.reshape(table.lookup(input1), shape=[1])
        inputs = {'input': input1}
        outputs = {'output': output1}

        signature_def_map = {
            'serving_default':
                tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs, outputs)
        }

        sess.run(table.init)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map)
        builder.save(False)

    export_dir = os.path.join(self.get_temp_dir(), 'saved_model_hash_table')

    def preprocessing_fn(inputs):
      x = inputs['x']
      output_col = tft.apply_saved_model(
          export_dir, x, tags=[tf.saved_model.tag_constants.SERVING])
      return {'out': output_col}

    save_model_with_hash_table(self, export_dir)
    input_data = [
        {'x': ['test_key']}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.string, [1], sch.FixedColumnRepresentation()),
    })
    expected_data = [
        {'out': 'test_value'}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'out': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testApplySavedModelMultiInputs(self):

    def save_model_with_multi_inputs(instance, export_dir):
      builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
      with instance.test_session(graph=tf.Graph()) as sess:
        input1 = tf.placeholder(dtype=tf.int64, shape=[3], name='myinput1')
        input2 = tf.placeholder(dtype=tf.int64, shape=[3], name='myinput2')
        input3 = tf.placeholder(dtype=tf.int64, shape=[3], name='myinput3')
        initializer = tf.constant_initializer([1, 2, 3])
        with tf.variable_scope('Model', reuse=None, initializer=initializer):
          v1 = tf.get_variable('v1', [3], dtype=tf.int64)
        o1 = tf.add(v1, input1, name='myadd1')
        o2 = tf.subtract(o1, input2, name='mysubtract1')
        output1 = tf.add(o2, input3, name='myadd2')
        inputs = {'name1': input1, 'name2': input2,
                  'name3': input3}
        outputs = {'single_output': output1}
        signature_def_map = {
            'serving_default':
                tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs, outputs)
        }
        sess.run(tf.global_variables_initializer())
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map)
        builder.save(False)

    export_dir = os.path.join(self.get_temp_dir(), 'saved_model_multi')

    def preprocessing_fn(inputs):
      x = inputs['x']
      y = inputs['y']
      z = inputs['z']
      sum_column = tft.apply_saved_model(
          export_dir, {'name1': x,
                       'name3': z,
                       'name2': y},
          tags=[tf.saved_model.tag_constants.SERVING])
      return {'sum': sum_column}

    save_model_with_multi_inputs(self, export_dir)
    input_data = [
        {'x': [1, 2, 3], 'y': [2, 3, 4], 'z': [1, 1, 1]},
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.int64, [3], sch.FixedColumnRepresentation()),
        'y': sch.ColumnSchema(tf.int64, [3], sch.FixedColumnRepresentation()),
        'z': sch.ColumnSchema(tf.int64, [3], sch.FixedColumnRepresentation()),
    })
    # [1, 2, 3] + [1, 2, 3] - [2, 3, 4] + [1, 1, 1] = [1, 2, 3]
    expected_data = [
        {'sum': [1, 2, 3]}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'sum': sch.ColumnSchema(tf.int64, [3], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testApplyFunctionWithCheckpoint(self):

    def tensor_fn(input1, input2):
      initializer = tf.constant_initializer([1, 2, 3])
      with tf.variable_scope('Model', reuse=None, initializer=initializer):
        v1 = tf.get_variable('v1', [3], dtype=tf.int64)
        v2 = tf.get_variable('v2', [3], dtype=tf.int64)
        o1 = tf.add(v1, v2, name='add1')
        o2 = tf.subtract(o1, input1, name='sub1')
        o3 = tf.subtract(o2, input2, name='sub2')
        return o3

    def save_checkpoint(instance, checkpoint_path):
      with instance.test_session(graph=tf.Graph()) as sess:
        input1 = tf.placeholder(dtype=tf.int64, shape=[3], name='myinput1')
        input2 = tf.placeholder(dtype=tf.int64, shape=[3], name='myinput2')
        tensor_fn(input1, input2)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint_path)

    checkpoint_path = os.path.join(self.get_temp_dir(), 'chk')

    def preprocessing_fn(inputs):
      x = inputs['x']
      y = inputs['y']
      out_value = tft.apply_function_with_checkpoint(
          tensor_fn, [x, y], checkpoint_path)
      return {'out': out_value}

    save_checkpoint(self, checkpoint_path)
    input_data = [
        {'x': [2, 2, 2], 'y': [-1, -3, 1]},
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.int64, [3], sch.FixedColumnRepresentation()),
        'y': sch.ColumnSchema(tf.int64, [3], sch.FixedColumnRepresentation()),
    })
    # [1, 2, 3] + [1, 2, 3] - [2, 2, 2] - [-1, -3, 1] = [1, 5, 3]
    expected_data = [
        {'out': [1, 5, 3]}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'out': sch.ColumnSchema(tf.int64, [3], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testMultipleLevelsOfAnalyzers(self):
    # Test a preprocessing function similar to scale_to_0_1 except that it
    # involves multiple interleavings of analyzers and transforms.
    def preprocessing_fn(inputs):
      scaled_to_0 = inputs['x'] - tft.min(inputs['x'])
      scaled_to_0_1 = scaled_to_0 / tft.max(scaled_to_0)
      return {'x_scaled': scaled_to_0_1}

    input_data = [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [
        {'x_scaled': 0.75},
        {'x_scaled': 0.0},
        {'x_scaled': 1.0},
        {'x_scaled': 0.25}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled': sch.ColumnSchema(tf.float32, [],
                                     sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testRawFeedDictInput(self):
    # Test the ability to feed raw data into AnalyzeDataset and TransformDataset
    # by using subclasses of these transforms which create batches of size 1.
    def preprocessing_fn(inputs):
      sequence_example = inputs['sequence_example']

      # Ordinarily this would have shape (batch_size,) since 'sequence_example'
      # was defined as a FixedLenFeature with shape ().  But since we specified
      # desired_batch_size, we can assume that the shape is (1,), and reshape
      # to ().
      sequence_example = tf.reshape(sequence_example, ())

      # Parse the sequence example.
      feature_spec = {
          'x': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string,
                                          default_value=None)
      }
      _, sequences = tf.parse_single_sequence_example(
          sequence_example, sequence_features=feature_spec)

      # Create a batch based on the sequence "x".
      return {'x': sequences['x']}

    def text_sequence_example_to_binary(text_proto):
      proto = text_format.Merge(text_proto, example_pb2.SequenceExample())
      return proto.SerializeToString()

    sequence_examples = [
        """
        feature_lists: {
          feature_list: {
            key: "x"
            value: {
              feature: {bytes_list: {value: 'ab'}}
              feature: {bytes_list: {value: ''}}
              feature: {bytes_list: {value: 'c'}}
              feature: {bytes_list: {value: 'd'}}
            }
          }
        }
        """,
        """
        feature_lists: {
          feature_list: {
            key: "x"
            value: {
              feature: {bytes_list: {value: 'ef'}}
              feature: {bytes_list: {value: 'g'}}
            }
          }
        }
        """
    ]
    input_data = [
        {'sequence_example': text_sequence_example_to_binary(sequence_example)}
        for sequence_example in sequence_examples]
    input_metadata = dataset_metadata.DatasetMetadata({
        'sequence_example': sch.ColumnSchema(tf.string, [],
                                             sch.FixedColumnRepresentation())
    })
    expected_data = [
        {'x': 'ab'},
        {'x': ''},
        {'x': 'c'},
        {'x': 'd'},
        {'x': 'ef'},
        {'x': 'g'}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })

    with beam_impl.Context(temp_dir=self.get_temp_dir(), desired_batch_size=1):
      transform_fn = ((input_data, input_metadata)
                      | beam_impl.AnalyzeDataset(preprocessing_fn))
      transformed_data, transformed_metadata = (
          ((input_data, input_metadata), transform_fn)
          | beam_impl.TransformDataset())

    self.assertDataCloseOrEqual(expected_data, transformed_data)
    # Now that the pipeline has run, transformed_metadata.deferred_metadata
    # should be a list containing a single DatasetMetadata with the full
    # metadata.
    assert len(transformed_metadata.deferred_metadata) == 1
    transformed_metadata = transformed_metadata.deferred_metadata[0]
    self.assertEqual(expected_metadata.schema.column_schemas,
                     transformed_metadata.schema.column_schemas)
    self.assertEqual(expected_metadata, transformed_metadata)

  def testAnalyzerBeforeMap(self):
    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    input_data = [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [
        {'x_scaled': 0.75},
        {'x_scaled': 0.0},
        {'x_scaled': 1.0},
        {'x_scaled': 0.25}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled': sch.ColumnSchema(tf.float32, [],
                                     sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testTransformWithExcludedOutputs(self):
    def preprocessing_fn(inputs):
      return {
          'x_scaled': tft.scale_to_0_1(inputs['x']),
          'y_scaled': tft.scale_to_0_1(inputs['y'])
      }

    # Run AnalyzeAndTransform on some input data and compare with expected
    # output.
    input_data = [{'x': 5, 'y': 1}, {'x': 1, 'y': 2}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'y': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transform_fn = (
          (input_data, input_metadata) | beam_impl.AnalyzeDataset(
              preprocessing_fn))

    # Take the transform function and use TransformDataset to apply it to
    # some eval data, with missing 'y' column.
    eval_data = [{'x': 6}]
    eval_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    transformed_eval_data, transformed_eval_metadata = (
        ((eval_data, eval_metadata), transform_fn)
        | beam_impl.TransformDataset(exclude_outputs=['y_scaled']))

    expected_transformed_eval_data = [{'x_scaled': 1.25}]
    expected_transformed_eval_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled': sch.ColumnSchema(tf.float32, [],
                                     sch.FixedColumnRepresentation())
    })
    self.assertDataCloseOrEqual(transformed_eval_data,
                                expected_transformed_eval_data)
    self.assertEqual(transformed_eval_metadata.dataset_metadata,
                     expected_transformed_eval_metadata)

  def testMapSparseColumns(self):
    # Define a transform that takes a sparse column and a varlen column, and
    # returns a combination of dense, sparse, and varlen columns.
    def preprocessing_fn(inputs):
      sparse_sum = tf.sparse_reduce_sum(inputs['sparse'], axis=1)
      return {
          'fixed': sparse_sum,  # Schema should be inferred.
          'varlen': inputs['varlen'],  # Schema should be inferred.
      }

    input_data = [
        {'sparse': ([0, 1], [0., 1.]), 'varlen': [0., 1.]},
        {'sparse': ([2, 3], [2., 3.]), 'varlen': [3., 4., 5.]},
        {'sparse': ([4, 5], [4., 5.]), 'varlen': [6., 7.]}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'sparse': sch.ColumnSchema(
            tf.float32, [10], sch.SparseColumnRepresentation(
                'val', [sch.SparseIndexField('idx', False)])),
        'varlen': sch.ColumnSchema(
            tf.float32, [None], sch.ListColumnRepresentation())
    })
    expected_data = [
        {'fixed': 1.0, 'varlen': [0., 1.]},
        {'fixed': 5.0, 'varlen': [3., 4., 5.]},
        {'fixed': 9.0, 'varlen': [6., 7.]}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'fixed': sch.ColumnSchema(
            tf.float32, None, sch.FixedColumnRepresentation()),
        'varlen': sch.ColumnSchema(
            tf.float32, [None], sch.ListColumnRepresentation()),
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testSingleMap(self):
    def preprocessing_fn(inputs):
      return {'ab': tf.multiply(inputs['a'], inputs['b'])}

    input_data = [
        {'a': 4, 'b': 3},
        {'a': 1, 'b': 2},
        {'a': 5, 'b': 6},
        {'a': 2, 'b': 3}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [
        {'ab': 12},
        {'ab': 2},
        {'ab': 30},
        {'ab': 6}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'ab': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testMapWithCond(self):
    def preprocessing_fn(inputs):
      return {'a': tf.cond(
          tf.constant(True), lambda: inputs['a'], lambda: inputs['b'])}

    input_data = [
        {'a': 4, 'b': 3},
        {'a': 1, 'b': 2},
        {'a': 5, 'b': 6},
        {'a': 2, 'b': 3}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [
        {'a': 4},
        {'a': 1},
        {'a': 5},
        {'a': 2}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)


  def testWithMoreThanDesiredBatchSize(self):
    def preprocessing_fn(inputs):
      return {
          'ab': tf.multiply(inputs['a'], inputs['b']),
          'i': tft.compute_and_apply_vocabulary(inputs['c'])
      }

    batch_size = 100
    num_instances = batch_size + 1
    input_data = [{
        'a': 2,
        'b': i,
        'c': '%.10i' % i,  # Front-padded to facilitate lexicographic sorting.
    } for i in range(num_instances)]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'c': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'ab': 2*i,
        'i': (len(input_data) - 1) - i,  # Due to reverse lexicographic sorting.
    } for i in range(len(input_data))]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'ab': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'i': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, num_instances - 1, True),
            [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        desired_batch_size=batch_size)

  def testWithUnicode(self):
    def preprocessing_fn(inputs):
      return {'a b': tf.string_join([inputs['a'], inputs['b']], separator=' ')}

    input_data = [{'a': 'Hello', 'b': 'world'}, {'a': 'Hello', 'b': u'κόσμε'}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation()),
    })
    expected_data = [
        {'a b': 'Hello world'},
        {'a b': u'Hello κόσμε'.encode('utf-8')}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'a b': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testComposedMaps(self):
    def preprocessing_fn(inputs):
      return {
          'a(b+c)': tf.multiply(
              inputs['a'], tf.add(inputs['b'], inputs['c']))
      }

    input_data = [{'a': 4, 'b': 3, 'c': 3}, {'a': 1, 'b': 2, 'c': 1}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'c': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{'a(b+c)': 24}, {'a(b+c)': 3}]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'a(b+c)': sch.ColumnSchema(
            tf.float32, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  @tft_unit.parameters((True,), (False,))
  def testScaleUnitInterval(self, elementwise):

    def preprocessing_fn(inputs):
      outputs = {}
      cols = ('x', 'y')
      for col, scaled_t in zip(
          cols,
          tf.unstack(
              tft.scale_to_0_1(
                  tf.stack([inputs[col] for col in cols], axis=1),
                  elementwise=elementwise),
              axis=1)):
        outputs[col + '_scaled'] = scaled_t
      return outputs

    input_data = [{
        'x': 4,
        'y': 5
    }, {
        'x': 1,
        'y': 2
    }, {
        'x': 5,
        'y': 6
    }, {
        'x': 2,
        'y': 3
    }]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'y': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    if elementwise:
      expected_data = [{
          'x_scaled': 0.75,
          'y_scaled': 0.75
      }, {
          'x_scaled': 0.0,
          'y_scaled': 0.0
      }, {
          'x_scaled': 1.0,
          'y_scaled': 1.0
      }, {
          'x_scaled': 0.25,
          'y_scaled': 0.25
      }]
    else:
      expected_data = [{
          'x_scaled': 0.6,
          'y_scaled': 0.8
      }, {
          'x_scaled': 0.0,
          'y_scaled': 0.2
      }, {
          'x_scaled': 0.8,
          'y_scaled': 1.0
      }, {
          'x_scaled': 0.2,
          'y_scaled': 0.4
      }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled':
            sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'y_scaled':
            sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  @tft_unit.parameters((True,), (False,))
  def testScaleMinMax(self, elementwise):

    def preprocessing_fn(inputs):
      outputs = {}
      cols = ('x', 'y')
      for col, scaled_t in zip(
          cols,
          tf.unstack(
              tft.scale_by_min_max(
                  tf.stack([inputs[col] for col in cols], axis=1),
                  output_min=-1,
                  output_max=1,
                  elementwise=elementwise),
              axis=1)):
        outputs[col + '_scaled'] = scaled_t
      return outputs

    input_data = [{
        'x': 4,
        'y': 8
    }, {
        'x': 1,
        'y': 5
    }, {
        'x': 5,
        'y': 9
    }, {
        'x': 2,
        'y': 6
    }]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'y': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    if elementwise:
      expected_data = [{
          'x_scaled': 0.5,
          'y_scaled': 0.5
      }, {
          'x_scaled': -1.0,
          'y_scaled': -1.0
      }, {
          'x_scaled': 1.0,
          'y_scaled': 1.0
      }, {
          'x_scaled': -0.5,
          'y_scaled': -0.5
      }]
    else:
      expected_data = [{
          'x_scaled': -0.25,
          'y_scaled': 0.75
      }, {
          'x_scaled': -1.0,
          'y_scaled': 0.0
      }, {
          'x_scaled': 0.0,
          'y_scaled': 1.0
      }, {
          'x_scaled': -0.75,
          'y_scaled': 0.25
      }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled':
            sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'y_scaled':
            sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testScaleMinMaxConstant(self):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_by_min_max(inputs['x'], 0, 10)}

    input_data = [{'x': 4}, {'x': 4}, {'x': 4}, {'x': 4}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'x_scaled': 5
    }, {
        'x_scaled': 5
    }, {
        'x_scaled': 5
    }, {
        'x_scaled': 5
    }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled':
            sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testScaleMinMaxConstantElementwise(self):

    def preprocessing_fn(inputs):
      outputs = {}
      cols = ('x', 'y')
      for col, scaled_t in zip(
          cols,
          tf.unstack(
              tft.scale_by_min_max(
                  tf.stack([inputs[col] for col in cols], axis=1),
                  output_min=0,
                  output_max=10,
                  elementwise=True),
              axis=1)):
        outputs[col + '_scaled'] = scaled_t
      return outputs

    input_data = [{
        'x': 4,
        'y': 1
    }, {
        'x': 4,
        'y': 1
    }, {
        'x': 4,
        'y': 2
    }, {
        'x': 4,
        'y': 2
    }]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'y': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'x_scaled': 5,
        'y_scaled': 0
    }, {
        'x_scaled': 5,
        'y_scaled': 0
    }, {
        'x_scaled': 5,
        'y_scaled': 10
    }, {
        'x_scaled': 5,
        'y_scaled': 10
    }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled':
            sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation()),
        'y_scaled':
            sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testScaleMinMaxError(self):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_by_min_max(inputs['x'], 2, 1)}

    input_data = [{'x': 1}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{'x_scaled': float('nan')}]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled':
            sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    with self.assertRaises(ValueError) as context:
      self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                            preprocessing_fn, expected_data,
                                            expected_metadata)
    self.assertTrue(
        'output_min must be less than output_max' in context.exception)

  @tft_unit.parameters(
      (tf.int16, tf.float32, True),
      (tf.int32, tf.float32, True),
      (tf.int64, tf.float32, True),
      (tf.float32, tf.float32, True),
      (tf.float64, tf.float64, True),
      (tf.int16, tf.float32, False),
      (tf.int32, tf.float32, False),
      (tf.int64, tf.float32, False),
      (tf.float32, tf.float32, False),
      (tf.float64, tf.float64, False),
  )
  def testScaleToZScore(self, input_dtype, output_dtype, elementwise):

    def preprocessing_fn(inputs):
      outputs = {}
      cols = ('x', 'y')
      for col, scaled_t in zip(cols,
                               tf.unstack(
                                   tft.scale_to_z_score(
                                       tf.stack(
                                           [inputs[col] for col in cols],
                                           axis=1),
                                       elementwise=elementwise),
                                   axis=1)):
        outputs[col + '_scaled'] = scaled_t
      return outputs

    if elementwise:
      input_data = [{
          'x': -4,
          'y': 4
      }, {
          'x': 10,
          'y': -10
      }, {
          'x': 2,
          'y': -2
      }, {
          'x': 4,
          'y': -4
      }]
      # Mean(x) = 3, Mean(y) = -3
      # Var(x) = Var(y) = (7^2 + 7^2 + 1^2 + 1^2) / 4 = 25
      # StdDev(x) = StdDev(y) = 5
      expected_data = [{
          'x_scaled': -1.4,  # (-4 - 3) / 5
          'y_scaled': 1.4  # (4 + 3) / 5
      }, {
          'x_scaled': 1.4,  # (10 - 3) / 5
          'y_scaled': -1.4  # (-10 + 3) / 5
      }, {
          'x_scaled': -0.2,  # (2 - 3) / 5
          'y_scaled': 0.2  # (-2 + 3) / 5
      }, {
          'x_scaled': 0.2,  # (4 - 3) / 5
          'y_scaled': -0.2  # (-4 + 3) / 5
      }]
    else:
      input_data = [{
          'x': -4,
          'y': 10
      }, {
          'x': 2,
          'y': 4
      }]
      # Mean = 3
      # Var = (7^2 + 7^2 + 1^2 + 1^2) / 4 = 25
      # Std Dev = 5
      expected_data = [{
          'x_scaled': -1.4,  # (-4 - 3) / 5
          'y_scaled': 1.4  # (10 - 3) / 5
      }, {
          'x_scaled': -0.2,  # (2 - 3) / 5
          'y_scaled': 0.2  # (4 - 3) / 5
      }]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(input_dtype, [], sch.FixedColumnRepresentation()),
        'y': sch.ColumnSchema(input_dtype, [], sch.FixedColumnRepresentation())
    })
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled':
            sch.ColumnSchema(output_dtype, [], sch.FixedColumnRepresentation()),
        'y_scaled':
            sch.ColumnSchema(output_dtype, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  @parameterized.named_parameters(('Int64In', tf.int64, {
      'min': tf.int64,
      'max': tf.int64,
      'sum': tf.int64,
      'size': tf.int64,
      'mean': tf.float32,
      'var': tf.float32
  }), ('Int32In', tf.int32, {
      'min': tf.int32,
      'max': tf.int32,
      'sum': tf.int64,
      'size': tf.int64,
      'mean': tf.float32,
      'var': tf.float32
  }), ('Int16In', tf.int16, {
      'min': tf.int16,
      'max': tf.int16,
      'sum': tf.int64,
      'size': tf.int64,
      'mean': tf.float32,
      'var': tf.float32
  }), ('Float64In', tf.float64, {
      'min': tf.float64,
      'max': tf.float64,
      'sum': tf.float64,
      'size': tf.int64,
      'mean': tf.float64,
      'var': tf.float64
  }), ('Float32In', tf.float32, {
      'min': tf.float32,
      'max': tf.float32,
      'sum': tf.float32,
      'size': tf.int64,
      'mean': tf.float32,
      'var': tf.float32
  }), ('Float16In', tf.float16, {
      'min': tf.float16,
      'max': tf.float16,
      'sum': tf.float32,
      'size': tf.int64,
      'mean': tf.float16,
      'var': tf.float16
  }))
  def testNumericAnalyzersWithScalarInputs(self, input_dtype, output_dtypes):

    def analyzer_fn(inputs):
      # Also test private analyzers that fuse min/max and mean/var analyzers.
      min_opt, max_opt = analyzers._min_and_max(inputs['a'])
      mean_opt, var_opt = analyzers._mean_and_var(inputs['a'])
      return {
          'min': tft.min(inputs['a']),
          'max': tft.max(inputs['a']),
          'min_opt': min_opt,
          'max_opt': max_opt,
          'sum': tft.sum(inputs['a']),
          'size': tft.size(inputs['a']),
          'mean': tft.mean(inputs['a']),
          'var': tft.var(inputs['a']),
          'mean_opt': mean_opt,
          'var_opt': var_opt
      }

    input_data = [{'a': 4}, {'a': 1}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(input_dtype, [], sch.FixedColumnRepresentation())
    })
    expected_outputs = {
        'min': np.array(1, output_dtypes['min'].as_numpy_dtype),
        'max': np.array(4, output_dtypes['max'].as_numpy_dtype),
        'min_opt': np.array(1, output_dtypes['min'].as_numpy_dtype),
        'max_opt': np.array(4, output_dtypes['max'].as_numpy_dtype),
        'sum': np.array(5, output_dtypes['sum'].as_numpy_dtype),
        'size': np.array(2, output_dtypes['size'].as_numpy_dtype),
        'mean': np.array(2.5, output_dtypes['mean'].as_numpy_dtype),
        'var': np.array(2.25, output_dtypes['var'].as_numpy_dtype),
        'mean_opt': np.array(2.5, output_dtypes['mean'].as_numpy_dtype),
        'var_opt': np.array(2.25, output_dtypes['var'].as_numpy_dtype)
    }
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testNumericAnalyzersWithInputsAndAxis(self):
    def analyzer_fn(inputs):
      # Also test private analyzers that fuse min/max and mean/var analyzers.
      min_opt, max_opt = analyzers._min_and_max(
          inputs['a'], reduce_instance_dims=False)
      mean_opt, var_opt = analyzers._mean_and_var(
          inputs['a'], reduce_instance_dims=False)
      return {
          'min': tft.min(inputs['a'], reduce_instance_dims=False),
          'max': tft.max(inputs['a'], reduce_instance_dims=False),
          'min_opt': min_opt,
          'max_opt': max_opt,
          'sum': tft.sum(inputs['a'], reduce_instance_dims=False),
          'size': tft.size(inputs['a'], reduce_instance_dims=False),
          'mean': tft.mean(inputs['a'], reduce_instance_dims=False),
          'var': tft.var(inputs['a'], reduce_instance_dims=False),
          'mean_opt': mean_opt,
          'var_opt': var_opt
      }

    input_data = [
        {'a': [8, 9, 3, 4]},
        {'a': [1, 2, 10, 11]}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [4], sch.FixedColumnRepresentation())
    })
    expected_outputs = {
        'min': np.array([1, 2, 3, 4], np.int64),
        'max': np.array([8, 9, 10, 11], np.int64),
        'min_opt': np.array([1, 2, 3, 4], np.int64),
        'max_opt': np.array([8, 9, 10, 11], np.int64),
        'sum': np.array([9, 11, 13, 15], np.int64),
        'size': np.array([2, 2, 2, 2], np.int64),
        'mean': np.array([4.5, 5.5, 6.5, 7.5], np.float32),
        'var': np.array([12.25, 12.25, 12.25, 12.25], np.float32),
        'mean_opt': np.array([4.5, 5.5, 6.5, 7.5], np.float32),
        'var_opt': np.array([12.25, 12.25, 12.25, 12.25], np.float32)
    }
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testNumericAnalyzersWithNDInputsAndAxis(self):
    def analyzer_fn(inputs):
      # Also test private analyzers that fuse min/max and mean/var analyzers.
      min_opt, max_opt = analyzers._min_and_max(
          inputs['a'], reduce_instance_dims=False)
      mean_opt, var_opt = analyzers._mean_and_var(
          inputs['a'], reduce_instance_dims=False)
      return {
          'min': tft.min(inputs['a'], reduce_instance_dims=False),
          'max': tft.max(inputs['a'], reduce_instance_dims=False),
          'min_opt': min_opt,
          'max_opt': max_opt,
          'sum': tft.sum(inputs['a'], reduce_instance_dims=False),
          'size': tft.size(inputs['a'], reduce_instance_dims=False),
          'mean': tft.mean(inputs['a'], reduce_instance_dims=False),
          'var': tft.var(inputs['a'], reduce_instance_dims=False),
          'mean_opt': mean_opt,
          'var_opt': var_opt
      }

    input_data = [
        {'a': [[8, 9], [3, 4]]},
        {'a': [[1, 2], [10, 11]]}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [2, 2], sch.FixedColumnRepresentation())
    })
    expected_outputs = {
        'min': np.array([[1, 2], [3, 4]], np.int64),
        'max': np.array([[8, 9], [10, 11]], np.int64),
        'min_opt': np.array([[1, 2], [3, 4]], np.int64),
        'max_opt': np.array([[8, 9], [10, 11]], np.int64),
        'sum': np.array([[9, 11], [13, 15]], np.int64),
        'size': np.array([[2, 2], [2, 2]], np.int64),
        'mean': np.array([[4.5, 5.5], [6.5, 7.5]], np.float32),
        'var': np.array([[12.25, 12.25], [12.25, 12.25]], np.float32),
        'mean_opt': np.array([[4.5, 5.5], [6.5, 7.5]], np.float32),
        'var_opt': np.array([[12.25, 12.25], [12.25, 12.25]], np.float32)
    }
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testNumericAnalyzersWithNDInputs(self):
    def analyzer_fn(inputs):
      # Also test private analyzers that fuse min/max and mean/var analyzers.
      min_opt, max_opt = analyzers._min_and_max(inputs['a'])
      mean_opt, var_opt = analyzers._mean_and_var(inputs['a'])
      return {
          'min': tft.min(inputs['a']),
          'max': tft.max(inputs['a']),
          'min_opt': min_opt,
          'max_opt': max_opt,
          'sum': tft.sum(inputs['a']),
          'size': tft.size(inputs['a']),
          'mean': tft.mean(inputs['a']),
          'var': tft.var(inputs['a']),
          'mean_opt': mean_opt,
          'var_opt': var_opt
      }

    input_data = [
        {'a': [[4, 5], [6, 7]]},
        {'a': [[1, 2], [3, 4]]}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [2, 2], sch.FixedColumnRepresentation())
    })
    expected_outputs = {
        'min': np.array(1, np.int64),
        'max': np.array(7, np.int64),
        'min_opt': np.array(1, np.int64),
        'max_opt': np.array(7, np.int64),
        'sum': np.array(32, np.int64),
        'size': np.array(8, np.int64),
        'mean': np.array(4.0, np.float32),
        'var': np.array(3.5, np.float32),
        'mean_opt': np.array(4.0, np.float32),
        'var_opt': np.array(3.5, np.float32)
    }
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testNumericMeanWithSparseTensor(self):

    def analyzer_fn(inputs):
      return {'mean': tft.mean(inputs['a'])}

    input_data = [{'a': [1, 5, 6]}, {'a': [1, 2]}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [None], sch.ListColumnRepresentation())
    })
    expected_outputs = {'mean': np.array(3., np.float32)}
    self.assertAnalyzerOutputs(input_data, input_metadata, analyzer_fn,
                               expected_outputs)

  def testNumericAnalyzersWithSparseInputs(self):
    def repeat(in_tensor, value):
      batch_size = tf.shape(in_tensor)[0]
      return tf.ones([batch_size], value.dtype) * value

    input_data = [{'a': [4, 5, 6]}, {'a': [1, 2]}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [None], sch.ListColumnRepresentation())
    })
    input_dataset = (input_data, input_metadata)

    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      with self.assertRaises(TypeError):
        def min_fn(inputs):
          return {'min': repeat(inputs['a'], tft.min(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(min_fn)

      with self.assertRaises(TypeError):
        def max_fn(inputs):
          return {'max': repeat(inputs['a'], tft.max(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(max_fn)

      with self.assertRaises(TypeError):
        def sum_fn(inputs):
          return {'sum': repeat(inputs['a'], tft.sum(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(sum_fn)

      with self.assertRaises(TypeError):
        def size_fn(inputs):
          return {'size': repeat(inputs['a'], tft.size(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(size_fn)

      with self.assertRaises(TypeError):
        def mean_fn(inputs):
          return {
              'mean':
                  repeat(inputs['a'],
                         tft.mean(inputs['a'], reduce_instance_dims=False))
          }
        _ = input_dataset | beam_impl.AnalyzeDataset(mean_fn)

      with self.assertRaises(TypeError):
        def var_fn(inputs):
          return {'var': repeat(inputs['a'], tft.var(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(var_fn)

  def testStringToTFIDF(self):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.string_split(inputs['a']))
      out_index, out_values = tft.tfidf(inputs_as_ints, 6)
      return {
          'tf_idf': out_values,
          'index': out_index
      }
    input_data = [{'a': 'hello hello world'},
                  {'a': 'hello goodbye hello world'},
                  {'a': 'I like pie pie pie'}]
    input_schema = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })

    # IDFs
    # hello = log(4/3) = 0.28768
    # world = log(4/3)
    # goodbye = log(4/2) = 0.69314
    # I = log(4/2)
    # like = log(4/2)
    # pie = log(4/2)
    log_4_over_2 = 1.69314718056
    log_4_over_3 = 1.28768207245
    expected_transformed_data = [{
        'tf_idf': [(2/3)*log_4_over_3, (1/3)*log_4_over_3],
        'index': [0, 2]
    }, {
        'tf_idf': [(2/4)*log_4_over_3, (1/4)*log_4_over_3, (1/4)*log_4_over_2],
        'index': [0, 2, 4]
    }, {
        'tf_idf': [(3/5)*log_4_over_2, (1/5)*log_4_over_2, (1/5)*log_4_over_2],
        'index': [1, 3, 5]
    }]
    expected_transformed_schema = dataset_metadata.DatasetMetadata({
        'tf_idf': sch.ColumnSchema(tf.float32, [None],
                                   sch.ListColumnRepresentation()),
        'index': sch.ColumnSchema(tf.int64, [None],
                                  sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_schema, preprocessing_fn, expected_transformed_data,
        expected_transformed_schema)

  def testTFIDFNoData(self):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.string_split(inputs['a']))
      out_index, out_values = tft.tfidf(inputs_as_ints, 6)
      return {
          'tf_idf': out_values,
          'index': out_index
      }
    input_data = [{'a': ''}]
    input_schema = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    expected_transformed_data = [{'tf_idf': [], 'index': []}]
    expected_transformed_schema = dataset_metadata.DatasetMetadata({
        'tf_idf': sch.ColumnSchema(tf.float32, [None],
                                   sch.ListColumnRepresentation()),
        'index': sch.ColumnSchema(tf.int64, [None],
                                  sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_schema, preprocessing_fn, expected_transformed_data,
        expected_transformed_schema)

  def testStringToTFIDFEmptyDoc(self):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.string_split(inputs['a']))
      out_index, out_values = tft.tfidf(inputs_as_ints, 6)
      return {
          'tf_idf': out_values,
          'index': out_index
      }
    input_data = [{'a': 'hello hello world'},
                  {'a': ''},
                  {'a': 'hello goodbye hello world'},
                  {'a': 'I like pie pie pie'}]
    input_schema = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })

    log_5_over_2 = 1.91629073187
    log_5_over_3 = 1.51082562376
    expected_transformed_data = [{
        'tf_idf': [(2/3)*log_5_over_3, (1/3)*log_5_over_3],
        'index': [0, 2]
    }, {
        'tf_idf': [],
        'index': []
    }, {
        'tf_idf': [(2/4)*log_5_over_3, (1/4)*log_5_over_3, (1/4)*log_5_over_2],
        'index': [0, 2, 4]
    }, {
        'tf_idf': [(3/5)*log_5_over_2, (1/5)*log_5_over_2, (1/5)*log_5_over_2],
        'index': [1, 3, 5]
    }]
    expected_transformed_schema = dataset_metadata.DatasetMetadata({
        'tf_idf': sch.ColumnSchema(tf.float32, [None],
                                   sch.ListColumnRepresentation()),
        'index': sch.ColumnSchema(tf.int64, [None],
                                  sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_schema, preprocessing_fn, expected_transformed_data,
        expected_transformed_schema)

  def testIntToTFIDF(self):
    def preprocessing_fn(inputs):
      out_index, out_values = tft.tfidf(inputs['a'], 13)
      return {'tf_idf': out_values, 'index': out_index}
    input_data = [{'a': [2, 2, 0]},
                  {'a': [2, 6, 2, 0]},
                  {'a': [8, 10, 12, 12, 12]},
                 ]
    input_schema = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [None],
                              sch.ListColumnRepresentation())})
    log_4_over_2 = 1.69314718056
    log_4_over_3 = 1.28768207245
    expected_data = [{
        'tf_idf': [(1/3)*log_4_over_3, (2/3)*log_4_over_3],
        'index': [0, 2]
    }, {
        'tf_idf': [(1/4)*log_4_over_3, (2/4)*log_4_over_3, (1/4)*log_4_over_2],
        'index': [0, 2, 6]
    }, {
        'tf_idf': [(1/5)*log_4_over_2, (1/5)*log_4_over_2, (3/5)*log_4_over_2],
        'index': [8, 10, 12]
    }]
    expected_schema = dataset_metadata.DatasetMetadata({
        'tf_idf': sch.ColumnSchema(tf.float32, [None],
                                   sch.ListColumnRepresentation()),
        'index': sch.ColumnSchema(tf.int64, [None],
                                  sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_schema, preprocessing_fn, expected_data,
        expected_schema)

  def testIntToTFIDFWithoutSmoothing(self):
    def preprocessing_fn(inputs):
      out_index, out_values = tft.tfidf(inputs['a'], 13, smooth=False)
      return {'tf_idf': out_values, 'index': out_index}
    input_data = [{'a': [2, 2, 0]},
                  {'a': [2, 6, 2, 0]},
                  {'a': [8, 10, 12, 12, 12]},
                 ]
    input_schema = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [None],
                              sch.ListColumnRepresentation())})
    log_3_over_2 = 1.4054651081
    log_3 = 2.0986122886
    expected_data = [{
        'tf_idf': [(1/3)*log_3_over_2, (2/3)*log_3_over_2],
        'index': [0, 2]
    }, {
        'tf_idf': [(1/4)*log_3_over_2, (2/4)*log_3_over_2, (1/4)*log_3],
        'index': [0, 2, 6]
    }, {
        'tf_idf': [(1/5)*log_3, (1/5)*log_3, (3/5)*log_3],
        'index': [8, 10, 12]
    }]
    expected_schema = dataset_metadata.DatasetMetadata({
        'tf_idf': sch.ColumnSchema(tf.float32, [None],
                                   sch.ListColumnRepresentation()),
        'index': sch.ColumnSchema(tf.int64, [None],
                                  sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_schema, preprocessing_fn, expected_data,
        expected_schema)

  def testTFIDFWithOOV(self):
    test_vocab_size = 3
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.string_split(inputs['a']), top_k=test_vocab_size)
      out_index, out_values = tft.tfidf(inputs_as_ints,
                                        test_vocab_size+1)
      return {
          'tf_idf': out_values,
          'index': out_index
      }
    input_data = [{'a': 'hello hello world'},
                  {'a': 'hello goodbye hello world'},
                  {'a': 'I like pie pie pie'}]
    input_schema = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })

    # IDFs
    # hello = log(3/3) = 0
    # pie = log(3/2) = 0.4054651081
    # world = log(3/3) = 0
    # OOV - goodbye, I, like = log(3/3)
    log_4_over_2 = 1.69314718056
    log_4_over_3 = 1.28768207245
    expected_transformed_data = [{
        'tf_idf': [(2/3)*log_4_over_3, (1/3)*log_4_over_3],
        'index': [0, 2]
    }, {
        'tf_idf': [(2/4)*log_4_over_3, (1/4)*log_4_over_3, (1/4)*log_4_over_3],
        'index': [0, 2, 3]
    }, {
        'tf_idf': [(3/5)*log_4_over_2, (2/5)*log_4_over_3],
        'index': [1, 3]
    }]
    expected_transformed_schema = dataset_metadata.DatasetMetadata({
        'tf_idf': sch.ColumnSchema(tf.float32, [None],
                                   sch.ListColumnRepresentation()),
        'index': sch.ColumnSchema(tf.int64, [None],
                                  sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_schema, preprocessing_fn, expected_transformed_data,
        expected_transformed_schema)

  def testTFIDFWithNegatives(self):
    def preprocessing_fn(inputs):
      out_index, out_values = tft.tfidf(inputs['a'], 14)
      return {
          'tf_idf': out_values,
          'index': out_index
      }
    input_data = [{'a': [2, 2, -4]},
                  {'a': [2, 6, 2, -1]},
                  {'a': [8, 10, 12, 12, 12]},
                 ]
    input_schema = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [None],
                              sch.ListColumnRepresentation())})

    log_4_over_2 = 1.69314718056
    log_4_over_3 = 1.28768207245
    # NOTE: -4 mod 14 = 10
    expected_transformed_data = [{
        'tf_idf': [(2/3)*log_4_over_3, (1/3)*log_4_over_3],
        'index': [2, 10]
    }, {
        'tf_idf': [(2/4)*log_4_over_3, (1/4)*log_4_over_2, (1/4)*log_4_over_2],
        'index': [2, 6, 13]
    }, {
        'tf_idf': [(1/5)*log_4_over_2, (1/5)*log_4_over_3, (3/5)*log_4_over_2],
        'index': [8, 10, 12]
    }]
    expected_transformed_schema = dataset_metadata.DatasetMetadata({
        'tf_idf': sch.ColumnSchema(tf.float32, [None],
                                   sch.ListColumnRepresentation()),
        'index': sch.ColumnSchema(tf.int64, [None],
                                  sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_schema, preprocessing_fn, expected_transformed_data,
        expected_transformed_schema)

  def testVocabularyAnalyzer(self):
    input_data = [
        {'a': 'hello'},
        {'a': 'world'},
        {'a': 'hello'},
        {'a': 'hello'},
        {'a': 'goodbye'},
        {'a': 'world'},
        {'a': 'aaaaa'},
        # Verify the analyzer can handle (dont-ignore) a space-only token.
        {'a': ' '},
        # Verify the analyzer can handle (ignore) the empty string.
        {'a': ''},
        # Verify the analyzer can handle (ignore) tokens that contain \n.
        {'a': '\n'},
        {'a': 'hi \n ho \n'},
        {'a': ' \r'},
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 4, True),
            [], sch.FixedColumnRepresentation())
    })

    # Assert empty string with default_value=-1
    def preprocessing_fn(inputs):
      return {'index': tft.compute_and_apply_vocabulary(inputs['a'])}

    expected_data = [
        {
            'index': 0
        },
        {
            'index': 1
        },
        {
            'index': 0
        },
        {
            'index': 0
        },
        {
            'index': 2
        },
        {
            'index': 1
        },
        {
            'index': 3
        },
        {
            'index': 4
        },
        # The empty string maps to compute_and_apply_vocabulary(
        #     default_value=-1).
        {
            'index': -1
        },
        # The tokens that contain \n map to
        # compute_and_apply_vocabulary(default_value=-1).
        {
            'index': -1
        },
        {
            'index': -1
        },
        {
            'index': -1
        }
    ]
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

    # Assert empty string with num_oov_buckets=1
    def preprocessing_fn_oov(inputs):
      return {
          'index':
              tft.compute_and_apply_vocabulary(
                  inputs['a'], num_oov_buckets=1, vocab_filename='my_vocab')
      }
    expected_data = [
        {'index': 0},
        {'index': 1},
        {'index': 0},
        {'index': 0},
        {'index': 2},
        {'index': 1},
        {'index': 3},
        {'index': 4},
        # The empty string maps to the oov bucket.
        {'index': 5},
        # The tokens that contain \n map to the oov bucket.
        {'index': 5},
        {'index': 5},
        {'index': 5}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index': sch.ColumnSchema(
            sch.IntDomain(tf.int64, 0, 5, True),
            [], sch.FixedColumnRepresentation())
    })
    expected_vocab_file_contents = {
        'my_vocab': ['hello\n', 'world\n', 'goodbye\n', 'aaaaa\n', ' \n']
    }
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn_oov, expected_data,
        expected_metadata,
        expected_vocab_file_contents=expected_vocab_file_contents)

  def testCreateApplyVocab(self):
    input_data = [
        {'a': 'hello', 'b': 'world', 'c': 'aaaaa'},
        {'a': 'good', 'b': '', 'c': 'hello'},
        {'a': 'goodbye', 'b': 'hello', 'c': '\n'},
        {'a': ' ', 'b': 'aaaaa', 'c': 'bbbbb'}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation()),
        'c': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    vocab_filename = 'test_compute_and_apply_vocabulary'
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index_a': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True),
            [], sch.FixedColumnRepresentation()),
        'index_b': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True),
            [], sch.FixedColumnRepresentation())
    })

    def preprocessing_fn(inputs):
      deferred_vocab_and_filename = tft.vocabulary(
          tf.concat([inputs['a'], inputs['b'], inputs['c']], 0),
          vocab_filename=vocab_filename)
      return {
          'index_a':
              tft.apply_vocabulary(inputs['a'], deferred_vocab_and_filename),
          'index_b':
              tft.apply_vocabulary(inputs['b'], deferred_vocab_and_filename)
      }

    expected_data = [
        # For tied frequencies, larger (lexicographic) items come first.
        # Index 5 corresponds to the word bbbbb.
        {'index_a': 0, 'index_b': 2},
        {'index_a': 4, 'index_b': -1},
        {'index_a': 3, 'index_b': 0},
        {'index_a': 6, 'index_b': 1}
    ]
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  # Example on how to use the vocab frequency as part of the transform
  # function.
  def testCreateVocabWithFrequency(self):
    input_data = [
        {'a': 'hello', 'b': 'world', 'c': 'aaaaa'},
        {'a': 'good', 'b': '', 'c': 'hello'},
        {'a': 'goodbye', 'b': 'hello', 'c': '\n'},
        {'a': '_', 'b': 'aaaaa', 'c': 'bbbbb'}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation()),
        'c': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    vocab_filename = 'test_vocab_with_frequency'
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index_a': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True),
            [], sch.FixedColumnRepresentation()),
        'frequency_a': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True),
            [], sch.FixedColumnRepresentation()),
        'index_b': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True),
            [], sch.FixedColumnRepresentation()),
        'frequency_b': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True),
            [], sch.FixedColumnRepresentation())
    })

    def preprocessing_fn(inputs):
      deferred_vocab_and_filename = tft.vocabulary(
          tf.concat([inputs['a'], inputs['b'], inputs['c']], 0),
          vocab_filename=vocab_filename,
          store_frequency=True)

      def _apply_vocab(y, deferred_vocab_filename_tensor):
        # NOTE: Please be aware that TextFileInitializer assigns a special
        # meaning to the constant tf.contrib.lookup.TextFileIndex.LINE_NUMBER.
        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.TextFileInitializer(
                deferred_vocab_filename_tensor,
                tf.string, 1,
                tf.int64, tf.contrib.lookup.TextFileIndex.LINE_NUMBER,
                delimiter=' '), default_value=-1)
        table_size = table.size()
        return table.lookup(y), table_size

      def _apply_frequency(y, deferred_vocab_filename_tensor):
        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.TextFileInitializer(
                deferred_vocab_filename_tensor,
                tf.string, 1,
                tf.int64, 0,
                delimiter=' '), default_value=-1)
        table_size = table.size()
        return table.lookup(y), table_size

      return {
          'index_a':
              tft.apply_vocabulary(
                  inputs['a'],
                  deferred_vocab_and_filename,
                  lookup_fn=_apply_vocab),
          'frequency_a':
              tft.apply_vocabulary(
                  inputs['a'],
                  deferred_vocab_and_filename,
                  lookup_fn=_apply_frequency),
          'index_b':
              tft.apply_vocabulary(
                  inputs['b'],
                  deferred_vocab_and_filename,
                  lookup_fn=_apply_vocab),
          'frequency_b':
              tft.apply_vocabulary(
                  inputs['b'],
                  deferred_vocab_and_filename,
                  lookup_fn=_apply_frequency),
      }

    expected_data = [
        # For tied frequencies, larger (lexicographic) items come first.
        # Index 5 corresponds to the word bbbbb.
        {'index_a': 0, 'frequency_a': 3, 'index_b': 2, 'frequency_b': 1},
        {'index_a': 4, 'frequency_a': 1, 'index_b': -1, 'frequency_b': -1},
        {'index_a': 3, 'frequency_a': 1, 'index_b': 0, 'frequency_b': 3},
        {'index_a': 6, 'frequency_a': 1, 'index_b': 1, 'frequency_b': 2}
    ]
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testAssets(self):
    def preprocessing_fn(inputs):
      return {
          'index':
              tft.compute_and_apply_vocabulary(inputs['a']),
          'index_2':
              tft.compute_and_apply_vocabulary(
                  inputs['b'], vocab_filename='index_2_file')
      }

    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })

    transform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')
    tft_tmp_dir = os.path.join(self.get_temp_dir(), 'temp_dir')
    with beam_impl.Context(temp_dir=tft_tmp_dir):
      with beam.Pipeline() as pipeline:
        input_data = pipeline | beam.Create([
            {'a': 'hello', 'b': 'hi'},
            {'a': 'world', 'b': 'ho'}
        ])
        transform_fn = (
            (input_data, input_metadata)
            | beam_impl.AnalyzeDataset(preprocessing_fn))
        _ = transform_fn | transform_fn_io.WriteTransformFn(transform_fn_dir)

    # Remove the temporary directories, including temporary save models and
    # assets created by tf.Transform as part of the analysis pipeline execution.
    self.assertTrue(os.path.isdir(tft_tmp_dir))
    tf.gfile.DeleteRecursively(tft_tmp_dir)
    self.assertFalse(os.path.isdir(tft_tmp_dir))

    # Assert that the transform function can be read and metadata is as
    # expected.
    expected_output_metadata = dataset_metadata.DatasetMetadata({
        'index': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 1, True), [],
            sch.FixedColumnRepresentation()),
        'index_2': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 1, True), [],
            sch.FixedColumnRepresentation())
    })
    with beam.Pipeline() as pipeline:
      _, metadata = (
          pipeline | transform_fn_io.ReadTransformFn(transform_fn_dir))
      self.assertMetadataEqual(metadata, expected_output_metadata)

    # Finally assert that the output model contains the expected assets and
    # the model can be loaded.
    saved_model_path = os.path.join(transform_fn_dir, 'transform_fn')
    saved_model = saved_model_loader.parse_saved_model(saved_model_path)
    meta_graph_def = saved_model_loader.choose_meta_graph_def(
        saved_model, [constants.TRANSFORM_TAG])
    asset_tensors = saved_model_loader.get_asset_tensors(
        saved_model_path, meta_graph_def)

    # Assert that the assets directory contains the expected asset files
    assets_path = os.path.join(saved_model_path, 'assets')
    self.assertTrue(os.path.isdir(assets_path))
    self.assertEqual(
        ['index_2_file', 'vocab_compute_and_apply_vocabulary_vocabulary'],
        sorted(os.listdir(assets_path)))

    # Verify that the paths are actually there.
    for asset_tensor in asset_tensors.values():
      self.assertTrue(os.path.isfile(asset_tensor))

  def testVocabularyAnalyzerWithNDInputs(self):
    def preprocessing_fn(inputs):
      return {'index': tft.compute_and_apply_vocabulary(inputs['a'])}

    input_data = [
        {'a': [['some', 'say'], ['the', 'world']]},
        {'a': [['will', 'end'], ['in', 'fire']]},
        {'a': [['some', 'say'], ['in', 'ice']]},
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [2, 2],
                              sch.FixedColumnRepresentation())
    })
    expected_data = [
        {'index': [[0, 1], [5, 3]]},
        {'index': [[4, 8], [2, 7]]},
        {'index': [[0, 1], [2, 6]]},
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 8, True),
            [2, 2], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testVocabularyAnalyzerWithTokenization(self):
    def preprocessing_fn(inputs):
      return {
          'index':
              tft.compute_and_apply_vocabulary(tf.string_split(inputs['a']))
      }

    input_data = [{'a': 'hello hello world'}, {'a': 'hello goodbye world'}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{'index': [0, 0, 1]}, {'index': [0, 2, 1]}]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 2, True),
            [None], sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testVocabularyAnalyzerWithTopK(self):
    def preprocessing_fn(inputs):
      return {
          'index1':
              tft.compute_and_apply_vocabulary(
                  tf.string_split(inputs['a']), default_value=-99, top_k=2),

          # As above but using a string for top_k (and changing the
          # default_value to showcase things).
          'index2':
              tft.compute_and_apply_vocabulary(
                  tf.string_split(inputs['a']), default_value=-9, top_k='2')
      }

    input_data = [
        {'a': 'hello hello world'},
        {'a': 'hello goodbye world'},
        {'a': 'hello goodbye foo'}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    # Generated vocab (ordered by frequency, then value) should be:
    # ["hello", "world", "goodbye", "foo"]. After applying top_k=2, this becomes
    # ["hello", "world"].
    expected_data = [
        {'index1': [0, 0, 1], 'index2': [0, 0, 1]},
        {'index1': [0, -99, 1], 'index2': [0, -9, 1]},
        {'index1': [0, -99, -99], 'index2': [0, -9, -9]}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index1': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -99, 1, True),
            [None], sch.ListColumnRepresentation()),
        'index2': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -9, 1, True),
            [None], sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testVocabularyAnalyzerWithFrequencyThreshold(self):
    def preprocessing_fn(inputs):
      return {
          'index1':
              tft.compute_and_apply_vocabulary(
                  tf.string_split(inputs['a']),
                  default_value=-99,
                  frequency_threshold=2),

          # As above but using a string for frequency_threshold (and changing
          # the default_value to showcase things).
          'index2':
              tft.compute_and_apply_vocabulary(
                  tf.string_split(inputs['a']),
                  default_value=-9,
                  frequency_threshold='2')
      }

    input_data = [
        {'a': 'hello hello world'},
        {'a': 'hello goodbye world'},
        {'a': 'hello goodbye foo'}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    # Generated vocab (ordered by frequency, then value) should be:
    # ["hello", "world", "goodbye", "foo"]. After applying frequency_threshold=2
    # this becomes
    # ["hello", "world", "goodbye"].
    expected_data = [
        {'index1': [0, 0, 1], 'index2': [0, 0, 1]},
        {'index1': [0, 2, 1], 'index2': [0, 2, 1]},
        {'index1': [0, 2, -99], 'index2': [0, 2, -9]}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index1': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -99, 2, True),
            [None], sch.ListColumnRepresentation()),
        'index2': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -9, 2, True),
            [None], sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testVocabularyAnalyzerWithFrequencyThresholdTooHigh(self):
    # Expected to return an empty dict due to too high threshold.
    def preprocessing_fn(inputs):
      return {
          'index1':
              tft.compute_and_apply_vocabulary(
                  tf.string_split(inputs['a']),
                  default_value=-99,
                  frequency_threshold=77),

          # As above but using a string for frequency_threshold (and changing
          # the default_value to showcase things).
          'index2':
              tft.compute_and_apply_vocabulary(
                  tf.string_split(inputs['a']),
                  default_value=-9,
                  frequency_threshold='77')
      }

    input_data = [
        {'a': 'hello hello world'},
        {'a': 'hello goodbye world'},
        {'a': 'hello goodbye foo'}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    # Generated vocab (ordered by frequency, then value) should be:
    # ["hello", "world", "goodbye", "foo"]. After applying frequency_threshold=2
    # this becomes empty.
    expected_data = [
        {'index1': [-99, -99, -99], 'index2': [-9, -9, -9]},
        {'index1': [-99, -99, -99], 'index2': [-9, -9, -9]},
        {'index1': [-99, -99, -99], 'index2': [-9, -9, -9]}
    ]
    # Note the vocabs are empty but the tables have size 1 so max_value is 1.
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index1': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -99, 0, True),
            [None], sch.ListColumnRepresentation()),
        'index2': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -9, 0, True),
            [None], sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testVocabularyAnalyzerWithHighFrequencyThresholdAndOOVBuckets(self):
    def preprocessing_fn(inputs):
      return {
          'index1':
              tft.compute_and_apply_vocabulary(
                  tf.string_split(inputs['a']),
                  default_value=-99,
                  top_k=1,
                  num_oov_buckets=3)
      }

    input_data = [
        {'a': 'hello hello world world'},
        {'a': 'hello tarkus toccata'},
        {'a': 'hello goodbye foo'}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    # Generated vocab (ordered by frequency, then value) should be:
    # ["hello", "world", "goodbye", "foo", "tarkus", "toccata"]. After applying
    # top_k =1 this becomes ["hello"] plus three OOV buckets.
    # The specific output values here depend on the hash of the words, and the
    # test will break if the hash changes.
    expected_data = [
        {'index1': [0, 0, 2, 2]},
        {'index1': [0, 3, 1]},
        {'index1': [0, 2, 1]},
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index1': sch.ColumnSchema(
            sch.IntDomain(tf.int64, 0, 3, True), [None],
            sch.ListColumnRepresentation()),
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testPipelineWithoutAutomaterialization(self):
    # The tests in BaseTFTransformImplTest, when run with the beam
    # implementation, pass lists instead of PCollections and thus invoke
    # automaterialization where each call to a beam PTransform will implicitly
    # run its own pipeline.
    #
    # In order to test the case where PCollections are not materialized in
    # between calls to the tf.Transform PTransforms, we include a test that is
    # not based on automaterialization.
    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    with beam.Pipeline() as pipeline:
      input_data = pipeline | 'CreateTrainingData' >> beam.Create(
          [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}])
      metadata = dataset_metadata.DatasetMetadata({
          'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
      })
      with beam_impl.Context(temp_dir=self.get_temp_dir()):
        transform_fn = (
            (input_data, metadata)
            | 'AnalyzeDataset' >> beam_impl.AnalyzeDataset(preprocessing_fn))

      # Run transform_columns on some eval dataset.
      eval_data = pipeline | 'CreateEvalData' >> beam.Create(
          [{'x': 6}, {'x': 3}])
      transformed_eval_data, _ = (
          ((eval_data, metadata), transform_fn)
          | 'TransformDataset' >> beam_impl.TransformDataset())
      expected_data = [{'x_scaled': 1.25}, {'x_scaled': 0.5}]
      beam_test_util.assert_that(
          transformed_eval_data, beam_test_util.equal_to(expected_data))

  def testTransformFnExportAndImportRoundtrip(self):
    transform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')
    metadata_dir = os.path.join(self.get_temp_dir(), 'export_metadata')

    with beam.Pipeline() as pipeline:
      def preprocessing_fn(inputs):
        return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

      metadata = dataset_metadata.DatasetMetadata({
          'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
      })
      data = pipeline | 'CreateTrainingData' >> beam.Create(
          [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}])
      with beam_impl.Context(temp_dir=self.get_temp_dir()):
        _, transform_fn = (
            (data, metadata)
            | 'AnalyzeAndTransform'
            >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

      _ = transform_fn | transform_fn_io.WriteTransformFn(transform_fn_dir)
      _ = metadata | beam_metadata_io.WriteMetadata(metadata_dir,
                                                    pipeline=pipeline)

    with beam.Pipeline() as pipeline:
      transform_fn = pipeline | transform_fn_io.ReadTransformFn(
          transform_fn_dir)
      # We have to load metadata in non-deferred manner to use it as an input to
      # TransformDataset.
      metadata = metadata_io.read_metadata(metadata_dir)
      # Run transform_columns on some eval dataset.
      eval_data = pipeline | 'CreateEvalData' >> beam.Create(
          [{'x': 6}, {'x': 3}])
      transformed_eval_data, _ = (
          ((eval_data, metadata), transform_fn)
          | 'Transform' >> beam_impl.TransformDataset())
      expected_data = [{'x_scaled': 1.25}, {'x_scaled': 0.5}]
      beam_test_util.assert_that(
          transformed_eval_data, beam_test_util.equal_to(expected_data))

  def testRunExportedGraph(self):
    # Run analyze_and_transform_columns on some dataset.
    def preprocessing_fn(inputs):
      x_scaled = tft.scale_to_0_1(inputs['x'])
      y_sum = tf.sparse_reduce_sum(inputs['y'], axis=1)
      z_copy = tf.SparseTensor(
          inputs['z'].indices, inputs['z'].values, inputs['z'].dense_shape)
      return {'x_scaled': x_scaled, 'y_sum': y_sum, 'z_copy': z_copy}

    metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(
            tf.float32, [], sch.FixedColumnRepresentation()),
        'y': sch.ColumnSchema(
            tf.float32, [10], sch.SparseColumnRepresentation(
                'val_copy', [sch.SparseIndexField('idx_copy', False)])),
        'z': sch.ColumnSchema(
            tf.float32, [None], sch.ListColumnRepresentation()),
    })
    data = [
        {'x': 4, 'y': ([0, 1], [0., 1.]), 'z': [2., 4., 6.], 'P': 17},
        {'x': 1, 'y': ([2, 3], [2., 3.]), 'z': [8.], 'P': 17},
        {'x': 5, 'y': ([4, 5], [4., 5.]), 'z': [1., 2., 3.], 'P': 17},
    ]
    with beam_impl.Context(temp_dir=self.get_temp_dir(),
                           passthrough_keys={'P'}):
      (transformed_dataset, _), transform_fn = (
          (data, metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    self.assertTrue(
        all(instance['P'] == 17 for instance in transformed_dataset))

    export_dir = os.path.join(self.get_temp_dir(), 'export')
    _ = transform_fn | transform_fn_io.WriteTransformFn(export_dir)

    # Load the exported graph, and apply it to a batch of data.
    g = tf.Graph()
    with g.as_default():
      with tf.Session():
        inputs, outputs = (
            saved_transform_io.partially_apply_saved_transform_internal(
                os.path.join(export_dir, 'transform_fn'), {}))
      x, y, z = inputs['x'], inputs['y'], inputs['z']
      feed = {
          x: [6., 3., 0., 1.],
          y: tf.SparseTensorValue(
              indices=[[0, 6], [0, 7], [1, 8]],
              values=[6., 7., 8.],
              dense_shape=[2, 10]),
          z: tf.SparseTensorValue(
              indices=[[0, 1], [0, 2], [4, 10]],
              values=[1., 2., 3.],
              dense_shape=[4, 10]),
      }

      sess = tf.Session()
      with sess.as_default():
        result = sess.run(outputs, feed_dict=feed)

      expected_transformed_data = {
          'x_scaled': [1.25, 0.5, -0.25, 0.0],
          'y_sum': [13.0, 8.0],
          'z_copy': tf.SparseTensorValue(
              indices=[[0, 1], [0, 2], [4, 10]],
              values=[1., 2., 3.],
              dense_shape=[4, 10]),
      }
      self.assertDataCloseOrEqual([expected_transformed_data], [result])

      # Verify that it breaks predictably if we feed unbatched data.
      with self.assertRaises(ValueError):
        feed = {
            x: 6.,
            y: tf.SparseTensorValue(indices=[[6], [7]], values=[6., 7.],
                                    dense_shape=[10])
        }
        sess = tf.Session()
        with sess.as_default():
          _ = sess.run(outputs, feed_dict=feed)

  def testNestedContextCreateBaseTempDir(self):
    level_1_dir = self.get_temp_dir()
    with beam_impl.Context(temp_dir=level_1_dir):
      self.assertEqual(
          os.path.join(level_1_dir, beam_impl.Context._TEMP_SUBDIR),
          beam_impl.Context.create_base_temp_dir())
      level_2_dir = self.get_temp_dir()
      with beam_impl.Context(temp_dir=level_2_dir):
        self.assertEqual(
            os.path.join(level_2_dir, beam_impl.Context._TEMP_SUBDIR),
            beam_impl.Context.create_base_temp_dir())
      self.assertEqual(
          os.path.join(level_1_dir, beam_impl.Context._TEMP_SUBDIR),
          beam_impl.Context.create_base_temp_dir())
    with self.assertRaises(ValueError):
      beam_impl.Context.create_base_temp_dir()

  @tft_unit.parameters(
      # Test for all integral types, each type is in a separate testcase to
      # increase parallelism of test shards (and reduce test time from ~250
      # seconds to ~80 seconds)
      *_construct_test_bucketization_parameters())
  def testBucketization(self, test_inputs, expected_boundaries, do_shuffle,
                        epsilon, should_apply, is_manual_boundaries,
                        input_dtype):
    test_inputs = list(test_inputs)

    # Shuffle the input to add randomness to input generated with
    # simple range().
    if do_shuffle:
      random.shuffle(test_inputs)

    def preprocessing_fn(inputs):

      x = inputs['x']
      num_buckets = len(expected_boundaries) + 1
      if should_apply:
        if is_manual_boundaries:
          bucket_boundaries = expected_boundaries
        else:
          bucket_boundaries = tft.quantiles(inputs['x'], num_buckets, epsilon)
        result = tft.apply_buckets(x, bucket_boundaries)
      else:
        result = tft.bucketize(x, num_buckets=num_buckets, epsilon=epsilon)
      return {'q_b': result}

    input_data = [{'x': [x]} for x in test_inputs]

    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(
            input_dtype, [1], sch.FixedColumnRepresentation())
    })

    # Sort the input based on value, index is used to create expected_data.
    indexed_input = enumerate(test_inputs)
    sorted_list = sorted(indexed_input,
                         cmp=lambda (xi, xv), (yi, yv): cmp(xv, yv))

    # Expected data has the same size as input, one bucket per input value.
    expected_data = [None] * len(test_inputs)
    bucket = 0
    for (index, x) in sorted_list:
      # Increment the bucket number when crossing the boundary
      if (bucket < len(expected_boundaries) and
          x >= expected_boundaries[bucket]):
        bucket += 1
      expected_data[index] = {'q_b': [bucket]}

    expected_metadata = dataset_metadata.DatasetMetadata({
        'q_b': sch.ColumnSchema(
            sch.IntDomain(tf.int64, 0, len(expected_boundaries), True),
            [1], sch.FixedColumnRepresentation())
    })

    @contextlib.contextmanager
    def no_assert():
      yield None

    assertion = no_assert()
    if input_dtype == tf.float16:
      assertion = self.assertRaisesRegexp(
          TypeError, '.*DataType float16 not in list of allowed values.*')

    with assertion:
      self.assertAnalyzeAndTransformResults(
          input_data,
          input_metadata,
          preprocessing_fn,
          expected_data,
          expected_metadata,
          desired_batch_size=1000)

  @tft_unit.parameters(
      # Test for all integral types, each type is in a separate testcase to
      # increase parallelism of test shards and reduce test time.
      (tf.int32,),
      (tf.int64,),
      (tf.float32,),
      (tf.float64,),
      (tf.double,),
  )
  def testQuantileBuckets(self, input_dtype):

    def analyzer_fn(inputs):
      return {'q_b': tft.quantiles(inputs['x'], num_buckets=3, epsilon=0.00001)}

    # NOTE: We force 3 batches: data has 3000 elements and we request a batch
    # size of 1000.
    input_data = [{'x': [x]} for  x in range(1, 3000)]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(input_dtype, [1], sch.FixedColumnRepresentation())
    })
    # The expected data has 2 boundaries that divides the data into 3 buckets.
    expected_outputs = {'q_b': np.array([[1001, 2001]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=1000)

  def testQuantileBucketsWithKey(self):
    def analyzer_fn(inputs):
      key_vocab, q_b = analyzers._quantiles_per_key(
          inputs['x'], inputs['key'], num_buckets=3, epsilon=0.00001)
      return {
          'key_vocab': key_vocab,
          'q_b': q_b
      }

    # NOTE: We force 10 batches: data has 100 elements and we request a batch
    # size of 10.
    input_data = [{'x': [x], 'key': 'a' if x < 50 else 'b'}
                  for x in range(1, 100)]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.int64, [1], sch.FixedColumnRepresentation()),
        'key': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    # The expected data has 2 boundaries that divides the data into 3 buckets.
    expected_outputs = {
        'key_vocab': np.array(['a', 'b'], np.object),
        'q_b': np.array([[18, 34], [67, 84]], np.float32)
    }
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=10)

  def testVocabularyWithFrequency(self):
    outfile = 'vocabulary_with_frequency'
    def preprocessing_fn(inputs):

      # Force the analyzer to be executed, and store the frequency file as a
      # side-effect.
      _ = tft.vocabulary(
          inputs['a'], vocab_filename=outfile, store_frequency=True)
      _ = tft.vocabulary(inputs['a'], store_frequency=True)
      _ = tft.vocabulary(inputs['b'], store_frequency=True)

      # The following must not produce frequency output, just the vocab words.
      _ = tft.vocabulary(inputs['b'])
      a_int = tft.compute_and_apply_vocabulary(inputs['a'])

      # Return input unchanged, this preprocessing_fn is a no-op except for
      # computing uniques.
      return {'a_int': a_int}

    def check_asset_file_contents(assets_path, filename, expected):
      assets_file = os.path.join(assets_path, filename)
      with tf.gfile.GFile(assets_file, 'r') as f:
        contents = f.read()

      self.assertMultiLineEqual(expected, contents)

    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })

    tft_tmp_dir = os.path.join(self.get_temp_dir(), 'temp_dir')
    transform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')

    with beam_impl.Context(temp_dir=tft_tmp_dir):
      with beam.Pipeline() as pipeline:
        input_data = pipeline | beam.Create([
            {'a': 'hello', 'b': 'hi'},
            {'a': 'world', 'b': 'ho ho'},
            {'a': 'hello', 'b': 'ho ho'},
        ])
        transform_fn = (
            (input_data, input_metadata)
            | beam_impl.AnalyzeDataset(preprocessing_fn))
        _ = transform_fn | transform_fn_io.WriteTransformFn(transform_fn_dir)

    self.assertTrue(os.path.isdir(tft_tmp_dir))

    saved_model_path = os.path.join(transform_fn_dir, 'transform_fn')
    assets_path = os.path.join(saved_model_path, 'assets')
    self.assertTrue(os.path.isdir(assets_path))
    six.assertCountEqual(self, [
        outfile, 'vocab_frequency_vocabulary_1', 'vocab_frequency_vocabulary_2',
        'vocab_compute_and_apply_vocabulary_vocabulary', 'vocab_vocabulary_3'
    ], os.listdir(assets_path))

    check_asset_file_contents(assets_path, outfile,
                              '2 hello\n1 world\n')

    check_asset_file_contents(assets_path, 'vocab_frequency_vocabulary_1',
                              '2 hello\n1 world\n')

    check_asset_file_contents(assets_path, 'vocab_frequency_vocabulary_2',
                              '2 ho ho\n1 hi\n')

    check_asset_file_contents(assets_path, 'vocab_vocabulary_3',
                              'ho ho\nhi\n')

    check_asset_file_contents(assets_path,
                              'vocab_compute_and_apply_vocabulary_vocabulary',
                              'hello\nworld\n')

  def testCovarianceTwoDimensions(self):
    def analyzer_fn(inputs):
      return {'y': tft.covariance(inputs['x'], dtype=tf.float64)}

    input_data = [{'x': x} for x in [[0, 0], [4, 0], [2, -2], [2, 2]]]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float64,
                              [2],
                              sch.FixedColumnRepresentation())
    })
    expected_outputs = {'y': np.array([[2, 0], [0, 2]], np.float64)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testCovarianceOneDimension(self):
    def analyzer_fn(inputs):
      return {'y': tft.covariance(inputs['x'], dtype=tf.float64)}

    input_data = [{'x': x} for x in [[0], [2], [4], [6]]]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float64,
                              [1],
                              sch.FixedColumnRepresentation())
    })
    expected_outputs = {'y': np.array([[5]], np.float64)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testPCAThreeToTwoDimensions(self):
    def analyzer_fn(inputs):
      return {'y': tft.pca(inputs['x'], 2, dtype=tf.float64)}

    input_data = [{'x': x}
                  for x in  [[0, 0, 1], [4, 0, 1], [2, -1, 1], [2, 1, 1]]]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float64,
                              [3],
                              sch.FixedColumnRepresentation())
    })
    expected_outputs = {'y': np.array([[1, 0], [0, 1], [0, 0]], np.float64)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def _asssert_quantile_boundaries(
      self, test_inputs, expected_boundaries, input_dtype, num_buckets=None):

    if not num_buckets:
      num_buckets = len(expected_boundaries) + 1

    def preprocessing_fn(inputs):
      return {
          'q_b': tft.quantiles(inputs['x'], num_buckets, epsilon=0.0001)
      }

    input_data = [{'x': [x]} for x in test_inputs]

    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(
            input_dtype, [1], sch.FixedColumnRepresentation())
    })

    # Expected data has the same size as input, one bucket per input value.
    batch_size = 1000
    expected_data = []
    num_batches = int(math.ceil(len(test_inputs) / float(batch_size)))

    for _ in range(num_batches):
      expected_data += [{'q_b': expected_boundaries}]

    expected_metadata = None

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        desired_batch_size=batch_size)

  def testBucketizationForTightSequence(self):
    # Divide a tight 1..N sequence into different number of buckets.
    self._asssert_quantile_boundaries(
        [1, 2, 3, 4], [3], tf.int32, num_buckets=2)
    self._asssert_quantile_boundaries(
        [1, 2, 3, 4], [3, 4], tf.int32, num_buckets=3)
    self._asssert_quantile_boundaries(
        [1, 2, 3, 4], [2, 3, 4], tf.int32, num_buckets=4)
    self._asssert_quantile_boundaries(
        [1, 2, 3, 4], [1, 2, 3, 4], tf.int32, num_buckets=5)
    # Request more number of buckets than there are inputs.
    self._asssert_quantile_boundaries(
        [1, 2, 3, 4], [1, 2, 3, 4], tf.int32, num_buckets=6)
    self._asssert_quantile_boundaries(
        [1, 2, 3, 4], [1, 2, 3, 4], tf.int32, num_buckets=10)

  def testBucketizationEqualDistributionInSequence(self):
    # Input pattern is of the form [1, 1, 1, ..., 2, 2, 2, ..., 3, 3, 3, ...]
    inputs = []
    for i in range(1, 101):
      inputs += [i] * 100
    # Expect 100 equally spaced buckets.
    expected_buckets = range(1, 101)
    self._asssert_quantile_boundaries(
        inputs, expected_buckets, tf.int32, num_buckets=101)

  def testBucketizationEqualDistributionInterleaved(self):
    # Input pattern is of the form [1, 2, 3, ..., 1, 2, 3, ..., 1, 2, 3, ...]
    sequence = range(1, 101)
    inputs = []
    for _ in range(1, 101):
      inputs += sequence
    # Expect 100 equally spaced buckets.
    expected_buckets = range(1, 101)
    self._asssert_quantile_boundaries(
        inputs, expected_buckets, tf.int32, num_buckets=101)

  def testBucketizationSpecificDistribution(self):
    # Distribution of input values.
    # This distribution is taken from one of the user pipelines.
    dist = (
        # Format: ((<min-value-in-range>, <max-value-in-range>), num-values)
        ((0.51, 0.67), 4013),
        ((0.67, 0.84), 2321),
        ((0.84, 1.01), 7145),
        ((1.01, 1.17), 64524),
        ((1.17, 1.34), 42886),
        ((1.34, 1.51), 154809),
        ((1.51, 1.67), 382678),
        ((1.67, 1.84), 582744),
        ((1.84, 2.01), 252221),
        ((2.01, 2.17), 7299))

    inputs = []
    for (mn, mx), num in dist:
      for _ in range(num//100):
        inputs += [random.uniform(mn, mx)]
    # NOTE: for this input data, the number of boundaries returned is more
    # than that needed to form the number of buckets. Below, we request 5
    # buckets, which translates to 4 boundaries. But, due to approximate
    # quantiles, the number of boundaries returned is 5.
    expected_boundaries = [1.520135, 1.646375, 1.738915, 1.827343, 2.166727]
    self._asssert_quantile_boundaries(
        inputs, expected_boundaries, tf.float32, num_buckets=5)



if __name__ == '__main__':
  unittest.main()
