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

import itertools
import os

# GOOGLE-INITIALIZATION

import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np
import six
from six.moves import range

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import schema_inference
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from google.protobuf import text_format
import unittest
from tensorflow.core.example import example_pb2
from tensorflow.python.ops import lookup_ops
from tensorflow_metadata.proto.v0 import schema_pb2

if common.IS_ANNOTATIONS_PB_AVAILABLE:
  from tensorflow_transform import annotations_pb2  # pylint: disable=g-import-not-at-top


_TFXIO_NAMED_PARAMETERS = [
    dict(testcase_name='WithTFXIO', use_tfxio=True),
    dict(testcase_name='NoTFXIO', use_tfxio=False),
]

_SCALE_TO_Z_SCORE_TEST_CASES = [
    dict(testcase_name='int16',
         input_data=np.array([[1], [1], [2], [2]], np.int16),
         output_data=np.array([[-1.0], [-1.0], [1.0], [1.0]], np.float32),
         elementwise=False),
    dict(testcase_name='int32',
         input_data=np.array([[1], [1], [2], [2]], np.int32),
         output_data=np.array([[-1.0], [-1.0], [1.0], [1.0]], np.float32),
         elementwise=False),
    dict(testcase_name='int64',
         input_data=np.array([[1], [1], [2], [2]], np.int64),
         output_data=np.array([[-1.0], [-1.0], [1.0], [1.0]], np.float32),
         elementwise=False),
    dict(testcase_name='float32',
         input_data=np.array([[1], [1], [2], [2]], np.float32),
         output_data=np.array([[-1.0], [-1.0], [1.0], [1.0]], np.float32),
         elementwise=False),
    dict(testcase_name='float64',
         input_data=np.array([[1], [1], [2], [2]], np.float64),
         output_data=np.array([[-1.0], [-1.0], [1.0], [1.0]], np.float64),
         elementwise=False),
    dict(testcase_name='vector',
         input_data=np.array([[1, 2], [3, 4]], np.float32),
         output_data=np.array([[-3, -1], [1, 3]] / np.sqrt(5.0), np.float32),
         elementwise=False),
    dict(testcase_name='vector_elementwise',
         input_data=np.array([[1, 2], [3, 4]], np.float32),
         output_data=np.array([[-1.0, -1.0], [1.0, 1.0]], np.float32),
         elementwise=True),
    dict(testcase_name='zero_varance',
         input_data=np.array([[3], [3], [3], [3]], np.float32),
         output_data=np.array([[0], [0], [0], [0]], np.float32),
         elementwise=False),
    dict(testcase_name='zero_variance_elementwise',
         input_data=np.array([[3, 4], [3, 4]], np.float32),
         output_data=np.array([[0, 0], [0, 0]], np.float32),
         elementwise=True),
]


def sum_output_dtype(input_dtype):
  """Returns the output dtype for tft.sum."""
  return input_dtype if input_dtype.is_floating else tf.int64


def _mean_output_dtype(input_dtype):
  """Returns the output dtype for tft.mean (and similar functions)."""
  return tf.float64 if input_dtype == tf.float64 else tf.float32


class BeamImplTest(tft_unit.TransformTestCase):

  def setUp(self):
    tf.compat.v1.logging.info('Starting test case: %s', self._testMethodName)

    self._context = beam_impl.Context(use_deep_copy_optimization=True)
    self._context.__enter__()

  def tearDown(self):
    self._context.__exit__()

  def _SkipIfExternalEnvironmentAnd(self, predicate, reason):
    if predicate and tft_unit.is_external_environment():
      raise unittest.SkipTest(reason)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testApplySavedModelSingleInput(self, use_tfxio):
    def save_model_with_single_input(instance, export_dir):
      builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
      with tf.compat.v1.Graph().as_default() as graph:
        with instance.test_session(graph=graph) as sess:
          input1 = tf.compat.v1.placeholder(
              dtype=tf.int64, shape=[3], name='myinput1')
          initializer = tf.compat.v1.constant_initializer([1, 2, 3])
          with tf.compat.v1.variable_scope(
              'Model', reuse=None, initializer=initializer):
            v1 = tf.compat.v1.get_variable('v1', [3], dtype=tf.int64)
          output1 = tf.add(v1, input1, name='myadd1')
          inputs = {'single_input': input1}
          outputs = {'single_output': output1}
          signature_def_map = {
              'serving_default':
                  tf.compat.v1.saved_model.signature_def_utils
                  .predict_signature_def(inputs, outputs)
          }
          sess.run(tf.compat.v1.global_variables_initializer())
          builder.add_meta_graph_and_variables(
              sess, [tf.saved_model.SERVING],
              signature_def_map=signature_def_map)
          builder.save(False)

    export_dir = os.path.join(self.get_temp_dir(), 'saved_model_single')

    def preprocessing_fn(inputs):
      x = inputs['x']
      output_col = tft.apply_saved_model(
          export_dir, x, tags=[tf.saved_model.SERVING])
      return {'out': output_col}

    save_model_with_single_input(self, export_dir)
    input_data = [
        {'x': [1, 2, 3]},
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([3], tf.int64),
    })
    # [1, 2, 3] + [1, 2, 3] = [2, 4, 6]
    expected_data = [
        {'out': [2, 4, 6]}
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'out': tf.io.FixedLenFeature([3], tf.int64)})
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testApplySavedModelWithHashTable(self, use_tfxio):
    def save_model_with_hash_table(instance, export_dir):
      builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
      with tf.compat.v1.Graph().as_default() as graph:
        with instance.test_session(graph=graph) as sess:
          key = tf.constant('test_key', shape=[1])
          value = tf.constant('test_value', shape=[1])
          table = lookup_ops.HashTable(
              lookup_ops.KeyValueTensorInitializer(key, value), '__MISSING__')

          input1 = tf.compat.v1.placeholder(
              dtype=tf.string, shape=[1], name='myinput')
          output1 = tf.reshape(table.lookup(input1), shape=[1])
          inputs = {'input': input1}
          outputs = {'output': output1}

          signature_def_map = {
              'serving_default':
                  tf.compat.v1.saved_model.signature_def_utils
                  .predict_signature_def(inputs, outputs)
          }

          sess.run(table.init)
          builder.add_meta_graph_and_variables(
              sess, [tf.saved_model.SERVING],
              signature_def_map=signature_def_map)
          builder.save(False)

    export_dir = os.path.join(self.get_temp_dir(), 'saved_model_hash_table')

    def preprocessing_fn(inputs):
      x = inputs['x']
      output_col = tft.apply_saved_model(
          export_dir, x, tags=[tf.saved_model.SERVING])
      return {'out': output_col}

    save_model_with_hash_table(self, export_dir)
    input_data = [
        {'x': ['test_key']}
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([1], tf.string),
    })
    expected_data = [
        {'out': b'test_value'}
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'out': tf.io.FixedLenFeature([], tf.string)})
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testApplySavedModelMultiInputs(self, use_tfxio):

    def save_model_with_multi_inputs(instance, export_dir):
      builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
      with tf.compat.v1.Graph().as_default() as graph:
        with instance.test_session(graph=graph) as sess:
          input1 = tf.compat.v1.placeholder(
              dtype=tf.int64, shape=[3], name='myinput1')
          input2 = tf.compat.v1.placeholder(
              dtype=tf.int64, shape=[3], name='myinput2')
          input3 = tf.compat.v1.placeholder(
              dtype=tf.int64, shape=[3], name='myinput3')
          initializer = tf.compat.v1.constant_initializer([1, 2, 3])
          with tf.compat.v1.variable_scope(
              'Model', reuse=None, initializer=initializer):
            v1 = tf.compat.v1.get_variable('v1', [3], dtype=tf.int64)
          o1 = tf.add(v1, input1, name='myadd1')
          o2 = tf.subtract(o1, input2, name='mysubtract1')
          output1 = tf.add(o2, input3, name='myadd2')
          inputs = {'name1': input1, 'name2': input2,
                    'name3': input3}
          outputs = {'single_output': output1}
          signature_def_map = {
              'serving_default':
                  tf.compat.v1.saved_model.signature_def_utils
                  .predict_signature_def(inputs, outputs)
          }
          sess.run(tf.compat.v1.global_variables_initializer())
          builder.add_meta_graph_and_variables(
              sess, [tf.saved_model.SERVING],
              signature_def_map=signature_def_map)
          builder.save(False)

    export_dir = os.path.join(self.get_temp_dir(), 'saved_model_multi')

    def preprocessing_fn(inputs):
      x = inputs['x']
      y = inputs['y']
      z = inputs['z']
      sum_column = tft.apply_saved_model(
          export_dir, {
              'name1': x,
              'name3': z,
              'name2': y
          },
          tags=[tf.saved_model.SERVING])
      return {'sum': sum_column}

    save_model_with_multi_inputs(self, export_dir)
    input_data = [
        {'x': [1, 2, 3], 'y': [2, 3, 4], 'z': [1, 1, 1]},
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([3], tf.int64),
        'y': tf.io.FixedLenFeature([3], tf.int64),
        'z': tf.io.FixedLenFeature([3], tf.int64),
    })
    # [1, 2, 3] + [1, 2, 3] - [2, 3, 4] + [1, 1, 1] = [1, 2, 3]
    expected_data = [
        {'sum': [1, 2, 3]}
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'sum': tf.io.FixedLenFeature([3], tf.int64)})
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testApplyFunctionWithCheckpoint(self, use_tfxio):

    def tensor_fn(input1, input2):
      initializer = tf.compat.v1.constant_initializer([1, 2, 3])
      with tf.compat.v1.variable_scope(
          'Model', reuse=None, initializer=initializer):
        v1 = tf.compat.v1.get_variable('v1', [3], dtype=tf.int64)
        v2 = tf.compat.v1.get_variable('v2', [3], dtype=tf.int64)
        o1 = tf.add(v1, v2, name='add1')
        o2 = tf.subtract(o1, input1, name='sub1')
        o3 = tf.subtract(o2, input2, name='sub2')
        return o3

    def save_checkpoint(instance, checkpoint_path):
      with tf.compat.v1.Graph().as_default() as graph:
        with instance.test_session(graph=graph) as sess:
          input1 = tf.compat.v1.placeholder(
              dtype=tf.int64, shape=[3], name='myinput1')
          input2 = tf.compat.v1.placeholder(
              dtype=tf.int64, shape=[3], name='myinput2')
          tensor_fn(input1, input2)
          saver = tf.compat.v1.train.Saver()
          sess.run(tf.compat.v1.global_variables_initializer())
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
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([3], tf.int64),
        'y': tf.io.FixedLenFeature([3], tf.int64),
    })
    # [1, 2, 3] + [1, 2, 3] - [2, 2, 2] - [-1, -3, 1] = [1, 5, 3]
    expected_data = [
        {'out': [1, 5, 3]}
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'out': tf.io.FixedLenFeature([3], tf.int64)})
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(
      *tft_unit.cross_named_parameters([
          dict(testcase_name='NoDeepCopy', with_deep_copy=False),
          dict(testcase_name='WithDeepCopy', with_deep_copy=True),
      ], _TFXIO_NAMED_PARAMETERS))
  def testMultipleLevelsOfAnalyzers(self, with_deep_copy, use_tfxio):
    # Test a preprocessing function similar to scale_to_0_1 except that it
    # involves multiple interleavings of analyzers and transforms.
    def preprocessing_fn(inputs):
      scaled_to_0 = inputs['x'] - tft.min(inputs['x'])
      scaled_to_0_1 = scaled_to_0 / tft.max(scaled_to_0)
      return {'x_scaled': scaled_to_0_1}

    input_data = [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.float32)})
    expected_data = [
        {'x_scaled': 0.75},
        {'x_scaled': 0.0},
        {'x_scaled': 1.0},
        {'x_scaled': 0.25}
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([], tf.float32)})
    with beam_impl.Context(use_deep_copy_optimization=with_deep_copy):
      # NOTE: In order to correctly test deep_copy here, we can't pass test_data
      # to assertAnalyzeAndTransformResults.
      # Not passing test_data to assertAnalyzeAndTransformResults means that
      # tft.AnalyzeAndTransform is called, exercising the right code path.
      self.assertAnalyzeAndTransformResults(
          input_data, input_metadata, preprocessing_fn, expected_data,
          expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testRawFeedDictInput(self, use_tfxio):
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
          'x':
              tf.io.FixedLenSequenceFeature(
                  shape=[], dtype=tf.string, default_value=None)
      }
      _, sequences = tf.io.parse_single_sequence_example(
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
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'sequence_example': tf.io.FixedLenFeature([], tf.string)})
    expected_data = [
        {'x': b'ab'},
        {'x': b''},
        {'x': b'c'},
        {'x': b'd'},
        {'x': b'ef'},
        {'x': b'g'}
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.string)})

    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata, desired_batch_size=1, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testTransformWithExcludedOutputs(self, use_tfxio):
    def preprocessing_fn(inputs):
      return {
          'x_scaled': tft.scale_to_0_1(inputs['x']),
          'y_scaled': tft.scale_to_0_1(inputs['y'])
      }

    # Run AnalyzeAndTransform on some input data and compare with expected
    # output.
    input_data = [{'x': 5, 'y': 1}, {'x': 1, 'y': 2}]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32)
    })
    with beam_impl.Context(temp_dir=self.get_temp_dir(), use_tfxio=use_tfxio):
      if use_tfxio:
        input_data, input_metadata = self.convert_to_tfxio_api_inputs(
            input_data, input_metadata)
      transform_fn = (
          (input_data, input_metadata) | beam_impl.AnalyzeDataset(
              preprocessing_fn))

    # Take the transform function and use TransformDataset to apply it to
    # some eval data, with missing 'y' column.
    eval_data = [{'x': 6}]
    eval_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.float32)})
    transformed_eval_data, transformed_eval_metadata = (
        ((eval_data, eval_metadata), transform_fn)
        | beam_impl.TransformDataset(exclude_outputs=['y_scaled']))

    expected_transformed_eval_data = [{'x_scaled': 1.25}]
    expected_transformed_eval_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([], tf.float32)})
    self.assertDataCloseOrEqual(transformed_eval_data,
                                expected_transformed_eval_data)
    self.assertEqual(transformed_eval_metadata.dataset_metadata,
                     expected_transformed_eval_metadata)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testMapWithCond(self, use_tfxio):
    def preprocessing_fn(inputs):
      return {
          'a':
              tf.cond(
                  pred=tf.constant(True),
                  true_fn=lambda: inputs['a'],
                  false_fn=lambda: inputs['b'])
      }

    input_data = [
        {'a': 4, 'b': 3},
        {'a': 1, 'b': 2},
        {'a': 5, 'b': 6},
        {'a': 2, 'b': 3}
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.float32),
        'b': tf.io.FixedLenFeature([], tf.float32)
    })
    expected_data = [
        {'a': 4},
        {'a': 1},
        {'a': 5},
        {'a': 2}
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.float32)})
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testPyFuncs(self, use_tfxio):
    def my_multiply(x, y):
      return x*y

    def my_add(x, y):
      return x+y

    def preprocessing_fn(inputs):
      result = {
          'a+b': tft.apply_pyfunc(
              my_add, tf.float32, True, 'add', inputs['a'], inputs['b']),
          'a+c': tft.apply_pyfunc(
              my_add, tf.float32, True, 'add', inputs['a'], inputs['c']),
          'ab': tft.apply_pyfunc(
              my_multiply, tf.float32, False, 'multiply',
              inputs['a'], inputs['b']),
          'sum_scaled': tft.scale_to_0_1(
              tft.apply_pyfunc(
                  my_add, tf.float32, True, 'add', inputs['a'], inputs['c']))
      }
      for value in result.values():
        value.set_shape([1,])
      return result

    input_data = [
        {'a': 4, 'b': 3, 'c': 2},
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 5, 'b': 6, 'c': 7},
        {'a': 2, 'b': 3, 'c': 4}
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.float32),
        'b': tf.io.FixedLenFeature([], tf.float32),
        'c': tf.io.FixedLenFeature([], tf.float32)
    })
    expected_data = [
        {'ab': 12, 'a+b': 7, 'a+c': 6, 'sum_scaled': 0.25},
        {'ab': 2, 'a+b': 3, 'a+c': 4, 'sum_scaled': 0},
        {'ab': 30, 'a+b': 11, 'a+c': 12, 'sum_scaled': 1},
        {'ab': 6, 'a+b': 5, 'a+c': 6, 'sum_scaled': 0.25}
    ]
    # When calling tf.py_func, the output shape is set to unknown.
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'ab': tf.io.FixedLenFeature([], tf.float32),
        'a+b': tf.io.FixedLenFeature([], tf.float32),
        'a+c': tf.io.FixedLenFeature([], tf.float32),
        'sum_scaled': tf.io.FixedLenFeature([], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testWithMoreThanDesiredBatchSize(self, use_tfxio):
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
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.float32),
        'b': tf.io.FixedLenFeature([], tf.float32),
        'c': tf.io.FixedLenFeature([], tf.string)
    })
    expected_data = [{
        'ab': 2*i,
        'i': (len(input_data) - 1) - i,  # Due to reverse lexicographic sorting.
    } for i in range(len(input_data))]
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'ab': tf.io.FixedLenFeature([], tf.float32),
        'i': tf.io.FixedLenFeature([], tf.int64),
    }, {
        'i':
            schema_pb2.IntDomain(
                min=-1, max=num_instances - 1, is_categorical=True)
    })
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        desired_batch_size=batch_size,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testWithUnicode(self, use_tfxio):
    def preprocessing_fn(inputs):
      return {'a b': tf.compat.v1.strings.join(
          [inputs['a'], inputs['b']], separator=' ')}

    input_data = [{'a': 'Hello', 'b': 'world'}, {'a': 'Hello', 'b': u'κόσμε'}]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.string),
        'b': tf.io.FixedLenFeature([], tf.string),
    })
    expected_data = [
        {'a b': b'Hello world'},
        {'a b': u'Hello κόσμε'.encode('utf-8')}
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'a b': tf.io.FixedLenFeature([], tf.string)})
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.parameters(*tft_unit.cross_parameters(
      [(True,), (False,)], [(True,), (False,)]))
  def testScaleUnitInterval(self, elementwise, use_tfxio):

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
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32)
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
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature([], tf.float32),
        'y_scaled': tf.io.FixedLenFeature([], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata,
                                          use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testScaleUnitIntervalPerKey(self, use_tfxio):

    def preprocessing_fn(inputs):
      outputs = {}
      cols = ('x', 'y')
      for col, scaled_t in zip(
          cols,
          tf.unstack(
              tft.scale_to_0_1_per_key(
                  tf.stack([inputs[col] for col in cols], axis=1),
                  inputs['key'],
                  elementwise=False),
              axis=1)):
        outputs[col + '_scaled'] = scaled_t
      return outputs

    input_data = [{
        'x': 4,
        'y': 5,
        'key': 'a'
    }, {
        'x': 1,
        'y': 2,
        'key': 'a'
    }, {
        'x': 5,
        'y': 6,
        'key': 'a'
    }, {
        'x': 2,
        'y': 3,
        'key': 'a'
    }, {
        'x': 25,
        'y': -25,
        'key': 'b'
    }, {
        'x': 5,
        'y': 0,
        'key': 'b'
    }]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'key': tf.io.FixedLenFeature([], tf.string)
    })
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
    }, {
        'x_scaled': 1.0,
        'y_scaled': 0.0
    }, {
        'x_scaled': 0.6,
        'y_scaled': 0.5
    }]
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature([], tf.float32),
        'y_scaled': tf.io.FixedLenFeature([], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata,
                                          use_tfxio=use_tfxio)

  @tft_unit.parameters(*tft_unit.cross_parameters(
      [(True,), (False,)], [(True,), (False,)]))
  def testScaleMinMax(self, elementwise, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

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
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32)
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
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature([], tf.float32),
        'y_scaled': tf.io.FixedLenFeature([], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters([
      dict(testcase_name='_empty_filename',
           key_vocabulary_filename=''),
      dict(testcase_name='_nonempty_filename',
           key_vocabulary_filename='per_key'),
      dict(testcase_name='_none_filename',
           key_vocabulary_filename=None)
  ], _TFXIO_NAMED_PARAMETERS))
  def testScaleMinMaxPerKey(self, key_vocabulary_filename, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    def preprocessing_fn(inputs):
      outputs = {}
      cols = ('x', 'y')
      for col, scaled_t in zip(
          cols,
          tf.unstack(
              tft.scale_by_min_max_per_key(
                  tf.stack([inputs[col] for col in cols], axis=1),
                  inputs['key'],
                  output_min=-1,
                  output_max=1,
                  elementwise=False,
                  key_vocabulary_filename=key_vocabulary_filename),
              axis=1)):
        outputs[col + '_scaled'] = scaled_t
      return outputs

    input_data = [{
        'x': 4,
        'y': 8,
        'key': 'a'
    }, {
        'x': 1,
        'y': 5,
        'key': 'a'
    }, {
        'x': 5,
        'y': 9,
        'key': 'a'
    }, {
        'x': 2,
        'y': 6,
        'key': 'a'
    }, {
        'x': -2,
        'y': 0,
        'key': 'b'
    }, {
        'x': 0,
        'y': 2,
        'key': 'b'
    }]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'key': tf.io.FixedLenFeature([], tf.string)
    })

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
    }, {
        'x_scaled': -1.0,
        'y_scaled': 0.0
    }, {
        'x_scaled': 0.0,
        'y_scaled': 1.0
    }]
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature([], tf.float32),
        'y_scaled': tf.io.FixedLenFeature([], tf.float32)
    })
    if key_vocabulary_filename:
      per_key_vocab_contents = {key_vocabulary_filename:
                                    [(b'a', [-1.0, 9.0]), (b'b', [2.0, 2.0])]}
    else:
      per_key_vocab_contents = None
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata,
        expected_vocab_file_contents=per_key_vocab_contents, use_tfxio=True)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters([
      dict(
          testcase_name='sparse_key',
          input_data=[{
              'idx': [0, 1],
              'val': [-4, 4],
              'key_idx': [0, 1],
              'key': ['a', 'a']
          }, {
              'idx': [0, 1],
              'val': [2, 1],
              'key_idx': [0, 1],
              'key': ['a', 'b']
          }, {
              'idx': [0, 1],
              'val': [-1, 4],
              'key_idx': [0, 1],
              'key': ['b', 'a']
          }],
          input_metadata=tft_unit.metadata_from_feature_spec({
              'x':
                  tf.io.SparseFeature(
                      'idx', 'val', tft_unit.canonical_numeric_dtype(
                          tf.float32), 4),
              'key':
                  tf.io.SparseFeature('key_idx', 'key', tf.string, 4)
          }),
          expected_data=[{
              'x_scaled': [0., 1., 0, 0]
          }, {
              'x_scaled': [.75, 1., 0, 0]
          }, {
              'x_scaled': [0., 1., 0, 0]
          }]),
      dict(
          testcase_name='dense_key',
          input_data=[{
              'idx': [0, 1],
              'val': [-4, 4],
              'key': 'a'
          }, {
              'idx': [0, 1],
              'val': [2, 1],
              'key': 'a'
          }, {
              'idx': [0, 1],
              'val': [-1, 4],
              'key': 'b'
          }],
          input_metadata=tft_unit.metadata_from_feature_spec({
              'x':
                  tf.io.SparseFeature(
                      'idx', 'val', tft_unit.canonical_numeric_dtype(
                          tf.float32), 4),
              'key':
                  tf.io.FixedLenFeature([], tf.string)
          }),
          expected_data=[{
              'x_scaled': [0., 1., 0, 0]
          }, {
              'x_scaled': [.75, .625, 0, 0]
          }, {
              'x_scaled': [0., 1., 0, 0]
          }]),
  ], _TFXIO_NAMED_PARAMETERS))
  def testScaleMinMaxSparsePerKey(
      self, input_data, input_metadata, expected_data, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    def preprocessing_fn(inputs):
      x_scaled = tf.sparse.to_dense(
          tft.scale_to_0_1_per_key(inputs['x'], inputs['key']))
      x_scaled.set_shape([None, 4])
      return {'x_scaled': x_scaled}

    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([4], tf.float32)})

    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata,
                                          use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testScaleMinMaxConstant(self, use_tfxio):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_by_min_max(inputs['x'], 0, 10)}

    input_data = [{'x': 4}, {'x': 4}, {'x': 4}, {'x': 4}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.float32)})
    expected_data = [{
        'x_scaled': 5
    }, {
        'x_scaled': 5
    }, {
        'x_scaled': 5
    }, {
        'x_scaled': 5
    }]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([], tf.float32)})
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata,
                                          use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testScaleMinMaxConstantElementwise(self, use_tfxio):

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
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32)
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
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature([], tf.float32),
        'y_scaled': tf.io.FixedLenFeature([], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata,
                                          use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testScaleMinMaxError(self, use_tfxio):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_by_min_max(inputs['x'], 2, 1)}

    input_data = [{'x': 1}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.float32)})
    expected_data = [{'x_scaled': float('nan')}]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([], tf.float32)})
    with self.assertRaises(ValueError) as context:
      self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                            preprocessing_fn, expected_data,
                                            expected_metadata,
                                            use_tfxio=use_tfxio)
    self.assertTrue(
        'output_min must be less than output_max' in str(context.exception))

  @tft_unit.named_parameters(
      *tft_unit.cross_named_parameters(_SCALE_TO_Z_SCORE_TEST_CASES,
                                       _TFXIO_NAMED_PARAMETERS))
  def testScaleToZScore(self, input_data, output_data, elementwise, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    def preprocessing_fn(inputs):
      x = inputs['x']
      x_cast = tf.cast(x, tf.as_dtype(input_data.dtype))
      x_scaled = tft.scale_to_z_score(x_cast, elementwise=elementwise)
      self.assertEqual(x_scaled.dtype, tf.as_dtype(output_data.dtype))
      return {'x_scaled': tf.cast(x_scaled, tf.float32)}

    input_data_dicts = [{'x': x} for x in input_data]
    expected_data_dicts = [{'x_scaled': x_scaled} for x_scaled in output_data]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x':
            tf.io.FixedLenFeature(
                input_data.shape[1:],
                tft_unit.canonical_numeric_dtype(tf.as_dtype(
                    input_data.dtype))),
    })
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature(output_data.shape[1:], tf.float32),
    })
    self.assertAnalyzeAndTransformResults(
        input_data_dicts, input_metadata,
        preprocessing_fn, expected_data_dicts, expected_metadata,
        use_tfxio=use_tfxio)

  @tft_unit.parameters(*itertools.product([
      tf.int16,
      tf.int32,
      tf.int64,
      tf.float32,
      tf.float64,
  ], (True, False), (True, False)))
  def testScaleToZScoreSparse(self, input_dtype, elementwise, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    def preprocessing_fn(inputs):
      z_score = tf.sparse.to_dense(
          tft.scale_to_z_score(
              tf.cast(inputs['x'], input_dtype), elementwise=elementwise),
          default_value=np.nan)
      z_score.set_shape([None, 4])
      self.assertEqual(z_score.dtype, _mean_output_dtype(input_dtype))
      return {
          'x_scaled': tf.cast(z_score, tf.float32)
      }

    input_data = [
        {'idx': [0, 1], 'val': [-4, 10]},
        {'idx': [0, 1], 'val': [2, 4]},
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x':
            tf.io.SparseFeature('idx', 'val',
                                tft_unit.canonical_numeric_dtype(input_dtype),
                                4)
    })
    if elementwise:
      # Mean(x) = [-1, 7]
      # Var(x) = [9, 9]
      # StdDev(x) = [3, 3]
      expected_data = [
          {
              'x_scaled': [-1., 1.,
                           float('nan'),
                           float('nan')]  # [(-4 +1 ) / 3, (10 -7) / 3]
          },
          {
              'x_scaled': [1., -1.,
                           float('nan'),
                           float('nan')]  # [(2 + 1) / 3, (4 - 7) / 3]
          }
      ]
    else:
      # Mean = 3
      # Var = 25
      # Std Dev = 5
      expected_data = [
          {
              'x_scaled': [-1.4, 1.4, float('nan'),
                           float('nan')]  # [(-4 - 3) / 5, (10 - 3) / 5]
          },
          {
              'x_scaled': [-.2, .2, float('nan'),
                           float('nan')]  # [(2 - 3) / 5, (4 - 3) / 5]
          }
      ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([4], tf.float32)})
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata,
                                          use_tfxio=use_tfxio)

  @tft_unit.parameters(*itertools.product([
      tf.int16,
      tf.int32,
      tf.int64,
      tf.float32,
      tf.float64,
  ], [True, False]))
  def testScaleToZScoreSparsePerDenseKey(self, input_dtype, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')
    # TODO(b/131852830) Add elementwise tests.
    def preprocessing_fn(inputs):

      def scale_to_z_score_per_key(tensor, key):
        z_score = tft.scale_to_z_score_per_key(
            tf.cast(tensor, input_dtype), key=key, elementwise=False)
        self.assertEqual(z_score.dtype, _mean_output_dtype(input_dtype))
        return tf.cast(z_score, tf.float32)

      return {
          'x_scaled': scale_to_z_score_per_key(inputs['x'], inputs['key']),
          'y_scaled': scale_to_z_score_per_key(inputs['y'], inputs['key']),
      }
    np_dtype = input_dtype.as_numpy_dtype
    input_data = [{
        'x': np.array([-4, 2], dtype=np_dtype),
        'y': np.array([0, 0], dtype=np_dtype),
        'key': 'a',
    }, {
        'x': np.array([10, 4], dtype=np_dtype),
        'y': np.array([0, 0], dtype=np_dtype),
        'key': 'a',
    }, {
        'x': np.array([1, -1], dtype=np_dtype),
        'y': np.array([0, 0], dtype=np_dtype),
        'key': 'b',
    }]
    # Mean(x) = 3, Mean(y) = 0
    # Var(x) = (-7^2 + -1^2 + 7^2 + 1^2) / 4 = 25, Var(y) = 0
    # StdDev(x) = 5, StdDev(y) = 0
    # 'b':
    # Mean(x) = 0, Mean(y) = 0
    # Var(x) = 1, Var(y) = 0
    # StdDev(x) = 1, StdDev(y) = 0
    expected_data = [
        {
            'x_scaled': [-1.4, -.2],  # [(-4 - 3) / 5, (2 - 3) / 5]
            'y_scaled': [0., 0.],
        },
        {
            'x_scaled': [1.4, .2],  # [(10 - 3) / 5, (4 - 3) / 5]
            'y_scaled': [0., 0.],
        },
        {
            'x_scaled': [1., -1.],  # [(1 - 0) / 1, (-1 - 0) / 1]
            'y_scaled': [0., 0.],
        }
    ]

    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.VarLenFeature(tft_unit.canonical_numeric_dtype(input_dtype)),
        'y': tf.io.VarLenFeature(tft_unit.canonical_numeric_dtype(input_dtype)),
        'key': tf.io.FixedLenFeature([], tf.string),
    })
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.VarLenFeature(tf.float32),
        'y_scaled': tf.io.VarLenFeature(tf.float32),
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata,
                                          use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters([
      dict(testcase_name='_empty_filename',
           key_vocabulary_filename=''),
      dict(testcase_name='_nonempty_filename',
           key_vocabulary_filename='per_key'),
      dict(testcase_name='_none_filename',
           key_vocabulary_filename=None)
  ], _TFXIO_NAMED_PARAMETERS))
  def testScaleToZScorePerKey(self, key_vocabulary_filename, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')
    # TODO(b/131852830) Add elementwise tests.
    def preprocessing_fn(inputs):

      def scale_to_z_score_per_key(tensor, key, var_name=''):
        if key_vocabulary_filename is None:
          filename = None
        else:
          filename = key_vocabulary_filename + var_name
        z_score = tft.scale_to_z_score_per_key(
            tf.cast(tensor, tf.float32), key=key, elementwise=False,
            key_vocabulary_filename=filename)
        self.assertEqual(z_score.dtype, tf.float32)
        return z_score

      return {
          'x_scaled': scale_to_z_score_per_key(inputs['x'], inputs['key'], 'x'),
          'y_scaled': scale_to_z_score_per_key(inputs['y'], inputs['key'], 'y'),
          's_scaled': scale_to_z_score_per_key(inputs['s'], inputs['key'], 's'),
      }

    np_dtype = np.float32
    input_data = [{
        'x': np.array([-4], dtype=np_dtype),
        'y': np.array([0], dtype=np_dtype),
        's': 3,
        'key': 'a',
    }, {
        'x': np.array([10], dtype=np_dtype),
        'y': np.array([0], dtype=np_dtype),
        's': -3,
        'key': 'a',
    }, {
        'x': np.array([1], dtype=np_dtype),
        'y': np.array([0], dtype=np_dtype),
        's': 3,
        'key': 'b',
    }, {
        'x': np.array([2], dtype=np_dtype),
        'y': np.array([0], dtype=np_dtype),
        's': 3,
        'key': 'a',
    }, {
        'x': np.array([4], dtype=np_dtype),
        'y': np.array([0], dtype=np_dtype),
        's': -3,
        'key': 'a',
    }, {
        'x': np.array([-1], dtype=np_dtype),
        'y': np.array([0], dtype=np_dtype),
        's': -3,
        'key': 'b',
    }]
    # 'a':
    # Mean(x) = 3, Mean(y) = 0
    # Var(x) = (-7^2 + -1^2 + 7^2 + 1^2) / 4 = 25, Var(y) = 0
    # StdDev(x) = 5, StdDev(y) = 0
    # 'b':
    # Mean(x) = 0, Mean(y) = 0
    # Var(x) = 1, Var(y) = 0
    # StdDev(x) = 1, StdDev(y) = 0
    expected_data = [
        {
            'x_scaled': [-1.4],  # [(-4 - 3) / 5, (2 - 3) / 5]
            'y_scaled': [0.],
            's_scaled': 1.,
        },
        {
            'x_scaled': [1.4],  # [(10 - 3) / 5, (4 - 3) / 5]
            'y_scaled': [0.],
            's_scaled': -1.,
        },
        {
            'x_scaled': [1.],  # [(1 - 0) / 1, (-1 - 0) / 1]
            'y_scaled': [0.],
            's_scaled': 1.,
        },
        {
            'x_scaled': [-.2],  # [(-4 - 3) / 5, (2 - 3) / 5]
            'y_scaled': [0.],
            's_scaled': 1.,
        },
        {
            'x_scaled': [.2],  # [(10 - 3) / 5, (4 - 3) / 5]
            'y_scaled': [0.],
            's_scaled': -1.,
        },
        {
            'x_scaled': [-1.],  # [(1 - 0) / 1, (-1 - 0) / 1]
            'y_scaled': [0.],
            's_scaled': -1.,
        }
    ]

    input_metadata = tft_unit.metadata_from_feature_spec({
        'x':
            tf.io.FixedLenFeature([1],
                                  tft_unit.canonical_numeric_dtype(tf.float32)),
        'y':
            tf.io.FixedLenFeature([1],
                                  tft_unit.canonical_numeric_dtype(tf.float32)),
        's':
            tf.io.FixedLenFeature([],
                                  tft_unit.canonical_numeric_dtype(tf.float32)),
        'key':
            tf.io.FixedLenFeature([], tf.string),
    })
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature([1], tf.float32),
        'y_scaled': tf.io.FixedLenFeature([1], tf.float32),
        's_scaled': tf.io.FixedLenFeature([], tf.float32),
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata,
                                          use_tfxio=use_tfxio)

  @tft_unit.parameters(*itertools.product([
      tf.int16,
      tf.int32,
      tf.int64,
      tf.float32,
      tf.float64,
  ], [True, False]))
  def testScaleToZScoreSparsePerKey(self, input_dtype, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')
    # TODO(b/131852830) Add elementwise tests.
    def preprocessing_fn(inputs):
      z_score = tf.sparse.to_dense(
          tft.scale_to_z_score_per_key(
              tf.cast(inputs['x'], input_dtype),
              inputs['key'],
              elementwise=False),
          default_value=np.nan)
      z_score.set_shape([None, 4])
      self.assertEqual(z_score.dtype, _mean_output_dtype(input_dtype))
      return {
          'x_scaled': tf.cast(z_score, tf.float32)
      }

    input_data = [
        {'idx': [0, 1], 'val': [-4, 10], 'key_idx': [0, 1], 'key': ['a', 'a']},
        {'idx': [0, 1], 'val': [2, 1], 'key_idx': [0, 1], 'key': ['a', 'b']},
        {'idx': [0, 1], 'val': [-1, 4], 'key_idx': [0, 1], 'key': ['b', 'a']},
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'key':
            tf.io.SparseFeature('key_idx', 'key', tf.string, 4),
        'x':
            tf.io.SparseFeature('idx', 'val',
                                tft_unit.canonical_numeric_dtype(input_dtype),
                                4)
    })
    # 'a':
    # Mean = 3
    # Var = 25
    # Std Dev = 5
    # 'b':
    # Mean = 0
    # Var = 1
    # Std Dev = 1
    expected_data = [
        {
            'x_scaled': [-1.4, 1.4, float('nan'),
                         float('nan')]  # [(-4 - 3) / 5, (10 - 3) / 5]
        },
        {
            'x_scaled': [-.2, 1., float('nan'),
                         float('nan')]  # [(2 - 3) / 5, (1 - 0) / 1]
        },
        {
            'x_scaled': [-1., .2,
                         float('nan'),
                         float('nan')]  # [(-1 - 0) / 1, (4 - 3) / 5]
        }
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([4], tf.float32)})
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata,
                                          use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testMeanAndVar(self, use_tfxio):
    def analyzer_fn(inputs):
      mean, var = analyzers._mean_and_var(inputs['x'])
      return {
          'mean': mean,
          'var': var
      }

    # NOTE: We force 10 batches: data has 100 elements and we request a batch
    # size of 10.
    input_data = [{'x': [x]}
                  for x in range(1, 101)]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([1], tf.int64)
    })
    expected_outputs = {
        'mean': np.float32(50.5),
        'var': np.float32(833.25)
    }
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=10,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testMeanAndVarPerKey(self, use_tfxio):
    def analyzer_fn(inputs):
      key_vocab, mean, var = analyzers._mean_and_var_per_key(
          inputs['x'], inputs['key'])
      return {
          'key_vocab': key_vocab,
          'mean': mean,
          'var': tf.round(100 * var) / 100.0
      }

    # NOTE: We force 10 batches: data has 100 elements and we request a batch
    # size of 10.
    input_data = [{'x': [x], 'key': 'a' if x < 50 else 'b'}
                  for x in range(1, 101)]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([1], tf.int64),
        'key': tf.io.FixedLenFeature([], tf.string)
    })
    expected_outputs = {
        'key_vocab': np.array([b'a', b'b'], np.object),
        'mean': np.array([25, 75], np.float32),
        'var': np.array([200, 216.67], np.float32)
    }
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=10,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters([
      {'testcase_name': '_string',
       'input_data': [{'key': 'a' if x < 25 else 'b'} for x in range(100)],
       'input_metadata': tft_unit.metadata_from_feature_spec(
           {'key': tf.io.FixedLenFeature([], tf.string)}),
       'expected_outputs': {
           'elements': np.array([b'a', b'b'], np.object),
           'counts': np.array([25, 75], np.int64)
       }
      },
      {'testcase_name': '_int',
       'input_data': [{'key': 0 if x < 25 else 1} for x in range(100)],
       'input_metadata': tft_unit.metadata_from_feature_spec(
           {'key': tf.io.FixedLenFeature([], tf.int64)}),
       'expected_outputs': {
           'elements': np.array([0, 1], np.int64),
           'counts': np.array([25, 75], np.int64)
       }
      },
  ], _TFXIO_NAMED_PARAMETERS))
  def testCountPerKey(self, input_data, input_metadata, expected_outputs,
                      use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    def analyzer_fn(inputs):
      elements, counts = analyzers.count_per_key(inputs['key'])
      return {
          'elements': elements,
          'counts': counts
      }
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters([
      {
          'testcase_name': '_uniform',
          'input_data': [{
              'x': [x]
          } for x in range(10, 100)],
          'feature_spec': {
              'x': tf.io.FixedLenFeature([1], tf.int64)
          },
          'boundaries': 10 * np.arange(11, dtype=np.float32),
          'categorical': False,
          'expected_outputs': {
              'hist':
                  10 * np.array([0] + [1] * 9, np.int64),
              'boundaries':
                  10 * np.arange(11, dtype=np.float32).reshape((1, 11))
          }
      },
      {
          'testcase_name': '_categorical_string',
          'input_data': [{
              'x': [str(x % 10) + '_']
          } for x in range(1, 101)],
          'feature_spec': {
              'x': tf.io.FixedLenFeature([1], tf.string)
          },
          'boundaries': None,
          'categorical': True,
          'expected_outputs': {
              'hist':
                  10 * np.ones(10, np.int64),
              'boundaries':
                  np.asarray(
                      sorted([
                          tf.compat.as_bytes(str(x % 10) + '_')
                          for x in range(10)
                      ]),
                      dtype=np.object)
          },
      },
      {
          'testcase_name': '_categorical_int',
          'input_data': [{
              'x': [(x % 10)]
          } for x in range(1, 101)],
          'feature_spec': {
              'x': tf.io.FixedLenFeature([1], tf.int64)
          },
          'boundaries': None,
          'categorical': True,
          'expected_outputs': {
              'hist': 10 * np.ones(10, np.int64),
              'boundaries': np.arange(10)
          }
      },
  ], _TFXIO_NAMED_PARAMETERS))
  def testHistograms(self, input_data, feature_spec, boundaries, categorical,
                     expected_outputs, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    def analyzer_fn(inputs):
      counts, bucket_boundaries = analyzers.histogram(tf.stack(inputs['x']),
                                                      categorical=categorical,
                                                      boundaries=boundaries)
      if not categorical:
        bucket_boundaries = tf.math.round(bucket_boundaries)
      return {'hist': counts, 'boundaries': bucket_boundaries}

    input_metadata = tft_unit.metadata_from_feature_spec(feature_spec)
    self.assertAnalyzerOutputs(input_data,
                               input_metadata,
                               analyzer_fn,
                               expected_outputs,
                               use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testProbCategorical(self, use_tfxio):
    def preprocessing_fn(inputs):
      return {'probs': tft.estimated_probability_density(inputs['x'],
                                                         categorical=True)}

    # NOTE: We force 10 batches: data has 100 elements and we request a batch
    # size of 10.
    input_data = [{'x': [str(x % 10) + '_']} for x in range(1, 101)]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([1], tf.string)
    })
    expected_outputs = [{
        'probs': np.array(np.ones((1)) / 10.0, np.float32)
    } for _ in range(100)]
    self.assertAnalyzeAndTransformResults(input_data,
                                          input_metadata,
                                          preprocessing_fn,
                                          expected_outputs,
                                          desired_batch_size=10,
                                          use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testProbTenBoundaries(self, use_tfxio):
    # If we draw uniformly from a range (0, 100], the expected density is 0.01.
    def preprocessing_fn(inputs):
      return {'probs': tft.estimated_probability_density(
          inputs['x'], boundaries=list(range(0, 101, 10)))}

    # NOTE: We force 10 batches: data has 100 elements and we request a batch
    # size of 10.
    input_data = [{'x': [x]} for x in range(100)]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([1], tf.int64)
    })
    expected_outputs = [{
        'probs': np.array(np.ones((1)) / (100.0), np.float32)
    } for _ in range(100)]
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_outputs,
        desired_batch_size=10,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters([
      {'testcase_name': 'uniform',
       'boundaries': 6,
       'input_data': [{'x': [x]} for x in range(100)],
       'expected_outputs': [{'probs': np.array(np.ones((1)) / 99.0, np.float32)
                            } for _ in range(100)]
      },
      {'testcase_name': 'nonuniform_with_zeros',
       'boundaries': 5,
       'input_data': [{'x': [x]} for x in list(range(25)) + (
           list(range(50, 75)) + list(range(50, 75)) + list(range(75, 100)))],
       'expected_outputs': [{'probs': np.ones((1), np.float32) / 99.0 * (
           2.0 if 24 < i < 75 else 1.0)} for i in range(100)]
      },
      {'testcase_name': 'empty',
       'boundaries': 5,
       'input_data': [],
       'expected_outputs': []
      },
  ], _TFXIO_NAMED_PARAMETERS))
  def testProbUnknownBoundaries(
      self, input_data, expected_outputs, boundaries, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')
    # Test 1 has 100 points over a range of 99; test 2 is an uneven distribution
    def preprocessing_fn(inputs):
      return {'probs': tft.estimated_probability_density(inputs['x'],
                                                         boundaries=boundaries)}

    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([1], tf.int64)
    })

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_outputs,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters([
      dict(
          testcase_name='Int64In',
          input_dtype=tf.int64,
          output_dtypes={
              'min': tf.int64,
              'max': tf.int64,
              'sum': tf.int64,
              'size': tf.int64,
              'mean': tf.float32,
              'var': tf.float32
          }),
      dict(
          testcase_name='Int32In',
          input_dtype=tf.int32,
          output_dtypes={
              'min': tf.int32,
              'max': tf.int32,
              'sum': tf.int64,
              'size': tf.int64,
              'mean': tf.float32,
              'var': tf.float32
          }),
      dict(
          testcase_name='Int16In',
          input_dtype=tf.int16,
          output_dtypes={
              'min': tf.int16,
              'max': tf.int16,
              'sum': tf.int64,
              'size': tf.int64,
              'mean': tf.float32,
              'var': tf.float32
          }),
      dict(
          testcase_name='Float64In',
          input_dtype=tf.float64,
          output_dtypes={
              'min': tf.float64,
              'max': tf.float64,
              'sum': tf.float64,
              'size': tf.int64,
              'mean': tf.float64,
              'var': tf.float64
          }),
      dict(
          testcase_name='Float32In',
          input_dtype=tf.float32,
          output_dtypes={
              'min': tf.float32,
              'max': tf.float32,
              'sum': tf.float32,
              'size': tf.int64,
              'mean': tf.float32,
              'var': tf.float32
          }),
      dict(
          testcase_name='Float16In',
          input_dtype=tf.float16,
          output_dtypes={
              'min': tf.float16,
              'max': tf.float16,
              'sum': tf.float32,
              'size': tf.int64,
              'mean': tf.float16,
              'var': tf.float16
          })
  ], _TFXIO_NAMED_PARAMETERS))
  def testNumericAnalyzersWithScalarInputs(
      self, input_dtype, output_dtypes, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    def analyzer_fn(inputs):
      a = tf.cast(inputs['a'], input_dtype)

      def assert_and_cast_dtype(tensor, out_dtype):
        self.assertEqual(tensor.dtype, out_dtype)
        return tf.cast(tensor, tft_unit.canonical_numeric_dtype(out_dtype))

      return {
          'min': assert_and_cast_dtype(tft.min(a),
                                       output_dtypes['min']),
          'max': assert_and_cast_dtype(tft.max(a),
                                       output_dtypes['max']),
          'sum': assert_and_cast_dtype(tft.sum(a),
                                       output_dtypes['sum']),
          'size': assert_and_cast_dtype(tft.size(a),
                                        output_dtypes['size']),
          'mean': assert_and_cast_dtype(tft.mean(a),
                                        output_dtypes['mean']),
          'var': assert_and_cast_dtype(tft.var(a),
                                       output_dtypes['var']),
      }

    input_data = [{'a': 4}, {'a': 1}]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a':
            tf.io.FixedLenFeature([],
                                  tft_unit.canonical_numeric_dtype(input_dtype))
    })
    expected_outputs = {
        'min':
            np.array(
                1,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['min']).as_numpy_dtype),
        'max':
            np.array(
                4,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['max']).as_numpy_dtype),
        'sum':
            np.array(
                5,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['sum']).as_numpy_dtype),
        'size':
            np.array(
                2,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['size']).as_numpy_dtype),
        'mean':
            np.array(
                2.5,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['mean']).as_numpy_dtype),
        'var':
            np.array(
                2.25,
                tft_unit.canonical_numeric_dtype(
                    output_dtypes['var']).as_numpy_dtype),
    }

    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs,
        use_tfxio=use_tfxio)

  @tft_unit.parameters(*itertools.product([
      tf.int16,
      tf.int32,
      tf.int64,
      tf.float32,
      tf.float64,
      tf.uint8,
      tf.uint16,
  ], (True, False), (True, False)))
  def testNumericAnalyzersWithSparseInputs(self, input_dtype,
                                           reduce_instance_dims, use_tfxio):
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    def analyzer_fn(inputs):
      return {
          'min':
              tft.min(inputs['a'], reduce_instance_dims=reduce_instance_dims),
          'max':
              tft.max(inputs['a'], reduce_instance_dims=reduce_instance_dims),
          'sum':
              tft.sum(inputs['a'], reduce_instance_dims=reduce_instance_dims),
          'size':
              tft.size(inputs['a'], reduce_instance_dims=reduce_instance_dims),
          'mean':
              tft.mean(inputs['a'], reduce_instance_dims=reduce_instance_dims),
          'var':
              tft.var(inputs['a'], reduce_instance_dims=reduce_instance_dims),
      }

    input_val_dtype = input_dtype.as_numpy_dtype
    output_dtype = tft_unit.canonical_numeric_dtype(input_dtype).as_numpy_dtype
    input_data = [
        {'idx': [0, 1], 'val': np.array([0, 1], dtype=input_val_dtype)},
        {'idx': [1, 3], 'val': np.array([2, 3], dtype=input_val_dtype)},
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a':
            tf.io.SparseFeature('idx', 'val',
                                tft_unit.canonical_numeric_dtype(input_dtype),
                                4)
    })
    if reduce_instance_dims:
      expected_outputs = {
          'min': np.array(0., output_dtype),
          'max': np.array(3., output_dtype),
          'sum': np.array(6., output_dtype),
          'size': np.array(4, np.int64),
          'mean': np.array(1.5, np.float32),
          'var': np.array(1.25, np.float32),
      }
    else:
      if input_dtype.is_floating:
        missing_value_max = float('nan')
        missing_value_min = float('nan')
      else:
        missing_value_max = np.iinfo(output_dtype).min
        missing_value_min = np.iinfo(output_dtype).max
      expected_outputs = {
          'min': np.array([0., 1., missing_value_min, 3.], output_dtype),
          'max': np.array([0., 2., missing_value_max, 3.], output_dtype),
          'sum': np.array([0., 3., 0., 3.], output_dtype),
          'size': np.array([1, 2, 0, 1], np.int64),
          'mean': np.array([0., 1.5, float('nan'), 3.], np.float32),
          'var': np.array([0., 0.25, float('nan'), 0.], np.float32),
      }
    self.assertAnalyzerOutputs(input_data, input_metadata, analyzer_fn,
                               expected_outputs, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testNumericAnalyzersWithInputsAndAxis(self, use_tfxio):
    def analyzer_fn(inputs):
      return {
          'min': tft.min(inputs['a'], reduce_instance_dims=False),
          'max': tft.max(inputs['a'], reduce_instance_dims=False),
          'sum': tft.sum(inputs['a'], reduce_instance_dims=False),
          'size': tft.size(inputs['a'], reduce_instance_dims=False),
          'mean': tft.mean(inputs['a'], reduce_instance_dims=False),
          'var': tft.var(inputs['a'], reduce_instance_dims=False),
      }

    input_data = [
        {'a': [8, 9, 3, 4]},
        {'a': [1, 2, 10, 11]}
    ]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([4], tf.int64)})
    expected_outputs = {
        'min': np.array([1, 2, 3, 4], np.int64),
        'max': np.array([8, 9, 10, 11], np.int64),
        'sum': np.array([9, 11, 13, 15], np.int64),
        'size': np.array([2, 2, 2, 2], np.int64),
        'mean': np.array([4.5, 5.5, 6.5, 7.5], np.float32),
        'var': np.array([12.25, 12.25, 12.25, 12.25], np.float32),
    }
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testNumericAnalyzersWithNDInputsAndAxis(self, use_tfxio):
    def analyzer_fn(inputs):
      return {
          'min': tft.min(inputs['a'], reduce_instance_dims=False),
          'max': tft.max(inputs['a'], reduce_instance_dims=False),
          'sum': tft.sum(inputs['a'], reduce_instance_dims=False),
          'size': tft.size(inputs['a'], reduce_instance_dims=False),
          'mean': tft.mean(inputs['a'], reduce_instance_dims=False),
          'var': tft.var(inputs['a'], reduce_instance_dims=False),
      }

    input_data = [
        {'a': [[8, 9], [3, 4]]},
        {'a': [[1, 2], [10, 11]]}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([2, 2], tf.int64)})
    expected_outputs = {
        'min': np.array([[1, 2], [3, 4]], np.int64),
        'max': np.array([[8, 9], [10, 11]], np.int64),
        'sum': np.array([[9, 11], [13, 15]], np.int64),
        'size': np.array([[2, 2], [2, 2]], np.int64),
        'mean': np.array([[4.5, 5.5], [6.5, 7.5]], np.float32),
        'var': np.array([[12.25, 12.25], [12.25, 12.25]], np.float32),
    }
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testNumericAnalyzersWithShape1NDInputsAndAxis(self, use_tfxio):

    def analyzer_fn(inputs):
      return {
          'min': tft.min(inputs['a'], reduce_instance_dims=False),
          'max': tft.max(inputs['a'], reduce_instance_dims=False),
          'sum': tft.sum(inputs['a'], reduce_instance_dims=False),
          'size': tft.size(inputs['a'], reduce_instance_dims=False),
          'mean': tft.mean(inputs['a'], reduce_instance_dims=False),
          'var': tft.var(inputs['a'], reduce_instance_dims=False),
      }

    input_data = [{'a': [[8, 9]]}, {'a': [[1, 2]]}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([1, 2], tf.int64)})
    expected_outputs = {
        'min': np.array([[1, 2]], np.int64),
        'max': np.array([[8, 9]], np.int64),
        'sum': np.array([[9, 11]], np.int64),
        'size': np.array([[2, 2]], np.int64),
        'mean': np.array([[4.5, 5.5]], np.float32),
        'var': np.array([[12.25, 12.25]], np.float32),
    }
    self.assertAnalyzerOutputs(input_data, input_metadata, analyzer_fn,
                               expected_outputs, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testNumericAnalyzersWithNDInputs(self, use_tfxio):
    def analyzer_fn(inputs):
      return {
          'min': tft.min(inputs['a']),
          'max': tft.max(inputs['a']),
          'sum': tft.sum(inputs['a']),
          'size': tft.size(inputs['a']),
          'mean': tft.mean(inputs['a']),
          'var': tft.var(inputs['a']),
      }

    input_data = [
        {'a': [[4, 5], [6, 7]]},
        {'a': [[1, 2], [3, 4]]}
    ]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([2, 2], tf.int64)})
    expected_outputs = {
        'min': np.array(1, np.int64),
        'max': np.array(7, np.int64),
        'sum': np.array(32, np.int64),
        'size': np.array(8, np.int64),
        'mean': np.array(4.0, np.float32),
        'var': np.array(3.5, np.float32),
    }
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testNumericMeanWithSparseTensorReduceFalseOverflow(self, use_tfxio):

    def analyzer_fn(inputs):
      return {'mean': tft.mean(tf.cast(inputs['sparse'], tf.int32), False)}

    input_data = [
        {'idx': [0, 1], 'val': [1, 1]},
        {'idx': [1, 3], 'val': [2147483647, 3]},
    ]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'sparse': tf.io.SparseFeature('idx', 'val', tf.int64, 4)})
    expected_outputs = {
        'mean': np.array([1., 1073741824., float('nan'), 3.], np.float32)
    }
    self.assertAnalyzerOutputs(input_data, input_metadata, analyzer_fn,
                               expected_outputs, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testStringToTFIDF(self, use_tfxio):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.compat.v1.strings.split(inputs['a']))
      out_index, out_values = tft.tfidf(inputs_as_ints, 6)
      return {
          'tf_idf': out_values,
          'index': out_index
      }
    input_data = [{'a': 'hello hello world'},
                  {'a': 'hello goodbye hello world'},
                  {'a': 'I like pie pie pie'}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.string)})

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
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'tf_idf': tf.io.VarLenFeature(tf.float32),
        'index': tf.io.VarLenFeature(tf.int64)
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn,
        expected_transformed_data, expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testTFIDFNoData(self, use_tfxio):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.compat.v1.strings.split(inputs['a']))
      out_index, out_values = tft.tfidf(inputs_as_ints, 6)
      return {
          'tf_idf': out_values,
          'index': out_index
      }
    input_data = [{'a': ''}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.string)})
    expected_transformed_data = [{'tf_idf': [], 'index': []}]
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'tf_idf': tf.io.VarLenFeature(tf.float32),
        'index': tf.io.VarLenFeature(tf.int64)
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_transformed_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testStringToTFIDFEmptyDoc(self, use_tfxio):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.compat.v1.strings.split(inputs['a']))
      out_index, out_values = tft.tfidf(inputs_as_ints, 6)
      return {
          'tf_idf': out_values,
          'index': out_index
      }
    input_data = [{'a': 'hello hello world'},
                  {'a': ''},
                  {'a': 'hello goodbye hello world'},
                  {'a': 'I like pie pie pie'}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.string)})

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
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'tf_idf': tf.io.VarLenFeature(tf.float32),
        'index': tf.io.VarLenFeature(tf.int64)
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn,
        expected_transformed_data, expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testIntToTFIDF(self, use_tfxio):
    def preprocessing_fn(inputs):
      out_index, out_values = tft.tfidf(inputs['a'], 13)
      return {'tf_idf': out_values, 'index': out_index}
    input_data = [{'a': [2, 2, 0]},
                  {'a': [2, 6, 2, 0]},
                  {'a': [8, 10, 12, 12, 12]},
                 ]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.VarLenFeature(tf.int64)})
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
    expected_schema = tft_unit.metadata_from_feature_spec({
        'tf_idf': tf.io.VarLenFeature(tf.float32),
        'index': tf.io.VarLenFeature(tf.int64)
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_schema, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testIntToTFIDFWithoutSmoothing(self, use_tfxio):
    def preprocessing_fn(inputs):
      out_index, out_values = tft.tfidf(inputs['a'], 13, smooth=False)
      return {'tf_idf': out_values, 'index': out_index}
    input_data = [{'a': [2, 2, 0]},
                  {'a': [2, 6, 2, 0]},
                  {'a': [8, 10, 12, 12, 12]},
                 ]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.VarLenFeature(tf.int64)})
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
    expected_schema = tft_unit.metadata_from_feature_spec({
        'tf_idf': tf.io.VarLenFeature(tf.float32),
        'index': tf.io.VarLenFeature(tf.int64)
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_schema, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testTFIDFWithOOV(self, use_tfxio):
    test_vocab_size = 3
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.compat.v1.strings.split(inputs['a']), top_k=test_vocab_size)
      out_index, out_values = tft.tfidf(inputs_as_ints,
                                        test_vocab_size+1)
      return {
          'tf_idf': out_values,
          'index': out_index
      }
    input_data = [{'a': 'hello hello world'},
                  {'a': 'hello goodbye hello world'},
                  {'a': 'I like pie pie pie'}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.string)})

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
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'tf_idf': tf.io.VarLenFeature(tf.float32),
        'index': tf.io.VarLenFeature(tf.int64)
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_transformed_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testTFIDFWithNegatives(self, use_tfxio):
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
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.VarLenFeature(tf.int64)})

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
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'tf_idf': tf.io.VarLenFeature(tf.float32),
        'index': tf.io.VarLenFeature(tf.int64)
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn,
        expected_transformed_data, expected_metadata, use_tfxio=use_tfxio)

  # From testVocabularyAnalyzerEmptyVocab
  _EMPTY_VOCABULARY_PARAMS = tft_unit.cross_named_parameters(
      [
          dict(testcase_name='string',
               x_data=['a', 'b'],
               x_feature_spec=tf.io.FixedLenFeature([], tf.string)),
          dict(testcase_name='int64',
               x_data=[1, 2],
               x_feature_spec=tf.io.FixedLenFeature([], tf.int64)),
      ],
      [
          dict(testcase_name='empty_vocabulary',
               index_data=[-1, -1],
               index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
               index_domain=schema_pb2.IntDomain(min=-1, max=0,
                                                 is_categorical=True),
               frequency_threshold=5),
      ])

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters([
      # NOTE: Since these tests are a refactoring of existing tests, each test
      # case parameter (or parameters where the original test was parameterized
      # or tested multiple calls to tft.compute_and_apply_vocabulary) has a
      # comment indicating the test case that it is based on.  This preserves
      # the ability to track the proveance of the test case parameters in the
      # git history.
      # TODO(KesterTong): Remove these annotations and the above comment.
      # From testVocabularyAnalyzerWithLabelsAndTopK
      dict(testcase_name='string_feature_with_label_top_2',
           x_data=[b'hello', b'hello', b'hello', b'goodbye', b'aaaaa',
                   b'aaaaa', b'goodbye', b'goodbye', b'aaaaa', b'aaaaa',
                   b'goodbye', b'goodbye'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[-1, -1, -1, 0, 1, 1, 0, 0, 0, 1, 1, 0],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=1,
                                             is_categorical=True),
           top_k=2),
      dict(testcase_name='string_feature_with_label_top_1',
           x_data=[b'hello', b'hello', b'hello', b'goodbye', b'aaaaa',
                   b'aaaaa', b'goodbye', b'goodbye', b'aaaaa', b'aaaaa',
                   b'goodbye', b'goodbye'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[-1, -1, -1, 0, -1, -1, 0, 0, 0, -1, -1, 0],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=0,
                                             is_categorical=True),
           top_k=1),
      dict(testcase_name='int_feature_with_label_top_2',
           x_data=[3, 3, 3, 1, 2, 2, 1, 1, 2, 2, 1, 1],
           x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[-1, -1, -1, 0, 1, 1, 0, 0, 0, 1, 1, 0],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=1,
                                             is_categorical=True),
           top_k=2),
      # From testVocabularyAnalyzerWithMultiDimensionalInputs
      dict(testcase_name='varlen_feature',
           x_data=[[b'world', b'hello', b'hello'], [b'hello', b'world', b'foo'],
                   [], [b'hello']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[1, 0, 0], [0, 1, -99], [], [0]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=1,
                                             is_categorical=True),
           default_value=-99,
           top_k=2),
      dict(testcase_name='vector_feature',
           x_data=[[b'world', b'hello', b'hello'], [b'hello', b'world', b'moo'],
                   [b'hello', b'hello', b'foo'], [b'world', b'foo', b'moo']],
           x_feature_spec=tf.io.FixedLenFeature([3], tf.string),
           index_data=[[1, 0, 0], [0, 1, -99], [0, 0, -99], [1, -99, -99]],
           index_feature_spec=tf.io.FixedLenFeature([3], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=1,
                                             is_categorical=True),
           default_value=-99,
           top_k=2),
      dict(testcase_name='varlen_feature_with_labels',
           x_data=[[b'hello', b'world', b'bye', b'moo'],
                   [b'world', b'moo', b'foo'], [b'hello', b'foo', b'moo'],
                   [b'moo']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           label_data=[1, 0, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[[0, -99, 1, -99], [-99, -99, -99], [0, -99, -99], [-99]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=1,
                                             is_categorical=True),
           default_value=-99,
           top_k=2),
      dict(testcase_name='vector_feature_with_labels',
           x_data=[[b'world', b'hello', b'hi'], [b'hello', b'world', b'moo'],
                   [b'hello', b'bye', b'foo'], [b'world', b'foo', b'moo']],
           x_feature_spec=tf.io.FixedLenFeature([3], tf.string),
           label_data=[1, 0, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[[-99, -99, 1], [-99, -99, 0], [-99, -99, -99],
                       [-99, -99, 0]],
           index_feature_spec=tf.io.FixedLenFeature([3], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=1,
                                             is_categorical=True),
           default_value=-99,
           top_k=2),
      dict(testcase_name='varlen_integer_feature_with_labels',
           x_data=[[0, 1, 3, 2], [1, 2, 4], [0, 4, 2], [2]],
           x_feature_spec=tf.io.VarLenFeature(tf.int64),
           label_data=[1, 0, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[[0, -99, 1, -99], [-99, -99, -99], [0, -99, -99], [-99]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=1,
                                             is_categorical=True),
           default_value=-99,
           top_k=2),
      dict(testcase_name='varlen_feature_with_some_empty_feature_values',
           x_data=[[b'world', b'hello', b'hi', b'moo'], [],
                   [b'world', b'hello', b'foo'], []],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           label_data=[1, 0, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[[0, 1, -99, -99], [], [0, 1, -99], []],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=1,
                                             is_categorical=True),
           default_value=-99,
           top_k=2),
      # From testSparseVocabularyWithMultiClassLabels
      dict(testcase_name='varlen_with_multiclass_labels',
           x_data=[[1, 2, 3, 5], [1, 4, 5], [1, 2], [1, 2], [1, 3, 5],
                   [1, 4, 3], [1, 3]],
           x_feature_spec=tf.io.VarLenFeature(tf.int64),
           label_data=[1, 0, 1, 1, 4, 5, 4],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[[-1, 0, 2, 3], [-1, 1, 3], [-1, 0], [-1, 0], [-1, 2, 3],
                       [-1, 1, 2], [-1, 2]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=3,
                                             is_categorical=True),
           top_k=4),
      # From testVocabularyAnalyzerWithLabelsAndWeights
      dict(testcase_name='labels_and_weights',
           x_data=[b'hello', b'hello', b'hello', b'goodbye', b'aaaaa',
                   b'aaaaa', b'goodbye', b'goodbye', b'aaaaa', b'aaaaa',
                   b'goodbye', b'goodbye'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           weight_data=[0.3, 0.4, 0.3, 1.2, 0.6, 0.7, 1.0, 1.0, 0.6, 0.7, 1.0,
                        1.0],
           weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
           index_data=[2, 2, 2, 1, 0, 0, 1, 1, 0, 0, 1, 1],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=2,
                                             is_categorical=True)),
      # From testVocabularyAnalyzerWithWeights
      dict(testcase_name='string_feature_with_weights',
           x_data=[b'hello', b'world', b'goodbye', b'aaaaa', b'aaaaa',
                   b'goodbye'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           weight_data=[1.0, .5, 1.0, .26, .25, 1.5],
           weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
           index_data=[1, 3, 0, 2, 2, 0],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=3,
                                             is_categorical=True)),
      dict(testcase_name='int64_feature_with_weights',
           x_data=[2, 1, 3, 4, 4, 3],
           x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           weight_data=[1.0, .5, 1.0, .26, .25, 1.5],
           weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
           index_data=[1, 3, 0, 2, 2, 0],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=3,
                                             is_categorical=True)),
      # From testVocabularyAnalyzer
      dict(testcase_name='whitespace_newlines_and_empty_strings',
           x_data=[b'hello', b'world', b'hello', b'hello', b'goodbye', b'world',
                   b'aaaaa', b' ', b'', b'\n', b'hi \n ho \n', '\r'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           # The empty string and strings containing newlines map to default
           # value because the vocab cannot contain them.
           index_data=[0, 1, 0, 0, 2, 1, 3, 4, -1, -1, -1, -1],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=4,
                                             is_categorical=True)),
      # From testVocabularyAnalyzerOOV
      dict(testcase_name='whitespace_newlines_and_empty_strings_oov_buckets',
           x_data=[b'hello', b'world', b'hello', b'hello', b'goodbye', b'world',
                   b'aaaaa', b' ', b'', b'\n', b'hi \n ho \n', '\r'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           # The empty string and strings containing newlines map to OOV because
           # the vocab cannot contain them.
           index_data=[0, 1, 0, 0, 2, 1, 3, 4, 5, 5, 5, 5],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=0, max=5,
                                             is_categorical=True),
           num_oov_buckets=1,
           vocab_filename='my_vocab',
           expected_vocab_file_contents={
               'my_vocab': [b'hello', b'world', b'goodbye', b'aaaaa', b' ']
           }),
      # From testVocabularyAnalyzerPositiveNegativeIntegers
      dict(testcase_name='positive_and_negative_integers',
           x_data=[13, 14, 13, 13, 12, 14, 11, 10, 10, -10, -10, -20],
           x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[0, 1, 0, 0, 4, 1, 5, 2, 2, 3, 3, 6],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=6,
                                             is_categorical=True),
           vocab_filename='my_vocab',
           expected_vocab_file_contents={
               'my_vocab': [b'13', b'14', b'10', b'-10', b'12', b'11', b'-20']
           }),
      # From testVocabularyAnalyzerWithNDInputs
      dict(testcase_name='rank_2',
           x_data=[[[b'some', b'say'], [b'the', b'world']],
                   [[b'will', b'end'], [b'in', b'fire']],
                   [[b'some', b'say'], [b'in', b'ice']]],
           x_feature_spec=tf.io.FixedLenFeature([2, 2], tf.string),
           index_data=[[[0, 1], [5, 3]],
                       [[4, 8], [2, 7]],
                       [[0, 1], [2, 6]]],
           index_feature_spec=tf.io.FixedLenFeature([2, 2], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=8,
                                             is_categorical=True)),
      # From testVocabularyAnalyzerWithTopK
      dict(testcase_name='top_k',
           x_data=[[b'hello', b'hello', b'world'],
                   [b'hello', b'goodbye', b'world'],
                   [b'hello', b'goodbye', b'foo']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[0, 0, 1], [0, -99, 1], [0, -99, -99]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=1,
                                             is_categorical=True),
           default_value=-99,
           top_k=2),
      dict(testcase_name='top_k_specified_as_str',
           x_data=[[b'hello', b'hello', b'world'],
                   [b'hello', b'goodbye', b'world'],
                   [b'hello', b'goodbye', b'foo']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[0, 0, 1], [0, -9, 1], [0, -9, -9]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-9, max=1,
                                             is_categorical=True),
           default_value=-9,
           top_k='2'),
      # From testVocabularyAnalyzerWithFrequencyThreshold
      dict(testcase_name='frequency_threshold',
           x_data=[[b'hello', b'hello', b'world'],
                   [b'hello', b'goodbye', b'world'],
                   [b'hello', b'goodbye', b'foo']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[0, 0, 1], [0, 2, 1], [0, 2, -99]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=2,
                                             is_categorical=True),
           default_value=-99,
           frequency_threshold=2),
      dict(testcase_name='frequency_threshold_specified_with_str',
           x_data=[[b'hello', b'hello', b'world'],
                   [b'hello', b'goodbye', b'world'],
                   [b'hello', b'goodbye', b'foo']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[0, 0, 1], [0, 2, 1], [0, 2, -9]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-9, max=2,
                                             is_categorical=True),
           default_value=-9,
           frequency_threshold='2'),
      # From testVocabularyAnalyzerWithFrequencyThresholdTooHigh
      dict(testcase_name='empty_vocabulary_from_high_frequency_threshold',
           x_data=[[b'hello', b'hello', b'world'],
                   [b'hello', b'goodbye', b'world'],
                   [b'hello', b'goodbye', b'foo']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[-99, -99, -99], [-99, -99, -99], [-99, -99, -99]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=0,
                                             is_categorical=True),
           default_value=-99,
           frequency_threshold=77),
      # From testVocabularyAnalyzerWithHighFrequencyThresholdAndOOVBuckets
      dict(testcase_name='top_k_and_oov',
           x_data=[[b'hello', b'hello', b'world', b'world'],
                   [b'hello', b'tarkus', b'toccata'],
                   [b'hello', b'goodbye', b'foo']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           # Generated vocab (ordered by frequency, then value) should be:
           # ["hello", "world", "goodbye", "foo", "tarkus", "toccata"]. After
           # applying top_k =1 this becomes ["hello"] plus three OOV buckets.
           # The specific output values here depend on the hash of the words,
           # and the test will break if the hash changes.
           index_data=[[0, 0, 2, 2], [0, 3, 1], [0, 2, 1]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=0, max=3,
                                             is_categorical=True),
           default_value=-99,
           top_k=1,
           num_oov_buckets=3),
      # From testVocabularyAnalyzerWithKeyFn
      dict(testcase_name='key_fn',
           x_data=[['a_X_1', 'a_X_1', 'a_X_2', 'b_X_1', 'b_X_2'],
                   ['a_X_1', 'a_X_1', 'a_X_2', 'a_X_2'], ['b_X_2']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[0, 0, 1, -99, 2], [0, 0, 1, 1], [2]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=2,
                                             is_categorical=True),
           coverage_top_k=1,
           default_value=-99,
           key_fn=lambda s: s.split(b'_X_')[0],
           frequency_threshold=3),
      # from testVocabularyAnalyzerWithKeyFnAndMultiCoverageTopK
      dict(testcase_name='key_fn_and_multi_coverage_top_k',
           x_data=[['a_X_1', 'a_X_1', 'a_X_2', 'b_X_1', 'b_X_2'],
                   ['a_X_1', 'a_X_1', 'a_X_2', 'a_X_2', 'a_X_3'], ['b_X_2']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[0, 0, 1, 3, 2], [0, 0, 1, 1, -99], [2]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=3,
                                             is_categorical=True),
           coverage_top_k=2,
           default_value=-99,
           key_fn=lambda s: s.split(b'_X_')[0],
           frequency_threshold=300),
      # from testVocabularyAnalyzerWithKeyFnAndTopK
      dict(testcase_name='key_fn_and_top_k',
           x_data=[['a_X_1', 'a_X_1', 'a_X_2', 'b_X_1', 'b_X_2'],
                   ['a_X_1', 'a_X_1', 'a_X_2', 'a_X_2'],
                   ['b_X_2', 'b_X_2', 'b_X_2', 'b_X_2', 'c_X_1']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[1, 1, -99, -99, 0], [1, 1, -99, -99], [0, 0, 0, 0, 2]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=2,
                                             is_categorical=True),
           coverage_top_k=1,
           default_value=-99,
           key_fn=lambda s: s.split(b'_X_')[0],
           top_k=2),
      # from testVocabularyAnalyzerWithKeyFnMultiCoverageTopK
      dict(testcase_name='key_fn_multi_coverage_top_k',
           x_data=[['0_X_a', '0_X_a', '5_X_a', '6_X_a', '6_X_a', '0_X_a'],
                   ['0_X_a', '2_X_a', '2_X_a', '2_X_a', '0_X_a', '5_X_a'],
                   ['1_X_b', '1_X_b', '3_X_b', '3_X_b', '0_X_b', '1_X_b',
                    '1_X_b']],
           x_feature_spec=tf.io.VarLenFeature(tf.string),
           index_data=[[0, 0, -99, -99, -99, 0], [0, 2, 2, 2, 0, -99],
                       [1, 1, 3, 3, -99, 1, 1]],
           index_feature_spec=tf.io.VarLenFeature(tf.int64),
           index_domain=schema_pb2.IntDomain(min=-99, max=3,
                                             is_categorical=True),
           coverage_top_k=2,
           default_value=-99,
           key_fn=lambda s: s.split(b'_X_')[1],
           frequency_threshold=4),
      # from testVocabularyAnalyzerWithKeyFnAndLabels
      dict(testcase_name='key_fn_and_labels',
           x_data=['aaa', 'aaa', 'aaa', 'aab', 'aba', 'aba', 'aab', 'aab',
                   'aba', 'abc', 'abc', 'aab'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_data=[0, 0, 0, -1, -1, -1, -1, -1, -1, 1, 1, -1],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=1,
                                             is_categorical=True),
           coverage_top_k=1,
           key_fn=lambda s: s[:2],
           frequency_threshold=3),
      # from testVocabularyAnalyzerWithKeyFnAndWeights
      dict(testcase_name='key_fn_and_weights',
           x_data=['xa', 'xa', 'xb', 'ya', 'yb', 'yc'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           weight_data=[1.0, 0.5, 3.0, 0.6, 0.25, 0.5],
           weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
           index_data=[1, 1, 0, -1, -1, -1],
           index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           index_domain=schema_pb2.IntDomain(min=-1, max=1,
                                             is_categorical=True),
           coverage_top_k=1,
           key_fn=lambda s: s[0],
           frequency_threshold=1.5,
           coverage_frequency_threshold=1),
  ] + _EMPTY_VOCABULARY_PARAMS, _TFXIO_NAMED_PARAMETERS))
  def testComputeAndApplyVocabulary(
      self, x_data, x_feature_spec, index_data, index_feature_spec,
      index_domain, use_tfxio, label_data=None, label_feature_spec=None,
      weight_data=None, weight_feature_spec=None,
      expected_vocab_file_contents=None, **kwargs):
    """Test tft.compute_and_apply_vocabulary with various inputs."""
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    input_data = [{'x': x} for x in x_data]
    input_feature_spec = {'x': x_feature_spec}
    expected_data = [{'index': index} for index in index_data]
    expected_feature_spec = {'index': index_feature_spec}
    expected_domains = {'index': index_domain}

    if label_data is not None:
      for idx, label in enumerate(label_data):
        input_data[idx]['label'] = label
      input_feature_spec['label'] = label_feature_spec

    if weight_data is not None:
      for idx, weight in enumerate(weight_data):
        input_data[idx]['weights'] = weight
      input_feature_spec['weights'] = weight_feature_spec

    input_metadata = tft_unit.metadata_from_feature_spec(input_feature_spec)
    expected_metadata = tft_unit.metadata_from_feature_spec(
        expected_feature_spec, expected_domains)

    def preprocessing_fn(inputs):
      x = inputs['x']
      labels = inputs.get('label')
      weights = inputs.get('weights')
      index = tft.compute_and_apply_vocabulary(
          x, labels=labels, weights=weights, **kwargs)
      return {'index': index}

    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata,
        expected_vocab_file_contents=expected_vocab_file_contents,
        use_tfxio=use_tfxio)

  # From testVocabularyAnalyzerStringVsIntegerFeature
  _WITH_LABEL_PARAMS = tft_unit.cross_named_parameters(
      [
          dict(testcase_name='string',
               x_data=[
                   b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
                   b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye',
                   b'goodbye'
               ],
               x_feature_spec=tf.io.FixedLenFeature([], tf.string),
               expected_vocab_file_contents=[(b'goodbye', 1.975322),
                                             (b'aaaaa', 1.6600708),
                                             (b'hello', 1.2450531)]),
          dict(testcase_name='int64',
               x_data=[3, 3, 3, 1, 2, 2, 1, 1, 2, 2, 1, 1],
               x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
               expected_vocab_file_contents=[(b'1', 1.975322),
                                             (b'2', 1.6600708),
                                             (b'3', 1.2450531)]),
      ],
      [
          dict(testcase_name='with_label',
               label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
               label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
               min_diff_from_avg=0.0,
               store_frequency=True),
      ])

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters([
      # NOTE: Since these tests are a refactoring of existing tests, each test
      # case parameter (or parameters where the original test was parameterized
      # or tested multiple calls to tft.vocabulary) has a comment indicating the
      # test case that it is based on.  This preserves the ability to track the
      # proveance of the test case parameters in the git history.
      # TODO(KesterTong): Remove these annotations and the above comment.
      # From testVocabularyWithMutualInformation
      dict(testcase_name='unadjusted_mi_binary_label',
           x_data=[
               b'informative', b'informative', b'informative', b'uninformative',
               b'uninformative', b'uninformative', b'uninformative',
               b'uninformative_rare', b'uninformative_rare'
           ],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 1, 0, 1, 1, 0, 0, 1],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           expected_vocab_file_contents=[
               (b'informative', 1.7548264),
               (b'uninformative', 0.33985),
               (b'uninformative_rare', 0.169925),
           ],
           min_diff_from_avg=0.0,
           use_adjusted_mutual_info=False,
           store_frequency=True),
      dict(testcase_name='unadjusted_mi_multi_class_label',
           x_data=[
               b'good_predictor_of_0', b'good_predictor_of_0',
               b'good_predictor_of_0', b'good_predictor_of_1',
               b'good_predictor_of_2', b'good_predictor_of_2',
               b'good_predictor_of_2', b'good_predictor_of_1',
               b'good_predictor_of_1', b'weak_predictor_of_1',
               b'good_predictor_of_0', b'good_predictor_of_1',
               b'good_predictor_of_1', b'good_predictor_of_1',
               b'weak_predictor_of_1'
           ],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[0, 0, 0, 1, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           expected_vocab_file_contents=[
               (b'good_predictor_of_2', 6.9656615),
               (b'good_predictor_of_1', 6.5969831),
               (b'good_predictor_of_0', 6.3396921),
               (b'weak_predictor_of_1', 0.684463),
           ],
           min_diff_from_avg=0.0,
           use_adjusted_mutual_info=False,
           store_frequency=True),
      dict(testcase_name='unadjusted_mi_binary_label_with_weights',
           x_data=[
               b'informative_1', b'informative_1', b'informative_0',
               b'informative_0', b'uninformative', b'uninformative',
               b'informative_by_weight', b'informative_by_weight'
           ],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 0, 0, 0, 1, 0, 1],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           # uninformative and informative_by_weight have the same co-occurrence
           # relationship with the label but will have different importance
           # values due to the weighting.
           expected_vocab_file_contents=[
               (b'informative_0', 3.1698803),
               (b'informative_1', 1.1698843),
               (b'informative_by_weight', 0.6096405),
               (b'uninformative', 0.169925),
           ],
           weight_data=[1, 1, 1, 1, 1, 1, 1, 5],
           weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
           min_diff_from_avg=0.0,
           use_adjusted_mutual_info=False,
           store_frequency=True),
      dict(testcase_name='unadjusted_mi_binary_label_min_diff_from_avg',
           x_data=[
               b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
               b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye',
               b'goodbye'
           ],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           # All features are weak predictors, so all are adjusted to zero.
           expected_vocab_file_contents=[
               (b'hello', 0.0),
               (b'goodbye', 0.0),
               (b'aaaaa', 0.0),
           ],
           use_adjusted_mutual_info=False,
           min_diff_from_avg=2.0,
           store_frequency=True),
      dict(testcase_name='adjusted_mi_binary_label',
           x_data=[
               b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
               b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye',
               b'goodbye'
           ],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           expected_vocab_file_contents=[
               (b'goodbye', 1.4070791),
               (b'aaaaa', 0.9987449),
               (b'hello', 0.5017179),
           ],
           min_diff_from_avg=0.0,
           use_adjusted_mutual_info=True,
           store_frequency=True),
      dict(testcase_name='adjusted_mi_binary_label_int64_feature',
           x_data=[3, 3, 3, 1, 2, 2, 1, 1, 2, 2, 1, 1],
           x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           expected_vocab_file_contents=[
               (b'1', 1.4070791),
               (b'2', 0.9987449),
               (b'3', 0.5017179),
           ],
           min_diff_from_avg=0.0,
           use_adjusted_mutual_info=True,
           store_frequency=True),
      dict(testcase_name='adjusted_mi_multi_class_label',
           x_data=[
               b'good_predictor_of_0', b'good_predictor_of_0',
               b'good_predictor_of_0', b'good_predictor_of_1',
               b'good_predictor_of_2', b'good_predictor_of_2',
               b'good_predictor_of_2', b'good_predictor_of_1',
               b'good_predictor_of_1', b'weak_predictor_of_1',
               b'good_predictor_of_0', b'good_predictor_of_1',
               b'good_predictor_of_1', b'good_predictor_of_1',
               b'weak_predictor_of_1'
           ],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[0, 0, 0, 1, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           expected_vocab_file_contents=[
               (b'good_predictor_of_1', 5.4800903),
               (b'good_predictor_of_2', 5.386102),
               (b'good_predictor_of_0', 4.9054723),
               (b'weak_predictor_of_1', -0.9748023),
           ],
           min_diff_from_avg=0.0,
           use_adjusted_mutual_info=True,
           store_frequency=True),
      # TODO(b/128831096): Determine correct interaction between AMI and weights
      dict(testcase_name='adjusted_mi_binary_label_with_weights',
           x_data=[
               b'informative_1', b'informative_1', b'informative_0',
               b'informative_0', b'uninformative', b'uninformative',
               b'informative_by_weight', b'informative_by_weight'
           ],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 0, 0, 0, 1, 0, 1],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           weight_data=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
           weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
           # uninformative and informative_by_weight have the same co-occurrence
           # relationship with the label but will have different importance
           # values due to the weighting.
           expected_vocab_file_contents=[
               (b'informative_0', 2.3029856),
               (b'informative_1', 0.3029896),
               (b'informative_by_weight', 0.1713041),
               (b'uninformative', -0.6969697),
           ],
           min_diff_from_avg=0.0,
           use_adjusted_mutual_info=True,
           store_frequency=True),
      dict(testcase_name='adjusted_mi_min_diff_from_avg',
           x_data=[
               b'good_predictor_of_0', b'good_predictor_of_0',
               b'good_predictor_of_0', b'good_predictor_of_1',
               b'good_predictor_of_0', b'good_predictor_of_1',
               b'good_predictor_of_1', b'good_predictor_of_1',
               b'good_predictor_of_1', b'good_predictor_of_0',
               b'good_predictor_of_1', b'good_predictor_of_1',
               b'good_predictor_of_1', b'weak_predictor_of_1',
               b'weak_predictor_of_1'
           ],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           # With min_diff_from_avg, the small AMI value is regularized to 0
           expected_vocab_file_contents=[
               (b'good_predictor_of_0', 1.8322128),
               (b'good_predictor_of_1', 1.7554416),
               (b'weak_predictor_of_1', 0),
           ],
           use_adjusted_mutual_info=True,
           min_diff_from_avg=1.0,
           store_frequency=True),
      # From testVocabularyAnalyzerWithLabelsWeightsAndFrequency
      dict(testcase_name='labels_weight_and_frequency',
           x_data=[
               b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
               b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye',
               b'goodbye'
           ],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
           label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           weight_data=[
               0.3, 0.4, 0.3, 1.2, 0.6, 0.7, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0
           ],
           weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
           expected_vocab_file_contents=[
               (b'aaaaa', 1.5637185),
               (b'goodbye', 0.8699492),
               (b'hello', 0.6014302),
           ],
           min_diff_from_avg=0.0,
           store_frequency=True),
      # From testVocabularyWithFrequencyAndFingerprintShuffle
      # fingerprints by which each of the tokens will be sorted if fingerprint
      # shuffling is used.
      # 'ho ho': '1b3dd735ddff70d90f3b7ba5ebf65df521d6ca4d'
      # 'world': '7c211433f02071597741e6ff5a8ea34789abbf43'
      # 'hello': 'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d'
      # 'hi': 'c22b5f9178342609428d6f51b2c5af4c0bde6a42'
      # '1': '356a192b7913b04c54574d18c28d46e6395428ab'
      # '2': 'da4b9237bacccdf19c0760cab7aec4a8359010b0'
      # '3': '77de68daecd823babbb58edb1c8e14d7106e83bb'
      dict(testcase_name='string_feature_with_frequency_and_shuffle',
           x_data=[b'world', b'hello', b'hello'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           expected_vocab_file_contents=[(b'world', 1), (b'hello', 2)],
           fingerprint_shuffle=True,
           store_frequency=True),
      dict(testcase_name='string_feature_with_frequency_and_no_shuffle',
           x_data=[b'hi', b'ho ho', b'ho ho'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           expected_vocab_file_contents=[(b'ho ho', 2), (b'hi', 1)],
           store_frequency=True),
      dict(testcase_name='string_feature_with_no_frequency_and_shuffle',
           x_data=[b'world', b'hello', b'hello'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           expected_vocab_file_contents=[b'world', b'hello'],
           fingerprint_shuffle=True),
      dict(testcase_name='string_feature_with_no_frequency_and_no_shuffle',
           x_data=[b'world', b'hello', b'hello'],
           x_feature_spec=tf.io.FixedLenFeature([], tf.string),
           expected_vocab_file_contents=[b'hello', b'world']),
      dict(testcase_name='int_feature_with_frequency_and_shuffle',
           x_data=[1, 2, 2, 3],
           x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           expected_vocab_file_contents=[(b'1', 1), (b'3', 1), (b'2', 2)],
           fingerprint_shuffle=True,
           store_frequency=True),
      dict(testcase_name='int_feature_with_frequency_and_no_shuffle',
           x_data=[2, 1, 1, 1],
           x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           expected_vocab_file_contents=[(b'1', 3), (b'2', 1)],
           store_frequency=True),
      dict(testcase_name='int_feature_with_no_frequency_and_shuffle',
           x_data=[1, 2, 2, 3],
           x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           expected_vocab_file_contents=[b'1', b'3', b'2'],
           fingerprint_shuffle=True),
      dict(testcase_name='int_feature_with_no_frequency_and_no_shuffle',
           x_data=[1, 2, 2, 3],
           x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
           expected_vocab_file_contents=[b'2', b'3', b'1']),
  ] + _WITH_LABEL_PARAMS, _TFXIO_NAMED_PARAMETERS))
  def testVocabulary(
      self, x_data, x_feature_spec, use_tfxio, label_data=None,
      label_feature_spec=None, weight_data=None, weight_feature_spec=None,
      expected_vocab_file_contents=None, **kwargs):
    """Test tft.Vocabulary with various inputs."""
    self._SkipIfExternalEnvironmentAnd(
        use_tfxio, 'Skipping large test cases; b/147698868')

    input_data = [{'x': x} for x in x_data]
    input_feature_spec = {'x': x_feature_spec}

    if label_data is not None:
      for idx, label in enumerate(label_data):
        input_data[idx]['label'] = label
      input_feature_spec['label'] = label_feature_spec

    if weight_data is not None:
      for idx, weight in enumerate(weight_data):
        input_data[idx]['weights'] = weight
      input_feature_spec['weights'] = weight_feature_spec

    input_metadata = tft_unit.metadata_from_feature_spec(input_feature_spec)

    def preprocessing_fn(inputs):
      x = inputs['x']
      labels = inputs.get('label')
      weights = inputs.get('weights')
      # Note even though the return value is not used, calling tft.vocabulary
      # will generate the vocabulary as a side effect, and since we have named
      # this vocabulary it can be looked up using public APIs.
      tft.vocabulary(
          x, labels=labels, weights=weights, vocab_filename='my_vocab',
          **kwargs)
      return inputs

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        input_data,  # expected output data is same as input data
        input_metadata,  # expected output metadata is ame as input metadata
        expected_vocab_file_contents={
            'my_vocab': expected_vocab_file_contents},
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testJointVocabularyForMultipleFeatures(self, use_tfxio):
    input_data = [
        {'a': 'hello', 'b': 'world', 'c': 'aaaaa'},
        {'a': 'good', 'b': '', 'c': 'hello'},
        {'a': 'goodbye', 'b': 'hello', 'c': '\n'},
        {'a': ' ', 'b': 'aaaaa', 'c': 'bbbbb'}
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.string),
        'b': tf.io.FixedLenFeature([], tf.string),
        'c': tf.io.FixedLenFeature([], tf.string)
    })
    vocab_filename = 'test_compute_and_apply_vocabulary'
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'index_a': tf.io.FixedLenFeature([], tf.int64),
        'index_b': tf.io.FixedLenFeature([], tf.int64),
    }, {
        'index_a': schema_pb2.IntDomain(min=-1, max=6, is_categorical=True),
        'index_b': schema_pb2.IntDomain(min=-1, max=6, is_categorical=True),
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
        expected_metadata, use_tfxio=use_tfxio)

  # Example on how to use the vocab frequency as part of the transform
  # function.
  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testCreateVocabWithFrequency(self, use_tfxio):
    input_data = [
        {'a': 'hello', 'b': 'world', 'c': 'aaaaa'},
        {'a': 'good', 'b': '', 'c': 'hello'},
        {'a': 'goodbye', 'b': 'hello', 'c': '\n'},
        {'a': '_', 'b': 'aaaaa', 'c': 'bbbbb'}
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.string),
        'b': tf.io.FixedLenFeature([], tf.string),
        'c': tf.io.FixedLenFeature([], tf.string)
    })
    vocab_filename = 'test_vocab_with_frequency'
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'index_a': tf.io.FixedLenFeature([], tf.int64),
        'index_b': tf.io.FixedLenFeature([], tf.int64),
        'frequency_a': tf.io.FixedLenFeature([], tf.int64),
        'frequency_b': tf.io.FixedLenFeature([], tf.int64),
    }, {
        'index_a': schema_pb2.IntDomain(min=-1, max=6, is_categorical=True),
        'index_b': schema_pb2.IntDomain(min=-1, max=6, is_categorical=True),
        'frequency_a': schema_pb2.IntDomain(min=-1, max=6, is_categorical=True),
        'frequency_b': schema_pb2.IntDomain(min=-1, max=6, is_categorical=True),
    })

    def preprocessing_fn(inputs):
      deferred_vocab_and_filename = tft.vocabulary(
          tf.concat([inputs['a'], inputs['b'], inputs['c']], 0),
          vocab_filename=vocab_filename,
          store_frequency=True)

      def _apply_vocab(y, deferred_vocab_filename_tensor):
        # NOTE: Please be aware that TextFileInitializer assigns a special
        # meaning to the constant lookup_ops.TextFileIndex.LINE_NUMBER.
        table = lookup_ops.HashTable(
            lookup_ops.TextFileInitializer(
                deferred_vocab_filename_tensor,
                tf.string,
                1,
                tf.int64,
                lookup_ops.TextFileIndex.LINE_NUMBER,
                delimiter=' '),
            default_value=-1)
        table_size = table.size()
        return table.lookup(y), table_size

      def _apply_frequency(y, deferred_vocab_filename_tensor):
        table = lookup_ops.HashTable(
            lookup_ops.TextFileInitializer(
                deferred_vocab_filename_tensor,
                tf.string,
                1,
                tf.int64,
                0,
                delimiter=' '),
            default_value=-1)
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
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testVocabularyAnalyzerWithTokenization(self, use_tfxio):
    def preprocessing_fn(inputs):
      return {
          'index':
              tft.compute_and_apply_vocabulary(
                  tf.compat.v1.strings.split(inputs['a']))
      }

    input_data = [{'a': 'hello hello world'}, {'a': 'hello goodbye world'}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.string)})
    expected_data = [{'index': [0, 0, 1]}, {'index': [0, 2, 1]}]

    expected_metadata = tft_unit.metadata_from_feature_spec({
        'index': tf.io.VarLenFeature(tf.int64),
    }, {
        'index': schema_pb2.IntDomain(min=-1, max=2, is_categorical=True),
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testPipelineWithoutAutomaterialization(self, use_tfxio):
    # Other tests pass lists instead of PCollections and thus invoke
    # automaterialization where each call to a beam PTransform will implicitly
    # run its own pipeline.
    #
    # In order to test the case where PCollections are not materialized in
    # between calls to the tf.Transform PTransforms, we include a test that is
    # not based on automaterialization.
    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    def equal_to(expected):

      def _equal(actual):
        dict_key_fn = lambda d: sorted(d.items())
        sorted_expected = sorted(expected, key=dict_key_fn)
        sorted_actual = sorted(actual, key=dict_key_fn)
        if sorted_expected != sorted_actual:
          raise ValueError('Failed assert: %s == %s' % (expected, actual))
      return _equal

    with self._makeTestPipeline() as pipeline:
      input_data = pipeline | 'CreateTrainingData' >> beam.Create(
          [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}])
      metadata = tft_unit.metadata_from_feature_spec(
          {'x': tf.io.FixedLenFeature([], tf.float32)})
      with beam_impl.Context(temp_dir=self.get_temp_dir(), use_tfxio=use_tfxio):
        if use_tfxio:
          legacy_metadata = metadata
          input_data, metadata = self.convert_to_tfxio_api_inputs(
              input_data, metadata, 'input_data')
        transform_fn = (
            (input_data, metadata)
            | 'AnalyzeDataset' >> beam_impl.AnalyzeDataset(preprocessing_fn))

        # Run transform_columns on some eval dataset.
        eval_data = pipeline | 'CreateEvalData' >> beam.Create(
            [{'x': 6}, {'x': 3}])
        if use_tfxio:
          eval_data, _ = self.convert_to_tfxio_api_inputs(
              eval_data, legacy_metadata, 'eval_data')
        transformed_eval_data, _ = (
            ((eval_data, metadata), transform_fn)
            | 'TransformDataset' >> beam_impl.TransformDataset())
        expected_data = [{'x_scaled': 1.25}, {'x_scaled': 0.5}]
        beam_test_util.assert_that(
            transformed_eval_data, equal_to(expected_data))

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

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testVocabularyWithFrequency(self, use_tfxio):
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
      with tf.io.gfile.GFile(assets_file, 'r') as f:
        contents = f.read()

      self.assertMultiLineEqual(expected, contents)

    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.string),
        'b': tf.io.FixedLenFeature([], tf.string)
    })

    tft_tmp_dir = os.path.join(self.get_temp_dir(), 'temp_dir')
    transform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')

    with beam_impl.Context(temp_dir=tft_tmp_dir, use_tfxio=use_tfxio):
      with self._makeTestPipeline() as pipeline:
        input_data = pipeline | beam.Create([
            {'a': 'hello', 'b': 'hi'},
            {'a': 'world', 'b': 'ho ho'},
            {'a': 'hello', 'b': 'ho ho'},
        ])
        if use_tfxio:
          input_data, input_metadata = self.convert_to_tfxio_api_inputs(
              input_data, input_metadata)
        transform_fn = (
            (input_data, input_metadata)
            | beam_impl.AnalyzeDataset(preprocessing_fn))
        _ = transform_fn | transform_fn_io.WriteTransformFn(transform_fn_dir)

    self.assertTrue(os.path.isdir(tft_tmp_dir))

    saved_model_path = os.path.join(transform_fn_dir,
                                    tft.TFTransformOutput.TRANSFORM_FN_DIR)
    assets_path = os.path.join(saved_model_path,
                               tf.saved_model.ASSETS_DIRECTORY)
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

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testCovarianceTwoDimensions(self, use_tfxio):
    def analyzer_fn(inputs):
      return {'y': tft.covariance(inputs['x'], dtype=tf.float32)}

    input_data = [{'x': x} for x in [[0, 0], [4, 0], [2, -2], [2, 2]]]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([2], tf.float32)})
    expected_outputs = {'y': np.array([[2, 0], [0, 2]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testCovarianceOneDimension(self, use_tfxio):
    def analyzer_fn(inputs):
      return {'y': tft.covariance(inputs['x'], dtype=tf.float32)}

    input_data = [{'x': x} for x in [[0], [2], [4], [6]]]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([1], tf.float32)})
    expected_outputs = {'y': np.array([[5]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs,
        use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testPCAThreeToTwoDimensions(self, use_tfxio):
    def analyzer_fn(inputs):
      return {'y': tft.pca(inputs['x'], 2, dtype=tf.float32)}

    input_data = [{'x': x}
                  for x in  [[0, 0, 1], [4, 0, 1], [2, -1, 1], [2, 1, 1]]]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([3], tf.float32)})
    expected_outputs = {'y': np.array([[1, 0], [0, 1], [0, 0]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs,
        use_tfxio=use_tfxio)

  class _SumCombiner(beam.PTransform):

    @staticmethod
    def _flatten_fn(batch_values):
      for value in zip(*batch_values):
        yield value

    @staticmethod
    def _sum_fn(values):
      return np.sum(list(values), axis=0)

    @staticmethod
    def _extract_outputs(sums):
      return [beam.pvalue.TaggedOutput('0', sums[0]),
              beam.pvalue.TaggedOutput('1', sums[1])]

    def expand(self, pcoll):
      output_tuple = (
          pcoll
          | beam.FlatMap(self._flatten_fn)
          | beam.CombineGlobally(self._sum_fn)
          | beam.FlatMap(self._extract_outputs).with_outputs('0', '1'))
      return (output_tuple['0'], output_tuple['1'])

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testPTransformAnalyzer(self, use_tfxio):

    def analyzer_fn(inputs):
      outputs = analyzers.ptransform_analyzer([inputs['x'], inputs['y']],
                                              [tf.int64, tf.int64],
                                              [[], []],
                                              self._SumCombiner())
      return {'x_sum': outputs[0], 'y_sum': outputs[1]}

    # NOTE: We force 10 batches: data has 100 elements and we request a batch
    # size of 10.
    input_data = [{'x': 1, 'y': i} for i in range(100)]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.int64),
        'y': tf.io.FixedLenFeature([], tf.int64)
    })
    expected_outputs = {
        'x_sum': np.array(100, np.int64),
        'y_sum': np.array(4950, np.int64)
    }
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=10, use_tfxio=use_tfxio)

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testVocabularyWithKeyFnAndFrequency(self, use_tfxio):
    def key_fn(string):
      return string.split(b'_X_')[1]

    outfile = 'vocabulary_with_frequency'

    def preprocessing_fn(inputs):

      # Force the analyzer to be executed, and store the frequency file as a
      # side-effect.

      _ = tft.vocabulary(
          tf.compat.v1.strings.split(inputs['a']),
          coverage_top_k=1,
          key_fn=key_fn,
          frequency_threshold=4,
          vocab_filename=outfile,
          store_frequency=True)

      _ = tft.vocabulary(
          tf.compat.v1.strings.split(inputs['a']),
          coverage_top_k=1,
          key_fn=key_fn,
          frequency_threshold=4,
          store_frequency=True)

      a_int = tft.compute_and_apply_vocabulary(
          tf.compat.v1.strings.split(inputs['a']),
          coverage_top_k=1,
          key_fn=key_fn,
          frequency_threshold=4)

      # Return input unchanged, this preprocessing_fn is a no-op except for
      # computing uniques.
      return {'a_int': a_int}

    def check_asset_file_contents(assets_path, filename, expected):
      assets_file = os.path.join(assets_path, filename)
      with tf.io.gfile.GFile(assets_file, 'r') as f:
        contents = f.read()

      self.assertMultiLineEqual(expected, contents)

    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.string)})

    tft_tmp_dir = os.path.join(self.get_temp_dir(), 'temp_dir')
    transform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')

    with beam_impl.Context(temp_dir=tft_tmp_dir, use_tfxio=use_tfxio):
      with self._makeTestPipeline() as pipeline:
        input_data = pipeline | beam.Create([
            {'a': '1_X_a 1_X_a 2_X_a 1_X_b 2_X_b'},
            {'a': '1_X_a 1_X_a 2_X_a 2_X_a'},
            {'a': '2_X_b 3_X_c 4_X_c'}
        ])
        if use_tfxio:
          input_data, input_metadata = self.convert_to_tfxio_api_inputs(
              input_data, input_metadata)
        transform_fn = (
            (input_data, input_metadata)
            | beam_impl.AnalyzeDataset(preprocessing_fn))
        _ = transform_fn | transform_fn_io.WriteTransformFn(transform_fn_dir)

    self.assertTrue(os.path.isdir(tft_tmp_dir))

    saved_model_path = os.path.join(transform_fn_dir,
                                    tft.TFTransformOutput.TRANSFORM_FN_DIR)
    assets_path = os.path.join(saved_model_path,
                               tf.saved_model.ASSETS_DIRECTORY)
    self.assertTrue(os.path.isdir(assets_path))

    check_asset_file_contents(assets_path, outfile,
                              '4 1_X_a\n2 2_X_b\n1 4_X_c\n')

  @unittest.skipIf(not common.IS_ANNOTATIONS_PB_AVAILABLE,
                     'Schema annotations are not available')
  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testSavedModelWithAnnotations(self, use_tfxio):
    """Test serialization/deserialization as a saved model with annotations."""
    def preprocessing_fn(inputs):
      # Bucketization applies annotations to the output schema
      return {
          'x_bucketized': tft.bucketize(inputs['x'], num_buckets=4),
          'y_vocab': tft.compute_and_apply_vocabulary(inputs['y']),
      }

    input_data = [{
        'x': 1,
        'y': 'foo',
    }, {
        'x': 2,
        'y': 'bar',
    }, {
        'x': 3,
        'y': 'foo',
    }, {
        'x': 4,
        'y': 'foo',
    }]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.string),
    })
    temp_dir = self.get_temp_dir()
    # Force a batch size of 1 to ensure that occurences are correctly aggregated
    # across batches when computing the total vocabulary size.
    with beam_impl.Context(temp_dir=temp_dir, desired_batch_size=1,
                           use_tfxio=use_tfxio):
      if use_tfxio:
        input_data, input_metadata = self.convert_to_tfxio_api_inputs(
            input_data, input_metadata)
      transform_fn = ((input_data, input_metadata)
                      | beam_impl.AnalyzeDataset(preprocessing_fn))
      #  Write transform_fn to serialize annotation collections to SavedModel
      _ = transform_fn | transform_fn_io.WriteTransformFn(temp_dir)

    # Ensure that the annotations survive the round trip to SavedModel.
    tf_transform_output = tft.TFTransformOutput(temp_dir)
    savedmodel_dir = tf_transform_output.transform_savedmodel_dir
    schema = beam_impl._infer_metadata_from_saved_model(savedmodel_dir)._schema
    self.assertLen(schema.feature, 2)
    for feature in schema.feature:
      if feature.name == 'x_bucketized':
        self.assertLen(feature.annotation.extra_metadata, 1)
        for annotation in feature.annotation.extra_metadata:
          message = annotations_pb2.BucketBoundaries()
          annotation.Unpack(message)
          self.assertAllClose(list(message.boundaries), [2, 3, 4])
      elif feature.name == 'y_vocab':
        self.assertLen(feature.annotation.extra_metadata, 0)
      else:
        raise ValueError('Unexpected feature with metadata: {}'.format(
            feature.name))
    # Vocabularies create a top-level schema annotation for each vocab file.
    self.assertLen(schema.annotation.extra_metadata, 1)
    message = annotations_pb2.VocabularyMetadata()
    annotation = schema.annotation.extra_metadata[0]
    annotation.Unpack(message)
    self.assertEqual(message.unfiltered_vocabulary_size, 2)

  @unittest.skipIf(not common.IS_ANNOTATIONS_PB_AVAILABLE,
                     'Schema annotations are not available')
  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testSavedModelWithGlobalAnnotations(self, use_tfxio):
    def preprocessing_fn(inputs):
      # Add some arbitrary annotation data at the global schema level.
      boundaries = tf.constant([[1.0]])
      message_type = annotations_pb2.BucketBoundaries.DESCRIPTOR.full_name
      sizes = tf.expand_dims([tf.size(boundaries)], axis=0)
      message_proto = tf.raw_ops.EncodeProto(
          sizes=sizes, values=[tf.cast(boundaries, tf.float32)],
          field_names=['boundaries'], message_type=message_type)[0]
      type_url = os.path.join('type.googleapis.com', message_type)
      schema_inference.annotate(type_url, message_proto)
      return {
          'x_scaled': tft.scale_by_min_max(inputs['x']),
      }

    input_data = [{'x': 1}, {'x': 2}, {'x': 3}, {'x': 4}]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
    })
    temp_dir = self.get_temp_dir()
    with beam_impl.Context(temp_dir=temp_dir):
      transform_fn = ((input_data, input_metadata)
                      | beam_impl.AnalyzeDataset(preprocessing_fn))
      #  Write transform_fn to serialize annotation collections to SavedModel
      _ = transform_fn | transform_fn_io.WriteTransformFn(temp_dir)

    # Ensure that global annotations survive the round trip to SavedModel.
    tf_transform_output = tft.TFTransformOutput(temp_dir)
    savedmodel_dir = tf_transform_output.transform_savedmodel_dir
    schema = beam_impl._infer_metadata_from_saved_model(savedmodel_dir)._schema
    self.assertLen(schema.annotation.extra_metadata, 1)
    for annotation in schema.annotation.extra_metadata:
      message = annotations_pb2.BucketBoundaries()
      annotation.Unpack(message)
      self.assertAllClose(list(message.boundaries), [1])

  @tft_unit.named_parameters(*_TFXIO_NAMED_PARAMETERS)
  def testPipelineAPICounters(self, use_tfxio):

    def preprocessing_fn(inputs):
      _ = tft.vocabulary(inputs['a'])
      return {
          'a_int': tft.compute_and_apply_vocabulary(inputs['a']),
          'x_scaled': tft.scale_to_0_1(inputs['x']),
          'y_scaled': tft.scale_to_0_1(inputs['y'])
      }

    with self._makeTestPipeline() as pipeline:
      input_data = pipeline | 'CreateTrainingData' >> beam.Create([{
          'x': 4,
          'y': 5,
          'a': 'hello'
      }, {
          'x': 1,
          'y': 3,
          'a': 'world'
      }])
      metadata = tft_unit.metadata_from_feature_spec({
          'x': tf.io.FixedLenFeature([], tf.float32),
          'y': tf.io.FixedLenFeature([], tf.float32),
          'a': tf.io.FixedLenFeature([], tf.string)
      })
      with beam_impl.Context(temp_dir=self.get_temp_dir(), use_tfxio=use_tfxio):
        if use_tfxio:
          input_data, metadata = self.convert_to_tfxio_api_inputs(
              input_data, metadata)
        _ = ((input_data, metadata)
             | 'AnalyzeDataset' >> beam_impl.AnalyzeDataset(preprocessing_fn))

    metrics = pipeline.metrics
    self.assertMetricsCounterEqual(metrics, 'tft_analyzer_vocabulary', 1)
    self.assertMetricsCounterEqual(metrics, 'tft_mapper_scale_to_0_1', 2)
    self.assertMetricsCounterEqual(metrics,
                                   'tft_mapper_compute_and_apply_vocabulary', 1)
    # compute_and_apply_vocabulary implicitly calls apply_vocabulary.
    # We check that that call is not logged.
    self.assertMetricsCounterEqual(metrics, 'tft_mapper_apply_vocabulary', 0)

  def testHandleBatchError(self):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
    })
    pipeline = self._makeTestPipeline()
    input_data = pipeline | 'CreateTrainingData' >> beam.Create([{
        'x': 1
    }, {
        'x': [4, 1]
    }])
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      _ = ((input_data, metadata)
           | 'AnalyzeDataset' >> beam_impl.AnalyzeDataset(preprocessing_fn))
    # Exception type depends on the running being used.
    with self.assertRaisesRegexp(
        (RuntimeError, ValueError),
        'An error occured while trying to apply the transformation:'):
      pipeline.run()


if __name__ == '__main__':
  tft_unit.main()
