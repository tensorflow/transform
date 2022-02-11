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

import contextlib
import itertools
import math
import os
from typing import Tuple

import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np
import pyarrow as pa
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import pretrained_models
from tensorflow_transform import schema_inference
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tfx_bsl.coders import example_coder
from tfx_bsl.tfxio import tensor_adapter

from google.protobuf import text_format
import unittest
from tensorflow_metadata.proto.v0 import schema_pb2

if common.IS_ANNOTATIONS_PB_AVAILABLE:
  from tensorflow_transform import annotations_pb2  # pylint: disable=g-import-not-at-top


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
    dict(testcase_name='zero_variance',
         input_data=np.array([[3], [3], [3], [3]], np.float32),
         output_data=np.array([[0], [0], [0], [0]], np.float32),
         elementwise=False),
    dict(testcase_name='zero_variance_elementwise',
         input_data=np.array([[3, 4], [3, 4]], np.float32),
         output_data=np.array([[0, 0], [0, 0]], np.float32),
         elementwise=True),
]

_SCALE_TO_Z_SCORE_NAN_TEST_CASES = [
    dict(
        testcase_name='with_nans',
        input_data=np.array([[1], [np.nan], [np.nan], [2]], np.float32),
        output_data=np.array([[-1.0], [np.nan], [np.nan], [1.0]], np.float32),
        elementwise=False),
    dict(
        testcase_name='with_nans_elementwise',
        input_data=np.array([[1, np.nan], [np.nan, 2]], np.float32),
        output_data=np.array([[0, np.nan], [np.nan, 0]], np.float32),
        elementwise=True),
]


def _sigmoid(x):
  return 1 / (1 + np.exp(-x))


def sum_output_dtype(input_dtype):
  """Returns the output dtype for tft.sum."""
  return input_dtype if input_dtype.is_floating else tf.int64


def _mean_output_dtype(input_dtype):
  """Returns the output dtype for tft.mean (and similar functions)."""
  return tf.float64 if input_dtype == tf.float64 else tf.float32


class BeamImplTest(tft_unit.TransformTestCase):

  def setUp(self):
    super().setUp()
    tf.compat.v1.logging.info('Starting test case: %s', self._testMethodName)
    self._context = tft_beam.Context(use_deep_copy_optimization=True)
    self._context.__enter__()

  def tearDown(self):
    super().tearDown()
    self._context.__exit__()

  def _OutputRecordBatches(self):
    return False

  def _SkipIfOutputRecordBatches(self):
    if self._OutputRecordBatches():
      raise unittest.SkipTest(
          'Test is disabled when TFT outputs `pa.RecordBatch`es to avoid '
          'duplicated testing: it does not exercise `TransformDataset` or '
          '`AnalyzeAndTransformDataset`.')

  def _MakeTransformOutputAssertFn(self, expected, sort=False):

    def _assert_fn(actual):
      if sort:
        dict_key_fn = lambda d: sorted(d.items())
        expected_sorted = sorted(expected, key=dict_key_fn)
        actual_sorted = sorted(actual, key=dict_key_fn)
        self.assertCountEqual(expected_sorted, actual_sorted)
      else:
        self.assertCountEqual(expected, actual)

    return _assert_fn

  def testApplySavedModelSingleInput(self):
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
      output_col = pretrained_models.apply_saved_model(
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
        expected_metadata)

  def testApplySavedModelWithHashTable(self):
    def save_model_with_hash_table(instance, export_dir):
      builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
      with tf.compat.v1.Graph().as_default() as graph:
        with instance.test_session(graph=graph) as sess:
          key = tf.constant('test_key', shape=[1])
          value = tf.constant('test_value', shape=[1])
          table = tf.lookup.StaticHashTable(
              tf.lookup.KeyValueTensorInitializer(key, value), '__MISSING__')

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

          sess.run(tf.compat.v1.tables_initializer())
          builder.add_meta_graph_and_variables(
              sess, [tf.saved_model.SERVING],
              signature_def_map=signature_def_map)
          builder.save(False)

    export_dir = os.path.join(self.get_temp_dir(), 'saved_model_hash_table')

    def preprocessing_fn(inputs):
      x = inputs['x']
      output_col = pretrained_models.apply_saved_model(
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
        expected_metadata)

  def testApplySavedModelMultiInputs(self):

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
      sum_column = pretrained_models.apply_saved_model(
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
        expected_metadata)

  def testApplyFunctionWithCheckpoint(self):

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
      out_value = pretrained_models.apply_function_with_checkpoint(
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
        expected_metadata)

  @tft_unit.named_parameters(
      dict(testcase_name='NoDeepCopy', with_deep_copy=False),
      dict(testcase_name='WithDeepCopy', with_deep_copy=True),
  )
  def testMultipleLevelsOfAnalyzers(self, with_deep_copy):
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
    with tft_beam.Context(use_deep_copy_optimization=with_deep_copy):
      # NOTE: In order to correctly test deep_copy here, we can't pass test_data
      # to assertAnalyzeAndTransformResults.
      # Not passing test_data to assertAnalyzeAndTransformResults means that
      # tft.AnalyzeAndTransform is called, exercising the right code path.
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
          'x':
              tf.io.FixedLenSequenceFeature(
                  shape=[], dtype=tf.string, default_value=None)
      }
      _, sequences = tf.io.parse_single_sequence_example(
          sequence_example, sequence_features=feature_spec)

      # Create a batch based on the sequence "x".
      return {'x': sequences['x']}

    def text_sequence_example_to_binary(text_proto):
      proto = text_format.Merge(text_proto, tf.train.SequenceExample())
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
        expected_metadata, desired_batch_size=1)

  def testTransformWithExcludedOutputs(self):
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
    with tft_beam.Context(temp_dir=self.get_temp_dir()):
      transform_fn = ((input_data, input_metadata)
                      | tft_beam.AnalyzeDataset(preprocessing_fn))

    # Take the transform function and use TransformDataset to apply it to
    # some eval data, with missing 'y' column.
    eval_data = [{'x': 6}]
    eval_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.float32)})
    transformed_eval_data, transformed_eval_metadata = (
        ((eval_data, eval_metadata), transform_fn)
        | tft_beam.TransformDataset(
            exclude_outputs=['y_scaled'],
            output_record_batches=self._OutputRecordBatches()))

    if self._OutputRecordBatches():
      expected_transformed_eval_data = {'x_scaled': [[1.25]]}
      self.assertLen(transformed_eval_data, 1)
      # Contains RecordBatch and unary pass-through features dict.
      self.assertLen(transformed_eval_data[0], 2)
      self.assertDictEqual(transformed_eval_data[0][0].to_pydict(),
                           expected_transformed_eval_data)
      self.assertDictEqual(transformed_eval_data[0][1], {})
    else:
      expected_transformed_eval_data = [{'x_scaled': 1.25}]
      self.assertDataCloseOrEqual(transformed_eval_data,
                                  expected_transformed_eval_data)
    expected_transformed_eval_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([], tf.float32)})
    self.assertEqual(transformed_eval_metadata.dataset_metadata,
                     expected_transformed_eval_metadata)

  def testMapWithCond(self):
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
        expected_metadata)

  def testPyFuncs(self):
    if not tft_unit.is_tf_api_version_1():
      raise unittest.SkipTest('Test disabled when TF 2.x behavior enabled.')

    def my_multiply(x, y):
      return x*y

    def my_add(x, y):
      return x+y

    def my_list_return(x, y):
      return [x, y, 2 * x, 2 * y]

    def preprocessing_fn(inputs):
      result = {
          'a+b':
              tft.apply_pyfunc(my_add, tf.float32, True, 'add', inputs['a'],
                               inputs['b']),
          'a+c':
              tft.apply_pyfunc(my_add, tf.float32, True, 'add', inputs['a'],
                               inputs['c']),
          'ab':
              tft.apply_pyfunc(my_multiply, tf.float32, False, 'multiply',
                               inputs['a'], inputs['b']),
          'sum_scaled':
              tft.scale_to_0_1(
                  tft.apply_pyfunc(my_add, tf.float32, True, 'add', inputs['a'],
                                   inputs['c'])),
          'list':
              tf.reduce_sum(
                  tft.apply_pyfunc(
                      my_list_return,
                      [tf.float32, tf.float32, tf.float32, tf.float32], True,
                      'my_list_return', inputs['a'], inputs['b']),
                  axis=0),
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
        {'ab': 12, 'a+b': 7, 'a+c': 6, 'list': 21, 'sum_scaled': 0.25},
        {'ab': 2, 'a+b': 3, 'a+c': 4, 'list': 9, 'sum_scaled': 0},
        {'ab': 30, 'a+b': 11, 'a+c': 12, 'list': 33, 'sum_scaled': 1},
        {'ab': 6, 'a+b': 5, 'a+c': 6, 'list': 15, 'sum_scaled': 0.25}
    ]
    # When calling tf.py_func, the output shape is set to unknown.
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'ab': tf.io.FixedLenFeature([], tf.float32),
        'a+b': tf.io.FixedLenFeature([], tf.float32),
        'a+c': tf.io.FixedLenFeature([], tf.float32),
        'list': tf.io.FixedLenFeature([], tf.float32),
        'sum_scaled': tf.io.FixedLenFeature([], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testAssertsNoReturnPyFunc(self):
    # Asserts that apply_pyfunc raises an exception if the passed function does
    # not return anything.
    if not tft_unit.is_tf_api_version_1():
      raise unittest.SkipTest('Test disabled when TF 2.x behavior enabled.')

    self._SkipIfOutputRecordBatches()

    def bad_func():
      return None

    with self.assertRaises(ValueError):
      tft.apply_pyfunc(bad_func, [], False, 'bad_func')

  def testWithMoreThanDesiredBatchSize(self):
    def preprocessing_fn(inputs):
      return {
          'ab': tf.multiply(inputs['a'], inputs['b']),
          'i': tft.compute_and_apply_vocabulary(inputs['c'])
      }

    batch_size = 100
    num_instances = batch_size + 1
    # pylint: disable=g-complex-comprehension
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
    # pylint: enable=g-complex-comprehension
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
        desired_batch_size=batch_size)

  def testWithUnicode(self):
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
        expected_metadata)

  def testNpArrayInput(self):

    def preprocessing_fn(inputs):
      return {'a b': tf.compat.v1.strings.join(
          [inputs['a'], inputs['b']], separator=' ')}

    input_data = [{
        'a': np.array('Hello', dtype=object),
        'b': np.array('world', dtype=object)
    }, {
        'a': np.array('Hello', dtype=object),
        'b': np.array(u'κόσμε', dtype=object)
    }]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.string),
        'b': tf.io.FixedLenFeature([], tf.string),
    })
    expected_data = [{
        'a b': np.array(b'Hello world', dtype=object)
    }, {
        'a b': np.array(u'Hello κόσμε'.encode('utf-8'), dtype=object)
    }]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'a b': tf.io.FixedLenFeature([], tf.string)})
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  @tft_unit.parameters((True,), (False,))
  def testScaleUnitInterval(self, elementwise):

    def preprocessing_fn(inputs):
      outputs = {}
      stacked_input = tf.stack([inputs['x'], inputs['y']], axis=1)
      result = tft.scale_to_0_1(stacked_input, elementwise=elementwise)
      outputs['x_scaled'], outputs['y_scaled'] = tf.unstack(result, axis=1)
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
                                          expected_metadata)

  def testScaleUnitIntervalPerKey(self):

    def preprocessing_fn(inputs):
      outputs = {}
      stacked_input = tf.stack([inputs['x'], inputs['y']], axis=1)
      result = tft.scale_to_0_1_per_key(
          stacked_input, inputs['key'], elementwise=False)
      outputs['x_scaled'], outputs['y_scaled'] = tf.unstack(result, axis=1)
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
                                          expected_metadata)

  @tft_unit.parameters((True,), (False,))
  def testScaleMinMax(self, elementwise):
    def preprocessing_fn(inputs):
      outputs = {}
      stacked_input = tf.stack([inputs['x'], inputs['y']], axis=1)
      result = tft.scale_by_min_max(
          stacked_input, output_min=-1, output_max=1, elementwise=elementwise)
      outputs['x_scaled'], outputs['y_scaled'] = tf.unstack(result, axis=1)
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

  @tft_unit.named_parameters(
      dict(testcase_name='_empty_filename',
           key_vocabulary_filename=''),
      dict(testcase_name='_nonempty_filename',
           key_vocabulary_filename='per_key'),
      dict(testcase_name='_none_filename',
           key_vocabulary_filename=None)
  )
  def testScaleMinMaxPerKey(self, key_vocabulary_filename):
    def preprocessing_fn(inputs):
      outputs = {}
      stacked_input = tf.stack([inputs['x'], inputs['y']], axis=1)
      result = tft.scale_by_min_max_per_key(
          stacked_input,
          inputs['key'],
          output_min=-1,
          output_max=1,
          elementwise=False,
          key_vocabulary_filename=key_vocabulary_filename)
      outputs['x_scaled'], outputs['y_scaled'] = tf.unstack(result, axis=1)
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
        expected_vocab_file_contents=per_key_vocab_contents)

  def testScalePerKeySparse(self):
    def preprocessing_fn(inputs):
      return {
          'scaled_by_min_max':
              tft.scale_by_min_max_per_key(
                  inputs['x'], inputs['key'], output_min=-1, output_max=1),
          'scaled_to_0_1':
              tft.scale_to_0_1_per_key(inputs['x'], inputs['key']),
          'scaled_to_z_score':
              tft.scale_to_z_score_per_key(inputs['x'], inputs['key']),
      }

    input_data = [{
        'val': [4, 8],
        's': ['a', 'a']
    }, {
        'val': [1, 5],
        's': ['a', 'a']
    }, {
        'val': [5, 9],
        's': ['a', 'a']
    }, {
        'val': [2, 6],
        's': ['a', 'a']
    }, {
        'val': [-2, 0],
        's': ['b', 'b']
    }, {
        'val': [0, 2],
        's': ['b', 'b']
    }]
    indices = [([x % 2] * 2, [x % 3] * 2) for x in range(len(input_data))]
    indices_x = [{'idx_x_0': a, 'idx_x_1': b} for a, b in indices]
    indices_key = [{'idx_key_0': a, 'idx_key_1': b} for a, b in indices]
    input_data = [{**a, **b, **c}
                  for a, b, c in zip(input_data, indices_x, indices_key)]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x':
            tf.io.SparseFeature(['idx_x_0', 'idx_x_1'], 'val', tf.float32,
                                (2, 3)),
        'key':
            tf.io.SparseFeature(['idx_key_0', 'idx_key_1'], 's', tf.string,
                                (2, 3))
    })

    output_names = ['scaled_by_min_max', 'scaled_to_0_1', 'scaled_to_z_score']
    expected_indices_prefix = [
        (('$sparse_indices_0', a), ('$sparse_indices_1', b)) for a, b in indices
    ]
    expected_indices = []
    for idx0, idx1 in expected_indices_prefix:
      instance = {}
      for n in output_names:
        instance.update({n + idx0[0]: idx0[1]})
        instance.update({n + idx1[0]: idx1[1]})
      expected_indices.append(instance)

    expected_data = [{
        'scaled_by_min_max$sparse_values': [-0.25, 0.75],
        'scaled_to_0_1$sparse_values':
            np.array([3. / 8., 7. / 8]),
        'scaled_to_z_score$sparse_values':
            np.array([-1. / math.sqrt(6.5), 3. / math.sqrt(6.5)])
    }, {
        'scaled_by_min_max$sparse_values': [-1.0, 0.0],
        'scaled_to_0_1$sparse_values': np.array([0., 0.5]),
        'scaled_to_z_score$sparse_values': np.array([-4. / math.sqrt(6.5), 0.]),
    }, {
        'scaled_by_min_max$sparse_values': [0.0, 1.0],
        'scaled_to_0_1$sparse_values': np.array([0.5, 1.]),
        'scaled_to_z_score$sparse_values': np.array([0., 4. / math.sqrt(6.5)]),
    }, {
        'scaled_by_min_max$sparse_values': [-0.75, 0.25],
        'scaled_to_0_1$sparse_values':
            np.array([1. / 8., 5. / 8.]),
        'scaled_to_z_score$sparse_values':
            np.array([-3. / math.sqrt(6.5), 1. / math.sqrt(6.5)]),
    }, {
        'scaled_by_min_max$sparse_values': np.array([-1., 0.]),
        'scaled_to_0_1$sparse_values': np.array([0., 0.5]),
        'scaled_to_z_score$sparse_values': np.array([-2. / math.sqrt(2), 0.]),
    }, {
        'scaled_by_min_max$sparse_values': [0.0, 1.0],
        'scaled_to_0_1$sparse_values': np.array([0.5, 1.]),
        'scaled_to_z_score$sparse_values': np.array([0., 2. / math.sqrt(2)]),
    }]
    expected_data = [{**a, **b}
                     for a, b in zip(expected_data, expected_indices)]
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        beam_pipeline=beam.Pipeline())

  @tft_unit.named_parameters(
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
  )
  def testScaleMinMaxSparsePerKey(
      self, input_data, input_metadata, expected_data):
    def preprocessing_fn(inputs):
      x_scaled = tf.sparse.to_dense(
          tft.scale_to_0_1_per_key(inputs['x'], inputs['key']))
      x_scaled.set_shape([None, 4])
      return {'x_scaled': x_scaled}

    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([4], tf.float32)})

    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters(
      [
          dict(
              testcase_name='dense_key',
              input_data=[{
                  'x_val': [-4, 4],
                  'x_row_lengths': [0, 2],
                  'key': 'a',
              }, {
                  'x_val': [0, 1],
                  'x_row_lengths': [1, 1],
                  'key': 'a',
              }, {
                  'x_val': [-4, 1, 1],
                  'x_row_lengths': [3],
                  'key': 'b',
              }],
              make_key_spec=lambda: tf.io.FixedLenFeature([], tf.string),
              expected_data=[{
                  'scaled_by_min_max$ragged_values': [-1., 1.],
                  'scaled_by_min_max$row_lengths_1': [0, 2],
                  'scaled_to_0_1$ragged_values': [0., 1.],
                  'scaled_to_0_1$row_lengths_1': [0, 2],
                  'scaled_to_z_score$ragged_values': [-1.4852968, 1.310556],
                  'scaled_to_z_score$row_lengths_1': [0, 2],
              }, {
                  'scaled_by_min_max$ragged_values': [0., 0.25],
                  'scaled_by_min_max$row_lengths_1': [1, 1],
                  'scaled_to_0_1$ragged_values': [0.5, 0.625],
                  'scaled_to_0_1$row_lengths_1': [1, 1],
                  'scaled_to_z_score$ragged_values': [-0.0873704, 0.26211122],
                  'scaled_to_z_score$row_lengths_1': [1, 1],
              }, {
                  'scaled_by_min_max$ragged_values': [-1., 1., 1.],
                  'scaled_by_min_max$row_lengths_1': [3],
                  'scaled_to_0_1$ragged_values': [0., 1., 1.],
                  'scaled_to_0_1$row_lengths_1': [3],
                  'scaled_to_z_score$ragged_values':
                      [-1.4142135, 0.7071068, 0.7071068],
                  'scaled_to_z_score$row_lengths_1': [3]
              }],
          ),
          dict(
              testcase_name='ragged_key',
              input_data=[{
                  'x_val': [-4, 4],
                  'x_row_lengths': [0, 2],
                  'key_val': ['a', 'a'],
                  'key_row_lengths': [0, 2],
              }, {
                  'x_val': [0, 1],
                  'x_row_lengths': [1, 1],
                  'key_val': ['a', 'b'],
                  'key_row_lengths': [1, 1],
              }, {
                  'x_val': [-4, 1, 1],
                  'x_row_lengths': [3],
                  'key_val': ['b', 'a', 'b'],
                  'key_row_lengths': [3],
              }],
              make_key_spec=lambda: tf.io.RaggedFeature(  # pylint: disable=g-long-lambda
                  tf.string,
                  value_key='key_val',
                  partitions=[
                      tf.io.RaggedFeature.RowLengths('key_row_lengths')  # pytype: disable=attribute-error
                  ]),
              expected_data=[{
                  'scaled_by_min_max$ragged_values': [-1., 1.],
                  'scaled_by_min_max$row_lengths_1': [0, 2],
                  'scaled_to_0_1$ragged_values': [0., 1.],
                  'scaled_to_0_1$row_lengths_1': [0, 2],
                  'scaled_to_z_score$ragged_values': [-1.4852968, 1.310556],
                  'scaled_to_z_score$row_lengths_1': [0, 2],
              }, {
                  'scaled_by_min_max$ragged_values': [0., 1.],
                  'scaled_by_min_max$row_lengths_1': [1, 1],
                  'scaled_to_0_1$ragged_values': [0.5, 1.],
                  'scaled_to_0_1$row_lengths_1': [1, 1],
                  'scaled_to_z_score$ragged_values': [-0.0873704, 0.7071068],
                  'scaled_to_z_score$row_lengths_1': [1, 1],
              }, {
                  'scaled_by_min_max$ragged_values': [-1., 0.25, 1.],
                  'scaled_by_min_max$row_lengths_1': [3],
                  'scaled_to_0_1$ragged_values': [0., 0.625, 1.],
                  'scaled_to_0_1$row_lengths_1': [3],
                  'scaled_to_z_score$ragged_values':
                      [-1.4142135, 0.26211122, 0.7071068],
                  'scaled_to_z_score$row_lengths_1': [3]
              }]),
      ],
      [
          dict(testcase_name='int16', input_dtype=tf.int16),
          dict(testcase_name='int32', input_dtype=tf.int32),
          dict(testcase_name='int64', input_dtype=tf.int64),
          dict(testcase_name='float32', input_dtype=tf.float32),
          dict(testcase_name='float64', input_dtype=tf.float64),
      ]))
  def testScalePerKeyRagged(self, input_data, make_key_spec, expected_data,
                            input_dtype):
    make_x_spec = lambda: tf.io.RaggedFeature(  # pylint: disable=g-long-lambda
        tft_unit.canonical_numeric_dtype(input_dtype),
        value_key='x_val',
        partitions=[
            tf.io.RaggedFeature.RowLengths('x_row_lengths')  # pytype: disable=attribute-error
        ])
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tft_unit.make_feature_spec_wrapper(make_x_spec),
        'key': tft_unit.make_feature_spec_wrapper(make_key_spec)
    })

    def preprocessing_fn(inputs):
      scaled_to_z_score = tft.scale_to_z_score_per_key(
          tf.cast(inputs['x'], input_dtype), inputs['key'])
      self.assertEqual(scaled_to_z_score.dtype, _mean_output_dtype(input_dtype))
      return {
          'scaled_by_min_max':
              tft.scale_by_min_max_per_key(
                  tf.cast(inputs['x'], input_dtype),
                  inputs['key'],
                  output_min=-1,
                  output_max=1),
          'scaled_to_0_1':
              tft.scale_to_0_1_per_key(
                  tf.cast(inputs['x'], input_dtype), inputs['key']),
          'scaled_to_z_score':
              tf.cast(scaled_to_z_score, tf.float32),
      }

    expected_specs = {}
    for output_name in ('scaled_by_min_max', 'scaled_to_0_1',
                        'scaled_to_z_score'):
      expected_specs[output_name] = tf.io.RaggedFeature(
          tf.float32,
          value_key='{}$ragged_values'.format(output_name),
          partitions=[
              tf.io.RaggedFeature.RowLengths(  # pytype: disable=attribute-error
                  '{}$row_lengths_1'.format(output_name))
          ])
    expected_metadata = tft_unit.metadata_from_feature_spec(expected_specs)
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testScaleMinMaxConstant(self):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_by_min_max(inputs['x'], 0, 10)}

    input_data = [{'x': 4}, {'x': 4}, {'x': 4}, {'x': 4}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.float32)})
    expected_data = [{
        'x_scaled': 9.8201379
    }, {
        'x_scaled': 9.8201379
    }, {
        'x_scaled': 9.8201379
    }, {
        'x_scaled': 9.8201379
    }]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([], tf.float32)})
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testScaleMinMaxConstantElementwise(self):

    def preprocessing_fn(inputs):
      outputs = {}
      stacked_input = tf.stack([inputs['x'], inputs['y']], axis=1)
      result = tft.scale_by_min_max(
          stacked_input, output_min=0, output_max=10, elementwise=True)
      outputs['x_scaled'], outputs['y_scaled'] = tf.unstack(result, axis=1)
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
        'x_scaled': 9.8201379,
        'y_scaled': 0
    }, {
        'x_scaled': 9.8201379,
        'y_scaled': 0
    }, {
        'x_scaled': 9.8201379,
        'y_scaled': 10
    }, {
        'x_scaled': 9.8201379,
        'y_scaled': 10
    }]
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature([], tf.float32),
        'y_scaled': tf.io.FixedLenFeature([], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testScaleMinMaxError(self):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_by_min_max(inputs['x'], 2, 1)}

    input_data = [{'x': 1}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.float32)})
    expected_data = [{'x_scaled': float('nan')}]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([], tf.float32)})
    with self.assertRaisesRegexp(  # pylint: disable=g-error-prone-assert-raises
        ValueError, 'output_min must be less than output_max'):
      self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                            preprocessing_fn, expected_data,
                                            expected_metadata)

  def testScaleMinMaxWithEmptyInputs(self):
    # x is repeated `multiple` times to test elementwise mapping.
    multiple = 3

    def preprocessing_fn(inputs):
      return {
          'x_scaled':
              tft.scale_by_min_max(inputs['x']),
          'x_scaled_elementwise':
              tft.scale_by_min_max(
                  tf.tile(inputs['x'], [1, multiple]), elementwise=True)
      }

    input_data = []
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([1], tf.float32)})
    test_data = [{'x': [100]}, {'x': [1]}, {'x': [12]}]
    expected_data = [{'x_scaled': [v], 'x_scaled_elementwise': [v] * multiple}
                     for v in [1., 0.7310585, 0.9999938]]
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature([1], tf.float32),
        'x_scaled_elementwise': tf.io.FixedLenFeature([multiple], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        test_data=test_data)

  @tft_unit.named_parameters(*(_SCALE_TO_Z_SCORE_TEST_CASES +
                               _SCALE_TO_Z_SCORE_NAN_TEST_CASES))
  def testScaleToZScore(self, input_data, output_data, elementwise):

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
        preprocessing_fn, expected_data_dicts, expected_metadata)

  @tft_unit.parameters(*itertools.product([
      tf.int16,
      tf.int32,
      tf.int64,
      tf.float32,
      tf.float64,
  ], (True, False)))
  def testScaleToZScoreSparse(self, input_dtype, elementwise):
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
    if input_dtype.is_floating:
      input_data.append({'idx': [0, 1], 'val': [np.nan, np.nan]})
      expected_data.append({
          'x_scaled': [float('nan'),
                       float('nan'),
                       float('nan'),
                       float('nan')]
      })
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([4], tf.float32)})
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  @tft_unit.parameters(
      (tf.int16,),
      (tf.int32,),
      (tf.int64,),
      (tf.float32,),
      (tf.float64,),
  )
  def testScaleToZScoreSparsePerDenseKey(self, input_dtype):
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
                                          expected_metadata)

  @tft_unit.named_parameters(
      dict(testcase_name='_empty_filename',
           key_vocabulary_filename=''),
      dict(testcase_name='_nonempty_filename',
           key_vocabulary_filename='per_key'),
      dict(testcase_name='_none_filename',
           key_vocabulary_filename=None)
  )
  def testScaleToZScorePerKey(self, key_vocabulary_filename):
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
    input_data = [
        {
            'x': np.array([-4], dtype=np_dtype),
            'y': np.array([0], dtype=np_dtype),
            's': 3,
            'key': 'a',
        },
        {
            'x': np.array([10], dtype=np_dtype),
            'y': np.array([0], dtype=np_dtype),
            's': -3,
            'key': 'a',
        },
        {
            'x': np.array([1], dtype=np_dtype),
            'y': np.array([0], dtype=np_dtype),
            's': 3,
            'key': 'b',
        },
        {
            'x': np.array([2], dtype=np_dtype),
            'y': np.array([0], dtype=np_dtype),
            's': 3,
            'key': 'a',
        },
        {
            'x': np.array([4], dtype=np_dtype),
            'y': np.array([0], dtype=np_dtype),
            's': -3,
            'key': 'a',
        },
        {
            'x': np.array([-1], dtype=np_dtype),
            'y': np.array([0], dtype=np_dtype),
            's': -3,
            'key': 'b',
        },
        {
            'x': np.array([np.nan], dtype=np_dtype),
            'y': np.array([np.nan], dtype=np_dtype),
            's': np.nan,
            'key': 'b',
        },
    ]
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
        },
        {
            'x_scaled': [np.nan],
            'y_scaled': [np.nan],
            's_scaled': np.nan,
        },
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
                                          expected_metadata)

  @tft_unit.parameters(
      (tf.int16,),
      (tf.int32,),
      (tf.int64,),
      (tf.float32,),
      (tf.float64,),
  )
  def testScaleToZScoreSparsePerKey(self, input_dtype):
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
    if input_dtype.is_floating:
      input_data.append({
          'idx': [0, 1],
          'val': [np.nan, np.nan],
          'key_idx': [0, 1],
          'key': ['a', 'b']
      })
      expected_data.append({
          'x_scaled': [float('nan'),
                       float('nan'),
                       float('nan'),
                       float('nan')]
      })
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {'x_scaled': tf.io.FixedLenFeature([4], tf.float32)})
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testScaleToZScoreWithEmptyInputs(self):
    # x is repeated `multiple` times to test elementwise mapping.
    multiple = 3

    def preprocessing_fn(inputs):
      return {
          'x_scaled':
              tft.scale_to_z_score(inputs['x']),
          'x_scaled_elementwise':
              tft.scale_to_z_score(
                  tf.tile(inputs['x'], [1, multiple]), elementwise=True)
      }

    input_data = []
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([1], tf.float32)})
    test_data = [{'x': [100]}, {'x': [1]}, {'x': [12]}]
    expected_data = [{'x_scaled': [v], 'x_scaled_elementwise': [v] * multiple}
                     for v in [100., 1., 12.]]
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x_scaled': tf.io.FixedLenFeature([1], tf.float32),
        'x_scaled_elementwise': tf.io.FixedLenFeature([multiple], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        test_data=test_data)

  def testMeanAndVar(self):
    self._SkipIfOutputRecordBatches()

    def analyzer_fn(inputs):
      mean, var = analyzers._mean_and_var(inputs['x'])
      return {
          'mean': mean,
          'var': var
      }

    # NOTE: We force 11 batches: data has 110 elements and we request a batch
    # size of 10.
    input_data = [{'x': [x if x < 101 else np.nan]} for x in range(1, 111)]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([1], tf.float32)})
    expected_outputs = {
        'mean': np.float32(50.5),
        'var': np.float32(833.25)
    }
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        desired_batch_size=10)

  def testMeanAndVarPerKey(self):
    self._SkipIfOutputRecordBatches()

    def analyzer_fn(inputs):
      key_vocab, mean, var = analyzers._mean_and_var_per_key(
          inputs['x'], inputs['key'])
      return {
          'key_vocab': key_vocab,
          'mean': mean,
          'var': tf.round(100 * var) / 100.0
      }

    # NOTE: We force 12 batches: data has 120 elements and we request a batch
    # size of 10.
    input_data = [{'x': [x], 'key': 'a' if x < 50 else 'b'}
                  for x in range(1, 101)] + [{'x': [np.nan], 'key': 'a'}] * 20
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([1], tf.float32),
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
        desired_batch_size=10)

  @tft_unit.parameters((True,), (False,))
  def testPerKeyWithOOVKeys(self, use_vocabulary):
    def preprocessing_fn(inputs):
      result = {}
      result['x_scaled'] = tft.scale_to_0_1_per_key(
          inputs['x'],
          inputs['key'],
          elementwise=False,
          key_vocabulary_filename='a' if use_vocabulary else None)
      result['x_z_score'] = tft.scale_to_z_score_per_key(
          inputs['x'],
          inputs['key'],
          elementwise=False,
          key_vocabulary_filename='b' if use_vocabulary else None)
      # TODO(b/179891014): Add key_vocabulary_filename to bucketize_per_key once
      # implemented.
      result['x_bucketized'] = tft.bucketize_per_key(inputs['x'], inputs['key'],
                                                     3)
      return result

    input_data = [
        dict(x=4, key='a'),
        dict(x=1, key='a'),
        dict(x=5, key='a'),
        dict(x=2, key='a'),
        dict(x=25, key='b'),
        dict(x=5, key='b')
    ]
    test_data = input_data + [dict(x=5, key='oov')]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
        'key': tf.io.FixedLenFeature([], tf.string)
    })

    expected_data = [{
        'x_scaled': 0.75,
        'x_z_score': 0.6324555,
        'x_bucketized': 2,
    }, {
        'x_scaled': 0.0,
        'x_z_score': -1.264911,
        'x_bucketized': 0,
    }, {
        'x_scaled': 1.0,
        'x_z_score': 1.264911,
        'x_bucketized': 2,
    }, {
        'x_scaled': 0.25,
        'x_z_score': -0.6324555,
        'x_bucketized': 1,
    }, {
        'x_scaled': 1.0,
        'x_z_score': 1.0,
        'x_bucketized': 2,
    }, {
        'x_scaled': 0.0,
        'x_z_score': -1.0,
        'x_bucketized': 1,
    }, {
        'x_scaled': _sigmoid(5),
        'x_z_score': 5.0,
        'x_bucketized': -1,
    }]
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        test_data=test_data)

  @tft_unit.named_parameters(
      dict(
          testcase_name='_string',
          input_data=[{
              'key': 'a' if x < 25 else 'b'
          } for x in range(100)],
          input_metadata=tft_unit.metadata_from_feature_spec(
              {'key': tf.io.FixedLenFeature([], tf.string)}),
          expected_outputs={
              'elements': np.array([b'a', b'b'], np.object),
              'counts': np.array([25, 75], np.int64)
          }),
      dict(
          testcase_name='_int',
          input_data=[{
              'key': 0 if x < 25 else 1
          } for x in range(100)],
          input_metadata=tft_unit.metadata_from_feature_spec(
              {'key': tf.io.FixedLenFeature([], tf.int64)}),
          expected_outputs={
              'elements': np.array([0, 1], np.int64),
              'counts': np.array([25, 75], np.int64)
          }),
      dict(
          testcase_name='_int_sparse',
          input_data=[{
              'key': [0] if x < 25 else [1]
          } for x in range(100)],
          input_metadata=tft_unit.metadata_from_feature_spec(
              {'key': tf.io.VarLenFeature(tf.int64)}),
          expected_outputs={
              'elements': np.array([0, 1], np.int64),
              'counts': np.array([25, 75], np.int64)
          }),
      dict(
          testcase_name='_3d_sparse',
          input_data=[
              {  # pylint: disable=g-complex-comprehension
                  'key': [0, 1] if x < 25 else [1],
                  'idx0': [0, 1] if x < 25 else [0],
                  'idx1': [0, 1] if x < 25 else [0]
              } for x in range(100)
          ],
          input_metadata=tft_unit.metadata_from_feature_spec({
              'key':
                  tf.io.SparseFeature(['idx0', 'idx1'], 'key', tf.int64, [2, 2])
          }),
          expected_outputs={
              'elements': np.array([0, 1], np.int64),
              'counts': np.array([25, 100], np.int64)
          },
      ),
  )
  def testCountPerKey(self, input_data, input_metadata, expected_outputs):
    self._SkipIfOutputRecordBatches()

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
        expected_outputs)

  @tft_unit.named_parameters(
      dict(
          testcase_name='_uniform',
          input_data=[{
              'x': [x]
          } for x in range(10, 100)],
          make_feature_spec=lambda: tf.io.FixedLenFeature([1], tf.int64),
          boundaries=10 * np.arange(11, dtype=np.float32),
          categorical=False,
          expected_outputs={
              'hist':
                  10 * np.array([0] + [1] * 9, np.int64),
              'boundaries':
                  10 * np.arange(11, dtype=np.float32).reshape((1, 11))
          }),
      dict(
          testcase_name='_categorical_string',
          input_data=[{
              'x': [str(x % 10) + '_']
          } for x in range(1, 101)],
          make_feature_spec=lambda: tf.io.FixedLenFeature([1], tf.string),
          boundaries=None,
          categorical=True,
          expected_outputs={
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
      ),
      dict(
          testcase_name='_categorical_int',
          input_data=[{
              'x': [(x % 10)]
          } for x in range(1, 101)],
          make_feature_spec=lambda: tf.io.FixedLenFeature([1], tf.int64),
          boundaries=None,
          categorical=True,
          expected_outputs={
              'hist': 10 * np.ones(10, np.int64),
              'boundaries': np.arange(10)
          }),
      dict(
          testcase_name='_sparse',
          input_data=[{  # pylint: disable=g-complex-comprehension
              'val': [(x % 10)],
              'idx0': [(x % 2)],
              'idx1': [((x + 1) % 2)]
          } for x in range(1, 101)],
          make_feature_spec=lambda: tf.io.SparseFeature(  # pylint: disable=g-long-lambda
              ['idx0', 'idx1'], 'val', tf.int64, [2, 2]),
          boundaries=None,
          categorical=True,
          expected_outputs={
              'hist': 10 * np.ones(10, np.int64),
              'boundaries': np.arange(10)
          }),
      dict(
          testcase_name='_ragged',
          input_data=[{  # pylint: disable=g-complex-comprehension
              'val': [x % 10, 9 - (x % 10)],
              'row_lengths': [0, 1, 1],
          } for x in range(1, 101)],
          make_feature_spec=lambda: tf.io.RaggedFeature(  # pylint: disable=g-long-lambda
              tf.int64,
              value_key='val',
              partitions=[
                  tf.io.RaggedFeature.RowLengths('row_lengths')  # pytype: disable=attribute-error
              ])
          ,
          boundaries=None,
          categorical=True,
          expected_outputs={
              'hist': 20 * np.ones(10, np.int64),
              'boundaries': np.arange(10)
          }),
  )
  def testHistograms(self, input_data, make_feature_spec, boundaries,
                     categorical, expected_outputs):
    self._SkipIfOutputRecordBatches()
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tft_unit.make_feature_spec_wrapper(make_feature_spec)})

    def analyzer_fn(inputs):
      counts, bucket_boundaries = analyzers.histogram(
          inputs['x'], categorical=categorical, boundaries=boundaries)
      if not categorical:
        bucket_boundaries = tf.math.round(bucket_boundaries)
      return {'hist': counts, 'boundaries': bucket_boundaries}

    self.assertAnalyzerOutputs(input_data,
                               input_metadata,
                               analyzer_fn,
                               expected_outputs)

  def testProbCategoricalInt(self):
    def preprocessing_fn(inputs):
      return {'probs': tft.estimated_probability_density(inputs['x'],
                                                         categorical=True)}

    # NOTE: We force 10 batches: data has 100 elements and we request a batch
    # size of 10.
    input_data = [{'x': [x % 10]} for x in range(1, 101)]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([1], tf.int64)
    })
    expected_outputs = [{
        'probs': np.array(np.ones(1) / 10.0, np.float32)
    } for _ in range(100)]
    self.assertAnalyzeAndTransformResults(input_data,
                                          input_metadata,
                                          preprocessing_fn,
                                          expected_outputs,
                                          desired_batch_size=10)

  def testProbCategorical(self):
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
        'probs': np.array(np.ones(1) / 10.0, np.float32)
    } for _ in range(100)]
    self.assertAnalyzeAndTransformResults(input_data,
                                          input_metadata,
                                          preprocessing_fn,
                                          expected_outputs,
                                          desired_batch_size=10)

  def testProbTenBoundaries(self):
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
        'probs': np.array(np.ones(1) / (100.0), np.float32)
    } for _ in range(100)]
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_outputs,
        desired_batch_size=10)

  @tft_unit.named_parameters(
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
  )
  def testProbUnknownBoundaries(
      self, input_data, expected_outputs, boundaries):
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
        expected_outputs)

  @tft_unit.named_parameters(
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
  )
  def testNumericAnalyzersWithScalarInputs(self, input_dtype, output_dtypes):
    self._SkipIfOutputRecordBatches()

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
        input_data, input_metadata, analyzer_fn, expected_outputs)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters(
      [
          dict(
              testcase_name='sparse',
              input_data=[
                  {
                      'idx0': [0, 1],
                      'idx1': [0, 1],
                      'val': [0, 1],
                  },
                  {
                      'idx0': [1, 2],
                      'idx1': [1, 3],
                      'val': [2, 3],
                  },
              ],
              make_feature_spec=lambda dtype: tf.io.SparseFeature(  # pylint: disable=g-long-lambda
                  ['idx0', 'idx1'], 'val', dtype, (3, 4)),
              expected_outputs={
                  'min': 0.,
                  'max': 3.,
                  'sum': 6.,
                  'size': 4,
                  'mean': 1.5,
                  'var': 1.25,
              },
              reduce_instance_dims=True,
          ),
          dict(
              testcase_name='sparse_elementwise',
              input_data=[
                  {
                      'idx0': [0, 1],
                      'idx1': [0, 1],
                      'val': [0, 1],
                  },
                  {
                      'idx0': [1, 2],
                      'idx1': [1, 3],
                      'val': [2, 3],
                  },
              ],
              make_feature_spec=lambda dtype: tf.io.SparseFeature(  # pylint: disable=g-long-lambda
                  ['idx0', 'idx1'], 'val', dtype, (3, 4)),
              expected_outputs={
                  # We use np.nan in place of missing values here but replace
                  # them accordingly to the dtype in the test.
                  'min': [[0., np.nan, np.nan, np.nan],
                          [np.nan, 1., np.nan, np.nan],
                          [np.nan, np.nan, np.nan, 3.]],
                  'max': [[0., np.nan, np.nan, np.nan],
                          [np.nan, 2., np.nan, np.nan],
                          [np.nan, np.nan, np.nan, 3.]],
                  'sum': [[0., 0., 0., 0.], [0., 3., 0., 0.], [0., 0., 0., 3.]],
                  'size': [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 1]],
                  'mean': [[0., np.nan, np.nan, np.nan],
                           [np.nan, 1.5, np.nan, np.nan],
                           [np.nan, np.nan, np.nan, 3.]],
                  'var': [[0., np.nan, np.nan, np.nan],
                          [np.nan, 0.25, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, 0.]],
              },
              reduce_instance_dims=False,
          ),
          dict(
              testcase_name='ragged',
              input_data=[
                  {
                      'val': [0., 2., 3.],
                      'row_lengths': [0, 3],
                  },
                  {
                      'val': [3., 3., 1.],
                      'row_lengths': [3],
                  },
              ],
              make_feature_spec=lambda dtype: tf.io.RaggedFeature(  # pylint: disable=g-long-lambda
                  dtype,
                  value_key='val',
                  partitions=[tf.io.RaggedFeature.RowLengths('row_lengths')]),  # pytype: disable=attribute-error
              expected_outputs={
                  'min': 0.,
                  'max': 3.,
                  'sum': 12.,
                  'size': 6,
                  'mean': 2.,
                  'var': 1.333333,
              },
              reduce_instance_dims=True,
          )
      ],
      [
          dict(testcase_name='int16', input_dtype=tf.int16),
          dict(testcase_name='int32', input_dtype=tf.int32),
          dict(testcase_name='int64', input_dtype=tf.int64),
          dict(testcase_name='float32', input_dtype=tf.float32),
          dict(testcase_name='tf.float64', input_dtype=tf.float64),
          dict(testcase_name='tf.uint8', input_dtype=tf.uint8),
          dict(testcase_name='tf.uint16', input_dtype=tf.uint16),
      ]))
  def testNumericAnalyzersWithCompositeInputs(self, input_data,
                                              make_feature_spec,
                                              expected_outputs,
                                              reduce_instance_dims,
                                              input_dtype):
    self._SkipIfOutputRecordBatches()
    output_dtype = tft_unit.canonical_numeric_dtype(input_dtype)
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tft_unit.make_feature_spec_wrapper(make_feature_spec, output_dtype)
    })

    def analyzer_fn(inputs):
      return {
          'min': tft.min(inputs['a'], reduce_instance_dims),
          'max': tft.max(inputs['a'], reduce_instance_dims),
          'sum': tft.sum(inputs['a'], reduce_instance_dims),
          'size': tft.size(inputs['a'], reduce_instance_dims),
          'mean': tft.mean(inputs['a'], reduce_instance_dims),
          'var': tft.var(inputs['a'], reduce_instance_dims),
      }

    input_val_dtype = input_dtype.as_numpy_dtype
    # Cast input values to appropriate type.
    for instance in input_data:
      instance['val'] = np.array(instance['val'], input_val_dtype)
    if not reduce_instance_dims:
      if input_dtype.is_floating:
        missing_value_max = float('nan')
        missing_value_min = float('nan')
      else:
        missing_value_max = np.iinfo(output_dtype.as_numpy_dtype).min
        missing_value_min = np.iinfo(output_dtype.as_numpy_dtype).max
      # Replace NaNs with proper missing values.
      for row in expected_outputs['min']:
        for idx in range(len(row)):
          if np.isnan(row[idx]):
            row[idx] = missing_value_min
      for row in expected_outputs['max']:
        for idx in range(len(row)):
          if np.isnan(row[idx]):
            row[idx] = missing_value_max
    for op in ('min', 'max', 'sum'):
      expected_outputs[op] = np.array(expected_outputs[op],
                                      output_dtype.as_numpy_dtype)
    expected_outputs['size'] = np.array(expected_outputs['size'], np.int64)
    expected_outputs['mean'] = np.array(expected_outputs['mean'], np.float32)
    expected_outputs['var'] = np.array(expected_outputs['var'], np.float32)
    self.assertAnalyzerOutputs(input_data, input_metadata, analyzer_fn,
                               expected_outputs)

  @tft_unit.named_parameters(
      dict(
          testcase_name='sparse',
          input_data=[
              {
                  'idx0': [0, 1],
                  'idx1': [0, 1],
                  'val': np.array([0, 1], dtype=np.int64)
              },
              {
                  'idx0': [1, 2],
                  'idx1': [1, 3],
                  'val': np.array([2, 3], dtype=np.int64)
              },
          ],
          make_feature_spec=lambda: tf.io.SparseFeature(  # pylint: disable=g-long-lambda
              ['idx0', 'idx1'], 'val', tf.int64, (3, 4)),
          elementwise=False,
          expected_outputs=[{
              'scale_to_0_1$sparse_indices_0':
                  np.array([0, 1]),
              'scale_to_0_1$sparse_indices_1':
                  np.array([0, 1]),
              'scale_to_z_score$sparse_indices_0':
                  np.array([0, 1]),
              'scale_to_z_score$sparse_indices_1':
                  np.array([0, 1]),
              'scale_by_min_max$sparse_indices_0':
                  np.array([0, 1]),
              'scale_by_min_max$sparse_indices_1':
                  np.array([0, 1]),
              'scale_to_0_1$sparse_values':
                  np.array([0., 1. / 3.], dtype=np.float32),
              'scale_to_z_score$sparse_values':
                  np.array([-1.5 / np.sqrt(1.25), -0.5 / np.sqrt(1.25)],
                           dtype=np.float32),
              'scale_by_min_max$sparse_values':
                  np.array([0., 1. / 3.], dtype=np.float32),
          }, {
              'scale_to_0_1$sparse_indices_0':
                  np.array([1, 2]),
              'scale_to_0_1$sparse_indices_1':
                  np.array([1, 3]),
              'scale_to_z_score$sparse_indices_0':
                  np.array([1, 2]),
              'scale_to_z_score$sparse_indices_1':
                  np.array([1, 3]),
              'scale_by_min_max$sparse_indices_0':
                  np.array([1, 2]),
              'scale_by_min_max$sparse_indices_1':
                  np.array([1, 3]),
              'scale_to_0_1$sparse_values':
                  np.array([2. / 3., 1.], dtype=np.float32),
              'scale_to_z_score$sparse_values':
                  np.array([.5 / np.sqrt(1.25), 1.5 / np.sqrt(1.25)],
                           dtype=np.float32),
              'scale_by_min_max$sparse_values':
                  np.array([2. / 3., 1.], dtype=np.float32)
          }]),
      dict(
          testcase_name='sparse_elementwise',
          input_data=[
              {
                  'idx0': [0, 1],
                  'idx1': [0, 1],
                  'val': np.array([0, 1], dtype=np.int64)
              },
              {
                  'idx0': [1, 2],
                  'idx1': [1, 3],
                  'val': np.array([2, 3], dtype=np.int64)
              },
          ],
          make_feature_spec=lambda: tf.io.SparseFeature(  # pylint: disable=g-long-lambda
              ['idx0', 'idx1'], 'val', tf.int64, (3, 4)),
          elementwise=True,
          expected_outputs=[{
              'scale_to_0_1$sparse_indices_0':
                  np.array([0, 1]),
              'scale_to_0_1$sparse_indices_1':
                  np.array([0, 1]),
              'scale_to_z_score$sparse_indices_0':
                  np.array([0, 1]),
              'scale_to_z_score$sparse_indices_1':
                  np.array([0, 1]),
              'scale_by_min_max$sparse_indices_0':
                  np.array([0, 1]),
              'scale_by_min_max$sparse_indices_1':
                  np.array([0, 1]),
              'scale_to_0_1$sparse_values':
                  np.array([0.5, 0.], dtype=np.float32),
              'scale_to_z_score$sparse_values':
                  np.array([0, -1], dtype=np.float32),
              'scale_by_min_max$sparse_values':
                  np.array([0.5, 0.], dtype=np.float32),
          }, {
              'scale_to_0_1$sparse_indices_0':
                  np.array([1, 2]),
              'scale_to_0_1$sparse_indices_1':
                  np.array([1, 3]),
              'scale_to_z_score$sparse_indices_0':
                  np.array([1, 2]),
              'scale_to_z_score$sparse_indices_1':
                  np.array([1, 3]),
              'scale_by_min_max$sparse_indices_0':
                  np.array([1, 2]),
              'scale_by_min_max$sparse_indices_1':
                  np.array([1, 3]),
              'scale_to_0_1$sparse_values':
                  np.array([1., _sigmoid(3)], dtype=np.float32),
              'scale_to_z_score$sparse_values':
                  np.array([1, 0], dtype=np.float32),
              'scale_by_min_max$sparse_values':
                  np.array([1., _sigmoid(3)], dtype=np.float32),
          }]),
      dict(
          testcase_name='ragged',
          input_data=[
              {
                  'val': [0., 2., 3.],
                  'row_lengths': [0, 3],
              },
              {
                  'val': [3., 3., 1.],
                  'row_lengths': [3],
              },
          ],
          make_feature_spec=lambda: tf.io.RaggedFeature(  # pylint: disable=g-long-lambda
              tf.float32,
              value_key='val',
              partitions=[tf.io.RaggedFeature.RowLengths('row_lengths')]),  # pytype: disable=attribute-error
          elementwise=False,
          expected_outputs=[{
              'scale_by_min_max$ragged_values': [0., 0.6666667, 1.],
              'scale_to_z_score$row_lengths_1': [0, 3],
              'scale_to_0_1$row_lengths_1': [0, 3],
              'scale_to_0_1$ragged_values': [0., 0.6666667, 1.],
              'scale_to_z_score$ragged_values': [-1.7320509, 0., 0.86602545],
              'scale_by_min_max$row_lengths_1': [0, 3],
          }, {
              'scale_to_0_1$row_lengths_1': [3],
              'scale_by_min_max$row_lengths_1': [3],
              'scale_to_z_score$ragged_values': [
                  0.86602545, 0.86602545, -0.86602545
              ],
              'scale_to_z_score$row_lengths_1': [3],
              'scale_to_0_1$ragged_values': [1., 1., 0.33333334],
              'scale_by_min_max$ragged_values': [1., 1., 0.33333334],
          }],
      ),
      dict(
          testcase_name='ragged_uniform',
          input_data=[
              {
                  'val': [0., 2., 3., 11., 2., 7.],
              },
              {
                  'val': [3., 1., 2.],
              },
          ],
          make_feature_spec=lambda: tf.io.RaggedFeature(  # pylint: disable=g-long-lambda
              tf.float32,
              value_key='val',
              partitions=[
                  tf.io.RaggedFeature.UniformRowLength(3),  # pytype: disable=attribute-error
              ]),
          elementwise=False,
          expected_outputs=[{
              'scale_by_min_max$ragged_values': [
                  0., 0.18181819, 0.27272728, 1., 0.18181819, 0.6363636
              ],
              'scale_to_z_score$ragged_values': [
                  -1.0645443, -0.4464218, -0.13736054, 2.3351295, -0.4464218,
                  1.0988845
              ],
              'scale_to_0_1$ragged_values': [
                  0., 0.18181819, 0.27272728, 1., 0.18181819, 0.6363636
              ],
          }, {
              'scale_to_0_1$ragged_values': [
                  0.27272728, 0.09090909, 0.18181819
              ],
              'scale_by_min_max$ragged_values': [
                  0.27272728, 0.09090909, 0.18181819
              ],
              'scale_to_z_score$ragged_values': [
                  -0.13736054, -0.7554831, -0.4464218
              ],
          }],
      ),
      dict(
          testcase_name='2d_ragged_uniform',
          input_data=[
              {
                  'val': [0., 2., 3., 1., 2., 7.],
                  'row_lengths': [0, 2, 0, 1],
              },
              {
                  'val': [3., 3., 1., 2.],
                  'row_lengths': [2],
              },
          ],
          make_feature_spec=lambda: tf.io.RaggedFeature(  # pylint: disable=g-long-lambda
              tf.float32,
              value_key='val',
              partitions=[
                  tf.io.RaggedFeature.RowLengths('row_lengths'),  # pytype: disable=attribute-error
                  tf.io.RaggedFeature.UniformRowLength(2),  # pytype: disable=attribute-error
              ],
              # Note that row splits are always encoded as int64 since we only
              # support this integral type in outputs. We modify the default
              # `row_splits_dtype` (tf.int32) here to make sure it still works.
              row_splits_dtype=tf.int64),
          elementwise=False,
          expected_outputs=[{
              'scale_by_min_max$ragged_values': [
                  0., 0.285714, 0.428571, 0.142857, 0.285714, 1.
              ],
              'scale_by_min_max$row_lengths_1': [0, 4, 0, 2],
              'scale_to_z_score$row_lengths_1': [0, 4, 0, 2],
              'scale_to_z_score$ragged_values': [
                  -1.3333334, -0.22222228, 0.33333328, -0.77777785, -0.22222228,
                  2.5555556
              ],
              'scale_to_0_1$row_lengths_1': [0, 4, 0, 2],
              'scale_to_0_1$ragged_values': [
                  0., 0.2857143, 0.42857143, 0.14285715, 0.2857143, 1.
              ],
          }, {
              'scale_to_0_1$ragged_values': [
                  0.42857143, 0.42857143, 0.14285715, 0.2857143
              ],
              'scale_to_0_1$row_lengths_1': [4],
              'scale_by_min_max$ragged_values': [
                  0.42857143, 0.42857143, 0.14285715, 0.2857143
              ],
              'scale_by_min_max$row_lengths_1': [4],
              'scale_to_z_score$ragged_values': [
                  0.33333328, 0.33333328, -0.77777785, -0.22222228
              ],
              'scale_to_z_score$row_lengths_1': [4],
          }],
      ),
  )
  def testNumericMappersWithCompositeInputs(self, input_data, make_feature_spec,
                                            elementwise, expected_outputs):
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tft_unit.make_feature_spec_wrapper(make_feature_spec)})

    def preprocessing_fn(inputs):
      return {
          'scale_to_0_1':
              tft.scale_to_0_1(inputs['a'], elementwise=elementwise),
          'scale_to_z_score':
              tft.scale_to_z_score(inputs['a'], elementwise=elementwise),
          'scale_by_min_max':
              tft.scale_by_min_max(inputs['a'], elementwise=elementwise),
      }

    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_outputs)

  def testNumericAnalyzersWithInputsAndAxis(self):
    self._SkipIfOutputRecordBatches()

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
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testNumericAnalyzersWithNDInputsAndAxis(self):
    self._SkipIfOutputRecordBatches()

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
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testNumericAnalyzersWithShape1NDInputsAndAxis(self):
    self._SkipIfOutputRecordBatches()

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
                               expected_outputs)

  def testNumericAnalyzersWithNDInputs(self):
    self._SkipIfOutputRecordBatches()

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
        input_data, input_metadata, analyzer_fn, expected_outputs)

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters(
      [
          dict(testcase_name='int64', input_dtype=tf.int64),
          dict(testcase_name='float32', input_dtype=tf.float32)
      ],
      [
          dict(testcase_name='scalar', input_shape=[]),
          dict(testcase_name='ND', input_shape=[2, 3])
      ],
      [
          dict(testcase_name='elementwise', reduce_instance_dims=False),
          dict(testcase_name='not_elementwise', reduce_instance_dims=True)
      ]))
  def testNumericAnalyzersWithEmptyInputs(self, input_dtype, input_shape,
                                          reduce_instance_dims):
    self._SkipIfOutputRecordBatches()

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

    input_data = []
    canonical_dtype = tft_unit.canonical_numeric_dtype(input_dtype)
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a':
            tf.io.FixedLenFeature(input_shape, canonical_dtype)
    })
    input_val_dtype = input_dtype.as_numpy_dtype
    output_shape = [] if reduce_instance_dims else input_shape
    output_dtype = canonical_dtype.as_numpy_dtype
    default_min = np.inf if input_dtype.is_floating else canonical_dtype.max
    default_max = -np.inf if input_dtype.is_floating else canonical_dtype.min
    expected_outputs = {
        'min': np.full(output_shape, default_min, output_dtype),
        'max': np.full(output_shape, default_max, output_dtype),
        'sum': np.full(output_shape, 0, output_dtype),
        'size': np.full(output_shape, 0, np.int64),
        'mean': np.full(output_shape, 0, np.float32),
        'var': np.full(output_shape, 0, np.float32),
    }
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        test_data=[{
            'a': np.zeros(input_shape, input_val_dtype)
        }, {
            'a': np.ones(input_shape, input_val_dtype)
        }])

  def testNumericMeanWithSparseTensorReduceFalseOverflow(self):
    self._SkipIfOutputRecordBatches()

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
                               expected_outputs)

  def testStringToTFIDF(self):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.compat.v1.strings.split(inputs['a']))
      out_index, out_values = tft.tfidf(
          inputs_as_ints,
          tft.get_num_buckets_for_transformed_feature(inputs_as_ints))
      return {
          'tf_idf': out_values,
          'index': out_index,
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
        expected_transformed_data, expected_metadata)

  def testTFIDFNoData(self):
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
        expected_metadata)

  def testStringToTFIDFEmptyDoc(self):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.compat.v1.strings.split(inputs['a']))
      out_index, out_values = tft.tfidf(
          inputs_as_ints,
          tft.get_num_buckets_for_transformed_feature(inputs_as_ints))
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
        expected_transformed_data, expected_metadata)

  def testIntToTFIDF(self):
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
        expected_schema)

  def testIntToTFIDFWithoutSmoothing(self):
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
        expected_schema)

  def testTFIDFWithOOV(self):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.compute_and_apply_vocabulary(
          tf.compat.v1.strings.split(inputs['a']), top_k=3)
      out_index, out_values = tft.tfidf(
          inputs_as_ints,
          tft.get_num_buckets_for_transformed_feature(inputs_as_ints) + 1)
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
        expected_metadata)

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
        expected_transformed_data, expected_metadata)

  def testCovarianceTwoDimensions(self):
    self._SkipIfOutputRecordBatches()

    def analyzer_fn(inputs):
      return {'y': tft.covariance(inputs['x'], dtype=tf.float32)}

    input_data = [{'x': x} for x in [[0, 0], [4, 0], [2, -2], [2, 2]]]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([2], tf.float32)})
    expected_outputs = {'y': np.array([[2, 0], [0, 2]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testCovarianceOneDimension(self):
    self._SkipIfOutputRecordBatches()

    def analyzer_fn(inputs):
      return {'y': tft.covariance(inputs['x'], dtype=tf.float32)}

    input_data = [{'x': x} for x in [[0], [2], [4], [6]]]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([1], tf.float32)})
    expected_outputs = {'y': np.array([[5]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testCovarianceOneDimensionWithEmptyInputs(self):
    self._SkipIfOutputRecordBatches()

    def analyzer_fn(inputs):
      return {'y': tft.covariance(inputs['x'], dtype=tf.float32)}

    input_data = []
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([1], tf.float32)})
    test_data = [{'x': [1]}, {'x': [2]}]
    expected_outputs = {'y': np.array([[0]], dtype=np.float32)}
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        test_data=test_data)

  def testPCAThreeToTwoDimensions(self):
    self._SkipIfOutputRecordBatches()

    def analyzer_fn(inputs):
      return {'y': tft.pca(inputs['x'], 2, dtype=tf.float32)}

    input_data = [{'x': x}
                  for x in  [[0, 0, 1], [4, 0, 1], [2, -1, 1], [2, 1, 1]]]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([3], tf.float32)})
    expected_outputs = {'y': np.array([[1, 0], [0, 1], [0, 0]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testPCAThreeToTwoDimensionsWithEmptyInputs(self):
    self._SkipIfOutputRecordBatches()

    def analyzer_fn(inputs):
      return {'y': tft.pca(inputs['x'], 2, dtype=tf.float32)}

    input_data = []
    test_data = [{'x': x} for x in
                 [[0, 0, 1], [4, 0, 1], [2, -1, 1], [2, 1, 1]]]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([3], tf.float32)})
    expected_outputs = {'y': np.array([[1, 0], [0, 1], [0, 0]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data,
        input_metadata,
        analyzer_fn,
        expected_outputs,
        test_data=test_data)

  class _SumCombiner(tft_beam.experimental.PTransformAnalyzer):

    def __init__(self):
      super().__init__()
      self.base_temp_dir_in_expand = None

    def _extract_outputs(self, sums):
      return [beam.pvalue.TaggedOutput('0', sums[0]),
              beam.pvalue.TaggedOutput('1', sums[1])]

    def expand(self, pcoll: beam.PCollection[Tuple[np.ndarray, np.ndarray]]):
      self.base_temp_dir_in_expand = self.base_temp_dir
      return (pcoll
              | beam.FlatMap(lambda baches: list(zip(*baches)))
              |
              beam.CombineGlobally(lambda values: np.sum(list(values), axis=0))
              | beam.FlatMap(self._extract_outputs).with_outputs('0', '1'))

  def testPTransformAnalyzer(self):
    self._SkipIfOutputRecordBatches()

    sum_combiner = self._SumCombiner()

    def analyzer_fn(inputs):
      outputs = tft.experimental.ptransform_analyzer([inputs['x'], inputs['y']],
                                                     sum_combiner,
                                                     [tf.int64, tf.int64],
                                                     [[], []])
      return {'x_sum': outputs[0], 'y_sum': outputs[1]}

    input_data = [{'x': 1, 'y': i} for i in range(100)]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.int64),
        'y': tf.io.FixedLenFeature([], tf.int64)
    })
    expected_outputs = {
        'x_sum': np.array(100, np.int64),
        'y_sum': np.array(4950, np.int64)
    }
    self.assertIsNone(sum_combiner.base_temp_dir_in_expand)
    self.assertAnalyzerOutputs(input_data, input_metadata, analyzer_fn,
                               expected_outputs)
    self.assertIsNotNone(sum_combiner.base_temp_dir_in_expand)
    self.assertStartsWith(sum_combiner.base_temp_dir_in_expand,
                          self.get_temp_dir())

  @tft_unit.named_parameters(
      dict(
          testcase_name='ArrayOutput',
          output_fn=lambda x: np.array(x, np.int64)),
      dict(testcase_name='ListOutput', output_fn=list),
  )
  def testPTransformAnalyzerMultiDimOutput(self, output_fn):
    self._SkipIfOutputRecordBatches()

    class _SimpleSumCombiner(tft_beam.experimental.PTransformAnalyzer):

      def expand(self, pcoll: beam.PCollection[Tuple[np.ndarray, np.ndarray]]):
        return (
            pcoll
            | beam.FlatMap(lambda baches: list(zip(*baches)))
            | beam.CombineGlobally(lambda values: np.sum(list(values), axis=0))
            | beam.combiners.ToList()
            | beam.Map(output_fn))

    sum_combiner = _SimpleSumCombiner()

    def analyzer_fn(inputs):
      outputs, = tft.experimental.ptransform_analyzer(
          [inputs['x'], inputs['y']], sum_combiner, [tf.int64], [[1, 2]])
      return {'x_y_sums': outputs}

    input_data = [{'x': 1, 'y': i} for i in range(100)]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.int64),
        'y': tf.io.FixedLenFeature([], tf.int64)
    })
    expected_outputs = {
        'x_y_sums': np.array([[100, 4950]], np.int64),
    }
    self.assertAnalyzerOutputs(input_data, input_metadata, analyzer_fn,
                               expected_outputs)

  @unittest.skipIf(not common.IS_ANNOTATIONS_PB_AVAILABLE,
                     'Schema annotations are not available')
  def testSavedModelWithAnnotations(self):
    """Test serialization/deserialization as a saved model with annotations."""
    self._SkipIfOutputRecordBatches()

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
    with tft_beam.Context(temp_dir=temp_dir, desired_batch_size=1):
      transform_fn = ((input_data, input_metadata)
                      | tft_beam.AnalyzeDataset(preprocessing_fn))
      #  Write transform_fn to serialize annotation collections to SavedModel
      _ = transform_fn | transform_fn_io.WriteTransformFn(temp_dir)

    # Ensure that the annotations survive the round trip to SavedModel.
    tf_transform_output = tft.TFTransformOutput(temp_dir)
    schema = tf_transform_output.transformed_metadata._schema
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
  def testSavedModelWithGlobalAnnotations(self):
    self._SkipIfOutputRecordBatches()

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
    with tft_beam.Context(temp_dir=temp_dir):
      transform_fn = ((input_data, input_metadata)
                      | tft_beam.AnalyzeDataset(preprocessing_fn))
      #  Write transform_fn to serialize annotation collections to SavedModel
      _ = transform_fn | transform_fn_io.WriteTransformFn(temp_dir)

    # Ensure that global annotations survive the round trip to SavedModel.
    tf_transform_output = tft.TFTransformOutput(temp_dir)
    schema = tf_transform_output.transformed_metadata._schema
    self.assertLen(schema.annotation.extra_metadata, 1)
    for annotation in schema.annotation.extra_metadata:
      message = annotations_pb2.BucketBoundaries()
      annotation.Unpack(message)
      self.assertAllClose(list(message.boundaries), [1])

  def testPipelineAPICounters(self):
    self._SkipIfOutputRecordBatches()

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
      with tft_beam.Context(temp_dir=self.get_temp_dir()):
        _ = ((input_data, metadata)
             | 'AnalyzeDataset' >> tft_beam.AnalyzeDataset(preprocessing_fn))

    metrics = pipeline.metrics
    self.assertMetricsCounterEqual(metrics, 'tft_analyzer_vocabulary', 1)
    self.assertMetricsCounterEqual(metrics, 'tft_mapper_scale_to_0_1', 2)
    self.assertMetricsCounterEqual(metrics,
                                   'tft_mapper_compute_and_apply_vocabulary', 1)
    # compute_and_apply_vocabulary implicitly calls apply_vocabulary.
    # We check that that call is not logged.
    self.assertMetricsCounterEqual(metrics, 'tft_mapper_apply_vocabulary', 0)

  def testNumBytesCounter(self):
    self._SkipIfOutputRecordBatches()

    test_data = [
        pa.RecordBatch.from_arrays([
            pa.array([[4]], type=pa.large_list(pa.float32())),
            pa.array([[5]], type=pa.large_list(pa.float32())),
            pa.array([['hello']], type=pa.large_list(pa.large_binary()))
        ], ['x', 'y', 'a']),
        pa.RecordBatch.from_arrays([
            pa.array([[1]], type=pa.large_list(pa.float32())),
            pa.array([[3]], type=pa.large_list(pa.float32())),
            pa.array([['world']], type=pa.large_list(pa.large_binary()))
        ], ['x', 'y', 'a'])
    ]
    tensor_representations = {
        name: text_format.Parse(
            f'dense_tensor {{ column_name: \"{name}\" shape {{}} }}',
            schema_pb2.TensorRepresentation()) for name in ('x', 'y', 'a')
    }
    expected_input_size = sum(rb.nbytes for rb in test_data)

    def preprocessing_fn(inputs):
      _ = tft.vocabulary(inputs['a'])
      return {
          'a_int': tft.compute_and_apply_vocabulary(inputs['a']),
          'x_scaled': tft.scale_to_0_1(inputs['x']),
          'y_scaled': tft.scale_to_0_1(inputs['y'])
      }

    with self._makeTestPipeline() as pipeline:
      input_data = pipeline | 'CreateTrainingData' >> beam.Create(test_data)
      tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
          test_data[0].schema, tensor_representations)

      with tft_beam.Context(temp_dir=self.get_temp_dir()):
        _ = ((input_data, tensor_adapter_config)
             | 'AnalyzeDataset' >> tft_beam.AnalyzeDataset(preprocessing_fn))

    metrics = pipeline.metrics
    self.assertMetricsCounterEqual(metrics, 'analysis_input_bytes',
                                   expected_input_size)

  def testHandleBatchError(self):
    self._SkipIfOutputRecordBatches()

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    # Exception type depends on the running being used.
    with self.assertRaisesRegexp(
        (RuntimeError, ValueError, TypeError), 'has type list'):
      # TODO(b/149997088): Remove this explicit use of DirectRunner.
      with beam.Pipeline() as pipeline:
        metadata = tft_unit.metadata_from_feature_spec({
            'x': tf.io.FixedLenFeature([], tf.float32),
        })

        input_data = pipeline | 'CreateTrainingData' >> beam.Create([{
            'x': 1
        }, {
            'x': [4, 1]
        }])
        with tft_beam.Context(temp_dir=self.get_temp_dir()):
          _ = ((input_data, metadata)
               | 'AnalyzeDataset' >> tft_beam.AnalyzeDataset(preprocessing_fn))

  def testPassthroughKeys(self):
    passthrough_key1 = '__passthrough__'
    passthrough_key2 = '__passthrough_not_in_input_record_batch__'

    def preprocessing_fn(inputs):
      self.assertNotIn(passthrough_key1, inputs)
      self.assertNotIn(passthrough_key2, inputs)
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    x_data = [0., 1., 2.]
    passthrough_data = [1, None, 3]
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[x] for x in x_data], type=pa.list_(pa.float32())),
        pa.array([None if p is None else [p] for p in passthrough_data],
                 type=pa.list_(pa.int64())),
    ], ['x', passthrough_key1])
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        input_record_batch.schema,
        {'x': text_format.Parse(
            'dense_tensor { column_name: "x" shape {} }',
            schema_pb2.TensorRepresentation())})

    with self._makeTestPipeline() as pipeline:
      input_data = (
          pipeline | beam.Create([input_record_batch]))
      with tft_beam.Context(
          temp_dir=self.get_temp_dir(),
          passthrough_keys=set([passthrough_key1, passthrough_key2])):
        (transformed_data,
         _), _ = ((input_data, tensor_adapter_config)
                  | tft_beam.AnalyzeAndTransformDataset(
                      preprocessing_fn,
                      output_record_batches=self._OutputRecordBatches()))
        expected_data = [{'x_scaled': x / 2.0, passthrough_key1: p}
                         for x, p in zip(x_data, passthrough_data)]
        beam_test_util.assert_that(
            transformed_data, self._MakeTransformOutputAssertFn(expected_data))

  def test3dSparseWithTFXIO(self):
    x_data = [0., 1., 2.]
    x_idx0 = [0, 0, 1]
    x_idx1 = [0, 0, 1]
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[x] for x in x_idx0], type=pa.list_(pa.int64())),
        pa.array([[x] for x in x_idx1], type=pa.list_(pa.int64())),
        pa.array([[x] for x in x_data], type=pa.list_(pa.float32())),
    ], ['x_idx0', 'x_idx1', 'x_val'])
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        input_record_batch.schema, {
            'x':
                text_format.Parse(
                    """
                    sparse_tensor {
                      index_column_names: ["x_idx0", "x_idx1"]
                      value_column_name: "x_val"
                      dense_shape {
                        dim {
                          size: 5
                        }
                        dim {
                          size: 5
                        }
                      }
                    }""", schema_pb2.TensorRepresentation())
        })
    expected_data = [
        {  # pylint: disable=g-complex-comprehension
            'x$sparse_values': x,
            'x$sparse_indices_0': idx0,
            'x$sparse_indices_1': idx1
        } for idx0, idx1, x in zip(x_idx0, x_idx1, x_data)
    ]

    materialize_path = os.path.join(self.get_temp_dir(), 'transformed_data')
    transform_output_path = os.path.join(self.get_temp_dir(),
                                         'transform_output')
    with self._makeTestPipeline() as pipeline:
      input_data = (pipeline | beam.Create([input_record_batch]))
      with tft_beam.Context(temp_dir=self.get_temp_dir()):
        (transformed_data, transformed_metadata), transform_fn = (
            (input_data, tensor_adapter_config)
            | tft_beam.AnalyzeAndTransformDataset(
                lambda inputs: inputs,
                output_record_batches=self._OutputRecordBatches()))
        if self._OutputRecordBatches():

          def record_batch_to_examples(data_batch):
            # Ignore unary pass-through features.
            record_batch, _ = data_batch
            return example_coder.RecordBatchToExamples(record_batch)

          transformed_and_serialized = (
              transformed_data |
              'EncodeTransformedData' >> beam.FlatMap(record_batch_to_examples))
        else:
          transformed_data_coder = tft.coders.ExampleProtoCoder(
              transformed_metadata.schema)
          transformed_and_serialized = (
              transformed_data | 'EncodeTransformedData' >> beam.Map(
                  transformed_data_coder.encode))

        _ = (
            transformed_and_serialized
            | 'Write' >> beam.io.WriteToTFRecord(
                materialize_path, shard_name_template=''))
        _ = (
            transform_fn
            | 'WriteTransformFn' >>
            tft.beam.WriteTransformFn(transform_output_path))

        expected_metadata = text_format.Parse(
            """
            feature {
              name: "x$sparse_indices_0"
              type: INT
            }
            feature {
              name: "x$sparse_indices_1"
              type: INT
            }
            feature {
              name: "x$sparse_values"
              type: FLOAT
            }
            sparse_feature {
              name: "x"
              index_feature {
                name: "x$sparse_indices_0"
              }
              index_feature {
                name: "x$sparse_indices_1"
              }
              is_sorted: true
              value_feature {
                name: "x$sparse_values"
              }
            }""", schema_pb2.Schema())
        if not tft_unit.is_external_environment():
          expected_metadata.generate_legacy_feature_spec = False

        self.assertProtoEquals(transformed_metadata.schema, expected_metadata)

        beam_test_util.assert_that(
            transformed_data, self._MakeTransformOutputAssertFn(expected_data))

        def _assert_schemas_equal_fn(schema_dict_list):
          self.assertEqual(1, len(schema_dict_list))
          self.assertProtoEquals(schema_dict_list[0].schema, expected_metadata)

        beam_test_util.assert_that(
            transformed_metadata.deferred_metadata,
            _assert_schemas_equal_fn,
            label='assert_deferred_metadata')

    with tf.Graph().as_default():
      dataset = tf.data.TFRecordDataset(materialize_path)
      tft_out = tft.TFTransformOutput(transform_output_path)
      transformed_feature_spec = tft_out.transformed_feature_spec()
      self.assertEqual(
          transformed_feature_spec, {
              'x':
                  tf.io.SparseFeature(
                      ['x$sparse_indices_0', 'x$sparse_indices_1'],
                      'x$sparse_values',
                      tf.float32, [-1, -1],
                      already_sorted=True)
          })

      def parse_fn(serialized_input):
        result = tf.io.parse_single_example(serialized_input,
                                            transformed_feature_spec)['x']
        return result.indices, result.values, result.dense_shape

      dataset = dataset.map(parse_fn).batch(len(x_data))
      transformed_sparse_components = tf.data.experimental.get_single_element(
          dataset)
      with tf.compat.v1.Session():
        transformed_sparse_components = [
            t.eval() for t in transformed_sparse_components
        ]
    expected_sparse_components = [
        np.array([[arr] for arr in zip(x_idx0, x_idx1)]),
        np.array([[x] for x in x_data]),
        np.array([[-1, -1]] * len(x_data))
    ]
    self.assertLen(transformed_sparse_components,
                   len(expected_sparse_components))
    for transformed, expected in zip(transformed_sparse_components,
                                     expected_sparse_components):
      self.assertAllEqual(expected[0], transformed[0])
      self.assertAllEqual(expected[1], transformed[1])
      self.assertAllEqual(expected[2], transformed[2])

  def testRaggedWithTFXIO(self):
    x_data = [[[1], [], [2, 3]], [[]]]
    y_data = [[[1, 2]], [[3, 4], [], [5, 6]]]
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array(x_data, type=pa.large_list(pa.large_list(pa.int64()))),
        pa.array(y_data, type=pa.large_list(pa.large_list(pa.float32())))
    ], ['x', 'y'])
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        input_record_batch.schema, {
            'x':
                text_format.Parse(
                    """ragged_tensor {
                          feature_path {
                            step: "x"
                          }
                          row_partition_dtype: INT64
                        }""", schema_pb2.TensorRepresentation()),
            'y':
                text_format.Parse(
                    """ragged_tensor {
                          feature_path {
                            step: "y"
                          }
                          row_partition_dtype: INT64
                          partition {
                            uniform_row_length: 2
                          }
                        }""", schema_pb2.TensorRepresentation())
        })

    def preprocessing_fn(inputs):
      return {
          'x_ones': tf.ones_like(inputs['x']),
          'y_ones': tf.ones_like(inputs['y'])
      }

    if common_types.is_ragged_feature_available():
      expected_data = [
          {
              'x_ones$ragged_values': [1, 1, 1],
              'x_ones$row_lengths_1': [1, 0, 2],
              'y_ones$ragged_values': [1, 1],
              'y_ones$row_lengths_1': [2],
          },
          {
              'x_ones$ragged_values': [],
              'x_ones$row_lengths_1': [0],
              'y_ones$ragged_values': [1, 1, 1, 1],
              'y_ones$row_lengths_1': [2, 0, 2],
          },
      ]
      expected_metadata = tft_unit.metadata_from_feature_spec({
          'x_ones':
              tf.io.RaggedFeature(
                  tf.int64,
                  value_key='x_ones$ragged_values',
                  partitions=[
                      tf.io.RaggedFeature.RowLengths('x_ones$row_lengths_1')  # pytype: disable=attribute-error
                  ]),
          'y_ones':
              tf.io.RaggedFeature(
                  tf.float32,
                  value_key='y_ones$ragged_values',
                  partitions=[
                      tf.io.RaggedFeature.RowLengths('y_ones$row_lengths_1'),  # pytype: disable=attribute-error
                      tf.io.RaggedFeature.UniformRowLength(2),  # pytype: disable=attribute-error
                  ]),
      })
      self.assertAnalyzeAndTransformResults([input_record_batch],
                                            tensor_adapter_config,
                                            preprocessing_fn,
                                            expected_data=expected_data,
                                            expected_metadata=expected_metadata)
    else:
      transform_output_path = os.path.join(self.get_temp_dir(),
                                           'transform_output')
      with self._makeTestPipeline() as pipeline:
        input_data = (pipeline | beam.Create([input_record_batch]))
        with tft_beam.Context(temp_dir=self.get_temp_dir()):
          transform_fn = (
              (input_data, tensor_adapter_config)
              | 'AnalyzeDataset' >> tft_beam.AnalyzeDataset(preprocessing_fn))
          _ = transform_fn | 'WriteTransform' >> tft.beam.WriteTransformFn(
              transform_output_path)

          _, transformed_metadata = transform_fn

          expected_metadata = text_format.Parse(
              """
                feature {
                  name: "x_ones"
                  type: INT
                  annotation {
                    tag: "ragged_tensor"
                  }
                }
                feature {
                  name: "y_ones"
                  type: FLOAT
                  annotation {
                    tag: "ragged_tensor"
                  }
                }""", schema_pb2.Schema())

          if not tft_unit.is_external_environment():
            expected_metadata.generate_legacy_feature_spec = False

          self.assertProtoEquals(transformed_metadata.schema, expected_metadata)

          def _assert_schemas_equal_fn(schema_dict_list):
            self.assertEqual(1, len(schema_dict_list))
            self.assertProtoEquals(schema_dict_list[0].schema,
                                   expected_metadata)

          beam_test_util.assert_that(
              transformed_metadata.deferred_metadata,
              _assert_schemas_equal_fn,
              label='assert_deferred_metadata')

      with tf.Graph().as_default():
        tft_out = tft.TFTransformOutput(transform_output_path)
        inputs = {
            'x': tf.ragged.constant([[[1], [], [2, 3]], [[1]]], dtype=tf.int64)
        }
        outputs = tft_out.transform_raw_features(inputs)
        flat_outputs = tf.nest.flatten(
            outputs['x_ones'], expand_composites=True)
        expected_flat_outputs = tf.nest.flatten(
            tf.ragged.constant([[[1], [], [1, 1]], [[1]]], dtype=tf.int64),
            expand_composites=True)
        with tf.compat.v1.Session():
          for expected, transformed in zip(expected_flat_outputs, flat_outputs):
            self.assertAllEqual(expected.eval(), transformed.eval())

  def testPipelineWithoutAutomaterialization(self):
    # Other tests pass lists instead of PCollections and thus invoke
    # automaterialization where each call to a beam PTransform will implicitly
    # run its own pipeline.
    #
    # In order to test the case where PCollections are not materialized in
    # between calls to the tf.Transform PTransforms, we include a test that is
    # not based on automaterialization.
    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    with self._makeTestPipeline() as pipeline:
      input_data = pipeline | 'CreateTrainingData' >> beam.Create(
          [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}])
      metadata = tft_unit.metadata_from_feature_spec(
          {'x': tf.io.FixedLenFeature([], tf.float32)})
      with tft_beam.Context(temp_dir=self.get_temp_dir()):
        transform_fn = (
            (input_data, metadata)
            | 'AnalyzeDataset' >> tft_beam.AnalyzeDataset(preprocessing_fn))

        # Run transform_columns on some eval dataset.
        eval_data = pipeline | 'CreateEvalData' >> beam.Create(
            [{'x': 6}, {'x': 3}])
        transformed_eval_data, _ = (
            ((eval_data, metadata), transform_fn)
            | 'TransformDataset' >> tft_beam.TransformDataset(
                output_record_batches=self._OutputRecordBatches()))
        expected_data = [{'x_scaled': 1.25}, {'x_scaled': 0.5}]
        beam_test_util.assert_that(
            transformed_eval_data,
            self._MakeTransformOutputAssertFn(expected_data, sort=True))

  def testModifyInputs(self):

    def preprocessing_fn(inputs):
      inputs['x_center'] = inputs['x'] - tft.mean(inputs['x'])
      return inputs

    input_data = [{'x': 1}, {'x': 2}, {'x': 3}, {'x': 4}, {'x': 5}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.float32)})
    expected_outputs = [{
        'x': 1,
        'x_center': -2
    }, {
        'x': 2,
        'x_center': -1
    }, {
        'x': 3,
        'x_center': 0
    }, {
        'x': 4,
        'x_center': 1
    }, {
        'x': 5,
        'x_center': 2
    }]
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_outputs)

  def testEmptySchema(self):
    with self.assertRaisesRegexp(  # pylint: disable=g-error-prone-assert-raises
        ValueError, 'The input metadata is empty.'):
      self.assertAnalyzeAndTransformResults(
          input_data=[{'x': x} for x in range(5)],
          input_metadata=tft_unit.metadata_from_feature_spec({}),
          preprocessing_fn=lambda inputs: inputs)  # pyformat: disable

  def testLoadKerasModelInPreprocessingFn(self):

    if tft_unit.is_tf_api_version_1():
      raise unittest.SkipTest(
          '`tft.make_and_track_object` is only supported when TF2 behavior is '
          'enabled.')

    def _create_model(features, target):
      inputs = [
          tf.keras.Input(shape=(1,), name=f, dtype=tf.float32) for f in features
      ]
      x = tf.keras.layers.Concatenate()(inputs)
      x = tf.keras.layers.Dense(64, activation='relu')(x)
      outputs = tf.keras.layers.Dense(1, activation='sigmoid', name=target)(x)
      model = tf.keras.Model(inputs=inputs, outputs=outputs)
      model.compile(
          loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

      n = 50
      model.fit(
          {
              f: tf.constant([np.random.uniform() for _ in range(n)
                             ]) for f in features
          },
          {target: tf.constant([np.random.randint(2) for _ in range(n)])},
      )
      return model

    test_base_dir = os.path.join(self.get_temp_dir(), self._testMethodName)
    # Create and save a test Keras model
    features = ['f1', 'f2']
    target = 't'
    keras_model = _create_model(features, target)
    keras_model_dir = os.path.join(test_base_dir, 'keras_model')
    keras_model.save(keras_model_dir)

    def preprocessing_fn(inputs):
      model = tft.make_and_track_object(
          lambda: tf.keras.models.load_model(keras_model_dir), name='keras')
      return {'prediction': model(inputs)}

    input_data = [{'f1': 1.0, 'f2': 0.0}, {'f1': 2.0, 'f2': 3.0}]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'f1': tf.io.FixedLenFeature([], tf.float32),
        'f2': tf.io.FixedLenFeature([], tf.float32)
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn)

  def test_non_deterministic_preprocessing_fn_without_name(self):

    idx = 0

    def get_features():
      nonlocal idx
      features = ['f1', 'f2', 'f3']
      result = features[idx:] + features[:idx]
      idx = 0 if idx == 2 else idx + 1
      return result

    def preprocessing_fn(inputs):
      features = get_features()

      outputs = {}
      for f in features:
        outputs[f] = inputs[f] - tft.mean(inputs[f])
      return outputs

    input_data = [{'f1': 0, 'f2': 10, 'f3': 20}, {'f1': 2, 'f2': 12, 'f3': 22}]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'f1': tf.io.FixedLenFeature([], tf.float32),
        'f2': tf.io.FixedLenFeature([], tf.float32),
        'f3': tf.io.FixedLenFeature([], tf.float32)
    })
    expected_outputs = [{
        'f1': -1,
        'f2': -1,
        'f3': -1
    }, {
        'f1': 1,
        'f2': 1,
        'f3': 1
    }]

    with contextlib.ExitStack() as stack:
      if not tft_unit.is_tf_api_version_1():
        stack.enter_context(
            self.assertRaisesRegex(
                RuntimeError, 'analyzers.*appears to be non-deterministic'))
      self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                            preprocessing_fn, expected_outputs)

  def test_non_deterministic_preprocessing_fn_with_name(self):

    idx = 0

    def get_features():
      nonlocal idx
      features = ['f1', 'f2', 'f3']
      result = features[idx:] + features[:idx]
      idx = 0 if idx == 2 else idx + 1
      return result

    def preprocessing_fn(inputs):
      features = get_features()

      outputs = {}
      for f in features:
        outputs[f] = inputs[f] - tft.mean(inputs[f], name=f)
      return outputs

    input_data = [{'f1': 0, 'f2': 10, 'f3': 20}, {'f1': 2, 'f2': 12, 'f3': 22}]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'f1': tf.io.FixedLenFeature([], tf.float32),
        'f2': tf.io.FixedLenFeature([], tf.float32),
        'f3': tf.io.FixedLenFeature([], tf.float32)
    })
    expected_outputs = [{
        'f1': -1,
        'f2': -1,
        'f3': -1
    }, {
        'f1': 1,
        'f2': 1,
        'f3': 1
    }]

    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_outputs)


if __name__ == '__main__':
  tft_unit.main()
