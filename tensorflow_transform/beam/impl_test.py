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
import pyarrow as pa
from six.moves import range
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import schema_inference
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
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


def sum_output_dtype(input_dtype):
  """Returns the output dtype for tft.sum."""
  return input_dtype if input_dtype.is_floating else tf.int64


def _mean_output_dtype(input_dtype):
  """Returns the output dtype for tft.mean (and similar functions)."""
  return tf.float64 if input_dtype == tf.float64 else tf.float32


class BeamImplTest(tft_unit.TransformTestCase):

  def setUp(self):
    tf.compat.v1.logging.info('Starting test case: %s', self._testMethodName)
    self._context = beam_impl.Context(
        use_deep_copy_optimization=True, force_tf_compat_v1=True)
    self._context.__enter__()
    super(BeamImplTest, self).setUp()

  def tearDown(self):
    self._context.__exit__()

  def _UseTFCompatV1(self):
    return True

  # This is an override that passes force_tf_compat_v1 to the overridden method.
  def assertAnalyzeAndTransformResults(self, *args, **kwargs):
    kwargs['force_tf_compat_v1'] = self._UseTFCompatV1()
    return super(BeamImplTest,
                 self).assertAnalyzeAndTransformResults(*args, **kwargs)

  # This is an override that passes force_tf_compat_v1 to the overridden method.
  def assertAnalyzerOutputs(self, *args, **kwargs):
    kwargs['force_tf_compat_v1'] = self._UseTFCompatV1()
    return super(BeamImplTest, self).assertAnalyzerOutputs(*args, **kwargs)

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
    with beam_impl.Context(use_deep_copy_optimization=with_deep_copy):
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
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transform_fn = ((input_data, input_metadata)
                      | beam_impl.AnalyzeDataset(preprocessing_fn))

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
    if not self._UseTFCompatV1():
      raise unittest.SkipTest('Test disabled when TF 2.x behavior enabled.')

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
        expected_vocab_file_contents=per_key_vocab_contents)

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
    with self.assertRaises(ValueError) as context:
      self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                            preprocessing_fn, expected_data,
                                            expected_metadata)
    self.assertTrue(
        'output_min must be less than output_max' in str(context.exception))

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

  @tft_unit.named_parameters(*_SCALE_TO_Z_SCORE_TEST_CASES)
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
                     for v in [100, 1, 12]]
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
        desired_batch_size=10)

  def testMeanAndVarPerKey(self):
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
        desired_batch_size=10)

  @tft_unit.named_parameters(
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
      {'testcase_name': '_int_sparse',
       'input_data': [{'key': [0] if x < 25 else [1]} for x in range(100)],
       'input_metadata': tft_unit.metadata_from_feature_spec(
           {'key': tf.io.VarLenFeature(tf.int64)}),
       'expected_outputs': {
           'elements': np.array([0, 1], np.int64),
           'counts': np.array([25, 75], np.int64)
       }
      },
  )  # pyformat: disable
  def testCountPerKey(self, input_data, input_metadata, expected_outputs):
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
  )
  def testHistograms(self, input_data, feature_spec, boundaries, categorical,
                     expected_outputs):
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
  def testNumericAnalyzersWithScalarInputs(
      self, input_dtype, output_dtypes):
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

  @tft_unit.parameters(*itertools.product([
      tf.int16,
      tf.int32,
      tf.int64,
      tf.float32,
      tf.float64,
      tf.uint8,
      tf.uint16,
  ], (True, False)))
  def testNumericAnalyzersWithSparseInputs(self, input_dtype,
                                           reduce_instance_dims):
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
                               expected_outputs)

  def testNumericAnalyzersWithInputsAndAxis(self):
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

  def testCovarianceTwoDimensions(self):
    def analyzer_fn(inputs):
      return {'y': tft.covariance(inputs['x'], dtype=tf.float32)}

    input_data = [{'x': x} for x in [[0, 0], [4, 0], [2, -2], [2, 2]]]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([2], tf.float32)})
    expected_outputs = {'y': np.array([[2, 0], [0, 2]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testCovarianceOneDimension(self):
    def analyzer_fn(inputs):
      return {'y': tft.covariance(inputs['x'], dtype=tf.float32)}

    input_data = [{'x': x} for x in [[0], [2], [4], [6]]]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([1], tf.float32)})
    expected_outputs = {'y': np.array([[5]], np.float32)}
    self.assertAnalyzerOutputs(
        input_data, input_metadata, analyzer_fn, expected_outputs)

  def testCovarianceOneDimensionWithEmptyInputs(self):
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

  def testPTransformAnalyzer(self):

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
        desired_batch_size=10)

  @unittest.skipIf(not common.IS_ANNOTATIONS_PB_AVAILABLE,
                     'Schema annotations are not available')
  def testSavedModelWithAnnotations(self):
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
    with beam_impl.Context(temp_dir=temp_dir, desired_batch_size=1):
      transform_fn = ((input_data, input_metadata)
                      | beam_impl.AnalyzeDataset(preprocessing_fn))
      #  Write transform_fn to serialize annotation collections to SavedModel
      _ = transform_fn | transform_fn_io.WriteTransformFn(temp_dir)

    # Ensure that the annotations survive the round trip to SavedModel.
    tf_transform_output = tft.TFTransformOutput(temp_dir)
    savedmodel_dir = tf_transform_output.transform_savedmodel_dir
    schema = beam_impl._infer_metadata_from_saved_model(
        savedmodel_dir, use_tf_compat_v1=self._UseTFCompatV1())._schema
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
    schema = beam_impl._infer_metadata_from_saved_model(
        savedmodel_dir, use_tf_compat_v1=self._UseTFCompatV1())._schema
    self.assertLen(schema.annotation.extra_metadata, 1)
    for annotation in schema.annotation.extra_metadata:
      message = annotations_pb2.BucketBoundaries()
      annotation.Unpack(message)
      self.assertAllClose(list(message.boundaries), [1])

  def testPipelineAPICounters(self):

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
      with beam_impl.Context(temp_dir=self.get_temp_dir()):
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
        with beam_impl.Context(temp_dir=self.get_temp_dir()):
          _ = ((input_data, metadata)
               | 'AnalyzeDataset' >> beam_impl.AnalyzeDataset(preprocessing_fn))

  def testPassthroughKeys(self):
    passthrough_key = '__passthrough__'

    def preprocessing_fn(inputs):
      self.assertNotIn(passthrough_key, inputs)
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    x_data = [0., 1., 2.]
    passthrough_data = [1, None, 3]
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[x] for x in x_data], type=pa.list_(pa.float32())),
        pa.array([None if p is None else [p] for p in passthrough_data],
                 type=pa.list_(pa.int64())),
    ], ['x', passthrough_key])
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        input_record_batch.schema,
        {'x': text_format.Parse(
            'dense_tensor { column_name: "x" shape {} }',
            schema_pb2.TensorRepresentation())})
    expected_data = [{'x_scaled': x / 2.0, passthrough_key: p}
                     for x, p in zip(x_data, passthrough_data)]

    with self._makeTestPipeline() as pipeline:
      input_data = (
          pipeline | beam.Create([input_record_batch]))
      with beam_impl.Context(
          temp_dir=self.get_temp_dir(),
          passthrough_keys=set([passthrough_key])):
        (transformed_data, _), _ = (
            (input_data, tensor_adapter_config)
            | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

        def _assert_fn(output_data):
          self.assertCountEqual(expected_data, output_data)

        beam_test_util.assert_that(transformed_data, _assert_fn)

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
      with beam_impl.Context(temp_dir=self.get_temp_dir()):
        (transformed_data, transformed_metadata), transform_fn = (
            (input_data, tensor_adapter_config)
            | beam_impl.AnalyzeAndTransformDataset(lambda inputs: inputs))

        transformed_data_coder = tft.coders.ExampleProtoCoder(
            transformed_metadata.schema)

        _ = (
            transformed_data
            | 'EncodeTransformedData' >> beam.Map(transformed_data_coder.encode)
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

        def _assert_fn(output_data):
          self.assertCountEqual(expected_data, output_data)

        beam_test_util.assert_that(transformed_data, _assert_fn)

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
            transformed_eval_data, equal_to(expected_data))

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


if __name__ == '__main__':
  tft_unit.main()
