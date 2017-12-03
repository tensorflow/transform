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

import os
import random


import apache_beam as beam
from apache_beam.testing import util as beam_test_util

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import analyzer_impls as beam_analyzer_impls
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

import unittest
# pylint: enable=g-import-not-at-top


class BeamImplTest(tft_unit.TransformTestCase):

  def assertMetadataEqual(self, a, b):
    # Use extra assertEqual for schemas, since full metadata assertEqual error
    # message is not conducive to debugging.
    self.assertEqual(a.schema.column_schemas, b.schema.column_schemas)
    self.assertEqual(a, b)

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
    self.assertEqual(transformed_eval_metadata,
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
      return {'ab': tf.multiply(inputs['a'], inputs['b']),
              'i': tft.string_to_int(inputs['c'])}

    num_instances = beam_impl._DEFAULT_DESIRED_BATCH_SIZE + 1
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
            sch.IntDomain(tf.int64, -1, num_instances - 1, True,
                          'vocab_string_to_int_uniques'),
            [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

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

  def testScaleUnitInterval(self):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    input_data = [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'x_scaled': 0.75
    }, {
        'x_scaled': 0.0
    }, {
        'x_scaled': 1.0
    }, {
        'x_scaled': 0.25
    }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled':
            sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testScaleMinMax(self):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_by_min_max(inputs['x'], -1, 1)}

    input_data = [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'x_scaled': 0.5
    }, {
        'x_scaled': -1.0
    }, {
        'x_scaled': 1.0
    }, {
        'x_scaled': -0.5
    }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled':
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

  def testScaleToZScore_int64(self):
    self._testScaleToZScore(input_dtype=tf.int64, output_dtype=tf.float64)

  def testScaleToZScore_int32(self):
    self._testScaleToZScore(input_dtype=tf.int32, output_dtype=tf.float64)

  def testScaleToZScore_int16(self):
    self._testScaleToZScore(input_dtype=tf.int16, output_dtype=tf.float32)

  def testScaleToZScore_float64(self):
    self._testScaleToZScore(input_dtype=tf.float64, output_dtype=tf.float64)

  def testScaleToZScore_float32(self):
    self._testScaleToZScore(input_dtype=tf.float32, output_dtype=tf.float32)

  def _testScaleToZScore(self, input_dtype, output_dtype):

    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_z_score(inputs['x'])}

    input_data = [{'x': -4}, {'x': 10}, {'x': 2}, {'x': 4}]
    # Mean: 3
    # Var: (7^2 + 7^2 + 1^2 + 1^2) / 4 = 25
    # Std Dev: 5
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(input_dtype, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'x_scaled': -1.4  # (-4 - 3) / 5
    }, {
        'x_scaled': 1.4  # (10 - 3) / 5
    }, {
        'x_scaled': -0.2  # (2 - 3) / 5
    }, {
        'x_scaled': 0.2  # (4 - 3) / 5
    }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'x_scaled':
            sch.ColumnSchema(output_dtype, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testNumericAnalyzersWithScalarInputs_int64(self):
    self._testNumericAnalyzersWithScalarInputs(
        input_dtype=tf.int64,
        output_dtypes={
            'min': tf.int64,
            'max': tf.int64,
            'sum': tf.int64,
            'size': tf.int64,
            'mean': tf.float64,
            'var': tf.float64
        }
    )

  def testNumericAnalyzersWithScalarInputs_int32(self):
    self._testNumericAnalyzersWithScalarInputs(
        input_dtype=tf.int32,
        output_dtypes={
            'min': tf.int32,
            'max': tf.int32,
            'sum': tf.int32,
            'size': tf.int32,
            'mean': tf.float64,
            'var': tf.float64
        }
    )

  def testNumericAnalyzersWithScalarInputs_int16(self):
    self._testNumericAnalyzersWithScalarInputs(
        input_dtype=tf.int16,
        output_dtypes={
            'min': tf.int16,
            'max': tf.int16,
            'sum': tf.int16,
            'size': tf.int16,
            'mean': tf.float32,
            'var': tf.float32
        }
    )

  def testNumericAnalyzersWithScalarInputs_float64(self):
    self._testNumericAnalyzersWithScalarInputs(
        input_dtype=tf.float64,
        output_dtypes={
            'min': tf.float64,
            'max': tf.float64,
            'sum': tf.float64,
            'size': tf.float64,
            'mean': tf.float64,
            'var': tf.float64
        }
    )

  def testNumericAnalyzersWithScalarInputs_float32(self):
    self._testNumericAnalyzersWithScalarInputs(
        input_dtype=tf.float32,
        output_dtypes={
            'min': tf.float32,
            'max': tf.float32,
            'sum': tf.float32,
            'size': tf.float32,
            'mean': tf.float32,
            'var': tf.float32
        }
    )

  def _testNumericAnalyzersWithScalarInputs(self, input_dtype, output_dtypes):
    def preprocessing_fn(inputs):
      def repeat(in_tensor, value):
        batch_size = tf.shape(in_tensor)[0]
        return tf.ones([batch_size], dtype=value.dtype) * value

      return {
          'min': repeat(inputs['a'], tft.min(inputs['a'])),
          'max': repeat(inputs['a'], tft.max(inputs['a'])),
          'sum': repeat(inputs['a'], tft.sum(inputs['a'])),
          'size': repeat(inputs['a'], tft.size(inputs['a'])),
          'mean': repeat(inputs['a'], tft.mean(inputs['a'])),
          'var': repeat(inputs['a'], tft.var(inputs['a']))
      }

    input_data = [{'a': 4}, {'a': 1}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(input_dtype, [], sch.FixedColumnRepresentation())
    })
    expected_data = [
        {'min': 1, 'max': 4, 'sum': 5, 'size': 2, 'mean': 2.5, 'var': 2.25},
        {'min': 1, 'max': 4, 'sum': 5, 'size': 2, 'mean': 2.5, 'var': 2.25}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'min': sch.ColumnSchema(output_dtypes['min'], [],
                                sch.FixedColumnRepresentation()),
        'max': sch.ColumnSchema(output_dtypes['max'], [],
                                sch.FixedColumnRepresentation()),
        'sum': sch.ColumnSchema(output_dtypes['sum'], [],
                                sch.FixedColumnRepresentation()),
        'size': sch.ColumnSchema(output_dtypes['size'], [],
                                 sch.FixedColumnRepresentation()),
        'mean': sch.ColumnSchema(output_dtypes['mean'], [],
                                 sch.FixedColumnRepresentation()),
        'var': sch.ColumnSchema(output_dtypes['var'], [],
                                sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testNumericAnalyzersWithInputsAndAxis(self):
    def preprocessing_fn(inputs):
      def repeat(in_tensor, value):
        batch_size = tf.shape(in_tensor)[0]
        return tf.ones([batch_size, 1], dtype=value.dtype) * value

      return {
          'min':
              repeat(inputs['a'],
                     tft.min(inputs['a'], reduce_instance_dims=False)),
          'max':
              repeat(inputs['a'],
                     tft.max(inputs['a'], reduce_instance_dims=False)),
          'sum':
              repeat(inputs['a'],
                     tft.sum(inputs['a'], reduce_instance_dims=False)),
          'size':
              repeat(inputs['a'],
                     tft.size(inputs['a'], reduce_instance_dims=False)),
          'mean':
              repeat(inputs['a'],
                     tft.mean(inputs['a'], reduce_instance_dims=False)),
          'var':
              repeat(inputs['a'],
                     tft.var(inputs['a'], reduce_instance_dims=False))
      }

    input_data = [
        {'a': [8, 9, 3, 4]},
        {'a': [1, 2, 10, 11]}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [4], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'min': [1, 2, 3, 4],
        'max': [8, 9, 10, 11],
        'sum': [9, 11, 13, 15],
        'size': [2, 2, 2, 2],
        'mean': [4.5, 5.5, 6.5, 7.5],
        'var': [12.25, 12.25, 12.25, 12.25]
    }, {
        'min': [1, 2, 3, 4],
        'max': [8, 9, 10, 11],
        'sum': [9, 11, 13, 15],
        'size': [2, 2, 2, 2],
        'mean': [4.5, 5.5, 6.5, 7.5],
        'var': [12.25, 12.25, 12.25, 12.25]
    }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'min': sch.ColumnSchema(
            tf.int64, [4], sch.FixedColumnRepresentation()),
        'max': sch.ColumnSchema(
            tf.int64, [4], sch.FixedColumnRepresentation()),
        'sum': sch.ColumnSchema(
            tf.int64, [4], sch.FixedColumnRepresentation()),
        'size': sch.ColumnSchema(
            tf.int64, [4], sch.FixedColumnRepresentation()),
        'mean': sch.ColumnSchema(
            tf.float64, [4], sch.FixedColumnRepresentation()),
        'var': sch.ColumnSchema(
            tf.float64, [4], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testNumericAnalyzersWithNDInputsAndAxis(self):
    def preprocessing_fn(inputs):
      def repeat(in_tensor, value):
        batch_size = tf.shape(in_tensor)[0]
        expand = tf.expand_dims(value, 0)
        return tf.tile(expand, [batch_size, 1, 1])

      return {
          'min': repeat(inputs['a'],
                        tft.min(inputs['a'], reduce_instance_dims=False)),
          'max': repeat(inputs['a'],
                        tft.max(inputs['a'], reduce_instance_dims=False)),
          'sum': repeat(inputs['a'],
                        tft.sum(inputs['a'], reduce_instance_dims=False)),
          'size': repeat(inputs['a'],
                         tft.size(inputs['a'], reduce_instance_dims=False)),
          'mean': repeat(inputs['a'],
                         tft.mean(inputs['a'], reduce_instance_dims=False)),
          'var': repeat(inputs['a'],
                        tft.var(inputs['a'], reduce_instance_dims=False))
      }

    input_data = [
        {'a': [[8, 9], [3, 4]]},
        {'a': [[1, 2], [10, 11]]}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [2, 2], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'min': [[1, 2], [3, 4]],
        'max': [[8, 9], [10, 11]],
        'sum': [[9, 11], [13, 15]],
        'size': [[2, 2], [2, 2]],
        'mean': [[4.5, 5.5], [6.5, 7.5]],
        'var': [[12.25, 12.25], [12.25, 12.25]]
    }, {
        'min': [[1, 2], [3, 4]],
        'max': [[8, 9], [10, 11]],
        'sum': [[9, 11], [13, 15]],
        'size': [[2, 2], [2, 2]],
        'mean': [[4.5, 5.5], [6.5, 7.5]],
        'var': [[12.25, 12.25], [12.25, 12.25]]
    }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'min': sch.ColumnSchema(
            tf.int64, [2, 2], sch.FixedColumnRepresentation()),
        'max': sch.ColumnSchema(
            tf.int64, [2, 2], sch.FixedColumnRepresentation()),
        'sum': sch.ColumnSchema(
            tf.int64, [2, 2], sch.FixedColumnRepresentation()),
        'size': sch.ColumnSchema(
            tf.int64, [2, 2], sch.FixedColumnRepresentation()),
        'mean': sch.ColumnSchema(
            tf.float64, [2, 2], sch.FixedColumnRepresentation()),
        'var': sch.ColumnSchema(
            tf.float64, [2, 2], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testNumericAnalyzersWithNDInputs(self):
    def preprocessing_fn(inputs):
      def repeat(in_tensor, value):
        batch_size = tf.shape(in_tensor)[0]
        return tf.ones([batch_size], value.dtype) * value

      return {
          'min': repeat(inputs['a'], tft.min(inputs['a'])),
          'max': repeat(inputs['a'], tft.max(inputs['a'])),
          'sum': repeat(inputs['a'], tft.sum(inputs['a'])),
          'size': repeat(inputs['a'], tft.size(inputs['a'])),
          'mean': repeat(inputs['a'], tft.mean(inputs['a'])),
          'var': repeat(inputs['a'], tft.var(inputs['a']))
      }

    input_data = [
        {'a': [[4, 5], [6, 7]]},
        {'a': [[1, 2], [3, 4]]}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.int64, [2, 2], sch.FixedColumnRepresentation())
    })
    expected_data = [
        {'min': 1, 'max': 7, 'sum': 32, 'size': 8, 'mean': 4.0, 'var': 3.5},
        {'min': 1, 'max': 7, 'sum': 32, 'size': 8, 'mean': 4.0, 'var': 3.5}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'min': sch.ColumnSchema(tf.int64, [], sch.FixedColumnRepresentation()),
        'max': sch.ColumnSchema(tf.int64, [], sch.FixedColumnRepresentation()),
        'sum': sch.ColumnSchema(tf.int64, [], sch.FixedColumnRepresentation()),
        'size': sch.ColumnSchema(tf.int64, [], sch.FixedColumnRepresentation()),
        'mean': sch.ColumnSchema(tf.float64, [],
                                 sch.FixedColumnRepresentation()),
        'var': sch.ColumnSchema(tf.float64, [],
                                sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

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
          return {'mean': repeat(inputs['a'], tft.mean(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(mean_fn)

      with self.assertRaises(TypeError):
        def var_fn(inputs):
          return {'var': repeat(inputs['a'], tft.var(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(var_fn)

  def testStringToTFIDF(self):
    def preprocessing_fn(inputs):
      inputs_as_ints = tft.string_to_int(tf.string_split(inputs['a']))
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
      inputs_as_ints = tft.string_to_int(tf.string_split(inputs['a']))
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
      inputs_as_ints = tft.string_to_int(tf.string_split(inputs['a']))
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
      inputs_as_ints = tft.string_to_int(tf.string_split(inputs['a']),
                                         top_k=test_vocab_size)
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

  def testUniquesAnalyzer(self):
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
            sch.IntDomain(tf.int64, -1, 4, True,
                          'vocab_string_to_int_uniques'),
            [], sch.FixedColumnRepresentation())
    })

    # Assert empty string with default_value=-1
    def preprocessing_fn(inputs):
      return {
          'index': tft.string_to_int(inputs['a'])
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
        # The empty string maps to string_to_int(default_value=-1).
        {'index': -1},
        # The tokens that contain \n map to string_to_int(default_value=-1).
        {'index': -1},
        {'index': -1},
        {'index': -1}
    ]
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

    # Assert empty string with num_oov_buckets=1
    def preprocessing_fn_oov(inputs):
      return {
          'index': tft.string_to_int(inputs['a'], num_oov_buckets=1)
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
            sch.IntDomain(tf.int64, 0, 5, True, 'vocab_string_to_int_uniques'),
            [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn_oov, expected_data,
        expected_metadata)

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
    vocab_filename = 'test_string_to_int'
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index_a': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True, vocab_filename),
            [], sch.FixedColumnRepresentation()),
        'index_b': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True, vocab_filename),
            [], sch.FixedColumnRepresentation())
    })

    def preprocessing_fn(inputs):
      deferred_vocab_and_filename = tft.uniques(
          tf.concat([inputs['a'],
                     inputs['b'],
                     inputs['c']], 0),
          vocab_filename=vocab_filename)
      return {
          'index_a': tft.apply_vocab(inputs['a'], deferred_vocab_and_filename),
          'index_b': tft.apply_vocab(inputs['b'], deferred_vocab_and_filename)
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
            sch.IntDomain(tf.int64, -1, 6, True, vocab_filename),
            [], sch.FixedColumnRepresentation()),
        'frequency_a': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True, vocab_filename),
            [], sch.FixedColumnRepresentation()),
        'index_b': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True, vocab_filename),
            [], sch.FixedColumnRepresentation()),
        'frequency_b': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 6, True, vocab_filename),
            [], sch.FixedColumnRepresentation())
    })

    def preprocessing_fn(inputs):
      deferred_vocab_and_filename = tft.uniques(
          tf.concat([inputs['a'],
                     inputs['b'],
                     inputs['c']], 0),
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
          'index_a': tft.apply_vocab(inputs['a'],
                                     deferred_vocab_and_filename,
                                     lookup_fn=_apply_vocab),
          'frequency_a': tft.apply_vocab(inputs['a'],
                                         deferred_vocab_and_filename,
                                         lookup_fn=_apply_frequency),
          'index_b': tft.apply_vocab(inputs['b'],
                                     deferred_vocab_and_filename,
                                     lookup_fn=_apply_vocab),
          'frequency_b': tft.apply_vocab(inputs['b'],
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
          'index': tft.string_to_int(inputs['a']),
          'index_2': tft.string_to_int(
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
            sch.IntDomain(tf.int64, -1, 1, True,
                          'vocab_string_to_int_uniques'), [],
            sch.FixedColumnRepresentation()),
        'index_2': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 1, True, 'index_2_file'), [],
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
    self.assertEqual(['index_2_file', 'vocab_string_to_int_uniques'],
                     sorted(os.listdir(assets_path)))

    # Verify that the paths are actually there.
    for asset_tensor in asset_tensors.values():
      self.assertTrue(os.path.isfile(asset_tensor))

  def testUniquesAnalyzerWithNDInputs(self):
    def preprocessing_fn(inputs):
      return {
          'index': tft.string_to_int(inputs['a'])
      }

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
            sch.IntDomain(tf.int64, -1, 8, True,
                          'vocab_string_to_int_uniques'),
            [2, 2], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testUniquesAnalyzerWithTokenization(self):
    def preprocessing_fn(inputs):
      return {
          'index': tft.string_to_int(tf.string_split(inputs['a']))
      }

    input_data = [{'a': 'hello hello world'}, {'a': 'hello goodbye world'}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{'index': [0, 0, 1]}, {'index': [0, 2, 1]}]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -1, 2, True,
                          'vocab_string_to_int_uniques'),
            [None], sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testUniquesAnalyzerWithTopK(self):
    def preprocessing_fn(inputs):
      return {
          'index1': tft.string_to_int(tf.string_split(inputs['a']),
                                      default_value=-99, top_k=2),

          # As above but using a string for top_k (and changing the
          # default_value to showcase things).
          'index2': tft.string_to_int(tf.string_split(inputs['a']),
                                      default_value=-9, top_k='2')
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
            sch.IntDomain(tf.int64, -99, 1, True,
                          'vocab_string_to_int_uniques'),
            [None], sch.ListColumnRepresentation()),
        'index2': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -9, 1, True,
                          'vocab_string_to_int_1_uniques'),
            [None], sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testUniquesAnalyzerWithFrequencyThreshold(self):
    def preprocessing_fn(inputs):
      return {
          'index1': tft.string_to_int(tf.string_split(inputs['a']),
                                      default_value=-99, frequency_threshold=2),

          # As above but using a string for frequency_threshold (and changing
          # the default_value to showcase things).
          'index2': tft.string_to_int(tf.string_split(inputs['a']),
                                      default_value=-9, frequency_threshold='2')
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
            sch.IntDomain(tf.int64, -99, 2, True,
                          'vocab_string_to_int_uniques'),
            [None], sch.ListColumnRepresentation()),
        'index2': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -9, 2, True,
                          'vocab_string_to_int_1_uniques'),
            [None], sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testUniquesAnalyzerWithFrequencyThresholdTooHigh(self):
    # Expected to return an empty dict due to too high threshold.
    def preprocessing_fn(inputs):
      return {
          'index1':
              tft.string_to_int(
                  tf.string_split(inputs['a']),
                  default_value=-99,
                  frequency_threshold=77),

          # As above but using a string for frequency_threshold (and changing
          # the default_value to showcase things).
          'index2':
              tft.string_to_int(
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
            sch.IntDomain(tf.int64, -99, 0, True,
                          'vocab_string_to_int_uniques'),
            [None], sch.ListColumnRepresentation()),
        'index2': sch.ColumnSchema(
            sch.IntDomain(tf.int64, -9, 0, True,
                          'vocab_string_to_int_1_uniques'),
            [None], sch.ListColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def testUniquesAnalyzerWithHighFrequencyThresholdAndOOVBuckets(self):
    def preprocessing_fn(inputs):
      return {
          'index1':
              tft.string_to_int(
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
            sch.IntDomain(tf.int64, 0, 3, True,
                          'vocab_string_to_int_uniques'), [None],
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
            tf.float32, [None], sch.ListColumnRepresentation())
    })
    data = [
        {'x': 4, 'y': ([0, 1], [0., 1.]), 'z': [2., 4., 6.]},
        {'x': 1, 'y': ([2, 3], [2., 3.]), 'z': [8.]},
        {'x': 5, 'y': ([4, 5], [4., 5.]), 'z': [1., 2., 3.]}
    ]
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      _, transform_fn = (
          (data, metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    export_dir = os.path.join(self.get_temp_dir(), 'export')
    _ = transform_fn | transform_fn_io.WriteTransformFn(export_dir)

    # Load the exported graph, and apply it to a batch of data.
    g = tf.Graph()
    with g.as_default():
      with tf.Session():
        inputs, outputs = saved_transform_io.partially_apply_saved_transform(
            os.path.join(export_dir, 'transform_fn'), {})
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
              dense_shape=[4, 10])
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
              dense_shape=[4, 10])
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

  def _test_bucketization_helper(
      self, test_inputs, expected_boundaries, input_dtype, do_shuffle=False,
      epsilon=None):
    # Shuffle the input to add randomness to input generated with
    # simple range().
    if do_shuffle:
      random.shuffle(test_inputs)

    def preprocessing_fn(inputs):
      return {
          'q_b': tft.bucketize(inputs['x'],
                               num_buckets=len(expected_boundaries)+1,
                               epsilon=epsilon)
      }

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
            tf.int64, [1], sch.FixedColumnRepresentation())
    })

    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

  def _test_bucketization_for_type(self, input_dtype):
    self._test_bucketization_helper(range(1, 10), [4, 7], input_dtype)

    self._test_bucketization_helper(range(1, 100), [26, 51, 76], input_dtype)

    # The following is similar to range(1, 100) test above, except that
    # only odd numbers are in the input; so boundaries differ (26 -> 27 and
    # 76 -> 77).
    self._test_bucketization_helper(range(1, 100, 2), [27, 51, 77], input_dtype)

    # Test some inversely sorted inputs, and with different strides, and
    # boundaries/buckets.
    self._test_bucketization_helper(range(9, 0, -1), [4, 7], input_dtype)
    self._test_bucketization_helper(range(19, 0, -1), [11], input_dtype)
    self._test_bucketization_helper(range(99, 0, -1), [51], input_dtype)
    self._test_bucketization_helper(range(99, 0, -1), [34, 67], input_dtype)
    self._test_bucketization_helper(range(99, 0, -2), [34, 68], input_dtype)
    self._test_bucketization_helper(
        range(99, 0, -1), [11, 21, 31, 41, 51, 61, 71, 81, 91], input_dtype)

    # These tests do a random shuffle of the inputs, which must not affect the
    # boundaries (or the computed buckets)
    self._test_bucketization_helper(
        range(99, 0, -1), [11, 21, 31, 41, 51, 61, 71, 81, 91], input_dtype,
        do_shuffle=True)
    self._test_bucketization_helper(
        range(1, 100), [11, 21, 31, 41, 51, 61, 71, 81, 91], input_dtype,
        do_shuffle=True)

    # The following test is with multiple batches (3 batches with default
    # batch of 1000).
    self._test_bucketization_helper(range(1, 3000), [1503], input_dtype)
    self._test_bucketization_helper(range(1, 3000), [1001, 2001], input_dtype)

    # Test with specific error for bucket boundaries. This is same as
    # the test above with 3 batches and a single boundary, but with a stricter
    # error tolerance (0.001) than the default error (0.01). The result is that
    # the computed boundary in the test below is closer to the middle (1501)
    # than that computed by the boundary of 1503 above.
    self._test_bucketization_helper(range(1, 3000),
                                    [1501],
                                    input_dtype,
                                    epsilon=0.001)
    # Test with specific error for bucket boundaries, with more relaxed error
    # tolerance (0.1) than the default (0.01). Now the boundary diverges further
    # to 1519 (compared to boundary of 1501 with error 0.001, and boundary of
    # 1503 with error 0.01).
    self._test_bucketization_helper(range(1, 3000),
                                    [1519],
                                    input_dtype,
                                    epsilon=0.1)

  # Test for all integral types, each type is in a separate testcase to
  # increase parallelism of test shards (and reduce test time from ~250 seconds
  # to ~80 seconds)
  def testBucketizationInt32(self):
    self._test_bucketization_for_type(tf.int32)

  def testBucketizationInt64(self):
    self._test_bucketization_for_type(tf.int64)

  def testBucketizationFloat32(self):
    self._test_bucketization_for_type(tf.float32)

  def testBucketizationFloat64(self):
    self._test_bucketization_for_type(tf.float64)

  def testBucketizationDouble(self):
    self._test_bucketization_for_type(tf.double)

  def testBucketizationFloat16Failure(self):
    with self.assertRaisesRegexp(
        TypeError,
        '.*DataType float16 not in list of allowed values.*'):
      self._test_bucketization_for_type(tf.float16)

  def _test_quantile_buckets_helper(self, input_dtype):

    def preprocessing_fn(inputs):
      return {'q_b': tft.quantiles(inputs['x'], num_buckets=3, epsilon=0.00001)}

    # Current batch size is 1000, force 3 batches.
    input_data = []
    for x in range(1, 3000):
      input_data += [{'x': [x]}]

    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(input_dtype, [1], sch.FixedColumnRepresentation())
    })
    # The expected data has 2 boundaries that divides the data into 3 buckets.
    expected_data = [{'q_b': [1001, 2001]},
                     {'q_b': [1001, 2001]},
                     {'q_b': [1001, 2001]}]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'q_b': sch.ColumnSchema(
            tf.float32, [None], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  # Test for all integral types, each type is in a separate testcase to
  # increase parallelism of test shards and reduce test time.
  def testQuantileBuckets_int32(self):
    self._test_quantile_buckets_helper(tf.int32)

  def testQuantileBuckets_int64(self):
    self._test_quantile_buckets_helper(tf.int64)

  def testQuantileBuckets_float32(self):
    self._test_quantile_buckets_helper(tf.float32)

  def testQuantileBuckets_float64(self):
    self._test_quantile_buckets_helper(tf.float64)

  def testQuantileBuckets_double(self):
    self._test_quantile_buckets_helper(tf.double)


  def testUniquesWithFrequency(self):
    outfile = 'uniques_vocab_with_frequency'
    def preprocessing_fn(inputs):

      # Force the analyzer to be executed, and store the frequency file as a
      # side-effect.
      _ = tft.uniques(
          inputs['a'], vocab_filename=outfile, store_frequency=True)
      _ = tft.uniques(
          inputs['a'], store_frequency=True)
      _ = tft.uniques(
          inputs['b'], store_frequency=True)

      # The following must not produce frequency output, just the vocab words.
      _ = tft.uniques(inputs['b'])
      a_int = tft.string_to_int(inputs['a'])

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
    self.assertItemsEqual([outfile,
                           'vocab_frequency_uniques_1',
                           'vocab_frequency_uniques_2',
                           'vocab_string_to_int_uniques',
                           'vocab_uniques_3'],
                          os.listdir(assets_path))

    check_asset_file_contents(assets_path, outfile,
                              '2 hello\n1 world\n')

    check_asset_file_contents(assets_path, 'vocab_frequency_uniques_1',
                              '2 hello\n1 world\n')

    check_asset_file_contents(assets_path, 'vocab_frequency_uniques_2',
                              '2 ho ho\n1 hi\n')

    check_asset_file_contents(assets_path, 'vocab_uniques_3',
                              'ho ho\nhi\n')

    check_asset_file_contents(assets_path, 'vocab_string_to_int_uniques',
                              'hello\nworld\n')

  # Example to demonstrate QuantileCombiner which implements combiner methods
  # such as add_input() and merge_accumulators() that achieve computation
  # through TF Graph and execution using TF Sesssion, e.g. graphs that contain
  # TF Quantile Ops.
  #
  # Note this wraps a beam_analyzer_impls._ComputeQuantiles which is a
  # beam.CombineFn, as a _CombinerSpec, which then gets re-wrapped as a
  # beam.CombineFn.  This is roundabout but necessary to maintain this test
  # until we update the actual quantiles implementation.
  class _QuantileCombinerSpec(tft.CombinerSpec):

    def __init__(self):
      self._impl = beam_analyzer_impls._ComputeQuantiles(
          num_quantiles=2, epsilon=0.00001)

    def create_accumulator(self):
      return self._impl.create_accumulator()

    def add_input(self, summary, next_input):
      return self._impl.add_input(summary, next_input)

    def merge_accumulators(self, accumulators):
      return self._impl.merge_accumulators(accumulators)

    def extract_output(self, accumulator):
      return self._impl.extract_output(accumulator)

  def testQuantileViaCombineAnalyzer(self):

    def preprocessing_fn(inputs):
      return {
          'buckets': tft.combine_analyzer(
              inputs['x'], output_dtype=tf.int32, output_shape=[None],
              combiner_spec=self._QuantileCombinerSpec(), name='quantiles'),
      }

    input_dtype = tf.int32
    input_data = []
    for x in range(1, 1000):
      input_data += [{'x': [x]}]

    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(input_dtype, [1], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'buckets': [1, 501, 999]
    }]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'buckets': sch.ColumnSchema(
            tf.int32, [], sch.FixedColumnRepresentation()),
    })
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)


if __name__ == '__main__':
  unittest.main()
