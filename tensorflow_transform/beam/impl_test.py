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


# pylint: disable=g-import-not-at-top
import apache_beam as beam
try:
  from apache_beam.testing import util as beam_test_util
except ImportError:
  from apache_beam.transforms import util as beam_test_util

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema as sch

import unittest
from tensorflow.python.framework import test_util
# pylint: enable=g-import-not-at-top


class BeamImplTest(test_util.TensorFlowTestCase):

  def assertDataEqual(self, a_data, b_data):
    self.assertEqual(len(a_data), len(b_data),
                     'len(%r) != len(%r)' % (a_data, b_data))
    for a_row, b_row in zip(a_data, b_data):
      self.assertItemsEqual(a_row.keys(), b_row.keys())
      for key in a_row.keys():
        a_value = a_row[key]
        b_value = b_row[key]
        if isinstance(a_value, tuple):
          self.assertValuesCloseOrEqual(a_value[0], b_value[0])
          self.assertValuesCloseOrEqual(a_value[1], b_value[1])
        else:
          self.assertValuesCloseOrEqual(a_value, b_value)

  def assertValuesCloseOrEqual(self, a_value, b_value):
    if (isinstance(a_value, str) or
        isinstance(a_value, list) and a_value and isinstance(a_value[0], str)):
      self.assertAllEqual(a_value, b_value)
    else:
      self.assertAllClose(a_value, b_value)

  def assertAnalyzeAndTransformResults(
      self, input_data, input_metadata, preprocessing_fn, expected_data,
      expected_metadata):
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      # Note: we don't separately test AnalyzeDataset and TransformDataset as
      # AnalyzeAndTransformDataset currently simply composes these two
      # transforms.  If in future versions of the code, the implementation
      # differs, we should also run AnalyzeDataset and TransformDatset composed.
      (transformed_data, transformed_metadata), _ = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    self.assertDataEqual(expected_data, transformed_data)
    # Use extra assertEqual for schemas, since full metadata assertEqual error
    # message is not conducive to debugging.
    self.assertEqual(
        expected_metadata.schema.column_schemas,
        transformed_metadata.schema.column_schemas)
    self.assertEqual(expected_metadata, transformed_metadata)

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
    self.assertDataEqual(transformed_eval_data, expected_transformed_eval_data)
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

    input_data = [{
        'a': 2,
        'b': i,
        'c': '%.10i' % i,  # Front-padded to facilitate lexicographic sorting.
    } for i in range(beam_impl._DEFAULT_DESIRED_BATCH_SIZE + 1)]
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
        'i': sch.ColumnSchema(tf.int64, [], sch.FixedColumnRepresentation())
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
      return {'x_scaled': tft.scale_by_min_max(inputs['x'], -10, 10)}

    input_data = [{'x': 4}, {'x': 4}, {'x': 4}, {'x': 4}]
    input_metadata = dataset_metadata.DatasetMetadata({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    expected_data = [{
        'x_scaled': float('nan')
    }, {
        'x_scaled': float('nan')
    }, {
        'x_scaled': float('nan')
    }, {
        'x_scaled': float('nan')
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

  def testNumericAnalyzersWithScalarInputs_int64(self):
    self.numericAnalyzersWithScalarInputs(
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
    self.numericAnalyzersWithScalarInputs(
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
    self.numericAnalyzersWithScalarInputs(
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
    self.numericAnalyzersWithScalarInputs(
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
    self.numericAnalyzersWithScalarInputs(
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

  def numericAnalyzersWithScalarInputs(self, input_dtype, output_dtypes):
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
    log_4_over_2 = 0.69314718056
    log_4_over_3 = 0.28768207245
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

    log_5_over_2 = 0.91629073187
    log_5_over_3 = 0.51082562376
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
        'a': sch.ColumnSchema(tf.int64, [], sch.ListColumnRepresentation())})
    log_4_over_2 = 0.69314718056
    log_4_over_3 = 0.28768207245
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
        'a': sch.ColumnSchema(tf.int64, [], sch.ListColumnRepresentation())})
    log_3_over_2 = 0.4054651081
    log_3 = 1.0986122886
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
    log_4_over_2 = 0.69314718056
    log_4_over_3 = 0.28768207245
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
        'a': sch.ColumnSchema(tf.int64, [], sch.ListColumnRepresentation())})

    log_4_over_2 = 0.69314718056
    log_4_over_3 = 0.28768207245
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
    def preprocessing_fn(inputs):
      return {
          'index': tft.string_to_int(inputs['a'])
      }

    input_data = [
        {'a': 'hello'},
        {'a': 'world'},
        {'a': 'hello'},
        {'a': 'hello'},
        {'a': 'goodbye'},
        {'a': 'world'},
        {'a': 'aaaaa'}
    ]
    input_metadata = dataset_metadata.DatasetMetadata({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    expected_data = [
        {'index': 0},
        {'index': 1},
        {'index': 0},
        {'index': 0},
        {'index': 2},
        {'index': 1},
        {'index': 3}
    ]
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index': sch.ColumnSchema(tf.int64, [], sch.FixedColumnRepresentation())
    })
    self.assertAnalyzeAndTransformResults(
        input_data, input_metadata, preprocessing_fn, expected_data,
        expected_metadata)

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
        'index': sch.ColumnSchema(tf.int64, [2, 2],
                                  sch.FixedColumnRepresentation())
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
        'index': sch.ColumnSchema(tf.int64, [None],
                                  sch.ListColumnRepresentation())
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
        'index1': sch.ColumnSchema(tf.int64, [None],
                                   sch.ListColumnRepresentation()),
        'index2': sch.ColumnSchema(tf.int64, [None],
                                   sch.ListColumnRepresentation())
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
        'index1': sch.ColumnSchema(tf.int64, [None],
                                   sch.ListColumnRepresentation()),
        'index2': sch.ColumnSchema(tf.int64, [None],
                                   sch.ListColumnRepresentation())
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
    expected_metadata = dataset_metadata.DatasetMetadata({
        'index1': sch.ColumnSchema(tf.int64, [None],
                                   sch.ListColumnRepresentation()),
        'index2': sch.ColumnSchema(tf.int64, [None],
                                   sch.ListColumnRepresentation())
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
        'index1': sch.ColumnSchema(tf.int64, [None],
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
    tranform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')
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

      _ = transform_fn | transform_fn_io.WriteTransformFn(tranform_fn_dir)
      _ = metadata | beam_metadata_io.WriteMetadata(metadata_dir,
                                                    pipeline=pipeline)

    with beam.Pipeline() as pipeline:
      transform_fn = pipeline | transform_fn_io.ReadTransformFn(tranform_fn_dir)
      metadata = pipeline | beam_metadata_io.ReadMetadata(metadata_dir)
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
      self.assertDataEqual([expected_transformed_data], [result])

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


if __name__ == '__main__':
  unittest.main()
