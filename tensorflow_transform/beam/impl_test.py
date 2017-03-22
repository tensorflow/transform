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


import apache_beam as beam
from apache_beam.transforms import util as beam_test_util
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import impl_helper
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.io import beam_metadata_io
from tensorflow_transform.beam.io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema as sch

import unittest
from tensorflow.python.framework import test_util


class BeamImplTest(test_util.TensorFlowTestCase):

  def assertDatasetsEqual(self, a, b):
    """Asserts that two test datasets are equal.

    Args:
      a: A Dataset in the test format (see comments at top of file).
      b: A Dataset in the test format.
    """
    a_data, a_metadata = a
    b_data, b_metadata = b

    self.assertDataEqual(a_data, b_data)
    self.assertEqual(a_metadata.schema, b_metadata.schema)

  def assertDataEqual(self, a_data, b_data):
    self.assertEqual(len(a_data), len(b_data))
    for a_row, b_row in zip(a_data, b_data):
      self.assertItemsEqual(a_row.keys(), b_row.keys())
      for key in a_row.keys():
        a_value = a_row[key]
        b_value = b_row[key]
        if isinstance(a_value, tf.SparseTensorValue):
          self.assertAllEqual(a_value.indices, b_value.indices)
          self.assertAllEqual(a_value.values, b_value.values)
          self.assertAllEqual(a_value.dense_shape, b_value.dense_shape)
        else:
          self.assertAllEqual(a_value, b_value)

  def toMetadata(self, feature_spec):
    return dataset_metadata.DatasetMetadata(
        schema=sch.from_feature_spec(feature_spec))

  def testMultipleLevelsOfAnalysis(self):
    # Test a preprocessing function similar to scale_to_0_1 except that it
    # involves multiple interleavings of analyzers and transforms.
    def preprocessing_fn(inputs):
      scaled_to_0 = tft.map(lambda x, y: x - y,
                            inputs['x'], tft.min(inputs['x']))
      scaled_to_0_1 = tft.map(lambda x, y: x / y,
                              scaled_to_0, tft.max(scaled_to_0))
      return {'x_scaled': scaled_to_0_1}

    metadata = self.toMetadata({'x': tf.FixedLenFeature((), tf.float32, 0)})
    input_columns = [{'x': v} for v in [4, 1, 5, 2]]
    input_dataset = (input_columns, metadata)

    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed, _ = (
          input_dataset | beam_impl.AnalyzeAndTransformDataset(
              preprocessing_fn))

    output_columns, _ = transformed

    self.assertEqual(output_columns,
                     [{'x_scaled': v} for v in [0.75, 0.0, 1.0, 0.25]])

  def testAnalyzeBeforeTransform(self):
    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

    # Run AnalyzeAndTransform on some input data and compare with expected
    # output.
    input_data = [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}]
    input_metadata = self.toMetadata(
        {'x': tf.FixedLenFeature((), tf.float32, 0)})

    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, transform_fn = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [
        {'x_scaled': 0.75},
        {'x_scaled': 0.0},
        {'x_scaled': 1.0},
        {'x_scaled': 0.25}
    ]
    expected_transformed_metadata = self.toMetadata({
        'x_scaled': tf.FixedLenFeature((), tf.float32, None)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

    # Take the transform function and use TransformDataset to apply it to
    # some eval data, and compare with expected output.
    eval_data = [{'x': 6}, {'x': 3}]
    transformed_eval_dataset = (
        ((eval_data, input_metadata), transform_fn)
        | beam_impl.TransformDataset())

    expected_transformed_eval_data = [{'x_scaled': 1.25}, {'x_scaled': 0.5}]
    self.assertDatasetsEqual(
        transformed_eval_dataset,
        (expected_transformed_eval_data, expected_transformed_metadata))

    # Redo test with eval data, using AnalyzeDataset instead of
    # AnalyzeAndTransformDataset to genereate transform_fn.
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transform_fn = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeDataset(preprocessing_fn))
      transformed_eval_dataset = (
          ((eval_data, input_metadata), transform_fn)
          | beam_impl.TransformDataset())
    self.assertDatasetsEqual(
        transformed_eval_dataset,
        (expected_transformed_eval_data, expected_transformed_metadata))

  def testTransformWithExcludedOutputs(self):
    def preprocessing_fn(inputs):
      return {
          'x_scaled': tft.scale_to_0_1(inputs['x']),
          'y_scaled': tft.scale_to_0_1(inputs['y'])
      }

    # Run AnalyzeAndTransform on some input data and compare with expected
    # output.
    input_data = [{'x': 5, 'y': 1}, {'x': 1, 'y': 2}]
    input_metadata = self.toMetadata({
        'x': tf.FixedLenFeature((), tf.float32, 0),
        'y': tf.FixedLenFeature((), tf.float32, 0)
    })
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transform_fn = (
          (input_data, input_metadata) | beam_impl.AnalyzeDataset(
              preprocessing_fn))

    # Take the transform function and use TransformDataset to apply it to
    # some eval data, with missing 'y' column.
    eval_data = [{'x': 6}]
    eval_metadata = self.toMetadata({
        'x': tf.FixedLenFeature((), tf.float32, 0)
    })
    transformed_eval_dataset = (
        ((eval_data, eval_metadata), transform_fn)
        | beam_impl.TransformDataset(exclude_outputs=['y_scaled']))

    expected_transformed_eval_data = [{'x_scaled': 1.25}]
    expected_transformed_eval_schema = self.toMetadata({
        'x_scaled': tf.FixedLenFeature((), tf.float32, None)
    })
    self.assertDatasetsEqual(
        transformed_eval_dataset,
        (expected_transformed_eval_data, expected_transformed_eval_schema))

  def testTransformSparseColumns(self):
    # Define a transform that takes a sparse column and a varlen column, and
    # returns a combination of dense, sparse, and varlen columns.
    def preprocessing_fn(inputs):
      sparse_sum = tft.map(
          lambda x: tf.sparse_reduce_sum(x, axis=1), inputs['sparse'])
      sparse_copy = tft.map(
          lambda y: tf.SparseTensor(y.indices, y.values, y.dense_shape),
          inputs['sparse'])
      varlen_copy = tft.map(
          lambda y: tf.SparseTensor(y.indices, y.values, y.dense_shape),
          inputs['varlen'])

      sparse_copy.schema = sch.ColumnSchema(
          tf.float32, [10],
          sch.SparseColumnRepresentation(
              'val_copy', [sch.SparseIndexField('idx_copy', False)]))

      return {
          'fixed': sparse_sum,  # Schema should be inferred.
          'sparse': inputs['sparse'],  # Schema manually attached above.
          'varlen': inputs['varlen'],  # Schema should be inferred.
          'sparse_copy': sparse_copy,  # Schema should propagate from input.
          'varlen_copy': varlen_copy   # Schema should propagate from input.
      }

    # Run AnalyzeAndTransform on some input data and compare with expected
    # output.
    input_metadata = self.toMetadata({
        'sparse': tf.SparseFeature('idx', 'val', tf.float32, 10),
        'varlen': tf.VarLenFeature(tf.float32),
    })
    input_data = [
        {'sparse': ([0, 1], [0., 1.]), 'varlen': [0., 1.]},
        {'sparse': ([2, 3], [2., 3.]), 'varlen': [3., 4., 5.]},
        {'sparse': ([4, 5], [4., 5.]), 'varlen': [6., 7.]}
    ]
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, transform_fn = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_metadata = self.toMetadata({
        'fixed': tf.FixedLenFeature(None, tf.float32, None),
        'sparse': tf.SparseFeature('idx', 'val', tf.float32, 10),
        'varlen': tf.VarLenFeature(tf.float32),
        'sparse_copy': tf.SparseFeature('idx_copy', 'val_copy', tf.float32, 10),
        'varlen_copy': tf.VarLenFeature(tf.float32)
    })
    expected_transformed_data = [
        {'fixed': 1.0, 'sparse': ([0, 1], [0., 1.]), 'varlen': [0., 1.],
         'sparse_copy': ([0, 1], [0., 1.]), 'varlen_copy': [0., 1.]},
        {'fixed': 5.0, 'sparse': ([2, 3], [2., 3.]), 'varlen': [3., 4., 5.],
         'sparse_copy': ([2, 3], [2., 3.]), 'varlen_copy': [3., 4., 5.]},
        {'fixed': 9.0, 'sparse': ([4, 5], [4., 5.]), 'varlen': [6., 7.],
         'sparse_copy': ([4, 5], [4., 5.]), 'varlen_copy': [6., 7.]}
    ]
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

    # Take the transform function and use TransformDataset to apply it to
    # some eval data, and compare with expected output.
    eval_data = [
        {'sparse': ([0], [9.]), 'varlen': [9.]},
        {'sparse': ([], []), 'varlen': []},
        {'sparse': ([2, 4], [8., 7.]), 'varlen': [8., 7.]}
    ]
    transformed_eval_dataset = (
        ((eval_data, input_metadata), transform_fn)
        | beam_impl.TransformDataset())

    expected_transformed_eval_values = [
        {'fixed': 9., 'sparse': ([0], [9.]), 'varlen': [9.],
         'sparse_copy': ([0], [9.]), 'varlen_copy': [9.]},
        {'fixed': 0., 'sparse': ([], []), 'varlen': [],
         'sparse_copy': ([], []), 'varlen_copy': []},
        {'fixed': 15., 'sparse': ([2, 4], [8., 7.]), 'varlen': [8., 7.],
         'sparse_copy': ([2, 4], [8., 7.]), 'varlen_copy': [8., 7.]}
    ]
    self.assertDatasetsEqual(
        transformed_eval_dataset,
        (expected_transformed_eval_values, expected_transformed_metadata))

  def testTransform(self):
    # User defined preprocessing_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):
      return {'ab': tft.map(tf.multiply, inputs['a'], inputs['b'])}

    input_data = [{
        'a': 4,
        'b': 3
    }, {
        'a': 1,
        'b': 2
    }, {
        'a': 5,
        'b': 6
    }, {
        'a': 2,
        'b': 3
    }]
    input_metadata = self.toMetadata({
        'a': tf.FixedLenFeature((), tf.float32, 0),
        'b': tf.FixedLenFeature((), tf.float32, 0)
    })
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [{
        'ab': 12
    }, {
        'ab': 2
    }, {
        'ab': 30
    }, {
        'ab': 6
    }]
    expected_transformed_metadata = self.toMetadata({
        'ab': tf.FixedLenFeature((), tf.float32, None)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

  def testTransformMoreThanDesiredBatchSize(self):
    # User defined preprocessing_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):
      return {'ab': tft.map(tf.multiply, inputs['a'], inputs['b'])}

    input_data = [{
        'a': 1,
        'b': i
    } for i in range(beam_impl._DEFAULT_DESIRED_BATCH_SIZE + 1)]
    input_metadata = self.toMetadata({
        'a': tf.FixedLenFeature((), tf.float32, 0),
        'b': tf.FixedLenFeature((), tf.float32, 0)
    })
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [{'ab': i} for i in range(len(input_data))]
    expected_transformed_metadata = self.toMetadata({
        'ab': tf.FixedLenFeature((), tf.float32, None)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

  def testTransformUnicode(self):
    # User defined preprocessing_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):

      def tito_string_join(*tensors):
        return tf.string_join(tensors, separator=' ')

      return {'a b': tft.map(tito_string_join, inputs['a'], inputs['b'])}

    input_data = [{
        'a': 'Hello',
        'b': 'world'
    }, {
        'a': 'Hello',
        'b': u'κόσμε'
    }]
    input_metadata = self.toMetadata({
        'a': tf.FixedLenFeature((), tf.string),
        'b': tf.FixedLenFeature((), tf.string)
    })
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = ((input_data, input_metadata)
                                | beam_impl.AnalyzeAndTransformDataset(
                                    preprocessing_fn))

    expected_transformed_data = [{
        'a b': 'Hello world'
    }, {
        'a b': u'Hello κόσμε'.encode('utf-8')
    }]
    expected_transformed_metadata = self.toMetadata({
        'a b': tf.FixedLenFeature((), tf.string, None)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

  def testComposedTransforms(self):
    # User defined preprocessing_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):
      return {
          'a(b+c)':
              tft.map(tf.multiply, inputs['a'],
                      tft.map(tf.add, inputs['b'], inputs['c']))
      }

    input_data = [{'a': 4, 'b': 3, 'c': 3}, {'a': 1, 'b': 2, 'c': 1}]
    input_metadata = self.toMetadata({
        'a': tf.FixedLenFeature((), tf.float32, 0),
        'b': tf.FixedLenFeature((), tf.float32, 0),
        'c': tf.FixedLenFeature((), tf.float32, 0)
    })
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_metadata) |
          beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [{'a(b+c)': 24}, {'a(b+c)': 3}]
    expected_transformed_metadata = self.toMetadata({
        'a(b+c)': tf.FixedLenFeature((), tf.float32, None)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

  def testNumericAnalyzersWithScalarInputs(self):
    def preprocessing_fn(inputs):
      def repeat(in_tensor, value):
        batch_size = tf.shape(in_tensor)[0]
        return tf.ones([batch_size], dtype=value.dtype) * value

      return {
          'min': tft.map(repeat, inputs['a'], tft.min(inputs['a'])),
          'max': tft.map(repeat, inputs['a'], tft.max(inputs['a'])),
          'sum': tft.map(repeat, inputs['a'], tft.sum(inputs['a'])),
          'size': tft.map(repeat, inputs['a'], tft.size(inputs['a'])),
          'mean': tft.map(repeat, inputs['a'], tft.mean(inputs['a']))
      }

    input_data = [{'a': 4}, {'a': 1}]
    input_metadata = self.toMetadata(
        {'a': tf.FixedLenFeature((), tf.int64, 0)})
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_metadata) |
          beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [
        {'min': 1, 'max': 4, 'sum': 5, 'size': 2, 'mean': 2.5},
        {'min': 1, 'max': 4, 'sum': 5, 'size': 2, 'mean': 2.5}]
    expected_transformed_metadata = self.toMetadata({
        'min': tf.FixedLenFeature((), tf.int64, None),
        'max': tf.FixedLenFeature((), tf.int64, None),
        'sum': tf.FixedLenFeature((), tf.int64, None),
        'size': tf.FixedLenFeature((), tf.int64, None),
        'mean': tf.FixedLenFeature((), tf.float64, None)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

  def testNumericAnalyzersWithNDInputs(self):
    def preprocessing_fn(inputs):
      def repeat(in_tensor, value):
        batch_size = tf.shape(in_tensor)[0]
        return tf.ones([batch_size], value.dtype) * value

      return {
          'min': tft.map(repeat, inputs['a'], tft.min(inputs['a'])),
          'max': tft.map(repeat, inputs['a'], tft.max(inputs['a'])),
          'sum': tft.map(repeat, inputs['a'], tft.sum(inputs['a'])),
          'size': tft.map(repeat, inputs['a'], tft.size(inputs['a'])),
          'mean': tft.map(repeat, inputs['a'], tft.mean(inputs['a']))
      }

    input_data = [
        {'a': [[4, 5], [6, 7]]},
        {'a': [[1, 2], [3, 4]]}
    ]
    input_metadata = self.toMetadata(
        {'a': tf.FixedLenFeature((2, 2), tf.int64)})
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_metadata) |
          beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [
        {'min': 1, 'max': 7, 'sum': 32, 'size': 8, 'mean': 4.0},
        {'min': 1, 'max': 7, 'sum': 32, 'size': 8, 'mean': 4.0}]
    expected_transformed_metadata = self.toMetadata({
        'min': tf.FixedLenFeature((), tf.int64, None),
        'max': tf.FixedLenFeature((), tf.int64, None),
        'sum': tf.FixedLenFeature((), tf.int64, None),
        'size': tf.FixedLenFeature((), tf.int64, None),
        'mean': tf.FixedLenFeature((), tf.float64, None)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

  def testNumericAnalyzersWithSparseInputs(self):
    def repeat(in_tensor, value):
      batch_size = tf.shape(in_tensor)[0]
      return tf.ones([batch_size], value.dtype) * value

    input_data = [
        {'a': [4, 5, 6]},
        {'a': [1, 2]}
    ]
    input_metadata = self.toMetadata({'a': tf.VarLenFeature(tf.int64)})
    input_dataset = (input_data, input_metadata)

    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      with self.assertRaises(TypeError):
        def min_fn(inputs):
          return {'min': tft.map(repeat, inputs['a'], tft.min(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(min_fn)

      with self.assertRaises(TypeError):
        def max_fn(inputs):
          return {'max': tft.map(repeat, inputs['a'], tft.max(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(max_fn)

      with self.assertRaises(TypeError):
        def sum_fn(inputs):
          return {'sum': tft.map(repeat, inputs['a'], tft.sum(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(sum_fn)

      with self.assertRaises(TypeError):
        def size_fn(inputs):
          return {'size': tft.map(repeat, inputs['a'], tft.size(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(size_fn)

      with self.assertRaises(TypeError):
        def mean_fn(inputs):
          return {'mean': tft.map(repeat, inputs['a'], tft.mean(inputs['a']))}
        _ = input_dataset | beam_impl.AnalyzeDataset(mean_fn)

  def testUniquesAnalyzer(self):
    # User defined transform_fn accepts and returns a dict of Columns.
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
    input_metadata = self.toMetadata({
        'a': tf.FixedLenFeature((), tf.string),
    })
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [
        {'index': 0},
        {'index': 1},
        {'index': 0},
        {'index': 0},
        {'index': 2},
        {'index': 1},
        {'index': 3}
    ]
    expected_transformed_metadata = self.toMetadata({
        'index': tf.FixedLenFeature((), tf.int64)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

  def testUniquesAnalyzerWithNDInputs(self):
    # User defined transform_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):
      return {
          'index': tft.string_to_int(inputs['a'])
      }

    input_data = [
        {'a': [['some', 'say'], ['the', 'world']]},
        {'a': [['will', 'end'], ['in', 'fire']]},
        {'a': [['some', 'say'], ['in', 'ice']]},
    ]
    input_metadata = self.toMetadata({
        'a': tf.FixedLenFeature((2, 2), tf.string),
    })

    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [
        {'index': [[0, 1], [5, 3]]},
        {'index': [[4, 8], [2, 7]]},
        {'index': [[0, 1], [2, 6]]},
    ]
    expected_transformed_metadata = self.toMetadata({
        'index': tf.FixedLenFeature((2, 2), tf.int64)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

  def testUniquesAnalyzerWithTokenization(self):
    # User defined transform_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):
      return {
          'index': tft.string_to_int(tft.map(tf.string_split, inputs['a']))
      }

    input_data = [{'a': 'hello hello world'}, {'a': 'hello goodbye world'}]
    input_metadata = self.toMetadata({
        'a': tf.FixedLenFeature((), tf.string, ''),
    })

    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [{
        'index': [0, 0, 1],
    }, {
        'index': [0, 2, 1]
    }]
    expected_transformed_metadata = self.toMetadata(
        {'index': tf.VarLenFeature(tf.int64)})
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_metadata))

  def testUniquesAnalyzerWithTopK(self):
    # User defined transform_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):
      return {
          'index1': tft.string_to_int(tft.map(tf.string_split, inputs['a']),
                                      default_value=-99, top_k=2),

          # As above but using a string for top_k (and changing the
          # default_value to showcase things).
          'index2': tft.string_to_int(tft.map(tf.string_split, inputs['a']),
                                      default_value=-9, top_k='2')
      }

    input_data = [{'a': 'hello hello world'},
                  {'a': 'hello goodbye world'},
                  {'a': 'hello goodbye foo'}]
    input_schema = self.toMetadata({
        'a': tf.FixedLenFeature((), tf.string, ''),
    })

    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_schema)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    # Generated vocab (ordered by frequency, then value) should be:
    # ["hello", "world", "goodbye", "foo"]. After applying top_k=2, this becomes
    # ["hello", "world"].
    expected_transformed_data = [{
        'index1': [0, 0, 1],
        'index2': [0, 0, 1]
    }, {
        'index1': [0, -99, 1],
        'index2': [0, -9, 1]
    }, {
        'index1': [0, -99, -99],
        'index2': [0, -9, -9]
    }]
    expected_transformed_schema = self.toMetadata({
        'index1': tf.VarLenFeature(tf.int64),
        'index2': tf.VarLenFeature(tf.int64)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_schema))

  def testUniquesAnalyzerWithFrequencyThreshold(self):
    # User defined transform_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):
      return {
          'index1': tft.string_to_int(tft.map(tf.string_split, inputs['a']),
                                      default_value=-99, frequency_threshold=2),

          # As above but using a string for frequency_threshold (and changing
          # the default_value to showcase things).
          'index2': tft.string_to_int(tft.map(tf.string_split, inputs['a']),
                                      default_value=-9, frequency_threshold='2')
      }

    input_data = [{'a': 'hello hello world'},
                  {'a': 'hello goodbye world'},
                  {'a': 'hello goodbye foo'}]
    input_schema = self.toMetadata({
        'a': tf.FixedLenFeature((), tf.string, ''),
    })

    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_schema)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    # Generated vocab (ordered by frequency, then value) should be:
    # ["hello", "world", "goodbye", "foo"]. After applying frequency_threshold=2
    # this becomes
    # ["hello", "world", "goodbye"].
    expected_transformed_data = [{
        'index1': [0, 0, 1],
        'index2': [0, 0, 1]
    }, {
        'index1': [0, 2, 1],
        'index2': [0, 2, 1]
    }, {
        'index1': [0, 2, -99],
        'index2': [0, 2, -9]
    }]
    expected_transformed_schema = self.toMetadata({
        'index1': tf.VarLenFeature(tf.int64),
        'index2': tf.VarLenFeature(tf.int64)
    })
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_schema))

  def testUniquesAnalyzerWithFrequencyThresholdTooHigh(self):
    # User defined transform_fn accepts and returns a dict of Columns.
    # Expected to return an empty dict due to too high threshold.
    def preprocessing_fn(inputs):
      return {
          'index1':
              tft.string_to_int(
                  tft.map(tf.string_split, inputs['a']),
                  default_value=-99,
                  frequency_threshold=77),

          # As above but using a string for frequency_threshold (and changing
          # the default_value to showcase things).
          'index2':
              tft.string_to_int(
                  tft.map(tf.string_split, inputs['a']),
                  default_value=-9,
                  frequency_threshold='77')
      }

    input_data = [{
        'a': 'hello hello world'
    }, {
        'a': 'hello goodbye world'
    }, {
        'a': 'hello goodbye foo'
    }]
    input_schema = self.toMetadata({
        'a': tf.FixedLenFeature((), tf.string, ''),
    })

    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      transformed_dataset, _ = (
          (input_data, input_schema)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    # Generated vocab (ordered by frequency, then value) should be:
    # ["hello", "world", "goodbye", "foo"]. After applying frequency_threshold=2
    # this becomes empty.
    expected_transformed_data = [{
        'index1': [-99, -99, -99],
        'index2': [-9, -9, -9]
    }, {
        'index1': [-99, -99, -99],
        'index2': [-9, -9, -9]
    }, {
        'index1': [-99, -99, -99],
        'index2': [-9, -9, -9]
    }]
    expected_transformed_schema = self.toMetadata({
        'index1': tf.VarLenFeature(tf.int64),
        'index2': tf.VarLenFeature(tf.int64)
    })
    self.assertDatasetsEqual(transformed_dataset, (expected_transformed_data,
                                                   expected_transformed_schema))

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

    p = beam.Pipeline()
    metadata = self.toMetadata({'x': tf.FixedLenFeature((), tf.float32, 0)})
    columns = p | 'CreateTrainingData' >> beam.Create([{
        'x': v
    } for v in [4, 1, 5, 2]])
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      _, transform_fn = (
          (columns, metadata)
          | 'Analyze and Transform'
          >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    # Run transform_columns on some eval dataset.
    eval_data = p | 'CreateEvalData' >> beam.Create([{'x': v} for v in [6, 3]])
    transformed_eval_data, _ = (
        ((eval_data, metadata), transform_fn)
        | 'Transform' >> beam_impl.TransformDataset())
    p.run()
    expected_transformed_eval_data = [{'x_scaled': v} for v in [1.25, 0.5]]
    beam_test_util.assert_that(
        transformed_eval_data,
        beam_test_util.equal_to(expected_transformed_eval_data))

  def testTransformFnExportAndImportRoundtrip(self):
    tranform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')
    metadata_dir = os.path.join(self.get_temp_dir(), 'export_metadata')

    with beam.Pipeline() as p:
      def preprocessing_fn(inputs):
        return {'x_scaled': tft.scale_to_0_1(inputs['x'])}

      metadata = self.toMetadata({'x': tf.FixedLenFeature((), tf.float32, 0)})
      columns = p | 'CreateTrainingData' >> beam.Create([{
          'x': v
      } for v in [4, 1, 5, 2]])
      with beam_impl.Context(temp_dir=self.get_temp_dir()):
        _, transform_fn = (
            (columns, metadata)
            | 'Analyze and Transform'
            >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

      _ = transform_fn | transform_fn_io.WriteTransformFn(tranform_fn_dir)
      _ = metadata | beam_metadata_io.WriteMetadata(metadata_dir, pipeline=p)

    with beam.Pipeline() as p:
      transform_fn = p | transform_fn_io.ReadTransformFn(tranform_fn_dir)
      metadata = p | beam_metadata_io.ReadMetadata(metadata_dir)
      # Run transform_columns on some eval dataset.
      eval_data = p | 'CreateEvalData' >> beam.Create(
          [{'x': v} for v in [6, 3]])
      transformed_eval_data, _ = (
          ((eval_data, metadata), transform_fn)
          | 'Transform' >> beam_impl.TransformDataset())
      expected_transformed_eval_data = [{'x_scaled': v} for v in [1.25, 0.5]]
      beam_test_util.assert_that(
          transformed_eval_data,
          beam_test_util.equal_to(expected_transformed_eval_data))

  def testRunExportedGraph(self):
    # Run analyze_and_transform_columns on some dataset.
    def preprocessing_fn(inputs):
      x_scaled = tft.scale_to_0_1(inputs['x'])
      y_sum = tft.map(
          lambda y: tf.sparse_reduce_sum(y, axis=1), inputs['y'])
      z_copy = tft.map(
          lambda z: tf.SparseTensor(z.indices, z.values, z.dense_shape),
          inputs['z'])
      return {'x_scaled': x_scaled, 'y_sum': y_sum, 'z_copy': z_copy}

    metadata = self.toMetadata({
        'x': tf.FixedLenFeature((), tf.float32, 0),
        'y': tf.SparseFeature('idx', 'val', tf.float32, 10),
        'z': tf.VarLenFeature(tf.float32)
    })
    columns = [
        {'x': 4, 'y': ([0, 1], [0., 1.]), 'z': [2., 4., 6.]},
        {'x': 1, 'y': ([2, 3], [2., 3.]), 'z': [8.]},
        {'x': 5, 'y': ([4, 5], [4., 5.]), 'z': [1., 2., 3.]}
    ]
    with beam_impl.Context(temp_dir=self.get_temp_dir()):
      _, transform_fn = (
          (columns, metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    export_dir = os.path.join(self.get_temp_dir(), 'export')
    _ = transform_fn | transform_fn_io.WriteTransformFn(export_dir)

    # Load the exported graph, and apply it to a batch of data.
    g = tf.Graph()
    with g.as_default():
      inputs, outputs = impl_helper.load_transform_fn_def(
          os.path.join(export_dir, 'transform_fn'))
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
