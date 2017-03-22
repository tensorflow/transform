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
"""Tests for tensorflow_transform.impl_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os


import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import api
from tensorflow_transform import impl_helper
from tensorflow_transform import mappers
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_schema as sch

import unittest
from tensorflow.python.framework import test_util


class ImplHelperTest(test_util.TensorFlowTestCase):

  def assertShapesEqual(self, a, b):
    if a.dims is None and b.dims is None:
      # TensorShape(None) != TensorShape(None) so we can't use assertEqual in
      # this case. But for our purposes these shapes are equal.
      pass
    else:
      self.assertAllEqual(a.as_list(), b.as_list())

  def assertSparseValuesEqual(self, a, b):
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def toSchema(self, feature_spec):
    return sch.from_feature_spec(feature_spec)

  def testInferFeatureSchema(self):
    columns = {
        'a': api._InputColumn(tf.placeholder(tf.float32, (None,)), None),
        'b': api._InputColumn(tf.placeholder(tf.string, (1, 2, 3)), None),
        'c': api._InputColumn(tf.placeholder(tf.int64, None), None)
    }
    schema = impl_helper.infer_feature_schema(columns)
    expected_schema = sch.Schema(column_schemas={
        'a': sch.ColumnSchema(tf.float32, [],
                              sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.string, [2, 3],
                              sch.FixedColumnRepresentation()),
        'c': sch.ColumnSchema(tf.int64, None,
                              sch.FixedColumnRepresentation())
    })
    self.assertEqual(schema, expected_schema)

  def testInferFeatureSchemaBadRank(self):
    columns = {
        'a': api._InputColumn(tf.placeholder(tf.float32, ()), None),
    }
    with self.assertRaises(ValueError):
      _ = impl_helper.infer_feature_schema(columns)

  def testMakeFeedDict(self):
    tensors = {
        'a': tf.placeholder(tf.int64),
        'b': tf.placeholder(tf.float32),
        'c': tf.placeholder(tf.float32),
        'd': tf.placeholder(tf.float32),
        'e': tf.sparse_placeholder(tf.string),
        'f': tf.sparse_placeholder(tf.float32)
    }
    schema = self.toSchema({
        'a': tf.FixedLenFeature(None, tf.int64),
        'b': tf.FixedLenFeature([], tf.float32),
        'c': tf.FixedLenFeature([1], tf.float32),
        'd': tf.FixedLenFeature([2, 2], tf.float32),
        'e': tf.VarLenFeature(tf.string),
        'f': tf.SparseFeature('idx', 'val', tf.float32, 10)
    })

    # Feed some dense and sparse values.
    instances = [{
        'a': 100,
        'b': 1.0,
        'c': [2.0],
        'd': [[1.0, 2.0], [3.0, 4.0]],
        'e': ['doe', 'a', 'deer'],
        'f': ([2, 4, 8], [10.0, 20.0, 30.0])
    }, {
        'a': 100,
        'b': 2.0,
        'c': [4.0],
        'd': [[5.0, 6.0], [7.0, 8.0]],
        'e': ['a', 'female', 'deer'],
        'f': ([], [])
    }]

    feed_dict = impl_helper.make_feed_dict(tensors, schema, instances)
    self.assertSetEqual(set(six.iterkeys(feed_dict)),
                        set(six.itervalues(tensors)))
    self.assertAllEqual(feed_dict[tensors['a']], [100, 100])
    self.assertAllEqual(feed_dict[tensors['b']], [1.0, 2.0])
    self.assertAllEqual(feed_dict[tensors['c']], [[2.0], [4.0]])
    self.assertAllEqual(feed_dict[tensors['d']], [[[1.0, 2.0], [3.0, 4.0]],
                                                  [[5.0, 6.0], [7.0, 8.0]]])
    self.assertSparseValuesEqual(feed_dict[tensors['e']], tf.SparseTensorValue(
        indices=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        values=['doe', 'a', 'deer', 'a', 'female', 'deer'],
        dense_shape=(2, 3)))
    self.assertSparseValuesEqual(feed_dict[tensors['f']], tf.SparseTensorValue(
        indices=[(0, 2), (0, 4), (0, 8)], values=[10.0, 20.0, 30.0],
        dense_shape=(2, 10)))

    # Feed numpy versions of everything.
    instances = [{
        'a': np.int64(100),
        'b': np.array(1.0, np.float32),
        'c': np.array([2.0], np.float32),
        'd': np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
        'e': ['doe', 'a', 'deer'],
        'f': (np.array([2, 4, 8]), np.array([10.0, 20.0, 30.0])),
    }, {
        'a': np.int64(100),
        'b': np.array(2.0, np.float32),
        'c': np.array([4.0], np.float32),
        'd': np.array([[5.0, 6.0], [7.0, 8.0]], np.float32),
        'e': ['a', 'female', 'deer'],
        'f': (np.array([], np.int32), np.array([], np.float32))
    }]

    feed_dict = impl_helper.make_feed_dict(tensors, schema, instances)
    self.assertSetEqual(set(six.iterkeys(feed_dict)),
                        set(six.itervalues(tensors)))
    self.assertAllEqual(feed_dict[tensors['a']], [100, 100])
    self.assertAllEqual(feed_dict[tensors['b']], [1.0, 2.0])
    self.assertAllEqual(feed_dict[tensors['c']], [[2.0], [4.0]])
    self.assertAllEqual(feed_dict[tensors['d']], [[[1.0, 2.0], [3.0, 4.0]],
                                                  [[5.0, 6.0], [7.0, 8.0]]])
    self.assertSparseValuesEqual(feed_dict[tensors['e']], tf.SparseTensorValue(
        indices=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        values=['doe', 'a', 'deer', 'a', 'female', 'deer'],
        dense_shape=(2, 3)))
    self.assertSparseValuesEqual(feed_dict[tensors['f']], tf.SparseTensorValue(
        indices=[(0, 2), (0, 4), (0, 8)], values=[10.0, 20.0, 30.0],
        dense_shape=(2, 10)))

    # Feed some empty sparse values
    instances = [{
        'a': 100,
        'b': 5.0,
        'c': [1.0],
        'd': [[1.0, 2.0], [3.0, 4.0]],
        'e': [],
        'f': ([], [])
    }]
    feed_dict = impl_helper.make_feed_dict(tensors, schema, instances)
    self.assertSparseValuesEqual(feed_dict[tensors['e']], tf.SparseTensorValue(
        indices=np.empty([0, 2], np.int64), values=[], dense_shape=(1, 0)))
    self.assertSparseValuesEqual(feed_dict[tensors['f']], tf.SparseTensorValue(
        indices=np.empty([0, 2], np.int64), values=[], dense_shape=(1, 10)))

  def testMakeFeedDictError(self):
    # Missing features.
    tensors = {
        'a': tf.placeholder(tf.int64),
        'b': tf.placeholder(tf.int64)
    }
    schema = self.toSchema({
        'a': tf.FixedLenFeature([1], tf.int64),
        'b': tf.FixedLenFeature([1], tf.int64)
    })
    instances = [{'a': 100}]
    with self.assertRaises(KeyError):
      _ = impl_helper.make_feed_dict(tensors, schema, instances)

  def testMalformedSparseFeatures(self):
    tensors = {
        'a': tf.sparse_placeholder(tf.int64),
    }

    # Invalid indices.
    schema = self.toSchema({
        'a': tf.SparseFeature('idx', 'val', tf.float32, 10)
    })
    instances = [{'a': ([-1, 2], [1.0, 2.0])}]
    with self.assertRaisesRegexp(
        ValueError, 'has index .* out of range'):
      _ = impl_helper.make_feed_dict(tensors, schema, instances)

    instances = [{'a': ([11, 1], [1.0, 2.0])}]
    with self.assertRaisesRegexp(
        ValueError, 'has index .* out of range'):
      _ = impl_helper.make_feed_dict(tensors, schema, instances)

    # Indices and values of different lengths.
    schema = self.toSchema({
        'a': tf.SparseFeature('idx', 'val', tf.float32, 10)
    })
    instances = [{'a': ([1, 2], [1])}]
    with self.assertRaisesRegexp(
        ValueError, 'indices and values of different lengths'):
      _ = impl_helper.make_feed_dict(tensors, schema, instances)

    # Tuple of the wrong length.
    instances = [{'a': ([1], [2], [3])}]
    with self.assertRaisesRegexp(
        ValueError, 'too many values to unpack'):
      _ = impl_helper.make_feed_dict(tensors, schema, instances)

  def testMakeOutputDict(self):
    schema = self.toSchema({
        'a': tf.FixedLenFeature(None, tf.int64),
        'b': tf.FixedLenFeature([], tf.float32),
        'c': tf.FixedLenFeature([1], tf.float32),
        'd': tf.FixedLenFeature([2, 2], tf.float32),
        'e': tf.VarLenFeature(tf.string),
        'f': tf.SparseFeature('idx', 'val', tf.float32, 10)
    })

    fetches = {
        'a': np.array([100, 200]),
        'b': np.array([10.0, 20.0]),
        'c': np.array([[40.0], [80.0]]),
        'd': np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        'e': tf.SparseTensorValue(
            indices=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            values=['doe', 'a', 'deer', 'a', 'female', 'deer'],
            dense_shape=(2, 3)),
        'f': tf.SparseTensorValue(indices=[(0, 2), (0, 4), (0, 8)],
                                  values=[10.0, 20.0, 30.0],
                                  dense_shape=(2, 20))
    }
    output_dict = impl_helper.make_output_dict(schema, fetches)
    self.assertSetEqual(set(six.iterkeys(output_dict)),
                        set(['a', 'b', 'c', 'd', 'e', 'f']))
    self.assertAllEqual(output_dict['a'], [100, 200])
    self.assertAllEqual(output_dict['b'], [10.0, 20.0])
    self.assertAllEqual(output_dict['c'], [[40.0], [80.0]])
    self.assertAllEqual(output_dict['d'], [[[1.0, 2.0], [3.0, 4.0]],
                                           [[5.0, 6.0], [7.0, 8.0]]])
    self.assertAllEqual(output_dict['e'], [['doe', 'a', 'deer'],
                                           ['a', 'female', 'deer']])
    self.assertEqual(len(output_dict['f']), 2)
    self.assertAllEqual(output_dict['f'][0], [[2, 4, 8], []])
    self.assertAllEqual(output_dict['f'][1], [[10.0, 20.0, 30.0], []])

  def testMakeOutputDictError(self):
    # SparseTensor that cannot be represented as VarLenFeature.
    schema = self.toSchema({'a': tf.VarLenFeature(tf.string)})
    fetches = {
        'a': tf.SparseTensorValue(indices=[(0, 2), (0, 4), (0, 8)],
                                  values=[10.0, 20.0, 30.0],
                                  dense_shape=(1, 20))
    }
    with self.assertRaises(ValueError):
      _ = impl_helper.make_output_dict(schema, fetches)

  def testToInstanceDicts(self):
    batch_dict = {
        'a': [100, 200],
        'b': [10.0, 20.0],
        'c': [[40.0], [80.0]],
        'd': [[[1.0, 2.0], [3.0, 4.0]],
              [[5.0, 6.0], [7.0, 8.0]]],
        'e': [['doe', 'a', 'deer'],
              ['a', 'female', 'deer']],
        'f': ([[2, 4, 8], []],
              [[10.0, 20.0, 30.0], []])
    }

    instance_dicts = impl_helper.to_instance_dicts(batch_dict)
    self.assertEqual(2, len(instance_dicts))
    self.assertSetEqual(set(six.iterkeys(instance_dicts[0])),
                        set(['a', 'b', 'c', 'd', 'e', 'f']))
    self.assertAllEqual(instance_dicts[0]['a'], 100)
    self.assertAllEqual(instance_dicts[0]['b'], 10.0)
    self.assertAllEqual(instance_dicts[0]['c'], [40.0])
    self.assertAllEqual(instance_dicts[0]['d'], [[1.0, 2.0], [3.0, 4.0]])
    self.assertAllEqual(instance_dicts[0]['e'], ['doe', 'a', 'deer'])
    self.assertEqual(len(instance_dicts[0]['f']), 2)
    self.assertAllEqual(instance_dicts[0]['f'][0], [2, 4, 8])
    self.assertAllEqual(instance_dicts[0]['f'][1], [10.0, 20.0, 30.0])
    self.assertAllEqual(instance_dicts[1]['a'], 200)
    self.assertAllEqual(instance_dicts[1]['b'], 20.0)
    self.assertAllEqual(instance_dicts[1]['c'], [80.0])
    self.assertAllEqual(instance_dicts[1]['d'], [[5.0, 6.0], [7.0, 8.0]])
    self.assertAllEqual(instance_dicts[1]['e'], ['a', 'female', 'deer'])
    self.assertEqual(len(instance_dicts[1]['f']), 2)
    self.assertAllEqual(instance_dicts[1]['f'][0], [])
    self.assertAllEqual(instance_dicts[1]['f'][1], [])

  def testImportAndExportDense(self):
    # Export the function "z = x * y + x + y"
    def preprocessing_fn(inputs):
      return {
          'z': api.map(lambda x, y: x * y + x + y,
                       inputs['x'], inputs['y'])
      }
    input_schema = self.toSchema({
        'x': tf.FixedLenFeature((), tf.float32),
        'y': tf.FixedLenFeature((), tf.float32)
    })

    inputs, outputs = impl_helper.run_preprocessing_fn(
        preprocessing_fn, input_schema)
    saved_model_dir = os.path.join(self.get_temp_dir(), 'dense')
    _ = impl_helper.make_transform_fn_def(
        input_schema, inputs, outputs, saved_model_dir)

    # Import the function, applying it to constants for x and y.
    g = tf.Graph()
    with g.as_default():
      x = tf.constant(5, tf.float32, (1,))
      y = tf.constant(6, tf.float32, (1,))
      outputs = saved_transform_io.apply_saved_transform(
          saved_model_dir, {'x': x, 'y': y})
      z = outputs['z']

      sess = tf.Session()
      with sess.as_default():
        # Check result is 5 * 6 + 5 + 6 = 41.
        self.assertEqual(41, z.eval())

    # Import the graph, feeding it values for x and y.
    g = tf.Graph()
    with g.as_default():
      inputs, outputs = impl_helper.load_transform_fn_def(
          saved_model_dir)
      x = inputs['x']
      y = inputs['y']
      z = outputs['z']

      sess = tf.Session()
      with sess.as_default():
        # Check result is 5 * 6 + 5 + 6 = 41.
        self.assertEqual(41, sess.run(z, {x: [5], y: [6]}))

  def testImportAndExportSparse(self):
    # Export the function "z = x + y"
    def preprocessing_fn(inputs):
      return {
          'z': api.map(tf.sparse_add, inputs['x'], inputs['y'])
      }
    input_schema = self.toSchema({
        'x': tf.VarLenFeature(tf.float32),
        'y': tf.VarLenFeature(tf.float32)
    })

    inputs, outputs = impl_helper.run_preprocessing_fn(
        preprocessing_fn, input_schema)
    saved_model_dir = os.path.join(self.get_temp_dir(), 'sparse')
    _ = impl_helper.make_transform_fn_def(
        input_schema, inputs, outputs, saved_model_dir)

    # Import the function, applying it to constants for x and y.
    g = tf.Graph()
    with g.as_default():
      x = tf.SparseTensor(
          indices=[[0]],
          values=tf.constant(5, shape=(1,), dtype=tf.float32),
          dense_shape=[1])
      y = tf.SparseTensor(
          indices=[[0]],
          values=tf.constant(6, shape=(1,), dtype=tf.float32),
          dense_shape=[1])
      outputs = saved_transform_io.apply_saved_transform(
          saved_model_dir, {'x': x, 'y': y})
      z = outputs['z']

      sess = tf.Session()
      with sess.as_default():
        # Check result is 5 + 6 = 11.
        result = z.eval()
        self.assertEqual(result.indices, [[0]])
        self.assertEqual(result.values, [11])
        self.assertEqual(result.dense_shape, [1])

  def testImportAndExportWithTensorValueMapping(self):
    # Export the function "z = x * min(y) + x + min(y)" with min(y) replaced by
    # 6.
    def preprocessing_fn(inputs):
      return {
          'z': api.map(lambda x, y: x * y + x + y,
                       inputs['x'], analyzers.min(inputs['y']))
      }
    input_schema = self.toSchema({
        'x': tf.FixedLenFeature((), tf.float32),
        'y': tf.FixedLenFeature((), tf.float32)
    })

    inputs, outputs = impl_helper.run_preprocessing_fn(
        preprocessing_fn, input_schema)
    saved_model_dir = os.path.join(self.get_temp_dir(), 'replace_original')
    input_columns_to_statistics = impl_helper.make_transform_fn_def(
        input_schema, inputs, outputs, saved_model_dir)
    self.assertEqual(len(input_columns_to_statistics.keys()), 1)
    y_min_input_name = input_columns_to_statistics.keys()[0]

    g = tf.Graph()
    with g.as_default():
      x = tf.placeholder(tf.float32, ())
      y = tf.placeholder(tf.float32, ())
      z = x * y + x + y
    new_saved_model_dir = os.path.join(self.get_temp_dir(), 'replace_new')
    impl_helper.replace_tensors_with_constant_values(
        saved_model_dir, new_saved_model_dir,
        {y_min_input_name: impl_helper.ConstantTensorValue(6, tf.float32, ())})

    # Import the function, applying it to constants for x and y.
    g = tf.Graph()
    with g.as_default():
      x = tf.constant(5, tf.float32, (1,))
      y = tf.constant(1000, tf.float32, (1,))  #  Value is never used.
      outputs = saved_transform_io.apply_saved_transform(
          new_saved_model_dir, {'x': x, 'y': y})
      z = outputs['z']

      sess = tf.Session()
      with sess.as_default():
        # Check result is 5 * 6 + 5 + 6 = 41.
        self.assertEqual(41, z.eval())

  def testRunTransformFn(self):
    schema = self.toSchema({
        'dense_1': tf.FixedLenFeature((), tf.float32),
        'dense_2': tf.FixedLenFeature((1, 2), tf.int64),
        'var_len': tf.VarLenFeature(tf.string),
        'sparse': tf.SparseFeature('ix', 'val', tf.float32, 100)
    })
    def preprocessing_fn(inputs):
      return {
          'dense_out': mappers.scale_to_0_1(inputs['dense_1']),
          'sparse_out': api.map(lambda x: tf.sparse_reshape(x, (1, 10)),
                                inputs['sparse'])
      }

    inputs, outputs = impl_helper.run_preprocessing_fn(
        preprocessing_fn, schema)

    # Verify that the input placeholders have the correct types.
    expected_dtype_and_shape = {
        'dense_1': (tf.float32, tf.TensorShape([None])),
        'dense_2': (tf.int64, tf.TensorShape([None, 1, 2])),
        'var_len': (tf.string, tf.TensorShape(None)),
        'sparse': (tf.float32, tf.TensorShape(None)),
        'dense_out': (tf.float32, tf.TensorShape([None])),
        'sparse_out': (tf.float32, tf.TensorShape([None, None])),
    }

    for key, column in itertools.chain(six.iteritems(inputs),
                                       six.iteritems(outputs)):
      dtype, shape = expected_dtype_and_shape[key]
      self.assertEqual(column.tensor.dtype, dtype)
      self.assertShapesEqual(column.tensor.get_shape(), shape)

  def testRunTransformFnBadTransform(self):
    schema = self.toSchema({
        'x': tf.FixedLenFeature((3,), tf.float32),
    })
    def preprocessing_fn(inputs):
      return {
          'x_sum': api.map(tf.reduce_sum, inputs['x']),
      }

    # Verify that we raise if preprocessing_fn outputs a tensor with rank 0.
    with self.assertRaises(ValueError):
      _ = impl_helper.run_preprocessing_fn(preprocessing_fn, schema)


if __name__ == '__main__':
  unittest.main()
