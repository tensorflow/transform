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


import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import api
from tensorflow_transform import impl_helper
from tensorflow_transform import mappers
from tensorflow_transform.tf_metadata import dataset_schema as sch
import unittest
from tensorflow.contrib import lookup
from tensorflow.python.framework import test_util


class ImplHelperTest(test_util.TensorFlowTestCase):

  def assertSparseValuesEqual(self, a, b):
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def toSchema(self, feature_spec):
    return sch.from_feature_spec(feature_spec)

  def testInferFeatureSchema(self):
    d = tf.placeholder(tf.int64, None)
    tensors = {
        'a': tf.placeholder(tf.float32, (None,)),
        'b': tf.placeholder(tf.string, (1, 2, 3)),
        'c': tf.placeholder(tf.int64, None),
        'd': d
    }
    d_column_schema = sch.ColumnSchema(tf.int64, [1, 2, 3],
                                       sch.FixedColumnRepresentation())
    api.set_column_schema(d, d_column_schema)
    schema = impl_helper.infer_feature_schema(tf.get_default_graph(), tensors)
    expected_schema = sch.Schema(column_schemas={
        'a': sch.ColumnSchema(tf.float32, [],
                              sch.FixedColumnRepresentation()),
        'b': sch.ColumnSchema(tf.string, [2, 3],
                              sch.FixedColumnRepresentation()),
        'c': sch.ColumnSchema(tf.int64, None,
                              sch.FixedColumnRepresentation()),
        'd': sch.ColumnSchema(tf.int64, [1, 2, 3],
                              sch.FixedColumnRepresentation())
    })
    self.assertEqual(schema, expected_schema)

  def testInferFeatureSchemaBadRank(self):
    tensors = {
        'a': tf.placeholder(tf.float32, ()),
    }
    with self.assertRaises(ValueError):
      _ = impl_helper.infer_feature_schema(tf.get_default_graph(), tensors)

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
        'a': np.array([100, 200, 300]),
        'b': np.array([10.0, 20.0, 30.0]),
        'c': np.array([[40.0], [80.0], [120.0]]),
        'd': np.array([[[1.0, 2.0], [3.0, 4.0]],
                       [[5.0, 6.0], [7.0, 8.0]],
                       [[9.0, 10.0], [11.0, 12.0]]]),
        'e': tf.SparseTensorValue(
            indices=np.array([(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)]),
            values=np.array(['doe', 'a', 'deer', 'a', 'female', 'deer']),
            dense_shape=(3, 3)),
        'f': tf.SparseTensorValue(
            indices=np.array([(0, 2), (0, 4), (0, 8), (1, 8), (1, 4)]),
            values=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            dense_shape=(3, 20))
    }
    output_dict = impl_helper.make_output_dict(schema, fetches)
    self.assertSetEqual(set(six.iterkeys(output_dict)),
                        set(['a', 'b', 'c', 'd', 'e', 'f']))
    self.assertAllEqual(output_dict['a'], [100, 200, 300])
    self.assertAllEqual(output_dict['b'], [10.0, 20.0, 30.0])
    self.assertAllEqual(output_dict['c'], [[40.0], [80.0], [120.0]])
    self.assertAllEqual(output_dict['d'], [[[1.0, 2.0], [3.0, 4.0]],
                                           [[5.0, 6.0], [7.0, 8.0]],
                                           [[9.0, 10.0], [11.0, 12.0]]])
    self.assertAllEqual(output_dict['e'][0], ['doe', 'a', 'deer'])
    self.assertAllEqual(output_dict['e'][1], [])
    self.assertAllEqual(output_dict['e'][2], ['a', 'female', 'deer'])
    self.assertEqual(len(output_dict['f']), 2)
    self.assertAllEqual(output_dict['f'][0][0], [2, 4, 8])
    self.assertAllEqual(output_dict['f'][0][1], [8, 4])
    self.assertAllEqual(output_dict['f'][0][2], [])
    self.assertAllEqual(output_dict['f'][1][0], [10.0, 20.0, 30.0])
    self.assertAllEqual(output_dict['f'][1][1], [40.0, 50.0])
    self.assertAllEqual(output_dict['f'][1][2], [])

  def testMakeOutputDictError(self):
    schema = self.toSchema({'a': tf.VarLenFeature(tf.string)})

    # SparseTensor that cannot be represented as VarLenFeature.
    fetches = {
        'a': tf.SparseTensorValue(indices=np.array([(0, 2), (0, 4), (0, 8)]),
                                  values=np.array([10.0, 20.0, 30.0]),
                                  dense_shape=(1, 20))
    }
    with self.assertRaisesRegexp(
        ValueError, 'cannot be decoded by ListColumnRepresentation'):
      _ = impl_helper.make_output_dict(schema, fetches)

    # SparseTensor of invalid rank.
    fetches = {
        'a': tf.SparseTensorValue(
            indices=np.array([(0, 0, 1), (0, 0, 2), (0, 0, 3)]),
            values=np.array([10.0, 20.0, 30.0]),
            dense_shape=(1, 10, 10))
    }
    with self.assertRaisesRegexp(
        ValueError, 'cannot be decoded by ListColumnRepresentation'):
      _ = impl_helper.make_output_dict(schema, fetches)

    # SparseTensor with indices that are out of order.
    fetches = {
        'a': tf.SparseTensorValue(indices=np.array([(0, 2), (2, 4), (1, 8)]),
                                  values=np.array([10.0, 20.0, 30.0]),
                                  dense_shape=(3, 20))
    }
    with self.assertRaisesRegexp(
        ValueError, 'Encountered out-of-order sparse index'):
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

  def testCreatePhasesWithDegenerateFunctionApplication(self):
    # Tests the case of a function whose inputs and outputs overlap.
    def preprocessing_fn(inputs):
      return {
          'index': api.apply_function(lambda x: x, inputs['a'])
      }

    input_schema = sch.Schema({
        'a': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    graph, _, _ = impl_helper.run_preprocessing_fn(
        preprocessing_fn, input_schema)
    phases = impl_helper.create_phases(graph)
    self.assertEqual(len(phases), 0)

  def testCreatePhasesWithMultipleLevelsOfAnalyzers(self):
    # Test a preprocessing function similar to scale_to_0_1 except that it
    # involves multiple interleavings of analyzers and transforms.
    def preprocessing_fn(inputs):
      scaled_to_0 = inputs['x'] - analyzers.min(inputs['x'])
      scaled_to_0_1 = scaled_to_0 / analyzers.max(scaled_to_0)
      return {'x_scaled': scaled_to_0_1}

    input_schema = sch.Schema({
        'x': sch.ColumnSchema(tf.float32, [], sch.FixedColumnRepresentation())
    })
    graph, _, _ = impl_helper.run_preprocessing_fn(
        preprocessing_fn, input_schema)
    phases = impl_helper.create_phases(graph)
    self.assertEqual(len(phases), 2)
    self.assertEqual(len(phases[0].analyzers), 1)
    self.assertEqual(len(phases[1].analyzers), 1)

  def testCreatePhasesWithTable(self):
    # Test a preprocessing function with table that can only be run after the
    # first analyzer has run.  Note converting an integerized string into a
    # float doesn't make much sense, but is a legal tensorflow computation.
    def preprocessing_fn(inputs):
      integerized = mappers.string_to_int(inputs['x'])
      integerized = tf.to_float(integerized)
      scaled_to_0_1 = integerized / analyzers.max(integerized)
      return {'x_scaled': scaled_to_0_1}

    input_schema = sch.Schema({
        'x': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    graph, _, _ = impl_helper.run_preprocessing_fn(
        preprocessing_fn, input_schema)
    phases = impl_helper.create_phases(graph)
    self.assertEqual(len(phases), 2)
    self.assertEqual(len(phases[0].analyzers), 1)
    self.assertEqual(len(phases[1].analyzers), 1)
    self.assertEqual(len(phases[0].table_initializers), 0)
    self.assertEqual(len(phases[1].table_initializers), 1)

  def testCreatePhasesWithUnwrappedTable(self):
    # Test a preprocessing function with a table that is not wrapped in
    # `apply_function`.
    def preprocessing_fn(inputs):
      table = lookup.index_table_from_tensor(['a', 'b'])
      integerized = table.lookup(inputs['x'])
      return {'integerized': integerized}

    input_schema = sch.Schema({
        'x': sch.ColumnSchema(tf.string, [], sch.FixedColumnRepresentation())
    })
    graph, _, _ = impl_helper.run_preprocessing_fn(
        preprocessing_fn, input_schema)
    with self.assertRaisesRegexp(ValueError, 'Found table initializers'):
      _ = impl_helper.create_phases(graph)

  def testCreatePhasesWithLoop(self):
    # Test a preprocessing function with control flow.
    #
    # The loop represents
    #
    # i = 0
    # while i < 10:
    #   i += 1
    #   x += 1
    #
    # To get an error in the case where apply_function is not called, we have
    # to call an analyzer first (see testCreatePhasesWithUnwrappedLoop).  So
    # we also do so here.
    def preprocessing_fn(inputs):
      def _subtract_ten(x):
        i = tf.constant(0)
        c = lambda i, x: tf.less(i, 10)
        b = lambda i, x: (tf.add(i, 1), tf.add(x, -1))
        return tf.while_loop(c, b, [i, x])[1]
      scaled_to_0_1 = mappers.scale_to_0_1(
          api.apply_function(_subtract_ten, inputs['x']))
      return {'x_scaled': scaled_to_0_1}

    input_schema = sch.Schema({
        'x': sch.ColumnSchema(tf.int32, [], sch.FixedColumnRepresentation())
    })
    graph, _, _ = impl_helper.run_preprocessing_fn(
        preprocessing_fn, input_schema)
    phases = impl_helper.create_phases(graph)
    self.assertEqual(len(phases), 1)
    self.assertEqual(len(phases[0].analyzers), 2)

  def testCreatePhasesWithUnwrappedLoop(self):
    # Test a preprocessing function with control flow.
    #
    # The loop represents
    #
    # i = 0
    # while i < 10:
    #   i += 1
    #   x += 1
    #
    # We need to call an analyzer after the loop because only the transitive
    # parents of analyzers are inspected by create_phases
    def preprocessing_fn(inputs):
      def _subtract_ten(x):
        i = tf.constant(0)
        c = lambda i, x: tf.less(i, 10)
        b = lambda i, x: (tf.add(i, 1), tf.add(x, -1))
        return tf.while_loop(c, b, [i, x])[1]
      scaled_to_0_1 = mappers.scale_to_0_1(_subtract_ten(inputs['x']))
      return {'x_scaled': scaled_to_0_1}

    input_schema = sch.Schema({
        'x': sch.ColumnSchema(tf.int32, [], sch.FixedColumnRepresentation())
    })
    graph, _, _ = impl_helper.run_preprocessing_fn(
        preprocessing_fn, input_schema)
    with self.assertRaisesRegexp(ValueError, 'Cycle detected'):
      _ = impl_helper.create_phases(graph)

  def testRunPreprocessingFn(self):
    schema = self.toSchema({
        'dense_1': tf.FixedLenFeature((), tf.float32),
        'dense_2': tf.FixedLenFeature((1, 2), tf.int64),
        'var_len': tf.VarLenFeature(tf.string),
        'sparse': tf.SparseFeature('ix', 'val', tf.float32, 100)
    })
    def preprocessing_fn(inputs):
      return {
          'dense_out': mappers.scale_to_0_1(inputs['dense_1']),
          'sparse_out': tf.sparse_reshape(inputs['sparse'], (1, 10)),
      }

    _, inputs, outputs = impl_helper.run_preprocessing_fn(
        preprocessing_fn, schema)

    # Verify that the input placeholders have the correct types.
    expected_dtype_and_shape = {
        'dense_1': (tf.float32, tf.TensorShape([None])),
        'dense_2': (tf.int64, tf.TensorShape([None, 1, 2])),
        'var_len': (tf.string, tf.TensorShape([None, None])),
        'sparse': (tf.float32, tf.TensorShape([None, None])),
        'dense_out': (tf.float32, tf.TensorShape([None])),
        'sparse_out': (tf.float32, tf.TensorShape([None, None])),
    }

    for key, tensor in itertools.chain(six.iteritems(inputs),
                                       six.iteritems(outputs)):
      dtype, shape = expected_dtype_and_shape[key]
      self.assertEqual(tensor.dtype, dtype)
      tensor.get_shape().assert_is_compatible_with(shape)


if __name__ == '__main__':
  unittest.main()
