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
    schema = impl_helper.infer_feature_schema(tensors)
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
      impl_helper.infer_feature_schema(tensors)

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
      impl_helper.make_feed_dict(tensors, schema, instances)

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
      impl_helper.make_feed_dict(tensors, schema, instances)

    instances = [{'a': ([11, 1], [1.0, 2.0])}]
    with self.assertRaisesRegexp(
        ValueError, 'has index .* out of range'):
      impl_helper.make_feed_dict(tensors, schema, instances)

    # Indices and values of different lengths.
    schema = self.toSchema({
        'a': tf.SparseFeature('idx', 'val', tf.float32, 10)
    })
    instances = [{'a': ([1, 2], [1])}]
    with self.assertRaisesRegexp(
        ValueError, 'indices and values of different lengths'):
      impl_helper.make_feed_dict(tensors, schema, instances)

    # Tuple of the wrong length.
    instances = [{'a': ([1], [2], [3])}]
    with self.assertRaisesRegexp(
        ValueError, 'too many values to unpack'):
      impl_helper.make_feed_dict(tensors, schema, instances)

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
        'd': np.array([[[1.0, 2.0], [3.0, 4.0]],
                       [[5.0, 6.0], [7.0, 8.0]]]),
        'e': tf.SparseTensorValue(
            indices=np.array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
            values=np.array(['doe', 'a', 'deer', 'a', 'female', 'deer']),
            dense_shape=(2, 3)),
        'f': tf.SparseTensorValue(
            indices=np.array([(0, 2), (0, 4), (0, 8), (1, 4), (1, 8)]),
            values=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            dense_shape=(2, 20))
    }

    instance_dicts = impl_helper.to_instance_dicts(schema, fetches)
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
    self.assertAllEqual(instance_dicts[1]['f'][0], [4, 8])
    self.assertAllEqual(instance_dicts[1]['f'][1], [40.0, 50.0])

  def testMakeOutputDictErrorSparse(self):
    schema = self.toSchema({'a': tf.VarLenFeature(tf.string)})

    # SparseTensor that cannot be represented as VarLenFeature.
    fetches = {
        'a': tf.SparseTensorValue(indices=np.array([(0, 2), (0, 4), (0, 8)]),
                                  values=np.array([10.0, 20.0, 30.0]),
                                  dense_shape=(1, 20))
    }
    with self.assertRaisesRegexp(
        ValueError, 'cannot be decoded by ListColumnRepresentation'):
      impl_helper.to_instance_dicts(schema, fetches)

    # SparseTensor of invalid rank.
    fetches = {
        'a': tf.SparseTensorValue(
            indices=np.array([(0, 0, 1), (0, 0, 2), (0, 0, 3)]),
            values=np.array([10.0, 20.0, 30.0]),
            dense_shape=(1, 10, 10))
    }
    with self.assertRaisesRegexp(
        ValueError, 'cannot be decoded by ListColumnRepresentation'):
      impl_helper.to_instance_dicts(schema, fetches)

    # SparseTensor with indices that are out of order.
    fetches = {
        'a': tf.SparseTensorValue(indices=np.array([(0, 2), (2, 4), (1, 8)]),
                                  values=np.array([10.0, 20.0, 30.0]),
                                  dense_shape=(3, 20))
    }
    with self.assertRaisesRegexp(
        ValueError, 'Encountered out-of-order sparse index'):
      impl_helper.to_instance_dicts(schema, fetches)

    # SparseTensors with different batch dimension sizes.
    schema = self.toSchema({
        'a': tf.VarLenFeature(tf.string),
        'b': tf.VarLenFeature(tf.string)
    })
    fetches = {
        'a': tf.SparseTensorValue(indices=np.array([(0, 0)]),
                                  values=np.array([10.0]),
                                  dense_shape=(1, 20)),
        'b': tf.SparseTensorValue(indices=np.array([(0, 0)]),
                                  values=np.array([10.0]),
                                  dense_shape=(2, 20))
    }
    with self.assertRaisesRegexp(
        ValueError,
        r'Inconsistent batch sizes: "\w" had batch dimension \d, "\w" had batch'
        r' dimension \d'):
      impl_helper.to_instance_dicts(schema, fetches)

  def testMakeOutputDictErrorDense(self):
    schema = self.toSchema({
        'a': tf.FixedLenFeature((), tf.string),
        'b': tf.FixedLenFeature((), tf.string)
    })
    # Tensors with different batch dimension sizes.
    fetches = {
        'a': np.array([1]),
        'b': np.array([1, 2])
    }
    with self.assertRaisesRegexp(
        ValueError,
        r'Inconsistent batch sizes: "\w" had batch dimension \d, "\w" had batch'
        r' dimension \d'):
      impl_helper.to_instance_dicts(schema, fetches)

  def testCreatePhasesWithApplyFunctionWithOverlappingInputsAndOutputs(self):
    string_placeholder = tf.placeholder(tf.string, shape=(None,))
    def degenerate_function(x):
      """A function whose input tensors and output tensors overlap."""
      return x
    api.apply_function(degenerate_function, string_placeholder)

    phases = impl_helper.create_phases()
    self.assertEqual(len(phases), 0)

  def testCreatePhasesWithMultipleLevelsOfAnalyzers(self):
    # Create graph similar to calling scale_to_0_1 except involving multiple
    # interleavings of analyzers and transforms.
    float_placeholder = tf.placeholder(tf.float32, shape=(None,))
    scaled_to_0 = float_placeholder - analyzers.min(float_placeholder)
    scaled_to_0 / analyzers.max(scaled_to_0)  # pylint: disable=expression-not-assigned

    phases = impl_helper.create_phases()
    self.assertEqual(len(phases), 2)
    self.assertEqual(len(phases[0].analyzers), 1)
    self.assertEqual(len(phases[1].analyzers), 1)

  def testCreatePhasesWithTable(self):
    # Create a graph with table that can only be run after the first analyzer
    # has run.  Note converting an integerized string into a float doesn't make
    # much sense, but is a legal tensorflow computation.
    string_placeholder = tf.placeholder(tf.string, shape=(None,))
    integerized = mappers.string_to_int(string_placeholder)
    integerized = tf.to_float(integerized)
    integerized / analyzers.max(integerized)  # pylint: disable=expression-not-assigned

    phases = impl_helper.create_phases()
    self.assertEqual(len(phases), 2)
    self.assertEqual(len(phases[0].analyzers), 1)
    self.assertEqual(len(phases[1].analyzers), 1)
    self.assertEqual(len(phases[0].table_initializers), 0)
    self.assertEqual(len(phases[1].table_initializers), 1)

  def testCreatePhasesWithUnwrappedTable(self):
    # Create a graph with a table that is not wrapped in `apply_function`.
    string_placeholder = tf.placeholder(tf.string, shape=(None,))
    table = lookup.index_table_from_tensor(['a', 'b'])
    table.lookup(string_placeholder)

    with self.assertRaisesRegexp(ValueError, 'Found table initializers'):
      impl_helper.create_phases()

  def testCreatePhasesWithControlFlowOpsWrappedInApplyFunction(self):
    int_placeholder = tf.placeholder(tf.int64, shape=(None,))
    int_placeholder_minus_10 = api.apply_function(_subtract_ten,
                                                  int_placeholder)
    # We need to call an analyzer after the loop because only the transitive
    # parents of analyzers are inspected by create_phases
    mappers.scale_to_0_1(int_placeholder_minus_10)

    phases = impl_helper.create_phases()
    self.assertEqual(len(phases), 1)
    #  tft.scale_to_0_1 uses a single analyzer: analyzers._min_and_max.
    self.assertEqual(len(phases[0].analyzers), 1)

  def testCreatePhasesWithControlFlowOpsNotWrappedInApplyFunction(self):
    int_placeholder = tf.placeholder(tf.int64, shape=(None,))
    int_placeholder_minus_10 = _subtract_ten(int_placeholder)
    # We need to call an analyzer after the loop because only the transitive
    # parents of analyzers are inspected by create_phases
    mappers.scale_to_0_1(int_placeholder_minus_10)

    with self.assertRaisesRegexp(ValueError, 'Cycle detected'):
      impl_helper.create_phases()

  def testCopyTensorsCopiesProducesDifferentTensors(self):
    tensors = {
        'dense': tf.placeholder(tf.int64, (None,), name='my_dense_input'),
        'sparse': tf.sparse_placeholder(tf.int64, name='my_sparse_input')
    }
    copied_tensors = impl_helper.copy_tensors(tensors)

    self.assertNotEqual(tensors['dense'],
                        copied_tensors['dense'])
    self.assertNotEqual(tensors['sparse'].indices,
                        copied_tensors['sparse'].indices)
    self.assertNotEqual(tensors['sparse'].values,
                        copied_tensors['sparse'].values)
    self.assertNotEqual(tensors['sparse'].dense_shape,
                        copied_tensors['sparse'].dense_shape)

  def testCopyTensorsProducesEquivalentTensors(self):
    tensors = {
        'dense': tf.placeholder(tf.int64, (None,), name='my_dense_input'),
        'sparse': tf.sparse_placeholder(tf.int64, name='my_sparse_input')
    }
    copied_tensors = impl_helper.copy_tensors(tensors)

    with tf.Session() as session:
      dense_value = [1, 2]
      sparse_value = tf.SparseTensorValue(
          indices=[[0, 0], [0, 2], [1, 1]],
          values=[3, 4, 5],
          dense_shape=[2, 3])
      sample_tensors = session.run(copied_tensors, feed_dict={
          tensors['dense']: dense_value,
          tensors['sparse']: sparse_value
      })
      self.assertAllEqual(sample_tensors['dense'], dense_value)
      self.assertAllEqual(sample_tensors['sparse'].indices,
                          sparse_value.indices)
      self.assertAllEqual(sample_tensors['sparse'].values,
                          sparse_value.values)
      self.assertAllEqual(sample_tensors['sparse'].dense_shape,
                          sparse_value.dense_shape)


def _subtract_ten(x):
  """Subtracts 10 from x using control flow ops.

  This function is equivalent to "x - 10" but uses a tf.while_loop, in order
  to test the use of functions that involve control flow ops.

  Args:
    x: A tensor of integral type.

  Returns:
    A tensor representing x - 10.
  """
  def stop_condition(counter, x_minus_counter):
    del x_minus_counter  # unused
    return tf.less(counter, 10)
  def iteration(counter, x_minus_counter):
    return tf.add(counter, 1), tf.add(x_minus_counter, -1)
  initial_values = [tf.constant(0), x]
  return tf.while_loop(stop_condition, iteration, initial_values)[1]


if __name__ == '__main__':
  unittest.main()
