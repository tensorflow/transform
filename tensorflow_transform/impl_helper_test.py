"""Tests for tensorflow_transform.impl_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
from tensorflow_transform import api
from tensorflow_transform import impl_helper

import unittest
from tensorflow.core.protobuf import meta_graph_pb2

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

  def testInferFeatureSchema(self):
    tensors = {
        'a': tf.placeholder(tf.float32, (None,)),
        'b': tf.placeholder(tf.string, (1, 2, 3)),
        'c': tf.placeholder(tf.float32),
    }
    feature_spec = impl_helper.infer_feature_schema(tensors)
    self.assertEqual(sorted(feature_spec.keys()), ['a', 'b', 'c'])
    self.assertEqual(feature_spec['a'].dtype, tf.float32)
    self.assertEqual(feature_spec['a'].shape, ())
    self.assertEqual(feature_spec['b'].dtype, tf.string)
    self.assertEqual(feature_spec['b'].shape, (2, 3))
    self.assertEqual(feature_spec['c'].dtype, tf.float32)
    self.assertEqual(feature_spec['c'].shape, None)

  def testInferFeatureSchemaBadRank(self):
    tensors = {
        'a': tf.placeholder(tf.float32, ()),
    }
    with self.assertRaises(ValueError):
      _ = impl_helper.infer_feature_schema(tensors)

  def testMakeFeedDict(self):
    tensors = {
        'a': tf.placeholder(tf.int64),
        'b': tf.placeholder(tf.float32),
        'c': tf.sparse_placeholder(tf.string),
        'd': tf.sparse_placeholder(tf.float32)
    }
    schema = {
        'a': tf.FixedLenFeature([1], tf.int64),
        'b': tf.FixedLenFeature([2, 2], tf.float32),
        'c': tf.VarLenFeature(tf.string),
        'd': tf.SparseFeature('idx', 'val', tf.float32, 10)
    }

    # Feed some dense and sparse values.
    instance = {
        'a': 100,
        'b': [[1.0, 2.0], [3.0, 4.0]],
        'c': ['doe', 'a', 'deer'],
        'idx': [2, 4, 8],
        'val': [10.0, 20.0, 30.0]
    }
    feed_dict = impl_helper.make_feed_dict(tensors, schema, instance)
    self.assertSetEqual(set(feed_dict.keys()), set(tensors.values()))
    self.assertEqual(feed_dict[tensors['a']], [100])
    self.assertEqual(feed_dict[tensors['b']], [[[1.0, 2.0], [3.0, 4.0]]])
    self.assertSparseValuesEqual(feed_dict[tensors['c']], tf.SparseTensorValue(
        indices=[(0, 0), (0, 1), (0, 2)], values=['doe', 'a', 'deer'],
        dense_shape=(1, 3)))
    self.assertSparseValuesEqual(feed_dict[tensors['d']], tf.SparseTensorValue(
        indices=[(0, 2), (0, 4), (0, 8)], values=[10.0, 20.0, 30.0],
        dense_shape=(1, 10)))

    # Feed some empty sparse values
    instance = {
        'a': 100,
        'b': [[1.0, 2.0], [3.0, 4.0]],
        'c': [],
        'idx': [],
        'val': []
    }
    feed_dict = impl_helper.make_feed_dict(tensors, schema, instance)
    self.assertSparseValuesEqual(feed_dict[tensors['c']], tf.SparseTensorValue(
        indices=np.empty([0, 2], np.int64), values=[], dense_shape=(1, 0)))
    self.assertSparseValuesEqual(feed_dict[tensors['d']], tf.SparseTensorValue(
        indices=np.empty([0, 2], np.int64), values=[], dense_shape=(1, 10)))

  def testMakeFeedDictError(self):
    # Missing features.
    tensors = {
        'a': tf.placeholder(tf.int64),
        'b': tf.placeholder(tf.int64)
    }
    schema = {
        'a': tf.FixedLenFeature([1], tf.int64),
        'b': tf.FixedLenFeature([1], tf.int64)
    }
    instance = {'a': 100}
    with self.assertRaises(KeyError):
      _ = impl_helper.make_feed_dict(tensors, schema, instance)

  def testMakeOutputDict(self):
    schema = {
        'a': tf.FixedLenFeature([1], tf.int64),
        'b': tf.FixedLenFeature([2, 2], tf.float32),
        'c': tf.VarLenFeature(tf.string),
        'd': tf.SparseFeature('idx', 'val', tf.float32, 10)
    }

    fetches = {
        'a': [100],
        'b': [[[1.0, 2.0], [3.0, 4.0]]],
        'c': tf.SparseTensorValue(indices=[(0, 0), (0, 1), (0, 2)],
                                  values=['doe', 'a', 'deer'],
                                  dense_shape=(1, 3)),
        'd': tf.SparseTensorValue(indices=[(0, 2), (0, 4), (0, 8)],
                                  values=[10.0, 20.0, 30.0],
                                  dense_shape=(1, 20))
    }
    output_dict = impl_helper.make_output_dict(schema, fetches)
    self.assertSetEqual(set(output_dict.keys()),
                        set(['a', 'b', 'c', 'idx', 'val']))
    self.assertEqual(output_dict['a'], 100)
    self.assertEqual(output_dict['b'], [[1.0, 2.0], [3.0, 4.0]])
    self.assertEqual(output_dict['c'], ['doe', 'a', 'deer'])
    self.assertEqual(output_dict['idx'], [2, 4, 8])
    self.assertEqual(output_dict['val'], [10.0, 20.0, 30.0])

  def testMakeOutputDictError(self):
    # SparseTensor that cannot be represented as VarLenFeature.
    schema = {'a': tf.VarLenFeature(tf.string)}
    fetches = {
        'a': tf.SparseTensorValue(indices=[(0, 2), (0, 4), (0, 8)],
                                  values=[10.0, 20.0, 30.0],
                                  dense_shape=(1, 20))
    }
    with self.assertRaises(ValueError):
      _ = impl_helper.make_output_dict(schema, fetches)

  def testImportAndExportDense(self):
    # Export the function "z = x * y + x + y"
    g = tf.Graph()
    with g.as_default():
      x = tf.placeholder(tf.float32, ())
      y = tf.placeholder(tf.float32, ())
      z = x * y + x + y
    transform_fn_def = impl_helper.make_transform_fn_def(
        g, {'x': x, 'y': y}, {'z': z})

    # Import the function, applying it to constants for x and y.
    g = tf.Graph()
    with g.as_default():
      x = tf.constant(5, tf.float32)
      y = tf.constant(6, tf.float32)
      outputs = impl_helper.apply_transform_fn_def(
          transform_fn_def, {'x': x, 'y': y})
      z = outputs['z']

      sess = tf.Session()
      with sess.as_default():
        # Check result is 5 * 6 + 5 + 6 = 41.
        self.assertEqual(41, z.eval())

    # Import the graph, feeding it values for x and y.
    g = tf.Graph()
    with g.as_default():
      inputs, outputs = impl_helper.load_transform_fn_def(
          transform_fn_def)
      x = inputs['x']
      y = inputs['y']
      z = outputs['z']

      sess = tf.Session()
      with sess.as_default():
        # Check result is 5 * 6 + 5 + 6 = 41.
        self.assertEqual(41, sess.run(z, {x: 5, y: 6}))

  def testImportAndExportSparse(self):
    # Export the function "z = x + y"
    g = tf.Graph()
    with g.as_default():
      x = tf.sparse_placeholder(tf.float32)
      y = tf.sparse_placeholder(tf.float32)
      z = tf.sparse_add(x, y)
    transform_fn_def = impl_helper.make_transform_fn_def(
        g, {'x': x, 'y': y}, {'z': z})

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
      outputs = impl_helper.apply_transform_fn_def(
          transform_fn_def, {'x': x, 'y': y})
      z = outputs['z']

      sess = tf.Session()
      with sess.as_default():
        # Check result is 5 + 6 = 11.
        result = z.eval()
        self.assertEqual(result.indices, [[0]])
        self.assertEqual(result.values, [11])
        self.assertEqual(result.dense_shape, [1])

  def testImportAndExportWithTensorValueMapping(self):
    # Export the function "z = x * y + x + y" with y replaced by 6.
    g = tf.Graph()
    with g.as_default():
      x = tf.placeholder(tf.float32, ())
      y = tf.placeholder(tf.float32, ())
      z = x * y + x + y
    transform_fn_def = impl_helper.replace_tensors_with_constant_values(
        impl_helper.make_transform_fn_def(g, {'x': x}, {'z': z}),
        tensor_value_mapping={y.name: 6})

    # Import the function, applying it to constants for x and y.
    g = tf.Graph()
    with g.as_default():
      x = tf.constant(5, tf.float32)
      outputs = impl_helper.apply_transform_fn_def(
          transform_fn_def, {'x': x})
      z = outputs['z']

      sess = tf.Session()
      with sess.as_default():
        # Check result is 5 * 6 + 5 + 6 = 41.
        self.assertEqual(41, z.eval())

  def testRunTransformFn(self):
    schema = {
        'dense_1': tf.FixedLenFeature((), tf.float32),
        'dense_2': tf.FixedLenFeature((1, 2), tf.int64),
        'var_len': tf.VarLenFeature(tf.string),
        'sparse': tf.SparseFeature('ix', 'val', tf.float32, 100)
    }
    def preprocessing_fn(inputs):
      return {
          'dense_out': api.scale_to_0_1(inputs['dense_1']),
          'sparse_out': api.transform(lambda x: tf.sparse_reshape(x, (1, 10)),
                                      inputs['sparse'])
      }

    graph = tf.Graph()
    inputs, outputs = impl_helper.run_preprocessing_fn(
        preprocessing_fn, schema, graph)

    # Verify that the input placeholders have the correct types.
    expected_dtype_and_shape = {
        'dense_1': (tf.float32, tf.TensorShape([None])),
        'dense_2': (tf.int64, tf.TensorShape([None, 1, 2])),
        'var_len': (tf.string, tf.TensorShape(None)),
        'sparse': (tf.float32, tf.TensorShape(None)),
        'dense_out': (tf.float32, tf.TensorShape([None])),
        'sparse_out': (tf.float32, tf.TensorShape([None, None])),
    }

    for key, column in inputs.items() + outputs.items():
      dtype, shape = expected_dtype_and_shape[key]
      self.assertEqual(column.tensor.dtype, dtype)
      self.assertShapesEqual(column.tensor.get_shape(), shape)

  def testRunTransformFnBadSchema(self):
    schema = {
        'bad': tf.FixedLenSequenceFeature((), tf.float32)
    }
    def preprocessing_fn(inputs):
      return {
          'bad_out': api.scale_to_0_1(inputs['bad']),
      }

    # Verify that we raise if the schema has an unsupported feature spec.
    graph = tf.Graph()
    with self.assertRaises(ValueError):
      _ = impl_helper.run_preprocessing_fn(preprocessing_fn, schema, graph)

  def testRunTransformFnBadTransform(self):
    schema = {
        'x': tf.FixedLenFeature((3,), tf.float32),
    }
    def preprocessing_fn(inputs):
      return {
          'x_sum': api.transform(tf.reduce_sum, inputs['x']),
      }

    # Verify that we raise if preprocessing_fn outputs a tensor with rank 0.
    graph = tf.Graph()
    with self.assertRaises(ValueError):
      _ = impl_helper.run_preprocessing_fn(preprocessing_fn, schema, graph)

  def testOverridingInputsFails(self):
    # Export the function "z = x * y + x + y"
    g = tf.Graph()
    with g.as_default():
      x = tf.placeholder(tf.float32, ())
      y = tf.placeholder(tf.float32, ())
      z = x * y + x + y

    with self.assertRaises(ValueError):
      impl_helper.replace_tensors_with_constant_values(
          impl_helper.make_transform_fn_def(g, {'x': x}, {'z': z}),
          tensor_value_mapping={x.name: 6})

  def testTransformFnDef(self):
    # Export the function "z = x * y + x + y"
    g = tf.Graph()
    with g.as_default():
      x = tf.placeholder(tf.float32, ())
      y = tf.placeholder(tf.float32, ())
      _ = x * y + x + y

    # Build a metagraph and assert that the serialize / deserialize path works
    # correctly.
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    meta_graph_def.graph_def.CopyFrom(g.as_graph_def())

    transform_fn_def = impl_helper.TransformFnDef(meta_graph_def)
    self.assertEqual(meta_graph_def, transform_fn_def.meta_graph_def)


if __name__ == '__main__':
  unittest.main()
