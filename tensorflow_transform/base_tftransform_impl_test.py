"""Base class for tests of an implementation of TFTransform.

An implementation of TFTransform is a module that provides the Anaylze,
Transform etc. methods described in api.py.  To test an implementation, create
a subclass of BaseTFTransformImplTest and override its `impl` method to return
the module that implements the TFTransform API.

For testing purposes, implementations should be polymorphic in that they can
accept inputs in the test format, and in that case will return outputs in the
test format.  The test format for a Dataset is a pair where the first
element is a list of dicts, and the second is a feature spec (in the sense of
TensorFlow).

NOTE: in may be necessary to switch to using explicit testing methods such
as load_test_data and assert_equals_test_data if the polymorphism approach
does not suit future extensions of TFTransform or non-beam implementations.
Currently we exploit the built in polymorphism of beam in order for the
beam implementation to be polymorphic in this way.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_transform import api

from tensorflow.python.framework import test_util


class BaseTFTransformImplTest(test_util.TensorFlowTestCase):

  @property
  def impl(self):
    raise NotImplementedError(
        'Subclasses of ImplTest must provide an impl method')

  def assertDatasetsEqual(self, a, b):
    """Asserts that two test datasets are equal.

    Args:
      a: A Dataset in the test format (see comments at top of file).
      b: A Dataset in the test format.
    """
    a_data, a_schema = a
    b_data, b_schema = b

    self.assertDataEqual(a_data, b_data)
    self.assertEqual(a_schema, b_schema)

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

  def testMultipleLevelsOfAnalysis(self):
    # Test a preprocessing function similar to scale_to_0_1 except that it
    # involves multiple interleavings of analyzers and transforms.
    def preprocessing_fn(inputs):
      scaled_to_0 = api.transform(lambda x, y: x - y,
                                  inputs['x'], api.min(inputs['x']))
      scaled_to_0_1 = api.transform(lambda x, y: x / y,
                                    scaled_to_0, api.max(scaled_to_0))
      return {'x_scaled': scaled_to_0_1}

    schema = {'x': tf.FixedLenFeature((), tf.float32, 0)}
    input_columns = [{'x': v} for v in [4, 1, 5, 2]]
    input_dataset = (input_columns, schema)

    transformed, _ = (
        input_dataset | self.impl.AnalyzeAndTransformDataset(preprocessing_fn))

    output_columns, _ = transformed

    self.assertEqual(output_columns,
                     [{'x_scaled': v} for v in [0.75, 0.0, 1.0, 0.25]])

  def testAnalyzeBeforeTransform(self):
    def preprocessing_fn(inputs):
      return {'x_scaled': api.scale_to_0_1(inputs['x'])}

    # Run AnalyzeAndTransform on some input data and compare with expected
    # output.
    input_data = [{'x': 4}, {'x': 1}, {'x': 5}, {'x': 2}]
    input_schema = {'x': tf.FixedLenFeature((), tf.float32, 0)}
    transformed_dataset, transform_fn = (
        (input_data, input_schema)
        | self.impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [
        {'x_scaled': 0.75},
        {'x_scaled': 0.0},
        {'x_scaled': 1.0},
        {'x_scaled': 0.25}
    ]
    expected_transformed_schema = {
        'x_scaled': tf.FixedLenFeature((), tf.float32, None)
    }
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_schema))

    # Take the transform function and use TransformDataset to apply it to
    # some eval data, and compare with expected output.
    eval_data = [{'x': 6}, {'x': 3}]
    transformed_eval_dataset = (
        ((eval_data, input_schema), transform_fn)
        | self.impl.TransformDataset())

    expected_transformed_eval_data = [{'x_scaled': 1.25}, {'x_scaled': 0.5}]
    self.assertDatasetsEqual(
        transformed_eval_dataset,
        (expected_transformed_eval_data, expected_transformed_schema))

    # Redo test with eval data, using AnalyzeDataset instead of
    # AnalyzeAndTransformDataset to genereate transform_fn.
    transform_fn = (
        (input_data, input_schema) | self.impl.AnalyzeDataset(preprocessing_fn))
    transformed_eval_dataset = (
        ((eval_data, input_schema), transform_fn)
        | self.impl.TransformDataset())
    self.assertDatasetsEqual(
        transformed_eval_dataset,
        (expected_transformed_eval_data, expected_transformed_schema))

  def testTransformSparseColumns(self):
    # Define a transform that takes two sparse columns, and returns the sum of
    # one of them (as a dense tensor) and passes thru the other.
    def preprocessing_fn(inputs):
      x_sum = api.transform(lambda x: tf.sparse_reduce_sum(x, axis=1),
                            inputs['x'])
      y_copy = api.transform(
          lambda y: tf.SparseTensor(y.indices, y.values, y.dense_shape),
          inputs['y'])
      return {'x_sum': x_sum, 'y_copy': y_copy}

    # Run AnalyzeAndTransform on some input data and compare with expected
    # output.
    input_data = [
        {'val': [0., 1.], 'idx': [0, 1], 'y': [0., 1.]},
        {'val': [2., 3.], 'idx': [2, 3], 'y': [2., 3., 4., 5.]},
        {'val': [4., 5.], 'idx': [4, 5], 'y': [6., 7.]}
    ]
    input_schema = {
        'x': tf.SparseFeature('idx', 'val', tf.float32, 10),
        'y': tf.VarLenFeature(tf.float32),
    }
    transformed_dataset, transform_fn = (
        (input_data, input_schema)
        | self.impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [
        {'x_sum': 1.0, 'y_copy': [0., 1.]},
        {'x_sum': 5.0, 'y_copy': [2., 3., 4., 5.]},
        {'x_sum': 9.0, 'y_copy': [6., 7.]}
    ]
    expected_transformed_schema = {
        # We expect x_sum to have a shape of None because the shape of
        # sparse_reduce_sum is not statically known.
        'x_sum': tf.FixedLenFeature(None, tf.float32, None),
        'y_copy': tf.VarLenFeature(tf.float32)
    }
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_schema))

    # Take the transform function and use TransformDataset to apply it to
    # some eval data, and compare with expected output.
    eval_data = [
        {'val': [9.], 'idx': [0], 'y': [9.]},
        {'val': [], 'idx': [], 'y': []},
        {'val': [8., 7.], 'idx': [2, 4], 'y': [8., 7.]}
    ]
    transformed_eval_dataset = (
        ((eval_data, input_schema), transform_fn)
        | self.impl.TransformDataset())

    expected_transformed_eval_values = [
        {'x_sum': 9., 'y_copy': [9.]},
        {'x_sum': 0., 'y_copy': []},
        {'x_sum': 15., 'y_copy': [8., 7.]}
    ]
    self.assertDatasetsEqual(
        transformed_eval_dataset,
        (expected_transformed_eval_values, expected_transformed_schema))

  def testTransform(self):
    # User defined preprocessing_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):
      return {'ab': api.transform(tf.multiply, inputs['a'], inputs['b'])}

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
    input_schema = {
        'a': tf.FixedLenFeature((), tf.float32, 0),
        'b': tf.FixedLenFeature((), tf.float32, 0)
    }
    transformed_dataset, _ = (
        (input_data, input_schema)
        | self.impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [{
        'ab': 12
    }, {
        'ab': 2
    }, {
        'ab': 30
    }, {
        'ab': 6
    }]
    expected_transformed_schema = {
        'ab': tf.FixedLenFeature((), tf.float32, None)
    }
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_schema))

  def testComposedTransforms(self):
    # User defined preprocessing_fn accepts and returns a dict of Columns.
    def preprocessing_fn(inputs):
      return {
          'a(b+c)':
              api.transform(tf.multiply, inputs['a'],
                            api.transform(tf.add, inputs['b'], inputs['c']))
      }

    input_data = [{'a': 4, 'b': 3, 'c': 3}, {'a': 1, 'b': 2, 'c': 1}]
    input_schema = {
        'a': tf.FixedLenFeature((), tf.float32, 0),
        'b': tf.FixedLenFeature((), tf.float32, 0),
        'c': tf.FixedLenFeature((), tf.float32, 0)
    }
    transformed_dataset, _ = (
        (input_data, input_schema) |
        self.impl.AnalyzeAndTransformDataset(preprocessing_fn))

    expected_transformed_data = [{'a(b+c)': 24}, {'a(b+c)': 3}]
    expected_transformed_schema = {
        'a(b+c)': tf.FixedLenFeature((), tf.float32, None)
    }
    self.assertDatasetsEqual(
        transformed_dataset,
        (expected_transformed_data, expected_transformed_schema))
