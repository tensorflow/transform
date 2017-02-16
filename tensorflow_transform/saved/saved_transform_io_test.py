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
"""Tests for dataset_metadata.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow_transform.saved import saved_transform_io

import unittest
from tensorflow.contrib import lookup


def _create_test_saved_model():
  export_path = os.path.join(tempfile.mkdtemp(), 'export')

  with tf.Graph().as_default():
    with tf.Session().as_default() as session:
      input_float = tf.placeholder(tf.float32, shape=[1])
      output = (input_float - 2.0) / 5.0
      inputs = {'x': input_float}
      outputs = {'x_scaled': output}
      saved_transform_io.write_saved_transform_from_session(
          session, inputs, outputs, export_path)

  return export_path


class SavedTransformIOTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls._test_saved_model = _create_test_saved_model()

  def test_apply_saved_transform(self):
    with tf.Graph().as_default() as graph:
      with tf.Session().as_default():
        input_floats = tf.constant([1234.0])  # tf.float32
        input_features = {'x': input_floats}
        transformed_features = saved_transform_io.apply_saved_transform(
            self._test_saved_model, input_features)
        self.assertEqual(['x_scaled'], transformed_features.keys())
        result_tensor = transformed_features['x_scaled']
        self.assertTrue(isinstance(result_tensor, tf.Tensor))

        self.assertEqual(graph.get_tensor_by_name('Const:0'), input_floats)
        self.assertEqual(
            graph.get_tensor_by_name('transform/truediv:0'),
            result_tensor)

  def test_apply_transform_extra_features_no_passthrough(self):
    with self.assertRaises(ValueError):
      with tf.Graph().as_default():
        with tf.Session().as_default():
          input_floats = tf.constant([1234.0])  # tf.float32
          input_features = {'x': input_floats,
                            'extra_1': tf.constant('1'),
                            'extra_2': tf.constant('2')}
          _ = saved_transform_io.apply_saved_transform(
              self._test_saved_model, input_features)

  def test_apply_transform_missing_features(self):
    with self.assertRaises(ValueError):
      with tf.Graph().as_default():
        with tf.Session().as_default():
          input_features = {}
          saved_transform_io.apply_saved_transform(
              self._test_saved_model, input_features)

  def test_apply_transform_type_mismatch(self):
    with self.assertRaises(ValueError):
      with tf.Graph().as_default():
        with tf.Session().as_default():
          input_strings = tf.constant(['bogus'])  # tf.string
          input_features = {'x': input_strings}
          saved_transform_io.apply_saved_transform(
              self._test_saved_model, input_features)

  def test_apply_transform_shape_mismatch(self):
    with self.assertRaises(ValueError):
      with tf.Graph().as_default():
        with tf.Session().as_default():
          input_floats = tf.constant(1234.0)  # tf.float32
          input_features = {'x': input_floats}
          saved_transform_io.apply_saved_transform(
              self._test_saved_model, input_features)

  def test_dense_roundtrip(self):
    export_path = os.path.join(tempfile.mkdtemp(), 'export')

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        input_float = tf.placeholder(tf.float32)
        # show that unrelated & unmapped placeholders do not interfere
        tf.placeholder(tf.int64)
        output = input_float / 5.0
        inputs = {'input': input_float}
        outputs = {'output': output}
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        # Using a computed input gives confidence that the graphs are fused.
        input_float = tf.constant(25.0) * 2
        inputs = {'input': input_float}
        outputs = saved_transform_io.apply_saved_transform(export_path, inputs)
        result = session.run(outputs['output'])
        # (25 * 2) / 5 = 10
        self.assertEqual(10.0, result)

  def test_table_roundtrip(self):
    export_path = os.path.join(tempfile.mkdtemp(), 'export')

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        input_string = tf.placeholder(tf.string)
        # Map string through a table, in this case based on a constant tensor.
        table = lookup.string_to_index_table_from_tensor(
            tf.constant(['cat', 'dog', 'giraffe']))
        output = table.lookup(input_string)
        inputs = {'input': input_string}
        outputs = {'output': output}
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        # Using a computed input gives confidence that the graphs are fused.
        input_string = tf.constant('dog')
        inputs = {'input': input_string}
        outputs = saved_transform_io.apply_saved_transform(export_path, inputs)
        session.run(tf.tables_initializer())
        result = session.run(outputs['output'])
        self.assertEqual(1, result)

  def test_sparse_roundtrip(self):
    export_path = os.path.join(tempfile.mkdtemp(), 'export')

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        input_float = tf.sparse_placeholder(tf.float32)
        output = input_float / 5.0
        inputs = {'input': input_float}
        outputs = {'output': output}
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
        values = np.array([1.0, 2.0], dtype=np.float32)
        shape = np.array([7, 9, 2], dtype=np.int64)
        input_sparse = tf.SparseTensor(
            indices=indices, values=values, dense_shape=shape)

        # Using a computed input gives confidence that the graphs are fused
        inputs = {'input': input_sparse * 10}
        outputs = saved_transform_io.apply_saved_transform(export_path, inputs)
        output_sparse = outputs['output']
        self.assertTrue(isinstance(output_sparse, tf.SparseTensor))
        result = session.run(output_sparse)

        # indices and shape unchanged; values divided by 2
        self.assertEqual(indices.tolist(), result.indices.tolist())
        self.assertEqual([2.0, 4.0], result.values.tolist())
        self.assertEqual(shape.tolist(), result.dense_shape.tolist())


if __name__ == '__main__':
  unittest.main()
