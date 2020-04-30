# Copyright 2020 Google Inc. All Rights Reserved.
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
"""Tests for saved_transform_io_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

# GOOGLE-INITIALIZATION
import numpy as np
import tensorflow as tf
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.saved import saved_transform_io_v2
import tensorflow_transform.test_case as tft_test_case

from tensorflow.core.protobuf import meta_graph_pb2  # pylint: disable=g-direct-tensorflow-import


# TODO(b/123241798): Find an open-source compatible way to access
# FLAGS.test_tmpdir.
def _create_test_saved_model():
  export_path = os.path.join(tempfile.mkdtemp(), 'export')

  with tf.compat.v1.Graph().as_default():
    with tf.compat.v1.Session().as_default() as session:
      input_float = tf.compat.v1.placeholder(tf.float32, shape=[1])
      output = (input_float - 2.0) / 5.0
      inputs = {'x': input_float}
      outputs = {'x_scaled': output}
      saved_transform_io.write_saved_transform_from_session(
          session, inputs, outputs, export_path)

  return export_path


class SavedTransformIOV2Test(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    tft_test_case.skip_if_not_tf2('Tensorflow 2.x required.')
    cls._test_saved_model = _create_test_saved_model()
    cls._saved_model_loader = saved_transform_io_v2.SavedModelLoader(
        cls._test_saved_model)

  def test_apply_saved_transform(self):
    input_floats = tf.constant([1237.0])  # tf.float32
    input_features = {'x': input_floats}
    transformed_features = (
        self._saved_model_loader.apply_v1_transform_model_in_v2(input_features))
    self.assertEqual(['x_scaled'], list(transformed_features))
    result_tensor = transformed_features['x_scaled']
    self.assertIsInstance(result_tensor, tf.Tensor)
    self.assertAllEqual(result_tensor.numpy(), [247.0])

  def test_apply_transform_extra_features_no_passthrough(self):
    with self.assertRaises(ValueError):
      input_floats = tf.constant([1237.0])  # tf.float32
      input_features = {
          'x': input_floats,
          'extra_1': tf.constant('1'),
          'extra_2': tf.constant('2')
      }
      self._saved_model_loader.apply_v1_transform_model_in_v2(input_features)

  def test_apply_transform_type_mismatch(self):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      input_strings = tf.constant(['bogus'])  # tf.string
      input_features = {'x': input_strings}
      self._saved_model_loader.apply_v1_transform_model_in_v2(input_features)

  def test_apply_transform_shape_mismatch(self):
    with self.assertRaises(ValueError):
      input_floats = tf.constant(1237.0)  # tf.float32
      input_features = {'x': input_floats}
      self._saved_model_loader.apply_v1_transform_model_in_v2(input_features)

  def test_apply_saved_transform_to_tensor_inside_scope(self):
    with tf.compat.v1.name_scope('my_scope'):
      input_floats = tf.constant([1237.0])  # tf.float32
      input_features = {'x': input_floats}
      transformed_features = (
          self._saved_model_loader.apply_v1_transform_model_in_v2(
              input_features))
      self.assertEqual(['x_scaled'], list(transformed_features))
      result_tensor = transformed_features['x_scaled']
      self.assertIsInstance(result_tensor, tf.Tensor)
      self.assertAllEqual(result_tensor.numpy(), [247.0])

  def test_apply_saved_transform_to_tensor_outside_scope(self):
    input_floats = tf.constant([1237.0])  # tf.float32
    with tf.compat.v1.name_scope('my_scope'):
      input_features = {'x': input_floats}
      transformed_features = (
          self._saved_model_loader.apply_v1_transform_model_in_v2(
              input_features))
      self.assertEqual(['x_scaled'], list(transformed_features))
      result_tensor = transformed_features['x_scaled']
      self.assertIsInstance(result_tensor, tf.Tensor)
      self.assertAllEqual(result_tensor.numpy(), [247.0])

  def test_dense_roundtrip(self):
    export_path = os.path.join(tempfile.mkdtemp(), 'export')

    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        input_float = tf.compat.v1.placeholder(tf.float32)
        # show that unrelated & unmapped placeholders do not interfere
        tf.compat.v1.placeholder(tf.int64)
        output = input_float / 5.0
        inputs = {'input': input_float}
        outputs = {'output': output}
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)

    # Using a computed input gives confidence that the graphs are fused.
    input_float = tf.constant(25.0) * 2
    inputs = {'input': input_float}
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
    outputs = saved_model_loader.apply_v1_transform_model_in_v2(inputs)
    # (25 * 2) / 5 = 10
    self.assertEqual(10.0, outputs['output'].numpy())

  def test_table_roundtrip(self):
    export_path = os.path.join(tempfile.mkdtemp(), 'export')

    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        input_string = tf.compat.v1.placeholder(tf.string)
        # Map string through a table, in this case based on a constant tensor.
        table_keys = ['cat', 'dog', 'giraffe']
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=table_keys,
            values=tf.cast(tf.range(len(table_keys)), tf.int64),
            key_dtype=tf.string,
            value_dtype=tf.int64)
        table = tf.lookup.StaticHashTable(initializer, default_value=-1)

        output = table.lookup(input_string)
        inputs = {'input': input_string}
        outputs = {'output': output}
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)

    # Using a computed input gives confidence that the graphs are fused.
    input_string = tf.constant('dog')
    inputs = {'input': input_string}
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
    outputs = saved_model_loader.apply_v1_transform_model_in_v2(inputs)
    self.assertEqual(1, outputs['output'].numpy())

  def test_sparse_roundtrip(self):
    export_path = os.path.join(tempfile.mkdtemp(), 'export')

    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        input_float = tf.compat.v1.sparse_placeholder(tf.float32)
        output = input_float / 5.0
        inputs = {'input': input_float}
        outputs = {'output': output}
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)

    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([7, 9, 2], dtype=np.int64)
    input_sparse = tf.SparseTensor(
        indices=indices, values=values, dense_shape=shape)

    # Using a computed input gives confidence that the graphs are fused
    inputs = {'input': input_sparse * 10}
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
    outputs = saved_model_loader.apply_v1_transform_model_in_v2(inputs)
    result = outputs['output']
    self.assertIsInstance(result, tf.SparseTensor)

    # indices and shape unchanged; values multiplied by 10 and divided by 5
    self.assertEqual(indices.tolist(), result.indices.numpy().tolist())
    self.assertEqual([2.0, 4.0], result.values.numpy().tolist())
    self.assertEqual(shape.tolist(), result.dense_shape.numpy().tolist())

  def test_ragged_roundtrip(self):
    if not hasattr(meta_graph_pb2.TensorInfo, 'CompositeTensor'):
      self.skipTest('This version of TensorFlow does not support '
                    'CompositeTenors in TensorInfo.')
    export_path = os.path.join(tempfile.mkdtemp(), 'export')

    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        input_float = tf.compat.v1.ragged.placeholder(tf.float32, ragged_rank=1,
                                                      value_shape=[])
        output = input_float / 2.0
        inputs = {'input': input_float}
        outputs = {'output': output}
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)

    splits = np.array([0, 2, 3], dtype=np.int64)
    values = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    input_ragged = tf.RaggedTensor.from_row_splits(values, splits)

    # Using a computed input gives confidence that the graphs are fused
    inputs = {'input': input_ragged * 10}
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
    outputs = saved_model_loader.apply_v1_transform_model_in_v2(inputs)
    result = outputs['output']
    self.assertIsInstance(result, tf.RaggedTensor)

    # indices and shape unchanged; values multipled by 10 and divided by 2
    self.assertAllEqual(splits, result.row_splits)
    self.assertEqual([5.0, 10.0, 20.0], result.values.numpy().tolist())


if __name__ == '__main__':
  tf.test.main()
