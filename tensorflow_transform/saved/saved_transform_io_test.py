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
"""Tests for saved_transform_io."""

import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow_transform.saved import saved_transform_io

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import lookup_ops
# pylint: enable=g-direct-tensorflow-import


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


class SavedTransformIOTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls._test_saved_model = _create_test_saved_model()

  def test_apply_saved_transform(self):
    with tf.compat.v1.Graph().as_default() as graph:
      with tf.compat.v1.Session().as_default() as session:
        input_floats = tf.constant([1237.0])  # tf.float32
        input_features = {'x': input_floats}
        _, transformed_features = (
            saved_transform_io.partially_apply_saved_transform_internal(
                self._test_saved_model, input_features))
        self.assertEqual(['x_scaled'], list(transformed_features))
        result_tensor = transformed_features['x_scaled']
        self.assertIsInstance(result_tensor, tf.Tensor)

        self.assertAllEqual(session.run(result_tensor), [247.0])
        self.assertEqual(graph.get_tensor_by_name('Const:0'), input_floats)
        self.assertEqual(
            graph.get_tensor_by_name('transform/truediv:0'),
            result_tensor)

  def test_apply_transform_extra_features_no_passthrough(self):
    with self.assertRaises(ValueError):
      with tf.compat.v1.Graph().as_default():
        with tf.compat.v1.Session().as_default():
          input_floats = tf.constant([1234.0])  # tf.float32
          input_features = {'x': input_floats,
                            'extra_1': tf.constant('1'),
                            'extra_2': tf.constant('2')}
          saved_transform_io.partially_apply_saved_transform_internal(
              self._test_saved_model, input_features)

  def test_apply_transform_type_mismatch(self):
    with self.assertRaises(ValueError):
      with tf.compat.v1.Graph().as_default():
        with tf.compat.v1.Session().as_default():
          input_strings = tf.constant(['bogus'])  # tf.string
          input_features = {'x': input_strings}
          saved_transform_io.partially_apply_saved_transform_internal(
              self._test_saved_model, input_features)

  def test_apply_transform_shape_mismatch(self):
    with self.assertRaises(ValueError):
      with tf.compat.v1.Graph().as_default():
        with tf.compat.v1.Session().as_default():
          input_floats = tf.constant(1234.0)  # tf.float32
          input_features = {'x': input_floats}
          saved_transform_io.partially_apply_saved_transform_internal(
              self._test_saved_model, input_features)

  def test_apply_saved_transform_to_tensor_inside_scope(self):
    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.name_scope('my_scope'):
        with tf.compat.v1.Session().as_default() as session:
          input_floats = tf.constant([1237.0])  # tf.float32
          input_features = {'x': input_floats}
          _, transformed_features = (
              saved_transform_io.partially_apply_saved_transform_internal(
                  self._test_saved_model, input_features))
          self.assertEqual(['x_scaled'], list(transformed_features))
          result_tensor = transformed_features['x_scaled']
          self.assertAllEqual(session.run(result_tensor), [247.0])

  def test_apply_saved_transform_to_tensor_outside_scope(self):
    with tf.compat.v1.Graph().as_default():
      input_floats = tf.constant([1237.0])  # tf.float32
      with tf.compat.v1.name_scope('my_scope'):
        with tf.compat.v1.Session().as_default() as session:
          input_features = {'x': input_floats}
          _, transformed_features = (
              saved_transform_io.partially_apply_saved_transform_internal(
                  self._test_saved_model, input_features))
          self.assertEqual(['x_scaled'], list(transformed_features))
          result_tensor = transformed_features['x_scaled']
          self.assertAllEqual(session.run(result_tensor), [247.0])

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

    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        # Using a computed input gives confidence that the graphs are fused.
        input_float = tf.constant(25.0) * 2
        inputs = {'input': input_float}
        _, outputs = (
            saved_transform_io.partially_apply_saved_transform_internal(
                export_path, inputs))
        result = session.run(outputs['output'])
        # (25 * 2) / 5 = 10
        self.assertEqual(10.0, result)

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

    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        # Using a computed input gives confidence that the graphs are fused.
        input_string = tf.constant('dog')
        inputs = {'input': input_string}
        _, outputs = (
            saved_transform_io.partially_apply_saved_transform_internal(
                export_path, inputs))
        session.run(tf.compat.v1.tables_initializer())
        result = session.run(outputs['output'])
        self.assertEqual(1, result)

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

    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
        values = np.array([1.0, 2.0], dtype=np.float32)
        shape = np.array([7, 9, 2], dtype=np.int64)
        input_sparse = tf.SparseTensor(
            indices=indices, values=values, dense_shape=shape)

        # Using a computed input gives confidence that the graphs are fused
        inputs = {'input': input_sparse * 10}
        _, outputs = (
            saved_transform_io.partially_apply_saved_transform_internal(
                export_path, inputs))
        output_sparse = outputs['output']
        self.assertIsInstance(output_sparse, tf.SparseTensor)
        result = session.run(output_sparse)

        # indices and shape unchanged; values multiplied by 10 and divided by 5
        self.assertEqual(indices.tolist(), result.indices.tolist())
        self.assertEqual([2.0, 4.0], result.values.tolist())
        self.assertEqual(shape.tolist(), result.dense_shape.tolist())

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

    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        splits = np.array([0, 2, 3], dtype=np.int64)
        values = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        input_ragged = tf.RaggedTensor.from_row_splits(values, splits)

        # Using a computed input gives confidence that the graphs are fused
        inputs = {'input': input_ragged * 10}
        _, outputs = (
            saved_transform_io.partially_apply_saved_transform_internal(
                export_path, inputs))
        output_ragged = outputs['output']
        self.assertIsInstance(output_ragged, tf.RaggedTensor)
        result = session.run(output_ragged)

        # indices and shape unchanged; values multipled by 10 and divided by 2
        self.assertAllEqual(splits, result.row_splits)
        self.assertEqual([5.0, 10.0, 20.0], result.values.tolist())

  def test_stale_asset_collections_are_cleaned(self):
    vocabulary_file = os.path.join(
        tf.compat.as_bytes(self.get_temp_dir()), tf.compat.as_bytes('asset'))
    file_io.write_string_to_file(vocabulary_file, 'foo bar baz')

    export_path = os.path.join(tempfile.mkdtemp(), 'export')

    # create a SavedModel including assets
    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        input_string = tf.compat.v1.placeholder(tf.string)
        # Map string through a table loaded from an asset file
        initializer = tf.lookup.TextFileInitializer(
            vocabulary_file,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
        table = tf.lookup.StaticHashTable(initializer, default_value=12)
        table = lookup_ops.IdTableWithHashBuckets(table,
                                                  num_oov_buckets=12,
                                                  key_dtype=tf.string)
        output = table.lookup(input_string)
        inputs = {'input': input_string}
        outputs = {'output': output}
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)

    # Load it and save it again repeatedly, verifying that the asset collections
    # remain valid.
    for _ in [1, 2, 3]:
      with tf.compat.v1.Graph().as_default() as g:
        with tf.compat.v1.Session().as_default() as session:
          input_string = tf.constant('dog')
          inputs = {'input': input_string}
          _, outputs = (
              saved_transform_io.partially_apply_saved_transform_internal(
                  export_path, inputs))

          self.assertEqual(
              1, len(g.get_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS)))
          self.assertEqual(0, len(g.get_collection(tf.saved_model.ASSETS_KEY)))

          # Check that every ASSET_FILEPATHS refers to a Tensor in the graph.
          # If not, get_tensor_by_name() raises KeyError.
          for asset_path in g.get_collection(
              tf.compat.v1.GraphKeys.ASSET_FILEPATHS):
            tensor_name = asset_path.name
            g.get_tensor_by_name(tensor_name)

          export_path = os.path.join(tempfile.mkdtemp(), 'export')
          saved_transform_io.write_saved_transform_from_session(
              session, inputs, outputs, export_path)

if __name__ == '__main__':
  tf.test.main()
