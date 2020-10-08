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
import shutil
import tempfile

# GOOGLE-INITIALIZATION
import numpy as np
import six
import tensorflow as tf
from tensorflow_transform import impl_helper
from tensorflow_transform import test_case
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.saved import saved_transform_io_v2

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.lib.io import file_io
# pylint: enable=g-direct-tensorflow-import

_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES = [
    dict(testcase_name='_exported_in_tf1', exported_in_tf1=True),
    dict(testcase_name='_exported_in_tf2', exported_in_tf1=False)
]


# TODO(b/123241798): Find an open-source compatible way to access
# FLAGS.test_tmpdir.
def _create_test_saved_model(export_in_tf1,
                             input_specs,
                             foo,
                             export_path_suffix=None):
  if not export_path_suffix:
    export_path = os.path.join(tempfile.mkdtemp(), 'export')
  else:
    export_path = os.path.join(tempfile.mkdtemp(), export_path_suffix)
  if export_in_tf1:
    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        inputs = {}
        for key in six.iterkeys(input_specs):
          tensor_spec = input_specs[key]
          if isinstance(tensor_spec, tf.TensorSpec):
            inputs[key] = tf.compat.v1.placeholder(
                tensor_spec.dtype, shape=tensor_spec.shape)
          elif isinstance(tensor_spec, tf.SparseTensorSpec):
            inputs[key] = tf.compat.v1.sparse_placeholder(
                tensor_spec.dtype, shape=tensor_spec.shape)
          elif isinstance(tensor_spec, tf.RaggedTensorSpec):
            inputs[key] = tf.compat.v1.ragged.placeholder(
                tensor_spec._dtype, tensor_spec._ragged_rank, [])
          else:
            raise ValueError(
                'TypeSpecs specified should be one of `tf.TensorSpec`, '
                '`tf.SparseTensorSpec`, `tf.RaggedTensorSpec`')
        outputs = foo(inputs)
        # show that unrelated & unmapped placeholders do not interfere
        tf.compat.v1.placeholder(tf.int64)
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)
  else:
    impl_helper.trace_and_write_v2_saved_model(
        saved_model_dir=export_path,
        preprocessing_fn=foo,
        input_signature=input_specs,
        base_temp_dir=None,
        tensor_replacement_map=None,
        output_keys_to_name_map=None)
  return export_path


class SavedTransformIOV2Test(test_case.TransformTestCase):

  @classmethod
  def setUpClass(cls):
    test_case.skip_if_not_tf2('Tensorflow 2.x required.')
    input_specs = {
        'x': tf.TensorSpec([
            None,
        ], dtype=tf.float32)
    }

    def foo(inputs):
      output = (inputs['x'] - 2.0) / 5.0
      return {'x_scaled': output}

    cls._saved_model_path_v1 = _create_test_saved_model(True, input_specs, foo,
                                                        'export_v1')
    cls._saved_model_loader_v1 = saved_transform_io_v2.SavedModelLoader(
        cls._saved_model_path_v1)
    cls._saved_model_path_v2 = _create_test_saved_model(False, input_specs, foo,
                                                        'export_v2')
    cls._saved_model_loader_v2 = saved_transform_io_v2.SavedModelLoader(
        cls._saved_model_path_v2)

  def _get_saved_model_loader(self, exported_in_tf1):
    if exported_in_tf1:
      return self._saved_model_loader_v1
    return self._saved_model_loader_v2

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_apply_saved_transform(self, exported_in_tf1):
    input_floats = tf.constant([1237.0])  # tf.float32
    input_features = {'x': input_floats}
    transformed_features = (
        self._get_saved_model_loader(exported_in_tf1).apply_transform_model(
            input_features))
    self.assertEqual(['x_scaled'], list(transformed_features))
    result_tensor = transformed_features['x_scaled']
    self.assertIsInstance(result_tensor, tf.Tensor)
    self.assertAllEqual(result_tensor.numpy(), [247.0])

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_apply_saved_transform_dataset_map(self, exported_in_tf1):
    ds = tf.data.Dataset.from_tensor_slices({'x': [[1237.0]]})
    model_loader = self._get_saved_model_loader(exported_in_tf1)

    def map_fn(inputs):
      result = model_loader.apply_transform_model(inputs)
      self.assertEqual(['x_scaled'], list(result))
      result_tensor = result['x_scaled']
      self.assertIsInstance(result_tensor, tf.Tensor)
      self.assertEqual(result_tensor.shape.as_list(), [1])
      return result

    result_ds = ds.map(map_fn)
    self.assertAllEqual(
        list(result_ds.as_numpy_iterator()), [{
            'x_scaled': [247.0]
        }])

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_apply_transform_extra_features_no_passthrough(self, exported_in_tf1):
    with self.assertRaises(ValueError):
      input_floats = tf.constant([1237.0])  # tf.float32
      input_features = {
          'x': input_floats,
          'extra_1': tf.constant('1'),
          'extra_2': tf.constant('2')
      }
      self._get_saved_model_loader(exported_in_tf1).apply_transform_model(
          input_features)

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_apply_transform_type_mismatch(self, exported_in_tf1):
    if exported_in_tf1:
      exception_type = tf.errors.InvalidArgumentError
    else:
      exception_type = ValueError
    with self.assertRaises(exception_type):
      input_strings = tf.constant(['bogus'])  # tf.string
      input_features = {'x': input_strings}
      self._get_saved_model_loader(exported_in_tf1).apply_transform_model(
          input_features)

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_apply_transform_shape_mismatch(self, exported_in_tf1):
    with self.assertRaises(ValueError):
      input_floats = tf.constant(1237.0)  # tf.float32
      input_features = {'x': input_floats}
      self._get_saved_model_loader(exported_in_tf1).apply_transform_model(
          input_features)

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_apply_saved_transform_to_tensor_inside_scope(self, exported_in_tf1):
    with tf.compat.v1.name_scope('my_scope'):
      input_floats = tf.constant([1237.0])  # tf.float32
      input_features = {'x': input_floats}
      transformed_features = (
          self._get_saved_model_loader(exported_in_tf1).apply_transform_model(
              input_features))
      self.assertEqual(['x_scaled'], list(transformed_features))
      result_tensor = transformed_features['x_scaled']
      self.assertIsInstance(result_tensor, tf.Tensor)
      self.assertAllEqual(result_tensor.numpy(), [247.0])

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_apply_saved_transform_to_tensor_outside_scope(self, exported_in_tf1):
    input_floats = tf.constant([1237.0])  # tf.float32
    with tf.compat.v1.name_scope('my_scope'):
      input_features = {'x': input_floats}
      transformed_features = (
          self._get_saved_model_loader(exported_in_tf1).apply_transform_model(
              input_features))
      self.assertEqual(['x_scaled'], list(transformed_features))
      result_tensor = transformed_features['x_scaled']
      self.assertIsInstance(result_tensor, tf.Tensor)
      self.assertAllEqual(result_tensor.numpy(), [247.0])

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_dense_roundtrip(self, exported_in_tf1):
    input_specs = {'input': tf.TensorSpec([], dtype=tf.float32)}

    def foo(inputs):
      return {'output': inputs['input'] / 5.0}

    export_path = _create_test_saved_model(exported_in_tf1, input_specs, foo)

    # Using a computed input gives confidence that the graphs are fused.
    input_float = tf.constant(25.0) * 2
    inputs = {'input': input_float}
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
    outputs = saved_model_loader.apply_transform_model(inputs)
    # (25 * 2) / 5 = 10
    self.assertEqual(10.0, outputs['output'].numpy())

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_table_roundtrip(self, exported_in_tf1):
    input_specs = {'input': tf.TensorSpec([], dtype=tf.string)}

    def foo(inputs):
      table_keys = ['cat', 'dog', 'giraffe']
      initializer = tf.lookup.KeyValueTensorInitializer(
          keys=table_keys,
          values=tf.cast(tf.range(len(table_keys)), tf.int64),
          key_dtype=tf.string,
          value_dtype=tf.int64)
      table = tf.lookup.StaticHashTable(initializer, default_value=-1)
      return {'output': table.lookup(inputs['input'])}

    export_path = _create_test_saved_model(exported_in_tf1, input_specs, foo)

    # Using a computed input gives confidence that the graphs are fused.
    input_string = tf.constant('dog')
    inputs = {'input': input_string}
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
    outputs = saved_model_loader.apply_transform_model(inputs)
    self.assertEqual(1, outputs['output'].numpy())

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_sparse_roundtrip(self, exported_in_tf1):
    input_specs = {
        'input': tf.SparseTensorSpec([None, None, None], dtype=tf.float32)
    }

    def foo(inputs):
      return {'output': inputs['input'] / 5.0}

    export_path = _create_test_saved_model(exported_in_tf1, input_specs, foo)

    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([7, 9, 2], dtype=np.int64)
    input_sparse = tf.SparseTensor(
        indices=indices, values=values, dense_shape=shape)

    # Using a computed input gives confidence that the graphs are fused
    inputs = {'input': input_sparse * 10}
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
    outputs = saved_model_loader.apply_transform_model(inputs)
    result = outputs['output']
    self.assertIsInstance(result, tf.SparseTensor)

    # indices and shape unchanged; values multiplied by 10 and divided by 5
    self.assertEqual(indices.tolist(), result.indices.numpy().tolist())
    self.assertEqual([2.0, 4.0], result.values.numpy().tolist())
    self.assertEqual(shape.tolist(), result.dense_shape.numpy().tolist())

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_ragged_roundtrip(self, exported_in_tf1):
    if not hasattr(meta_graph_pb2.TensorInfo, 'CompositeTensor'):
      self.skipTest('This version of TensorFlow does not support '
                    'CompositeTenors in TensorInfo.')
    input_specs = {
        'input':
            tf.RaggedTensorSpec(
                shape=[None, None],
                dtype=tf.float32,
                ragged_rank=1,
                row_splits_dtype=tf.int64)
    }

    def foo(inputs):
      return {'output': inputs['input'] / 2.0}

    export_path = _create_test_saved_model(exported_in_tf1, input_specs, foo)

    splits = np.array([0, 2, 3], dtype=np.int64)
    values = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    input_ragged = tf.RaggedTensor.from_row_splits(values, splits)

    # Using a computed input gives confidence that the graphs are fused
    inputs = {'input': input_ragged * 10}
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
    outputs = saved_model_loader.apply_transform_model(inputs)
    result = outputs['output']
    self.assertIsInstance(result, tf.RaggedTensor)

    # indices and shape unchanged; values multipled by 10 and divided by 2
    self.assertAllEqual(splits, result.row_splits)
    self.assertEqual([5.0, 10.0, 20.0], result.values.numpy().tolist())

  def test_stale_asset_collections_are_cleaned(self):
    exported_in_tf_1 = False
    vocabulary_file = os.path.join(tempfile.mkdtemp(), 'asset')
    file_io.write_string_to_file(vocabulary_file, 'foo bar baz')

    input_specs = {'input': tf.TensorSpec([], dtype=tf.string)}

    def foo(inputs):
      initializer = tf.lookup.TextFileInitializer(
          vocabulary_file,
          key_dtype=tf.string,
          key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
          value_dtype=tf.int64,
          value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
      table = tf.lookup.StaticHashTable(initializer, default_value=12)
      return {'output': table.lookup(inputs['input'])}

    export_path = _create_test_saved_model(exported_in_tf_1, input_specs, foo)

    # Load it and save it again repeatedly, verifying that the asset collections
    # remain valid.
    for it in [1, 2, 3]:
      input_string = tf.constant('dog')
      inputs = {'input': input_string}
      saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
      outputs = saved_model_loader.apply_transform_model(inputs)
      self.assertEqual(12, outputs['output'])

      new_export_path = os.path.join(tempfile.mkdtemp(), 'export_' + str(it))
      tf.saved_model.save(saved_model_loader._imported, new_export_path)
      shutil.rmtree(export_path)
      export_path = new_export_path


if __name__ == '__main__':
  test_case.main()
