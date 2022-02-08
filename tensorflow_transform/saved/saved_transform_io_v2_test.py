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

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow_transform import graph_context
from tensorflow_transform import impl_helper
from tensorflow_transform import tf_utils
from tensorflow_transform import test_case
from tensorflow_transform.py_func.api import apply_pyfunc
from tensorflow_transform.saved import constants
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.saved import saved_transform_io_v2

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import script_ops
# pylint: enable=g-direct-tensorflow-import

_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES = [
    dict(testcase_name='_exported_in_tf1', exported_in_tf1=True),
    dict(testcase_name='_exported_in_tf2', exported_in_tf1=False)
]


def _get_preprocessing_fn_asset_table(asset_file):

  def construct_table(asset_path):
    initializer = tf.lookup.TextFileInitializer(
        asset_path,
        key_dtype=tf.string,
        key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
    return tf.lookup.StaticHashTable(initializer, default_value=-1)

  def preprocessing_fn(inputs):
    output, unused_table_size = tf_utils.construct_and_lookup_table(
        construct_table, asset_file, inputs['input'])
    return {'output': output}

  return preprocessing_fn


def _get_preprocessing_fn_non_asset_table(asset_file):
  del asset_file

  def preprocessing_fn(inputs):
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=['foo', 'bar', 'baz'],
        values=tf.cast(tf.range(3), tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64)
    table = tf.lookup.StaticHashTable(initializer, default_value=12)
    return {
        'output': table.lookup(inputs['input']),
    }

  return preprocessing_fn


_RE_EXPORT_TF2_TO_TF1_TEST_CASES = [
    dict(
        testcase_name='_asset_table',
        preprocessing_fn_getter=_get_preprocessing_fn_asset_table,
        expected_output=2,
        test_input='baz',
        asset_file_contents='foo\nbar\nbaz\n'),
    dict(
        testcase_name='_non_asset_table',
        preprocessing_fn_getter=_get_preprocessing_fn_non_asset_table,
        expected_output=2,
        test_input='baz'),
]


# TODO(b/123241798): Find an open-source compatible way to access
# FLAGS.test_tmpdir.
def _create_test_saved_model(export_in_tf1,
                             input_specs,
                             preprocessing_fn,
                             export_path_suffix=None,
                             base_dir=None):
  if not export_path_suffix:
    export_path = os.path.join(tempfile.mkdtemp(dir=base_dir), 'export')
  else:
    export_path = os.path.join(
        tempfile.mkdtemp(dir=base_dir), export_path_suffix)
  if export_in_tf1:
    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session().as_default() as session:
        inputs = {}
        for key in input_specs:
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
        outputs = preprocessing_fn(inputs)
        # show that unrelated & unmapped placeholders do not interfere
        tf.compat.v1.placeholder(tf.int64)
        saved_transform_io.write_saved_transform_from_session(
            session, inputs, outputs, export_path)
  else:
    module = tf.Module()
    tf_graph_context = graph_context.TFGraphContext(
        module_to_export=module, temp_dir=None, evaluated_replacements=None)
    transform_fn = impl_helper.get_traced_transform_fn(
        preprocessing_fn=preprocessing_fn,
        input_signature=input_specs,
        tf_graph_context=tf_graph_context,
        output_keys_to_name_map=None)

    saved_transform_io_v2.write_v2_saved_model(module, transform_fn,
                                               'transform_fn', export_path)
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

    def preprocessing_fn(inputs):
      output = (inputs['x'] - 2.0) / 5.0
      return {'x_scaled': output}

    cls._saved_model_path_v1 = _create_test_saved_model(True, input_specs,
                                                        preprocessing_fn,
                                                        'export_v1')
    cls._saved_model_path_v2 = _create_test_saved_model(False, input_specs,
                                                        preprocessing_fn,
                                                        'export_v2')

  def _get_saved_model_loader(self, exported_in_tf1):
    if exported_in_tf1:
      return saved_transform_io_v2.SavedModelLoader(self._saved_model_path_v1)
    return saved_transform_io_v2.SavedModelLoader(self._saved_model_path_v2)

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
    with self.assertRaises(tf.errors.InvalidArgumentError):
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

    def preprocessing_fn(inputs):
      return {'output': inputs['input'] / 5.0}

    export_path = _create_test_saved_model(
        exported_in_tf1,
        input_specs,
        preprocessing_fn,
        base_dir=self.get_temp_dir())

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

    def preprocessing_fn(inputs):
      table_keys = ['cat', 'dog', 'giraffe']
      initializer = tf.lookup.KeyValueTensorInitializer(
          keys=table_keys,
          values=tf.cast(tf.range(len(table_keys)), tf.int64),
          key_dtype=tf.string,
          value_dtype=tf.int64)
      table = tf.lookup.StaticHashTable(initializer, default_value=-1)
      return {'output': table.lookup(inputs['input'])}

    export_path = _create_test_saved_model(
        exported_in_tf1,
        input_specs,
        preprocessing_fn,
        base_dir=self.get_temp_dir())

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

    def preprocessing_fn(inputs):
      return {'output': inputs['input'] / 5.0}

    export_path = _create_test_saved_model(
        exported_in_tf1,
        input_specs,
        preprocessing_fn,
        base_dir=self.get_temp_dir())

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

    def preprocessing_fn(inputs):
      return {'output': inputs['input'] / 2.0}

    export_path = _create_test_saved_model(
        exported_in_tf1,
        input_specs,
        preprocessing_fn,
        base_dir=self.get_temp_dir())

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

  @test_case.named_parameters(*_TRANFORM_FN_EXPORT_TF_VERSION_TEST_CASES)
  def test_ragged_with_unfed(self, exported_in_tf1):
    input_specs = {
        'x': tf.RaggedTensorSpec([
            None,
            None,
        ], dtype=tf.float32),
        'y': tf.RaggedTensorSpec([
            None,
        ], dtype=tf.float32)
    }

    def preprocessing_fn(inputs):
      output = (inputs['x'] - 2.0) / 5.0
      return {'x_scaled': output, 'x_in': inputs['x'], 'y': inputs['y'] + 1}

    export_path = _create_test_saved_model(
        exported_in_tf1,
        input_specs,
        preprocessing_fn,
        base_dir=self.get_temp_dir())
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)

    # Missing 'y'.
    input_features = {'x': tf.ragged.constant([[1237.0]], ragged_rank=1)}
    transformed_features = (
        saved_model_loader.apply_transform_model(input_features))
    self.assertCountEqual(['x_in', 'x_scaled'], list(transformed_features))
    self.assertAllEqual(transformed_features['x_scaled'].numpy(), [[247.0]])
    self.assertAllEqual(transformed_features['x_in'].numpy(), [[1237.0]])

  @test_case.named_parameters(*_RE_EXPORT_TF2_TO_TF1_TEST_CASES)
  def test_re_export_tf2_saved_model_to_tf1(self,
                                            preprocessing_fn_getter,
                                            expected_output,
                                            test_input,
                                            asset_file_contents=None):

    asset_file = None
    if asset_file_contents is not None:
      asset_file_path = os.path.join(
          tempfile.mkdtemp(dir=self.get_temp_dir()), 'asset')
      file_io.write_string_to_file(asset_file_path, asset_file_contents)
      asset_file = tf.constant(asset_file_path)

    input_specs = {'input': tf.TensorSpec([], dtype=tf.string)}
    export_path = _create_test_saved_model(
        False,
        input_specs,
        preprocessing_fn_getter(asset_file),
        base_dir=self.get_temp_dir())

    if asset_file is not None:
      os.remove(asset_file.numpy())
    new_export_path = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()), 'export_v1')

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(
        new_export_path)
    # TODO(b/175844561): Investigate why the variable names need to be different
    # for the two graph and session contexts below.
    with tf.compat.v1.Graph().as_default() as g1:
      saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
      if asset_file_contents is not None:
        self.assertEqual(
            1, len(g1.get_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS)))
      with tf.compat.v1.Session().as_default() as s1:
        inputs = {'input': tf.compat.v1.placeholder(tf.string)}
        outputs = saved_model_loader.apply_transform_model(inputs)
        predict_signature_def = (
            tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                inputs, outputs))
        builder.add_meta_graph_and_variables(
            s1, ['graph_tag'],
            signature_def_map={'graph_signature': predict_signature_def},
            assets_collection=tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.ASSET_FILEPATHS),
            main_op=tf.compat.v1.tables_initializer())
    builder.save()

    shutil.rmtree(export_path)

    with tf.compat.v1.Graph().as_default() as g2:
      with tf.compat.v1.Session().as_default() as s2:
        meta_graph_def = tf.compat.v1.saved_model.loader.load(
            s2, ['graph_tag'], new_export_path)
        signature = meta_graph_def.signature_def['graph_signature']
        output = s2.run(
            g2.get_tensor_by_name(signature.outputs['output'].name),
            feed_dict={
                g2.get_tensor_by_name(signature.inputs['input'].name):
                    test_input
            })
        self.assertEqual(expected_output, output)
        if asset_file_contents is not None:
          self.assertEqual(
              1, len(g2.get_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS)))

  def test_stale_asset_collections_are_cleaned(self):
    vocabulary_file = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()), 'asset')
    file_io.write_string_to_file(vocabulary_file, 'foo bar baz')

    input_specs = {'input': tf.TensorSpec([], dtype=tf.string)}

    def preprocessing_fn(inputs):
      initializer = tf.lookup.TextFileInitializer(
          vocabulary_file,
          key_dtype=tf.string,
          key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
          value_dtype=tf.int64,
          value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
      table = tf.lookup.StaticHashTable(initializer, default_value=12)
      return {'output': table.lookup(inputs['input'])}

    export_path = _create_test_saved_model(
        False, input_specs, preprocessing_fn, base_dir=self.get_temp_dir())

    # Load it and save it again repeatedly, verifying that the asset collections
    # remain valid.
    for it in [1, 2, 3]:
      input_string = tf.constant('dog')
      inputs = {'input': input_string}
      saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)
      outputs = saved_model_loader.apply_transform_model(inputs)
      self.assertEqual(12, outputs['output'])

      new_export_path = os.path.join(
          tempfile.mkdtemp(dir=self.get_temp_dir()), 'export_' + str(it))
      tf.saved_model.save(saved_model_loader._imported, new_export_path)
      shutil.rmtree(export_path)
      export_path = new_export_path

  def test_finalize(self):
    input_keys = ['x']
    output_keys = ['x_scaled']

    input_specs = {
        'x': tf.TensorSpec([
            None,
        ], dtype=tf.float32),
        'y': tf.TensorSpec([
            None,
        ], dtype=tf.float32)
    }

    def preprocessing_fn(inputs):
      output = (inputs['x'] - 2.0) / 5.0
      return {'x_scaled': output, 'x_in': inputs['x'], 'y': inputs['y'] + 1}

    export_path = _create_test_saved_model(
        False, input_specs, preprocessing_fn, base_dir=self.get_temp_dir())
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(export_path)

    input_features = {'x': tf.constant([1237.0])}  # tf.float32
    transformed_features = (
        saved_model_loader.apply_transform_model(input_features))
    self.assertCountEqual(['x_in', 'x_scaled'], list(transformed_features))
    self.assertAllEqual(transformed_features['x_scaled'].numpy(), [247.0])
    self.assertAllEqual(transformed_features['x_in'].numpy(), [1237.0])

    # Since `finalize` is not thread-safe it is not recommended to call it after
    # `apply_transform_model` has already been invoked. This is only for unit
    # testing behavior differences.
    saved_model_loader.finalize(input_keys, output_keys)
    transformed_features = (
        saved_model_loader.apply_transform_model(input_features))
    self.assertEqual(['x_scaled'], list(transformed_features))
    self.assertAllEqual(transformed_features['x_scaled'].numpy(), [247.0])

  @test_case.named_parameters(
      dict(
          testcase_name='_strip_control_dependencies',
          strip_control_dependencies=True),
      dict(
          testcase_name='_keep_control_dependencies',
          strip_control_dependencies=False))
  def test_optimize_concrete_function(self, strip_control_dependencies):

    @tf.function(input_signature=[tf.TensorSpec([], dtype=tf.int64)])
    def func(x):
      z = x + 2
      with tf.init_scope():
        initializer = tf.lookup.KeyValueTensorInitializer([0, 1, 2],
                                                          ['a', 'b', 'c'],
                                                          key_dtype=tf.int64,
                                                          value_dtype=tf.string)
        table = tf.lookup.StaticHashTable(initializer, default_value='NAN')
      _ = table.lookup(x)
      return z

    concrete_function = func.get_concrete_function()
    optimized_function = saved_transform_io_v2.optimize_concrete_function(
        concrete_function,
        strip_control_dependencies=strip_control_dependencies)
    output = optimized_function(tf.constant(0, tf.int64))
    self.assertEqual(output, 2)

    if strip_control_dependencies:
      self.assertLess(
          len(optimized_function.graph.as_graph_def().node),
          len(concrete_function.graph.as_graph_def().node))
    else:
      self.assertEqual(
          len(optimized_function.graph.as_graph_def().node),
          len(concrete_function.graph.as_graph_def().node))

  def test_restore_from_v1_saved_model_with_pyfuncs(self):
    input_specs = {
        'a': tf.TensorSpec([
            None,
        ], dtype=tf.float32),
        'b': tf.TensorSpec([
            None,
        ], dtype=tf.float32),
    }

    def my_add(x, y):
      return x + y

    def func(inputs):
      result = {
          'a+b':
              apply_pyfunc(my_add, tf.float32, True, 'add', inputs['a'],
                           inputs['b'])
      }
      for value in result.values():
        value.set_shape([1])
      return result

    saved_model_path_v1 = _create_test_saved_model(True, input_specs, func,
                                                   'export_v1')
    # Clear PyFunc registry to mimic loading a SavedModel in a new runtime.
    script_ops._py_funcs._funcs.clear()  # pylint: disable=protected-access

    imported = tf.compat.v2.saved_model.load(saved_model_path_v1)
    imported_function = imported.signatures[constants.TRANSFORM_SIGNATURE]
    input_keys = ['a', 'b']
    inputs = [
        tf.constant([2.0], dtype=tf.float32),
        tf.constant([3.0], dtype=tf.float32)
    ]
    input_kwargs = {k: v for k, v in zip(input_keys, inputs)}
    expected_output = 5.0
    restored_function, _, _ = (
        saved_transform_io_v2._restore_from_v1_saved_model(
            imported_function, saved_model_path_v1))
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'callback.*pyfunc_'):
      imported_function(**input_kwargs)
    self.assertEqual(restored_function(*inputs)['a+b'], expected_output)

  def test_restore_from_v1_saved_model_without_pyfuncs(self):
    input_specs = {
        'a': tf.TensorSpec([
            None,
        ], dtype=tf.float32),
        'b': tf.TensorSpec([
            None,
        ], dtype=tf.float32),
    }

    def func(inputs):
      result = {'a+b': inputs['a'] + inputs['b']}
      for value in result.values():
        value.set_shape([1])
      return result

    saved_model_path_v1 = _create_test_saved_model(True, input_specs, func,
                                                   'export_v1')

    imported = tf.compat.v2.saved_model.load(saved_model_path_v1)
    imported_function = imported.signatures[constants.TRANSFORM_SIGNATURE]
    input_kwargs = {
        'a': tf.constant([2.0], dtype=tf.float32),
        'b': tf.constant([3.0], dtype=tf.float32)
    }
    expected_output = 5.0
    restored_function, _, _ = (
        saved_transform_io_v2._restore_from_v1_saved_model(
            imported_function, saved_model_path_v1))
    self.assertEqual(imported_function(**input_kwargs)['a+b'], expected_output)
    self.assertEqual(restored_function(**input_kwargs)['a+b'], expected_output)


if __name__ == '__main__':
  test_case.main()
