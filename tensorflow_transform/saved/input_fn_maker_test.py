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
"""Tests for schema_io_v1_json."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile

# GOOGLE-INITIALIZATION

import six
import tensorflow as tf

from tensorflow_transform import test_case
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata


class _MockSchema(object):
  """Mock object that allows feature specs not allowed by the actual Schema."""

  def __init__(self, feature_spec):
    self._feature_spec = feature_spec

  def as_feature_spec(self):
    return self._feature_spec


def _make_raw_schema(
    shape=None,
    should_add_unused_feature=False):
  feature_spec = {
      'raw_a': tf.FixedLenFeature(shape, tf.int64, 0),
      'raw_b': tf.FixedLenFeature(shape, tf.int64, 1),
      'raw_label': tf.FixedLenFeature(shape, tf.int64, -1),
  }
  if should_add_unused_feature:
    feature_spec['raw_unused'] = tf.FixedLenFeature(shape, tf.int64, 1)
  return _MockSchema(feature_spec=feature_spec)


def _make_transformed_schema(shape):
  feature_spec = {
      'transformed_a': tf.FixedLenFeature(shape, tf.int64),
      'transformed_b': tf.VarLenFeature(tf.int64),
      'transformed_label': tf.FixedLenFeature(shape, tf.int64),
  }
  return _MockSchema(feature_spec=feature_spec)


class InputFnMakerTest(test_case.TransformTestCase):

  def test_build_csv_transforming_serving_input_fn_with_defaults(self):
    feed_dict = [',,']

    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(schema=_make_raw_schema([]))

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(transform_savedmodel_dir)

    serving_input_fn = (
        input_fn_maker.build_csv_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            raw_keys=['raw_a', 'raw_b', 'raw_label'],
            transform_savedmodel_dir=transform_savedmodel_dir))

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, labels, inputs = serving_input_fn()

        self.assertCountEqual(
            outputs.keys(),
            {'transformed_a', 'transformed_b', 'transformed_label'})
        self.assertIsNone(labels)
        self.assertEqual(set(inputs.keys()), {'csv_example'})

        feed_inputs = {inputs['csv_example']: feed_dict}
        transformed_a, transformed_b, transformed_label = session.run(
            [outputs['transformed_a'], outputs['transformed_b'],
             outputs['transformed_label']],
            feed_dict=feed_inputs)

    self.assertEqual((1, 1), tuple(transformed_a.shape))
    # transformed_b is sparse so _convert_scalars_to_vectors did not fix it
    self.assertEqual((1,), tuple(transformed_b.dense_shape))
    self.assertEqual((1, 1), tuple(transformed_label.shape))

    transformed_b_dict = dict(zip([tuple(x)
                                   for x in transformed_b.indices.tolist()],
                                  transformed_b.values.tolist()))

    # Note the feed dict is empty. So these values come from the defaults
    # in _make_raw_schema()
    self.assertEqual(1, transformed_a[0][0])
    # transformed_b is sparse so _convert_scalars_to_vectors did not fix it
    self.assertEqual(-1, transformed_b_dict[(0,)])
    self.assertEqual(-1000, transformed_label[0][0])

  def test_build_csv_transforming_serving_input_fn_with_label(self):
    feed_dict = ['15,6,1', '12,17,2']

    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(schema=_make_raw_schema([]))

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(transform_savedmodel_dir)

    serving_input_fn = (
        input_fn_maker.build_csv_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            raw_keys=['raw_a', 'raw_b', 'raw_label'],
            transform_savedmodel_dir=transform_savedmodel_dir))

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, labels, inputs = serving_input_fn()

        self.assertCountEqual(
            outputs.keys(),
            {'transformed_a', 'transformed_b', 'transformed_label'})
        self.assertIsNone(labels)
        self.assertEqual(set(inputs.keys()), {'csv_example'})

        feed_inputs = {inputs['csv_example']: feed_dict}
        transformed_a, transformed_b, transformed_label = session.run(
            [outputs['transformed_a'], outputs['transformed_b'],
             outputs['transformed_label']],
            feed_dict=feed_inputs)

    batch_shape = (len(feed_dict), 1)

    # transformed_b is sparse so _convert_scalars_to_vectors did not fix it
    sparse_batch_shape = (len(feed_dict),)
    transformed_b_dict = dict(zip([tuple(x + [0])
                                   for x in transformed_b.indices.tolist()],
                                  transformed_b.values.tolist()))

    self.assertEqual(batch_shape, tuple(transformed_a.shape))
    self.assertEqual(sparse_batch_shape, tuple(transformed_b.dense_shape))
    self.assertEqual(batch_shape, tuple(transformed_label.shape))

    self.assertEqual(21, transformed_a[0][0])
    self.assertEqual(9, transformed_b_dict[(0, 0)])
    self.assertEqual(1000, transformed_label[0][0])
    self.assertEqual(29, transformed_a[1][0])
    self.assertEqual(-5, transformed_b_dict[(1, 0)])
    self.assertEqual(2000, transformed_label[1][0])

  def test_build_json_example_transforming_serving_input_fn(self):
    example_all = {
        'features': {
            'feature': {
                'raw_a': {
                    'int64List': {
                        'value': [42]
                    }
                },
                'raw_b': {
                    'int64List': {
                        'value': [43]
                    }
                },
                'raw_label': {
                    'int64List': {
                        'value': [44]
                    }
                }
            }
        }
    }
    # Default values for raw_a and raw_b come from _make_raw_schema()
    example_missing = {
        'features': {
            'feature': {
                'raw_label': {
                    'int64List': {
                        'value': [3]
                    }
                }
            }
        }
    }
    feed_dict = [json.dumps(example_all), json.dumps(example_missing)]

    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(schema=_make_raw_schema([]))

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(transform_savedmodel_dir)

    serving_input_fn = (
        input_fn_maker.build_json_example_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            raw_label_keys=[],
            raw_feature_keys=['raw_a', 'raw_b', 'raw_label'],
            transform_savedmodel_dir=transform_savedmodel_dir))

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, labels, inputs = serving_input_fn()

        self.assertCountEqual(
            outputs.keys(),
            {'transformed_a', 'transformed_b', 'transformed_label'})
        self.assertIsNone(labels)
        self.assertEqual(set(inputs.keys()), {'json_example'})

        feed_inputs = {inputs['json_example']: feed_dict}
        transformed_a, transformed_b, transformed_label = session.run(
            [outputs['transformed_a'], outputs['transformed_b'],
             outputs['transformed_label']],
            feed_dict=feed_inputs)

    batch_shape = (len(feed_dict), 1)

    # transformed_b is sparse so _convert_scalars_to_vectors did not fix it
    sparse_batch_shape = (len(feed_dict),)
    transformed_b_dict = dict(zip([tuple(x + [0])
                                   for x in transformed_b.indices.tolist()],
                                  transformed_b.values.tolist()))

    self.assertEqual(batch_shape, tuple(transformed_a.shape))
    self.assertEqual(sparse_batch_shape, tuple(transformed_b.dense_shape))
    self.assertEqual(batch_shape, tuple(transformed_label.shape))

    self.assertEqual(85, transformed_a[0][0])
    self.assertEqual(-1, transformed_b_dict[(0, 0)])
    self.assertEqual(44000, transformed_label[0][0])
    self.assertEqual(1, transformed_a[1][0])
    self.assertEqual(-1, transformed_b_dict[(1, 0)])
    self.assertEqual(3000, transformed_label[1][0])

  def test_build_parsing_transforming_serving_input_fn_scalars(self):
    self._test_build_parsing_transforming_serving_input_fn([])

  def test_build_parsing_transforming_serving_input_fn_scalars_with_label(self):
    self._test_build_parsing_transforming_serving_input_fn_with_label([])

  def test_build_parsing_transforming_serving_input_fn_vectors(self):
    self._test_build_parsing_transforming_serving_input_fn([1])

  def test_build_parsing_transforming_serving_input_fn_vectors_with_label(self):
    self._test_build_parsing_transforming_serving_input_fn_with_label([1])

  def _test_build_parsing_transforming_serving_input_fn_with_label(self, shape):
    # TODO(b/123241798): use TEST_TMPDIR
    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_raw_schema(shape))

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(transform_savedmodel_dir)

    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            transform_savedmodel_dir=transform_savedmodel_dir,
            raw_label_keys=[],  # Test labels are in output
            raw_feature_keys=None,
            convert_scalars_to_vectors=True))

    examples = [_create_serialized_example(d)
                for d in [
                    {'raw_a': 15, 'raw_b': 6, 'raw_label': 1},
                    {'raw_a': 12, 'raw_b': 17, 'raw_label': 2}]]

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, labels, inputs = serving_input_fn()

        self.assertCountEqual(
            set(outputs.keys()),
            {'transformed_a', 'transformed_b', 'transformed_label'})
        self.assertEqual(labels, None)
        self.assertEqual(set(inputs.keys()), {'examples'})

        feed_inputs = {inputs['examples']: examples}
        transformed_a, transformed_b, transformed_label = session.run(
            [outputs['transformed_a'], outputs['transformed_b'],
             outputs['transformed_label']],
            feed_dict=feed_inputs)

    batch_shape = (len(examples), 1)
    sparse_batch_shape = batch_shape

    if not shape:
      # transformed_b is sparse so _convert_scalars_to_vectors did not fix it
      sparse_batch_shape = sparse_batch_shape[:1]
      transformed_b_dict = dict(zip([tuple(x + [0])
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))
    else:
      transformed_b_dict = dict(zip([tuple(x)
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))

    self.assertEqual(batch_shape, tuple(transformed_a.shape))
    self.assertEqual(sparse_batch_shape, tuple(transformed_b.dense_shape))
    self.assertEqual(batch_shape, tuple(transformed_label.shape))

    self.assertEqual(21, transformed_a[0][0])
    self.assertEqual(9, transformed_b_dict[(0, 0)])
    self.assertEqual(1000, transformed_label[0][0])
    self.assertEqual(29, transformed_a[1][0])
    self.assertEqual(-5, transformed_b_dict[(1, 0)])
    self.assertEqual(2000, transformed_label[1][0])

  def _test_build_parsing_transforming_serving_input_fn(self, shape):
    # TODO(b/123241798): use TEST_TMPDIR
    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_raw_schema(shape, should_add_unused_feature=True))

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(
        transform_savedmodel_dir, should_add_unused_feature=True)

    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            transform_savedmodel_dir=transform_savedmodel_dir,
            raw_label_keys=['raw_label'],  # Labels are excluded
            raw_feature_keys=['raw_a', 'raw_b'],
            convert_scalars_to_vectors=True))

    examples = [_create_serialized_example(d)
                for d in [
                    {'raw_a': 15, 'raw_b': 6},
                    {'raw_a': 12, 'raw_b': 17}]]

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, labels, inputs = serving_input_fn()

        self.assertCountEqual(
            set(outputs.keys()),
            {'transformed_a', 'transformed_b', 'transformed_label'})
        self.assertEqual(labels, None)
        self.assertEqual(set(inputs.keys()), {'examples'})

        feed_inputs = {inputs['examples']: examples}
        transformed_a, transformed_b = session.run(
            [outputs['transformed_a'], outputs['transformed_b']],
            feed_dict=feed_inputs)

    batch_shape = (len(examples), 1)
    sparse_batch_shape = batch_shape

    if not shape:
      # transformed_b is sparse so _convert_scalars_to_vectors did not fix it
      sparse_batch_shape = sparse_batch_shape[:1]
      transformed_b_dict = dict(zip([tuple(x + [0])
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))
    else:
      transformed_b_dict = dict(zip([tuple(x)
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))

    self.assertEqual(batch_shape, tuple(transformed_a.shape))
    self.assertEqual(sparse_batch_shape, tuple(transformed_b.dense_shape))

    self.assertEqual(21, transformed_a[0][0])
    self.assertEqual(9, transformed_b_dict[(0, 0)])
    self.assertEqual(29, transformed_a[1][0])
    self.assertEqual(-5, transformed_b_dict[(1, 0)])

  def test_build_default_transforming_serving_input_fn_scalars(self):
    self._test_build_default_transforming_serving_input_fn(
        [], [[15, 12], [6, 17]])

  def test_build_default_transforming_serving_input_fn_scalars_with_label(self):
    self._test_build_default_transforming_serving_input_fn_with_label(
        [], [[15, 12], [6, 17], [1, 2]])

  def test_build_default_transforming_serving_input_fn_vectors(self):
    self._test_build_default_transforming_serving_input_fn(
        [1], [[[15], [12]], [[6], [17]]])

  def test_build_default_transforming_serving_input_fn_vectors_with_label(self):
    self._test_build_default_transforming_serving_input_fn_with_label(
        [1], [[[15], [12]], [[6], [17]], [[1], [2]]])

  def _test_build_default_transforming_serving_input_fn_with_label(
      self, shape, feed_input_values):
    # TODO(b/123241798): use TEST_TMPDIR
    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_raw_schema(shape))

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(transform_savedmodel_dir)

    serving_input_fn = (
        input_fn_maker.build_default_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            raw_label_keys=[],  # Test labels are in output
            raw_feature_keys=None,
            transform_savedmodel_dir=transform_savedmodel_dir,
            convert_scalars_to_vectors=True))

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, labels, inputs = serving_input_fn()

        self.assertCountEqual(
            set(outputs.keys()),
            {'transformed_a', 'transformed_b', 'transformed_label'})
        self.assertEqual(labels, None)
        self.assertEqual(set(inputs.keys()), {'raw_a', 'raw_b', 'raw_label'})

        feed_inputs = {inputs['raw_a']: feed_input_values[0],
                       inputs['raw_b']: feed_input_values[1],
                       inputs['raw_label']: feed_input_values[2]
                      }
        transformed_a, transformed_b, transformed_label = session.run(
            [outputs['transformed_a'], outputs['transformed_b'],
             outputs['transformed_label']],
            feed_dict=feed_inputs)

    batch_shape = (len(feed_input_values[0]), 1)
    sparse_batch_shape = batch_shape

    if not shape:
      # transformed_b is sparse so _convert_scalars_to_vectors did not fix it
      sparse_batch_shape = sparse_batch_shape[:1]
      transformed_b_dict = dict(zip([tuple(x + [0])
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))
    else:
      transformed_b_dict = dict(zip([tuple(x)
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))

    self.assertEqual(batch_shape, tuple(transformed_a.shape))
    self.assertEqual(sparse_batch_shape, tuple(transformed_b.dense_shape))
    self.assertEqual(batch_shape, tuple(transformed_label.shape))

    self.assertEqual(21, transformed_a[0][0])
    self.assertEqual(9, transformed_b_dict[(0, 0)])
    self.assertEqual(1000, transformed_label[0][0])
    self.assertEqual(29, transformed_a[1][0])
    self.assertEqual(-5, transformed_b_dict[(1, 0)])
    self.assertEqual(2000, transformed_label[1][0])

  def _test_build_default_transforming_serving_input_fn(
      self, shape, feed_input_values):
    # TODO(b/123241798): use TEST_TMPDIR
    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_raw_schema(shape, should_add_unused_feature=True))

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(
        transform_savedmodel_dir, should_add_unused_feature=True)

    serving_input_fn = (
        input_fn_maker.build_default_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            raw_label_keys=['raw_label'],
            raw_feature_keys=['raw_a', 'raw_b'],
            transform_savedmodel_dir=transform_savedmodel_dir,
            convert_scalars_to_vectors=True))

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, labels, inputs = serving_input_fn()

        self.assertCountEqual(
            set(outputs.keys()),
            {'transformed_a', 'transformed_b', 'transformed_label'})
        self.assertEqual(labels, None)
        self.assertEqual(set(inputs.keys()), {'raw_a', 'raw_b'})

        feed_inputs = {inputs['raw_a']: feed_input_values[0],
                       inputs['raw_b']: feed_input_values[1]}
        transformed_a, transformed_b = session.run(
            [outputs['transformed_a'], outputs['transformed_b']],
            feed_dict=feed_inputs)

        with self.assertRaises(Exception):
          session.run(outputs['transformed_label'])

    batch_shape = (len(feed_input_values[0]), 1)
    sparse_batch_shape = batch_shape

    if not shape:
      # transformed_b is sparse so _convert_scalars_to_vectors did not fix it
      sparse_batch_shape = sparse_batch_shape[:1]
      transformed_b_dict = dict(zip([tuple(x + [0])
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))
    else:
      transformed_b_dict = dict(zip([tuple(x)
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))

    self.assertEqual(batch_shape, tuple(transformed_a.shape))
    self.assertEqual(sparse_batch_shape, tuple(transformed_b.dense_shape))

    self.assertEqual(21, transformed_a[0][0])
    self.assertEqual(9, transformed_b_dict[(0, 0)])
    self.assertEqual(29, transformed_a[1][0])
    self.assertEqual(-5, transformed_b_dict[(1, 0)])

  def test_build_training_input_fn(self):
    # TODO(b/123241798): use TEST_TMPDIR
    basedir = tempfile.mkdtemp()

    # the transformed schema should be vectorized already.
    metadata = dataset_metadata.DatasetMetadata(
        schema=_make_transformed_schema([1]))
    data_file = os.path.join(basedir, 'data')
    examples = [_create_serialized_example(d)
                for d in [
                    {'transformed_a': 15,
                     'transformed_b': 6,
                     'transformed_label': 77},
                    {'transformed_a': 12,
                     'transformed_b': 17,
                     'transformed_label': 44}]]
    _write_tfrecord(data_file, examples)

    training_input_fn = (
        input_fn_maker.build_training_input_fn(
            metadata=metadata,
            file_pattern=[data_file],
            training_batch_size=128,
            label_keys=['transformed_label'],
            randomize_input=False))

    with tf.Graph().as_default():
      features, labels = training_input_fn()

      with tf.Session().as_default() as session:
        session.run(tf.initialize_all_variables())
        tf.train.start_queue_runners()
        transformed_a, transformed_b, transformed_label = session.run(
            [features['transformed_a'],
             features['transformed_b'],
             labels])

    self.assertEqual((128, 1), tuple(transformed_a.shape))
    self.assertEqual((128, 1), tuple(transformed_b.dense_shape))
    self.assertEqual((128, 1), tuple(transformed_label.shape))
    transformed_b_dict = dict(zip([tuple(x)
                                   for x in transformed_b.indices.tolist()],
                                  transformed_b.values.tolist()))

    self.assertEqual(15, transformed_a[0][0])
    self.assertEqual(6, transformed_b_dict[(0, 0)])
    self.assertEqual(77, transformed_label[0][0])
    self.assertEqual(12, transformed_a[1][0])
    self.assertEqual(17, transformed_b_dict[(1, 0)])
    self.assertEqual(44, transformed_label[1][0])

  def test_build_transforming_training_input_fn_scalars(self):
    self._test_build_transforming_training_input_fn([])

  def test_build_transforming_training_input_fn_vectors(self):
    self._test_build_transforming_training_input_fn([1])

  def _test_build_transforming_training_input_fn(self, shape):
    # TODO(b/123241798): use TEST_TMPDIR
    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_raw_schema(shape, should_add_unused_feature=True))

    # the transformed schema should be vectorized already.
    transformed_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_transformed_schema([1]))
    data_file = os.path.join(basedir, 'data')
    examples = [_create_serialized_example(d)
                for d in [
                    {'raw_a': 15,
                     'raw_b': 6,
                     'raw_label': 77},
                    {'raw_a': 12,
                     'raw_b': 17,
                     'raw_label': 44}]]
    _write_tfrecord(data_file, examples)

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(
        transform_savedmodel_dir, should_add_unused_feature=True)

    training_input_fn = (
        input_fn_maker.build_transforming_training_input_fn(
            raw_metadata=raw_metadata,
            transformed_metadata=transformed_metadata,
            transform_savedmodel_dir=transform_savedmodel_dir,
            raw_data_file_pattern=[data_file],
            training_batch_size=128,
            transformed_label_keys=['transformed_label'],
            randomize_input=False,
            convert_scalars_to_vectors=True))

    with tf.Graph().as_default():
      features, labels = training_input_fn()

      with tf.Session().as_default() as session:
        session.run(tf.initialize_all_variables())
        tf.train.start_queue_runners()
        transformed_a, transformed_b, transformed_label = session.run(
            [features['transformed_a'],
             features['transformed_b'],
             labels])

    batch_shape = (128, 1)
    sparse_batch_shape = batch_shape

    if not shape:
      # transformed_b is sparse so _convert_scalars_to_vectors did not fix it
      sparse_batch_shape = sparse_batch_shape[:1]
      transformed_b_dict = dict(zip([tuple(x + [0])
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))
    else:
      transformed_b_dict = dict(zip([tuple(x)
                                     for x in transformed_b.indices.tolist()],
                                    transformed_b.values.tolist()))

    self.assertEqual(batch_shape, tuple(transformed_a.shape))
    self.assertEqual(sparse_batch_shape, tuple(transformed_b.dense_shape))
    self.assertEqual(batch_shape, tuple(transformed_label.shape))

    self.assertEqual(21, transformed_a[0][0])
    self.assertEqual(9, transformed_b_dict[(0, 0)])
    self.assertEqual(77000, transformed_label[0][0])
    self.assertEqual(29, transformed_a[1][0])
    self.assertEqual(-5, transformed_b_dict[(1, 0)])
    self.assertEqual(44000, transformed_label[1][0])


def _write_tfrecord(data_file, examples, count=1):
  with tf.python_io.TFRecordWriter(data_file) as writer:
    for _ in range(count):
      for e in examples:
        writer.write(e)


def _create_serialized_example(data_dict):
  example1 = tf.train.Example()
  for k, v in six.iteritems(data_dict):
    example1.features.feature[k].int64_list.value.append(v)
  return example1.SerializeToString()


def _write_transform_savedmodel(transform_savedmodel_dir,
                                should_add_unused_feature=False):
  """Writes a TransformFn to the given directory.

  Args:
    transform_savedmodel_dir: A directory to save to.
    should_add_unused_feature: Whether or not an unused feature should be added
      to the inputs. This has to be in sync with the value of
      should_add_unused_feature used to invoke _make_raw_schema.
  """
  with tf.Graph().as_default():
    with tf.Session().as_default() as session:
      raw_a = tf.placeholder(tf.int64)
      raw_b = tf.placeholder(tf.int64)
      raw_label = tf.placeholder(tf.int64)
      transformed_a = raw_a + raw_b
      transformed_b_dense = raw_a - raw_b

      idx = tf.where(tf.not_equal(transformed_b_dense, 0))
      transformed_b_sparse = tf.SparseTensor(
          idx,
          tf.gather_nd(transformed_b_dense, idx),
          tf.shape(transformed_b_dense, out_type=tf.int64))

      # Ensure sparse shape is [batch_size, 1], not [batch_size,]
      # transformed_b_sparse_wide = tf.sparse_reshape(
      #     transformed_b_sparse,
      #     tf.concat([transformed_b_sparse.dense_shape, [1]], 0))

      transformed_label = raw_label * 1000
      inputs = {'raw_a': raw_a, 'raw_b': raw_b, 'raw_label': raw_label}

      if should_add_unused_feature:
        inputs['raw_unused'] = tf.placeholder(tf.int64)

      outputs = {'transformed_a': transformed_a,
                 'transformed_b': transformed_b_sparse,
                 # 'transformed_b_wide': transformed_b_sparse_wide,
                 'transformed_label': transformed_label}
      saved_transform_io.write_saved_transform_from_session(
          session, inputs, outputs, transform_savedmodel_dir)

if __name__ == '__main__':
  test_case.main()
