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

import os
import tempfile


import six
import tensorflow as tf

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema as sch
import unittest


class InputFnMakerTest(unittest.TestCase):

  def test_build_parsing_transforming_serving_input_fn(self):
    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_raw_schema())

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(transform_savedmodel_dir)

    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            transform_savedmodel_dir=transform_savedmodel_dir,
            raw_label_keys=['raw_label'],
            raw_feature_keys=['raw_a', 'raw_b']))

    examples = [_create_serialized_example(d)
                for d in [
                    {'raw_a': 15, 'raw_b': 5},
                    {'raw_a': 12, 'raw_b': 17}]]

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, _, inputs = serving_input_fn()
        feed_inputs = {inputs['examples']: examples}
        transformed_a, transformed_b = session.run(
            [outputs['transformed_a'], outputs['transformed_b']],
            feed_dict=feed_inputs)

    self.assertEqual(20, transformed_a[0][0])
    self.assertEqual(10, transformed_b[0][0])
    self.assertEqual(29, transformed_a[1][0])
    self.assertEqual(-5, transformed_b[1][0])

  def test_build_default_transforming_serving_input_fn(self):
    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_raw_schema())

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(transform_savedmodel_dir)

    serving_input_fn = (
        input_fn_maker.build_default_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            raw_label_keys=['raw_label'],
            raw_feature_keys=['raw_a', 'raw_b'],
            transform_savedmodel_dir=transform_savedmodel_dir))

    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, _, inputs = serving_input_fn()
        feed_inputs = {inputs['raw_a']: [[15], [12]],
                       inputs['raw_b']: [[5], [17]]}
        transformed_a, transformed_b = session.run(
            [outputs['transformed_a'], outputs['transformed_b']],
            feed_dict=feed_inputs)

    self.assertEqual(20, transformed_a[0][0])
    self.assertEqual(10, transformed_b[0][0])
    self.assertEqual(29, transformed_a[1][0])
    self.assertEqual(-5, transformed_b[1][0])

  def test_build_training_input_fn(self):
    basedir = tempfile.mkdtemp()

    metadata = dataset_metadata.DatasetMetadata(
        schema=_make_transformed_schema())
    data_file = os.path.join(basedir, 'data')
    examples = [_create_serialized_example(d)
                for d in [
                    {'transformed_a': 15,
                     'transformed_b': 5,
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

    self.assertEqual(15, transformed_a[0][0])
    self.assertEqual(5, transformed_b[0][0])
    self.assertEqual(77, transformed_label[0][0])
    self.assertEqual(12, transformed_a[1][0])
    self.assertEqual(17, transformed_b[1][0])
    self.assertEqual(44, transformed_label[1][0])

  def test_build_transforming_training_input_fn(self):
    basedir = tempfile.mkdtemp()

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_raw_schema())
    transformed_metadata = dataset_metadata.DatasetMetadata(
        schema=_make_transformed_schema())
    data_file = os.path.join(basedir, 'data')
    examples = [_create_serialized_example(d)
                for d in [
                    {'raw_a': 15,
                     'raw_b': 5,
                     'raw_label': 77},
                    {'raw_a': 12,
                     'raw_b': 17,
                     'raw_label': 44}]]
    _write_tfrecord(data_file, examples)

    transform_savedmodel_dir = os.path.join(basedir, 'transform-savedmodel')
    _write_transform_savedmodel(transform_savedmodel_dir)

    training_input_fn = (
        input_fn_maker.build_transforming_training_input_fn(
            raw_metadata=raw_metadata,
            transformed_metadata=transformed_metadata,
            transform_savedmodel_dir=transform_savedmodel_dir,
            raw_data_file_pattern=[data_file],
            training_batch_size=128,
            raw_label_keys=['raw_label'],
            transformed_label_keys=['transformed_label'],
            raw_feature_keys=['raw_a', 'raw_b'],
            transformed_feature_keys=['transformed_a', 'transformed_b'],
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

    self.assertEqual(20, transformed_a[0][0])
    self.assertEqual(10, transformed_b[0][0])
    self.assertEqual(77000, transformed_label[0][0])
    self.assertEqual(29, transformed_a[1][0])
    self.assertEqual(-5, transformed_b[1][0])
    self.assertEqual(44000, transformed_label[1][0])


def _write_tfrecord(data_file, examples, count=1):
  with tf.python_io.TFRecordWriter(data_file) as writer:
    for _ in xrange(count):
      for e in examples:
        writer.write(e)


def _create_serialized_example(data_dict):
  example1 = tf.train.Example()
  for k, v in six.iteritems(data_dict):
    example1.features.feature[k].int64_list.value.append(v)
  return example1.SerializeToString()


def _make_raw_schema():
  schema = sch.Schema()

  schema.column_schemas['raw_a'] = (
      sch.ColumnSchema(tf.int64, [1], sch.FixedColumnRepresentation()))

  schema.column_schemas['raw_b'] = (
      sch.ColumnSchema(tf.int64, [1], sch.FixedColumnRepresentation()))

  schema.column_schemas['raw_label'] = (
      sch.ColumnSchema(tf.int64, [1], sch.FixedColumnRepresentation()))

  return schema


def _make_transformed_schema():
  schema = sch.Schema()

  schema.column_schemas['transformed_a'] = (
      sch.ColumnSchema(tf.int64, [1], sch.FixedColumnRepresentation()))

  schema.column_schemas['transformed_b'] = (
      sch.ColumnSchema(tf.int64, [1], sch.FixedColumnRepresentation()))

  schema.column_schemas['transformed_label'] = (
      sch.ColumnSchema(tf.int64, [1], sch.FixedColumnRepresentation()))

  return schema


def _write_transform_savedmodel(transform_savedmodel_dir):
  with tf.Graph().as_default():
    with tf.Session().as_default() as session:
      raw_a = tf.placeholder(tf.int64)
      raw_b = tf.placeholder(tf.int64)
      raw_label = tf.placeholder(tf.int64)
      transformed_a = raw_a + raw_b
      transformed_b = raw_a - raw_b
      transformed_label = raw_label * 1000
      inputs = {'raw_a': raw_a, 'raw_b': raw_b, 'raw_label': raw_label}
      outputs = {'transformed_a': transformed_a,
                 'transformed_b': transformed_b,
                 'transformed_label': transformed_label}
      saved_transform_io.write_saved_transform_from_session(
          session, inputs, outputs, transform_savedmodel_dir)

if __name__ == '__main__':
  unittest.main()

