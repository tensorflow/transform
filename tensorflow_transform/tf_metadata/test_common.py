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
"""Common data and utilities for tf_metadata tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_transform.tf_metadata import dataset_schema as sch


test_feature_spec = {
    # FixedLenFeatures
    'fixed_bool_with_default': tf.FixedLenFeature(
        shape=[1], dtype=tf.bool, default_value=False),
    'fixed_bool_without_default': tf.FixedLenFeature(
        shape=[5], dtype=tf.bool),
    'fixed_int_with_default': tf.FixedLenFeature(
        shape=[1], dtype=tf.int64, default_value=0),
    'fixed_categorical_int_with_range': tf.FixedLenFeature(
        shape=[1], dtype=tf.int64, default_value=0),
    'fixed_categorical_int_with_vocab': tf.FixedLenFeature(
        shape=[1], dtype=tf.int64, default_value=0),
    'fixed_int_without_default': tf.FixedLenFeature(
        shape=[5], dtype=tf.int64),
    'fixed_float_with_default': tf.FixedLenFeature(
        shape=[1], dtype=tf.float32, default_value=0.0),
    'fixed_float_without_default': tf.FixedLenFeature(
        shape=[5], dtype=tf.float32),
    'fixed_string_with_default': tf.FixedLenFeature(
        shape=[1], dtype=tf.string, default_value='default'),
    'fixed_string_without_default': tf.FixedLenFeature(
        shape=[5], dtype=tf.string),
    '3d_fixed_int_without_default': tf.FixedLenFeature(
        shape=[5, 6, 7], dtype=tf.int64),

    # VarLenFeatures
    'var_bool': tf.VarLenFeature(dtype=tf.bool),
    'var_int': tf.VarLenFeature(dtype=tf.int64),
    'var_float': tf.VarLenFeature(dtype=tf.float32),
    'var_string': tf.VarLenFeature(dtype=tf.string),

    # SparseFeatures
    'sparse_bool': tf.SparseFeature(
        index_key='sparse_bool_index', value_key='sparse_bool_value',
        dtype=tf.bool, size=15, already_sorted=True),
    'sparse_int': tf.SparseFeature(
        index_key='sparse_int_index', value_key='sparse_int_value',
        dtype=tf.int64, size=150, already_sorted=False),
    'sparse_float': tf.SparseFeature(
        index_key='sparse_float_index', value_key='sparse_float_value',
        dtype=tf.float32, size=1500),
    'sparse_string': tf.SparseFeature(
        index_key='sparse_string_index', value_key='sparse_string_value',
        dtype=tf.string, size=15000, already_sorted=True),
}


def get_test_schema():
  return sch.from_feature_spec(test_feature_spec)


def get_manually_created_schema():
  """Provide a test schema built from scratch using the Schema classes."""
  schema = sch.Schema()

  # FixedLenFeatures
  schema.column_schemas['fixed_bool_with_default'] = (
      sch.ColumnSchema(tf.bool, [1], sch.FixedColumnRepresentation(
          default_value=False)))

  schema.column_schemas['fixed_bool_without_default'] = (
      sch.ColumnSchema(tf.bool, [5], sch.FixedColumnRepresentation()))

  schema.column_schemas['fixed_int_with_default'] = (
      sch.ColumnSchema(tf.int64, [1], sch.FixedColumnRepresentation(
          default_value=0)))

  schema.column_schemas['fixed_categorical_int_with_range'] = (
      sch.ColumnSchema(sch.IntDomain(tf.int64, -5, 10, True), [1],
                       sch.FixedColumnRepresentation(0)))

  schema.column_schemas['fixed_categorical_int_with_vocab'] = (
      sch.ColumnSchema(sch.IntDomain(tf.int64, vocabulary_file='test_filename'),
                       [1],
                       sch.FixedColumnRepresentation(0)))

  schema.column_schemas['fixed_int_without_default'] = (
      sch.ColumnSchema(tf.int64, [5], sch.FixedColumnRepresentation()))

  schema.column_schemas['fixed_float_with_default'] = (
      sch.ColumnSchema(tf.float32, [1], sch.FixedColumnRepresentation(
          default_value=0.0)))

  schema.column_schemas['fixed_float_without_default'] = (
      sch.ColumnSchema(tf.float32, [5], sch.FixedColumnRepresentation()))

  schema.column_schemas['fixed_string_with_default'] = (
      sch.ColumnSchema(tf.string, [1],
                       sch.FixedColumnRepresentation(default_value='default')))

  schema.column_schemas['fixed_string_without_default'] = (
      sch.ColumnSchema(tf.string, [5], sch.FixedColumnRepresentation()))

  schema.column_schemas['3d_fixed_int_without_default'] = (
      sch.ColumnSchema(tf.int64, [5, 6, 7], sch.FixedColumnRepresentation()))

  # VarLenFeatures
  schema.column_schemas['var_bool'] = (
      sch.ColumnSchema(tf.bool, None, sch.ListColumnRepresentation()))

  schema.column_schemas['var_int'] = (
      sch.ColumnSchema(tf.int64, None, sch.ListColumnRepresentation()))

  schema.column_schemas['var_float'] = (
      sch.ColumnSchema(tf.float32, None, sch.ListColumnRepresentation()))

  schema.column_schemas['var_string'] = (
      sch.ColumnSchema(tf.string, None, sch.ListColumnRepresentation()))

  # SparseFeatures
  schema.column_schemas['sparse_bool'] = (
      sch.ColumnSchema(
          tf.bool, [15],
          sch.SparseColumnRepresentation('sparse_bool_value',
                                         [sch.SparseIndexField(
                                             'sparse_bool_index', True)])))

  schema.column_schemas['sparse_int'] = (
      sch.ColumnSchema(
          tf.int64, [150],
          sch.SparseColumnRepresentation('sparse_int_value',
                                         [sch.SparseIndexField(
                                             'sparse_int_index', False)])))

  schema.column_schemas['sparse_float'] = (
      sch.ColumnSchema(
          tf.float32, [1500],
          sch.SparseColumnRepresentation('sparse_float_value',
                                         [sch.SparseIndexField(
                                             'sparse_float_index',
                                             False)])))

  schema.column_schemas['sparse_string'] = (
      sch.ColumnSchema(
          tf.string, [15000],
          sch.SparseColumnRepresentation('sparse_string_value',
                                         [sch.SparseIndexField(
                                             'sparse_string_index',
                                             True)])))

  return schema
