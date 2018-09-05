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
    'fixed_categorical_int_with_range': tf.FixedLenFeature(
        shape=[], dtype=tf.int64),
    'fixed_int': tf.FixedLenFeature(shape=[5], dtype=tf.int64),
    'fixed_float': tf.FixedLenFeature(shape=[5], dtype=tf.float32),
    'fixed_string': tf.FixedLenFeature(shape=[5], dtype=tf.string),

    # VarLenFeatures
    'var_int': tf.VarLenFeature(dtype=tf.int64),
    'var_float': tf.VarLenFeature(dtype=tf.float32),
    'var_string': tf.VarLenFeature(dtype=tf.string),
}


def get_test_schema():
  return sch.from_feature_spec(test_feature_spec)


def get_manually_created_schema():
  """Provide a test schema built from scratch using the Schema classes."""
  return sch.Schema({
      # FixedLenFeatures
      'fixed_categorical_int_with_range': sch.ColumnSchema(
          sch.IntDomain(tf.int64, -5, 10, True),
          [], sch.FixedColumnRepresentation()),
      'fixed_int': sch.ColumnSchema(
          tf.int64, [5], sch.FixedColumnRepresentation()),
      'fixed_float': sch.ColumnSchema(
          tf.float32, [5], sch.FixedColumnRepresentation()),
      'fixed_string': sch.ColumnSchema(
          tf.string, [5], sch.FixedColumnRepresentation()),
      # VarLenFeatures
      'var_int': sch.ColumnSchema(
          tf.int64, None, sch.ListColumnRepresentation()),
      'var_float': sch.ColumnSchema(
          tf.float32, None, sch.ListColumnRepresentation()),
      'var_string': sch.ColumnSchema(
          tf.string, None, sch.ListColumnRepresentation())
  })
