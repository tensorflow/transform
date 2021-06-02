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

import tensorflow as tf

from tensorflow_transform.tf_metadata import schema_utils


test_feature_spec = {
    # FixedLenFeatures
    'fixed_categorical_int_with_range':
        tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    'fixed_int':
        tf.io.FixedLenFeature(shape=[5], dtype=tf.int64),
    'fixed_float':
        tf.io.FixedLenFeature(shape=[5], dtype=tf.float32),
    'fixed_string':
        tf.io.FixedLenFeature(shape=[5], dtype=tf.string),

    # VarLenFeatures
    'var_int':
        tf.io.VarLenFeature(dtype=tf.int64),
    'var_float':
        tf.io.VarLenFeature(dtype=tf.float32),
    'var_string':
        tf.io.VarLenFeature(dtype=tf.string),
}


def get_test_schema():
  return schema_utils.schema_from_feature_spec(test_feature_spec)
