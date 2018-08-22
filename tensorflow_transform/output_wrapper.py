# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Utilities for consuming tf.Transform output during training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import tensorflow as tf
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import metadata_io


class TFTransformOutput(object):
  """A wrapper around the output of the tf.Transform.

  Args:
    transform_output_dir: The directory containig tf.Transform output.
  """

  # Locations relative to the base output directory, where outputs of
  # tf.Transform should be written in order to be read by TFTransformOutput.
  # WriteTransformFn will follow these conventions.
  TRANSFORMED_METADATA_DIR = 'transformed_metadata'
  TRANSFORM_FN_DIR = 'transform_fn'

  def __init__(self, transform_output_dir):
    self._transform_output_dir = transform_output_dir

    # Lazily constructed properties.
    self._transformed_metadata = None

  @property
  def transformed_metadata(self):
    """A DatasetMetadata."""
    if self._transformed_metadata is None:
      self._transformed_metadata = metadata_io.read_metadata(
          os.path.join(self._transform_output_dir,
                       self.TRANSFORMED_METADATA_DIR))
    return self._transformed_metadata

  @property
  def transform_savedmodel_dir(self):
    """A python str."""
    return os.path.join(self._transform_output_dir, self.TRANSFORM_FN_DIR)

  def transformed_feature_spec(self):
    """Returns a feature_spec for the transformed features.

    Returns:
      A dict from feature names to FixedLenFeature/SparseFeature/VarLenFeature.
    """
    return self.transformed_metadata.schema.as_feature_spec()

  def vocabulary_file_by_name(self, vocab_filename):
    """Returns the vocabulary file path created in the preprocessing function.

    `vocab_filename` must be the name used as the vocab_filename argument to
    tft.compute_and_apply_vocabulary or tft.vocabulary. By convention, this
    should be the name of the feature that the vocab was computed for, where
    possible.

    Args:
      vocab_filename: The relative filename to lookup.
    """
    return os.path.join(
        self.transform_savedmodel_dir, 'assets', vocab_filename)

  def vocabulary_size_by_name(self, vocab_filename):
    """Like vocabulary_file_by_name, but returns the size of vocabulary."""
    with tf.gfile.GFile(self.vocabulary_file_by_name(vocab_filename)) as f:
      return sum(1 for _ in f)

  def vocabulary_by_name(self, vocab_filename):
    """Like vocabulary_file_by_name but returns a list."""
    with tf.gfile.GFile(self.vocabulary_file_by_name(vocab_filename)) as f:
      return [l.rstrip() for l in f]

  def num_buckets_for_transformed_feature(self, name):
    """Returns the number of buckets for an integerized transformed feature."""
    # Do checks that this tensor can be wrapped in
    # sparse_column_with_integerized_feature
    column_schema = self.transformed_metadata.schema.column_schemas[name]
    if column_schema.domain.dtype != tf.int64:
      raise ValueError('Column {} has dtype {}, should be int64'.format(
          name, column_schema.domain.dtype))
    if column_schema.domain.min_value != 0:
      raise ValueError('Column {} has min value {}, should be 0'.format(
          name, column_schema.domain.min_value))
    return column_schema.domain.max_value + 1

  def transform_raw_features(self, raw_features):
    """Takes a dict of tensors representing raw features and transforms them.

    Takes a dictionary of `Tensor`s or `SparseTensor`s that represent the raw
    features, and applies the transformation defined by tf.Transform.

    Args:
      raw_features: A dict whose keys are feature names and values are `Tensor`s
          or `SparseTensor`s.

    Returns:
      A dict whose keys are feature names and values are `Tensor`s or
          `SparseTensor`s representing transformed features.
    """
    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform_internal(
            self.transform_savedmodel_dir, raw_features))
    return transformed_features

  def load_transform_graph(self):
    """Load the transform graph without replacing any placeholders.

    This is necessary to ensure that variables in the transform graph are
    included in the training checkpoint when using tf.Estimator.  This should
    be called in the training input_fn.
    """
    saved_transform_io.partially_apply_saved_transform_internal(
        self.transform_savedmodel_dir, {})

