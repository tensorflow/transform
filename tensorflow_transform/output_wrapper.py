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

# GOOGLE-INITIALIZATION

import six
import tensorflow as tf
from tensorflow_transform import graph_tools
from tensorflow_transform.analyzers import sanitized_vocab_filename
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.saved import saved_transform_io_v2
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import tf2
from tensorflow.python.framework import ops
# pylint: enable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


class TFTransformOutput(object):
  """A wrapper around the output of the tf.Transform."""

  # Locations relative to the base output directory, where outputs of
  # tf.Transform should be written in order to be read by TFTransformOutput.
  # WriteTransformFn will follow these conventions.
  TRANSFORMED_METADATA_DIR = 'transformed_metadata'
  TRANSFORM_FN_DIR = 'transform_fn'

  def __init__(self, transform_output_dir):
    """Init method for TFTransformOutput.

    Args:
      transform_output_dir: The directory containig tf.Transform output.
    """
    self._transform_output_dir = transform_output_dir

    # Lazily constructed properties.
    self._transformed_metadata = None
    self._raw_metadata = None
    self._transform_features_layer = None

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
    return schema_utils.schema_as_feature_spec(
        self.transformed_metadata.schema).feature_spec

  def transformed_domains(self):
    """Returns domains for the transformed features.

    Returns:
      A dict from feature names to schema_pb2.Domain.
    """
    return schema_utils.schema_as_feature_spec(
        self.transformed_metadata.schema).domains

  def vocabulary_file_by_name(self, vocab_filename):
    """Returns the vocabulary file path created in the preprocessing function.

    `vocab_filename` must be the name used as the vocab_filename argument to
    tft.compute_and_apply_vocabulary or tft.vocabulary. By convention, this
    should be the name of the feature that the vocab was computed for, where
    possible.

    Args:
      vocab_filename: The relative filename to lookup.
    """
    return os.path.join(self.transform_savedmodel_dir,
                        tf.saved_model.ASSETS_DIRECTORY,
                        sanitized_vocab_filename(filename=vocab_filename))

  def vocabulary_size_by_name(self, vocab_filename):
    """Like vocabulary_file_by_name, but returns the size of vocabulary."""
    with tf.io.gfile.GFile(self.vocabulary_file_by_name(vocab_filename),
                           'rb') as f:
      return sum(1 for _ in f)

  def vocabulary_by_name(self, vocab_filename):
    """Like vocabulary_file_by_name but returns a list."""
    with tf.io.gfile.GFile(self.vocabulary_file_by_name(vocab_filename),
                           'rb') as f:
      return [l.rstrip() for l in f]

  # TODO(KesterTong): Add test for this in output_wrapper_test.py
  def num_buckets_for_transformed_feature(self, name):
    """Returns the number of buckets for an integerized transformed feature."""
    # Do checks that this tensor can be wrapped in
    # sparse_column_with_integerized_feature
    domains = schema_utils.schema_as_feature_spec(
        self.transformed_metadata.schema).domains
    try:
      domain = domains[name]
    except KeyError:
      raise ValueError('Column {} did not have a domain provided.'.format(name))
    if not isinstance(domain, schema_pb2.IntDomain):
      raise ValueError('Column {} has domain {}, expected an IntDomain'.format(
          name, domain))
    if domain.min != 0:
      raise ValueError('Column {} has min value {}, should be 0'.format(
          name, domain.min))
    return domain.max + 1

  def transform_features_layer(self):
    """Creates a `TransformFeaturesLayer` from this transform output.

    If a `TransformFeaturesLayer` has already been created for self, the same
    one will be returned.

    Returns:
      A `TransformFeaturesLayer` instance.
    """
    if self._transform_features_layer is None:
      self._transform_features_layer = TransformFeaturesLayer(self)
    return self._transform_features_layer

  def transform_raw_features(self, raw_features, drop_unused_features=False):
    """Takes a dict of tensors representing raw features and transforms them.

    Takes a dictionary of `Tensor`s or `SparseTensor`s that represent the raw
    features, and applies the transformation defined by tf.Transform.

    By default it returns all transformed features defined by tf.Transform. To
    only return features transformed from the given 'raw_features', set
    `drop_unused_features` to True.

    Args:
      raw_features: A dict whose keys are feature names and values are `Tensor`s
        or `SparseTensor`s.
      drop_unused_features: If True, the result will be filtered. Only the
        features that are transformed from 'raw_features' will be included in
        the returned result. If a feature is transformed from multiple raw
        features (e.g, feature cross), it will only be included if all its base
        raw features are present in `raw_features`.

    Returns:
      A dict whose keys are feature names and values are `Tensor`s or
          `SparseTensor`s representing transformed features.
    """
    unbounded_raw_features, transformed_features = (
        saved_transform_io.partially_apply_saved_transform_internal(
            self.transform_savedmodel_dir, raw_features))
    if drop_unused_features:
      graph = tf.compat.v1.get_default_graph()
      graph_analyzer = graph_tools.InitializableGraphAnalyzer(
          graph, raw_features,
          [(t, False) for t in six.itervalues(unbounded_raw_features)])
      return {
          name: feature
          for name, feature in six.iteritems(transformed_features)
          if graph_analyzer.ready_to_run(feature)
      }
    else:
      return transformed_features

  def load_transform_graph(self):
    """Load the transform graph without replacing any placeholders.

    This is necessary to ensure that variables in the transform graph are
    included in the training checkpoint when using tf.Estimator.  This should
    be called in the training input_fn.
    """
    saved_transform_io.partially_apply_saved_transform_internal(
        self.transform_savedmodel_dir, {})

  RAW_METADATA_DIR = 'metadata'
  _FEATURE_STATS_PB = 'FeatureStats.pb'
  PRE_TRANSFORM_FEATURE_STATS_PATH = os.path.join(
      'pre_transform_feature_stats', _FEATURE_STATS_PB)
  POST_TRANSFORM_FEATURE_STATS_PATH = os.path.join(
      'post_transform_feature_stats', _FEATURE_STATS_PB)

  @property
  def raw_metadata(self):
    """A DatasetMetadata.

    Note: raw_metadata is not guaranteed to exist in the output of tf.transform
    and hence using this could fail, if raw_metadata is not present in
    TFTransformOutput.

    Returns:
      A DatasetMetadata
    """
    if self._raw_metadata is None:
      self._raw_metadata = metadata_io.read_metadata(
          os.path.join(self._transform_output_dir, self.RAW_METADATA_DIR))
    return self._raw_metadata

  def raw_feature_spec(self):
    """Returns a feature_spec for the raw features.

    Returns:
      A dict from feature names to FixedLenFeature/SparseFeature/VarLenFeature.
    """
    return schema_utils.schema_as_feature_spec(
        self.raw_metadata.schema).feature_spec

  def raw_domains(self):
    """Returns domains for the raw features.

    Returns:
      A dict from feature names to schema_pb2.Domain.
    """
    return schema_utils.schema_as_feature_spec(
        self.raw_metadata.schema).domains

  @property
  def pre_transform_statistics_path(self):
    """Returns the path to the pre-transform datum statistics.

    Note: pre_transform_statistics is not guaranteed to exist in the output of
    tf.transform and hence using this could fail, if pre_transform statistics is
    not present in TFTransformOutput.
    """
    return os.path.join(
        self._transform_output_dir, self.PRE_TRANSFORM_FEATURE_STATS_PATH)

  @property
  def post_transform_statistics_path(self):
    """Returns the path to the post-transform datum statistics.

    Note: post_transform_statistics is not guaranteed to exist in the output of
    tf.transform and hence using this could fail, if post_transform statistics
    is not present in TFTransformOutput.
    """
    return os.path.join(
        self._transform_output_dir, self.POST_TRANSFORM_FEATURE_STATS_PATH)


# TODO(zoyahav): Use register_keras_serializable directly once we no longer support
# TF<2.1.
def _maybe_register_keras_serializable(package):
  if hasattr(tf.keras.utils, 'register_keras_serializable'):
    return tf.keras.utils.register_keras_serializable(package=package)
  else:
    return lambda cls: cls


@_maybe_register_keras_serializable(package='TensorFlowTransform')
class TransformFeaturesLayer(tf.keras.layers.Layer):
  """A Keras layer for applying a tf.Transform output to input layers."""

  def __init__(self, tft_output):
    super(TransformFeaturesLayer, self).__init__(trainable=False)
    self._tft_output = tft_output
    # TODO(b/160294509): Use tf.compat.v1 when we stop supporting TF 1.15.
    if ops.executing_eagerly_outside_functions():
      self._check_tensorflow_version()
      self._saved_model_loader = saved_transform_io_v2.SavedModelLoader(
          tft_output.transform_savedmodel_dir)
      # The model must be tracked by assigning to an attribute of the Keras
      # layer. Hence, we track the attributes of _saved_model_loader here as
      # well.
      self._saved_model_loader_tracked_dict = self._saved_model_loader.__dict__

  def _check_tensorflow_version(self):
    """Check that we're using a compatible TF version.

    Raises a warning if either Tensorflow version is less that 2.0 or TF 2.x is
    not enabled.

    If TF 2.x is enabled, but version is < TF 2.3, raises a warning to indicate
    that resources may not be initialized.
    """
    major, minor, _ = tf.version.VERSION.split('.')
    if not (int(major) >= 2 and tf2.enabled()):
      tf.compat.v1.logging.warning(
          'Tensorflow version (%s) found. TransformFeaturesLayer is supported '
          'only for TF 2.x with TF 2.x behaviors enabled and may not work as '
          'intended.', tf.version.VERSION)
    elif int(major) == 2 and int(minor) < 3:
      # TODO(varshaan): Log a more specific warning.
      tf.compat.v1.logging.warning(
          'Tensorflow version (%s) found. TransformFeaturesLayer may not work '
          'as intended if the SavedModel contains an initialization op.',
          tf.version.VERSION)

  def call(self, inputs):
    # TODO(b/160294509): Use tf.compat.v1 when we stop supporting TF 1.15.
    if ops.executing_eagerly_outside_functions():
      return self._saved_model_loader.apply_transform_model(inputs)
    else:
      tf.compat.v1.logging.warning('Falling back to transform_raw_features...')
      return self._tft_output.transform_raw_features(
          inputs, drop_unused_features=True)
