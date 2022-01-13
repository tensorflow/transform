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

import json
import os
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import tensorflow as tf
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import graph_tools
from tensorflow_transform.analyzers import sanitized_vocab_filename
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.saved import saved_transform_io_v2
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.tools.docs import doc_controls
# pylint: enable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


def _get_tensor_value(tensor_or_eager_tensor: tf.Tensor) -> Any:
  if ops.executing_eagerly_outside_functions():
    return np.asarray(tensor_or_eager_tensor)
  else:
    with tf.compat.v1.Session():
      return tensor_or_eager_tensor.eval()


class _TransformedFeaturesDict(dict):
  """A wrapper around dict.

  Overrides pop to return None instead of throwing a KeyError when invoked with
  a key that is not found in the dictionary.

  NOTE: Do not use directly.
  """

  def pop(self, key, default=None):  # pylint: disable=useless-super-delegation
    return super().pop(key, default)


class TFTransformOutput:
  """A wrapper around the output of the tf.Transform."""

  # Locations relative to the base output directory, where outputs of
  # tf.Transform should be written in order to be read by TFTransformOutput.
  # WriteTransformFn will follow these conventions.
  TRANSFORMED_METADATA_DIR = 'transformed_metadata'
  TRANSFORM_FN_DIR = 'transform_fn'
  ASSET_MAP = 'asset_map'

  def __init__(self, transform_output_dir: str):
    """Init method for TFTransformOutput.

    Args:
      transform_output_dir: The directory containig tf.Transform output.
    """
    self._transform_output_dir = transform_output_dir

    # Lazily constructed properties.
    self._transformed_metadata = None
    self._raw_metadata = None
    self._transform_features_layer = None
    self._exported_as_v1_value = None
    self._transformed_domains = None

  @property
  def transformed_metadata(self) -> dataset_metadata.DatasetMetadata:
    """A DatasetMetadata."""
    if self._transformed_metadata is None:
      self._transformed_metadata = metadata_io.read_metadata(
          self._transformed_metadata_dir)
    return self._transformed_metadata

  @property
  def transform_savedmodel_dir(self) -> str:
    """A python str."""
    return os.path.join(self._transform_output_dir, self.TRANSFORM_FN_DIR)

  @property
  def _exported_as_v1(self) -> bool:
    """A boolean.

    Indicates whether the SavedModel was exported using TF 1.x or TF 2.x APIs.
    """
    if self._exported_as_v1_value is None:
      self._exported_as_v1_value = saved_transform_io.exported_as_v1(
          self.transform_savedmodel_dir)
    return self._exported_as_v1_value

  @property
  def _transformed_metadata_dir(self) -> str:
    return os.path.join(self._transform_output_dir,
                        self.TRANSFORMED_METADATA_DIR)

  def transformed_feature_spec(self) -> Dict[str, common_types.FeatureSpecType]:
    """Returns a feature_spec for the transformed features.

    Returns:
      A dict from feature names to FixedLenFeature/SparseFeature/VarLenFeature.
    """
    return schema_utils.schema_as_feature_spec(
        self.transformed_metadata.schema).feature_spec

  def transformed_domains(self) -> Dict[str, common_types.DomainType]:
    """Returns domains for the transformed features.

    Returns:
      A dict from feature names to one of schema_pb2.IntDomain,
      schema_pb2.StringDomain or schema_pb2.FloatDomain.
    """
    if self._transformed_domains is None:
      self._transformed_domains = schema_utils.schema_as_feature_spec(
          self.transformed_metadata.schema).domains
    return self._transformed_domains

  def vocabulary_file_by_name(self, vocab_filename: str) -> Optional[str]:
    """Returns the vocabulary file path created in the preprocessing function.

    `vocab_filename` must either be (i) the name used as the vocab_filename
    argument to tft.compute_and_apply_vocabulary / tft.vocabulary or (ii) the
    key used in tft.annotate_asset.

    When a mapping has been specified by calls to tft.annotate_asset, it will be
    checked first for the provided filename. If present, this filename will be
    used directly to construct a path.

    If the mapping does not exist or `vocab_filename` is not present within it,
    we will default to sanitizing `vocab_filename` and searching for files
    matching it within the assets directory.

    In either case, if the constructed path does not point to an existing file
    within the assets subdirectory, we will return a None.

    Args:
      vocab_filename: The vocabulary name to lookup.
    """
    mapping_path = os.path.join(self._transformed_metadata_dir, self.ASSET_MAP)

    mapping = {}
    if tf.io.gfile.exists(mapping_path):
      with tf.io.gfile.GFile(mapping_path) as f:
        mapping = json.loads(f.read())
        if vocab_filename in mapping:
          vocab_path = os.path.join(self.transform_savedmodel_dir,
                                    tf.saved_model.ASSETS_DIRECTORY,
                                    mapping[vocab_filename])
          if tf.io.gfile.exists(vocab_path):
            return vocab_path

    prefix = os.path.join(self.transform_savedmodel_dir,
                          tf.saved_model.ASSETS_DIRECTORY,
                          sanitized_vocab_filename(filename=vocab_filename))
    files = tf.io.gfile.glob(prefix) + tf.io.gfile.glob(
        '{}.tfrecord.gz'.format(prefix))
    if not files:
      return None
    if len(files) != 1:
      raise ValueError('Found too many vocabulary files: {}'.format(files))
    return files[0]

  def _vocabulary_size_from_annotations(self,
                                        vocab_filename: str) -> Optional[int]:
    """If vocabulary size is present in annotations return it, else None."""
    if not common.IS_ANNOTATIONS_PB_AVAILABLE:
      return None

    try:
      schema = self.transformed_metadata.schema
    except IOError:
      return None

    from tensorflow_transform import annotations_pb2  # pylint: disable=g-import-not-at-top
    for annotation in schema.annotation.extra_metadata:
      message = annotations_pb2.VocabularyMetadata()
      annotation.Unpack(message)
      # Check message.filtered_vocabulary_size is not 0 for backwards
      # compatibility.
      if (message.file_name == vocab_filename and
          message.filtered_vocabulary_size != 0):
        return message.filtered_vocabulary_size

    return None

  def vocabulary_size_by_name(self, vocab_filename: str) -> int:
    """Like vocabulary_file_by_name, but returns the size of vocabulary."""
    vocab_size_from_annotations = self._vocabulary_size_from_annotations(
        vocab_filename)
    if vocab_size_from_annotations is not None:
      return vocab_size_from_annotations

    vocab_path = self.vocabulary_file_by_name(vocab_filename)
    if not vocab_path:
      raise ValueError(
          'Could not compute vocabulary size for {}, does not exist'.format(
              vocab_filename))
    elif vocab_path.endswith('tfrecord.gz'):
      dataset = tf.data.TFRecordDataset(vocab_path, compression_type='GZIP')

      def reduce_fn(accum, elem):
        return tf.size(elem, out_type=tf.int64, name='vocabulary_size') + accum

      return _get_tensor_value(
          dataset.batch(tf.int32.max).reduce(
              tf.constant(0, tf.int64), reduce_fn))
    else:
      with tf.io.gfile.GFile(vocab_path, 'rb') as f:
        return sum(1 for _ in f)

  def vocabulary_by_name(self, vocab_filename: str) -> List[bytes]:
    """Like vocabulary_file_by_name but returns a list."""
    vocab_path = self.vocabulary_file_by_name(vocab_filename)
    if not vocab_path:
      raise ValueError('Could not read vocabulary: {}, does not exist'.format(
          vocab_filename))
    elif vocab_path.endswith('tfrecord.gz'):
      dataset = tf.data.TFRecordDataset(vocab_path, compression_type='GZIP')
      vocab_tensor = dataset.batch(tf.int32.max).reduce(
          tf.constant([], dtype=tf.string),
          lambda state, elem: tf.concat([state, elem], axis=-1))
      # Using as_numpy_iterator only works when executing eagerly.
      return _get_tensor_value(vocab_tensor).tolist()
    else:
      with tf.io.gfile.GFile(vocab_path, 'rb') as f:
        return [l.rstrip(os.linesep.encode('utf-8')) for l in f]

  # TODO(KesterTong): Add test for this in output_wrapper_test.py
  def num_buckets_for_transformed_feature(self, name: str) -> int:
    """Returns the number of buckets for an integerized transformed feature."""
    # Do checks that this tensor can be wrapped in
    # sparse_column_with_integerized_feature
    try:
      domain = self.transformed_domains()[name]
    except KeyError:
      raise ValueError('Column {} did not have a domain provided.'.format(name))
    if not isinstance(domain, schema_pb2.IntDomain):
      raise ValueError('Column {} has domain {}, expected an IntDomain'.format(
          name, domain))
    if domain.min != 0:
      raise ValueError('Column {} has min value {}, should be 0'.format(
          name, domain.min))
    return domain.max + 1

  def transform_features_layer(self) -> tf.keras.Model:
    """Creates a `TransformFeaturesLayer` from this transform output.

    If a `TransformFeaturesLayer` has already been created for self, the same
    one will be returned.

    Returns:
      A `TransformFeaturesLayer` instance.
    """
    if self._transform_features_layer is None:
      self._transform_features_layer = TransformFeaturesLayer(
          self, exported_as_v1=self._exported_as_v1)
    return self._transform_features_layer

  def transform_raw_features(
      self,
      raw_features: Mapping[str, common_types.TensorType],
      drop_unused_features: bool = True  # LEGACY_VALUE=False
  ) -> Dict[str, common_types.TensorType]:
    """Takes a dict of tensors representing raw features and transforms them.

    Takes a dictionary of `Tensor`s or `SparseTensor`s that represent the raw
    features, and applies the transformation defined by tf.Transform.

    If False it returns all transformed features defined by tf.Transform. To
    only return features transformed from the given 'raw_features', set
    `drop_unused_features` to True.

    Note: If eager execution is enabled and this API is invoked inside a
    tf.function or an API that uses tf.function such as dataset.map, please use
    `transform_features_layer` instead. It separates out loading of the
    transform graph and hence resources will not be initialized on each
    invocation. This can have significant performance improvement if the
    transform graph was exported as a TF1 SavedModel and guarantees correctness
    if it was exported as a TF2 SavedModel.

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
    if self._exported_as_v1:
      transformed_features = self._transform_raw_features_compat_v1(
          raw_features, drop_unused_features)
    else:
      tft_layer = self.transform_features_layer()
      if not drop_unused_features:
        tf.compat.v1.logging.warning(
            'Unused features are always dropped in the TF 2.x '
            'implementation. Ignoring value of drop_unused_features.')

      transformed_features = tft_layer(raw_features)
    return _TransformedFeaturesDict(transformed_features)

  def _transform_raw_features_compat_v1(
      self, raw_features: Mapping[str, common_types.TensorType],
      drop_unused_features: bool) -> Dict[str, common_types.TensorType]:
    """Takes a dict of tensors representing raw features and transforms them."""
    unbounded_raw_features, transformed_features = (
        saved_transform_io.partially_apply_saved_transform_internal(
            self.transform_savedmodel_dir, raw_features))
    if drop_unused_features:
      graph = tf.compat.v1.get_default_graph()
      graph_analyzer = graph_tools.InitializableGraphAnalyzer(
          graph, raw_features,
          [(t, False) for t in unbounded_raw_features.values()])
      return {
          name: feature
          for name, feature in transformed_features.items()
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
    if self._exported_as_v1 is None:
      self._exported_as_v1 = saved_transform_io.exported_as_v1(
          self.transform_savedmodel_dir)

    if self._exported_as_v1:
      saved_transform_io.partially_apply_saved_transform_internal(
          self.transform_savedmodel_dir, {})
    else:
      # Note: This should use the same mechanism as `transform_raw_features` to
      # load the SavedModel into the current graph context.
      _ = self.transform_features_layer()({})

  RAW_METADATA_DIR = 'metadata'
  _FEATURE_STATS_PB = 'FeatureStats.pb'
  PRE_TRANSFORM_FEATURE_STATS_PATH = os.path.join(
      'pre_transform_feature_stats', _FEATURE_STATS_PB)
  POST_TRANSFORM_FEATURE_STATS_PATH = os.path.join(
      'post_transform_feature_stats', _FEATURE_STATS_PB)

  @property
  def raw_metadata(self) -> dataset_metadata.DatasetMetadata:
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

  def raw_feature_spec(self) -> Dict[str, common_types.FeatureSpecType]:
    """Returns a feature_spec for the raw features.

    Returns:
      A dict from feature names to FixedLenFeature/SparseFeature/VarLenFeature.
    """
    return schema_utils.schema_as_feature_spec(
        self.raw_metadata.schema).feature_spec

  def raw_domains(self) -> Dict[str, common_types.DomainType]:
    """Returns domains for the raw features.

    Returns:
      A dict from feature names to one of schema_pb2.IntDomain,
      schema_pb2.StringDomain or schema_pb2.FloatDomain.
    """
    return schema_utils.schema_as_feature_spec(
        self.raw_metadata.schema).domains

  @property
  def pre_transform_statistics_path(self) -> str:
    """Returns the path to the pre-transform datum statistics.

    Note: pre_transform_statistics is not guaranteed to exist in the output of
    tf.transform and hence using this could fail, if pre_transform statistics is
    not present in TFTransformOutput.
    """
    return os.path.join(
        self._transform_output_dir, self.PRE_TRANSFORM_FEATURE_STATS_PATH)

  @property
  def post_transform_statistics_path(self) -> str:
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


def _check_tensorflow_version():
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


# TODO(b/162055065): Possibly switch back to inherit from Layer when possible.
@_maybe_register_keras_serializable(package='TensorFlowTransform')
class TransformFeaturesLayer(tf.keras.Model):
  """A Keras layer for applying a tf.Transform output to input layers."""

  def __init__(self,
               tft_output: TFTransformOutput,
               exported_as_v1: Optional[bool] = None):
    super().__init__(trainable=False)
    self._tft_output = tft_output
    if exported_as_v1 is None:
      self._exported_as_v1 = saved_transform_io.exported_as_v1(
          tft_output.transform_savedmodel_dir)
    else:
      self._exported_as_v1 = exported_as_v1
    self._saved_model_loader_value = None
    self._loaded_saved_model_graph = None
    # TODO(b/160294509): Use tf.compat.v1 when we stop supporting TF 1.15.
    if ops.executing_eagerly_outside_functions():
      _check_tensorflow_version()
      # The model must be tracked by assigning to an attribute of the Keras
      # layer. Hence, we track the attributes of _saved_model_loader here as
      # well.
      self._saved_model_loader_tracked_dict = self._saved_model_loader.__dict__

    # TODO(b/162055065): This is needed because otherwise we'd get an error in
    # some cases:
    # ValueError: Your Layer or Model is in an invalid state. This can happen
    # if you are interleaving estimator/non-estimator models or interleaving
    # models/layers made in tf.compat.v1.Graph.as_default() with models/layers
    # created outside of it. Converting a model to an estimator (via
    # model_to_estimator) invalidates all models/layers made before the
    # conversion (even if they were not the model converted to an estimator).
    # Similarly, making a layer or a model inside a a tf.compat.v1.Graph
    # invalidates all layers/models you previously made outside of the graph.
    self._originally_built_as_v1 = True

  @property
  def _saved_model_loader(self) -> saved_transform_io_v2.SavedModelLoader:
    """A `saved_transform_io_v2.SavedModelLoader`."""
    if self._saved_model_loader_value is None:
      self._saved_model_loader_value = saved_transform_io_v2.SavedModelLoader(
          self._tft_output.transform_savedmodel_dir)
      self._loaded_saved_model_graph = ops.get_default_graph()

    # TODO(b/160294509): Use tf.compat.v1 when we stop supporting TF 1.15.
    if ops.executing_eagerly_outside_functions():
      return self._saved_model_loader_value
    else:
      assert not self._exported_as_v1
      # TODO(b/149997088): Raise an exception once we no longer support using
      # the Keras layer with estimator based Trainer.
      tf.compat.v1.logging.warning('Loading a TF2 SavedModel but eager mode '
                                   'seems disabled.')
      # If exported as TF2 SavedModel but not invoked in eager mode,
      # re-initialize the saved_model_loader_value as __init__ could have been
      # called in a different graph context.
      default_graph = ops.get_default_graph()
      if (self._loaded_saved_model_graph is None or
          self._loaded_saved_model_graph is not default_graph):
        self._saved_model_loader_value = saved_transform_io_v2.SavedModelLoader(
            self._tft_output.transform_savedmodel_dir)
        self._loaded_saved_model_graph = default_graph
      return self._saved_model_loader_value

  def _init_batch_counters(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Overriding this method because Model's implementation creates variables.

    These Variables are not needed for TransformFeaturesLayer.
    """
    pass

  def call(
      self, inputs: Mapping[str, common_types.TensorType]
  ) -> Dict[str, common_types.TensorType]:

    if self._exported_as_v1 and not ops.executing_eagerly_outside_functions():
      tf.compat.v1.logging.warning('Falling back to transform_raw_features...')
      return self._tft_output._transform_raw_features_compat_v1(  # pylint: disable=protected-access
          inputs,
          drop_unused_features=True)
    else:
      return self._saved_model_loader.apply_transform_model(inputs)


def _make_method_override(name):

  @doc_controls.do_not_generate_docs
  def method_override(*args, **kwargs):
    raise NotImplementedError(name)

  return method_override


# TODO(zoyahav): Get rid of property attributes docs as well.
def _override_parent_methods(keep_items):
  """Makes inheritted attributes of the TFT layer unusable and undocumented."""
  for name in dir(tf.keras.Model):
    if name.startswith('_') or name in keep_items:
      continue
    if callable(getattr(tf.keras.Model, name)):
      setattr(TransformFeaturesLayer, name, _make_method_override(name))
    elif not isinstance(getattr(TransformFeaturesLayer, name), property):
      doc_controls.do_not_generate_docs(getattr(TransformFeaturesLayer, name))


_override_parent_methods(keep_items=[
    'call', 'build', 'compute_mask', 'add_loss', 'count_params',
    'finalize_state', 'save_spec'
])
