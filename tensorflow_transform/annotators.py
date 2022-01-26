# Copyright 2021 Google Inc. All Rights Reserved.
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
"""Functions that provide user annotations.

This module contains functions that are used in the preprocessing function to
annotate key aspects and make them easily accessible to downstream components.
"""

import contextlib
import os
from typing import Callable, List, Optional

import tensorflow as tf
from tensorflow_transform.graph_context import TFGraphContext

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.training.tracking import base
# pylint: enable=g-direct-tensorflow-import

__all__ = ['annotate_asset', 'make_and_track_object']

_ASSET_KEY_COLLECTION = 'tft_asset_key_collection'
_ASSET_FILENAME_COLLECTION = 'tft_asset_filename_collection'
# Thread-Hostile
_OBJECT_TRACKER = None
VOCABULARY_SIZE_BY_NAME_COLLECTION = 'tft_vocabulary_size_by_name_collection'


class ObjectTracker:
  """A class that tracks a list of trackable objects."""

  __slots__ = ['_trackable_objects']

  def __init__(self):
    self._trackable_objects = []

  @property
  def trackable_objects(self) -> List[base.Trackable]:
    return self._trackable_objects

  def add_trackable_object(self, trackable_object: base.Trackable,
                           name: Optional[str]):
    """Add `trackable_object` to list of objects tracked."""
    if name is None:
      self._trackable_objects.append(trackable_object)
    else:
      module = TFGraphContext.get_module_to_export()
      # The `preprocessing_fn` should always be invoked within a TFGraphContext.
      # If not, module will be None.
      if module is None:
        raise RuntimeError(
            f'No module found to track {name} with. Check that the '
            '`preprocessing_fn` is invoked within a `TFGraphContext` with a '
            'valid `TFGraphContext.module_to_export`.')
      if hasattr(module, name):
        raise ValueError(
            f'An object with name {name} is already being tracked. Check that a '
            'unique name was passed.')
      setattr(module, name, trackable_object)


# Thread-Hostile
@contextlib.contextmanager
def object_tracker_scope(object_tracker: ObjectTracker):
  """A context to manage trackable objects.

  Collects all trackable objects annotated using `track_object` within the body
  of its scope.

  Args:
    object_tracker: The passed in ObjectTracker object

  Yields:
    A scope in which the object_tracker is active.
  """
  global _OBJECT_TRACKER
  # Multiple nested object_tracker_scope calls are not expected.
  assert _OBJECT_TRACKER is None
  _OBJECT_TRACKER = object_tracker
  try:
    yield
  finally:
    _OBJECT_TRACKER = None


def _get_object(name: str) -> Optional[base.Trackable]:
  """If an object is being tracked using `name` return it, else None."""
  module = TFGraphContext.get_module_to_export()
  # The `preprocessing_fn` should always be invoked within a TFGraphContext. If
  # not, module will be None.
  if module is None:
    raise RuntimeError(
        f'No module found to track {name} with. Check that the `preprocessing_fn` is'
        ' invoked within a `TFGraphContext` with a valid '
        '`TFGraphContext.module_to_export`.')
  return getattr(module, name, None)


# Thread-Hostile
def track_object(trackable: base.Trackable, name: Optional[str]):
  """Add `trackable` to the object trackers active in this scope."""
  global _OBJECT_TRACKER
  # The transform tf.function should always be traced
  # (call to get_concrete_function) within an object_tracker_scope.
  assert _OBJECT_TRACKER is not None
  _OBJECT_TRACKER.add_trackable_object(trackable, name)


# Thread-Hostile
def make_and_track_object(trackable_factory_callable: Callable[[],
                                                               base.Trackable],
                          name: Optional[str] = None) -> base.Trackable:
  # pyformat: disable
  """Keeps track of the object created by invoking `trackable_factory_callable`.

  This API is only for use when Transform APIs are run with TF2 behaviors
  enabled and `tft_beam.Context.force_tf_compat_v1` is set to False.

  Use this API to track TF Trackable objects created in the `preprocessing_fn`
  such as tf.hub modules, tf.data.Dataset etc. This ensures they are serialized
  correctly when exporting to SavedModel.

  Args:
    trackable_factory_callable: A callable that creates and returns a Trackable
      object.
    name: (Optional) Provide a unique name to track this object with. If the
      Trackable object created is a Keras Layer or Model this is needed for
      proper tracking.

  Example:

  >>> def preprocessing_fn(inputs):
  ...   dataset = tft.make_and_track_object(
  ...       lambda: tf.data.Dataset.from_tensor_slices([1, 2, 3]))
  ...   with tf.init_scope():
  ...     dataset_list = list(dataset.as_numpy_iterator())
  ...   return {'x_0': dataset_list[0] + inputs['x']}
  >>> raw_data = [dict(x=1), dict(x=2), dict(x=3)]
  >>> feature_spec = dict(x=tf.io.FixedLenFeature([], tf.int64))
  >>> raw_data_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(
  ...     tft.tf_metadata.schema_utils.schema_from_feature_spec(feature_spec))
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp(),
  ...                       force_tf_compat_v1=False):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> transformed_data
  [{'x_0': 2}, {'x_0': 3}, {'x_0': 4}]

  Returns:
    The object returned when trackable_factory_callable is invoked. The object
    creation is lifted out to the eager context using `tf.init_scope`.
  """
  # pyformat: enable
  # TODO(b/165884902): Use tf.inside_function after dropping TF 1.15 support.
  if not isinstance(ops.get_default_graph(), func_graph.FuncGraph):
    raise ValueError('This API should only be invoked inside the user defined '
                     '`preprocessing_fn` with TF2 behaviors enabled and '
                     '`force_tf_compat_v1=False`. ')
  result = _get_object(name) if name is not None else None
  if result is None:
    with tf.init_scope():
      result = trackable_factory_callable()
      if name is None and isinstance(result, tf.keras.layers.Layer):
        raise ValueError(
            'Please pass a unique `name` to this API to ensure Keras objects '
            'are tracked correctly.')
      track_object(result, name)
  return result


def get_asset_annotations(graph: tf.Graph):
  """Obtains the asset annotations in the specified graph.

  Args:
    graph: A `tf.Graph` object.

  Returns:
    A dict that maps asset_keys to asset_filenames. Note that if multiple
    entries for the same key exist, later ones will override earlier ones.
  """
  asset_key_collection = graph.get_collection(_ASSET_KEY_COLLECTION)
  asset_filename_collection = graph.get_collection(_ASSET_FILENAME_COLLECTION)
  assert len(asset_key_collection) == len(
      asset_filename_collection
  ), 'Length of asset key and filename collections must match.'
  # Remove scope.
  annotations = {
      os.path.basename(key): os.path.basename(filename)
      for key, filename in zip(asset_key_collection, asset_filename_collection)
  }
  return annotations


def clear_asset_annotations(graph: tf.Graph):
  """Clears the asset annotations.

  Args:
    graph: A `tf.Graph` object.
  """
  graph.clear_collection(_ASSET_KEY_COLLECTION)
  graph.clear_collection(_ASSET_FILENAME_COLLECTION)


def annotate_asset(asset_key: str, asset_filename: str):
  """Creates mapping between user-defined keys and SavedModel assets.

  This mapping is made available in `BeamDatasetMetadata` and is also used to
  resolve vocabularies in `tft.TFTransformOutput`.

  Note: multiple mappings for the same key will overwrite the previous one.

  Args:
    asset_key: The key to associate with the asset.
    asset_filename: The filename as it appears within the assets/ subdirectory.
      Must be sanitized and complete (e.g. include the tfrecord.gz for suffix
      appropriate files).
  """
  tf.compat.v1.add_to_collection(_ASSET_KEY_COLLECTION, asset_key)
  tf.compat.v1.add_to_collection(_ASSET_FILENAME_COLLECTION, asset_filename)


def annotate_vocab_size(vocab_filename: str, vocab_size: tf.Tensor):
  """Adds annotation to retrieve the vocabulary size for `vocab_filename`."""
  tf.compat.v1.add_to_collection(VOCABULARY_SIZE_BY_NAME_COLLECTION,
                                 (vocab_filename, vocab_size))
