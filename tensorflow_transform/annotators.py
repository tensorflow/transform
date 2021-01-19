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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text

import tensorflow as tf

__all__ = ['annotate_asset']

_ASSET_KEY_COLLECTION = 'tft_asset_key_collection'
_ASSET_FILENAME_COLLECTION = 'tft_asset_filename_collection'


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


def annotate_asset(asset_key: Text, asset_filename: Text):
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
