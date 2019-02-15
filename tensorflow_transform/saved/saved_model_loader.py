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
"""Utility functions to build input_fns for use with tf.Learn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# GOOGLE-INITIALIZATION

from tensorflow.python.saved_model import loader_impl


# This file is forked and refactored from saved_model/loader_impl.py
# TODO(b/123242568): refactor, moving most of this back into saved_model.


def parse_saved_model(saved_model_dir):
  # pylint: disable=protected-access
  return loader_impl._parse_saved_model(saved_model_dir)
  # pylint: enable=protected-access


def choose_meta_graph_def(saved_model, tags):
  """Find a MetaGraphDef within the SavedModel with exactly matching tags.

  Args:
    saved_model: A `SavedModel` protocol buffer.
    tags: Set of string tags to identify the required MetaGraphDef. These should
        correspond to the tags used when saving the variables using the
        SavedModel `save()` API.
  Returns:
    The chosen `MetaGraphDef` protocol buffer.  This can be used to further
    extract signature-defs, collection-defs, etc.

  Raises:
    RuntimeError: MetaGraphDef associated with the tags cannot be found.
  """
  found_match = False
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set(tags):
      meta_graph_def_to_load = meta_graph_def
      found_match = True
      break

  if not found_match:
    raise RuntimeError('MetaGraphDef associated with tags '  + str(tags).strip(
        '[]') + ' could not be found in SavedModel')

  return meta_graph_def_to_load


def get_asset_tensors(saved_model_dir, meta_graph_def_to_load):
  try:
    return loader_impl.get_asset_tensors(saved_model_dir,
                                         meta_graph_def_to_load)
  # TODO(b/124491249): Remove this backwards compatibility once TFT 0.14 is
  # released.
  except AttributeError:
    return loader_impl._get_asset_tensors(  # pylint: disable=protected-access
        saved_model_dir, meta_graph_def_to_load)
