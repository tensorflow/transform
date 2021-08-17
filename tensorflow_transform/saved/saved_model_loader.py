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

from tensorflow_transform.saved import constants
from tensorflow.python.saved_model import loader_impl  # pylint: disable=g-direct-tensorflow-import


def parse_saved_model(saved_model_dir):
  return loader_impl.parse_saved_model(saved_model_dir)


def _choose_meta_graph_def_internal(saved_model, tags):
  """Find a MetaGraphDef within the SavedModel with exactly matching tags.

  Args:
    saved_model: A `SavedModel` protocol buffer.
    tags: Set of string tags to identify the required MetaGraphDef. These should
        correspond to the tags used when saving the variables using the
        SavedModel `save()` API.
  Returns:
    The chosen `MetaGraphDef` protocol buffer.  This can be used to further
    extract signature-defs, collection-defs, etc. If tags cannot be found,
    returns None.
  """
  result = None
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set(tags):
      result = meta_graph_def
      break

  return result


def choose_meta_graph_def(saved_model):
  """Find a MetaGraphDef in the SavedModel with tag `constants.TRANSFORM_TAG`.

  Args:
    saved_model: A `SavedModel` protocol buffer.

  Returns:
    The chosen `MetaGraphDef` protocol buffer.  This can be used to further
    extract signature-defs, collection-defs, etc. If tags cannot be found,
    returns None.
  """
  return _choose_meta_graph_def_internal(saved_model, [constants.TRANSFORM_TAG])


def choose_meta_graph_def_and_raise(saved_model):
  """Find a MetaGraphDef in the SavedModel with tag `constants.TRANSFORM_TAG`.

  Args:
    saved_model: A `SavedModel` protocol buffer.

  Returns:
    The chosen `MetaGraphDef` protocol buffer.  This can be used to further
    extract signature-defs, collection-defs, etc.

  Raises:
    RuntimeError: MetaGraphDef associated with the tags cannot be found.
  """
  result = choose_meta_graph_def(saved_model)

  if result is None:
    raise RuntimeError(
        'MetaGraphDef associated with tags {} could not be found in SavedModel'
        .format(constants.TRANSFORM_TAG))

  return result


def get_asset_tensors(saved_model_dir, meta_graph_def_to_load):
  return loader_impl.get_asset_tensors(saved_model_dir, meta_graph_def_to_load)
