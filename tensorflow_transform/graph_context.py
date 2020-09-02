# Copyright 2020 Google Inc. All Rights Reserved.
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
"""Context manager for TF Graph when it is being traced."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import collections
import os
from typing import Any, Dict, Optional

# GOOGLE-INITIALIZATION

import tensorflow as tf

_CURRENT_STATE = None


class TFGraphContext(object):
  """A context manager to pass global state to a TF graph when it is traced.

  This context manager is not thread safe.
  """

  class _State(
      collections.namedtuple('_State', [
          'temp_dir',
          'evaluated_replacements',
      ])):
    """A named tuple storing state passed to this context manager.

    Fields:
      temp_dir: The base path of the directory to write out any temporary files
        in this context block.
      evaluated_replacements: A subset of placeholders/temporary asset files in
        `analyzer_nodes.TENSOR_REPLACEMENTS` that have been evaluated in
        previous TFT phases.
    """
    pass

  _TEMP_SUBDIR = 'analyzer_temporary_assets'

  def __init__(self,
               temp_dir: str,
               evaluated_replacements: Optional[Dict[str, Any]] = None):
    self._temp_dir = temp_dir
    self._evaluated_replacements = evaluated_replacements

  def __enter__(self):
    global _CURRENT_STATE
    assert _CURRENT_STATE is None
    _CURRENT_STATE = self._State(
        temp_dir=self._temp_dir,
        evaluated_replacements=self._evaluated_replacements)

  def __exit__(self, *exn_info):
    global _CURRENT_STATE
    _CURRENT_STATE = None

  @classmethod
  def get_or_create_temp_dir(cls) -> str:
    """Generate a temporary location."""
    if _CURRENT_STATE is None or not _CURRENT_STATE.temp_dir:
      raise ValueError('A temp dir was requested, but no temp_dir was set. Use '
                       'the _TFGraphContext context manager.')
    result = os.path.join(_CURRENT_STATE.temp_dir, cls._TEMP_SUBDIR)
    tf.io.gfile.makedirs(result)
    return result

  @classmethod
  def get_evaluated_replacements(cls) -> Optional[Dict[str, Any]]:
    """Retrieves the value of evaluated_replacements if set.

    None otherwise.

    Returns:
      A dictionary from graph tensor names to evaluated values for these
      tensors. The keys are a subset of placeholders/temporary asset files in
      `analyzer_nodes.TENSOR_REPLACEMENTS` that have been evaluated in
      previous TFT phases.
    """
    if _CURRENT_STATE is None:
      return None
    return _CURRENT_STATE.evaluated_replacements
