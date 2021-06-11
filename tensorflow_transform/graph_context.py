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

import os
import threading
from typing import Any, Dict, Optional

import tensorflow as tf
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple


class TFGraphContext:
  """A context manager to pass global state to a TF graph when it is traced.

  All the attributes in this context are kept on a thread local state.

  Attributes:
    module_to_export: A tf.Module object that can be exported to a SavedModel
      and will be used to track objects created within this TF graph.
    temp_dir: The base path of the directory to write out any temporary files
      in this context block. If None, the TF graph in this context will be
      traced with placeholders for asset filepaths and is not serializable to a
      SavedModel.
    evaluated_replacements: A subset of placeholders/temporary asset files in
      `analyzer_nodes.TENSOR_REPLACEMENTS` that have been evaluated in
      previous TFT phases.

  Note that the temp dir should be accessible to worker jobs, e.g. if running
  with the Cloud Dataflow runner, the temp dir should be on GCS and should have
  permissions that allow both launcher and workers to access it.
  """

  class _State(
      tfx_namedtuple.namedtuple('_State', [
          'module_to_export',
          'temp_dir',
          'evaluated_replacements',
      ])):
    """A named tuple storing state passed to this context manager."""

    @classmethod
    def make_empty(cls):
      """Return `_State` object with all fields set to `None`."""
      return cls(*(None,) * len(cls._fields))

  _TEMP_SUBDIR = 'analyzer_temporary_assets'

  _thread_local = threading.local()

  def __init__(self,
               module_to_export: tf.Module,
               temp_dir: Optional[str] = None,
               evaluated_replacements: Optional[Dict[str, Any]] = None):
    self._module_to_export = module_to_export
    self._temp_dir = temp_dir
    self._evaluated_replacements = evaluated_replacements

  def __enter__(self):
    assert getattr(self._thread_local, 'current_state', None) is None
    self._thread_local.current_state = self._State(
        module_to_export=self._module_to_export,
        temp_dir=self._temp_dir,
        evaluated_replacements=self._evaluated_replacements)

  def __exit__(self, *exn_info):
    self._thread_local.current_state = None

  @property
  def module_to_export(self):
    return self._module_to_export

  @classmethod
  def _get_current_state(cls) -> 'TFGraphContext._State':
    if hasattr(cls._thread_local, 'current_state'):
      return cls._thread_local.current_state
    return cls._State.make_empty()

  @classmethod
  def get_or_create_temp_dir(cls) -> Optional[str]:
    """Generate a temporary location."""
    current_state = cls._get_current_state()
    if current_state.temp_dir is None:
      return None
    if not current_state.temp_dir:
      raise ValueError('A temp dir was requested, but empty temp_dir was set. '
                       'Use the TFGraphContext context manager.')
    result = os.path.join(current_state.temp_dir, cls._TEMP_SUBDIR)
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
    return cls._get_current_state().evaluated_replacements

  @classmethod
  def get_module_to_export(cls) -> Optional[tf.Module]:
    """Retrieves the value of module_to_export.

    None if called outside a TFGraphContext scope.

    Returns:
      A tf.Module object
    """
    return cls._get_current_state().module_to_export
