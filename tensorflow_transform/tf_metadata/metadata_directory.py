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
"""A prescribed directory structure for storing metadata in versioned formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


class DatasetMetadataDirectory(object):
  """A prescribed directory structure for storing metadata in versioned formats.
  """

  def __init__(self, basepath):
    self._basepath = basepath

  @property
  def assets_path(self):
    return os.path.join(self._basepath, 'assets')

  def version_dir(self, version):
    if version.version_flavor is not None:
      version_flavor_dir = version.version_key + '-' + version.version_flavor
    else:
      version_flavor_dir = version.version_key
    return DatasetMetadataVersionDirectory(
        os.path.join(self._basepath, version_flavor_dir))

  @property
  def basepath(self):
    return self._basepath


class DatasetMetadataVersionDirectory(object):
  """A prescribed directory structure for storing metadata in a known format."""

  def __init__(self, basepath):
    self._basepath = basepath

  def create(self):
    tf.gfile.MakeDirs(self._basepath)

  @property
  def schema_filename(self):
    return os.path.join(self._basepath, 'schema')

  @property
  def provenance_filename(self):
    return os.path.join(self._basepath, 'provenance')

  @property
  def statistics_path(self):
    return os.path.join(self._basepath, 'statistics')

  @property
  def anomalies_path(self):
    return os.path.join(self._basepath, 'anomalies')

  @property
  def problem_statements_path(self):
    return os.path.join(self._basepath, 'problem_statements')

