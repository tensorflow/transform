"""A prescribed directory structure for storing metadata in versioned formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


class DatasetMetadataDirectory(object):
  """A prescribed directory structure for storing metadata in versioned formats.
  """

  def __init__(self, basepath):
    self._basepath = basepath

  @property
  def assets_path(self):
    return os.path.join(self._basepath, "assets")

  def version_dir(self, version):
    return DatasetMetadataVersionDirectory(
        os.path.join(self._basepath, version.version_key))

  @property
  def basepath(self):
    return self._basepath


class DatasetMetadataVersionDirectory(object):
  """A prescribed directory structure for storing metadata in a known format."""

  def __init__(self, basepath):
    self._basepath = basepath

  def create(self):
    os.makedirs(self._basepath)

  @property
  def schema_filename(self):
    return os.path.join(self._basepath, "schema")

  @property
  def provenance_filename(self):
    return os.path.join(self._basepath, "provenance")

  @property
  def statistics_path(self):
    return os.path.join(self._basepath, "statistics")

  @property
  def anomalies_path(self):
    return os.path.join(self._basepath, "anomalies")

  @property
  def problem_statements_path(self):
    return os.path.join(self._basepath, "problem_statements")

