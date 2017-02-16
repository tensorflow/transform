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
"""Utilities to read and write metadata in standardized versioned formats."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_directory
from tensorflow_transform.tf_metadata import version_api
from tensorflow_transform.tf_metadata.v1_json import schema_io_v1_json

# The _all_versions dict registers metadata versions that this library knows
# about.  Typically all known versions will be written, and the most recent
# known version available in a given directory will be parsed.
_V1_JSON = version_api.MetadataVersion('v1', 'json',
                                       schema_io_v1_json.SchemaIOv1JSON(),
                                       None, None, None, None)
_all_versions = {'1_JSON': _V1_JSON}.items()  # make immutable


def read_metadata(paths, versions=_all_versions):
  """Load metadata from multiple paths into a new DatasetMetadata."""
  dm = dataset_metadata.DatasetMetadata()
  if isinstance(paths, list):
    _read_merge_all(dm, paths, versions)
  else:
    _read_merge(dm, paths, versions)
  return dm


def write_metadata(metadata, path, versions=_all_versions):
  """Write all known versions, for forward compatibility.

  Args:
    metadata: A `DatasetMetadata` to write.
    path: a path to a directory where metadata should be written.
    versions: a dict of {version_id: MetadataVersion}; defaults to all known
      versions.
  """
  basedir = metadata_directory.DatasetMetadataDirectory(path)
  for _, version in versions:
    vdir = basedir.version_dir(version)
    version.write(metadata, vdir)


def _read_merge_all(metadata, paths, versions=_all_versions):
  """Load metadata from multiple paths into a DatasetMetadata.

  Args:
    metadata: A `DatasetMetadata` to update.
    paths: a list of file paths, each pointing to a metadata directory
      having the prescribed structure.  Each one may provide different
      metadata versions.
    versions: a dict of {version_id: MetadataVersion}; defaults to all known
      versions.
  """
  for path in paths:
    _read_merge(metadata, path, versions)


def _read_merge(metadata, path, versions=_all_versions):
  """Load metadata from a path into a DatasetMetadata.

  Args:
    metadata: A `DatasetMetadata` to update.
    path: A metadata directory having the prescribed structure.  Each one may
      provide different metadata versions.
    versions: a dict of {version_id: MetadataVersion}; defaults to all known
      versions.
  """
  basedir = metadata_directory.DatasetMetadataDirectory(path)

  (_, version), = versions
  vdir = basedir.version_dir(version)
  other = version.read(vdir)
  metadata.merge(other)
