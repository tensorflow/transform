"""Utilities to read and write metadata in standardized versioned formats."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_directory
# from tensorflow_transform.tf_metadata.v1 import schema_io_v1

# The _all_versions dict registers metadata versions that this library knows
# about.  Typically all known versions will be written, and the most recent
# known version available in a given directory will be parsed.
# TODO(soergel): uncomment this in the followup CL that establishes v1.
# _V1 = version_api.MetadataVersion("v1", schema_io_v1.SchemaIOv1(),
#                                  None, None, None, None)

# Versions are incrementing integers starting with 1
# _all_versions = {1: _V1}.items()  # make immutable
_all_versions = {}.items()  # make immutable


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
  for _, version in versions.items():
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

  # TODO(soergel): # choose best version in common between this and the dir
  # best_version_number = 1
  # version = versions[best_version_number]
  (_, version), = versions.items()
  vdir = basedir.version_dir(version)
  other = version.read(vdir)
  metadata.merge(other)
