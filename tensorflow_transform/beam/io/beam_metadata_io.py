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
"""Transforms to read/write metadata from disk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
from tensorflow_transform.tf_metadata import metadata_io


class ReadMetadata(beam.PTransform):
  """A PTransform to read Metadata from disk."""

  def __init__(self, path):
    super(ReadMetadata, self).__init__()
    self._path = path

  def expand(self, pvalue):
    # Read metadata in non-deferred manner.  Note that since this reads the
    # whole metadata in a non-deferred manner, typically the roundtrip
    #
    # done = metadata | WriteMetadata(path)
    # metadata = p | ReadMetadata(path).must_follow(done)
    #
    # will fail as the metadata on disk will not be complete when the read is
    # done.
    return metadata_io.read_metadata(self._path)


class WriteMetadata(beam.PTransform):

  # NOTE: The pipeline metadata is required by PTransform given that all the
  # inpits are non-deferred.
  def __init__(self, path, pipeline):
    super(WriteMetadata, self).__init__()
    self._path = path
    self.pipeline = pipeline

  def _extract_input_pvalues(self, metadata):
    return metadata, []

  def expand(self, metadata):
    """A PTransform to write Metadata to disk."""
    metadata_io.write_metadata(metadata, self._path)
