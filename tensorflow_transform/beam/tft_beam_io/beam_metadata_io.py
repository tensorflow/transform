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
"""Transforms to read/write metadata from disk.

A write/read cycle will render all metadata deferred, but in general users
should avoid doing this anyway and pass around live metadata objects.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io


class WriteMetadata(beam.PTransform):
  """A PTransform to write Metadata to disk.

  Input can either be a DatasetMetadata or a tuple of properties.
  """

  # NOTE: The pipeline metadata is required by PTransform given that all the
  # inputs may be non-deferred.
  def __init__(self, path, pipeline):
    super(WriteMetadata, self).__init__()
    self._path = path
    self.pipeline = pipeline

  def _extract_input_pvalues(self, metadata_or_tuple):
    if isinstance(metadata_or_tuple, dataset_metadata.DatasetMetadata):
      return metadata_or_tuple, []
    else:
      return metadata_or_tuple, [metadata_or_tuple[1]]

  def expand(self, metadata_or_tuple):
    if isinstance(metadata_or_tuple, dataset_metadata.DatasetMetadata):
      metadata = metadata_or_tuple
      deferred_metadata = (
          self.pipeline | 'CreateEmptyDeferredMetadata' >> beam.Create([{}]))
    else:
      metadata, deferred_metadata = metadata_or_tuple

    def write_metadata(futures_dict, non_deferred_metadata, destination):
      unresolved_futures = non_deferred_metadata.substitute_futures(
          futures_dict)
      if unresolved_futures:
        raise ValueError(
            'Some futures were unresolved: %r' % unresolved_futures)
      metadata_io.write_metadata(non_deferred_metadata, destination)

    return (
        deferred_metadata
        | 'WriteMetadata' >> beam.Map(write_metadata, metadata, self._path))
