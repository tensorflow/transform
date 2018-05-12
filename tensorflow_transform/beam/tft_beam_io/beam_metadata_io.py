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

import collections

import apache_beam as beam
from tensorflow_transform.tf_metadata import metadata_io


class BeamDatasetMetadata(
    collections.namedtuple(
        'BeamDatasetMetadata',
        ['dataset_metadata', 'deferred_metadata'])):
  """A class like DatasetMetadata that also holds a dict of `PCollection`s.

  `deferred_metadata` is a PCollection containing a single DatasetMetadata.
  """

  @property
  def schema(self):
    return self.dataset_metadata.schema

  @schema.setter
  def schema(self, value):
    self.dataset_metadata.schema = value

  @property
  def provenance(self):
    return self.dataset_metadata.provenance

  @property
  def statistics(self):
    return self.dataset_metadata.statistics

  @property
  def anomalies(self):
    return self.dataset_metadata.anomalies

  @property
  def problem_statements(self):
    return self.dataset_metadata.problem_statements

  def merge(self, other):
    raise NotImplementedError


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

  def _extract_input_pvalues(self, metadata):
    pvalues = []
    if isinstance(metadata, BeamDatasetMetadata):
      pvalues.append(metadata.deferred_metadata)
    return metadata, pvalues

  def expand(self, metadata):
    if hasattr(metadata, 'deferred_metadata'):
      metadata_pcoll = metadata.deferred_metadata
    else:
      metadata_pcoll = self.pipeline | beam.Create([metadata])
    return metadata_pcoll | 'WriteMetadata' >> beam.Map(
        metadata_io.write_metadata, self._path)
