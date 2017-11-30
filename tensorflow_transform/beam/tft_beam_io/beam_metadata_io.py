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
import six
from tensorflow_transform.tf_metadata import metadata_io


class BeamDatasetMetadata(
    collections.namedtuple(
        'BeamDatasetMetadata', ['dataset_metadata', 'pcollections'])):
  """A class like DatasetMetadata that also holds a dict of `PCollection`s.

  DatasetMetadata allows values to be instances of the `Future` class which
  allows us to represent deferred objects.  This class allows us to also
  embed Beam values.  We do this by adding a dictionary, `pcollections` which
  maps the names of futures to Beam `PCollection`s.
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


class ResolveBeamFutures(beam.PTransform):
  """A PTransform to resolve futures of a DatasetMetadata."""

  # NOTE: The pipeline metadata is required by PTransform given that all the
  # inputs may be non-deferred.
  def __init__(self, pipeline):
    super(ResolveBeamFutures, self).__init__()
    self.pipeline = pipeline

  def _extract_input_pvalues(self, metadata):
    return metadata, getattr(metadata, 'pcollections', {}).values()

  def expand(self, metadata):
    if isinstance(metadata, BeamDatasetMetadata):
      pcollections = metadata.pcollections
      metadata = metadata.dataset_metadata
    else:
      pcollections = {}

    # Extract `PCollection`s from futures.
    tensor_value_pairs = []
    for name, pcoll in six.iteritems(pcollections):
      tensor_value_pairs.append(
          pcoll
          | 'AddName[%s]' % name >> beam.Map(lambda x, name=name: (name, x)))
    tensor_value_mapping = beam.pvalue.AsDict(
        tensor_value_pairs | 'MergeTensorValuePairs' >> beam.Flatten(
            pipeline=self.pipeline))

    def resolve_futures(dummy_input, updated_metadata, future_values):
      updated_metadata.substitute_futures(future_values)
      return updated_metadata

    return (self.pipeline
            | 'CreateSingleton' >> beam.Create([None])
            | 'ResolveFutures' >> beam.Map(resolve_futures, metadata,
                                           tensor_value_mapping))


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
    return metadata, getattr(metadata, 'pcollections', {}).values()

  def expand(self, metadata):
    return (metadata
            | 'ResolveBeamFutures' >> ResolveBeamFutures(self.pipeline)
            | 'WriteMetadata' >> beam.Map(metadata_io.write_metadata,
                                          self._path))
