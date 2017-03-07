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
"""In-memory representation of all metadata associated with a dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow_transform.tf_metadata import dataset_anomalies
from tensorflow_transform.tf_metadata import dataset_problem_statements
from tensorflow_transform.tf_metadata import dataset_provenance
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_statistics


class DatasetMetadata(collections.namedtuple(
    'DatasetMetadata',
    ['schema', 'provenance', 'statistics', 'anomalies', 'problem_statements'])):
  """A collection of metadata about a dataset.

  This is an in-memory representation that may be serialized and deserialized to
  and from a variety of disk representations.  These disk representations must
  conform to the directory structure encoded in `DatasetMetadataDirectory`, but
  may vary in the file formats they write within those directories.
  """

  def __new__(
      cls,
      schema=None,
      provenance=None,
      statistics=None,
      anomalies=None,
      problem_statements=None):
    if isinstance(schema, dict):
      schema = dataset_schema.Schema(schema)
    schema = schema or dataset_schema.Schema()
    provenance = provenance or dataset_provenance.Provenance()
    statistics = statistics or dataset_statistics.Statistics()
    anomalies = anomalies or dataset_anomalies.Anomalies()
    problem_statements = (
        problem_statements or dataset_problem_statements.ProblemStatements())
    return super(DatasetMetadata, cls).__new__(
        cls, schema, provenance, statistics, anomalies, problem_statements)

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self._asdict() == other._asdict()
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return self._asdict().__repr__()

  def merge(self, other):
    self.schema.merge(other.schema)
    self.provenance.merge(other.provenance)
    self.statistics.merge(other.statistics)
    self.anomalies.merge(other.anomalies)
    self.problem_statements.merge(other.problem_statements)
