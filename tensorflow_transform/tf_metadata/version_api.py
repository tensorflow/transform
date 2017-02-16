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
"""Representation of versioned metadata serialization strategies.

Specific serialization strategies should subclass the abstract *IO classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections


from tensorflow_transform.tf_metadata import dataset_metadata


class MetadataVersion(collections.namedtuple("MetadataVersion",
                                             ["version_key",
                                              "version_flavor",
                                              "schema_io",
                                              "statistics_io",
                                              "anomalies_io",
                                              "provenance_io",
                                              "problem_statements_io"])):
  """A specific metadata serialization format."""

  def read(self, vdir):
    """Read metadata from the given path.

    Args:
      vdir: A `DatasetMetadataVersionDirectory` from which the metadata should
        be read.

    Returns:
      A `DatasetMetadata` object.
    """

    schema = None
    provenance = None
    statistics = None
    anomalies = None
    problem_statements = None

    if self.schema_io is not None:
      schema = self.schema_io.read(
          vdir.schema_filename)
    if self.provenance_io is not None:
      provenance = self.provenance_io.read(
          vdir.provenance_filename)
    if self.statistics_io is not None:
      statistics = self.statistics_io.read(
          vdir.statistics_filename)
    if self.anomalies_io is not None:
      anomalies = self.anomalies_io.read(
          vdir.anomalies_filename)
    if self.problem_statements_io is not None:
      problem_statements = self.problem_statements_io.read(
          vdir.problem_statements_filename)

    return dataset_metadata.DatasetMetadata(
        schema=schema,
        statistics=statistics,
        anomalies=anomalies,
        provenance=provenance,
        problem_statements=problem_statements)

  def write(self, metadata, vdir):
    """Write metadata to a given path.

    Args:
      metadata: A `DatasetMetadata` to write.
      vdir: A `DatasetMetadataVersionDirectory` where the metadata should
        be written.
    """
    vdir.create()

    if self.schema_io is not None:
      self.schema_io.write(metadata.schema, vdir.schema_filename)
    if self.provenance_io is not None:
      self.provenance_io.write(metadata.provenance, vdir.provenance_filename)
    if self.statistics_io is not None:
      self.statistics_io.write(metadata.statistics, vdir.statistics_path)
    if self.anomalies_io is not None:
      self.anomalies_io.write(metadata.anomalies, vdir.anomalies_path)
    if self.problem_statements_io is not None:
      self.problem_statements_io.write(metadata.problem_statements,
                                       vdir.problem_statements_path)


class SchemaIO(object):
  """A SchemaIO represents a serialization strategy.

  It maps the in-memory `Schema` representation to and from a specific
  serialization format, such as certain protos, a JSON representation, etc.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def write(self, schema, path):
    """Write the schema to the given path.

    Args:
      schema: A `Schema` object to write.
      path: A path where the schema will be written as a single file (not a
        directory).  The implementation may append an appropriate filename
        extension (e.g. ".pbtxt", ".json") to the name.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def read(self, path):
    """Read the schema from the given path.

    Args:
      path: A path from which the schema should be read.  This path may exclude
        the implementation-specific filename extension.

    Returns:
      A `Schema` object.
    """
    raise NotImplementedError("Calling an abstract method.")


class ProvenanceIO(object):
  """A ProvenanceIO represents a serialization strategy.

  It maps the in-memory `Provenance` representation to and from a specific
  serialization format, such as certain protos, a JSON representation, etc.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def write(self, provenance, path):
    """Write the provenance to the given path.

    Args:
      provenance: A `Provenance` object to write.
      path: A path where the provenance will be written as a single file (not a
        directory).  The implementation may append an appropriate filename
        extension (e.g. ".pbtxt", ".json") to the name.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def read(self, path):
    """Read the provenance from the given path.

    Args:
      path: A path from which the provenance should be read.

    Returns:
      A `Provenance` object.
    """
    raise NotImplementedError("Calling an abstract method.")


class StatisticsIO(object):
  """A StatisticsIO represents a serialization strategy.

  It maps the in-memory `Statistics` representation to and from a specific
  serialization format, such as certain protos, a JSON representation, etc.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def write(self, statistics, path):
    """Write the statistics to the given path.

    Args:
      statistics: A `Statistics` object to write.
      path: A path where the statistics should be written.  The implementation
        will write files within a directory at this location.  The directory
        is expected to exist already.  Multiple files may be written within
        this directory.
    """
    # The implementation may choose the filenames it writes, but should take
    # care not to overwrite existing files.
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def read(self, path):
    """Read the statistics from the given path.

    Args:
      path: A path from which the statistics should be read, representing a
        directory that may contain multiple files.  All of these files will be
        read and their contents merged.

    Returns:
      A `Statistics` object.
    """
    raise NotImplementedError("Calling an abstract method.")


class AnomaliesIO(object):
  """An AnomaliesIO represents a serialization strategy.

  It maps the in-memory `Anomalies` representation to and from a specific
  serialization format, such as certain protos, a JSON representation, etc.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def write(self, anomalies, path):
    """Write the anomalies to the given path.

    Args:
      anomalies: An `Anomalies` object to write.
      path: A path where the anomalies should be written.  The implementation
        will write files within a directory at this location.  The directory
        is expected to exist already.  Multiple files may be written within
        this directory.
    """
    # The implementation may choose the filenames it writes, but should take
    # care not to overwrite existing files.
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def read(self, path):
    """Read the anomalies from the given path.

    Args:
      path: A path from which the anomalies should be read, representing a
        directory that may contain multiple files.  All of these files will be
        read and their contents merged.

    Returns:
      An `Anomalies` object.
    """
    raise NotImplementedError("Calling an abstract method.")


class ProblemStatementsIO(object):
  """A ProblemStatementsIO represents a serialization strategy.

  It maps the in-memory `ProblemStatements` representation to and from a
  specific serialization format, such as certain protos, a JSON representation,
  etc.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def write(self, problem_statements, path):
    """Write the problem_statements to the given path.

    Args:
      problem_statements: A `ProblemStatements` object to write.
      path: A path where the problem_statements should be written.  The
        implementation will write files within a directory at this location.
        The directory is expected to exist already.  Multiple files may be
        written within this directory.
    """
    # The implementation may choose the filenames it writes, but should take
    # care not to overwrite existing files.
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def read(self, path):
    """Read the problem_statements from the given path.

    Args:
      path: A path from which the problem_statements should be read,
        representing a directory that may contain multiple files.  All of these
        files will be read and their contents merged.

    Returns:
      A `ProblemStatements` object.
    """
    raise NotImplementedError("Calling an abstract method.")

