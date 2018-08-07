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
                                              "schema_io"])):
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

    if self.schema_io is not None:
      schema = self.schema_io.read(vdir.schema_filename)

    return dataset_metadata.DatasetMetadata(schema=schema)

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
