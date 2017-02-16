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
"""Serialization strategy mapping `Schema` to v1 JSON."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow_transform.tf_metadata import version_api
from tensorflow_transform.tf_metadata.v1_json import schema_io_v1_json_reader
from tensorflow_transform.tf_metadata.v1_json import schema_io_v1_json_writer

from tensorflow.python.lib.io import file_io


class SchemaIOv1JSON(version_api.SchemaIO):
  """Serialization strategy for the v1 Schema as JSON."""

  def write(self, schema, path):
    """Writes a v1 `Schema` to disk as JSON.

    The function converts the in-memory Schema representation to the v1 Schema
    JSON representation, and writes it to the specified path.

    Args:
      schema: The Schema to write.
      path: the filename to write to.
    """
    schema_as_json = schema_io_v1_json_writer.to_schema_json(schema)

    basedir = os.path.dirname(path)
    if not file_io.file_exists(basedir):
      file_io.recursive_create_dir(basedir)

    file_io.write_string_to_file(path + ".json", schema_as_json)

  def read(self, path):
    """Reads a v1 JSON schema from disk."""
    # Ensure that the Schema file exists
    if not file_io.file_exists(path + ".json"):
      raise IOError("v1 Schema file does not exist at: %s" % path)

    file_content = file_io.FileIO(path + ".json", "r").read()
    return schema_io_v1_json_reader.from_schema_json(file_content)
