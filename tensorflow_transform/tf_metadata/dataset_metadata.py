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

from tensorflow_transform.tf_metadata import dataset_schema


class DatasetMetadata(object):
  """A collection of metadata about a dataset.

  This is an in-memory representation that may be serialized and deserialized to
  and from a variety of disk representations.  These disk representations must
  conform to the directory structure encoded in `DatasetMetadataDirectory`, but
  may vary in the file formats they write within those directories.
  """

  def __init__(self, schema=None):
    self.schema = schema or dataset_schema.Schema()

  @property
  def schema(self):
    return self._schema

  @schema.setter
  def schema(self, value):
    if isinstance(value, dict):
      value = dataset_schema.Schema(value)
    self._schema = value

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return self.__dict__.__repr__()
