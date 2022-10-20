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

from typing import Mapping, Optional, Type, TypeVar

from tensorflow_transform import common_types
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_metadata.proto.v0 import schema_pb2

_DatasetMetadataType = TypeVar('_DatasetMetadataType', bound='DatasetMetadata')


class DatasetMetadata:
  """Metadata about a dataset used for the "instance dict" format.

  Caution: The "instance dict" format used with `DatasetMetadata` is much less
  efficient than TFXIO. For any serious workloads you should use TFXIO with a
  `tfxio.TensorAdapterConfig` instance as the metadata. Refer to
  [Get started with TF-Transform](https://www.tensorflow.org/tfx/transform/get_started#data_formats_and_schema)
  for more details.

  This is an in-memory representation that may be serialized and deserialized to
  and from a variety of disk representations.
  """

  def __init__(self, schema: schema_pb2.Schema):
    self._schema = schema
    self._output_record_batches = True

  @classmethod
  def from_feature_spec(
      cls: Type[_DatasetMetadataType],
      feature_spec: Mapping[str, common_types.FeatureSpecType],
      domains: Optional[Mapping[str, common_types.DomainType]] = None
  ) -> _DatasetMetadataType:
    """Creates a DatasetMetadata from a TF feature spec dict."""
    return cls(schema_utils.schema_from_feature_spec(feature_spec, domains))

  @property
  def schema(self) -> schema_pb2.Schema:
    return self._schema

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.schema == other.schema
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return self.__dict__.__repr__()
