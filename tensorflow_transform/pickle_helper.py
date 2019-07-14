# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Functions to fix pickling of certain objects (see b/121323638)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import copyreg
import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


_PROTO_CLASSES = [
    tf.compat.v1.ConfigProto,
    schema_pb2.Schema,
    statistics_pb2.DatasetFeatureStatistics
]


_PROTO_CLS_BY_NAME = {proto_cls.DESCRIPTOR.name: proto_cls
                      for proto_cls in _PROTO_CLASSES}


def _pickle_proto(proto):
  return _unpickle_proto, (proto.DESCRIPTOR.name, proto.SerializeToString())


def _unpickle_proto(name, serialized_proto):
  return _PROTO_CLS_BY_NAME[name].FromString(serialized_proto)


def fix_proto_pickling():
  """Fix pickling issues (see b/121323638)."""
  for proto_cls in _PROTO_CLASSES:
    copyreg.pickle(proto_cls, _pickle_proto)
