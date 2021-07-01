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

import copyreg
import tensorflow as tf
from tensorflow_transform import common
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

if common.IS_ANNOTATIONS_PB_AVAILABLE:
  from tensorflow_transform import annotations_pb2  # pylint: disable=g-import-not-at-top

_ANNOTATION_CLASSES = [
    annotations_pb2.VocabularyMetadata, annotations_pb2.BucketBoundaries
] if common.IS_ANNOTATIONS_PB_AVAILABLE else []

_PROTO_CLASSES = [
    tf.compat.v1.ConfigProto,
    schema_pb2.Schema,
    statistics_pb2.DatasetFeatureStatistics,
] + _ANNOTATION_CLASSES


_PROTO_CLS_BY_NAME = {proto_cls.DESCRIPTOR.name: proto_cls
                      for proto_cls in _PROTO_CLASSES}


def _pickle_proto(proto):
  return _unpickle_proto, (proto.DESCRIPTOR.name, proto.SerializeToString())


def _unpickle_proto(name, serialized_proto):
  return _PROTO_CLS_BY_NAME[name].FromString(serialized_proto)


def _pickle_tensor_spec(tensor_spec):
  return _unpickle_tensor_spec, (tensor_spec.shape.as_list(),
                                 tensor_spec.dtype.as_numpy_dtype)


def _unpickle_tensor_spec(shape, numpy_dtype):
  return tf.TensorSpec(shape, tf.as_dtype(numpy_dtype))


def fix_internal_object_pickling():
  """Fix pickling issues (see b/121323638)."""
  for proto_cls in _PROTO_CLASSES:
    copyreg.pickle(proto_cls, _pickle_proto)

  copyreg.pickle(tf.TensorSpec, _pickle_tensor_spec)
