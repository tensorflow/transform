# Copyright 2020 Google Inc. All Rights Reserved.
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
"""Common types in tf.transform."""

from typing import Iterable, TypeVar, Union, Any

import numpy as np
import tensorflow as tf
from typing_extensions import Literal

from tensorflow_metadata.proto.v0 import schema_pb2

# TODO(b/160294509): Stop using tracking.TrackableAsset when TF1.15 support is
# dropped.
if hasattr(tf.saved_model, 'Asset'):
  Asset = tf.saved_model.Asset  # pylint: disable=invalid-name
else:
  from tensorflow.python.training.tracking import tracking  # pylint: disable=g-direct-tensorflow-import, g-import-not-at-top
  Asset = tracking.TrackableAsset  # pylint: disable=invalid-name

# TODO(b/185719271): Define BucketBoundariesType at module level of mappers.py.
BucketBoundariesType = Union[tf.Tensor, Iterable[Union[int, float]]]

# TODO(b/160294509): RaggedFeature is not supported in TF 1.x. Clean this up
# once TF 1.x support is dropped.
if hasattr(tf.io, 'RaggedFeature'):
  FeatureSpecType = Union[tf.io.FixedLenFeature, tf.io.VarLenFeature,
                          tf.io.SparseFeature, tf.io.RaggedFeature]
  RaggedFeature = tf.io.RaggedFeature
else:
  FeatureSpecType = Union[tf.io.FixedLenFeature, tf.io.VarLenFeature,
                          tf.io.SparseFeature]
  RaggedFeature = Any

DomainType = Union[schema_pb2.IntDomain, schema_pb2.FloatDomain,
                   schema_pb2.StringDomain]
TensorType = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
ConsistentTensorType = TypeVar('ConsistentTensorType', tf.Tensor,
                               tf.SparseTensor, tf.RaggedTensor)
SparseTensorValueType = Union[tf.SparseTensor, tf.compat.v1.SparseTensorValue]
RaggedTensorValueType = Union[tf.RaggedTensor,
                              tf.compat.v1.ragged.RaggedTensorValue]
TensorValueType = Union[tf.Tensor, np.ndarray, SparseTensorValueType,
                        RaggedTensorValueType]
TemporaryAnalyzerOutputType = Union[tf.Tensor, Asset]
VocabularyFileFormatType = Literal['text', 'tfrecord_gzip']


def is_ragged_feature_available() -> bool:
  # TODO(b/160294509): RaggedFeature is not supported in TF 1.x. Clean this up
  # once TF 1.x support is dropped.
  return hasattr(tf.io, 'RaggedFeature')


def is_ragged_feature(spec: FeatureSpecType) -> bool:
  return (is_ragged_feature_available() and
          isinstance(spec, tf.io.RaggedFeature))
