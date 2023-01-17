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

from typing import Any, Dict, Iterable, List, TypeVar, Union, Optional

import numpy as np
import tensorflow as tf
from typing_extensions import Literal

from tensorflow_metadata.proto.v0 import schema_pb2

# Demonstrational per-row data formats.
PrimitiveType = Union[str, bytes, float, int]
InstanceValueType = Optional[
    Union[np.ndarray, np.generic, PrimitiveType, List[Any]]
]
InstanceDictType = Dict[str, InstanceValueType]

# TODO(b/185719271): Define BucketBoundariesType at module level of mappers.py.
BucketBoundariesType = Union[tf.Tensor, Iterable[Union[int, float]]]

FeatureSpecType = Union[tf.io.FixedLenFeature, tf.io.VarLenFeature,
                        tf.io.SparseFeature, tf.io.RaggedFeature]

DomainType = Union[schema_pb2.IntDomain, schema_pb2.FloatDomain,
                   schema_pb2.StringDomain]
TensorType = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
ConsistentTensorType = TypeVar(  # pylint: disable=invalid-name
    'ConsistentTensorType', tf.Tensor, tf.SparseTensor, tf.RaggedTensor)
SparseTensorValueType = Union[tf.SparseTensor, tf.compat.v1.SparseTensorValue]
RaggedTensorValueType = Union[tf.RaggedTensor,
                              tf.compat.v1.ragged.RaggedTensorValue]
TensorValueType = Union[tf.Tensor, np.ndarray, SparseTensorValueType,
                        RaggedTensorValueType]
TemporaryAnalyzerOutputType = Union[tf.Tensor, tf.saved_model.Asset]
VocabularyFileFormatType = Literal['text', 'tfrecord_gzip']
