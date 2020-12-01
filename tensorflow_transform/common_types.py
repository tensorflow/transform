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

from typing import Union

import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2

# TODO(b/160294509): Stop using tracking.TrackableAsset when TF1.15 support is
# dropped.
if hasattr(tf.saved_model, 'Asset'):
  Asset = tf.saved_model.Asset  # pylint: disable=invalid-name
else:
  from tensorflow.python.training.tracking import tracking  # pylint: disable=g-direct-tensorflow-import, g-import-not-at-top
  Asset = tracking.TrackableAsset  # pylint: disable=invalid-name

FeatureSpecType = Union[tf.io.FixedLenFeature, tf.io.VarLenFeature,
                        tf.io.SparseFeature]
DomainType = Union[schema_pb2.IntDomain, schema_pb2.FloatDomain,
                   schema_pb2.StringDomain]
TensorType = Union[tf.Tensor, tf.SparseTensor]
TemporaryAnalyzerOutputType = Union[tf.Tensor, Asset]
