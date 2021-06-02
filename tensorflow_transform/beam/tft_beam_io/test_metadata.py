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
"""Test metadata for tft_beam_io tests."""

import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURE_SPEC = {
    'fixed_column': tf.io.FixedLenFeature([3], tf.string),
    'list_columm': tf.io.VarLenFeature(tf.int64),
}

COMPLETE_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(
        _FEATURE_SPEC,
        domains={'list_columm': schema_pb2.IntDomain(min=-1, max=5)}))

INCOMPLETE_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(
        _FEATURE_SPEC,
        # Values will be overridden by those in COMPLETE_METADATA
        domains={'list_columm': schema_pb2.IntDomain(min=0, max=0)}))
