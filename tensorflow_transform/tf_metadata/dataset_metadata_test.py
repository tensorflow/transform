# Copyright 2022 Google Inc. All Rights Reserved.
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
"""Tests for dataset_metadata."""

import unittest

from tensorflow_transform.tf_metadata import dataset_metadata, test_common


class DatasetSchemaTest(unittest.TestCase):
    def test_sanity(self):
        metadata = dataset_metadata.DatasetMetadata.from_feature_spec(
            test_common.test_feature_spec
        )
        self.assertEqual(metadata.schema, test_common.get_test_schema())
