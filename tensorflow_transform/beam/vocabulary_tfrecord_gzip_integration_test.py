#
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
"""Tests for tfrecord_gzip tft.vocabulary and tft.compute_and_apply_vocabulary."""

from tensorflow_transform import tf_utils
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.beam import vocabulary_integration_test

import unittest


class TFRecordVocabularyIntegrationTest(
    vocabulary_integration_test.VocabularyIntegrationTest):

  def setUp(self):
    if (tft_unit.is_external_environment() and
        not tf_utils.is_vocabulary_tfrecord_supported() or
        tft_unit.is_tf_api_version_1()):
      raise unittest.SkipTest('Test requires async DatasetInitializer')
    super().setUp()

  def _VocabFormat(self):
    return 'tfrecord_gzip'


if __name__ == '__main__':
  tft_unit.main()
