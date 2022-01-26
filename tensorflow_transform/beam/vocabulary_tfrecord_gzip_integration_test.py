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

import tensorflow as tf
from tensorflow_transform import tf2_utils
from tensorflow_transform import tf_utils
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.beam import vocabulary_integration_test

import unittest

mock = tf.compat.v1.test.mock


class TFRecordVocabularyIntegrationTest(
    vocabulary_integration_test.VocabularyIntegrationTest):

  def setUp(self):
    # TODO(b/164921571): Remove mock once tfrecord vocabularies are supported in
    # all TF versions.
    if not tf2_utils.use_tf_compat_v1(force_tf_compat_v1=False):
      self.is_vocabulary_tfrecord_supported_patch = mock.patch(
          'tensorflow_transform.tf_utils.is_vocabulary_tfrecord_supported')
      mock_is_vocabulary_tfrecord_supported = (
          self.is_vocabulary_tfrecord_supported_patch.start())
      mock_is_vocabulary_tfrecord_supported.side_effect = lambda: True

    if (tft_unit.is_external_environment() and
        not tf_utils.is_vocabulary_tfrecord_supported() or
        tft_unit.is_tf_api_version_1()):
      raise unittest.SkipTest('Test requires async DatasetInitializer')
    super().setUp()

  def tearDown(self):
    if not tf2_utils.use_tf_compat_v1(force_tf_compat_v1=False):
      self.is_vocabulary_tfrecord_supported_patch.stop()
    super().tearDown()

  def _VocabFormat(self):
    return 'tfrecord_gzip'


if __name__ == '__main__':
  tft_unit.main()
