# Copyright 2021 Google Inc. All Rights Reserved.
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
"""Tests for tensorflow_transform.beam.context."""

import os

import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import tft_unit


class ContextTest(tft_unit.TransformTestCase):

  def testNestedContextCreateBaseTempDir(self):

    level_1_dir = self.get_temp_dir()
    with tft_beam.Context(temp_dir=level_1_dir):
      self.assertEqual(
          os.path.join(level_1_dir, tft_beam.Context._TEMP_SUBDIR),
          tft_beam.Context.create_base_temp_dir())
      level_2_dir = self.get_temp_dir()
      with tft_beam.Context(temp_dir=level_2_dir):
        self.assertEqual(
            os.path.join(level_2_dir, tft_beam.Context._TEMP_SUBDIR),
            tft_beam.Context.create_base_temp_dir())
      self.assertEqual(
          os.path.join(level_1_dir, tft_beam.Context._TEMP_SUBDIR),
          tft_beam.Context.create_base_temp_dir())
    with self.assertRaises(ValueError):
      tft_beam.Context.create_base_temp_dir()


if __name__ == '__main__':
  tft_unit.main()
