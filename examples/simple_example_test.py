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
"""Tests for simple_example."""

import tensorflow as tf
import simple_example
from tensorflow_transform.beam import tft_unit


_EXPECTED_TRANSFORMED_OUTPUT = [
    {
        'x_centered': 1.0,
        'y_normalized': 1.0,
        'x_centered_times_y_normalized': 1.0,
        's_integerized': 0,
    },
    {
        'x_centered': 0.0,
        'y_normalized': 0.5,
        'x_centered_times_y_normalized': 0.0,
        's_integerized': 1,
    },
    {
        'x_centered': -1.0,
        'y_normalized': 0.0,
        'x_centered_times_y_normalized': -0.0,
        's_integerized': 0,
    },
]


class SimpleExampleTest(tft_unit.TransformTestCase):

  def test_preprocessing_fn(self):
    self.assertAnalyzeAndTransformResults(simple_example._RAW_DATA,
                                          simple_example._RAW_DATA_METADATA,
                                          simple_example._preprocessing_fn,
                                          _EXPECTED_TRANSFORMED_OUTPUT)


class SimpleMainTest(tf.test.TestCase):

  def testMainDoesNotCrash(self):
    simple_example.main()


if __name__ == '__main__':
  tf.test.main()
