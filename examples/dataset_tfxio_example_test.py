# Copyright 2023 Google Inc. All Rights Reserved.
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
"""Tests for dataset_tfxio."""

import tensorflow as tf
import dataset_tfxio_example
from tensorflow_transform.beam import tft_unit


_EXPECTED_TRANSFORMED_OUTPUT = [
    {'x_scaled': 0.0, 'x_centered': -4.0},
    {'x_scaled': 0.125, 'x_centered': -3.0},
    {'x_scaled': 0.25, 'x_centered': -2.0},
    {'x_scaled': 0.375, 'x_centered': -1.0},
    {'x_scaled': 0.5, 'x_centered': 0.0},
    {'x_scaled': 0.625, 'x_centered': 1.0},
    {'x_scaled': 0.75, 'x_centered': 2.0},
    {'x_scaled': 0.875, 'x_centered': 3.0},
    {'x_scaled': 1.0, 'x_centered': 4.0},
]


class SimpleMainTest(tf.test.TestCase):

  def testMainDoesNotCrash(self):
    tft_unit.skip_if_not_tf2('Tensorflow 2.x required.')
    dataset_tfxio_example.main('')


class SimpleProcessingTest(tft_unit.TransformTestCase):

  # Asserts equal for each element. (Does not check batchwise.)
  def test_preprocessing_fn(self):
    tfxio = dataset_tfxio_example._make_tfxio()
    self.assertAnalyzeAndTransformResults(
        tfxio.BeamSource(),
        tfxio.TensorAdapterConfig(),
        dataset_tfxio_example._preprocessing_fn,
        _EXPECTED_TRANSFORMED_OUTPUT,
    )


if __name__ == '__main__':
  tf.test.main()
