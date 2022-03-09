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
"""Simple Example of tf.Transform usage."""

import pprint
import tempfile

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

_RAW_DATA_METADATA = tft.DatasetMetadata.from_feature_spec({
    's': tf.io.FixedLenFeature([], tf.string),
    'y': tf.io.FixedLenFeature([], tf.float32),
    'x': tf.io.FixedLenFeature([], tf.float32),
})

_RAW_DATA = [{
    'x': 1,
    'y': 1,
    's': 'hello'
}, {
    'x': 2,
    'y': 2,
    's': 'world'
}, {
    'x': 3,
    'y': 3,
    's': 'hello'
}]


def _preprocessing_fn(inputs):
  """Preprocess input columns into transformed columns."""
  x = inputs['x']
  y = inputs['y']
  s = inputs['s']
  x_centered = x - tft.mean(x)
  y_normalized = tft.scale_to_0_1(y)
  s_integerized = tft.compute_and_apply_vocabulary(s)
  x_centered_times_y_normalized = (x_centered * y_normalized)
  return {
      'x_centered': x_centered,
      'y_normalized': y_normalized,
      'x_centered_times_y_normalized': x_centered_times_y_normalized,
      's_integerized': s_integerized
  }


def main():

  with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    transformed_dataset, transform_fn = (  # pylint: disable=unused-variable
        (_RAW_DATA, _RAW_DATA_METADATA)
        | tft_beam.AnalyzeAndTransformDataset(_preprocessing_fn))

  transformed_data, transformed_metadata = transformed_dataset  # pylint: disable=unused-variable

  pprint.pprint(transformed_data)

if __name__ == '__main__':
  main()
