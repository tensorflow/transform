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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import tempfile

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as beam_impl
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


def preprocessing_fn(inputs):
  """Preprocess input columns into transformed columns."""
  x = inputs['x']
  y = inputs['y']
  s = inputs['s']
  x_centered = tft.map(lambda x, mean: x - mean, x, tft.mean(x))
  y_normalized = tft.scale_to_0_1(y)
  s_integerized = tft.string_to_int(s)
  x_centered_times_y_normalized = tft.map(lambda x, y: x * y,
                                          x_centered, y_normalized)
  return {
      'x_centered': x_centered,
      'y_normalized': y_normalized,
      'x_centered_times_y_normalized': x_centered_times_y_normalized,
      's_integerized': s_integerized
  }

raw_data = [
    {'x': 1, 'y': 1, 's': 'hello'},
    {'x': 2, 'y': 2, 's': 'world'},
    {'x': 3, 'y': 3, 's': 'hello'}
]

raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
    's': dataset_schema.ColumnSchema(
        dataset_schema.LogicalColumnSchema(
            dataset_schema.Domain(tf.string), dataset_schema.LogicalShape([])),
        dataset_schema.FixedColumnRepresentation()),
    'y': dataset_schema.ColumnSchema(
        dataset_schema.LogicalColumnSchema(
            dataset_schema.Domain(tf.float32), dataset_schema.LogicalShape([])),
        dataset_schema.FixedColumnRepresentation()),
    'x': dataset_schema.ColumnSchema(
        dataset_schema.LogicalColumnSchema(
            dataset_schema.Domain(tf.float32), dataset_schema.LogicalShape([])),
        dataset_schema.FixedColumnRepresentation())
}))

transformed_dataset, transform_fn = (
    (raw_data, raw_data_metadata) | beam_impl.AnalyzeAndTransformDataset(
        preprocessing_fn, tempfile.mkdtemp()))

transformed_data, transformed_metadata = transformed_dataset

pprint.pprint(transformed_data)
