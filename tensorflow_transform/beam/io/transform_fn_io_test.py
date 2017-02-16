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
"""Tests for tensorflow_transform.beam.io.transform_fn_io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile


import apache_beam as beam
import dill
import tensorflow as tf

import tensorflow_transform as tft
from tensorflow_transform import coders
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import io
from tensorflow_transform.beam.io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

import unittest


class CoderAssetsTest(unittest.TestCase):

  def _make_transform_fn(self, p, output_path):
    def preprocessing_fn(inputs):
      return {'x_scaled': tft.scale_to_0_1(inputs['x'])}
    schema = dataset_schema.from_feature_spec(
        {'x': tf.FixedLenFeature((), tf.float32, 0)})
    metadata = dataset_metadata.DatasetMetadata(schema=schema)
    columns = p | 'CreateTrainingData' >> beam.Create([{
        'x': v
    } for v in [4, 1, 5, 2]])
    _, result = (
        (columns, metadata)
        | 'AnalyzeAndTransform'
        >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn, output_path))
    coder = coders.CsvCoder(['x'], schema, delimiter='\t')
    return result, coder

  def testWriteTransformFnToSameLocation(self):
    # Write a transform_fn to the given path.
    base_dir = tempfile.mkdtemp()
    saved_model_path = os.path.join(base_dir, 'transform_fn_def')

    with self.assertRaisesRegexp(
        ValueError, 'Cannot write a TransformFn to its current location.'):
      with beam.Pipeline() as p:
        transform_fn, _ = self._make_transform_fn(p, saved_model_path)
        _ = transform_fn | transform_fn_io.WriteTransformFn(saved_model_path)

  def testCoderAssets(self):
    # Write a transform_fn to the given path.
    base_dir = tempfile.mkdtemp()
    saved_model_path = os.path.join(base_dir, 'transform_fn_def')

    with beam.Pipeline() as p:
      (transform_fn, _), coder = self._make_transform_fn(p, saved_model_path)
      _ = (transform_fn |
           'AppendCoderAssets' >> io.AppendCoderAssets([coder]))

    # Assert that coder assets got added to the transform_fn_def.
    self.assertTrue(os.path.isfile(
        os.path.join(saved_model_path,
                     'transform_fn',
                     transform_fn_io._ASSETS_EXTRA,
                     transform_fn_io._TF_TRANSFORM_CODERS_FILE_NAME)))

  def testAppendCoderAssetsAreCorrect(self):
    # Write a transform_fn to the given path.
    base_dir = tempfile.mkdtemp()
    saved_model_path = os.path.join(base_dir, 'transform_fn_load')

    # Load the coder from the saved assets and asssert it works.
    def assert_equal(transform_fn_def_dir, string, expected_value):
      pickled_coder = os.path.join(
          transform_fn_def_dir,
          transform_fn_io._ASSETS_EXTRA,
          transform_fn_io._TF_TRANSFORM_CODERS_FILE_NAME)
      with open(pickled_coder) as f:
        coders_dict = dill.load(f)
      self.assertEqual(expected_value, coders_dict['csv'].decode(string))
      self.assertEqual(expected_value, coders_dict['default'].decode(string))

    with beam.Pipeline() as p:
      (transform_fn, _), coder = self._make_transform_fn(p, saved_model_path)
      _ = (transform_fn |
           'AppendCoderAssets' >> io.AppendCoderAssets([coder]) |
           beam.Map(assert_equal, '1', {'x': 1}))

if __name__ == '__main__':
  unittest.main()
