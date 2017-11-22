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
"""Tests for transform_fn_io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import apache_beam as beam
from apache_beam.testing import util as beam_test_util

import tensorflow as tf
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import futures
from tensorflow_transform.tf_metadata import metadata_io

import unittest
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io

_TEST_METADATA = dataset_metadata.DatasetMetadata({
    'fixed_column': dataset_schema.ColumnSchema(
        tf.string, (1, 3, 2), dataset_schema.FixedColumnRepresentation()),
    'fixed_column_with_default': dataset_schema.ColumnSchema(
        tf.float32, (1, 3, 2), dataset_schema.FixedColumnRepresentation(123.4)),
    'list_columm': dataset_schema.ColumnSchema(
        tf.float32, (None,), dataset_schema.ListColumnRepresentation())
})

_TEST_METADATA_WITH_FUTURES = dataset_metadata.DatasetMetadata({
    'fixed_column': dataset_schema.ColumnSchema(
        tf.string, (1, 3, 2), dataset_schema.FixedColumnRepresentation()),
    'fixed_column_with_default': dataset_schema.ColumnSchema(
        tf.float32, (1, futures.Future('a'), 2),
        dataset_schema.FixedColumnRepresentation(123.4)),
    'list_columm': dataset_schema.ColumnSchema(
        tf.float32, (None,), dataset_schema.ListColumnRepresentation())
})


class BeamMetadataIoTest(test_util.TensorFlowTestCase):

  def assertMetadataEqual(self, a, b):
    # Use extra assertEqual for schemas, since full metadata assertEqual error
    # message is not conducive to debugging.
    self.assertEqual(a.schema.column_schemas, b.schema.column_schemas)
    self.assertEqual(a, b)

  def testReadTransformFn(self):
    path = self.get_temp_dir()
    # NOTE: we don't need to create or write to the transform_fn directory since
    # ReadTransformFn never inspects this directory.
    transform_fn_dir = os.path.join(path, transform_fn_io.TRANSFORM_FN_DIR)
    transformed_metadata_dir = os.path.join(
        path, transform_fn_io.TRANSFORMED_METADATA_DIR)
    metadata_io.write_metadata(_TEST_METADATA, transformed_metadata_dir)

    with beam.Pipeline() as pipeline:
      saved_model_dir_pcoll, metadata = (
          pipeline | transform_fn_io.ReadTransformFn(path))
      beam_test_util.assert_that(
          saved_model_dir_pcoll, beam_test_util.equal_to([transform_fn_dir]),
          label='AssertSavedModelDir')
      # NOTE: metadata is currently read in a non-deferred manner.
      self.assertEqual(metadata, _TEST_METADATA)

  def testWriteTransformFn(self):
    path = os.path.join(self.get_temp_dir(), 'output')

    with beam.Pipeline() as pipeline:
      # Create an empty directory for the source saved model dir.
      saved_model_dir = os.path.join(self.get_temp_dir(), 'source')
      file_io.recursive_create_dir(saved_model_dir)
      saved_model_dir_pcoll = (
          pipeline | 'CreateSavedModelDir' >> beam.Create([saved_model_dir]))
      metadata = beam_metadata_io.BeamDatasetMetadata(
          _TEST_METADATA_WITH_FUTURES,
          {
              'a': pipeline | 'CreateA' >> beam.Create([3]),
          })

      _ = ((saved_model_dir_pcoll, metadata)
           | transform_fn_io.WriteTransformFn(path))

    transformed_metadata_dir = os.path.join(
        path, transform_fn_io.TRANSFORMED_METADATA_DIR)
    metadata = metadata_io.read_metadata(transformed_metadata_dir)
    self.assertEqual(metadata, _TEST_METADATA)

    transform_fn_dir = os.path.join(path, transform_fn_io.TRANSFORM_FN_DIR)
    self.assertTrue(file_io.file_exists(transform_fn_dir))
    self.assertTrue(file_io.is_directory(transform_fn_dir))

if __name__ == '__main__':
  unittest.main()
