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
"""Tests for beam_metadata_io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import apache_beam as beam

import tensorflow as tf
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io

import unittest
from tensorflow.python.framework import test_util

_TEST_METADATA_COMPLETE = dataset_metadata.DatasetMetadata({
    'fixed_column': dataset_schema.ColumnSchema(
        tf.string, (3,), dataset_schema.FixedColumnRepresentation()),
    'list_columm': dataset_schema.ColumnSchema(
        dataset_schema.IntDomain(tf.int64, min_value=-1, max_value=5),
        (None,), dataset_schema.ListColumnRepresentation())
})

_TEST_METADATA = dataset_metadata.DatasetMetadata({
    'fixed_column': dataset_schema.ColumnSchema(
        tf.string, (3,), dataset_schema.FixedColumnRepresentation()),
    # zeros will be overriddden
    'list_columm': dataset_schema.ColumnSchema(
        dataset_schema.IntDomain(tf.int64, min_value=0, max_value=0),
        (None,), dataset_schema.ListColumnRepresentation())
})


class BeamMetadataIoTest(test_util.TensorFlowTestCase):

  def testWriteMetadataNonDeferred(self):
    # Write properties as metadata to disk.
    with beam.Pipeline() as pipeline:
      path = self.get_temp_dir()
      _ = (_TEST_METADATA_COMPLETE
           | beam_metadata_io.WriteMetadata(path, pipeline))
    # Load from disk and check that it is as expected.
    metadata = metadata_io.read_metadata(path)
    self.assertEqual(metadata, _TEST_METADATA_COMPLETE)

  def testWriteMetadataDeferredProperties(self):
    # Write deferred properties as metadata to disk.
    with beam.Pipeline() as pipeline:
      path = self.get_temp_dir()

      # Combine _TEST_METADATA with the complete (deferred) metadata.
      deferred_metadata = pipeline | 'CreateDeferredMetadata' >> beam.Create(
          [_TEST_METADATA_COMPLETE])
      metadata = beam_metadata_io.BeamDatasetMetadata(
          _TEST_METADATA, deferred_metadata)

      _ = metadata | beam_metadata_io.WriteMetadata(path, pipeline)
    # Load from disk and check that it is as expected.
    metadata = metadata_io.read_metadata(path)
    self.assertEqual(metadata, _TEST_METADATA_COMPLETE)


if __name__ == '__main__':
  unittest.main()
