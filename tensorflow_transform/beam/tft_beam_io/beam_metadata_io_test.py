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

# GOOGLE-INITIALIZATION

import apache_beam as beam
import tensorflow as tf

from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.beam.tft_beam_io import test_metadata
from tensorflow_transform.tf_metadata import metadata_io


class BeamMetadataIoTest(tf.test.TestCase):

  def testWriteMetadataNonDeferred(self):
    # Write metadata to disk using WriteMetadata PTransform.
    with beam.Pipeline() as pipeline:
      path = self.get_temp_dir()
      _ = (test_metadata.COMPLETE_METADATA
           | beam_metadata_io.WriteMetadata(path, pipeline))

    # Load from disk and check that it is as expected.
    metadata = metadata_io.read_metadata(path)
    self.assertEqual(metadata, test_metadata.COMPLETE_METADATA)

  def testWriteMetadataDeferred(self):
    # Write metadata to disk using WriteMetadata PTransform, combining
    # incomplete metadata with (deferred) complete metadata.
    with beam.Pipeline() as pipeline:
      path = self.get_temp_dir()
      deferred_metadata = pipeline | 'CreateDeferredMetadata' >> beam.Create(
          [test_metadata.COMPLETE_METADATA])
      metadata = beam_metadata_io.BeamDatasetMetadata(
          test_metadata.INCOMPLETE_METADATA, deferred_metadata)
      _ = metadata | beam_metadata_io.WriteMetadata(path, pipeline)

    # Load from disk and check that it is as expected.
    metadata = metadata_io.read_metadata(path)
    self.assertEqual(metadata, test_metadata.COMPLETE_METADATA)


if __name__ == '__main__':
  tf.test.main()
