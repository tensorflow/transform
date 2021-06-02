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

import json
import os

import apache_beam as beam
import tensorflow as tf
from tensorflow_transform import output_wrapper
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.beam.tft_beam_io import test_metadata
import tensorflow_transform.test_case as tft_test_case
from tensorflow_transform.tf_metadata import metadata_io

mock = tf.compat.v1.test.mock


class BeamMetadataIoTest(tft_unit.TransformTestCase):

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
    expected_asset_map = {'key': 'value'}
    with beam.Pipeline() as pipeline:
      path = self.get_temp_dir()
      deferred_metadata = pipeline | 'CreateDeferredMetadata' >> beam.Create(
          [test_metadata.COMPLETE_METADATA])
      metadata = beam_metadata_io.BeamDatasetMetadata(
          test_metadata.INCOMPLETE_METADATA, deferred_metadata,
          expected_asset_map)
      _ = metadata | beam_metadata_io.WriteMetadata(path, pipeline)

    # Load from disk and check that it is as expected.
    metadata = metadata_io.read_metadata(path)
    self.assertEqual(metadata, test_metadata.COMPLETE_METADATA)

    with tf.io.gfile.GFile(
        os.path.join(path, output_wrapper.TFTransformOutput.ASSET_MAP)) as f:
      asset_map = json.loads(f.read())
      self.assertDictEqual(asset_map, expected_asset_map)

  def testWriteMetadataIsRetryable(self):
    tft_test_case.skip_if_external_environment(
        'Retries are currently not available on this environment.')
    original_write_metadata = beam_metadata_io.metadata_io.write_metadata
    write_metadata_called_list = []

    def mock_write_metadata(metadata, path):
      """Mocks metadata_io.write_metadata to fail the first time it is called by this test, thus forcing a retry which should succeed."""
      if not write_metadata_called_list:
        write_metadata_called_list.append(True)
        original_write_metadata(metadata, path)
        raise ArithmeticError('Some error')
      return original_write_metadata(metadata, path)

    # Write metadata to disk using WriteMetadata PTransform.
    with mock.patch(
        'tensorflow_transform.tf_metadata.metadata_io.write_metadata',
        mock_write_metadata):
      with self._makeTestPipeline() as pipeline:
        path = self.get_temp_dir()
        _ = (
            test_metadata.COMPLETE_METADATA
            | beam_metadata_io.WriteMetadata(path, pipeline))

      # Load from disk and check that it is as expected.
      metadata = metadata_io.read_metadata(path)
      self.assertEqual(metadata, test_metadata.COMPLETE_METADATA)


if __name__ == '__main__':
  tf.test.main()
