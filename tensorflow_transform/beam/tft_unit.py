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
"""Library for testing Tensorflow Transform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

# GOOGLE-INITIALIZATION

import apache_beam as beam
from builtins import zip  # pylint: disable=redefined-builtin

import numpy as np
import six
import tensorflow as tf
import tensorflow_transform as tft

from tensorflow_transform import test_case
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io


parameters = test_case.parameters
named_parameters = test_case.named_parameters


class TransformTestCase(test_case.TransformTestCase):
  """Base test class for testing tf-transform preprocessing functions."""

  def _makeRunner(self):
    return None  # pylint: disable=unreachable

  def assertAnalyzeAndTransformResults(self,
                                       input_data,
                                       input_metadata,
                                       preprocessing_fn,
                                       expected_data=None,
                                       expected_metadata=None,
                                       expected_vocab_file_contents=None,
                                       expected_asset_file_contents=None,
                                       test_data=None,
                                       desired_batch_size=None,
                                       beam_pipeline=None,
                                       temp_dir=None):
    """Assert that input data and metadata is transformed as expected.

    This methods asserts transformed data and transformed metadata match
    with expected_data and expected_metadata.

    Args:
      input_data: A sequence of dicts whose values are
          either strings, lists of strings, numeric types or a pair of those.
      input_metadata: DatasetMetadata describing input_data.
      preprocessing_fn: A function taking a dict of tensors and returning
          a dict of tensors.
      expected_data: (optional) A dataset with the same type constraints as
          input_data, but representing the output after transformation.
          If supplied, transformed data is asserted to be equal.
      expected_metadata: (optional) DatasetMetadata describing the transformed
          data. If supplied, transformed metadata is asserted to be equal.
      expected_vocab_file_contents: (optional) A dictionary from vocab filenames
          to their expected content as a list of text lines or a list of tuples
          of frequency and text. Values should be the expected result of calling
          f.readlines() on the given asset files.
      expected_asset_file_contents: deprecated.  Use
          expected_vocab_file_contents.
      test_data: (optional) If this is provided then instead of calling
          AnalyzeAndTransformDataset with input_data, this function will call
          AnalyzeDataset with input_data and TransformDataset with test_data.
          Note that this is the case even if input_data and test_data are equal.
          test_data should also conform to input_metadata.
      desired_batch_size: (optional) A batch size to batch elements by. If not
          provided, a batch size will be computed automatically.
      beam_pipeline: (optional) A Beam Pipeline to use in this test.
      temp_dir: If set, it is used as output directory, else a new unique
          directory is created.
    Raises:
      AssertionError: if the expected data does not match the results of
          transforming input_data according to preprocessing_fn, or
          (if provided) if the expected metadata does not match.
      ValueError: if expected_vocab_file_contents and
          expected_asset_file_contents are both set.
    """
    if (expected_vocab_file_contents is not None and
        expected_asset_file_contents is not None):
      raise ValueError('only one of expected_asset_file_contents and '
                       'expected_asset_file_contents should be set')
    elif expected_asset_file_contents is not None:
      tf.logging.warn('expected_asset_file_contents is deprecated, use '
                      'expected_vocab_file_contents')

    expected_vocab_file_contents = (
        expected_vocab_file_contents or expected_asset_file_contents or {})
    del expected_asset_file_contents

    # Note: we don't separately test AnalyzeDataset and TransformDataset as
    # AnalyzeAndTransformDataset currently simply composes these two
    # transforms.  If in future versions of the code, the implementation
    # differs, we should also run AnalyzeDataset and TransformDatset composed.
    temp_dir = temp_dir or tempfile.mkdtemp(
        prefix=self._testMethodName, dir=self.get_temp_dir())
    with beam_pipeline or beam.Pipeline(runner=self._makeRunner()) as pipeline:
      with beam_impl.Context(
          temp_dir=temp_dir, desired_batch_size=desired_batch_size):
        input_data = pipeline | 'CreateInput' >> beam.Create(input_data)
        if test_data is None:
          (transformed_data, transformed_metadata), transform_fn = (
              (input_data, input_metadata)
              | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))
        else:
          transform_fn = ((input_data, input_metadata)
                          | beam_impl.AnalyzeDataset(preprocessing_fn))
          test_data = pipeline | 'CreateTest' >> beam.Create(test_data)
          transformed_data, transformed_metadata = (
              ((test_data, input_metadata), transform_fn)
              | beam_impl.TransformDataset())

        # Write transform_fn so we can test its assets
        _ = transform_fn | transform_fn_io.WriteTransformFn(temp_dir)

        if expected_data is not None:
          transformed_data_coder = tft.coders.ExampleProtoCoder(
              transformed_metadata.schema)

          transformed_data_path = os.path.join(temp_dir, 'transformed_data')
          _ = (
              transformed_data
              | beam.Map(transformed_data_coder.encode)
              | beam.io.tfrecordio.WriteToTFRecord(
                  transformed_data_path, shard_name_template=''))

    # TODO(ebreck) Log transformed_data somewhere.
    if expected_data is not None:
      examples = tf.python_io.tf_record_iterator(path=transformed_data_path)
      transformed_data = [transformed_data_coder.decode(x) for x in examples]
      self.assertDataCloseOrEqual(expected_data, transformed_data)

    tf_transform_output = tft.TFTransformOutput(temp_dir)
    if expected_metadata:
      self.assertEqual(expected_metadata,
                       tf_transform_output.transformed_metadata)

    for filename, file_contents in six.iteritems(expected_vocab_file_contents):
      full_filename = tf_transform_output.vocabulary_file_by_name(filename)
      with tf.gfile.Open(full_filename, 'rb') as f:
        file_lines = f.readlines()

        # Store frequency case.
        if isinstance(file_contents[0], tuple):
          word_and_frequency_list = []
          for content in file_lines:
            frequency, word = content.split(b' ', 1)
            word_and_frequency_list.append((word.strip(b'\n'),
                                            float(frequency.strip(b'\n'))))
          expected_words, expected_frequency = zip(*word_and_frequency_list)
          actual_words, actual_frequency = zip(*file_contents)
          self.assertAllEqual(expected_words, actual_words)
          np.testing.assert_almost_equal(expected_frequency, actual_frequency)
        else:
          file_lines = [content.strip(b'\n') for content in file_lines]
          self.assertAllEqual(file_lines, file_contents)
