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
"""Tests for sentiment_example_v2."""

import os
import shutil

import tensorflow as tf
import tensorflow_transform as tft
import sentiment_example_v2
from tensorflow_transform import test_case
import local_model_server


class SentimentExampleTest(test_case.TransformTestCase):

  def testSentimentExampleAccuracy(self):
    raw_data_dir = os.path.join(os.path.dirname(__file__), 'testdata/sentiment')
    working_dir = self.get_temp_dir()

    # Copy data from raw data directory to `working_dir`
    try:
      for filename in ['test_shuffled-00000-of-00001',
                       'train_shuffled-00000-of-00001']:
        shutil.copy(os.path.join(raw_data_dir, filename), working_dir)
    except FileNotFoundError:
      # We only use a small sample of the data for testing purposes.
      train_neg_filepattern = os.path.join(raw_data_dir, 'train/neg/10000*')
      train_pos_filepattern = os.path.join(raw_data_dir, 'train/pos/10000*')
      test_neg_filepattern = os.path.join(raw_data_dir, 'test/neg/10000*')
      test_pos_filepattern = os.path.join(raw_data_dir, 'test/pos/10000*')

      # Writes the shuffled data under working_dir in TFRecord format.
      sentiment_example_v2.read_and_shuffle_data(
          train_neg_filepattern,
          train_pos_filepattern,
          test_neg_filepattern,
          test_pos_filepattern,
          working_dir,
      )

    sentiment_example_v2.transform_data(working_dir)
    # TODO: b/323209255 - Remove this if clause once TF pulls the latest keras
    # nightly version.
    if not test_case.is_external_environment():
      model_path = os.path.join(
          working_dir, sentiment_example_v2.EXPORTED_MODEL_DIR
      )
      results = sentiment_example_v2.train_and_evaluate(
          working_dir,
          model_path,
          num_train_instances=1000,
          num_test_instances=1000,
      )
    if not test_case.is_external_environment():
      # Assert expected accuracy.
      self.assertGreaterEqual(results[1], 0.7)

      # Delete temp directory and transform_fn directory.  This ensures that the
      # test of serving the model below will only pass if the SavedModel saved
      # to sentiment_example_v2.EXPORTED_MODEL_DIR is hermetic, i.e does not
      # contain references to tft_temp and transform_fn.
      shutil.rmtree(
          os.path.join(working_dir, sentiment_example_v2.TRANSFORM_TEMP_DIR)
      )
      shutil.rmtree(
          os.path.join(working_dir, tft.TFTransformOutput.TRANSFORM_FN_DIR))

      if local_model_server.local_model_server_supported():
        model_name = 'my_model'
        with local_model_server.start_server(model_name, model_path) as address:
          # Use made up data chosen to give high probability of negative
          # sentiment.
          ascii_classification_request = """model_spec { name: "my_model" }
  input {
    example_list {
      examples {
        features {
          feature {
            key: "review"
            value: {
              bytes_list {
                value: "errible terrible terrible terrible terrible terrible terrible."
              }
            }
          }
        }
      }
    }
  }"""
          results = local_model_server.make_classification_request(
              address, ascii_classification_request)
          self.assertLen(results, 1)
          self.assertLen(results[0].classes, 2)
          self.assertEqual(results[0].classes[0].label, '0')
          self.assertGreater(results[0].classes[0].score, 0.8)
          self.assertEqual(results[0].classes[1].label, '1')
          self.assertLess(results[0].classes[1].score, 0.2)


if __name__ == '__main__':
  tf.test.main()
