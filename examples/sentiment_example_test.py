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
"""Tests for sentiment_example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

# GOOGLE-INITIALIZATION

import tensorflow as tf
import sentiment_example
import local_model_server


class SentimentExampleTest(tf.test.TestCase):

  def testSentimentExampleAccuracy(self):
    raw_data_dir = os.path.join(os.path.dirname(__file__), 'testdata/sentiment')
    working_dir = self.get_temp_dir()

    # Copy data from raw data directory to `working_dir`
    for filename in ['test_shuffled-00000-of-00001',
                     'train_shuffled-00000-of-00001']:
      shutil.copy(os.path.join(raw_data_dir, filename), working_dir)

    sentiment_example.transform_data(working_dir)
    results = sentiment_example.train_and_evaluate(
        working_dir, num_train_instances=1000, num_test_instances=1000)
    self.assertGreaterEqual(results['accuracy'], 0.7)

    if local_model_server.local_model_server_supported():
      model_name = 'my_model'
      model_path = os.path.join(working_dir,
                                sentiment_example.EXPORTED_MODEL_DIR)
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
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].classes), 2)
        self.assertEqual(results[0].classes[0].label, '0')
        self.assertGreater(results[0].classes[0].score, 0.8)
        self.assertEqual(results[0].classes[1].label, '1')
        self.assertLess(results[0].classes[1].score, 0.2)


if __name__ == '__main__':
  tf.test.main()
