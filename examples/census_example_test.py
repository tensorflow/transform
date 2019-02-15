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
"""Tests for census_example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# GOOGLE-INITIALIZATION

import tensorflow as tf
import census_example
import local_model_server


class CensusExampleTest(tf.test.TestCase):

  def testCensusExampleAccuracy(self):
    raw_data_dir = os.path.join(os.path.dirname(__file__), 'testdata/census')
    working_dir = self.get_temp_dir()

    train_data_file = os.path.join(raw_data_dir, 'adult.data')
    test_data_file = os.path.join(raw_data_dir, 'adult.test')

    census_example.transform_data(train_data_file, test_data_file, working_dir)
    results = census_example.train_and_evaluate(
        working_dir, num_train_instances=1000, num_test_instances=1000)
    self.assertGreaterEqual(results['accuracy'], 0.7)

    if local_model_server.local_model_server_supported():
      model_name = 'my_model'
      model_path = os.path.join(working_dir, census_example.EXPORTED_MODEL_DIR)
      with local_model_server.start_server(model_name, model_path) as address:
        # Use first row of test data set, which has high probability on label 1
        # (which corresponds to '<=50K').
        ascii_classification_request = """model_spec { name: "my_model" }
input {
  example_list {
    examples {
      features {
        feature {
          key: "age"
          value { float_list: { value: 25 } }
        }
        feature {
          key: "workclass"
          value { bytes_list: { value: "Private" } }
        }
        feature {
          key: "education"
          value { bytes_list: { value: "11th" } }
        }
        feature {
          key: "education-num"
          value { float_list: { value: 7 } }
        }
        feature {
          key: "marital-status"
          value { bytes_list: { value: "Never-married" } }
        }
        feature {
          key: "occupation"
          value { bytes_list: { value: "Machine-op-inspct" } }
        }
        feature {
          key: "relationship"
          value { bytes_list: { value: "Own-child" } }
        }
        feature {
          key: "race"
          value { bytes_list: { value: "Black" } }
        }
        feature {
          key: "sex"
          value { bytes_list: { value: "Male" } }
        }
        feature {
          key: "capital-gain"
          value { float_list: { value: 0 } }
        }
        feature {
          key: "capital-loss"
          value { float_list: { value: 0 } }
        }
        feature {
          key: "hours-per-week"
          value { float_list: { value: 40 } }
        }
        feature {
          key: "native-country"
          value { bytes_list: { value: "United-States" } }
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
        self.assertLess(results[0].classes[0].score, 0.01)
        self.assertEqual(results[0].classes[1].label, '1')
        self.assertGreater(results[0].classes[1].score, 0.99)


if __name__ == '__main__':
  tf.test.main()
