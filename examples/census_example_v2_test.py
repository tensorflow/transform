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
import census_example_v2


class CensusExampleV2Test(tf.test.TestCase):

  def testCensusExampleAccuracy(self):
    raw_data_dir = os.path.join(os.path.dirname(__file__), 'testdata/census')
    working_dir = self.get_temp_dir()

    train_data_file = os.path.join(raw_data_dir, 'adult.data')
    test_data_file = os.path.join(raw_data_dir, 'adult.test')

    census_example_v2.transform_data(train_data_file, test_data_file,
                                     working_dir)
    results = census_example_v2.train_and_evaluate(
        working_dir, num_train_instances=1000, num_test_instances=1000)

    self.assertGreaterEqual(results[1], 0.7)

    # TODO(b/143530879) Serve the keras model.

if __name__ == '__main__':
  tf.test.main()
