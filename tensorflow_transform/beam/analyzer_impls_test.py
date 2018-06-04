# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tests for tensorflow_transform.beam.analyzer_impls."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import apache_beam as beam

import numpy as np
from tensorflow_transform.beam import analyzer_impls
from tensorflow_transform.beam import tft_unit

import unittest


class AnalyzerImplsTest(tft_unit.TransformTestCase):

  def testSplitInputsByKey(self):
    inputs = [
        np.array(['my_key', 'my_other_key']),
        np.array([[1, 2], [3, 4]]),
        np.array([5, 6])
    ]
    split_inputs = list(analyzer_impls._split_inputs_by_key(inputs))
    self.assertEqual(len(split_inputs), 2)

    self.assertEqual(len(split_inputs[0]), 2)
    self.assertEqual(split_inputs[0][0], 'my_key')
    self.assertEqual(len(split_inputs[0][1]), 2)
    self.assertAllEqual(split_inputs[0][1][0], np.array([1, 2]))
    self.assertAllEqual(split_inputs[0][1][1], np.array(5))

    self.assertEqual(len(split_inputs[1]), 2)
    self.assertEqual(split_inputs[1][0], 'my_other_key')
    self.assertEqual(len(split_inputs[1][1]), 2)
    self.assertAllEqual(split_inputs[1][1][0], np.array([3, 4]))
    self.assertAllEqual(split_inputs[1][1][1], np.array(6))

  def testMergeOutputsByKey(self):
    outputs = [
        ('my_key', [np.array(20), np.array([21, 22])]),
        ('my_other_key', [np.array(23), np.array([24, 25])])
    ]
    outputs_pcoll = [outputs]
    merged_outputs_pcolls = tuple(outputs_pcoll | beam.FlatMap(
        analyzer_impls._merge_outputs_by_key, num_outputs=2).with_outputs(
            'key', '0', '1'))
    self.assertAllEqual(merged_outputs_pcolls[0][0],
                        np.array(['my_key', 'my_other_key']))
    self.assertAllEqual(merged_outputs_pcolls[1][0],
                        np.array([20, 23]))
    self.assertAllEqual(merged_outputs_pcolls[2][0],
                        np.array([[21, 22], [24, 25]]))


if __name__ == '__main__':
  unittest.main()
