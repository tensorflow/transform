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

import apache_beam as beam

import numpy as np
import tensorflow as tf
from tensorflow_transform.beam import analyzer_impls
from tensorflow_transform.beam import tft_unit


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
        analyzer_impls._merge_outputs_by_key,
        outputs_dtype=[tf.int64, tf.int64]).with_outputs('key', '0', '1'))
    self.assertAllEqual(merged_outputs_pcolls[0][0],
                        np.array(['my_key', 'my_other_key']))
    self.assertAllEqual(merged_outputs_pcolls[1][0],
                        np.array([20, 23]))
    self.assertAllEqual(merged_outputs_pcolls[2][0],
                        np.array([[21, 22], [24, 25]]))

  def testMergeOutputsByKeyEmptyInput(self):
    outputs = []
    outputs_pcoll = [outputs]
    merged_outputs_pcolls = tuple(outputs_pcoll | beam.FlatMap(
        analyzer_impls._merge_outputs_by_key,
        outputs_dtype=[tf.float32, tf.float32]).with_outputs('key', '0', '1'))
    self.assertAllEqual(merged_outputs_pcolls[0][0],
                        np.array([]))
    self.assertAllEqual(merged_outputs_pcolls[1][0], np.array([]))
    self.assertAllEqual(merged_outputs_pcolls[2][0], np.array([]))

  @tft_unit.named_parameters(
      dict(
          testcase_name='Increasing',
          input_boundaries=np.array([[1, 1.00000001], [1, 2]]),
          expected_boundaries=np.array([[1, 1.00000001], [1, 2]])),
      dict(
          testcase_name='Repeating',
          input_boundaries=np.array([[1, 1, 1], [4, 4, 4]]),
          expected_boundaries=np.array([[1, 1.000001, 1.000002],
                                        [4, 4.000001, 4.000002]])),
      dict(
          testcase_name='NonIncreasing',
          input_boundaries=np.array([[3, 5.1, 5.1], [4.01, 4.01, 4.2]]),
          expected_boundaries=np.array([[3, 5.1, 5.1000021],
                                        [4.01, 4.01000019, 4.20000019]]),
          atol=1e-6),
  )
  def testMakeStrictlyIncreasingBoundariesRows(self,
                                               input_boundaries,
                                               expected_boundaries,
                                               atol=None):
    result = analyzer_impls._make_strictly_increasing_boundaries_rows(
        input_boundaries)
    if atol is None:
      self.assertAllEqual(result, expected_boundaries)
    else:
      self.assertAllClose(result, expected_boundaries, atol=atol)

  @tft_unit.named_parameters(
      dict(
          testcase_name='Simple',
          input_boundaries=np.array([[0, 1, 2], [0, 1, 2]]),
          expected_boundaries=np.array([0, 0.5, 1, 1.5, 2]),
          expected_scales=np.array([0.5, 0.5]),
          expected_shifts=np.array([0, 1]),
          expected_num_buckets=np.array(4)),
      dict(
          testcase_name='Complex',
          input_boundaries=np.array([[0, 1, 2, 3], [3, 3, 3, 3], [2, 4, 6, 8]]),
          expected_boundaries=np.array([
              0, 0.33333333, 0.66666667, 1, 1.33333333, 1.66666667, 2,
              2.33333333, 2.66666667, 3
          ]),
          expected_scales=np.array([0.333333333, 333333.333, 0.166666667]),
          expected_shifts=np.array([0, -999999, 1.66666667]),
          expected_num_buckets=np.array(5)),
      dict(
          testcase_name='SingleBoundary',
          input_boundaries=np.array([[1], [2]]),
          expected_boundaries=np.array([0]),
          expected_scales=np.array([1., 1.]),
          expected_shifts=np.array([-1, -1]),
          expected_num_buckets=np.array(2)),
  )
  def testJoinBoundarieRows(self, input_boundaries, expected_boundaries,
                            expected_scales, expected_shifts,
                            expected_num_buckets):
    boundaries, scales, shifts, num_buckets = (
        analyzer_impls._join_boundary_rows(input_boundaries))
    self.assertAllClose(boundaries, expected_boundaries)
    self.assertAllClose(scales, expected_scales)
    self.assertAllClose(shifts, expected_shifts)
    self.assertAllEqual(num_buckets, expected_num_buckets)


if __name__ == '__main__':
  tft_unit.main()
