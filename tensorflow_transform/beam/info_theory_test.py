# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Tests for tensorflow_transform.beam.info_theory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# GOOGLE-INITIALIZATION

from tensorflow_transform.beam import info_theory
from tensorflow_transform.beam import tft_unit


import unittest


EPSILON = 1e-4


class InfoTheoryTest(tft_unit.TransformTestCase):

  def testHypergeometricPmf(self):
    expected_results = [(0, 0.75), (1, 0.25)]
    results = list(info_theory._hypergeometric_pmf(4, 1, 1))
    for expected_result, result in zip(expected_results, results):
      self.assertEqual(expected_result[0], result[0])
      self.assertNear(expected_result[1], result[1], EPSILON)

  def testHypergeometricPmf_LargeN(self):
    expected_results = [(0, 0.9508937), (1, 0.0482198), (2, 0.0008794),
                        (3, 7.1e-06), (4, 2.5e-08), (5, 0.0)]
    results = list(info_theory._hypergeometric_pmf(1000, 5, 10))
    for expected_result, result in zip(expected_results, results):
      self.assertEqual(expected_result[0], result[0])
      self.assertNear(expected_result[1], result[1], EPSILON)

  def testHypergeometricPmf_SumUpToOne(self):
    for x in range(1000, 10000):
      probs = [
          prob for _, prob in info_theory._hypergeometric_pmf(10000, x, 1000)
      ]
      sum_prob = sum(probs)
      self.assertNear(sum_prob, 1.0, EPSILON)

  def testCalculatePartialExpectedMutualInformation(self):

    # The two values co-occur in all observations, EMI is 0.
    self.assertNear(
        info_theory.calculate_partial_expected_mutual_information(10, 10, 10),
        0, EPSILON)

    # The two values co-occur no observations, EMI is 0
    self.assertNear(
        info_theory.calculate_partial_expected_mutual_information(10, 0, 0), 0,
        EPSILON)

    # The two values each appear 50% of the time.
    self.assertNear(
        info_theory.calculate_partial_expected_mutual_information(10, 5, 5),
        .215411, EPSILON)

    # The two values have differing frequencies.
    self.assertNear(
        info_theory.calculate_partial_expected_mutual_information(10, 2, 4),
        0.524209, EPSILON)

  @tft_unit.named_parameters(
      dict(
          testcase_name='strongly_positive_mi',
          cell_count=2,
          row_count=10,
          col_count=2,
          total_count=14,
          expected_mi=0.970854),
      dict(
          testcase_name='weakly_positive_mi',
          cell_count=4,
          row_count=15,
          col_count=6,
          total_count=25,
          expected_mi=0.608012),
      dict(
          testcase_name='strongly_negative_mi',
          cell_count=2,
          row_count=10,
          col_count=6,
          total_count=25,
          expected_mi=-0.526069),
      dict(
          testcase_name='weakly_negative_mi',
          cell_count=3,
          row_count=31,
          col_count=4,
          total_count=41,
          expected_mi=-0.0350454),
      dict(
          testcase_name='zero_mi',
          cell_count=4,
          row_count=8,
          col_count=8,
          total_count=16,
          expected_mi=0),
  )
  def test_mutual_information(self, cell_count, row_count, col_count,
                              total_count, expected_mi):
    per_cell_mi = info_theory.calculate_partial_mutual_information(
        cell_count, row_count, col_count, total_count)
    self.assertNear(per_cell_mi, expected_mi, EPSILON)


if __name__ == '__main__':
  unittest.main()
