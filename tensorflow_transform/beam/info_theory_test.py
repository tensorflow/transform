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


if __name__ == '__main__':
  unittest.main()
