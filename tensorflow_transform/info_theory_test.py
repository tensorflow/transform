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
"""Tests for tensorflow_transform.info_theory."""

from tensorflow_transform import info_theory
from tensorflow_transform import test_case


import unittest


EPSILON = 1e-4


def _make_hypergeometric_pmf_sum_up_to_one_parameters():
  start = 1000
  end = 10000
  range_length = end - start
  num_chunks = 15
  assert range_length % num_chunks == 0
  chunk_size = int(range_length / num_chunks)
  sub_ranges = [(x, x + chunk_size) for x in range(start, end, chunk_size)]
  return [  # pylint: disable=g-complex-comprehension
      dict(
          testcase_name='{}_to_{}'.format(a, b),
          test_range=range(a, b),
          n=end,
          y_j=start) for a, b in sub_ranges
  ]


class InfoTheoryTest(test_case.TransformTestCase):

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

  @test_case.named_parameters(
      *_make_hypergeometric_pmf_sum_up_to_one_parameters())
  def test_hypergeometric_pmf_sum_up_to_one(self, test_range, n, y_j):
    for x in test_range:
      probs = [prob for _, prob in info_theory._hypergeometric_pmf(n, x, y_j)]
      sum_prob = sum(probs)
      self.assertNear(sum_prob, 1.0, EPSILON)

  @test_case.named_parameters(
      dict(
          testcase_name='all_co_occur',
          n=10,
          x_i=10,
          y_j=10,
          expected=0,
      ),
      dict(
          testcase_name='2_co_occur_no_observations',
          n=10,
          x_i=0,
          y_j=0,
          expected=0,
      ),
      dict(
          testcase_name='2_values_appear_half_the_time',
          n=10,
          x_i=5,
          y_j=5,
          expected=0.215411,
      ),
      dict(
          testcase_name='2_values_differing_frequencies',
          n=10,
          x_i=2,
          y_j=4,
          expected=0.524209,
      ),
  )
  def test_calculate_partial_expected_mutual_information(
      self, n, x_i, y_j, expected):
    self.assertNear(
        info_theory.calculate_partial_expected_mutual_information(n, x_i, y_j),
        expected, EPSILON)

  @test_case.named_parameters(
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
