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
"""Tests for tensorflow_transform.test_case."""

import re

from tensorflow_transform import test_case

import unittest


class TftUnitTest(test_case.TransformTestCase):

  def testCrossNamedParameters(self):
    test_cases_1 = [
        {'testcase_name': 'a_1_b_1', 'a': 1, 'b': 1},
        {'testcase_name': 'a_3_b_3', 'a': 3, 'b': 3},
    ]
    test_cases_2 = [
        {'testcase_name': 'c_2', 'c': 2},
        {'testcase_name': 'c_4', 'c': 4},
    ]
    expected_cross = [
        {'testcase_name': 'a_1_b_1_c_2', 'a': 1, 'b': 1, 'c': 2},
        {'testcase_name': 'a_1_b_1_c_4', 'a': 1, 'b': 1, 'c': 4},
        {'testcase_name': 'a_3_b_3_c_2', 'a': 3, 'b': 3, 'c': 2},
        {'testcase_name': 'a_3_b_3_c_4', 'a': 3, 'b': 3, 'c': 4},
    ]
    self.assertEqual(
        test_case.cross_named_parameters(test_cases_1, test_cases_2),
        expected_cross)

  def testCrossParameters(self):
    test_cases_1 = [('a', 1), ('b', 2)]
    test_cases_2 = [(True,), (False,)]
    expected_cross = [
        ('a', 1, True), ('b', 2, True),
        ('a', 1, False), ('b', 2, False),
    ]
    self.assertCountEqual(
        test_case.cross_parameters(test_cases_1, test_cases_2), expected_cross)

  def testAssertDataCloseOrEqual(self):
    self.assertDataCloseOrEqual([{'a': 'first',
                                  'b': 1.0,
                                  'c': 5,
                                  'd': ('second', 2.0)},
                                 {'e': 2,
                                  'f': 3}],
                                [{'a': 'first',
                                  'b': 1.0000001,
                                  'c': 5,
                                  'd': ('second', 2.0000001)},
                                 {'e': 2,
                                  'f': 3}])
    with self.assertRaisesRegexp(AssertionError, r'len\(.*\) != len\(\[\]\)'):
      self.assertDataCloseOrEqual([{'a': 1}], [])
    with self.assertRaisesRegexp(
        AssertionError,
        re.compile('Element counts were not equal.*: Row 0', re.DOTALL)):
      self.assertDataCloseOrEqual([{'a': 1}], [{'b': 1}])
    with self.assertRaisesRegexp(
        AssertionError,
        re.compile('Not equal to tolerance.*: Row 0, key a', re.DOTALL)):
      self.assertDataCloseOrEqual([{'a': 1}], [{'a': 2}])

  @test_case.parameters((1, 'a'), (2, 'b'))
  def testSampleParametrizedTestMethod(self, my_arg, my_other_arg):
    self.assertIn((my_arg, my_other_arg), {(1, 'a'), (2, 'b')})


if __name__ == '__main__':
  unittest.main()
