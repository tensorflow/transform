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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow_transform import test_case

import unittest


class TftUnitTest(test_case.TransformTestCase):

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
