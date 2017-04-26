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
"""Tests for tensorflow_transform.mappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow_transform import mappers

import unittest
from tensorflow.python.framework import test_util


class MappersTest(test_util.TensorFlowTestCase):

  def testSegmentIndices(self):
    with tf.Session():
      self.assertAllEqual(
          mappers.segment_indices(tf.constant([0, 0, 1, 2, 2, 2],
                                              tf.int64)).eval(),
          [0, 1, 0, 0, 1, 2])
      self.assertAllEqual(
          mappers.segment_indices(tf.constant([], tf.int64)).eval(),
          [])

  def testNGrams(self):
    output_tensor = mappers.ngrams(
        tf.constant(['abc', 'def', 'fghijklm', 'z', '']), (1, 5))
    with tf.Session():
      output = output_tensor.eval()
      self.assertAllEqual(
          output.indices,
          [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
           [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
           [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7],
           [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15],
           [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21], [2, 22],
           [2, 23], [2, 24], [2, 25], [2, 26], [2, 27], [2, 28], [2, 29],
           [3, 0]])
      self.assertAllEqual(output.values, [
          'a', 'ab', 'abc', 'b', 'bc', 'c',
          'd', 'de', 'def', 'e', 'ef', 'f',
          'f', 'fg', 'fgh', 'fghi', 'fghij', 'g', 'gh', 'ghi', 'ghij', 'ghijk',
          'h', 'hi', 'hij', 'hijk', 'hijkl', 'i', 'ij', 'ijk', 'ijkl', 'ijklm',
          'j', 'jk', 'jkl', 'jklm', 'k', 'kl', 'klm', 'l', 'lm', 'm',
          'z'])
      self.assertAllEqual(output.dense_shape, [5, 30])

  def testNGramsMinSizeNotOne(self):
    output_tensor = mappers.ngrams(
        tf.constant(['abc', 'def', 'fghijklm', 'z', '']), (2, 5))
    with tf.Session():
      output = output_tensor.eval()
      self.assertAllEqual(
          output.indices,
          [[0, 0], [0, 1], [0, 2],
           [1, 0], [1, 1], [1, 2],
           [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7],
           [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15],
           [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21]])
      self.assertAllEqual(output.values, [
          'ab', 'abc', 'bc',
          'de', 'def', 'ef',
          'fg', 'fgh', 'fghi', 'fghij', 'gh', 'ghi', 'ghij', 'ghijk',
          'hi', 'hij', 'hijk', 'hijkl', 'ij', 'ijk', 'ijkl', 'ijklm',
          'jk', 'jkl', 'jklm', 'kl', 'klm', 'lm'])
      self.assertAllEqual(output.dense_shape, [5, 22])

  def testNGramsBadSizes(self):
    with self.assertRaisesRegexp(ValueError, 'Invalid ngram_range'):
      mappers.ngrams(tf.constant(['abc', 'def', 'fghijklm', 'z', '']), (0, 5))
    with self.assertRaisesRegexp(ValueError, 'Invalid ngram_range'):
      mappers.ngrams(tf.constant(['abc', 'def', 'fghijklm', 'z', '']), (6, 5))


if __name__ == '__main__':
  unittest.main()
