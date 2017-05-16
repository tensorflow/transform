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
"""Tests for tensorflow_transform.beam.analyzer_impls."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-import-not-at-top
import apache_beam as beam
try:
  from apache_beam.testing.util import assert_that
  from apache_beam.testing.util import equal_to
except ImportError:
  from apache_beam.transforms.util import assert_that
  from apache_beam.transforms.util import equal_to


import numpy as np
from tensorflow_transform.beam import analyzer_impls as impl
from tensorflow_transform.beam import impl as beam_impl


import unittest
from tensorflow.python.framework import test_util
# pylint: enable=g-import-not-at-top



class AnalyzerImplsTest(test_util.TensorFlowTestCase):

  def assertCombine(self, combine_fn, shards, expected):
    """Tests the provided combiner.

    Args:
      combine_fn: A beam.ComineFn to exercise.
      shards: A list of next_inputs to add via the combiner.
      expected: The expected output from extract_output.

    Exercises create_accumulator, add_input, merge_accumulators,
    and extract_output.
    """
    accumulators = [
        combine_fn.add_input(combine_fn.create_accumulator(), shard)
        for shard in shards]
    final_accumulator = combine_fn.merge_accumulators(accumulators)
    extracted = combine_fn.extract_output(final_accumulator)
    self.assertAllEqual(expected, extracted)

  def testCombineOnBatchSimple(self):
    lst_1 = [np.ones(6), np.ones(6)]
    lst_2 = [np.ones(6)]
    # pylint: disable=unused-variable
    out = [3 for i in range(6)]
    analyzer = impl._NumericCombineAnalyzerImpl._CombineOnBatchDim(np.sum)
    self.assertCombine(analyzer, [lst_1, lst_2], out)

  def testCombineOnBatchAllEmptyRow(self):
    analyzer = impl._NumericCombineAnalyzerImpl._CombineOnBatchDim(np.sum)
    self.assertCombine(analyzer, [[[]], [[]], [[]]], [])

  def testCombineOnBatchLotsOfData(self):
    # pylint: disable=unused-variable
    shards = [[np.ones(3)] for i in range(
        beam_impl._DEFAULT_DESIRED_BATCH_SIZE * 2)]
    out = [1 for i in range(3)]
    analyzer = impl._NumericCombineAnalyzerImpl._CombineOnBatchDim(np.min)
    self.assertCombine(analyzer, shards, out)


if __name__ == '__main__':
  unittest.main()
