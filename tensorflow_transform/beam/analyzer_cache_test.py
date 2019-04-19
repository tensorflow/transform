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
"""Tests for tensorflow_transform.beam.analyzer_cache."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# GOOGLE-INITIALIZATION
import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np

import tensorflow as tf

from tensorflow_transform import analyzer_nodes
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform import test_case


class AnalyzerCacheTest(test_case.TransformTestCase):

  def test_validate_dataset_keys(self):
    analyzer_cache.validate_dataset_keys(
        {'foo', 'Foo', 'A1', 'A_1', 'A.1', 'A-1'})

    for key in {'foo 1', 'foo@1', 'foo*', 'foo[]', 'foo/goo'}:
      with self.assertRaisesRegexp(
          ValueError, 'Dataset key .* does not match allowed pattern:'):
        analyzer_cache.validate_dataset_keys({key})

  @test_case.named_parameters(
      dict(
          testcase_name='JsonNumpyCacheCoder',
          coder_cls=analyzer_nodes.JsonNumpyCacheCoder,
          value=[1, 2.5, 3, '4']),
      dict(
          testcase_name='_VocabularyAccumulatorCoder',
          coder_cls=analyzer_nodes._VocabularyAccumulatorCoder,
          value=['A', 17]),
  )
  def test_coders_round_trip(self, coder_cls, value):
    coder = coder_cls()
    encoded = coder.encode_cache(value)
    np.testing.assert_equal(value, coder.decode_cache(encoded))

  def test_cache_helpers_round_trip(self):
    base_test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    with beam.Pipeline() as p:
      cache_pcoll_dict = {
          'dataset_key_0': {
              'a': p | 'CreateA' >> beam.Create([b'[1, 2, 3]']),
              'b': p | 'CreateB' >> beam.Create([b'[5]']),
          },
          'dataset_key_1': {
              'c': p | 'CreateC' >> beam.Create([b'[9, 5, 2, 1]']),
          },
      }
      _ = cache_pcoll_dict | analyzer_cache.WriteAnalysisCacheToFS(
          base_test_dir)

    with beam.Pipeline() as p:
      read_cache = p | analyzer_cache.ReadAnalysisCacheFromFS(
          base_test_dir, list(cache_pcoll_dict.keys()))

      def assert_equal_matcher(expected_encoded):

        def _assert_equal(encoded_cache_list):
          (encode_cache,) = encoded_cache_list
          self.assertEqual(expected_encoded, encode_cache)

        return _assert_equal

      beam_test_util.assert_that(
          read_cache['dataset_key_0']['a'],
          beam_test_util.equal_to([b'[1, 2, 3]']),
          label='AssertA')
      beam_test_util.assert_that(
          read_cache['dataset_key_0']['b'],
          assert_equal_matcher(b'[5]'),
          label='AssertB')
      beam_test_util.assert_that(
          read_cache['dataset_key_1']['c'],
          assert_equal_matcher(b'[9, 5, 2, 1]'),
          label='AssertC')

  def test_cache_helpers_with_alternative_io(self):

    class LocalSink(beam.PTransform):

      def __init__(self, path, file_name_suffix):
        del file_name_suffix
        self._path = path

      def expand(self, pcoll):

        def write_to_file(value):
          tf.io.gfile.makedirs(self._path)
          with open(os.path.join(self._path, 'cache'), 'w') as f:
            f.write(value)

        return pcoll | beam.Map(write_to_file)

    test_cache_dict = {'a': {'b': [str([17, 19, 27, 31])]}}

    class LocalSource(beam.PTransform):

      def __init__(self, path):
        del path

      def expand(self, pbegin):
        return pbegin | beam.Create([test_cache_dict['a']['b']])

    cache_dir = self.get_temp_dir()
    with beam.Pipeline() as p:
      _ = test_cache_dict | analyzer_cache.WriteAnalysisCacheToFS(
          cache_dir, sink=LocalSink)

      read_cache = p | analyzer_cache.ReadAnalysisCacheFromFS(
          cache_dir, list(test_cache_dict.keys()), source=LocalSource)

      self.assertItemsEqual(read_cache.keys(), ['a'])
      self.assertItemsEqual(read_cache['a'].keys(), ['b'])

      beam_test_util.assert_that(
          read_cache['a']['b'],
          beam_test_util.equal_to([test_cache_dict['a']['b']]))


if __name__ == '__main__':
  test_case.main()
