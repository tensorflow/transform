# coding=utf-8
#
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

import os

import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np

import tensorflow as tf

from tensorflow_transform import analyzer_nodes
from tensorflow_transform import analyzers
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform import test_case

mock = tf.compat.v1.test.mock


def _get_quantiles_accumulator():

  qcombiner = analyzers.QuantilesCombiner(
      num_quantiles=2,
      epsilon=0.01,
      bucket_numpy_dtype=np.float32,
      has_weights=False,
      output_shape=None,
      include_max_and_min=False,
      feature_shape=[1])
  accumulator = qcombiner.create_accumulator()
  return qcombiner.add_input(accumulator, [np.array([1.0, 2.0, 3.0])])


class AnalyzerCacheTest(test_case.TransformTestCase):

  def test_validate_dataset_keys(self):
    analyzer_cache.validate_dataset_keys({
        analyzer_cache.DatasetKey(k)
        for k in ('foo', 'Foo', 'A1', 'A_1', 'A.1', 'A-1', 'foo@1', 'foo*',
                  'foo[]', 'foo/goo')
    })

    for key in {analyzer_cache.DatasetKey(k) for k in ('^foo^', 'foo 1')}:
      with self.assertRaisesRegexp(
          ValueError, 'Dataset key .* does not match allowed pattern:'):
        analyzer_cache.validate_dataset_keys({key})

  @test_case.named_parameters(
      dict(
          testcase_name='JsonNumpyCacheCoder',
          coder=analyzer_nodes.JsonNumpyCacheCoder(),
          value=[1, 2.5, 3, '4']),
      dict(
          testcase_name='JsonNumpyCacheCoderNpArray',
          coder=analyzer_nodes.JsonNumpyCacheCoder(),
          value=np.array([1, 2.5, 3, '4'])),
      dict(
          testcase_name='JsonNumpyCacheCoderNestedNpTypes',
          coder=analyzer_nodes.JsonNumpyCacheCoder(),
          value=[np.int64(1), np.float32(2.5), 3, '4']),
      dict(
          testcase_name='_VocabularyAccumulatorCoderIntAccumulator',
          coder=analyzer_nodes._VocabularyAccumulatorCoder(),
          value=[b'A', 17]),
      dict(
          testcase_name='_VocabularyAccumulatorCoderIntAccumulatorNonUtf8',
          coder=analyzer_nodes._VocabularyAccumulatorCoder(),
          value=[b'\x8a', 29]),
      dict(
          testcase_name='_VocabularyAccumulatorCoderClassAccumulator',
          coder=analyzer_nodes._VocabularyAccumulatorCoder(),
          value=[
              b'A',
              analyzers._WeightedMeanAndVarAccumulator(
                  count=np.array(5),
                  mean=np.array([.4, .9, 1.5]),
                  variance=np.array([.1, .4, .5]),
                  weight=np.array(0.),
              )
          ]),
      dict(
          testcase_name='_QuantilesAccumulatorCoderClassAccumulator',
          coder=analyzers._QuantilesSketchCacheCoder(),
          value=_get_quantiles_accumulator()),
      dict(
          testcase_name='_CombinerPerKeyAccumulatorCoder',
          coder=analyzer_nodes._CombinerPerKeyAccumulatorCoder(
              analyzer_nodes.JsonNumpyCacheCoder()),
          value=[b'\x8a', [np.int64(1), np.float32(2.5), 3, '4']]),
  )
  def test_coders_round_trip(self, coder, value):
    encoded = coder.encode_cache(value)
    if isinstance(coder, analyzers._QuantilesSketchCacheCoder):
      # Quantiles accumulator becomes a different object after pickle round trip
      # and doesn't have a deep __eq__ defined. That's why we compare the output
      # of accumulator before and after pickling.
      np.testing.assert_equal(
          coder.decode_cache(encoded).GetQuantiles(10).to_pylist(),
          value.GetQuantiles(10).to_pylist())
    else:
      np.testing.assert_equal(coder.decode_cache(encoded), value)

  def test_cache_helpers_round_trip(self):
    base_test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    dataset_key_0 = analyzer_cache.DatasetKey('dataset_key_0')
    dataset_key_1 = analyzer_cache.DatasetKey('dataset_key_1')
    dataset_keys = (dataset_key_0, dataset_key_1)

    with beam.Pipeline() as p:
      cache_pcoll_dict = {
          dataset_key_0: {
              b'\x8a': p | 'CreateA' >> beam.Create([b'[1, 2, 3]']),
              b'\x8b': p | 'CreateB' >> beam.Create([b'[5]']),
              b'\x8b1': p | 'CreateB1' >> beam.Create([b'[6]']),
          },
          dataset_key_1: {
              b'\x8c': p | 'CreateC' >> beam.Create([b'[9, 5, 2, 1]']),
          },
      }

      _ = cache_pcoll_dict | analyzer_cache.WriteAnalysisCacheToFS(
          p, base_test_dir, dataset_keys)

    with beam.Pipeline() as p:
      read_cache = p | analyzer_cache.ReadAnalysisCacheFromFS(
          base_test_dir, list(cache_pcoll_dict.keys()),
          [b'\x8a', b'\x8b', b'\x8c'])

      beam_test_util.assert_that(
          read_cache[dataset_key_0][b'\x8a'],
          beam_test_util.equal_to([b'[1, 2, 3]']),
          label='AssertA')
      beam_test_util.assert_that(
          read_cache[dataset_key_0][b'\x8b'],
          beam_test_util.equal_to([b'[5]']),
          label='AssertB')
      beam_test_util.assert_that(
          read_cache[dataset_key_1][b'\x8c'],
          beam_test_util.equal_to([b'[9, 5, 2, 1]']),
          label='AssertC')

  def test_cache_write_empty(self):
    base_test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    with beam.Pipeline() as p:
      _ = {} | analyzer_cache.WriteAnalysisCacheToFS(
          p, base_test_dir, (analyzer_cache.DatasetKey('dataset_key_0'),))
    self.assertFalse(os.path.isdir(base_test_dir))

  def test_cache_merge(self):
    base_test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    dataset_key_0 = analyzer_cache.DatasetKey('dataset_key_0')
    dataset_key_1 = analyzer_cache.DatasetKey('dataset_key_1')
    dataset_keys = (dataset_key_0, dataset_key_1)
    cache_keys = list('abcd')

    def read_manifests():
      return [
          analyzer_cache._ManifestFile(
              analyzer_cache._get_dataset_cache_path(base_test_dir, key)).read()
          for key in dataset_keys
      ]

    with beam.Pipeline() as p:
      cache_pcoll_dict = {
          dataset_key_0: {
              'a': p | 'CreateA' >> beam.Create([b'a']),
              'b': p | 'CreateB' >> beam.Create([b'b']),
          },
          dataset_key_1: {
              'c': p | 'CreateC' >> beam.Create([b'c']),
              'd': p | 'CreateD' >> beam.Create([b'd']),
          },
      }
      _ = cache_pcoll_dict | analyzer_cache.WriteAnalysisCacheToFS(
          p, base_test_dir, dataset_keys)

    first_manifests = read_manifests()

    with beam.Pipeline() as p:
      cache_pcoll_dict = {
          dataset_key_0: {
              'c': p | 'CreateC' >> beam.Create([b'c']),
              'd': p | 'CreateD' >> beam.Create([b'd']),
          },
          dataset_key_1: {
              'a': p | 'CreateA' >> beam.Create([b'a']),
              'b': p | 'CreateB' >> beam.Create([b'b']),
          },
      }
      _ = cache_pcoll_dict | analyzer_cache.WriteAnalysisCacheToFS(
          p, base_test_dir, dataset_keys)

    second_manifests = read_manifests()
    self.assertEqual(len(first_manifests), len(second_manifests))
    for manifest_a, manifest_b in zip(first_manifests, second_manifests):
      for key_value_pair in manifest_a.items():
        self.assertIn(key_value_pair, manifest_b.items())

      self.assertEqual(2, len(manifest_a))
      self.assertCountEqual(range(len(manifest_a)), manifest_a.values())

      self.assertEqual(4, len(manifest_b))
      self.assertCountEqual(range(len(manifest_b)), manifest_b.values())
      self.assertCountEqual(cache_keys, manifest_b.keys())

  def test_cache_helpers_with_alternative_io(self):

    class LocalSink(beam.PTransform):

      def __init__(self, path):
        self._path = path

      def expand(self, pcoll):

        def write_to_file(value):
          tf.io.gfile.makedirs(self._path)
          with open(os.path.join(self._path, 'cache'), 'wb') as f:
            f.write(value)

        return pcoll | beam.Map(write_to_file)

    test_cache_dict = {
        analyzer_cache.DatasetKey('a'): {
            'b': [bytes([17, 19, 27, 31])]
        }
    }

    class LocalSource(beam.PTransform):

      def __init__(self, path):
        del path

      def expand(self, pbegin):
        return pbegin | beam.Create([test_cache_dict['a']['b']])

    dataset_keys = list(test_cache_dict.keys())
    cache_dir = self.get_temp_dir()
    with beam.Pipeline() as p:
      _ = test_cache_dict | analyzer_cache.WriteAnalysisCacheToFS(
          p, cache_dir, dataset_keys, sink=LocalSink)

      read_cache = p | analyzer_cache.ReadAnalysisCacheFromFS(
          cache_dir, dataset_keys, source=LocalSource)

      self.assertItemsEqual(read_cache.keys(), ['a'])
      self.assertItemsEqual(read_cache['a'].keys(), ['b'])

      beam_test_util.assert_that(
          read_cache['a']['b'],
          beam_test_util.equal_to([test_cache_dict['a']['b']]))


if __name__ == '__main__':
  test_case.main()
