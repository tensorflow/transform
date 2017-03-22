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
"""Benchmark test for tensorflow_transform.impl_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow as tf

from tensorflow_transform import impl_helper
from tensorflow_transform.internal import benchmark
from tensorflow_transform.tf_metadata import dataset_schema

import unittest

BATCH_SIZE = 1000


# pylint: disable=line-too-long
class ImplHelperBenchmark(benchmark.BmRegistry):
  r"""Parent class for the coder benchmarks.

  """
  # Keep the number of iterations and schema in sync across the benchmarks so
  # the magnitude of the performance results is comparable.
  #

  ITERATIONS = 2**8
  INPUT_SCHEMA = dataset_schema.from_feature_spec({
      'numeric': tf.FixedLenFeature(None, tf.int64),
      'numeric_0d': tf.FixedLenFeature([], tf.float32),
      'numeric_1d': tf.FixedLenFeature([1], tf.float32),
      'numeric_2d': tf.FixedLenFeature([2, 2], tf.float32),
      'varlen': tf.VarLenFeature(tf.string),
      'sparse': tf.SparseFeature('idx', 'val', tf.float32, 10)
  })


# Make sure you run this test with 'blaze test -c opt'
# before updating.
# Last update: 2017/03/14
# TestMakeFeedDictBenchmark with extra statement "pass" took 0.607286930084
# TestMakeFeedDictBenchmark with extra statement "gc.enable()" took 0.863233089447
@ImplHelperBenchmark.register(iterations=ImplHelperBenchmark.ITERATIONS)
class TestMakeFeedDictBenchmark(benchmark.Benchmark):
  _tensors = {
      'numeric': tf.placeholder(tf.int64),
      'numeric_0d': tf.placeholder(tf.float32),
      'numeric_1d': tf.placeholder(tf.float32),
      'numeric_2d': tf.placeholder(tf.float32),
      'varlen': tf.sparse_placeholder(tf.string),
      'sparse': tf.sparse_placeholder(tf.float32)
  }
  _instances = list(itertools.repeat({
      'numeric': 100,
      'numeric_0d': 1.0,
      'numeric_1d': [2.0],
      'numeric_2d': [[1.0, 2.0], [3.0, 4.0]],
      'varlen': ['doe', 'a', 'deer'],
      'sparse': ([2, 4, 8], [10.0, 20.0, 30.0])
  }, 1000))

  @classmethod
  def run_benchmark(cls):
    _ = impl_helper.make_feed_dict(cls._tensors,
                                   ImplHelperBenchmark.INPUT_SCHEMA,
                                   cls._instances)


# Make sure you run this test with 'blaze test -c opt'
# before updating.
# Last update: 2017/03/14
# TestMakeOutputDictBenchmark with extra statement "pass" took 1.68296504021
# TestMakeOutputDictBenchmark with extra statement "gc.enable()" took 1.99450588226
@ImplHelperBenchmark.register(iterations=ImplHelperBenchmark.ITERATIONS)
class TestMakeOutputDictBenchmark(benchmark.Benchmark):
  _fetches = {
      'numeric': np.array(list(itertools.repeat(100, BATCH_SIZE))),
      'numeric_0d': np.array(list(itertools.repeat(10.0, BATCH_SIZE))),
      'numeric_1d': np.array(list(itertools.repeat([40.0], BATCH_SIZE))),
      'numeric_2d': np.array(list(itertools.repeat([[1.0, 2.0], [3.0, 4.0]],
                                                   BATCH_SIZE))),
      'varlen': tf.SparseTensorValue(
          indices=np.array(list(itertools.product(
              xrange(BATCH_SIZE), xrange(0, 3)))),
          values=np.array(list(itertools.islice(
              itertools.cycle(['doe', 'a', 'deer']), 3*BATCH_SIZE))),
          dense_shape=(BATCH_SIZE, 3)),
      'sparse': tf.SparseTensorValue(
          indices=np.array(list(itertools.product(
              xrange(BATCH_SIZE), xrange(0, 6, 2)))),
          values=np.array(list(itertools.islice(
              itertools.cycle([10.0, 20.0, 30.0]), 3*BATCH_SIZE))),
          dense_shape=(BATCH_SIZE, 10)),
  }

  @classmethod
  def run_benchmark(cls):
    _ = impl_helper.make_output_dict(ImplHelperBenchmark.INPUT_SCHEMA,
                                     cls._fetches)
# pylint: enable=line-too-long


if __name__ == '__main__':
  unittest.main()
