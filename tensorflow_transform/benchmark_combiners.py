# Copyright 2020 Google Inc. All Rights Reserved.
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
# pylint: disable=line-too-long
r"""Benchmark test for combiners from tensorflow_transform.analyzers.

These benchmarks should be run with MCD Perflab (go/perflab).  Use the following
commandline to run the benchmarks, substituting python3 for python2 for Python 2
benchmarks. Note you must be a member of the perflab-machines MDB group to run
these benchmarks. See (go/perflab) for details.

blaze run -c opt --dynamic_mode=off \
  --run_under='perflab --constraints=arch=x86_64,platform_family=iota,platform_genus=sandybridge' \
  //third_party/py/tensorflow_transform:benchmark_combiners.python3

Last update: 2020/07/14

Python 2:
TestNumPyCombinerAddInput (gc disabled) took 6.62356901169
TestNumPyCombinerAddInput (gc enabled) took 6.85395503044
TestNumPyCombinerMergeAccumulators (gc disabled) took 8.283233881
TestNumPyCombinerMergeAccumulators (gc enabled) took 8.37501692772

Python 3:
TestNumPyCombinerAddInput (gc disabled) took 6.885694884695113
TestNumPyCombinerAddInput (gc enabled) took 6.812804839108139
TestNumPyCombinerMergeAccumulators (gc disabled) took 8.4131267699413
TestNumPyCombinerMergeAccumulators (gc enabled) took 8.452987399883568
"""
# pylint: enable=line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_transform import analyzers
from tensorflow_transform.google import benchmark

import unittest


class CombinersBenchmark(benchmark.BmRegistry):
  """Parent class for the combiner benchmarks."""


@CombinersBenchmark.register(iterations=benchmark.ITERATIONS)
class TestNumPyCombinerAddInput(benchmark.Benchmark):
  # Create an empty accumulator.
  _fn = np.sum
  _input_shape = (1000, 1000)
  _output_shapes = [_input_shape] * 5
  _combiner = analyzers.NumPyCombiner(
      fn=_fn,
      default_accumulator_value=0,
      output_dtypes=np.float32,
      output_shapes=_output_shapes)
  _accumulator = _combiner.create_accumulator()

  # Create batch values to be added to the accumulator and add them once to make
  # it non-empty.
  _batch_values = [np.full(_input_shape, 0.1212131)] * len(_output_shapes)
  _accumulator = _combiner.add_input(_accumulator, _batch_values)

  @classmethod
  def run_benchmark(cls):
    _ = cls._combiner.add_input(cls._accumulator, cls._batch_values)


@CombinersBenchmark.register(iterations=benchmark.ITERATIONS)
class TestNumPyCombinerMergeAccumulators(benchmark.Benchmark):
  _num_accumulators = 1000

  # Create an empty accumulator.
  _fn = np.sum
  _input_shape = (100, 100)
  _output_shapes = [_input_shape] * 2
  _combiner = analyzers.NumPyCombiner(
      fn=_fn,
      default_accumulator_value=0,
      output_dtypes=np.float32,
      output_shapes=_output_shapes)
  _accumulator = _combiner.create_accumulator()

  # Create batch values to be added to the accumulator and add them once to make
  # it non-empty.
  _batch_values = [np.full(_input_shape, 0.1212131)] * len(_output_shapes)
  _accumulator = _combiner.add_input(_accumulator, _batch_values)

  _accumulators = [_accumulator] * _num_accumulators

  @classmethod
  def run_benchmark(cls):
    _ = cls._combiner.merge_accumulators(cls._accumulators)


if __name__ == '__main__':
  unittest.main()
