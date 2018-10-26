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
"""Attributes for analyzers.

Attributes are objects that describe how to perform a full pass analysis over
some input tensors.  The attributes can be any object and is typically a
namedtuple.  User-defined attributes are possible but an implementation must be
defined (see register_ptransform in beam/common.py).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class Combiner(object):
  """Analyze using combiner function.

  This object mirrors a beam.CombineFn, that will receive a beam PCollection
  representing the batched input tensors.
  """

  def create_accumulator(self):
    """Return a fresh, empty accumulator.

    Returns: An empty accumulator.  This can be any Python value.
    """
    raise NotImplementedError

  def add_input(self, accumulator, batch_values):
    """Return result of folding a batch of inputs into accumulator.

    Args:
      accumulator: the current accumulator
      batch_values: A list of ndarrays representing the values of the inputs for
          a batch, which should be added to the accumulator.

    Returns: An accumulator that includes the batch of inputs.
    """
    raise NotImplementedError

  def merge_accumulators(self, accumulators):
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: the accumulators to merge

    Returns: The sole merged accumulator.
    """
    raise NotImplementedError

  def extract_output(self, accumulator):
    """Return result of converting accumulator into the output value.

    Args:
      accumulator: the final accumulator value.

    Returns: A list of ndarrays representing the result of this combiner.
    """
    raise NotImplementedError

  def num_outputs(self):
    """Return the number of outputs that are produced by extract_output.

    Returns: The number of outputs extract_output will produce.
    """
    raise NotImplementedError


Combine = collections.namedtuple('Combine', ['combiner', 'name'])

CombinePerKey = collections.namedtuple('CombinePerKey', ['combiner', 'name'])

Vocabulary = collections.namedtuple(
    'Vocabulary',
    ['top_k', 'frequency_threshold', 'vocab_filename', 'store_frequency',
     'vocab_ordering_type',
     'name'])

PTransform = collections.namedtuple('PTransform', ['ptransform', 'name'])
