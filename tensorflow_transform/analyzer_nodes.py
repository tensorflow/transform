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
"""Nodes that define analyzers.

`OperationNode`s are objects that describe how to perform a full pass analysis
over some input tensors.  They are described by an `OperationDef`.  This module
contains the `OperationDef` subclasses that define specific operations such as
computing a mean or vocabulary.  It also contains a special `OperationDef`,
`ExtractTensors` which represents the operation of extracting the values of a
tuple of `Tensor`s into a `PCollection`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections


import tensorflow as tf
from tensorflow_transform import nodes

TENSOR_REPLACEMENTS = 'tft_tensor_replacements'

TensorInfo = collections.namedtuple('TensorInfo',
                                    ['dtype', 'shape', 'is_asset_filepath'])


class TensorSource(
    collections.namedtuple('TensorSource', ['tensors', 'label']),
    nodes.OperationDef):
  """An `OperationDef` that defines extracting a tuple of tensor values.

  This `OperationDef` defines an operation that extracts the values of the given
  tensors into a PCollection of tuples of values.  It is used as a source for
  analyzers, which further transform

  This OperationDef accepts zero inputs and return a single output representing
  the PCollection of tuples of values.  It will be converted in
  tensorflow_transform.beam.analysis_graph_builder.build to an operation that
  extracts the tensors for a dictionary of tensors, after running a beam.ParDo
  to produce tensor values by running the graph on its inputs.

  Fields:
    tensors: The tensors whose values should be extracted.
    label: A unique label for this operation.
  """

  def __new__(cls, tensors, label=None):
    for tensor in tensors:
      if not isinstance(tensor, tf.Tensor):
        raise TypeError('tensor must be a Tensor, got {} of type {}'.format(
            tensor, type(tensor)))
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(TensorSource, cls).__new__(cls, tensors=tensors, label=label)


TensorSink = collections.namedtuple('TensorSink',
                                    ['tensor', 'future', 'is_asset_filepath'])


def bind_future_as_tensor(future, tensor_info, name=None):
  """Bind a future value as a tensor."""
  result = tf.placeholder(tensor_info.dtype, tensor_info.shape, name)
  tf.add_to_collection(
      TENSOR_REPLACEMENTS,
      TensorSink(result, future, tensor_info.is_asset_filepath))
  return result


class Combiner(object):
  """Analyze using combiner function.

  This object mirrors a beam.CombineFn, that will receive a beam PCollection
  representing the batched input tensors.
  """

  def __repr__(self):
    return '<{}>'.format(self.__class__.__name__)

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

  def output_tensor_infos(self):
    """Return the number of outputs that are produced by extract_output.

    Returns: The number of outputs extract_output will produce.
    """
    raise NotImplementedError


class AnalyzerDef(nodes.OperationDef):
  """A subclass of OperationDef whose outputs can be constant tensors.

  An AnalyzerDef is an OperationDef that also provides enough information to
  wrap each of its outputs as constant `Tensor`s in the graph.  By inserting
  the output of the AnalyzerDef back into the graph, the user can define
  multiple levels of anaylsis and transformation.

  All `OperationDef`s are placeholders for operations that will be implemented
  as `beam.PTransform`s.  This is done by a registration system.  The subclasses
  defined below that inherit from `AnalyzerDef` have there implementations
  registered in the module `tensorflow_transform.beam.analyzer_impls`.
  """

  __metaclass__ = abc.ABCMeta

  @property
  @abc.abstractmethod
  def output_tensor_infos(self):
    """A description on how to wrap the outputs of this AnalyzerDef.

    An `OperationDef` defines the number of outputs it creates.  An
    `AnalyzerDef` must implemented this property that defines not only the
    number of outputs but how to wrap each output as a tensor.
    """
    pass

  @property
  def num_outputs(self):
    """The number of outputs returned by this operation."""
    return len(self.output_tensor_infos)


class Combine(
    collections.namedtuple('Combine', ['combiner', 'label']), AnalyzerDef):
  """An analyzer that runs `beam.Combine`.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.Combine`.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._CombineImpl`.

  Fields:
    combiner: The Combiner to be applied to the inputs.
    label: A unique label for this operation.
  """

  def __new__(cls, combiner, label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(Combine, cls).__new__(cls, combiner=combiner, label=label)

  @property
  def output_tensor_infos(self):
    return self.combiner.output_tensor_infos()


class CombinePerKey(
    collections.namedtuple('CombinePerKey', ['combiner', 'label']),
    AnalyzerDef):
  """An analyzer that runs `beam.CombinePerKey`.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.CombinePerKey`.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._CombinePerKeyImpl`.

  Fields:
    combiner: The Combiner to be applied to the inputs.
    label: A unique label for this operation.
  """

  def __new__(cls, combiner, label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CombinePerKey, cls).__new__(
        cls, combiner=combiner, label=label)

  @property
  def output_tensor_infos(self):
    # Returns a key vocab and one output per combiner output.
    return [TensorInfo(tf.string, [None], False)
           ] + self.combiner.output_tensor_infos()


class Vocabulary(
    collections.namedtuple(
        'Vocabulary',
        [
            'top_k',
            'frequency_threshold',
            'vocab_filename',
            'store_frequency',
            'vocab_ordering_type',
            'use_adjusted_mutual_info',
            'min_diff_from_avg',
            'coverage_top_k',
            'coverage_frequency_threshold',
            'key_fn',
            'label'
        ]),
    AnalyzerDef):
  """OperationDef for computing a vocabulary of unique values.

  This analyzer computes a vocabulary composed of the unique values present in
  the input elements.  It selects a subset of the unique elements based on the
  provided parameters.  It may also accept a label and weight as input
  depending on the parameters.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls.VocabularyImpl`.

  See `tft.vocabulary` for a description of the parameters.
  """

  def __new__(
      cls,
      top_k,
      frequency_threshold,
      vocab_filename,
      store_frequency,
      vocab_ordering_type,
      use_adjusted_mutual_info,
      min_diff_from_avg,
      coverage_top_k,
      coverage_frequency_threshold,
      key_fn,
      label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(Vocabulary, cls).__new__(
        cls,
        top_k=top_k,
        frequency_threshold=frequency_threshold,
        vocab_filename=vocab_filename,
        store_frequency=store_frequency,
        vocab_ordering_type=vocab_ordering_type,
        use_adjusted_mutual_info=use_adjusted_mutual_info,
        min_diff_from_avg=min_diff_from_avg,
        coverage_top_k=coverage_top_k,
        coverage_frequency_threshold=coverage_frequency_threshold,
        key_fn=key_fn,
        label=label)

  @property
  def output_tensor_infos(self):
    return [TensorInfo(tf.string, [], True)]


class PTransform(
    collections.namedtuple('PTransform',
                           ['ptransform', 'output_tensor_info_list', 'label']),
    AnalyzerDef):
  """(Experimental) OperationDef for PTransform anaylzer.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._ptransform_impl`.

  Fields:
    ptransform: The `beam.PTransform` to be applied to the inputs.
    output_tensor_info_list: A list of `TensorInfo`s that defines the outputs of
        this `PTransform`.
    label: A unique label for this operation.
  """

  def __new__(cls, ptransform, output_tensor_info_list, label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(PTransform, cls).__new__(
        cls,
        ptransform=ptransform,
        output_tensor_info_list=output_tensor_info_list,
        label=label)

  @property
  def output_tensor_infos(self):
    return self.output_tensor_info_list
