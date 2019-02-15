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
import json

# GOOGLE-INITIALIZATION

import numpy as np
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


def get_input_tensors_value_nodes(tensor_inputs):
  return nodes.apply_operation(TensorSource, tensors=tensor_inputs)


TensorSink = collections.namedtuple('TensorSink',
                                    ['tensor', 'future', 'is_asset_filepath'])


def bind_future_as_tensor(future, tensor_info, name=None):
  """Bind a future value as a tensor."""
  result = tf.placeholder(tensor_info.dtype, tensor_info.shape, name)
  tf.add_to_collection(
      TENSOR_REPLACEMENTS,
      TensorSink(result, future, tensor_info.is_asset_filepath))
  return result


def wrap_as_tensor(output_value_node):
  analyzer_def = output_value_node.parent_operation.operation_def
  assert isinstance(analyzer_def, AnalyzerDef)
  return bind_future_as_tensor(
      output_value_node,
      analyzer_def.output_tensor_infos[output_value_node.value_index])


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

  @property
  def accumulator_coder(self):
    return JsonNumpyCacheCoder()


class CacheCoder(object):
  """A coder iterface for encoding and decoding cache items."""

  __metaclass__ = abc.ABCMeta

  def __repr__(self):
    return '<{}>'.format(self.__class__.__name__)

  @abc.abstractmethod
  def encode_cache(self, cache):
    pass

  @abc.abstractmethod
  def decode_cache(self, encoded_cache):
    pass


class JsonNumpyCacheCoder(CacheCoder):

  def encode_cache(self, accumulator):
    # Need to wrap in np.array and call tolist to make it JSON serializable.
    return tf.compat.as_bytes(json.dumps(np.array(accumulator).tolist()))

  def decode_cache(self, encoded_accumulator):
    return np.array(json.loads(tf.compat.as_text(encoded_accumulator)))


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


class CacheableCombineAccumulate(
    collections.namedtuple('CacheableCombineAccumulate', ['combiner', 'label']),
    AnalyzerDef):
  """An analyzer that runs a beam CombineFn to accumulate without merging.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.Combine`.

  Fields:
    combiner: The Combiner to be applies to the inputs.
    label: A unique label for this operation.
  """

  def __new__(cls, combiner, label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CacheableCombineAccumulate, cls).__new__(
        cls, combiner=combiner, label=label)

  @property
  def num_outputs(self):
    return 1

  @property
  def is_partitionable(self):
    return True

  @property
  def cache_coder(self):
    return self.combiner.accumulator_coder


class CacheableCombineMerge(
    collections.namedtuple('CacheableCombineMerge', ['combiner', 'label']),
    AnalyzerDef):
  """An analyzer that runs a beam CombineFn to only merge computed accumulators.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.Combine`.

  Fields:
    combiner: The Combiner to be applied to the inputs.
    label: A unique label for this operation.
  """

  def __new__(cls, combiner, label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CacheableCombineMerge, cls).__new__(
        cls, combiner=combiner, label=label)

  @property
  def output_tensor_infos(self):
    return self.combiner.output_tensor_infos()


class CacheableCombinePerKeyAccumulate(CacheableCombineAccumulate):
  """An analyzer that runs `beam.CombinePerKey` to accumulate without merging.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.CombinePerKey`.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._IntermediateAccumulateCombineImpl`.

  Fields:
    combiner: The Combiner to be applied to the inputs.
    label: A unique label for this operation.
  """

  def __new__(cls, combiner, label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CacheableCombinePerKeyAccumulate, cls).__new__(
        cls, combiner=combiner, label=label)


class CacheableCombinePerKeyMerge(CacheableCombineMerge):
  """An analyzer that runs `beam.CombinePerKey` to only merge accumulators.

  This analyzer reduces the values that it accepts as inputs, using the
  provided `Combiner`.  The `Combiner` is applied to the data by wrapping it as
  a `beam.CombineFn` and applying `beam.CombinePerKey`.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._MergeAccumulatorsCombinePerKeyImpl`

  Fields:
    combiner: The Combiner to use for merging and extracting outputs.
    label: A unique label for this operation.
  """

  def __new__(cls, combiner, label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CacheableCombinePerKeyMerge, cls).__new__(
        cls, combiner=combiner, label=label)

  @property
  def output_tensor_infos(self):
    # Returns a key vocab and one output per combiner output.
    return [TensorInfo(tf.string, (None,), False)] + [
        TensorInfo(info.dtype, (None,) + info.shape, info.is_asset_filepath)
        for info in self.combiner.output_tensor_infos()
    ]


class VocabularyAccumulate(
    collections.namedtuple('VocabularyAccumulate',
                           ['vocab_ordering_type', 'label']),
    nodes.OperationDef):
  """An operation that accumulates unique words with their frequency or weight.

  This operation is implemented by
  `tensorflow_transform.beam.analyzer_impls.VocabularyAccumulateImpl`.
  """

  def __new__(cls, vocab_ordering_type, label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(VocabularyAccumulate, cls).__new__(
        cls, vocab_ordering_type=vocab_ordering_type, label=label)

  @property
  def num_outputs(self):
    return 1

  @property
  def is_partitionable(self):
    return True

  @property
  def cache_coder(self):
    return _VocabularyAccumulatorCoder()


class _VocabularyAccumulatorCoder(CacheCoder):
  """Coder for vocabulary accumulators."""

  def encode_cache(self, accumulator):
    # Need to wrap in np.array and call tolist to make it JSON serializable.
    word, count = accumulator
    accumulator = (word.decode('utf-8'), count)
    return tf.compat.as_bytes(
        json.dumps(np.array(accumulator, dtype=object).tolist()))

  def decode_cache(self, encoded_accumulator):
    return np.array(
        json.loads(tf.compat.as_text(encoded_accumulator)), dtype=object)


class VocabularyMerge(
    collections.namedtuple('VocabularyMerge', [
        'vocab_ordering_type', 'use_adjusted_mutual_info', 'min_diff_from_avg',
        'label'
    ]), nodes.OperationDef):
  """An operation that merges the accumulators produced by VocabularyAccumulate.

  This operation operates on the output of VocabularyAccumulate and is
  implemented by `tensorflow_transform.beam.analyzer_impls.VocabularyMergeImpl`.

  See `tft.vocabulary` for a description of the parameters.
  """

  def __new__(cls,
              vocab_ordering_type,
              use_adjusted_mutual_info,
              min_diff_from_avg,
              label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(VocabularyMerge, cls).__new__(
        cls,
        vocab_ordering_type=vocab_ordering_type,
        use_adjusted_mutual_info=use_adjusted_mutual_info,
        min_diff_from_avg=min_diff_from_avg,
        label=label)

  @property
  def num_outputs(self):
    return 1


class VocabularyOrderAndFilter(
    collections.namedtuple('VocabularyOrderAndFilter', [
        'top_k', 'frequency_threshold', 'coverage_top_k',
        'coverage_frequency_threshold', 'key_fn', 'label'
    ]), nodes.OperationDef):
  """An operation that filters and orders a computed vocabulary.

  This operation operates on the output of VocabularyMerge and is implemented by
  `tensorflow_transform.beam.analyzer_impls.VocabularyOrderAndFilterImpl`.

  See `tft.vocabulary` for a description of the parameters.
  """

  def __new__(
      cls,
      top_k,
      frequency_threshold,
      coverage_top_k,
      coverage_frequency_threshold,
      key_fn,
      label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(VocabularyOrderAndFilter, cls).__new__(
        cls,
        top_k=top_k,
        frequency_threshold=frequency_threshold,
        coverage_top_k=coverage_top_k,
        coverage_frequency_threshold=coverage_frequency_threshold,
        key_fn=key_fn,
        label=label)

  @property
  def num_outputs(self):
    return 1


class VocabularyWrite(
    collections.namedtuple('VocabularyWrite',
                           ['vocab_filename', 'store_frequency', 'label']),
    AnalyzerDef):
  """An analyzer that writes vocabulary files from an accumulator.

  This operation operates on the output of VocabularyOrderAndFilter and is
  implemented by `tensorflow_transform.beam.analyzer_impls.VocabularyWriteImpl`.

  See `tft.vocabulary` for a description of the parameters.
  """

  def __new__(cls, vocab_filename, store_frequency, label=None):
    if label is None:
      scope = tf.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(VocabularyWrite, cls).__new__(
        cls,
        vocab_filename=vocab_filename,
        store_frequency=store_frequency,
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


class WriteCache(
    collections.namedtuple('WriteCache', ['path', 'coder', 'label']),
    nodes.OperationDef):
  """OperationDef for writing a cache object to a file system.

  Fields:
    path: A path to write the cache to.
    encode_cache_fn: A map function that will be used to encode a cache object.
    label: A unique label for this operation.
  """

  @property
  def is_partitionable(self):
    return True


class ReadCache(
    collections.namedtuple('ReadCache', ['path', 'coder', 'label']),
    nodes.OperationDef):
  """OperationDef for reading a cache object from a file system.

  Fields:
    path: A path to read the cache from.
    decode_cache_fn: A map function that will be used to decode a cache object.
    label: A unique label for this operation.
  """

  @property
  def is_partitionable(self):
    return True
