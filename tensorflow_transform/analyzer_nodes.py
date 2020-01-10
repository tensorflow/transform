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
import struct

# GOOGLE-INITIALIZATION

import numpy as np
import tensorflow as tf
from tensorflow_transform import nodes

TENSOR_REPLACEMENTS = 'tft_tensor_replacements'


class TensorInfo(collections.namedtuple(
    'TensorInfo', ['dtype', 'shape', 'is_asset_filepath'])):
  """A container for attributes of output tensors from analyzers.

  Fields:
    dtype: The TensorFlow dtype.
    shape: The shape of the tensor.
    is_asset_filepath: Whether it should be part of the filepath assets.
  """

  def __init__(self, dtype, shape, is_asset_filepath):
    del shape, is_asset_filepath

    if not isinstance(dtype, tf.DType):
      raise TypeError('dtype must be a TensorFlow dtype, got {}'.format(dtype))

    super(TensorInfo, self).__init__()


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
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(TensorSource, cls).__new__(cls, tensors=tensors, label=label)


def get_input_tensors_value_nodes(tensor_inputs):
  return nodes.apply_operation(TensorSource, tensors=tensor_inputs)


TensorSink = collections.namedtuple('TensorSink',
                                    ['tensor', 'future', 'is_asset_filepath'])


def bind_future_as_tensor(future, tensor_info, name=None):
  """Bind a future value as a tensor."""
  result = tf.compat.v1.placeholder(tensor_info.dtype, tensor_info.shape, name)
  tf.compat.v1.add_to_collection(
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
    """Return the number / types of outputs that are produced by extract_output.

    Returns: An iterable of `TensorInfo` describing how the outputs that
      extract_output will produce should be wrapped as `Tensor`s.

    Types are required to be TensorFlow dtypes.
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
  """An accumulator cache coder that can handle lists."""

  def _convert_numpy_dtype(self, x):
    if hasattr(x, 'tolist'):
      return x.tolist()
    return x

  def encode_cache(self, accumulator):
    if isinstance(accumulator, (list, tuple)):
      primitive_accumulator = [
          self._convert_numpy_dtype(a) for a in accumulator
      ]
    else:
      primitive_accumulator = self._convert_numpy_dtype(accumulator)
    # Need to wrap in np.array and call tolist to make it JSON serializable.
    return tf.compat.as_bytes(json.dumps(primitive_accumulator))

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


# We do the packing of combiners after the caching optimization. Hence, we don't
# name this operation as cacheable. The rationale behind doing the combiner
# packing after the cache optimization is that this optimization is more of a
# Beam execution level optimization and we want to keep it towards the end.
# So that, once Beam can automatically pack combines, we can remove this.
class PackedCombineAccumulate(
    collections.namedtuple('PackedCombineAccumulate', ['combiners', 'label']),
    AnalyzerDef):
  """An analyzer that packs a list of combiners into a single beam CombineFn.

  Fields:
    combiners:  A list of `analysis_graph_builder._CombinerOpWrapper` objects.
    label: A unique label for this operation.
  """

  def __new__(cls, combiners, label=None):
    if label is None:
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(PackedCombineAccumulate, cls).__new__(
        cls, combiners=combiners, label=label)

  @property
  def num_outputs(self):
    return 1

  # Note that this will not have any effect as packing of combiners is done
  # after the caching optimization.
  @property
  def is_partitionable(self):
    return True


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
      scope = tf.compat.v1.get_default_graph().get_name_scope()
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
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CacheableCombineMerge, cls).__new__(
        cls, combiner=combiner, label=label)

  @property
  def output_tensor_infos(self):
    return self.combiner.output_tensor_infos()


class _CombinerPerKeyAccumulatorCoder(CacheCoder):
  """Coder for per-key combiner accumulators."""

  def __init__(self, value_coder):
    self._combiner_coder = value_coder
    self._vocabulary_coder = _BaseKVCoder()

  def __repr__(self):
    return '<{}[{}[{}]]>'.format(self.__class__.__name__,
                                 repr(self._vocabulary_coder),
                                 repr(self._combiner_coder))

  def encode_cache(self, accumulator):
    key, value = accumulator
    encoded_value = self._combiner_coder.encode_cache(value)
    return self._vocabulary_coder.encode_cache((key, encoded_value))

  def decode_cache(self, encoded_accumulator):
    accumulator = self._vocabulary_coder.decode_cache(encoded_accumulator)
    key, encoded_value = accumulator
    value = self._combiner_coder.decode_cache(encoded_value)
    return (key, value)


class CacheableCombinePerKeyAccumulate(
    collections.namedtuple('CacheableCombinePerKeyAccumulate',
                           ['combiner', 'label']), AnalyzerDef):
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
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CacheableCombinePerKeyAccumulate, cls).__new__(
        cls, combiner=combiner, label=label)

  @property
  def num_outputs(self):
    return 1

  @property
  def is_partitionable(self):
    return True

  @property
  def cache_coder(self):
    return _CombinerPerKeyAccumulatorCoder(self.combiner.accumulator_coder)


class CacheableCombinePerKeyMerge(
    collections.namedtuple('CacheableCombinePerKeyMerge',
                           ['combiner', 'label']), nodes.OperationDef):
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
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CacheableCombinePerKeyMerge, cls).__new__(
        cls, combiner=combiner, label=label)


class CacheableCombinePerKeyFormatKeys(
    collections.namedtuple('CacheableCombinePerKeyFormatKeys',
                           ['combiner', 'label']), AnalyzerDef):
  """An analyzer that formats output for the non-stored per-key case.

  This analyzer converts the (key, output) pairs into a tuple of keys (of type
  string) and outputs.

  This analyzer is implemented by
  `tensorflow_transform.beam.analyzer_impls._CombinePerKeyFormatKeysImpl`

  Fields:
    combiner: The Combiner to use for extracting outputs.
    label: A unique label for this operation.
  """

  def __new__(cls, combiner, label=None):
    if label is None:
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CacheableCombinePerKeyFormatKeys, cls).__new__(
        cls, combiner=combiner, label=label)

  @property
  def output_tensor_infos(self):
    # Returns a key vocab and one output per combiner output.
    return [TensorInfo(tf.string, (None,), False)] + [
        TensorInfo(info.dtype, (None,) + info.shape, info.is_asset_filepath)
        for info in self.combiner.output_tensor_infos()
    ]


class CacheableCombinePerKeyFormatLarge(
    collections.namedtuple('CacheableCombinePerKeyFormatLarge', [
        'label'
    ]), nodes.OperationDef):
  """An analyzer that formats output prior to writing to file for per-key case.

  This operation operates on the output of CacheableCombinePerKeyAccumulate and
  is implemented by `tensorflow_transform.beam.analyzer_impls.
  _CombinePerKeyFormatLargeImpl`.
  """

  def __new__(cls, label=None):
    if label is None:
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(CacheableCombinePerKeyFormatLarge, cls).__new__(
        cls, label=label)

  @property
  def num_outputs(self):
    return 1


class ScaleAndFlattenPerKeyBucketBouandaries(
    collections.namedtuple('PostProcessPerKeyBucketBoundaries',
                           ['output_tensor_dtype', 'label']), AnalyzerDef):
  """An analyzer which takes quantile boundaries per key and combines them.

  It receives a 2-d array of boundaries, computes scales and shifts to each
  row separately, a new boundaries 1-d array which is a combination of
  boundaries for all the keys, and the number of buckets defined for each key.

  This outputs boundaries, scale_factor_per_key, shift_per_key, num_buckets.

  For example, for an input boundaries matrix, [[0, 1, 2], [0, 1, 2]] it will
  return:
  boundaries: [0, 0.5, 1, 1.5, 2]
  scale_factor_per_key: [0.5, 0.5]
  shift_per_key: [0, 1]
  num_buckets: 4

  So the transformation of each input x before computing its bucket should be:
  F(x, key) = x * scale_factor_per_key[key] + shift_per_key[key]
  """

  def __new__(cls, output_tensor_dtype, label=None):
    if label is None:
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(ScaleAndFlattenPerKeyBucketBouandaries, cls).__new__(
        cls, output_tensor_dtype=output_tensor_dtype, label=label)

  @property
  def output_tensor_infos(self):
    # Boundaries, scale_factor_per_key, shift_per_key, num_buckets.
    return [
        TensorInfo(self.output_tensor_dtype, (None,), is_asset_filepath=False)
    ] * 3 + [TensorInfo(tf.int64, (), is_asset_filepath=False)]


class VocabularyAccumulate(
    collections.namedtuple('VocabularyAccumulate',
                           ['vocab_ordering_type', 'input_dtype', 'label']),
    nodes.OperationDef):
  """An operation that accumulates unique words with their frequency or weight.

  This operation is implemented by
  `tensorflow_transform.beam.analyzer_impls._VocabularyAccumulateImpl`.
  """

  def __new__(cls, vocab_ordering_type, input_dtype=tf.string.name, label=None):
    if label is None:
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(VocabularyAccumulate, cls).__new__(
        cls,
        vocab_ordering_type=vocab_ordering_type,
        input_dtype=input_dtype,
        label=label)

  @property
  def num_outputs(self):
    return 1

  @property
  def is_partitionable(self):
    return True

  @property
  def cache_coder(self):
    return _VocabularyAccumulatorCoder(input_dtype=self.input_dtype)


class _BaseKVCoder(CacheCoder):
  """Coder for key-value based accumulators."""

  def __init__(self):
    self._lengths_prefix_format = 'qq'
    self._lengths_prefix_length = struct.calcsize(self._lengths_prefix_format)

  def encode_cache(self, accumulator):
    token, value = accumulator
    len_token, len_value = len(token), len(value)
    return struct.pack(
        '{}{}s{}s'.format(self._lengths_prefix_format, len_token, len_value),
        len_token, len_value, token, value)

  def decode_cache(self, encoded_accumulator):
    (len_token, len_value) = struct.unpack_from(
        self._lengths_prefix_format,
        encoded_accumulator[:self._lengths_prefix_length])
    accumulator = struct.unpack_from(
        '{}s{}s'.format(len_token, len_value),
        encoded_accumulator[self._lengths_prefix_length:])
    return accumulator


class _VocabularyAccumulatorCoder(_BaseKVCoder):
  """Coder for vocabulary accumulators."""

  def __init__(self, input_dtype=tf.string.name):
    self._input_dtype = tf.dtypes.as_dtype(input_dtype)
    super(_VocabularyAccumulatorCoder, self).__init__()

  def encode_cache(self, accumulator):
    token, value = accumulator
    if self._input_dtype is not tf.string:
      token = tf.compat.as_bytes(json.dumps(token))
    # If the value is a _WeightedMeanAndVarAccumulator, cast each field to a
    # list for serialization.
    if isinstance(value, tuple):
      value = [
          a.tolist()
          for a in (value.count, value.mean, value.variance, value.weight)
      ]
    value = tf.compat.as_bytes(json.dumps(value))
    return super(_VocabularyAccumulatorCoder, self).encode_cache((token, value))

  def decode_cache(self, encoded_accumulator):
    accumulator = super(_VocabularyAccumulatorCoder, self).decode_cache(
        encoded_accumulator)
    token, value = accumulator
    if self._input_dtype is not tf.string:
      token = json.loads(tf.compat.as_text(token))

    value = json.loads(tf.compat.as_text(value))
    if isinstance(value, list):
      # If the value is a _WeightedMeanAndVarAccumulator (serialized to tuple),
      # cast each field back to a np.array.
      (count, mean, variance, weight) = value
      value = (np.array(count), np.array(mean), np.array(variance),
               np.array(weight))
    return token, value


class VocabularyCount(
    collections.namedtuple('VocabularyCount', ['label']), nodes.OperationDef):
  """An operation counts the total number of tokens in a vocabulary.

  This operation takes in the output of VocabularyAccumulate and is implemented
  by `tensorflow_transform.beam.analyzer_impls._VocabularyCountImpl`.

  The output of this operation is a singleton Integer.
  """

  def __new__(cls, label=None):
    if label is None:
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(VocabularyCount, cls).__new__(cls, label=label)

  @property
  def num_outputs(self):
    return 1


class VocabularyMerge(
    collections.namedtuple('VocabularyMerge', [
        'vocab_ordering_type', 'use_adjusted_mutual_info', 'min_diff_from_avg',
        'label'
    ]), nodes.OperationDef):
  """An operation that merges the accumulators produced by VocabularyAccumulate.

  This operation operates on the output of VocabularyAccumulate and is
  implemented by `tensorflow_transform.beam.analyzer_impls._VocabularyMergeImpl`
  .

  See `tft.vocabulary` for a description of the parameters.
  """

  def __new__(cls,
              vocab_ordering_type,
              use_adjusted_mutual_info,
              min_diff_from_avg,
              label=None):
    if label is None:
      scope = tf.compat.v1.get_default_graph().get_name_scope()
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


class VocabularyPrune(
    collections.namedtuple('VocabularyPrune', [
        'top_k', 'frequency_threshold', 'coverage_top_k',
        'coverage_frequency_threshold', 'key_fn', 'label'
    ]), nodes.OperationDef):
  """An operation that filters and orders a computed vocabulary.

  This operation operates on the output of VocabularyMerge and is implemented by
  `tensorflow_transform.beam.analyzer_impls._VocabularyPruneImpl`.

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
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(VocabularyPrune, cls).__new__(
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


class VocabularyOrderAndWrite(
    collections.namedtuple('VocabularyOrderAndWrite', [
        'vocab_filename',
        'store_frequency',
        'input_dtype',
        'label',
        'fingerprint_shuffle',
    ]), AnalyzerDef):
  """An analyzer that writes vocabulary files from an accumulator.

  This operation operates on the output of VocabularyPrune and is implemented by
  `tensorflow_transform.beam.analyzer_impls._VocabularyOrderAndWriteImpl`.

  See `tft.vocabulary` for a description of the parameters.
  """

  def __new__(cls,
              vocab_filename,
              store_frequency,
              fingerprint_shuffle,
              input_dtype=tf.string.name,
              label=None):
    if label is None:
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(VocabularyOrderAndWrite, cls).__new__(
        cls,
        vocab_filename=vocab_filename,
        store_frequency=store_frequency,
        fingerprint_shuffle=fingerprint_shuffle,
        input_dtype=input_dtype,
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
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
    return super(PTransform, cls).__new__(
        cls,
        ptransform=ptransform,
        output_tensor_info_list=output_tensor_info_list,
        label=label)

  @property
  def output_tensor_infos(self):
    return self.output_tensor_info_list


class EncodeCache(
    collections.namedtuple('EncodeCache', ['coder', 'label']),
    nodes.OperationDef):
  """OperationDef for encoding a cache instance.

  Fields:
    coder: An instance of CacheCoder used to encode cache.
    label: A unique label for this operation.
  """

  @property
  def is_partitionable(self):
    return True


class DecodeCache(
    collections.namedtuple('DecodeCache',
                           ['dataset_key', 'cache_key', 'coder', 'label']),
    nodes.OperationDef):
  """OperationDef for decoding a cache instance.

  Fields:
    coder: An instance of CacheCoder used to decode cache.
    label: A unique label for this operation.
  """

  def get_field_str(self, field_name):
    if field_name == 'cache_key':
      return '<bytes>'
    return super(DecodeCache, self).get_field_str(field_name)

  @property
  def is_partitionable(self):
    return True
