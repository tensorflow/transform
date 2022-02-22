# Copyright 2021 Google Inc. All Rights Reserved.
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
"""Experimental functions that involve a full pass over the dataset.

This module contains functions that are used in the preprocessing function, to
define a full pass operation such as computing the sum, min, max or unique
values of a tensor over the entire dataset.  This is implemented by a reduction
operation in the Beam implementation.

From the user's point of view, an analyzer appears as a regular TensorFlow
function, i.e. it accepts and returns tensors.  However it is represented in
the graph as a `Analyzer` which is not a TensorFlow op, but a placeholder for
the computation that takes place outside of TensorFlow.
"""

from typing import Any, Collection, List, Optional, Tuple, Union, Iterable

import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import nodes
from tensorflow_transform import tf_utils
from tfx_bsl import sketches

# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple
from typing_extensions import Protocol

__all__ = [
    'PTransformAnalyzerCacheCoder',
    'SimpleJsonPTransformAnalyzerCacheCoder',
    'CacheablePTransformAnalyzer',
    'ptransform_analyzer',
    'approximate_vocabulary',
]

PTransformAnalyzerCacheCoder = analyzer_nodes.CacheCoder
SimpleJsonPTransformAnalyzerCacheCoder = analyzer_nodes.JsonNumpyCacheCoder

_APPROXIMATE_VOCAB_FILENAME_PREFIX = 'approx_vocab_'
_APPROXIMATE_VOCAB_FREQUENCY_FILENAME_PREFIX = 'approx_vocab_frequency_'


class _BeamPTransform(Protocol):
  """Pytype for `beam.PTransform` without depending on beam in this module.
  """

  def expand(self, pcol: Any) -> Any:
    ...

  def default_label(self) -> str:
    ...


# TODO(zoyahav): Add an example for using this API.
class CacheablePTransformAnalyzer(
    tfx_namedtuple.TypedNamedTuple(
        'PTransformCachedAnalyzer',
        [('make_accumulators_ptransform', _BeamPTransform),
         ('merge_accumulators_ptransform', _BeamPTransform),
         ('extract_output_ptransform', _BeamPTransform),
         ('cache_coder', PTransformAnalyzerCacheCoder)])):
  """A PTransformAnalyzer which enables analyzer cache.

  WARNING: This should only be used if the analyzer can correctly be separated
  into make_accumulators, merge_accumulators and extract_output stages.
  1. make_accumulators_ptransform: this is a `beam.PTransform` which maps data
     to a more compact mergeable representation (accumulator). Mergeable here
     means that it is possible to combine multiple representations produced from
     a partition of the dataset into a representation of the entire dataset.
  1. merge_accumulators_ptransform: this is a `beam.PTransform` which operates
     on a collection of accumulators, i.e. the results of both the
     make_accumulators_ptransform and merge_accumulators_ptransform stages,
     and produces a single reduced accumulator. This operation must be
     associative and commutative in order to have reliably reproducible results.
  1. extract_output: this is a `beam.PTransform` which operates on the result of
     the merge_accumulators_ptransform stage, and produces the outputs of the
     analyzer. These outputs must be consistent with the `output_dtypes` and
     `output_shapes` provided to `ptransform_analyzer`.

  This container also holds a `cache_coder` (`PTransformAnalyzerCacheCoder`)
  which can encode outputs and decode the inputs of the
  `merge_accumulators_ptransform` stage.
  In many cases, `SimpleJsonPTransformAnalyzerCacheCoder` would be sufficient.

  To ensure the correctness of this analyzer, the following must hold:
  merge(make({D1, ..., Dn})) == merge({make(D1), ..., make(Dn)})
  """
  __slots__ = ()


def _apply_analyzer(ptransform: Union[_BeamPTransform,
                                      CacheablePTransformAnalyzer],
                    *tensor_inputs: common_types.TensorType,
                    **analyzer_def_kwargs: Any) -> Tuple[tf.Tensor, ...]:
  """Applies the analyzer over the whole dataset.

  Args:
    ptransform: A class inheriting from analyzer_nodes.AnalyzerDef or
      CacheablePTransformAnalyzer that should be applied.
    *tensor_inputs: A list of input `Tensor`s or `CompositeTensor`s.
    **analyzer_def_kwargs: KW arguments to use when constructing
      analyzer_def_cls.

  Returns:
    A list of `Tensor`s representing the values of the analysis result.
  """
  input_values_node = analyzer_nodes.get_input_tensors_value_nodes(
      tensor_inputs)
  if isinstance(ptransform, CacheablePTransformAnalyzer):
    with tf.compat.v1.name_scope('make_accumulators'):
      make_accumulators_value_node = nodes.apply_multi_output_operation(
          analyzer_nodes.PTransform,
          input_values_node,
          ptransform=ptransform.make_accumulators_ptransform,
          is_partitionable=True,
          **analyzer_def_kwargs)
    with tf.compat.v1.name_scope('local_merge_accumulators'):
      cached_value_nodes = nodes.apply_multi_output_operation(
          analyzer_nodes.PTransform,
          *make_accumulators_value_node,
          ptransform=ptransform.merge_accumulators_ptransform,
          is_partitionable=True,
          cache_coder=ptransform.cache_coder,
          **analyzer_def_kwargs)
    with tf.compat.v1.name_scope('global_merge_accumulators'):
      merge_output_value_nodes = nodes.apply_multi_output_operation(
          analyzer_nodes.PTransform,
          *cached_value_nodes,
          ptransform=ptransform.merge_accumulators_ptransform,
          is_partitionable=False,
          **analyzer_def_kwargs)
    with tf.compat.v1.name_scope('extract_output'):
      output_value_nodes = nodes.apply_multi_output_operation(
          analyzer_nodes.PTransform,
          *merge_output_value_nodes,
          ptransform=ptransform.extract_output_ptransform,
          is_partitionable=False,
          **analyzer_def_kwargs)
  else:
    output_value_nodes = nodes.apply_multi_output_operation(
        analyzer_nodes.PTransform,
        input_values_node,
        ptransform=ptransform,
        is_partitionable=False,
        **analyzer_def_kwargs)
  return tuple(map(analyzer_nodes.wrap_as_tensor, output_value_nodes))


# TODO(b/164921571): Support output assets in tfrecord format.
@common.log_api_use(common.ANALYZER_COLLECTION)
def ptransform_analyzer(
    inputs: Collection[tf.Tensor],
    ptransform: Union[_BeamPTransform, CacheablePTransformAnalyzer],
    output_dtypes: Collection[tf.dtypes.DType],
    output_shapes: Collection[List[int]],
    output_asset_default_values: Optional[Collection[Optional[bytes]]] = None,
    name: Optional[str] = None):
  # pylint: disable=line-too-long
  """Applies a user-provided PTransform over the whole dataset.

  WARNING: This is experimental.

  Note that in order to have asset files copied correctly, any outputs that
  represent asset filenames must be added to the `tf.GraphKeys.ASSET_FILEPATHS`
  collection by the caller if using Transform's APIs in compat v1 mode.

  Example:

  >>> class MeanPerKey(beam.PTransform):
  ...   def expand(self, pcoll: beam.PCollection[Tuple[np.ndarray, np.ndarray]]) -> Tuple[beam.PCollection[np.ndarray], beam.PCollection[np.ndarray]]:
  ...     def extract_output(key_value_pairs):
  ...       keys, values = zip(*key_value_pairs)
  ...       return [beam.TaggedOutput('keys', keys),
  ...               beam.TaggedOutput('values', values)]
  ...     return tuple(
  ...         pcoll
  ...         | 'ZipAndFlatten' >> beam.FlatMap(lambda batches: list(zip(*batches)))
  ...         | 'MeanPerKey' >> beam.CombinePerKey(beam.combiners.MeanCombineFn())
  ...         | 'ToList' >> beam.combiners.ToList()
  ...         | 'Extract' >> beam.FlatMap(extract_output).with_outputs(
  ...             'keys', 'values'))
  >>> def preprocessing_fn(inputs):
  ...   outputs = tft.experimental.ptransform_analyzer(
  ...       inputs=[inputs['s'], inputs['x']],
  ...       ptransform=MeanPerKey(),
  ...       output_dtypes=[tf.string, tf.float32],
  ...       output_shapes=[[2], [2]])
  ...   (keys, means) = outputs
  ...   mean_a = tf.reshape(tf.gather(means, tf.where(keys == 'a')), [])
  ...   return { 'x/mean_a': inputs['x'] / mean_a }
  >>> raw_data = [dict(x=1, s='a'), dict(x=8, s='b'), dict(x=3, s='a')]
  >>> feature_spec = dict(
  ...     x=tf.io.FixedLenFeature([], tf.float32),
  ...     s=tf.io.FixedLenFeature([], tf.string))
  >>> raw_data_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(
  ...     tft.tf_metadata.schema_utils.schema_from_feature_spec(feature_spec))
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> transformed_data
  [{'x/mean_a': 0.5}, {'x/mean_a': 4.0}, {'x/mean_a': 1.5}]

  Args:
    inputs: An ordered collection of input `Tensor`s.
    ptransform: A Beam PTransform that accepts a Beam PCollection where each
      element is a tuple of `ndarray`s.  Each element in the tuple contains a
      batch of values for the corresponding input tensor of the analyzer and
      maintain their shapes and dtypes.
      It returns a `PCollection`, or a tuple of `PCollections`, each containing
      a single element which is an `ndarray` or a list of primitive types. The
      contents of these output `PCollection`s must be consistent with the given
      values of `output_dtypes` and `output_shapes`.
      It may inherit from `tft_beam.experimental.PTransformAnalyzer` if access
      to a temp base directory is needed.
      Alternatively, it could be an instance of
      `tft.experimental.CacheablePTransformAnalyzer` in order to enable cache
      for this analyzer, when analyzer cache is enabled for this pipeline.
    output_dtypes: An ordered collection of TensorFlow dtypes of the output of
      the analyzer.
    output_shapes: An ordered collection of shapes of the output of the
      analyzer. Must have the same length as output_dtypes.
    output_asset_default_values: (Optional) An ordered collection of optional
      `bytes` aligned with output_dtypes/output_shapes. Every item in this
      collection which is not `None` indicates that the output is a TF asset
      path, and its value would be used as the default value of this asset file
      prior to analysis.
    name: (Optional) Similar to a TF op name.  Used to define a unique scope for
      this analyzer, which can be used for debugging info.

  Returns:
    A list of output `Tensor`s.  These will have `dtype` and `shape` as
      specified by `output_dtypes` and `output_shapes`.

  Raises:
    ValueError: If output_dtypes and output_shapes have different lengths.
  """
  # pylint: enable=line-too-long
  if len(output_dtypes) != len(output_shapes):
    raise ValueError('output_dtypes ({}) and output_shapes ({}) had different'
                     ' lengths'.format(output_dtypes, output_shapes))
  if output_asset_default_values is not None:
    if len(output_asset_default_values) != len(output_dtypes):
      raise ValueError(
          'output_dtypes ({}) and output_asset_default_values ({}) had '
          'different lengths'.format(output_dtypes,
                                     output_asset_default_values))
    output_asset_default_values = [
        analyzer_nodes.TemporaryAssetInfo(value, 'text')
        for value in output_asset_default_values
    ]
  else:
    output_asset_default_values = [None] * len(output_dtypes)
  with tf.compat.v1.name_scope(name, 'ptransform'):
    output_tensor_infos = [
        analyzer_nodes.TensorInfo(dtype, shape, default_asset_content)
        for dtype, shape, default_asset_content in zip(
            output_dtypes, output_shapes, output_asset_default_values)
    ]
    return _apply_analyzer(
        ptransform, *inputs, output_tensor_info_list=output_tensor_infos)


def _get_approx_vocab_filename(vocab_filename: Optional[str],
                               store_frequency: bool) -> str:
  """Returns a sanitized vocabulary filename with appropriate prefix applied.

  Args:
    vocab_filename: The file name for the approximate vocabulary file. If None,
      the "approximate_vocabulary" scope name in the context of this graph will
      be used as the file name.
    store_frequency: A bool that is true when the vocabulary for which this
      generates a filename stores term frequency. False otherwise.

  Returns:
    A valid filename.
  """
  if vocab_filename is not None:
    prefix = None
  elif store_frequency:
    prefix = _APPROXIMATE_VOCAB_FILENAME_PREFIX
  else:
    prefix = _APPROXIMATE_VOCAB_FREQUENCY_FILENAME_PREFIX

  # Make the file name path safe.
  return analyzers.sanitized_vocab_filename(vocab_filename, prefix=prefix)


@common.log_api_use(common.ANALYZER_COLLECTION)
def approximate_vocabulary(
    x: common_types.TensorType,
    top_k: int,
    vocab_filename: Optional[str] = None,
    store_frequency: bool = False,
    weights: Optional[tf.Tensor] = None,
    file_format: common_types.VocabularyFileFormatType = analyzers
    .DEFAULT_VOCABULARY_FILE_FORMAT,
    name: Optional[str] = None) -> common_types.TemporaryAnalyzerOutputType:
  r"""Computes the unique values of a `Tensor` over the whole dataset.

  Approximately computes the unique values taken by `x`, which can be a `Tensor`
  or `CompositeTensor` of any size.  The unique values will be aggregated over
  all dimensions of `x` and all instances.

  This analyzer provides an approximate alternative to `tft.vocabulary` that can
  be more efficient with smaller `top_k` and/or smaller number of unique
  elements in `x`. As a rule of thumb, `approximate_vocabulary` becomes more
  efficient than `tft.vocabulary` if `top_k` or the number of unique elements in
  `x` is smaller than 2*10^5. Moreover, this analyzer is subject to combiner
  packing optimization that does not apply to `tft.vocabulary`. Caching is also
  more efficient with the approximate implementation since the filtration
  happens before writing out cache. Output artifact of `approximate_vocabulary`
  is consistent with `tft.vocabulary` and can be used in `tft.apply_vocabulary`
  mapper.

  Implementation of this analyzer is based on the Misra-Gries algorithm [1]. It
  stores at most `top_k` elements with lower bound frequency estimates at a
  time. The algorithm keeps track of the approximation error `delta` such that
  for any item x with true frequency X:

              frequency[x] <= X <= frequency[x] + delta,
              delta <= (m - m') / (top_k + 1),

  where m is the total frequency of the items in the dataset and m' is the sum
  of the lower bound estimates in `frequency` [2]. For datasets that are Zipfian
  distributed with parameter `a`, the algorithm provides an expected value of
  delta = m / (top_k ^ a) [3].

  [1]
  https://www.cs.utexas.edu/users/misra/scannedPdf.dir/FindRepeatedElements.pdf
  [2] http://www.cohenwang.com/edith/bigdataclass2013/lectures/lecture1.pdf
  [3] http://dimacs.rutgers.edu/~graham/pubs/papers/countersj.pdf

  In case `file_format` is 'text' and one of the tokens contains the '\n' or
  '\r' characters or is empty it will be discarded.

  If an integer `Tensor` is provided, its semantic type should be categorical
  not a continuous/numeric, since computing a vocabulary over a continuous
  feature is not appropriate.

  The unique values are sorted by decreasing frequency and then reverse
  lexicographical order (e.g. [('a', 5), ('c', 3), ('b', 3)]). This is true even
  if `x` is numerical dtype (e.g. [('3', 5), ('2', 3), ('111', 3)]).

  Args:
    x: A categorical/discrete input `Tensor` or `CompositeTensor` with dtype
      tf.string or tf.int[8|16|32|64].
    top_k: Limit the generated vocabulary to the first `top_k` elements. Note
      that if `top_k` is larger than the number of unique elements in `x`, then
      the result will be exact.
    vocab_filename: The file name for the vocabulary file. If None, a file name
      will be chosen based on the current scope. If not None, should be unique
      within a given preprocessing function. NOTE: To make your pipelines
        resilient to implementation details please set `vocab_filename` when you
        are using the vocab_filename on a downstream component.
    store_frequency: If True, frequency of the words is stored in the vocabulary
      file. Each line in the file will be of the form 'frequency word'. NOTE: if
        this is True then the computed vocabulary cannot be used with
        `tft.apply_vocabulary` directly, since frequencies are added to the
        beginning of each row of the vocabulary, which the mapper will not
        ignore.
    weights: (Optional) Weights `Tensor` for the vocabulary. It must have the
      same shape as x.
    file_format: (Optional) A str. The format of the resulting vocabulary file.
      Accepted formats are: 'tfrecord_gzip', 'text'. 'tfrecord_gzip' requires
        tensorflow>=2.4. The default value is 'text'.
    name: (Optional) A name for this operation.

  Returns:
    The path name for the vocabulary file containing the unique values of `x`.

  Raises:
    ValueError: If `top_k` is negative.
      If `file_format` is not in the list of allowed formats.
      If x.dtype is not string or integral.
  """

  if top_k <= 0:
    raise ValueError('top_k must be positive, but got: %r' % top_k)
  elif top_k > analyzers.LARGE_VOCAB_TOP_K:
    raise ValueError('Provided top_k threshold is too large for the '
                     'approximate calculation: if the expected number of '
                     'unique elements is larger than top_k, tft.vocabulary may '
                     'be more efficient. Maximum allowed top_k is {}'.format(
                         analyzers.LARGE_VOCAB_TOP_K))

  if file_format not in analyzers.ALLOWED_VOCABULARY_FILE_FORMATS:
    raise ValueError(
        '"{}" is not an accepted file_format. It should be one of: {}'.format(
            file_format, analyzers.ALLOWED_VOCABULARY_FILE_FORMATS))

  if x.dtype != tf.string and not x.dtype.is_integer:
    raise ValueError('expected tf.string or integer but got %r' % x.dtype)

  with tf.compat.v1.name_scope(name, 'approximate_vocabulary'):
    vocabulary_key = vocab_filename
    vocab_filename = _get_approx_vocab_filename(vocab_filename, store_frequency)
    analyzer_inputs = _get_approximate_vocabulary_analyzer_inputs(
        x=x, file_format=file_format, weights=weights)
    return _approximate_vocabulary_analyzer_nodes(
        analyzer_inputs=analyzer_inputs,
        input_dtype=x.dtype.name,
        vocab_filename=vocab_filename,
        top_k=top_k,
        store_frequency=store_frequency,
        file_format=file_format,
        vocabulary_key=vocabulary_key)


def _approximate_vocabulary_analyzer_nodes(
    analyzer_inputs: Collection[tf.Tensor], input_dtype: tf.dtypes.DType,
    vocab_filename: str, top_k: int, store_frequency: bool,
    file_format: common_types.VocabularyFileFormatType,
    vocabulary_key: str) -> common_types.TemporaryAnalyzerOutputType:
  """Internal helper for analyzing vocab. See `vocabulary` doc string."""
  if (file_format == 'tfrecord_gzip' and
      not tf_utils.is_vocabulary_tfrecord_supported()):
    raise ValueError(
        'Vocabulary file_format "tfrecord_gzip" requires TF version >= 2.4')

  # TODO(b/208879020): Add vocabulary size annotation for this analyzer.
  analyzers.register_vocab(
      vocab_filename, vocabulary_key=vocabulary_key, file_format=file_format)

  outputs_value_nodes = analyzers.apply_cacheable_combine_operation(
      _VocabularyCombiner(top_k, input_dtype), *analyzer_inputs)

  flattened_outputs_value_node = nodes.apply_operation(
      analyzer_nodes.FlattenLists, *outputs_value_nodes)

  vocab_filename_node = nodes.apply_operation(
      analyzer_nodes.VocabularyOrderAndWrite,
      flattened_outputs_value_node,
      vocab_filename=vocab_filename,
      store_frequency=store_frequency,
      input_dtype=input_dtype,
      file_format=file_format,
      fingerprint_shuffle=False,
      input_is_sorted=True)

  return analyzer_nodes.wrap_as_tensor(vocab_filename_node)


class _MisraGriesSketchCoder(analyzer_nodes.CacheCoder):
  """Cache coder for the approximate vocabulary accumulator."""

  def encode_cache(self, accumulator: sketches.MisraGriesSketch) -> bytes:
    return accumulator.Serialize()

  def decode_cache(self,
                   encoded_accumulator: bytes) -> sketches.MisraGriesSketch:
    return sketches.MisraGriesSketch.Deserialize(encoded_accumulator)


class _VocabularyCombiner(analyzer_nodes.Combiner):
  """Approximately computes unique values on the PCollection."""

  def __init__(self, top_k: int, input_dtype: tf.dtypes.DType):
    self._top_k = top_k
    self._input_dtype = input_dtype

  def create_accumulator(self) -> sketches.MisraGriesSketch:
    return sketches.MisraGriesSketch(self._top_k)

  def add_input(
      self, accumulator: sketches.MisraGriesSketch,
      next_input: Tuple[np.ndarray, np.ndarray]) -> sketches.MisraGriesSketch:
    items, weights = next_input
    if items.size:
      accumulator.AddValues(pa.array(items), pa.array(weights, pa.float32()))
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[sketches.MisraGriesSketch]
  ) -> sketches.MisraGriesSketch:
    # Make sure that `accumulators` is an iterator (so that the position is
    # remembered).
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      result.Merge(accumulator)
    return result

  def extract_output(self,
                     accumulator: sketches.MisraGriesSketch) -> np.ndarray:
    estimate = accumulator.Estimate()
    estimate.validate()
    result = np.dstack(reversed(estimate.flatten()))
    if not result.size:
      return np.array(
          [[analyzers.get_empy_vocabulary_dummy_value(self._input_dtype)]],
          dtype=object)
    else:
      return result

  def output_tensor_infos(self) -> List[analyzer_nodes.TensorInfo]:
    return [analyzer_nodes.TensorInfo(tf.string, [None, 2], None)]

  @property
  def accumulator_coder(self) -> _MisraGriesSketchCoder:
    return _MisraGriesSketchCoder()


def _get_approximate_vocabulary_analyzer_inputs(
    x: common_types.TensorType,
    file_format: common_types.VocabularyFileFormatType,
    weights: Optional[common_types.TensorType] = None,
) -> Tuple[common_types.TensorType, common_types.TensorType]:
  """Helper for constructing approximate vocabulary inputs from tensors.

  Args:
    x: `Tensor` or `CompositeTensor` to compute vocabulary over.
    file_format: The format of the resulting vocabulary file.
      'tfrecord_gzip' requires tensorflow>=2.4.
    weights: Optional `Tensor` of weights.

  Returns:
    A list of batch-reduced `Tensor`s to feed to vocabulary analysis.
  """
  filter_regex = analyzers.get_vocab_newline_characters_regex(
      x.dtype, file_format)
  reduced_batch = tf_utils.reduce_batch_weighted_counts(
      x, weights=weights, force=True, filter_regex=filter_regex)
  assert reduced_batch.summed_positive_per_x_and_y is None
  if weights is None:
    assert reduced_batch.summed_weights_per_x is None
    return (reduced_batch.unique_x, reduced_batch.counts_per_x)
  else:
    return (reduced_batch.unique_x, reduced_batch.summed_weights_per_x)
