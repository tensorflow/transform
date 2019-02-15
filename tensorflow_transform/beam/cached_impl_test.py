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
"""Tests for cached tf.Transform analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
# GOOGLE-INITIALIZATION
import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import impl_helper
from tensorflow_transform import nodes
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import analysis_graph_builder
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform import test_case
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


def _write_cache(name, dataset_key, values, input_cache_dir):
  dataset_cache_dir = os.path.join(input_cache_dir, dataset_key)

  if not tf.gfile.IsDirectory(dataset_cache_dir):
    tf.gfile.MakeDirs(dataset_cache_dir)
  cache_file = '{}-00000-of-00001.gz'.format(
      os.path.join(dataset_cache_dir, name))

  with tf.io.TFRecordWriter(cache_file, 'GZIP') as writer:
    writer.write(tf.compat.as_bytes(json.dumps(values)))


def _preprocessing_fn_for_common_optimize_traversal(inputs):
  _ = tft.vocabulary(inputs['s'])
  x = inputs['x']
  x_mean = tft.mean(x, name='x')
  x_square_deviations = tf.square(x - x_mean)
  x_var = tft.mean(x_square_deviations, name='x_square_deviations')
  x_normalized = (x - x_mean) / tf.sqrt(x_var)
  return {'x_normalized': x_normalized}


def _write_cache_for_common_optimize_traversal(location, dataset_keys):
  _write_cache('__v0__CacheableCombineAccumulate--x-mean_and_var--',
               dataset_keys[0], [1., 1., 1.], location)


_OPTIMIZE_TRAVERSAL_COMMON_CASE = dict(
    testcase_name='common',
    feature_spec={
        'x': tf.FixedLenFeature([], tf.float32),
        's': tf.FixedLenFeature([], tf.string)
    },
    preprocessing_fn=_preprocessing_fn_for_common_optimize_traversal,
    write_cache_fn=_write_cache_for_common_optimize_traversal,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('vocabulary/Reshape', \<tf.Tensor 'vocabulary/Reshape:0' shape=(?,) dtype=string\>), ('x/mean_and_var/Cast', \<tf.Tensor 'x/mean_and_var/Cast:0' shape=() dtype=float32\>), ('x/mean_and_var/truediv', \<tf.Tensor 'x/mean_and_var/truediv:0' shape=() dtype=float32\>), ('x/mean_and_var/truediv_1', \<tf.Tensor 'x/mean_and_var/truediv_1:0' shape=() dtype=float32\>)])|label: CreateSavedModelForAnalyzerInputs[0]}"];
"ApplySavedModel[0][span-0]" [label="{ApplySavedModel|dataset_key: span-0|phase: 0|label: ApplySavedModel[0][span-0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0][span-0]";
"TensorSource[vocabulary][span-0]" [label="{ExtractFromDict|keys: ('vocabulary/Reshape',)|label: TensorSource[vocabulary][span-0]|partitionable: True}"];
"ApplySavedModel[0][span-0]" -> "TensorSource[vocabulary][span-0]";
"VocabularyAccumulate[vocabulary][span-0]" [label="{VocabularyAccumulate|vocab_ordering_type: 1|label: VocabularyAccumulate[vocabulary][span-0]|partitionable: True}"];
"TensorSource[vocabulary][span-0]" -> "VocabularyAccumulate[vocabulary][span-0]";
"WriteCache[VocabularyAccumulate[vocabulary]][span-0]" [label="{WriteCache|path: span-0/__v0__VocabularyAccumulate--vocabulary--|coder: \<_VocabularyAccumulatorCoder\>|label: WriteCache[VocabularyAccumulate[vocabulary]][span-0]|partitionable: True}"];
"VocabularyAccumulate[vocabulary][span-0]" -> "WriteCache[VocabularyAccumulate[vocabulary]][span-0]";
"ApplySavedModel[0][span-1]" [label="{ApplySavedModel|dataset_key: span-1|phase: 0|label: ApplySavedModel[0][span-1]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0][span-1]";
"TensorSource[vocabulary][span-1]" [label="{ExtractFromDict|keys: ('vocabulary/Reshape',)|label: TensorSource[vocabulary][span-1]|partitionable: True}"];
"ApplySavedModel[0][span-1]" -> "TensorSource[vocabulary][span-1]";
"VocabularyAccumulate[vocabulary][span-1]" [label="{VocabularyAccumulate|vocab_ordering_type: 1|label: VocabularyAccumulate[vocabulary][span-1]|partitionable: True}"];
"TensorSource[vocabulary][span-1]" -> "VocabularyAccumulate[vocabulary][span-1]";
"WriteCache[VocabularyAccumulate[vocabulary]][span-1]" [label="{WriteCache|path: span-1/__v0__VocabularyAccumulate--vocabulary--|coder: \<_VocabularyAccumulatorCoder\>|label: WriteCache[VocabularyAccumulate[vocabulary]][span-1]|partitionable: True}"];
"VocabularyAccumulate[vocabulary][span-1]" -> "WriteCache[VocabularyAccumulate[vocabulary]][span-1]";
"FlattenCache[VocabularyMerge[vocabulary]]" [label="{Flatten|label: FlattenCache[VocabularyMerge[vocabulary]]|partitionable: True}"];
"WriteCache[VocabularyAccumulate[vocabulary]][span-0]" -> "FlattenCache[VocabularyMerge[vocabulary]]";
"WriteCache[VocabularyAccumulate[vocabulary]][span-1]" -> "FlattenCache[VocabularyMerge[vocabulary]]";
"VocabularyMerge[vocabulary]" [label="{VocabularyMerge|vocab_ordering_type: 1|use_adjusted_mutual_info: False|min_diff_from_avg: 0.0|label: VocabularyMerge[vocabulary]}"];
"FlattenCache[VocabularyMerge[vocabulary]]" -> "VocabularyMerge[vocabulary]";
"VocabularyOrderAndFilter[vocabulary]" [label="{VocabularyOrderAndFilter|top_k: None|frequency_threshold: None|coverage_top_k: None|coverage_frequency_threshold: None|key_fn: None|label: VocabularyOrderAndFilter[vocabulary]}"];
"VocabularyMerge[vocabulary]" -> "VocabularyOrderAndFilter[vocabulary]";
"VocabularyWrite[vocabulary]" [label="{VocabularyWrite|vocab_filename: vocab_vocabulary|store_frequency: False|label: VocabularyWrite[vocabulary]}"];
"VocabularyOrderAndFilter[vocabulary]" -> "VocabularyWrite[vocabulary]";
"CreateTensorBinding[vocabulary/Placeholder]" [label="{CreateTensorBinding|tensor: vocabulary/Placeholder:0|is_asset_filepath: True|label: CreateTensorBinding[vocabulary/Placeholder]}"];
"VocabularyWrite[vocabulary]" -> "CreateTensorBinding[vocabulary/Placeholder]";
"ReadCache[CacheableCombineAccumulate[x/mean_and_var]][span-0]" [label="{ReadCache|path: span-0/__v0__CacheableCombineAccumulate--x-mean_and_var--|coder: \<JsonNumpyCacheCoder\>|label: ReadCache[CacheableCombineAccumulate[x/mean_and_var]][span-0]|partitionable: True}"];
"TensorSource[x/mean_and_var][span-1]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast', 'x/mean_and_var/truediv', 'x/mean_and_var/truediv_1')|label: TensorSource[x/mean_and_var][span-1]|partitionable: True}"];
"ApplySavedModel[0][span-1]" -> "TensorSource[x/mean_and_var][span-1]";
"CacheableCombineAccumulate[x/mean_and_var][span-1]" [label="{CacheableCombineAccumulate|combiner: \<MeanAndVarCombiner\>|label: CacheableCombineAccumulate[x/mean_and_var][span-1]|partitionable: True}"];
"TensorSource[x/mean_and_var][span-1]" -> "CacheableCombineAccumulate[x/mean_and_var][span-1]";
"WriteCache[CacheableCombineAccumulate[x/mean_and_var]][span-1]" [label="{WriteCache|path: span-1/__v0__CacheableCombineAccumulate--x-mean_and_var--|coder: \<JsonNumpyCacheCoder\>|label: WriteCache[CacheableCombineAccumulate[x/mean_and_var]][span-1]|partitionable: True}"];
"CacheableCombineAccumulate[x/mean_and_var][span-1]" -> "WriteCache[CacheableCombineAccumulate[x/mean_and_var]][span-1]";
"FlattenCache[CacheableCombineMerge[x/mean_and_var]]" [label="{Flatten|label: FlattenCache[CacheableCombineMerge[x/mean_and_var]]|partitionable: True}"];
"ReadCache[CacheableCombineAccumulate[x/mean_and_var]][span-0]" -> "FlattenCache[CacheableCombineMerge[x/mean_and_var]]";
"WriteCache[CacheableCombineAccumulate[x/mean_and_var]][span-1]" -> "FlattenCache[CacheableCombineMerge[x/mean_and_var]]";
"CacheableCombineMerge[x/mean_and_var]" [label="{CacheableCombineMerge|combiner: \<MeanAndVarCombiner\>|label: CacheableCombineMerge[x/mean_and_var]|{<0>0|<1>1}}"];
"FlattenCache[CacheableCombineMerge[x/mean_and_var]]" -> "CacheableCombineMerge[x/mean_and_var]";
"CreateTensorBinding[x/mean_and_var/Placeholder]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder]}"];
"CacheableCombineMerge[x/mean_and_var]":0 -> "CreateTensorBinding[x/mean_and_var/Placeholder]";
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder_1:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder_1]}"];
"CacheableCombineMerge[x/mean_and_var]":1 -> "CreateTensorBinding[x/mean_and_var/Placeholder_1]";
"CreateSavedModelForAnalyzerInputs[1]" [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x_square_deviations/mean_and_var/Cast', \<tf.Tensor 'x_square_deviations/mean_and_var/Cast:0' shape=() dtype=float32\>), ('x_square_deviations/mean_and_var/truediv', \<tf.Tensor 'x_square_deviations/mean_and_var/truediv:0' shape=() dtype=float32\>), ('x_square_deviations/mean_and_var/truediv_1', \<tf.Tensor 'x_square_deviations/mean_and_var/truediv_1:0' shape=() dtype=float32\>)])|label: CreateSavedModelForAnalyzerInputs[1]}"];
"CreateTensorBinding[vocabulary/Placeholder]" -> "CreateSavedModelForAnalyzerInputs[1]";
"CreateTensorBinding[x/mean_and_var/Placeholder]" -> "CreateSavedModelForAnalyzerInputs[1]";
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" -> "CreateSavedModelForAnalyzerInputs[1]";
"ApplySavedModel[1]" [label="{ApplySavedModel|dataset_key: None|phase: 1|label: ApplySavedModel[1]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[1]" -> "ApplySavedModel[1]";
"TensorSource[x_square_deviations/mean_and_var]" [label="{ExtractFromDict|keys: ('x_square_deviations/mean_and_var/Cast', 'x_square_deviations/mean_and_var/truediv', 'x_square_deviations/mean_and_var/truediv_1')|label: TensorSource[x_square_deviations/mean_and_var]|partitionable: True}"];
"ApplySavedModel[1]" -> "TensorSource[x_square_deviations/mean_and_var]";
"CacheableCombineAccumulate[x_square_deviations/mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<MeanAndVarCombiner\>|label: CacheableCombineAccumulate[x_square_deviations/mean_and_var]|partitionable: True}"];
"TensorSource[x_square_deviations/mean_and_var]" -> "CacheableCombineAccumulate[x_square_deviations/mean_and_var]";
"CacheableCombineMerge[x_square_deviations/mean_and_var]" [label="{CacheableCombineMerge|combiner: \<MeanAndVarCombiner\>|label: CacheableCombineMerge[x_square_deviations/mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineAccumulate[x_square_deviations/mean_and_var]" -> "CacheableCombineMerge[x_square_deviations/mean_and_var]";
"CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder]" [label="{CreateTensorBinding|tensor: x_square_deviations/mean_and_var/Placeholder:0|is_asset_filepath: False|label: CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder]}"];
"CacheableCombineMerge[x_square_deviations/mean_and_var]":0 -> "CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder]";
"CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder_1]" [label="{CreateTensorBinding|tensor: x_square_deviations/mean_and_var/Placeholder_1:0|is_asset_filepath: False|label: CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder_1]}"];
"CacheableCombineMerge[x_square_deviations/mean_and_var]":1 -> "CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder_1]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x_normalized', \<tf.Tensor 'truediv:0' shape=(?,) dtype=float32\>)])|label: CreateSavedModel}"];
"CreateTensorBinding[vocabulary/Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x/mean_and_var/Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder_1]" -> CreateSavedModel;
}
""")


def _preprocessing_fn_for_generalized_chained_ptransforms(inputs):

  class FakeChainablePartitionable(
      collections.namedtuple('FakeChainablePartitionable', ['label']),
      nodes.OperationDef):

    def __new__(cls, label=None):
      if label is None:
        scope = tf.get_default_graph().get_name_scope()
        label = '{}[{}]'.format(cls.__name__, scope)
      return super(FakeChainablePartitionable, cls).__new__(cls, label=label)

    @property
    def num_outputs(self):
      return 1

    @property
    def is_partitionable(self):
      return True

  class FakeChainableCacheable(
      collections.namedtuple('FakeChainableCacheable', ['label']),
      nodes.OperationDef):

    def __new__(cls, label=None):
      if label is None:
        scope = tf.get_default_graph().get_name_scope()
        label = '{}[{}]'.format(cls.__name__, scope)
      return super(FakeChainableCacheable, cls).__new__(cls, label=label)

    @property
    def num_outputs(self):
      return 1

    @property
    def is_partitionable(self):
      return True

    @property
    def cache_coder(self):
      return 'Not-a-coder-but-thats-ok!'

  class FakeChainable(
      collections.namedtuple('FakeChainable', ['label']), nodes.OperationDef):

    def __new__(cls, label=None):
      if label is None:
        scope = tf.get_default_graph().get_name_scope()
        label = '{}[{}]'.format(cls.__name__, scope)
      return super(FakeChainable, cls).__new__(cls, label=label)

    @property
    def num_outputs(self):
      return 1

    @property
    def is_partitionable(self):
      return False

  with tf.name_scope('x'):
    input_values_node = nodes.apply_operation(
        analyzer_nodes.TensorSource, tensors=[inputs['x']])
    with tf.name_scope('partitionable1'):
      partitionable_outputs = nodes.apply_multi_output_operation(
          FakeChainablePartitionable, input_values_node)
    with tf.name_scope('cacheable1'):
      intermediate_cached_value_node = nodes.apply_multi_output_operation(
          FakeChainableCacheable, *partitionable_outputs)
    with tf.name_scope('partitionable2'):
      partitionable_outputs = nodes.apply_multi_output_operation(
          FakeChainablePartitionable, *intermediate_cached_value_node)
    with tf.name_scope('cacheable2'):
      cached_value_node = nodes.apply_multi_output_operation(
          FakeChainableCacheable, *partitionable_outputs)
    with tf.name_scope('partitionable3'):
      output_value_node = nodes.apply_multi_output_operation(
          FakeChainablePartitionable, *cached_value_node)
    with tf.name_scope('merge'):
      output_value_node = nodes.apply_operation(FakeChainable,
                                                *output_value_node)
    with tf.name_scope('not-cacheable'):
      non_cached_output = nodes.apply_operation(FakeChainable,
                                                input_values_node)
    x_chained = analyzer_nodes.bind_future_as_tensor(
        output_value_node, analyzer_nodes.TensorInfo(tf.float32, (17, 27),
                                                     False))
    x_plain = analyzer_nodes.bind_future_as_tensor(
        non_cached_output, analyzer_nodes.TensorInfo(tf.int64, (7, 13), False))
    return {'x_chained': x_chained, 'x_plain': x_plain}


_OPTIMIZE_TRAVERSAL_GENERALIZED_CHAINED_PTRANSFORMS_CASE = dict(
    testcase_name='generalized_chained_ptransforms',
    feature_spec={'x': tf.FixedLenFeature([], tf.float32)},
    preprocessing_fn=_preprocessing_fn_for_generalized_chained_ptransforms,
    write_cache_fn=None,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('inputs/x', \<tf.Tensor 'inputs/x:0' shape=(?,) dtype=float32\>)])|label: CreateSavedModelForAnalyzerInputs[0]}"];
"ApplySavedModel[0][span-1]" [label="{ApplySavedModel|dataset_key: span-1|phase: 0|label: ApplySavedModel[0][span-1]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0][span-1]";
"TensorSource[x][span-1]" [label="{ExtractFromDict|keys: ('inputs/x',)|label: TensorSource[x][span-1]|partitionable: True}"];
"ApplySavedModel[0][span-1]" -> "TensorSource[x][span-1]";
"FakeChainablePartitionable[x/partitionable1][span-1]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable1][span-1]|partitionable: True}"];
"TensorSource[x][span-1]" -> "FakeChainablePartitionable[x/partitionable1][span-1]";
"FakeChainableCacheable[x/cacheable1][span-1]" [label="{FakeChainableCacheable|label: FakeChainableCacheable[x/cacheable1][span-1]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable1][span-1]" -> "FakeChainableCacheable[x/cacheable1][span-1]";
"WriteCache[FakeChainableCacheable[x/cacheable1]][span-1]" [label="{WriteCache|path: span-1/__v0__FakeChainableCacheable--x-cacheable1--|coder: Not-a-coder-but-thats-ok!|label: WriteCache[FakeChainableCacheable[x/cacheable1]][span-1]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable1][span-1]" -> "WriteCache[FakeChainableCacheable[x/cacheable1]][span-1]";
"FakeChainablePartitionable[x/partitionable2][span-1]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable2][span-1]|partitionable: True}"];
"WriteCache[FakeChainableCacheable[x/cacheable1]][span-1]" -> "FakeChainablePartitionable[x/partitionable2][span-1]";
"FakeChainableCacheable[x/cacheable2][span-1]" [label="{FakeChainableCacheable|label: FakeChainableCacheable[x/cacheable2][span-1]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable2][span-1]" -> "FakeChainableCacheable[x/cacheable2][span-1]";
"WriteCache[FakeChainableCacheable[x/cacheable2]][span-1]" [label="{WriteCache|path: span-1/__v0__FakeChainableCacheable--x-cacheable2--|coder: Not-a-coder-but-thats-ok!|label: WriteCache[FakeChainableCacheable[x/cacheable2]][span-1]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable2][span-1]" -> "WriteCache[FakeChainableCacheable[x/cacheable2]][span-1]";
"FakeChainablePartitionable[x/partitionable3][span-1]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable3][span-1]|partitionable: True}"];
"WriteCache[FakeChainableCacheable[x/cacheable2]][span-1]" -> "FakeChainablePartitionable[x/partitionable3][span-1]";
"ApplySavedModel[0][span-0]" [label="{ApplySavedModel|dataset_key: span-0|phase: 0|label: ApplySavedModel[0][span-0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0][span-0]";
"TensorSource[x][span-0]" [label="{ExtractFromDict|keys: ('inputs/x',)|label: TensorSource[x][span-0]|partitionable: True}"];
"ApplySavedModel[0][span-0]" -> "TensorSource[x][span-0]";
"FakeChainablePartitionable[x/partitionable1][span-0]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable1][span-0]|partitionable: True}"];
"TensorSource[x][span-0]" -> "FakeChainablePartitionable[x/partitionable1][span-0]";
"FakeChainableCacheable[x/cacheable1][span-0]" [label="{FakeChainableCacheable|label: FakeChainableCacheable[x/cacheable1][span-0]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable1][span-0]" -> "FakeChainableCacheable[x/cacheable1][span-0]";
"WriteCache[FakeChainableCacheable[x/cacheable1]][span-0]" [label="{WriteCache|path: span-0/__v0__FakeChainableCacheable--x-cacheable1--|coder: Not-a-coder-but-thats-ok!|label: WriteCache[FakeChainableCacheable[x/cacheable1]][span-0]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable1][span-0]" -> "WriteCache[FakeChainableCacheable[x/cacheable1]][span-0]";
"FakeChainablePartitionable[x/partitionable2][span-0]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable2][span-0]|partitionable: True}"];
"WriteCache[FakeChainableCacheable[x/cacheable1]][span-0]" -> "FakeChainablePartitionable[x/partitionable2][span-0]";
"FakeChainableCacheable[x/cacheable2][span-0]" [label="{FakeChainableCacheable|label: FakeChainableCacheable[x/cacheable2][span-0]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable2][span-0]" -> "FakeChainableCacheable[x/cacheable2][span-0]";
"WriteCache[FakeChainableCacheable[x/cacheable2]][span-0]" [label="{WriteCache|path: span-0/__v0__FakeChainableCacheable--x-cacheable2--|coder: Not-a-coder-but-thats-ok!|label: WriteCache[FakeChainableCacheable[x/cacheable2]][span-0]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable2][span-0]" -> "WriteCache[FakeChainableCacheable[x/cacheable2]][span-0]";
"FakeChainablePartitionable[x/partitionable3][span-0]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable3][span-0]|partitionable: True}"];
"WriteCache[FakeChainableCacheable[x/cacheable2]][span-0]" -> "FakeChainablePartitionable[x/partitionable3][span-0]";
"FlattenCache[FakeChainable[x/merge]]" [label="{Flatten|label: FlattenCache[FakeChainable[x/merge]]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable3][span-1]" -> "FlattenCache[FakeChainable[x/merge]]";
"FakeChainablePartitionable[x/partitionable3][span-0]" -> "FlattenCache[FakeChainable[x/merge]]";
"FakeChainable[x/merge]" [label="{FakeChainable|label: FakeChainable[x/merge]}"];
"FlattenCache[FakeChainable[x/merge]]" -> "FakeChainable[x/merge]";
"CreateTensorBinding[x/Placeholder]" [label="{CreateTensorBinding|tensor: x/Placeholder:0|is_asset_filepath: False|label: CreateTensorBinding[x/Placeholder]}"];
"FakeChainable[x/merge]" -> "CreateTensorBinding[x/Placeholder]";
"ApplySavedModel[0]" [label="{ApplySavedModel|dataset_key: None|phase: 0|label: ApplySavedModel[0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0]";
"TensorSource[x]" [label="{ExtractFromDict|keys: ('inputs/x',)|label: TensorSource[x]|partitionable: True}"];
"ApplySavedModel[0]" -> "TensorSource[x]";
"FakeChainable[x/not-cacheable]" [label="{FakeChainable|label: FakeChainable[x/not-cacheable]}"];
"TensorSource[x]" -> "FakeChainable[x/not-cacheable]";
"CreateTensorBinding[x/Placeholder_1]" [label="{CreateTensorBinding|tensor: x/Placeholder_1:0|is_asset_filepath: False|label: CreateTensorBinding[x/Placeholder_1]}"];
"FakeChainable[x/not-cacheable]" -> "CreateTensorBinding[x/Placeholder_1]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x_chained', \<tf.Tensor 'x/Placeholder:0' shape=(17, 27) dtype=float32\>), ('x_plain', \<tf.Tensor 'x/Placeholder_1:0' shape=(7, 13) dtype=int64\>)])|label: CreateSavedModel}"];
"CreateTensorBinding[x/Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x/Placeholder_1]" -> CreateSavedModel;
}
""")

_OPTIMIZE_TRAVERSAL_TEST_CASES = [
    _OPTIMIZE_TRAVERSAL_COMMON_CASE,
    _OPTIMIZE_TRAVERSAL_GENERALIZED_CHAINED_PTRANSFORMS_CASE,
]


class CachedImplTest(test_case.TransformTestCase):

  def setUp(self):
    super(CachedImplTest, self).setUp()
    self.base_test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

  def _make_cache_location(self,
                           input_cache_dir_name=None,
                           output_cache_dir_name=None):
    return beam_impl.CacheLocation(
        input_cache_dir=os.path.join(self.base_test_dir, input_cache_dir_name or
                                     'input_cache'),
        output_cache_dir=os.path.join(self.base_test_dir,
                                      output_cache_dir_name or 'output_cache'))

  def test_single_phase_mixed_analyzer_run_once(self):
    cache_location = self._make_cache_location()

    span_0_key = 'span-0'
    span_1_key = 'span-1'

    # TODO(b/37788560): Get these names programmatically.
    _write_cache('__v0__CacheableCombineAccumulate--x_1-mean_and_var--',
                 span_0_key, [2.0, 1.0, 9.0], cache_location.input_cache_dir)
    _write_cache('__v0__CacheableCombineAccumulate--x-x--', span_0_key,
                 [2.0, 4.0], cache_location.input_cache_dir)
    _write_cache('__v0__CacheableCombineAccumulate--y_1-mean_and_var--',
                 span_0_key, [2.0, -1.5, 6.25], cache_location.input_cache_dir)
    _write_cache('__v0__CacheableCombineAccumulate--y-y--', span_0_key,
                 [4.0, 1.0], cache_location.input_cache_dir)

    def preprocessing_fn(inputs):

      integerized_s = tft.compute_and_apply_vocabulary(inputs['s'])

      _ = tft.bucketize(inputs['x'], 2, name='bucketize')

      return {
          'integerized_s':
              integerized_s,
          'x_min':
              tft.min(inputs['x'], name='x') + tf.zeros_like(inputs['x']),
          'x_mean':
              tft.mean(inputs['x'], name='x') + tf.zeros_like(inputs['x']),
          'y_min':
              tft.min(inputs['y'], name='y') + tf.zeros_like(inputs['y']),
          'y_mean':
              tft.mean(inputs['y'], name='y') + tf.zeros_like(inputs['y']),
      }

    # Run AnalyzeAndTransform on some input data and compare with expected
    # output.
    input_data = [{'x': 12, 'y': 1, 's': 'c'}, {'x': 10, 'y': 1, 's': 'c'}]
    input_metadata = dataset_metadata.DatasetMetadata(
        dataset_schema.from_feature_spec({
            'x': tf.FixedLenFeature([], tf.float32),
            'y': tf.FixedLenFeature([], tf.float32),
            's': tf.FixedLenFeature([], tf.string),
        }))
    input_data_dict = {
        span_0_key: [{
            'x': -2,
            'y': 1,
            's': 'b',
        }, {
            'x': 4,
            'y': -4,
            's': 'b',
        }],
        span_1_key: input_data,
    }
    with beam_impl.Context(temp_dir=self.get_temp_dir()):

      flat_data = input_data_dict.values() | 'Flatten' >> beam.Flatten()

      transform_fn = ((flat_data, input_data_dict, input_metadata) |
                      (beam_impl.AnalyzeDatasetWithCache(
                          preprocessing_fn, cache_location)))

    transformed_dataset = (((input_data_dict[span_1_key], input_metadata),
                            transform_fn)
                           | beam_impl.TransformDataset())

    transformed_data, unused_transformed_metadata = transformed_dataset

    exepected_transformed_data = [
        {
            'x_mean': 6.0,
            'x_min': -2.0,
            'y_mean': -0.25,
            'y_min': -4.0,
            'integerized_s': 0,
        },
        {
            'x_mean': 6.0,
            'x_min': -2.0,
            'y_mean': -0.25,
            'y_min': -4.0,
            'integerized_s': 0,
        },
    ]
    self.assertDataCloseOrEqual(transformed_data, exepected_transformed_data)

    transform_fn_dir = os.path.join(self.base_test_dir, 'transform_fn')
    _ = transform_fn | tft_beam.WriteTransformFn(transform_fn_dir)

  def test_single_phase_run_twice(self):

    cache_location = self._make_cache_location('input_cache_1',
                                               'output_cache_1')

    span_0_key = 'span-0'
    span_1_key = 'span-1'

    def preprocessing_fn(inputs):

      _ = tft.vocabulary(inputs['s'])

      _ = tft.bucketize(inputs['x'], 2, name='bucketize')

      return {
          'x_min':
              tft.min(inputs['x'], name='x') + tf.zeros_like(inputs['x']),
          'x_mean':
              tft.mean(inputs['x'], name='x') + tf.zeros_like(inputs['x']),
          'y_min':
              tft.min(inputs['y'], name='y') + tf.zeros_like(inputs['y']),
          'y_mean':
              tft.mean(inputs['y'], name='y') + tf.zeros_like(inputs['y']),
      }

    input_data = [{'x': 12, 'y': 1, 's': 'b'}, {'x': 10, 'y': 1, 's': 'c'}]
    input_metadata = dataset_metadata.DatasetMetadata(
        dataset_schema.from_feature_spec({
            'x': tf.FixedLenFeature([], tf.float32),
            'y': tf.FixedLenFeature([], tf.float32),
            's': tf.FixedLenFeature([], tf.string),
        }))
    input_data_dict = {
        span_0_key: [{
            'x': -2,
            'y': 1,
            's': 'a',
        }, {
            'x': 4,
            'y': -4,
            's': 'a',
        }],
        span_1_key: input_data,
    }
    with beam_impl.Context(temp_dir=self.get_temp_dir()):

      flat_data = input_data_dict.values() | 'Flatten' >> beam.Flatten()

      transform_fn = ((flat_data, input_data_dict, input_metadata) |
                      (beam_impl.AnalyzeDatasetWithCache(
                          preprocessing_fn, cache_location)))

    transformed_dataset = (((input_data_dict[span_1_key], input_metadata),
                            transform_fn)
                           | beam_impl.TransformDataset())

    transformed_data, unused_transformed_metadata = transformed_dataset

    exepected_transformed_data = [
        {
            'x_mean': 6.0,
            'x_min': -2.0,
            'y_mean': -0.25,
            'y_min': -4.0,
        },
        {
            'x_mean': 6.0,
            'x_min': -2.0,
            'y_mean': -0.25,
            'y_min': -4.0,
        },
    ]
    self.assertDataCloseOrEqual(transformed_data, exepected_transformed_data)

    transform_fn_dir = os.path.join(self.base_test_dir, 'transform_fn')
    _ = transform_fn | tft_beam.WriteTransformFn(transform_fn_dir)

    for key in input_data_dict:
      key_cache_dir = os.path.join(cache_location.output_cache_dir, key)
      self.assertTrue(tf.gfile.IsDirectory(key_cache_dir))
      self.assertEqual(len(tf.gfile.ListDirectory(key_cache_dir)), 6)

    cache_location = self._make_cache_location('output_cache_1',
                                               'output_cache_2')

    with beam_impl.Context(temp_dir=self.get_temp_dir()):

      flat_data = input_data_dict.values() | 'Flatten' >> beam.Flatten()

      transform_fn = ((flat_data, input_data_dict, input_metadata) |
                      (beam_impl.AnalyzeDatasetWithCache(
                          preprocessing_fn, cache_location)))

    transformed_dataset = (((input_data_dict[span_1_key], input_metadata),
                            transform_fn)
                           | beam_impl.TransformDataset())
    transformed_data, unused_transformed_metadata = transformed_dataset
    self.assertDataCloseOrEqual(transformed_data, exepected_transformed_data)

    self.assertFalse(tf.gfile.IsDirectory(cache_location.output_cache_dir))

  def test_non_frequency_vocabulary_merge(self):
    """This test compares vocabularies produced with and without cache."""

    mi_vocab_name = 'mutual_information_vocab'
    adjusted_mi_vocab_name = 'adjusted_mutual_information_vocab'
    weighted_frequency_vocab_name = 'weighted_frequency_vocab'

    def preprocessing_fn(inputs):
      _ = tft.vocabulary(
          inputs['s'],
          labels=inputs['label'],
          store_frequency=True,
          vocab_filename=mi_vocab_name,
          min_diff_from_avg=0.1,
          use_adjusted_mutual_info=False)

      _ = tft.vocabulary(
          inputs['s'],
          labels=inputs['label'],
          store_frequency=True,
          vocab_filename=adjusted_mi_vocab_name,
          min_diff_from_avg=1.0,
          use_adjusted_mutual_info=True)

      _ = tft.vocabulary(
          inputs['s'],
          weights=inputs['weight'],
          store_frequency=True,
          vocab_filename=weighted_frequency_vocab_name,
          use_adjusted_mutual_info=False)
      return inputs

    span_0_key = 'span-0'
    span_1_key = 'span-1'

    input_data = [
        dict(s='a', weight=1, label=1),
        dict(s='a', weight=0.5, label=1),
        dict(s='b', weight=0.75, label=1),
        dict(s='b', weight=1, label=0),
    ]
    input_metadata = dataset_metadata.DatasetMetadata(
        dataset_schema.from_feature_spec({
            's': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'weight': tf.FixedLenFeature([], tf.float32),
        }))
    input_data_dict = {
        span_0_key: input_data,
        span_1_key: input_data,
    }
    with beam_impl.Context(temp_dir=self.get_temp_dir()):

      flat_data = input_data_dict.values() | 'Flatten' >> beam.Flatten()

      transform_fn_with_cache = ((flat_data, input_data_dict, input_metadata) |
                                 (beam_impl.AnalyzeDatasetWithCache(
                                     preprocessing_fn,
                                     self._make_cache_location())))

      transform_fn_no_cache = ((input_data * 2, input_metadata) |
                               (beam_impl.AnalyzeDataset(preprocessing_fn)))

    transform_fn_with_cache_dir = os.path.join(self.base_test_dir,
                                               'transform_fn_with_cache')
    _ = transform_fn_with_cache | tft_beam.WriteTransformFn(
        transform_fn_with_cache_dir)

    transform_fn_no_cache_dir = os.path.join(self.base_test_dir,
                                             'transform_fn_no_cache')
    _ = transform_fn_no_cache | tft_beam.WriteTransformFn(
        transform_fn_no_cache_dir)

    tft_output_cache = tft.TFTransformOutput(transform_fn_with_cache_dir)
    tft_output_no_cache = tft.TFTransformOutput(transform_fn_no_cache_dir)

    for vocab_filename in (mi_vocab_name, adjusted_mi_vocab_name,
                           weighted_frequency_vocab_name):
      cache_path = tft_output_cache.vocabulary_file_by_name(vocab_filename)
      no_cache_path = tft_output_no_cache.vocabulary_file_by_name(
          vocab_filename)
      with tf.gfile.Open(cache_path, 'rb') as f1, tf.gfile.Open(
          no_cache_path, 'rb') as f2:
        self.assertEqual(
            f1.readlines(), f2.readlines(),
            'vocab with cache != vocab without cache for: {}'.format(
                vocab_filename))

  @test_case.named_parameters(*_OPTIMIZE_TRAVERSAL_TEST_CASES)
  def test_optimize_traversal(self, feature_spec, preprocessing_fn,
                              write_cache_fn, expected_dot_graph_str):
    cache_location = self._make_cache_location()
    span_0_key, span_1_key = 'span-0', 'span-1'
    if write_cache_fn is not None:
      write_cache_fn(cache_location.input_cache_dir, [span_0_key, span_1_key])

    with tf.name_scope('inputs'):
      input_signature = impl_helper.feature_spec_as_batched_placeholders(
          feature_spec)
    output_signature = preprocessing_fn(input_signature)
    transform_fn_future = analysis_graph_builder.build(
        tf.get_default_graph(), input_signature, output_signature,
        {span_0_key, span_1_key}, cache_location)

    dot_string = nodes.get_dot_graph([transform_fn_future]).to_string()
    self.WriteRenderedDotFile(dot_string)

    self.assertSameElements(
        dot_string.split('\n'),
        expected_dot_graph_str.split('\n'),
        msg='Result dot graph is:\n{}'.format(dot_string))


if __name__ == '__main__':
  test_case.main()
