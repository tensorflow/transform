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
"""Tests for tensorflow_transform.analysis_graph_builder."""

import os

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import impl_helper
from tensorflow_transform import nodes
from tensorflow_transform import tf2_utils
from tensorflow_transform.beam import analysis_graph_builder
from tensorflow_transform import test_case
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple

mock = tf.compat.v1.test.mock


def _preprocessing_fn_with_no_analyzers(inputs):
  x = inputs['x']
  x_plus_1 = x + 1
  return {'x_plus_1': x_plus_1}


_NO_ANALYZERS_CASE = dict(
    testcase_name='with_no_analyzers',
    feature_spec={'x': tf.io.FixedLenFeature([], tf.float32)},
    preprocessing_fn=_preprocessing_fn_with_no_analyzers,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_plus_1', \"Tensor\<shape: [None], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
}
""",
    expected_dot_graph_str_tf2=r"""digraph G {
directed=True;
node [shape=Mrecord];
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_plus_1', \"Tensor\<shape: [None], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
}
""")


def _preprocessing_fn_with_one_analyzer(inputs):

  @tf.function
  def _plus_one(x):
    return x + 1

  x = _plus_one(inputs['x'])
  x_mean = tft.mean(x, name='x')
  x_centered = x - x_mean
  return {'x_centered': x_centered}


_ONE_ANALYZER_CASE = dict(
    testcase_name='with_one_analyzer',
    feature_spec={'x': tf.io.FixedLenFeature([], tf.float32)},
    preprocessing_fn=_preprocessing_fn_with_one_analyzer,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x/mean_and_var/Cast_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
"ApplySavedModel[Phase0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase0]";
"TensorSource[x#mean_and_var]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast_1', 'x/mean_and_var/div_no_nan', 'x/mean_and_var/div_no_nan_1', 'x/mean_and_var/zeros')|label: TensorSource[x#mean_and_var]|partitionable: True}"];
"ApplySavedModel[Phase0]" -> "TensorSource[x#mean_and_var]";
"CacheableCombineAccumulate[x#mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x#mean_and_var]|partitionable: True}"];
"TensorSource[x#mean_and_var]" -> "CacheableCombineAccumulate[x#mean_and_var]";
"CacheableCombineMerge[x#mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x#mean_and_var]}"];
"CacheableCombineAccumulate[x#mean_and_var]" -> "CacheableCombineMerge[x#mean_and_var]";
"ExtractCombineMergeOutputs[x#mean_and_var]" [label="{ExtractCombineMergeOutputs|output_tensor_info_list: [TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None), TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None)]|label: ExtractCombineMergeOutputs[x#mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineMerge[x#mean_and_var]" -> "ExtractCombineMergeOutputs[x#mean_and_var]";
"CreateTensorBinding[x#mean_and_var#Placeholder]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/Placeholder:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#Placeholder]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":0 -> "CreateTensorBinding[x#mean_and_var#Placeholder]";
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/Placeholder_1:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#Placeholder_1]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":1 -> "CreateTensorBinding[x#mean_and_var#Placeholder_1]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_centered', \"Tensor\<shape: [None], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x#mean_and_var#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" -> CreateSavedModel;
}
""",
    expected_dot_graph_str_tf2=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x/mean_and_var/Cast_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
"ApplySavedModel[Phase0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase0]";
"TensorSource[x#mean_and_var]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast_1', 'x/mean_and_var/div_no_nan', 'x/mean_and_var/div_no_nan_1', 'x/mean_and_var/zeros')|label: TensorSource[x#mean_and_var]|partitionable: True}"];
"ApplySavedModel[Phase0]" -> "TensorSource[x#mean_and_var]";
"CacheableCombineAccumulate[x#mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x#mean_and_var]|partitionable: True}"];
"TensorSource[x#mean_and_var]" -> "CacheableCombineAccumulate[x#mean_and_var]";
"CacheableCombineMerge[x#mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x#mean_and_var]}"];
"CacheableCombineAccumulate[x#mean_and_var]" -> "CacheableCombineMerge[x#mean_and_var]";
"ExtractCombineMergeOutputs[x#mean_and_var]" [label="{ExtractCombineMergeOutputs|output_tensor_info_list: [TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None), TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None)]|label: ExtractCombineMergeOutputs[x#mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineMerge[x#mean_and_var]" -> "ExtractCombineMergeOutputs[x#mean_and_var]";
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/temporary_analyzer_output/PlaceholderWithDefault:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":0 -> "CreateTensorBinding[x#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]";
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/temporary_analyzer_output_1/PlaceholderWithDefault:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":1 -> "CreateTensorBinding[x#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_centered', \"Tensor\<shape: [None], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]" -> CreateSavedModel;
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]" -> CreateSavedModel;
}
""")


def _preprocessing_fn_with_table(inputs):
  x = inputs['x']
  x_vocab = tft.vocabulary(x, name='x')
  initializer = tf.lookup.TextFileInitializer(
      x_vocab,
      key_dtype=tf.string,
      key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
      value_dtype=tf.int64,
      value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  table = tf.lookup.StaticHashTable(initializer, default_value=-1)
  x_integerized = table.lookup(x)
  return {'x_integerized': x_integerized}


_WITH_TABLE_CASE = dict(
    testcase_name='with_table',
    feature_spec={'x': tf.io.FixedLenFeature([], tf.string)},
    preprocessing_fn=_preprocessing_fn_with_table,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x/boolean_mask/GatherV2', \"Tensor\<shape: [None], \<dtype: 'string'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
"ApplySavedModel[Phase0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase0]";
"TensorSource[x]" [label="{ExtractFromDict|keys: ('x/boolean_mask/GatherV2',)|label: TensorSource[x]|partitionable: True}"];
"ApplySavedModel[Phase0]" -> "TensorSource[x]";
"VocabularyAccumulate[x]" [label="{VocabularyAccumulate|vocab_ordering_type: 1|input_dtype: string|label: VocabularyAccumulate[x]|partitionable: True}"];
"TensorSource[x]" -> "VocabularyAccumulate[x]";
"VocabularyMerge[x]" [label="{VocabularyMerge|vocab_ordering_type: 1|use_adjusted_mutual_info: False|min_diff_from_avg: None|label: VocabularyMerge[x]}"];
"VocabularyAccumulate[x]" -> "VocabularyMerge[x]";
"VocabularyCountUnfiltered[x]" [label="{VocabularyCount|label: VocabularyCountUnfiltered[x]}"];
"VocabularyMerge[x]" -> "VocabularyCountUnfiltered[x]";
"CreateTensorBinding[x#vocab_x_unpruned_vocab_size]" [label="{CreateTensorBinding|tensor_name: x/vocab_x_unpruned_vocab_size:0|dtype_enum: 9|is_asset_filepath: False|label: CreateTensorBinding[x#vocab_x_unpruned_vocab_size]}"];
"VocabularyCountUnfiltered[x]" -> "CreateTensorBinding[x#vocab_x_unpruned_vocab_size]";
"VocabularyPrune[x]" [label="{VocabularyPrune|top_k: None|frequency_threshold: 0|informativeness_threshold: -inf|coverage_top_k: None|coverage_frequency_threshold: 0|coverage_informativeness_threshold: -inf|key_fn: None|input_dtype: string|label: VocabularyPrune[x]}"];
"VocabularyMerge[x]" -> "VocabularyPrune[x]";
"VocabularyCountFiltered[x]" [label="{VocabularyCount|label: VocabularyCountFiltered[x]}"];
"VocabularyPrune[x]" -> "VocabularyCountFiltered[x]";
"CreateTensorBinding[x#vocab_x_pruned_vocab_size]" [label="{CreateTensorBinding|tensor_name: x/vocab_x_pruned_vocab_size:0|dtype_enum: 9|is_asset_filepath: False|label: CreateTensorBinding[x#vocab_x_pruned_vocab_size]}"];
"VocabularyCountFiltered[x]" -> "CreateTensorBinding[x#vocab_x_pruned_vocab_size]";
"VocabularyOrderAndWrite[x]" [label="{VocabularyOrderAndWrite|vocab_filename: vocab_x|store_frequency: False|input_dtype: string|label: VocabularyOrderAndWrite[x]|fingerprint_shuffle: False|file_format: text|input_is_sorted: False}"];
"VocabularyPrune[x]" -> "VocabularyOrderAndWrite[x]";
"CreateTensorBinding[x#Placeholder]" [label="{CreateTensorBinding|tensor_name: x/Placeholder:0|dtype_enum: 7|is_asset_filepath: True|label: CreateTensorBinding[x#Placeholder]}"];
"VocabularyOrderAndWrite[x]" -> "CreateTensorBinding[x#Placeholder]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 1|output_signature: OrderedDict([('x_integerized', \"Tensor\<shape: [None], \<dtype: 'int64'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x#vocab_x_unpruned_vocab_size]" -> CreateSavedModel;
"CreateTensorBinding[x#vocab_x_pruned_vocab_size]" -> CreateSavedModel;
"CreateTensorBinding[x#Placeholder]" -> CreateSavedModel;
}
""",
    expected_dot_graph_str_tf2=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x/boolean_mask/GatherV2', \"Tensor\<shape: [None], \<dtype: 'string'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
"ApplySavedModel[Phase0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase0]";
"TensorSource[x]" [label="{ExtractFromDict|keys: ('x/boolean_mask/GatherV2',)|label: TensorSource[x]|partitionable: True}"];
"ApplySavedModel[Phase0]" -> "TensorSource[x]";
"VocabularyAccumulate[x]" [label="{VocabularyAccumulate|vocab_ordering_type: 1|input_dtype: string|label: VocabularyAccumulate[x]|partitionable: True}"];
"TensorSource[x]" -> "VocabularyAccumulate[x]";
"VocabularyMerge[x]" [label="{VocabularyMerge|vocab_ordering_type: 1|use_adjusted_mutual_info: False|min_diff_from_avg: None|label: VocabularyMerge[x]}"];
"VocabularyAccumulate[x]" -> "VocabularyMerge[x]";
"VocabularyCountUnfiltered[x]" [label="{VocabularyCount|label: VocabularyCountUnfiltered[x]}"];
"VocabularyMerge[x]" -> "VocabularyCountUnfiltered[x]";
"CreateTensorBinding[x#temporary_analyzer_output#vocab_x_unpruned_vocab_size]" [label="{CreateTensorBinding|tensor_name: x/temporary_analyzer_output/vocab_x_unpruned_vocab_size:0|dtype_enum: 9|is_asset_filepath: False|label: CreateTensorBinding[x#temporary_analyzer_output#vocab_x_unpruned_vocab_size]}"];
"VocabularyCountUnfiltered[x]" -> "CreateTensorBinding[x#temporary_analyzer_output#vocab_x_unpruned_vocab_size]";
"VocabularyPrune[x]" [label="{VocabularyPrune|top_k: None|frequency_threshold: 0|informativeness_threshold: -inf|coverage_top_k: None|coverage_frequency_threshold: 0|coverage_informativeness_threshold: -inf|key_fn: None|input_dtype: string|label: VocabularyPrune[x]}"];
"VocabularyMerge[x]" -> "VocabularyPrune[x]";
"VocabularyCountFiltered[x]" [label="{VocabularyCount|label: VocabularyCountFiltered[x]}"];
"VocabularyPrune[x]" -> "VocabularyCountFiltered[x]";
"CreateTensorBinding[x#temporary_analyzer_output_1#vocab_x_pruned_vocab_size]" [label="{CreateTensorBinding|tensor_name: x/temporary_analyzer_output_1/vocab_x_pruned_vocab_size:0|dtype_enum: 9|is_asset_filepath: False|label: CreateTensorBinding[x#temporary_analyzer_output_1#vocab_x_pruned_vocab_size]}"];
"VocabularyCountFiltered[x]" -> "CreateTensorBinding[x#temporary_analyzer_output_1#vocab_x_pruned_vocab_size]";
"VocabularyOrderAndWrite[x]" [label="{VocabularyOrderAndWrite|vocab_filename: vocab_x|store_frequency: False|input_dtype: string|label: VocabularyOrderAndWrite[x]|fingerprint_shuffle: False|file_format: text|input_is_sorted: False}"];
"VocabularyPrune[x]" -> "VocabularyOrderAndWrite[x]";
"CreateTensorBinding[x#temporary_analyzer_output_2#Const]" [label="{CreateTensorBinding|tensor_name: x/temporary_analyzer_output_2/Const:0|dtype_enum: 7|is_asset_filepath: True|label: CreateTensorBinding[x#temporary_analyzer_output_2#Const]}"];
"VocabularyOrderAndWrite[x]" -> "CreateTensorBinding[x#temporary_analyzer_output_2#Const]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 1|output_signature: OrderedDict([('x_integerized', \"Tensor\<shape: [None], \<dtype: 'int64'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x#temporary_analyzer_output#vocab_x_unpruned_vocab_size]" -> CreateSavedModel;
"CreateTensorBinding[x#temporary_analyzer_output_1#vocab_x_pruned_vocab_size]" -> CreateSavedModel;
"CreateTensorBinding[x#temporary_analyzer_output_2#Const]" -> CreateSavedModel;
}
""")


def _preprocessing_fn_with_two_phases(inputs):
  x = inputs['x']
  x_mean = tft.mean(x, name='x')
  x_square_deviations = tf.square(x - x_mean)
  x_var = tft.mean(x_square_deviations, name='x_square_deviations')
  x_normalized = (x - x_mean) / tf.sqrt(x_var)
  return {'x_normalized': x_normalized}


_TWO_PHASES_CASE = dict(
    testcase_name='with_two_phases',
    feature_spec={'x': tf.io.FixedLenFeature([], tf.float32)},
    preprocessing_fn=_preprocessing_fn_with_two_phases,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x/mean_and_var/Cast_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
"ApplySavedModel[Phase0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase0]";
"TensorSource[x#mean_and_var]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast_1', 'x/mean_and_var/div_no_nan', 'x/mean_and_var/div_no_nan_1', 'x/mean_and_var/zeros')|label: TensorSource[x#mean_and_var]|partitionable: True}"];
"ApplySavedModel[Phase0]" -> "TensorSource[x#mean_and_var]";
"CacheableCombineAccumulate[x#mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x#mean_and_var]|partitionable: True}"];
"TensorSource[x#mean_and_var]" -> "CacheableCombineAccumulate[x#mean_and_var]";
"CacheableCombineMerge[x#mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x#mean_and_var]}"];
"CacheableCombineAccumulate[x#mean_and_var]" -> "CacheableCombineMerge[x#mean_and_var]";
"ExtractCombineMergeOutputs[x#mean_and_var]" [label="{ExtractCombineMergeOutputs|output_tensor_info_list: [TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None), TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None)]|label: ExtractCombineMergeOutputs[x#mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineMerge[x#mean_and_var]" -> "ExtractCombineMergeOutputs[x#mean_and_var]";
"CreateTensorBinding[x#mean_and_var#Placeholder]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/Placeholder:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#Placeholder]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":0 -> "CreateTensorBinding[x#mean_and_var#Placeholder]";
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/Placeholder_1:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#Placeholder_1]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":1 -> "CreateTensorBinding[x#mean_and_var#Placeholder_1]";
"CreateSavedModelForAnalyzerInputs[Phase1]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_square_deviations/mean_and_var/Cast_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/div_no_nan', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/div_no_nan_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase1]}"];
"CreateTensorBinding[x#mean_and_var#Placeholder]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"ApplySavedModel[Phase1]" [label="{ApplySavedModel|phase: 1|label: ApplySavedModel[Phase1]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase1]" -> "ApplySavedModel[Phase1]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase1]";
"TensorSource[x_square_deviations#mean_and_var]" [label="{ExtractFromDict|keys: ('x_square_deviations/mean_and_var/Cast_1', 'x_square_deviations/mean_and_var/div_no_nan', 'x_square_deviations/mean_and_var/div_no_nan_1', 'x_square_deviations/mean_and_var/zeros')|label: TensorSource[x_square_deviations#mean_and_var]|partitionable: True}"];
"ApplySavedModel[Phase1]" -> "TensorSource[x_square_deviations#mean_and_var]";
"CacheableCombineAccumulate[x_square_deviations#mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x_square_deviations#mean_and_var]|partitionable: True}"];
"TensorSource[x_square_deviations#mean_and_var]" -> "CacheableCombineAccumulate[x_square_deviations#mean_and_var]";
"CacheableCombineMerge[x_square_deviations#mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x_square_deviations#mean_and_var]}"];
"CacheableCombineAccumulate[x_square_deviations#mean_and_var]" -> "CacheableCombineMerge[x_square_deviations#mean_and_var]";
"ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]" [label="{ExtractCombineMergeOutputs|output_tensor_info_list: [TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None), TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None)]|label: ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineMerge[x_square_deviations#mean_and_var]" -> "ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]";
"CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder]" [label="{CreateTensorBinding|tensor_name: x_square_deviations/mean_and_var/Placeholder:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder]}"];
"ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]":0 -> "CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder]";
"CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder_1]" [label="{CreateTensorBinding|tensor_name: x_square_deviations/mean_and_var/Placeholder_1:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder_1]}"];
"ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]":1 -> "CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder_1]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_normalized', \"Tensor\<shape: [None], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x#mean_and_var#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder_1]" -> CreateSavedModel;
}
""",
    expected_dot_graph_str_tf2=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x/mean_and_var/Cast_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
"ApplySavedModel[Phase0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase0]";
"TensorSource[x#mean_and_var]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast_1', 'x/mean_and_var/div_no_nan', 'x/mean_and_var/div_no_nan_1', 'x/mean_and_var/zeros')|label: TensorSource[x#mean_and_var]|partitionable: True}"];
"ApplySavedModel[Phase0]" -> "TensorSource[x#mean_and_var]";
"CacheableCombineAccumulate[x#mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x#mean_and_var]|partitionable: True}"];
"TensorSource[x#mean_and_var]" -> "CacheableCombineAccumulate[x#mean_and_var]";
"CacheableCombineMerge[x#mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x#mean_and_var]}"];
"CacheableCombineAccumulate[x#mean_and_var]" -> "CacheableCombineMerge[x#mean_and_var]";
"ExtractCombineMergeOutputs[x#mean_and_var]" [label="{ExtractCombineMergeOutputs|output_tensor_info_list: [TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None), TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None)]|label: ExtractCombineMergeOutputs[x#mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineMerge[x#mean_and_var]" -> "ExtractCombineMergeOutputs[x#mean_and_var]";
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/temporary_analyzer_output/PlaceholderWithDefault:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":0 -> "CreateTensorBinding[x#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]";
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/temporary_analyzer_output_1/PlaceholderWithDefault:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":1 -> "CreateTensorBinding[x#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]";
"CreateSavedModelForAnalyzerInputs[Phase1]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_square_deviations/mean_and_var/Cast_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/div_no_nan', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/div_no_nan_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase1]}"];
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"ApplySavedModel[Phase1]" [label="{ApplySavedModel|phase: 1|label: ApplySavedModel[Phase1]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase1]" -> "ApplySavedModel[Phase1]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase1]";
"TensorSource[x_square_deviations#mean_and_var]" [label="{ExtractFromDict|keys: ('x_square_deviations/mean_and_var/Cast_1', 'x_square_deviations/mean_and_var/div_no_nan', 'x_square_deviations/mean_and_var/div_no_nan_1', 'x_square_deviations/mean_and_var/zeros')|label: TensorSource[x_square_deviations#mean_and_var]|partitionable: True}"];
"ApplySavedModel[Phase1]" -> "TensorSource[x_square_deviations#mean_and_var]";
"CacheableCombineAccumulate[x_square_deviations#mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x_square_deviations#mean_and_var]|partitionable: True}"];
"TensorSource[x_square_deviations#mean_and_var]" -> "CacheableCombineAccumulate[x_square_deviations#mean_and_var]";
"CacheableCombineMerge[x_square_deviations#mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x_square_deviations#mean_and_var]}"];
"CacheableCombineAccumulate[x_square_deviations#mean_and_var]" -> "CacheableCombineMerge[x_square_deviations#mean_and_var]";
"ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]" [label="{ExtractCombineMergeOutputs|output_tensor_info_list: [TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None), TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None)]|label: ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineMerge[x_square_deviations#mean_and_var]" -> "ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]";
"CreateTensorBinding[x_square_deviations#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]" [label="{CreateTensorBinding|tensor_name: x_square_deviations/mean_and_var/temporary_analyzer_output/PlaceholderWithDefault:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x_square_deviations#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]}"];
"ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]":0 -> "CreateTensorBinding[x_square_deviations#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]";
"CreateTensorBinding[x_square_deviations#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]" [label="{CreateTensorBinding|tensor_name: x_square_deviations/mean_and_var/temporary_analyzer_output_1/PlaceholderWithDefault:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x_square_deviations#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]}"];
"ExtractCombineMergeOutputs[x_square_deviations#mean_and_var]":1 -> "CreateTensorBinding[x_square_deviations#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_normalized', \"Tensor\<shape: [None], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]" -> CreateSavedModel;
"CreateTensorBinding[x#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations#mean_and_var#temporary_analyzer_output#PlaceholderWithDefault]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations#mean_and_var#temporary_analyzer_output_1#PlaceholderWithDefault]" -> CreateSavedModel;
}
""")


def _preprocessing_fn_with_chained_ptransforms(inputs):

  class FakeChainable(
      tfx_namedtuple.namedtuple('FakeChainable', ['label']),
      nodes.OperationDef):

    def __new__(cls):
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
      return super(FakeChainable, cls).__new__(cls, label=label)

  with tf.compat.v1.name_scope('x'):
    input_values_node = nodes.apply_operation(
        analyzer_nodes.TensorSource, tensors=[inputs['x']])
    with tf.compat.v1.name_scope('ptransform1'):
      intermediate_value_node = nodes.apply_operation(FakeChainable,
                                                      input_values_node)
    with tf.compat.v1.name_scope('ptransform2'):
      output_value_node = nodes.apply_operation(FakeChainable,
                                                intermediate_value_node)
    x_chained = analyzer_nodes.bind_future_as_tensor(
        output_value_node, analyzer_nodes.TensorInfo(tf.float32, (17, 27),
                                                     None))
    return {'x_chained': x_chained}


_CHAINED_PTRANSFORMS_CASE = dict(
    testcase_name='with_chained_ptransforms',
    feature_spec={'x': tf.io.FixedLenFeature([], tf.int64)},
    preprocessing_fn=_preprocessing_fn_with_chained_ptransforms,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('inputs/inputs/x_copy', \"Tensor\<shape: [None], \<dtype: 'int64'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
"ApplySavedModel[Phase0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase0]";
"TensorSource[x]" [label="{ExtractFromDict|keys: ('inputs/inputs/x_copy',)|label: TensorSource[x]|partitionable: True}"];
"ApplySavedModel[Phase0]" -> "TensorSource[x]";
"FakeChainable[x/ptransform1]" [label="{FakeChainable|label: FakeChainable[x/ptransform1]}"];
"TensorSource[x]" -> "FakeChainable[x/ptransform1]";
"FakeChainable[x/ptransform2]" [label="{FakeChainable|label: FakeChainable[x/ptransform2]}"];
"FakeChainable[x/ptransform1]" -> "FakeChainable[x/ptransform2]";
"CreateTensorBinding[x#Placeholder]" [label="{CreateTensorBinding|tensor_name: x/Placeholder:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#Placeholder]}"];
"FakeChainable[x/ptransform2]" -> "CreateTensorBinding[x#Placeholder]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_chained', \"Tensor\<shape: [17, 27], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x#Placeholder]" -> CreateSavedModel;
}
""",
    expected_dot_graph_str_tf2=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('inputs_copy', \"Tensor\<shape: [None], \<dtype: 'int64'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
"ApplySavedModel[Phase0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase0]";
"TensorSource[x]" [label="{ExtractFromDict|keys: ('inputs_copy',)|label: TensorSource[x]|partitionable: True}"];
"ApplySavedModel[Phase0]" -> "TensorSource[x]";
"FakeChainable[x/ptransform1]" [label="{FakeChainable|label: FakeChainable[x/ptransform1]}"];
"TensorSource[x]" -> "FakeChainable[x/ptransform1]";
"FakeChainable[x/ptransform2]" [label="{FakeChainable|label: FakeChainable[x/ptransform2]}"];
"FakeChainable[x/ptransform1]" -> "FakeChainable[x/ptransform2]";
"CreateTensorBinding[x#temporary_analyzer_output#PlaceholderWithDefault]" [label="{CreateTensorBinding|tensor_name: x/temporary_analyzer_output/PlaceholderWithDefault:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#temporary_analyzer_output#PlaceholderWithDefault]}"];
"FakeChainable[x/ptransform2]" -> "CreateTensorBinding[x#temporary_analyzer_output#PlaceholderWithDefault]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_chained', \"Tensor\<shape: [17, 27], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x#temporary_analyzer_output#PlaceholderWithDefault]" -> CreateSavedModel;
}
""")

_ANALYZE_TEST_CASES = [
    _NO_ANALYZERS_CASE,
    _ONE_ANALYZER_CASE,
    _WITH_TABLE_CASE,
    _TWO_PHASES_CASE,
    _CHAINED_PTRANSFORMS_CASE,
]


class AnalysisGraphBuilderTest(test_case.TransformTestCase):

  @test_case.named_parameters(
      *test_case.cross_named_parameters(_ANALYZE_TEST_CASES, [
          dict(testcase_name='tf_compat_v1', use_tf_compat_v1=True),
          dict(testcase_name='tf2', use_tf_compat_v1=False)
      ]))
  def test_build(self, feature_spec, preprocessing_fn, expected_dot_graph_str,
                 expected_dot_graph_str_tf2, use_tf_compat_v1):
    if not use_tf_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')
    specs = (
        feature_spec if use_tf_compat_v1 else
        impl_helper.get_type_specs_from_feature_specs(feature_spec))
    graph, structured_inputs, structured_outputs = (
        impl_helper.trace_preprocessing_function(
            preprocessing_fn,
            specs,
            use_tf_compat_v1=use_tf_compat_v1,
            base_temp_dir=os.path.join(self.get_temp_dir(),
                                       self._testMethodName)))
    transform_fn_future, unused_cache = analysis_graph_builder.build(
        graph, structured_inputs, structured_outputs)

    dot_string = nodes.get_dot_graph([transform_fn_future]).to_string()
    self.WriteRenderedDotFile(dot_string)
    self.assertMultiLineEqual(
        msg='Result dot graph is:\n{}'.format(dot_string),
        first=dot_string,
        second=(expected_dot_graph_str
                if use_tf_compat_v1 else expected_dot_graph_str_tf2))

  @test_case.named_parameters(*test_case.cross_named_parameters(
      [
          dict(
              testcase_name='one_dataset_cached_single_phase',
              preprocessing_fn=_preprocessing_fn_with_one_analyzer,
              full_dataset_keys=['a', 'b'],
              cached_dataset_keys=['a'],
              expected_dataset_keys=['b'],
          ),
          dict(
              testcase_name='all_datasets_cached_single_phase',
              preprocessing_fn=_preprocessing_fn_with_one_analyzer,
              full_dataset_keys=['a', 'b'],
              cached_dataset_keys=['a', 'b'],
              expected_dataset_keys=[],
          ),
          dict(
              testcase_name='mixed_single_phase',
              preprocessing_fn=lambda d: dict(  # pylint: disable=g-long-lambda
                  list(_preprocessing_fn_with_chained_ptransforms(d).items()) +
                  list(_preprocessing_fn_with_one_analyzer(d).items())),
              full_dataset_keys=['a', 'b'],
              cached_dataset_keys=['a', 'b'],
              expected_dataset_keys=['a', 'b'],
          ),
          dict(
              testcase_name='multi_phase',
              preprocessing_fn=_preprocessing_fn_with_two_phases,
              full_dataset_keys=['a', 'b'],
              cached_dataset_keys=['a', 'b'],
              expected_dataset_keys=['a', 'b'],
          )
      ],
      [
          dict(testcase_name='tf_compat_v1', use_tf_compat_v1=True),
          dict(testcase_name='tf2', use_tf_compat_v1=False)
      ]))
  def test_get_analysis_dataset_keys(self, preprocessing_fn, full_dataset_keys,
                                     cached_dataset_keys, expected_dataset_keys,
                                     use_tf_compat_v1):
    if not use_tf_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')
    full_dataset_keys = [
        analysis_graph_builder.analyzer_cache.DatasetKey(k)
        for k in full_dataset_keys
    ]
    # We force all dataset keys with entries in the cache dict will have a cache
    # hit.
    mocked_cache_entry_key = b'M'
    input_cache = {
        key: {
            mocked_cache_entry_key: 'C'
        } for key in cached_dataset_keys
    }
    feature_spec = {'x': tf.io.FixedLenFeature([], tf.float32)}
    specs = (
        feature_spec if use_tf_compat_v1 else
        impl_helper.get_type_specs_from_feature_specs(feature_spec))
    with mock.patch(
        'tensorflow_transform.beam.analysis_graph_builder.'
        'analyzer_cache.make_cache_entry_key',
        return_value=mocked_cache_entry_key):
      dataset_keys = (
          analysis_graph_builder.get_analysis_dataset_keys(
              preprocessing_fn,
              specs,
              full_dataset_keys,
              input_cache,
              force_tf_compat_v1=use_tf_compat_v1))

    dot_string = nodes.get_dot_graph([analysis_graph_builder._ANALYSIS_GRAPH
                                     ]).to_string()
    self.WriteRenderedDotFile(dot_string)
    self.assertCountEqual(expected_dataset_keys, dataset_keys)

  @test_case.named_parameters(
      dict(testcase_name='tf_compat_v1', use_tf_compat_v1=True),
      dict(testcase_name='tf2', use_tf_compat_v1=False))
  def test_get_analysis_cache_entry_keys(self, use_tf_compat_v1):
    if not use_tf_compat_v1:
      test_case.skip_if_not_tf2('Tensorflow 2.x required')
    full_dataset_keys = ['a', 'b']
    def preprocessing_fn(inputs):
      return {'x': tft.scale_to_0_1(inputs['x'])}
    mocked_cache_entry_key = 'A'
    def mocked_make_cache_entry_key(_):
      return mocked_cache_entry_key
    feature_spec = {'x': tf.io.FixedLenFeature([], tf.float32)}
    specs = (
        feature_spec if use_tf_compat_v1 else
        impl_helper.get_type_specs_from_feature_specs(feature_spec))
    with mock.patch(
        'tensorflow_transform.beam.analysis_graph_builder.'
        'analyzer_cache.make_cache_entry_key',
        side_effect=mocked_make_cache_entry_key):
      cache_entry_keys = (
          analysis_graph_builder.get_analysis_cache_entry_keys(
              preprocessing_fn,
              specs,
              full_dataset_keys,
              force_tf_compat_v1=use_tf_compat_v1))

    dot_string = nodes.get_dot_graph([analysis_graph_builder._ANALYSIS_GRAPH
                                     ]).to_string()
    self.WriteRenderedDotFile(dot_string)
    self.assertCountEqual(cache_entry_keys, [mocked_cache_entry_key])

  def test_duplicate_label_error(self):

    def _preprocessing_fn(inputs):

      class _Analyzer(
          tfx_namedtuple.namedtuple('_Analyzer', ['label']),
          nodes.OperationDef):
        pass

      input_values_node = nodes.apply_operation(
          analyzer_nodes.TensorSource, tensors=[inputs['x']])
      intermediate_value_node = nodes.apply_operation(
          _Analyzer, input_values_node, label='SameLabel')
      output_value_node = nodes.apply_operation(
          _Analyzer, intermediate_value_node, label='SameLabel')
      x_chained = analyzer_nodes.bind_future_as_tensor(
          output_value_node,
          analyzer_nodes.TensorInfo(tf.float32, (17, 27), None))
      return {'x_chained': x_chained}

    feature_spec = {'x': tf.io.FixedLenFeature([], tf.float32)}
    use_tf_compat_v1 = tf2_utils.use_tf_compat_v1(False)
    specs = (
        feature_spec if use_tf_compat_v1 else
        impl_helper.get_type_specs_from_feature_specs(feature_spec))
    graph, structured_inputs, structured_outputs = (
        impl_helper.trace_preprocessing_function(
            _preprocessing_fn,
            specs,
            use_tf_compat_v1=use_tf_compat_v1,
            base_temp_dir=os.path.join(self.get_temp_dir(),
                                       self._testMethodName)))
    with self.assertRaisesRegex(AssertionError, 'SameLabel'):
      _ = analysis_graph_builder.build(graph, structured_inputs,
                                       structured_outputs)


if __name__ == '__main__':
  test_case.main()
