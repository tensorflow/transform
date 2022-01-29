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
"""Tests for cached tf.Transform analysis."""

import functools
import os
import struct
from typing import Callable, Mapping, List
import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import common_types
from tensorflow_transform import impl_helper
from tensorflow_transform import nodes
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import analysis_graph_builder
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple

mock = tf.compat.v1.test.mock

_SINGLE_PHASE_NUM_SAVED_MODELS = 2
_ZERO_PHASE_NUM_SAVED_MODELS = 1


def _make_cache_key(cache_identifier):
  return analyzer_cache._CACHE_VERSION + cache_identifier + b'-HASH'


def _encode_vocabulary_accumulator(token_bytes, value_bytes):
  return struct.pack('qq{}s{}s'.format(len(token_bytes), len(value_bytes)),
                     len(token_bytes), len(value_bytes), token_bytes,
                     value_bytes)


def _preprocessing_fn_for_common_optimize_traversal(inputs):
  _ = tft.vocabulary(inputs['s'])
  x = inputs['x']
  x_mean = tft.mean(x, name='x')
  x_square_deviations = tf.square(x - x_mean)

  # 2nd analysis phase defined here.
  x_var = tft.mean(x_square_deviations, name='x_square_deviations')
  x_normalized = (x - x_mean) / tf.sqrt(x_var)
  return {'x_normalized': x_normalized}


_OPTIMIZE_TRAVERSAL_COMMON_CASE = dict(
    testcase_name='common',
    feature_spec={
        'x': tf.io.FixedLenFeature([], tf.float32),
        's': tf.io.FixedLenFeature([], tf.string)
    },
    preprocessing_fn=_preprocessing_fn_for_common_optimize_traversal,
    dataset_input_cache_dicts=[{
        _make_cache_key(b'CacheableCombineAccumulate[x#mean_and_var]'):
            'cache hit',
    }],
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('vocabulary/boolean_mask/GatherV2', \"Tensor\<shape: [None], \<dtype: 'string'\>\>\"), ('x/mean_and_var/Cast_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/div_no_nan_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[AnalysisIndex0]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='span-0')|label: ExtractInputForSavedModel[AnalysisIndex0]}"];
"ApplySavedModel[Phase0][AnalysisIndex0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0][AnalysisIndex0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0][AnalysisIndex0]";
"ExtractInputForSavedModel[AnalysisIndex0]" -> "ApplySavedModel[Phase0][AnalysisIndex0]";
"TensorSource[vocabulary][AnalysisIndex0]" [label="{ExtractFromDict|keys: ('vocabulary/boolean_mask/GatherV2',)|label: TensorSource[vocabulary][AnalysisIndex0]|partitionable: True}"];
"ApplySavedModel[Phase0][AnalysisIndex0]" -> "TensorSource[vocabulary][AnalysisIndex0]";
"VocabularyAccumulate[vocabulary][AnalysisIndex0]" [label="{VocabularyAccumulate|vocab_ordering_type: 1|input_dtype: string|label: VocabularyAccumulate[vocabulary][AnalysisIndex0]|partitionable: True}"];
"TensorSource[vocabulary][AnalysisIndex0]" -> "VocabularyAccumulate[vocabulary][AnalysisIndex0]";
"ExtractInputForSavedModel[AnalysisIndex1]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='span-1')|label: ExtractInputForSavedModel[AnalysisIndex1]}"];
"ApplySavedModel[Phase0][AnalysisIndex1]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0][AnalysisIndex1]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0][AnalysisIndex1]";
"ExtractInputForSavedModel[AnalysisIndex1]" -> "ApplySavedModel[Phase0][AnalysisIndex1]";
"TensorSource[vocabulary][AnalysisIndex1]" [label="{ExtractFromDict|keys: ('vocabulary/boolean_mask/GatherV2',)|label: TensorSource[vocabulary][AnalysisIndex1]|partitionable: True}"];
"ApplySavedModel[Phase0][AnalysisIndex1]" -> "TensorSource[vocabulary][AnalysisIndex1]";
"VocabularyAccumulate[vocabulary][AnalysisIndex1]" [label="{VocabularyAccumulate|vocab_ordering_type: 1|input_dtype: string|label: VocabularyAccumulate[vocabulary][AnalysisIndex1]|partitionable: True}"];
"TensorSource[vocabulary][AnalysisIndex1]" -> "VocabularyAccumulate[vocabulary][AnalysisIndex1]";
"FlattenCache[VocabularyMerge[vocabulary]]" [label="{Flatten|label: FlattenCache[VocabularyMerge[vocabulary]]|partitionable: True}"];
"VocabularyAccumulate[vocabulary][AnalysisIndex0]" -> "FlattenCache[VocabularyMerge[vocabulary]]";
"VocabularyAccumulate[vocabulary][AnalysisIndex1]" -> "FlattenCache[VocabularyMerge[vocabulary]]";
"VocabularyMerge[vocabulary]" [label="{VocabularyMerge|vocab_ordering_type: 1|use_adjusted_mutual_info: False|min_diff_from_avg: None|label: VocabularyMerge[vocabulary]}"];
"FlattenCache[VocabularyMerge[vocabulary]]" -> "VocabularyMerge[vocabulary]";
"VocabularyCountUnfiltered[vocabulary]" [label="{VocabularyCount|label: VocabularyCountUnfiltered[vocabulary]}"];
"VocabularyMerge[vocabulary]" -> "VocabularyCountUnfiltered[vocabulary]";
"CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]" [label="{CreateTensorBinding|tensor_name: vocabulary/vocab_vocabulary_unpruned_vocab_size:0|dtype_enum: 9|is_asset_filepath: False|label: CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]}"];
"VocabularyCountUnfiltered[vocabulary]" -> "CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]";
"VocabularyPrune[vocabulary]" [label="{VocabularyPrune|top_k: None|frequency_threshold: 0|informativeness_threshold: -inf|coverage_top_k: None|coverage_frequency_threshold: 0|coverage_informativeness_threshold: -inf|key_fn: None|input_dtype: string|label: VocabularyPrune[vocabulary]}"];
"VocabularyMerge[vocabulary]" -> "VocabularyPrune[vocabulary]";
"VocabularyCountFiltered[vocabulary]" [label="{VocabularyCount|label: VocabularyCountFiltered[vocabulary]}"];
"VocabularyPrune[vocabulary]" -> "VocabularyCountFiltered[vocabulary]";
"CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]" [label="{CreateTensorBinding|tensor_name: vocabulary/vocab_vocabulary_pruned_vocab_size:0|dtype_enum: 9|is_asset_filepath: False|label: CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]}"];
"VocabularyCountFiltered[vocabulary]" -> "CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]";
"VocabularyOrderAndWrite[vocabulary]" [label="{VocabularyOrderAndWrite|vocab_filename: vocab_vocabulary|store_frequency: False|input_dtype: string|label: VocabularyOrderAndWrite[vocabulary]|fingerprint_shuffle: False|file_format: text|input_is_sorted: False}"];
"VocabularyPrune[vocabulary]" -> "VocabularyOrderAndWrite[vocabulary]";
"CreateTensorBinding[vocabulary#Placeholder]" [label="{CreateTensorBinding|tensor_name: vocabulary/Placeholder:0|dtype_enum: 7|is_asset_filepath: True|label: CreateTensorBinding[vocabulary#Placeholder]}"];
"VocabularyOrderAndWrite[vocabulary]" -> "CreateTensorBinding[vocabulary#Placeholder]";
"DecodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex0]" [label="{DecodeCache|dataset_key: DatasetKey(key='span-0')|cache_key: \<bytes\>|coder: \<JsonNumpyCacheCoder\>|label: DecodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex0]|partitionable: True}"];
"TensorSource[x#mean_and_var][AnalysisIndex1]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast_1', 'x/mean_and_var/div_no_nan', 'x/mean_and_var/div_no_nan_1', 'x/mean_and_var/zeros')|label: TensorSource[x#mean_and_var][AnalysisIndex1]|partitionable: True}"];
"ApplySavedModel[Phase0][AnalysisIndex1]" -> "TensorSource[x#mean_and_var][AnalysisIndex1]";
"CacheableCombineAccumulate[x#mean_and_var][AnalysisIndex1]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x#mean_and_var][AnalysisIndex1]|partitionable: True}"];
"TensorSource[x#mean_and_var][AnalysisIndex1]" -> "CacheableCombineAccumulate[x#mean_and_var][AnalysisIndex1]";
"FlattenCache[CacheableCombineMerge[x#mean_and_var]]" [label="{Flatten|label: FlattenCache[CacheableCombineMerge[x#mean_and_var]]|partitionable: True}"];
"DecodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex0]" -> "FlattenCache[CacheableCombineMerge[x#mean_and_var]]";
"CacheableCombineAccumulate[x#mean_and_var][AnalysisIndex1]" -> "FlattenCache[CacheableCombineMerge[x#mean_and_var]]";
"CacheableCombineMerge[x#mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x#mean_and_var]}"];
"FlattenCache[CacheableCombineMerge[x#mean_and_var]]" -> "CacheableCombineMerge[x#mean_and_var]";
"ExtractCombineMergeOutputs[x#mean_and_var]" [label="{ExtractCombineMergeOutputs|output_tensor_info_list: [TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None), TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None)]|label: ExtractCombineMergeOutputs[x#mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineMerge[x#mean_and_var]" -> "ExtractCombineMergeOutputs[x#mean_and_var]";
"CreateTensorBinding[x#mean_and_var#Placeholder]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/Placeholder:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#Placeholder]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":0 -> "CreateTensorBinding[x#mean_and_var#Placeholder]";
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/Placeholder_1:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#Placeholder_1]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":1 -> "CreateTensorBinding[x#mean_and_var#Placeholder_1]";
"CreateSavedModelForAnalyzerInputs[Phase1]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_square_deviations/mean_and_var/Cast_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/div_no_nan', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/div_no_nan_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase1]}"];
"CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[vocabulary#Placeholder]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[x#mean_and_var#Placeholder]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
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
"CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]" -> CreateSavedModel;
"CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]" -> CreateSavedModel;
"CreateTensorBinding[vocabulary#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x#mean_and_var#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder_1]" -> CreateSavedModel;
"EncodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex1]" [label="{EncodeCache|coder: \<JsonNumpyCacheCoder\>|label: EncodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex1]|partitionable: True}"];
"CacheableCombineAccumulate[x#mean_and_var][AnalysisIndex1]" -> "EncodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex1]";
"EncodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex0]" [label="{EncodeCache|coder: \<_VocabularyAccumulatorCoder\>|label: EncodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex0]|partitionable: True}"];
"VocabularyAccumulate[vocabulary][AnalysisIndex0]" -> "EncodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex0]";
"EncodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex1]" [label="{EncodeCache|coder: \<_VocabularyAccumulatorCoder\>|label: EncodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex1]|partitionable: True}"];
"VocabularyAccumulate[vocabulary][AnalysisIndex1]" -> "EncodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex1]";
}
""")

_OPTIMIZE_TRAVERSAL_MULTI_PHASE_FULL_CACHE_HIT_CASE = dict(
    testcase_name='multi_phase_full_cache_coverage',
    feature_spec={
        'x': tf.io.FixedLenFeature([], tf.float32),
        's': tf.io.FixedLenFeature([], tf.string)
    },
    preprocessing_fn=_preprocessing_fn_for_common_optimize_traversal,
    dataset_input_cache_dicts=[{
        _make_cache_key(b'CacheableCombineAccumulate[x#mean_and_var]'):
            'cache hit',
        _make_cache_key(b'VocabularyAccumulate[vocabulary]'):
            'cache hit',
    }] * 2,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"DecodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex0]" [label="{DecodeCache|dataset_key: DatasetKey(key='span-0')|cache_key: \<bytes\>|coder: \<_VocabularyAccumulatorCoder\>|label: DecodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex0]|partitionable: True}"];
"DecodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex1]" [label="{DecodeCache|dataset_key: DatasetKey(key='span-1')|cache_key: \<bytes\>|coder: \<_VocabularyAccumulatorCoder\>|label: DecodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex1]|partitionable: True}"];
"FlattenCache[VocabularyMerge[vocabulary]]" [label="{Flatten|label: FlattenCache[VocabularyMerge[vocabulary]]|partitionable: True}"];
"DecodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex0]" -> "FlattenCache[VocabularyMerge[vocabulary]]";
"DecodeCache[VocabularyAccumulate[vocabulary]][AnalysisIndex1]" -> "FlattenCache[VocabularyMerge[vocabulary]]";
"VocabularyMerge[vocabulary]" [label="{VocabularyMerge|vocab_ordering_type: 1|use_adjusted_mutual_info: False|min_diff_from_avg: None|label: VocabularyMerge[vocabulary]}"];
"FlattenCache[VocabularyMerge[vocabulary]]" -> "VocabularyMerge[vocabulary]";
"VocabularyCountUnfiltered[vocabulary]" [label="{VocabularyCount|label: VocabularyCountUnfiltered[vocabulary]}"];
"VocabularyMerge[vocabulary]" -> "VocabularyCountUnfiltered[vocabulary]";
"CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]" [label="{CreateTensorBinding|tensor_name: vocabulary/vocab_vocabulary_unpruned_vocab_size:0|dtype_enum: 9|is_asset_filepath: False|label: CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]}"];
"VocabularyCountUnfiltered[vocabulary]" -> "CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]";
"VocabularyPrune[vocabulary]" [label="{VocabularyPrune|top_k: None|frequency_threshold: 0|informativeness_threshold: -inf|coverage_top_k: None|coverage_frequency_threshold: 0|coverage_informativeness_threshold: -inf|key_fn: None|input_dtype: string|label: VocabularyPrune[vocabulary]}"];
"VocabularyMerge[vocabulary]" -> "VocabularyPrune[vocabulary]";
"VocabularyCountFiltered[vocabulary]" [label="{VocabularyCount|label: VocabularyCountFiltered[vocabulary]}"];
"VocabularyPrune[vocabulary]" -> "VocabularyCountFiltered[vocabulary]";
"CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]" [label="{CreateTensorBinding|tensor_name: vocabulary/vocab_vocabulary_pruned_vocab_size:0|dtype_enum: 9|is_asset_filepath: False|label: CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]}"];
"VocabularyCountFiltered[vocabulary]" -> "CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]";
"VocabularyOrderAndWrite[vocabulary]" [label="{VocabularyOrderAndWrite|vocab_filename: vocab_vocabulary|store_frequency: False|input_dtype: string|label: VocabularyOrderAndWrite[vocabulary]|fingerprint_shuffle: False|file_format: text|input_is_sorted: False}"];
"VocabularyPrune[vocabulary]" -> "VocabularyOrderAndWrite[vocabulary]";
"CreateTensorBinding[vocabulary#Placeholder]" [label="{CreateTensorBinding|tensor_name: vocabulary/Placeholder:0|dtype_enum: 7|is_asset_filepath: True|label: CreateTensorBinding[vocabulary#Placeholder]}"];
"VocabularyOrderAndWrite[vocabulary]" -> "CreateTensorBinding[vocabulary#Placeholder]";
"DecodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex0]" [label="{DecodeCache|dataset_key: DatasetKey(key='span-0')|cache_key: \<bytes\>|coder: \<JsonNumpyCacheCoder\>|label: DecodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex0]|partitionable: True}"];
"DecodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex1]" [label="{DecodeCache|dataset_key: DatasetKey(key='span-1')|cache_key: \<bytes\>|coder: \<JsonNumpyCacheCoder\>|label: DecodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex1]|partitionable: True}"];
"FlattenCache[CacheableCombineMerge[x#mean_and_var]]" [label="{Flatten|label: FlattenCache[CacheableCombineMerge[x#mean_and_var]]|partitionable: True}"];
"DecodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex0]" -> "FlattenCache[CacheableCombineMerge[x#mean_and_var]]";
"DecodeCache[CacheableCombineAccumulate[x#mean_and_var]][AnalysisIndex1]" -> "FlattenCache[CacheableCombineMerge[x#mean_and_var]]";
"CacheableCombineMerge[x#mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x#mean_and_var]}"];
"FlattenCache[CacheableCombineMerge[x#mean_and_var]]" -> "CacheableCombineMerge[x#mean_and_var]";
"ExtractCombineMergeOutputs[x#mean_and_var]" [label="{ExtractCombineMergeOutputs|output_tensor_info_list: [TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None), TensorInfo(dtype=tf.float32, shape=(), temporary_asset_info=None)]|label: ExtractCombineMergeOutputs[x#mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineMerge[x#mean_and_var]" -> "ExtractCombineMergeOutputs[x#mean_and_var]";
"CreateTensorBinding[x#mean_and_var#Placeholder]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/Placeholder:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#Placeholder]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":0 -> "CreateTensorBinding[x#mean_and_var#Placeholder]";
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" [label="{CreateTensorBinding|tensor_name: x/mean_and_var/Placeholder_1:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#mean_and_var#Placeholder_1]}"];
"ExtractCombineMergeOutputs[x#mean_and_var]":1 -> "CreateTensorBinding[x#mean_and_var#Placeholder_1]";
"CreateSavedModelForAnalyzerInputs[Phase1]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_square_deviations/mean_and_var/Cast_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/div_no_nan', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/div_no_nan_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase1]}"];
"CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[vocabulary#Placeholder]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[x#mean_and_var#Placeholder]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" -> "CreateSavedModelForAnalyzerInputs[Phase1]";
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
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
"CreateTensorBinding[vocabulary#vocab_vocabulary_unpruned_vocab_size]" -> CreateSavedModel;
"CreateTensorBinding[vocabulary#vocab_vocabulary_pruned_vocab_size]" -> CreateSavedModel;
"CreateTensorBinding[vocabulary#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x#mean_and_var#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x#mean_and_var#Placeholder_1]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations#mean_and_var#Placeholder_1]" -> CreateSavedModel;
}
""")

_TF_VERSION_NAMED_PARAMETERS = [
    dict(testcase_name='CompatV1', use_tf_compat_v1=True),
    dict(testcase_name='V2', use_tf_compat_v1=False),
]


def _preprocessing_fn_for_generalized_chained_ptransforms(inputs):

  class FakeChainablePartitionable(
      tfx_namedtuple.namedtuple('FakeChainablePartitionable', ['label']),
      nodes.OperationDef):

    def __new__(cls):
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
      return super(FakeChainablePartitionable, cls).__new__(cls, label=label)

    @property
    def num_outputs(self):
      return 1

    @property
    def is_partitionable(self):
      return True

  class FakeChainableCacheable(
      tfx_namedtuple.namedtuple('FakeChainableCacheable', ['label']),
      nodes.OperationDef):

    def __new__(cls):
      scope = tf.compat.v1.get_default_graph().get_name_scope()
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
      tfx_namedtuple.namedtuple('FakeChainable', ['label']),
      nodes.OperationDef):

    def __new__(cls):
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      label = '{}[{}]'.format(cls.__name__, scope)
      return super(FakeChainable, cls).__new__(cls, label=label)

    @property
    def num_outputs(self):
      return 1

    @property
    def is_partitionable(self):
      return False

  with tf.compat.v1.name_scope('x'):
    input_values_node = nodes.apply_operation(
        analyzer_nodes.TensorSource, tensors=[inputs['x']])
    with tf.compat.v1.name_scope('partitionable1'):
      partitionable_outputs = nodes.apply_multi_output_operation(
          FakeChainablePartitionable, input_values_node)
    with tf.compat.v1.name_scope('cacheable1'):
      intermediate_cached_value_node = nodes.apply_multi_output_operation(
          FakeChainableCacheable, *partitionable_outputs)
    with tf.compat.v1.name_scope('partitionable2'):
      partitionable_outputs = nodes.apply_multi_output_operation(
          FakeChainablePartitionable, *intermediate_cached_value_node)
    with tf.compat.v1.name_scope('cacheable2'):
      cached_value_node = nodes.apply_multi_output_operation(
          FakeChainableCacheable, *partitionable_outputs)
    with tf.compat.v1.name_scope('partitionable3'):
      output_value_node = nodes.apply_multi_output_operation(
          FakeChainablePartitionable, *cached_value_node)
    with tf.compat.v1.name_scope('merge'):
      output_value_node = nodes.apply_operation(FakeChainable,
                                                *output_value_node)
    with tf.compat.v1.name_scope('not-cacheable'):
      non_cached_output = nodes.apply_operation(FakeChainable,
                                                input_values_node)
    x_chained = analyzer_nodes.bind_future_as_tensor(
        output_value_node, analyzer_nodes.TensorInfo(tf.float32, (17, 27),
                                                     None))
    x_plain = analyzer_nodes.bind_future_as_tensor(
        non_cached_output, analyzer_nodes.TensorInfo(tf.int64, (7, 13), None))
    return {'x_chained': x_chained, 'x_plain': x_plain}


_OPTIMIZE_TRAVERSAL_GENERALIZED_CHAINED_PTRANSFORMS_CASE = dict(
    testcase_name='generalized_chained_ptransforms',
    feature_spec={'x': tf.io.FixedLenFeature([], tf.float32)},
    preprocessing_fn=_preprocessing_fn_for_generalized_chained_ptransforms,
    dataset_input_cache_dicts=None,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[Phase0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('inputs/inputs/x_copy', \"Tensor\<shape: [None], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[Phase0]}"];
"ExtractInputForSavedModel[AnalysisIndex0]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='span-0')|label: ExtractInputForSavedModel[AnalysisIndex0]}"];
"ApplySavedModel[Phase0][AnalysisIndex0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0][AnalysisIndex0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0][AnalysisIndex0]";
"ExtractInputForSavedModel[AnalysisIndex0]" -> "ApplySavedModel[Phase0][AnalysisIndex0]";
"TensorSource[x][AnalysisIndex0]" [label="{ExtractFromDict|keys: ('inputs/inputs/x_copy',)|label: TensorSource[x][AnalysisIndex0]|partitionable: True}"];
"ApplySavedModel[Phase0][AnalysisIndex0]" -> "TensorSource[x][AnalysisIndex0]";
"FakeChainablePartitionable[x/partitionable1][AnalysisIndex0]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable1][AnalysisIndex0]|partitionable: True}"];
"TensorSource[x][AnalysisIndex0]" -> "FakeChainablePartitionable[x/partitionable1][AnalysisIndex0]";
"FakeChainableCacheable[x/cacheable1][AnalysisIndex0]" [label="{FakeChainableCacheable|label: FakeChainableCacheable[x/cacheable1][AnalysisIndex0]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable1][AnalysisIndex0]" -> "FakeChainableCacheable[x/cacheable1][AnalysisIndex0]";
"FakeChainablePartitionable[x/partitionable2][AnalysisIndex0]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable2][AnalysisIndex0]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable1][AnalysisIndex0]" -> "FakeChainablePartitionable[x/partitionable2][AnalysisIndex0]";
"FakeChainableCacheable[x/cacheable2][AnalysisIndex0]" [label="{FakeChainableCacheable|label: FakeChainableCacheable[x/cacheable2][AnalysisIndex0]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable2][AnalysisIndex0]" -> "FakeChainableCacheable[x/cacheable2][AnalysisIndex0]";
"FakeChainablePartitionable[x/partitionable3][AnalysisIndex0]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable3][AnalysisIndex0]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable2][AnalysisIndex0]" -> "FakeChainablePartitionable[x/partitionable3][AnalysisIndex0]";
"ExtractInputForSavedModel[AnalysisIndex1]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='span-1')|label: ExtractInputForSavedModel[AnalysisIndex1]}"];
"ApplySavedModel[Phase0][AnalysisIndex1]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0][AnalysisIndex1]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0][AnalysisIndex1]";
"ExtractInputForSavedModel[AnalysisIndex1]" -> "ApplySavedModel[Phase0][AnalysisIndex1]";
"TensorSource[x][AnalysisIndex1]" [label="{ExtractFromDict|keys: ('inputs/inputs/x_copy',)|label: TensorSource[x][AnalysisIndex1]|partitionable: True}"];
"ApplySavedModel[Phase0][AnalysisIndex1]" -> "TensorSource[x][AnalysisIndex1]";
"FakeChainablePartitionable[x/partitionable1][AnalysisIndex1]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable1][AnalysisIndex1]|partitionable: True}"];
"TensorSource[x][AnalysisIndex1]" -> "FakeChainablePartitionable[x/partitionable1][AnalysisIndex1]";
"FakeChainableCacheable[x/cacheable1][AnalysisIndex1]" [label="{FakeChainableCacheable|label: FakeChainableCacheable[x/cacheable1][AnalysisIndex1]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable1][AnalysisIndex1]" -> "FakeChainableCacheable[x/cacheable1][AnalysisIndex1]";
"FakeChainablePartitionable[x/partitionable2][AnalysisIndex1]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable2][AnalysisIndex1]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable1][AnalysisIndex1]" -> "FakeChainablePartitionable[x/partitionable2][AnalysisIndex1]";
"FakeChainableCacheable[x/cacheable2][AnalysisIndex1]" [label="{FakeChainableCacheable|label: FakeChainableCacheable[x/cacheable2][AnalysisIndex1]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable2][AnalysisIndex1]" -> "FakeChainableCacheable[x/cacheable2][AnalysisIndex1]";
"FakeChainablePartitionable[x/partitionable3][AnalysisIndex1]" [label="{FakeChainablePartitionable|label: FakeChainablePartitionable[x/partitionable3][AnalysisIndex1]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable2][AnalysisIndex1]" -> "FakeChainablePartitionable[x/partitionable3][AnalysisIndex1]";
"FlattenCache[FakeChainable[x/merge]]" [label="{Flatten|label: FlattenCache[FakeChainable[x/merge]]|partitionable: True}"];
"FakeChainablePartitionable[x/partitionable3][AnalysisIndex0]" -> "FlattenCache[FakeChainable[x/merge]]";
"FakeChainablePartitionable[x/partitionable3][AnalysisIndex1]" -> "FlattenCache[FakeChainable[x/merge]]";
"FakeChainable[x/merge]" [label="{FakeChainable|label: FakeChainable[x/merge]}"];
"FlattenCache[FakeChainable[x/merge]]" -> "FakeChainable[x/merge]";
"CreateTensorBinding[x#Placeholder]" [label="{CreateTensorBinding|tensor_name: x/Placeholder:0|dtype_enum: 1|is_asset_filepath: False|label: CreateTensorBinding[x#Placeholder]}"];
"FakeChainable[x/merge]" -> "CreateTensorBinding[x#Placeholder]";
"ExtractInputForSavedModel[FlattenedDataset]" [label="{ExtractInputForSavedModel|dataset_key: DatasetKey(key='FlattenedDataset')|label: ExtractInputForSavedModel[FlattenedDataset]}"];
"ApplySavedModel[Phase0]" [label="{ApplySavedModel|phase: 0|label: ApplySavedModel[Phase0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[Phase0]" -> "ApplySavedModel[Phase0]";
"ExtractInputForSavedModel[FlattenedDataset]" -> "ApplySavedModel[Phase0]";
"TensorSource[x]" [label="{ExtractFromDict|keys: ('inputs/inputs/x_copy',)|label: TensorSource[x]|partitionable: True}"];
"ApplySavedModel[Phase0]" -> "TensorSource[x]";
"FakeChainable[x/not-cacheable]" [label="{FakeChainable|label: FakeChainable[x/not-cacheable]}"];
"TensorSource[x]" -> "FakeChainable[x/not-cacheable]";
"CreateTensorBinding[x#Placeholder_1]" [label="{CreateTensorBinding|tensor_name: x/Placeholder_1:0|dtype_enum: 9|is_asset_filepath: False|label: CreateTensorBinding[x#Placeholder_1]}"];
"FakeChainable[x/not-cacheable]" -> "CreateTensorBinding[x#Placeholder_1]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_chained', \"Tensor\<shape: [17, 27], \<dtype: 'float32'\>\>\"), ('x_plain', \"Tensor\<shape: [7, 13], \<dtype: 'int64'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x#Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x#Placeholder_1]" -> CreateSavedModel;
"EncodeCache[FakeChainableCacheable[x/cacheable1]][AnalysisIndex0]" [label="{EncodeCache|coder: Not-a-coder-but-thats-ok!|label: EncodeCache[FakeChainableCacheable[x/cacheable1]][AnalysisIndex0]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable1][AnalysisIndex0]" -> "EncodeCache[FakeChainableCacheable[x/cacheable1]][AnalysisIndex0]";
"EncodeCache[FakeChainableCacheable[x/cacheable1]][AnalysisIndex1]" [label="{EncodeCache|coder: Not-a-coder-but-thats-ok!|label: EncodeCache[FakeChainableCacheable[x/cacheable1]][AnalysisIndex1]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable1][AnalysisIndex1]" -> "EncodeCache[FakeChainableCacheable[x/cacheable1]][AnalysisIndex1]";
"EncodeCache[FakeChainableCacheable[x/cacheable2]][AnalysisIndex0]" [label="{EncodeCache|coder: Not-a-coder-but-thats-ok!|label: EncodeCache[FakeChainableCacheable[x/cacheable2]][AnalysisIndex0]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable2][AnalysisIndex0]" -> "EncodeCache[FakeChainableCacheable[x/cacheable2]][AnalysisIndex0]";
"EncodeCache[FakeChainableCacheable[x/cacheable2]][AnalysisIndex1]" [label="{EncodeCache|coder: Not-a-coder-but-thats-ok!|label: EncodeCache[FakeChainableCacheable[x/cacheable2]][AnalysisIndex1]|partitionable: True}"];
"FakeChainableCacheable[x/cacheable2][AnalysisIndex1]" -> "EncodeCache[FakeChainableCacheable[x/cacheable2]][AnalysisIndex1]";
}
""")

_OPTIMIZE_TRAVERSAL_TEST_CASES = [
    _OPTIMIZE_TRAVERSAL_COMMON_CASE,
    _OPTIMIZE_TRAVERSAL_MULTI_PHASE_FULL_CACHE_HIT_CASE,
    _OPTIMIZE_TRAVERSAL_GENERALIZED_CHAINED_PTRANSFORMS_CASE,
]


def mock_out_cache_hash(test_fn):

  def _make_next_hashed_path_for_test(*unused_args):
    return b'HASH'

  def _run_test(*args, **kwargs):
    with mock.patch.object(analysis_graph_builder._OptimizeVisitor,
                           '_make_next_hashed_path',
                           _make_next_hashed_path_for_test):
      return test_fn(*args, **kwargs)

  return _run_test


class CachedImplTest(tft_unit.TransformTestCase):

  def setUp(self):
    super().setUp()
    self.base_test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._cache_dir = os.path.join(self.base_test_dir, 'cache')
    self._running_index = 0

    self._context = tft_beam.Context(temp_dir=self.get_temp_dir())
    self._context.__enter__()

  def tearDown(self):
    self._context.__exit__()
    super().tearDown()

  def _get_running_index(self):
    self._running_index += 1
    return self._running_index

  def _publish_rendered_dot_graph_file(self,
                                       preprocessing_fn,
                                       feature_spec,
                                       dataset_keys,
                                       pcoll_cache_dict,
                                       use_tf_compat_v1=True):
    specs = feature_spec
    base_temp_dir = None
    if not use_tf_compat_v1:
      specs = impl_helper.get_type_specs_from_feature_specs(specs)
      base_temp_dir = self.base_test_dir
    graph, structured_inputs, structured_outputs = (
        impl_helper.trace_preprocessing_function(
            preprocessing_fn,
            specs,
            use_tf_compat_v1=use_tf_compat_v1,
            base_temp_dir=base_temp_dir))
    transform_fn_future, cache_output_dict = analysis_graph_builder.build(
        graph, structured_inputs, structured_outputs, dataset_keys,
        pcoll_cache_dict)
    dot_string = nodes.get_dot_graph(
        [transform_fn_future] +
        sorted(cache_output_dict.values(), key=str)).to_string()
    tf.io.gfile.makedirs(self.base_test_dir)
    output_file = os.path.join(
        self.base_test_dir,
        'rendered_graph_{}.svg'.format(self._get_running_index()))
    self.WriteRenderedDotFile(dot_string, output_file=output_file)
    return dot_string

  _RunPipelineResult = tfx_namedtuple.namedtuple(  # pylint: disable=invalid-name
      '_RunPipelineResult', ['cache_output', 'pipeline'])

  def _run_pipeline(self,
                    feature_spec,
                    input_data_dict,
                    preprocessing_fn,
                    cache_dict=None,
                    should_read_cache=False,
                    datasets_to_transform=None,
                    expected_transform_data=None,
                    expected_cache=None,
                    transform_fn_output_dir=None,
                    use_tf_compat_v1=True):
    """Runs an analysis pipeline with cache.

    Args:
      feature_spec: A feature_spec for the input data.
      input_data_dict: Dict[str, List[Dict[str, primitive]]] the input data used
        for analysis.
      preprocessing_fn: The preprocessing_fn used for analysis.
      cache_dict: Dict[str, Dict[str, List[bytes]]], input cache dict. If
        provided, should_read_cache must be False.
      should_read_cache: A bool indicating if the pipeline should read cache. If
        True, cache_dict must be False.
      datasets_to_transform: List[str], list of dataset keys to transform.
      expected_transform_data: List[Dict[str, primitive]], the expected
        transformed data, should be the same for each dataset.
      expected_cache: Dict[str, Dict[str, bytes]], expected encoded cache.
      transform_fn_output_dir: A directory where the output transform_fn should
        be written to, if not provided it will not be written.
      use_tf_compat_v1: If True, TFT's public APIs (e.g. AnalyzeDataset) will
        use Tensorflow in compat.v1 mode. Defaults to `True`.

    Returns:
      A _RunPipelineResult.
    """
    input_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec))
    with self._TestPipeline() as p:
      with tft_beam.Context(force_tf_compat_v1=use_tf_compat_v1):

        # Wraps each value in input_data_dict as a PCollection.
        input_data_pcoll_dict = {}
        for a, b in input_data_dict.items():
          pcoll = p | a.key >> beam.Create(b)
          input_data_pcoll_dict[a] = pcoll

        pcoll_cache_dict = {}

        # If provided with a cache dictionary this wraps cache entries in
        # PCollections.
        if cache_dict is not None:
          assert not should_read_cache
          for dataset in cache_dict:
            cache_entry = {}
            for idx, (k, v) in enumerate(cache_dict[dataset].items()):
              cache_entry[k] = (
                  p |
                  'CreateCache[{}][{}]'.format(dataset, idx) >> beam.Create(v))
            pcoll_cache_dict[dataset] = cache_entry

        # If requested, reads cache from the test cache directory.
        if should_read_cache:
          assert cache_dict is None
          pcoll_cache_dict = p | analyzer_cache.ReadAnalysisCacheFromFS(
              self._cache_dir, list(input_data_dict.keys()))

        self._publish_rendered_dot_graph_file(
            preprocessing_fn,
            feature_spec,
            set(input_data_dict.keys()),
            pcoll_cache_dict,
            use_tf_compat_v1=use_tf_compat_v1)

        transform_fn, cache_output = (
            (input_data_pcoll_dict, pcoll_cache_dict, input_metadata)
            | 'Analyze' >> tft_beam.AnalyzeDatasetWithCache(preprocessing_fn))
        _ = (
            cache_output
            | 'WriteCache' >> analyzer_cache.WriteAnalysisCacheToFS(
                p, self._cache_dir))

        # Transforms the requested datasets.
        if datasets_to_transform is None:
          transformed_dataset = None
        else:
          flattened_transform_data = (
              [input_data_pcoll_dict[d] for d in datasets_to_transform]
              | 'FlattenTransformData' >> beam.Flatten())
          transformed_dataset = ((
              (flattened_transform_data, input_metadata), transform_fn)
                                 | 'Transform' >> tft_beam.TransformDataset())

        # Validate the transformed data is as expected. This requires providing
        # datasets_to_transform.
        if expected_transform_data is not None:
          assert transformed_dataset is not None
          transformed_data, unused_transformed_metadata = transformed_dataset
          beam_test_util.assert_that(
              transformed_data,
              beam_test_util.equal_to(expected_transform_data))

        if expected_cache is not None:
          for dataset in expected_cache:
            self.assertCountEqual(cache_output[dataset].keys(),
                                  expected_cache[dataset].keys())
            for idx, (key, value) in enumerate(expected_cache[dataset].items()):
              beam_test_util.assert_that(
                  cache_output[dataset][key],
                  beam_test_util.equal_to(value),
                  label='AssertCache[{}][{}]'.format(dataset, idx))

        # Write transform_fn if provided with an output directory.
        if transform_fn_output_dir is not None:
          _ = transform_fn | tft_beam.WriteTransformFn(transform_fn_output_dir)

        return self._RunPipelineResult(cache_output, p)

  @tft_unit.named_parameters(_TF_VERSION_NAMED_PARAMETERS)
  @mock_out_cache_hash
  def test_single_phase_mixed_analyzer_run_once(self, use_tf_compat_v1):
    if not use_tf_compat_v1:
      tft_unit.skip_if_not_tf2('Tensorflow 2.x required.')
    span_0_key = analyzer_cache.DatasetKey('span-0')
    span_1_key = analyzer_cache.DatasetKey('span-1')

    def preprocessing_fn(inputs):

      _ = tft.bucketize(inputs['x'], 2, name='bucketize')

      return {
          'integerized_s':
              tft.compute_and_apply_vocabulary(inputs['s']),
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
    input_data = [{'x': 12, 'y': 1, 's': 'd'}, {'x': 10, 'y': 1, 's': 'c'}]
    feature_spec = {
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        's': tf.io.FixedLenFeature([], tf.string),
    }
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

    cache_dict = {
        span_0_key: {
            _make_cache_key(b'CacheableCombineAccumulate[x_1#mean_and_var]'): [
                b'[2.0, 1.0, 9.0, 0.0]'
            ],
            _make_cache_key(b'CacheableCombineAccumulate[x#x]'): [
                b'[2.0, 4.0]'
            ],
            _make_cache_key(b'CacheableCombineAccumulate[y_1#mean_and_var]'): [
                b'[2.0, -1.5, 6.25, 0.0]'
            ],
            _make_cache_key(b'CacheableCombineAccumulate[y#y]'): [
                b'[4.0, 1.0]'
            ],
        },
        span_1_key: {},
    }

    expected_transformed = [
        {
            'x_mean': 6.0,
            'x_min': -2.0,
            'y_mean': -0.25,
            'y_min': -4.0,
            'integerized_s': 1,
        },
        {
            'x_mean': 6.0,
            'x_min': -2.0,
            'y_mean': -0.25,
            'y_min': -4.0,
            'integerized_s': 2,
        },
    ]

    run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        preprocessing_fn,
        cache_dict=cache_dict,
        datasets_to_transform=[span_1_key],
        expected_transform_data=expected_transformed,
        transform_fn_output_dir=os.path.join(self.base_test_dir,
                                             'transform_fn'),
        use_tf_compat_v1=use_tf_compat_v1)

    # The output cache should not have entries for the cache that is present
    # in the input cache.
    self.assertEqual(
        len(run_result.cache_output[span_0_key]),
        len(run_result.cache_output[span_1_key]) - 4)

    p = run_result.pipeline
    # 4 from analyzing 2 spans, and 2 from transform.
    self.assertMetricsCounterEqual(p.metrics, 'num_instances', 6)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_decoded', 4)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_encoded', 8)
    self.assertMetricsCounterEqual(p.metrics, 'saved_models_created',
                                   _SINGLE_PHASE_NUM_SAVED_MODELS)
    self.assertMetricsCounterEqual(
        p.metrics, 'num_packed_accumulate_combiners', 1)
    self.assertMetricsCounterEqual(
        p.metrics, 'num_packed_merge_combiners', 1)

  @tft_unit.named_parameters(_TF_VERSION_NAMED_PARAMETERS)
  def test_single_phase_run_twice(self, use_tf_compat_v1):
    if not use_tf_compat_v1:
      tft_unit.skip_if_not_tf2('Tensorflow 2.x required.')
    span_0_key = analyzer_cache.DatasetKey('span-0')
    span_1_key = analyzer_cache.DatasetKey('span-1')
    span_2_key = analyzer_cache.DatasetKey('span-2')

    def preprocessing_fn(inputs):

      _ = tft.vocabulary(inputs['s'], vocab_filename='vocab1')

      _ = tft.bucketize(inputs['x'], 2, name='bucketize')

      y_cov = tft.covariance(tf.expand_dims(inputs['y'], axis=1), tf.float32)
      return {
          'x_min':
              tft.min(inputs['x'], name='x') + tf.zeros_like(inputs['x']),
          'x_mean':
              tft.mean(inputs['x'], name='x') + tf.zeros_like(inputs['x']),
          'y_min':
              tft.min(inputs['y'], name='y') + tf.zeros_like(inputs['y']),
          'y_mean':
              tft.mean(inputs['y'], name='y') + tf.zeros_like(inputs['y']),
          'y_cov':
              tf.math.reduce_sum(y_cov) + tf.zeros_like(inputs['y']),
          's_integerized':
              tft.compute_and_apply_vocabulary(
                  inputs['s'],
                  labels=inputs['label'],
                  use_adjusted_mutual_info=True),
      }

    feature_spec = {
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        's': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    input_data_dict = {
        span_0_key: [],
        span_1_key: [{
            'x': -2,
            'y': 1,
            's': 'a',
            'label': 0,
        }, {
            'x': 4,
            'y': -4,
            's': 'a',
            'label': 1,
        }, {
            'x': 5,
            'y': 11,
            's': 'a',
            'label': 1,
        }, {
            'x': 1,
            'y': -4,
            's': u'ȟᎥ𝒋ǩľḿꞑȯ𝘱𝑞𝗋𝘴'.encode('utf-8'),
            'label': 1,
        }],
        span_2_key: [{
            'x': 12,
            'y': 1,
            's': u'ȟᎥ𝒋ǩľḿꞑȯ𝘱𝑞𝗋𝘴'.encode('utf-8'),
            'label': 0
        }, {
            'x': 10,
            'y': 1,
            's': 'c',
            'label': 1
        }],
    }
    expected_vocabulary_contents = np.array(
        [b'a', u'ȟᎥ𝒋ǩľḿꞑȯ𝘱𝑞𝗋𝘴'.encode('utf-8'), b'c'],
        dtype=object)

    expected_transformed_data = [
        {
            'x_mean': 5.0,
            'x_min': -2.0,
            'y_mean': 1.0,
            'y_min': -4.0,
            'y_cov': 25.0,
            's_integerized': 0,
        },
        {
            'x_mean': 5.0,
            'x_min': -2.0,
            'y_mean': 1.0,
            'y_min': -4.0,
            'y_cov': 25.0,
            's_integerized': 2,
        },
    ]

    transform_fn_dir = os.path.join(self.base_test_dir, 'transform_fn_1')

    first_run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        preprocessing_fn,
        datasets_to_transform=[span_2_key],
        expected_transform_data=expected_transformed_data,
        transform_fn_output_dir=transform_fn_dir,
        use_tf_compat_v1=use_tf_compat_v1)

    for key in input_data_dict:
      self.assertIn(key, first_run_result.cache_output)
      self.assertEqual(8, len(first_run_result.cache_output[key]))

    tf_transform_output = tft.TFTransformOutput(transform_fn_dir)
    vocab1_path = tf_transform_output.vocabulary_file_by_name('vocab1')
    self.AssertVocabularyContents(vocab1_path, expected_vocabulary_contents)

    p = first_run_result.pipeline
    # 6 from analyzing 3 spans, and 2 from transform.
    self.assertMetricsCounterEqual(p.metrics, 'num_instances', 8)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_decoded', 0)
    # 8 entries for each of 3 spans. Note that default values for the empty span
    # are also encoded.
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_encoded', 24)
    self.assertMetricsCounterEqual(p.metrics, 'saved_models_created',
                                   _SINGLE_PHASE_NUM_SAVED_MODELS)

    transform_fn_dir = os.path.join(self.base_test_dir, 'transform_fn_2')
    second_run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        preprocessing_fn,
        should_read_cache=True,
        datasets_to_transform=[span_2_key],
        expected_transform_data=expected_transformed_data,
        transform_fn_output_dir=transform_fn_dir,
        use_tf_compat_v1=use_tf_compat_v1)

    tf_transform_output = tft.TFTransformOutput(transform_fn_dir)
    vocab1_path = tf_transform_output.vocabulary_file_by_name('vocab1')
    self.AssertVocabularyContents(vocab1_path, expected_vocabulary_contents)

    self.assertFalse(second_run_result.cache_output)

    p = second_run_result.pipeline
    # Only 2 from transform.
    self.assertMetricsCounterEqual(p.metrics, 'num_instances', 2)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_decoded', 24)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_encoded', 0)

    # The root CreateSavedModel is optimized away because the data doesn't get
    # processed at all (only cache).
    self.assertMetricsCounterEqual(p.metrics, 'saved_models_created',
                                   _ZERO_PHASE_NUM_SAVED_MODELS)

  @tft_unit.named_parameters(_TF_VERSION_NAMED_PARAMETERS)
  @mock_out_cache_hash
  def test_caching_vocab_for_integer_categorical(self, use_tf_compat_v1):
    if not use_tf_compat_v1:
      tft_unit.skip_if_not_tf2('Tensorflow 2.x required.')
    span_0_key = analyzer_cache.DatasetKey('span-0')
    span_1_key = analyzer_cache.DatasetKey('span-1')

    def preprocessing_fn(inputs):
      return {
          'x_vocab':
              tft.compute_and_apply_vocabulary(
                  inputs['x'], frequency_threshold=2)
      }

    feature_spec = {'x': tf.io.FixedLenFeature([], tf.int64)}
    input_data_dict = {
        span_0_key: [{
            'x': -2,
        }, {
            'x': -4,
        }, {
            'x': -1,
        }, {
            'x': 4,
        }],
        span_1_key: [{
            'x': -2,
        }, {
            'x': -1,
        }, {
            'x': 6,
        }, {
            'x': 7,
        }],
    }
    expected_transformed_data = [{
        'x_vocab': 0,
    }, {
        'x_vocab': 1,
    }, {
        'x_vocab': -1,
    }, {
        'x_vocab': -1,
    }]

    cache_dict = {
        span_0_key: {
            _make_cache_key(
                b'VocabularyAccumulate[compute_and_apply_vocabulary#vocabulary]'
            ): [
                _encode_vocabulary_accumulator(b'-2', b'2'),
                _encode_vocabulary_accumulator(b'-4', b'1'),
                _encode_vocabulary_accumulator(b'-1', b'1'),
                _encode_vocabulary_accumulator(b'4', b'1'),
            ]
        },
        span_1_key: {},
    }

    run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        preprocessing_fn,
        cache_dict=cache_dict,
        datasets_to_transform=[span_1_key],
        expected_transform_data=expected_transformed_data,
        use_tf_compat_v1=use_tf_compat_v1)

    self.assertNotIn(span_0_key, run_result.cache_output)

    p = run_result.pipeline
    # 4 from analysis since 1 span was completely cached, and 4 from transform.
    self.assertMetricsCounterEqual(p.metrics, 'num_instances', 8)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_decoded', 1)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_encoded', 1)
    self.assertMetricsCounterEqual(p.metrics, 'saved_models_created',
                                   _SINGLE_PHASE_NUM_SAVED_MODELS)

  @tft_unit.named_parameters(_TF_VERSION_NAMED_PARAMETERS)
  @mock_out_cache_hash
  def test_non_frequency_vocabulary_merge(self, use_tf_compat_v1):
    """This test compares vocabularies produced with and without cache."""
    if not use_tf_compat_v1:
      tft_unit.skip_if_not_tf2('Tensorflow 2.x required.')
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
          use_adjusted_mutual_info=False,
          name='with_mi')

      _ = tft.vocabulary(
          inputs['s'],
          labels=inputs['label'],
          store_frequency=True,
          vocab_filename=adjusted_mi_vocab_name,
          min_diff_from_avg=1.0,
          use_adjusted_mutual_info=True,
          name='with_adjusted_mi')

      _ = tft.vocabulary(
          inputs['s'],
          weights=inputs['weight'],
          store_frequency=True,
          vocab_filename=weighted_frequency_vocab_name,
          use_adjusted_mutual_info=False,
          name='with_weight')
      return inputs

    span_0_key = analyzer_cache.DatasetKey('span-0')
    span_1_key = analyzer_cache.DatasetKey('span-1')

    input_data = [
        dict(s='a', weight=1, label=1),
        dict(s='a', weight=0.5, label=1),
        dict(s='b', weight=0.75, label=1),
        dict(s='b', weight=1, label=0),
    ]
    feature_spec = {
        's': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'weight': tf.io.FixedLenFeature([], tf.float32),
    }
    input_data_dict = {
        span_0_key: input_data,
        span_1_key: input_data,
    }
    transform_fn_with_cache_dir = os.path.join(self.base_test_dir,
                                               'transform_fn_with_cache')

    expected_accumulators = {
        _make_cache_key(b'VocabularyAccumulate[with_mi]'): [
            _encode_vocabulary_accumulator(b'a',
                                           b'[2, [0.0, 1.0], [0.0, 0.0], 1.0]'),
            _encode_vocabulary_accumulator(b'b',
                                           b'[2, [0.5, 0.5], [0.0, 0.0], 1.0]'),
            _encode_vocabulary_accumulator(
                b'global_y_count_sentinel',
                b'[4, [0.25, 0.75], [0.0, 0.0], 1.0]'),
        ],
        _make_cache_key(b'VocabularyAccumulate[with_adjusted_mi]'): [
            _encode_vocabulary_accumulator(b'a',
                                           b'[2, [0.0, 1.0], [0.0, 0.0], 1.0]'),
            _encode_vocabulary_accumulator(b'b',
                                           b'[2, [0.5, 0.5], [0.0, 0.0], 1.0]'),
            _encode_vocabulary_accumulator(
                b'global_y_count_sentinel',
                b'[4, [0.25, 0.75], [0.0, 0.0], 1.0]'),
        ],
        _make_cache_key(b'VocabularyAccumulate[with_weight]'): [
            _encode_vocabulary_accumulator(b'a', b'1.5'),
            _encode_vocabulary_accumulator(b'b', b'1.75')
        ],
    }
    expected_cache = {
        span: expected_accumulators for span in [span_0_key, span_1_key]
    }

    run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        preprocessing_fn,
        transform_fn_output_dir=transform_fn_with_cache_dir,
        expected_cache=expected_cache,
        use_tf_compat_v1=use_tf_compat_v1)

    p = run_result.pipeline
    # 4 from analysis on each of the input spans.
    self.assertMetricsCounterEqual(p.metrics, 'num_instances', 8)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_decoded', 0)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_encoded', 6)
    self.assertMetricsCounterEqual(p.metrics, 'saved_models_created',
                                   _SINGLE_PHASE_NUM_SAVED_MODELS)

    with self._TestPipeline() as p:
      with tft_beam.Context():
        flat_data = p | 'CreateInputData' >> beam.Create(input_data * 2)

        input_metadata = dataset_metadata.DatasetMetadata(
            schema_utils.schema_from_feature_spec(feature_spec))
        transform_fn_no_cache = ((flat_data, input_metadata)
                                 | tft_beam.AnalyzeDataset(preprocessing_fn))

        transform_fn_no_cache_dir = os.path.join(self.base_test_dir,
                                                 'transform_fn_no_cache')
        _ = transform_fn_no_cache | tft_beam.WriteTransformFn(
            transform_fn_no_cache_dir)

    # 4 from analysis on each of the input spans.
    self.assertMetricsCounterEqual(p.metrics, 'num_instances', 8)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_decoded', 0)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_encoded', 0)
    self.assertMetricsCounterEqual(p.metrics, 'saved_models_created',
                                   _SINGLE_PHASE_NUM_SAVED_MODELS)

    tft_output_cache = tft.TFTransformOutput(transform_fn_with_cache_dir)
    tft_output_no_cache = tft.TFTransformOutput(transform_fn_no_cache_dir)

    for vocab_filename in (mi_vocab_name, adjusted_mi_vocab_name,
                           weighted_frequency_vocab_name):
      cache_path = tft_output_cache.vocabulary_file_by_name(vocab_filename)
      no_cache_path = tft_output_no_cache.vocabulary_file_by_name(
          vocab_filename)
      with tf.io.gfile.GFile(cache_path, 'rb') as f1, tf.io.gfile.GFile(
          no_cache_path, 'rb') as f2:
        self.assertEqual(
            f1.readlines(), f2.readlines(),
            'vocab with cache != vocab without cache for: {}'.format(
                vocab_filename))

  @tft_unit.named_parameters(_TF_VERSION_NAMED_PARAMETERS)
  @mock_out_cache_hash
  def test_cached_ptransform_analyzer(self, use_tf_compat_v1):
    if not use_tf_compat_v1:
      tft_unit.skip_if_not_tf2('Tensorflow 2.x required.')

    class _AnalyzerMakeAccumulators(beam.PTransform):

      def expand(self, pcoll):
        input_sum = pcoll | beam.FlatMap(
            sum) | 'ReduceSum' >> beam.CombineGlobally(sum)
        size = pcoll | beam.Map(
            np.size) | 'ReduceCount' >> beam.CombineGlobally(sum)

        return (pcoll.pipeline
                | beam.Create([None])
                | beam.Map(
                    lambda _, a, b: (a, b),  # pyformat: disable
                    beam.pvalue.AsSingleton(input_sum),
                    beam.pvalue.AsSingleton(size)))

    class _AnalyzerMergeAccumulators(beam.PTransform):

      def expand(self, pcoll):

        def merge(x):
          zipped = list(zip(*x))
          assert len(zipped) == 2, zipped
          return sum(zipped[0]), sum(zipped[1])

        return pcoll | beam.CombineGlobally(merge)

    class _AnalyzerExtractOutput(beam.PTransform):

      def expand(self, pcoll):

        return pcoll | beam.Map(lambda p: p[0] / p[1])

    analyzer = tft.experimental.CacheablePTransformAnalyzer(
        make_accumulators_ptransform=_AnalyzerMakeAccumulators(),
        merge_accumulators_ptransform=_AnalyzerMergeAccumulators(),
        extract_output_ptransform=_AnalyzerExtractOutput(),
        cache_coder=tft.experimental.SimpleJsonPTransformAnalyzerCacheCoder())

    def preprocessing_fn(inputs):
      y = tft.experimental.ptransform_analyzer([inputs['x']], analyzer,
                                               [tf.int64], [[]])
      return {'y': tf.zeros_like(inputs['x']) + y}

    feature_spec = {'x': tf.io.FixedLenFeature([], tf.int64)}
    span_0_key = analyzer_cache.DatasetKey('span-0')
    input_data_dict = {span_0_key: [{'x': x} for x in range(7)]}
    expected_cache_dict = {
        span_0_key: {
            _make_cache_key(b'PTransform[ptransform#local_merge_accumulators]'):
                [b'[21, 7]'],
        },
    }
    expected_transformed_data = [{'y': 3} for _ in range(7)]
    transform_fn_dir = os.path.join(self.base_test_dir, 'transform_fn_0')
    first_run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        preprocessing_fn,
        datasets_to_transform=[span_0_key],
        expected_transform_data=expected_transformed_data,
        expected_cache=expected_cache_dict,
        transform_fn_output_dir=transform_fn_dir,
        use_tf_compat_v1=use_tf_compat_v1)
    p = first_run_result.pipeline
    # Incremented for both analysis and transform (7 * 2).
    self.assertMetricsCounterEqual(p.metrics, 'num_instances', 14)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_decoded', 0)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_encoded', 1)

    transform_fn_dir = os.path.join(self.base_test_dir, 'transform_fn_1')
    first_run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        preprocessing_fn,
        should_read_cache=True,
        datasets_to_transform=[span_0_key],
        expected_transform_data=expected_transformed_data,
        expected_cache={},
        transform_fn_output_dir=transform_fn_dir,
        use_tf_compat_v1=use_tf_compat_v1)
    p = first_run_result.pipeline
    # This time analysis is skipped due to cache, only transform dataset counts.
    self.assertMetricsCounterEqual(p.metrics, 'num_instances', 7)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_decoded', 1)
    self.assertMetricsCounterEqual(p.metrics, 'cache_entries_encoded', 0)

  @tft_unit.named_parameters(*_OPTIMIZE_TRAVERSAL_TEST_CASES)
  @mock_out_cache_hash
  def test_optimize_traversal(
      self, feature_spec: Mapping[str, common_types.FeatureSpecType],
      preprocessing_fn: Callable[[Mapping[str, common_types.TensorType]],
                                 Mapping[str, common_types.TensorType]],
      dataset_input_cache_dicts: List[Mapping[str, str]],
      expected_dot_graph_str: str):
    dataset_keys = [
        analyzer_cache.DatasetKey('span-0'),
        analyzer_cache.DatasetKey('span-1')
    ]
    if dataset_input_cache_dicts is not None:
      cache = {
          key: cache_dict
          for key, cache_dict in zip(dataset_keys, dataset_input_cache_dicts)
      }
    else:
      cache = {}
    dot_string = self._publish_rendered_dot_graph_file(preprocessing_fn,
                                                       feature_spec,
                                                       set(dataset_keys), cache)

    self.assertSameElements(
        expected_dot_graph_str.split('\n'),
        dot_string.split('\n'),
        msg='Result dot graph is:\n{}'.format(dot_string))

  def test_no_data_needed(self):
    span_0_key = analyzer_cache.DatasetKey('span-0')
    span_1_key = analyzer_cache.DatasetKey('span-1')

    def preprocessing_fn(inputs):
      return {k: tf.identity(v) for k, v in inputs.items()}

    input_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec({
            'x': tf.io.FixedLenFeature([], tf.float32),
        }))
    input_data_dict = {
        span_0_key: None,
        span_1_key: None,
    }

    with self._TestPipeline() as p:
      cache_dict = {
          span_0_key: {},
          span_1_key: {},
      }

      _, output_cache = ((input_data_dict, cache_dict, input_metadata)
                         | 'Analyze' >> tft_beam.AnalyzeDatasetWithCache(
                             preprocessing_fn, pipeline=p))
      self.assertFalse(output_cache)

  @tft_unit.named_parameters(_TF_VERSION_NAMED_PARAMETERS)
  def test_tf_function_works_with_cache(self, use_tf_compat_v1):
    if not use_tf_compat_v1:
      tft_unit.skip_if_not_tf2('Tensorflow 2.x required.')

    def preprocessing_fn(inputs, should_add_one):

      @tf.function
      def identity(x):
        if should_add_one:
          x = x + 1
        return x

      return {
          'x_mean':
              tft.mean(identity(inputs['x']), name='x') +
              tf.zeros_like(inputs['x'])
      }

    feature_spec = {'x': tf.io.FixedLenFeature([], tf.float32)}
    input_data_dict = {
        analyzer_cache.DatasetKey('span-0'): [dict(x=-2), dict(x=4)]
    }
    run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        functools.partial(preprocessing_fn, should_add_one=False),
        use_tf_compat_v1=use_tf_compat_v1)
    first_cache_output, p1 = run_result.cache_output, run_result.pipeline

    for key in input_data_dict:
      self.assertIn(key, first_cache_output)
      self.assertEqual(1, len(first_cache_output[key]))

    self.assertMetricsCounterEqual(p1.metrics, 'num_instances', 2)
    self.assertMetricsCounterEqual(p1.metrics, 'cache_entries_decoded', 0)
    self.assertMetricsCounterEqual(p1.metrics, 'cache_entries_encoded', 1)
    self.assertMetricsCounterEqual(p1.metrics, 'saved_models_created',
                                   _SINGLE_PHASE_NUM_SAVED_MODELS)

    # Cache is still valid since the contents of the tf.function are the same.
    run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        functools.partial(preprocessing_fn, should_add_one=False),
        should_read_cache=True,
        use_tf_compat_v1=use_tf_compat_v1)
    second_cache_output, p2 = run_result.cache_output, run_result.pipeline

    self.assertFalse(second_cache_output)

    self.assertMetricsCounterEqual(p2.metrics, 'num_instances', 0)
    self.assertMetricsCounterEqual(p2.metrics, 'cache_entries_decoded', 1)
    self.assertMetricsCounterEqual(p2.metrics, 'cache_entries_encoded', 0)
    self.assertMetricsCounterEqual(p2.metrics, 'saved_models_created',
                                   _ZERO_PHASE_NUM_SAVED_MODELS)

    # Modifying the tf.function contents causes cache invalidation.
    run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        functools.partial(preprocessing_fn, should_add_one=True),
        should_read_cache=True,
        use_tf_compat_v1=use_tf_compat_v1)
    third_output_cache, p3 = run_result.cache_output, run_result.pipeline

    for key in input_data_dict:
      self.assertIn(key, third_output_cache)
      self.assertEqual(1, len(third_output_cache[key]))

    self.assertMetricsCounterEqual(p3.metrics, 'num_instances', 2)
    self.assertMetricsCounterEqual(p3.metrics, 'cache_entries_decoded', 0)
    self.assertMetricsCounterEqual(p3.metrics, 'cache_entries_encoded', 1)
    self.assertMetricsCounterEqual(p3.metrics, 'saved_models_created',
                                   _SINGLE_PHASE_NUM_SAVED_MODELS)

  @tft_unit.named_parameters(_TF_VERSION_NAMED_PARAMETERS)
  def test_changing_constant_fails_cache(self, use_tf_compat_v1):
    if not use_tf_compat_v1:
      tft_unit.skip_if_not_tf2('Tensorflow 2.x required.')

    def make_preprocessing_fn(string):

      def preprocessing_fn(inputs):
        constant_str = tf.tile(tf.constant([string]), tf.shape(inputs['s']))
        joined = tf.strings.join([inputs['s'], constant_str])
        return {'id': tft.compute_and_apply_vocabulary(joined)}

      return preprocessing_fn

    feature_spec = {'s': tf.io.FixedLenFeature([], tf.string)}
    input_data_dict = {
        analyzer_cache.DatasetKey('span-0'): [dict(s='a'),
                                              dict(s='b')]
    }

    run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        make_preprocessing_fn('1st_run'),
        use_tf_compat_v1=use_tf_compat_v1)
    first_cache_output, p1 = run_result.cache_output, run_result.pipeline

    for key in input_data_dict:
      self.assertIn(key, first_cache_output)
      self.assertEqual(1, len(first_cache_output[key]))

    self.assertMetricsCounterEqual(p1.metrics, 'num_instances', 2)
    self.assertMetricsCounterEqual(p1.metrics, 'cache_entries_decoded', 0)
    self.assertMetricsCounterEqual(p1.metrics, 'cache_entries_encoded', 1)
    self.assertMetricsCounterEqual(p1.metrics, 'saved_models_created',
                                   _SINGLE_PHASE_NUM_SAVED_MODELS)

    run_result = self._run_pipeline(
        feature_spec,
        input_data_dict,
        make_preprocessing_fn('2nd_run'),
        use_tf_compat_v1=use_tf_compat_v1)
    second_cache_output, p2 = run_result.cache_output, run_result.pipeline

    # We expect a full output cache again because tf.function in the
    # preprocessing_fn broke that cache entry.
    for key in input_data_dict:
      self.assertIn(key, second_cache_output)
      self.assertEqual(1, len(second_cache_output[key]))

    self.assertMetricsCounterEqual(p2.metrics, 'num_instances', 2)
    self.assertMetricsCounterEqual(p2.metrics, 'cache_entries_decoded', 0)
    self.assertMetricsCounterEqual(p2.metrics, 'cache_entries_encoded', 1)
    self.assertMetricsCounterEqual(p2.metrics, 'saved_models_created',
                                   _SINGLE_PHASE_NUM_SAVED_MODELS)


if __name__ == '__main__':
  tft_unit.main()
