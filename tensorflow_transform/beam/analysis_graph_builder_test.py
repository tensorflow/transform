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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
# GOOGLE-INITIALIZATION
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import analyzer_nodes
from tensorflow_transform import impl_helper
from tensorflow_transform import nodes
from tensorflow_transform.beam import analysis_graph_builder
from tensorflow_transform import test_case

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
""")


def _preprocessing_fn_with_one_analyzer(inputs):
  x = inputs['x']
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
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x/mean_and_var/Cast', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/truediv', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/truediv_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[0]}"];
"ApplySavedModel[0]" [label="{ApplySavedModel|dataset_key: None|phase: 0|label: ApplySavedModel[0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0]";
"TensorSource[x/mean_and_var]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast', 'x/mean_and_var/truediv', 'x/mean_and_var/truediv_1', 'x/mean_and_var/zeros')|label: TensorSource[x/mean_and_var]|partitionable: True}"];
"ApplySavedModel[0]" -> "TensorSource[x/mean_and_var]";
"CacheableCombineAccumulate[x/mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x/mean_and_var]|partitionable: True}"];
"TensorSource[x/mean_and_var]" -> "CacheableCombineAccumulate[x/mean_and_var]";
"CacheableCombineMerge[x/mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x/mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineAccumulate[x/mean_and_var]" -> "CacheableCombineMerge[x/mean_and_var]";
"CreateTensorBinding[x/mean_and_var/Placeholder]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder]}"];
"CacheableCombineMerge[x/mean_and_var]":0 -> "CreateTensorBinding[x/mean_and_var/Placeholder]";
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder_1:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder_1]}"];
"CacheableCombineMerge[x/mean_and_var]":1 -> "CreateTensorBinding[x/mean_and_var/Placeholder_1]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_centered', \"Tensor\<shape: [None], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x/mean_and_var/Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" -> CreateSavedModel;
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
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x/Reshape', \"Tensor\<shape: [None], \<dtype: 'string'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[0]}"];
"ApplySavedModel[0]" [label="{ApplySavedModel|dataset_key: None|phase: 0|label: ApplySavedModel[0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0]";
"TensorSource[x]" [label="{ExtractFromDict|keys: ('x/Reshape',)|label: TensorSource[x]|partitionable: True}"];
"ApplySavedModel[0]" -> "TensorSource[x]";
"VocabularyAccumulate[x]" [label="{VocabularyAccumulate|vocab_ordering_type: 1|input_dtype: string|label: VocabularyAccumulate[x]|partitionable: True}"];
"TensorSource[x]" -> "VocabularyAccumulate[x]";
"VocabularyMerge[x]" [label="{VocabularyMerge|vocab_ordering_type: 1|use_adjusted_mutual_info: False|min_diff_from_avg: None|label: VocabularyMerge[x]}"];
"VocabularyAccumulate[x]" -> "VocabularyMerge[x]";
"VocabularyOrderAndFilter[x]" [label="{VocabularyOrderAndFilter|top_k: None|frequency_threshold: None|coverage_top_k: None|coverage_frequency_threshold: None|key_fn: None|label: VocabularyOrderAndFilter[x]}"];
"VocabularyMerge[x]" -> "VocabularyOrderAndFilter[x]";
"VocabularyWrite[x]" [label="{VocabularyWrite|vocab_filename: vocab_x|store_frequency: False|input_dtype: string|label: VocabularyWrite[x]|fingerprint_shuffle: False}"];
"VocabularyOrderAndFilter[x]" -> "VocabularyWrite[x]";
"CreateTensorBinding[x/Placeholder]" [label="{CreateTensorBinding|tensor: x/Placeholder:0|is_asset_filepath: True|label: CreateTensorBinding[x/Placeholder]}"];
"VocabularyWrite[x]" -> "CreateTensorBinding[x/Placeholder]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 1|output_signature: OrderedDict([('x_integerized', \"Tensor\<shape: [None], \<dtype: 'int64'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x/Placeholder]" -> CreateSavedModel;
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
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x/mean_and_var/Cast', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/truediv', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/truediv_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[0]}"];
"ApplySavedModel[0]" [label="{ApplySavedModel|dataset_key: None|phase: 0|label: ApplySavedModel[0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0]";
"TensorSource[x/mean_and_var]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast', 'x/mean_and_var/truediv', 'x/mean_and_var/truediv_1', 'x/mean_and_var/zeros')|label: TensorSource[x/mean_and_var]|partitionable: True}"];
"ApplySavedModel[0]" -> "TensorSource[x/mean_and_var]";
"CacheableCombineAccumulate[x/mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x/mean_and_var]|partitionable: True}"];
"TensorSource[x/mean_and_var]" -> "CacheableCombineAccumulate[x/mean_and_var]";
"CacheableCombineMerge[x/mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x/mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineAccumulate[x/mean_and_var]" -> "CacheableCombineMerge[x/mean_and_var]";
"CreateTensorBinding[x/mean_and_var/Placeholder]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder]}"];
"CacheableCombineMerge[x/mean_and_var]":0 -> "CreateTensorBinding[x/mean_and_var/Placeholder]";
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder_1:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder_1]}"];
"CacheableCombineMerge[x/mean_and_var]":1 -> "CreateTensorBinding[x/mean_and_var/Placeholder_1]";
"CreateSavedModelForAnalyzerInputs[1]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_square_deviations/mean_and_var/Cast', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/truediv', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/truediv_1', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\"), ('x_square_deviations/mean_and_var/zeros', \"Tensor\<shape: [], \<dtype: 'float32'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[1]}"];
"CreateTensorBinding[x/mean_and_var/Placeholder]" -> "CreateSavedModelForAnalyzerInputs[1]";
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" -> "CreateSavedModelForAnalyzerInputs[1]";
"ApplySavedModel[1]" [label="{ApplySavedModel|dataset_key: None|phase: 1|label: ApplySavedModel[1]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[1]" -> "ApplySavedModel[1]";
"TensorSource[x_square_deviations/mean_and_var]" [label="{ExtractFromDict|keys: ('x_square_deviations/mean_and_var/Cast', 'x_square_deviations/mean_and_var/truediv', 'x_square_deviations/mean_and_var/truediv_1', 'x_square_deviations/mean_and_var/zeros')|label: TensorSource[x_square_deviations/mean_and_var]|partitionable: True}"];
"ApplySavedModel[1]" -> "TensorSource[x_square_deviations/mean_and_var]";
"CacheableCombineAccumulate[x_square_deviations/mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineAccumulate[x_square_deviations/mean_and_var]|partitionable: True}"];
"TensorSource[x_square_deviations/mean_and_var]" -> "CacheableCombineAccumulate[x_square_deviations/mean_and_var]";
"CacheableCombineMerge[x_square_deviations/mean_and_var]" [label="{CacheableCombineMerge|combiner: \<WeightedMeanAndVarCombiner\>|label: CacheableCombineMerge[x_square_deviations/mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineAccumulate[x_square_deviations/mean_and_var]" -> "CacheableCombineMerge[x_square_deviations/mean_and_var]";
"CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder]" [label="{CreateTensorBinding|tensor: x_square_deviations/mean_and_var/Placeholder:0|is_asset_filepath: False|label: CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder]}"];
"CacheableCombineMerge[x_square_deviations/mean_and_var]":0 -> "CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder]";
"CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder_1]" [label="{CreateTensorBinding|tensor: x_square_deviations/mean_and_var/Placeholder_1:0|is_asset_filepath: False|label: CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder_1]}"];
"CacheableCombineMerge[x_square_deviations/mean_and_var]":1 -> "CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder_1]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_normalized', \"Tensor\<shape: [None], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x/mean_and_var/Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x_square_deviations/mean_and_var/Placeholder_1]" -> CreateSavedModel;
}
""")


def _preprocessing_fn_with_chained_ptransforms(inputs):

  class FakeChainable(
      collections.namedtuple('FakeChainable', ['label']), nodes.OperationDef):

    def __new__(cls, label=None):
      if label is None:
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
        output_value_node,
        analyzer_nodes.TensorInfo(tf.float32, (17, 27), False))
    return {'x_chained': x_chained}


_CHAINED_PTRANSFORMS_CASE = dict(
    testcase_name='with_chained_ptransforms',
    feature_spec={'x': tf.io.FixedLenFeature([], tf.int64)},
    preprocessing_fn=_preprocessing_fn_with_chained_ptransforms,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('inputs/x', \"Tensor\<shape: [None], \<dtype: 'int64'\>\>\")])|label: CreateSavedModelForAnalyzerInputs[0]}"];
"ApplySavedModel[0]" [label="{ApplySavedModel|dataset_key: None|phase: 0|label: ApplySavedModel[0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0]";
"TensorSource[x]" [label="{ExtractFromDict|keys: ('inputs/x',)|label: TensorSource[x]|partitionable: True}"];
"ApplySavedModel[0]" -> "TensorSource[x]";
"FakeChainable[x/ptransform1]" [label="{FakeChainable|label: FakeChainable[x/ptransform1]}"];
"TensorSource[x]" -> "FakeChainable[x/ptransform1]";
"FakeChainable[x/ptransform2]" [label="{FakeChainable|label: FakeChainable[x/ptransform2]}"];
"FakeChainable[x/ptransform1]" -> "FakeChainable[x/ptransform2]";
"CreateTensorBinding[x/Placeholder]" [label="{CreateTensorBinding|tensor: x/Placeholder:0|is_asset_filepath: False|label: CreateTensorBinding[x/Placeholder]}"];
"FakeChainable[x/ptransform2]" -> "CreateTensorBinding[x/Placeholder]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: 0|output_signature: OrderedDict([('x_chained', \"Tensor\<shape: [17, 27], \<dtype: 'float32'\>\>\")])|label: CreateSavedModel}"];
"CreateTensorBinding[x/Placeholder]" -> CreateSavedModel;
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

  @test_case.named_parameters(*_ANALYZE_TEST_CASES)
  def test_build(self, feature_spec, preprocessing_fn, expected_dot_graph_str):
    with tf.compat.v1.name_scope('inputs'):
      input_signature = impl_helper.feature_spec_as_batched_placeholders(
          feature_spec)
    output_signature = preprocessing_fn(input_signature)
    transform_fn_future, unused_cache = analysis_graph_builder.build(
        tf.compat.v1.get_default_graph(), input_signature, output_signature)

    dot_string = nodes.get_dot_graph([transform_fn_future]).to_string()
    self.WriteRenderedDotFile(dot_string)

    self.assertMultiLineEqual(
        msg='Result dot graph is:\n{}'.format(dot_string),
        first=dot_string,
        second=expected_dot_graph_str)

  @test_case.named_parameters(
      dict(
          testcase_name='one_dataset_cached_single_phase',
          preprocessing_fn=_preprocessing_fn_with_one_analyzer,
          full_dataset_keys=['a', 'b'],
          cached_dataset_keys=['a'],
          expected_dataset_keys=['b'],
          expected_flat_data_required=False,
      ),
      dict(
          testcase_name='all_datasets_cached_single_phase',
          preprocessing_fn=_preprocessing_fn_with_one_analyzer,
          full_dataset_keys=['a', 'b'],
          cached_dataset_keys=['a', 'b'],
          expected_dataset_keys=[],
          expected_flat_data_required=False,
      ),
      dict(
          testcase_name='mixed_single_phase',
          preprocessing_fn=lambda d: dict(  # pylint: disable=g-long-lambda
              list(_preprocessing_fn_with_chained_ptransforms(d).items()) +
              list(_preprocessing_fn_with_one_analyzer(d).items())),
          full_dataset_keys=['a', 'b'],
          cached_dataset_keys=['a', 'b'],
          expected_dataset_keys=['a', 'b'],
          expected_flat_data_required=True,
      ),
      dict(
          testcase_name='multi_phase',
          preprocessing_fn=_preprocessing_fn_with_two_phases,
          full_dataset_keys=['a', 'b'],
          cached_dataset_keys=['a', 'b'],
          expected_dataset_keys=['a', 'b'],
          expected_flat_data_required=True,
      ),
  )
  def test_get_analysis_dataset_keys(self, preprocessing_fn, full_dataset_keys,
                                     cached_dataset_keys, expected_dataset_keys,
                                     expected_flat_data_required):
    # We force all dataset keys with entries in the cache dict will have a cache
    # hit.
    mocked_cache_entry_key = b'M'
    input_cache = {
        key: {
            mocked_cache_entry_key: 'C'
        } for key in cached_dataset_keys
    }
    feature_spec = {'x': tf.io.FixedLenFeature([], tf.float32)}
    with mock.patch(
        'tensorflow_transform.beam.analysis_graph_builder.'
        'analyzer_cache.make_cache_entry_key',
        return_value=mocked_cache_entry_key):
      dataset_keys, flat_data_required = (
          analysis_graph_builder.get_analysis_dataset_keys(
              preprocessing_fn, feature_spec, full_dataset_keys, input_cache))

    dot_string = nodes.get_dot_graph([analysis_graph_builder._ANALYSIS_GRAPH
                                     ]).to_string()
    self.WriteRenderedDotFile(dot_string)

    self.assertCountEqual(expected_dataset_keys, dataset_keys)
    self.assertEqual(expected_flat_data_required, flat_data_required)


if __name__ == '__main__':
  # TODO(b/133440043): Remove this once TFT supports eager execution.
  tf.compat.v1.disable_eager_execution()
  test_case.main()
