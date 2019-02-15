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


def _preprocessing_fn_with_no_analyzers(inputs):
  x = inputs['x']
  x_plus_1 = x + 1
  return {'x_plus_1': x_plus_1}


_NO_ANALYZERS_CASE = dict(
    testcase_name='with_no_analyzers',
    feature_spec={'x': tf.FixedLenFeature([], tf.float32)},
    preprocessing_fn=_preprocessing_fn_with_no_analyzers,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
CreateSavedModel [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x_plus_1', \<tf.Tensor 'add:0' shape=(?,) dtype=float32\>)])|label: CreateSavedModel}"];
}
""")


def _preprocessing_fn_with_one_analyzer(inputs):
  x = inputs['x']
  x_mean = tft.mean(x, name='x')
  x_centered = x - x_mean
  return {'x_centered': x_centered}


_ONE_ANALYZER_CASE = dict(
    testcase_name='with_one_analyzer',
    feature_spec={'x': tf.FixedLenFeature([], tf.float32)},
    preprocessing_fn=_preprocessing_fn_with_one_analyzer,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x/mean_and_var/Cast', \<tf.Tensor 'x/mean_and_var/Cast:0' shape=() dtype=float32\>), ('x/mean_and_var/truediv', \<tf.Tensor 'x/mean_and_var/truediv:0' shape=() dtype=float32\>), ('x/mean_and_var/truediv_1', \<tf.Tensor 'x/mean_and_var/truediv_1:0' shape=() dtype=float32\>)])|label: CreateSavedModelForAnalyzerInputs[0]}"];
"ApplySavedModel[0]" [label="{ApplySavedModel|dataset_key: None|phase: 0|label: ApplySavedModel[0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0]";
"TensorSource[x/mean_and_var]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast', 'x/mean_and_var/truediv', 'x/mean_and_var/truediv_1')|label: TensorSource[x/mean_and_var]|partitionable: True}"];
"ApplySavedModel[0]" -> "TensorSource[x/mean_and_var]";
"CacheableCombineAccumulate[x/mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<MeanAndVarCombiner\>|label: CacheableCombineAccumulate[x/mean_and_var]|partitionable: True}"];
"TensorSource[x/mean_and_var]" -> "CacheableCombineAccumulate[x/mean_and_var]";
"CacheableCombineMerge[x/mean_and_var]" [label="{CacheableCombineMerge|combiner: \<MeanAndVarCombiner\>|label: CacheableCombineMerge[x/mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineAccumulate[x/mean_and_var]" -> "CacheableCombineMerge[x/mean_and_var]";
"CreateTensorBinding[x/mean_and_var/Placeholder]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder]}"];
"CacheableCombineMerge[x/mean_and_var]":0 -> "CreateTensorBinding[x/mean_and_var/Placeholder]";
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder_1:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder_1]}"];
"CacheableCombineMerge[x/mean_and_var]":1 -> "CreateTensorBinding[x/mean_and_var/Placeholder_1]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x_centered', \<tf.Tensor 'sub:0' shape=(?,) dtype=float32\>)])|label: CreateSavedModel}"];
"CreateTensorBinding[x/mean_and_var/Placeholder]" -> CreateSavedModel;
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" -> CreateSavedModel;
}
""")


def _preprocessing_fn_with_table(inputs):
  x = inputs['x']
  x_vocab = tft.vocabulary(x, name='x')
  table = tf.contrib.lookup.index_table_from_file(x_vocab)
  x_integerized = table.lookup(x)
  return {'x_integerized': x_integerized}


_WITH_TABLE_CASE = dict(
    testcase_name='with_table',
    feature_spec={'x': tf.FixedLenFeature([], tf.string)},
    preprocessing_fn=_preprocessing_fn_with_table,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x/Reshape', \<tf.Tensor 'x/Reshape:0' shape=(?,) dtype=string\>)])|label: CreateSavedModelForAnalyzerInputs[0]}"];
"ApplySavedModel[0]" [label="{ApplySavedModel|dataset_key: None|phase: 0|label: ApplySavedModel[0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0]";
"TensorSource[x]" [label="{ExtractFromDict|keys: ('x/Reshape',)|label: TensorSource[x]|partitionable: True}"];
"ApplySavedModel[0]" -> "TensorSource[x]";
"VocabularyAccumulate[x]" [label="{VocabularyAccumulate|vocab_ordering_type: 1|label: VocabularyAccumulate[x]|partitionable: True}"];
"TensorSource[x]" -> "VocabularyAccumulate[x]";
"VocabularyMerge[x]" [label="{VocabularyMerge|vocab_ordering_type: 1|use_adjusted_mutual_info: False|min_diff_from_avg: 0.0|label: VocabularyMerge[x]}"];
"VocabularyAccumulate[x]" -> "VocabularyMerge[x]";
"VocabularyOrderAndFilter[x]" [label="{VocabularyOrderAndFilter|top_k: None|frequency_threshold: None|coverage_top_k: None|coverage_frequency_threshold: None|key_fn: None|label: VocabularyOrderAndFilter[x]}"];
"VocabularyMerge[x]" -> "VocabularyOrderAndFilter[x]";
"VocabularyWrite[x]" [label="{VocabularyWrite|vocab_filename: vocab_x|store_frequency: False|label: VocabularyWrite[x]}"];
"VocabularyOrderAndFilter[x]" -> "VocabularyWrite[x]";
"CreateTensorBinding[x/Placeholder]" [label="{CreateTensorBinding|tensor: x/Placeholder:0|is_asset_filepath: True|label: CreateTensorBinding[x/Placeholder]}"];
"VocabularyWrite[x]" -> "CreateTensorBinding[x/Placeholder]";
CreateSavedModel [label="{CreateSavedModel|table_initializers: (\<tf.Operation 'string_to_index/hash_table/table_init' type=InitializeTableFromTextFileV2\>,)|output_signature: OrderedDict([('x_integerized', \<tf.Tensor 'hash_table_Lookup:0' shape=(?,) dtype=int64\>)])|label: CreateSavedModel}"];
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
    feature_spec={'x': tf.FixedLenFeature([], tf.float32)},
    preprocessing_fn=_preprocessing_fn_with_two_phases,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x/mean_and_var/Cast', \<tf.Tensor 'x/mean_and_var/Cast:0' shape=() dtype=float32\>), ('x/mean_and_var/truediv', \<tf.Tensor 'x/mean_and_var/truediv:0' shape=() dtype=float32\>), ('x/mean_and_var/truediv_1', \<tf.Tensor 'x/mean_and_var/truediv_1:0' shape=() dtype=float32\>)])|label: CreateSavedModelForAnalyzerInputs[0]}"];
"ApplySavedModel[0]" [label="{ApplySavedModel|dataset_key: None|phase: 0|label: ApplySavedModel[0]|partitionable: True}"];
"CreateSavedModelForAnalyzerInputs[0]" -> "ApplySavedModel[0]";
"TensorSource[x/mean_and_var]" [label="{ExtractFromDict|keys: ('x/mean_and_var/Cast', 'x/mean_and_var/truediv', 'x/mean_and_var/truediv_1')|label: TensorSource[x/mean_and_var]|partitionable: True}"];
"ApplySavedModel[0]" -> "TensorSource[x/mean_and_var]";
"CacheableCombineAccumulate[x/mean_and_var]" [label="{CacheableCombineAccumulate|combiner: \<MeanAndVarCombiner\>|label: CacheableCombineAccumulate[x/mean_and_var]|partitionable: True}"];
"TensorSource[x/mean_and_var]" -> "CacheableCombineAccumulate[x/mean_and_var]";
"CacheableCombineMerge[x/mean_and_var]" [label="{CacheableCombineMerge|combiner: \<MeanAndVarCombiner\>|label: CacheableCombineMerge[x/mean_and_var]|{<0>0|<1>1}}"];
"CacheableCombineAccumulate[x/mean_and_var]" -> "CacheableCombineMerge[x/mean_and_var]";
"CreateTensorBinding[x/mean_and_var/Placeholder]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder]}"];
"CacheableCombineMerge[x/mean_and_var]":0 -> "CreateTensorBinding[x/mean_and_var/Placeholder]";
"CreateTensorBinding[x/mean_and_var/Placeholder_1]" [label="{CreateTensorBinding|tensor: x/mean_and_var/Placeholder_1:0|is_asset_filepath: False|label: CreateTensorBinding[x/mean_and_var/Placeholder_1]}"];
"CacheableCombineMerge[x/mean_and_var]":1 -> "CreateTensorBinding[x/mean_and_var/Placeholder_1]";
"CreateSavedModelForAnalyzerInputs[1]" [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x_square_deviations/mean_and_var/Cast', \<tf.Tensor 'x_square_deviations/mean_and_var/Cast:0' shape=() dtype=float32\>), ('x_square_deviations/mean_and_var/truediv', \<tf.Tensor 'x_square_deviations/mean_and_var/truediv:0' shape=() dtype=float32\>), ('x_square_deviations/mean_and_var/truediv_1', \<tf.Tensor 'x_square_deviations/mean_and_var/truediv_1:0' shape=() dtype=float32\>)])|label: CreateSavedModelForAnalyzerInputs[1]}"];
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
        scope = tf.get_default_graph().get_name_scope()
        label = '{}[{}]'.format(cls.__name__, scope)
      return super(FakeChainable, cls).__new__(cls, label=label)

  with tf.name_scope('x'):
    input_values_node = nodes.apply_operation(
        analyzer_nodes.TensorSource, tensors=[inputs['x']])
    with tf.name_scope('ptransform1'):
      intermediate_value_node = nodes.apply_operation(FakeChainable,
                                                      input_values_node)
    with tf.name_scope('ptransform2'):
      output_value_node = nodes.apply_operation(FakeChainable,
                                                intermediate_value_node)
    x_chained = analyzer_nodes.bind_future_as_tensor(
        output_value_node,
        analyzer_nodes.TensorInfo(tf.float32, (17, 27), False))
    return {'x_chained': x_chained}


_CHAINED_PTRANSFORMS_CASE = dict(
    testcase_name='with_chained_ptransforms',
    feature_spec={'x': tf.FixedLenFeature([], tf.int64)},
    preprocessing_fn=_preprocessing_fn_with_chained_ptransforms,
    expected_dot_graph_str=r"""digraph G {
directed=True;
node [shape=Mrecord];
"CreateSavedModelForAnalyzerInputs[0]" [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('inputs/x', \<tf.Tensor 'inputs/x:0' shape=(?,) dtype=int64\>)])|label: CreateSavedModelForAnalyzerInputs[0]}"];
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
CreateSavedModel [label="{CreateSavedModel|table_initializers: ()|output_signature: OrderedDict([('x_chained', \<tf.Tensor 'x/Placeholder:0' shape=(17, 27) dtype=float32\>)])|label: CreateSavedModel}"];
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
    with tf.name_scope('inputs'):
      input_signature = impl_helper.feature_spec_as_batched_placeholders(
          feature_spec)
    output_signature = preprocessing_fn(input_signature)
    transform_fn_future = analysis_graph_builder.build(
        tf.get_default_graph(), input_signature, output_signature)

    dot_string = nodes.get_dot_graph([transform_fn_future]).to_string()
    self.WriteRenderedDotFile(dot_string)

    self.assertMultiLineEqual(
        msg='Result dot graph is:\n{}'.format(dot_string),
        first=dot_string,
        second=expected_dot_graph_str)


if __name__ == '__main__':
  test_case.main()
