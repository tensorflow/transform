# coding=utf-8
#
# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Tests for tft.vocabulary and tft.compute_and_apply_vocabulary."""

import os

import apache_beam as beam

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import common_types
from tensorflow_transform.beam import analyzer_impls
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

from tensorflow_metadata.proto.v0 import schema_pb2

_COMPOSITE_COMPUTE_AND_APPLY_VOCABULARY_TEST_CASES = [
    dict(
        testcase_name='sparse',
        input_data=[
            {
                'val': ['hello'],
                'idx0': [0],
                'idx1': [0]
            },
            {
                'val': ['world'],
                'idx0': [1],
                'idx1': [1]
            },
            {
                'val': ['hello', 'goodbye'],
                'idx0': [0, 1],
                'idx1': [1, 2]
            },
            {
                'val': ['hello', 'goodbye', ' '],
                'idx0': [0, 1, 1],
                'idx1': [0, 1, 2]
            },
        ],
        input_metadata=tft_unit.metadata_from_feature_spec({
            'x': tf.io.SparseFeature(['idx0', 'idx1'], 'val', tf.string, [2, 3])
        }),
        expected_data=[{
            'index$sparse_indices_0': [0],
            'index$sparse_indices_1': [0],
            'index$sparse_values': [0],
        }, {
            'index$sparse_indices_0': [1],
            'index$sparse_indices_1': [1],
            'index$sparse_values': [2],
        }, {
            'index$sparse_indices_0': [0, 1],
            'index$sparse_indices_1': [1, 2],
            'index$sparse_values': [0, 1],
        }, {
            'index$sparse_indices_0': [0, 1, 1],
            'index$sparse_indices_1': [0, 1, 2],
            'index$sparse_values': [0, 1, 3],
        }],
        expected_vocab_file_contents={
            'my_vocab': [b'hello', b'goodbye', b'world', b' ']
        }),
]

if common_types.is_ragged_feature_available():
  _COMPOSITE_COMPUTE_AND_APPLY_VOCABULARY_TEST_CASES.append(
      dict(
          testcase_name='ragged',
          input_data=[
              {
                  'val': ['hello', ' '],
                  'row_lengths': [1, 0, 1]
              },
              {
                  'val': ['world'],
                  'row_lengths': [0, 1]
              },
              {
                  'val': ['hello', 'goodbye'],
                  'row_lengths': [2, 0, 0]
              },
              {
                  'val': ['hello', 'goodbye', ' '],
                  'row_lengths': [0, 2, 1]
              },
          ],
          input_metadata=tft_unit.metadata_from_feature_spec({
              'x':
                  tf.io.RaggedFeature(
                      tf.string,
                      value_key='val',
                      partitions=[
                          tf.io.RaggedFeature.RowLengths('row_lengths')  # pytype: disable=attribute-error
                      ])
          }),
          expected_data=[
              {
                  'index$ragged_values': [0, 2],
                  'index$row_lengths_1': [1, 0, 1]
              },
              {
                  'index$ragged_values': [3],
                  'index$row_lengths_1': [0, 1]
              },
              {
                  'index$ragged_values': [0, 1],
                  'index$row_lengths_1': [2, 0, 0]
              },
              {
                  'index$ragged_values': [0, 1, 2],
                  'index$row_lengths_1': [0, 2, 1]
              },
          ],
          expected_vocab_file_contents={
              'my_vocab': [b'hello', b'goodbye', b' ', b'world']
          }))


class VocabularyIntegrationTest(tft_unit.TransformTestCase):

  def setUp(self):
    tf.compat.v1.logging.info('Starting test case: %s', self._testMethodName)
    super().setUp()

  def _VocabFormat(self):
    return 'text'

  _WITH_LABEL_PARAMS = tft_unit.cross_named_parameters([
      dict(
          testcase_name='_string',
          x_data=[
              b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
              b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye', b'goodbye'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          expected_vocab_file_contents=[(b'goodbye', 1.9753224),
                                        (b'aaaaa', 1.6600707),
                                        (b'hello', 1.2450531)]),
      dict(
          testcase_name='_int64',
          x_data=[3, 3, 3, 1, 2, 2, 1, 1, 2, 2, 1, 1],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[(b'1', 1.9753224), (b'2', 1.6600707),
                                        (b'3', 1.2450531)]),
  ], [
      dict(
          testcase_name='with_label',
          label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          min_diff_from_avg=0.0,
          store_frequency=True),
  ])

  @tft_unit.named_parameters(*([
      dict(
          testcase_name='_unadjusted_mi_binary_label',
          x_data=[
              b'informative', b'informative', b'informative', b'uninformative',
              b'uninformative', b'uninformative', b'uninformative',
              b'uninformative_rare', b'uninformative_rare'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 1, 0, 1, 1, 0, 0, 1],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[
              (b'informative', 1.7548264),
              (b'uninformative', 0.33985),
              (b'uninformative_rare', 0.169925),
          ],
          min_diff_from_avg=0.0,
          use_adjusted_mutual_info=False,
          store_frequency=True),
      dict(
          testcase_name='_unadjusted_mi_multi_class_label',
          x_data=[
              b'good_predictor_of_0', b'good_predictor_of_0',
              b'good_predictor_of_0', b'good_predictor_of_1',
              b'good_predictor_of_2', b'good_predictor_of_2',
              b'good_predictor_of_2', b'good_predictor_of_1',
              b'good_predictor_of_1', b'weak_predictor_of_1',
              b'good_predictor_of_0', b'good_predictor_of_1',
              b'good_predictor_of_1', b'good_predictor_of_1',
              b'weak_predictor_of_1'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[0, 0, 0, 1, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[
              (b'good_predictor_of_2', 6.9656613),
              (b'good_predictor_of_1', 6.5969828),
              (b'good_predictor_of_0', 6.339692),
              (b'weak_predictor_of_1', 0.684463),
          ],
          min_diff_from_avg=0.0,
          use_adjusted_mutual_info=False,
          store_frequency=True),
      dict(
          testcase_name='_unadjusted_mi_binary_label_with_weights',
          x_data=[
              b'informative_1', b'informative_1', b'informative_0',
              b'informative_0', b'uninformative', b'uninformative',
              b'informative_by_weight', b'informative_by_weight'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 0, 0, 0, 1, 0, 1],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          # uninformative and informative_by_weight have the same co-occurrence
          # relationship with the label but will have different importance
          # values due to the weighting.
          expected_vocab_file_contents=[
              (b'informative_0', 3.1698803),
              (b'informative_1', 1.1698843),
              (b'informative_by_weight', 0.6096405),
              (b'uninformative', 0.169925),
          ],
          weight_data=[1, 1, 1, 1, 1, 1, 1, 5],
          weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
          min_diff_from_avg=0.0,
          use_adjusted_mutual_info=False,
          store_frequency=True),
      dict(
          testcase_name='_unadjusted_mi_binary_label_min_diff_from_avg',
          x_data=[
              b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
              b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye', b'goodbye'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          # All features are weak predictors, so all are adjusted to zero.
          expected_vocab_file_contents=[
              (b'hello', 0.0),
              (b'goodbye', 0.0),
              (b'aaaaa', 0.0),
          ],
          use_adjusted_mutual_info=False,
          min_diff_from_avg=2.0,
          store_frequency=True),
      dict(
          testcase_name='_adjusted_mi_binary_label',
          x_data=[
              b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
              b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye', b'goodbye'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[
              (b'goodbye', 1.4070794),
              (b'aaaaa', 0.9987448),
              (b'hello', 0.5017178),
          ],
          min_diff_from_avg=0.0,
          use_adjusted_mutual_info=True,
          store_frequency=True),
      dict(
          testcase_name='_adjusted_mi_binary_label_int64_feature',
          x_data=[3, 3, 3, 1, 2, 2, 1, 1, 2, 2, 1, 1],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[
              (b'1', 1.4070794),
              (b'2', 0.9987448),
              (b'3', 0.5017178),
          ],
          min_diff_from_avg=0.0,
          use_adjusted_mutual_info=True,
          store_frequency=True),
      dict(
          testcase_name='_adjusted_mi_multi_class_label',
          x_data=[
              b'good_predictor_of_0', b'good_predictor_of_0',
              b'good_predictor_of_0', b'good_predictor_of_1',
              b'good_predictor_of_2', b'good_predictor_of_2',
              b'good_predictor_of_2', b'good_predictor_of_1',
              b'good_predictor_of_1', b'weak_predictor_of_1',
              b'good_predictor_of_0', b'good_predictor_of_1',
              b'good_predictor_of_1', b'good_predictor_of_1',
              b'weak_predictor_of_1'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[0, 0, 0, 1, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[
              (b'good_predictor_of_1', 5.4800901),
              (b'good_predictor_of_2', 5.3861019),
              (b'good_predictor_of_0', 4.9054722),
              (b'weak_predictor_of_1', -0.9748023),
          ],
          min_diff_from_avg=0.0,
          use_adjusted_mutual_info=True,
          store_frequency=True),
      # TODO(b/128831096): Determine correct interaction between AMI and weights
      dict(
          testcase_name='_adjusted_mi_binary_label_with_weights',
          x_data=[
              b'informative_1', b'informative_1', b'informative_0',
              b'informative_0', b'uninformative', b'uninformative',
              b'informative_by_weight', b'informative_by_weight'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 0, 0, 0, 1, 0, 1],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          weight_data=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
          weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
          # uninformative and informative_by_weight have the same co-occurrence
          # relationship with the label but will have different importance
          # values due to the weighting.
          expected_vocab_file_contents=[
              (b'informative_0', 2.3029856),
              (b'informative_1', 0.3029896),
              (b'informative_by_weight', 0.1713041),
              (b'uninformative', -0.6969697),
          ],
          min_diff_from_avg=0.0,
          use_adjusted_mutual_info=True,
          store_frequency=True),
      dict(
          testcase_name='_adjusted_mi_min_diff_from_avg',
          x_data=[
              b'good_predictor_of_0', b'good_predictor_of_0',
              b'good_predictor_of_0', b'good_predictor_of_1',
              b'good_predictor_of_0', b'good_predictor_of_1',
              b'good_predictor_of_1', b'good_predictor_of_1',
              b'good_predictor_of_1', b'good_predictor_of_0',
              b'good_predictor_of_1', b'good_predictor_of_1',
              b'good_predictor_of_1', b'weak_predictor_of_1',
              b'weak_predictor_of_1'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          # With min_diff_from_avg, the small AMI value is regularized to 0
          expected_vocab_file_contents=[
              (b'good_predictor_of_0', 1.8322128),
              (b'good_predictor_of_1', 1.7554416),
              (b'weak_predictor_of_1', 0),
          ],
          use_adjusted_mutual_info=True,
          min_diff_from_avg=1.0,
          store_frequency=True),
      dict(
          testcase_name='_labels_weight_and_frequency',
          x_data=[
              b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
              b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye', b'goodbye'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          weight_data=[
              0.3, 0.4, 0.3, 1.2, 0.6, 0.7, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0
          ],
          weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
          expected_vocab_file_contents=[
              (b'aaaaa', 1.5637185),
              (b'goodbye', 0.8699492),
              (b'hello', 0.6014302),
          ],
          min_diff_from_avg=0.0,
          store_frequency=True),
      # fingerprints by which each of the tokens will be sorted if fingerprint
      # shuffling is used.
      # 'ho ho': '1b3dd735ddff70d90f3b7ba5ebf65df521d6ca4d'
      # 'world': '7c211433f02071597741e6ff5a8ea34789abbf43'
      # 'hello': 'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d'
      # 'hi': 'c22b5f9178342609428d6f51b2c5af4c0bde6a42'
      # '1': '356a192b7913b04c54574d18c28d46e6395428ab'
      # '2': 'da4b9237bacccdf19c0760cab7aec4a8359010b0'
      # '3': '77de68daecd823babbb58edb1c8e14d7106e83bb'
      dict(
          testcase_name='_string_feature_with_frequency_and_shuffle',
          x_data=[b'world', b'hello', b'hello'],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          expected_vocab_file_contents=[(b'world', 1), (b'hello', 2)],
          fingerprint_shuffle=True,
          store_frequency=True),
      dict(
          testcase_name='_string_feature_with_frequency_and_no_shuffle',
          x_data=[b'hi', b'ho ho', b'ho ho'],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          expected_vocab_file_contents=[(b'ho ho', 2), (b'hi', 1)],
          store_frequency=True),
      dict(
          testcase_name='_string_feature_with_no_frequency_and_shuffle',
          x_data=[b'world', b'hello', b'hello'],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          expected_vocab_file_contents=[b'world', b'hello'],
          fingerprint_shuffle=True),
      dict(
          testcase_name='_string_feature_with_no_frequency_and_no_shuffle',
          x_data=[b'world', b'hello', b'hello'],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          expected_vocab_file_contents=[b'hello', b'world']),
      dict(
          testcase_name='_int_feature_with_frequency_and_shuffle',
          x_data=[1, 2, 2, 3],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[(b'1', 1), (b'3', 1), (b'2', 2)],
          fingerprint_shuffle=True,
          store_frequency=True),
      dict(
          testcase_name='_int_feature_with_frequency_and_no_shuffle',
          x_data=[2, 1, 1, 1],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[(b'1', 3), (b'2', 1)],
          store_frequency=True),
      dict(
          testcase_name='_int_feature_with_no_frequency_and_shuffle',
          x_data=[1, 2, 2, 3],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[b'1', b'3', b'2'],
          fingerprint_shuffle=True),
      dict(
          testcase_name='_int_feature_with_no_frequency_and_no_shuffle',
          x_data=[1, 2, 2, 3],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[b'2', b'3', b'1']),
      dict(
          testcase_name='_int_feature_with_top_k',
          x_data=[111, 2, 2, 3],
          top_k=2,
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          expected_vocab_file_contents=[b'2', b'3']),
  ] + _WITH_LABEL_PARAMS))
  def testVocabulary(self,
                     x_data,
                     x_feature_spec,
                     label_data=None,
                     label_feature_spec=None,
                     weight_data=None,
                     weight_feature_spec=None,
                     expected_vocab_file_contents=None,
                     **kwargs):
    """Test tft.Vocabulary with various inputs."""

    input_data = [{'x': x} for x in x_data]
    input_feature_spec = {'x': x_feature_spec}

    if label_data is not None:
      for idx, label in enumerate(label_data):
        input_data[idx]['label'] = label
      input_feature_spec['label'] = label_feature_spec

    if weight_data is not None:
      for idx, weight in enumerate(weight_data):
        input_data[idx]['weights'] = weight
      input_feature_spec['weights'] = weight_feature_spec

    input_metadata = tft_unit.metadata_from_feature_spec(input_feature_spec)

    def preprocessing_fn(inputs):
      x = inputs['x']
      labels = inputs.get('label')
      weights = inputs.get('weights')
      # Note even though the return value is not used, calling tft.vocabulary
      # will generate the vocabulary as a side effect, and since we have named
      # this vocabulary it can be looked up using public APIs.
      tft.vocabulary(
          x,
          labels=labels,
          weights=weights,
          vocab_filename='my_vocab',
          file_format=self._VocabFormat(),
          **kwargs)
      return inputs

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        input_data,  # expected output data is same as input data
        input_metadata,  # expected output metadata is same as input metadata
        expected_vocab_file_contents={'my_vocab': expected_vocab_file_contents})

  @tft_unit.named_parameters(*tft_unit.cross_named_parameters(
      [
          dict(
              testcase_name='_string',
              input_data=[{
                  'x': b'hello'
              }, {
                  'x': b'hello'
              }, {
                  'x': b'hello'
              }, {
                  'x': b'goodbye'
              }, {
                  'x': b'aaaaa'
              }, {
                  'x': b'aaaaa'
              }, {
                  'x': b'goodbye'
              }, {
                  'x': b'goodbye'
              }, {
                  'x': b'aaaaa'
              }, {
                  'x': b'aaaaa'
              }, {
                  'x': b'goodbye'
              }, {
                  'x': b'goodbye'
              }],
              make_feature_spec=lambda:  # pylint: disable=g-long-lambda
              {'x': tf.io.FixedLenFeature([], tf.string)},
              top_k=2,
              make_expected_vocab_fn=(
                  lambda _: [(b'goodbye', 5), (b'aaaaa', 4)])),
          dict(
              testcase_name='_int',
              input_data=[{
                  'x': 1
              }, {
                  'x': 2
              }, {
                  'x': 2
              }, {
                  'x': 3
              }, {
                  'x': 1
              }],
              make_feature_spec=lambda:  # pylint: disable=g-long-lambda
              {'x': tf.io.FixedLenFeature([], tf.int64)},
              top_k=2,
              make_expected_vocab_fn=lambda _: [(b'1', 2), (b'2', 2)]),
          dict(
              testcase_name='_weights',
              input_data=[
                  {
                      'x': b'hello',
                      'weights': 1.4
                  },
                  {
                      'x': b'hello',
                      'weights': 0.5
                  },
                  {
                      'x': b'hello',
                      'weights': 1.12
                  },
                  {
                      'x': b'goodbye',
                      'weights': 0.123
                  },
                  {
                      'x': b'aaaaa',
                      'weights': 0.3
                  },
                  {
                      'x': b'aaaaa',
                      'weights': 1.123
                  },
                  {
                      'x': b'goodbye',
                      'weights': 0.1
                  },
                  {
                      'x': b'goodbye',
                      'weights': 0.00001
                  },
              ],
              make_feature_spec=lambda: {  # pylint: disable=g-long-lambda
                  'x': tf.io.FixedLenFeature([], tf.string),
                  'weights': tf.io.FixedLenFeature([], tf.float32)
              },
              top_k=2,
              make_expected_vocab_fn=(
                  lambda _: [(b'hello', 3.02), (b'aaaaa', 1.423)])),
          dict(
              testcase_name='_large_top_k',
              input_data=[{
                  'x': b'hello'
              }, {
                  'x': b'hello'
              }, {
                  'x': b'hello'
              }, {
                  'x': b' '
              }, {
                  'x': b'aaaaa'
              }, {
                  'x': b'aaaaa'
              }, {
                  'x': b'goodbye'
              }, {
                  'x': b'goodbye'
              }, {
                  'x': b' '
              }, {
                  'x': b''
              }, {
                  'x': b'goodbye'
              }, {
                  'x': b'goodbye'
              }],
              make_feature_spec=lambda:  # pylint: disable=g-long-lambda
              {'x': tf.io.FixedLenFeature([], tf.string)},
              top_k=100,
              make_expected_vocab_fn=lambda file_format:  # pylint: disable=g-long-lambda
              ([(b'goodbye', 4), (b'hello', 3), (b' ', 2),  # pylint: disable=g-long-ternary
                (b'aaaaa', 2)] if file_format == 'text' else [(b'goodbye', 4),
                                                              (b'hello', 3),
                                                              (b' ', 2),
                                                              (b'aaaaa', 2),
                                                              (b'', 1)])),
          dict(
              testcase_name='_ragged',
              input_data=[
                  {
                      'x$ragged_values': ['hello', ' '],
                      'x$row_lengths_1': [1, 0, 1]
                  },
                  {
                      'x$ragged_values': ['hello'],
                      'x$row_lengths_1': [0, 1]
                  },
                  {
                      'x$ragged_values': ['hello', 'goodbye'],
                      'x$row_lengths_1': [2, 0, 0]
                  },
                  {
                      'x$ragged_values': ['hello', 'hello', ' ', ' '],
                      'x$row_lengths_1': [0, 2, 2]
                  },
              ],
              make_feature_spec=lambda: {  # pylint: disable=g-long-lambda
                  'x':
                      tf.io.RaggedFeature(
                          tf.string,
                          value_key='x$ragged_values',
                          partitions=[
                              tf.io.RaggedFeature.RowLengths('x$row_lengths_1')  # pytype: disable=attribute-error
                          ])
              },
              top_k=2,
              make_expected_vocab_fn=lambda _: [(b'hello', 5), (b' ', 3)]),
          dict(
              testcase_name='_sparse',
              input_data=[{
                  'x$sparse_indices_0': [0, 1],
                  'x$sparse_indices_1': [2, 3],
                  'x$sparse_values': [-4, 4],
              }, {
                  'x$sparse_indices_0': [0, 1],
                  'x$sparse_indices_1': [4, 1],
                  'x$sparse_values': [2, 2],
              }, {
                  'x$sparse_indices_0': [0, 1],
                  'x$sparse_indices_1': [0, 3],
                  'x$sparse_values': [2, 4],
              }],
              make_feature_spec=lambda: {  # pylint: disable=g-long-lambda
                  'x':
                      tf.io.SparseFeature([
                          'x$sparse_indices_0', 'x$sparse_indices_1'
                      ], 'x$sparse_values', tf.int64, [5, 5])
              },
              top_k=2,
              make_expected_vocab_fn=lambda _: [(b'2', 3), (b'4', 2)]),
          dict(
              testcase_name='_newline_chars',
              input_data=[{'x': b'aaaaa\n'},
                          {'x': b'\n\n'},
                          {'x': b''},
                          {'x': b' '},
                          {'x': b' '},
                          {'x': b'aaaaa\n'},
                          {'x': b'aaaaa\n'},
                          {'x': b'aaaaa'},
                          {'x': b'goo\rdbye'},
                          {'x': b' '},
                          {'x': b' '},
                          {'x': b'aaaaa\n'}],
              make_feature_spec=(
                  lambda: {'x': tf.io.FixedLenFeature([], tf.string)}),
              top_k=6,
              make_expected_vocab_fn=(
                  lambda file_format: [(b' ', 4), (b'aaaaa', 1)]  # pylint: disable=g-long-lambda,g-long-ternary
                  if file_format == 'text' else [(b' ', 4), (b'aaaaa\n', 4),
                                                 (b'', 1), (b'\n\n', 1),
                                                 (b'aaaaa', 1),
                                                 (b'goo\rdbye', 1)])),
      ],
      [
          dict(testcase_name='no_frequency', store_frequency=False),
          dict(testcase_name='with_frequency', store_frequency=True)
      ]))
  def testApproximateVocabulary(self, input_data, make_feature_spec, top_k,
                                make_expected_vocab_fn, store_frequency):
    input_metadata = tft_unit.metadata_from_feature_spec(
        tft_unit.make_feature_spec_wrapper(make_feature_spec))

    def preprocessing_fn(inputs):
      x = inputs['x']
      weights = inputs.get('weights')
      # Note even though the return value is not used, calling
      # tft.experimental.approximate_vocabulary will generate the vocabulary as
      # a side effect, and since we have named this vocabulary it can be looked
      # up using public APIs.
      tft.experimental.approximate_vocabulary(
          x,
          top_k,
          store_frequency=store_frequency,
          weights=weights,
          vocab_filename='my_approximate_vocab',
          file_format=self._VocabFormat())
      return inputs

    expected_vocab_file_contents = make_expected_vocab_fn(self._VocabFormat())
    if not store_frequency:
      expected_vocab_file_contents = [
          token for token, _ in expected_vocab_file_contents
      ]
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_vocab_file_contents={
            'my_approximate_vocab': expected_vocab_file_contents
        })

  def testComputeAndApplyApproximateVocabulary(self):
    input_data = [{'x': 'a'}] * 2 + [{'x': 'b'}] * 3
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.string)})

    def preprocessing_fn(inputs):
      index = tft.experimental.compute_and_apply_approximate_vocabulary(
          inputs['x'],
          top_k=2,
          file_format=self._VocabFormat(),
          num_oov_buckets=1)
      return {'index': index}

    expected_data = [{'index': 1}] * 2 + [{'index': 0}] * 3 + [{'index': 2}]

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        test_data=input_data + [{'x': 'c'}])  # pyformat: disable

  def testEmptyComputeAndApplyApproximateVocabulary(self):
    input_data = [{'x': ''}] * 3
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.string)})

    def preprocessing_fn(inputs):
      index = tft.experimental.compute_and_apply_approximate_vocabulary(
          inputs['x'],
          top_k=2,
          file_format=self._VocabFormat(),
          num_oov_buckets=1)
      return {'index': index}

    # We only filter empty tokens for `text` format.
    expected_data = [{'index': 1 if self._VocabFormat() == 'text' else 0}] * 3
    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data)

  def testJointVocabularyForMultipleFeatures(self):
    input_data = [{
        'a': 'hello',
        'b': 'world',
        'c': 'aaaaa'
    }, {
        'a': 'good',
        'b': '',
        'c': 'hello'
    }, {
        'a': 'goodbye',
        'b': 'hello',
        'c': '\n'
    }, {
        'a': ' ',
        'b': 'aaaaa',
        'c': 'bbbbb'
    }]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.string),
        'b': tf.io.FixedLenFeature([], tf.string),
        'c': tf.io.FixedLenFeature([], tf.string)
    })
    vocab_filename = 'test_compute_and_apply_vocabulary'

    def preprocessing_fn(inputs):
      deferred_vocab_and_filename = tft.vocabulary(
          tf.concat([inputs['a'], inputs['b'], inputs['c']], 0),
          vocab_filename=vocab_filename,
          file_format=self._VocabFormat())
      return {
          'index_a':
              tft.apply_vocabulary(
                  inputs['a'],
                  deferred_vocab_and_filename,
                  file_format=self._VocabFormat()),
          'index_b':
              tft.apply_vocabulary(
                  inputs['b'],
                  deferred_vocab_and_filename,
                  file_format=self._VocabFormat())
      }

    expected_vocab = [
        b'hello', b'aaaaa', b'world', b'goodbye', b'good', b'bbbbb', b' ',
        b'\n', b''
    ]
    empty_index = len(expected_vocab) - 1
    if self._VocabFormat() == 'text':
      expected_vocab = expected_vocab[:-2]
      empty_index = -1
    max_index = len(expected_vocab) - 1
    expected_data = [
        # For tied frequencies, larger (lexicographic) items come first.
        {
            'index_a': 0,  # hello
            'index_b': 2  # world
        },
        {
            'index_a': 4,  # good
            'index_b': empty_index  # ''
        },
        {
            'index_a': 3,  # goodbye
            'index_b': 0  # hello
        },
        {
            'index_a': 6,  # ' '
            'index_b': 1  # aaaaa
        },
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {
            'index_a': tf.io.FixedLenFeature([], tf.int64),
            'index_b': tf.io.FixedLenFeature([], tf.int64),
        }, {
            'index_a':
                schema_pb2.IntDomain(
                    min=-1, max=max_index, is_categorical=True),
            'index_b':
                schema_pb2.IntDomain(
                    min=-1, max=max_index, is_categorical=True),
        })
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        expected_vocab_file_contents={vocab_filename: expected_vocab})

  _EMPTY_VOCABULARY_PARAMS = tft_unit.cross_named_parameters([
      dict(
          testcase_name='_string',
          x_data=['a', 'b'],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string)),
      dict(
          testcase_name='_int64',
          x_data=[1, 2],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64)),
  ], [
      dict(
          testcase_name='empty_vocabulary',
          index_data=[-1, -1],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=0, is_categorical=True),
          frequency_threshold=5),
  ])

  @tft_unit.named_parameters(*([
      dict(
          testcase_name='_string_feature_with_label_top_2',
          x_data=[
              b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
              b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye', b'goodbye'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[-1, -1, -1, 0, 1, 1, 0, 0, 0, 1, 1, 0],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=1, is_categorical=True),
          top_k=2),
      dict(
          testcase_name='_string_feature_with_label_top_1',
          x_data=[
              b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
              b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye', b'goodbye'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[-1, -1, -1, 0, -1, -1, 0, 0, 0, -1, -1, 0],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=0, is_categorical=True),
          top_k=1),
      dict(
          testcase_name='_int_feature_with_label_top_2',
          x_data=[3, 3, 3, 1, 2, 2, 1, 1, 2, 2, 1, 1],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[-1, -1, -1, 0, 1, 1, 0, 0, 0, 1, 1, 0],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=1, is_categorical=True),
          top_k=2),
      dict(
          testcase_name='_varlen_feature',
          x_data=[[b'world', b'hello', b'hello'], [b'hello', b'world', b'foo'],
                  [], [b'hello']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[1, 0, 0], [0, 1, -99], [], [0]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=1, is_categorical=True),
          default_value=-99,
          top_k=2),
      dict(
          testcase_name='_vector_feature',
          x_data=[[b'world', b'hello', b'hello'], [b'hello', b'world', b'moo'],
                  [b'hello', b'hello', b'foo'], [b'world', b'foo', b'moo']],
          x_feature_spec=tf.io.FixedLenFeature([3], tf.string),
          index_data=[[1, 0, 0], [0, 1, -99], [0, 0, -99], [1, -99, -99]],
          index_feature_spec=tf.io.FixedLenFeature([3], tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=1, is_categorical=True),
          default_value=-99,
          top_k=2),
      dict(
          testcase_name='_varlen_feature_with_labels',
          x_data=[[b'hello', b'world', b'bye', b'moo'],
                  [b'world', b'moo', b'foo'], [b'hello', b'foo', b'moo'],
                  [b'moo']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          label_data=[1, 0, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[[0, -99, 1, -99], [-99, -99, -99], [0, -99, -99], [-99]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=1, is_categorical=True),
          default_value=-99,
          top_k=2),
      dict(
          testcase_name='_vector_feature_with_labels',
          x_data=[[b'world', b'hello', b'hi'], [b'hello', b'world', b'moo'],
                  [b'hello', b'bye', b'foo'], [b'world', b'foo', b'moo']],
          x_feature_spec=tf.io.FixedLenFeature([3], tf.string),
          label_data=[1, 0, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[[-99, -99, 1], [-99, -99, 0], [-99, -99, -99],
                      [-99, -99, 0]],
          index_feature_spec=tf.io.FixedLenFeature([3], tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=1, is_categorical=True),
          default_value=-99,
          top_k=2),
      dict(
          testcase_name='_varlen_integer_feature_with_labels',
          x_data=[[0, 1, 3, 2], [1, 2, 4], [0, 4, 2], [2]],
          x_feature_spec=tf.io.VarLenFeature(tf.int64),
          label_data=[1, 0, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[[0, -99, 1, -99], [-99, -99, -99], [0, -99, -99], [-99]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=1, is_categorical=True),
          default_value=-99,
          top_k=2),
      dict(
          testcase_name='_varlen_feature_with_some_empty_feature_values',
          x_data=[[b'world', b'hello', b'hi', b'moo'], [],
                  [b'world', b'hello', b'foo'], []],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          label_data=[1, 0, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[[0, 1, -99, -99], [], [0, 1, -99], []],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=1, is_categorical=True),
          default_value=-99,
          top_k=2),
      dict(
          testcase_name='_varlen_with_multiclass_labels',
          x_data=[[1, 2, 3, 5], [1, 4, 5], [1, 2], [1, 2], [1, 3, 5], [1, 4, 3],
                  [1, 3]],
          x_feature_spec=tf.io.VarLenFeature(tf.int64),
          label_data=[1, 0, 1, 1, 4, 5, 4],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[[-1, 0, 2, 3], [-1, 1, 3], [-1, 0], [-1, 0], [-1, 2, 3],
                      [-1, 1, 2], [-1, 2]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=3, is_categorical=True),
          top_k=4),
      dict(
          testcase_name='_labels_and_weights',
          x_data=[
              b'hello', b'hello', b'hello', b'goodbye', b'aaaaa', b'aaaaa',
              b'goodbye', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye', b'goodbye'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          weight_data=[
              0.3, 0.4, 0.3, 1.2, 0.6, 0.7, 1.0, 1.0, 0.6, 0.7, 1.0, 1.0
          ],
          weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
          index_data=[2, 2, 2, 1, 0, 0, 1, 1, 0, 0, 1, 1],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=2,
                                            is_categorical=True)),
      dict(
          testcase_name='_string_feature_with_weights',
          x_data=[
              b'hello', b'world', b'goodbye', b'aaaaa', b'aaaaa', b'goodbye'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          weight_data=[1.0, .5, 1.0, .26, .25, 1.5],
          weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
          index_data=[1, 3, 0, 2, 2, 0],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=3,
                                            is_categorical=True)),
      dict(
          testcase_name='_int64_feature_with_weights',
          x_data=[2, 1, 3, 4, 4, 3],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          weight_data=[1.0, .5, 1.0, .26, .25, 1.5],
          weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
          index_data=[1, 3, 0, 2, 2, 0],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=3,
                                            is_categorical=True)),
      dict(
          testcase_name='_whitespace_newlines_and_empty_strings_text',
          x_data=[
              b'hello', b'world', b'hello', b'hello', b'goodbye', b'world',
              b'aaaaa', b' ', b'', b'\n', b'hi \n ho \n', '\r'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          # The empty string and strings containing newlines map to default
          # value because the vocab cannot contain them.
          index_data=[0, 1, 0, 0, 2, 1, 3, 4, -1, -1, -1, -1],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=4, is_categorical=True),
          vocab_filename='my_vocab',
          expected_vocab_file_contents={
              'my_vocab': [b'hello', b'world', b'goodbye', b'aaaaa', b' ']
          },
          required_format='text'),
      dict(
          testcase_name='_whitespace_newlines_and_empty_strings_tfrecord',
          x_data=[
              b'hello', b'world', b'hello', b'hello', b'goodbye', b'world',
              b'aaaaa', b' ', b'', b'\n', b'hi \n ho \n', b'\r'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          index_data=[0, 0, 0, 1, 1, 8, 3, 2, 4, 5, 6, 7],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=8, is_categorical=True),
          vocab_filename='my_vocab',
          expected_vocab_file_contents={
              'my_vocab': [
                  b'hello', b'world', b'hi \n ho \n', b'goodbye', b'aaaaa',
                  b' ', b'\r', b'\n', b''
              ]
          },
          required_format='tfrecord_gzip'),
      dict(
          testcase_name='_whitespace_newlines_empty_oov_buckets_text',
          x_data=[
              b'hello', b'world', b'hello', b'hello', b'goodbye', b'world',
              b'aaaaa', b' ', b'', b'\n', b'hi \n ho \n', '\r'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          index_data=[0, 1, 0, 0, 2, 1, 3, 4, 5, 5, 5, 5],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=0, max=5, is_categorical=True),
          num_oov_buckets=1,
          vocab_filename='my_vocab',
          expected_vocab_file_contents={
              'my_vocab': [b'hello', b'world', b'goodbye', b'aaaaa', b' ']
          },
          required_format='text'),
      dict(
          testcase_name='_whitespace_newlines_empty_oov_buckets_tfrecord',
          x_data=[
              b'hello', b'world', b'hello', b'hello', b'goodbye', b'world',
              b'aaaaa', b' ', b'', b'\n', b'hi \n ho \n', '\r'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          index_data=[0, 0, 1, 0, 1, 8, 3, 2, 4, 5, 6, 7],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=0, max=9, is_categorical=True),
          num_oov_buckets=1,
          vocab_filename='my_vocab',
          expected_vocab_file_contents={
              'my_vocab': [
                  b'hello', b'world', b'hi \n ho \n', b'goodbye', b'aaaaa',
                  b' ', b'\r', b'\n', b''
              ]
          },
          required_format='tfrecord_gzip'),
      dict(
          testcase_name='_positive_and_negative_integers',
          x_data=[13, 14, 13, 13, 12, 14, 11, 10, 10, -10, -10, -20],
          x_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[0, 1, 0, 0, 4, 1, 5, 2, 2, 3, 3, 6],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=6, is_categorical=True),
          vocab_filename='my_vocab',
          expected_vocab_file_contents={
              'my_vocab': [b'13', b'14', b'10', b'-10', b'12', b'11', b'-20']
          }),
      dict(
          testcase_name='_rank_2',
          x_data=[[[b'some', b'say'], [b'the', b'world']],
                  [[b'will', b'end'], [b'in', b'fire']],
                  [[b'some', b'say'], [b'in', b'ice']]],
          x_feature_spec=tf.io.FixedLenFeature([2, 2], tf.string),
          index_data=[[[0, 1], [5, 3]], [[4, 8], [2, 7]], [[0, 1], [2, 6]]],
          index_feature_spec=tf.io.FixedLenFeature([2, 2], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=8,
                                            is_categorical=True)),
      dict(
          testcase_name='_top_k',
          x_data=[[b'hello', b'hello', b'world'],
                  [b'hello', b'goodbye', b'world'],
                  [b'hello', b'goodbye', b'foo']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[0, 0, 1], [0, -99, 1], [0, -99, -99]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=1, is_categorical=True),
          default_value=-99,
          top_k=2),
      dict(
          testcase_name='_top_k_specified_as_str',
          x_data=[[b'hello', b'hello', b'world'],
                  [b'hello', b'goodbye', b'world'],
                  [b'hello', b'goodbye', b'foo']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[0, 0, 1], [0, -9, 1], [0, -9, -9]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(min=-9, max=1, is_categorical=True),
          default_value=-9,
          top_k='2'),
      dict(
          testcase_name='_frequency_threshold',
          x_data=[[b'hello', b'hello', b'world'],
                  [b'hello', b'goodbye', b'world'],
                  [b'hello', b'goodbye', b'foo']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[0, 0, 1], [0, 2, 1], [0, 2, -99]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=2, is_categorical=True),
          default_value=-99,
          frequency_threshold=2),
      dict(
          testcase_name='_frequency_threshold_specified_with_str',
          x_data=[[b'hello', b'hello', b'world'],
                  [b'hello', b'goodbye', b'world'],
                  [b'hello', b'goodbye', b'foo']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[0, 0, 1], [0, 2, 1], [0, 2, -9]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(min=-9, max=2, is_categorical=True),
          default_value=-9,
          frequency_threshold='2'),
      dict(
          testcase_name='_empty_vocabulary_from_high_frequency_threshold',
          x_data=[[b'hello', b'hello', b'world'],
                  [b'hello', b'goodbye', b'world'],
                  [b'hello', b'goodbye', b'foo']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[-99, -99, -99], [-99, -99, -99], [-99, -99, -99]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=0, is_categorical=True),
          default_value=-99,
          frequency_threshold=77),
      dict(
          testcase_name='_top_k_and_oov',
          x_data=[[b'hello', b'hello', b'world', b'world'],
                  [b'hello', b'tarkus', b'toccata'],
                  [b'hello', b'goodbye', b'foo']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          # Generated vocab (ordered by frequency, then value) should be:
          # ["hello", "world", "goodbye", "foo", "tarkus", "toccata"]. After
          # applying top_k =1 this becomes ["hello"] plus three OOV buckets.
          # The specific output values here depend on the hash of the words,
          # and the test will break if the hash changes.
          index_data=[[0, 0, 2, 2], [0, 3, 1], [0, 2, 1]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(min=0, max=3, is_categorical=True),
          default_value=-99,
          top_k=1,
          num_oov_buckets=3),
      dict(
          testcase_name='_key_fn',
          x_data=[['a_X_1', 'a_X_1', 'a_X_2', 'b_X_1', 'b_X_2'],
                  ['a_X_1', 'a_X_1', 'a_X_2', 'a_X_2'], ['b_X_2']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[0, 0, 1, -99, 2], [0, 0, 1, 1], [2]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=2, is_categorical=True),
          coverage_top_k=1,
          default_value=-99,
          key_fn=lambda s: s.split(b'_X_')[0],
          frequency_threshold=3),
      dict(
          testcase_name='_key_fn_and_multi_coverage_top_k',
          x_data=[['a_X_1', 'a_X_1', 'a_X_2', 'b_X_1', 'b_X_2'],
                  ['a_X_1', 'a_X_1', 'a_X_2', 'a_X_2', 'a_X_3'], ['b_X_2']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[0, 0, 1, 3, 2], [0, 0, 1, 1, -99], [2]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=3, is_categorical=True),
          coverage_top_k=2,
          default_value=-99,
          key_fn=lambda s: s.split(b'_X_')[0],
          frequency_threshold=300),
      dict(
          testcase_name='_key_fn_and_top_k',
          x_data=[['a_X_1', 'a_X_1', 'a_X_2', 'b_X_1', 'b_X_2'],
                  ['a_X_1', 'a_X_1', 'a_X_2', 'a_X_2'],
                  ['b_X_2', 'b_X_2', 'b_X_2', 'b_X_2', 'c_X_1']],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[1, 1, -99, -99, 0], [1, 1, -99, -99], [0, 0, 0, 0, 2]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=2, is_categorical=True),
          coverage_top_k=1,
          default_value=-99,
          key_fn=lambda s: s.split(b'_X_')[0],
          top_k=2),
      dict(
          testcase_name='_key_fn_multi_coverage_top_k',
          x_data=[
              ['0_X_a', '0_X_a', '5_X_a', '6_X_a', '6_X_a', '0_X_a'],
              ['0_X_a', '2_X_a', '2_X_a', '2_X_a', '0_X_a', '5_X_a'],
              ['1_X_b', '1_X_b', '3_X_b', '3_X_b', '0_X_b', '1_X_b', '1_X_b']
          ],
          x_feature_spec=tf.io.VarLenFeature(tf.string),
          index_data=[[0, 0, -99, -99, -99, 0], [0, 2, 2, 2, 0, -99],
                      [1, 1, 3, 3, -99, 1, 1]],
          index_feature_spec=tf.io.VarLenFeature(tf.int64),
          index_domain=schema_pb2.IntDomain(
              min=-99, max=3, is_categorical=True),
          coverage_top_k=2,
          default_value=-99,
          key_fn=lambda s: s.split(b'_X_')[1],
          frequency_threshold=4),
      dict(
          testcase_name='_key_fn_and_labels',
          x_data=[
              'aaa', 'aaa', 'aaa', 'aab', 'aba', 'aba', 'aab', 'aab', 'aba',
              'abc', 'abc', 'aab'
          ],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          label_data=[1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
          label_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_data=[0, 0, 0, -1, -1, -1, -1, -1, -1, 1, 1, -1],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=1, is_categorical=True),
          coverage_top_k=1,
          key_fn=lambda s: s[:2],
          frequency_threshold=3),
      dict(
          testcase_name='_key_fn_and_weights',
          x_data=['xa', 'xa', 'xb', 'ya', 'yb', 'yc'],
          x_feature_spec=tf.io.FixedLenFeature([], tf.string),
          weight_data=[1.0, 0.5, 3.0, 0.6, 0.25, 0.5],
          weight_feature_spec=tf.io.FixedLenFeature([], tf.float32),
          index_data=[1, 1, 0, -1, -1, -1],
          index_feature_spec=tf.io.FixedLenFeature([], tf.int64),
          index_domain=schema_pb2.IntDomain(min=-1, max=1, is_categorical=True),
          coverage_top_k=1,
          key_fn=lambda s: s[0],
          frequency_threshold=1.5,
          coverage_frequency_threshold=1),
  ] + _EMPTY_VOCABULARY_PARAMS))
  def testComputeAndApplyVocabulary(self,
                                    x_data,
                                    x_feature_spec,
                                    index_data,
                                    index_feature_spec,
                                    index_domain,
                                    label_data=None,
                                    label_feature_spec=None,
                                    weight_data=None,
                                    weight_feature_spec=None,
                                    expected_vocab_file_contents=None,
                                    required_format=None,
                                    **kwargs):
    """Test tft.compute_and_apply_vocabulary with various inputs."""
    if required_format is not None and required_format != self._VocabFormat():
      raise tft_unit.SkipTest('Test only applicable to format: {}.'.format(
          self._VocabFormat()))

    input_data = [{'x': x} for x in x_data]
    input_feature_spec = {'x': x_feature_spec}
    expected_data = [{'index': index} for index in index_data]
    expected_feature_spec = {'index': index_feature_spec}
    expected_domains = {'index': index_domain}

    if label_data is not None:
      for idx, label in enumerate(label_data):
        input_data[idx]['label'] = label
      input_feature_spec['label'] = label_feature_spec

    if weight_data is not None:
      for idx, weight in enumerate(weight_data):
        input_data[idx]['weights'] = weight
      input_feature_spec['weights'] = weight_feature_spec

    input_metadata = tft_unit.metadata_from_feature_spec(input_feature_spec)
    expected_metadata = tft_unit.metadata_from_feature_spec(
        expected_feature_spec, expected_domains)

    def preprocessing_fn(inputs):
      x = inputs['x']
      labels = inputs.get('label')
      weights = inputs.get('weights')
      index = tft.compute_and_apply_vocabulary(
          x,
          labels=labels,
          weights=weights,
          file_format=self._VocabFormat(),
          **kwargs)
      return {'index': index}

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        expected_vocab_file_contents=expected_vocab_file_contents)

  @tft_unit.named_parameters(*_COMPOSITE_COMPUTE_AND_APPLY_VOCABULARY_TEST_CASES
                            )
  def testCompositeComputeAndApplyVocabulary(self, input_data, input_metadata,
                                             expected_data,
                                             expected_vocab_file_contents):

    def preprocessing_fn(inputs):
      index = tft.compute_and_apply_vocabulary(
          inputs['x'],
          file_format=self._VocabFormat(),
          vocab_filename='my_vocab')
      return {'index': index}

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_vocab_file_contents=expected_vocab_file_contents)

  # Example on how to use the vocab frequency as part of the transform
  # function.
  def testCreateVocabWithFrequency(self):
    input_data = [
        {'a': 'hello', 'b': 'world', 'c': 'aaaaa'},
        {'a': 'good', 'b': '', 'c': 'hello'},
        {'a': 'goodbye', 'b': 'hello', 'c': '\n'},
        {'a': '_', 'b': 'aaaaa', 'c': 'bbbbb'}
    ]
    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.string),
        'b': tf.io.FixedLenFeature([], tf.string),
        'c': tf.io.FixedLenFeature([], tf.string)
    })
    vocab_filename = 'test_vocab_with_frequency'

    def preprocessing_fn(inputs):
      deferred_vocab_and_filename = tft.vocabulary(
          tf.concat([inputs['a'], inputs['b'], inputs['c']], 0),
          vocab_filename=vocab_filename,
          store_frequency=True,
          file_format=self._VocabFormat())

      def _make_table_initializer(filename_tensor, is_frequency_value):
        if self._VocabFormat() == 'text':
          return tf.lookup.TextFileInitializer(
              filename=filename_tensor,
              key_dtype=tf.string,
              key_index=1,
              value_dtype=tf.int64,
              value_index=(0 if is_frequency_value else
                           tf.lookup.TextFileIndex.LINE_NUMBER),
              delimiter=' ')
        elif self._VocabFormat() == 'tfrecord_gzip':
          return tft.tf_utils.make_tfrecord_vocabulary_lookup_initializer(
              filename_tensor,
              return_indicator_as_value=is_frequency_value,
              has_indicator=True)

      def _apply_vocab(y, deferred_vocab_filename_tensor):
        initializer = _make_table_initializer(deferred_vocab_filename_tensor,
                                              False)
        table = tf.lookup.StaticHashTable(initializer, default_value=-1)
        table_size = table.size()
        return table.lookup(y), table_size

      def _apply_frequency(y, deferred_vocab_filename_tensor):
        initializer = _make_table_initializer(deferred_vocab_filename_tensor,
                                              True)
        table = tf.lookup.StaticHashTable(initializer, default_value=-1)
        return table.lookup(y), table.size()

      return {
          'index_a':
              tft.apply_vocabulary(
                  inputs['a'],
                  deferred_vocab_and_filename,
                  lookup_fn=_apply_vocab,
                  file_format=self._VocabFormat()),
          'frequency_a':
              tft.apply_vocabulary(
                  inputs['a'],
                  deferred_vocab_and_filename,
                  lookup_fn=_apply_frequency,
                  file_format=self._VocabFormat()),
          'index_b':
              tft.apply_vocabulary(
                  inputs['b'],
                  deferred_vocab_and_filename,
                  lookup_fn=_apply_vocab,
                  file_format=self._VocabFormat()),
          'frequency_b':
              tft.apply_vocabulary(
                  inputs['b'],
                  deferred_vocab_and_filename,
                  lookup_fn=_apply_frequency,
                  file_format=self._VocabFormat()),
      }

    expected_vocab = [(b'hello', 3), (b'aaaaa', 2), (b'world', 1),
                      (b'goodbye', 1), (b'good', 1), (b'bbbbb', 1), (b'_', 1),
                      (b'\n', 1), (b'', 1)]
    if self._VocabFormat() == 'text':
      expected_vocab = expected_vocab[:-2]
      empty_index = -1
      empty_frequency = -1
    else:
      empty_index = 8
      empty_frequency = 1
    expected_data = [
        # For tied frequencies, larger (lexicographic) items come first.
        {
            'index_a': 0,
            'frequency_a': 3,
            'index_b': 2,
            'frequency_b': 1
        },
        {
            'index_a': 4,
            'frequency_a': 1,
            'index_b': empty_index,
            'frequency_b': empty_frequency
        },
        {
            'index_a': 3,
            'frequency_a': 1,
            'index_b': 0,
            'frequency_b': 3
        },
        {
            'index_a': 6,
            'frequency_a': 1,
            'index_b': 1,
            'frequency_b': 2
        }
    ]
    size = len(expected_vocab) - 1
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {
            'index_a': tf.io.FixedLenFeature([], tf.int64),
            'index_b': tf.io.FixedLenFeature([], tf.int64),
            'frequency_a': tf.io.FixedLenFeature([], tf.int64),
            'frequency_b': tf.io.FixedLenFeature([], tf.int64),
        }, {
            'index_a':
                schema_pb2.IntDomain(min=-1, max=size, is_categorical=True),
            'index_b':
                schema_pb2.IntDomain(min=-1, max=size, is_categorical=True),
            'frequency_a':
                schema_pb2.IntDomain(min=-1, max=size, is_categorical=True),
            'frequency_b':
                schema_pb2.IntDomain(min=-1, max=size, is_categorical=True),
        })

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        expected_vocab_file_contents={vocab_filename: expected_vocab})

  def testVocabularyAnalyzerWithTokenization(self):
    def preprocessing_fn(inputs):
      return {
          'index':
              tft.compute_and_apply_vocabulary(
                  tf.compat.v1.strings.split(inputs['a']),
                  file_format=self._VocabFormat(),
                  vocab_filename='my_vocab')
      }

    input_data = [{'a': 'hello hello world'}, {'a': 'hello goodbye world'}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.string)})
    expected_data = [{'index': [0, 0, 1]}, {'index': [0, 2, 1]}]

    expected_metadata = tft_unit.metadata_from_feature_spec({
        'index': tf.io.VarLenFeature(tf.int64),
    }, {
        'index': schema_pb2.IntDomain(min=-1, max=2, is_categorical=True),
    })
    expected_vocabulary = {'my_vocab': [b'hello', b'world', b'goodbye']}
    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata=expected_metadata,
        expected_vocab_file_contents=expected_vocabulary)

  def testVocabularyWithFrequency(self):
    outfile = 'vocabulary_with_frequency'
    def preprocessing_fn(inputs):

      # Force the analyzer to be executed, and store the frequency file as a
      # side-effect.
      _ = tft.vocabulary(
          inputs['a'],
          vocab_filename=outfile,
          store_frequency=True,
          file_format=self._VocabFormat())
      _ = tft.vocabulary(
          inputs['a'], store_frequency=True, file_format=self._VocabFormat())
      _ = tft.vocabulary(
          inputs['b'], store_frequency=True, file_format=self._VocabFormat())

      # The following must not produce frequency output, just the vocab words.
      _ = tft.vocabulary(inputs['b'], file_format=self._VocabFormat())
      a_int = tft.compute_and_apply_vocabulary(
          inputs['a'], file_format=self._VocabFormat())

      # Return input unchanged, this preprocessing_fn is a no-op except for
      # computing uniques.
      return {'a_int': a_int}

    input_metadata = tft_unit.metadata_from_feature_spec({
        'a': tf.io.FixedLenFeature([], tf.string),
        'b': tf.io.FixedLenFeature([], tf.string)
    })

    tft_tmp_dir = os.path.join(self.get_temp_dir(), 'temp_dir')
    transform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')

    with beam_impl.Context(temp_dir=tft_tmp_dir):
      with self._makeTestPipeline() as pipeline:
        input_data = pipeline | beam.Create([
            {'a': 'hello', 'b': 'hi'},
            {'a': 'world', 'b': 'ho ho'},
            {'a': 'hello', 'b': 'ho ho'},
        ])
        transform_fn = (
            (input_data, input_metadata)
            | beam_impl.AnalyzeDataset(preprocessing_fn))
        _ = transform_fn | transform_fn_io.WriteTransformFn(transform_fn_dir)

    self.assertTrue(os.path.isdir(tft_tmp_dir))

    tft_output = tft.TFTransformOutput(transform_fn_dir)
    assets_path = os.path.join(tft_output.transform_savedmodel_dir,
                               tf.saved_model.ASSETS_DIRECTORY)
    self.assertTrue(os.path.isdir(assets_path))

    self.assertEqual([b'2 hello', b'1 world'],
                     tft_output.vocabulary_by_name(outfile))

    self.assertEqual(
        [b'2 hello', b'1 world'],
        tft_output.vocabulary_by_name('vocab_frequency_vocabulary_1'))

    self.assertEqual(
        [b'2 ho ho', b'1 hi'],
        tft_output.vocabulary_by_name('vocab_frequency_vocabulary_2'))

    self.assertEqual([b'ho ho', b'hi'],
                     tft_output.vocabulary_by_name('vocab_vocabulary_3'))

    self.assertEqual([b'hello', b'world'],
                     tft_output.vocabulary_by_name(
                         'vocab_compute_and_apply_vocabulary_vocabulary'))

  def testVocabularyWithKeyFnAndFrequency(self):
    def key_fn(string):
      return string.split(b'_X_')[1]

    outfile = 'vocabulary_with_frequency'

    def preprocessing_fn(inputs):

      # Force the analyzer to be executed, and store the frequency file as a
      # side-effect.

      _ = tft.vocabulary(
          tf.compat.v1.strings.split(inputs['a']),
          coverage_top_k=1,
          key_fn=key_fn,
          frequency_threshold=4,
          vocab_filename=outfile,
          store_frequency=True,
          file_format=self._VocabFormat())

      _ = tft.vocabulary(
          tf.compat.v1.strings.split(inputs['a']),
          coverage_top_k=1,
          key_fn=key_fn,
          frequency_threshold=4,
          store_frequency=True,
          file_format=self._VocabFormat())

      a_int = tft.compute_and_apply_vocabulary(
          tf.compat.v1.strings.split(inputs['a']),
          coverage_top_k=1,
          key_fn=key_fn,
          frequency_threshold=4,
          file_format=self._VocabFormat())

      # Return input unchanged, this preprocessing_fn is a no-op except for
      # computing uniques.
      return {'a_int': a_int}

    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.string)})

    tft_tmp_dir = os.path.join(self.get_temp_dir(), 'temp_dir')
    transform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')

    with beam_impl.Context(temp_dir=tft_tmp_dir):
      with self._makeTestPipeline() as pipeline:
        input_data = pipeline | beam.Create([
            {'a': '1_X_a 1_X_a 2_X_a 1_X_b 2_X_b'},
            {'a': '1_X_a 1_X_a 2_X_a 2_X_a'},
            {'a': '2_X_b 3_X_c 4_X_c'}
        ])
        transform_fn = (
            (input_data, input_metadata)
            | beam_impl.AnalyzeDataset(preprocessing_fn))
        _ = transform_fn | transform_fn_io.WriteTransformFn(transform_fn_dir)

    self.assertTrue(os.path.isdir(tft_tmp_dir))

    tft_output = tft.TFTransformOutput(transform_fn_dir)
    assets_path = os.path.join(tft_output.transform_savedmodel_dir,
                               tf.saved_model.ASSETS_DIRECTORY)
    self.assertTrue(os.path.isdir(assets_path))

    self.assertEqual([b'4 1_X_a', b'2 2_X_b', b'1 4_X_c'],
                     tft_output.vocabulary_by_name(outfile))

  def testVocabularyAnnotations(self):
    outfile = 'vocab.file'
    # Sanitization of vocabulary file names replaces '.' with '_'.
    annotation_file = 'vocab_file'
    if self._VocabFormat() == 'tfrecord_gzip':
      annotation_file = '{}.tfrecord.gz'.format(annotation_file)

    def preprocessing_fn(inputs):
      _ = tft.vocabulary(
          inputs['a'], vocab_filename=outfile, file_format=self._VocabFormat())
      tft.annotate_asset('key_1', annotation_file)
      return inputs

    input_metadata = tft_unit.metadata_from_feature_spec(
        {'a': tf.io.FixedLenFeature([], tf.string)})

    tft_tmp_dir = os.path.join(self.get_temp_dir(), 'temp_dir')
    transform_fn_dir = os.path.join(self.get_temp_dir(), 'export_transform_fn')

    with beam_impl.Context(temp_dir=tft_tmp_dir):
      with self._makeTestPipeline() as pipeline:
        input_data = pipeline | beam.Create([
            {
                'a': 'hello',
            },
            {
                'a': 'world',
            },
            {
                'a': 'hello',
            },
        ])
        transform_fn = ((input_data, input_metadata)
                        | beam_impl.AnalyzeDataset(preprocessing_fn))
        _, metadata = transform_fn
        self.assertDictEqual(metadata.asset_map, {
            'key_1': annotation_file,
            outfile: annotation_file
        })

        _ = transform_fn | transform_fn_io.WriteTransformFn(transform_fn_dir)

    self.assertTrue(os.path.isdir(tft_tmp_dir))

    tft_output = tft.TFTransformOutput(transform_fn_dir)
    assets_path = os.path.join(tft_output.transform_savedmodel_dir,
                               tf.saved_model.ASSETS_DIRECTORY)
    self.assertTrue(os.path.isdir(assets_path))

    self.assertEqual([b'hello', b'world'],
                     tft_output.vocabulary_by_name('key_1'))

  def testVocabularyPreSort(self):
    input_data = [
        dict(x=b'foo'),
        dict(x=b'hello'),
        dict(x=b'aaaaa'),
        dict(x=b'goodbye'),
        dict(x=b'bar'),
        dict(x=b'hello'),
        dict(x=b'goodbye'),
        dict(x=b'hello'),
        dict(x=b'hello'),
        dict(x=b'goodbye'),
        dict(x=b'aaaaa'),
    ]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.string)})
    expected_vocab_file_contents = [(b'hello', 4), (b'goodbye', 3),
                                    (b'aaaaa', 2), (b'foo', 1), (b'bar', 1)]

    def preprocessing_fn(inputs):
      tft.vocabulary(
          inputs['x'],
          vocab_filename='my_vocab',
          file_format=self._VocabFormat(),
          store_frequency=True)
      return inputs

    with tf.compat.v1.test.mock.patch.object(analyzer_impls,
                                             '_PRESORT_BATCH_SIZE', 2):
      self.assertAnalyzeAndTransformResults(
          input_data,
          input_metadata,
          preprocessing_fn,
          input_data,
          input_metadata,
          expected_vocab_file_contents={
              'my_vocab': expected_vocab_file_contents
          })

  def testVocabularyWithUserDefinedLookupFnFeedsSecondAnalyzer(self):
    input_data = [
        dict(x=b'bar'),
        dict(x=b'foo'),
        dict(x=b'bar'),
        dict(x=b'bar'),
        dict(x=b'foo'),
    ]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.string)})
    expected_data = [
        dict(x=b'bar', x_int=0, x_int_mean=0.4),
        dict(x=b'bar', x_int=0, x_int_mean=0.4),
        dict(x=b'bar', x_int=0, x_int_mean=0.4),
        dict(x=b'foo', x_int=1, x_int_mean=0.4),
        dict(x=b'foo', x_int=1, x_int_mean=0.4),
    ]
    expected_vocab_file_contents = [(b'bar'), (b'foo')]
    size = len(expected_vocab_file_contents) - 1
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {
            'x': tf.io.FixedLenFeature([], tf.string),
            'x_int': tf.io.FixedLenFeature([], tf.int64),
            'x_int_mean': tf.io.FixedLenFeature([], tf.float32)
        },
        domains={
            'x_int': schema_pb2.IntDomain(
                min=-1, max=size, is_categorical=True)
        })

    def preprocessing_fn(inputs):

      def _make_table_initializer(filename_tensor):
        if self._VocabFormat() == 'text':
          return tf.lookup.TextFileInitializer(
              filename=filename_tensor,
              key_dtype=tf.string,
              key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
              value_dtype=tf.int64,
              value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
        elif self._VocabFormat() == 'tfrecord_gzip':
          return tft.tf_utils.make_tfrecord_vocabulary_lookup_initializer(
              filename_tensor, return_indicator_as_value=False)

      def _apply_vocab(y, deferred_vocab_filename_tensor):
        initializer = _make_table_initializer(deferred_vocab_filename_tensor)
        table = tf.lookup.StaticHashTable(initializer, default_value=-1)
        table_size = table.size()
        return table.lookup(y), table_size

      deferred_vocab_and_filename = tft.vocabulary(
          inputs['x'],
          vocab_filename='my_vocab',
          file_format=self._VocabFormat())
      x_int = tft.apply_vocabulary(
          inputs['x'],
          deferred_vocab_and_filename,
          lookup_fn=_apply_vocab,
          file_format=self._VocabFormat())

      x_int_mean = tf.zeros_like(x_int, dtype=tf.float32) + tft.mean(x_int)
      return {'x': inputs['x'], 'x_int': x_int, 'x_int_mean': x_int_mean}

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        expected_vocab_file_contents={'my_vocab': expected_vocab_file_contents})

  def testVocabularyWithTableDefinedInPreprocessingFnFeedsSecondAnalyzer(self):
    if self._VocabFormat() != 'text':
      raise tft_unit.SkipTest('Test only applicable to text format.')

    input_data = [
        dict(x=b'bar'),
        dict(x=b'foo'),
        dict(x=b'bar'),
        dict(x=b'bar'),
        dict(x=b'foo'),
    ]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.string)})
    expected_data = [
        dict(x=b'bar', x_int=0, x_int_mean=0.4),
        dict(x=b'bar', x_int=0, x_int_mean=0.4),
        dict(x=b'bar', x_int=0, x_int_mean=0.4),
        dict(x=b'foo', x_int=1, x_int_mean=0.4),
        dict(x=b'foo', x_int=1, x_int_mean=0.4),
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.string),
        'x_int': tf.io.FixedLenFeature([], tf.int64),
        'x_int_mean': tf.io.FixedLenFeature([], tf.float32)
    })
    expected_vocab_file_contents = [(b'bar'), (b'foo')]

    def preprocessing_fn(inputs):
      vocab_path = tft.vocabulary(
          inputs['x'],
          vocab_filename='my_vocab',
          file_format=self._VocabFormat())
      initializer = tf.lookup.TextFileInitializer(
          vocab_path,
          key_dtype=tf.string,
          key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
          value_dtype=tf.int64,
          value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
      table = tf.lookup.StaticHashTable(initializer, default_value=-1)
      x_int = table.lookup(inputs['x'])
      x_int_mean = tf.zeros_like(x_int, dtype=tf.float32) + tft.mean(x_int)
      return {'x': inputs['x'], 'x_int': x_int, 'x_int_mean': x_int_mean}

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        expected_vocab_file_contents={'my_vocab': expected_vocab_file_contents})

  def testStringOpsWithAutomaticControlDependencies(self):

    def preprocessing_fn(inputs):
      month_str = tf.strings.substr(
          inputs['date'], pos=5, len=3, unit='UTF8_CHAR')

      # The table created here will add an automatic control dependency.
      month_int = tft.compute_and_apply_vocabulary(month_str)
      return {'month_int': month_int}

    input_data = [{'date': '2021-May-31'}, {'date': '2021-Jun-01'}]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'date': tf.io.FixedLenFeature([], tf.string)})
    expected_data = [{'month_int': 0}, {'month_int': 1}]
    max_index = len(expected_data) - 1
    expected_metadata = tft_unit.metadata_from_feature_spec(
        {
            'month_int': tf.io.FixedLenFeature([], tf.int64),
        }, {
            'month_int':
                schema_pb2.IntDomain(
                    min=-1, max=max_index, is_categorical=True),
        })

    self.assertAnalyzeAndTransformResults(input_data, input_metadata,
                                          preprocessing_fn, expected_data,
                                          expected_metadata)

  def testVocabularyOneHotEncoding(self):

    input_data = [
        dict(x=b'bar'),
        dict(x=b'foo'),
        dict(x=b'bar'),
        dict(x=b'bar'),
        dict(x=b'foo'),
    ]
    input_metadata = tft_unit.metadata_from_feature_spec(
        {'x': tf.io.FixedLenFeature([], tf.string)})
    expected_data = [
        dict(x=b'bar', x_encoded=[1], x_encoded_centered=[0.4]),
        dict(x=b'bar', x_encoded=[1], x_encoded_centered=[0.4]),
        dict(x=b'bar', x_encoded=[1], x_encoded_centered=[0.4]),
        dict(x=b'foo', x_encoded=[0], x_encoded_centered=[-0.6]),
        dict(x=b'foo', x_encoded=[0], x_encoded_centered=[-0.6]),
    ]
    expected_metadata = tft_unit.metadata_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.string),
        'x_encoded': tf.io.FixedLenFeature([1], tf.int64),
        'x_encoded_centered': tf.io.FixedLenFeature([1], tf.float32),
    })
    expected_vocab_file_contents = [(b'bar')]

    def preprocessing_fn(inputs):
      x_int = tft.compute_and_apply_vocabulary(
          inputs['x'],
          vocab_filename='my_vocab',
          file_format=self._VocabFormat(),
          frequency_threshold=3)

      depth = tft.experimental.get_vocabulary_size_by_name('my_vocab')
      x_encoded = tf.one_hot(
          x_int, depth=tf.cast(depth, tf.int32), dtype=tf.int64)
      # Add a second phase that depends on vocabulary size.
      x_encoded_centered = (
          tf.cast(x_encoded, dtype=tf.float32) - tft.mean(x_encoded))
      return {
          'x': inputs['x'],
          'x_encoded': x_encoded,
          'x_encoded_centered': x_encoded_centered
      }

    self.assertAnalyzeAndTransformResults(
        input_data,
        input_metadata,
        preprocessing_fn,
        expected_data,
        expected_metadata,
        expected_vocab_file_contents={'my_vocab': expected_vocab_file_contents})


if __name__ == '__main__':
  tft_unit.main()
