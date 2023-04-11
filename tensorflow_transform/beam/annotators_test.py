# coding=utf-8
#
# Copyright 2023 Google Inc. All Rights Reserved.
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
"""Tests for tft annotators."""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import tft_unit
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


_TF_VERSION_NAMED_PARAMETERS = [
    dict(testcase_name='CompatV1', use_tf_compat_v1=True),
    dict(testcase_name='V2', use_tf_compat_v1=False),
]


class AnnotatorsTest(tft_unit.TransformTestCase):

  @tft_unit.named_parameters(*_TF_VERSION_NAMED_PARAMETERS)
  def test_annotate_sparse_outputs(self, use_tf_compat_v1):
    def preprocessing_fn(inputs):
      outputs = inputs.copy()
      x = tf.sparse.expand_dims(inputs['x'], -1)
      outputs['x'] = x
      tft.experimental.annotate_sparse_output_shape(x, tf.constant([1, 1]))
      tft.experimental.annotate_sparse_output_shape(outputs['y'], [17])
      tft.experimental.annotate_true_sparse_output(outputs['z'])
      return outputs

    input_data_dicts = [dict(x=[1], y=[2], z=[3], t=[4]) for x in range(10)]
    input_metadata = tft.DatasetMetadata.from_feature_spec({
        'x': tf.io.VarLenFeature(tf.int64),
        'y': tf.io.VarLenFeature(tf.int64),
        'z': tf.io.VarLenFeature(tf.int64),
        't': tf.io.VarLenFeature(tf.int64),
    })
    schema = text_format.Parse(
        """
        feature {
          name: "t"
          type: INT
        }
        feature {
          name: "x$sparse_indices_0"
          type: INT
          int_domain {
            min: 0
            max: 0
          }
        }
        feature {
          name: "x$sparse_indices_1"
          type: INT
          int_domain {
            min: 0
            max: 0
          }
        }
        feature {
          name: "x$sparse_values"
          type: INT
        }
        feature {
          name: "y$sparse_indices_0"
          type: INT
          int_domain {
            min: 0
            max: 16
          }
        }
        feature {
          name: "y$sparse_values"
          type: INT
        }
        feature {
          name: "z$sparse_indices_0"
          type: INT
        }
        feature {
          name: "z$sparse_values"
          type: INT
        }
        sparse_feature {
          name: "x"
          index_feature {
            name: "x$sparse_indices_0"
          }
          index_feature {
            name: "x$sparse_indices_1"
          }
          is_sorted: true
          value_feature {
            name: "x$sparse_values"
          }
        }
        sparse_feature {
          name: "y"
          index_feature {
            name: "y$sparse_indices_0"
          }
          is_sorted: true
          value_feature {
            name: "y$sparse_values"
          }
        }
        sparse_feature {
          name: "z"
          index_feature {
            name: "z$sparse_indices_0"
          }
          is_sorted: true
          value_feature {
            name: "z$sparse_values"
          }
        }
    """,
        schema_pb2.Schema(),
    )
    if not tft_unit.is_external_environment():
      schema.generate_legacy_feature_spec = False
    self.assertAnalyzeAndTransformResults(
        input_data_dicts,
        input_metadata,
        preprocessing_fn,
        expected_metadata=tft.DatasetMetadata(schema),
        force_tf_compat_v1=use_tf_compat_v1,
        output_record_batches=True,
    )

  @tft_unit.named_parameters(*_TF_VERSION_NAMED_PARAMETERS)
  def test_conflicting_sparse_outputs_annotations(self, use_tf_compat_v1):
    def preprocessing_fn(inputs):
      tft.experimental.annotate_sparse_output_shape(inputs['x'], [3])
      tft.experimental.annotate_sparse_output_shape(inputs['x'], [17])
      tft.experimental.annotate_true_sparse_output(inputs['x'])
      return inputs

    input_data_dicts = [dict(x=[1]) for x in range(10)]
    input_metadata = tft.DatasetMetadata.from_feature_spec(
        {
            'x': tf.io.VarLenFeature(tf.int64),
        }
    )
    schema = text_format.Parse(
        """
      feature {
        name: "x$sparse_indices_0"
        type: INT
        int_domain {
          min: 0
          max: 16
        }
      }
      feature {
        name: "x$sparse_values"
        type: INT
      }
      sparse_feature {
        name: "x"
        index_feature {
          name: "x$sparse_indices_0"
        }
        is_sorted: true
        value_feature {
          name: "x$sparse_values"
        }
      }
    """,
        schema_pb2.Schema(),
    )
    if not tft_unit.is_external_environment():
      schema.generate_legacy_feature_spec = False
    self.assertAnalyzeAndTransformResults(
        input_data_dicts,
        input_metadata,
        preprocessing_fn,
        expected_metadata=tft.DatasetMetadata(schema),
        force_tf_compat_v1=use_tf_compat_v1,
        output_record_batches=True,
    )

  @tft_unit.named_parameters(*_TF_VERSION_NAMED_PARAMETERS)
  def test_invalid_sparse_outputs_annotations(self, use_tf_compat_v1):
    def preprocessing_fn(inputs):
      tft.experimental.annotate_sparse_output_shape(inputs['x'], [3, 42])
      return inputs

    input_data_dicts = [dict(x=[1]) for x in range(10)]
    input_metadata = tft.DatasetMetadata.from_feature_spec(
        {
            'x': tf.io.VarLenFeature(tf.int64),
        }
    )
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        r'Annotated shape \[3, 42\] was expected to have rank 1',
    ):
      self.assertAnalyzeAndTransformResults(
          input_data_dicts,
          input_metadata,
          preprocessing_fn,
          force_tf_compat_v1=use_tf_compat_v1,
      )

  @tft_unit.named_parameters(
      dict(
          testcase_name='sanity',
          values=['hello', 'world', 'world'],
          expected_size=2,
      ),
      dict(
          testcase_name='single_token',
          values=['hello', 'hello', 'hello'],
          expected_size=1,
      ),
      dict(
          testcase_name='empty',
          values=['', '', ''],
          expected_size=1,
      ),
  )
  def test_get_vocabulary_size_by_name(self, values, expected_size):
    vocab_filename = 'vocab'

    def preprocessing_fn(inputs):
      tft.vocabulary(inputs['s'], vocab_filename=vocab_filename)
      size = tf.zeros_like(
          inputs['s'], dtype=tf.int64
      ) + tft.experimental.get_vocabulary_size_by_name(vocab_filename)
      return {'size': size}

    input_data_dicts = [dict(s=v) for v in values]
    input_metadata = tft.DatasetMetadata.from_feature_spec({
        's': tf.io.FixedLenFeature([], tf.string),
    })
    expected_data = [{
        'size': expected_size,
    }] * len(values)
    self.assertAnalyzeAndTransformResults(
        input_data_dicts,
        input_metadata,
        preprocessing_fn,
        force_tf_compat_v1=False,
        expected_data=expected_data,
    )


if __name__ == '__main__':
  tft_unit.main()
