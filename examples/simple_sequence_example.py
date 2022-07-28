# Copyright 2022 Google Inc. All Rights Reserved.
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
"""Example of reading SequenceExample in tf.Transform."""

import os
import tempfile

from absl import logging
import apache_beam as beam
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tfx_bsl.public import tfxio

from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf import text_format

_TRANSFORM_TEMP_DIR = 'tft_temp'
_SCHEMA = text_format.Parse(
    """
  feature {
    name: "int_feature"
    type: INT
    value_count {
      min: 1
      max: 1
    }
  }
  feature {
    name: "float_feature"
    type: FLOAT
    value_count {
      min: 4
      max: 4
    }
  }
  feature {
    name: "##SEQUENCE##"
    type: STRUCT
    struct_domain {
      feature {
        name: "int_feature"
        type: INT
        value_count {
          min: 0
          max: 2
        }
      }
      feature {
        name: "string_feature"
        type: BYTES
        value_count {
          min: 0
          max: 2
        }
      }
    }
  }
  tensor_representation_group {
    key: ""
    value {
      tensor_representation {
        key: "int_feature"
        value { varlen_sparse_tensor { column_name: "int_feature" } }
      }
      tensor_representation {
        key: "float_feature"
        value { varlen_sparse_tensor { column_name: "float_feature" } }
      }
      tensor_representation {
        key: "seq_string_feature"
        value { ragged_tensor {
                    feature_path { step: "##SEQUENCE##" step: "string_feature" } } }
      }
      tensor_representation {
        key: "seq_int_feature"
        value { ragged_tensor {
                    feature_path { step: "##SEQUENCE##" step: "int_feature" } } }
      }
    }
  }
""", schema_pb2.Schema())

_TELEMETRY_DESCRIPTORS = ['TFT', 'SequenceExample']


def _print_record_batch(data):
  logging.info(data.to_pydict())


def _make_tfxio(schema):
  """Creates TFXIO for SequenceExample.

  Args:
    schema: A TFMD Schema describing the dataset.

  Returns:
   TFSequenceExampleRecord TFXIO Instance.

  The data_tfrecord.gz file holds Serialized SequenceExample as below:
    context {
      feature { key: "int_feature" value { int64_list { value: [0] } } }
      feature {
        key: "float_feature"
        value { float_list { value: [1.0, 2.0, 3.0, 4.0] } }
      }
    }
    feature_lists {
      feature_list {
        key: "int_feature"
        value {
          feature { int64_list { value: [1, 2] } }
          feature { int64_list { value: [3, 4] } }
        }
      }
      feature_list {
        key: "string_feature"
        value {
          feature { bytes_list { value: ["Hello", "World"] } }
          feature { bytes_list { value: [] } }
        }
      }
    }
  """
  sequence_example_file = os.path.join(
      os.path.dirname(__file__), 'testdata/sequence_example/data_tfrecord.gz')
  return tfxio.TFSequenceExampleRecord(
      sequence_example_file,
      schema=schema,
      telemetry_descriptors=_TELEMETRY_DESCRIPTORS)


def _preprocessing_fn(inputs):
  """Preprocess input columns into transformed columns.

  Args:
    inputs: Input Tensors.

  Returns:
   Dictionary of respective transformed inputs

  Example:
    `int_features`: tft.scale_to_0_1(...)
      Input:    [[[0]],   [[1]], [[2]]]
      Output:   [[[0]], [[0.5]], [[1]]]

    `float_features`: tft.scale_to_0_1(.., elementwise = True)
      Input:  [
          [[1.0, 2.0, 3.0, 4.0]],
          [[2.0, 3.0, 4.0, 5.0]],
          [[3.0, 4.0, 0.0, 0.0]]
      ]
      Output: [
          [[0.0, 0.0, 0.75, 0.8]],
          [[0.5, 0.5,  1.0, 1.0]],
          [[1.0, 1.0,  0.0, 0.0]]
      ]

    `seq_int_feature`: tft.scale_by_min_max(...)
      Input: [
          [ [1, 2],   [3, 4]],
          [ [5, 6],   [7, 8]],
          [[9, 10], [11, 12]]
      ]
      Output: [
          [[   0.0, 0.0909], [0.1818, 0.2727]],
          [[0.3636, 0.4545], [0.5454, 0.6363]],
          [[0.7272, 0.8181], [0.9090,    1.0]]
      ]

    `seq_string_feature`: tft.compute_and_apply_vocabulary(...)
      Input: [
          [[ b'Hello', b'World'], []],
          [[   b'foo',   b'bar'], []],
          [[b'tensor',  b'flow'], []]
      ]
      Output: [
          [[[5, 4], []]],
          [[[1, 3], []]],
          [[[0, 2], []]]
      ]
  """
  return {
      'transformed_seq_int_feature':
          tft.scale_by_min_max(inputs['seq_int_feature']),
      'transformed_seq_string_feature':
          tft.compute_and_apply_vocabulary(inputs['seq_string_feature']),
      'transformed_float_feature':
          tft.scale_to_0_1(inputs['float_feature'], elementwise=True),
      'transformed_int_feature':
          tft.scale_to_0_1(inputs['int_feature']),
  }


def _transform_data(sequence_example_tfxio):
  """Transform the data and output transformed values.

  Args:
    sequence_example_tfxio: tfxio.TFSequenceExampleRecord Object
  """

  with beam.Pipeline() as pipeline:
    with tft_beam.Context(
        temp_dir=os.path.join(tempfile.mkdtemp(), _TRANSFORM_TEMP_DIR)):

      raw_data = pipeline | 'ReadAndDecode' >> sequence_example_tfxio.BeamSource(
      )
      _ = raw_data | 'PrintInputData' >> beam.Map(_print_record_batch)

      (transformed_data,
       _), _ = ((raw_data, sequence_example_tfxio.TensorAdapterConfig())
                | 'AnalyzeAndTransform' >> tft_beam.AnalyzeAndTransformDataset(
                    _preprocessing_fn, output_record_batches=True))

      # Drop empty pass-through features dictionary that is not relevant
      # for this example.
      transformed_data = transformed_data | 'ExtractRecordBatch' >> beam.Keys()
      _ = transformed_data | 'PrintTransformedData' >> beam.Map(
          _print_record_batch)


def main():
  _transform_data(_make_tfxio(_SCHEMA))


if __name__ == '__main__':
  main()
