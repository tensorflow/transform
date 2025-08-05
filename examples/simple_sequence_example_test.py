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
"""Tests for simple_example."""

import simple_sequence_example
import tensorflow as tf

from tensorflow_transform.beam import tft_unit

_EXPECTED_TRANSFORMED_OUTPUT = [
    {
        "transformed_seq_int_feature$ragged_values": [
            0.0,
            0.09090909,
            0.18181818,
            0.27272727,
        ],
        "transformed_seq_int_feature$row_lengths_1": [2, 2],
        "transformed_seq_string_feature$ragged_values": [5, 4],
        "transformed_seq_string_feature$row_lengths_1": [2, 0],
        "transformed_float_feature": [0.0, 0.0, 0.75, 0.8],
        "transformed_int_feature": [0],
    },
    {
        "transformed_seq_int_feature$ragged_values": [
            0.36363636,
            0.45454545,
            0.54545454,
            0.63636363,
        ],
        "transformed_seq_int_feature$row_lengths_1": [2, 2],
        "transformed_seq_string_feature$ragged_values": [1, 3],
        "transformed_seq_string_feature$row_lengths_1": [2, 0],
        "transformed_float_feature": [0.5, 0.5, 1.0, 1.0],
        "transformed_int_feature": [0.5],
    },
    {
        "transformed_seq_int_feature$ragged_values": [
            0.72727272,
            0.81818181,
            0.90909090,
            1.0,
        ],
        "transformed_seq_int_feature$row_lengths_1": [2, 2],
        "transformed_seq_string_feature$ragged_values": [0, 2],
        "transformed_seq_string_feature$row_lengths_1": [2, 0],
        "transformed_float_feature": [1.0, 1.0, 0.0, 0.0],
        "transformed_int_feature": [1],
    },
]


class SimpleMainTest(tf.test.TestCase):
    def testMainDoesNotCrash(self):
        tft_unit.skip_if_not_tf2("Tensorflow 2.x required.")
        simple_sequence_example.main()


class SimpleSequenceExampleTest(tft_unit.TransformTestCase):
    def testPreprocessingFn(self):
        tft_unit.skip_if_not_tf2("Tensorflow 2.x required.")
        tfxio = simple_sequence_example._make_tfxio(simple_sequence_example._SCHEMA)
        self.assertAnalyzeAndTransformResults(
            tfxio.BeamSource(),
            tfxio.TensorAdapterConfig(),
            simple_sequence_example._preprocessing_fn,
            output_record_batches=True,
            expected_data=_EXPECTED_TRANSFORMED_OUTPUT,
        )


if __name__ == "__main__":
    tf.test.main()
