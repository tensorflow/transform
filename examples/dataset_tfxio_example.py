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
"""Simple Example of DatasetTFXIO usage."""

import pprint
import tempfile

import apache_beam as beam
import tensorflow as tf
from absl import app
from tfx_bsl.tfxio import dataset_tfxio

import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam


def _print_record_batch(data):
    pprint.pprint(data.to_pydict())


def _preprocessing_fn(inputs):
    return {
        "x_centered": tf.cast(inputs["feature0"], tf.float32)
        - tft.mean(inputs["feature0"]),
        "x_scaled": tft.scale_by_min_max(inputs["feature0"]),
    }


def _make_tfxio() -> dataset_tfxio.DatasetTFXIO:
    """Make DatasetTFXIO."""
    num_elements = 9
    batch_size = 2
    dataset = tf.data.Dataset.range(num_elements).batch(batch_size)

    return dataset_tfxio.DatasetTFXIO(dataset=dataset)


def main(args):
    del args

    input_tfxio = _make_tfxio()

    # User-Defined Processing Pipeline
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
            raw_dataset = (
                pipeline | "ReadRecordBatch" >> input_tfxio.BeamSource(batch_size=5),
                input_tfxio.TensorAdapterConfig(),
            )
            (transformed_data, _), _ = (
                raw_dataset
                | "AnalyzeAndTransform"
                >> tft_beam.AnalyzeAndTransformDataset(
                    _preprocessing_fn, output_record_batches=True
                )
            )
            transformed_data = transformed_data | "ExtractRecordBatch" >> beam.Keys()
            _ = transformed_data | "PrintTransformedData" >> beam.Map(
                _print_record_batch
            )


if __name__ == "__main__":
    app.run(main)
