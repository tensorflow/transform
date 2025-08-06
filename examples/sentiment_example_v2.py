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
"""Example of sentiment analysis using IMDB movie review dataset."""

# pylint: disable=g-bad-import-order
import argparse
import math
import os
import pprint
import tempfile

import apache_beam as beam
import tensorflow as tf
from absl import logging
from tfx_bsl.public import tfxio

import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.keras_lib import tf_keras

VOCAB_SIZE = 20000
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_EPOCHS = 200
NUM_TRAIN_INSTANCES = 25000
NUM_TEST_INSTANCES = 25000

REVIEW_KEY = "review"
REVIEW_WEIGHT_KEY = "review_weight"
LABEL_KEY = "label"

RAW_DATA_FEATURE_SPEC = {
    REVIEW_KEY: tf.io.FixedLenFeature([], tf.string),
    LABEL_KEY: tf.io.FixedLenFeature([], tf.int64),
}

SCHEMA = tft.DatasetMetadata.from_feature_spec(RAW_DATA_FEATURE_SPEC).schema

DELIMITERS = ".,!?() "

# Names of temp files
SHUFFLED_TRAIN_DATA_FILEBASE = "train_shuffled"
SHUFFLED_TEST_DATA_FILEBASE = "test_shuffled"
TRANSFORMED_TRAIN_DATA_FILEBASE = "train_transformed"
TRANSFORMED_TEST_DATA_FILEBASE = "test_transformed"
TRANSFORM_TEMP_DIR = "tft_temp"
EXPORTED_MODEL_DIR = "exported_model_dir"

# Functions for preprocessing


# pylint: disable=invalid-name
@beam.ptransform_fn
def Shuffle(pcoll):
    """Shuffles a PCollection.  Collection should not contain duplicates."""
    return (
        pcoll
        | "PairWithHash" >> beam.Map(lambda x: (hash(x), x))
        | "GroupByHash" >> beam.GroupByKey()
        | "DropHash" >> beam.FlatMap(lambda hash_and_values: hash_and_values[1])
    )


# pylint: disable=invalid-name
@beam.ptransform_fn
def ReadAndShuffleData(pcoll, filepatterns):
    """Read a train or test dataset from disk and shuffle it."""
    # NOTE: we pass filepatterns as a tuple instead of two args, as the current
    # version of beam assumes that if the first arg to a ptransfrom_fn is a
    # string, then that string is the label.
    neg_filepattern, pos_filepattern = filepatterns

    # Read from each file pattern and create a tuple of the review text and the
    # correct label.
    negative_examples = (
        pcoll
        | "ReadNegativeExamples" >> beam.io.ReadFromText(neg_filepattern)
        | "PairWithZero" >> beam.Map(lambda review: (review, 0))
    )
    positive_examples = (
        pcoll
        | "ReadPositiveExamples" >> beam.io.ReadFromText(pos_filepattern)
        | "PairWithOne" >> beam.Map(lambda review: (review, 1))
    )
    all_examples = [negative_examples, positive_examples] | "Merge" >> beam.Flatten()

    # Shuffle the data.  Note that the data does in fact contain duplicate reviews
    # for reasons that are unclear.  This means that NUM_TRAIN_INSTANCES and
    # NUM_TRAIN_INSTANCES are slightly wrong for the preprocessed data.
    # pylint: disable=no-value-for-parameter
    shuffled_examples = (
        all_examples | "Distinct" >> beam.Distinct() | "Shuffle" >> Shuffle()
    )

    # Put the data in the format that can be accepted directly by tf.Transform.
    return shuffled_examples | "MakeInstances" >> beam.Map(
        lambda p: {REVIEW_KEY: p[0], LABEL_KEY: p[1]}
    )


def read_and_shuffle_data(
    train_neg_filepattern: str,
    train_pos_filepattern: str,
    test_neg_filepattern: str,
    test_pos_filepattern: str,
    working_dir: str,
):
    """Read and shuffle the data and write out as a TFRecord of Example protos.

    Read in the data from the positive and negative examples on disk, shuffle it
    and write it out in TFRecord format.
    transform it using a preprocessing pipeline that removes punctuation,
    tokenizes and maps tokens to int64 values indices.

    Args:
    ----
      train_neg_filepattern: Filepattern for training data negative examples
      train_pos_filepattern: Filepattern for training data positive examples
      test_neg_filepattern: Filepattern for test data negative examples
      test_pos_filepattern: Filepattern for test data positive examples
      working_dir: Directory to write shuffled data to
    """
    with beam.Pipeline() as pipeline:
        coder = tft.coders.ExampleProtoCoder(SCHEMA)

        # pylint: disable=no-value-for-parameter
        _ = (
            pipeline
            | "ReadAndShuffleTrain"
            >> ReadAndShuffleData((train_neg_filepattern, train_pos_filepattern))
            | "EncodeTrainData" >> beam.Map(coder.encode)
            | "WriteTrainData"
            >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, SHUFFLED_TRAIN_DATA_FILEBASE)
            )
        )

        _ = (
            pipeline
            | "ReadAndShuffleTest"
            >> ReadAndShuffleData((test_neg_filepattern, test_pos_filepattern))
            | "EncodeTestData" >> beam.Map(coder.encode)
            | "WriteTestData"
            >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, SHUFFLED_TEST_DATA_FILEBASE)
            )
        )
        # pylint: enable=no-value-for-parameter


def transform_data(working_dir: str):
    """Transform the data and write out as a TFRecord of Example protos.

    Read in the data from the positive and negative examples on disk, and
    transform it using a preprocessing pipeline that removes punctuation,
    tokenizes and maps tokens to int64 values indices.

    Args:
    ----
      working_dir: Directory to read shuffled data from and write transformed data
          and metadata to.
    """
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir=os.path.join(working_dir, TRANSFORM_TEMP_DIR)):
            tfxio_train_data = tfxio.TFExampleRecord(
                file_pattern=os.path.join(
                    working_dir, SHUFFLED_TRAIN_DATA_FILEBASE + "*"
                ),
                schema=SCHEMA,
            )
            train_data = pipeline | "TFXIORead[Train]" >> tfxio_train_data.BeamSource()

            tfxio_test_data = tfxio.TFExampleRecord(
                file_pattern=os.path.join(
                    working_dir, SHUFFLED_TEST_DATA_FILEBASE + "*"
                ),
                schema=SCHEMA,
            )
            test_data = pipeline | "TFXIORead[Test]" >> tfxio_test_data.BeamSource()

            def preprocessing_fn(inputs):
                """Preprocess input columns into transformed columns."""
                review = inputs[REVIEW_KEY]

                # Here tf.compat.v1.string_split behaves differently from
                # tf.strings.split.
                review_tokens = tf.compat.v1.string_split(review, DELIMITERS)
                review_indices = tft.compute_and_apply_vocabulary(
                    review_tokens, top_k=VOCAB_SIZE
                )
                # Add one for the oov bucket created by compute_and_apply_vocabulary.
                review_bow_indices, review_weight = tft.tfidf(
                    review_indices, VOCAB_SIZE + 1
                )
                return {
                    REVIEW_KEY: review_bow_indices,
                    REVIEW_WEIGHT_KEY: review_weight,
                    LABEL_KEY: tf.one_hot(inputs[LABEL_KEY], 2),
                }

            # The TFXIO output format is chosen for improved performance.
            transformed_train_data, transform_fn = (
                train_data,
                tfxio_train_data.TensorAdapterConfig(),
            ) | "AnalyzeAndTransform" >> tft_beam.AnalyzeAndTransformDataset(
                preprocessing_fn, output_record_batches=True
            )

            transformed_test_data = (
                (test_data, tfxio_test_data.TensorAdapterConfig()),
                transform_fn,
            ) | "Transform" >> tft_beam.TransformDataset(output_record_batches=True)

            # Extract transformed RecordBatches, encode and write them to the given
            # directory.
            _ = (
                transformed_train_data
                | "EncodeTrainData" >> tft_beam.EncodeTransformedDataset()
                | "WriteTrainData"
                >> beam.io.WriteToTFRecord(
                    os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)
                )
            )

            _ = (
                transformed_test_data
                | "EncodeTestData" >> tft_beam.EncodeTransformedDataset()
                | "WriteTestData"
                >> beam.io.WriteToTFRecord(
                    os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)
                )
            )

            # Will write a SavedModel and metadata to two subdirectories of
            # working_dir, given by tft.TRANSFORM_FN_DIR and
            # tft.TRANSFORMED_METADATA_DIR respectively.
            _ = transform_fn | "WriteTransformFn" >> tft_beam.WriteTransformFn(
                working_dir
            )


# Functions for training


def _input_fn(
    tf_transform_output: tft.TFTransformOutput,
    transformed_examples: str,
    batch_size: int,
):
    """Creates an input function reading from transformed data.

    Args:
    ----
      tf_transform_output: Wrapper around output of tf.Transform.
      transformed_examples: Base filename of examples.
      batch_size: Batch size.

    Returns:
    -------
      The input function for training or eval.
    """
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=transformed_examples,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=tf.data.TFRecordDataset,
        label_key=LABEL_KEY,
        shuffle=True,
    ).prefetch(tf.data.experimental.AUTOTUNE)


def export_serving_model(
    tf_transform_output: tft.TFTransformOutput,
    model: tf_keras.Model,
    output_dir: str,
):
    """Creates an input function reading from raw data.

    Args:
    ----
      tf_transform_output: Wrapper around output of tf.Transform.
      model: The keras model to export.
      output_dir: A path to export the model to.

    Returns:
    -------
      The serving input function.
    """
    # The layer has to be saved to the model for keras tracking purpases.
    model.tft_layer = tf_transform_output.transform_features_layer()

    raw_feature_spec = RAW_DATA_FEATURE_SPEC.copy()
    # Remove label since it is not available during serving.
    raw_feature_spec.pop(LABEL_KEY)

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Serving tf.function model wrapper."""
        parsed_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)
        classes_names = tf.constant([["0", "1"]])
        classes = tf.tile(classes_names, [tf.shape(outputs)[0], 1])
        return {"classes": classes, "scores": outputs}

    concrete_serving_fn = serve_tf_examples_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="inputs")
    )
    signatures = {"serving_default": concrete_serving_fn}

    # This is required in order to make this model servable with model_server.
    versioned_output_dir = os.path.join(output_dir, "1")
    model.save(versioned_output_dir, save_format="tf", signatures=signatures)


def train_and_evaluate(
    working_dir: str,
    output_dir: str,
    num_train_instances: int = NUM_TRAIN_INSTANCES,
    num_test_instances: int = NUM_TEST_INSTANCES,
):
    """Train the model on training data and evaluate on test data.

    Args:
    ----
      working_dir: Directory to read transformed data and metadata from.
      output_dir: A directory where the output should be exported to.
      num_train_instances: Number of instances in train set
      num_test_instances: Number of instances in test set

    Returns:
    -------
      The results from the estimator's 'evaluate' method
    """
    tf_transform_output = tft.TFTransformOutput(working_dir)
    train_data_path_pattern = os.path.join(
        working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE + "*"
    )
    test_data_path_pattern = os.path.join(
        working_dir, TRANSFORMED_TEST_DATA_FILEBASE + "*"
    )

    train_dataset = _input_fn(
        tf_transform_output, train_data_path_pattern, batch_size=TRAIN_BATCH_SIZE
    )
    validation_dataset = _input_fn(
        tf_transform_output, test_data_path_pattern, batch_size=TRAIN_BATCH_SIZE
    )

    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(LABEL_KEY)

    review_input = tf_keras.layers.Input(
        shape=[None], name=REVIEW_KEY, dtype=tf.int64, sparse=True
    )
    review_weight_input = tf_keras.layers.Input(
        shape=[None], name=REVIEW_WEIGHT_KEY, dtype=tf.float32, sparse=True
    )
    count_layer = tf.keras.layers.CategoryEncoding(
        num_tokens=VOCAB_SIZE + 1, output_mode="count"
    )
    embedding_layer = tf.keras.layers.Dense(4, use_bias=False)
    embedding = embedding_layer(
        count_layer(review_input, count_weights=review_weight_input)
    )
    output = tf_keras.layers.Dense(100, activation="relu")(embedding)
    output = tf_keras.layers.Dense(70, activation="relu")(output)
    output = tf_keras.layers.Dense(50, activation="relu")(output)
    output = tf_keras.layers.Dense(20, activation="relu")(output)
    output = tf_keras.layers.Dense(2, activation="sigmoid")(output)
    model = tf_keras.Model(
        inputs={
            REVIEW_KEY: review_input,
            REVIEW_WEIGHT_KEY: review_weight_input,
        },
        outputs=output,
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    logging.info(model.summary())

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=TRAIN_NUM_EPOCHS,
        steps_per_epoch=math.ceil(num_train_instances / TRAIN_BATCH_SIZE),
        validation_steps=math.ceil(num_test_instances / TRAIN_BATCH_SIZE),
    )

    # Export the model.
    export_serving_model(tf_transform_output, model, output_dir)

    return model.evaluate(validation_dataset, steps=num_test_instances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_data_dir", help="path to directory containing input data"
    )
    parser.add_argument(
        "--working_dir", help="path to directory to hold transformed data"
    )
    args = parser.parse_args()

    if args.working_dir:
        working_dir = args.working_dir
    else:
        working_dir = tempfile.mkdtemp(dir=args.input_data_dir)

    train_neg_filepattern = os.path.join(args.input_data_dir, "train/neg/*")
    train_pos_filepattern = os.path.join(args.input_data_dir, "train/pos/*")
    test_neg_filepattern = os.path.join(args.input_data_dir, "test/neg/*")
    test_pos_filepattern = os.path.join(args.input_data_dir, "test/pos/*")

    read_and_shuffle_data(
        train_neg_filepattern,
        train_pos_filepattern,
        test_neg_filepattern,
        test_pos_filepattern,
        working_dir,
    )
    transform_data(working_dir)
    exported_model_dir = os.path.join(working_dir, EXPORTED_MODEL_DIR)
    results = train_and_evaluate(working_dir, exported_model_dir)

    pprint.pprint(results)


if __name__ == "__main__":
    main()
