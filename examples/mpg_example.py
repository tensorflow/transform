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
"""Example using auto-mpg data from UCI repository."""

# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tempfile

import tensorflow as tf
import tensorflow_transform as tft
from apache_beam.io import textio
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import csv_coder
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

import apache_beam as beam

# to download and prepare the data:
#  curl https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data|grep -v "?"|sed -E -e 's/[[:blank:]]{2,}/,/g'|sed  -E -e $'s/\t/,/g' | head -n340 > auto-mpg.csv
#  curl https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data|grep -v "?"|sed -E -e 's/[[:blank:]]{2,}/,/g'|sed  -E -e $'s/\t/,/g' | tail -n50 > auto-mpg-test.csv

ordered_columns = [
    'mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name'
]

CATEGORICAL_FEATURE_KEYS = [
    'cylinders', 'year', 'name', 'origin'
]

NUMERIC_FEATURE_KEYS = [
    'displacement', 'horsepower', 'weight', 'acceleration'
]

LABEL_KEY = 'mpg'


def _create_raw_metadata():
    """Create a DatasetMetadata for the raw data."""
    column_schemas = {
        key: dataset_schema.ColumnSchema(
            tf.string, [], dataset_schema.FixedColumnRepresentation())
        for key in CATEGORICAL_FEATURE_KEYS
    }
    column_schemas.update({
        key: dataset_schema.ColumnSchema(
            tf.float32, [], dataset_schema.FixedColumnRepresentation())
        for key in NUMERIC_FEATURE_KEYS
    })
    column_schemas[LABEL_KEY] = dataset_schema.ColumnSchema(
        tf.float32, [], dataset_schema.FixedColumnRepresentation())
    raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema(
        column_schemas))
    return raw_data_metadata


RAW_DATA_METADATA = _create_raw_metadata()

# Constants used for training.  Note that the number of instances will be
# computed by tf.Transform in future versions, in which case it can be read from
# the metadata.  Similarly BUCKET_SIZES will not be needed as this information
# will be stored in the metadata for each of the columns.  The bucket size
# includes all listed categories in the dataset description as well as one extra
# for "?" which represents unknown.
BATCH_SIZE = 5
TRAIN_NUM_EPOCHS = 20
NUM_TRAIN_INSTANCES = 340
NUM_TEST_INSTANCES = 50
BUCKET_SIZES = [5, 12, 1024, 3]

EXPORTED_MODEL_DIR = 'exported_model_dir'


def create_transform_fn(train_data_file, working_dir):
    """Create a transform function that can be run on-the-fly while training

    Read in the data using the CSV reader, and transform it using a
    preprocessing pipeline that scales numeric data and converts categorical data
    from strings to int64 values indices, by creating a vocabulary for each
    category.

    Args:
      train_data_file: File containing training data
      working_dir: Directory to write transformed data and metadata to
    """

    def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        outputs = {}

        # Scale numeric columns to have range [0, 1].
        for key in NUMERIC_FEATURE_KEYS:
            outputs[key] = tft.scale_to_0_1(inputs[key])

        # For all categorical columns except the label column, we use
        # tft.string_to_int which computes the set of unique values and uses this
        # to convert the strings to indices.
        for key in CATEGORICAL_FEATURE_KEYS:
            outputs[key] = tft.string_to_int(inputs[key])

        # For the label column we provide the mapping from string to index.
        outputs[LABEL_KEY] = inputs[LABEL_KEY]

        return outputs

    # The "with" block will create a pipeline, and run that pipeline at the exit
    # of the block.
    with beam.Pipeline() as pipeline:
        with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
            # Create a coder to read the mpg data with the schema.  To do this we
            # need to list all columns in order since the schema doesn't specify the
            # order of columns in the csv.
            converter = csv_coder.CsvCoder(ordered_columns, RAW_DATA_METADATA.schema)

            # Read in raw data and convert using CSV converter.  Note that we apply
            # some Beam transformations here, which will not be encoded in the TF
            # graph since we don't do the from within tf.Transform's methods
            # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
            # to get data into a format that the CSV converter can read, in particular
            # removing empty lines and removing spaces after commas.
            raw_data = (
                    pipeline
                    | 'ReadTrainData' >> textio.ReadFromText(train_data_file)
                    | 'FilterTrainData' >> beam.Filter(lambda line: line)
                    | 'FixCommasTrainData' >> beam.Map(
                lambda line: line.replace(', ', ','))
                    | 'DecodeTrainData' >> beam.Map(converter.decode))

            # Combine data and schema into a dataset tuple.  Note that we already used
            # the schema to read the CSV data, but we also need it to interpret
            # raw_data.
            raw_dataset = (raw_data, RAW_DATA_METADATA)
            transformed_dataset, transform_fn = (
                    raw_dataset | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))
            transformed_data, transformed_metadata = transformed_dataset

            # Will write a SavedModel and metadata to two subdirectories of
            # working_dir, given by transform_fn_io.TRANSFORM_FN_DIR and
            # transform_fn_io.TRANSFORMED_METADATA_DIR respectively.
            _ = (
                    transform_fn
                    | 'WriteTransformFn' >>
                    transform_fn_io.WriteTransformFn(working_dir))


def file_decode_csv(line):
    columns_default_values = [[0.0], ["4"], [0.0], [0.0], [0.0], [0.0], ["70"], ["1"], ["<unkown>"]]

    parsed_line = tf.decode_csv(line, columns_default_values)
    features = parsed_line

    d = dict(zip(ordered_columns, features))

    label = d[LABEL_KEY]
    del d[LABEL_KEY]

    return d, label


def _make_training_input_fn(working_dir, csv_file, batch_size):
    dataset = (tf.data.TextLineDataset(csv_file, buffer_size=8 * 1048576))

    dataset = dataset.shuffle(NUM_TRAIN_INSTANCES)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(file_decode_csv, batch_size, num_parallel_batches=4))
    dataset = dataset.prefetch(4)

    raw_features, raw_label = dataset.make_one_shot_iterator().get_next()

    _, transformed_features = saved_transform_io.partially_apply_saved_transform(
        os.path.join(working_dir, transform_fn_io.TRANSFORM_FN_DIR), raw_features)
    return transformed_features, raw_label


def _make_serving_input_fn(working_dir):
    """Creates an input function reading from raw data.

    Args:
      working_dir: Directory to read transformed metadata from.

    Returns:
      The serving input function.
    """
    raw_feature_spec = RAW_DATA_METADATA.schema.as_feature_spec()
    # Remove label since it is not available during serving.
    raw_feature_spec.pop(LABEL_KEY)

    def serving_input_fn():
        """Input function for serving."""
        # Get raw features by generating the basic serving input_fn and calling it.
        # Here we generate an input_fn that expects a parsed Example proto to be fed
        # to the model at serving time.  See also
        # input_fn_utils.build_default_serving_input_fn.
        raw_input_fn = input_fn_utils.build_parsing_serving_input_fn(
            raw_feature_spec)
        raw_features, _, default_inputs = raw_input_fn()

        # Apply the transform function that was used to generate the materialized
        # data.
        _, transformed_features = (
            saved_transform_io.partially_apply_saved_transform(
                os.path.join(working_dir, transform_fn_io.TRANSFORM_FN_DIR),
                raw_features))

        return tf.estimator.export.ServingInputReceiver(transformed_features, default_inputs)

    return serving_input_fn


def train_and_evaluate(working_dir, num_train_instances=NUM_TRAIN_INSTANCES,
                       num_test_instances=NUM_TEST_INSTANCES):
    """Train the model on training data and evaluate on eval data.

    Args:
      working_dir: Directory to read transformed data and metadata from and to
          write exported model to.
      num_train_instances: Number of instances in train set
      num_test_instances: Number of instances in test set

    Returns:
    """

    one_hot_columns = [
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(key=key, num_buckets=num_buckets))
        for key, num_buckets in zip(CATEGORICAL_FEATURE_KEYS, BUCKET_SIZES)]

    real_valued_columns = [tf.feature_column.numeric_column(key, shape=())
                           for key in NUMERIC_FEATURE_KEYS]

    estimator = tf.estimator.DNNRegressor(
        feature_columns=real_valued_columns + one_hot_columns,
        model_dir=os.path.join(working_dir, "logs_directory"),
        optimizer=tf.train.AdamOptimizer(),
        hidden_units=[10, 5])

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: _make_training_input_fn(working_dir, "auto-mpg.csv", BATCH_SIZE),
        max_steps=TRAIN_NUM_EPOCHS * num_train_instances / BATCH_SIZE)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: _make_training_input_fn(working_dir, "auto-mpg-test.csv", BATCH_SIZE),
        throttle_secs=10, steps=num_test_instances / BATCH_SIZE)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Export the model.
    serving_input_fn = _make_serving_input_fn(working_dir)
    exported_model_dir = os.path.join(working_dir, EXPORTED_MODEL_DIR)
    estimator.export_savedmodel(exported_model_dir, serving_input_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_data_dir',
        help='path to directory containing input data')
    parser.add_argument(
        '--working_dir',
        help='optional, path to directory to hold transformed data')
    args = parser.parse_args()

    if args.working_dir:
        working_dir = args.working_dir
    else:
        working_dir = tempfile.mkdtemp(dir=args.input_data_dir)

    train_data_file = os.path.join(args.input_data_dir, 'auto-mpg.csv')

    # Will write a SavedModel and metadata to two subdirectories of
    # working_dir, given by transform_fn_io.TRANSFORM_FN_DIR and
    # transform_fn_io.TRANSFORMED_METADATA_DIR respectively.
    create_transform_fn(train_data_file, working_dir)

    # will transform features on the fly using the transform_fn created above
    train_and_evaluate(working_dir)

if __name__ == '__main__':
    main()
