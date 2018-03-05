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
"""Example using census data from UCI repository."""

# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import tempfile


import tensorflow as tf
import tensorflow_transform as tft
from apache_beam.io import textio
from apache_beam.io import tfrecordio
from tensorflow.contrib import learn
from tensorflow.contrib import lookup
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io

import apache_beam as beam

CATEGORICAL_FEATURE_KEYS = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
]
NUMERIC_FEATURE_KEYS = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week'
]
LABEL_KEY = 'label'


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
      tf.string, [], dataset_schema.FixedColumnRepresentation())
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
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_EPOCHS = 200
NUM_TRAIN_INSTANCES = 32561
NUM_TEST_INSTANCES = 16281
BUCKET_SIZES = [9, 17, 8, 15, 17, 6, 3, 43]

# Names of temp files
TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
EXPORTED_MODEL_DIR = 'exported_model_dir'

# Functions for preprocessing


def transform_data(train_data_file, test_data_file, working_dir):
  """Transform the data and write out as a TFRecord of Example protos.

  Read in the data using the CSV reader, and transform it using a
  preprocessing pipeline that scales numeric data and converts categorical data
  from strings to int64 values indices, by creating a vocabulary for each
  category.

  Args:
    train_data_file: File containing training data
    test_data_file: File containing test data
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
    def convert_label(label):
      table = lookup.index_table_from_tensor(['>50K', '<=50K'])
      return table.lookup(label)
    outputs[LABEL_KEY] = tft.apply_function(convert_label, inputs[LABEL_KEY])

    return outputs

  # The "with" block will create a pipeline, and run that pipeline at the exit
  # of the block.
  with beam.Pipeline() as pipeline:
    with tft.Context(temp_dir=tempfile.mkdtemp()):
      # Create a coder to read the census data with the schema.  To do this we
      # need to list all columns in order since the schema doesn't specify the
      # order of columns in the csv.
      ordered_columns = [
          'age', 'workclass', 'fnlwgt', 'education', 'education-num',
          'marital-status', 'occupation', 'relationship', 'race', 'sex',
          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
          'label'
      ]
      converter = tft.CsvCoder(ordered_columns, RAW_DATA_METADATA.schema)

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
          raw_dataset | tft.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data, transformed_metadata = transformed_dataset

      _ = transformed_data | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(
          os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE),
          coder=tft.ExampleProtoCoder(transformed_metadata.schema))

      # Now apply transform function to test data.  In this case we also remove
      # the header line from the CSV file and the trailing period at the end of
      # each line.
      raw_test_data = (
          pipeline
          | 'ReadTestData' >> textio.ReadFromText(test_data_file)
          | 'FilterTestData' >> beam.Filter(
              lambda line: line and line != '|1x3 Cross validator')
          | 'FixCommasTestData' >> beam.Map(
              lambda line: line.replace(', ', ','))
          | 'RemoveTrailingPeriodsTestData' >> beam.Map(lambda line: line[:-1])
          | 'DecodeTestData' >> beam.Map(converter.decode))

      raw_test_dataset = (raw_test_data, RAW_DATA_METADATA)

      transformed_test_dataset = (
          (raw_test_dataset, transform_fn) | tft.TransformDataset())
      # Don't need transformed data schema, it's the same as before.
      transformed_test_data, _ = transformed_test_dataset

      _ = transformed_test_data | 'WriteTestData' >> tfrecordio.WriteToTFRecord(
          os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE),
          coder=tft.ExampleProtoCoder(transformed_metadata.schema))

      # Will write a SavedModel and metadata to two subdirectories of
      # working_dir, given by tft.TRANSFORM_FN_DIR and
      # tft.TRANSFORMED_METADATA_DIR respectively.
      _ = (
          transform_fn
          | 'WriteTransformFn' >>
          tft.WriteTransformFn(working_dir))

# Functions for training


def _make_training_input_fn(working_dir, filebase, batch_size):
  """Creates an input function reading from transformed data.

  Args:
    working_dir: Directory to read transformed data and metadata from and to
        write exported model to.
    filebase: Base filename (relative to `working_dir`) of examples.
    batch_size: Batch size.

  Returns:
    The input function for training or eval.
  """
  transformed_metadata = metadata_io.read_metadata(
      os.path.join(working_dir, tft.TRANSFORMED_METADATA_DIR))
  transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

  def input_fn():
    """Input function for training and eval."""
    transformed_features = tf.contrib.learn.io.read_batch_features(
        os.path.join(working_dir, filebase + '*'),
        batch_size, transformed_feature_spec, tf.TFRecordReader)

    # Extract features and label from the transformed tensors.
    transformed_labels = transformed_features.pop(LABEL_KEY)

    return transformed_features, transformed_labels

  return input_fn


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
        tft.partially_apply_saved_transform(
            os.path.join(working_dir, tft.TRANSFORM_FN_DIR), raw_features))

    return input_fn_utils.InputFnOps(transformed_features, None, default_inputs)

  return serving_input_fn


def train_and_evaluate(working_dir, num_train_instances=NUM_TRAIN_INSTANCES,
                       num_test_instances=NUM_TEST_INSTANCES):
  """Train the model on training data and evaluate on test data.

  Args:
    working_dir: Directory to read transformed data and metadata from and to
        write exported model to.
    num_train_instances: Number of instances in train set
    num_test_instances: Number of instances in test set

  Returns:
    The results from the estimator's 'evaluate' method
  """

  # Wrap scalars as real valued columns.
  real_valued_columns = [tf.feature_column.numeric_column(key, shape=())
                         for key in NUMERIC_FEATURE_KEYS]

  # Wrap categorical columns.  Note the combiner is irrelevant since the input
  # only has one value set per feature per instance.
  one_hot_columns = [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=num_buckets)
      for key, num_buckets in zip(CATEGORICAL_FEATURE_KEYS, BUCKET_SIZES)]

  estimator = learn.LinearClassifier(real_valued_columns + one_hot_columns)

  # Fit the model using the default optimizer.
  train_input_fn = _make_training_input_fn(
      working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE,
      batch_size=TRAIN_BATCH_SIZE)
  estimator.fit(
      input_fn=train_input_fn,
      max_steps=TRAIN_NUM_EPOCHS * num_train_instances / TRAIN_BATCH_SIZE)

  # Evaluate model on test dataset.
  eval_input_fn = _make_training_input_fn(
      working_dir, TRANSFORMED_TEST_DATA_FILEBASE,
      batch_size=1)

  # Export the model.
  serving_input_fn = _make_serving_input_fn(working_dir)
  exported_model_dir = os.path.join(working_dir, EXPORTED_MODEL_DIR)
  estimator.export_savedmodel(exported_model_dir, serving_input_fn)

  return estimator.evaluate(input_fn=eval_input_fn, steps=num_test_instances)


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

  train_data_file = os.path.join(args.input_data_dir, 'adult.data')
  test_data_file = os.path.join(args.input_data_dir, 'adult.test')

  transform_data(train_data_file, test_data_file, working_dir)

  results = train_and_evaluate(working_dir)

  pprint.pprint(results)

if __name__ == '__main__':
  main()
