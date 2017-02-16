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

import os
import pprint
import sys
import tempfile

import tensorflow as tf
import tensorflow_transform as tft
from apache_beam.io import textio
from apache_beam.io import tfrecordio
from tensorflow.contrib import learn
from tensorflow.contrib import lookup
from tensorflow.contrib.layers import feature_column
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.io import beam_metadata_io
from tensorflow_transform.coders import csv_coder
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io

import apache_beam as beam

CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
]
NUMERIC_COLUMNS = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week'
]
LABEL_COLUMN = 'label'

# Constants used for training.  Note that the number of instances will be
# computed by tf.Transform in future versions, in which case it can be read from
# the metadata.  Similarly BUCKET_SIZES will not be needed as this information
# will be stored in the metadata for each of the columns.  The bucket size
# includes all listed categories in the dataset description as well as one extra
# for "?" which represents unknown.
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_EPOCHS = 200
NUM_TRAIN_INSTANCES = 32561
NUM_EVAL_INSTANCES = 16281
BUCKET_SIZES = [9, 17, 8, 15, 17, 6, 3, 43]


def transform_data(train_data_file, eval_data_file,
                   transformed_train_data_base, transformed_eval_data_base,
                   transformed_metadata_dir):
  """Transform the cleaned data and write out as a TFRecord of Example protos.

  Read in the cleaned data using the CSV reader, and transform it using a
  preprocessing pipeline that scales numeric data and coverts categorical data
  from strings to int64 values indices, by creating a vocabulary for each
  category.

  Args:
    train_data_file: File containing training data
    eval_data_file: File containing evaluation data
    transformed_train_data_base: Base filename for transformed training data
        shards
    transformed_eval_data_base: Base filename for cleaned evaluation data
        shards
    transformed_metadata_dir: Directory where metadata for transformed data
        should be written.
  """
  raw_data_schema = {
      key: dataset_schema.ColumnSchema(
          dataset_schema.LogicalColumnSchema(
              dataset_schema.Domain(tf.string),
              dataset_schema.LogicalShape([])),
          dataset_schema.FixedColumnRepresentation())
      for key in CATEGORICAL_COLUMNS
  }
  raw_data_schema.update({
      key: dataset_schema.ColumnSchema(
          dataset_schema.LogicalColumnSchema(
              dataset_schema.Domain(tf.float32),
              dataset_schema.LogicalShape([])),
          dataset_schema.FixedColumnRepresentation())
      for key in NUMERIC_COLUMNS
  })
  raw_data_schema[LABEL_COLUMN] = dataset_schema.ColumnSchema(
      dataset_schema.LogicalColumnSchema(
          dataset_schema.Domain(tf.string),
          dataset_schema.LogicalShape([])),
      dataset_schema.FixedColumnRepresentation())
  raw_data_schema = dataset_schema.Schema(raw_data_schema)
  raw_data_metadata = dataset_metadata.DatasetMetadata(raw_data_schema)

  def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}

    # Scale numeric columns to have range [0, 1].
    for key in NUMERIC_COLUMNS:
      outputs[key] = tft.scale_to_0_1(inputs[key])

    # For all categorical columns except the label column, we use
    # tft.string_to_int which computes the set of unique values and uses this
    # to convert the strings to indices.
    for key in CATEGORICAL_COLUMNS:
      outputs[key] = tft.string_to_int(inputs[key])

    # Update outputs of both kinds to convert from shape (batch,), i.e. a batch
    # of scalars, to shape (batch, 1), i.e. a batch of vectors of length 1.
    # This is needed so the output can be easily wrapped in `FeatureColumn`s.
    for key in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
      outputs[key] = tft.map(lambda x: tf.expand_dims(x, -1), outputs[key])

    # For the label column we provide the mapping from string to index.
    def convert_label(label):
      table = lookup.string_to_index_table_from_tensor(['>50K', '<=50K'])
      return table.lookup(label)
    outputs[LABEL_COLUMN] = tft.map(convert_label, inputs[LABEL_COLUMN])

    return outputs

  # The "with" block will create a pipeline, and run that pipeline at the exit
  # of the block.
  with beam.Pipeline() as p:
    # Create a coder to read the census data with the schema.  To do this we
    # need to list all columns in order since the schema doesn't specify the
    # order of columns in the csv.
    ordered_columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'label'
    ]
    converter = csv_coder.CsvCoder(ordered_columns, raw_data_schema)

    # Read in raw data and convert using CSV converter.  Note that we apply some
    # Beam transformations here, which will not be encoded in the TF graph since
    # we don't do the from within tf.Transform's methods (AnalyzeDataset,
    # TransformDataset etc.).  These transformations are just to get data into
    # a format that the CSV converter can read, in particular removing empty
    # lines and removing spaces after commas.
    raw_data = (
        p
        | 'ReadTrainData' >> textio.ReadFromText(train_data_file)
        | 'FilterTrainData' >> beam.Filter(lambda line: line)
        | 'FixCommasTrainData' >> beam.Map(lambda line: line.replace(', ', ','))
        | 'DecodeTrainData' >> beam.Map(converter.decode))

    # Combine data and schema into a dataset tuple.  Note that we already used
    # the schema to read the CSV data, but we also need it to interpret
    # raw_data.
    raw_dataset = (raw_data, raw_data_metadata)
    transformed_dataset, transform_fn = (
        raw_dataset | beam_impl.AnalyzeAndTransformDataset(
            preprocessing_fn, output_dir=os.path.join(tempfile.mkdtemp())))
    transformed_data, transformed_metadata = transformed_dataset

    _ = transformed_data | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(
        transformed_train_data_base,
        coder=example_proto_coder.ExampleProtoCoder(
            transformed_metadata.schema))

    # Now apply transform function to eval data.  In this case we also remove
    # the header line from the CSV file and the trailing period at the end of
    # each line.
    raw_eval_data = (
        p
        | 'ReadEvalData' >> textio.ReadFromText(eval_data_file)
        | 'FilterEvalData' >> beam.Filter(
            lambda line: line and line != '|1x3 Cross validator')
        | 'FixCommasEvalData' >> beam.Map(lambda line: line.replace(', ', ','))
        | 'RemoveTrailingPeriodsEvalData' >> beam.Map(lambda line: line[:-1])
        | 'DecodeEvalData' >> beam.Map(converter.decode))

    raw_eval_dataset = (raw_eval_data, raw_data_metadata)

    transformed_eval_dataset = (
        (raw_eval_dataset, transform_fn) | beam_impl.TransformDataset())
    # Don't need transformed data schema, it's the same as before.
    transformed_eval_data, _ = transformed_eval_dataset

    _ = transformed_eval_data | 'WriteEvalData' >> tfrecordio.WriteToTFRecord(
        transformed_eval_data_base,
        coder=example_proto_coder.ExampleProtoCoder(
            transformed_metadata.schema))

    _ = (
        transformed_metadata
        | 'WriteMetadata' >> beam_metadata_io.WriteMetadata(
            transformed_metadata_dir, pipeline=p))


def train_and_evaluate(transformed_train_data_base, transformed_eval_data_base,
                       transformed_metadata_dir):
  """Train the model on training data and evaluate on evaluation data.

  Args:
    transformed_train_data_base: Base filename for transformed training data
        shards
    transformed_eval_data_base: Base filename for cleaned evaluation data
        shards
    transformed_metadata_dir: Directory containing transformed data metadata.

  Returns:
    The results from the estimator's 'evaluate' method.
  """

  # Wrap scalars as real valued columns.
  real_valued_columns = [feature_column.real_valued_column(key)
                         for key in NUMERIC_COLUMNS]

  # Wrap categorical columns.
  one_hot_columns = [
      feature_column.sparse_column_with_integerized_feature(
          key, bucket_size=bucket_size)
      for key, bucket_size in zip(CATEGORICAL_COLUMNS, BUCKET_SIZES)]

  estimator = learn.LinearClassifier(real_valued_columns + one_hot_columns)

  transformed_metadata = metadata_io.read_metadata(transformed_metadata_dir)
  train_input_fn = input_fn_maker.build_training_input_fn(
      transformed_metadata,
      transformed_train_data_base + '*',
      training_batch_size=TRAIN_BATCH_SIZE,
      label_keys=['label'])

  # Estimate the model using the default optimizer.
  estimator.fit(
      input_fn=train_input_fn,
      max_steps=TRAIN_NUM_EPOCHS * NUM_TRAIN_INSTANCES / TRAIN_BATCH_SIZE)

  # Evaluate model on eval dataset.
  eval_input_fn = input_fn_maker.build_training_input_fn(
      transformed_metadata,
      transformed_eval_data_base + '*',
      training_batch_size=1,
      label_keys=['label'])

  return estimator.evaluate(input_fn=eval_input_fn, steps=NUM_EVAL_INSTANCES)


def main(argv):
  census_data_dir = argv[1]
  temp_dir = tempfile.mkdtemp(dir=census_data_dir)

  train_data_file = os.path.join(census_data_dir, 'adult.data')
  eval_data_file = os.path.join(census_data_dir, 'adult.test')
  transformed_train_data_base = os.path.join(temp_dir, 'adult.data.transformed')
  transformed_eval_data_base = os.path.join(temp_dir, 'adult.test.transformed')
  transformed_metadata_dir = os.path.join(temp_dir, 'metadata')

  transform_data(train_data_file, eval_data_file, transformed_train_data_base,
                 transformed_eval_data_base, transformed_metadata_dir)

  results = train_and_evaluate(transformed_train_data_base,
                               transformed_eval_data_base,
                               transformed_metadata_dir)

  pprint.pprint(results)

if __name__ == '__main__':
  main(sys.argv)
