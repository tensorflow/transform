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
import argparse
import os
import tempfile

import apache_beam as beam
import tensorflow.compat.v2 as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.coders.example_coder import RecordBatchToExamples
from tfx_bsl.public import tfxio

CATEGORICAL_FEATURE_KEYS = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]
NUMERIC_FEATURE_KEYS = [
    'age',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
]
OPTIONAL_NUMERIC_FEATURE_KEYS = [
    'education-num',
]
LABEL_KEY = 'label'


ORDERED_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label'
]


RAW_DATA_FEATURE_SPEC = dict([(name, tf.io.FixedLenFeature([], tf.string))
                              for name in CATEGORICAL_FEATURE_KEYS] +
                             [(name, tf.io.FixedLenFeature([], tf.float32))
                              for name in NUMERIC_FEATURE_KEYS] +
                             [(name, tf.io.VarLenFeature(tf.float32))
                              for name in OPTIONAL_NUMERIC_FEATURE_KEYS] +
                             [(LABEL_KEY,
                               tf.io.FixedLenFeature([], tf.string))])

_SCHEMA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC)).schema

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
NUM_OOV_BUCKETS = 1

# Names of temp files
TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
EXPORTED_MODEL_DIR = 'exported_model_dir'

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_data_dir', help='path to directory containing input data')
parser.add_argument(
    '--working_dir',
    help='optional, path to directory to hold transformed data')


def get_args():
  return parser.parse_args()


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
    # Since we are modifying some features and leaving others unchanged, we
    # start by setting `outputs` to a copy of `inputs.
    outputs = inputs.copy()

    # Scale numeric columns to have range [0, 1].
    for key in NUMERIC_FEATURE_KEYS:
      outputs[key] = tft.scale_to_0_1(inputs[key])

    for key in OPTIONAL_NUMERIC_FEATURE_KEYS:
      # This is a SparseTensor because it is optional. Here we fill in a default
      # value when it is missing.
      sparse = tf.sparse.SparseTensor(inputs[key].indices, inputs[key].values,
                                      [inputs[key].dense_shape[0], 1])
      dense = tf.sparse.to_dense(sp_input=sparse, default_value=0.)
      # Reshaping from a batch of vectors of size 1 to a batch to scalars.
      dense = tf.squeeze(dense, axis=1)
      outputs[key] = tft.scale_to_0_1(dense)

    # For all categorical columns except the label column, we generate a
    # vocabulary, and convert the string feature to a one-hot encoding.
    for key in CATEGORICAL_FEATURE_KEYS:
      integerized = tft.compute_and_apply_vocabulary(
          tf.strings.strip(inputs[key]),
          num_oov_buckets=NUM_OOV_BUCKETS,
          vocab_filename=key)
      depth = (
          tft.experimental.get_vocabulary_size_by_name(key) + NUM_OOV_BUCKETS)
      one_hot_encoded = tf.one_hot(
          integerized,
          depth=tf.cast(depth, tf.int32),
          on_value=1.0,
          off_value=0.0)
      # This output is now one-hot encoded. If saving transformed data to disk,
      # this can incur significant memory cost.
      outputs[key] = tf.reshape(one_hot_encoded, [-1, depth])

    # For the label column we provide the mapping from string to index.
    table_keys = ['>50K', '<=50K']
    with tf.init_scope():
      initializer = tf.lookup.KeyValueTensorInitializer(
          keys=table_keys,
          values=tf.cast(tf.range(len(table_keys)), tf.int64),
          key_dtype=tf.string,
          value_dtype=tf.int64)
      table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    # Remove trailing periods for test data when the data is read with tf.data.
    label_str = tf.strings.regex_replace(inputs[LABEL_KEY], r'\.', '')
    label_str = tf.strings.strip(label_str)
    data_labels = table.lookup(label_str)
    transformed_label = tf.one_hot(
        indices=data_labels, depth=len(table_keys), on_value=1.0, off_value=0.0)
    outputs[LABEL_KEY] = tf.reshape(transformed_label, [-1, len(table_keys)])

    return outputs

  # The "with" block will create a pipeline, and run that pipeline at the exit
  # of the block.
  with beam.Pipeline() as pipeline:
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      # Create a TFXIO to read the census data with the schema. To do this we
      # need to list all columns in order since the schema doesn't specify the
      # order of columns in the csv.
      # We first read CSV files and use BeamRecordCsvTFXIO whose .BeamSource()
      # accepts a PCollection[bytes] because we need to patch the records first
      # (see "FixCommasTrainData" below). Otherwise, tfxio.CsvTFXIO can be used
      # to both read the CSV files and parse them to TFT inputs:
      # csv_tfxio = tfxio.CsvTFXIO(...)
      # raw_data = (pipeline | 'ToRecordBatches' >> csv_tfxio.BeamSource())
      csv_tfxio = tfxio.BeamRecordCsvTFXIO(
          physical_format='text',
          column_names=ORDERED_CSV_COLUMNS,
          schema=_SCHEMA)

      # Read in raw data and convert using CSV TFXIO.  Note that we apply
      # some Beam transformations here, which will not be encoded in the TF
      # graph since we don't do the from within tf.Transform's methods
      # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
      # to get data into a format that the CSV TFXIO can read, in particular
      # removing spaces after commas.
      raw_data = (
          pipeline
          | 'ReadTrainData' >> beam.io.ReadFromText(
              train_data_file, coder=beam.coders.BytesCoder())
          | 'FixCommasTrainData' >> beam.Map(
              lambda line: line.replace(b', ', b','))
          | 'DecodeTrainData' >> csv_tfxio.BeamSource())

      # Combine data and schema into a dataset tuple.  Note that we already used
      # the schema to read the CSV data, but we also need it to interpret
      # raw_data.
      raw_dataset = (raw_data, csv_tfxio.TensorAdapterConfig())

      # The TFXIO output format is chosen for improved performance.
      transformed_dataset, transform_fn = (
          raw_dataset | tft_beam.AnalyzeAndTransformDataset(
              preprocessing_fn, output_record_batches=True))

      # Transformed metadata is not necessary for encoding.
      transformed_data, _ = transformed_dataset

      # Extract transformed RecordBatches, encode and write them to the given
      # directory.
      _ = (
          transformed_data
          | 'EncodeTrainData' >>
          beam.FlatMapTuple(lambda batch, _: RecordBatchToExamples(batch))
          | 'WriteTrainData' >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

      # Now apply transform function to test data.  In this case we remove the
      # trailing period at the end of each line, and also ignore the header line
      # that is present in the test data file.
      raw_test_data = (
          pipeline
          | 'ReadTestData' >> beam.io.ReadFromText(
              test_data_file, skip_header_lines=1,
              coder=beam.coders.BytesCoder())
          | 'FixCommasTestData' >> beam.Map(
              lambda line: line.replace(b', ', b','))
          | 'RemoveTrailingPeriodsTestData' >> beam.Map(lambda line: line[:-1])
          | 'DecodeTestData' >> csv_tfxio.BeamSource())

      raw_test_dataset = (raw_test_data, csv_tfxio.TensorAdapterConfig())

      # The TFXIO output format is chosen for improved performance.
      transformed_test_dataset = (
          (raw_test_dataset, transform_fn)
          | tft_beam.TransformDataset(output_record_batches=True))

      # Transformed metadata is not necessary for encoding.
      transformed_test_data, _ = transformed_test_dataset

      # Extract transformed RecordBatches, encode and write them to the given
      # directory.
      _ = (
          transformed_test_data
          | 'EncodeTestData' >>
          beam.FlatMapTuple(lambda batch, _: RecordBatchToExamples(batch))
          | 'WriteTestData' >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)))

      # Will write a SavedModel and metadata to working_dir, which can then
      # be read by the tft.TFTransformOutput class.
      _ = (
          transform_fn
          | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))
