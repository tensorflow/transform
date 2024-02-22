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
import math
import os
import pprint
import tempfile

from absl import logging
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.keras_lib import tf_keras
import argparse

import apache_beam as beam
import tensorflow.compat.v2 as tf
import tensorflow_transform.beam as tft_beam
from tfx_bsl.public import tfxio

# Functions for training

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
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'label',
]


RAW_DATA_FEATURE_SPEC = dict(
    [
        (name, tf.io.FixedLenFeature([], tf.string))
        for name in CATEGORICAL_FEATURE_KEYS
    ]
    + [
        (name, tf.io.FixedLenFeature([], tf.float32))
        for name in NUMERIC_FEATURE_KEYS
    ]
    + [
        (
            name,  # pylint: disable=g-complex-comprehension
            tf.io.RaggedFeature(
                tf.float32,
                value_key=name,
                partitions=[],
                row_splits_dtype=tf.int64,
            ),
        )
        for name in OPTIONAL_NUMERIC_FEATURE_KEYS
    ]
    + [(LABEL_KEY, tf.io.FixedLenFeature([], tf.string))]
)

_SCHEMA = tft.DatasetMetadata.from_feature_spec(RAW_DATA_FEATURE_SPEC).schema

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
    '--input_data_dir', help='path to directory containing input data'
)
parser.add_argument(
    '--working_dir', help='optional, path to directory to hold transformed data'
)


def get_args():
  return parser.parse_args()


# Functions for preprocessing


def transform_data(train_data_file: str, test_data_file: str, working_dir: str):
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
      # This is a RaggedTensor because it is optional. Here we fill in a default
      # value when it is missing, after scaling it.
      outputs[key] = tft.scale_to_0_1(inputs[key]).to_tensor(
          default_value=0.0, shape=[None, 1]
      )

    # For all categorical columns except the label column, we generate a
    # vocabulary, and convert the string feature to a one-hot encoding.
    for key in CATEGORICAL_FEATURE_KEYS:
      integerized = tft.compute_and_apply_vocabulary(
          tf.strings.strip(inputs[key]),
          num_oov_buckets=NUM_OOV_BUCKETS,
          vocab_filename=key,
      )
      depth = (
          tft.experimental.get_vocabulary_size_by_name(key) + NUM_OOV_BUCKETS
      )
      one_hot_encoded = tf.one_hot(
          integerized,
          depth=tf.cast(depth, tf.int32),
          on_value=1,
          off_value=0,
          dtype=tf.int64,
      )
      # Saving one-hot encoded outputs as sparse in order to avoid large dense
      # (mostly empty) tensors. This is especially important when saving
      # transformed data to disk.
      outputs[key] = tf.sparse.from_dense(
          tf.reshape(one_hot_encoded, [-1, depth])
      )
      tft.experimental.annotate_sparse_output_shape(outputs[key], depth)

    # For the label column we provide the mapping from string to index.
    table_keys = ['>50K', '<=50K']
    with tf.init_scope():
      initializer = tf.lookup.KeyValueTensorInitializer(
          keys=table_keys,
          values=tf.cast(tf.range(len(table_keys)), tf.int64),
          key_dtype=tf.string,
          value_dtype=tf.int64,
      )
      table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    # Remove trailing periods for test data when the data is read with tf.data.
    label_str = tf.strings.regex_replace(inputs[LABEL_KEY], r'\.', '')
    label_str = tf.strings.strip(label_str)
    data_labels = table.lookup(label_str)
    transformed_label = tf.one_hot(
        indices=data_labels, depth=len(table_keys), on_value=1.0, off_value=0.0
    )
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
          schema=_SCHEMA,
      )

      # Read in raw data and convert using CSV TFXIO.  Note that we apply
      # some Beam transformations here, which will not be encoded in the TF
      # graph since we don't do the from within tf.Transform's methods
      # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
      # to get data into a format that the CSV TFXIO can read, in particular
      # removing spaces after commas.
      raw_data = (
          pipeline
          | 'ReadTrainData'
          >> beam.io.ReadFromText(
              train_data_file, coder=beam.coders.BytesCoder()
          )
          | 'FixCommasTrainData'
          >> beam.Map(lambda line: line.replace(b', ', b','))
          | 'DecodeTrainData' >> csv_tfxio.BeamSource()
      )

      # Combine data and schema into a dataset tuple.  Note that we already used
      # the schema to read the CSV data, but we also need it to interpret
      # raw_data.
      raw_dataset = (raw_data, csv_tfxio.TensorAdapterConfig())

      # The TFXIO output format is chosen for improved performance.
      transformed_dataset, transform_fn = (
          raw_dataset
          | tft_beam.AnalyzeAndTransformDataset(
              preprocessing_fn, output_record_batches=True
          )
      )

      # Extract transformed RecordBatches, encode and write them to the given
      # directory.
      _ = (
          transformed_dataset
          | 'EncodeTrainData' >> tft_beam.EncodeTransformedDataset()
          | 'WriteTrainData'
          >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)
          )
      )

      # Now apply transform function to test data.  In this case we remove the
      # trailing period at the end of each line, and also ignore the header line
      # that is present in the test data file.
      raw_test_data = (
          pipeline
          | 'ReadTestData'
          >> beam.io.ReadFromText(
              test_data_file,
              skip_header_lines=1,
              coder=beam.coders.BytesCoder(),
          )
          | 'FixCommasTestData'
          >> beam.Map(lambda line: line.replace(b', ', b','))
          | 'RemoveTrailingPeriodsTestData' >> beam.Map(lambda line: line[:-1])
          | 'DecodeTestData' >> csv_tfxio.BeamSource()
      )

      raw_test_dataset = (raw_test_data, csv_tfxio.TensorAdapterConfig())

      # The TFXIO output format is chosen for improved performance.
      transformed_test_dataset = (
          raw_test_dataset,
          transform_fn,
      ) | tft_beam.TransformDataset(output_record_batches=True)

      # Extract transformed RecordBatches, encode and write them to the given
      # directory.
      _ = (
          transformed_test_dataset
          | 'EncodeTestData' >> tft_beam.EncodeTransformedDataset()
          | 'WriteTestData'
          >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)
          )
      )

      # Will write a SavedModel and metadata to working_dir, which can then
      # be read by the tft.TFTransformOutput class.
      _ = transform_fn | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
          working_dir
      )


def input_fn(
    tf_transform_output: tft.TFTransformOutput,
    transformed_examples_pattern: str,
    batch_size: int,
):
  """An input function reading from transformed data, converting to model input.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    transformed_examples_pattern: Base filename of examples.
    batch_size: Batch size.

  Returns:
    The input data for training or eval, in the form of k.
  """
  return tf.data.experimental.make_batched_features_dataset(
      file_pattern=transformed_examples_pattern,
      batch_size=batch_size,
      features=tf_transform_output.transformed_feature_spec(),
      reader=tf.data.TFRecordDataset,
      label_key=LABEL_KEY,
      shuffle=True,
  ).prefetch(tf.data.experimental.AUTOTUNE)


def input_fn_raw(
    tf_transform_output: tft.TFTransformOutput,
    raw_examples_pattern: str,
    batch_size: int,
):
  """An input function reading from raw data, converting to model input.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    raw_examples_pattern: Base filename of examples.
    batch_size: Batch size.

  Returns:
    The input data for training or eval, in the form of k.
  """

  def get_ordered_raw_data_dtypes():
    result = []
    for col in ORDERED_CSV_COLUMNS:
      if col not in RAW_DATA_FEATURE_SPEC:
        result.append(0.0)
        continue
      spec = RAW_DATA_FEATURE_SPEC[col]
      if isinstance(spec, tf.io.FixedLenFeature):
        result.append(spec.dtype)
      else:
        result.append(0.0)
    return result

  dataset = tf.data.experimental.make_csv_dataset(
      file_pattern=raw_examples_pattern,
      batch_size=batch_size,
      column_names=ORDERED_CSV_COLUMNS,
      column_defaults=get_ordered_raw_data_dtypes(),
      prefetch_buffer_size=0,
      ignore_errors=True,
  )

  tft_layer = tf_transform_output.transform_features_layer()

  def transform_dataset(data):
    raw_features = {}
    for key, val in data.items():
      if key not in RAW_DATA_FEATURE_SPEC:
        continue
      if isinstance(RAW_DATA_FEATURE_SPEC[key], tf.io.RaggedFeature):
        # make_csv_dataset will set the value to 0 when it's missing.
        raw_features[key] = tf.RaggedTensor.from_tensor(
            tf.expand_dims(val, axis=-1), padding=0)
        continue
      raw_features[key] = val
    transformed_features = tft_layer(raw_features)
    data_labels = transformed_features.pop(LABEL_KEY)
    return (transformed_features, data_labels)

  return dataset.map(
      transform_dataset,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
          tf.data.experimental.AUTOTUNE)


def export_serving_model(
    tf_transform_output: tft.TFTransformOutput,
    model: tf_keras.Model,
    output_dir: str,
):
  """Exports a keras model for serving.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    model: A keras model to export for serving.
    output_dir: A directory where the model will be exported to.
  """
  # The layer has to be saved to the model for keras tracking purpases.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Serving tf.function model wrapper."""
    feature_spec = RAW_DATA_FEATURE_SPEC.copy()
    feature_spec.pop(LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    outputs = model(transformed_features)
    classes_names = tf.constant([['0', '1']])
    classes = tf.tile(classes_names, [tf.shape(outputs)[0], 1])
    return {'classes': classes, 'scores': outputs}

  concrete_serving_fn = serve_tf_examples_fn.get_concrete_function(
      tf.TensorSpec(shape=[None], dtype=tf.string, name='inputs'))
  signatures = {'serving_default': concrete_serving_fn}

  # This is required in order to make this model servable with model_server.
  versioned_output_dir = os.path.join(output_dir, '1')
  model.save(versioned_output_dir, save_format='tf', signatures=signatures)


def train_and_evaluate(
    raw_train_eval_data_path_pattern: tuple[str, str],
    transformed_train_eval_data_path_pattern: tuple[str, str],
    output_dir: str,
    transform_output_dir: str,
    num_train_instances: int = NUM_TRAIN_INSTANCES,
    num_test_instances: int = NUM_TEST_INSTANCES,
):
  """Train the model on training data and evaluate on test data.

  Args:
    raw_train_eval_data_path_pattern: A pair of patterns of raw
      (train data file paths, eval data file paths) in CSV format.
    transformed_train_eval_data_path_pattern: A pair of patterns of transformed
      (train data file paths, eval data file paths) in TFRecord format.
    output_dir: A directory where the output should be exported to.
    transform_output_dir: The location of the Transform output.
    num_train_instances: Number of instances in train set
    num_test_instances: Number of instances in test set

  Returns:
    The results from the estimator's 'evaluate' method
  """
  if not ((raw_train_eval_data_path_pattern is None) ^
          (transformed_train_eval_data_path_pattern is None)):
    raise ValueError(
        'Exactly one of raw_train_eval_data_path_pattern and '
        'transformed_train_eval_data_path_pattern should be provided')
  tf_transform_output = tft.TFTransformOutput(transform_output_dir)

  if raw_train_eval_data_path_pattern is not None:
    selected_input_fn = input_fn_raw
    (train_data_path_pattern,
     eval_data_path_pattern) = raw_train_eval_data_path_pattern
  else:
    selected_input_fn = input_fn
    (train_data_path_pattern,
     eval_data_path_pattern) = transformed_train_eval_data_path_pattern

  train_dataset = selected_input_fn(
      tf_transform_output, train_data_path_pattern, batch_size=TRAIN_BATCH_SIZE
  )

  # Evaluate model on test dataset.
  validation_dataset = selected_input_fn(
      tf_transform_output, eval_data_path_pattern, batch_size=TRAIN_BATCH_SIZE
  )

  feature_spec = tf_transform_output.transformed_feature_spec().copy()
  feature_spec.pop(LABEL_KEY)

  inputs = {}
  sparse_inputs = {}
  dense_inputs = {}
  for key, spec in feature_spec.items():
    if isinstance(spec, tf.io.FixedLenFeature):
      # TODO(b/208879020): Move into schema such that spec.shape is [1] and not
      # [] for scalars.
      inputs[key] = tf_keras.layers.Input(
          shape=spec.shape or [1], name=key, dtype=spec.dtype)
      dense_inputs[key] = inputs[key]
    elif isinstance(spec, tf.io.SparseFeature):
      inputs[key] = tf_keras.layers.Input(
          shape=spec.size, name=key, dtype=spec.dtype, sparse=True
      )
      sparse_inputs[key] = inputs[key]
    else:
      raise ValueError('Spec type is not supported: ', key, spec)

  outputs = [
      tf_keras.layers.Dense(10, activation='relu')(x)
      for x in tf.nest.flatten(sparse_inputs)
  ]
  stacked_inputs = tf.concat(tf.nest.flatten(dense_inputs) + outputs, axis=1)
  output = tf_keras.layers.Dense(100, activation='relu')(stacked_inputs)
  output = tf_keras.layers.Dense(70, activation='relu')(output)
  output = tf_keras.layers.Dense(50, activation='relu')(output)
  output = tf_keras.layers.Dense(20, activation='relu')(output)
  output = tf_keras.layers.Dense(2, activation='sigmoid')(output)
  model = tf_keras.Model(inputs=inputs, outputs=output)

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
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


def main(
    input_data_dir: str,
    working_dir: str,
    read_raw_data_for_training: bool = True,
    num_train_instances: int = NUM_TRAIN_INSTANCES,
    num_test_instances: int = NUM_TEST_INSTANCES,
):
  if not working_dir:
    working_dir = tempfile.mkdtemp(dir=input_data_dir)

  train_data_file = os.path.join(input_data_dir, 'adult.data')
  test_data_file = os.path.join(input_data_dir, 'adult.test')

  transform_data(train_data_file, test_data_file, working_dir)

  if read_raw_data_for_training:
    raw_train_and_eval_patterns = (train_data_file, test_data_file)
    transformed_train_and_eval_patterns = None
  else:
    train_pattern = os.path.join(
        working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE + '*'
    )
    eval_pattern = os.path.join(
        working_dir, TRANSFORMED_TEST_DATA_FILEBASE + '*'
    )
    raw_train_and_eval_patterns = None
    transformed_train_and_eval_patterns = (train_pattern, eval_pattern)
  output_dir = os.path.join(working_dir, EXPORTED_MODEL_DIR)
  results = train_and_evaluate(
      raw_train_and_eval_patterns,
      transformed_train_and_eval_patterns,
      output_dir,
      working_dir,
      num_train_instances=num_train_instances,
      num_test_instances=num_test_instances)

  pprint.pprint(results)


if __name__ == '__main__':
  args = get_args()
  main(args.input_data_dir, args.working_dir)
