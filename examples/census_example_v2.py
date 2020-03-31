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
import math
import os
import pprint
import tempfile

# GOOGLE-INITIALIZATION

from absl import logging
import apache_beam as beam
import tensorflow.compat.v2 as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

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


class MapAndFilterErrors(beam.PTransform):
  """Like beam.Map but filters out erros in the map_fn."""

  class _MapAndFilterErrorsDoFn(beam.DoFn):
    """Count the bad examples using a beam metric."""

    def __init__(self, fn):
      self._fn = fn
      # Create a counter to measure number of bad elements.
      self._bad_elements_counter = beam.metrics.Metrics.counter(
          'census_example', 'bad_elements')

    def process(self, element):
      try:
        yield self._fn(element)
      except Exception:  # pylint: disable=broad-except
        # Catch any exception the above call.
        self._bad_elements_counter.inc(1)

  def __init__(self, fn):
    self._fn = fn

  def expand(self, pcoll):
    return pcoll | beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn))


RAW_DATA_FEATURE_SPEC = dict([(name, tf.io.FixedLenFeature([], tf.string))
                              for name in CATEGORICAL_FEATURE_KEYS] +
                             [(name, tf.io.FixedLenFeature([], tf.float32))
                              for name in NUMERIC_FEATURE_KEYS] +
                             [(name, tf.io.VarLenFeature(tf.float32))
                              for name in OPTIONAL_NUMERIC_FEATURE_KEYS] +
                             [(LABEL_KEY,
                               tf.io.FixedLenFeature([], tf.string))])

RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC))

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
    # Since we are modifying some features and leaving others unchanged, we
    # start by setting `outputs` to a copy of `inputs.
    outputs = inputs.copy()

    # Scale numeric columns to have range [0, 1].
    for key in NUMERIC_FEATURE_KEYS:
      outputs[key] = tft.scale_to_0_1(outputs[key])

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
    # vocabulary but do not modify the feature.  This vocabulary is instead
    # used in the trainer, by means of a feature column, to convert the feature
    # from a string to an integer id.

    for key in CATEGORICAL_FEATURE_KEYS:
      outputs[key] = tft.compute_and_apply_vocabulary(inputs[key],
                                                      vocab_filename=key)

    # TODO(b/149019258): Can't have the label depend on a table lookup op.

    return outputs

  # The "with" block will create a pipeline, and run that pipeline at the exit
  # of the block.
  with beam.Pipeline() as pipeline:
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      # Create a coder to read the census data with the schema.  To do this we
      # need to list all columns in order since the schema doesn't specify the
      # order of columns in the csv.
      ordered_columns = [
          'age', 'workclass', 'fnlwgt', 'education', 'education-num',
          'marital-status', 'occupation', 'relationship', 'race', 'sex',
          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
          'label'
      ]
      converter = tft.coders.CsvCoder(ordered_columns, RAW_DATA_METADATA.schema)

      # Read in raw data and convert using CSV converter.  Note that we apply
      # some Beam transformations here, which will not be encoded in the TF
      # graph since we don't do the from within tf.Transform's methods
      # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
      # to get data into a format that the CSV converter can read, in particular
      # removing spaces after commas.
      #
      # We use MapAndFilterErrors instead of Map to filter out decode errors in
      # convert.decode which should only occur for the trailing blank line.
      raw_data = (
          pipeline
          | 'ReadTrainData' >> beam.io.ReadFromText(train_data_file)
          | 'FixCommasTrainData' >> beam.Map(
              lambda line: line.replace(', ', ','))
          | 'DecodeTrainData' >> MapAndFilterErrors(converter.decode))

      # Combine data and schema into a dataset tuple.  Note that we already used
      # the schema to read the CSV data, but we also need it to interpret
      # raw_data.
      raw_dataset = (raw_data, RAW_DATA_METADATA)
      transformed_dataset, transform_fn = (
          raw_dataset | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data, transformed_metadata = transformed_dataset
      transformed_data_coder = tft.coders.ExampleProtoCoder(
          transformed_metadata.schema)

      _ = (
          transformed_data
          | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
          | 'WriteTrainData' >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

      # Now apply transform function to test data.  In this case we remove the
      # trailing period at the end of each line, and also ignore the header line
      # that is present in the test data file.
      raw_test_data = (
          pipeline
          | 'ReadTestData' >> beam.io.ReadFromText(test_data_file,
                                                   skip_header_lines=1)
          | 'FixCommasTestData' >> beam.Map(
              lambda line: line.replace(', ', ','))
          | 'RemoveTrailingPeriodsTestData' >> beam.Map(lambda line: line[:-1])
          | 'DecodeTestData' >> MapAndFilterErrors(converter.decode))

      raw_test_dataset = (raw_test_data, RAW_DATA_METADATA)

      transformed_test_dataset = (
          (raw_test_dataset, transform_fn) | tft_beam.TransformDataset())
      # Don't need transformed data schema, it's the same as before.
      transformed_test_data, _ = transformed_test_dataset

      _ = (
          transformed_test_data
          | 'EncodeTestData' >> beam.Map(transformed_data_coder.encode)
          | 'WriteTestData' >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)))

      # Will write a SavedModel and metadata to working_dir, which can then
      # be read by the tft.TFTransformOutput class.
      _ = (
          transform_fn
          | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))


# Functions for training


def input_fn(feature_spec, transformed_examples, batch_size):
  """An input function reading from transformed data, converting to model input.

  Args:
    feature_spec: FeatureSpec for the input data.
    transformed_examples: Base filename of examples.
    batch_size: Batch size.

  Returns:
    The input data for training or eval, in the form of k.
  """
  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=transformed_examples,
      batch_size=batch_size,
      features=feature_spec,
      reader=tf.data.TFRecordDataset,
      label_key=LABEL_KEY,
      shuffle=True)

  def format_dataset(data, data_labels):
    """Transforms the label feature."""
    # For the label column we provide the mapping from string to index.
    table_keys = ['>50K', '<=50K']
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=table_keys,
        values=tf.cast(tf.range(len(table_keys)), tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64)
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    data_labels = table.lookup(data_labels)
    data_labels = tf.one_hot(
        indices=data_labels, depth=len(table_keys), on_value=1.0, off_value=0.0)
    return (data, data_labels)

  return dataset.map(format_dataset)


def export_serving_model(tf_transform_output, model, output_dir):
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
    transformed_features.pop(LABEL_KEY)
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


def train_and_evaluate(train_data_path_pattern,
                       eval_data_path_pattern,
                       output_dir,
                       transform_output_dir,
                       num_train_instances=NUM_TRAIN_INSTANCES,
                       num_test_instances=NUM_TEST_INSTANCES):
  """Train the model on training data and evaluate on test data.

  Args:
    train_data_path_pattern: Pattern of train data file paths.
    eval_data_path_pattern: Pattern of eval data file paths.
    output_dir: A directory where the output should be exported to.
    transform_output_dir: The location of the Transform output.
    num_train_instances: Number of instances in train set
    num_test_instances: Number of instances in test set

  Returns:
    The results from the estimator's 'evaluate' method
  """
  tf_transform_output = tft.TFTransformOutput(transform_output_dir)

  train_dataset = input_fn(
      tf_transform_output.transformed_feature_spec(),
      train_data_path_pattern,
      batch_size=TRAIN_BATCH_SIZE)

  # Evaluate model on test dataset.
  validation_dataset = input_fn(
      tf_transform_output.transformed_feature_spec(),
      eval_data_path_pattern,
      batch_size=TRAIN_BATCH_SIZE)

  feature_spec = tf_transform_output.transformed_feature_spec().copy()
  feature_spec.pop(LABEL_KEY)

  inputs = {}
  for key, spec in feature_spec.items():
    if isinstance(spec, tf.io.VarLenFeature):
      inputs[key] = tf.keras.layers.Input(
          shape=spec.shape, name=key, dtype=spec.dtype, sparse=True)
    elif isinstance(spec, tf.io.FixedLenFeature):
      inputs[key] = tf.keras.layers.Input(
          shape=spec.shape, name=key, dtype=spec.dtype)
    else:
      raise ValueError('Spec type is not supported: ', key, spec)

  stacked_inputs = tf.stack([tf.cast(x, tf.float32) for x in inputs.values()],
                            axis=-1)

  output = tf.keras.layers.Dense(100, activation='relu')(stacked_inputs)
  output = tf.keras.layers.Dense(70, activation='relu')(output)
  output = tf.keras.layers.Dense(50, activation='relu')(output)
  output = tf.keras.layers.Dense(20, activation='relu')(output)
  output = tf.keras.layers.Dense(2, activation='sigmoid')(output)
  model = tf.keras.Model(inputs=inputs, outputs=output)

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  logging.info(model.summary())

  model.fit(train_dataset, validation_data=validation_dataset,
            epochs=TRAIN_NUM_EPOCHS,
            steps_per_epoch=math.ceil(num_train_instances / TRAIN_BATCH_SIZE),
            validation_steps=math.ceil(num_test_instances / TRAIN_BATCH_SIZE))

  # Export the model.
  export_serving_model(tf_transform_output, model, output_dir)

  return model.evaluate(validation_dataset, steps=num_test_instances)


def main(input_data_dir,
         working_dir,
         num_train_instances=NUM_TRAIN_INSTANCES,
         num_test_instances=NUM_TEST_INSTANCES):
  if not working_dir:
    working_dir = tempfile.mkdtemp(dir=input_data_dir)

  train_data_file = os.path.join(input_data_dir, 'adult.data')
  test_data_file = os.path.join(input_data_dir, 'adult.test')

  transform_data(train_data_file, test_data_file, working_dir)

  train_pattern = os.path.join(working_dir,
                               TRANSFORMED_TRAIN_DATA_FILEBASE + '*')
  eval_pattern = os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE + '*')
  output_dir = os.path.join(working_dir, EXPORTED_MODEL_DIR)
  results = train_and_evaluate(train_pattern, eval_pattern, output_dir,
                               working_dir, num_train_instances,
                               num_test_instances)

  pprint.pprint(results)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'input_data_dir',
      help='path to directory containing input data')
  parser.add_argument(
      '--working_dir',
      help='optional, path to directory to hold transformed data')
  args = parser.parse_args()
  main(args.input_data_dir, args.working_dir)
