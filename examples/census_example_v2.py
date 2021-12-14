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
import census_example_common as common

# Functions for training


def input_fn(tf_transform_output, transformed_examples_pattern, batch_size):
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
      label_key=common.LABEL_KEY,
      shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)


def input_fn_raw(tf_transform_output, raw_examples_pattern, batch_size):
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
    for col in common.ORDERED_CSV_COLUMNS:
      if col not in common.RAW_DATA_FEATURE_SPEC:
        result.append(0.0)
        continue
      spec = common.RAW_DATA_FEATURE_SPEC[col]
      if isinstance(spec, tf.io.FixedLenFeature):
        result.append(spec.dtype)
      else:
        result.append(0.0)
    return result

  dataset = tf.data.experimental.make_csv_dataset(
      file_pattern=raw_examples_pattern,
      batch_size=batch_size,
      column_names=common.ORDERED_CSV_COLUMNS,
      column_defaults=get_ordered_raw_data_dtypes(),
      prefetch_buffer_size=0,
      ignore_errors=True)

  tft_layer = tf_transform_output.transform_features_layer()

  def transform_dataset(data):
    raw_features = {}
    for key, val in data.items():
      if key not in common.RAW_DATA_FEATURE_SPEC:
        continue
      if isinstance(common.RAW_DATA_FEATURE_SPEC[key], tf.io.VarLenFeature):
        # TODO(b/169666856): Remove conversion to sparse once ragged tensors are
        # natively supported.
        raw_features[key] = tf.RaggedTensor.from_tensor(
            tf.expand_dims(val, -1)).to_sparse()
        continue
      raw_features[key] = val
    transformed_features = tft_layer(raw_features)
    data_labels = transformed_features.pop(common.LABEL_KEY)
    return (transformed_features, data_labels)

  return dataset.map(
      transform_dataset,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
          tf.data.experimental.AUTOTUNE)


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
    feature_spec = common.RAW_DATA_FEATURE_SPEC.copy()
    feature_spec.pop(common.LABEL_KEY)
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


def train_and_evaluate(raw_train_eval_data_path_pattern,
                       transformed_train_eval_data_path_pattern,
                       output_dir,
                       transform_output_dir,
                       num_train_instances=common.NUM_TRAIN_INSTANCES,
                       num_test_instances=common.NUM_TEST_INSTANCES):
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
      tf_transform_output,
      train_data_path_pattern,
      batch_size=common.TRAIN_BATCH_SIZE)

  # Evaluate model on test dataset.
  validation_dataset = selected_input_fn(
      tf_transform_output,
      eval_data_path_pattern,
      batch_size=common.TRAIN_BATCH_SIZE)

  feature_spec = tf_transform_output.transformed_feature_spec().copy()
  feature_spec.pop(common.LABEL_KEY)

  inputs = {}
  for key, spec in feature_spec.items():
    if isinstance(spec, tf.io.VarLenFeature):
      inputs[key] = tf.keras.layers.Input(
          shape=[None], name=key, dtype=spec.dtype, sparse=True)
    elif isinstance(spec, tf.io.FixedLenFeature):
      # TODO(b/208879020): Move into schema such that spec.shape is [1] and not
      # [] for scalars.
      inputs[key] = tf.keras.layers.Input(
          shape=spec.shape or [1], name=key, dtype=spec.dtype)
    else:
      raise ValueError('Spec type is not supported: ', key, spec)

  stacked_inputs = tf.concat(tf.nest.flatten(inputs), axis=1)
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

  model.fit(
      train_dataset,
      validation_data=validation_dataset,
      epochs=common.TRAIN_NUM_EPOCHS,
      steps_per_epoch=math.ceil(num_train_instances / common.TRAIN_BATCH_SIZE),
      validation_steps=math.ceil(num_test_instances / common.TRAIN_BATCH_SIZE))

  # Export the model.
  export_serving_model(tf_transform_output, model, output_dir)

  return model.evaluate(validation_dataset, steps=num_test_instances)


def main(input_data_dir,
         working_dir,
         read_raw_data_for_training=True,
         num_train_instances=common.NUM_TRAIN_INSTANCES,
         num_test_instances=common.NUM_TEST_INSTANCES):
  if not working_dir:
    working_dir = tempfile.mkdtemp(dir=input_data_dir)

  train_data_file = os.path.join(input_data_dir, 'adult.data')
  test_data_file = os.path.join(input_data_dir, 'adult.test')

  common.transform_data(train_data_file, test_data_file, working_dir)

  if read_raw_data_for_training:
    raw_train_and_eval_patterns = (train_data_file, test_data_file)
    transformed_train_and_eval_patterns = None
  else:
    train_pattern = os.path.join(working_dir,
                                 common.TRANSFORMED_TRAIN_DATA_FILEBASE + '*')
    eval_pattern = os.path.join(working_dir,
                                common.TRANSFORMED_TEST_DATA_FILEBASE + '*')
    raw_train_and_eval_patterns = None
    transformed_train_and_eval_patterns = (train_pattern, eval_pattern)
  output_dir = os.path.join(working_dir, common.EXPORTED_MODEL_DIR)
  results = train_and_evaluate(
      raw_train_and_eval_patterns,
      transformed_train_and_eval_patterns,
      output_dir,
      working_dir,
      num_train_instances=num_train_instances,
      num_test_instances=num_test_instances)

  pprint.pprint(results)


if __name__ == '__main__':
  args = common.get_args()
  main(args.input_data_dir, args.working_dir)
