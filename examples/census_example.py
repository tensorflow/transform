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
import os
import pprint
import tempfile

import tensorflow as tf
from tensorflow import estimator as tf_estimator
import tensorflow_transform as tft
import census_example_common as common

# Functions for training


def _make_inputs_dense(transformed_features):
  return {
      k: tf.sparse.to_dense(v) if isinstance(v, tf.SparseTensor) else v
      for k, v in transformed_features.items()
  }
# pylint: disable=g-deprecated-tf-checker


def _make_training_input_fn(tf_transform_output, transformed_examples,
                            batch_size):
  """Creates an input function reading from transformed data.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    transformed_examples: Base filename of examples.
    batch_size: Batch size.

  Returns:
    The input function for training or eval.
  """
  def input_fn():
    """Input function for training and eval."""
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=transformed_examples,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=tf.data.TFRecordDataset,
        shuffle=True)

    transformed_features = _make_inputs_dense(
        tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    )

    # Extract features and label from the transformed tensors.
    # TODO(b/30367437): make transformed_labels a dict.
    transformed_labels = tf.where(
        tf.equal(transformed_features.pop(common.LABEL_KEY), 1))

    return transformed_features, transformed_labels[:, 1]

  return input_fn


def _make_serving_input_fn(tf_transform_output):
  """Creates an input function reading from raw data.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.

  Returns:
    The serving input function.
  """
  raw_feature_spec = common.RAW_DATA_FEATURE_SPEC.copy()
  # Remove label since it is not available during serving.
  raw_feature_spec.pop(common.LABEL_KEY)

  def serving_input_fn():
    """Input function for serving."""
    # Get raw features by generating the basic serving input_fn and calling it.
    # Here we generate an input_fn that expects a parsed Example proto to be fed
    # to the model at serving time.  See also
    # tf.estimator.export.build_raw_serving_input_receiver_fn.
    raw_input_fn = tf_estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    # Apply the transform function that was used to generate the materialized
    # data.
    raw_features = serving_input_receiver.features
    transformed_features = _make_inputs_dense(
        tf_transform_output.transform_raw_features(raw_features)
    )

    return tf_estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)

  return serving_input_fn


def get_feature_columns(tf_transform_output):
  """Returns the FeatureColumns for the model.

  Args:
    tf_transform_output: A `TFTransformOutput` object.

  Returns:
    A list of FeatureColumns.
  """
  feature_spec = tf_transform_output.transformed_feature_spec()
  # Wrap scalars as real valued columns.
  def get_shape(spec):
    if isinstance(spec, tf.io.SparseFeature):
      return spec.size
    return spec.shape

  return [
      tf.feature_column.numeric_column(key, shape=get_shape(feature_spec[key]))
      for key in (common.NUMERIC_FEATURE_KEYS + common.CATEGORICAL_FEATURE_KEYS)
  ]


def train_and_evaluate(working_dir,
                       num_train_instances=common.NUM_TRAIN_INSTANCES,
                       num_test_instances=common.NUM_TEST_INSTANCES):
  """Train the model on training data and evaluate on test data.

  Args:
    working_dir: Directory to read transformed data and metadata from and to
        write exported model to.
    num_train_instances: Number of instances in train set
    num_test_instances: Number of instances in test set

  Returns:
    The results from the estimator's 'evaluate' method
  """
  tf_transform_output = tft.TFTransformOutput(working_dir)

  run_config = tf_estimator.RunConfig()

  estimator = tf_estimator.LinearClassifier(
      feature_columns=get_feature_columns(tf_transform_output),
      config=run_config,
      loss_reduction=tf.losses.Reduction.SUM)

  # Fit the model using the default optimizer.
  train_input_fn = _make_training_input_fn(
      tf_transform_output,
      os.path.join(working_dir, common.TRANSFORMED_TRAIN_DATA_FILEBASE + '*'),
      batch_size=common.TRAIN_BATCH_SIZE)
  estimator.train(
      input_fn=train_input_fn,
      max_steps=common.TRAIN_NUM_EPOCHS * num_train_instances /
      common.TRAIN_BATCH_SIZE)

  # Evaluate model on test dataset.
  eval_input_fn = _make_training_input_fn(
      tf_transform_output,
      os.path.join(working_dir, common.TRANSFORMED_TEST_DATA_FILEBASE + '*'),
      batch_size=1)

  # Export the model.
  serving_input_fn = _make_serving_input_fn(tf_transform_output)
  exported_model_dir = os.path.join(working_dir, common.EXPORTED_MODEL_DIR)
  estimator.export_saved_model(exported_model_dir, serving_input_fn)

  return estimator.evaluate(input_fn=eval_input_fn, steps=num_test_instances)


def main():
  args = common.get_args()
  if args.working_dir:
    working_dir = args.working_dir
  else:
    working_dir = tempfile.mkdtemp(dir=args.input_data_dir)

  train_data_file = os.path.join(args.input_data_dir, 'adult.data')
  test_data_file = os.path.join(args.input_data_dir, 'adult.test')

  common.transform_data(train_data_file, test_data_file, working_dir)

  results = train_and_evaluate(working_dir)

  pprint.pprint(results)

if __name__ == '__main__':
  main()
