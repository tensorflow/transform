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
"""Utility functions to build input_fns for use with tf.Learn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_schema

from tensorflow.contrib.learn.python.learn.utils import input_fn_utils


def _convert_scalars_to_vectors(features):
  """Vectorize scalar columns to meet FeatureColumns input requirements."""
  def maybe_expand_dims(tensor):
    # Ignore the SparseTensor case.  In principle it's possible to have a
    # rank-1 SparseTensor that needs to be expanded, but this is very
    # unlikely.
    if isinstance(tensor, tf.Tensor) and tensor.get_shape().ndims == 1:
      tensor = tf.expand_dims(tensor, -1)
    return tensor

  return {name: maybe_expand_dims(tensor)
          for name, tensor in six.iteritems(features)}


def build_csv_transforming_serving_input_fn(
    raw_metadata,
    transform_savedmodel_dir,
    raw_keys,
    field_delim=",",
    convert_scalars_to_vectors=True):
  """Creates input_fn that applies transforms to raw data in csv format.

  CSV files have many restrictions and are not suitable for every input source.
  Consider using build_parsing_transforming_serving_input_fn (which is good for
  input sources of tensorflow records containing tf.example) or
  build_default_transforming_serving_input_fn (which is good for input sources
  like json that list each input tensor).

  CSV input sources have the following restrictions:
    * Only columns with schema tf.FixedLenFeature colums are supported
    * Text columns containing the delimiter must be wrapped in '"'
    * If a string contains a double quote, the double quote must be escaped with
      another double quote, for example: the first column in
      '"quick ""brown"" fox",1,2' becomes 'quick "brown" fox'
    * White space is kept. So a text column "label ," is parsed to 'label '

  Args:
    raw_metadata: a `DatasetMetadata` object describing the raw data.
    transform_savedmodel_dir: a SavedModel directory produced by tf.Transform
      embodying a transformation function to be applied to incoming raw data.
    raw_keys: A list of string keys of the raw labels to be used. The order in
      the list matches the parsing order in the csv file.
    field_delim: Delimiter to separate fields in a record.
    convert_scalars_to_vectors: Boolean specifying whether this input_fn should
      convert scalars into 1-d vectors.  This is necessary if the inputs will be
      used with `FeatureColumn`s as `FeatureColumn`s cannot accept scalar
      inputs. Default: True.

  Raises:
    ValueError: if columns cannot be saved in a csv file.

  Returns:
    An input_fn suitable for serving that applies transforms to raw data in
    CSV format.
  """
  if not raw_keys:
    raise ValueError("raw_keys must be set.")

  column_schemas = raw_metadata.schema.column_schemas

  # Check for errors.
  for k in raw_keys:
    if k not in column_schemas:
      raise ValueError("Key %s does not exist in the schema" % k)
    if not isinstance(column_schemas[k].representation,
                      dataset_schema.FixedColumnRepresentation):
      raise ValueError(("CSV files can only support tensors of fixed size"
                        "which %s is not.") % k)
    shape = column_schemas[k].tf_shape().as_list()
    if shape and shape != [1]:
      # Column is not a scalar-like value. shape == [] or [1] is ok.
      raise ValueError(("CSV files can only support features that are scalars "
                        "having shape []. %s has shape %s")
                       % (k, shape))

  def default_transforming_serving_input_fn():
    """Serving input_fn that applies transforms to raw data in Tensors."""

    record_defaults = []
    for k in raw_keys:
      if column_schemas[k].representation.default_value is not None:
        # Note that 0 and '' are valid defaults.
        value = tf.constant([column_schemas[k].representation.default_value],
                            dtype=column_schemas[k].domain.dtype)
      else:
        value = tf.constant([], dtype=column_schemas[k].domain.dtype)
      record_defaults.append(value)

    placeholder = tf.placeholder(dtype=tf.string, shape=(None,),
                                 name="csv_input_placeholder")
    parsed_tensors = tf.decode_csv(placeholder, record_defaults,
                                   field_delim=field_delim)

    raw_serving_features = {k: v for k, v in zip(raw_keys, parsed_tensors)}

    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            transform_savedmodel_dir, raw_serving_features))

    if convert_scalars_to_vectors:
      transformed_features = _convert_scalars_to_vectors(transformed_features)

    return input_fn_utils.InputFnOps(
        transformed_features, None, {"csv_example": placeholder})

  return default_transforming_serving_input_fn


def build_json_example_transforming_serving_input_fn(
    raw_metadata,
    transform_savedmodel_dir,
    raw_label_keys,
    raw_feature_keys=None,
    convert_scalars_to_vectors=True):
  """Creates input_fn that applies transforms to raw data formatted in json.

  The json is formatted as tf.examples. For example, one input row could contain
  the string for

  {"features": {"feature": {"name": {"int64List": {"value": [42]}}}}}

  which encodes an example containing only feature column 'name' with value 42.

  Args:
    raw_metadata: a `DatasetMetadata` object describing the raw data.
    transform_savedmodel_dir: a SavedModel directory produced by tf.Transform
      embodying a transformation function to be applied to incoming raw data.
    raw_label_keys: A list of string keys of the raw labels to be used. These
      labels are removed from the serving graph. To build a serving function
      that expects labels in the input at serving time, pass raw_labels_keys=[].
    raw_feature_keys: A list of string keys of the raw features to be used.
      If None or empty, defaults to all features except labels.
    convert_scalars_to_vectors: Boolean specifying whether this input_fn should
      convert scalars into 1-d vectors.  This is necessary if the inputs will be
      used with `FeatureColumn`s as `FeatureColumn`s cannot accept scalar
      inputs. Default: True.

  Returns:
    An input_fn suitable for serving that applies transforms to raw data in
    tf.Examples.
  """

  raw_feature_spec = raw_metadata.schema.as_feature_spec()
  raw_feature_keys = _prepare_feature_keys(raw_metadata,
                                           raw_label_keys,
                                           raw_feature_keys)
  raw_serving_feature_spec = {key: raw_feature_spec[key]
                              for key in raw_feature_keys}

  def _serving_input_fn():
    """Applies transforms to raw data in json-example strings."""

    json_example_placeholder = tf.placeholder(tf.string, shape=[None])
    example_strings = tf.decode_json_example(json_example_placeholder)
    raw_features = tf.parse_example(example_strings, raw_serving_feature_spec)
    inputs = {"json_example": json_example_placeholder}

    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            transform_savedmodel_dir, raw_features))

    if convert_scalars_to_vectors:
      transformed_features = _convert_scalars_to_vectors(transformed_features)

    return input_fn_utils.InputFnOps(transformed_features, None, inputs)

  return _serving_input_fn


def build_parsing_transforming_serving_input_fn(
    raw_metadata,
    transform_savedmodel_dir,
    raw_label_keys,
    raw_feature_keys=None,
    convert_scalars_to_vectors=True):
  """Creates input_fn that applies transforms to raw data in tf.Examples.

  Args:
    raw_metadata: a `DatasetMetadata` object describing the raw data.
    transform_savedmodel_dir: a SavedModel directory produced by tf.Transform
      embodying a transformation function to be applied to incoming raw data.
    raw_label_keys: A list of string keys of the raw labels to be used. These
      labels are removed from the serving graph. To build a serving function
      that expects labels in the input at serving time, pass raw_labels_keys=[].
    raw_feature_keys: A list of string keys of the raw features to be used.
      If None or empty, defaults to all features except labels.
    convert_scalars_to_vectors: Boolean specifying whether this input_fn should
      convert scalars into 1-d vectors.  This is necessary if the inputs will be
      used with `FeatureColumn`s as `FeatureColumn`s cannot accept scalar
      inputs. Default: True.

  Returns:
    An input_fn suitable for serving that applies transforms to raw data in
    tf.Examples.
  """

  raw_feature_spec = raw_metadata.schema.as_feature_spec()
  raw_feature_keys = _prepare_feature_keys(raw_metadata,
                                           raw_label_keys,
                                           raw_feature_keys)
  raw_serving_feature_spec = {key: raw_feature_spec[key]
                              for key in raw_feature_keys}

  def parsing_transforming_serving_input_fn():
    """Serving input_fn that applies transforms to raw data in tf.Examples."""
    raw_input_fn = input_fn_utils.build_parsing_serving_input_fn(
        raw_serving_feature_spec, default_batch_size=None)
    raw_features, _, inputs = raw_input_fn()
    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            transform_savedmodel_dir, raw_features))

    if convert_scalars_to_vectors:
      transformed_features = _convert_scalars_to_vectors(transformed_features)

    return input_fn_utils.InputFnOps(transformed_features, None, inputs)

  return parsing_transforming_serving_input_fn


def build_default_transforming_serving_input_fn(
    raw_metadata,
    transform_savedmodel_dir,
    raw_label_keys,
    raw_feature_keys=None,
    convert_scalars_to_vectors=True):
  """Creates input_fn that applies transforms to raw data in Tensors.

  Args:
    raw_metadata: a `DatasetMetadata` object describing the raw data.
    transform_savedmodel_dir: a SavedModel directory produced by tf.Transform
      embodying a transformation function to be applied to incoming raw data.
    raw_label_keys: A list of string keys of the raw labels to be used. These
      labels are removed from the serving graph. To build a serving function
      that expects labels in the input at serving time, pass raw_labels_keys=[].
    raw_feature_keys: A list of string keys of the raw features to be used.
      If None or empty, defaults to all features except labels.
    convert_scalars_to_vectors: Boolean specifying whether this input_fn should
      convert scalars into 1-d vectors.  This is necessary if the inputs will be
      used with `FeatureColumn`s as `FeatureColumn`s cannot accept scalar
      inputs. Default: True.

  Returns:
    An input_fn suitable for serving that applies transforms to raw data in
    tf.Examples.

  Raises:
    ValueError: if raw_label_keys is not provided.
  """
  if raw_label_keys is None:
    raise ValueError("raw_label_keys must be specified.")
  if raw_feature_keys is None:
    raw_feature_keys = list(
        set(six.iterkeys(raw_metadata.schema.column_schemas))
        - set(raw_label_keys))

  def default_transforming_serving_input_fn():
    """Serving input_fn that applies transforms to raw data in Tensors."""

    raw_serving_features = {
        k: v
        for k, v in six.iteritems(raw_metadata.schema.as_batched_placeholders())
        if k in raw_feature_keys}
    sparse_serving_features = [t for t in raw_serving_features
                               if isinstance(t, tf.SparseTensor)]
    if sparse_serving_features:
      raise ValueError("Feeding sparse tensors directly at serving time is not "
                       "supported.")
    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            transform_savedmodel_dir, raw_serving_features))

    if convert_scalars_to_vectors:
      transformed_features = _convert_scalars_to_vectors(transformed_features)

    return input_fn_utils.InputFnOps(
        transformed_features, None, raw_serving_features)

  return default_transforming_serving_input_fn


def build_training_input_fn(metadata,
                            file_pattern,
                            training_batch_size,
                            label_keys,
                            feature_keys=None,
                            reader=tf.TFRecordReader,
                            key_feature_name=None,
                            convert_scalars_to_vectors=True,
                            **read_batch_features_args):
  """Creates an input_fn that reads training data based on its metadata.

  Args:
    metadata: a `DatasetMetadata` object describing the data.
    file_pattern: List of files or pattern of file paths containing
        `Example` records. See `tf.gfile.Glob` for pattern rules.
    training_batch_size: An int or scalar `Tensor` specifying the batch size to
      use.
    label_keys: A list of string keys of the labels to be used.
    feature_keys: A list of string keys of the features to be used.
      If None or empty, defaults to all features except labels.
    reader: A function or class that returns an object with
      `read` method, (filename tensor) -> (example tensor).
    key_feature_name: A name to use to add a key column to the features dict.
      Defaults to None, meaning no key column will be created.
    convert_scalars_to_vectors: Boolean specifying whether this input_fn should
      convert scalars into 1-d vectors.  This is necessary if the inputs will be
      used with `FeatureColumn`s as `FeatureColumn`s cannot accept scalar
      inputs. Default: True.
    **read_batch_features_args: any additional arguments to be passed through to
      `read_batch_features()`, including e.g. queue parameters.

  Returns:
    An input_fn suitable for training that reads training data.
  """
  feature_spec = metadata.schema.as_feature_spec()
  feature_keys = _prepare_feature_keys(metadata, label_keys, feature_keys)

  training_feature_spec = {key: feature_spec[key]
                           for key in feature_keys + label_keys}

  def training_input_fn():
    """A training input function that reads materialized transformed data."""

    if key_feature_name is not None:
      keys, data = tf.contrib.learn.io.read_keyed_batch_features(
          file_pattern, training_batch_size, training_feature_spec,
          reader, **read_batch_features_args)
    else:
      data = tf.contrib.learn.io.read_batch_features(
          file_pattern, training_batch_size, training_feature_spec, reader,
          **read_batch_features_args)

    features = {k: v for k, v in six.iteritems(data) if k in feature_keys}
    labels = {k: v for k, v in six.iteritems(data) if k in label_keys}

    if convert_scalars_to_vectors:
      features = _convert_scalars_to_vectors(features)
      labels = _convert_scalars_to_vectors(labels)

    if key_feature_name is not None:
      features[key_feature_name] = keys

    if len(labels) == 1:
      (_, labels), = labels.items()
    return features, labels

  return training_input_fn


def build_transforming_training_input_fn(raw_metadata,
                                         transformed_metadata,
                                         transform_savedmodel_dir,
                                         raw_data_file_pattern,
                                         training_batch_size,
                                         raw_label_keys,
                                         transformed_label_keys,
                                         raw_feature_keys=None,
                                         transformed_feature_keys=None,
                                         reader=tf.TFRecordReader,
                                         key_feature_name=None,
                                         convert_scalars_to_vectors=True,
                                         **read_batch_features_args):
  """Creates training input_fn that reads raw data and applies transforms.

  Args:
    raw_metadata: a `DatasetMetadata` object describing the raw data.
    transformed_metadata: a `DatasetMetadata` object describing the raw data.
    transform_savedmodel_dir: a SavedModel directory produced by tf.Transform
      embodying a transformation function to be applied to incoming raw data.
    raw_data_file_pattern: List of files or pattern of file paths containing
        `Example` records. See `tf.gfile.Glob` for pattern rules.
    training_batch_size: An int or scalar `Tensor` specifying the batch size to
      use.
    raw_label_keys: A list of string keys of the raw labels to be used.
    transformed_label_keys: A list of string keys of the transformed labels to
      be used.
    raw_feature_keys: A list of string keys of the raw features to be used.
      If None or empty, defaults to all features except labels.
    transformed_feature_keys: A list of string keys of the transformed features
      to be used.  If None or empty, defaults to all features except labels.
    reader: A function or class that returns an object with
      `read` method, (filename tensor) -> (example tensor).
    key_feature_name: A name to use to add a key column to the features dict.
      Defaults to None, meaning no key column will be created.
    convert_scalars_to_vectors: Boolean specifying whether this input_fn should
      convert scalars into 1-d vectors.  This is necessary if the inputs will be
      used with `FeatureColumn`s as `FeatureColumn`s cannot accept scalar
      inputs. Default: True.
    **read_batch_features_args: any additional arguments to be passed through to
      `read_batch_features()`, including e.g. queue parameters.

  Returns:
    An input_fn suitable for training that reads raw training data and applies
    transforms.
  """

  raw_feature_spec = raw_metadata.schema.as_feature_spec()
  raw_feature_keys = _prepare_feature_keys(raw_metadata,
                                           raw_label_keys, raw_feature_keys)
  raw_training_feature_spec = {
      key: raw_feature_spec[key]
      for key in raw_feature_keys + raw_label_keys}

  transformed_feature_keys = _prepare_feature_keys(
      transformed_metadata, transformed_label_keys, transformed_feature_keys)

  def raw_training_input_fn():
    """Training input function that reads raw data and applies transforms."""

    if key_feature_name is not None:
      keys, raw_data = tf.contrib.learn.io.read_keyed_batch_features(
          raw_data_file_pattern, training_batch_size, raw_training_feature_spec,
          reader, **read_batch_features_args)
    else:
      raw_data = tf.contrib.learn.io.read_batch_features(
          raw_data_file_pattern, training_batch_size, raw_training_feature_spec,
          reader, **read_batch_features_args)
    transformed_data = saved_transform_io.apply_saved_transform(
        transform_savedmodel_dir, raw_data)

    transformed_features = {
        k: v for k, v in six.iteritems(transformed_data)
        if k in transformed_feature_keys}
    transformed_labels = {
        k: v for k, v in six.iteritems(transformed_data)
        if k in transformed_label_keys}

    if convert_scalars_to_vectors:
      transformed_features = _convert_scalars_to_vectors(transformed_features)
      transformed_labels = _convert_scalars_to_vectors(transformed_labels)

    if key_feature_name is not None:
      transformed_features[key_feature_name] = keys

    if len(transformed_labels) == 1:
      (_, transformed_labels), = transformed_labels.items()
    return transformed_features, transformed_labels

  return raw_training_input_fn


def _prepare_feature_keys(metadata, label_keys, feature_keys=None):
  """Infer feature keys if needed, and sanity-check label and feature keys."""
  if label_keys is None:
    raise ValueError("label_keys must be specified.")
  if feature_keys is None:
    feature_keys = list(
        set(six.iterkeys(metadata.schema.column_schemas)) - set(label_keys))
  overlap_keys = set(label_keys) & set(feature_keys)
  if overlap_keys:
    raise ValueError("Keys cannot be used as both a feature and a "
                     "label: {}".format(overlap_keys))

  return feature_keys
