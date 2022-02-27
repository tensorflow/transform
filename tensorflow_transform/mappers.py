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
"""The core public API of TFTransform.  Provide functions to transform tensors.

The core tf.Transform API requires a user to construct a
"preprocessing function" that accepts and returns `Tensor`s.  This function is
built by composing regular functions built from TensorFlow ops, as well as
special functions we refer to as `Analyzer`s.  `Analyzer`s behave similarly to
TensorFlow ops but require a full pass over the whole dataset to compute their
output value.  The analyzers are defined in analyzers.py, while this module
provides helper functions that call analyzers and then use the results of the
anaylzers to transform the original data.

The user-defined preprocessing function should accept and return `Tensor`s that
are batches from the dataset, whose batch size may vary.  For example the
following preprocessing function centers the input 'x' while returning 'y'
unchanged.

import tensorflow_transform as tft

def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']

  # Apply the `mean` analyzer to obtain the mean x.
  x_mean = tft.mean(x)

  # Subtract the mean.
  x_centered = x - mean

  # Return a new dictionary containing x_centered, and y unchanged
  return {
    'x_centered': x_centered,
    'y': y
  }

This user-defined function then must be run using an implementation based on
some distributed computation framework.  The canonical implementation uses
Apache Beam as the underlying framework.  See beam/impl.py for how to use the
Beam implementation.
"""

import os
from typing import Any, Callable, Iterable, Optional, Tuple, Union


import tensorflow as tf
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import common_types
from tensorflow_transform import gaussianization
from tensorflow_transform import schema_inference
from tensorflow_transform import tf_utils
# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple


@common.log_api_use(common.MAPPER_COLLECTION)
def scale_to_gaussian(
    x: common_types.ConsistentTensorType,
    elementwise: bool = False,
    name: Optional[str] = None,
    output_dtype: Optional[tf.DType] = None
) -> common_types.ConsistentTensorType:
  """Returns an (approximately) normal column with mean to 0 and variance 1.

  We transform the column to values that are approximately distributed
  according to a standard normal distribution.
  The transformation is obtained by applying the moments method to estimate
  the parameters of a Tukey HH distribution and applying the inverse of the
  estimated function to the column values.
  The method is partially described in

  Georg M. Georgm "The Lambert Way to Gaussianize Heavy-Tailed Data with the
  Inverse of Tukey's h Transformation as a Special Case," The Scientific World
  Journal, Vol. 2015, Hindawi Publishing Corporation.

  We use the L-moments instead of conventional moments to be able to deal with
  long-tailed distributions. The expressions of the L-moments for the Tukey HH
  distribution is in

  Todd C. Headrick, and Mohan D. Pant. "Characterizing Tukey H and
  HH-Distributions through L-Moments and the L-Correlation," ISRN Applied
  Mathematics, vol. 2012, 2012. doi:10.5402/2012/980153

  Note that the transformation to Gaussian is applied only if the column has
  long-tails. If this is not the case, for instance if values are uniformly
  distributed, the values are only normalized using the z score. This applies
  also to the cases where only one of the tails is long; the other tail is only
  rescaled but not non linearly transformed.
  Also, if the analysis set is empty, the transformation is set to to leave the
  input vaules unchanged.

  Args:
    x: A numeric `Tensor` or `CompositeTensor`.
    elementwise: If true, scales each element of the tensor independently;
        otherwise uses the parameters of the whole tensor.
    name: (Optional) A name for this operation.
    output_dtype: (Optional) If not None, casts the output tensor to this type.

  Returns:
    A `Tensor` or `CompositeTensor` containing the input column transformed to
    be approximately standard distributed (i.e. a Gaussian with mean 0 and
    variance 1). If `x` is floating point, the mean will have the same type as
    `x`. If `x` is integral, the output is cast to tf.float32.

    Note that TFLearn generally permits only tf.int64 and tf.float32, so casting
    this scaler's output may be necessary.
  """
  with tf.compat.v1.name_scope(name, 'scale_to_gaussian'):
    return _scale_to_gaussian_internal(
        x=x,
        elementwise=elementwise,
        output_dtype=output_dtype)


def _scale_to_gaussian_internal(
    x: common_types.ConsistentTensorType,
    elementwise: bool = False,
    output_dtype: Optional[tf.DType] = None
) -> common_types.ConsistentTensorType:
  """Implementation for scale_to_gaussian."""
  # x_mean will be float16, float32, or float64, depending on type of x.
  x_loc, x_scale, hl, hr = analyzers._tukey_parameters(  # pylint: disable=protected-access
      x, reduce_instance_dims=not elementwise, output_dtype=output_dtype)

  compose_result_fn = _make_composite_tensor_wrapper_if_composite(x)
  x_values = tf_utils.get_values(x)

  x_var = analyzers.var(x, reduce_instance_dims=not elementwise,
                        output_dtype=output_dtype)

  if isinstance(x, tf.SparseTensor):
    if elementwise:
      x_loc = tf.gather_nd(x_loc, x.indices[:, 1:])
      x_scale = tf.gather_nd(x_scale, x.indices[:, 1:])
      hl = tf.gather_nd(hl, x.indices[:, 1:])
      hr = tf.gather_nd(hr, x.indices[:, 1:])
      x_var = tf.gather_nd(x_var, x.indices[:, 1:])
  elif isinstance(x, tf.RaggedTensor):
    if elementwise:
      raise NotImplementedError(
          'Elementwise scale_to_gaussian does not support RaggedTensors.')

  numerator = tf.cast(x_values, x_loc.dtype) - x_loc
  is_long_tailed = tf.math.logical_or(hl > 0.0, hr > 0.0)

  # If the distribution is long-tailed, we apply the robust scale computed
  # with L-moments; otherwise, we scale using the standard deviation so that
  # we obtain the same result of scale_to_z_score.
  denominator = tf.where(is_long_tailed, x_scale, tf.sqrt(x_var))
  cond = tf.not_equal(denominator, 0)

  if cond.shape.as_list() != x_values.shape.as_list():
    # Repeats cond when necessary across the batch dimension for it to be
    # compatible with the shape of numerator.
    cond = tf.cast(
        tf.zeros_like(numerator) + tf.cast(cond, numerator.dtype),
        dtype=tf.bool)

  scaled_values = tf.where(cond, tf.divide(numerator, denominator),
                           numerator)
  gaussianized_values = gaussianization.inverse_tukey_hh(scaled_values, hl, hr)
  return compose_result_fn(gaussianized_values)


@common.log_api_use(common.MAPPER_COLLECTION)
def sparse_tensor_to_dense_with_shape(
    x: tf.SparseTensor,
    shape: Union[tf.TensorShape, Iterable[int]],
    default_value: Union[tf.Tensor, int, float, str] = 0) -> tf.Tensor:
  """Converts a `SparseTensor` into a dense tensor and sets its shape.

  Args:
    x: A `SparseTensor`.
    shape: The desired shape of the densified `Tensor`.
    default_value: (Optional) Value to set for indices not specified. Defaults
      to zero.

  Returns:
    A `Tensor` with the desired shape.

  Raises:
    ValueError: If input is not a `SparseTensor`.
  """
  if not isinstance(x, tf.SparseTensor):
    raise ValueError('input must be a SparseTensor')
  new_dense_shape = [
      x.dense_shape[i] if size is None else size
      for i, size in enumerate(shape)
  ]
  dense = tf.raw_ops.SparseToDense(
      sparse_indices=x.indices,
      output_shape=new_dense_shape,
      sparse_values=x.values,
      default_value=default_value)
  dense.set_shape(shape)
  return dense


@common.log_api_use(common.MAPPER_COLLECTION)
def sparse_tensor_left_align(sparse_tensor: tf.SparseTensor) -> tf.SparseTensor:
  """Re-arranges a `tf.SparseTensor` and returns a left-aligned version of it.

  This mapper can be useful when returning a sparse tensor that may not be
  left-aligned from a preprocessing_fn.

  Args:
    sparse_tensor: A 2D `tf.SparseTensor`.

  Raises:
    ValueError if `sparse_tensor` is not 2D.

  Returns:
    A left-aligned version of sparse_tensor as a `tf.SparseTensor`.
  """
  if sparse_tensor.get_shape().ndims != 2:
    raise ValueError('sparse_tensor_left_align requires a 2D input')
  reordered_tensor = tf.sparse.reorder(sparse_tensor)
  transposed_indices = tf.transpose(reordered_tensor.indices)
  row_indices = transposed_indices[0]
  row_counts = tf.unique_with_counts(row_indices, out_idx=tf.int64).count
  column_indices = tf.ragged.range(row_counts).flat_values
  return tf.SparseTensor(
      indices=tf.transpose(tf.stack([row_indices, column_indices])),
      values=reordered_tensor.values,
      dense_shape=reordered_tensor.dense_shape)


@common.log_api_use(common.MAPPER_COLLECTION)
def scale_by_min_max(
    x: common_types.ConsistentTensorType,
    output_min: float = 0.0,
    output_max: float = 1.0,
    elementwise: bool = False,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  """Scale a numerical column into the range [output_min, output_max].

  Args:
    x: A numeric `Tensor` or `CompositeTensor`.
    output_min: The minimum of the range of output values.
    output_max: The maximum of the range of output values.
    elementwise: If true, scale each element of the tensor independently.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the input column scaled to [output_min, output_max].
    If the analysis dataset is empty or contains a singe distinct value, then
    `x` is scaled using a sigmoid function.

  Raises:
    ValueError: If output_min, output_max have the wrong order.
  """
  with tf.compat.v1.name_scope(name, 'scale_by_min_max'):
    return _scale_by_min_max_internal(
        x,
        key=None,
        output_min=output_min,
        output_max=output_max,
        elementwise=elementwise,
        key_vocabulary_filename=None)


@common.log_api_use(common.MAPPER_COLLECTION)
def scale_by_min_max_per_key(
    x: common_types.ConsistentTensorType,
    key: common_types.TensorType,
    output_min: float = 0.0,
    output_max: float = 1.0,
    elementwise: bool = False,
    key_vocabulary_filename: Optional[str] = None,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  # pyformat: disable
  """Scale a numerical column into a predefined range on a per-key basis.

  Args:
    x: A numeric `Tensor` or `CompositeTensor`.
    key: A `Tensor` or `CompositeTensor` of dtype tf.string.
        Must meet one of the following conditions:
        0. key is None
        1. Both x and key are dense,
        2. Both x and key are composite and `key` must exactly match `x` in
           everything except values,
        3. The axis=1 index of each x matches its index of dense key.
    output_min: The minimum of the range of output values.
    output_max: The maximum of the range of output values.
    elementwise: If true, scale each element of the tensor independently.
    key_vocabulary_filename: (Optional) The file name for the per-key file.
      If None, this combiner will assume the keys fit in memory and will not
      store the analyzer result in a file. If '', a file name will be chosen
      based on the current TensorFlow scope. If not '', it should be unique
      within a given preprocessing function.
    name: (Optional) A name for this operation.

  Example:

  >>> def preprocessing_fn(inputs):
  ...   return {
  ...      'scaled': tft.scale_by_min_max_per_key(inputs['x'], inputs['s'])
  ...   }
  >>> raw_data = [dict(x=1, s='a'), dict(x=0, s='b'), dict(x=3, s='a')]
  >>> feature_spec = dict(
  ...     x=tf.io.FixedLenFeature([], tf.float32),
  ...     s=tf.io.FixedLenFeature([], tf.string))
  >>> raw_data_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(
  ...     tft.tf_metadata.schema_utils.schema_from_feature_spec(feature_spec))
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> transformed_data
  [{'scaled': 0.0}, {'scaled': 0.5}, {'scaled': 1.0}]

  Returns:
    A `Tensor`  or `CompositeTensor` containing the input column scaled to
    [output_min, output_max] on a per-key basis if a key is provided. If the
    analysis dataset is empty, a certain key contains a single distinct value or
    the computed key vocabulary doesn't have an entry for `key`, then `x` is
    scaled using a sigmoid function.

  Raises:
    ValueError: If output_min, output_max have the wrong order.
    NotImplementedError: If elementwise is True and key is not None.
    InvalidArgumentError: If indices of sparse x and key do not match.
  """
  # pyformat: enable
  with tf.compat.v1.name_scope(name, 'scale_by_min_max_per_key'):
    if key is None:
      raise ValueError('key is None, call `tft.scale_by_min_max` instead')
    return _scale_by_min_max_internal(
        x,
        key=key,
        output_min=output_min,
        output_max=output_max,
        elementwise=elementwise,
        key_vocabulary_filename=key_vocabulary_filename)


def _scale_by_min_max_internal(
    x: common_types.ConsistentTensorType,
    key: Optional[common_types.TensorType],
    output_min: float,
    output_max: float,
    elementwise: bool,
    key_vocabulary_filename: Optional[str] = None
) -> common_types.ConsistentTensorType:
  """Implementation for scale_by_min_max."""
  if output_min >= output_max:
    raise ValueError('output_min must be less than output_max')

  x = tf.cast(x, tf.float32)
  if key is None:
    min_x_value, max_x_value = analyzers._min_and_max(  # pylint: disable=protected-access
        x,
        reduce_instance_dims=not elementwise)
  else:
    if elementwise:
      raise NotImplementedError('Per-key elementwise reduction not supported')
    key_values = analyzers._min_and_max_per_key(  # pylint: disable=protected-access
        x,
        key,
        reduce_instance_dims=True,
        key_vocabulary_filename=key_vocabulary_filename)
    if key_vocabulary_filename is None:
      key_vocab, min_x_value, max_x_value = key_values
      # Missing keys will translate to 0 for both min and max which will be
      # ignored below in the tf.where.
      min_x_value, max_x_value = tf_utils.map_per_key_reductions(
          (min_x_value, max_x_value), key, key_vocab, x)
    else:
      minus_min_max_for_key = tf_utils.apply_per_key_vocabulary(
          key_values, key, target_ndims=x.get_shape().ndims)
      min_x_value, max_x_value = (
          -minus_min_max_for_key[:, 0], minus_min_max_for_key[:, 1])

  compose_result_fn = _make_composite_tensor_wrapper_if_composite(x)
  x_values = tf_utils.get_values(x)
  if isinstance(x, tf.SparseTensor):
    if elementwise:
      min_x_value = tf.gather_nd(
          tf.broadcast_to(min_x_value, x.dense_shape), x.indices)
      max_x_value = tf.gather_nd(
          tf.broadcast_to(max_x_value, x.dense_shape), x.indices)
  elif isinstance(x, tf.RaggedTensor):
    if elementwise:
      raise NotImplementedError(
          'Elementwise min_and_max does not support RaggedTensors.')

  # If min>=max, then the corresponding input to the min_and_max analyzer either
  # was empty and the analyzer returned default values, or contained only one
  # distinct value. In this case we scale x by applying a sigmoid function which
  # is continuous, increasing and maps (-inf, inf) -> (0, 1). Its output is
  # then projected on the requested range. Note that both the options of
  # tf.where are computed, which means that this will compute unused NaNs.
  numerator = tf.cast(x_values, min_x_value.dtype) - min_x_value
  where_cond = min_x_value < max_x_value
  where_cond = tf.cast(
      tf.zeros_like(numerator) + tf.cast(where_cond, numerator.dtype),
      dtype=tf.bool)
  scaled_result = tf.where(where_cond, numerator / (max_x_value - min_x_value),
                           tf.math.sigmoid(x_values))

  return compose_result_fn((scaled_result * (output_max - output_min)) +
                           output_min)


@common.log_api_use(common.MAPPER_COLLECTION)
def scale_to_0_1(
    x: common_types.ConsistentTensorType,
    elementwise: bool = False,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  """Returns a column which is the input column scaled to have range [0,1].

  Args:
    x: A numeric `Tensor` or `CompositeTensor`.
    elementwise: If true, scale each element of the tensor independently.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `CompositeTensor` containing the input column scaled to
    [0, 1]. If the analysis dataset is empty or contains a single distinct
    value, then `x` is scaled using a sigmoid function.
  """
  with tf.compat.v1.name_scope(name, 'scale_to_0_1'):
    return _scale_by_min_max_internal(
        x,
        key=None,
        output_min=0,
        output_max=1,
        elementwise=elementwise,
        key_vocabulary_filename=None)


@common.log_api_use(common.MAPPER_COLLECTION)
def scale_to_0_1_per_key(
    x: common_types.ConsistentTensorType,
    key: common_types.TensorType,
    elementwise: bool = False,
    key_vocabulary_filename: Optional[str] = None,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  # pyformat: disable
  """Returns a column which is the input column scaled to have range [0,1].

  Args:
    x: A numeric `Tensor` or `CompositeTensor`.
    key: A `Tensor` or `CompositeTensor` of type string.
    elementwise: If true, scale each element of the tensor independently.
    key_vocabulary_filename: (Optional) The file name for the per-key file. If
      None, this combiner will assume the keys fit in memory and will not store
      the analyzer result in a file. If '', a file name will be chosen based on
      the current TensorFlow scope. If not '', it should be unique within a
      given preprocessing function.
    name: (Optional) A name for this operation.

  Example:

  >>> def preprocessing_fn(inputs):
  ...   return {
  ...      'scaled': tft.scale_to_0_1_per_key(inputs['x'], inputs['s'])
  ...   }
  >>> raw_data = [dict(x=1, s='a'), dict(x=0, s='b'), dict(x=3, s='a')]
  >>> feature_spec = dict(
  ...     x=tf.io.FixedLenFeature([], tf.float32),
  ...     s=tf.io.FixedLenFeature([], tf.string))
  >>> raw_data_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(
  ...     tft.tf_metadata.schema_utils.schema_from_feature_spec(feature_spec))
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> transformed_data
  [{'scaled': 0.0}, {'scaled': 0.5}, {'scaled': 1.0}]

  Returns:
    A `Tensor` or `CompositeTensor` containing the input column scaled to [0, 1],
    per key. If the analysis dataset is empty, contains a single distinct value
    or the computed key vocabulary doesn't have an entry for `key`, then `x` is
    scaled using a sigmoid function.
  """
  # pyformat: enable
  with tf.compat.v1.name_scope(name, 'scale_to_0_1_per_key'):
    if key is None:
      raise ValueError('key is None, call `tft.scale_to_0_1` instead')
    return _scale_by_min_max_internal(
        x,
        key=key,
        output_min=0,
        output_max=1,
        elementwise=elementwise,
        key_vocabulary_filename=key_vocabulary_filename)


@common.log_api_use(common.MAPPER_COLLECTION)
def scale_to_z_score(
    x: common_types.ConsistentTensorType,
    elementwise: bool = False,
    name: Optional[str] = None,
    output_dtype: Optional[tf.DType] = None
) -> common_types.ConsistentTensorType:
  """Returns a standardized column with mean 0 and variance 1.

  Scaling to z-score subtracts out the mean and divides by standard deviation.
  Note that the standard deviation computed here is based on the biased variance
  (0 delta degrees of freedom), as computed by analyzers.var.

  Args:
    x: A numeric `Tensor` or `CompositeTensor`.
    elementwise: If true, scales each element of the tensor independently;
        otherwise uses the mean and variance of the whole tensor.
    name: (Optional) A name for this operation.
    output_dtype: (Optional) If not None, casts the output tensor to this type.

  Returns:
    A `Tensor` or `CompositeTensor` containing the input column scaled to mean 0
    and variance 1 (standard deviation 1), given by: (x - mean(x)) / std_dev(x).
    If `x` is floating point, the mean will have the same type as `x`. If `x` is
    integral, the output is cast to tf.float32. If the analysis dataset is empty
    or contains a single distinct value, then the input is returned without
    scaling.

    Note that TFLearn generally permits only tf.int64 and tf.float32, so casting
    this scaler's output may be necessary.
  """
  with tf.compat.v1.name_scope(name, 'scale_to_z_score'):
    return _scale_to_z_score_internal(
        x=x,
        key=None,
        elementwise=elementwise,
        key_vocabulary_filename=None,
        output_dtype=output_dtype)


@common.log_api_use(common.MAPPER_COLLECTION)
def scale_to_z_score_per_key(
    x: common_types.ConsistentTensorType,
    key: common_types.TensorType,
    elementwise: bool = False,
    key_vocabulary_filename: Optional[str] = None,
    name: Optional[str] = None,
    output_dtype: Optional[tf.DType] = None
) -> common_types.ConsistentTensorType:
  """Returns a standardized column with mean 0 and variance 1, grouped per key.

  Scaling to z-score subtracts out the mean and divides by standard deviation.
  Note that the standard deviation computed here is based on the biased variance
  (0 delta degrees of freedom), as computed by analyzers.var.

  Args:
    x: A numeric `Tensor` or `CompositeTensor`.
    key: A Tensor or `CompositeTensor` of dtype tf.string.
        Must meet one of the following conditions:
        0. key is None
        1. Both x and key are dense,
        2. Both x and key are sparse and `key` must exactly match `x` in
        everything except values,
        3. The axis=1 index of each x matches its index of dense key.
    elementwise: If true, scales each element of the tensor independently;
        otherwise uses the mean and variance of the whole tensor.
        Currently, not supported for per-key operations.
    key_vocabulary_filename: (Optional) The file name for the per-key file.
      If None, this combiner will assume the keys fit in memory and will not
      store the analyzer result in a file. If '', a file name will be chosen
      based on the current TensorFlow scope. If not '', it should be unique
      within a given preprocessing function.
    name: (Optional) A name for this operation.
    output_dtype: (Optional) If not None, casts the output tensor to this type.

  Returns:
    A `Tensor` or `CompositeTensor` containing the input column scaled to mean 0
    and variance 1 (standard deviation 1), grouped per key if a key is provided.

    That is, for all keys k: (x - mean(x)) / std_dev(x) for all x with key k.
    If `x` is floating point, the mean will have the same type as `x`. If `x` is
    integral, the output is cast to tf.float32. If the analysis dataset is
    empty, contains a single distinct value or the computed key vocabulary
    doesn't have an entry for `key`, then the input is returned without scaling.

    Note that TFLearn generally permits only tf.int64 and tf.float32, so casting
    this scaler's output may be necessary.
  """
  with tf.compat.v1.name_scope(name, 'scale_to_z_score_per_key'):
    if key is None:
      raise ValueError('key is None, call `tft.scale_to_z_score` instead')
    return _scale_to_z_score_internal(
        x=x,
        key=key,
        elementwise=elementwise,
        key_vocabulary_filename=key_vocabulary_filename,
        output_dtype=output_dtype)


def _scale_to_z_score_internal(
    x: common_types.ConsistentTensorType,
    key: Optional[common_types.TensorType], elementwise: bool,
    key_vocabulary_filename: Optional[str],
    output_dtype: Optional[tf.DType]) -> common_types.ConsistentTensorType:
  """Implementation for scale_to_z_score."""
  # x_mean will be float16, float32, or float64, depending on type of x
  if key is None:
    x_mean, x_var = analyzers._mean_and_var(  # pylint: disable=protected-access
        x,
        reduce_instance_dims=not elementwise,
        output_dtype=output_dtype)
  else:
    if elementwise:
      raise NotImplementedError('Per-key elementwise reduction not supported')

    mean_and_var_per_key_result = analyzers._mean_and_var_per_key(  # pylint: disable=protected-access
        x, key, key_vocabulary_filename=key_vocabulary_filename,
        output_dtype=output_dtype)

    if key_vocabulary_filename is None:
      # Missing keys will translate to 0 for both mean and var which will be
      # ignored below in the tf.where.
      key_vocab, key_means, key_vars = mean_and_var_per_key_result
      x_mean, x_var = tf_utils.map_per_key_reductions((key_means, key_vars),
                                                      key, key_vocab, x)
    else:
      mean_var_for_key = tf_utils.apply_per_key_vocabulary(
          mean_and_var_per_key_result, key, target_ndims=x.get_shape().ndims)
      x_mean, x_var = (mean_var_for_key[:, 0], mean_var_for_key[:, 1])

  compose_result_fn = _make_composite_tensor_wrapper_if_composite(x)
  x_values = tf_utils.get_values(x)

  if isinstance(x, tf.SparseTensor):
    if elementwise:
      x_mean = tf.gather_nd(tf.broadcast_to(x_mean, x.dense_shape), x.indices)
      x_var = tf.gather_nd(tf.broadcast_to(x_var, x.dense_shape), x.indices)
  elif isinstance(x, tf.RaggedTensor):
    if elementwise:
      raise NotImplementedError(
          'Elementwise scale_to_z_score does not support RaggedTensors')

  numerator = tf.cast(x_values, x_mean.dtype) - x_mean
  denominator = tf.sqrt(x_var)
  cond = tf.not_equal(denominator, 0)

  if cond.shape.as_list() != x_values.shape.as_list():
    # Repeats cond when necessary across the batch dimension for it to be
    # compatible with the shape of numerator.
    cond = tf.cast(
        tf.zeros_like(numerator) + tf.cast(cond, numerator.dtype),
        dtype=tf.bool)

  deviation_values = tf.where(cond, tf.divide(numerator, denominator),
                              numerator)
  return compose_result_fn(deviation_values)


@common.log_api_use(common.MAPPER_COLLECTION)
def tfidf(
    x: tf.SparseTensor,
    vocab_size: int,
    smooth: bool = True,
    name: Optional[str] = None) -> Tuple[tf.SparseTensor, tf.SparseTensor]:
  # pyformat: disable
  """Maps the terms in x to their term frequency * inverse document frequency.

  The term frequency of a term in a document is calculated as
  (count of term in document) / (document size)

  The inverse document frequency of a term is, by default, calculated as
  1 + log((corpus size + 1) / (count of documents containing term + 1)).


  Example usage:

  >>> def preprocessing_fn(inputs):
  ...   integerized = tft.compute_and_apply_vocabulary(inputs['x'])
  ...   vocab_size = tft.get_num_buckets_for_transformed_feature(integerized)
  ...   vocab_index, tfidf_weight = tft.tfidf(integerized, vocab_size)
  ...   return {
  ...      'index': vocab_index,
  ...      'tf_idf': tfidf_weight,
  ...      'integerized': integerized,
  ...   }
  >>> raw_data = [dict(x=["I", "like", "pie", "pie", "pie"]),
  ...             dict(x=["yum", "yum", "pie"])]
  >>> feature_spec = dict(x=tf.io.VarLenFeature(tf.string))
  >>> raw_data_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(
  ...     tft.tf_metadata.schema_utils.schema_from_feature_spec(feature_spec))
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> transformed_data
  [{'index': array([0, 2, 3]), 'integerized': array([3, 2, 0, 0, 0]),
    'tf_idf': array([0.6, 0.28109303, 0.28109303], dtype=float32)},
   {'index': array([0, 1]), 'integerized': array([1, 1, 0]),
    'tf_idf': array([0.33333334, 0.9369768 ], dtype=float32)}]

    ```
    example strings: [["I", "like", "pie", "pie", "pie"], ["yum", "yum", "pie]]
    in: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                              [1, 0], [1, 1], [1, 2]],
                     values=[1, 2, 0, 0, 0, 3, 3, 0])
    out: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                      values=[1, 2, 0, 3, 0])
         SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                      values=[(1/5)*(log(3/2)+1), (1/5)*(log(3/2)+1), (3/5),
                              (2/3)*(log(3/2)+1), (1/3)]
    ```

    NOTE: the first doc's duplicate "pie" strings have been combined to
    one output, as have the second doc's duplicate "yum" strings.

  Args:
    x: A 2D `SparseTensor` representing int64 values (most likely that are the
        result of calling `compute_and_apply_vocabulary` on a tokenized string).
    vocab_size: An int - the count of vocab used to turn the string into int64s
        including any OOV buckets.
    smooth: A bool indicating if the inverse document frequency should be
        smoothed. If True, which is the default, then the idf is calculated as
        1 + log((corpus size + 1) / (document frequency of term + 1)).
        Otherwise, the idf is
        1 +log((corpus size) / (document frequency of term)), which could
        result in a division by zero error.
    name: (Optional) A name for this operation.

  Returns:
    Two `SparseTensor`s with indices [index_in_batch, index_in_bag_of_words].
    The first has values vocab_index, which is taken from input `x`.
    The second has values tfidf_weight.

  Raises:
    ValueError if `x` does not have 2 dimensions.
  """
  # pyformat: enable
  if x.get_shape().ndims != 2:
    raise ValueError('tft.tfidf requires a 2D SparseTensor input. '
                     'Input had {} dimensions.'.format(x.get_shape().ndims))

  def _to_vocab_range(x):
    """Enforces that the vocab_ids in x are positive."""
    return tf.SparseTensor(
        indices=x.indices,
        values=tf.math.mod(x.values, vocab_size),
        dense_shape=x.dense_shape)

  with tf.compat.v1.name_scope(name, 'tfidf'):
    cleaned_input = _to_vocab_range(x)

    term_frequencies = _to_term_frequency(cleaned_input, vocab_size)

    count_docs_with_term_column = _count_docs_with_term(term_frequencies)
    # Expand dims to get around the min_tensor_rank checks
    sizes = tf.expand_dims(tf.shape(input=cleaned_input)[0], 0)
    # [batch, vocab] - tfidf
    tfidfs = _to_tfidf(term_frequencies,
                       analyzers.sum(count_docs_with_term_column,
                                     reduce_instance_dims=False),
                       analyzers.sum(sizes),
                       smooth)
    return _split_tfidfs_to_outputs(tfidfs)


def _split_tfidfs_to_outputs(
    tfidfs: tf.SparseTensor) -> Tuple[tf.SparseTensor, tf.SparseTensor]:
  """Splits [batch, vocab]-weight into [batch, bow]-vocab & [batch, bow]-tfidf.

  Args:
    tfidfs: the `SparseTensor` output of _to_tfidf
  Returns:
    Two `SparseTensor`s with indices [index_in_batch, index_in_bag_of_words].
    The first has values vocab_index, which is taken from input `x`.
    The second has values tfidf_weight.
  """
  # Split tfidfs tensor into [batch, dummy] -> vocab & [batch, dummy] -> tfidf
  # The "dummy" index counts from 0 to the number of unique tokens in the doc.
  # So example doc ["I", "like", "pie", "pie", "pie"], with 3 unique tokens,
  # will have "dummy" indices [0, 1, 2]. The particular dummy index that any
  # token receives is not important, only that the tfidf value and vocab index
  # have the *same* dummy index, so that feature_column can apply the weight to
  # the correct vocab item.
  dummy_index = segment_indices(tfidfs.indices[:, 0])
  out_index = tf.concat(
      [tf.expand_dims(tfidfs.indices[:, 0], 1),
       tf.expand_dims(dummy_index, 1)], 1)

  out_shape_second_dim = tf.maximum(
      tf.reduce_max(input_tensor=dummy_index), -1) + 1
  out_shape = tf.stack([tfidfs.dense_shape[0], out_shape_second_dim])
  out_shape.set_shape([2])

  de_duped_indicies_out = tf.SparseTensor(  # NOTYPO ('indices')
      indices=out_index,
      values=tfidfs.indices[:, 1],
      dense_shape=out_shape)
  de_duped_tfidf_out = tf.SparseTensor(
      indices=out_index,
      values=tfidfs.values,
      dense_shape=out_shape)
  return de_duped_indicies_out, de_duped_tfidf_out  # NOTYPO ('indices')


def _to_term_frequency(x: tf.SparseTensor,
                       vocab_size: Union[int, tf.Tensor]) -> tf.SparseTensor:
  """Creates a SparseTensor of term frequency for every doc/term pair.

  Args:
    x : a SparseTensor of int64 representing string indices in vocab.
    vocab_size: A scalar int64 Tensor - the count of vocab used to turn the
        string into int64s including any OOV buckets.

  Returns:
    a SparseTensor with the count of times a term appears in a document at
        indices <doc_index_in_batch>, <term_index_in_vocab>,
        with size (num_docs_in_batch, vocab_size).
  """
  # Construct intermediary sparse tensor with indices
  # [<doc>, <term_index_in_doc>, <vocab_id>] and tf.ones values.
  vocab_size = tf.convert_to_tensor(value=vocab_size, dtype=tf.int64)
  split_indices = tf.cast(
      tf.split(x.indices, axis=1, num_or_size_splits=2), dtype=tf.int64)
  expanded_values = tf.cast(tf.expand_dims(x.values, 1), dtype=tf.int64)
  next_index = tf.concat(
      [split_indices[0], split_indices[1], expanded_values], axis=1)

  next_values = tf.ones_like(x.values)
  expanded_vocab_size = tf.expand_dims(vocab_size, 0)
  next_shape = tf.concat(
      [x.dense_shape, expanded_vocab_size], 0)

  next_tensor = tf.SparseTensor(
      indices=tf.cast(next_index, dtype=tf.int64),
      values=next_values,
      dense_shape=next_shape)

  # Take the intermediary tensor and reduce over the term_index_in_doc
  # dimension. This produces a tensor with indices [<doc_id>, <term_id>]
  # and values [count_of_term_in_doc] and shape batch x vocab_size
  term_count_per_doc = tf.compat.v1.sparse_reduce_sum_sparse(next_tensor, 1)

  dense_doc_sizes = tf.cast(
      tf.sparse.reduce_sum(
          tf.SparseTensor(
              indices=x.indices,
              values=tf.ones_like(x.values),
              dense_shape=x.dense_shape), 1),
      dtype=tf.float64)

  gather_indices = term_count_per_doc.indices[:, 0]
  gathered_doc_sizes = tf.gather(dense_doc_sizes, gather_indices)

  term_frequency = (
      tf.cast(term_count_per_doc.values, dtype=tf.float64) /
      tf.cast(gathered_doc_sizes, dtype=tf.float64))
  return tf.SparseTensor(
      indices=term_count_per_doc.indices,
      values=term_frequency,
      dense_shape=term_count_per_doc.dense_shape)


def _to_tfidf(term_frequency: tf.SparseTensor, reduced_term_freq: tf.Tensor,
              corpus_size: tf.Tensor, smooth: bool) -> tf.SparseTensor:
  """Calculates the inverse document frequency of terms in the corpus.

  Args:
    term_frequency: The `SparseTensor` output of _to_term_frequency.
    reduced_term_freq: A `Tensor` of shape (vocabSize,) that represents the
        count of the number of documents with each term.
    corpus_size: A scalar count of the number of documents in the corpus.
    smooth: A bool indicating if the idf value should be smoothed. See
        tfidf_weights documentation for details.

  Returns:
    A `SparseTensor` with indices=<doc_index_in_batch>, <term_index_in_vocab>,
    values=term frequency * inverse document frequency,
    and shape=(batch, vocab_size)
  """
  # The idf tensor has shape (vocab_size,)
  if smooth:
    idf = tf.math.log((tf.cast(corpus_size, dtype=tf.float64) + 1.0) /
                      (1.0 + tf.cast(reduced_term_freq, dtype=tf.float64))) + 1
  else:
    idf = tf.math.log(
        tf.cast(corpus_size, dtype=tf.float64) /
        (tf.cast(reduced_term_freq, dtype=tf.float64))) + 1

  gathered_idfs = tf.gather(tf.squeeze(idf), term_frequency.indices[:, 1])
  tfidf_values = (tf.cast(term_frequency.values, tf.float32)
                  * tf.cast(gathered_idfs, tf.float32))

  return tf.SparseTensor(
      indices=term_frequency.indices,
      values=tfidf_values,
      dense_shape=term_frequency.dense_shape)


def _count_docs_with_term(term_frequency: tf.SparseTensor) -> tf.Tensor:
  """Computes the number of documents in a batch that contain each term.

  Args:
    term_frequency: The `SparseTensor` output of _to_term_frequency.
  Returns:
    A `Tensor` of shape (vocab_size,) that contains the number of documents in
    the batch that contain each term.
  """
  count_of_doc_inter = tf.SparseTensor(
      indices=term_frequency.indices,
      values=tf.ones_like(term_frequency.values),
      dense_shape=term_frequency.dense_shape)
  out = tf.sparse.reduce_sum(count_of_doc_inter, axis=0)
  return tf.expand_dims(out, 0)


@common.log_api_use(common.MAPPER_COLLECTION)
def compute_and_apply_vocabulary(
    x: common_types.ConsistentTensorType,
    default_value: Any = -1,
    top_k: Optional[int] = None,
    frequency_threshold: Optional[int] = None,
    num_oov_buckets: int = 0,
    vocab_filename: Optional[str] = None,
    weights: Optional[tf.Tensor] = None,
    labels: Optional[tf.Tensor] = None,
    use_adjusted_mutual_info: bool = False,
    min_diff_from_avg: float = 0.0,
    coverage_top_k: Optional[int] = None,
    coverage_frequency_threshold: Optional[int] = None,
    key_fn: Optional[Callable[[Any], Any]] = None,
    fingerprint_shuffle: bool = False,
    file_format: common_types.VocabularyFileFormatType = analyzers
    .DEFAULT_VOCABULARY_FILE_FORMAT,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  r"""Generates a vocabulary for `x` and maps it to an integer with this vocab.

  In case one of the tokens contains the '\n' or '\r' characters or is empty it
  will be discarded since we are currently writing the vocabularies as text
  files. This behavior will likely be fixed/improved in the future.

  Note that this function will cause a vocabulary to be computed.  For large
  datasets it is highly recommended to either set frequency_threshold or top_k
  to control the size of the vocabulary, and also the run time of this
  operation.

  Args:
    x: A `Tensor` or `CompositeTensor` of type tf.string or tf.int[8|16|32|64].
    default_value: The value to use for out-of-vocabulary values, unless
      'num_oov_buckets' is greater than zero.
    top_k: Limit the generated vocabulary to the first `top_k` elements. If set
      to None, the full vocabulary is generated.
    frequency_threshold: Limit the generated vocabulary only to elements whose
      absolute frequency is >= to the supplied threshold. If set to None, the
      full vocabulary is generated.  Absolute frequency means the number of
      occurences of the element in the dataset, as opposed to the proportion of
      instances that contain that element. If labels are provided and the vocab
      is computed using mutual information, tokens are filtered if their mutual
      information with the label is < the supplied threshold.
    num_oov_buckets:  Any lookup of an out-of-vocabulary token will return a
      bucket ID based on its hash if `num_oov_buckets` is greater than zero.
      Otherwise it is assigned the `default_value`.
    vocab_filename: The file name for the vocabulary file. If None, a name based
      on the scope name in the context of this graph will be used as the
      file name. If not None, should be unique within a given preprocessing
      function.
      NOTE in order to make your pipelines resilient to implementation details
      please set `vocab_filename` when you are using the vocab_filename on a
      downstream component.
    weights: (Optional) Weights `Tensor` for the vocabulary. It must have the
      same shape as x.
    labels: (Optional) A `Tensor` of labels for the vocabulary. If provided,
      the vocabulary is calculated based on mutual information with the label,
      rather than frequency. The labels must have the same batch dimension as x.
      If x is sparse, labels should be a 1D tensor reflecting row-wise labels.
      If x is dense, labels can either be a 1D tensor of row-wise labels, or
      a dense tensor of the identical shape as x (i.e. element-wise labels).
      Labels should be a discrete integerized tensor (If the label is numeric,
      it should first be bucketized; If the label is a string, an integer
      vocabulary should first be applied). Note: `CompositeTensor` labels are
      not yet supported (b/134931826). WARNING: when labels are provided, the
      frequency_threshold argument functions as a mutual information threshold,
      which is a float. TODO(b/116308354): Fix confusing naming.
    use_adjusted_mutual_info: If true, use adjusted mutual information.
    min_diff_from_avg: Mutual information of a feature will be adjusted to zero
      whenever the difference between count of the feature with any label and
      its expected count is lower than min_diff_from_average.
    coverage_top_k: (Optional), (Experimental) The minimum number of elements
      per key to be included in the vocabulary.
    coverage_frequency_threshold: (Optional), (Experimental) Limit the coverage
      arm of the vocabulary only to elements whose absolute frequency is >= this
      threshold for a given key.
    key_fn: (Optional), (Experimental) A fn that takes in a single entry of `x`
      and returns the corresponding key for coverage calculation. If this is
      `None`, no coverage arm is added to the vocabulary.
    fingerprint_shuffle: (Optional), (Experimental) Whether to sort the
      vocabularies by fingerprint instead of counts. This is useful for load
      balancing on the training parameter servers. Shuffle only happens while
      writing the files, so all the filters above will still take effect.
    file_format: (Optional) A str. The format of the resulting vocabulary file.
      Accepted formats are: 'tfrecord_gzip', 'text'. 'tfrecord_gzip' requires
      tensorflow>=2.4.
      The default value is 'text'.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `CompositeTensor` where each string value is mapped to an
    integer. Each unique string value that appears in the vocabulary
    is mapped to a different integer and integers are consecutive starting from
    zero. String value not in the vocabulary is assigned default_value.
    Alternatively, if num_oov_buckets is specified, out of vocabulary strings
    are hashed to values in [vocab_size, vocab_size + num_oov_buckets) for an
    overall range of [0, vocab_size + num_oov_buckets).

  Raises:
    ValueError: If `top_k` or `frequency_threshold` is negative.
      If `coverage_top_k` or `coverage_frequency_threshold` is negative.
  """
  with tf.compat.v1.name_scope(name, 'compute_and_apply_vocabulary'):
    deferred_vocab_and_filename = analyzers.vocabulary(
        x=x,
        top_k=top_k,
        frequency_threshold=frequency_threshold,
        vocab_filename=vocab_filename,
        weights=weights,
        labels=labels,
        use_adjusted_mutual_info=use_adjusted_mutual_info,
        min_diff_from_avg=min_diff_from_avg,
        coverage_top_k=coverage_top_k,
        coverage_frequency_threshold=coverage_frequency_threshold,
        key_fn=key_fn,
        fingerprint_shuffle=fingerprint_shuffle,
        file_format=file_format)
    return apply_vocabulary(
        x,
        deferred_vocab_and_filename,
        default_value,
        num_oov_buckets,
        file_format=file_format)


@common.log_api_use(common.MAPPER_COLLECTION)
def apply_vocabulary(
    x: common_types.ConsistentTensorType,
    deferred_vocab_filename_tensor: common_types.TemporaryAnalyzerOutputType,
    default_value: Any = -1,
    num_oov_buckets: int = 0,
    lookup_fn: Optional[Callable[[common_types.TensorType, tf.Tensor],
                                 Tuple[tf.Tensor, tf.Tensor]]] = None,
    file_format: common_types.VocabularyFileFormatType = analyzers
    .DEFAULT_VOCABULARY_FILE_FORMAT,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  r"""Maps `x` to a vocabulary specified by the deferred tensor.

  This function also writes domain statistics about the vocabulary min and max
  values. Note that the min and max are inclusive, and depend on the vocab size,
  num_oov_buckets and default_value.

  Args:
    x: A categorical `Tensor` or `CompositeTensor` of type tf.string or
      tf.int[8|16|32|64] to which the vocabulary transformation should be
      applied. The column names are those intended for the transformed tensors.
    deferred_vocab_filename_tensor: The deferred vocab filename tensor as
      returned by `tft.vocabulary`, as long as the frequencies were not stored.
    default_value: The value to use for out-of-vocabulary values, unless
      'num_oov_buckets' is greater than zero.
    num_oov_buckets:  Any lookup of an out-of-vocabulary token will return a
      bucket ID based on its hash if `num_oov_buckets` is greater than zero.
      Otherwise it is assigned the `default_value`.
    lookup_fn: Optional lookup function, if specified it should take a tensor
      and a deferred vocab filename as an input and return a lookup `op` along
      with the table size, by default `apply_vocabulary` constructs a
      StaticHashTable for the table lookup.
    file_format: (Optional) A str. The format of the given vocabulary.
      Accepted formats are: 'tfrecord_gzip', 'text'.
      The default value is 'text'.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `CompositeTensor` where each string value is mapped to an
    integer. Each unique string value that appears in the vocabulary
    is mapped to a different integer and integers are consecutive
    starting from zero, and string value not in the vocabulary is
    assigned default_value.
  """
  if (file_format == 'tfrecord_gzip' and
      not tf_utils.is_vocabulary_tfrecord_supported()):
    raise ValueError(
        'Vocabulary file_format "tfrecord_gzip" not yet supported for '
        f'{tf.version.VERSION}.')
  with tf.compat.v1.name_scope(name, 'apply_vocab'):
    if x.dtype != tf.string and not x.dtype.is_integer:
      raise ValueError('expected tf.string or tf.int[8|16|32|64] but got %r' %
                       x.dtype)

    if lookup_fn:
      result, table_size = tf_utils.lookup_table(
          lookup_fn, deferred_vocab_filename_tensor, x)
    else:
      if (deferred_vocab_filename_tensor is None or
          (isinstance(deferred_vocab_filename_tensor,
                      (bytes, str)) and not deferred_vocab_filename_tensor)):
        raise ValueError('`deferred_vocab_filename_tensor` must not be empty.')

      def _construct_table(asset_filepath):
        if file_format == 'tfrecord_gzip':
          initializer = tf_utils.make_tfrecord_vocabulary_lookup_initializer(
              asset_filepath, x.dtype)
        elif file_format == 'text':
          initializer = tf.lookup.TextFileInitializer(
              asset_filepath,
              key_dtype=x.dtype,
              key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
              value_dtype=tf.int64,
              value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
        else:
          raise ValueError(
              '"{}" is not an accepted file_format. It should be one of: {}'
              .format(file_format, analyzers.ALLOWED_VOCABULARY_FILE_FORMATS))

        if num_oov_buckets > 0:
          table = tf.lookup.StaticVocabularyTable(
              initializer,
              num_oov_buckets=num_oov_buckets,
              lookup_key_dtype=x.dtype)
        else:
          table = tf.lookup.StaticHashTable(
              initializer, default_value=default_value)
        return table

      compose_result_fn = _make_composite_tensor_wrapper_if_composite(x)
      x_values = tf_utils.get_values(x)
      result, table_size = tf_utils.construct_and_lookup_table(
          _construct_table, deferred_vocab_filename_tensor, x_values)
      result = compose_result_fn(result)

    # Specify schema overrides which will override the values in the schema
    # with the min and max values, which are deferred as they are only known
    # once the analyzer has run.
    #
    # `table_size` includes the num oov buckets.  The default value is only used
    # if num_oov_buckets <= 0.
    min_value = tf.constant(0, tf.int64)
    max_value = table_size - 1
    if num_oov_buckets <= 0:
      min_value = tf.minimum(min_value, default_value)
      max_value = tf.maximum(max_value, default_value)
    schema_inference.set_tensor_schema_override(
        tf_utils.get_values(result), min_value, max_value)
    return result


@common.log_api_use(common.MAPPER_COLLECTION)
def get_num_buckets_for_transformed_feature(
    transformed_feature: common_types.TensorType) -> tf.Tensor:
  # pyformat: disable
  """Provides the number of buckets for a transformed feature if annotated.

  This for example can be used for the direct output of `tft.bucketize`,
  `tft.apply_buckets`, `tft.compute_and_apply_vocabulary`,
  `tft.apply_vocabulary`.
  These methods annotate the transformed feature with additional information.
  If the given `transformed_feature` isn't annotated, this method will fail.

  Example:

  >>> def preprocessing_fn(inputs):
  ...   bucketized = tft.bucketize(inputs['x'], num_buckets=3)
  ...   integerized = tft.compute_and_apply_vocabulary(inputs['x'])
  ...   zeros = tf.zeros_like(inputs['x'], tf.int64)
  ...   return {
  ...      'bucketized': bucketized,
  ...      'bucketized_num_buckets': (
  ...         zeros + tft.get_num_buckets_for_transformed_feature(bucketized)),
  ...      'integerized': integerized,
  ...      'integerized_num_buckets': (
  ...         zeros + tft.get_num_buckets_for_transformed_feature(integerized)),
  ...   }
  >>> raw_data = [dict(x=3),dict(x=23)]
  >>> feature_spec = dict(x=tf.io.FixedLenFeature([], tf.int64))
  >>> raw_data_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(
  ...     tft.tf_metadata.schema_utils.schema_from_feature_spec(feature_spec))
  >>> with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  ...   transformed_dataset, transform_fn = (
  ...       (raw_data, raw_data_metadata)
  ...       | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
  >>> transformed_data, transformed_metadata = transformed_dataset
  >>> transformed_data
  [{'bucketized': 1, 'bucketized_num_buckets': 3,
   'integerized': 0, 'integerized_num_buckets': 2},
  {'bucketized': 2, 'bucketized_num_buckets': 3,
   'integerized': 1, 'integerized_num_buckets': 2}]

  Args:
    transformed_feature: A `Tensor` or `SparseTensor` which is the direct output
      of `tft.bucketize`, `tft.apply_buckets`,
      `tft.compute_and_apply_vocabulary` or `tft.apply_vocabulary`.

  Raises:
    ValueError: If the given tensor has not been annotated a the number of
    buckets.

  Returns:
    A `Tensor` with the number of buckets for the given `transformed_feature`.
  """
  # pyformat: enable
  # Adding 1 to the 2nd Tensor of the returned pair in order to compute max + 1.
  return tf.cast(
      schema_inference.get_tensor_schema_override(transformed_feature)[1] + 1,
      tf.int64)


@common.log_api_use(common.MAPPER_COLLECTION)
def segment_indices(segment_ids: tf.Tensor,
                    name: Optional[str] = None) -> tf.Tensor:
  """Returns a `Tensor` of indices within each segment.

  segment_ids should be a sequence of non-decreasing non-negative integers that
  define a set of segments, e.g. [0, 0, 1, 2, 2, 2] defines 3 segments of length
  2, 1 and 3.  The return value is a `Tensor` containing the indices within each
  segment.

  Example:

  >>> result = tft.segment_indices(tf.constant([0, 0, 1, 2, 2, 2]))
  >>> print(result)
  tf.Tensor([0 1 0 0 1 2], shape=(6,), dtype=int32)

  Args:
    segment_ids: A 1-d `Tensor` containing an non-decreasing sequence of
        non-negative integers with type `tf.int32` or `tf.int64`.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` containing the indices within each segment.
  """
  ndims = segment_ids.get_shape().ndims
  if ndims != 1 and ndims is not None:
    raise ValueError(
        'segment_indices requires a 1-dimensional input. '
        'segment_indices has {} dimensions.'.format(ndims))
  with tf.compat.v1.name_scope(name, 'segment_indices'):
    # TODO(KesterTong): This is a fundamental operation for segments, write a C++
    # op to do this.
    # TODO(KesterTong): Add a check that segment_ids are increasing.
    segment_lengths = tf.math.segment_sum(
        tf.ones_like(segment_ids), segment_ids)
    segment_starts = tf.gather(tf.concat([[0], tf.cumsum(segment_lengths)], 0),
                               segment_ids)
    return (tf.range(tf.size(input=segment_ids, out_type=segment_ids.dtype)) -
            segment_starts)


@common.log_api_use(common.MAPPER_COLLECTION)
def deduplicate_tensor_per_row(input_tensor, name=None):
  """Deduplicates each row (0-th dimension) of the provided tensor.

  Args:
    input_tensor: A two-dimensional `Tensor` or `SparseTensor`. The first
      dimension is assumed to be the batch or "row" dimension, and deduplication
      is done on the 2nd dimension. If the Tensor is 1D it is returned as the
      equivalent `SparseTensor` since the "row" is a scalar can't be further
      deduplicated.
    name: Optional name for the operation.

  Returns:
    A  `SparseTensor` containing the unique set of values from each
      row of the input. Note: the original order of the input may not be
      preserved.
  """
  with tf.compat.v1.name_scope(name, 'deduplicate_per_row'):

    if isinstance(input_tensor, tf.SparseTensor):
      batch_dim = tf.cast(input_tensor.dense_shape[0], tf.int32)
      rank = input_tensor.dense_shape.shape[0]
    else:
      batch_dim = tf.cast(tf.shape(input_tensor)[0], tf.int32)
      rank = input_tensor.shape.rank

    def _univalent_dense_to_sparse(batch_dim, input_tensor):
      """Helper to convert a 1D dense `Tensor` to a `SparseTensor`."""
      indices = tf.cast(
          tf.stack([
              tf.range(batch_dim, dtype=tf.int32),
              tf.zeros(batch_dim, dtype=tf.int32)
          ],
                   axis=1),
          dtype=tf.int64)

      return tf.SparseTensor(
          indices=indices, values=input_tensor, dense_shape=(batch_dim, 1))

    if rank is not None:
      # If the rank is known at graph construction time, and it's rank 1, there
      # is no deduplication to be done so we can return early.
      if rank <= 1:
        if isinstance(input_tensor, tf.SparseTensor):
          return input_tensor
        # Even though we are just returning as is, we convert to a SparseTensor
        # to ensure consistent output type.
        return _univalent_dense_to_sparse(batch_dim, input_tensor)
      if rank > 2:
        raise ValueError(
            'Deduplication assumes a rank 2 tensor, got {}.'.format(rank))
      return _deduplicate_tensor_per_row(input_tensor, batch_dim)

    if isinstance(input_tensor, tf.SparseTensor):
      return _deduplicate_tensor_per_row(input_tensor, batch_dim)
    else:
      # Again check for rank 1 tensor (that doesn't need deduplication), this
      # time handling inputs where rank isn't known until execution time.
      dynamic_rank = tf.rank(input_tensor)
      return tf.cond(
          tf.equal(dynamic_rank, 1),
          lambda: _univalent_dense_to_sparse(batch_dim, input_tensor),
          lambda: _deduplicate_tensor_per_row(input_tensor, batch_dim),
      )


_DedupRowLoopArgs = tfx_namedtuple.namedtuple(
    'DedupRowLoopArgs',
    [
        'index',  # Index representing the row of input_tensor to be processed.
        'input_tensor',  # `Tensor` or `SparseTensor` to be deuplicated per row.
        'indices',  # `TensorArray` containing indices of each deduplicated row.
        'values',  # `TensorArray` containing values of each deduplicated row.
        'max_unique',  # Tracks the maximum size of any row.
    ])


class _DedupRowLoopVars(_DedupRowLoopArgs):
  """Loop variables for _deduplicate_per_row."""
  pass


def _deduplicate_tensor_per_row(input_tensor, batch_dim):
  """Helper function for deduplicating each row of the provided tensor.

  For each input row, computes the unique values and set them in positions 0
  through num_unique - 1 within the row.

  Args:
    input_tensor: A `Tensor` or `SparseTensor` to be deuplicated per row.
    batch_dim: The batch dimension or number of "rows" in the batch.

  Returns:
    A  `SparseTensor` containing the unique set of values from each
      row of the input. Note: the original order of the input may not be
      preserved.
  """
  max_unique = tf.constant(0, dtype=tf.int64)
  values = tf.TensorArray(
      size=batch_dim,
      dtype=input_tensor.dtype,
      element_shape=[None],
      infer_shape=False)
  indices = tf.TensorArray(
      size=batch_dim,
      dtype=tf.int64,
      element_shape=[None, 2],
      infer_shape=False)

  def _deduplicate_row(dedup_row_loop_vars):
    """Deduplicates the values in the i-th row of the input.

    Args:
      dedup_row_loop_vars: A _DedupRowLoopVars NamedTuple.

    Returns:
      Updated version of the _DedupRowLoopVars for the loop iteration.
    """
    index, input_tensor, indices, values, max_unique = dedup_row_loop_vars
    if isinstance(input_tensor, tf.SparseTensor):

      row = tf.sparse.slice(input_tensor, [index, 0],
                            [1, input_tensor.dense_shape[1]])
      row_values, _ = tf.unique(row.values)
    else:
      row = input_tensor[index]
      row_values, _ = tf.unique(row)

    # Keep track of the maximum number of unique elements in a row, as this
    # will determine the resulting dense shape.
    max_unique = tf.cast(
        tf.maximum(tf.cast(tf.shape(row_values)[0], tf.int64), max_unique),
        tf.int64)
    column_indices = tf.cast(
        tf.expand_dims(tf.range(tf.shape(row_values)[0]), axis=1), tf.int64)
    row_indices = tf.fill(tf.shape(column_indices), tf.cast(index, tf.int64))
    values = values.write(index, row_values)
    indices = indices.write(index, tf.concat([row_indices, column_indices], 1))
    return [
        _DedupRowLoopVars(index + 1, input_tensor, indices, values, max_unique)
    ]

  index = tf.constant(0, tf.int32)
  (loop_output,) = tf.while_loop(
      lambda loop_args: loop_args.index < batch_dim,
      _deduplicate_row,
      [_DedupRowLoopVars(index, input_tensor, indices, values, max_unique)],
      back_prop=False)

  dense_shape = tf.convert_to_tensor(
      [tf.cast(batch_dim, tf.int64),
       tf.cast(loop_output.max_unique, tf.int64)],
      dtype=tf.int64)
  return tf.SparseTensor(
      indices=tf.cast(loop_output.indices.concat(), tf.int64),
      values=loop_output.values.concat(),
      dense_shape=dense_shape)


@common.log_api_use(common.MAPPER_COLLECTION)
def bag_of_words(tokens: tf.SparseTensor,
                 ngram_range: Tuple[int, int],
                 separator: str,
                 name: Optional[str] = None) -> tf.SparseTensor:
  """Computes a bag of "words" based on the specified ngram configuration.

  A light wrapper around tft.ngrams. First computes ngrams, then transforms the
  ngram representation (list semantics) into a Bag of Words (set semantics) per
  row. Each row reflects the set of *unique* ngrams present in an input record.

  See tft.ngrams for more information.

  Args:
    tokens: a two-dimensional `SparseTensor` of dtype `tf.string` containing
      tokens that will be used to construct a bag of words.
    ngram_range: A pair with the range (inclusive) of ngram sizes to compute.
    separator: a string that will be inserted between tokens when ngrams are
      constructed.
    name: (Optional) A name for this operation.

  Returns:
    A `SparseTensor` containing the unique set of ngrams from each row of the
      input. Note: the original order of the ngrams may not be preserved.
  """
  if tokens.get_shape().ndims != 2:
    raise ValueError('bag_of_words requires `tokens` to be 2-dimensional')
  with tf.compat.v1.name_scope(name, 'bag_of_words'):
    # First compute the ngram representation, which will contain ordered and
    # possibly duplicated ngrams per row.
    all_ngrams = ngrams(tokens, ngram_range, separator)
    # Then deduplicate the ngrams in each row.
    return deduplicate_tensor_per_row(all_ngrams)


@common.log_api_use(common.MAPPER_COLLECTION)
def ngrams(tokens: tf.SparseTensor,
           ngram_range: Tuple[int, int],
           separator: str,
           name: Optional[str] = None) -> tf.SparseTensor:
  """Create a `SparseTensor` of n-grams.

  Given a `SparseTensor` of tokens, returns a `SparseTensor` containing the
  ngrams that can be constructed from each row.

  `separator` is inserted between each pair of tokens, so " " would be an
  appropriate choice if the tokens are words, while "" would be an appropriate
  choice if they are characters.

  Example:

  >>> tokens = tf.SparseTensor(
  ...         indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]],
  ...         values=['One', 'was', 'Johnny', 'Two', 'was', 'a', 'rat'],
  ...         dense_shape=[2, 4])
  >>> print(tft.ngrams(tokens, ngram_range=(1, 3), separator=' '))
  SparseTensor(indices=tf.Tensor(
      [[0 0] [0 1] [0 2] [0 3] [0 4] [0 5]
       [1 0] [1 1] [1 2] [1 3] [1 4] [1 5] [1 6] [1 7] [1 8]],
       shape=(15, 2), dtype=int64),
    values=tf.Tensor(
      [b'One' b'One was' b'One was Johnny' b'was' b'was Johnny' b'Johnny' b'Two'
       b'Two was' b'Two was a' b'was' b'was a' b'was a rat' b'a' b'a rat'
       b'rat'], shape=(15,), dtype=string),
    dense_shape=tf.Tensor([2 9], shape=(2,), dtype=int64))

  Args:
    tokens: a two-dimensional`SparseTensor` of dtype `tf.string` containing
      tokens that will be used to construct ngrams.
    ngram_range: A pair with the range (inclusive) of ngram sizes to return.
    separator: a string that will be inserted between tokens when ngrams are
      constructed.
    name: (Optional) A name for this operation.

  Returns:
    A `SparseTensor` containing all ngrams from each row of the input. Note:
    if an ngram appears multiple times in the input row, it will be present the
    same number of times in the output. For unique ngrams, see tft.bag_of_words.

  Raises:
    ValueError: if `tokens` is not 2D.
    ValueError: if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]
  """
  # This function is implemented as follows.  Assume we start with the following
  # `SparseTensor`:
  #
  # indices=[[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [2, 1], [2, 2]]
  # values=['a', 'b', 'c', 'd', 'q', 'x', 'y', 'z']
  # dense_shape=[3, 4]
  #
  # First we then create shifts of the values and first column of indices,
  # buffering to avoid overrunning the end of the array, so the shifted values
  # (if we are ngrams up to size 3) are
  #
  # shifted_batch_indices[0]=[0, 0, 0, 0, 1, 2, 2, 2]
  # shifted_tokens[0]=['a', 'b', 'c', 'd', 'q', 'x', 'y', 'z']
  #
  # shifted_batch_indices[1]=[0, 0, 0, 1, 2, 2, 2, -1]
  # shifted_tokens[1]=['b', 'c', 'd', 'q', 'x', 'y', 'z', '']
  #
  # shifted_batch_indices[2]=[0, 0, 1, 2, 2, 2, -1, -1]
  # shifted_tokens[2]=['c', 'd', 'q', 'x', 'y', 'z', '', '']
  #
  # These shifted ngrams are used to create the ngrams as follows.  We use
  # tf.string_join to join shifted_tokens[:k] to create k-grams. The `separator`
  # string is inserted between each pair of tokens in the k-gram.
  # The batch that the first of these belonged to is given by
  # shifted_batch_indices[0]. However some of these will cross the boundaries
  # between 'batches' and so we we create a boolean mask which is True when
  # shifted_indices[:k] are all equal.
  #
  # This results in tensors of ngrams, their batch indices and a boolean mask,
  # which we then use to construct the output SparseTensor.
  if tokens.get_shape().ndims != 2:
    raise ValueError('ngrams requires `tokens` to be 2-dimensional')
  with tf.compat.v1.name_scope(name, 'ngrams'):
    if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
      raise ValueError('Invalid ngram_range: %r' % (ngram_range,))

    def _sliding_windows(values, num_shifts, fill_value):
      buffered_values = tf.concat(
          [values, tf.fill([num_shifts - 1], fill_value)], 0)
      return [
          tf.slice(buffered_values, [i], tf.shape(input=values))
          for i in range(num_shifts)
      ]

    shifted_batch_indices = _sliding_windows(
        tokens.indices[:, 0], ngram_range[1] + 1,
        tf.constant(-1, dtype=tf.int64))
    shifted_tokens = _sliding_windows(tokens.values, ngram_range[1] + 1, '')

    # Construct a tensor of the form
    # [['a', 'ab, 'abc'], ['b', 'bcd', cde'], ...]
    def _string_join(tensors):
      if tensors:
        return tf.strings.join(tensors, separator=separator)
      else:
        return

    ngrams_array = [_string_join(shifted_tokens[:k])
                    for k in range(ngram_range[0], ngram_range[1] + 1)]
    ngrams_tensor = tf.stack(ngrams_array, 1)

    # Construct a boolean mask for whether each ngram in ngram_tensor is valid,
    # in that each character came from the same batch.
    valid_ngram = tf.equal(
        tf.math.cumprod(
            tf.cast(
                tf.equal(
                    tf.stack(shifted_batch_indices, 1),
                    tf.expand_dims(shifted_batch_indices[0], 1)),
                dtype=tf.int32),
            axis=1), 1)
    valid_ngram = valid_ngram[:, (ngram_range[0] - 1):ngram_range[1]]

    # Construct a tensor with the batch that each ngram in ngram_tensor belongs
    # to.
    batch_indices = tf.tile(tf.expand_dims(tokens.indices[:, 0], 1),
                            [1, ngram_range[1] + 1 - ngram_range[0]])

    # Apply the boolean mask and construct a SparseTensor with the given indices
    # and values, where another index is added to give the position within a
    # batch.
    batch_indices = tf.boolean_mask(tensor=batch_indices, mask=valid_ngram)
    ngrams_tensor = tf.boolean_mask(tensor=ngrams_tensor, mask=valid_ngram)
    instance_indices = segment_indices(batch_indices)
    dense_shape_second_dim = tf.maximum(
        tf.reduce_max(input_tensor=instance_indices), -1) + 1
    return tf.SparseTensor(
        indices=tf.stack([batch_indices, instance_indices], 1),
        values=ngrams_tensor,
        dense_shape=tf.stack(
            [tokens.dense_shape[0], dense_shape_second_dim]))


@common.log_api_use(common.MAPPER_COLLECTION)
def word_count(tokens: Union[tf.SparseTensor, tf.RaggedTensor],
               name: Optional[str] = None) -> tf.Tensor:
  # pyformat: disable
  """Find the token count of each document/row.

  `tokens` is either a `RaggedTensor` or `SparseTensor`, representing tokenized
  strings. This function simply returns size of each row, so the dtype is not
  constrained to string.

  Example:
  >>> sparse = tf.SparseTensor(indices=[[0, 0], [0, 1], [2, 2]],
  ...                          values=['a', 'b', 'c'], dense_shape=(4, 4))
  >>> tft.word_count(sparse)
  <tf.Tensor: shape=(4,), dtype=int64, numpy=array([2, 0, 1, 0])>

  Args:
    tokens: either
      (1) a `SparseTensor`, or
      (2) a `RaggedTensor` with ragged rank of 1, non-ragged rank of 1
      of dtype `tf.string` containing tokens to be counted
    name: (Optional) A name for this operation.

  Returns:
    A one-dimensional `Tensor` the token counts of each row.

  Raises:
    ValueError: if tokens is neither sparse nor ragged
  """
  # pyformat: enable
  with tf.compat.v1.name_scope(name, 'word_count'):
    if isinstance(tokens, tf.RaggedTensor):
      return tokens.row_lengths()
    elif isinstance(tokens, tf.SparseTensor):
      result = tf.sparse.reduce_sum(
          tf.SparseTensor(indices=tokens.indices,
                          values=tf.ones_like(tokens.values, dtype=tf.int64),
                          dense_shape=tokens.dense_shape),
          axis=list(range(1, tokens.get_shape().ndims)))
      result.set_shape([tokens.shape[0]])
      return result
    else:
      raise ValueError('Invalid token tensor')


@common.log_api_use(common.MAPPER_COLLECTION)
def hash_strings(
    strings: common_types.ConsistentTensorType,
    hash_buckets: int,
    key: Optional[Iterable[int]] = None,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  """Hash strings into buckets.

  Args:
    strings: a `Tensor` or `CompositeTensor` of dtype `tf.string`.
    hash_buckets: the number of hash buckets.
    key: optional. An array of two Python `uint64`. If passed, output will be
      a deterministic function of `strings` and `key`. Note that hashing will be
      slower if this value is specified.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `CompositeTensor` of dtype `tf.int64` with the same shape as
    the input `strings`.

  Raises:
    TypeError: if `strings` is not a `Tensor` or `CompositeTensor` of dtype
    `tf.string`.
  """
  if (not isinstance(strings, (tf.Tensor, tf.SparseTensor, tf.RaggedTensor)) or
      strings.dtype != tf.string):
    raise TypeError(
        'Input to hash_strings must be a Tensor or CompositeTensor of dtype '
        'string; got {}'.format(strings.dtype))
  if isinstance(strings, tf.Tensor):
    if name is None:
      name = 'hash_strings'
    if key is None:
      return tf.strings.to_hash_bucket_fast(strings, hash_buckets, name=name)
    return tf.strings.to_hash_bucket_strong(
        strings, hash_buckets, key, name=name)
  else:
    compose_result_fn = _make_composite_tensor_wrapper_if_composite(strings)
    values = tf_utils.get_values(strings)
    return compose_result_fn(hash_strings(values, hash_buckets, key))


@common.log_api_use(common.MAPPER_COLLECTION)
def bucketize(x: common_types.ConsistentTensorType,
              num_buckets: int,
              epsilon: Optional[float] = None,
              weights: Optional[tf.Tensor] = None,
              elementwise: bool = False,
              name: Optional[str] = None) -> common_types.ConsistentTensorType:
  """Returns a bucketized column, with a bucket index assigned to each input.

  Args:
    x: A numeric input `Tensor` or `CompositeTensor` whose values should be
      mapped to buckets.  For a `CompositeTensor` only non-missing values will
      be included in the quantiles computation, and the result of `bucketize`
      will be a `CompositeTensor` with non-missing values mapped to buckets. If
      elementwise=True then `x` must be dense.
    num_buckets: Values in the input `x` are divided into approximately
      equal-sized buckets, where the number of buckets is `num_buckets`.
    epsilon: (Optional) Error tolerance, typically a small fraction close to
      zero. If a value is not specified by the caller, a suitable value is
      computed based on experimental results.  For `num_buckets` less than 100,
      the value of 0.01 is chosen to handle a dataset of up to ~1 trillion input
      data values.  If `num_buckets` is larger, then epsilon is set to
      (1/`num_buckets`) to enforce a stricter error tolerance, because more
      buckets will result in smaller range for each bucket, and so we want the
      boundaries to be less fuzzy. See analyzers.quantiles() for details.
    weights: (Optional) Weights tensor for the quantiles. Tensor must have the
      same shape as x.
    elementwise: (Optional) If true, bucketize each element of the tensor
      independently.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` of the same shape as `x`, with each element in the
    returned tensor representing the bucketized value. Bucketized value is
    in the range [0, actual_num_buckets). Sometimes the actual number of buckets
    can be different than num_buckets hint, for example in case the number of
    distinct values is smaller than num_buckets, or in cases where the
    input values are not uniformly distributed.
    NaN values are mapped to the last bucket. Values with NaN weights are
    ignored in bucket boundaries calculation.

  Raises:
    TypeError: If num_buckets is not an int.
    ValueError: If value of num_buckets is not > 1.
    ValueError: If elementwise=True and x is a `CompositeTensor`.
  """
  with tf.compat.v1.name_scope(name, 'bucketize'):
    if not isinstance(num_buckets, int):
      raise TypeError('num_buckets must be an int, got %s' % type(num_buckets))

    if num_buckets < 1:
      raise ValueError('Invalid num_buckets %d' % num_buckets)

    if isinstance(x, (tf.SparseTensor, tf.RaggedTensor)) and elementwise:
      raise ValueError(
          'bucketize requires `x` to be dense if `elementwise=True`')

    if epsilon is None:
      # See explanation in args documentation for epsilon.
      epsilon = min(1.0 / num_buckets, 0.01)

    x_values = tf_utils.get_values(x)
    bucket_boundaries = analyzers.quantiles(
        x_values,
        num_buckets,
        epsilon,
        weights,
        reduce_instance_dims=not elementwise)

    if not elementwise:
      return apply_buckets(x, bucket_boundaries)

    num_features = tf.math.reduce_prod(x.get_shape()[1:])
    bucket_boundaries = tf.reshape(bucket_boundaries, [num_features, -1])
    x_reshaped = tf.reshape(x, [-1, num_features])
    bucketized = []
    for idx, boundaries in enumerate(tf.unstack(bucket_boundaries, axis=0)):
      bucketized.append(apply_buckets(x_reshaped[:, idx],
                                      tf.expand_dims(boundaries, axis=0)))
    return tf.reshape(tf.stack(bucketized, axis=1),
                      [-1] + x.get_shape().as_list()[1:])


# TODO(b/179891014): Implement key_vocabulary_filename for bucketize_per_key.
@common.log_api_use(common.MAPPER_COLLECTION)
def bucketize_per_key(
    x: common_types.ConsistentTensorType,
    key: common_types.ConsistentTensorType,
    num_buckets: int,
    epsilon: Optional[float] = None,
    weights: Optional[common_types.ConsistentTensorType] = None,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  """Returns a bucketized column, with a bucket index assigned to each input.

  Args:
    x: A numeric input `Tensor` or `CompositeTensor` with rank 1, whose values
      should be mapped to buckets.  `CompositeTensor`s will have their
      non-missing values mapped and missing values left as missing.
    key: A Tensor or `CompositeTensor` with the same shape as `x` and dtype
      tf.string.  If `x` is a `CompositeTensor`, `key` must exactly match `x` in
      everything except values, i.e. indices and dense_shape or nested row
      splits must be identical.
    num_buckets: Values in the input `x` are divided into approximately
      equal-sized buckets, where the number of buckets is num_buckets.
    epsilon: (Optional) see `bucketize`.
    weights: (Optional) A Tensor or `CompositeTensor` with the same shape as `x`
      and dtype tf.float32. Used as weights for quantiles calculation. If `x` is
      a `CompositeTensor`, `weights` must exactly match `x` in everything except
      values.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `CompositeTensor` of the same shape as `x`, with each element
    in the returned tensor representing the bucketized value. Bucketized value
    is in the range [0, actual_num_buckets). If the computed key vocabulary
    doesn't have an entry for `key` then the resulting bucket is -1.

  Raises:
    ValueError: If value of num_buckets is not > 1.
  """
  with tf.compat.v1.name_scope(name, 'bucketize_per_key'):
    if not isinstance(num_buckets, int):
      raise TypeError(
          'num_buckets must be an int, got {}'.format(type(num_buckets)))

    if num_buckets < 1:
      raise ValueError('Invalid num_buckets {}'.format(num_buckets))

    if epsilon is None:
      # See explanation in args documentation for epsilon.
      epsilon = min(1.0 / num_buckets, 0.01)

    (key_vocab, bucket_boundaries, scale_factor_per_key, shift_per_key,
     actual_num_buckets) = (
         analyzers._quantiles_per_key(  # pylint: disable=protected-access
             tf_utils.get_values(x),
             tf_utils.get_values(key),
             num_buckets,
             epsilon,
             weights=tf_utils.get_values(weights)))
    return _apply_buckets_with_keys(x, key, key_vocab, bucket_boundaries,
                                    scale_factor_per_key, shift_per_key,
                                    actual_num_buckets)


def _make_composite_tensor_wrapper_if_composite(
    x: common_types.ConsistentTensorType
) -> Callable[[tf.Tensor], common_types.ConsistentTensorType]:
  """Produces a function to wrap values in the composite structure of x."""
  if isinstance(x, tf.SparseTensor):
    return lambda values: tf.SparseTensor(x.indices, values, x.dense_shape)
  elif isinstance(x, tf.RaggedTensor):

    def from_nested_row_splits(values):
      return tf.RaggedTensor.from_nested_row_splits(values, x.nested_row_splits)

    return from_nested_row_splits
  else:
    return lambda values: values


def _fill_shape(value, shape, dtype):
  return tf.cast(tf.fill(shape, value), dtype)


def _apply_buckets_with_keys(
    x: common_types.ConsistentTensorType,
    key: common_types.ConsistentTensorType,
    key_vocab: tf.Tensor,
    bucket_boundaries: tf.Tensor,
    scale_factor_per_key: tf.Tensor,
    shift_per_key: tf.Tensor,
    num_buckets: int,
    name: Optional[int] = None) -> common_types.ConsistentTensorType:
  """Bucketize a Tensor or CompositeTensor where boundaries depend on the index.

  Args:
    x: A 1-d Tensor or CompositeTensor.
    key: A 1-d Tensor or CompositeTensor with the same size as x.
    key_vocab: A vocab containing all keys.  Must be exhaustive, an out-of-vocab
      entry in `key` will cause a crash.
    bucket_boundaries: A rank-1 Tensor.
    scale_factor_per_key: A rank-1 Tensor of shape (key_size,).
    shift_per_key: A rank-1 Tensor of shape (key_size,).
    num_buckets: A scalar.
    name: (Optional) A name for this operation.

  Returns:
    A tensor with the same shape as `x` and dtype tf.int64. If any value in
    `key` is not present in `key_vocab` then the resulting bucket will be -1.
  """
  with tf.compat.v1.name_scope(name, 'apply_buckets_with_keys'):
    x_values = tf.cast(tf_utils.get_values(x), tf.float32)
    compose_result_fn = _make_composite_tensor_wrapper_if_composite(x)
    key_values = tf_utils.get_values(key)

    # Convert `key_values` to indices in key_vocab.
    key_indices = tf_utils.lookup_key(key_values, key_vocab)

    adjusted_key_indices = tf.where(
        key_indices < 0, _fill_shape(0, tf.shape(key_indices), tf.int64),
        key_indices)

    # Apply the per-key offsets to x, which produces offset buckets (where the
    # bucket offset is an integer offset).  Then remove this offset to get the
    # actual per-key buckets for x.
    scale_factors = tf.gather(scale_factor_per_key, adjusted_key_indices)
    shifts = tf.gather(shift_per_key, adjusted_key_indices)

    transformed_x = x_values * scale_factors + shifts

    offset_buckets = tf_utils.assign_buckets(
        transformed_x, bucket_boundaries, side=tf_utils.Side.RIGHT)

    max_bucket = num_buckets - 1

    # Shift the bucket numbers back to the correct range [0, num_buckets].
    # We use max_bucket-1 due to different keys sharing 1 boundary.
    corrected_buckets = offset_buckets - (
        (max_bucket - 1) * adjusted_key_indices)
    bucketized_values = tf.clip_by_value(corrected_buckets, 0, max_bucket)

    # Set values with missing keys as -1.
    bucketized_values = tf.where(key_indices < 0, key_indices,
                                 bucketized_values)

    # Attach the relevant metadata to result, so that the corresponding
    # output feature will have this metadata set.
    min_value = tf.constant(0, tf.int64)
    schema_inference.set_tensor_schema_override(
        bucketized_values, min_value, max_bucket)

    return compose_result_fn(bucketized_values)


@common.log_api_use(common.MAPPER_COLLECTION)
def apply_buckets_with_interpolation(
    x: common_types.ConsistentTensorType,
    bucket_boundaries: common_types.BucketBoundariesType,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  """Interpolates within the provided buckets and then normalizes to 0 to 1.

  A method for normalizing continuous numeric data to the range [0, 1].
  Numeric values are first bucketized according to the provided boundaries, then
  linearly interpolated within their respective bucket ranges. Finally, the
  interpolated values are normalized to the range [0, 1]. Values that are
  less than or equal to the lowest boundary, or greater than or equal to the
  highest boundary, will be mapped to 0 and 1 respectively. NaN values will be
  mapped to the middle of the range (.5).

  This is a non-linear approach to normalization that is less sensitive to
  outliers than min-max or z-score scaling. When outliers are present, standard
  forms of normalization can leave the majority of the data compressed into a
  very small segment of the output range, whereas this approach tends to spread
  out the more frequent values (if quantile buckets are used). Note that
  distance relationships in the raw data are not necessarily preserved (data
  points that close to each other in the raw feature space may not be equally
  close in the transformed feature space). This means that unlike linear
  normalization methods, correlations between features may be distorted by the
  transformation. This scaling method may help with stability and minimize
  exploding gradients in neural networks.

  Args:
    x: A numeric input `Tensor`/`CompositeTensor` (tf.float[32|64],
      tf.int[32|64]).
    bucket_boundaries: Sorted bucket boundaries as a rank-2 `Tensor` or list.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `CompositeTensor` of the same shape as `x`, normalized to the
      range [0, 1]. If the input x is tf.float64, the returned values will be
      tf.float64. Otherwise, returned values are tf.float32.

  """
  with tf.compat.v1.name_scope(name, 'buckets_with_interpolation'):
    bucket_boundaries = tf.convert_to_tensor(bucket_boundaries)
    tf.compat.v1.assert_rank(bucket_boundaries, 2)
    x_values = tf_utils.get_values(x)
    compose_result_fn = _make_composite_tensor_wrapper_if_composite(x)
    if not (x_values.dtype.is_floating or x_values.dtype.is_integer):
      raise ValueError(
          'Input tensor to be normalized must be numeric, got {}.'.format(
              x_values.dtype))
    # Remove any non-finite boundaries.
    if bucket_boundaries.dtype in (tf.float64, tf.float32):
      bucket_boundaries = tf.expand_dims(
          tf.gather_nd(bucket_boundaries,
                       tf.where(tf.math.is_finite(bucket_boundaries))),
          axis=0)
    return_type = tf.float64 if x.dtype == tf.float64 else tf.float32
    num_boundaries = tf.cast(
        tf.shape(bucket_boundaries)[1], dtype=tf.int64, name='num_boundaries')
    assert_some_finite_boundaries = tf.compat.v1.assert_greater(
        num_boundaries,
        tf.constant(0, tf.int64),
        name='assert_1_or_more_finite_boundaries')
    with tf.control_dependencies([assert_some_finite_boundaries]):
      bucket_indices = tf_utils.assign_buckets(
          x_values, bucket_boundaries, side=tf_utils.Side.RIGHT)
      # Get max, min, and width of the corresponding bucket for each element.
      bucket_max = tf.cast(
          tf.gather(
              tf.concat([bucket_boundaries[0], bucket_boundaries[:, -1]],
                        axis=0), bucket_indices), return_type)
      bucket_min = tf.cast(
          tf.gather(
              tf.concat([bucket_boundaries[:, 0], bucket_boundaries[0]],
                        axis=0), bucket_indices), return_type)
    bucket_width = bucket_max - bucket_min
    zeros = tf.zeros_like(x_values, dtype=return_type)
    ones = tf.ones_like(x_values, dtype=return_type)

    # Linearly interpolate each value within its respective bucket range.
    interpolation_value = (
        (tf.cast(x_values, return_type) - bucket_min) / bucket_width)
    bucket_interpolation = tf.compat.v1.verify_tensor_all_finite(
        tf.where(
            # If bucket index is first or last, which represents "less than
            # min" and "greater than max" respectively, the bucket logically
            # has an infinite width and we can't meaningfully interpolate.
            tf.logical_or(
                tf.equal(bucket_indices, 0),
                tf.equal(bucket_indices, num_boundaries)),
            zeros,
            tf.where(
                # If the bucket width is zero due to numerical imprecision,
                # there is no point in interpolating
                tf.equal(bucket_width, 0.0),
                ones / 2.0,
                # Finally, for a bucket with a valid width, we can interpolate.
                interpolation_value)),
        'bucket_interpolation')
    bucket_indices_with_interpolation = tf.cast(
        tf.maximum(bucket_indices - 1, 0), return_type) + bucket_interpolation

    # Normalize the interpolated values to the range [0, 1].
    denominator = tf.cast(tf.maximum(num_boundaries - 1, 1), return_type)
    normalized_values = bucket_indices_with_interpolation / denominator
    if x_values.dtype.is_floating:
      # Impute NaNs with .5, the middle value of the normalized output range.
      imputed_values = tf.ones_like(x_values, dtype=return_type) / 2.0
      normalized_values = tf.where(
          tf.math.is_nan(x_values), imputed_values, normalized_values)
    # If there is only one boundary, all values < the boundary are 0, all values
    # >= the boundary are 1.
    single_boundary_values = lambda: tf.where(  # pylint: disable=g-long-lambda
        tf.equal(bucket_indices, 0), zeros, ones)
    normalized_result = tf.cond(
        tf.equal(num_boundaries, 1),
        single_boundary_values, lambda: normalized_values)
    return compose_result_fn(normalized_result)


@common.log_api_use(common.MAPPER_COLLECTION)
def apply_buckets(
    x: common_types.ConsistentTensorType,
    bucket_boundaries: common_types.BucketBoundariesType,
    name: Optional[str] = None) -> common_types.ConsistentTensorType:
  """Returns a bucketized column, with a bucket index assigned to each input.

  Each element `e` in `x` is mapped to a positive index `i` for which
  `bucket_boundaries[i-1] <= e < bucket_boundaries[i]`, if it exists.
  If `e < bucket_boundaries[0]`, then `e` is mapped to `0`. If
  `e >= bucket_boundaries[-1]`, then `e` is mapped to `len(bucket_boundaries)`.
  NaNs are mapped to `len(bucket_boundaries)`.

  Example:

  >>> x = tf.constant([[4.0, float('nan'), 1.0], [float('-inf'), 7.5, 10.0]])
  >>> bucket_boundaries = tf.constant([[2.0, 5.0, 10.0]])
  >>> tft.apply_buckets(x, bucket_boundaries)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[1, 3, 0],
         [0, 2, 3]])>

  Args:
    x: A numeric input `Tensor` or `CompositeTensor` whose values should be
        mapped to buckets.  For `CompositeTensor`s, the non-missing values will
        be mapped to buckets and missing value left missing.
    bucket_boundaries: A rank 2 `Tensor` or list representing the bucket
        boundaries sorted in ascending order.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` or `CompositeTensor` of the same shape as `x`, with each element
    in the returned tensor representing the bucketized value. Bucketized value
    is in the range [0, len(bucket_boundaries)].
  """
  with tf.compat.v1.name_scope(name, 'apply_buckets'):
    bucket_boundaries = tf.convert_to_tensor(bucket_boundaries)
    tf.compat.v1.assert_rank(bucket_boundaries, 2)

    bucketized_values = tf_utils.assign_buckets(
        tf_utils.get_values(x), bucket_boundaries, side=tf_utils.Side.RIGHT)

    # Attach the relevant metadata to result, so that the corresponding
    # output feature will have this metadata set.
    min_value = tf.constant(0, tf.int64)
    max_value = tf.shape(input=bucket_boundaries)[1]
    schema_inference.set_tensor_schema_override(
        bucketized_values, min_value, max_value)
    _annotate_buckets(bucketized_values, bucket_boundaries)
    compose_result_fn = _make_composite_tensor_wrapper_if_composite(x)
    return compose_result_fn(bucketized_values)


def _annotate_buckets(x: tf.Tensor, bucket_boundaries: tf.Tensor) -> None:
  """Annotates a bucketized tensor with the boundaries that were applied.

  Creates a deferred annotation for the specified tensor.

  Args:
    x: The tensor to annotate.
    bucket_boundaries: A tensor of boundaries that were used to bucketize x.
  """
  # The annotations proto currently isn't available in OSS builds, so schema
  # annotations are not supported.
  if not common.IS_ANNOTATIONS_PB_AVAILABLE:
    return
  from tensorflow_transform import annotations_pb2  # pylint: disable=g-import-not-at-top
  message_type = annotations_pb2.BucketBoundaries.DESCRIPTOR.full_name

  # The BucketBoundaries annotation expects a float field.
  bucket_boundaries = tf.cast(bucket_boundaries, tf.float32)
  # Some callers provide rank 2 boundaries like [[.25], [.5], [.75], [1.]],
  # whereas we expect rank 2 boundaries like [[.25, .5, .75, 1.]]
  bucket_boundaries = tf.reshape(bucket_boundaries, [-1])
  bucket_boundaries = tf.expand_dims(bucket_boundaries, 0)
  size = (tf.shape(bucket_boundaries)[1],)
  message_proto = tf.raw_ops.EncodeProto(sizes=[size],
                                         values=[bucket_boundaries],
                                         field_names=['boundaries'],
                                         message_type=message_type)
  assert message_proto.shape == [1]
  message_proto = message_proto[0]

  type_url = os.path.join(common.ANNOTATION_PREFIX_URL, message_type)
  schema_inference.annotate(type_url, message_proto, tensor=x)


@common.log_api_use(common.MAPPER_COLLECTION)
def estimated_probability_density(x: tf.Tensor,
                                  boundaries: Optional[Union[tf.Tensor,
                                                             int]] = None,
                                  categorical: bool = False,
                                  name: Optional[str] = None) -> tf.Tensor:
  """Computes an approximate probability density at each x, given the bins.

  Using this type of fixed-interval method has several benefits compared to
    bucketization, although may not always be preferred.
    1. Quantiles does not work on categorical data.
    2. The quantiles algorithm does not currently operate on multiple features
    jointly, only independently.

  Ex: Outlier detection in a multi-modal or arbitrary distribution.
    Imagine a value x where a simple model is highly predictive of a target y
    within certain densely populated ranges. Outside these ranges, we may want
    to treat the data differently, but there are too few samples for the model
    to detect them by case-by-case treatment.
    One option would be to use the density estimate for this purpose:

    outputs['x_density'] = tft.estimated_prob(inputs['x'], bins=100)
    outputs['outlier_x'] = tf.where(outputs['x_density'] < OUTLIER_THRESHOLD,
                                    tf.constant([1]), tf.constant([0]))

    This exercise uses a single variable for illustration, but a direct density
    metric would become more useful with higher dimensions.

  Note that we normalize by average bin_width to arrive at a probability density
  estimate. The result resembles a pdf, not the probability that a value falls
  in the bucket (except in the categorical case).

  Args:
    x: A `Tensor`.
    boundaries: (Optional) A `Tensor` or int used to approximate the density.
        If possible provide boundaries as a Tensor of multiple sorted values.
        Will default to 10 intervals over the 0-1 range, or find the min/max
        if an int is provided (not recommended because multi-phase analysis is
        inefficient). If the boundaries are known as potentially arbitrary
        interval boundaries, sizes are assumed to be equal. If the sizes are
        unequal, density may be inaccurate. Ignored if `categorical` is true.
    categorical: (Optional) A `bool` that will treat x as categorical if true.
    name: (Optional) A name for this operation.

  Returns:
    A `Tensor` the same shape as x, the probability density estimate at x (or
    probability mass estimate if `categorical` is True).

  Raises:
    NotImplementedError: If `x` is CompositeTensor.
  """
  with tf.compat.v1.name_scope(name, 'estimated_probability_density'):
    if isinstance(x, (tf.SparseTensor, tf.RaggedTensor)):
      raise NotImplementedError(
          'estimated probability density does not support Composite Tensors')
    if x.get_shape().ndims > 1 and x.shape[-1] > 1:
      raise NotImplementedError(
          'estimated probability density does not support multiple dimensions')

    counts, boundaries = analyzers.histogram(x, boundaries=boundaries,
                                             categorical=categorical)

    xdims = x.get_shape().ndims
    counts = tf.cast(counts, tf.float32)
    probabilities = counts / tf.reduce_sum(counts)

    x = tf.reshape(x, [-1])

    if categorical:
      bucket_indices = tf_utils.lookup_key(x, boundaries)
      bucket_densities = probabilities
    else:
      # We need to compute the bin width so that density does not depend on
      # number of intervals.
      bin_width = tf.cast(boundaries[0, -1] - boundaries[0, 0], tf.float32) / (
          tf.cast(tf.size(probabilities), tf.float32))
      bucket_densities = probabilities / bin_width

      bucket_indices = tf_utils.assign_buckets(
          tf.cast(x, tf.float32),
          analyzers.remove_leftmost_boundary(boundaries))
    bucket_indices = tf_utils._align_dims(bucket_indices, xdims)  # pylint: disable=protected-access

    # In the categorical case, when keys are missing, the indices may be -1,
    # therefore we replace those with 0 in order to use tf.gather.
    adjusted_bucket_indices = tf.where(
        bucket_indices < 0, _fill_shape(0, tf.shape(bucket_indices), tf.int64),
        bucket_indices)
    bucket_densities = tf.gather(bucket_densities, adjusted_bucket_indices)
    return tf.where(bucket_indices < 0,
                    _fill_shape(0, tf.shape(bucket_indices), tf.float32),
                    bucket_densities)
