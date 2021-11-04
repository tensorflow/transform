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
"""Library for Tensorflow Transform test cases."""

from builtins import zip  # pylint: disable=redefined-builtin,g-importing-member
import functools
import inspect
import itertools
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import unittest
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import tf2
from tensorflow.python.eager import context
# pylint: enable=g-direct-tensorflow-import

main = tf.test.main

named_parameters = parameterized.named_parameters
SkipTest = unittest.SkipTest


def is_tf_api_version_1():
  return hasattr(tf, 'Session')


def cross_named_parameters(*args):
  """Cross a list of lists of dicts suitable for @named_parameters.

  Takes a list of lists, where each list is suitable as an input to
  @named_parameters, and crosses them, forming a new name for each crossed test
  case.

  Args:
    *args: A list of lists of dicts.

  Returns:
    A list of dicts.
  """
  def _cross_test_cases(parameters_list):
    """Cross a list of test case parameters."""
    crossed_parameters = parameters_list[0].copy()
    for current_parameters in parameters_list[1:]:
      for name, value in current_parameters.items():
        if name == 'testcase_name':
          crossed_parameters[name] = '{}_{}'.format(
              crossed_parameters[name], value)
        else:
          assert name not in crossed_parameters, name
          crossed_parameters[name] = value
    return crossed_parameters
  return list(map(_cross_test_cases, itertools.product(*args)))


def parameters(*testcases):
  """like parameterized.parameters but tests show arg names.

  Only works for class methods without *args or **kwargs.

  Args:
    *testcases: The input to parameterized.parameters().

  Returns:
    A wrapper function which passes the arguments through as a dictionary.
  """

  def wrapper(fn):
    """Constructs and returns the arguments as a dictionary."""
    arg_names = inspect.getargspec(fn).args
    if arg_names[0] != 'self':
      raise ValueError(
          'First argument to test is expected to be "self", but is {}'.format(
              arg_names[0]))
    arg_names = arg_names[1:]

    def to_arg_dict(testcase):
      testcase = tuple(testcase)
      if len(testcase) != len(arg_names):
        raise ValueError(
            'The number of arguments to parameterized test do not match the '
            'number of expected arguments: {} != {}, arguments: {}, names: {}'.
            format(len(testcase), len(arg_names), testcase, arg_names))
      return dict(zip(arg_names, testcase))

    testcases_with_names = [to_arg_dict(testcase) for testcase in testcases]
    return parameterized.parameters(*testcases_with_names)(fn)

  return wrapper


def cross_parameters(*args):
  """Cross a sequence of sequences of parameters suitable for @parameters."""
  for p in itertools.product(*args):
    yield functools.reduce(lambda x, y: x + y, p)


def _make_placeholder(tensor_spec):
  """Create a placeholder for the given tensor_spec."""

  if isinstance(tensor_spec, tf.SparseTensorSpec):
    return tf.compat.v1.sparse_placeholder(
        shape=tensor_spec.shape, dtype=tensor_spec.dtype)
  if isinstance(tensor_spec, tf.RaggedTensorSpec):
    # TODO(b/160294509): Switch to public APIs once TF 1 support is dropped.
    return tf.compat.v1.ragged.placeholder(
        tensor_spec._dtype, tensor_spec._ragged_rank, value_shape=())  # pylint: disable=protected-access
  else:
    return tf.compat.v1.placeholder(
        shape=tensor_spec.shape, dtype=tensor_spec.dtype)


def _graph_function_handler(input_signature):
  """Run the given function in graph mode, utilizing placeholders.

  Args:
    input_signature: A possibly nested sequence of `tf.TensorSpec` objects
      specifying the shapes and dtypes of the Tensors that will be supplied to
      this function.

  Returns:
    A wrapper function that accepts arguments specified by `input_signature`.
  """
  def wrapper(fn):
    """Decorator that runs decorated function in graph mode."""
    def _run_graph(*inputs):
      with context.graph_mode():  # pylint: disable=missing-docstring
        assert len(input_signature) == len(inputs)
        placeholders = list(map(_make_placeholder, input_signature))
        output_tensor = fn(*placeholders)
        with tf.compat.v1.Session() as sess:
          sess.run(tf.compat.v1.tables_initializer())
          return sess.run(output_tensor,
                          feed_dict=dict(zip(placeholders, inputs)))

    return _run_graph
  return wrapper


def _ragged_value_as_constant(value, dtype):
  if isinstance(value, tf.compat.v1.ragged.RaggedTensorValue):
    return tf.RaggedTensor.from_row_splits(
        values=_ragged_value_as_constant(value.values, dtype),
        row_splits=tf.constant(value.row_splits, dtype=tf.int64))
  else:
    return tf.constant(value, dtype=dtype)


def _wrap_as_constant(value, tensor_spec):
  """Wrap a value as a constant, using tensor_spec for shape and type info."""
  if isinstance(tensor_spec, tf.SparseTensorSpec):
    result = tf.SparseTensor(
        indices=tf.constant(value.indices, dtype=tf.int64),
        values=tf.constant(value.values, dtype=tensor_spec.dtype),
        dense_shape=tf.constant(value.dense_shape, dtype=tf.int64))
  elif isinstance(tensor_spec, tf.RaggedTensorSpec):
    # TODO(b/160294509): Switch to public APIs once TF 1 support is dropped.
    result = _ragged_value_as_constant(value, tensor_spec._dtype)  # pylint: disable=protected-access
  else:
    result = tf.constant(value, dtype=tensor_spec.dtype)
    result.shape.assert_is_compatible_with(tensor_spec.shape)
  return result


def _eager_function_handler(input_signature):
  """Run the given function in eager mode.

  Args:
    input_signature: A possibly nested sequence of `tf.TensorSpec` objects
      specifying the shapes and dtypes of the Tensors that will be supplied to
      this function.

  Returns:
    A wrapper function that accepts arguments specified by `input_signature`.
  """
  def wrapper(fn):
    """Decorator that runs decorated function in eager mode."""
    def _run_eagerly(*inputs):  # pylint: disable=missing-docstring
      with context.eager_mode():
        constants = [_wrap_as_constant(value, tensor_spec)
                     for value, tensor_spec in zip(inputs, input_signature)]
        output = fn(*constants)
        if hasattr(output, '_make'):
          return output._make([np.asarray(tensor) for tensor in output])
        if isinstance(output, (tuple, list)):
          return [
              tensor.to_list()
              if isinstance(tensor, tf.RaggedTensor) else np.asarray(tensor)
              for tensor in output
          ]
        elif isinstance(output, tf.RaggedTensor):
          return output.to_list()
        else:
          return np.asarray(output)

    return _run_eagerly
  return wrapper


def _tf_function_function_handler(input_signature):
  """Call function in eager mode, but also wrapped in `tf.function`."""
  def wrapper(fn):
    wrapped_fn = tf.function(fn, input_signature)
    return _eager_function_handler(input_signature)(wrapped_fn)
  return wrapper


FUNCTION_HANDLERS = [
    dict(testcase_name='graph',
         function_handler=_graph_function_handler),
    dict(testcase_name='eager',
         function_handler=_eager_function_handler),
    dict(testcase_name='tf_function',
         function_handler=_tf_function_function_handler)
]


def is_external_environment():
  return not os.environ.get('TEST_WORKSPACE', '').startswith('google')


def skip_if_external_environment(reason):
  if is_external_environment():
    raise unittest.SkipTest(reason)


def skip_if_not_tf2(reason):
  major, _, _ = tf.version.VERSION.split('.')
  if not (int(major) >= 2 and tf2.enabled()) or is_tf_api_version_1():
    raise unittest.SkipTest(reason)


def cross_with_function_handlers(parameters_list):
  """Cross named parameters with all function handlers.

  Takes a list of parameters suitable as an input to @named_parameters,
  and crosses it with the set of function handlers.
  A parameterized test function that uses this should have a parameter named
  `function_handler`.

  Args:
    parameters_list: A list of dicts.

  Returns:
    A list of dicts.
  """
  return cross_named_parameters(parameters_list, FUNCTION_HANDLERS)


class TransformTestCase(parameterized.TestCase, tf.test.TestCase):
  """Base test class for testing tf-transform code."""

  # Display context for failing rows in data assertions.
  longMessage = True  # pylint: disable=invalid-name

  def assertDataCloseOrEqual(self, a_data, b_data):
    """Assert two datasets contain nearly equal values.

    Args:
      a_data: a sequence of dicts whose values are
              either strings, lists of strings, numeric types or a pair of
              those.
      b_data: same types as a_data

    Raises:
      AssertionError: if the two datasets are not the same.
    """
    msg = ''
    try:
      sorted_a, sorted_b = self._SortedData(a_data), self._SortedData(b_data)
      self.assertEqual(
          len(sorted_a), len(sorted_b), 'len(%r) != len(%r)' % (a_data, b_data))
      for i, (a_row, b_row) in enumerate(zip(sorted_a, sorted_b)):
        self.assertCountEqual(a_row.keys(), b_row.keys(), msg='Row %d' % i)
        for key in a_row.keys():
          a_value = a_row[key]
          b_value = b_row[key]
          msg = 'Row %d, key %s' % (i, key)
          if isinstance(a_value, tuple):
            self._assertValuesCloseOrEqual(a_value[0], b_value[0], msg=msg)
            self._assertValuesCloseOrEqual(a_value[1], b_value[1], msg=msg)
          else:
            self._assertValuesCloseOrEqual(a_value, b_value, msg=msg)
    except (AssertionError, TypeError) as e:
      message = '{}\nCompared:\n{}\nvs.\n{}'.format(msg, a_data, b_data)
      e.args = ((e.args[0] + ' : ' + message,) + e.args[1:])
      raise e

  def _assertValuesCloseOrEqual(self, a_value, b_value, msg=None):
    if (isinstance(a_value, (bytes, str)) or isinstance(a_value, list) and
        a_value and isinstance(a_value[0], (bytes, str)) or
        isinstance(a_value, np.ndarray) and a_value.dtype == np.object):
      self.assertAllEqual(a_value, b_value)
    else:
      # TODO(varshaan): Change atol only for tests for which 1e-6 is too strict.
      self.assertAllClose(a_value, b_value, atol=1e-5)

  def AssertVocabularyContents(self, vocab_file_path, file_contents):
    if vocab_file_path.endswith('.tfrecord.gz'):
      file_lines = list(
          tf.data.TFRecordDataset(vocab_file_path,
                                  compression_type='GZIP').as_numpy_iterator())
    else:
      with tf.io.gfile.GFile(vocab_file_path, 'rb') as f:
        file_lines = f.read().splitlines()

    # Store frequency case.
    if isinstance(file_contents[0], tuple):
      word_and_frequency_list = []
      for content in file_lines:
        frequency, word = content.split(b' ', 1)
        # Split by comma for when the vocabulary file stores the result of
        # per-key analyzers.
        values = list(map(float, frequency.split(b',')))
        word_and_frequency_list.append(
            (word, values[0] if len(values) == 1 else values))

      expected_words, expected_frequency = zip(*word_and_frequency_list)
      actual_words, actual_frequency = zip(*file_contents)
      self.assertAllEqual(expected_words, actual_words)
      np.testing.assert_almost_equal(
          expected_frequency, actual_frequency, decimal=6)
    else:
      self.assertAllEqual(file_contents, file_lines)

  def WriteRenderedDotFile(self, dot_string, output_file=None):
    tf.compat.v1.logging.info(
        'Writing a rendered dot file is not yet supported.')

  def _NumpyArraysToLists(self, maybe_arrays):
    return [
        x.tolist() if isinstance(x, np.ndarray) else x for x in maybe_arrays]

  def _SortedDicts(self, list_of_dicts):
    # Sorts dicts by their unordered (key, value) pairs. We use string ordering
    # to ensure consistent comparison of NaNs with numbers.
    return sorted(list_of_dicts, key=lambda d: str(sorted(d.items())))

  def _SortedData(self, list_of_dicts_of_arrays):
    list_of_values = [
        self._NumpyArraysToLists(d.values()) for d in list_of_dicts_of_arrays
    ]
    list_of_keys = [d.keys() for d in list_of_dicts_of_arrays]
    unsorted_dict_list = [
        dict(zip(a, b)) for a, b in zip(list_of_keys, list_of_values)
    ]
    return self._SortedDicts(unsorted_dict_list)
