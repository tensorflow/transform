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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import itertools

# GOOGLE-INITIALIZATION

from absl.testing import parameterized
from builtins import zip  # pylint: disable=redefined-builtin

import numpy as np
import six
import tensorflow as tf

main = tf.test.main

named_parameters = parameterized.named_parameters


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


def function_handler(input_specs=None):
  """Run the given function with one of several possible modes and specs.

  We want to test functions in at least these three modes:
  1. In 1.x graph mode, utilizing placeholders
  2. In 2.x graph mode
  3. In 2.x eager mode, utilizing tf.function

  Since placeholders/sessions do not exist in their traditional sense in TF 2.0,
  we need to break out how we manage the inputs here. In practice, that means we
  assign placeholders with the same properties as each input_spec in 1.x, set up
  a tf.function with those input_specs in eager mode.

  Args:
    input_specs: A list of (str, `tf.TensorSpec`) tuples that specify input to
      fn, along with the name of each arg. Must be in correct order.

  Returns:
    A wrapper function that accepts arguments specified by *input_specs.

  Note: using tf.function requires the inputs to be in correct order, as
  input_signature can only be a list.
  """
  def wrapper(fn):
    """Runs either in eager mode with constants or a graph with placeholders."""

    def _map_tf_constant(*inputs):
      constant_inputs = []
      assert len(inputs) == len(input_specs)
      for value, spec in zip(inputs, input_specs):

        constant_input = tf.constant(value, dtype=spec[1].dtype)
        constant_input.shape.assert_is_compatible_with(spec[1].shape)
        constant_inputs.append(constant_input)

      return fn(*constant_inputs)

    if tf.executing_eagerly():
      return _map_tf_constant

    else:
      placeholders = {}
      for input_name, input_spec in input_specs:
        placeholders[input_name] = tf.compat.v1.placeholder(
            shape=input_spec.shape, dtype=input_spec.dtype, name=input_name)

      def _session_function(*feed_list):
        assert len(input_specs) == len(feed_list)
        result = fn(**placeholders)
        with tf.compat.v1.Session() as sess:
          return sess.run(
              result,
              feed_dict=dict((placeholders[input_name], value) for (
                  (input_name, _), value) in zip(input_specs, feed_list)))

      return _session_function

  return wrapper


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
    a_data, b_data = self._SortedData(a_data), self._SortedData(b_data)
    self.assertEqual(
        len(a_data), len(b_data), 'len(%r) != len(%r)' % (a_data, b_data))
    for i, (a_row, b_row) in enumerate(zip(a_data, b_data)):
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

  def _assertValuesCloseOrEqual(self, a_value, b_value, msg=None):
    try:
      if (isinstance(a_value, (six.binary_type, six.text_type)) or
          isinstance(a_value, list) and a_value and
          isinstance(a_value[0], (six.binary_type, six.text_type)) or
          isinstance(a_value, np.ndarray) and a_value.dtype == np.object):
        self.assertAllEqual(a_value, b_value)
      else:
        self.assertAllClose(a_value, b_value)
    except (AssertionError, TypeError) as e:
      if msg:
        e.args = ((e.args[0] + ' : ' + msg,) + e.args[1:])
      raise

  def AssertVocabularyContents(self, vocab_file_path, file_contents):
    with tf.io.gfile.GFile(vocab_file_path, 'rb') as f:
      file_lines = f.readlines()

      # Store frequency case.
      if isinstance(file_contents[0], tuple):
        word_and_frequency_list = []
        for content in file_lines:
          frequency, word = content.split(b' ', 1)
          word_and_frequency_list.append(
              (word.strip(b'\n'), float(frequency.strip(b'\n'))))
        expected_words, expected_frequency = zip(*word_and_frequency_list)
        actual_words, actual_frequency = zip(*file_contents)
        self.assertAllEqual(expected_words, actual_words)
        np.testing.assert_almost_equal(expected_frequency, actual_frequency)
      else:
        file_lines = [content.strip(b'\n') for content in file_lines]
        self.assertAllEqual(file_lines, file_contents)

  def WriteRenderedDotFile(self, dot_string, output_file=None):
    tf.compat.v1.logging.info(
        'Writing a rendered dot file is not yet supported.')

  def _NumpyArraysToLists(self, maybe_arrays):
    return [
        x.tolist() if isinstance(x, np.ndarray) else x for x in maybe_arrays]

  def _SortedDicts(self, list_of_dicts):
    # Sorts dicts by their unordered (key, value) pairs.
    return sorted(list_of_dicts, key=lambda d: sorted(d.items()))

  def _SortedData(self, list_of_dicts_of_arrays):
    list_of_values = [
        self._NumpyArraysToLists(d.values()) for d in list_of_dicts_of_arrays
    ]
    list_of_keys = [d.keys() for d in list_of_dicts_of_arrays]
    unsorted_dict_list = [
        dict(zip(a, b)) for a, b in zip(list_of_keys, list_of_values)
    ]
    return self._SortedDicts(unsorted_dict_list)
