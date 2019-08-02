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
"""Tests for tensorflow_transform.graph_tools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

# GOOGLE-INITIALIZATION

import six

import tensorflow as tf
from tensorflow_transform import graph_tools
from tensorflow_transform import test_case

from tensorflow.python.ops import control_flow_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import

mock = tf.test.mock


def _create_lookup_table_from_file(filename):
  initializer = tf.lookup.TextFileInitializer(
      filename,
      key_dtype=tf.string,
      key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
      value_dtype=tf.int64,
      value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  return tf.lookup.StaticHashTable(initializer, default_value=-1)


def _create_graph_with_y_function_of_x():
  x = tf.compat.v1.placeholder(tf.int64)
  y = x + 1
  return {'x': x, 'y': y}


def _create_graph_with_y_function_of_x_with_unused_inputs():
  x = tf.compat.v1.placeholder(tf.int64)
  x2 = tf.compat.v1.placeholder(tf.int64)
  x_unused = tf.compat.v1.placeholder(tf.int64)
  y = x + 1
  z = x2 + 2
  return {'x': x, 'x2': x2, 'x_unused': x_unused, 'y': y, 'z': z}


def _create_graph_with_y_function_of_x_sparse():
  x = tf.compat.v1.sparse_placeholder(tf.int64)
  y = tf.sparse.reduce_sum(x) + 1
  return {'x': x, 'y': y}


def _create_graph_with_y_sparse_function_of_x_sparse():
  x = tf.compat.v1.sparse_placeholder(tf.int64)
  y = tf.SparseTensor(
      indices=x.indices,
      values=x.values + 1,
      dense_shape=x.dense_shape)
  return {
      'x': x,
      'y': y,
      'z': tf.sparse.add(y, tf.ones(y.dense_shape, tf.int64))
  }


def _create_graph_with_y_function_of_x_and_table():
  filename = tf.compat.v1.placeholder(tf.string, ())
  table = _create_lookup_table_from_file(filename)
  x = tf.compat.v1.placeholder(tf.string, (None,))
  y = table.lookup(x)
  return {'filename': filename, 'x': x, 'y': y}


def _create_graph_with_y_function_of_x_and_table_in_first_phase():
  table = _create_lookup_table_from_file('not_a_file_name_but_ok')
  x = tf.compat.v1.placeholder(tf.string, (None,))
  y = table.lookup(x)
  return {'x': x, 'y': y}


def _create_graph_with_y_function_of_x_and_untracked_table():
  filename = tf.compat.v1.placeholder(tf.string, ())
  table = _create_lookup_table_from_file(filename)

  x = tf.compat.v1.placeholder(tf.string, (None,))
  y = table.lookup(x)
  del tf.compat.v1.get_collection_ref(
      tf.compat.v1.GraphKeys.TABLE_INITIALIZERS)[:]
  return {'filename': filename, 'x': x, 'y': y}


def _create_graph_with_table_initialized_by_table_output():
  filename = tf.compat.v1.placeholder(tf.string, ())
  table1 = _create_lookup_table_from_file(filename)

  # Use output from the first table to initialize the second table.
  keys = ['a', 'b', 'c']
  tensor_keys = tf.as_string(
      table1.lookup(tf.constant(keys, tf.string)))
  initializer2 = tf.lookup.KeyValueTensorInitializer(
      keys=tensor_keys,
      values=tf.range(len(keys), dtype=tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  table2 = tf.lookup.StaticHashTable(initializer2, default_value=-1)
  x = tf.compat.v1.placeholder(tf.string, (None,))
  y = table2.lookup(x)
  return {'filename': filename, 'x': x, 'y': y}


def _create_graph_with_assert_equal():
  x = tf.compat.v1.placeholder(tf.int64)
  y = tf.compat.v1.placeholder(tf.int64)
  z = control_flow_ops.with_dependencies([tf.compat.v1.assert_equal(x, y)], x)
  return {'x': x, 'y': y, 'z': z}


def _create_graph_with_y_function_of_x_with_tf_while():
  x = tf.compat.v1.placeholder(tf.int64, ())

  # Subtract 10 from x using a tf.while_loop.
  def stop_condition(counter, x_minus_counter):
    del x_minus_counter  # unused
    return tf.less(counter, 10)
  def iteration(counter, x_minus_counter):
    return tf.add(counter, 1), tf.add(x_minus_counter, -1)
  initial_values = [tf.constant(0), x]
  final_values = tf.while_loop(
      cond=stop_condition, body=iteration, loop_vars=initial_values)

  y = final_values[1]
  return {'x': x, 'y': y}


class _SubStrMatcher(collections.namedtuple('_EitherMatcher', ['sub'])):

  def __eq__(self, other):
    if self.sub in other:
      return True
    tf.logging.error('{} is not in {}'.format(self.sub, other))
    return False


class _Matcher(object):

  __metaclass__ = abc.ABCMeta

  def _future_proof(self, value):
    if isinstance(value, (six.text_type, str, bytes)):
      new_to_old = {}
      for new, old in new_to_old.items():
        value = value.replace(new, old)
    return value

  @abc.abstractmethod
  def expected_fields(self, other):
    raise NotImplementedError

  @abc.abstractproperty
  def expected_fields_values(self):
    raise NotImplementedError

  @abc.abstractproperty
  def expected_class(self):
    raise NotImplementedError

  def __eq__(self, other):
    if not isinstance(other, self.expected_class):
      tf.logging.error('Types do not match, got: %s, expected: %s', type(other),
                       self.expected_class)
      return False

    future_expected_fields = tuple(
        self._future_proof(f) for f in self.expected_fields_values)
    if (self.expected_fields_values != self.expected_fields(other) and
        future_expected_fields != self.expected_fields(other)):
      tf.logging.error('Fields do not match: %s != %s',
                       self.expected_fields_values, self.expected_fields(other))
      return False

    return True


class _TensorMatcher(_Matcher, collections.namedtuple('_TensorMatcher',
                                                      ['name'])):

  def expected_fields(self, other):
    return (str(other.name),)

  @property
  def expected_fields_values(self):
    return tuple(self)

  @property
  def expected_class(self):
    return tf.Tensor


class _OpMatcher(_Matcher, collections.namedtuple('_OpMatcher', ['name'])):

  def expected_fields(self, other):
    return (str(other.name),)

  @property
  def expected_fields_values(self):
    return tuple(self)

  @property
  def expected_class(self):
    return tf.Operation


class GraphToolsTest(test_case.TransformTestCase):

  @test_case.named_parameters(
      dict(
          testcase_name='y_function_of_x_nothing_ready',
          create_graph_fn=_create_graph_with_y_function_of_x,
          feeds=[],
          replaced_tensors_ready={'x': False},
          should_be_ready={'y': False},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_unused_input_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_with_unused_inputs,
          feeds=[],
          replaced_tensors_ready={
              'x': False,
              'x2': True,
              'x_unused': True
          },
          should_be_ready={
              'y': False,
              'z': True
          },
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_no_feeds_y_is_ready',
          create_graph_fn=_create_graph_with_y_function_of_x,
          feeds=[],
          replaced_tensors_ready={'x': True},
          should_be_ready={'y': True},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_feeds_x_y_is_ready',
          create_graph_fn=_create_graph_with_y_function_of_x,
          feeds=['x'],
          replaced_tensors_ready={},
          should_be_ready={'y': True},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_sparse_nothing_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_sparse,
          feeds=[],
          replaced_tensors_ready={'x': False},
          should_be_ready={'y': False},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_sparse_no_feeds_y_is_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_sparse,
          feeds=[],
          replaced_tensors_ready={'x': True},
          should_be_ready={'y': True},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_sparse_feeds_x_y_is_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_sparse,
          feeds=['x'],
          replaced_tensors_ready={},
          should_be_ready={'y': True},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_sparse_function_of_x_sparse_nothing_ready',
          create_graph_fn=_create_graph_with_y_sparse_function_of_x_sparse,
          feeds=[],
          replaced_tensors_ready={'x': False},
          should_be_ready={'y': False},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_sparse_function_of_x_sparse_no_feeds_y_is_ready',
          create_graph_fn=_create_graph_with_y_sparse_function_of_x_sparse,
          feeds=[],
          replaced_tensors_ready={'x': True},
          should_be_ready={'y': True},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_sparse_function_of_x_sparse_feeds_x_y_is_ready',
          create_graph_fn=_create_graph_with_y_sparse_function_of_x_sparse,
          feeds=['x'],
          replaced_tensors_ready={},
          should_be_ready={'y': True},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_with_tf_while_nothing_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_with_tf_while,
          feeds=[],
          replaced_tensors_ready={'x': False},
          should_be_ready={'y': False},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_with_tf_while_no_feeds_y_is_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_with_tf_while,
          feeds=[],
          replaced_tensors_ready={'x': True},
          should_be_ready={'y': True},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_with_tf_while_feeds_x_y_is_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_with_tf_while,
          feeds=['x'],
          replaced_tensors_ready={},
          should_be_ready={'y': True},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_and_table_nothing_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_and_table,
          feeds=[],
          replaced_tensors_ready={
              'x': False,
              'filename': False
          },
          should_be_ready={'y': False},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_and_table_filename_ready_y_is_not',
          create_graph_fn=_create_graph_with_y_function_of_x_and_table,
          feeds=[],
          replaced_tensors_ready={
              'x': False,
              'filename': True
          },
          should_be_ready={'y': False},
          num_ready_table_initializers=1),
      dict(
          testcase_name='y_function_of_x_and_table_x_ready_filename_is_not',
          create_graph_fn=_create_graph_with_y_function_of_x_and_table,
          feeds=[],
          replaced_tensors_ready={
              'x': True,
              'filename': False
          },
          should_be_ready={'y': False},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_and_table_everything_is_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_and_table,
          feeds=[],
          replaced_tensors_ready={
              'x': True,
              'filename': True
          },
          should_be_ready={'y': True},
          num_ready_table_initializers=1),
      dict(
          testcase_name='y_function_of_x_and_table_feeds_x_nothing_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_and_table,
          feeds=['x'],
          replaced_tensors_ready={'filename': False},
          should_be_ready={'y': False},
          num_ready_table_initializers=0),
      dict(
          testcase_name='y_function_of_x_and_table_feeds_x_everything_ready',
          create_graph_fn=_create_graph_with_y_function_of_x_and_table,
          feeds=['x'],
          replaced_tensors_ready={'filename': True},
          should_be_ready={'y': True},
          num_ready_table_initializers=1),
      dict(
          testcase_name='assert_equal',
          create_graph_fn=_create_graph_with_assert_equal,
          feeds=['x', 'y'],
          replaced_tensors_ready={},
          should_be_ready={'z': True},
          num_ready_table_initializers=0),
  )
  def testDetermineReadyTensorsAndTableInitializers(
      self, create_graph_fn, feeds, replaced_tensors_ready, should_be_ready,
      num_ready_table_initializers):
    """Test determine_ready_tensors_and_table_initializers.

    Args:
      create_graph_fn: A function that adds ops to a graph and returns a dict
          mapping tensor names to `Tensor` or `SparseTensor`s.
      feeds: A list of keys in the dict returned by create_graph_fn that are fed
          in the main run (but not table initialization run).
      replaced_tensors_ready: A dict whose keys are keys in the dict returned by
          create_graph_fn and values are a bools indicating whether that tensor
          is ready to be replaced in this phase.
      should_be_ready: A dict whose keys are keys in the dict returned by
          create_graph_fn and value are bools indicating whether a tensor can be
          calculated in this phase.
      num_ready_table_initializers: The number of table initializers that are
          ready to run in the table initialization run of this phase.
    """
    tensors = create_graph_fn()
    replaced_tensors_ready = {tensors[name]: ready
                              for name, ready in replaced_tensors_ready.items()}

    graph_analyzer = graph_tools.InitializableGraphAnalyzer(
        tf.compat.v1.get_default_graph(), {x: tensors[x] for x in feeds},
        replaced_tensors_ready)
    self.assertEqual(len(graph_analyzer.ready_table_initializers),
                     num_ready_table_initializers)

    for name, ready in should_be_ready.items():
      tensor = tensors[name]
      self.assertEqual(graph_analyzer.ready_to_run(tensor), ready)

  @test_case.parameters(
      (_create_graph_with_y_function_of_x_and_table,
       [], {'x': False},
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_y_function_of_x_and_table,
       [], {'x': True},
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_y_function_of_x_and_table,
       ['filename'], {'x': False},
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_y_function_of_x_and_table,
       ['filename'], {'x': True},
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_y_function_of_x_and_table,
       ['filename', 'x'], {},
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_table_initialized_by_table_output,
       ['x'], {'filename': True},
       'tables are initialized in one pass')
  )
  def testInitializableGraphAnalyzerConstructorRaises(
      self, create_graph_fn, feeds, replaced_tensors_ready,
      error_msg_regex):
    """Test determine_ready_tensors_and_table_initializers.

    Args:
      create_graph_fn: A function that adds ops to a graph and returns a dict
          mapping tensor names to `Tensor` or `SparseTensor`s.
      feeds: A list of keys in the dict returned by create_graph_fn that are fed
          in the main run (but not table initialization run).
      replaced_tensors_ready: A dict whose keys are keys in the dict returned by
          create_graph_fn and values are a bools indicating whether that tensor
          is ready to be replaced in this phase.
      error_msg_regex: The expected error message.
    """
    tensors = create_graph_fn()
    replaced_tensors_ready = {tensors[name]: ready
                              for name, ready in replaced_tensors_ready.items()}
    with self.assertRaisesRegexp(ValueError, error_msg_regex):
      graph_tools.InitializableGraphAnalyzer(tf.compat.v1.get_default_graph(),
                                             {x: tensors[x] for x in feeds},
                                             replaced_tensors_ready)

  @test_case.parameters(
      (_create_graph_with_y_function_of_x, [], {}, 'y',
       'may have be caused by manually adding a placeholder to the graph'),
      (_create_graph_with_y_function_of_x_and_untracked_table,
       ['x'], {'filename': True}, 'y',
       'may be caused by adding an initializable table without'),
  )
  def testInitializableGraphAnalyzerReadyToRunRaises(
      self, create_graph_fn, feeds, replaced_tensors_ready, fetch,
      error_msg_regex):
    """Test determine_ready_tensors_and_table_initializers.

    Args:
      create_graph_fn: A function that adds ops to a graph and returns a dict
          mapping tensor names to `Tensor` or `SparseTensor`s.
      feeds: A list of keys in the dict returned by create_graph_fn that are fed
          in the main run (but not table initialization run).
      replaced_tensors_ready: A dict whose keys are keys in the dict returned by
          create_graph_fn and values are a bools indicating whether that tensor
          is ready to be replaced in this phase.
      fetch: The tensor to fetch.  Should be a key in the dict returned by
          create_graph_fn.
      error_msg_regex: The expected error message.
    """
    tensors = create_graph_fn()
    replaced_tensors_ready = {tensors[name]: ready
                              for name, ready in replaced_tensors_ready.items()}
    graph_analyzer = graph_tools.InitializableGraphAnalyzer(
        tf.compat.v1.get_default_graph(), {x: tensors[x] for x in feeds},
        replaced_tensors_ready)
    with self.assertRaisesRegexp(ValueError, error_msg_regex):
      tensor = tensors[fetch]
      graph_analyzer.ready_to_run(tensor)

  @test_case.named_parameters(
      dict(
          testcase_name='y_function_of_x',
          create_graph_fn=_create_graph_with_y_function_of_x,
          feeds=['x'],
          fetches=['y'],
          expected_dependent_inputs=['x']),
      dict(
          testcase_name='y_function_of_x_with_unused_inputs',
          create_graph_fn=_create_graph_with_y_function_of_x_with_unused_inputs,
          feeds=['x', 'x2', 'x_unused'],
          fetches=['y', 'z'],
          expected_dependent_inputs=['x', 'x2']),
      dict(
          testcase_name='y_function_of_sparse_x',
          create_graph_fn=_create_graph_with_y_function_of_x_sparse,
          feeds=['x'],
          fetches=['y'],
          expected_dependent_inputs=['x']),
      dict(
          testcase_name='y_sparse_function_of_sparse_x',
          create_graph_fn=_create_graph_with_y_sparse_function_of_x_sparse,
          feeds=['x'],
          fetches=['y'],
          expected_dependent_inputs=['x']),
      dict(
          testcase_name='z_function_of_x_y_with_control_dependencies',
          create_graph_fn=_create_graph_with_assert_equal,
          feeds=['x', 'y'],
          fetches=['z'],
          expected_dependent_inputs=['x', 'y']),
      dict(
          testcase_name='y_function_of_x_with_tf_while',
          create_graph_fn=_create_graph_with_y_function_of_x_with_tf_while,
          feeds=['x'],
          fetches=['y'],
          expected_dependent_inputs=['x']),
  )
  def testGetDependentInputs(self, create_graph_fn, feeds, fetches,
                             expected_dependent_inputs):
    tensors = create_graph_fn()
    got = graph_tools.get_dependent_inputs(tf.compat.v1.get_default_graph(),
                                           {x: tensors[x] for x in feeds},
                                           {y: tensors[y] for y in fetches})
    self.assertCountEqual(expected_dependent_inputs, got.keys())
    for input_name in expected_dependent_inputs:
      self.assertEqual(tensors[input_name], got[input_name])


class GraphToolsTestUniquePath(test_case.TransformTestCase):

  @test_case.named_parameters(
      dict(
          testcase_name='y_function_of_x',
          create_graph_fn=_create_graph_with_y_function_of_x,
          feeds=['x'],
          replaced_tensors_ready={'x': False},
          expected_calls_dict={
              'x': [mock.call('x$tensor'),],
              'y': [
                  mock.call(_OpMatcher('add/y'), parents=[]),
                  mock.call(_TensorMatcher('add/y:0'), parents=[u'add/y']),
                  mock.call('x$tensor'),
                  mock.call(
                      _OpMatcher('add'), parents=['x$tensor', u'add/y:0']),
                  mock.call(_TensorMatcher('add:0'), parents=[u'add']),
              ]
          }),
      dict(
          testcase_name='y_function_of_x_sparse',
          create_graph_fn=_create_graph_with_y_function_of_x_sparse,
          feeds=['x'],
          replaced_tensors_ready={'x': False},
          expected_calls_dict={
              'y': [
                  mock.call(_OpMatcher('add/y'), parents=[]),
                  mock.call(_TensorMatcher('add/y:0'), parents=[u'add/y']),
                  mock.call(_OpMatcher('range/delta'), parents=[]),
                  mock.call(
                      _TensorMatcher('range/delta:0'),
                      parents=[u'range/delta']),
                  mock.call('x$dense_shape'),
                  mock.call(_OpMatcher('Rank'), parents=['x$dense_shape']),
                  mock.call(_TensorMatcher('Rank:0'), parents=[u'Rank']),
                  mock.call(_OpMatcher('range/start'), parents=[]),
                  mock.call(
                      _TensorMatcher('range/start:0'),
                      parents=[u'range/start']),
                  mock.call(
                      _OpMatcher('range'),
                      parents=[u'range/start:0', u'Rank:0', u'range/delta:0']),
                  mock.call(_TensorMatcher('range:0'), parents=[u'range']),
                  mock.call('x$values'),
                  mock.call('x$indices'),
                  mock.call(
                      _OpMatcher('SparseReduceSum'),
                      parents=[
                          'x$indices', 'x$values', 'x$dense_shape', u'range:0'
                      ]),
                  mock.call(
                      _TensorMatcher('SparseReduceSum:0'),
                      parents=[u'SparseReduceSum']),
                  mock.call(
                      _OpMatcher('add'),
                      parents=[u'SparseReduceSum:0', u'add/y:0']),
                  mock.call(_TensorMatcher('add:0'), parents=[u'add']),
              ]
          }),
      dict(
          testcase_name='y_sparse_function_of_x_sparse',
          create_graph_fn=_create_graph_with_y_sparse_function_of_x_sparse,
          feeds=['x'],
          replaced_tensors_ready={'x': False},
          expected_calls_dict={
              'z': [
                  mock.call(_OpMatcher('ones/Const'), parents=[]),
                  mock.call(
                      _TensorMatcher('ones/Const:0'), parents=[u'ones/Const']),
                  mock.call('x$dense_shape'),
                  mock.call(
                      _OpMatcher('ones'),
                      parents=['x$dense_shape', u'ones/Const:0']),
                  mock.call(_TensorMatcher('ones:0'), parents=[u'ones']),
                  mock.call(_OpMatcher('add/y'), parents=[]),
                  mock.call(_TensorMatcher('add/y:0'), parents=[u'add/y']),
                  mock.call('x$values'),
                  mock.call(
                      _OpMatcher('add'), parents=['x$values', u'add/y:0']),
                  mock.call(_TensorMatcher('add:0'), parents=[u'add']),
                  mock.call('x$indices'),
                  mock.call(
                      _OpMatcher('SparseTensorDenseAdd'),
                      parents=[
                          'x$indices', u'add:0', 'x$dense_shape', u'ones:0'
                      ]),
                  mock.call(
                      _TensorMatcher('SparseTensorDenseAdd:0'),
                      parents=[u'SparseTensorDenseAdd']),
              ],
          }),
      dict(
          testcase_name='y_function_of_x_with_tf_while',
          create_graph_fn=_create_graph_with_y_function_of_x_with_tf_while,
          feeds=['x'],
          replaced_tensors_ready={'x': False},
          expected_calls_dict={
              'y': [
                  mock.call(_TensorMatcher('while/Merge:0'), parents=[]),
                  mock.call(
                      _OpMatcher('while/Switch'), parents=[
                          u'while/Merge:0',
                      ]),
                  mock.call(
                      _TensorMatcher('while/Switch:1'),
                      parents=[u'while/Switch']),
                  mock.call(
                      _OpMatcher('while/Identity'),
                      parents=[u'while/Switch:1']),
                  mock.call(
                      _OpMatcher('while/Add/y'), parents=[u'while/Identity']),
                  mock.call(
                      _TensorMatcher('while/Add/y:0'),
                      parents=[u'while/Add/y']),
                  mock.call(
                      _TensorMatcher('while/Identity:0'),
                      parents=[u'while/Identity']),
                  mock.call(
                      _OpMatcher('while/Add'),
                      parents=[u'while/Identity:0', u'while/Add/y:0']),
                  mock.call(
                      _TensorMatcher('while/Add:0'), parents=[u'while/Add']),
                  mock.call(
                      _OpMatcher('while/NextIteration'),
                      parents=[u'while/Add:0']),
                  mock.call(
                      _TensorMatcher('while/NextIteration:0'),
                      parents=[u'while/NextIteration']),
                  mock.call(_OpMatcher('Const'), parents=[]),
                  mock.call(_TensorMatcher('Const:0'), parents=[u'Const']),
                  mock.call(_OpMatcher('while/Enter'), parents=[u'Const:0']),
                  mock.call(
                      _TensorMatcher('while/Enter:0'),
                      parents=[u'while/Enter']),
                  mock.call(
                      _OpMatcher('while/Merge'),
                      parents=[u'while/Enter:0', u'while/NextIteration:0']),
                  mock.call(
                      _OpMatcher('while/Less/y'), parents=[u'while/Merge']),
                  mock.call(
                      _TensorMatcher('while/Less/y:0'),
                      parents=[u'while/Less/y']),
                  mock.call(
                      _OpMatcher('while/Less'),
                      parents=[u'while/Merge:0', u'while/Less/y:0']),
                  mock.call(
                      _TensorMatcher('while/Less:0'), parents=[u'while/Less']),
                  mock.call(
                      _OpMatcher('while/LoopCond'), parents=[u'while/Less:0']),
                  mock.call(
                      _TensorMatcher('while/LoopCond:0'),
                      parents=[u'while/LoopCond']),
                  mock.call(
                      _OpMatcher('while/Add_1/y'), parents=[u'while/Identity']),
                  mock.call(
                      _TensorMatcher('while/Add_1/y:0'),
                      parents=[u'while/Add_1/y']),
                  mock.call(_TensorMatcher('while/Switch_1:1'), parents=[]),
                  mock.call(
                      _OpMatcher('while/Identity_1'),
                      parents=[u'while/Switch_1:1']),
                  mock.call(
                      _TensorMatcher('while/Identity_1:0'),
                      parents=[u'while/Identity_1']),
                  mock.call(
                      _OpMatcher('while/Add_1'),
                      parents=[u'while/Identity_1:0', u'while/Add_1/y:0']),
                  mock.call(
                      _TensorMatcher('while/Add_1:0'),
                      parents=[u'while/Add_1']),
                  mock.call(
                      _OpMatcher('while/NextIteration_1'),
                      parents=[u'while/Add_1:0']),
                  mock.call(
                      _TensorMatcher('while/NextIteration_1:0'),
                      parents=[u'while/NextIteration_1']),
                  mock.call('x$tensor'),
                  mock.call(_OpMatcher('while/Enter_1'), parents=['x$tensor']),
                  mock.call(
                      _TensorMatcher('while/Enter_1:0'),
                      parents=[u'while/Enter_1']),
                  mock.call(
                      _OpMatcher('while/Merge_1'),
                      parents=[u'while/Enter_1:0', u'while/NextIteration_1:0']),
                  mock.call(
                      _TensorMatcher('while/Merge_1:0'),
                      parents=[u'while/Merge_1']),
                  mock.call(
                      _OpMatcher('while/Switch_1'),
                      parents=[u'while/Merge_1:0', u'while/LoopCond:0']),
                  mock.call(
                      _TensorMatcher('while/Switch_1:0'),
                      parents=[u'while/Switch_1']),
                  mock.call(
                      _OpMatcher('while/Exit_1'),
                      parents=[u'while/Switch_1:0']),
                  mock.call(
                      _TensorMatcher('while/Exit_1:0'),
                      parents=[u'while/Exit_1']),
              ],
          }),
      dict(
          testcase_name='y_function_of_x_and_table',
          create_graph_fn=_create_graph_with_y_function_of_x_and_table_in_first_phase,
          feeds=['x'],
          replaced_tensors_ready={'x': False},
          expected_calls_dict={
              'x': [
                  mock.call(
                      _OpMatcher('asset_path'),
                      parents=[]),
                  mock.call(
                      _TensorMatcher('asset_path:0'),
                      parents=['asset_path']),
                  mock.call(
                      _OpMatcher('hash_table'),
                      parents=['asset_path:0']),
                  mock.call('x$tensor'),
              ],
              'y': [
                  mock.call(
                      _OpMatcher('asset_path'),
                      parents=[]),
                  mock.call(
                      _TensorMatcher('asset_path:0'),
                      parents=['asset_path']),
                  mock.call(
                      _OpMatcher('hash_table'),
                      parents=['asset_path:0']),
                  mock.call(
                      _OpMatcher('Const'),
                      parents=[]),
                  mock.call(
                      _TensorMatcher('Const:0'),
                      parents=['Const']),
                  mock.call('x$tensor'),
                  mock.call('hash_table'),
                  mock.call(
                      _TensorMatcher('hash_table:0'),
                      parents=['hash_table']),
                  mock.call(
                      _OpMatcher('hash_table_Lookup/LookupTableFindV2'),
                      parents=[
                          'hash_table:0', 'x$tensor', 'Const:0'
                      ]),
                  mock.call(
                      _TensorMatcher('hash_table_Lookup/LookupTableFindV2:0'),
                      parents=['hash_table_Lookup/LookupTableFindV2']),
              ],
          }),
      dict(
          testcase_name='with_assert_equal',
          create_graph_fn=_create_graph_with_assert_equal,
          feeds=['x', 'y'],
          replaced_tensors_ready={
              'x': False,
              'y': False
          },
          expected_calls_dict={
              'x': [mock.call('x$tensor'),],
              'y': [mock.call('y$tensor'),],
              'z': [
                  mock.call(_OpMatcher('assert_equal/range/delta'), parents=[]),
                  mock.call(
                      _TensorMatcher('assert_equal/range/delta:0'),
                      parents=[u'assert_equal/range/delta']),
                  mock.call('y$tensor'),
                  mock.call('x$tensor'),
                  mock.call(
                      _OpMatcher('assert_equal/Equal'),
                      parents=['x$tensor', 'y$tensor']),
                  mock.call(
                      _TensorMatcher('assert_equal/Equal:0'),
                      parents=[u'assert_equal/Equal']),
                  mock.call(
                      _OpMatcher('assert_equal/Rank'),
                      parents=[u'assert_equal/Equal:0']),
                  mock.call(
                      _TensorMatcher('assert_equal/Rank:0'),
                      parents=[u'assert_equal/Rank']),
                  mock.call(_OpMatcher('assert_equal/range/start'), parents=[]),
                  mock.call(
                      _TensorMatcher('assert_equal/range/start:0'),
                      parents=[u'assert_equal/range/start']),
                  mock.call(
                      _OpMatcher('assert_equal/range'),
                      parents=[
                          u'assert_equal/range/start:0', u'assert_equal/Rank:0',
                          u'assert_equal/range/delta:0'
                      ]),
                  mock.call(
                      _TensorMatcher('assert_equal/range:0'),
                      parents=[u'assert_equal/range']),
                  mock.call(
                      _OpMatcher('assert_equal/All'),
                      parents=[
                          u'assert_equal/Equal:0', u'assert_equal/range:0'
                      ]),
                  mock.call(
                      _TensorMatcher('assert_equal/All:0'),
                      parents=[u'assert_equal/All']),
                  mock.call(
                      _OpMatcher('assert_equal/Assert/AssertGuard/Switch'),
                      parents=[u'assert_equal/All:0', u'assert_equal/All:0']),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/Switch:1'),
                      parents=[u'assert_equal/Assert/AssertGuard/Switch']),
                  mock.call(
                      _OpMatcher('assert_equal/Assert/AssertGuard/switch_t'),
                      parents=[u'assert_equal/Assert/AssertGuard/Switch:1']),
                  mock.call(
                      _OpMatcher('assert_equal/Assert/AssertGuard/NoOp'),
                      parents=[u'assert_equal/Assert/AssertGuard/switch_t']),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/switch_t:0'),
                      parents=[u'assert_equal/Assert/AssertGuard/switch_t']),
                  mock.call(
                      _OpMatcher(
                          'assert_equal/Assert/AssertGuard/control_dependency'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/switch_t:0',
                          u'assert_equal/Assert/AssertGuard/NoOp'
                      ]),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/control_dependency:0'
                      ),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/control_dependency'
                      ]),
                  mock.call(
                      _OpMatcher('assert_equal/Assert/AssertGuard/pred_id'),
                      parents=[u'assert_equal/All:0']),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/pred_id:0'),
                      parents=[u'assert_equal/Assert/AssertGuard/pred_id']),
                  mock.call(
                      _OpMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/Switch_2'),
                      parents=[
                          'y$tensor',
                          u'assert_equal/Assert/AssertGuard/pred_id:0'
                      ]),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/Switch_2:0'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/Assert/Switch_2'
                      ]),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/Switch:0'),
                      parents=[u'assert_equal/Assert/AssertGuard/Switch']),
                  mock.call(
                      _OpMatcher('assert_equal/Assert/AssertGuard/switch_f'),
                      parents=[u'assert_equal/Assert/AssertGuard/Switch:0']),
                  mock.call(
                      _OpMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/data_4'),
                      parents=[u'assert_equal/Assert/AssertGuard/switch_f']),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/data_4:0'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/Assert/data_4'
                      ]),
                  mock.call(
                      _OpMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/Switch_1'),
                      parents=[
                          'x$tensor',
                          u'assert_equal/Assert/AssertGuard/pred_id:0'
                      ]),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/Switch_1:0'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/Assert/Switch_1'
                      ]),
                  mock.call(
                      _OpMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/data_2'),
                      parents=[u'assert_equal/Assert/AssertGuard/switch_f']),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/data_2:0'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/Assert/data_2'
                      ]),
                  mock.call(
                      _OpMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/data_1'),
                      parents=[u'assert_equal/Assert/AssertGuard/switch_f']),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/data_1:0'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/Assert/data_1'
                      ]),
                  mock.call(
                      _OpMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/data_0'),
                      parents=[u'assert_equal/Assert/AssertGuard/switch_f']),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/data_0:0'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/Assert/data_0'
                      ]),
                  mock.call(
                      _OpMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/Switch'),
                      parents=[
                          u'assert_equal/All:0',
                          u'assert_equal/Assert/AssertGuard/pred_id:0'
                      ]),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/Assert/Switch:0'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/Assert/Switch'
                      ]),
                  mock.call(
                      _OpMatcher('assert_equal/Assert/AssertGuard/Assert'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/Assert/Switch:0',
                          u'assert_equal/Assert/AssertGuard/Assert/data_0:0',
                          u'assert_equal/Assert/AssertGuard/Assert/data_1:0',
                          u'assert_equal/Assert/AssertGuard/Assert/data_2:0',
                          u'assert_equal/Assert/AssertGuard/Assert/Switch_1:0',
                          u'assert_equal/Assert/AssertGuard/Assert/data_4:0',
                          u'assert_equal/Assert/AssertGuard/Assert/Switch_2:0'
                      ]),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/switch_f:0'),
                      parents=[u'assert_equal/Assert/AssertGuard/switch_f']),
                  mock.call(
                      _OpMatcher(
                          'assert_equal/Assert/AssertGuard/control_dependency_1'
                      ),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/switch_f:0',
                          u'assert_equal/Assert/AssertGuard/Assert'
                      ]),
                  mock.call(
                      _TensorMatcher(
                          'assert_equal/Assert/AssertGuard/control_dependency_1:0'
                      ),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/control_dependency_1'
                      ]),
                  mock.call(
                      _OpMatcher('assert_equal/Assert/AssertGuard/Merge'),
                      parents=[
                          u'assert_equal/Assert/AssertGuard/control_dependency_1:0',
                          u'assert_equal/Assert/AssertGuard/control_dependency:0'
                      ]),
                  mock.call(
                      _OpMatcher('control_dependency'),
                      parents=[
                          'x$tensor', u'assert_equal/Assert/AssertGuard/Merge'
                      ]),
                  mock.call(
                      _TensorMatcher('control_dependency:0'),
                      parents=[u'control_dependency']),
              ],
          }),
  )
  def testGetUniquePath(self, create_graph_fn, feeds, replaced_tensors_ready,
                        expected_calls_dict):

    tensors = create_graph_fn()
    replaced_tensors_ready = {
        tensors[name]: ready for name, ready in replaced_tensors_ready.items()
    }

    for name in expected_calls_dict:

      # This is used to construct the debugging string below.
      actual_needed_matchers_to_pass = []

      def describe_path_fn(x, parents=None):
        if parents is None:
          parents_str = ''
        else:
          parents_str = ', parents={}'.format(
              list(map(_value_to_matcher, parents)))
        actual_needed_matchers_to_pass.append('({}{}),'.format(  # pylint: disable=cell-var-from-loop
            _value_to_matcher(x, True), parents_str))

        if isinstance(x, tf.Operation):
          return x.node_def.name
        if isinstance(x, tf.Tensor):
          self.assertLessEqual(len(parents), 1)
          return x.name
        if isinstance(x, (six.text_type, str, bytes)):
          return x
        raise ValueError('Unexpected type: {}'.format(x))

      path_cb_mock = mock.MagicMock(side_effect=describe_path_fn)

      graph_analyzer = graph_tools.InitializableGraphAnalyzer(
          tf.compat.v1.get_default_graph(), {x: tensors[x] for x in feeds},
          replaced_tensors_ready, path_cb_mock)

      graph_analyzer.get_unique_path(tensors[name])

      try:
        path_cb_mock.assert_has_calls(expected_calls_dict[name])
        self.assertEqual(
            path_cb_mock.call_count, len(expected_calls_dict[name]),
            'Number of expected calls != number of actual calls for {}: {}'
            .format(name, path_cb_mock.call_args_list))
      except AssertionError:
        tf.logging.error(
            'The following is a list of matchers for {}:\n{}'.format(
                name, '\n'.join(actual_needed_matchers_to_pass)))
        raise


def _value_to_matcher(value, add_quotes=False):
  """Returns a matcher for the value - used for debugging failures."""
  if isinstance(value, tf.Operation):
    return _OpMatcher(str(value.node_def.name))
  if isinstance(value, tf.Tensor):
    return _TensorMatcher(str(value.name))
  if isinstance(value, (six.text_type, str, bytes)):
    if add_quotes:
      return '\'{}\''.format(value)
    else:
      return value
  raise ValueError('Cannot get a matcher for: {}, {}'.format(
      type(value), value))


if __name__ == '__main__':
  # TODO(b/133440043): Remove this once TFT supports eager execution.
  tf.compat.v1.disable_eager_execution()
  # TODO(b/138856464): Enable this with v2 control flow.
  # TODO(b/138848475): Replace this with tf.compat.v1.disable_control_flow_v2()
  control_flow_util.ENABLE_CONTROL_FLOW_V2 = False
  test_case.main()
