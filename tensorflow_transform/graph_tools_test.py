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


import tensorflow as tf
from tensorflow_transform import graph_tools
from tensorflow_transform import test_case

import unittest
from tensorflow.python.ops import control_flow_ops


def _create_graph_with_y_function_of_x():
  x = tf.placeholder(tf.int64)
  y = x + 1
  return {'x': x, 'y': y}


def _create_graph_with_y_function_of_x_sparse():
  x = tf.sparse_placeholder(tf.int64)
  y = tf.sparse_reduce_sum(x) + 1
  return {'x': x, 'y': y}


def _create_graph_with_y_sparse_function_of_x_sparse():
  x = tf.sparse_placeholder(tf.int64)
  y = tf.SparseTensor(
      indices=x.indices,
      values=x.values + 1,
      dense_shape=x.dense_shape)
  return {'x': x, 'y': y}


def _create_graph_with_y_function_of_x_and_table():
  filename = tf.placeholder(tf.string, ())
  table = tf.contrib.lookup.index_table_from_file(filename)
  x = tf.placeholder(tf.string, (None,))
  y = table.lookup(x)
  return {'filename': filename, 'x': x, 'y': y}


def _create_graph_with_y_function_of_x_and_untracked_table():
  filename = tf.placeholder(tf.string, ())
  table = tf.contrib.lookup.index_table_from_file(filename)
  x = tf.placeholder(tf.string, (None,))
  y = table.lookup(x)
  del tf.get_collection_ref(tf.GraphKeys.TABLE_INITIALIZERS)[:]
  return {'filename': filename, 'x': x, 'y': y}


def _create_graph_with_table_initialized_by_table_output():
  filename = tf.placeholder(tf.string, ())
  table1 = tf.contrib.lookup.index_table_from_file(filename)
  # Use output from the first table to initialize the second table.
  tensor_keys = tf.as_string(
      table1.lookup(tf.constant(['a', 'b', 'c'], tf.string)))
  table2 = tf.contrib.lookup.index_table_from_tensor(tensor_keys)
  x = tf.placeholder(tf.string, (None,))
  y = table2.lookup(x)
  return {'filename': filename, 'x': x, 'y': y}


def _create_graph_with_assert_equal():
  x = tf.placeholder(tf.int64)
  y = tf.placeholder(tf.int64)
  z = control_flow_ops.with_dependencies([tf.assert_equal(x, y)], x)
  return {'x': x, 'y': y, 'z': z}


def _create_graph_with_y_function_of_x_with_tf_while():
  x = tf.placeholder(tf.int64, ())

  # Subtract 10 from x using a tf.while_loop.
  def stop_condition(counter, x_minus_counter):
    del x_minus_counter  # unused
    return tf.less(counter, 10)
  def iteration(counter, x_minus_counter):
    return tf.add(counter, 1), tf.add(x_minus_counter, -1)
  initial_values = [tf.constant(0), x]
  final_values = tf.while_loop(stop_condition, iteration, initial_values)

  y = final_values[1]
  return {'x': x, 'y': y}


class GraphToolsTest(test_case.TransformTestCase):

  @test_case.parameters(
      (_create_graph_with_y_function_of_x, [], {'x': False}, {'y': False}, 0),
      (_create_graph_with_y_function_of_x, [], {'x': True}, {'y': True}, 0),
      (_create_graph_with_y_function_of_x, ['x'], {}, {'y': True}, 0),
      (_create_graph_with_y_function_of_x_sparse,
       [], {'x': False}, {'y': False}, 0),
      (_create_graph_with_y_function_of_x_sparse,
       [], {'x': True}, {'y': True}, 0),
      (_create_graph_with_y_function_of_x_sparse,
       ['x'], {}, {'y': True}, 0),
      (_create_graph_with_y_sparse_function_of_x_sparse,
       [], {'x': False}, {'y': False}, 0),
      (_create_graph_with_y_sparse_function_of_x_sparse,
       [], {'x': True}, {'y': True}, 0),
      (_create_graph_with_y_sparse_function_of_x_sparse,
       ['x'], {}, {'y': True}, 0),
      (_create_graph_with_y_function_of_x_with_tf_while,
       [], {'x': False}, {'y': False}, 0),
      (_create_graph_with_y_function_of_x_with_tf_while,
       [], {'x': True}, {'y': True}, 0),
      (_create_graph_with_y_function_of_x_with_tf_while,
       ['x'], {}, {'y': True}, 0),
      (_create_graph_with_y_function_of_x_and_table,
       [], {'x': False, 'filename': False}, {'y': False}, 0),
      (_create_graph_with_y_function_of_x_and_table,
       [], {'x': False, 'filename': True}, {'y': False}, 1),
      (_create_graph_with_y_function_of_x_and_table,
       [], {'x': True, 'filename': False}, {'y': False}, 0),
      (_create_graph_with_y_function_of_x_and_table,
       [], {'x': True, 'filename': True}, {'y': True}, 1),
      (_create_graph_with_y_function_of_x_and_table,
       ['x'], {'filename': False}, {'y': False}, 0),
      (_create_graph_with_y_function_of_x_and_table,
       ['x'], {'filename': True}, {'y': True}, 1),
      (_create_graph_with_assert_equal,
       ['x', 'y'], {}, {'z': True}, 0),
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
      should_be_ready: A dict dict whose keys are keys in the dict returned by
          create_graph_fn and value are bools indicating whether a tensor can be
          calculated in this phase.
      num_ready_table_initializers: The number of table initializers that are
          ready to run in the table initialization run of this phase.
    """
    tensors = create_graph_fn()
    feeds = [tensors[name] for name in feeds]
    replaced_tensors_ready = {tensors[name]: ready
                              for name, ready in replaced_tensors_ready.items()}
    fetches = [tensors[name] for name in should_be_ready]
    expected_ready_tensors = [
        tensors[name] for name in should_be_ready if should_be_ready[name]]
    ready_table_initializers, ready_tensors = (
        graph_tools.determine_ready_tensors_and_table_initializers(
            tf.get_default_graph(), fetches, feeds, replaced_tensors_ready))
    self.assertEqual(len(ready_table_initializers),
                     num_ready_table_initializers)
    self.assertCountEqual(ready_tensors, expected_ready_tensors)

  @test_case.parameters(
      (_create_graph_with_y_function_of_x, [], {}, ['y'],
       'may have be caused by manually adding a placeholder to the graph'),
      (_create_graph_with_y_function_of_x_and_table,
       [], {'x': False}, ['y'],
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_y_function_of_x_and_table,
       [], {'x': True}, ['y'],
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_y_function_of_x_and_table,
       ['filename'], {'x': False}, ['y'],
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_y_function_of_x_and_table,
       ['filename'], {'x': True}, ['y'],
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_y_function_of_x_and_table,
       ['filename', 'x'], {}, ['y'],
       'placeholders will not be fed during table initialization'),
      (_create_graph_with_y_function_of_x_and_untracked_table,
       ['x'], {'filename': True}, ['y'],
       'may be caused by adding an initializable table without'),
      (_create_graph_with_table_initialized_by_table_output,
       ['x'], {'filename': True}, ['y'],
       'tables are initialized in one pass')
  )
  def testDetermineReadyTensorsAndTableInitializersRaises(
      self, create_graph_fn, feeds, replaced_tensors_ready, fetches,
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
      fetches: A list keys in the dict returned by create_graph_fn to determine
          ready status for.
      error_msg_regex: The expected error message.
    """
    tensors = create_graph_fn()
    feeds = [tensors[name] for name in feeds]
    fetches = [tensors[name] for name in fetches]
    replaced_tensors_ready = {tensors[name]: ready
                              for name, ready in replaced_tensors_ready.items()}
    with self.assertRaisesRegexp(ValueError, error_msg_regex):
      graph_tools.determine_ready_tensors_and_table_initializers(
          tf.get_default_graph(), fetches, feeds, replaced_tensors_ready)


if __name__ == '__main__':
  unittest.main()
