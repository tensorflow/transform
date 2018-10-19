# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Framework for an abstract graph of operations.

This graph resembles the TF graph but is much simpler.  It is used to define the
execution graph that will ultimately be translated to a Beam execution graph.
However we need this intermediate data structure as Beam graphs are not easy to
construct, introspect or manipulate.  So we provide a very lightweight
framework in this module instead.

The framework is a graph with two kinds of node `OperationNode` and `ValueNode`.
An `OperationNode` has inputs and outputs that are `ValueNode`s. Each
`ValueNode` has exactly one parent `OperationNode`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections


class ValueNode(collections.namedtuple(
    'ValueNode', ['parent_operation', 'value_index'])):
  """A placeholder that will ultimately be translated to a PCollection.

  Args:
    parent_operation: The `OperationNode` that produces this value.
    value_index: The index of this value in the outputs of `parent_operation`.
  """

  def __init__(self, parent_operation, value_index):
    if not isinstance(parent_operation, OperationNode):
      raise TypeError(
          'parent_operation must be a OperationNode, got {} of type {}'.format(
              parent_operation, type(parent_operation)))
    num_outputs = parent_operation.operation_def.num_outputs
    if not (0 <= value_index and value_index < num_outputs):
      raise ValueError(
          'value_index was {} but parent_operation had {} outputs'.format(
              value_index, num_outputs))
    super(ValueNode, self).__init__()


class OperationDef(object):
  """The definition of an operation.

  This class contains all the information needed to run an operation, except
  the number of inputs and their values.
  """

  @property
  def num_outputs(self):
    """The number of outputs returned by this operation."""
    return 1


class OperationNode(object):
  """A placeholder that will ultimately be translated to a PTransform.

  Args:
    operation_def: An `OperationDef`.
    inputs: An iterable of `ValueNode`s.
  """

  def __init__(self, operation_def, inputs):
    self._operation_def = operation_def
    self._inputs = inputs
    if not isinstance(operation_def, OperationDef):
      raise TypeError(
          'operation_def must be an OperationDef, got {} of type {}'.format(
              operation_def, type(operation_def)))
    if not isinstance(inputs, tuple):
      raise TypeError(
          'inputs must be a tuple, got {} of type {}'.format(
              inputs, type(inputs)))
    for value_node in inputs:
      if not isinstance(value_node, ValueNode):
        raise TypeError(
            'Inputs to Operation must be a ValueNode, got {} of type {}'.format(
                value_node, type(value_node)))

  @property
  def operation_def(self):
    return self._operation_def

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    """A tuple of `ValueNode`s representing outputs of this operation."""
    return tuple(ValueNode(self, value_index)
                 for value_index in range(self.operation_def.num_outputs))

  def __repr__(self):
    return 'OperationNode(operation_def={}, inputs=[...])'.format(
        self.operation_def)


def apply_operation(operation_def_cls, *args, **kwargs):
  """Applies an operation to some inputs and returns its output.

  This function is syntactic sugar on top of the constructor for OperationNode.
  The operation must return a single output.

  Args:
    operation_def_cls: A class that is a subclass of `OperationDef`.
    *args: The inputs to the `OperationNode`.
    **kwargs: Constructor args for `operation_def_cls`.

  Returns:
    The output of the `OperationNode` that was constructed.
  """
  (result,) = apply_multi_output_operation(operation_def_cls, *args, **kwargs)
  return result


def apply_multi_output_operation(operation_def_cls, *args, **kwargs):
  """Like `apply_operation` but returns a tuple of outputs."""
  return OperationNode(operation_def_cls(**kwargs), args).outputs


class Visitor(object):
  """Class to visit nodes in the graph."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def validate_value(self, value):
    """Validate the value of a ValueNode.

    Should raise an error if `value` is invalid.

    Args:
      value: An element of the tuple returned by visit.
    """
    pass

  @abc.abstractmethod
  def visit(self, operation_def, input_values):
    """Visits an `OperationNode` in the graph.

    Called once for each `OperationNode` in the graph that is visited.  Will
    be called with the `operation_def` of that `OperationNode`, and values
    determined by cached recursive calls to the `OperationNode`s that produce
    each input `ValueNode` of the current `OperationNode`.

    Args:
      operation_def: The `OperationDef` of the current `OperationNode`.
      input_values: Values corresponding to each input of the current
          `OperationNode`.

    Returns:
      A tuple of values corresponding to the outputs of the current
          `OperationNode`.
    """
    pass


class Traverser(object):
  """Class to traverse the DAG of nodes.

  Args:
    visitor: A `Visitor` object.
  """

  def __init__(self, visitor):
    self._cached_value_nodes_values = {}
    self._stack = []
    self._visitor = visitor

  def visit_value_node(self, value_node):
    """Visit a value node, and return a corresponding value.

    Args:
      value_node: A `ValueNode`.

    Returns:
      A value corresponding to `value_node` determined by the implementation of
          the abstract `visit` method.
    """
    if value_node not in self._cached_value_nodes_values:
      self._visit_operation(value_node.parent_operation)
    return self._cached_value_nodes_values[value_node]

  def _visit_operation(self, operation):
    """Visit an `OperationNode`."""
    if operation in self._stack:
      cycle = self._stack[self._stack.index(operation):] + [operation]
      # For readability, just print the cycle of `operation_def`s
      cycle = [operation.operation_def for operation in cycle]
      raise AssertionError('Cycle detected: {}'.format(cycle))
    self._stack.append(operation)
    input_values = tuple(map(self.visit_value_node, operation.inputs))
    assert operation is self._stack.pop()
    output_values = self._visitor.visit(operation.operation_def, input_values)
    outputs = operation.outputs

    if not isinstance(output_values, tuple):
      raise ValueError(
          'When running operation {} expected visitor to return a tuple, got '
          '{} of type {}'.format(operation.operation_def, output_values,
                                 type(output_values)))
    if len(output_values) != len(outputs):
      raise ValueError(
          'Operation {} has {} outputs but visitor returned {} values: '
          '{}'.format(operation.operation_def, len(outputs),
                      len(output_values), output_values))

    for output, value in zip(outputs, output_values):
      self._visitor.validate_value(value)
      self._cached_value_nodes_values[output] = value
