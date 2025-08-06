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

import abc
import collections
import dataclasses
from typing import Any, Collection, Dict, List, Optional, Tuple

import pydot


class OperationDef(metaclass=abc.ABCMeta):
    """The definition of an operation.

    This class contains all the information needed to run an operation, except
    the number of inputs and their values.  A subclass should document

      - How many inputs it expects, and what they should contain.
      - What it outputs, as a function of its inputs.

    An OperationDef is just a specification and does not contain the actual
    computation.
    """

    @property
    def num_outputs(self) -> int:
        """The number of outputs returned by this operation."""
        return 1

    @abc.abstractproperty
    def label(self) -> str:
        """A unique label for this operation in the graph."""
        pass

    def get_field_str(self, field_name: str) -> str:
        """Returns a str representation of the requested field."""
        return getattr(self, field_name)

    @property
    def is_partitionable(self) -> bool:
        """If True, means that this operation can be applied on partitioned data.

        Being able to be applied on partitioned data means that partitioning the
        data, running this operation on each of the data subsets independently, and
        then having the next operation get the flattened results as inputs would be
        equivalent to running this operation on the entire data and passing the
        result to the next operation.

        Returns
        -------
          A bool indicating whether or not this operation is partitionable.
        """
        return False

    @property
    def cache_coder(self) -> Optional[object]:
        """A CacheCoder object used to cache outputs returned by this operation.

        If this doesn't return None, then:
          * num_outputs has to be 1
          * is_partitionable has to be True.
        """
        return None


@dataclasses.dataclass(frozen=True)
class ValueNode:
    """A placeholder that will ultimately be translated to a PCollection.

    Attributes
    ----------
      parent_operation: The `OperationNode` that produces this value.
      value_index: The index of this value in the outputs of `parent_operation`.
    """

    parent_operation: "OperationNode"
    value_index: int

    def __post_init__(self):
        num_outputs = self.parent_operation.operation_def.num_outputs
        if not (self.value_index >= 0 and self.value_index < num_outputs):
            raise ValueError(
                "value_index was {} but parent_operation had {} outputs".format(
                    self.value_index, num_outputs
                )
            )


class OperationNode:
    """A placeholder that will ultimately be translated to a PTransform.

    Attributes
    ----------
      operation_def: An `OperationDef`.
      inputs: A tuple of `ValueNode`s.
    """

    def __init__(self, operation_def, inputs):
        self._operation_def = operation_def
        self._inputs = inputs
        if not isinstance(operation_def, OperationDef):
            raise TypeError(
                "operation_def must be an OperationDef, got {} of type {}".format(
                    operation_def, type(operation_def)
                )
            )
        if not isinstance(inputs, tuple):
            raise TypeError(
                "inputs must be a tuple, got {} of type {}".format(inputs, type(inputs))
            )
        for value_node in inputs:
            if not isinstance(value_node, ValueNode):
                raise TypeError(
                    "Inputs to Operation must be a ValueNode, got {} of type {}".format(
                        value_node, type(value_node)
                    )
                )

    def __repr__(self):
        return "{}(operation_def={}, inputs={})".format(
            self.__class__.__name__, self.operation_def, self.inputs
        )

    @property
    def operation_def(self):
        return self._operation_def

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        """A tuple of `ValueNode`s representing outputs of this operation."""
        return tuple(
            ValueNode(self, value_index)
            for value_index in range(self.operation_def.num_outputs)
        )


def apply_operation(operation_def_cls, *args, **kwargs):
    """Applies an operation to some inputs and returns its output.

    This function is syntactic sugar on top of the constructor for OperationNode.
    The operation must return a single output.

    Args:
    ----
      operation_def_cls: A class that is a subclass of `OperationDef`.
      *args: The inputs to the `OperationNode`.
      **kwargs: Constructor args for `operation_def_cls`.

    Returns:
    -------
      The output of the `OperationNode` that was constructed.
    """
    (result,) = apply_multi_output_operation(operation_def_cls, *args, **kwargs)
    return result


def apply_multi_output_operation(operation_def_cls, *args, **kwargs):
    """Like `apply_operation` but returns a tuple of outputs."""
    try:
        return OperationNode(operation_def_cls(**kwargs), args).outputs
    except TypeError as e:
        raise RuntimeError(
            "Failed to apply Operation {}, with error: {}".format(
                operation_def_cls, str(e)
            )
        )


class Visitor(metaclass=abc.ABCMeta):
    """Class to visit nodes in the graph."""

    @abc.abstractmethod
    def validate_value(self, value):
        """Validate the value of a ValueNode.

        Should raise an error if `value` is invalid.

        Args:
        ----
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
        ----
          operation_def: The `OperationDef` of the current `OperationNode`.
          input_values: Values corresponding to each input of the current
              `OperationNode`.

        Returns:
        -------
          A tuple of values corresponding to the outputs of the current
              `OperationNode`.
        """
        pass


class Traverser:
    """Class to traverse the DAG of nodes."""

    def __init__(self, visitor: Visitor):
        """Init method for Traverser.

        Args:
        ----
          visitor: A `Visitor` object.
        """
        self._cached_value_nodes_values: Dict[ValueNode, Any] = {}
        self._stack: List[OperationNode] = []
        self._visitor = visitor

    def visit_value_node(self, value_node: ValueNode):
        """Visit a value node, and return a corresponding value.

        Args:
        ----
          value_node: A `ValueNode`.

        Returns:
        -------
          A value corresponding to `value_node` determined by the implementation of
              the abstract `visit` method.
        """
        return self._maybe_visit_value_node(value_node)

    def _maybe_visit_value_node(self, value_node: ValueNode):
        """Visit a value node if not cached, and return a corresponding value.

        Args:
        ----
          value_node: A `ValueNode`.

        Returns:
        -------
          A value corresponding to `value_node` determined by the implementation of
              the abstract `visit` method.
        """
        if value_node not in self._cached_value_nodes_values:
            self._visit_operation(value_node.parent_operation)
        return self._cached_value_nodes_values[value_node]

    def _visit_operation(self, operation: OperationNode):
        """Visit an `OperationNode`."""
        if operation in self._stack:
            cycle = self._stack[self._stack.index(operation) :] + [operation]
            # For readability, just print the label of `operation_def`s
            cycle = ", ".join(operation.operation_def.label for operation in cycle)
            raise AssertionError("Cycle detected: [{}]".format(cycle))
        self._stack.append(operation)
        input_values = tuple(map(self._maybe_visit_value_node, operation.inputs))
        assert operation is self._stack.pop()
        output_values = self._visitor.visit(operation.operation_def, input_values)
        outputs = operation.outputs

        # Expect a tuple of outputs.  Since ValueNode and OperationDef are both
        # subclasses of tuple, we also explicitly disallow them, since returning
        # a single ValueNode or OperationDef is almost certainly an error.
        try:
            _ = iter(output_values)
            output_iterable = not isinstance(output_values, str)
        except TypeError:
            output_iterable = False
        if not output_iterable or isinstance(output_values, (ValueNode, OperationDef)):
            raise ValueError(
                "When running operation {} expected visitor to return a tuple, got "
                "{} of type {}".format(
                    operation.operation_def.label, output_values, type(output_values)
                )
            )
        # DoOutputsTuple doesn't work with len().
        if hasattr(output_values, "__len__") and len(output_values) != len(outputs):
            raise ValueError(
                "Operation {} has {} outputs but visitor returned {} values: "
                "{}".format(
                    operation.operation_def,
                    len(outputs),
                    len(output_values),
                    output_values,
                )
            )

        for output, value in zip(outputs, output_values):
            self._visitor.validate_value(value)
            self._cached_value_nodes_values[output] = value


def _escape(line: str) -> str:
    for char in "<>{}":
        line = line.replace(char, "\\%s" % char)
    return line


class _PrintGraphVisitor(Visitor):
    """Visitor to produce a human readable string for a graph."""

    def __init__(self):
        self._print_result = ""
        self._dot_graph = pydot.Dot(directed=True)
        self._dot_graph.obj_dict = collections.OrderedDict(
            sorted(self._dot_graph.obj_dict.items(), key=lambda t: t[0])
        )
        self._dot_graph.set_node_defaults(shape="Mrecord")
        super().__init__()

    def get_dot_graph(self) -> pydot.Dot:
        return self._dot_graph

    def visit(self, operation_def, input_nodes) -> Tuple[pydot.Node, ...]:
        num_outputs = operation_def.num_outputs
        node_name = operation_def.label

        display_label_rows = [operation_def.__class__.__name__] + [
            _escape("%s: %s" % (field, operation_def.get_field_str(field)))
            for field in operation_def._fields
        ]

        if operation_def.is_partitionable:
            display_label_rows.append("partitionable: %s" % True)

        if num_outputs != 1:
            ports = "|".join("<{0}>{0}".format(idx) for idx in range(num_outputs))
            display_label_rows.append("{%s}" % ports)
        display_label = "{%s}" % "|".join(display_label_rows)

        node = pydot.Node(node_name, label=display_label)

        self._dot_graph.add_node(node)

        for input_node in input_nodes:
            self._dot_graph.add_edge(pydot.Edge(input_node, node))

        if num_outputs == 1:
            return (node,)
        else:
            return tuple(
                pydot.Node(obj_dict={"name": '"{}":{}'.format(node_name, idx)})
                for idx in range(num_outputs)
            )

    def validate_value(self, value: pydot.Node):
        assert isinstance(value, pydot.Node)


def get_dot_graph(leaf_nodes: Collection[ValueNode]) -> pydot.Dot:
    """Utility to print a graph in a human readable manner.

    The format resembles a sequence of calls to apply_operation or
    apply_multi_output_operation.

    Args:
    ----
      leaf_nodes: A list of leaf `ValueNode`s to define the graph.  The graph will
        be the transitive parents of the leaf nodes.

    Returns:
    -------
      A human readable summary of the graph.
    """
    visitor = _PrintGraphVisitor()
    traverser = Traverser(visitor)
    for value_node in leaf_nodes:
        traverser.visit_value_node(value_node)
    return visitor.get_dot_graph()


class _CountGraphNodes(Visitor):
    """Visitor which counts the graph nodes."""

    num_nodes = 0

    def visit(self, operation_def: OperationDef, _) -> Tuple[int]:
        self.num_nodes += 1
        return tuple(1 for _ in range(operation_def.num_outputs))

    def validate_value(self, value: int):
        pass


def count_graph_nodes(leaf_nodes: Collection[ValueNode]) -> int:
    """Counts the number of graph nodes.

    Note: these nodes only include the TFT graph nodes, it doesn't count beam
    nodes constructed directly.

    Args:
    ----
      leaf_nodes: A list of leaf `ValueNode`s to define the graph.  The graph will
        be the transitive parents of the leaf nodes.

    Returns:
    -------
      The count of TFT graph nodes.
    """
    visitor = _CountGraphNodes()
    traverser = Traverser(visitor)
    for value_node in leaf_nodes:
        traverser.visit_value_node(value_node)
    return visitor.num_nodes
