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
"""Tests for tensorflow_transform.nodes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# GOOGLE-INITIALIZATION

import tensorflow as tf
from tensorflow_transform import nodes
from tensorflow_transform import test_case

mock = tf.test.mock


class _Concat(collections.namedtuple('_Concat', ['label']), nodes.OperationDef):
  pass


class _Swap(collections.namedtuple('_Swap', ['label']), nodes.OperationDef):

  @property
  def num_outputs(self):
    return 2


class _Constant(collections.namedtuple('_Constant', ['value', 'label']),
                nodes.OperationDef):
  pass


class _Identity(collections.namedtuple('_Identity', ['label']),
                nodes.OperationDef):
  pass


class NodesTest(test_case.TransformTestCase):

  def testApplyOperationWithKwarg(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    op = a.parent_operation
    self.assertEqual(a.value_index, 0)
    self.assertEqual(op.operation_def, _Constant('a', 'Constant[a]'))
    self.assertEqual(op.inputs, ())
    self.assertEqual(op.outputs, (a,))

  def testApplyOperationWithTupleOutput(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    b = nodes.apply_operation(_Constant, value='b', label='Constant[b]')
    b_copy, a_copy = nodes.apply_multi_output_operation(
        _Swap, a, b, label='Swap')
    op = b_copy.parent_operation
    self.assertEqual(b_copy.value_index, 0)
    self.assertEqual(a_copy.parent_operation, op)
    self.assertEqual(a_copy.value_index, 1)
    self.assertEqual(op.operation_def, _Swap('Swap'))
    self.assertEqual(op.inputs, (a, b))
    self.assertEqual(op.outputs, (b_copy, a_copy))

  def testOperationNodeWithBadOperatonDef(self):
    with self.assertRaisesRegexp(
        TypeError, 'operation_def must be an OperationDef, got'):
      nodes.OperationNode('not a operation_def', ())

  def testOperationNodeWithBadInput(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    with self.assertRaisesRegexp(
        TypeError, 'Inputs to Operation must be a ValueNode, got'):
      nodes.OperationNode(_Concat(label='Concat'), (a, 'not a value_node'))

  def testOperationNodeWithBadInputs(self):
    with self.assertRaisesRegexp(
        TypeError, 'inputs must be a tuple, got'):
      nodes.OperationNode(_Concat(label='Concat'), 'not a tuple')

  def testValueNodeWithBadParent(self):
    with self.assertRaisesRegexp(
        TypeError, 'parent_operation must be a OperationNode, got'):
      nodes.ValueNode('not an operation node', 0)

  def testValueNodeWithNegativeValueIndex(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'value_index was -1 but parent_operation had 1 outputs'):
      nodes.ValueNode(a.parent_operation, -1)

  def testValueNodeWithTooHighValueIndex(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'value_index was 2 but parent_operation had 1 outputs'):
      nodes.ValueNode(a.parent_operation, 2)

  def testTraverserSimpleGraph(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    mock_visitor = mock.MagicMock()
    mock_visitor.visit.side_effect = [('a',)]
    nodes.Traverser(mock_visitor).visit_value_node(a)
    mock_visitor.assert_has_calls([
        mock.call.visit(_Constant('a', 'Constant[a]'), ()),
        mock.call.validate_value('a'),
    ])

  def testTraverserComplexGraph(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    b = nodes.apply_operation(_Constant, value='b', label='Constant[b]')
    c = nodes.apply_operation(_Constant, value='c', label='Constant[c]')
    b_copy, a_copy = nodes.apply_multi_output_operation(
        _Swap, a, b, label='Swap')
    b_a = nodes.apply_operation(_Concat, b_copy, a_copy, label='Concat[0]')
    b_a_c = nodes.apply_operation(_Concat, b_a, c, label='Concat[1]')

    mock_visitor = mock.MagicMock()
    mock_visitor.visit.side_effect = [
        ('a',), ('b',), ('b', 'a'), ('ba',), ('c',), ('bac',)]

    nodes.Traverser(mock_visitor).visit_value_node(b_a_c)

    mock_visitor.assert_has_calls([
        mock.call.visit(_Constant('a', 'Constant[a]'), ()),
        mock.call.validate_value('a'),
        mock.call.visit(_Constant('b', 'Constant[b]'), ()),
        mock.call.validate_value('b'),
        mock.call.visit(_Swap('Swap'), ('a', 'b')),
        mock.call.validate_value('b'),
        mock.call.validate_value('a'),
        mock.call.visit(_Concat('Concat[0]'), ('b', 'a')),
        mock.call.validate_value('ba'),
        mock.call.visit(_Constant('c', 'Constant[c]'), ()),
        mock.call.validate_value('c'),
        mock.call.visit(_Concat('Concat[1]'), ('ba', 'c')),
        mock.call.validate_value('bac'),
    ])

  def testTraverserComplexGraphMultipleCalls(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    b = nodes.apply_operation(_Constant, value='b', label='Constant[b]')
    c = nodes.apply_operation(_Constant, value='c', label='Constant[c]')
    b_copy, a_copy = nodes.apply_multi_output_operation(
        _Swap, a, b, label='Swap')
    b_a = nodes.apply_operation(_Concat, b_copy, a_copy, label='Concat[0]')
    b_a_c = nodes.apply_operation(_Concat, b_a, c, label='Concat[1]')

    mock_visitor = mock.MagicMock()
    mock_visitor.visit.side_effect = [
        ('a',), ('b',), ('b', 'a'), ('ba',), ('c',), ('bac',)]

    traverser = nodes.Traverser(mock_visitor)
    traverser.visit_value_node(b_a)
    traverser.visit_value_node(b_a_c)

    mock_visitor.assert_has_calls([
        mock.call.visit(_Constant('a', 'Constant[a]'), ()),
        mock.call.validate_value('a'),
        mock.call.visit(_Constant('b', 'Constant[b]'), ()),
        mock.call.validate_value('b'),
        mock.call.visit(_Swap('Swap'), ('a', 'b')),
        mock.call.validate_value('b'),
        mock.call.validate_value('a'),
        mock.call.visit(_Concat('Concat[0]'), ('b', 'a')),
        mock.call.validate_value('ba'),
        mock.call.visit(_Constant('c', 'Constant[c]'), ()),
        mock.call.validate_value('c'),
        mock.call.visit(_Concat('Concat[1]'), ('ba', 'c')),
        mock.call.validate_value('bac'),
    ])

  def testTraverserOutputsNotATuple(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')

    mock_visitor = mock.MagicMock()
    mock_visitor.visit.side_effect = ['not a tuple']

    with self.assertRaisesRegexp(
        ValueError, r'expected visitor to return a tuple, got'):
      nodes.Traverser(mock_visitor).visit_value_node(a)

  def testTraverserBadNumOutputs(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    mock_visitor = mock.MagicMock()
    mock_visitor.visit.side_effect = [('a', 'b')]

    with self.assertRaisesRegexp(
        ValueError, 'has 1 outputs but visitor returned 2 values: '):
      nodes.Traverser(mock_visitor).visit_value_node(a)

  def testTraverserCycle(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    x_0 = nodes.apply_operation(_Identity, a, label='Identity[0]')
    x_1 = nodes.apply_operation(_Identity, x_0, label='Identity[1]')
    x_2 = nodes.apply_operation(_Identity, x_1, label='Identity[2]')
    x_0.parent_operation._inputs = (x_2,)

    mock_visitor = mock.MagicMock()
    mock_visitor.visit.return_value = ('x',)

    with self.assertRaisesWithLiteralMatch(
        AssertionError,
        'Cycle detected: [Identity[2], Identity[1], Identity[0], Identity[2]]'):
      nodes.Traverser(mock_visitor).visit_value_node(x_2)

  def testGetDotGraph(self):
    a = nodes.apply_operation(_Constant, value='a', label='Constant[a]')
    b = nodes.apply_operation(_Constant, value='b', label='Constant[b]')
    b_copy, a_copy = nodes.apply_multi_output_operation(
        _Swap, a, b, label='Swap[0]')
    b_copy2, unused_a_copy2 = nodes.apply_multi_output_operation(
        _Swap, a_copy, b_copy, label='Swap[1]')
    dot_string = nodes.get_dot_graph([b_copy2]).to_string()
    self.WriteRenderedDotFile(dot_string)

    self.assertMultiLineEqual(
        dot_string,
        """\
digraph G {
directed=True;
node [shape=Mrecord];
"Constant[a]" [label="{_Constant|value: a|label: Constant[a]}"];
"Constant[b]" [label="{_Constant|value: b|label: Constant[b]}"];
"Swap[0]" [label="{_Swap|label: Swap[0]|{<0>0|<1>1}}"];
"Constant[a]" -> "Swap[0]";
"Constant[b]" -> "Swap[0]";
"Swap[1]" [label="{_Swap|label: Swap[1]|{<0>0|<1>1}}"];
"Swap[0]":1 -> "Swap[1]";
"Swap[0]":0 -> "Swap[1]";
}
""",
        msg='Result dot graph is:\n{}'.format(dot_string))


if __name__ == '__main__':
  test_case.main()
