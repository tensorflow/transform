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
"""Unit tests for tensorflow_transform.beam.deep_copy."""

import collections

import apache_beam as beam
from apache_beam import pvalue
from apache_beam.transforms import resources
from tensorflow_transform.beam import deep_copy
from tensorflow_transform.beam import test_helpers
import unittest


# pylint: disable=g-long-lambda
class DeepCopyTest(unittest.TestCase):

  @staticmethod
  def _MakeBeamPipeline():
    return beam.Pipeline(**test_helpers.make_test_beam_pipeline_kwargs())

  # _CountingIdentityFn and _InitializeCounts are declared as class-level
  # methods to avoid Beam serialization issues, which would occur if an
  # individual object instance were referenced in a lambda.  In such a case,
  # the object would be serialized and deserialized, so that mutations would
  # not be propagated correctly for the subsequent verification step.
  @staticmethod
  def _CountingIdentityFn(label, x):
    DeepCopyTest._counts[label] += 1
    return x

  @staticmethod
  def _MakeAdd1CountingIdentityFn(label):

    def Add1CountingIdentityFn(x_y):
      (x, y) = x_y
      return DeepCopyTest._CountingIdentityFn(label, (x + 1, y))

    return Add1CountingIdentityFn

  @staticmethod
  def _InitializeCounts():
    DeepCopyTest._counts = collections.defaultdict(int)

  def setUp(self):
    DeepCopyTest._InitializeCounts()

  def testBasicDeepCopy(self):
    with DeepCopyTest._MakeBeamPipeline() as p:
      grouped = (p
                 | beam.Create([(1, 'a'), (2, 'b'), (3, 'c')])
                 | beam.Map(
                     lambda x: DeepCopyTest._CountingIdentityFn(
                         'PreGroup', x))
                 | beam.GroupByKey())
      modified = (
          grouped
          |
          'Add1' >> beam.Map(DeepCopyTest._MakeAdd1CountingIdentityFn('Add1'))
          |
          'Add2' >> beam.Map(DeepCopyTest._MakeAdd1CountingIdentityFn('Add2')))
      copied = deep_copy.deep_copy(modified)

      # pylint: disable=expression-not-assigned
      modified | 'Add3' >> beam.Map(
          DeepCopyTest._MakeAdd1CountingIdentityFn('Add3'))
      # pylint: enable=expression-not-assigned

      # Check labels.
      self.assertEqual(copied.producer.full_label, 'Add2.Copy[0]')
      self.assertEqual(copied.producer.inputs[0].producer.full_label,
                       'Add1.Copy[0]')

      # Check that deep copy was performed.
      self.assertIsNot(copied.producer.inputs[0], modified.producer.inputs[0])

      # Check that copy stops at materialization boundary.
      self.assertIs(copied.producer.inputs[0].producer.inputs[0],
                    modified.producer.inputs[0].producer.inputs[0])

    # Check counts of processed items.
    self.assertEqual(DeepCopyTest._counts['PreGroup'], 3)
    self.assertEqual(DeepCopyTest._counts['Add1'], 6)
    self.assertEqual(DeepCopyTest._counts['Add2'], 6)
    self.assertEqual(DeepCopyTest._counts['Add3'], 3)

  def testMultipleCopies(self):
    with DeepCopyTest._MakeBeamPipeline() as p:
      grouped = (p
                 | beam.Create([(1, 'a'), (2, 'b'), (3, 'c')])
                 | beam.Map(lambda x: DeepCopyTest._CountingIdentityFn(
                     'PreGroup', x))
                 | beam.GroupByKey())
      modified = (
          grouped
          |
          'Add1' >> beam.Map(DeepCopyTest._MakeAdd1CountingIdentityFn('Add1'))
          |
          'Add2' >> beam.Map(DeepCopyTest._MakeAdd1CountingIdentityFn('Add2')))

      num_copies = 6

      for i in range(num_copies):
        copied = deep_copy.deep_copy(modified)
        self.assertEqual(copied.producer.full_label, 'Add2.Copy[%d]' % i)
        self.assertEqual(copied.producer.inputs[0].producer.full_label,
                         'Add1.Copy[%d]' % i)

    self.assertEqual(DeepCopyTest._counts['PreGroup'], 3)
    self.assertEqual(DeepCopyTest._counts['Add1'], 3 * (num_copies + 1))
    self.assertEqual(DeepCopyTest._counts['Add2'], 3 * (num_copies + 1))

  def testFlatten(self):
    with DeepCopyTest._MakeBeamPipeline() as p:
      create_1 = p | 'Create1' >> beam.Create([(1, 'a'), (2, 'b')])
      create_2 = p | 'Create2' >> beam.Create([(3, 'c')])
      created = (create_1, create_2) | 'Flatten1' >> beam.Flatten()
      grouped1 = (created
                  | 'PreGroup1' >> beam.Map(
                      lambda x: DeepCopyTest._CountingIdentityFn(
                          'PreGroup1', x))
                  | 'GBK1' >> beam.GroupByKey())
      grouped2 = (p
                  | beam.Create([(1, 'a'), (2, 'b'), (3, 'c')])
                  | 'PreGroup2' >> beam.Map(
                      lambda x: DeepCopyTest._CountingIdentityFn(
                          'PreGroup2', x))
                  | 'GBK2' >> beam.GroupByKey())
      modified1 = (
          grouped1
          |
          'Add1' >> beam.Map(DeepCopyTest._MakeAdd1CountingIdentityFn('Add1')))
      modified2 = (
          grouped2
          |
          'Add2' >> beam.Map(DeepCopyTest._MakeAdd1CountingIdentityFn('Add2')))
      flattened = (modified1, modified2) | 'Flatten2' >> beam.Flatten()
      modified3 = (
          flattened
          |
          'Add3' >> beam.Map(DeepCopyTest._MakeAdd1CountingIdentityFn('Add3')))

      copied = deep_copy.deep_copy(modified3)

      # Check that deep copy was performed.
      self.assertIsNot(copied.producer.inputs[0], modified3.producer.inputs[0])
      self.assertIsNot(copied.producer.inputs[0].producer.inputs[0],
                       modified3.producer.inputs[0].producer.inputs[0])
      self.assertIsNot(copied.producer.inputs[0].producer.inputs[1],
                       modified3.producer.inputs[0].producer.inputs[1])

      # Check that copy stops at materialization boundary.
      self.assertIs(
          copied.producer.inputs[0].producer.inputs[0].producer.inputs[0],
          modified3.producer.inputs[0].producer.inputs[0].producer.inputs[0])
      self.assertIs(
          copied.producer.inputs[0].producer.inputs[1].producer.inputs[0],
          modified3.producer.inputs[0].producer.inputs[1].producer.inputs[0])

    # Check counts of processed items.
    self.assertEqual(DeepCopyTest._counts['PreGroup1'], 3)
    self.assertEqual(DeepCopyTest._counts['PreGroup2'], 3)
    self.assertEqual(DeepCopyTest._counts['Add1'], 6)
    self.assertEqual(DeepCopyTest._counts['Add2'], 6)
    self.assertEqual(DeepCopyTest._counts['Add3'], 12)

  def testEachPTransformCopiedOnce(self):
    with DeepCopyTest._MakeBeamPipeline() as p:
      created = p | 'Create1' >> beam.Create([(1, 'a'), (2, 'b')])
      modified1 = (created
                   | 'Transform1' >> beam.Map(
                       lambda x: DeepCopyTest._CountingIdentityFn(
                           'Transform1', x)))
      partition_fn = lambda element, partitions: element[0] % partitions
      p1, p2 = (modified1
                | 'Partition' >> beam.Partition(partition_fn, 2))
      merged = (p1, p2) | 'Flatten1' >> beam.Flatten()
      modified2 = (merged
                   | 'Transform2' >> beam.Map(
                       lambda x: DeepCopyTest._CountingIdentityFn(
                           'Transform2', x)))

      copied = deep_copy.deep_copy(modified2)

      # Check that deep copy was performed.
      self.assertIsNot(copied.producer.inputs[0], modified2.producer.inputs[0])
      self.assertIsNot(copied.producer.inputs[0].producer.inputs[0],
                       modified2.producer.inputs[0].producer.inputs[0])
      self.assertIsNot(copied.producer.inputs[0].producer.inputs[1],
                       modified2.producer.inputs[0].producer.inputs[1])

    # Check counts of processed items.
    self.assertEqual(DeepCopyTest._counts['Transform1'], 4)
    self.assertEqual(DeepCopyTest._counts['Transform2'], 4)

  def testCombineGlobally(self):
    with DeepCopyTest._MakeBeamPipeline() as p:
      combined = (p
                  | beam.Create([1, 2, 3])
                  | beam.Map(
                      lambda x: DeepCopyTest._CountingIdentityFn(
                          'PreCombine', x))
                  | beam.WindowInto(beam.window.FixedWindows(5, 0))
                  | beam.CombineGlobally(
                      beam.transforms.combiners.MeanCombineFn()
                  ).without_defaults()
                  | beam.Map(
                      lambda x: DeepCopyTest._CountingIdentityFn(
                          'PostCombine', x)))
      copied = deep_copy.deep_copy(combined)

      # Check that deep copy was performed.
      self.assertIsNot(combined, copied)
      self.assertIsNot(combined.producer.inputs[0], copied.producer.inputs[0])
      self.assertEqual(combined.producer.inputs[0].producer.full_label,
                       'CombineGlobally(MeanCombineFn)/UnKey')
      self.assertEqual(copied.producer.inputs[0].producer.full_label,
                       'CombineGlobally(MeanCombineFn)/UnKey.Copy[0]')

      # Check that deep copy stops at materialization boundary.
      self.assertIs(combined.producer.inputs[0].producer.inputs[0],
                    copied.producer.inputs[0].producer.inputs[0])
      self.assertEqual(
          str(combined.producer.inputs[0].producer.inputs[0]),
          ('PCollection[CombineGlobally(MeanCombineFn)/CombinePerKey/Combine/'
           'ParDo(CombineValuesDoFn).None]'))
      self.assertIs(combined.producer.inputs[0].producer.inputs[0].producer,
                    copied.producer.inputs[0].producer.inputs[0].producer)
      self.assertEqual(
          copied.producer.inputs[0].producer.inputs[0].producer.full_label,
          ('CombineGlobally(MeanCombineFn)/CombinePerKey/Combine/'
           'ParDo(CombineValuesDoFn)'))

    # Check counts of processed items.
    self.assertEqual(DeepCopyTest._counts['PreCombine'], 3)
    self.assertEqual(DeepCopyTest._counts['PostCombine'], 2)

  def testSideInputNotCopied(self):
    with DeepCopyTest._MakeBeamPipeline() as p:
      side = (p
              | 'CreateSide' >> beam.Create(['s1', 's2', 's3'])
              | beam.Map(
                  lambda x: DeepCopyTest._CountingIdentityFn(
                      'SideInput', x)))
      main = (p
              | 'CreateMain' >> beam.Create([1, 2, 3])
              | beam.Map(
                  lambda x: DeepCopyTest._CountingIdentityFn(
                      'Main', x))
              | beam.Map(lambda e, s: (e, list(s)),
                         pvalue.AsList(side)))
      copied = deep_copy.deep_copy(main)

      # Check that deep copy was performed.
      self.assertIsNot(main, copied)
      self.assertIsNot(main.producer, copied.producer)

      # Check that deep copy stops at the side input materialization boundary.
      self.assertIs(main.producer.side_inputs[0],
                    copied.producer.side_inputs[0])
      self.assertIs(main.producer.side_inputs[0].pvalue, side)

    # Check counts of processed items.
    self.assertEqual(DeepCopyTest._counts['SideInput'], 3)
    self.assertEqual(DeepCopyTest._counts['Main'], 6)

  def testDeepCopyTags(self):
    if not resources.ResourceHint.is_registered('tags'):
      self.skipTest('Resource hint tags are not available.')

    with DeepCopyTest._MakeBeamPipeline() as p:
      grouped = (
          p | beam.Create([(1, 'a'), (2, 'b'), (3, 'c')])
          | beam.Map(lambda x: DeepCopyTest._CountingIdentityFn('PreGroup', x)))

      modified = (
          grouped
          |
          'Add1' >> beam.Map(DeepCopyTest._MakeAdd1CountingIdentityFn('Add1'))
          |
          'Add2' >> beam.Map(DeepCopyTest._MakeAdd1CountingIdentityFn('Add2')))

      num_copies = 6

      for i in range(num_copies):
        copied = deep_copy.deep_copy(modified)
        # Check labels.
        self.assertEqual(copied.producer.full_label, 'Add2.Copy[%d]' % i)
        self.assertEqual(copied.producer.inputs[0].producer.full_label,
                         'Add1.Copy[%d]' % i)

        # Check resource hints.
        self.assertEqual(modified.producer.resource_hints,
                         {'beam:resources:tags:v1': b'DeepCopy.Original'})
        self.assertEqual(modified.producer.inputs[0].producer.resource_hints,
                         {'beam:resources:tags:v1': b'DeepCopy.Original'})
        self.assertEqual(copied.producer.resource_hints,
                         {'beam:resources:tags:v1': b'DeepCopy.Copy[%d]' % i})
        self.assertEqual(copied.producer.inputs[0].producer.resource_hints,
                         {'beam:resources:tags:v1': b'DeepCopy.Copy[%d]' % i})

      # pylint: disable=expression-not-assigned
      modified | 'Add3' >> beam.Map(
          DeepCopyTest._MakeAdd1CountingIdentityFn('Add3'))
      # pylint: enable=expression-not-assigned

    # Check counts of processed items. Without the materialization boundary,
    # e.g. GroupByKey, PreGroup is also copied.
    self.assertEqual(DeepCopyTest._counts['PreGroup'], 3 * (num_copies + 1))
    self.assertEqual(DeepCopyTest._counts['Add1'], 3 * (num_copies + 1))
    self.assertEqual(DeepCopyTest._counts['Add2'], 3 * (num_copies + 1))
    self.assertEqual(DeepCopyTest._counts['Add3'], 3)

if __name__ == '__main__':
  unittest.main()
