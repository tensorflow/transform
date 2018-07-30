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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


import apache_beam as beam
from tensorflow_transform.beam import deep_copy
import unittest


# pylint: disable=g-long-lambda
class DeepCopyTest(unittest.TestCase):

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
  def _InitializeCounts():
    DeepCopyTest._counts = collections.defaultdict(int)

  def setUp(self):
    DeepCopyTest._InitializeCounts()

  def testBasicDeepCopy(self):
    with beam.Pipeline() as p:
      grouped = (p
                 | beam.Create([(1, 'a'), (2, 'b'), (3, 'c')])
                 | beam.Map(
                     lambda x: DeepCopyTest._CountingIdentityFn(
                         'PreGroup', x))
                 | beam.GroupByKey())
      modified = (grouped
                  | 'Add1' >> beam.Map(
                      lambda (x, y): DeepCopyTest._CountingIdentityFn(
                          'Add1', (x+1, y)))
                  | 'Add2' >> beam.Map(
                      lambda (x, y): DeepCopyTest._CountingIdentityFn(
                          'Add2', (x+1, y))))
      copied = deep_copy.deep_copy(modified)

      # pylint: disable=expression-not-assigned
      (modified | 'Add3' >> beam.Map(
          lambda (x, y): DeepCopyTest._CountingIdentityFn('Add3', (x+1, y))))
      # pylint: enable=expression-not-assigned

      # Check labels.
      self.assertEqual(copied.producer.full_label, 'Add2.Copy')
      self.assertEqual(copied.producer.inputs[0].producer.full_label,
                       'Add1.Copy')

      # Check that deep copy was performed.
      self.assertNotEqual(copied.producer.inputs[0],
                          modified.producer.inputs[0])

      # Check that copy stops at materialization boundary.
      self.assertEqual(copied.producer.inputs[0].producer.inputs[0],
                       modified.producer.inputs[0].producer.inputs[0])

    # Check counts of processed items.
    self.assertEqual(DeepCopyTest._counts['PreGroup'], 3)
    self.assertEqual(DeepCopyTest._counts['Add1'], 6)
    self.assertEqual(DeepCopyTest._counts['Add2'], 6)
    self.assertEqual(DeepCopyTest._counts['Add3'], 3)

  def testMultipleCopies(self):
    with beam.Pipeline() as p:
      grouped = (p
                 | beam.Create([(1, 'a'), (2, 'b'), (3, 'c')])
                 | beam.Map(lambda x: DeepCopyTest._CountingIdentityFn(
                     'PreGroup', x))
                 | beam.GroupByKey())
      modified = (grouped
                  | 'Add1' >> beam.Map(
                      lambda (x, y): DeepCopyTest._CountingIdentityFn(
                          'Add1', (x+1, y)))
                  | 'Add2' >> beam.Map(
                      lambda (x, y): DeepCopyTest._CountingIdentityFn(
                          'Add2', (x+1, y))))

      num_copies = 6

      first_copy = deep_copy.deep_copy(modified)
      self.assertEqual(first_copy.producer.full_label, 'Add2.Copy')
      self.assertEqual(first_copy.producer.inputs[0].producer.full_label,
                       'Add1.Copy')

      for i in range(num_copies - 1):
        copied = deep_copy.deep_copy(modified)
        self.assertEqual(copied.producer.full_label, 'Add2.Copy%d' % i)
        self.assertEqual(copied.producer.inputs[0].producer.full_label,
                         'Add1.Copy%d' % i)

    self.assertEqual(DeepCopyTest._counts['PreGroup'], 3)
    self.assertEqual(DeepCopyTest._counts['Add1'], 3 * (num_copies + 1))
    self.assertEqual(DeepCopyTest._counts['Add2'], 3 * (num_copies + 1))

  def testFlatten(self):
    with beam.Pipeline() as p:
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
      modified1 = (grouped1
                   | 'Add1' >> beam.Map(
                       lambda (x, y): DeepCopyTest._CountingIdentityFn(
                           'Add1', (x+1, y))))
      modified2 = (grouped2
                   | 'Add2' >> beam.Map(
                       lambda (x, y): DeepCopyTest._CountingIdentityFn(
                           'Add2', (x+1, y))))
      flattened = (modified1, modified2) | 'Flatten2' >> beam.Flatten()
      modified3 = (flattened
                   | 'Add3' >> beam.Map(
                       lambda (x, y): DeepCopyTest._CountingIdentityFn(
                           'Add3', (x+1, y))))

      copied = deep_copy.deep_copy(modified3)

      # Check that deep copy was performed.
      self.assertNotEqual(copied.producer.inputs[0],
                          modified3.producer.inputs[0])
      self.assertNotEqual(copied.producer.inputs[0].producer.inputs[0],
                          modified3.producer.inputs[0].producer.inputs[0])
      self.assertNotEqual(copied.producer.inputs[0].producer.inputs[1],
                          modified3.producer.inputs[0].producer.inputs[1])

      # Check that copy stops at materialization boundary.
      self.assertEqual(
          copied.producer.inputs[0].producer.inputs[0].producer.inputs[0],
          modified3.producer.inputs[0].producer.inputs[0].producer.inputs[0])
      self.assertEqual(
          copied.producer.inputs[0].producer.inputs[1].producer.inputs[0],
          modified3.producer.inputs[0].producer.inputs[1].producer.inputs[0])

    # Check counts of processed items.
    self.assertEqual(DeepCopyTest._counts['PreGroup1'], 3)
    self.assertEqual(DeepCopyTest._counts['PreGroup2'], 3)
    self.assertEqual(DeepCopyTest._counts['Add1'], 6)
    self.assertEqual(DeepCopyTest._counts['Add2'], 6)
    self.assertEqual(DeepCopyTest._counts['Add3'], 12)

  def testCombineGlobally(self):
    with beam.Pipeline() as p:
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
      self.assertNotEqual(combined, copied)
      self.assertNotEqual(combined.producer.inputs[0],
                          copied.producer.inputs[0])
      self.assertEqual(copied.producer.inputs[0].producer.full_label,
                       'CombineGlobally(MeanCombineFn)/UnKey.Copy')
      self.assertNotEqual(combined.producer.inputs[0].producer.inputs[0],
                          copied.producer.inputs[0].producer.inputs[0])

      # Check that deep copy stops at materialization boundary.
      self.assertEqual(
          combined.producer.inputs[0].producer.inputs[0].producer,
          copied.producer.inputs[0].producer.inputs[0].producer)
      self.assertEqual(
          copied.producer.inputs[0].producer.inputs[0].producer.full_label,
          ('CombineGlobally(MeanCombineFn)/CombinePerKey/Combine/'
           'ParDo(CombineValuesDoFn)'))

    # Check counts of processed items.
    self.assertEqual(DeepCopyTest._counts['PreCombine'], 3)
    self.assertEqual(DeepCopyTest._counts['PostCombine'], 2)


if __name__ == '__main__':
  unittest.main()
