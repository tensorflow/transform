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
"""Tests for tensorflow_transform.mappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow_transform import mappers

import unittest
from tensorflow.python.framework import test_util


class MappersTest(test_util.TensorFlowTestCase):

  def assertSparseOutput(self, expected_indices, expected_values,
                         expected_shape, actual_sparse_tensor, close_values):
    with tf.Session() as sess:
      sess.run(tf.tables_initializer())
      actual = actual_sparse_tensor.eval()
      self.assertAllEqual(expected_indices, actual.indices)
      self.assertAllEqual(expected_shape, actual.dense_shape)
      if close_values:
        self.assertAllClose(expected_values, actual.values)
      else:
        self.assertAllEqual(expected_values, actual.values)

  def testSegmentIndices(self):
    with tf.Session():
      self.assertAllEqual(
          mappers.segment_indices(tf.constant([0, 0, 1, 2, 2, 2], tf.int64),
                                  name='test_name').eval(),
          [0, 1, 0, 0, 1, 2])
      self.assertAllEqual(
          mappers.segment_indices(tf.constant([], tf.int64)).eval(),
          [])

  def testSegmentIndicesSkipOne(self):
    input_tensor = tf.constant([0, 0, 2, 2])
    with tf.Session():
      self.assertAllEqual([0, 1, 0, 1],
                          mappers.segment_indices(input_tensor).eval())

  def testNGramsEmpty(self):
    output_tensor = mappers.ngrams(tf.string_split(tf.constant([''])),
                                   (1, 5), '')
    with tf.Session():
      output = output_tensor.eval()
      self.assertEqual((0, 2), output.indices.shape)
      self.assertAllEqual([1, 0], output.dense_shape)
      self.assertEqual(0, len(output.values))

  def testNGrams(self):
    string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
    tokenized_tensor = tf.string_split(string_tensor, delimiter='')
    output_tensor = mappers.ngrams(
        tokens=tokenized_tensor,
        ngram_range=(1, 5),
        separator='')
    self.assertSparseOutput(
        expected_indices=[
            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
            [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7],
            [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14],
            [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21],
            [2, 22], [2, 23], [2, 24], [2, 25], [2, 26], [2, 27], [2, 28],
            [2, 29], [3, 0]],
        expected_values=[
            'a', 'ab', 'abc', 'b', 'bc', 'c',
            'd', 'de', 'def', 'e', 'ef', 'f',
            'f', 'fg', 'fgh', 'fghi', 'fghij', 'g', 'gh', 'ghi', 'ghij',
            'ghijk', 'h', 'hi', 'hij', 'hijk', 'hijkl', 'i', 'ij', 'ijk',
            'ijkl', 'ijklm', 'j', 'jk', 'jkl', 'jklm', 'k', 'kl', 'klm', 'l',
            'lm', 'm', 'z'],
        expected_shape=[5, 30],
        actual_sparse_tensor=output_tensor,
        close_values=False)

  def testNGramsMinSizeNotOne(self):
    string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
    tokenized_tensor = tf.string_split(string_tensor, delimiter='')
    output_tensor = mappers.ngrams(
        tokens=tokenized_tensor,
        ngram_range=(2, 5),
        separator='')
    self.assertSparseOutput(
        expected_indices=[
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7],
            [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14],
            [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21]],
        expected_values=[
            'ab', 'abc', 'bc',
            'de', 'def', 'ef',
            'fg', 'fgh', 'fghi', 'fghij', 'gh', 'ghi', 'ghij', 'ghijk',
            'hi', 'hij', 'hijk', 'hijkl', 'ij', 'ijk', 'ijkl', 'ijklm',
            'jk', 'jkl', 'jklm', 'kl', 'klm', 'lm'],
        expected_shape=[5, 22],
        actual_sparse_tensor=output_tensor,
        close_values=False)

  def testNGramsWithSpaceSeparator(self):
    string_tensor = tf.constant(['One was Johnny', 'Two was a rat'])
    tokenized_tensor = tf.string_split(string_tensor, delimiter=' ')
    output_tensor = mappers.ngrams(
        tokens=tokenized_tensor,
        ngram_range=(1, 2),
        separator=' ')
    with tf.Session():
      output = output_tensor.eval()
      self.assertAllEqual(
          output.indices,
          [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
           [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
      self.assertAllEqual(output.values, [
          'One', 'One was', 'was', 'was Johnny', 'Johnny',
          'Two', 'Two was', 'was', 'was a', 'a', 'a rat', 'rat'])
      self.assertAllEqual(output.dense_shape, [2, 7])

  def testNGramsBadSizes(self):
    string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
    tokenized_tensor = tf.string_split(string_tensor, delimiter='')
    with self.assertRaisesRegexp(ValueError, 'Invalid ngram_range'):
      mappers.ngrams(tokenized_tensor, (0, 5), separator='')
    with self.assertRaisesRegexp(ValueError, 'Invalid ngram_range'):
      mappers.ngrams(tokenized_tensor, (6, 5), separator='')

  def testTermFrequency(self):
    input_tensor = tf.SparseTensor(
        [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1]],
        [1, 2, 0, 0, 0, 3, 0],
        [2, 5])
    self.assertSparseOutput(
        expected_indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 3]],
        expected_values=[(3/5), (1/5), (1/5), (1/2), (1/2)],
        expected_shape=[2, 4],
        actual_sparse_tensor=mappers._to_term_frequency(input_tensor, 4),
        close_values=True)

  def testTermFrequencyUnusedTerm(self):
    input_tensor = tf.SparseTensor(
        [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1]],
        [4, 2, 0, 0, 0, 3, 0],
        [2, 5])
    self.assertSparseOutput(
        expected_indices=[[0, 0], [0, 2], [0, 4], [1, 0], [1, 3]],
        expected_values=[(3/5), (1/5), (1/5), (1/2), (1/2)],
        expected_shape=[2, 5],
        actual_sparse_tensor=mappers._to_term_frequency(input_tensor, 5),
        close_values=True)

  def testCountDocsWithTerm(self):
    input_tensor = tf.SparseTensor(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 3]],
        [(3/5), (1/5), (1/5), (1/2), (1/2)],
        [2, 4])
    output_tensor = mappers._count_docs_with_term(input_tensor)
    with tf.Session():
      output = output_tensor.eval()
      self.assertAllEqual([[2, 1, 1, 1]], output)

  def testCountDocsWithTermUnusedTerm(self):
    input_tensor = tf.SparseTensor(
        [[0, 0], [0, 2], [1, 0], [1, 3]],
        [(3/5), (1/5), (1/2), (1/2)],
        [2, 4])
    output_tensor = mappers._count_docs_with_term(input_tensor)
    with tf.Session():
      output = output_tensor.eval()
      self.assertAllEqual([[2, 0, 1, 1]], output)

  def testToTFIDF(self):
    term_freq = tf.SparseTensor(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 3]],
        [(3/5), (1/5), (1/5), (1/2), (1/2)],
        [2, 4])
    reduced_term_freq = tf.constant([[2, 1, 1, 1]])
    output_tensor = mappers._to_tfidf(term_freq, reduced_term_freq, 2, True)
    log_3_over_2 = 1.4054651
    self.assertSparseOutput(
        expected_indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 3]],
        expected_values=[(3/5), (1/5)*log_3_over_2, (1/5)*log_3_over_2,
                         (1/2), (1/2)*log_3_over_2],
        expected_shape=[2, 4],
        actual_sparse_tensor=output_tensor,
        close_values=True)

  def testToTFIDFNotSmooth(self):
    term_freq = tf.SparseTensor(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 3]],
        [(3/5), (1/5), (1/5), (1/2), (1/2)],
        [2, 4])
    reduced_term_freq = tf.constant([[2, 1, 1, 1]])
    output_tensor = mappers._to_tfidf(term_freq, reduced_term_freq, 2, False)
    log_2_over_1 = 1.6931471
    self.assertSparseOutput(
        expected_indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 3]],
        expected_values=[(3/5), (1/5)*log_2_over_1, (1/5)*log_2_over_1,
                         (1/2), (1/2)*log_2_over_1],
        expected_shape=[2, 4],
        actual_sparse_tensor=output_tensor,
        close_values=True)

  def testSplitTFIDF(self):
    tfidfs = tf.SparseTensor(
        [[0, 0], [0, 1], [2, 1], [2, 2]],
        [0.23104906, 0.19178806, 0.14384104, 0.34657359],
        [3, 4])

    out_index, out_weight = mappers._split_tfidfs_to_outputs(tfidfs)
    self.assertSparseOutput(
        expected_indices=[[0, 0], [0, 1], [2, 0], [2, 1]],
        expected_values=[0, 1, 1, 2],
        expected_shape=[3, 2],
        actual_sparse_tensor=out_index,
        close_values=False)
    self.assertSparseOutput(
        expected_indices=[[0, 0], [0, 1], [2, 0], [2, 1]],
        expected_values=[0.23104906, 0.19178806, 0.14384104, 0.34657359],
        expected_shape=[3, 2],
        actual_sparse_tensor=out_weight,
        close_values=True)

  def testSplitTFIDFWithEmptyInput(self):
    with tf.Graph().as_default():
      tfidf = tf.SparseTensor(
          values=tf.constant([], shape=[0], dtype=tf.float32),
          indices=tf.constant([], shape=[0, 2], dtype=tf.int64),
          dense_shape=[2, 0])

      _, weights = mappers._split_tfidfs_to_outputs(tfidf)

      with self.test_session() as sess:
        weights_shape = sess.run(weights.dense_shape)
    self.assertAllEqual(weights_shape, [2, 0])

  def testHashStringsNoKeyDenseInput(self):
    strings = tf.constant(['Car', 'Bus', 'Tree'])
    expected_output = [8, 4, 5]

    hash_buckets = 11
    hashed_strings = mappers.hash_strings(strings, hash_buckets)
    with self.test_session() as sess:
      output = sess.run(hashed_strings)
      self.assertAllEqual(expected_output, output)

  def testHashStringsNoKeySparseInput(self):
    strings = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]],
                              values=['Dog', 'Cat', ''],
                              dense_shape=[2, 2])
    hash_buckets = 17
    expected_indices = [[0, 0], [0, 1], [1, 0]]
    expected_values = [12, 4, 11]
    expected_shape = [2, 2]
    hashed_strings = mappers.hash_strings(strings, hash_buckets)
    self.assertSparseOutput(
        expected_indices=expected_indices,
        expected_values=expected_values,
        expected_shape=expected_shape,
        actual_sparse_tensor=hashed_strings,
        close_values=False)

  def testHashStringsWithKeyDenseInput(self):
    strings = tf.constant(['Cake', 'Pie', 'Sundae'])
    expected_output = [6, 5, 6]
    hash_buckets = 11
    hashed_strings = mappers.hash_strings(strings, hash_buckets, key=[123, 456])
    with self.test_session() as sess:
      output = sess.run(hashed_strings)
      self.assertAllEqual(expected_output, output)

  def testHashStringsWithKeySparseInput(self):
    strings = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
                              values=['$$$', '%^#', '&$!#@', '$$$'],
                              dense_shape=[3, 2])
    hash_buckets = 173
    expected_indices = [[0, 0], [0, 1], [1, 0], [2, 0]]
    expected_values = [16, 156, 9, 16]
    expected_shape = [3, 2]
    hashed_strings = mappers.hash_strings(strings, hash_buckets, key=[321, 555])
    self.assertSparseOutput(
        expected_indices=expected_indices,
        expected_values=expected_values,
        expected_shape=expected_shape,
        actual_sparse_tensor=hashed_strings,
        close_values=False)

  def testLookupKey(self):
    keys = tf.constant(['a', 'a', 'a', 'b', 'b', 'b', 'b'])
    key_vocab = tf.constant(['a', 'b'])
    key_indices = mappers._lookup_key(keys, key_vocab)
    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      output = sess.run(key_indices)
      self.assertAllEqual([0, 0, 0, 1, 1, 1, 1], output)

  def testStackBucketBoundaries(self):
    bucket_boundaries = tf.constant([[0, .1, .2], [.1, .2, .3]],
                                    dtype=tf.float32)
    combined_boundaries, offsets = mappers._combine_bucket_boundaries(
        bucket_boundaries, epsilon=0.03)
    with self.test_session() as sess:
      self.assertAllClose([0, 0.1, 0.2, 0.23, 0.33, 0.43],
                          sess.run(combined_boundaries))
      self.assertAllClose([0, 0.13], sess.run(offsets))

  def testApplyBucketsWithKeys(self):
    values = tf.constant(
        [-100, -0.05, 0.05, 0.25, 0.15, 100, -100, 4.3, 4.5, 4.4, 4.6, 100],
        dtype=tf.float32)
    keys = tf.constant(
        ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b'])
    key_vocab = tf.constant(['a', 'b'])
    bucket_boundaries = tf.constant([[0, .1, .2], [4.33, 4.43, 4.53]],
                                    dtype=tf.float32)
    buckets = mappers._apply_buckets_with_keys(values, keys, key_vocab,
                                               bucket_boundaries)
    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      output = sess.run(buckets)
      self.assertAllEqual([0, 0, 1, 3, 2, 3, 0, 0, 2, 1, 3, 3], output)

  def testSparseTensorToDenseWithShape(self):
    with tf.Graph().as_default():
      sparse = tf.sparse_placeholder(tf.int64, shape=[None, None])
      dense = mappers.sparse_tensor_to_dense_with_shape(sparse, [None, 5])
      self.assertAllEqual(dense.get_shape().as_list(), [None, 5])


if __name__ == '__main__':
  unittest.main()
