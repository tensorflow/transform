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

import numpy as np

import tensorflow as tf
from tensorflow_transform import mappers
from tensorflow_transform import test_case

mock = tf.compat.v1.test.mock


class MappersTest(test_case.TransformTestCase):

  def assertSparseOutput(self, expected_indices, expected_values,
                         expected_shape, actual_sparse_tensor, close_values):
    actual = self.evaluate(actual_sparse_tensor)
    self.assertAllEqual(expected_indices, actual.indices)
    self.assertAllEqual(expected_shape, actual.dense_shape)
    if close_values:
      self.assertAllClose(expected_values, actual.values)
    else:
      self.assertAllEqual(expected_values, actual.values)

  def testSegmentIndices(self):
    with tf.compat.v1.Graph().as_default():
      with tf.compat.v1.Session():
        self.assertAllEqual(
            mappers.segment_indices(tf.constant([0, 0, 1, 2, 2, 2], tf.int64),
                                    name='test_name').eval(),
            [0, 1, 0, 0, 1, 2])
        self.assertAllEqual(
            mappers.segment_indices(tf.constant([], tf.int64)).eval(),
            [])

  def testSegmentIndicesSkipOne(self):
    with tf.compat.v1.Graph().as_default():
      input_tensor = tf.constant([0, 0, 2, 2])
      with tf.compat.v1.Session():
        self.assertAllEqual([0, 1, 0, 1],
                            mappers.segment_indices(input_tensor).eval())

  def testNGramsEmpty(self):
    with tf.compat.v1.Graph().as_default():
      output_tensor = mappers.ngrams(
          tf.compat.v1.strings.split(tf.constant([''])), (1, 5), '')
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertEqual((0, 2), output.indices.shape)
        self.assertAllEqual([1, 0], output.dense_shape)
        self.assertEqual(0, len(output.values))

  def testNGrams(self):
    with tf.compat.v1.Graph().as_default():
      string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
      tokenized_tensor = tf.compat.v1.string_split(string_tensor, delimiter='')
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
              b'a', b'ab', b'abc', b'b', b'bc', b'c', b'd', b'de', b'def', b'e',
              b'ef', b'f', b'f', b'fg', b'fgh', b'fghi', b'fghij', b'g', b'gh',
              b'ghi', b'ghij', b'ghijk', b'h', b'hi', b'hij', b'hijk', b'hijkl',
              b'i', b'ij', b'ijk', b'ijkl', b'ijklm', b'j', b'jk', b'jkl',
              b'jklm', b'k', b'kl', b'klm', b'l', b'lm', b'm', b'z'
          ],
          expected_shape=[5, 30],
          actual_sparse_tensor=output_tensor,
          close_values=False)

  def testNGramsMinSizeNotOne(self):
    with tf.compat.v1.Graph().as_default():
      string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
      tokenized_tensor = tf.compat.v1.string_split(string_tensor, delimiter='')
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
              b'ab', b'abc', b'bc', b'de', b'def', b'ef', b'fg', b'fgh',
              b'fghi', b'fghij', b'gh', b'ghi', b'ghij', b'ghijk', b'hi',
              b'hij', b'hijk', b'hijkl', b'ij', b'ijk', b'ijkl', b'ijklm',
              b'jk', b'jkl', b'jklm', b'kl', b'klm', b'lm'
          ],
          expected_shape=[5, 22],
          actual_sparse_tensor=output_tensor,
          close_values=False)

  def testNGramsWithSpaceSeparator(self):
    with tf.compat.v1.Graph().as_default():
      string_tensor = tf.constant(['One was Johnny', 'Two was a rat'])
      tokenized_tensor = tf.compat.v1.strings.split(string_tensor, sep=' ')
      output_tensor = mappers.ngrams(
          tokens=tokenized_tensor,
          ngram_range=(1, 2),
          separator=' ')
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertAllEqual(
            output.indices,
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
             [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
        self.assertAllEqual(output.values, [
            b'One', b'One was', b'was', b'was Johnny', b'Johnny', b'Two',
            b'Two was', b'was', b'was a', b'a', b'a rat', b'rat'
        ])
        self.assertAllEqual(output.dense_shape, [2, 7])

  def testNGramsWithRepeatedTokensPerRow(self):
    with tf.compat.v1.Graph().as_default():
      string_tensor = tf.constant(['Cats or dogs or bunnies', 'Cats not rats'])
      tokenized_tensor = tf.compat.v1.strings.split(string_tensor, sep=' ')
      output_tensor = mappers.ngrams(
          tokens=tokenized_tensor, ngram_range=(1, 1), separator=' ')
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertAllEqual(output.indices, [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 0],
            [1, 1],
            [1, 2],
        ])
        # Note: the ngram "or" is represented twice for the first document.
        self.assertAllEqual(output.values, [
            b'Cats', b'or', b'dogs', b'or', b'bunnies', b'Cats', b'not', b'rats'
        ])
        self.assertAllEqual(output.dense_shape, [2, 5])

  def testNGramsBadSizes(self):
    string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
    tokenized_tensor = tf.compat.v1.string_split(string_tensor, delimiter='')
    with self.assertRaisesRegexp(ValueError, 'Invalid ngram_range'):
      mappers.ngrams(tokenized_tensor, (0, 5), separator='')
    with self.assertRaisesRegexp(ValueError, 'Invalid ngram_range'):
      mappers.ngrams(tokenized_tensor, (6, 5), separator='')

  def testNGramsBagOfWordsEmpty(self):
    with tf.compat.v1.Graph().as_default():
      string_tensor = tf.constant([], dtype=tf.string)
      tokenized_tensor = tf.compat.v1.string_split(string_tensor, delimiter='')
      ngrams = mappers.ngrams(tokenized_tensor, (1, 2), separator='')
      bow = mappers.bag_of_words(tokenized_tensor, (1, 2), separator='')
      with tf.compat.v1.Session():
        ngrams_output = ngrams.eval()
        bow_output = bow.eval()
        self.assertAllEqual(ngrams_output.values, [])
        self.assertAllEqual(bow_output.values, [])
        self.assertAllEqual(ngrams_output.dense_shape, [0, 0])
        self.assertAllEqual(bow_output.dense_shape, [0, 0])

  @test_case.named_parameters(
      dict(
          testcase_name='bag_of_words',
          strings=['snakes or dogs and bunnies', 'cats not rats'],
          expected_output_indices=[
              [0, 0],
              [0, 1],
              [0, 2],
              [0, 3],
              [0, 4],
              [1, 0],
              [1, 1],
              [1, 2],
          ],
          expected_output_values=[
              b'snakes', b'or', b'dogs', b'and', b'bunnies', b'cats', b'not',
              b'rats'
          ]),
      dict(
          testcase_name='bag_of_words_duplicates_within_rows',
          strings=['Cats or dogs or bunnies', 'Cats not rats'],
          expected_output_indices=[
              [0, 0],
              [0, 1],
              [0, 2],
              [0, 3],
              [1, 0],
              [1, 1],
              [1, 2],
          ],
          expected_output_values=[
              b'Cats', b'or', b'dogs', b'bunnies', b'Cats', b'not', b'rats'
          ]),
      dict(
          testcase_name='bag_of_words_duplicates_across_rows',
          strings=['cats or dogs or cats', 'cats or dogs'],
          expected_output_indices=[
              [0, 0],
              [0, 1],
              [0, 2],
              [1, 0],
              [1, 1],
              [1, 2],
          ],
          expected_output_values=[
              b'cats', b'or', b'dogs', b'cats', b'or', b'dogs'
          ]),
      dict(
          testcase_name='bag_of_words_some_empty',
          strings=['boots and cats and boots and cats', '', 'cats or dogs', ''],
          expected_output_indices=[
              [0, 0],
              [0, 1],
              [0, 2],
              [2, 0],
              [2, 1],
              [2, 2],
          ],
          expected_output_values=[
              b'boots', b'and', b'cats', b'cats', b'or', b'dogs'
          ]),
      dict(
          testcase_name='bag_of_words_bigrams',
          strings=['i like cats and i like cats to pet', 'i like cats'],
          expected_output_indices=[
              [0, 0],
              [0, 1],
              [0, 2],
              [0, 3],
              [0, 4],
              [0, 5],
              [1, 0],
              [1, 1],
          ],
          # bigrams 'i like' and 'like cats' appear twice in the input but only
          # once in the output for that row.
          expected_output_values=[
              b'i like',
              b'like cats',
              b'cats and',
              b'and i',
              b'cats to',
              b'to pet',
              b'i like',
              b'like cats',
          ],
          ngram_range=[2, 2]),
  )
  def testBagOfWords(self,
                     strings,
                     expected_output_indices,
                     expected_output_values,
                     ngram_range=(1, 1),
                     separator=' '):
    with tf.compat.v1.Graph().as_default():
      string_tensor = tf.constant(strings, dtype=tf.string)
      tokenized_tensor = tf.compat.v1.string_split(
          string_tensor, delimiter=separator)
      output_tensor = mappers.bag_of_words(
          tokens=tokenized_tensor, ngram_range=ngram_range, separator=separator)
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertAllEqual(output.indices, expected_output_indices)
        self.assertAllEqual(output.values, expected_output_values)

  @test_case.named_parameters(
      dict(
          testcase_name='deduplicate_no_op',
          indices=[
              [0, 0],
              [1, 0],
              [1, 1],
              [1, 2],
          ],
          values=[b'foo', b'bar', b'biz', b'buzz'],
          dense_shape=[2, 3],
          expected_output_indices=[
              [0, 0],
              [1, 0],
              [1, 1],
              [1, 2],
          ],
          expected_output_values=[b'foo', b'bar', b'biz', b'buzz'],
          expected_output_shape=[2, 3],
      ),
      dict(
          testcase_name='deduplicate_integers',
          indices=[
              [1, 0],
              [3, 1],
              [3, 2],
              [4, 4],
              [4, 1],
          ],
          values=[1, 1, 1, 0, 0],
          dense_shape=[5, 5],
          expected_output_indices=[
              [1, 0],
              [3, 0],
              [4, 0],
          ],
          expected_output_values=[1, 1, 0],
          expected_output_shape=[5, 1],
      ),
      dict(
          testcase_name='deduplicate_empty_rows',
          indices=[
              [0, 0],
              [2, 1],
              [2, 2],
              [2, 4],
              [4, 1],
          ],
          values=[b'foo', b'bar', b'biz', b'bar', b'foo'],
          dense_shape=[5, 5],
          expected_output_indices=[
              [0, 0],
              [2, 0],
              [2, 1],
              [4, 0],
          ],
          expected_output_values=[b'foo', b'bar', b'biz', b'foo'],
          expected_output_shape=[5, 2],
      ),
      dict(
          testcase_name='deduplicate_shape_change',
          indices=[
              [0, 0],
              [0, 3],
              [1, 0],
              [1, 1],
              [1, 2],
          ],
          values=[b'foo', b'foo', b'bar', b'buzz', b'bar'],
          dense_shape=[2, 4],
          expected_output_indices=[
              [0, 0],
              [1, 0],
              [1, 1],
          ],
          expected_output_values=[b'foo', b'bar', b'buzz'],
          expected_output_shape=[2, 2],
      ))
  def testDedupeSparseTensorPerRow(self, indices, values, dense_shape,
                                   expected_output_indices,
                                   expected_output_values,
                                   expected_output_shape):
    with tf.compat.v1.Graph().as_default():
      sp_input = tf.SparseTensor(
          indices=indices, values=values, dense_shape=dense_shape)
      output_tensor = mappers.deduplicate_tensor_per_row(sp_input)
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertAllEqual(output.indices, expected_output_indices)
        self.assertAllEqual(output.values, expected_output_values)
        self.assertAllEqual(output.dense_shape, expected_output_shape)

  @test_case.named_parameters(
      dict(
          testcase_name='deduplicate_no_op',
          values=[[b'a', b'b'], [b'c', b'd']],
          expected_indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
          expected_output=[b'a', b'b', b'c', b'd'],
      ),
      # Note: because the first dimension is the batch/row dimension, a 1D
      # tensor is always returned as is (since there's only 1 value per row).
      dict(
          testcase_name='deduplicate_1D',
          values=[b'a', b'b', b'a', b'd'],
          expected_indices=[[0, 0], [1, 0], [2, 0], [3, 0]],
          expected_output=[b'a', b'b', b'a', b'd'],
      ),
      dict(
          testcase_name='deduplicate',
          values=[[b'a', b'b', b'a', b'b'], [b'c', b'c', b'd', b'd']],
          expected_indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
          expected_output=[b'a', b'b', b'c', b'd'],
      ),
      dict(
          testcase_name='deduplicate_different_sizes',
          # 2 uniques in the first row, 3 in the second row.
          values=[[b'a', b'b', b'a', b'b'], [b'c', b'a', b'd', b'd']],
          expected_indices=[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
          expected_output=[b'a', b'b', b'c', b'a', b'd'],
      ),
      dict(
          testcase_name='deduplicate_keeps_dups_across_rows',
          values=[[b'a', b'b', b'a', b'b'], [b'b', b'a', b'b', b'b']],
          expected_indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
          expected_output=[b'a', b'b', b'b', b'a'],
      ),
  )
  def testDedupeDenseTensorPerRow(self, values, expected_indices,
                                  expected_output):
    with tf.compat.v1.Graph().as_default():
      dense_input = tf.constant(values)
      output_tensor = mappers.deduplicate_tensor_per_row(dense_input)
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertAllEqual(output.indices, expected_indices)
        self.assertAllEqual(output.values, expected_output)

  def testDedup3dInputRaises(self):
    dense_input = tf.constant([[[b'a', b'a'], [b'b', b'b']],
                               [[b'a', b'a'], [b'd', b'd']]])
    with self.assertRaises(ValueError):
      mappers.deduplicate_tensor_per_row(dense_input)

  def testWordCountEmpty(self):
    with tf.compat.v1.Graph().as_default():
      output_tensor = mappers.word_count(
          tf.compat.v1.string_split(tf.constant([''])))
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertEqual(1, len(output))
        self.assertEqual(0, sum(output))

  def testWordCount(self):
    with tf.compat.v1.Graph().as_default():
      string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
      tokenized_tensor = tf.compat.v1.string_split(string_tensor, delimiter='')
      output_tensor = mappers.word_count(tokenized_tensor)
      output_3d_tensor = mappers.word_count(
          tf.sparse.expand_dims(
              tf.sparse.expand_dims(tokenized_tensor, axis=1), axis=1))
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertEqual(5, len(output))
        self.assertEqual(15, sum(output))
        self.assertAllEqual(output, [3, 3, 8, 1, 0])
        self.assertAllEqual(output, output_3d_tensor.eval())

  def testWordCountRagged(self):
    with tf.compat.v1.Graph().as_default():
      string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
      tokenized_tensor = tf.RaggedTensor.from_sparse(
          tf.compat.v1.string_split(string_tensor, delimiter=''))
      output_tensor = mappers.word_count(tokenized_tensor)
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertEqual(5, len(output))
        self.assertEqual(15, sum(output))
        self.assertAllEqual(output, [3, 3, 8, 1, 0])

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
    with tf.compat.v1.Graph().as_default():
      input_tensor = tf.SparseTensor(
          [[0, 0], [0, 1], [0, 2], [1, 0], [1, 3]],
          [(3/5), (1/5), (1/5), (1/2), (1/2)],
          [2, 4])
      output_tensor = mappers._count_docs_with_term(input_tensor)
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertAllEqual([[2, 1, 1, 1]], output)

  def testCountDocsWithTermUnusedTerm(self):
    with tf.compat.v1.Graph().as_default():
      input_tensor = tf.SparseTensor(
          [[0, 0], [0, 2], [1, 0], [1, 3]],
          [(3/5), (1/5), (1/2), (1/2)],
          [2, 4])
      output_tensor = mappers._count_docs_with_term(input_tensor)
      with tf.compat.v1.Session():
        output = output_tensor.eval()
        self.assertAllEqual([[2, 0, 1, 1]], output)

  def testToTFIDF(self):
    term_freq = tf.SparseTensor(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 3]],
        [(3/5), (1/5), (1/5), (1/2), (1/2)],
        [2, 4])
    reduced_term_freq = tf.constant([[2, 1, 1, 1]])
    output_tensor = mappers._to_tfidf(term_freq, reduced_term_freq,
                                      tf.constant(2), True)
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
    output_tensor = mappers._to_tfidf(term_freq, reduced_term_freq,
                                      tf.constant(2), False)
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
    # TODO(b/123242111): rewrite this test using public functions.
    with tf.compat.v1.Graph().as_default():
      tfidf = tf.SparseTensor(
          values=tf.constant([], shape=[0], dtype=tf.float32),
          indices=tf.constant([], shape=[0, 2], dtype=tf.int64),
          dense_shape=[2, 0])

      _, weights = mappers._split_tfidfs_to_outputs(tfidf)

      with self.test_session() as sess:
        weights_shape = sess.run(weights.dense_shape)
    self.assertAllEqual(weights_shape, [2, 0])

  def testHashStringsNoKeyDenseInput(self):
    with tf.compat.v1.Graph().as_default():
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

  def testHashStringsNoKeyRaggedInput(self):
    strings = tf.RaggedTensor.from_row_splits(
        values=['Dog', 'Cat', ''], row_splits=[0, 1, 1, 1, 1, 3])
    hash_buckets = 17
    expected_hashed_strings = tf.RaggedTensor.from_row_splits(
        values=[12, 4, 11], row_splits=[0, 1, 1, 1, 1, 3])
    hashed_strings = mappers.hash_strings(strings, hash_buckets)
    self.assertAllEqual(expected_hashed_strings, hashed_strings)

  def testHashStringsWithKeyDenseInput(self):
    with tf.compat.v1.Graph().as_default():
      strings = tf.constant(['Cake', 'Pie', 'Sundae'])
      expected_output = [6, 5, 6]
      hash_buckets = 11
      hashed_strings = mappers.hash_strings(
          strings, hash_buckets, key=[123, 456])
      with self.test_session() as sess:
        output = sess.run(hashed_strings)
        self.assertAllEqual(expected_output, output)

  def testHashStringsWithKeySparseInput(self):
    strings = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 1], [1, 1, 0], [2, 1, 0]],
        values=['$$$', '%^#', '&$!#@', '$$$'],
        dense_shape=[3, 3, 2])
    hash_buckets = 173
    expected_indices = strings.indices
    expected_values = [16, 156, 9, 16]
    expected_shape = strings.dense_shape
    hashed_strings = mappers.hash_strings(strings, hash_buckets, key=[321, 555])
    self.assertSparseOutput(
        expected_indices=expected_indices,
        expected_values=expected_values,
        expected_shape=expected_shape,
        actual_sparse_tensor=hashed_strings,
        close_values=False)

  def testHashStringsWithKeyRaggedInput(self):
    strings = tf.RaggedTensor.from_row_splits(
        values=['$$$', '%^#', '&$!#@', '$$$'], row_splits=[0, 1, 1, 2, 2, 4])
    hash_buckets = 173
    expected_hashed_strings = tf.RaggedTensor.from_row_splits(
        values=[16, 156, 9, 16], row_splits=[0, 1, 1, 2, 2, 4])
    hashed_strings = mappers.hash_strings(strings, hash_buckets, key=[321, 555])
    self.assertAllEqual(expected_hashed_strings, hashed_strings)

  @test_case.named_parameters(
      dict(
          testcase_name='few_buckets',
          x=4,
          bucket_boundaries=[[5]],
          expected_buckets=0),
      dict(
          testcase_name='large_buckets',
          x=50_000_000,
          bucket_boundaries=[[0, 50_000_001, 100_000_001]],
          expected_buckets=1),
      dict(
          testcase_name='with_nans',
          x=[4.0, float('nan'), float('-inf'), 7.5, 10.0],
          bucket_boundaries=[[2, 5, 8]],
          expected_buckets=[1, 3, 0, 2, 3]),
      dict(
          testcase_name='with_inf_boundary',
          x=[4.0, float('-inf'), .8, 7.5, 10.0],
          bucket_boundaries=[[float('-inf'), 2, 5, 8]],
          expected_buckets=[2, 1, 1, 3, 4]),
  )
  def testApplyBuckets(self, x, bucket_boundaries, expected_buckets):
    x = tf.constant(x)
    bucket_boundaries = tf.constant(bucket_boundaries)
    expected_buckets = tf.constant(expected_buckets, dtype=tf.int64)
    buckets = mappers.apply_buckets(x, bucket_boundaries)
    self.assertAllEqual(buckets, expected_buckets)

  def testApplybucketsToSparseTensor(self):
    inputs = tf.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 1], [2, 2, 2]],
        values=[10, 20, -1],
        dense_shape=[3, 3, 4])
    quantiles = [-10, 0, 13]
    bucketized = mappers.apply_buckets(inputs, [quantiles])
    self.assertSparseOutput(
        inputs.indices,
        tf.constant([2, 3, 1]),
        inputs.dense_shape,
        bucketized,
        close_values=False)

  def testApplybucketsToRaggedTensor(self):
    inputs = tf.RaggedTensor.from_row_splits(
        values=tf.RaggedTensor.from_row_splits(
            values=[10, 20, -1], row_splits=[0, 1, 1, 2, 2, 3]),
        row_splits=[0, 1, 1, 2, 3, 5])
    quantiles = [-10, 0, 13]
    expected_bucketized = tf.RaggedTensor.from_row_splits(
        values=tf.RaggedTensor.from_row_splits(
            values=[2, 3, 1], row_splits=[0, 1, 1, 2, 2, 3]),
        row_splits=[0, 1, 1, 2, 3, 5])
    bucketized = mappers.apply_buckets(inputs, [quantiles])
    self.assertAllEqual(expected_bucketized, bucketized)

  def testApplyBucketsWithKeys(self):
    with tf.compat.v1.Graph().as_default():
      values = tf.constant([
          -100, -0.05, 0.05, 0.25, 0.15, 100, -100, 0, 4.3, 4.5, 4.4, 4.6, 100
      ],
                           dtype=tf.float32)
      keys = tf.constant([
          'a', 'a', 'a', 'a', 'a', 'a', 'b', 'missing', 'b', 'b', 'b', 'b', 'b'
      ])
      key_vocab = tf.constant(['a', 'b'])
      # Pre-normalization boundaries: [[0, 0.1, 0.2], [4.33, 4.43, 4.53]]
      bucket_boundaries = tf.constant([0.0, 0.5, 1.0, 1.5, 2.0],
                                      dtype=tf.float32)
      scales = 1.0 / (
          tf.constant([0.2, 4.53], dtype=tf.float32) -
          tf.constant([0, 4.33], dtype=tf.float32))
      shifts = tf.constant([0, 1.0 - (4.33 * 5)], dtype=tf.float32)
      num_buckets = tf.constant(4, dtype=tf.int64)
      buckets = mappers._apply_buckets_with_keys(values, keys, key_vocab,
                                                 bucket_boundaries, scales,
                                                 shifts, num_buckets)
      with self.test_session() as sess:
        sess.run(tf.compat.v1.tables_initializer())
        output = sess.run(buckets)
        self.assertAllEqual([0, 0, 1, 3, 2, 3, 0, -1, 0, 2, 1, 3, 3], output)

  @test_case.named_parameters(
      dict(
          testcase_name='single_input_value',
          x=1,
          boundaries=[0, 2],
          expected_results=.5),
      dict(
          testcase_name='single_boundary',
          x=[-1, 9, 10, 11],
          boundaries=[10],
          expected_results=[0, 0, 1, 1]),
      dict(
          testcase_name='out_of_bounds',
          x=[-1111, 0, 5, 9, 10, 11, 15, 19, 20, 21, 1111],
          boundaries=[10, 20],
          expected_results=[0, 0, 0, 0, 0, .1, 0.5, .9, 1, 1, 1]),
      dict(
          testcase_name='2d_input',
          x=[[15, 10], [20, 17], [-1111, 21]],
          boundaries=[10, 20],
          expected_results=[[0.5, 0], [1, .7], [0, 1]]),
      dict(
          testcase_name='integer_input',
          x=[15, 20, 25],
          boundaries=[10, 20],
          expected_results=[.5, 1, 1],
          input_dtype=tf.int64),
      dict(
          testcase_name='float_input',
          x=[-10, 0, 0.1, 2.3, 4.5, 6.7, 8.9, 10, 100],
          boundaries=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          expected_results=[0, 0, 0.01, 0.23, 0.45, 0.67, 0.89, 1, 1]),
      dict(
          testcase_name='float_input_with_nans',
          x=[
              float('-inf'), -10, 0, 0.1, 2.3,
              float('nan'), 4.5, 6.7, 8.9, 10, 100,
              float('inf')
          ],
          boundaries=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          expected_results=[0, 0, 0, 0.01, 0.23, .5, 0.45, 0.67, 0.89, 1, 1,
                            1]),
      dict(
          testcase_name='float_input_with_inf_boundaries',
          x=[
              float('-inf'),
              float('-inf'),
              float(0),
              float('-inf'),
          ],
          boundaries=[float('-inf'), 0],
          expected_results=[0, 0, 1, 0]),
      dict(
          testcase_name='float_input_with_nan_boundaries',
          x=[
              float('-inf'),
              float('nan'),
              float(0),
              float(1),
          ],
          boundaries=[float('nan'), 0, 1],
          expected_results=[0, .5, 0, 1]),
      dict(
          testcase_name='integer_boundaries',
          x=[15, 20, 25],
          boundaries=[10, 20],
          expected_results=[.5, 1, 1],
          boundaries_dtype=tf.int64),
      dict(
          testcase_name='negative_boundaries',
          x=[-10, -5, -3, 0, 2, 4, 8, 12, 18],
          boundaries=[-20, -4, 1, 4, 20],
          expected_results=[
              0.15625, 0.234375, .3, .45, 0.583333, .75, 0.8125, .875, 0.96875
          ]),
      dict(
          testcase_name='interpolates_properly',
          x=[-1111, 10, 50, 100, 1000, 9000, 10000, 1293817391],
          boundaries=[10, 100, 1000, 10000],
          expected_results=[
              0, 0, (4.0 / 9 / 3), (1.0 / 3), (2.0 / 3), ((2 + 8.0 / 9) / 3), 1,
              1
          ],
          boundaries_dtype=tf.int64),
  )
  def testApplyBucketsWithInterpolation(self,
                                        x,
                                        boundaries,
                                        expected_results,
                                        input_dtype=tf.float32,
                                        boundaries_dtype=tf.float32):
    with tf.compat.v1.Graph().as_default():
      with self.test_session() as sess:
        x = tf.constant(x, dtype=input_dtype)
        boundaries = tf.constant([boundaries], dtype=boundaries_dtype)
        output = mappers.apply_buckets_with_interpolation(x, boundaries)
        self.assertAllClose(sess.run(output), expected_results, 1e-6)

  def testApplyBucketsWithInterpolationAllNanBoundariesRaises(self):
    with tf.compat.v1.Graph().as_default():
      with self.test_session() as sess:
        x = tf.constant([float('-inf'), float('nan'), 0.0, 1.0])
        boundaries = tf.constant([[float('nan'), float('nan'), float('nan')]])
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                     'num_boundaries'):
          sess.run(mappers.apply_buckets_with_interpolation(x, boundaries))

  def testApplyBucketsWithInterpolationRaises(self):
    # We should raise an exception if you try to scale a non-numeric tensor.
    with self.test_session():
      x = tf.constant(['a', 'b', 'c'], dtype=tf.string)
      boundaries = tf.constant([.2, .4], dtype=tf.float32)
      with self.assertRaises(ValueError):
        mappers.apply_buckets_with_interpolation(x, boundaries)

  def testApplyBucketsWithInterpolationSparseTensor(self):
    with tf.compat.v1.Graph().as_default():
      with self.test_session() as sess:
        x = tf.SparseTensor(
            indices=[[0, 0, 0], [1, 1, 2], [3, 1, 4], [1, 1, 4], [6, 1, 1],
                     [3, 1, 2]],
            values=[15, 10, 20, 17, -1111, 21],
            dense_shape=[7, 3, 5])
        boundaries = [[10, 20]]
        output = mappers.apply_buckets_with_interpolation(x, boundaries)
        expected_results = tf.SparseTensor(
            indices=x.indices,
            values=[.5, 0, 1, .7, 0, 1],
            dense_shape=x.dense_shape)
        actual_results = sess.run(output)
        self.assertAllClose(actual_results.values,
                            expected_results.values,
                            1e-6)
        self.assertAllEqual(actual_results.indices, expected_results.indices)
        self.assertAllEqual(actual_results.dense_shape,
                            expected_results.dense_shape)

  def testApplyBucketsWithInterpolationRaggedTensor(self):
    inputs = tf.RaggedTensor.from_row_splits(
        values=[15, 10, 20, 17, -1111, 21], row_splits=[0, 1, 1, 2, 4, 5, 6])
    boundaries = [[10, 20]]
    expected_bucketized = tf.RaggedTensor.from_row_splits(
        values=[.5, 0, 1, .7, 0, 1], row_splits=[0, 1, 1, 2, 4, 5, 6])
    bucketized = mappers.apply_buckets_with_interpolation(inputs, boundaries)
    self.assertAllEqual(expected_bucketized, bucketized)

  def testBucketsWithInterpolationUnknownShapeBoundary(self):
    with tf.compat.v1.Graph().as_default():
      with self.test_session() as sess:
        x = tf.constant([0, 1, 5, 12], dtype=tf.float32)
        # The shape used to generate the boundaries is random, and therefore
        # the size of the boundaries tensor is not known.
        num_boundaries = tf.random.uniform([1], 1, 2, dtype=tf.int64)[0]
        boundaries = tf.random.uniform([1, num_boundaries], 0, 10)
        # We don't assert anything about the outcome because we're intentionally
        # using randomized boundaries, but we ensure the operations succeed.
        _ = sess.run(mappers.apply_buckets_with_interpolation(x, boundaries))

  def testSparseTensorToDenseWithShape(self):
    with tf.compat.v1.Graph().as_default():
      sparse = tf.compat.v1.sparse_placeholder(
          tf.int64, shape=[None, None, None])
      dense = mappers.sparse_tensor_to_dense_with_shape(sparse, [None, 5, 6])
      self.assertAllEqual(dense.get_shape().as_list(), [None, 5, 6])

  def testSparseTensorLeftAlign(self):
    with tf.compat.v1.Graph().as_default():
      with self.test_session() as sess:
        x = tf.SparseTensor(
            indices=[[0, 3], [1, 2], [1, 4], [3, 2], [3, 4], [5, 0], [6, 1]],
            values=[15, 10, 20, 17, -1111, 13, 21],
            dense_shape=[7, 5])
        y = mappers.sparse_tensor_left_align(x)
        expected_indices = [[0, 0], [1, 0], [1, 1], [3, 0], [3, 1], [5, 0],
                            [6, 0]]
        self.assertAllEqual(sess.run(y.indices), expected_indices)

  def testEstimatedProbabilityDensityMissingKey(self):
    input_size = 5

    with tf.compat.v1.Graph().as_default():
      input_data = tf.constant([[str(x + 1)] for x in range(input_size)])

      count = tf.constant([3] * input_size, tf.int64)
      boundaries = tf.as_string(tf.range(input_size))
      with mock.patch.object(
          mappers.analyzers, 'histogram', side_effect=[(count, boundaries)]):

        result = mappers.estimated_probability_density(
            input_data, categorical=True)

      expected = np.array([[0.2], [0.2], [0.2], [0.2], [0.]], np.float32)
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.tables_initializer())
        self.assertAllEqual(expected, sess.run(result))


if __name__ == '__main__':
  test_case.main()
