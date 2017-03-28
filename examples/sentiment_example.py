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
"""Example of sentiment analysis using IMDB movie review dataset."""

# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import tempfile

import tensorflow as tf
import tensorflow_transform as tft
from apache_beam.io import textio
from apache_beam.io import tfrecordio
from tensorflow.contrib import learn
from tensorflow.contrib.layers import feature_column
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io

import apache_beam as beam

VOCAB_SIZE = 20000
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_EPOCHS = 200
NUM_TRAIN_INSTANCES = 25000
NUM_TEST_INSTANCES = 25000

REVIEW_COLUMN = 'review'
LABEL_COLUMN = 'label'

PUNCTUATION_CHARACTERS = ['.', ',', '!', '?', '(', ')']


# pylint: disable=invalid-name
@beam.ptransform_fn
def Shuffle(pcoll):
  """Shuffles a PCollection.  Collection should not contain duplicates."""
  return (pcoll
          | 'PairWithHash' >> beam.Map(lambda x: (hash(x), x))
          | 'GroupByHash' >> beam.GroupByKey()
          | 'DropHash' >> beam.FlatMap(lambda (k, vs): vs))


# pylint: disable=invalid-name
@beam.ptransform_fn
def ReadAndShuffleData(pcoll, filepatterns):
  """Read a train or test dataset from disk and shuffle it."""
  # NOTE: we pass filepatterns as a tuple instead of two args, as the current
  # version of beam assumes that if the first arg to a ptransfrom_fn is a
  # string, then that string is the label.
  neg_filepattern, pos_filepattern = filepatterns

  # Read from each file pattern and create a tuple of the review text and the
  # correct label.
  negative_examples = (
      pcoll
      | 'ReadNegativeExamples' >> textio.ReadFromText(neg_filepattern)
      | 'PairWithZero' >> beam.Map(lambda review: (review, 0)))
  positive_examples = (
      pcoll
      | 'ReadPositiveExamples' >> textio.ReadFromText(pos_filepattern)
      | 'PairWithOne' >> beam.Map(lambda review: (review, 1)))
  all_examples = (
      [negative_examples, positive_examples] | 'Merge' >> beam.Flatten())

  # Shuffle the data.  Note that the data does in fact contain duplicate reviews
  # for reasons that are unclear.  This means that NUM_TRAIN_INSTANCES and
  # NUM_TRAIN_INSTANCES are slightly wrong for the preprocessed data.
  # pylint: disable=no-value-for-parameter
  shuffled_examples = (
      all_examples
      | 'RemoveDuplicates' >> beam.RemoveDuplicates()
      | 'Shuffle' >> Shuffle())

  # Put the data in the format that can be accepted directly by tf.Transform.
  return shuffled_examples | 'MakeInstances' >> beam.Map(
      lambda p: {REVIEW_COLUMN: p[0], LABEL_COLUMN: p[1]})


def transform_data(train_neg_filepattern, train_pos_filepattern,
                   test_neg_filepattern, test_pos_filepattern,
                   transformed_train_filebase, transformed_test_filebase,
                   transformed_metadata_dir):
  """Transform the data and write out as a TFRecord of Example protos.

  Read in the data from the positive and negative examples on disk, and
  transform it using a preprocessing pipeline that removes punctuation,
  tokenizes and maps tokens to int64 values indices.

  Args:
    train_neg_filepattern: Filepattern for training data negative examples
    train_pos_filepattern: Filepattern for training data positive examples
    test_neg_filepattern: Filepattern for test data negative examples
    test_pos_filepattern: Filepattern for test data positive examples
    transformed_train_filebase: Base filename for transformed training data
        shards
    transformed_test_filebase: Base filename for transformed test data shards
    transformed_metadata_dir: Directory where metadata for transformed data
        should be written
  """

  with beam.Pipeline() as pipeline:
    with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
      # pylint: disable=no-value-for-parameter
      train_data = pipeline | 'ReadTrain' >> ReadAndShuffleData(
          (train_neg_filepattern, train_pos_filepattern))
      # pylint: disable=no-value-for-parameter
      test_data = pipeline | 'ReadTest' >> ReadAndShuffleData(
          (test_neg_filepattern, test_pos_filepattern))

      metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
          REVIEW_COLUMN: dataset_schema.ColumnSchema(
              tf.string, [], dataset_schema.FixedColumnRepresentation()),
          LABEL_COLUMN: dataset_schema.ColumnSchema(
              tf.int64, [], dataset_schema.FixedColumnRepresentation()),
      }))

      def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        review = inputs[REVIEW_COLUMN]

        def remove_character(s, char):
          """Remove a character from a string.

          Args:
            s: A SparseTensor of rank 1 of type tf.string
            char: A string of length 1

          Returns:
            The string `s` with the given character removed (i.e. replaced by
            '')
          """
          # Hacky implementation where we split and rejoin.
          split = tf.string_split(s, char)
          rejoined = tf.reduce_join(
              tf.sparse_to_dense(
                  split.indices, split.dense_shape, split.values, ''),
              1)
          return rejoined

        def remove_punctuation(s):
          """Remove puncuation from a string.

          Args:
            s: A SparseTensor of rank 1 of type tf.string

          Returns:
            The string `s` with punctuation removed.
          """
          for char in PUNCTUATION_CHARACTERS:
            s = remove_character(s, char)
          return s

        cleaned_review = tft.map(remove_punctuation, review)
        review_tokens = tft.map(tf.string_split, cleaned_review)
        review_indices = tft.string_to_int(review_tokens, top_k=VOCAB_SIZE)
        return {
            REVIEW_COLUMN: review_indices,
            LABEL_COLUMN: inputs[LABEL_COLUMN]
        }

      (transformed_train_data, transformed_metadata), transform_fn = (
          (train_data, metadata)
          | 'AnalyzeAndTransform' >> beam_impl.AnalyzeAndTransformDataset(
              preprocessing_fn))

      transformed_test_data, _ = (
          ((test_data, metadata), transform_fn)
          | 'Transform' >> beam_impl.TransformDataset())

      _ = (
          transformed_train_data
          | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(
              transformed_train_filebase,
              coder=example_proto_coder.ExampleProtoCoder(
                  transformed_metadata.schema)))

      _ = (
          transformed_test_data
          | 'WriteTestData' >> tfrecordio.WriteToTFRecord(
              transformed_test_filebase,
              coder=example_proto_coder.ExampleProtoCoder(
                  transformed_metadata.schema)))

      _ = (
          transformed_metadata
          | 'WriteMetadata' >> beam_metadata_io.WriteMetadata(
              transformed_metadata_dir, pipeline=pipeline))


def train_and_evaluate(transformed_train_filepattern,
                       transformed_test_filepattern,
                       transformed_metadata_dir):
  """Train the model on training data and evaluate on evaluation data.

  Args:
    transformed_train_filepattern: Base filename for transformed training data
        shards
    transformed_test_filepattern: Base filename for transformed evaluation data
        shards
    transformed_metadata_dir: Directory containing transformed data metadata

  Returns:
    The results from the estimator's 'evaluate' method
  """
  # Unrecognized tokens are represented by -1, but
  # sparse_column_with_integerized_feature uses the mod operator to map integers
  # to the range [0, bucket_size).  By choosing bucket_size=VOCAB_SIZE + 1, we
  # represent unrecognized tokens as VOCAB_SIZE.
  review_column = feature_column.sparse_column_with_integerized_feature(
      REVIEW_COLUMN,
      bucket_size=VOCAB_SIZE + 1,
      combiner='sqrtn')

  estimator = learn.LinearClassifier([review_column])

  transformed_metadata = metadata_io.read_metadata(transformed_metadata_dir)
  train_input_fn = input_fn_maker.build_training_input_fn(
      transformed_metadata,
      transformed_train_filepattern,
      training_batch_size=TRAIN_BATCH_SIZE,
      label_keys=[LABEL_COLUMN])

  # Estimate the model using the default optimizer.
  estimator.fit(
      input_fn=train_input_fn,
      max_steps=TRAIN_NUM_EPOCHS * NUM_TRAIN_INSTANCES / TRAIN_BATCH_SIZE)

  # Evaluate model on eval dataset.
  eval_input_fn = input_fn_maker.build_training_input_fn(
      transformed_metadata,
      transformed_test_filepattern,
      training_batch_size=1,
      label_keys=[LABEL_COLUMN])

  return estimator.evaluate(input_fn=eval_input_fn, steps=NUM_TEST_INSTANCES)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input_data_dir',
                      help='path to directory containing input data')
  parser.add_argument('--transformed_data_dir',
                      help='path to directory to hold transformed data')
  args = parser.parse_args()

  if args.transformed_data_dir:
    transformed_data_dir = args.transformed_data_dir
  else:
    transformed_data_dir = tempfile.mkdtemp(dir=args.input_data_dir)

  train_neg_filepattern = os.path.join(args.input_data_dir, 'train/neg/*')
  train_pos_filepattern = os.path.join(args.input_data_dir, 'train/pos/*')
  test_neg_filepattern = os.path.join(args.input_data_dir, 'test/neg/*')
  test_pos_filepattern = os.path.join(args.input_data_dir, 'test/pos/*')
  transformed_train_filebase = os.path.join(transformed_data_dir,
                                            'train_transformed')
  transformed_test_filebase = os.path.join(transformed_data_dir,
                                           'test_transformed')
  transformed_metadata_dir = os.path.join(transformed_data_dir, 'metadata')

  transform_data(train_neg_filepattern, train_pos_filepattern,
                 test_neg_filepattern, test_pos_filepattern,
                 transformed_train_filebase, transformed_test_filebase,
                 transformed_metadata_dir)

  results = train_and_evaluate(transformed_train_filebase + '*',
                               transformed_test_filebase + '*',
                               transformed_metadata_dir)

  pprint.pprint(results)


if __name__ == '__main__':
  main()
