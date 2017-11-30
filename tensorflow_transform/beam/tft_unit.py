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
"""Library for testing Tensorflow Transform."""


from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow.python.framework import test_util


class TransformTestCase(test_util.TensorFlowTestCase):
  """Base test class for testing tf-transform preprocessing functions."""

  # Display context for failing rows in data assertions.
  longMessage = True  # pylint: disable=invalid-name

  def assertDataCloseOrEqual(self, a_data, b_data):
    """Assert two datasets contain nearly equal values.

    Args:
      a_data: a sequence of dicts whose values are
              either strings, lists of strings, numeric types or a pair of
              those.
      b_data: same types as a_data

    Raises:
      AssertionError: if the two datasets are not the same.
    """
    self.assertEqual(len(a_data), len(b_data),
                     'len(%r) != len(%r)' % (a_data, b_data))
    for i, (a_row, b_row) in enumerate(zip(a_data, b_data)):
      self.assertItemsEqual(a_row.keys(), b_row.keys(), msg='Row %d' % i)
      for key in a_row.keys():
        a_value = a_row[key]
        b_value = b_row[key]
        msg = 'Row %d, key %s' % (i, key)
        if isinstance(a_value, tuple):
          self._assertValuesCloseOrEqual(a_value[0], b_value[0], msg=msg)
          self._assertValuesCloseOrEqual(a_value[1], b_value[1], msg=msg)
        else:
          self._assertValuesCloseOrEqual(a_value, b_value, msg=msg)

  def _assertValuesCloseOrEqual(self, a_value, b_value, msg=None):
    try:
      if (isinstance(a_value, str) or
          isinstance(a_value, list) and a_value and
          isinstance(a_value[0], str)):
        self.assertAllEqual(a_value, b_value)
      else:
        self.assertAllClose(a_value, b_value)
    except (AssertionError, TypeError) as e:
      if msg:
        e.args = ((e.args[0] + ' : ' + msg,) + e.args[1:])
      raise

  def _resolveDeferredMetadata(self, transformed_metadata):
    """Asserts that there is no unresolved metadata."""
    # We should be able to call ResolveBeamFutures in all cases, but because
    # we are using Beam's automaterialization, we don't have access to an
    # explicit pipeline.  Therefore we only call ResolveBeamFutures when we
    # are sure that transformed_metadata contains at least one element.
    if transformed_metadata.pcollections:
      transformed_metadata = (
          (transformed_metadata | beam_metadata_io.ResolveBeamFutures(None))[0])
    else:
      transformed_metadata = transformed_metadata.dataset_metadata

    # No more unresolved metadata should remain.
    unresolved_futures = transformed_metadata.substitute_futures({})
    self.assertEqual(unresolved_futures, [])
    return transformed_metadata

  def assertAnalyzeAndTransformResults(self,
                                       input_data,
                                       input_metadata,
                                       preprocessing_fn,
                                       expected_data=None,
                                       expected_metadata=None,
                                       only_check_core_metadata=False):
    """Assert that input data and metadata is transformed as expected.

    This methods asserts transformed data and transformed metadata match
    with expected_data and expected_metadata.

    Args:
      input_data: A sequence of dicts whose values are
          either strings, lists of strings, numeric types or a pair of those.
      input_metadata: DatasetMetadata describing input_data.
      preprocessing_fn: A function taking a dict of tensors and returning
          a dict of tensors.
      expected_data: (optional) A dataset with the same type constraints as
          input_data, but representing the output after transformation.
          If supplied, transformed data is asserted to be equal.
      expected_metadata: (optional) DatasetMetadata describing the transformed
          data. If supplied, transformed metadata is asserted to be equal.
      only_check_core_metadata: A boolean to indicate if all elements in
          the transformed metadata is asserted to be equal to expected metadata.
          If True, only transformed feature names, dtypes and representations
          are asserted.
    Raises:
      AssertionError: if the expected data does not match the results of
          transforming input_data according to preprocessing_fn, or
          (if provided) if the expected metadata does not match.
    """
    # Note: we don't separately test AnalyzeDataset and TransformDataset as
    # AnalyzeAndTransformDataset currently simply composes these two
    # transforms.  If in future versions of the code, the implementation
    # differs, we should also run AnalyzeDataset and TransformDatset composed.
    #
    # Also, the dataset_metadata that is returned along with
    # `transformed_data` is incomplete as it does not contain the deferred
    # components, so we instead inspect the metadata returned along with the
    # transform function.
    temp_dir = self.get_temp_dir()
    with beam_impl.Context(temp_dir=temp_dir):
      (transformed_data, _), (_, transformed_metadata) = (
          (input_data, input_metadata)
          | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

    if expected_data:
      self.assertDataCloseOrEqual(expected_data, transformed_data)

    if not expected_metadata:
      return

    transformed_metadata = self._resolveDeferredMetadata(transformed_metadata)

    if only_check_core_metadata:
      # preprocessing_fn may add metadata to column schema only relevant to
      # internal implementation such as vocabulary_file. As such, only check
      # feature names, dtypes and representations are as expected.
      self.assertSameElements(transformed_metadata.schema.column_schemas.keys(),
                              expected_metadata.schema.column_schemas.keys())

      for k, v in transformed_metadata.schema.column_schemas.iteritems():
        expected_schema = expected_metadata.schema.column_schemas[k]

        self.assertEqual(expected_schema.representation, v.representation,
                         "representation doesn't match for feature '%s'" % k)
        self.assertEqual(expected_schema.domain.dtype, v.domain.dtype,
                         "dtype doesn't match for feature '%s'" % k)

    else:
      # Check the entire DatasetMetadata is as expected.
      # Use extra assertEqual for schemas, since full metadata assertEqual
      # error message is not conducive to debugging.
      self.assertEqual(expected_metadata.schema.column_schemas,
                       transformed_metadata.schema.column_schemas)

      self.assertEqual(expected_metadata, transformed_metadata)
