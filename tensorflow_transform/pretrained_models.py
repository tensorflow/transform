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
"""Public API functions related to pretrained tensorflow models.

These APIs provides ways to use pretrained models as as a transformation
function. Users can use a SavedModel specified by model dir, metagraph tags,
signatures, or use a tensor-in-tensor-out function which includes variables
together with a specified checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow_transform import api
from tensorflow_transform import impl_helper


def map_with_saved_model(model_dir, input_columns, tags, signature_name=None,
                         output_keys_in_signature=None):
  """Applies a SavedModel to some columns.

  Applies a SavedModel to `input_columns`. The SavedModel is specified with
  `model_dir`, `tags` and `signature_name`. Note that the SavedModel will be
  converted to an all-constants graph, so ops requiring graph collections, such
  as table lookup (which requires a table init op being added to
  TABLE_INITIALIZERS collection), are not supported.

  Args:
    model_dir: A path containing a SavedModel.
    input_columns: Input `Column`s used as model input tensors. If there are
       multiple inputs from model signature, it is a map with keys as the names
       from signature and values as input `Column`s. If there is only one input
       from signature, it needs to be the input `Column`.
    tags: The tags specifying which metagraph to load from the SavedModel.
    signature_name: Specify signature of the loaded model. The default value
       None can be used if there is only one signature in the MetaGraphDef.
    output_keys_in_signature: A list of strings which should be a subset of
       the outputs in the signature of the SavedModel. The returned `Column`s
       will correspond to specified output tensors, in the same order. The
       default value None can be used if there is only one output from
       signature.

  Returns:
    Like tft.map, returns a `Column` representing the application of the
    SavedModel.

  Raises:
    ValueError: if
    `input_columns` is invalid type, or
    `signature_name` is None but the SavedModel contains multiple signature, or
    `input_columns` do not match the signature inputs, or
    `output_keys_in_signature` is not a subset of the signature outputs.
  """

  if isinstance(input_columns, dict):
    # Sort input columns so the pipeline is deterministic.
    input_keys_in_signature_sorted = sorted(input_columns.keys())
    input_columns_sorted = [input_columns[k] for k in
                            input_keys_in_signature_sorted]
  elif isinstance(input_columns, api.Column):
    input_keys_in_signature_sorted = None
    input_columns_sorted = [input_columns]
  else:
    raise ValueError(
        'Expect "input_columns" to be dict or tft.Column but got %s.'
        % type(input_columns))
  tensor_fn = impl_helper.make_tensor_func_from_saved_model(
      model_dir, tags, signature_name=signature_name,
      input_keys_in_signature=input_keys_in_signature_sorted,
      output_keys_in_signature=output_keys_in_signature)
  return api.map(tensor_fn, *input_columns_sorted)


def map_with_checkpoint(input_tensor_func, input_columns, checkpoint,
                        include=None, exclude=None):
  """Applies a tensor-in-tensor-out function with variables to some columns.

  Variable values are loaded from the given checkpoint path. Note that the
  input_tensor_func, together with the checkpoint, will be converted to an
  all-constants graph, so ops requiring graph collections, such as table lookup
  (which requires a table init op being added to TABLE_INITIALIZERS collection),
  are not supported.

  Args:
    input_tensor_func: A tensor-in-tensor-out function that may contain
       variables.
    input_columns: A list of `Column`s to apply the `input_tensor_func` to.
    checkpoint: The checkpoint path to load variables from.
    include: An optional list/tuple of scope strings for filtering which
       variables from the VARIABLES collection to include. If None, all
       variables will be included.
    exclude: An optional list/tuple of scope strings for filtering which
       variables from the VARIABLES collection to exclude. If None, no variables
       will be excluded.

  Returns:
    Like tft.map, returns a `Column` representing the output of the
    `input_tensor_func`.

  Raises:
    ValueError if the input tensor-in-tensor-out function adds to
       TABLE_INITIALIZERS collections.
  """

  tensor_func = impl_helper.make_tensor_func_from_checkpoint(
      input_tensor_func, checkpoint, include, exclude)

  return api.map(tensor_func, *input_columns)
