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
"""Init module for TF.Transform."""

# pylint: disable=wildcard-import
from tensorflow_transform import coders, experimental
from tensorflow_transform.analyzers import *
from tensorflow_transform.annotators import *
from tensorflow_transform.inspect_preprocessing_fn import *
from tensorflow_transform.mappers import *
from tensorflow_transform.output_wrapper import (
    TFTransformOutput,
    TransformFeaturesLayer,
)
from tensorflow_transform.py_func.api import apply_pyfunc
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata

# pylint: enable=wildcard-import
# Import version string.
from tensorflow_transform.version import __version__

# TF 2.6 split support for filesystems such as Amazon S3 out to the
# `tensorflow_io` package. Hence, this import is needed wherever we touch the
# filesystem.
try:
    import tensorflow_io as _  # pytype: disable=import-error # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
    pass

try:
    from tensorflow_transform import (
        google,  # pytype: disable=import-error # pylint: disable=g-import-not-at-top
    )
except ImportError:
    pass

__all__ = [
    "annotate_asset",
    "apply_buckets",
    "apply_buckets_with_interpolation",
    "apply_pyfunc",
    "apply_vocabulary",
    "bag_of_words",
    "bucketize",
    "bucketize_per_key",
    "compute_and_apply_vocabulary",
    "count_per_key",
    "covariance",
    "DatasetMetadata",
    "deduplicate_tensor_per_row",
    "estimated_probability_density",
    "get_analyze_input_columns",
    "get_num_buckets_for_transformed_feature",
    "get_transform_input_columns",
    "hash_strings",
    "histogram",
    "make_and_track_object",
    "max",
    "mean",
    "min",
    "ngrams",
    "pca",
    "quantiles",
    "scale_by_min_max",
    "scale_by_min_max_per_key",
    "scale_to_0_1",
    "scale_to_0_1_per_key",
    "scale_to_gaussian",
    "scale_to_z_score",
    "scale_to_z_score_per_key",
    "segment_indices",
    "size",
    "sparse_tensor_left_align",
    "sparse_tensor_to_dense_with_shape",
    "sum",
    "tfidf",
    "TFTransformOutput",
    "TransformFeaturesLayer",
    "tukey_h_params",
    "tukey_location",
    "tukey_scale",
    "var",
    "__version__",
    "vocabulary",
    "word_count",
]
