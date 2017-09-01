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
"""Constants and types shared by tf.Transform Beam package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from apache_beam.typehints import Union
from six import integer_types
from six import string_types

NUMERIC_TYPE = Union[float, Union[integer_types]]
PRIMITIVE_TYPE = Union[NUMERIC_TYPE, Union[string_types]]
