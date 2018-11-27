# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Package Setup script for tf.Transform.
"""
from setuptools import find_packages
from setuptools import setup

# Tensorflow transform version.
__version__ = '0.12.0dev'


def _make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # six, and protobuf) with TF.
  return [
      'absl-py>=0.1.6',
      'apache-beam[gcp]>=2.8,<3',
      'numpy>=1.13.3,<2',

      'protobuf>=3.6.0,<4',

      'six>=1.10,<2',

      'tensorflow-metadata>=0.9,<0.10',


      'pydot>=1.2.0,<1.3',
  ]

_LONG_DESCRIPTION = """\
*TensorFlow Transform* is a library for preprocessing data with TensorFlow.
`tf.Transform` is useful for data that requires a full-pass, such as:

* Normalize an input value by mean and standard deviation.
* Convert strings to integers by generating a vocabulary over all input values.
* Convert floats to integers by assigning them to buckets based on the observed
  data distribution.

TensorFlow has built-in support for manipulations on a single example or a batch
of examples. `tf.Transform` extends these capabilities to support full-passes
over the example data.

https://github.com/tensorflow/transform/blob/master/README.md
"""


setup(
    name='tensorflow-transform',
    version=__version__,
    author='Google Inc.',
    author_email='tf-transform-feedback@google.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 2 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    namespace_packages=[],
    install_requires=_make_required_install_packages(),
    python_requires='>=2.7,<3',
    packages=find_packages(),
    include_package_data=True,
    description='A library for data preprocessing with TensorFlow',
    long_description=_LONG_DESCRIPTION,
    keywords='tensorflow transform tfx',
    url='https://www.tensorflow.org/tfx/transform',
    download_url='https://pypi.org/project/tensorflow-transform',
    requires=[])
