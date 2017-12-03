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
"""Package Setup script for the tf.Transform binary.
"""
from setuptools import find_packages
from setuptools import setup

# Tensorflow transform version.
__version__ = '0.4.0'


def _make_required_install_packages():
  return [
      'apache-beam[gcp]>=2.2,<3',

      # Protobuf libraries < 3.3 contain some map-related data corruption bugs
      # (b/35874111).
      'protobuf>=3.3,<4',

      # Six 1.11.0 incompatible with apitools.
      'six>=1.9,<1.11',

  ]


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
    requires=[])
