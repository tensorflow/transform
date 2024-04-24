# Version 1.15.0

## Major Features and Improvements

*   Added support for sparse labels in AMI vocabulary computation.

## Bug Fixes and Other Changes

*   Bumped the Ubuntu version on which `tensorflow_transform` is tested to 20.04
    (previously was 16.04).
*   Explicitly use Keras 2 or `tf_keras`` if Keras 3 is installed.
*   Added python 3.11 support.
*   Depends on `tensorflow 2.15`.
*   Enable passing `tf.saved_model.SaveOptions` to model saving functionality.
*   Census and sentiment examples updated to only use Keras instead of
    estimator.
*   Depends on `apache-beam[gcp]>=2.53.0,<3` for Python 3.11 and on 
    `apache-beam[gcp]>=2.47.0,<3` for 3.9 and 3.10.
*   Depends on `protobuf>=4.25.2,<5` for Python 3.11 and on `protobuf>3.20.3,<5`
    for 3.9 and 3.10.

## Breaking Changes

*   Existing analyzer cache is automatically invalidated.

## Deprecations

*   Deprecated python 3.8 support.

