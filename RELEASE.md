# Version 0.25.0

## Major Features and Improvements

*   Updated the "Getting Started" guide and examples to demonstrate the support
    for both the "instance dict" and the "TFXIO" format. Users are encouraged to
    start using the "TFXIO" format, expecially in cases where
    [pre-canned TFXIO implementations](https://www.tensorflow.org/tfx/tfx_bsl/api_docs/python/tfx_bsl/public/tfxio)
    is available as it offers better performance.
*   From this release TFT will also be hosting nightly packages on
    https://pypi-nightly.tensorflow.org. To install the nightly package use the
    following command:

    ```
    pip install -i https://pypi-nightly.tensorflow.org/simple tensorflow-transform
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of TFT available on PyPI by running the
    command `pip install tensorflow-transform` .

## Bug Fixes and Other Changes

*   `TFTransformOutput.transform_raw_features` and `TransformFeaturesLayer` can
    be used when a transform fn is exported as a TF2 SavedModel and imported in
    graph mode.
*   Utility methods in `tft.inspect_preprocessing_fn` now take an optional
    parameter `force_tf_compat_v1`. If this is False, the `preprocessing_fn` is
    traced using tf.function in TF 2.x when TF 2 behaviors are enabled.
*   Switching to a wrapper for `collections.namedtuple` to ensure compatibility
    with PySpark which modifies classes produced by the factory.
*   Caching has been disabled for `tft.tukey_h_params`, `tft.tukey_location` and
    `tft.tukey_scale` due to the cached accumulator being non-deterministic.
*   Track variables created within the `preprocessing_fn` in the native TF 2
    implementation.
*   `TFTransformOutput.transform_raw_features` returns a wrapped python dict
    that overrides pop to return None instead of raising a KeyError when called
    with a key not found in the dictionary. This is done as preparation for
    switching the default value of `drop_unused_features` to True.
*   Vocabularies written in `tfrecord_gzip` format no longer filter out entries
    that are empty or that include a newline character.
*   Depends on `apache-beam[gcp]>=2.25,<3`.
*   Depends on `tensorflow-metadata>=0.25,<0.26`.
*   Depends on `tfx-bsl>=0.25,<0.26`.

## Breaking changes

*   N/A

## Deprecations

*   The `decode` method of the available coders (`tft.coders.CsvCoder` and
    `tft.coders.ExampleProtoCoder`) has been deprecated and removed.
    [Canned TFXIO implementations](https://www.tensorflow.org/tfx/tfx_bsl/api_docs/python/tfx_bsl/public/tfxio)
    should be used to read and decode data instead.
