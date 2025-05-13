# Transform library for non-TFX users

Transform is available as a standalone library.

-   [Getting Started with TensorFlow
    Transform](get_started)
-   [TensorFlow Transform API
    Reference](api_docs/python/tft)

The [`tft`](api_docs/python/tft)
module documentation is the only module that is relevant to TFX users.
The [`tft_beam`](api_docs/python/tft_beam)
module is relevant only when using Transform as a standalone library.
Typically, a TFX user constructs a `preprocessing_fn`, and the rest of the 
Transform library calls are made by the Transform component.

You can also use the Apache Beam `MLTransform`
class to preprocess data for training and inference. The
`MLTransform` class wraps multiple TFX data
processing transforms in one class. For more information, see
[Preprocess data with
MLTransform](https://beam.apache.org/documentation/ml/preprocess-data)
in the Apache Beam documentation.
