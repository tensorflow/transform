"""Pytest configuration and shared fixtures for TensorFlow Transform tests."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Any, Dict

import pytest
import tensorflow as tf


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory that will be cleaned up after test.
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """Create a temporary file for testing.
    
    Yields:
        Path: Path to the temporary file that will be cleaned up after test.
    """
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    yield Path(temp_path)
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing.
    
    Returns:
        Dict[str, Any]: A sample configuration dictionary.
    """
    return {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "features": ["feature1", "feature2", "feature3"],
        "label": "target",
        "model_dir": "/tmp/model",
        "preprocessing": {
            "normalize": True,
            "scale": True,
            "bucketize": False
        }
    }


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Generate sample data for testing transformations.
    
    Returns:
        Dict[str, Any]: A dictionary containing sample features and labels.
    """
    return {
        "numeric_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        "categorical_feature": ["A", "B", "A", "C", "B"],
        "text_feature": ["hello world", "tensorflow transform", "test data"],
        "label": [0, 1, 0, 1, 1]
    }


@pytest.fixture
def tf_example_data() -> Generator[str, None, None]:
    """Create a temporary TFRecord file with example data.
    
    Yields:
        str: Path to the temporary TFRecord file.
    """
    import tensorflow as tf
    
    temp_path = tempfile.mktemp(suffix=".tfrecord")
    
    # Create sample TF examples
    writer = tf.io.TFRecordWriter(temp_path)
    
    for i in range(5):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "numeric_feature": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[float(i)])
                    ),
                    "categorical_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[f"category_{i}".encode()])
                    ),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[i % 2])
                    )
                }
            )
        )
        writer.write(example.SerializeToString())
    
    writer.close()
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_transform_output_dir(temp_dir: Path) -> Path:
    """Create a mock transform output directory structure.
    
    Args:
        temp_dir: Temporary directory fixture.
        
    Returns:
        Path: Path to the transform output directory.
    """
    output_dir = temp_dir / "transform_output"
    output_dir.mkdir()
    
    # Create expected subdirectories
    (output_dir / "transformed_metadata").mkdir()
    (output_dir / "transform_fn").mkdir()
    (output_dir / "transformed_data").mkdir()
    
    return output_dir


@pytest.fixture(autouse=True)
def reset_tensorflow_state():
    """Reset TensorFlow state between tests to avoid interference."""
    yield
    tf.keras.backend.clear_session()


@pytest.fixture
def mock_preprocessing_fn():
    """Provide a simple preprocessing function for testing.
    
    Returns:
        callable: A preprocessing function that applies basic transformations.
    """
    def preprocessing_fn(inputs):
        """Simple preprocessing function for testing."""
        import tensorflow_transform as tft
        
        outputs = {}
        
        # Normalize numeric features
        if "numeric_feature" in inputs:
            outputs["numeric_feature_normalized"] = tft.scale_to_z_score(
                inputs["numeric_feature"]
            )
        
        # Vocabulary for categorical features
        if "categorical_feature" in inputs:
            outputs["categorical_feature_integerized"] = tft.compute_and_apply_vocabulary(
                inputs["categorical_feature"]
            )
        
        # Pass through labels
        if "label" in inputs:
            outputs["label"] = inputs["label"]
        
        return outputs
    
    return preprocessing_fn


@pytest.fixture
def mock_schema():
    """Provide a mock schema for testing.
    
    Returns:
        schema_pb2.Schema: A simple schema for testing.
    """
    from tensorflow_metadata.proto.v0 import schema_pb2
    
    schema = schema_pb2.Schema()
    
    # Add numeric feature
    numeric_feature = schema.feature.add()
    numeric_feature.name = "numeric_feature"
    numeric_feature.type = schema_pb2.FLOAT
    
    # Add categorical feature
    categorical_feature = schema.feature.add()
    categorical_feature.name = "categorical_feature"
    categorical_feature.type = schema_pb2.BYTES
    
    # Add label
    label_feature = schema.feature.add()
    label_feature.name = "label"
    label_feature.type = schema_pb2.INT
    
    return schema


# Markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (may require external resources)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker for tests with "slow" in their name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)