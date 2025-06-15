# TensorFlow Transform Testing Infrastructure

This directory contains the testing infrastructure for TensorFlow Transform.

## Structure

```
tests/
├── README.md               # This file
├── __init__.py            # Package marker
├── conftest.py            # Shared pytest fixtures and configuration
├── test_setup_validation.py  # Full validation tests (requires all dependencies)
├── test_minimal_setup.py     # Minimal tests that work without all dependencies
├── unit/                  # Unit tests directory
│   └── __init__.py
└── integration/           # Integration tests directory
    └── __init__.py
```

## Running Tests

### Using Poetry Scripts

```bash
# Run all tests
poetry run test

# Alternative command (both work)
poetry run tests

# Run specific test file
poetry run test tests/test_minimal_setup.py

# Run with specific markers
poetry run test -m unit
poetry run test -m "not slow"
```

### Using pytest directly

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=tensorflow_transform

# Run without coverage (useful for debugging)
python -m pytest --no-cov
```

## Test Markers

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests that may require external resources
- `@pytest.mark.slow` - Tests that take a long time to run

## Available Fixtures

See `conftest.py` for all available fixtures. Key fixtures include:

- `temp_dir` - Temporary directory that's cleaned up after test
- `temp_file` - Temporary file that's cleaned up after test
- `mock_config` - Sample configuration dictionary
- `sample_data` - Sample data for testing transformations
- `tf_example_data` - Temporary TFRecord file with example data
- `mock_preprocessing_fn` - Simple preprocessing function for testing
- `mock_schema` - Simple schema for testing

## Coverage Configuration

Coverage is configured to:
- Require 80% minimum coverage
- Generate HTML reports in `htmlcov/`
- Generate XML report as `coverage.xml`
- Exclude test files and common patterns from coverage

## Known Issues

### ARM64 Architecture Support

Some dependencies like `tfx-bsl` may not have pre-built wheels for ARM64 architecture (e.g., Apple Silicon Macs, ARM Linux). If you encounter installation issues:

1. Try running the minimal test suite: `poetry run test tests/test_minimal_setup.py --no-cov`
2. Consider using x86_64 emulation or a compatible environment
3. Build dependencies from source if needed

## Writing New Tests

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use appropriate markers (`@pytest.mark.unit`, etc.)
4. Import and use fixtures from `conftest.py`
5. Follow existing test patterns and naming conventions