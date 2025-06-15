"""Validation tests to verify the testing infrastructure is properly set up."""

import os
import sys
from pathlib import Path

import pytest


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure setup."""
    
    def test_pytest_is_installed(self):
        """Verify pytest is installed and importable."""
        import pytest
        assert pytest.__version__
    
    def test_pytest_cov_is_installed(self):
        """Verify pytest-cov is installed and importable."""
        import pytest_cov
        assert pytest_cov
    
    def test_pytest_mock_is_installed(self):
        """Verify pytest-mock is installed and importable."""
        import pytest_mock
        assert pytest_mock
    
    def test_project_structure_exists(self):
        """Verify the expected project structure exists."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories
        assert project_root.exists()
        assert (project_root / "tensorflow_transform").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check configuration files
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / ".gitignore").exists()
    
    def test_conftest_fixtures_available(self, temp_dir, mock_config, sample_data):
        """Verify conftest fixtures are available and working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "batch_size" in mock_config
        assert "features" in mock_config
        
        # Test sample_data fixture
        assert isinstance(sample_data, dict)
        assert "numeric_feature" in sample_data
        assert "categorical_feature" in sample_data
    
    def test_markers_are_registered(self):
        """Verify custom markers are properly registered."""
        markers = [mark.name for mark in pytest.mark._markers]
        assert "unit" in markers
        assert "integration" in markers
        assert "slow" in markers
    
    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Test that unit marker can be applied."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Test that integration marker can be applied."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Test that slow marker can be applied."""
        assert True
    
    def test_tensorflow_import(self):
        """Verify TensorFlow can be imported."""
        import tensorflow as tf
        assert tf.__version__
    
    def test_tensorflow_transform_import(self):
        """Verify tensorflow_transform can be imported."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import tensorflow_transform as tft
        assert tft.__version__
    
    def test_coverage_configuration(self):
        """Verify coverage is properly configured."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        with open(pyproject_path, "r") as f:
            content = f.read()
            
        # Check coverage configuration exists
        assert "[tool.coverage.run]" in content
        assert "[tool.coverage.report]" in content
        assert "source = [\"tensorflow_transform\"]" in content
        assert "--cov-fail-under=80" in content
    
    def test_pytest_configuration(self):
        """Verify pytest is properly configured."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        with open(pyproject_path, "r") as f:
            content = f.read()
            
        # Check pytest configuration exists
        assert "[tool.pytest.ini_options]" in content
        assert "testpaths = [\"tests\"]" in content
        assert "--strict-markers" in content
    
    def test_poetry_scripts_configured(self):
        """Verify Poetry scripts are properly configured."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        with open(pyproject_path, "r") as f:
            content = f.read()
            
        # Check poetry scripts exist
        assert "[tool.poetry.scripts]" in content
        assert 'test = "pytest:main"' in content
        assert 'tests = "pytest:main"' in content


class TestMockingCapabilities:
    """Test class to validate mocking capabilities."""
    
    def test_pytest_mock_fixture(self, mocker):
        """Test that pytest-mock mocker fixture works."""
        mock_func = mocker.Mock(return_value="mocked")
        assert mock_func() == "mocked"
        mock_func.assert_called_once()
    
    def test_mock_patch(self, mocker):
        """Test that patching with mocker works."""
        mock_os = mocker.patch("os.path.exists")
        mock_os.return_value = True
        
        result = os.path.exists("/fake/path")
        assert result is True
        mock_os.assert_called_with("/fake/path")


class TestTempFileHandling:
    """Test class to validate temporary file handling."""
    
    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temp_dir is created and will be cleaned up."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_temp_file_cleanup(self, temp_file):
        """Test that temp_file is created and will be cleaned up."""
        assert temp_file.exists()
        temp_file.write_text("temporary content")
        assert temp_file.read_text() == "temporary content"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])