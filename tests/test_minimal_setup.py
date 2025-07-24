"""Minimal validation tests that can run without all dependencies."""

import os
import sys
from pathlib import Path

import pytest


class TestMinimalSetup:
    """Minimal tests to validate basic setup."""
    
    def test_pytest_works(self):
        """Basic test to ensure pytest is functional."""
        assert 1 + 1 == 2
    
    def test_project_structure(self):
        """Test that basic project structure is in place."""
        root = Path(__file__).parent.parent
        assert (root / "tests").exists()
        assert (root / "pyproject.toml").exists()
        assert (root / ".gitignore").exists()
    
    def test_fixtures_work(self, tmp_path):
        """Test that basic pytest fixtures work."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        assert test_file.read_text() == "hello"
    
    @pytest.mark.unit
    def test_markers_work(self):
        """Test that custom markers are functional."""
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])