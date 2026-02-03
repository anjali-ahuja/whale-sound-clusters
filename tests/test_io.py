"""Tests for I/O modules."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whalesound_cluster.io.downloader import GCSDownloader


def test_gcs_downloader_init():
    """Test GCS downloader initialization."""
    downloader = GCSDownloader(
        bucket_name="test-bucket",
        prefix="test/prefix",
        output_dir="test_output",
        method="python",
    )
    
    assert downloader.bucket_name == "test-bucket"
    assert downloader.prefix == "test/prefix"
    assert downloader.output_dir.exists()


def test_extract_date_from_path():
    """Test date extraction from paths."""
    downloader = GCSDownloader(
        bucket_name="test",
        prefix="",
        output_dir="test",
        method="python",
    )
    
    # Test various date formats
    assert downloader._extract_date_from_path("file_20200101.wav") is not None
    assert downloader._extract_date_from_path("file_2020-01-01.wav") is not None
    assert downloader._extract_date_from_path("file_2020_01_01.wav") is not None
    assert downloader._extract_date_from_path("file_no_date.wav") is None


def test_metadata_integrity():
    """Test that metadata is preserved through pipeline."""
    # This is a placeholder test - in a real scenario, you'd test
    # that metadata flows through download -> segment -> featurize -> cluster
    pass




