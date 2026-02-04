"""Tests for feature extraction."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from whalesound_cluster.audio.io import save_audio
from whalesound_cluster.features.extractor import FeatureExtractor


def test_feature_extractor_logmel():
    """Test log-mel feature extraction."""
    extractor = FeatureExtractor(
        method="logmel",
        sample_rate=4000,
        n_mels=64,
        summarize="mean_std",
    )
    
    # Create test audio
    sample_rate = 4000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    features = extractor.extract_logmel(audio)
    
    assert len(features) == 128  # 64 mean + 64 std
    assert np.all(np.isfinite(features))


def test_feature_extractor_from_file():
    """Test feature extraction from file."""
    extractor = FeatureExtractor(
        method="logmel",
        sample_rate=4000,
        n_mels=32,
        summarize="mean_std",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test audio file
        file_path = Path(tmpdir) / "test.wav"
        sample_rate = 4000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        save_audio(audio, file_path, sample_rate)
        
        # Extract features
        result = extractor.extract_from_file(file_path, {"test": "metadata"})
        
        assert "features" in result
        assert "segment_id" in result
        assert "feature_dim" in result
        assert result["test"] == "metadata"
        assert len(result["features"]) > 0


def test_feature_extractor_batch():
    """Test batch feature extraction."""
    extractor = FeatureExtractor(
        method="logmel",
        sample_rate=4000,
        n_mels=32,
        summarize="mean_std",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple test files
        file_paths = []
        sample_rate = 4000
        for i in range(3):
            file_path = Path(tmpdir) / f"test_{i}.wav"
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * (440 + i * 100) * t)
            save_audio(audio, file_path, sample_rate)
            file_paths.append(file_path)
        
        # Extract features
        df = extractor.extract_batch(file_paths)
        
        assert len(df) == 3
        assert "segment_id" in df.columns
        assert any(c.startswith("feat_") for c in df.columns)


def test_feature_output_shapes():
    """Test feature output shapes are consistent."""
    extractor_mean_std = FeatureExtractor(
        method="logmel",
        n_mels=64,
        summarize="mean_std",
    )
    
    extractor_flatten = FeatureExtractor(
        method="logmel",
        n_mels=64,
        summarize="flatten",
    )
    
    # Create test audio
    sample_rate = 4000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    features_mean_std = extractor_mean_std.extract_logmel(audio)
    features_flatten = extractor_flatten.extract_logmel(audio)
    
    assert len(features_mean_std) == 128  # 64 * 2
    assert len(features_flatten) > len(features_mean_std)  # Flattened is larger





