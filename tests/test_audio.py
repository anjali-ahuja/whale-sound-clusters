"""Tests for audio processing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from whalesound_cluster.audio.io import load_audio, save_audio
from whalesound_cluster.audio.segmenter import AudioSegmenter


def test_load_save_audio():
    """Test audio loading and saving."""
    # Create test audio
    sample_rate = 4000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.wav")
        
        # Save
        save_audio(audio, file_path, sample_rate)
        assert file_path.exists()
        
        # Load
        loaded_audio, loaded_sr = load_audio(file_path)
        assert loaded_sr == sample_rate
        assert len(loaded_audio) == len(audio)
        assert np.allclose(loaded_audio, audio, atol=0.1)


def test_segmenter_energy():
    """Test energy computation."""
    segmenter = AudioSegmenter(sample_rate=4000)
    
    # Create audio with silence and signal
    sample_rate = 4000
    duration = 2.0
    audio = np.zeros(int(sample_rate * duration))
    audio[int(sample_rate * 0.5) : int(sample_rate * 1.5)] = 0.5
    
    energy = segmenter.compute_energy(audio)
    assert len(energy) > 0
    assert energy.max() > 0


def test_segmenter_find_segments():
    """Test segment finding."""
    segmenter = AudioSegmenter(
        sample_rate=4000,
        energy_threshold=0.01,
        min_duration_ms=500,
    )
    
    # Create audio with two segments
    sample_rate = 4000
    duration = 3.0
    audio = np.zeros(int(sample_rate * duration))
    audio[int(sample_rate * 0.5) : int(sample_rate * 1.0)] = 0.5
    audio[int(sample_rate * 2.0) : int(sample_rate * 2.5)] = 0.5
    
    energy = segmenter.compute_energy(audio)
    segments = segmenter.find_segments(energy)
    
    assert len(segments) >= 1  # Should find at least one segment


def test_segmenter_segment_audio():
    """Test full segmentation."""
    segmenter = AudioSegmenter(
        sample_rate=4000,
        min_duration_ms=200,
        max_duration_ms=2000,
    )
    
    # Create test audio
    sample_rate = 4000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    segments = segmenter.segment_audio(audio, "test.wav", {})
    
    assert len(segments) > 0
    for segment in segments:
        assert segment.start_time < segment.end_time
        assert len(segment.audio) > 0
        assert segment.segment_id is not None


def test_bandpass_filter():
    """Test bandpass filtering."""
    segmenter = AudioSegmenter(sample_rate=4000)
    
    # Create test signal
    sample_rate = 4000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1000 * t)
    
    filtered = segmenter.apply_bandpass(audio, fmin=50, fmax=500, sample_rate=sample_rate)
    
    assert len(filtered) == len(audio)
    assert not np.allclose(filtered, audio)  # Should be different





