"""Audio I/O functions."""

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from whalesound_cluster.utils.logging import setup_logging

logger = setup_logging()


def load_audio(
    file_path: str | Path,
    sample_rate: int | None = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (resample if different)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    import librosa
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check if file is empty
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {file_path}")
    
    try:
        # Use librosa for resampling support
        y, sr = librosa.load(str(file_path), sr=sample_rate, mono=mono)
        logger.debug(f"Loaded {file_path}: shape={y.shape}, sr={sr}")
        return y, sr
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    file_path: str | Path,
    sample_rate: int,
    format: str = "WAV",
) -> None:
    """
    Save audio array to file.
    
    Args:
        audio: Audio array (1D or 2D)
        file_path: Output file path
        sample_rate: Sample rate
        format: File format (WAV, FLAC, etc.)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure audio is 2D for soundfile
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    
    try:
        sf.write(str(file_path), audio, sample_rate, format=format)
        logger.debug(f"Saved audio to {file_path}: shape={audio.shape}, sr={sample_rate}")
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")
        raise

