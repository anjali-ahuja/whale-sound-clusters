"""Feature extraction from audio segments."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

import librosa

from whalesound_cluster.audio.io import load_audio
from whalesound_cluster.utils.logging import setup_logging

logger = setup_logging()


class FeatureExtractor:
    """Extract features from audio segments."""

    def __init__(
        self,
        method: str = "logmel",
        sample_rate: int = 4000,
        n_mels: int = 64,
        fmin: float = 20,
        fmax: float = 2000,
        hop_length: int = 512,
        n_fft: int = 2048,
        summarize: str = "mean_std",
    ):
        """
        Initialize feature extractor.
        
        Args:
            method: "logmel" or "embedding"
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
            hop_length: Hop length for STFT
            n_fft: FFT window size
            summarize: "mean_std" or "flatten"
        """
        self.method = method
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.summarize = summarize

    def extract_logmel(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel spectrogram features.
        
        Args:
            audio: Audio array
            
        Returns:
            Feature vector
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        
        # Convert to log scale
        logmel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Summarize
        if self.summarize == "mean_std":
            # Mean and std across time
            mean = np.mean(logmel, axis=1)
            std = np.std(logmel, axis=1)
            features = np.concatenate([mean, std])
        elif self.summarize == "flatten":
            # Flatten entire spectrogram
            features = logmel.flatten()
        else:
            raise ValueError(f"Unknown summarize method: {self.summarize}")
        
        return features

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from audio.
        
        Args:
            audio: Audio array
            
        Returns:
            Feature vector
        """
        if self.method == "logmel":
            return self.extract_logmel(audio)
        elif self.method == "embedding":
            # Placeholder for embedding model
            logger.warning("Embedding method not yet implemented, using logmel")
            return self.extract_logmel(audio)
        else:
            raise ValueError(f"Unknown feature method: {self.method}")

    def extract_from_file(
        self,
        file_path: str | Path,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Extract features from audio file.
        
        Args:
            file_path: Path to audio file
            metadata: Additional metadata
            
        Returns:
            Dictionary with features and metadata
        """
        file_path = Path(file_path)
        
        # Load audio
        audio, sr = load_audio(file_path, sample_rate=self.sample_rate)
        
        if sr != self.sample_rate:
            logger.warning(f"Sample rate mismatch: {sr} != {self.sample_rate}")
        
        # Extract features
        features = self.extract_features(audio)
        
        # Build result
        result = {
            "segment_id": file_path.stem,
            "features": features,
            "feature_dim": len(features),
            "sample_rate": sr,
            "duration": len(audio) / sr,
        }
        
        if metadata:
            result.update(metadata)
        
        return result

    def extract_batch(
        self,
        segment_files: List[Path],
        metadata_list: Optional[List[Dict]] = None,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Extract features from multiple segments.
        
        Args:
            segment_files: List of segment file paths
            metadata_list: Optional list of metadata dicts
            output_path: Optional path to save parquet file
            
        Returns:
            DataFrame with features and metadata
        """
        if metadata_list is None:
            metadata_list = [{}] * len(segment_files)
        
        results = []
        
        for file_path, metadata in tqdm(
            zip(segment_files, metadata_list),
            desc="Extracting features",
            total=len(segment_files),
        ):
            try:
                result = self.extract_from_file(file_path, metadata)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to extract features from {file_path}: {e}")
                continue
        
        # Convert to DataFrame
        # Separate features array from metadata
        feature_arrays = [r.pop("features") for r in results]
        feature_dim = results[0]["feature_dim"] if results else 0
        
        # Create DataFrame with metadata
        df = pd.DataFrame(results)
        
        # Add feature columns
        feature_df = pd.DataFrame(
            np.array(feature_arrays),
            columns=[f"feat_{i}" for i in range(feature_dim)],
        )
        
        df = pd.concat([df, feature_df], axis=1)
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved features to {output_path}")
        
        return df

