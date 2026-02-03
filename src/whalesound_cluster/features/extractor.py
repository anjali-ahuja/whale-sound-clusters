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

    def extract_from_segment_metadata(
        self,
        source_file: str | Path,
        start_time: float,
        end_time: float,
        segment_id: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Extract features from a segment of a source audio file.
        
        Args:
            source_file: Path to source audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            segment_id: Segment identifier
            metadata: Additional metadata
            
        Returns:
            Dictionary with features and metadata
        """
        source_file = Path(source_file)
        
        # Load full audio
        audio, sr = load_audio(source_file, sample_rate=self.sample_rate)
        
        if sr != self.sample_rate:
            logger.warning(f"Sample rate mismatch: {sr} != {self.sample_rate}")
        
        # Extract segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = audio[start_sample:end_sample]
        
        if len(segment_audio) == 0:
            raise ValueError(f"Empty segment: {start_time} to {end_time} in {source_file}")
        
        # Extract features
        features = self.extract_features(segment_audio)
        
        # Build result
        result = {
            "segment_id": segment_id,
            "features": features,
            "feature_dim": len(features),
            "sample_rate": sr,
            "duration": len(segment_audio) / sr,
            "start_time": start_time,
            "end_time": end_time,
            "source_file": str(source_file),
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

    def extract_batch_from_metadata(
        self,
        segments_metadata: List[Dict],
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Extract features from segments using metadata (extracts from source files).
        
        Args:
            segments_metadata: List of segment metadata dicts with:
                - segment_id: Segment identifier
                - source_file: Path to source audio file
                - start_time: Start time in seconds
                - end_time: End time in seconds
            output_path: Optional path to save parquet file
            
        Returns:
            DataFrame with features and metadata
        """
        results = []
        
        # Group segments by source file to avoid loading the same file multiple times
        from collections import defaultdict
        segments_by_source = defaultdict(list)
        for seg_meta in segments_metadata:
            source_file = seg_meta.get("source_file")
            if source_file:
                # Resolve and normalize source file path
                source_path = Path(source_file)
                if source_path.exists():
                    # Use resolved absolute path as key for consistent grouping
                    source_key = source_path.resolve()
                else:
                    # If file doesn't exist, still try to use it (will fail later with better error)
                    source_key = source_path.resolve() if source_path.is_absolute() else source_path
                segments_by_source[source_key].append(seg_meta)
            else:
                logger.warning(f"Segment {seg_meta.get('segment_id')} missing source_file, skipping")
        
        # Process segments grouped by source file
        for source_file, segs in tqdm(
            segments_by_source.items(),
            desc="Processing source files",
            total=len(segments_by_source),
        ):
            if not source_file.exists():
                logger.warning(f"Source file not found: {source_file}, skipping {len(segs)} segments")
                continue
            
            # Load full audio once per source file
            try:
                full_audio, sr = load_audio(source_file, sample_rate=self.sample_rate)
            except Exception as e:
                logger.warning(f"Failed to load {source_file}: {e}, skipping {len(segs)} segments")
                continue
            
            # Extract features for each segment from this source file
            for seg_meta in tqdm(
                segs,
                desc=f"Extracting from {source_file.name}",
                leave=False,
            ):
                try:
                    start_time = seg_meta.get("start_time", 0)
                    end_time = seg_meta.get("end_time")
                    segment_id = seg_meta.get("segment_id", "unknown")
                    
                    if end_time is None:
                        logger.warning(f"Segment {segment_id} missing end_time, skipping")
                        continue
                    
                    # Extract segment
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment_audio = full_audio[start_sample:end_sample]
                    
                    if len(segment_audio) == 0:
                        logger.warning(f"Empty segment {segment_id}, skipping")
                        continue
                    
                    # Extract features
                    features = self.extract_features(segment_audio)
                    
                    # Build result
                    result = {
                        "segment_id": segment_id,
                        "features": features,
                        "feature_dim": len(features),
                        "sample_rate": sr,
                        "duration": len(segment_audio) / sr,
                        "start_time": start_time,
                        "end_time": end_time,
                        "source_file": str(source_file),
                    }
                    
                    # Add other metadata
                    for key, value in seg_meta.items():
                        if key not in result:
                            result[key] = value
                    
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to extract features from segment {seg_meta.get('segment_id')}: {e}")
                    continue
        
        if not results:
            logger.error("No features extracted")
            return pd.DataFrame()
        
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




