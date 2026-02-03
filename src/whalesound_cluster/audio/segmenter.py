"""Audio segmentation based on energy thresholds."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy import signal

from whalesound_cluster.audio.io import load_audio, save_audio
from whalesound_cluster.utils.logging import setup_logging

logger = setup_logging()


@dataclass
class Segment:
    """Represents an audio segment."""
    
    start_time: float  # seconds
    end_time: float  # seconds
    audio: np.ndarray
    source_file: str
    segment_id: str
    metadata: dict


class AudioSegmenter:
    """Segment audio into units based on energy thresholds."""

    def __init__(
        self,
        frame_length_ms: int = 50,
        hop_length_ms: int = 25,
        energy_threshold: float = 0.01,
        min_duration_ms: int = 500,
        max_duration_ms: int = 3000,
        merge_gap_ms: int = 200,
        sample_rate: int = 4000,
    ):
        """
        Initialize segmenter.
        
        Args:
            frame_length_ms: Frame length in milliseconds
            hop_length_ms: Hop length in milliseconds
            energy_threshold: RMS energy threshold (normalized)
            min_duration_ms: Minimum segment duration
            max_duration_ms: Maximum segment duration
            merge_gap_ms: Merge segments closer than this gap
            sample_rate: Audio sample rate
        """
        self.frame_length_ms = frame_length_ms
        self.hop_length_ms = hop_length_ms
        self.energy_threshold = energy_threshold
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.merge_gap_ms = merge_gap_ms
        self.sample_rate = sample_rate
        
        # Convert to samples
        self.frame_length = int(frame_length_ms * sample_rate / 1000)
        self.hop_length = int(hop_length_ms * sample_rate / 1000)
        self.min_duration = int(min_duration_ms * sample_rate / 1000)
        self.max_duration = int(max_duration_ms * sample_rate / 1000)
        self.merge_gap = int(merge_gap_ms * sample_rate / 1000)

    def compute_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute short-time RMS energy.
        
        Args:
            audio: Audio array
            
        Returns:
            Energy array (one value per frame)
        """
        # Validate audio
        if len(audio) == 0:
            logger.warning("Empty audio array provided")
            return np.array([])
        
        # Check for invalid values and work on a copy
        audio_clean = audio.copy()
        if np.any(np.isnan(audio_clean)) or np.any(np.isinf(audio_clean)):
            logger.warning("Audio contains NaN or Inf values, replacing with zeros")
            audio_clean = np.nan_to_num(audio_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check if audio is too short
        if len(audio_clean) < self.frame_length:
            logger.warning(
                f"Audio too short for frame length: {len(audio_clean)} < {self.frame_length} samples"
            )
            # Return a single frame with RMS of entire audio
            rms = np.sqrt(np.mean(audio_clean**2))
            return np.array([rms])
        
        # Normalize audio
        audio_normalized = audio_clean
        max_val = np.abs(audio_normalized).max()
        if max_val > 0:
            audio_normalized = audio_normalized / (max_val + 1e-10)
        
        # Frame audio
        frames = []
        num_frames = max(0, len(audio_normalized) - self.frame_length + 1)
        for i in range(0, num_frames, self.hop_length):
            frame = audio_normalized[i : i + self.frame_length]
            rms = np.sqrt(np.mean(frame**2))
            frames.append(rms)
        
        return np.array(frames)

    def find_segments(self, energy: np.ndarray) -> List[tuple[int, int]]:
        """
        Find segment boundaries from energy array.
        
        Args:
            energy: Energy array
            
        Returns:
            List of (start_frame, end_frame) tuples
        """
        # Find frames above threshold
        above_threshold = energy > self.energy_threshold
        
        # Find transitions
        segments = []
        in_segment = False
        start_frame = 0
        
        for i, above in enumerate(above_threshold):
            if above and not in_segment:
                # Start of segment
                start_frame = i
                in_segment = True
            elif not above and in_segment:
                # End of segment
                end_frame = i
                duration = (end_frame - start_frame) * self.hop_length
                
                if duration >= self.min_duration:
                    segments.append((start_frame, end_frame))
                
                in_segment = False
        
        # Handle segment that extends to end
        if in_segment:
            end_frame = len(energy)
            duration = (end_frame - start_frame) * self.hop_length
            if duration >= self.min_duration:
                segments.append((start_frame, end_frame))
        
        # Merge close segments
        if segments:
            segments = self._merge_segments(segments)
        
        return segments

    def _merge_segments(self, segments: List[tuple[int, int]]) -> List[tuple[int, int]]:
        """Merge segments separated by small gaps."""
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for start, end in segments[1:]:
            last_start, last_end = merged[-1]
            gap = (start - last_end) * self.hop_length
            
            if gap <= self.merge_gap:
                # Merge
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))
        
        return merged

    def segment_audio(
        self,
        audio: np.ndarray,
        source_file: str,
        metadata: Optional[dict] = None,
    ) -> List[Segment]:
        """
        Segment audio into units.
        
        Args:
            audio: Audio array
            source_file: Source file path
            metadata: Additional metadata
            
        Returns:
            List of Segment objects
        """
        if metadata is None:
            metadata = {}
        
        # Validate audio
        if len(audio) == 0:
            logger.warning(f"Empty audio array for {source_file}, skipping segmentation")
            return []
        
        if len(audio) < self.min_duration:
            logger.warning(
                f"Audio too short for minimum duration: {len(audio)} < {self.min_duration} samples "
                f"({len(audio)/self.sample_rate:.2f}s < {self.min_duration_ms/1000:.2f}s) for {source_file}"
            )
            return []
        
        # Compute energy
        energy = self.compute_energy(audio)
        
        # Check if energy computation failed
        if len(energy) == 0:
            logger.warning(f"No energy frames computed for {source_file}, skipping segmentation")
            return []
        
        # Find segments
        frame_segments = self.find_segments(energy)
        
        # Extract audio segments
        segments = []
        for idx, (start_frame, end_frame) in enumerate(frame_segments):
            # Convert frame indices to sample indices
            start_sample = start_frame * self.hop_length
            end_sample = min(end_frame * self.hop_length + self.frame_length, len(audio))
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            
            # Check duration
            duration_samples = len(segment_audio)
            if duration_samples > self.max_duration:
                # Truncate
                segment_audio = segment_audio[: self.max_duration]
                duration_samples = self.max_duration
            
            if duration_samples < self.min_duration:
                continue
            
            # Create segment ID
            source_stem = Path(source_file).stem
            segment_id = f"{source_stem}_seg{idx:04d}"
            
            # Create segment
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            segment = Segment(
                start_time=start_time,
                end_time=end_time,
                audio=segment_audio,
                source_file=source_file,
                segment_id=segment_id,
                metadata={
                    **metadata,
                    "frame_start": start_frame,
                    "frame_end": end_frame,
                    "duration_samples": duration_samples,
                    "duration_seconds": end_time - start_time,
                },
            )
            
            segments.append(segment)
        
        logger.info(f"Segmented {source_file}: {len(segments)} segments")
        return segments

    def segment_file(
        self,
        file_path: str | Path,
        output_dir: str | Path,
        metadata: Optional[dict] = None,
    ) -> List[Segment]:
        """
        Load audio file and segment it.
        
        Args:
            file_path: Input audio file
            output_dir: Directory to save segments
            metadata: Additional metadata
            
        Returns:
            List of Segment objects
        """
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        try:
            audio, sr = load_audio(file_path, sample_rate=self.sample_rate)
        except Exception as e:
            logger.error(f"Failed to load audio from {file_path}: {e}")
            raise
        
        # Validate loaded audio
        if len(audio) == 0:
            raise ValueError(f"Loaded audio is empty: {file_path}")
        
        if sr != self.sample_rate:
            logger.warning(f"Sample rate mismatch: expected {self.sample_rate}, got {sr}")
            self.sample_rate = sr
            # Recompute frame/hop lengths
            self.frame_length = int(self.frame_length_ms * sr / 1000)
            self.hop_length = int(self.hop_length_ms * sr / 1000)
            self.min_duration = int(self.min_duration_ms * sr / 1000)
            self.max_duration = int(self.max_duration_ms * sr / 1000)
            self.merge_gap = int(self.merge_gap_ms * sr / 1000)
        
        # Add file metadata
        if metadata is None:
            metadata = {}
        
        metadata["source_file"] = str(file_path)
        metadata["sample_rate"] = sr
        metadata["original_length"] = len(audio) / sr
        
        # Segment
        segments = self.segment_audio(audio, str(file_path), metadata)
        
        # Save segments
        saved_count = 0
        for segment in segments:
            output_path = output_dir / f"{segment.segment_id}.wav"
            try:
                save_audio(segment.audio, output_path, self.sample_rate)
                # Verify file was actually created
                if not output_path.exists():
                    logger.error(f"Segment file was not created: {output_path}")
                    raise FileNotFoundError(f"Segment file was not created: {output_path}")
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save segment {segment.segment_id} to {output_path}: {e}")
                raise
        
        if saved_count > 0:
            logger.debug(f"Saved {saved_count} segments from {file_path.name}")
        
        return segments

    def apply_bandpass(
        self,
        audio: np.ndarray,
        fmin: float,
        fmax: float,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Apply bandpass filter.
        
        Args:
            audio: Audio array
            fmin: Low cutoff frequency
            fmax: High cutoff frequency
            sample_rate: Sample rate
            
        Returns:
            Filtered audio
        """
        # Validate inputs
        if len(audio) == 0:
            logger.warning("Empty audio array provided to bandpass filter")
            return audio
        
        if fmin >= fmax:
            logger.warning(f"Invalid frequency range: fmin={fmin} >= fmax={fmax}, skipping filter")
            return audio
        
        if fmax >= sample_rate / 2:
            logger.warning(
                f"fmax ({fmax} Hz) >= Nyquist frequency ({sample_rate/2} Hz), "
                "adjusting to Nyquist - 1 Hz"
            )
            fmax = sample_rate / 2 - 1
        
        nyquist = sample_rate / 2
        low = max(0.01, fmin / nyquist)  # Ensure low > 0
        high = min(0.99, fmax / nyquist)  # Ensure high < 1
        
        if low >= high:
            logger.warning(f"Invalid normalized frequency range: low={low} >= high={high}, skipping filter")
            return audio
        
        try:
            # Design Butterworth filter
            sos = signal.butter(4, [low, high], btype="band", output="sos")
            filtered = signal.sosfilt(sos, audio)
            return filtered
        except Exception as e:
            logger.error(f"Bandpass filter failed: {e}, returning original audio")
            return audio

