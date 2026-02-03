"""CLI entrypoints for pipeline commands."""

import json
import os
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from whalesound_cluster.audio.segmenter import AudioSegmenter
from whalesound_cluster.cluster.pipeline import ClusteringPipeline
from whalesound_cluster.features.extractor import FeatureExtractor
from whalesound_cluster.io.downloader import GCSDownloader
from whalesound_cluster.utils.config import load_config
from whalesound_cluster.utils.logging import setup_logging

logger = setup_logging()


@click.group()
def cli():
    """WhaleSound Clusters CLI."""
    pass


@cli.command()
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--max-files", type=int, default=None, help="Maximum files to download")
@click.option("--skip-existing", is_flag=True, default=None, help="Skip files that already exist locally")
def download(config: Optional[str], max_files: Optional[int], skip_existing: Optional[bool]):
    """Download audio files from GCS."""
    cfg = load_config(config)
    data_cfg = cfg["data"]
    
    # Override with CLI args
    if max_files:
        data_cfg["max_files"] = max_files
    
    # Use CLI flag if provided, otherwise use config, default to False
    if skip_existing is None:
        skip_existing = data_cfg.get("skip_existing", False)
    
    downloader = GCSDownloader(
        bucket_name=data_cfg["gcs_bucket"],
        prefix=data_cfg["gcs_prefix"],
        output_dir=data_cfg["output_dir"],
        method=data_cfg["download_method"],
    )
    
    downloaded = downloader.download_batch(
        max_files=data_cfg.get("max_files"),
        date_range=data_cfg.get("date_range"),
        skip_existing=skip_existing,
    )
    
    logger.info(f"Downloaded {len(downloaded)} files")


@cli.command()
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--input-dir", type=str, default=None, help="Input directory")
@click.option("--output-dir", type=str, default=None, help="Output directory")
def segment(config: Optional[str], input_dir: Optional[str], output_dir: Optional[str]):
    """Segment audio files into units."""
    cfg = load_config(config)
    audio_cfg = cfg["audio"]
    seg_cfg = cfg["segmentation"]
    
    # Override with CLI args
    if input_dir is None:
        input_dir = cfg["data"]["output_dir"]
    if output_dir is None:
        output_dir = seg_cfg["output_dir"]
    
    # Initialize segmenter
    segmenter = AudioSegmenter(
        frame_length_ms=seg_cfg["frame_length_ms"],
        hop_length_ms=seg_cfg["hop_length_ms"],
        energy_threshold=seg_cfg["energy_threshold"],
        min_duration_ms=seg_cfg["min_duration_ms"],
        max_duration_ms=seg_cfg["max_duration_ms"],
        merge_gap_ms=seg_cfg["merge_gap_ms"],
        sample_rate=audio_cfg["sample_rate"],
    )
    
    # Find audio files (both .wav and .flac)
    input_path = Path(input_dir)
    all_audio_files = list(input_path.glob("**/*.wav")) + list(input_path.glob("**/*.flac"))
    
    # Filter out empty files
    audio_files = []
    empty_files = []
    for f in all_audio_files:
        try:
            if f.stat().st_size == 0:
                empty_files.append(f)
            else:
                audio_files.append(f)
        except OSError:
            # File might have been deleted or is inaccessible
            continue
    
    if empty_files:
        logger.warning(
            f"Skipping {len(empty_files)} empty files. "
            "These may be from failed downloads."
        )
    
    if not audio_files:
        logger.warning(f"No valid audio files found in {input_dir}")
        if empty_files:
            logger.warning(
                f"Found {len(empty_files)} empty files. "
                "Check your download configuration and GCS bucket access."
            )
        return
    
    logger.info(f"Segmenting {len(audio_files)} audio files")
    
    # Clear existing metadata and segment files at the start
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_file = output_path / "segments_metadata.json"
    
    # Create a separate temp directory for filtered audio files (completely isolated from segments)
    temp_dir = output_path / ".temp_filtered"
    temp_dir.mkdir(exist_ok=True)
    
    # Remove existing metadata file
    if metadata_file.exists():
        logger.info("Clearing existing metadata file")
        metadata_file.unlink()
    
    # Remove existing segment files
    existing_segments = list(output_path.glob("*_seg*.wav"))
    if existing_segments:
        logger.info(f"Clearing {len(existing_segments)} existing segment files")
        for seg_file in existing_segments:
            try:
                seg_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove {seg_file}: {e}")
    
    # Segment each file
    all_segments = []
    
    for audio_file in audio_files:
        temp_path = None
        try:
            # Apply bandpass if enabled
            if audio_cfg["bandpass"]["enabled"]:
                import numpy as np
                from whalesound_cluster.audio.io import load_audio
                
                audio, sr = load_audio(audio_file, sample_rate=audio_cfg["sample_rate"])
                audio = segmenter.apply_bandpass(
                    audio,
                    audio_cfg["bandpass"]["fmin"],
                    audio_cfg["bandpass"]["fmax"],
                    sr,
                )
                # Save filtered audio temporarily in isolated temp directory
                from whalesound_cluster.audio.io import save_audio
                # Use unique temp filename per source file
                temp_path = temp_dir / f"{os.getpid()}_{Path(audio_file).stem}.wav"
                save_audio(audio, temp_path, sr)
                audio_file = temp_path
            
            segments = segmenter.segment_file(
                audio_file,
                output_dir,
                metadata={"source_file": str(audio_file)},
            )
            all_segments.extend(segments)
            
            # Clean up temp file immediately after processing (segments are already saved)
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to segment {audio_file}: {e}")
            # Clean up temp file even on error
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove temp file {temp_path} after error: {cleanup_error}")
            continue
    
    # Clean up temp directory (should be empty, but remove any leftover files)
    try:
        if temp_dir.exists():
            remaining_files = list(temp_dir.glob("*.wav"))
            if remaining_files:
                logger.warning(f"Cleaning up {len(remaining_files)} leftover temp files")
                for f in remaining_files:
                    try:
                        f.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {f}: {e}")
            # Try to remove the directory (will fail if not empty, which is fine)
            try:
                temp_dir.rmdir()
            except OSError:
                pass  # Directory not empty, that's okay
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
    
    # Save metadata
    segments_metadata = [
        {
            "segment_id": s.segment_id,
            "source_file": s.source_file,
            "start_time": s.start_time,
            "end_time": s.end_time,
            **s.metadata,
        }
        for s in all_segments
    ]
    
    # Verify segment files exist
    segment_files = list(output_path.glob("*_seg*.wav"))
    segment_file_ids = {f.stem for f in segment_files}
    segment_metadata_ids = {s["segment_id"] for s in segments_metadata}
    
    missing_files = segment_metadata_ids - segment_file_ids
    if missing_files:
        logger.warning(
            f"Found {len(missing_files)} segments in metadata but missing segment files. "
            f"First few missing: {list(missing_files)[:5]}"
        )
        # Only include segments that have files in the metadata
        segments_metadata = [
            s for s in segments_metadata if s["segment_id"] in segment_file_ids
        ]
        logger.warning(
            f"Reduced metadata to {len(segments_metadata)} segments that have corresponding files"
        )
    
    with open(metadata_file, "w") as f:
        json.dump(segments_metadata, f, indent=2)
    
    logger.info(
        f"Created {len(all_segments)} segments, "
        f"{len(segment_files)} segment files found, "
        f"{len(segments_metadata)} segments in metadata"
    )


@cli.command()
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--segments-dir", type=str, default=None, help="Segments directory")
@click.option("--output-path", type=str, default=None, help="Output parquet path")
@click.option("--use-full-audio", is_flag=True, default=False, help="Skip segmentation and use full audio files instead")
@click.option("--audio-dir", type=str, default=None, help="Directory containing full audio files (used with --use-full-audio)")
def featurize(
    config: Optional[str],
    segments_dir: Optional[str],
    output_path: Optional[str],
    use_full_audio: bool,
    audio_dir: Optional[str],
):
    """Extract features from segments or full audio files."""
    cfg = load_config(config)
    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]
    
    # Override with CLI args
    if output_path is None:
        output_path = Path(feat_cfg["output_dir"]) / "features.parquet"
    else:
        output_path = Path(output_path)
    
    # Initialize extractor
    if feat_cfg["method"] == "logmel":
        extractor = FeatureExtractor(
            method="logmel",
            sample_rate=audio_cfg["sample_rate"],
            n_mels=feat_cfg["logmel"]["n_mels"],
            fmin=feat_cfg["logmel"]["fmin"],
            fmax=feat_cfg["logmel"]["fmax"],
            hop_length=feat_cfg["logmel"]["hop_length"],
            n_fft=feat_cfg["logmel"]["n_fft"],
            summarize=feat_cfg["logmel"]["summarize"],
        )
    else:
        extractor = FeatureExtractor(method=feat_cfg["method"])
    
    if use_full_audio:
        # Use full audio files instead of segments
        if audio_dir is None:
            audio_dir = cfg["data"]["output_dir"]
        
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            logger.error(f"Audio directory not found: {audio_dir}")
            return
        
        # Find audio files (both .wav and .flac)
        audio_files = list(audio_path.glob("**/*.wav")) + list(audio_path.glob("**/*.flac"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir}")
            return
        
        logger.info(f"Extracting features from {len(audio_files)} full audio files")
        
        # Create minimal metadata for full audio files
        metadata_list = []
        for audio_file in audio_files:
            metadata = {
                "source_file": str(audio_file),
                "segment_id": audio_file.stem,
            }
            metadata_list.append(metadata)
        
        df = extractor.extract_batch(audio_files, metadata_list, output_path)
    else:
        # Use segments (original behavior)
        if segments_dir is None:
            segments_dir = cfg["segmentation"]["output_dir"]
        
        # Load segment metadata
        metadata_file = Path(segments_dir) / "segments_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                segments_metadata = json.load(f)
        else:
            segments_metadata = []
        
        # Find segment files
        segments_path = Path(segments_dir)
        segment_files = list(segments_path.glob("*.wav"))
        
        # Filter out temp files
        segment_files = [f for f in segment_files if not f.name.startswith(".temp")]
        
        if not segment_files and segments_metadata:
            # No segment files found, but we have metadata - extract from source files
            logger.info(
                f"No segment files found in {segments_dir}, but found {len(segments_metadata)} segments in metadata. "
                "Extracting features from source audio files using segment timestamps."
            )
            
            # Resolve source file paths
            data_dir = Path(cfg["data"]["output_dir"])
            resolved_metadata = []
            for seg_meta in segments_metadata:
                source_file = seg_meta.get("source_file", "")
                if source_file:
                    source_path = Path(source_file)
                    resolved_path = None
                    
                    # Check if it's a temp file (starts with .temp_filtered_)
                    if source_path.name.startswith(".temp_filtered_"):
                        # Extract original filename: .temp_filtered_<pid>_<original>.wav -> <original>.flac
                        # Pattern: .temp_filtered_<numbers>_<original_filename>.wav
                        temp_name = source_path.stem  # Remove .wav extension
                        # Remove .temp_filtered_ prefix
                        if temp_name.startswith(".temp_filtered_"):
                            # Find the first underscore after the prefix, then find the next one after the PID
                            # .temp_filtered_3292_SanctSound_CI01_01_671379494_20181122T145632Z
                            # We want: SanctSound_CI01_01_671379494_20181122T145632Z
                            parts = temp_name.replace(".temp_filtered_", "", 1).split("_", 1)
                            if len(parts) == 2:
                                # parts[0] is the PID, parts[1] is the original filename
                                original_name = parts[1]
                                # Try to find the original file (could be .flac or .wav)
                                for ext in [".flac", ".wav"]:
                                    found_files = list(data_dir.glob(f"**/{original_name}{ext}"))
                                    if found_files:
                                        resolved_path = found_files[0]
                                        break
                    
                    # If not resolved yet, try direct path resolution
                    if not resolved_path:
                        if source_path.is_absolute() and source_path.exists():
                            resolved_path = source_path
                        elif (data_dir / source_path).exists():
                            resolved_path = data_dir / source_path
                        elif (segments_path / source_path).exists():
                            resolved_path = segments_path / source_path
                        else:
                            # Try to find by filename
                            filename = source_path.name
                            found_files = list(data_dir.glob(f"**/{filename}"))
                            if found_files:
                                resolved_path = found_files[0]
                    
                    if resolved_path and resolved_path.exists():
                        seg_meta["source_file"] = str(resolved_path)
                        resolved_metadata.append(seg_meta)
                    else:
                        logger.warning(f"Could not resolve source file: {source_file}, skipping segment")
                        continue
                else:
                    logger.warning(f"Segment {seg_meta.get('segment_id')} missing source_file, skipping")
            
            if not resolved_metadata:
                logger.error("Could not resolve any source files from metadata")
                return
            
            logger.info(f"Resolved {len(resolved_metadata)} segments from {len(segments_metadata)} total segments")
            
            # Optionally limit number of segments for testing/faster processing
            max_segments = feat_cfg.get("max_segments")
            if max_segments and len(resolved_metadata) > max_segments:
                logger.info(f"Limiting to {max_segments} segments (out of {len(resolved_metadata)} total)")
                resolved_metadata = resolved_metadata[:max_segments]
            else:
                logger.info(f"Processing all {len(resolved_metadata)} segments")
            
            logger.info(f"Extracting features from {len(resolved_metadata)} segments using source files")
            df = extractor.extract_batch_from_metadata(resolved_metadata, output_path)
        elif segment_files:
            # Use existing segment files
            logger.info(f"Extracting features from {len(segment_files)} segment files")
            
            # Create metadata dict
            metadata_dict = {m["segment_id"]: m for m in segments_metadata}
            
            # Extract features
            metadata_list = [metadata_dict.get(f.stem, {}) for f in segment_files]
            df = extractor.extract_batch(segment_files, metadata_list, output_path)
        else:
            logger.error(f"No segment files found in {segments_dir} and no metadata available")
            return
    
    logger.info(f"Extracted features: {df.shape}")


@cli.command()
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--features-path", type=str, default=None, help="Features parquet path")
@click.option("--output-dir", type=str, default=None, help="Output directory")
def cluster(config: Optional[str], features_path: Optional[str], output_dir: Optional[str]):
    """Run clustering pipeline."""
    cfg = load_config(config)
    cluster_cfg = cfg["clustering"]
    
    # Override with CLI args
    if features_path is None:
        features_path = Path(cfg["features"]["output_dir"]) / "features.parquet"
    else:
        features_path = Path(features_path)
    
    if output_dir is None:
        output_dir = cluster_cfg["output_dir"]
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return
    
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    
    # Initialize pipeline
    pipeline = ClusteringPipeline(
        random_seed=cluster_cfg["random_seed"],
        umap_n_components=cluster_cfg["umap"]["n_components"],
        umap_n_neighbors=cluster_cfg["umap"]["n_neighbors"],
        umap_min_dist=cluster_cfg["umap"]["min_dist"],
        umap_metric=cluster_cfg["umap"]["metric"],
        hdbscan_min_cluster_size=cluster_cfg["hdbscan"]["min_cluster_size"],
        hdbscan_min_samples=cluster_cfg["hdbscan"]["min_samples"],
        hdbscan_cluster_selection_method=cluster_cfg["hdbscan"]["cluster_selection_method"],
        hdbscan_metric=cluster_cfg["hdbscan"]["metric"],
    )
    
    # Run pipeline
    result_df = pipeline.run_pipeline(
        df,
        Path(output_dir),
        n_exemplars=cluster_cfg["exemplars"]["n_per_cluster"],
    )
    
    logger.info(f"Clustering complete: {len(result_df)} segments, {len(set(result_df['cluster_id']))} clusters")


@cli.command()
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--port", type=int, default=8501, help="Port for Streamlit")
def ui(config: Optional[str], port: int):
    """Launch Streamlit UI."""
    import subprocess
    import sys
    
    cfg = load_config(config)
    ui_cfg = cfg.get("ui", {})
    
    cluster_path = Path(cfg["clustering"]["output_dir"]) / "clusters.parquet"
    segments_dir = Path(cfg["segmentation"]["output_dir"])
    exemplars_path = Path(cfg["clustering"]["output_dir"]) / "exemplars.json"
    
    # Run streamlit
    app_path = Path(__file__).parent / "ui" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--",
        str(cluster_path),
        str(segments_dir),
        str(exemplars_path),
    ]
    
    subprocess.run(cmd)


@cli.command()
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--skip-existing", is_flag=True, default=None, help="Skip downloading files that already exist locally")
def pipeline(config: Optional[str], skip_existing: Optional[bool]):
    """Run full pipeline: download -> segment -> featurize -> cluster."""
    logger.info("Running full pipeline...")
    
    # Download
    logger.info("Step 1/4: Downloading audio files...")
    download.callback(config=config, max_files=None, skip_existing=skip_existing)
    
    # Segment
    logger.info("Step 2/4: Segmenting audio files...")
    segment.callback(config=config, input_dir=None, output_dir=None)
    
    # Featurize
    logger.info("Step 3/4: Extracting features...")
    featurize.callback(config=config, segments_dir=None, output_path=None)
    
    # Cluster
    logger.info("Step 4/4: Clustering...")
    cluster.callback(config=config, features_path=None, output_dir=None)
    
    logger.info("Pipeline complete!")


# Entry points for poetry scripts (direct function calls)
def download_cmd():
    """Entry point for whale-download command."""
    import sys
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download audio files from GCS")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum files to download")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist locally")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    
    # Parse known args only (in case there are other args we don't know about)
    args, _ = parser.parse_known_args()
    
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    
    # Override with CLI args
    max_files = args.max_files if args.max_files is not None else data_cfg.get("max_files")
    
    # Use CLI flag if provided, otherwise use config
    # Check if --skip-existing was explicitly provided in command line
    skip_existing_provided = "--skip-existing" in sys.argv
    if skip_existing_provided:
        skip_existing = args.skip_existing
    else:
        skip_existing = data_cfg.get("skip_existing", False)
    
    downloader = GCSDownloader(
        bucket_name=data_cfg["gcs_bucket"],
        prefix=data_cfg["gcs_prefix"],
        output_dir=data_cfg["output_dir"],
        method=data_cfg["download_method"],
    )
    downloaded = downloader.download_batch(
        max_files=max_files,
        date_range=data_cfg.get("date_range"),
        skip_existing=skip_existing,
    )
    logger.info(f"Downloaded {len(downloaded)} files")


def _process_single_audio_file(args):
    """
    Helper function to process a single audio file (for multiprocessing).
    
    Args:
        args: Tuple of (audio_file, seg_cfg, audio_cfg, output_dir)
    
    Returns:
        List of Segment objects or None if failed
    """
    audio_file, seg_cfg, audio_cfg, output_dir = args
    
    try:
        from pathlib import Path
        from whalesound_cluster.audio.segmenter import AudioSegmenter
        from whalesound_cluster.audio.io import load_audio, save_audio
        from whalesound_cluster.utils.logging import setup_logging
        import numpy as np
        
        logger = setup_logging()
        
        segmenter = AudioSegmenter(
            frame_length_ms=seg_cfg["frame_length_ms"],
            hop_length_ms=seg_cfg["hop_length_ms"],
            energy_threshold=seg_cfg["energy_threshold"],
            min_duration_ms=seg_cfg["min_duration_ms"],
            max_duration_ms=seg_cfg["max_duration_ms"],
            merge_gap_ms=seg_cfg["merge_gap_ms"],
            sample_rate=audio_cfg["sample_rate"],
        )
        
        # Handle bandpass filtering if enabled
        if audio_cfg["bandpass"]["enabled"]:
            audio, sr = load_audio(audio_file, sample_rate=audio_cfg["sample_rate"])
            audio = segmenter.apply_bandpass(
                audio,
                audio_cfg["bandpass"]["fmin"],
                audio_cfg["bandpass"]["fmax"],
                sr,
            )
            
            # Ensure segmenter uses the actual sample rate
            if sr != segmenter.sample_rate:
                segmenter.sample_rate = sr
                segmenter.frame_length = int(segmenter.frame_length_ms * sr / 1000)
                segmenter.hop_length = int(segmenter.hop_length_ms * sr / 1000)
                segmenter.min_duration = int(segmenter.min_duration_ms * sr / 1000)
                segmenter.max_duration = int(segmenter.max_duration_ms * sr / 1000)
                segmenter.merge_gap = int(segmenter.merge_gap_ms * sr / 1000)
            
            metadata = {
                "source_file": str(audio_file),
                "sample_rate": sr,
                "original_length": len(audio) / sr,
            }
            
            segments = segmenter.segment_audio(audio, str(audio_file), metadata)
            
            # Save segments (use original filename stem for segment IDs)
            output_path = Path(output_dir)
            saved_count = 0
            for segment in segments:
                seg_path = output_path / f"{segment.segment_id}.wav"
                save_audio(segment.audio, seg_path, segmenter.sample_rate)
                if not seg_path.exists():
                    raise FileNotFoundError(f"Segment file was not created: {seg_path}")
                saved_count += 1
            
            if saved_count > 0:
                logger.debug(f"Saved {saved_count} segments from {Path(audio_file).name}")
            
            return segments
        
        # If bandpass is disabled, use the existing segmenter file path for efficiency
        segments = segmenter.segment_file(
            audio_file,
            output_dir,
            metadata={"source_file": str(audio_file)},
        )
        
        return segments
    except Exception as e:
        logger.error(f"Failed to segment {audio_file}: {e}")
        return None


def segment_cmd():
    """Entry point for whale-segment command."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    cfg = load_config()
    audio_cfg = cfg["audio"]
    seg_cfg = cfg["segmentation"]
    input_dir = cfg["data"]["output_dir"]
    output_dir = seg_cfg["output_dir"]
    
    input_path = Path(input_dir)
    audio_files = list(input_path.glob("**/*.wav")) + list(input_path.glob("**/*.flac"))
    
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return
    
    # Check for large files and warn about memory usage
    large_files = []
    total_size_gb = 0
    for f in audio_files:
        try:
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size_gb += size_mb / 1024
            if size_mb > 500:  # Files larger than 500MB
                large_files.append((f, size_mb))
        except OSError:
            continue
    
    if large_files:
        logger.warning(
            f"Found {len(large_files)} large audio files (>500MB). "
            f"Total data size: ~{total_size_gb:.1f}GB. "
            f"Processing many large files simultaneously may require significant memory. "
            f"Consider reducing max_workers if you encounter memory issues."
        )
    
    # Determine number of workers with safe defaults to prevent system overload
    cpu_count = os.cpu_count() or 4
    max_workers_config = seg_cfg.get("max_workers")
    
    if max_workers_config is None:
        # Conservative default: use min(4, max(1, cpu_count - 2)) to leave resources for system
        # This prevents overwhelming the system with too many large audio files in memory
        # Formula ensures: at most 4 workers, at least 1 worker, and leaves 2 cores for system
        max_workers = min(4, max(1, cpu_count - 2))
        logger.info(
            f"Auto-detected {cpu_count} CPU cores. Using {max_workers} workers by default "
            f"(conservative setting to prevent system overload). "
            f"Set 'max_workers' in config to override."
        )
    else:
        # User-specified, but cap at CPU count for safety
        max_workers = min(int(max_workers_config), cpu_count)
        if max_workers >= cpu_count:
            logger.warning(
                f"Using {max_workers} workers (all CPU cores). "
                f"This may cause high memory usage and system slowdown. "
                f"Consider using fewer workers (e.g., {min(4, max(1, cpu_count - 2))}) for large files."
            )
        elif max_workers > 6:
            logger.warning(
                f"Using {max_workers} workers. Processing large audio files ({len(audio_files)} files) "
                f"with many workers may cause high memory usage. Monitor system resources."
            )
    
    logger.info(f"Segmenting {len(audio_files)} audio files with {max_workers} workers")
    
    # Clear existing metadata and segment files at the start
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_file = output_path / "segments_metadata.json"
    
    # Remove existing metadata file
    if metadata_file.exists():
        logger.info("Clearing existing metadata file")
        metadata_file.unlink()
    
    # Remove existing segment files (but keep temp files for now, they'll be cleaned up later)
    existing_segments = list(output_path.glob("*_seg*.wav"))
    if existing_segments:
        logger.info(f"Clearing {len(existing_segments)} existing segment files")
        for seg_file in existing_segments:
            try:
                seg_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove {seg_file}: {e}")
    
    all_segments = []
    
    # Prepare arguments for parallel processing
    process_args = [
        (audio_file, seg_cfg, audio_cfg, output_dir)
        for audio_file in audio_files
    ]
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(_process_single_audio_file, args): args[0]
            for args in process_args
        }
        
        # Process results with progress bar
        with tqdm(total=len(audio_files), desc="Segmenting files") as pbar:
            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    segments = future.result()
                    if segments is not None:
                        all_segments.extend(segments)
                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")
                finally:
                    pbar.update(1)
    
    segments_metadata = [
        {
            "segment_id": s.segment_id,
            "source_file": s.source_file,
            "start_time": s.start_time,
            "end_time": s.end_time,
            **s.metadata,
        }
        for s in all_segments
    ]
    
    # Verify segment files exist
    output_path = Path(output_dir)
    segment_files = list(output_path.glob("*_seg*.wav"))
    segment_file_ids = {f.stem for f in segment_files}
    segment_metadata_ids = {s["segment_id"] for s in segments_metadata}
    
    missing_files = segment_metadata_ids - segment_file_ids
    if missing_files:
        logger.warning(
            f"Found {len(missing_files)} segments in metadata but missing segment files. "
            f"First few missing: {list(missing_files)[:5]}"
        )
        # Only include segments that have files in the metadata
        segments_metadata = [
            s for s in segments_metadata if s["segment_id"] in segment_file_ids
        ]
        logger.warning(
            f"Reduced metadata to {len(segments_metadata)} segments that have corresponding files"
        )
    
    with open(metadata_file, "w") as f:
        json.dump(segments_metadata, f, indent=2)
    
    logger.info(
        f"Created {len(all_segments)} segments, "
        f"{len(segment_files)} segment files found, "
        f"{len(segments_metadata)} segments in metadata"
    )


def featurize_cmd():
    """Entry point for whale-featurize command."""
    import sys
    
    # Parse command-line arguments for flags
    use_full_audio = "--use-full-audio" in sys.argv
    audio_dir = None
    if "--audio-dir" in sys.argv:
        idx = sys.argv.index("--audio-dir")
        if idx + 1 < len(sys.argv):
            audio_dir = sys.argv[idx + 1]
    
    cfg = load_config()
    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]
    output_path = Path(feat_cfg["output_dir"]) / "features.parquet"
    
    # Override config with CLI args if provided
    if use_full_audio:
        feat_cfg["use_full_audio"] = True
    if audio_dir:
        feat_cfg["audio_dir"] = audio_dir
    
    # Initialize extractor
    if feat_cfg["method"] == "logmel":
        extractor = FeatureExtractor(
            method="logmel",
            sample_rate=audio_cfg["sample_rate"],
            n_mels=feat_cfg["logmel"]["n_mels"],
            fmin=feat_cfg["logmel"]["fmin"],
            fmax=feat_cfg["logmel"]["fmax"],
            hop_length=feat_cfg["logmel"]["hop_length"],
            n_fft=feat_cfg["logmel"]["n_fft"],
            summarize=feat_cfg["logmel"]["summarize"],
        )
    else:
        extractor = FeatureExtractor(method=feat_cfg["method"])
    
    # Check if we should use full audio files
    use_full_audio = feat_cfg.get("use_full_audio", False)
    
    if use_full_audio:
        # Use full audio files instead of segments
        audio_dir = feat_cfg.get("audio_dir") or cfg["data"]["output_dir"]
        audio_path = Path(audio_dir)
        
        if not audio_path.exists():
            logger.error(f"Audio directory not found: {audio_dir}")
            return
        
        # Find audio files (both .wav and .flac)
        audio_files = list(audio_path.glob("**/*.wav")) + list(audio_path.glob("**/*.flac"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir}")
            return
        
        logger.info(f"Extracting features from {len(audio_files)} full audio files")
        
        # Create minimal metadata for full audio files
        metadata_list = []
        for audio_file in audio_files:
            metadata = {
                "source_file": str(audio_file),
                "segment_id": audio_file.stem,
            }
            metadata_list.append(metadata)
        
        df = extractor.extract_batch(audio_files, metadata_list, output_path)
    else:
        # Use segments (original behavior)
        segments_dir = cfg["segmentation"]["output_dir"]
        
        # Load segment metadata
        metadata_file = Path(segments_dir) / "segments_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                segments_metadata = json.load(f)
        else:
            segments_metadata = []
        
        segments_path = Path(segments_dir)
        segment_files = list(segments_path.glob("*.wav"))
        
        # Filter out temp files
        segment_files = [f for f in segment_files if not f.name.startswith(".temp")]
        
        if not segment_files and segments_metadata:
            # No segment files found, but we have metadata - extract from source files
            logger.info(
                f"No segment files found in {segments_dir}, but found {len(segments_metadata)} segments in metadata. "
                "Extracting features from source audio files using segment timestamps."
            )
            
            # Resolve source file paths
            data_dir = Path(cfg["data"]["output_dir"])
            resolved_metadata = []
            for seg_meta in segments_metadata:
                source_file = seg_meta.get("source_file", "")
                if source_file:
                    source_path = Path(source_file)
                    resolved_path = None
                    
                    # Check if it's a temp file (starts with .temp_filtered_)
                    if source_path.name.startswith(".temp_filtered_"):
                        # Extract original filename: .temp_filtered_<pid>_<original>.wav -> <original>.flac
                        temp_name = source_path.stem
                        if temp_name.startswith(".temp_filtered_"):
                            parts = temp_name.replace(".temp_filtered_", "", 1).split("_", 1)
                            if len(parts) == 2:
                                original_name = parts[1]
                                # Try to find the original file (could be .flac or .wav)
                                for ext in [".flac", ".wav"]:
                                    found_files = list(data_dir.glob(f"**/{original_name}{ext}"))
                                    if found_files:
                                        resolved_path = found_files[0]
                                        break
                    
                    # If not resolved yet, try direct path resolution
                    if not resolved_path:
                        if source_path.is_absolute() and source_path.exists():
                            resolved_path = source_path
                        elif (data_dir / source_path).exists():
                            resolved_path = data_dir / source_path
                        elif (segments_path / source_path).exists():
                            resolved_path = segments_path / source_path
                        else:
                            # Try to find by filename
                            filename = source_path.name
                            found_files = list(data_dir.glob(f"**/{filename}"))
                            if found_files:
                                resolved_path = found_files[0]
                    
                    if resolved_path and resolved_path.exists():
                        seg_meta["source_file"] = str(resolved_path)
                        resolved_metadata.append(seg_meta)
                    else:
                        logger.warning(f"Could not resolve source file: {source_file}, skipping segment")
                        continue
                else:
                    logger.warning(f"Segment {seg_meta.get('segment_id')} missing source_file, skipping")
            
            if not resolved_metadata:
                logger.error("Could not resolve any source files from metadata")
                return
            
            logger.info(f"Resolved {len(resolved_metadata)} segments from {len(segments_metadata)} total segments")
            
            # Optionally limit number of segments for testing/faster processing
            max_segments = feat_cfg.get("max_segments")
            if max_segments and len(resolved_metadata) > max_segments:
                logger.info(f"Limiting to {max_segments} segments (out of {len(resolved_metadata)} total)")
                resolved_metadata = resolved_metadata[:max_segments]
            else:
                logger.info(f"Processing all {len(resolved_metadata)} segments")
            
            logger.info(f"Extracting features from {len(resolved_metadata)} segments using source files")
            df = extractor.extract_batch_from_metadata(resolved_metadata, output_path)
        elif segment_files:
            # Use existing segment files
            logger.info(f"Extracting features from {len(segment_files)} segment files")
            metadata_dict = {m["segment_id"]: m for m in segments_metadata}
            metadata_list = [metadata_dict.get(f.stem, {}) for f in segment_files]
            df = extractor.extract_batch(segment_files, metadata_list, output_path)
        else:
            logger.error(f"No segment files found in {segments_dir} and no metadata available")
            return
    
    logger.info(f"Extracted features: {df.shape}")


def cluster_cmd():
    """Entry point for whale-cluster command."""
    cfg = load_config()
    cluster_cfg = cfg["clustering"]
    features_path = Path(cfg["features"]["output_dir"]) / "features.parquet"
    output_dir = cluster_cfg["output_dir"]
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return
    
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    
    pipeline = ClusteringPipeline(
        random_seed=cluster_cfg["random_seed"],
        umap_n_components=cluster_cfg["umap"]["n_components"],
        umap_n_neighbors=cluster_cfg["umap"]["n_neighbors"],
        umap_min_dist=cluster_cfg["umap"]["min_dist"],
        umap_metric=cluster_cfg["umap"]["metric"],
        hdbscan_min_cluster_size=cluster_cfg["hdbscan"]["min_cluster_size"],
        hdbscan_min_samples=cluster_cfg["hdbscan"]["min_samples"],
        hdbscan_cluster_selection_method=cluster_cfg["hdbscan"]["cluster_selection_method"],
        hdbscan_metric=cluster_cfg["hdbscan"]["metric"],
    )
    
    result_df = pipeline.run_pipeline(
        df,
        Path(output_dir),
        n_exemplars=cluster_cfg["exemplars"]["n_per_cluster"],
    )
    
    logger.info(f"Clustering complete: {len(result_df)} segments, {len(set(result_df['cluster_id']))} clusters")


def ui_cmd():
    """Entry point for whale-ui command."""
    import subprocess
    import sys
    from pathlib import Path
    
    cfg = load_config()
    ui_cfg = cfg.get("ui", {})
    port = ui_cfg.get("port", 8501)
    
    cluster_path = Path(cfg["clustering"]["output_dir"]) / "clusters.parquet"
    segments_dir = Path(cfg["segmentation"]["output_dir"])
    exemplars_path = Path(cfg["clustering"]["output_dir"]) / "exemplars.json"
    
    # Use streamlit_app.py as entry point
    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--",
        str(cluster_path),
        str(segments_dir),
        str(exemplars_path),
    ]
    
    subprocess.run(cmd)


def pipeline_cmd():
    """Entry point for whale-pipeline command."""
    logger.info("Running full pipeline...")
    download_cmd()
    logger.info("Step 2/4: Segmenting audio files...")
    segment_cmd()
    logger.info("Step 3/4: Extracting features...")
    featurize_cmd()
    logger.info("Step 4/4: Clustering...")
    cluster_cmd()
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    cli()

