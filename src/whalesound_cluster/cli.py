"""CLI entrypoints for pipeline commands."""

import json
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
    
    # Segment each file
    all_segments = []
    metadata_file = Path(output_dir) / "segments_metadata.json"
    
    for audio_file in audio_files:
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
                # Save filtered audio temporarily
                from whalesound_cluster.audio.io import save_audio
                temp_path = Path(output_dir) / ".temp_filtered.wav"
                save_audio(audio, temp_path, sr)
                audio_file = temp_path
            
            segments = segmenter.segment_file(
                audio_file,
                output_dir,
                metadata={"source_file": str(audio_file)},
            )
            all_segments.extend(segments)
        except Exception as e:
            logger.error(f"Failed to segment {audio_file}: {e}")
            continue
    
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
    
    with open(metadata_file, "w") as f:
        json.dump(segments_metadata, f, indent=2)
    
    logger.info(f"Created {len(all_segments)} segments")


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
        
        if not segment_files:
            logger.warning(f"No segment files found in {segments_dir}")
            return
        
        logger.info(f"Extracting features from {len(segment_files)} segments")
        
        # Create metadata dict
        metadata_dict = {m["segment_id"]: m for m in segments_metadata}
        
        # Extract features
        metadata_list = [metadata_dict.get(f.stem, {}) for f in segment_files]
        df = extractor.extract_batch(segment_files, metadata_list, output_path)
    
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
        import os
        
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
            # Use a unique temp file per process to avoid conflicts
            temp_path = Path(output_dir) / f".temp_filtered_{os.getpid()}_{Path(audio_file).stem}.wav"
            save_audio(audio, temp_path, sr)
            audio_file = temp_path
        
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
    
    # Determine number of workers with safe upper limit (CPU count)
    cpu_count = os.cpu_count() or 4
    max_workers_config = seg_cfg.get("max_workers")
    if max_workers_config is None:
        # Auto-detect: use CPU count
        max_workers = cpu_count
    else:
        # User-specified, but cap at CPU count for safety
        max_workers = min(int(max_workers_config), cpu_count)
    
    logger.info(f"Segmenting {len(audio_files)} audio files with {max_workers} workers")
    
    all_segments = []
    metadata_file = Path(output_dir) / "segments_metadata.json"
    
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
    
    # Clean up any temp files
    temp_files = list(Path(output_dir).glob(".temp_filtered_*.wav"))
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except Exception:
            pass
    
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
    
    with open(metadata_file, "w") as f:
        json.dump(segments_metadata, f, indent=2)
    
    logger.info(f"Created {len(all_segments)} segments")


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
        
        if not segment_files:
            logger.warning(f"No segment files found in {segments_dir}")
            return
        
        logger.info(f"Extracting features from {len(segment_files)} segments")
        metadata_dict = {m["segment_id"]: m for m in segments_metadata}
        
        metadata_list = [metadata_dict.get(f.stem, {}) for f in segment_files]
        df = extractor.extract_batch(segment_files, metadata_list, output_path)
    
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

