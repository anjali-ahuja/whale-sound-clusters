# WhaleSound Clusters

An exploratory bioacoustics project for discovering recurring acoustic motifs (clusters) in humpback whale songs from SanctSound audio data.

## Project Philosophy

**This project is NOT about "translating whale language."** 

We are exploring acoustic patterns—recurring motifs in whale vocalizations—through unsupervised clustering. The clusters represent acoustic similarity, not semantic meaning. Contextual associations, behavioral interpretations, and deeper analysis come later. This is a tool for discovery, not translation.

## Overview

This project processes 30-second audio files (WAV or FLAC) from SanctSound stored in Google Cloud Storage (`gs://noaa-passive-bioacoustic/sanctsound/audio`), segments them into short acoustic units (1-3 seconds), extracts features, and clusters them to discover recurring patterns. An interactive UI allows exploration of the discovered clusters.

### Key Features

- **GCS Downloader**: Download curated subsets of audio files from public GCS buckets
- **Audio Segmentation**: Energy-based segmentation with configurable thresholds
- **Feature Extraction**: Log-mel spectrogram features (with optional embedding support)
- **Clustering**: UMAP dimensionality reduction + HDBSCAN clustering
- **Interactive UI**: Streamlit app with museum-like design for exploring clusters
- **Reproducible**: Deterministic runs with seed handling and caching

## Requirements

- Python >= 3.11
- Poetry (for dependency management)
- Optional: `gsutil` (for alternative download method)

## Quick Start

### 1. Setup

```bash
# Install poetry if needed (recommended: via pipx)
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry

# Alternative: via official installer (may have SSL issues on macOS)
# curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
make setup
# or
poetry install
```

### 2. Configure

Copy the default config and customize:

```bash
cp configs/default.yaml configs/local.yaml
# Edit configs/local.yaml as needed
```

Key settings:
- `data.max_files`: Number of audio files to download (default: 200)
- `data.gcs_bucket`: GCS bucket name (default: `"noaa-passive-bioacoustic"`)
- `data.gcs_prefix`: Path prefix within bucket (default: `"sanctsound/audio"`)
- `audio.sample_rate`: Target sample rate (default: 4000 Hz)
- `segmentation.*`: Segmentation parameters
- `clustering.*`: Clustering parameters

### 3. Run Pipeline

```bash
# Full pipeline (download -> segment -> featurize -> cluster)
make run-pipeline

# Or step by step:
make download      # Download audio files
make segment       # Segment into units
make featurize     # Extract features
make cluster       # Run clustering
```

### 4. Launch UI

```bash
make run-ui
# or
poetry run whale-ui
```

The UI will open at `http://localhost:8501`

## Usage

### Command Line Interface

The project provides several CLI commands:

```bash
# Download audio files
poetry run whale-download --max-files 200

# Segment audio files
poetry run whale-segment

# Extract features from segments
poetry run whale-featurize

# Extract features from full audio files (skip segmentation)
poetry run whale-featurize --use-full-audio --audio-dir data/downloads/ci01/sanctsound_ci01_01/audio

# Run clustering
poetry run whale-cluster

# Launch UI
poetry run whale-ui

# Run full pipeline
poetry run whale-pipeline
```

### Configuration

Configuration is managed via YAML files in `configs/`. The system looks for:
1. `configs/local.yaml` (user-specific, gitignored)
2. `configs/default.yaml` (default settings)

Key configuration sections:

- **data**: GCS bucket settings, download parameters
- **audio**: Sample rate, bandpass filtering
- **segmentation**: Energy thresholds, duration limits
- **features**: Feature extraction method and parameters
  - `use_full_audio`: Skip segmentation and use full audio files (default: `false`)
  - `audio_dir`: Directory with full audio files when `use_full_audio: true` (default: `null`, uses `data.output_dir`)
- **clustering**: UMAP and HDBSCAN parameters
- **ui**: UI display settings

### Data Flow

**Standard pipeline (with segmentation):**
```
gs://noaa-passive-bioacoustic/sanctsound/audio/
    ↓ (download)
data/downloads/*.wav, *.flac
    ↓ (segment)
data/segments/*.wav + metadata.json
    ↓ (featurize)
data/features/features.parquet
    ↓ (cluster)
data/clusters/clusters.parquet + exemplars.json + summary.json
    ↓ (ui)
Interactive exploration
```

**Alternative pipeline (skip segmentation):**
```
gs://noaa-passive-bioacoustic/sanctsound/audio/
    ↓ (download)
data/downloads/*.wav, *.flac
    ↓ (featurize --use-full-audio)
data/features/features.parquet  # One feature vector per audio file
    ↓ (cluster)
data/clusters/clusters.parquet + exemplars.json + summary.json
    ↓ (ui)
Interactive exploration
```

## Project Structure

```
WhaleSoundClusters/
├── src/whalesound_cluster/
│   ├── io/              # GCS downloader
│   ├── audio/           # Audio I/O, segmentation
│   ├── features/        # Feature extraction
│   ├── cluster/         # Clustering pipeline
│   ├── ui/              # Streamlit app
│   ├── utils/           # Config, logging
│   └── cli.py           # CLI entrypoints
├── configs/             # Configuration files
├── data/                # Data directory (gitignored)
│   ├── downloads/       # Downloaded audio files (WAV/FLAC)
│   ├── segments/        # Segmented audio units
│   ├── features/        # Extracted features
│   └── clusters/        # Clustering results
├── tests/               # Unit tests
├── pyproject.toml       # Poetry configuration
├── Makefile             # Common tasks
└── README.md
```

## Development

### Linting and Formatting

```bash
make lint      # Run ruff linter
make format    # Format code with ruff
```

### Testing

```bash
make test      # Run tests with coverage
```

### Cleanup

```bash
make clean     # Remove generated files and caches
```

## Data Source

The project downloads audio files from the **NOAA Passive Bioacoustic** Google Cloud Storage bucket:

- **Bucket**: `noaa-passive-bioacoustic`
- **Path**: `sanctsound/audio/`
- **Full GCS path**: `gs://noaa-passive-bioacoustic/sanctsound/audio/`
- **Access**: Public bucket (no authentication required)
- **Browser access**: [View in Google Cloud Console](https://console.cloud.google.com/storage/browser/noaa-passive-bioacoustic/sanctsound)

The SanctSound dataset contains passive acoustic monitoring data from various marine sanctuaries, including humpback whale vocalizations.

## GCS Access

The project supports two methods for downloading from GCS:

1. **Python-native** (default): Uses `google-cloud-storage` library
   - For public buckets: automatically uses anonymous access
   - For private buckets: requires Application Default Credentials
     ```bash
     gcloud auth application-default login
     ```

2. **gsutil**: Shells out to `gsutil` command
   - Requires `gsutil` to be installed and configured
   - Set `data.download_method: "gsutil"` in config

## Audio Processing

### Supported Formats

The pipeline supports both **WAV** and **FLAC** audio files. Files are automatically detected and processed using `librosa`, which handles format conversion and resampling as needed.

### Segmentation

Segments are extracted based on:
- Short-time RMS energy above a threshold
- Minimum/maximum duration constraints
- Gap merging for close segments

### Feature Extraction

Default: Log-mel spectrogram with mean+std summarization
- Configurable: `n_mels`, `fmin`, `fmax`, `hop_length`, `n_fft`
- Summarization: `mean_std` (fixed-size vector) or `flatten` (variable-size)

**Skip Segmentation Option**: You can skip the segmentation step and extract features directly from full audio files:
```bash
poetry run whale-featurize --use-full-audio --audio-dir <path-to-audio-files>
```

Or configure in `configs/local.yaml`:
```yaml
features:
  use_full_audio: true
  audio_dir: "data/downloads/ci01/sanctsound_ci01_01/audio"  # or null to use data.output_dir
```

This extracts one feature vector per audio file (instead of per segment), useful for clustering at the file level rather than segment level.

Optional: Embedding models (placeholder for future implementation)

### Clustering

- **UMAP**: Reduces features to 2D/3D for visualization
- **HDBSCAN**: Density-based clustering on original features
- **Exemplars**: Representative segments per cluster (centroid or diverse selection)

## UI Features

The Streamlit UI provides:

- **Explorer Tab**:
  - Interactive 2D scatter plot of UMAP embeddings
  - Color-coded by cluster
  - Hover metadata
  - Click-to-play audio with spectrogram visualization
  - Filters: cluster, station, date range

- **Gallery Tab**:
  - Cluster exemplars displayed in a grid
  - Audio playback for each exemplar
  - Museum-like elegant design

## Limitations and Future Work

### Current Limitations

- Embedding model support is placeholder (defaults to log-mel)
- No video/context layer integration yet
- UI uses click-to-play (browser autoplay restrictions)

### Future Enhancements

- Add pretrained audio embedding models (OpenL3, etc.)
- Integrate video/contextual data layers
- Support for additional audio formats (currently supports WAV and FLAC)
- Batch processing optimizations
- Export capabilities (CSV, JSON, audio playlists)

## Contributing

This is an exploratory research project. Contributions welcome for:
- Feature extraction improvements
- Clustering algorithm alternatives
- UI enhancements
- Documentation improvements

## License

[Add your license here]

## Acknowledgments

- **SanctSound project** for the audio data
- **NOAA** for making the data publicly available via Google Cloud Storagesanctsound`

## Citation

If you use this project, please cite:

```
WhaleSound Clusters: Exploratory bioacoustics for humpback whale song units
[Your citation details]
```

