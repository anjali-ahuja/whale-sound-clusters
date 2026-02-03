"""Tests for clustering pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from whalesound_cluster.cluster.pipeline import ClusteringPipeline


def test_umap_reduction():
    """Test UMAP dimensionality reduction."""
    pipeline = ClusteringPipeline(random_seed=42)
    
    # Create synthetic features
    n_samples = 100
    n_features = 64
    X = np.random.randn(n_samples, n_features)
    
    embeddings = pipeline.fit_transform(X)
    
    assert embeddings.shape == (n_samples, 2)  # Default 2D
    assert np.all(np.isfinite(embeddings))


def test_hdbscan_clustering():
    """Test HDBSCAN clustering."""
    pipeline = ClusteringPipeline(
        random_seed=42,
        hdbscan_min_cluster_size=5,
    )
    
    # Create synthetic clustered data
    n_samples = 50
    n_features = 32
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure
    X[:20] += 2
    X[20:40] -= 2
    
    labels = pipeline.cluster(X)
    
    assert len(labels) == n_samples
    assert labels.dtype in [np.int32, np.int64]
    assert labels.min() >= -1  # -1 is noise


def test_clustering_pipeline():
    """Test full clustering pipeline."""
    pipeline = ClusteringPipeline(random_seed=42)
    
    # Create synthetic features DataFrame
    n_samples = 50
    n_features = 32
    features = np.random.randn(n_samples, n_features)
    
    df = pd.DataFrame(
        {
            "segment_id": [f"seg_{i}" for i in range(n_samples)],
            **{f"feat_{i}": features[:, i] for i in range(n_features)},
        }
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        result_df = pipeline.run_pipeline(df, output_dir, n_exemplars=3)
        
        assert "umap_x" in result_df.columns
        assert "umap_y" in result_df.columns
        assert "cluster_id" in result_df.columns
        assert "cluster_confidence" in result_df.columns
        assert len(result_df) == n_samples
        
        # Check output files
        assert (output_dir / "clusters.parquet").exists()
        assert (output_dir / "exemplars.json").exists()
        assert (output_dir / "summary.json").exists()


def test_exemplar_selection():
    """Test exemplar selection."""
    pipeline = ClusteringPipeline(random_seed=42)
    
    # Create synthetic clustered data
    n_samples = 30
    n_features = 16
    X = np.random.randn(n_samples, n_features)
    
    # Create two clusters
    X[:15] += 2
    X[15:] -= 2
    
    labels = np.array([0] * 15 + [1] * 15)
    
    exemplars = pipeline.find_exemplars(X, labels, n_per_cluster=3, method="centroid")
    
    assert 0 in exemplars
    assert 1 in exemplars
    assert len(exemplars[0]) == 3
    assert len(exemplars[1]) == 3


def test_reproducibility():
    """Test clustering reproducibility with same seed."""
    # Create synthetic data
    n_samples = 40
    n_features = 32
    X = np.random.RandomState(42).randn(n_samples, n_features)
    
    pipeline1 = ClusteringPipeline(random_seed=42)
    pipeline2 = ClusteringPipeline(random_seed=42)
    
    labels1 = pipeline1.cluster(X)
    labels2 = pipeline2.cluster(X)
    
    # Should produce same results
    np.testing.assert_array_equal(labels1, labels2)


