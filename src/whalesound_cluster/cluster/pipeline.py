"""Clustering pipeline: UMAP + HDBSCAN."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hdbscan import HDBSCAN
from umap import UMAP

from whalesound_cluster.utils.logging import setup_logging

logger = setup_logging()


class ClusteringPipeline:
    """Pipeline for dimensionality reduction and clustering."""

    def __init__(
        self,
        random_seed: int = 42,
        umap_n_components: int = 2,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "euclidean",
        hdbscan_min_cluster_size: int = 10,
        hdbscan_min_samples: int = 5,
        hdbscan_cluster_selection_method: str = "eom",
        hdbscan_metric: str = "euclidean",
    ):
        """
        Initialize clustering pipeline.
        
        Args:
            random_seed: Random seed for reproducibility
            umap_n_components: UMAP output dimensions
            umap_n_neighbors: UMAP neighbors parameter
            umap_min_dist: UMAP minimum distance
            umap_metric: UMAP distance metric
            hdbscan_min_cluster_size: Minimum cluster size
            hdbscan_min_samples: Minimum samples in cluster
            hdbscan_cluster_selection_method: "eom" or "leaf"
            hdbscan_metric: HDBSCAN distance metric
        """
        self.random_seed = random_seed
        
        # UMAP
        self.umap = UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_seed,
        )
        
        # HDBSCAN
        self.hdbscan = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_method=hdbscan_cluster_selection_method,
            metric=hdbscan_metric,
        )

    def fit_transform(
        self,
        features: np.ndarray | pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Apply UMAP dimensionality reduction.
        
        Args:
            features: Feature array or DataFrame
            feature_cols: Column names if features is DataFrame
            
        Returns:
            UMAP embeddings (n_samples, n_components)
        """
        if isinstance(features, pd.DataFrame):
            if feature_cols is None:
                # Auto-detect feature columns
                feature_cols = [c for c in features.columns if c.startswith("feat_")]
            X = features[feature_cols].values
        else:
            X = features
        
        logger.info(f"Applying UMAP to {X.shape[0]} samples, {X.shape[1]} features")
        embeddings = self.umap.fit_transform(X)
        logger.info(f"UMAP embeddings shape: {embeddings.shape}")
        
        return embeddings

    def cluster(
        self,
        features: np.ndarray | pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Apply HDBSCAN clustering.
        
        Args:
            features: Feature array or DataFrame
            feature_cols: Column names if features is DataFrame
            
        Returns:
            Cluster labels (-1 for noise)
        """
        if isinstance(features, pd.DataFrame):
            if feature_cols is None:
                feature_cols = [c for c in features.columns if c.startswith("feat_")]
            X = features[feature_cols].values
        else:
            X = features
        
        logger.info(f"Clustering {X.shape[0]} samples")
        labels = self.hdbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
        
        return labels

    def compute_cluster_confidence(self, labels: np.ndarray) -> np.ndarray:
        """
        Compute cluster membership confidence (from HDBSCAN).
        
        Args:
            labels: Cluster labels
            
        Returns:
            Confidence scores
        """
        if hasattr(self.hdbscan, "probabilities_"):
            return self.hdbscan.probabilities_
        else:
            # Fallback: binary confidence
            return (labels >= 0).astype(float)

    def find_exemplars(
        self,
        features: np.ndarray | pd.DataFrame,
        labels: np.ndarray,
        n_per_cluster: int = 5,
        method: str = "centroid",
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[int, List[int]]:
        """
        Find exemplar segments for each cluster.
        
        Args:
            features: Feature array or DataFrame
            labels: Cluster labels
            n_per_cluster: Number of exemplars per cluster
            method: "centroid" or "diverse"
            feature_cols: Column names if features is DataFrame
            
        Returns:
            Dict mapping cluster_id to list of exemplar indices
        """
        if isinstance(features, pd.DataFrame):
            if feature_cols is None:
                feature_cols = [c for c in features.columns if c.startswith("feat_")]
            X = features[feature_cols].values
        else:
            X = features
        
        exemplars = {}
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # Skip noise
            
            cluster_mask = labels == cluster_id
            cluster_features = X[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if method == "centroid":
                # Find points closest to centroid
                centroid = np.mean(cluster_features, axis=0)
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                exemplar_indices = np.argsort(distances)[:n_per_cluster]
            elif method == "diverse":
                # Greedy selection for diversity
                exemplar_indices = self._greedy_diverse_selection(
                    cluster_features, n_per_cluster
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            exemplars[int(cluster_id)] = cluster_indices[exemplar_indices].tolist()
        
        return exemplars

    def _greedy_diverse_selection(
        self, features: np.ndarray, n: int
    ) -> np.ndarray:
        """Greedy selection of diverse points."""
        if len(features) <= n:
            return np.arange(len(features))
        
        # Start with point farthest from mean
        centroid = np.mean(features, axis=0)
        distances = np.linalg.norm(features - centroid, axis=1)
        selected = [np.argmax(distances)]
        
        # Greedily add points farthest from already selected
        for _ in range(n - 1):
            distances_to_selected = []
            for i, feat in enumerate(features):
                if i in selected:
                    continue
                min_dist = min(
                    np.linalg.norm(feat - features[j]) for j in selected
                )
                distances_to_selected.append((i, min_dist))
            
            if not distances_to_selected:
                break
            
            next_idx = max(distances_to_selected, key=lambda x: x[1])[0]
            selected.append(next_idx)
        
        return np.array(selected)

    def run_pipeline(
        self,
        features_df: pd.DataFrame,
        output_dir: Path,
        feature_cols: Optional[List[str]] = None,
        n_exemplars: int = 5,
    ) -> pd.DataFrame:
        """
        Run full clustering pipeline.
        
        Args:
            features_df: DataFrame with features and metadata
            output_dir: Output directory
            feature_cols: Feature column names
            n_exemplars: Number of exemplars per cluster
            
        Returns:
            DataFrame with embeddings and cluster labels
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract feature columns
        if feature_cols is None:
            feature_cols = [c for c in features_df.columns if c.startswith("feat_")]
        
        # UMAP
        embeddings = self.fit_transform(features_df, feature_cols)
        
        # Add UMAP columns
        if embeddings.shape[1] == 2:
            features_df["umap_x"] = embeddings[:, 0]
            features_df["umap_y"] = embeddings[:, 1]
        else:
            for i in range(embeddings.shape[1]):
                features_df[f"umap_{i}"] = embeddings[:, i]
        
        # HDBSCAN - cluster on UMAP embeddings so clusters align with visualization
        logger.info(f"Clustering on UMAP embeddings ({embeddings.shape[1]}D)")
        labels = self.hdbscan.fit_predict(embeddings)
        features_df["cluster_id"] = labels
        
        # Confidence
        confidence = self.compute_cluster_confidence(labels)
        features_df["cluster_confidence"] = confidence
        
        # Exemplars
        exemplars = self.find_exemplars(
            features_df, labels, n_per_cluster=n_exemplars
        )
        
        # Save results
        output_path = output_dir / "clusters.parquet"
        features_df.to_parquet(output_path, index=False)
        logger.info(f"Saved clustering results to {output_path}")
        
        # Save exemplars
        exemplars_path = output_dir / "exemplars.json"
        with open(exemplars_path, "w") as f:
            json.dump(exemplars, f, indent=2)
        logger.info(f"Saved exemplars to {exemplars_path}")
        
        # Save summary
        summary = {
            "n_samples": len(features_df),
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_noise": int(np.sum(labels == -1)),
            "cluster_sizes": {
                int(k): int(v)
                for k, v in zip(*np.unique(labels, return_counts=True))
                if k != -1
            },
        }
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")
        
        return features_df

