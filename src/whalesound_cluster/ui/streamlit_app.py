"""Streamlit entry point for UI."""

import sys
from pathlib import Path

from whalesound_cluster.ui.app import run_app
from whalesound_cluster.utils.config import load_config

# Streamlit will execute this file, so we set up the app here
if __name__ == "__main__":
    # Try to load config for default paths
    try:
        cfg = load_config()
        cluster_path = str(Path(cfg["clustering"]["output_dir"]) / "clusters.parquet")
        segments_dir = str(Path(cfg["segmentation"]["output_dir"]))
        exemplars_path = str(Path(cfg["clustering"]["output_dir"]) / "exemplars.json")
    except Exception:
        # Fallback to defaults
        cluster_path = "data/clusters/clusters.parquet"
        segments_dir = "data/segments"
        exemplars_path = "data/clusters/exemplars.json"
    
    # Get paths from command line args if provided
    if len(sys.argv) > 1:
        cluster_path = sys.argv[1]
        if len(sys.argv) > 2:
            segments_dir = sys.argv[2]
        if len(sys.argv) > 3:
            exemplars_path = sys.argv[3]
    
    run_app(cluster_path, segments_dir, exemplars_path)

