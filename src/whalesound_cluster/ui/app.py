"""Streamlit app for exploring whale sound clusters."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import librosa
import soundfile as sf

from whalesound_cluster.utils.config import load_config
from whalesound_cluster.utils.logging import setup_logging

logger = setup_logging()


@st.cache_data
def load_clusters(cluster_path: Path) -> pd.DataFrame:
    """Load clustering results."""
    return pd.read_parquet(cluster_path)


@st.cache_data
def load_exemplars(exemplars_path: Path) -> dict:
    """Load exemplars."""
    with open(exemplars_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_audio_segment(segment_path: Path) -> tuple[np.ndarray, int]:
    """Load audio segment."""
    audio, sr = sf.read(str(segment_path))
    return audio, sr


def create_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """Create spectrogram for visualization."""
    stft = librosa.stft(audio, hop_length=512, n_fft=2048)
    magnitude = np.abs(stft)
    mel_spec = librosa.feature.melspectrogram(
        S=magnitude**2, sr=sr, n_mels=64, fmax=2000
    )
    logmel = librosa.power_to_db(mel_spec, ref=np.max)
    return logmel


def run_app(
    cluster_path: str | None = None,
    segments_dir: str | None = None,
    exemplars_path: str | None = None,
):
    """
    Run Streamlit app.
    
    Args:
        cluster_path: Path to clusters parquet file
        segments_dir: Directory containing segment audio files
        exemplars_path: Path to exemplars JSON
    """
    # Load defaults from config if not provided
    if cluster_path is None or segments_dir is None or exemplars_path is None:
        try:
            from whalesound_cluster.utils.config import load_config
            cfg = load_config()
            if cluster_path is None:
                cluster_path = str(Path(cfg["clustering"]["output_dir"]) / "clusters.parquet")
            if segments_dir is None:
                segments_dir = str(Path(cfg["segmentation"]["output_dir"]))
            if exemplars_path is None:
                exemplars_path = str(Path(cfg["clustering"]["output_dir"]) / "exemplars.json")
        except Exception:
            # Fallback defaults
            if cluster_path is None:
                cluster_path = "data/clusters/clusters.parquet"
            if segments_dir is None:
                segments_dir = "data/segments"
            if exemplars_path is None:
                exemplars_path = "data/clusters/exemplars.json"
    
    st.set_page_config(
        page_title="WhaleSound Clusters",
        page_icon="🐋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for museum-like design
    st.markdown(
        """
        <style>
        .main {
            background-color: #fafafa;
        }
        h1 {
            font-family: 'Georgia', serif;
            color: #2c3e50;
            font-weight: 300;
            letter-spacing: 1px;
        }
        h2 {
            font-family: 'Georgia', serif;
            color: #34495e;
            font-weight: 300;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 0.5rem 1.5rem;
            font-weight: 400;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Header
    st.title("🐋 WhaleSound Clusters")
    st.markdown(
        """
        <p style='font-size: 1.1em; color: #7f8c8d; font-style: italic;'>
        Exploring acoustic motifs in humpback whale songs from SanctSound
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    # Load data
    cluster_path = Path(cluster_path)
    segments_dir = Path(segments_dir)
    exemplars_path = Path(exemplars_path)
    
    if not cluster_path.exists():
        st.error(f"Clusters file not found: {cluster_path}")
        st.info("Please run the clustering pipeline first: `make cluster`")
        return
    
    try:
        df = load_clusters(cluster_path)
    except Exception as e:
        st.error(f"Failed to load clusters: {e}")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Cluster filter
    unique_clusters = sorted([c for c in df["cluster_id"].unique() if c >= 0])
    selected_clusters = st.sidebar.multiselect(
        "Clusters",
        unique_clusters,
        default=unique_clusters[:10] if len(unique_clusters) > 10 else unique_clusters,
    )
    
    # Station filter (if available)
    if "station" in df.columns:
        stations = df["station"].unique()
        selected_stations = st.sidebar.multiselect("Stations", stations, default=stations)
    else:
        selected_stations = None
    
    # Date filter (if available)
    if "timestamp" in df.columns:
        dates = pd.to_datetime(df["timestamp"]).dt.date
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(dates.min(), dates.max()),
            min_value=dates.min(),
            max_value=dates.max(),
        )
    else:
        date_range = None
    
    # Apply filters
    filtered_df = df.copy()
    if selected_clusters:
        filtered_df = filtered_df[filtered_df["cluster_id"].isin(selected_clusters + [-1])]
    if selected_stations is not None:
        filtered_df = filtered_df[filtered_df["station"].isin(selected_stations)]
    if date_range and len(date_range) == 2:
        if "timestamp" in filtered_df.columns:
            dates = pd.to_datetime(filtered_df["timestamp"]).dt.date
            filtered_df = filtered_df[
                (dates >= date_range[0]) & (dates <= date_range[1])
            ]
    
    # Limit display
    max_display = st.sidebar.slider("Max points to display", 100, 10000, 500)
    if len(filtered_df) > max_display:
        filtered_df = filtered_df.sample(n=max_display, random_state=42)
    
    # Main content
    tab1, tab2 = st.tabs(["Explorer", "Gallery"])
    
    with tab1:
        st.header("Cluster Explorer")
        
        # Convert cluster_id to string for discrete coloring
        plot_df = filtered_df.copy()
        plot_df["cluster_id_str"] = plot_df["cluster_id"].astype(str)
        
        # Create explicit color map for clusters (noise=-1 gets gray, clusters get distinct colors)
        unique_clusters = sorted(plot_df["cluster_id_str"].unique())
        color_map = {}
        cluster_colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
        
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == "-1":
                color_map[cluster_id] = "#888888"  # Gray for noise
            else:
                color_map[cluster_id] = cluster_colors[i % len(cluster_colors)]
        
        # Scatter plot with discrete colors
        fig = px.scatter(
            plot_df,
            x="umap_x",
            y="umap_y",
            color="cluster_id_str",
            hover_data=["segment_id", "cluster_confidence"],
            color_discrete_map=color_map,
            title="UMAP Embedding of Audio Segments",
            labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2", "cluster_id_str": "Cluster ID"},
        )
        
        fig.update_layout(
            height=600,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Arial", size=12),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Selected point info
        st.subheader("Segment Details")
        
        # Random segment button
        if st.button("🎲 Play Random Segment"):
            random_idx = np.random.randint(0, len(filtered_df))
            selected_segment = filtered_df.iloc[random_idx]
        else:
            # Default to first segment
            selected_segment = filtered_df.iloc[0] if len(filtered_df) > 0 else None
        
        if selected_segment is not None:
            segment_id = selected_segment["segment_id"]
            segment_path = segments_dir / f"{segment_id}.wav"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Segment ID:** {segment_id}")
                st.write(f"**Cluster:** {selected_segment['cluster_id']}")
                st.write(f"**Confidence:** {selected_segment['cluster_confidence']:.3f}")
                if "duration" in selected_segment:
                    st.write(f"**Duration:** {selected_segment['duration']:.2f}s")
            
            with col2:
                if segment_path.exists():
                    audio, sr = load_audio_segment(segment_path)
                    st.audio(audio, sample_rate=sr)
                    
                    # Spectrogram
                    spec = create_spectrogram(audio, sr)
                    fig_spec = go.Figure(
                        data=go.Heatmap(
                            z=spec,
                            colorscale="Viridis",
                            showscale=False,
                        )
                    )
                    fig_spec.update_layout(
                        height=200,
                        xaxis_title="Time",
                        yaxis_title="Frequency (mel)",
                        margin=dict(l=0, r=0, t=0, b=0),
                    )
                    st.plotly_chart(fig_spec, use_container_width=True)
                else:
                    st.warning(f"Audio file not found: {segment_path}")
    
    with tab2:
        st.header("Gallery")
        st.markdown(
            """
            <p style='color: #7f8c8d;'>
            Cluster exemplars: representative segments from each discovered motif
            </p>
            """,
            unsafe_allow_html=True,
        )
        
        if not exemplars_path.exists():
            st.warning("Exemplars file not found. Run clustering with exemplars enabled.")
        else:
            try:
                exemplars = load_exemplars(exemplars_path)
                
                # Display clusters
                for cluster_id in sorted(exemplars.keys()):
                    cluster_id_int = int(cluster_id)
                    cluster_df = df[df["cluster_id"] == cluster_id_int]
                    
                    if len(cluster_df) == 0:
                        continue
                    
                    st.subheader(f"Cluster {cluster_id_int} ({len(cluster_df)} segments)")
                    
                    # Exemplar segments
                    exemplar_indices = exemplars[cluster_id]
                    exemplar_segments = df.iloc[exemplar_indices]
                    
                    # Create grid of exemplars
                    cols = st.columns(min(5, len(exemplar_segments)))
                    
                    for idx, (col, (_, exemplar)) in enumerate(zip(cols, exemplar_segments.iterrows())):
                        with col:
                            segment_id = exemplar["segment_id"]
                            segment_path = segments_dir / f"{segment_id}.wav"
                            
                            if segment_path.exists():
                                audio, sr = load_audio_segment(segment_path)
                                st.audio(audio, sample_rate=sr)
                                st.caption(f"{segment_id}")
                            else:
                                st.caption(f"{segment_id} (audio not found)")
                    
                    st.markdown("---")
            
            except Exception as e:
                st.error(f"Failed to load exemplars: {e}")


# Streamlit will call run_app() directly when run via streamlit run

