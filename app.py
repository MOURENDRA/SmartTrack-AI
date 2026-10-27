"""
SmartTrack AI — Occlusion-Aware Multi-Object Tracking
Streamlit Web Application

Launch: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import json
import time
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_config, generate_color, Timer
from src.video_processor import VideoProcessor
from src.evaluation import (
    get_dataset_status, evaluate_mot_sequence, run_comparison,
    MOT17_SEQUENCES, DENSITY_GROUPS
)

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="SmartTrack AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark professional theme */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }
    
    .stMetric {
        background-color: #1a1a2e;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #16213e;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border: 1px solid #0f3460;
    }
    
    .header-title {
        color: #e94560;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    
    .header-subtitle {
        color: #a0a0b0;
        font-size: 1rem;
        margin-top: 0;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #0f0f23;
    }
    
    .status-active { color: #00d26a; font-weight: bold; }
    .status-idle { color: #a0a0b0; }
    .status-error { color: #e94560; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ───────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'config': load_config(),
        'processor': None,
        'processing': False,
        'processed': False,
        'results': None,
        'frame_metrics': [],
        'output_path': None,
        'current_frame_idx': 0,
        'tracker_mode': 'enhanced',
        'detection_conf': 0.5,
        'association_gate': 0.7,
        'max_lost_age': 60,
        'motion_weight': 0.6,
        'appearance_weight': 0.4,
        'eval_results': None,
        'comparison_results': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─── Sidebar ────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with configuration controls."""
    with st.sidebar:
        st.markdown("## 🎯 SmartTrack AI")
        st.markdown("*Identity-Preserving Multi-Object Tracking*")
        st.divider()
        
        # ── Tracker Mode ──
        st.markdown("### Tracker Configuration")
        
        mode = st.radio(
            "Tracking Method",
            options=['enhanced', 'baseline'],
            index=0 if st.session_state.tracker_mode == 'enhanced' else 1,
            help=(
                "**Baseline**: IoU-only motion matching (standard SORT-like).\n\n"
                "**Enhanced**: Combined motion + appearance + confidence-aware "
                "association with track memory and re-association."
            )
        )
        st.session_state.tracker_mode = mode
        
        if mode == 'enhanced':
            st.success("✨ Enhanced tracker: motion + appearance + confidence gating")
        else:
            st.info("📐 Baseline tracker: IoU motion-only matching")
        
        st.divider()
        
        # ── Detection Settings ──
        st.markdown("### Detection Settings")
        
        st.session_state.detection_conf = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1, max_value=0.9, value=st.session_state.detection_conf,
            step=0.05,
            help="Minimum confidence to accept a person detection. "
                 "Higher = fewer false detections but may miss people."
        )
        
        st.divider()
        
        # ── Association Settings ──
        st.markdown("### Association Settings")
        
        st.session_state.association_gate = st.slider(
            "Association Gate Threshold",
            min_value=0.3, max_value=1.0, value=st.session_state.association_gate,
            step=0.05,
            help="Maximum cost to accept a detection-to-track match. "
                 "Lower = stricter matching, fewer false associations."
        )
        
        if mode == 'enhanced':
            st.session_state.motion_weight = st.slider(
                "Motion Weight (λ_motion)",
                min_value=0.0, max_value=1.0, value=st.session_state.motion_weight,
                step=0.1,
                help="Weight of motion (IoU) cost in combined association."
            )
            st.session_state.appearance_weight = round(
                1.0 - st.session_state.motion_weight, 1
            )
            st.caption(f"Appearance Weight: {st.session_state.appearance_weight}")
        
        st.divider()
        
        # ── Track Memory ──
        st.markdown("### Track Memory")
        
        st.session_state.max_lost_age = st.slider(
            "Lost Track Retention (frames)",
            min_value=10, max_value=150, value=st.session_state.max_lost_age,
            step=5,
            help="How many frames to keep a lost track before removing it. "
                 "Higher = better re-association but more memory usage."
        )
        
        st.divider()
        
        # ── Controls ──
        if st.button("🔄 Reset Tracker", use_container_width=True):
            st.session_state.processor = None
            st.session_state.processing = False
            st.session_state.processed = False
            st.session_state.results = None
            st.session_state.frame_metrics = []
            st.session_state.output_path = None
            st.rerun()
        
        st.divider()
        st.caption("SmartTrack AI v1.0 | PS-2 Project")


def build_config() -> dict:
    """Build config dict from session state."""
    config = load_config()
    config['detector']['confidence_threshold'] = st.session_state.detection_conf
    config['tracker']['mode'] = st.session_state.tracker_mode
    config['association']['gate_threshold'] = st.session_state.association_gate
    config['association']['lambda_iou'] = st.session_state.motion_weight
    config['association']['lambda_appearance'] = st.session_state.appearance_weight
    config['track_memory']['max_lost_age'] = st.session_state.max_lost_age
    return config


# ─── Main Content ───────────────────────────────────────────
def render_header():
    """Render the main header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="header-title">🎯 SmartTrack AI</p>', 
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="header-subtitle">'
            'Occlusion-Aware Multi-Object Tracking with Identity Preservation'
            '</p>',
            unsafe_allow_html=True
        )
    with col2:
        mode = st.session_state.tracker_mode
        status = "Processing..." if st.session_state.processing else (
            "Complete" if st.session_state.processed else "Ready"
        )
        st.metric("Mode", mode.capitalize())
        st.metric("Status", status)


def render_video_processing():
    """Render the video upload and processing section."""
    st.markdown("## 📹 Video Processing")
    
    tab1, tab2 = st.tabs(["Upload Video", "Webcam (Experimental)"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            # Show video info
            cap = cv2.VideoCapture(input_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Resolution", f"{width}×{height}")
                col2.metric("FPS", f"{fps:.1f}")
                col3.metric("Frames", frame_count)
                col4.metric("Duration", f"{duration:.1f}s")
                
                # Process button
                if st.button("🚀 Start Tracking", type="primary", 
                             use_container_width=True,
                             disabled=st.session_state.processing):
                    process_video(input_path)
            else:
                st.error("❌ Could not open video file. Check the format and codec.")
                cap.release()
    
    with tab2:
        st.info(
            "📷 Webcam support is experimental. It requires a connected camera "
            "and may not work in all browser environments with Streamlit."
        )
        if st.button("Start Webcam Capture", disabled=True):
            st.warning("Webcam processing requires local display access. "
                      "Use the CLI script for webcam: "
                      "`python scripts/run_tracking.py --input 0`")


def process_video(input_path: str):
    """Process the uploaded video through the tracking pipeline."""
    st.session_state.processing = True
    
    config = build_config()
    
    # Initialize processor
    processor = VideoProcessor(config)
    st.session_state.processor = processor
    
    if not processor.detector.is_ready:
        st.warning(
            "⚠️ YOLO model not loaded. Install `ultralytics` and ensure "
            "model weights are available. Processing will produce empty detections."
        )
    
    # Prepare output
    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = config['tracker']['mode']
    output_filename = f"tracked_{mode}_{timestamp}.mp4"
    output_path = os.path.join('outputs', output_filename)
    
    # Progress bar
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    metrics_placeholder = st.empty()
    frame_display = st.empty()
    
    frame_metrics_list = []
    
    def progress_callback(frame_idx, total_frames, tracks, frame_time):
        progress = frame_idx / total_frames
        fps = 1.0 / frame_time if frame_time > 0 else 0
        n_active = len([t for t in tracks if hasattr(t, 'state')]) if tracks else 0
        
        progress_bar.progress(progress, 
                             text=f"Frame {frame_idx}/{total_frames} | "
                                  f"FPS: {fps:.1f} | Tracks: {n_active}")
        
        # Collect frame metrics
        track_data = []
        if tracks:
            for t in tracks:
                track_data.append({
                    'track_id': t.track_id,
                    'bbox': t.bbox.tolist() if hasattr(t.bbox, 'tolist') else list(t.bbox),
                    'state': t.state.name if hasattr(t.state, 'name') else str(t.state),
                    'confidence': getattr(t, 'confidence', 0),
                })
        
        frame_metrics_list.append({
            'frame_idx': frame_idx,
            'n_detections': n_active,
            'n_active_tracks': n_active,
            'frame_time': frame_time,
            'tracks': track_data,
        })
    
    try:
        results = processor.process_video(
            input_path, output_path, 
            progress_callback=progress_callback
        )
        
        st.session_state.results = results
        st.session_state.frame_metrics = frame_metrics_list
        st.session_state.output_path = output_path
        st.session_state.processed = True
        st.session_state.processing = False
        
        progress_bar.progress(1.0, text="✅ Processing complete!")
        st.success(f"✅ Video processed successfully! Output: `{output_path}`")
        
    except Exception as e:
        st.session_state.processing = False
        st.error(f"❌ Processing failed: {str(e)}")
        import traceback
        st.expander("Error Details").code(traceback.format_exc())


def render_results():
    """Render processing results and metrics."""
    if not st.session_state.processed or st.session_state.results is None:
        return
    
    results = st.session_state.results
    
    st.markdown("## 📊 Results")
    
    # ── Summary Metrics ──
    col1, col2, col3, col4, col5 = st.columns(5)
    
    stats = results.get('tracker_stats', {})
    col1.metric("Frames Processed", results.get('processed_frames', 0))
    col2.metric("Processing FPS", f"{results.get('avg_fps', 0):.1f}")
    col3.metric("Active Tracks", stats.get('confirmed', 0))
    col4.metric("Lost Tracks", stats.get('lost', 0))
    col5.metric("Total IDs Created", stats.get('total_tracks_created', 
                stats.get('confirmed', 0) + stats.get('lost', 0) + stats.get('removed', 0)))
    
    # ── Output Video ──
    st.markdown("### 🎬 Processed Video")
    
    output_path = st.session_state.output_path
    if output_path and os.path.exists(output_path):
        st.video(output_path)
        
        # Download button
        with open(output_path, 'rb') as f:
            st.download_button(
                "⬇️ Download Processed Video",
                data=f.read(),
                file_name=os.path.basename(output_path),
                mime="video/mp4"
            )
    else:
        st.warning("Output video file not found. Check the outputs/ directory.")
    
    # ── Charts ──
    st.markdown("### 📈 Analytics")
    
    frame_metrics = st.session_state.frame_metrics
    if frame_metrics:
        render_analytics_charts(frame_metrics)
    
    # ── Export ──
    render_export_section(results, frame_metrics)


def render_analytics_charts(frame_metrics: list):
    """Render analytics charts from frame metrics."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    frames = [m['frame_idx'] for m in frame_metrics]
    n_tracks = [m['n_active_tracks'] for m in frame_metrics]
    frame_times = [m['frame_time'] * 1000 for m in frame_metrics]  # ms
    fps_values = [1000.0 / t if t > 0 else 0 for t in frame_times]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames, y=n_tracks,
            mode='lines', name='Active Tracks',
            line=dict(color='#e94560', width=2)
        ))
        fig.update_layout(
            title="Active Tracks Over Time",
            xaxis_title="Frame",
            yaxis_title="Count",
            template="plotly_dark",
            height=350,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames, y=fps_values,
            mode='lines', name='FPS',
            line=dict(color='#00d26a', width=2)
        ))
        fig.update_layout(
            title="Processing Speed (FPS)",
            xaxis_title="Frame",
            yaxis_title="FPS",
            template="plotly_dark",
            height=350,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Frame time histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=frame_times,
        nbinsx=50,
        marker_color='#533483',
        name='Frame Time'
    ))
    fig.update_layout(
        title="Frame Processing Time Distribution",
        xaxis_title="Processing Time (ms)",
        yaxis_title="Count",
        template="plotly_dark",
        height=300,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_export_section(results: dict, frame_metrics: list):
    """Render export buttons and options."""
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        if frame_metrics:
            csv_rows = []
            for fm in frame_metrics:
                for track in fm.get('tracks', []):
                    bbox = track.get('bbox', [0, 0, 0, 0])
                    csv_rows.append({
                        'frame_id': fm['frame_idx'],
                        'track_id': track['track_id'],
                        'x1': bbox[0] if len(bbox) > 0 else 0,
                        'y1': bbox[1] if len(bbox) > 1 else 0,
                        'x2': bbox[2] if len(bbox) > 2 else 0,
                        'y2': bbox[3] if len(bbox) > 3 else 0,
                        'confidence': track.get('confidence', 0),
                        'state': track.get('state', ''),
                    })
            
            if csv_rows:
                df = pd.DataFrame(csv_rows)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_data,
                    file_name="tracking_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("No tracks to export.")
    
    with col2:
        # MOT Format Export
        if frame_metrics:
            mot_lines = []
            for fm in frame_metrics:
                for track in fm.get('tracks', []):
                    bbox = track.get('bbox', [0, 0, 0, 0])
                    x1 = bbox[0] if len(bbox) > 0 else 0
                    y1 = bbox[1] if len(bbox) > 1 else 0
                    x2 = bbox[2] if len(bbox) > 2 else 0
                    y2 = bbox[3] if len(bbox) > 3 else 0
                    w = x2 - x1
                    h = y2 - y1
                    conf = track.get('confidence', 1.0)
                    mot_lines.append(
                        f"{fm['frame_idx']},{track['track_id']},"
                        f"{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                        f"{conf:.4f},-1,-1,-1"
                    )
            
            if mot_lines:
                st.download_button(
                    "⬇️ Download MOT Format",
                    data="\n".join(mot_lines),
                    file_name="tracking_results.txt",
                    mime="text/plain"
                )
    
    with col3:
        # JSON Summary
        summary = {
            'project': 'SmartTrack AI',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'mode': st.session_state.tracker_mode,
                'detection_confidence': st.session_state.detection_conf,
                'association_gate': st.session_state.association_gate,
                'max_lost_age': st.session_state.max_lost_age,
                'motion_weight': st.session_state.motion_weight,
                'appearance_weight': st.session_state.appearance_weight,
            },
            'results': {
                'total_frames': results.get('total_frames', 0),
                'processed_frames': results.get('processed_frames', 0),
                'avg_fps': results.get('avg_fps', 0),
                'tracker_stats': results.get('tracker_stats', {}),
            }
        }
        st.download_button(
            "⬇️ Download JSON Summary",
            data=json.dumps(summary, indent=2),
            file_name="tracking_summary.json",
            mime="application/json"
        )


def render_evaluation():
    """Render the evaluation page."""
    st.markdown("## 🔬 Benchmark Evaluation")
    
    st.info(
        "📂 **Dataset Setup**: Download MOT17 from "
        "[motchallenge.net](https://motchallenge.net/data/MOT17/) "
        "and extract to `data/datasets/MOT17/`"
    )
    
    dataset_root = st.text_input(
        "Dataset Root Path",
        value="data/datasets/MOT17",
        help="Path to the MOT dataset directory"
    )
    
    # Check dataset status
    if st.button("🔍 Check Dataset Status"):
        status = get_dataset_status(dataset_root)
        
        if not status['exists']:
            st.error(f"Dataset directory not found: `{dataset_root}`")
            st.markdown(
                "**To set up MOT17:**\n"
                "1. Download from https://motchallenge.net/data/MOT17/\n"
                "2. Extract the archive\n"
                "3. Place `train/` folder under `data/datasets/MOT17/`\n\n"
                "**Expected structure:**\n"
                "```\n"
                "data/datasets/MOT17/\n"
                "  train/\n"
                "    MOT17-02-SDP/\n"
                "      img1/\n"
                "      gt/gt.txt\n"
                "      seqinfo.ini\n"
                "```"
            )
        else:
            n17 = status['mot17_available']
            st.success(f"Found {n17} MOT17 sequence(s)")
            
            for seq, info in status['mot17'].items():
                if info['available']:
                    gt_text = "✓ GT" if info['has_ground_truth'] else "✗ No GT"
                    st.markdown(f"  ✅ {seq} ({gt_text})")
    
    st.divider()
    
    # Evaluation controls
    col1, col2 = st.columns(2)
    
    with col1:
        eval_seq = st.selectbox(
            "Select Sequence",
            options=MOT17_SEQUENCES,
            help="Choose a MOT17 sequence to evaluate"
        )
    
    with col2:
        eval_mode = st.radio(
            "Evaluation Mode",
            options=['Single Sequence', 'Baseline vs Enhanced Comparison'],
            horizontal=True
        )
    
    if eval_mode == 'Single Sequence':
        tracker_mode = st.radio("Tracker", ['enhanced', 'baseline'], horizontal=True)
        
        if st.button("▶️ Run Evaluation", type="primary"):
            config = build_config()
            config['tracker']['mode'] = tracker_mode
            
            progress_bar = st.progress(0, text="Starting evaluation...")
            
            def progress_cb(idx, total):
                progress_bar.progress(
                    idx / total, 
                    text=f"Processing frame {idx}/{total}..."
                )
            
            with st.spinner("Running tracker on sequence..."):
                result = evaluate_mot_sequence(
                    dataset_root, eval_seq, config, progress_cb
                )
            
            if result is None:
                st.error(f"Sequence {eval_seq} not found in {dataset_root}")
            else:
                progress_bar.progress(1.0, text="✅ Evaluation complete!")
                st.session_state.eval_results = result
                
                # Display results
                display_eval_results(result)
    
    else:
        if st.button("▶️ Run Comparison", type="primary"):
            config = build_config()
            
            progress_bar = st.progress(0, text="Starting comparison...")
            status_text = st.empty()
            
            def progress_cb(seq_name, mode, idx, total):
                progress_bar.progress(
                    idx / total,
                    text=f"[{mode}] {seq_name}: Frame {idx}/{total}"
                )
            
            with st.spinner("Running baseline and enhanced trackers..."):
                comp_results = run_comparison(
                    dataset_root, [eval_seq], config, progress_cb
                )
            
            st.session_state.comparison_results = comp_results
            progress_bar.progress(1.0, text="✅ Comparison complete!")
            
            display_comparison_results(comp_results)
    
    # Show previous results
    if st.session_state.eval_results and eval_mode == 'Single Sequence':
        st.divider()
        st.markdown("### Previous Evaluation Results")
        display_eval_results(st.session_state.eval_results)
    
    if st.session_state.comparison_results and eval_mode != 'Single Sequence':
        st.divider()
        st.markdown("### Previous Comparison Results")
        display_comparison_results(st.session_state.comparison_results)


def display_eval_results(result: dict):
    """Display evaluation results."""
    rt = result.get('runtime', {})
    metrics = result.get('metrics')
    source = result.get('metrics_source', 'N/A')
    
    st.markdown(f"**Sequence**: {result.get('sequence', 'N/A')}")
    st.markdown(f"**Metrics source**: {source}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames", rt.get('processed_frames', 0))
    col2.metric("Processing FPS", f"{rt.get('processing_fps', 0):.1f}")
    col3.metric("Total Time", f"{rt.get('total_time_s', 0):.1f}s")
    
    if metrics:
        st.markdown("#### Tracking Metrics")
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        
        mota = metrics.get('MOTA')
        idf1 = metrics.get('IDF1')
        hota = metrics.get('HOTA')
        idsw = metrics.get('num_id_switches')
        
        mcol1.metric("MOTA", f"{mota}%" if mota is not None else "N/A")
        mcol2.metric("IDF1", f"{idf1}%" if idf1 is not None else "N/A")
        mcol3.metric("HOTA", f"{hota}%" if hota is not None else "N/A")
        mcol4.metric("ID Switches", idsw if idsw is not None else "N/A")
        
        col1, col2 = st.columns(2)
        col1.metric("Precision", f"{metrics.get('precision', 'N/A')}%")
        col2.metric("Recall", f"{metrics.get('recall', 'N/A')}%")
        
        col1.metric("False Positives", metrics.get('num_false_positives', 'N/A'))
        col2.metric("False Negatives", metrics.get('num_false_negatives', 'N/A'))
    else:
        st.warning(
            "No ground truth available for this sequence. "
            "Download ground truth annotations to enable metric computation."
        )


def display_comparison_results(results: dict):
    """Display baseline vs enhanced comparison."""
    comparison = results.get('comparison', {})
    
    if not comparison:
        st.warning("No comparison data available. Both trackers may have failed.")
        return
    
    for seq_name, comp in comparison.items():
        st.markdown(f"#### {seq_name}")
        
        if not comp:
            st.info("No metrics available for comparison.")
            continue
        
        # Create comparison table
        rows = []
        for metric, info in comp.items():
            better_icon = "✅" if info['better'] == 'enhanced' else (
                "⬅️" if info['better'] == 'baseline' else "🔄"
            )
            rows.append({
                'Metric': metric,
                'Baseline': info['baseline'],
                'Enhanced': info['enhanced'],
                'Difference': f"{info['difference']:+.2f}",
                'Better': f"{better_icon} {info['better']}",
            })
        
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Bar chart comparison
        import plotly.graph_objects as go
        
        metrics_to_plot = [m for m in ['MOTA', 'IDF1', 'HOTA'] if m in comp]
        if metrics_to_plot:
            fig = go.Figure(data=[
                go.Bar(
                    name='Baseline',
                    x=metrics_to_plot,
                    y=[comp[m]['baseline'] for m in metrics_to_plot],
                    marker_color='#636efa'
                ),
                go.Bar(
                    name='Enhanced',
                    x=metrics_to_plot,
                    y=[comp[m]['enhanced'] for m in metrics_to_plot],
                    marker_color='#e94560'
                )
            ])
            fig.update_layout(
                title="Baseline vs Enhanced Comparison",
                yaxis_title="Score (%)",
                barmode='group',
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_help():
    """Render the help and information page."""
    st.markdown("## ℹ️ Help & Information")
    
    with st.expander("🔧 Tracking Pipeline", expanded=True):
        st.markdown("""
        **SmartTrack AI** processes video frames one at a time (online/causal tracking):
        
        1. **Detection**: YOLOv8 detects people in each frame → bounding boxes
        2. **Prediction**: Kalman filter predicts where each tracked person should be
        3. **Feature Extraction**: Visual appearance features are extracted from each detection
        4. **Data Association**: Hungarian algorithm matches detections to existing tracks using:
           - **Motion cost**: How far the detection is from the predicted position (IoU)
           - **Appearance cost**: How similar the person looks to the tracked appearance
        5. **Track Update**: Matched tracks are updated; unmatched detections create new tracks
        6. **Track Memory**: Lost tracks are kept for a configurable time for re-association
        """)
    
    with st.expander("🔄 What is an ID Switch?"):
        st.markdown("""
        An **ID switch** happens when the tracker assigns person A's identity to person B 
        (or vice versa). This often occurs during **occlusion** — when two people walk 
        past each other and overlap in the camera view.
        
        **Example**: Person #1 and Person #2 cross paths. While they overlap, the tracker 
        loses track of who is who. After they separate, Person #1 might be labeled as #2.
        
        SmartTrack AI addresses this with:
        - **Appearance features**: Remembering what each person looks like
        - **Track memory**: Keeping lost tracks alive for re-association
        - **Confidence gating**: Refusing uncertain matches instead of guessing
        """)
    
    with st.expander("📐 Algorithms Used"):
        st.markdown("""
        - **YOLOv8**: Real-time object detection neural network (person class)
        - **Kalman Filter**: Predicts object motion using position and velocity state
        - **Hungarian Algorithm**: Optimal assignment between detections and tracks
        - **Cosine Distance**: Measures appearance feature similarity
        - **IoU (Intersection over Union)**: Measures spatial overlap between boxes
        """)
    
    with st.expander("📊 Metrics Explained"):
        st.markdown("""
        - **MOTA** (Multi-Object Tracking Accuracy): Overall tracking quality. 
          Accounts for false positives, misses, and ID switches. Higher is better.
        - **IDF1** (ID F1 Score): How consistently correct identities are maintained. 
          Higher is better.
        - **HOTA** (Higher Order Tracking Accuracy): Balanced metric combining 
          detection and association quality. Higher is better.
        - **ID Switches**: Number of times a tracked identity incorrectly changes. 
          Lower is better.
        - **FPS**: Frames processed per second. Indicates processing speed.
        """)
    
    with st.expander("🛠️ Setup Instructions"):
        st.code("""
# 1. Create virtual environment
python -m venv venv

# 2. Activate it (Windows PowerShell)
.\\venv\\Scripts\\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py

# 5. Run tests
pytest tests/ -v

# 6. For benchmark evaluation, download MOT17:
#    https://motchallenge.net/data/MOT17/
#    Extract to data/datasets/MOT17/
        """, language="bash")
    
    with st.expander("⚠️ Troubleshooting"):
        st.markdown("""
        **"ultralytics not installed"**: Run `pip install ultralytics`
        
        **"Could not load model"**: Ensure internet access for first-time model download, 
        or place `yolov8n.pt` in the project directory.
        
        **"Video codec not supported"**: Install ffmpeg or use MP4 (H.264) format.
        
        **Slow processing**: 
        - Use a smaller model (yolov8n is already the smallest)
        - Reduce input resolution in config
        - Use GPU if available (`pip install torch` with CUDA)
        
        **Out of memory**: Process shorter videos or reduce resolution.
        
        **Streamlit errors**: Try `pip install --upgrade streamlit`
        """)


# ─── Main App ───────────────────────────────────────────────
def main():
    """Main application entry point."""
    render_sidebar()
    render_header()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎬 Process Video", "📊 Results & Analytics", 
        "🔬 Evaluation", "ℹ️ Help"
    ])
    
    with tab1:
        render_video_processing()
    
    with tab2:
        render_results()
        if not st.session_state.processed:
            st.info("📹 Upload and process a video to see results here.")
    
    with tab3:
        render_evaluation()
    
    with tab4:
        render_help()


if __name__ == "__main__":
    main()
