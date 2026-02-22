
import json
import os
import tempfile
import time
import math
import uuid
import threading
import concurrent.futures
from pathlib import Path
from typing import Optional

# Load environment variables
from dotenv import load_dotenv

import av
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from narrator.cactus_vl import CactusVL

APP_TITLE = "🌵 Cactus On-Device Video Narrator"
DEFAULT_MODEL_DIR = "weights/lfm2.5-vl-1.6b"  # adjust to your local weights folder name

# Removing the global TMP_DIR completely. We will use tempfile.

@st.cache_resource(show_spinner="Loading VLM to Unified Memory...")
def get_vlm(model_dir: str) -> CactusVL:
    """Cache the model in memory so it doesn't OOM on page reload/multiple tabs."""
    vl = CactusVL(model_dir)
    vl.load()
    return vl

def _normalize_weights_dir(model_dir: str) -> str:
    return str(Path(model_dir).expanduser())


def _resize_for_vlm(rgb: np.ndarray, target_short_side: int = 256) -> Image.Image:
    '''
    The Local Apple Neural Engine model operates on fixed shapes (256 patches). 
    We must center-crop the frame into a strict square to avoid NPU crashes.
    '''
    img = Image.fromarray(rgb)
    w, h = img.size
    short = min(w, h)
    
    # Scale first
    scale = target_short_side / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h))
    
    # Then Center Crop to perfectly square (target_short_side x target_short_side)
    left = (new_w - target_short_side) / 2
    top = (new_h - target_short_side) / 2
    right = (new_w + target_short_side) / 2
    bottom = (new_h + target_short_side) / 2
    
    return img.crop((left, top, right, bottom))


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_rgb: Optional[np.ndarray] = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        self.latest_rgb = bgr[:, :, ::-1]
        return frame


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Live webcam → sampled frames → on-device VLM → short narration.")

with st.sidebar:
    st.header("Settings")
    model_dir = _normalize_weights_dir(st.text_input("Local model weights directory", value=DEFAULT_MODEL_DIR))

    sample_every = st.slider("Analyze every N seconds", 0.3, 5.0, 5.0, 0.1)
    max_tokens = st.slider("Max tokens", 32, 256, 96, 16)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    short_side = st.slider("Resize short side (px)", 128, 512, 256, 32)
    
    st.divider()
    st.subheader("Input Source")
    input_source = st.radio("Select Video Source", ["Live Webcam", "Video File"])
    
    uploaded_video = None
    if input_source == "Video File":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "mkv"])
    
    st.divider()
    st.subheader("Hybrid Cloud Fallback")
    st.write("If the local model drops below this confidence, Vertex AI takes over.")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.70, 0.01)
    cooldown_seconds = st.slider("Cloud API Cooldown (s)", 1, 60, 10, 1)

    st.divider()
    st.subheader("Grounding DINO Settings")
    st.write("Controls the noun extraction length for the final transcript summary.")
    dino_max_tokens = st.slider("DINO Max Tokens", 128, 2048, 1024, 128)

    prompt = st.text_area(
        "Prompt",
        value="Describe what is happening in this scene in 1-2 sentences. Be concrete and avoid guessing.",
        height=90,
    )

    st.divider()
    st.write("Tip: keep resize ~256px and max_tokens <= 128 for low latency.")

col1, col2 = st.columns([1, 1])

process_btn = False

with col1:
    st.subheader("Camera / Video")
    ctx = None
    if input_source == "Live Webcam":
        ctx = webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        if uploaded_video:
            # Provide a quick way to play it manually in the UI
            st.video(uploaded_video)
            process_btn = st.button("Process Video 🚀", type="primary", use_container_width=True)
            stop_placeholder = st.empty()
        else:
            st.info("Please upload a video file in the sidebar to begin.")

with col2:
    st.subheader("Narration")
    frame_box = st.empty()
    narration_box = st.empty()
    metrics_box = st.empty()

st.divider()
st.subheader("Real-Time Transcript")
transcript_container = st.container(height=300)
transcript_box = transcript_container.empty()

# Load the model via the cached function instead of session state
vl = get_vlm(model_dir)

transcript_history = []

# helper for inference and UI updates
def _run_inference(frame_rgb: np.ndarray, video_timestamp: Optional[float] = None, transcript_data_list: Optional[list] = None):
    img = _resize_for_vlm(frame_rgb, target_short_side=short_side)
    frame_box.image(img, caption="Analyzing this frame...")

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        img.save(tmp.name, quality=85)
        
        out = vl.describe_image(
            image_path=tmp.name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            cooldown_seconds=cooldown_seconds,
        )
    dt_ms = (time.time() - t0) * 1000.0

    if out.get("success") is True:
        text = (out.get("response") or "").strip()
        
        if out.get("cloud_handoff") is True:
            prefix = "☁️ **[Gemini Fallback]**"
            color = "blue"
        else:
            prefix = "📱 **[On-Device]**"
            color = "green"
            
        narration_box.markdown(f":{color}[{prefix}] {text or '(empty response)'}")
        
        if video_timestamp is not None:
            # Format the 0-indexed seconds from the video to a cleanly readable format like `00:15`
            mins = int(video_timestamp // 60)
            secs = int(video_timestamp % 60)
            timestamp = f"{mins:02d}:{secs:02d}"
        else:
            timestamp = time.strftime("%H:%M:%S")
            
        transcript_history.insert(0, f"**{timestamp}** | :{color}[{prefix}] {text or '(empty response)'}")
        if len(transcript_history) > 50:
            transcript_history.pop()
        
        transcript_box.markdown("  \n".join(transcript_history))
        
        # If a list was provided, append the output payload in JSON-RPC format
        if transcript_data_list is not None:
            transcript_data_list.append({
                "jsonrpc": "2.0",
                "method": "notify_transcript",
                "params": {
                    "timestamp": timestamp,
                    "text": text,
                    "source": "Gemini Fallback" if out.get("cloud_handoff") else "On-Device",
                    "confidence": out.get("confidence", 1.0)
                }
            })
    else:
        narration_box.error(out.get("error") or "Inference failed")
        if "raw" in out:
            st.code(str(out["raw"])[:1000])

    metrics_box.json({
        "total_time_ms": out.get("total_time_ms", dt_ms),
        "time_to_first_token_ms": out.get("time_to_first_token_ms"),
        "prefill_tps": out.get("prefill_tps"),
        "decode_tps": out.get("decode_tps"),
        "ram_usage_mb": out.get("ram_usage_mb"),
        "confidence": out.get("confidence"),
        "cloud_handoff": out.get("cloud_handoff"),
    })

# --- EXECUTION LOOP ---
if input_source == "Live Webcam":
    last_t = 0.0
    
    # We need to maintain this list across Streamlit reruns so we don't overwrite it every frame.
    if "webcam_transcript" not in st.session_state:
        st.session_state["webcam_transcript"] = []
        
    session_id = st.session_state.get("session_id", int(time.time()))
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = session_id
        
    start_time = st.session_state.get("start_time", time.time())
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = start_time
        
    while ctx and ctx.state.playing:
        if not ctx.video_processor:
            time.sleep(0.1)
            continue

        vp: VideoProcessor = ctx.video_processor
        if vp.latest_rgb is None:
            time.sleep(0.05)
            continue

        now = time.time()
        if now - last_t < sample_every:
            time.sleep(0.1)
            continue
        
        last_t = now
        elapsed_seconds = now - st.session_state["start_time"]
        
        _run_inference(
            vp.latest_rgb, 
            video_timestamp=elapsed_seconds, 
            transcript_data_list=st.session_state["webcam_transcript"]
        )
        
        # Save it continuously so it isn't lost when the user hits 'Stop'
        if len(st.session_state["webcam_transcript"]) > 0:
            os.makedirs("transcripts", exist_ok=True)
            out_filename = f"transcripts/webcam_session_{session_id}.json"
            with open(out_filename, "w") as f:
                json.dump(st.session_state["webcam_transcript"], f, indent=2)
                    
elif input_source == "Video File" and uploaded_video is not None:
    if process_btn:
        st.session_state["video_processing_active"] = True
        st.session_state["video_transcript"] = []
        st.session_state["video_target_pts"] = 0.0
        st.session_state["video_frames_processed"] = 0
        st.session_state["video_summary_done"] = False
        st.session_state["video_dino_res"] = None
        st.session_state["video_rcp_res"] = None
        
    is_processing = st.session_state.get("video_processing_active", False)
    
    if is_processing:
        if stop_placeholder.button("Stop Processing 🛑", type="secondary", use_container_width=True, help="Click to cancel video processing."):
            st.session_state["video_processing_active"] = False
            st.session_state["video_summary_done"] = True
        else:
            with st.spinner("Extracting and analyzing frames..."):
                # Save the uploaded video temporarily to read it with av
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tf:
                    tf.write(uploaded_video.read())
                    tf.flush()
                    
                    try:
                        container = av.open(tf.name)
                        stream = container.streams.video[0]
                        
                        target_pts_sec = st.session_state["video_target_pts"]
                        frames_processed = st.session_state["video_frames_processed"]
                        
                        # Add a progress bar to the UI so it doesn't look frozen
                        progress_bar = st.progress(0.0, text="Analyzing video...")
                        duration_sec = float(stream.duration * stream.time_base) if stream.duration else 1.0
                        
                        while target_pts_sec < duration_sec:
                            offset = int(target_pts_sec / stream.time_base)
                            # M4 Pro Hardware Optimization: Jump immediately to the nearest keyframe 
                            container.seek(offset, stream=stream, any_frame=False)
                            
                            frame = None
                            for f in container.decode(stream):
                                if f.time >= target_pts_sec:
                                    frame = f
                                    break
                                    
                            if frame is None:
                                break  # Reached EOF where no further frames satisfy the constraint
                                
                            pts_seconds = frame.time
                            rgb = frame.to_ndarray(format="rgb24")
                            _run_inference(rgb, video_timestamp=pts_seconds, transcript_data_list=st.session_state["video_transcript"])
                            frames_processed += 1
                            st.session_state["video_frames_processed"] = frames_processed
                            
                            # Update progress bar
                            pct = min(1.0, float(pts_seconds) / duration_sec) if duration_sec > 0 else 0.0
                            progress_bar.progress(pct, text=f"Analyzed {frames_processed} frames... ({pts_seconds:.1f}s / {duration_sec:.1f}s)")
                            
                            # Artificially sleep for a moment to let the UI breathe
                            time.sleep(0.5)
                            
                            # Lock the next target payload strictly against the actual evaluated timestamp
                            target_pts_sec = pts_seconds + sample_every
                            st.session_state["video_target_pts"] = target_pts_sec
                            
                        progress_bar.progress(1.0, text=f"Finished! Analyzed {frames_processed} total frames.")
                        st.session_state["video_processing_active"] = False
                        st.session_state["video_summary_done"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        st.session_state["video_processing_active"] = False

    if st.session_state.get("video_summary_done", False) and not st.session_state.get("video_processing_active", False):
        transcript_data = st.session_state.get("video_transcript", [])
        if transcript_data:
            os.makedirs("transcripts", exist_ok=True)
            out_filename = f"transcripts/{Path(uploaded_video.name).stem}_transcript.json"
            with open(out_filename, "w") as f:
                json.dump(transcript_data, f, indent=2)
            st.success(f"Video processing complete. Transcript saved locally to `{out_filename}`!")
            
            # Check if we need to generate telemetry
            if st.session_state.get("video_dino_res") is None or st.session_state.get("video_rcp_res") is None:
                with st.spinner("Generating Grounding DINO Prompt and RCP Telemetry..."):
                    full_text = " ".join([item["params"]["text"] for item in transcript_data if item.get("params", {}).get("text")])
                    
                    # Convert JSON-RPC format into heavily structured string for the RCP prompt
                    rcp_text = "\n".join([
                        f"[{item['params']['timestamp']}] {item['params']['text']} (Confidence: {item['params'].get('confidence', 'N/A')})"
                        for item in transcript_data if 'params' in item
                    ])
                    
                    if full_text.strip():
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                            future_dino = executor.submit(vl.generate_dino_prompt, full_text, dino_max_tokens)
                            future_rcp = executor.submit(vl.generate_rcp_telemetry, rcp_text)
                            st.session_state["video_dino_res"] = future_dino.result()
                            st.session_state["video_rcp_res"] = future_rcp.result()
                            
            dino_res = st.session_state.get("video_dino_res")
            rcp_res = st.session_state.get("video_rcp_res")
            
            if dino_res and rcp_res:
                st.subheader("Automated Output Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    if dino_res.get("success"):
                        dino_text = dino_res["response"]
                        st.success("Grounding DINO Prompt:")
                        st.code(dino_text)
                        dino_filename = f"transcripts/{Path(uploaded_video.name).stem}_dino_prompt.txt"
                        with open(dino_filename, "w") as f:
                            f.write(dino_text)
                        st.caption(f"Saved to `{dino_filename}`")
                        # Sync to session_state so PathGuard HUD can auto-detect
                        st.session_state["cactus_dino_prompt_text"] = dino_text
                        st.session_state["cactus_dino_prompt_file"] = dino_filename
                    else:
                        st.error(f"Failed to generate DINO prompt: {dino_res.get('error')}")
                        
                with col2:
                    if rcp_res.get("success"):
                        rcp_json = rcp_res["response"]
                        st.success("RCP Site Telemetry:")
                        with st.expander("View RCP JSON Payload", expanded=False):
                            st.json(rcp_json)
                        rcp_filename = f"transcripts/{Path(uploaded_video.name).stem}_rcp.json"
                        with open(rcp_filename, "w") as f:
                            json.dump(rcp_json, f, indent=2)
                        st.caption(f"Saved to `{rcp_filename}`")
                    else:
                        st.error(f"Failed to generate RCP Telemetry: {rcp_res.get('error')}")
        else:
            st.info("Video processing stopped before any frames were successfully analyzed.")

# Summarization Hooks for Live Webcam (manual trigger since stream is infinite)
if input_source == "Live Webcam" and st.session_state.get("webcam_transcript"):
    st.divider()
    if st.button("Summarize Session & Generate Telemetry", type="primary", use_container_width=True):
        with st.spinner("Generating Grounding DINO Prompt and RCP Telemetry from session history..."):
            transcript_data = st.session_state["webcam_transcript"]
            full_text = " ".join([item["params"]["text"] for item in transcript_data if item.get("params", {}).get("text")])
            
            # Convert JSON-RPC format into heavily structured string for the RCP prompt
            rcp_text = "\n".join([
                f"[{item['params']['timestamp']}] {item['params']['text']} (Confidence: {item['params'].get('confidence', 'N/A')})"
                for item in transcript_data if 'params' in item
            ])
            
            if full_text.strip():
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_dino = executor.submit(vl.generate_dino_prompt, full_text, dino_max_tokens)
                    future_rcp = executor.submit(vl.generate_rcp_telemetry, rcp_text)
                    
                    dino_res = future_dino.result()
                    rcp_res = future_rcp.result()
                    
                st.subheader("Automated Output Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    if dino_res.get("success"):
                        dino_text = dino_res["response"]
                        st.success("Grounding DINO Prompt:")
                        st.code(dino_text)
                        
                        session_id = st.session_state.get("session_id")
                        dino_filename = f"transcripts/webcam_session_{session_id}_dino_prompt.txt"
                        try:
                            with open(dino_filename, "w") as f:
                                f.write(dino_text)
                            st.caption(f"Saved to `{dino_filename}`")
                            # Sync to session_state so PathGuard HUD can auto-detect
                            st.session_state["cactus_dino_prompt_text"] = dino_text
                            st.session_state["cactus_dino_prompt_file"] = dino_filename
                        except Exception as e:
                            st.error(f"Failed to save text artifact: {e}")
                    else:
                        st.error(f"Failed to generate DINO prompt: {dino_res.get('error')}")
                        
                with col2:
                    if rcp_res.get("success"):
                        rcp_json = rcp_res["response"]
                        st.success("RCP Site Telemetry:")
                        with st.expander("View RCP JSON Payload", expanded=False):
                            st.json(rcp_json)
                        
                        session_id = st.session_state.get("session_id")
                        rcp_filename = f"transcripts/webcam_session_{session_id}_rcp.json"
                        try:
                            with open(rcp_filename, "w") as f:
                                json.dump(rcp_json, f, indent=2)
                            st.caption(f"Saved to `{rcp_filename}`")
                        except Exception as e:
                            st.error(f"Failed to save JSON artifact: {e}")
                    else:
                        st.error(f"Failed to generate RCP Telemetry: {rcp_res.get('error')}")
            else:
                st.warning("No transcript data to summarize yet. Let the camera run longer!")
