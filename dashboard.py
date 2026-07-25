import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.camera import Camera
from src.face_detection import FaceDetector

from src.heart_rate import (
    extract_forehead,
    update_signal,
    green_signal,
    smooth_bpm,
    get_signal_quality,
    check_roi_quality,
    get_bpm_status,
    get_measurement_progress,
    reset_buffers,
    get_sample_count,
)

from src.signal_processing import estimate_bpm


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Heart Rate AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(220,38,38,.10), transparent 30%),
            radial-gradient(circle at 85% 20%, rgba(14,165,233,.08), transparent 30%),
            #071019;
    }

    [data-testid="stSidebar"] {
        background: #09131d;
        border-right: 1px solid #1e293b;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
        color: #f8fafc;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .card {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid #263548;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,.20);
    }

    .bpm-card {
        text-align: center;
        background:
            linear-gradient(
                145deg,
                rgba(127,29,29,.30),
                rgba(15,23,42,.95)
            );
        border: 1px solid rgba(239,68,68,.35);
        border-radius: 22px;
        padding: 28px;
    }

    .bpm-number {
        font-size: 72px;
        line-height: 1;
        font-weight: 800;
        color: #fb7185;
    }

    .bpm-label {
        color: #94a3b8;
        font-size: 17px;
        margin-top: 8px;
    }

    .normal-status {
        display: inline-block;
        margin-top: 16px;
        padding: 8px 18px;
        border-radius: 100px;
        color: #4ade80;
        background: rgba(34,197,94,.12);
        border: 1px solid rgba(34,197,94,.35);
        font-weight: 700;
    }

    .low-status {
        display: inline-block;
        margin-top: 16px;
        padding: 8px 18px;
        border-radius: 100px;
        color: #38bdf8;
        background: rgba(14,165,233,.12);
        border: 1px solid rgba(14,165,233,.35);
        font-weight: 700;
    }

    .high-status {
        display: inline-block;
        margin-top: 16px;
        padding: 8px 18px;
        border-radius: 100px;
        color: #fb7185;
        background: rgba(239,68,68,.12);
        border: 1px solid rgba(239,68,68,.35);
        font-weight: 700;
    }

    .calculating-status {
        display: inline-block;
        margin-top: 16px;
        padding: 8px 18px;
        border-radius: 100px;
        color: #facc15;
        background: rgba(250,204,21,.10);
        border: 1px solid rgba(250,204,21,.30);
        font-weight: 700;
    }

    .section-title {
        color: #e2e8f0;
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .system-ready {
        color: #4ade80;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "running": False,
    "camera": None,
    "detector": None,
    "bpm": None,
    "bpm_values": [],
    "history": [],
    "start_time": None,
    "last_log_time": 0.0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("❤️ Heart Rate AI")

    st.caption(
        "Contactless heart-rate estimation using "
        "facial video and rPPG."
    )

    st.divider()

    st.subheader("Measurement")

    required_samples = st.slider(
        "Required samples",
        min_value=180,
        max_value=300,
        value=240,
        step=10,
    )

    show_roi = st.toggle(
        "Show forehead ROI",
        value=True,
    )

    show_signal = st.toggle(
        "Show live signal",
        value=True,
    )

    st.divider()

    st.info(
        "For better readings:\n\n"
        "• Keep your face still\n"
        "• Look toward the camera\n"
        "• Use steady lighting\n"
        "• Avoid strong backlighting"
    )

    st.warning(
        "This project is for educational/research use "
        "and is not a medical diagnostic device."
    )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([5, 1])

with header_left:

    st.markdown(
        '<div class="main-title">❤️ Heart Rate AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Real-Time Contactless Cardiovascular Monitoring'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:

    st.markdown(
        '<div class="system-ready">● SYSTEM READY</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# CONTROL BUTTONS
# ============================================================

c1, c2, c3, c4 = st.columns([1, 1, 1, 3])

with c1:

    start_clicked = st.button(
        "▶ Start",
        use_container_width=True,
        type="primary",
    )

with c2:

    stop_clicked = st.button(
        "■ Stop",
        use_container_width=True,
    )

with c3:

    reset_clicked = st.button(
        "↻ Reset",
        use_container_width=True,
    )


# ============================================================
# START
# ============================================================

if start_clicked and not st.session_state.running:

    reset_buffers()

    st.session_state.bpm = None
    st.session_state.bpm_values = []
    st.session_state.start_time = time.time()
    st.session_state.last_log_time = 0.0

    camera = Camera()

    if camera.open():

        st.session_state.camera = camera
        st.session_state.detector = FaceDetector()
        st.session_state.running = True

    else:

        st.error(
            "Could not open the camera. "
            "Check macOS camera permissions."
        )


# ============================================================
# STOP
# ============================================================

if stop_clicked:

    st.session_state.running = False

    if st.session_state.camera is not None:
        st.session_state.camera.release()

    if st.session_state.detector is not None:
        st.session_state.detector.close()

    st.session_state.camera = None
    st.session_state.detector = None


# ============================================================
# RESET
# ============================================================

if reset_clicked:

    st.session_state.running = False

    if st.session_state.camera is not None:
        st.session_state.camera.release()

    if st.session_state.detector is not None:
        st.session_state.detector.close()

    st.session_state.camera = None
    st.session_state.detector = None

    reset_buffers()

    st.session_state.bpm = None
    st.session_state.bpm_values = []
    st.session_state.history = []
    st.session_state.start_time = None

    st.rerun()


# ============================================================
# DASHBOARD LAYOUT
# ============================================================

camera_col, bpm_col = st.columns([1.7, 1])

with camera_col:

    st.markdown(
        '<div class="section-title">Live Camera</div>',
        unsafe_allow_html=True,
    )

    camera_placeholder = st.empty()

    roi_message = st.empty()


with bpm_col:

    st.markdown(
        '<div class="section-title">Current Heart Rate</div>',
        unsafe_allow_html=True,
    )

    bpm_placeholder = st.empty()

    quality_placeholder = st.empty()

    progress_placeholder = st.empty()


st.divider()


# ============================================================
# SIGNAL GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">Live Pulse Signal</div>',
    unsafe_allow_html=True,
)

signal_placeholder = st.empty()


# ============================================================
# STATISTICS
# ============================================================

st.markdown(
    '<div class="section-title">Session Statistics</div>',
    unsafe_allow_html=True,
)

stat1, stat2, stat3, stat4 = st.columns(4)

avg_placeholder = stat1.empty()
min_placeholder = stat2.empty()
max_placeholder = stat3.empty()
duration_placeholder = stat4.empty()


# ============================================================
# HISTORY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Measurement History</div>',
    unsafe_allow_html=True,
)

history_placeholder = st.empty()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_bpm(bpm):

    if bpm is None:

        bpm_placeholder.markdown(
            """
            <div class="bpm-card">
                <div class="bpm-number">--</div>
                <div class="bpm-label">BPM</div>
                <div class="calculating-status">
                    ● CALCULATING
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    status = get_bpm_status(bpm)

    css_class = {
        "NORMAL": "normal-status",
        "LOW": "low-status",
        "HIGH": "high-status",
    }.get(
        status,
        "calculating-status",
    )

    bpm_placeholder.markdown(
        f"""
        <div class="bpm-card">
            <div class="bpm-number">{bpm:.1f}</div>
            <div class="bpm-label">BEATS PER MINUTE</div>
            <div class="{css_class}">
                ● {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal():

    if not show_signal:
        signal_placeholder.info(
            "Live signal display is disabled."
        )
        return

    values = list(green_signal)

    if len(values) < 2:
        signal_placeholder.info(
            "Waiting for pulse signal..."
        )
        return

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            y=values,
            mode="lines",
            name="Green intensity",
        )
    )

    figure.update_layout(
        height=300,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        paper_bgcolor="#071019",
        plot_bgcolor="#0f172a",
        font=dict(
            color="#cbd5e1",
        ),
        xaxis_title="Samples",
        yaxis_title="Green Intensity",
        showlegend=False,
    )

    signal_placeholder.plotly_chart(
        figure,
        use_container_width=True,
        key=f"signal_{time.time_ns()}",
    )


def render_statistics():

    values = st.session_state.bpm_values

    if values:

        avg_bpm = np.mean(values)
        min_bpm = np.min(values)
        max_bpm = np.max(values)

        avg_placeholder.metric(
            "Average BPM",
            f"{avg_bpm:.1f}",
        )

        min_placeholder.metric(
            "Minimum BPM",
            f"{min_bpm:.1f}",
        )

        max_placeholder.metric(
            "Maximum BPM",
            f"{max_bpm:.1f}",
        )

    else:

        avg_placeholder.metric(
            "Average BPM",
            "--",
        )

        min_placeholder.metric(
            "Minimum BPM",
            "--",
        )

        max_placeholder.metric(
            "Maximum BPM",
            "--",
        )

    if st.session_state.start_time:

        duration = (
            time.time()
            - st.session_state.start_time
        )

        duration_placeholder.metric(
            "Session",
            f"{int(duration)} sec",
        )

    else:

        duration_placeholder.metric(
            "Session",
            "0 sec",
        )


def render_history():

    history = st.session_state.history

    if not history:
        history_placeholder.info(
            "No BPM readings recorded yet."
        )
        return

    df = pd.DataFrame(history)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["BPM"],
            mode="lines+markers",
            name="BPM",
        )
    )

    figure.update_layout(
        height=300,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        paper_bgcolor="#071019",
        plot_bgcolor="#0f172a",
        font=dict(
            color="#cbd5e1",
        ),
        xaxis_title="Time",
        yaxis_title="BPM",
    )

    history_placeholder.plotly_chart(
        figure,
        use_container_width=True,
        key=f"history_{time.time_ns()}",
    )


# ============================================================
# INITIAL DISPLAY
# ============================================================

render_bpm(st.session_state.bpm)
render_signal()
render_statistics()
render_history()


# ============================================================
# LIVE MEASUREMENT
# ============================================================

if st.session_state.running:

    camera = st.session_state.camera
    detector = st.session_state.detector

    success, frame = camera.read()

    if success and frame is not None:

        # Mirror camera
        frame = cv2.flip(
            frame,
            1,
        )

        landmarks = detector.detect(frame)

        if landmarks is not None:

            roi = extract_forehead(
                frame,
                landmarks,
            )

            roi_ok, roi_status = check_roi_quality(
                roi
            )

            if roi_ok:

                value = update_signal(roi)

                if value is not None:

                    roi_message.success(
                        "Forehead detected • Signal collecting"
                    )

                # --------------------------------------------
                # Draw forehead rectangle approximately
                # --------------------------------------------

                if show_roi:

                    h, w = frame.shape[:2]

                    forehead = landmarks.landmark[10]

                    cx = int(
                        forehead.x * w
                    )

                    cy = int(
                        forehead.y * h
                    )

                    cv2.rectangle(
                        frame,
                        (cx - 35, cy - 15),
                        (cx + 35, cy + 35),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        "FOREHEAD ROI",
                        (cx - 50, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # --------------------------------------------
                # BPM calculation
                # --------------------------------------------

                if get_sample_count() >= required_samples:

                    fps = camera.get_fps()

                    estimated_bpm = estimate_bpm(
                        list(green_signal),
                        fps,
                    )

                    bpm = smooth_bpm(
                        estimated_bpm
                    )

                    if bpm is not None:

                        st.session_state.bpm = bpm

                        st.session_state.bpm_values.append(
                            bpm
                        )

                        # Prevent unlimited statistics data
                        st.session_state.bpm_values = (
                            st.session_state.bpm_values[-300:]
                        )

                        # Add history approximately once/sec
                        current_time = time.time()

                        if (
                            current_time
                            - st.session_state.last_log_time
                            >= 1
                        ):

                            st.session_state.history.append(
                                {
                                    "Time": datetime.now().strftime(
                                        "%H:%M:%S"
                                    ),
                                    "BPM": bpm,
                                }
                            )

                            st.session_state.last_log_time = (
                                current_time
                            )

            else:

                roi_message.warning(
                    roi_status
                )

        else:

            roi_message.warning(
                "Face not detected. Look toward the camera."
            )

        # BGR -> RGB for Streamlit
        display_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        camera_placeholder.image(
            display_frame,
            channels="RGB",
            use_container_width=True,
        )

        # --------------------------------------------
        # Quality
        # --------------------------------------------

        quality, quality_score = (
            get_signal_quality()
        )

        quality_placeholder.metric(
            "Signal Quality",
            f"{quality} • {quality_score}%",
        )

        # --------------------------------------------
        # Progress
        # --------------------------------------------

        progress = get_measurement_progress(
            required_samples
        )

        progress_placeholder.progress(
            progress / 100,
            text=f"Measurement progress: {progress}%",
        )

        # --------------------------------------------
        # Refresh components
        # --------------------------------------------

        render_bpm(
            st.session_state.bpm
        )

        render_signal()

        render_statistics()

        render_history()

        # Small delay reduces CPU usage
        time.sleep(0.03)

        st.rerun()

    else:

        st.error(
            "Camera frame could not be read."
        )

        st.session_state.running = False


else:

    camera_placeholder.info(
        "Camera is stopped. Click **Start** to begin measurement."
    )