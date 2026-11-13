import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.camera import Camera
from src.face_detection import FaceDetector
from src.signal_processing import estimate_bpm, estimate_chrom_bpm


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Rate AI",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MEASUREMENT SETTINGS
# ============================================================

MIN_SAMPLES = 180
MAX_SAMPLES = 300

# BPM is calculated approximately once every second
BPM_UPDATE_SECONDS = 1.0

# Graph updates slower than camera to reduce flickering
GRAPH_UPDATE_SECONDS = 0.75

# Number of BPM candidates needed before locking
LOCK_WINDOW = 6

# Maximum standard deviation allowed for lock
LOCK_STD_THRESHOLD = 4.0

# Reject sudden BPM jumps
MAX_BPM_JUMP = 20.0

MIN_VALID_BPM = 45.0
MAX_VALID_BPM = 180.0

# Reset signal if face disappears too long
FACE_LOST_RESET_SECONDS = 2.0


# ============================================================
# CSS
# ============================================================

THEMES = {
    "Ocean": {
        "page": "#eef6f8", "sidebar": "#dceef1", "surface": "#ffffff",
        "text": "#183344", "muted": "#526b78", "primary": "#087f8c",
        "accent": "#13a89e", "border": "#c9e0e5", "bpm_bg": "#e2f5f3",
        "bpm": "#087f8c", "info_bg": "#dceefb", "info_text": "#164f82",
    },
    "Midnight": {
        "page": "#111827", "sidebar": "#182235", "surface": "#202c40",
        "text": "#f3f6fc", "muted": "#b6c3d6", "primary": "#4f8cff",
        "accent": "#a78bfa", "border": "#36445c", "bpm_bg": "#2b2448",
        "bpm": "#c4b5fd", "info_bg": "#203b60", "info_text": "#dbeafe",
    },
    "Sage": {
        "page": "#f3f5ed", "sidebar": "#e5eadb", "surface": "#ffffff",
        "text": "#293d30", "muted": "#617363", "primary": "#477653",
        "accent": "#93a96b", "border": "#d5dfca", "bpm_bg": "#eaf2df",
        "bpm": "#477653", "info_bg": "#e1ebd8", "info_text": "#304d35",
    },
    "Minimal": {
        "page": "#f4f6f8", "sidebar": "#e9edf1", "surface": "#ffffff",
        "text": "#202b38", "muted": "#5e6b79", "primary": "#334e68",
        "accent": "#0f8b8d", "border": "#d7dfe7", "bpm_bg": "#e4f3f2",
        "bpm": "#087f8c", "info_bg": "#e3edf7", "info_text": "#284d70",
    },

    "Cyberpunk": {
        "page": "#100d1c", "sidebar": "#19132b", "surface": "#211a35",
        "text": "#f5f0ff", "muted": "#c0b4dc", "primary": "#a855f7",
        "accent": "#22d3ee", "border": "#44305f", "bpm_bg": "#261b3d",
        "bpm": "#22d3ee", "info_bg": "#202744", "info_text": "#dbeafe",
    },
    "Deep Ocean": {
        "page": "#071923", "sidebar": "#0b2533", "surface": "#103141",
        "text": "#eaf7fc", "muted": "#a8c5d1", "primary": "#0ea5b7",
        "accent": "#38d9c5", "border": "#245060", "bpm_bg": "#103542",
        "bpm": "#38d9c5", "info_bg": "#16384c", "info_text": "#d8f1ff",
    },
    "AMOLED Black": {
        "page": "#000000", "sidebar": "#080b0a", "surface": "#111614",
        "text": "#f4fff8", "muted": "#b2c5b9", "primary": "#00a86b",
        "accent": "#65f0ad", "border": "#294638", "bpm_bg": "#0d2118",
        "bpm": "#65f0ad", "info_bg": "#10271d", "info_text": "#d7ffea",
    },
}

if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = "Ocean"

selected_theme = st.session_state.selected_theme
theme = THEMES[selected_theme]

st.markdown(
    f"""
<style>
:root {{
    color-scheme: {"dark" if selected_theme in ("Midnight", "Cyberpunk", "Deep Ocean", "AMOLED Black") else "light"};
}}
.stApp {{
    background: {theme["page"]} !important;
    color: {theme["text"]} !important;
}}
.block-container {{
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}}
section[data-testid="stSidebar"] {{
    background: {theme["sidebar"]} !important;
    border-right: 1px solid {theme["border"]};
}}
.stApp, .stApp p, .stApp label, .stApp li,
.stApp [data-testid="stMarkdownContainer"] {{
    color: {theme["text"]};
}}
.main-title {{
    color: {theme["text"]} !important;
    font-size: clamp(32px, 4vw, 46px);
    font-weight: 800;
    letter-spacing: -1px;
}}
.subtitle {{
    color: {theme["muted"]} !important;
    font-size: 17px;
    margin-bottom: 22px;
}}
.system-ready {{
    color: {theme["primary"]} !important;
    font-weight: 700;
    text-align: right;
    padding-top: 15px;
}}
h1, h2, h3 {{
    color: {theme["text"]} !important;
}}
.bpm-card {{
    border: 1px solid {theme["border"]};
    border-radius: 22px;
    padding: 30px 20px;
    text-align: center;
    background: {theme["bpm_bg"]};
    min-height: 230px;
}}
.bpm-number {{
    font-size: clamp(48px, 5vw, 64px);
    font-weight: 800;
    color: {theme["bpm"]};
    line-height: 1.1;
}}
.bpm-unit {{
    color: {theme["muted"]};
    font-size: 15px;
    margin-top: 10px;
}}
.badge {{
    display: inline-block;
    margin-top: 18px;
    padding: 8px 18px;
    border-radius: 25px;
    font-weight: 700;
}}
.normal, .locked, .detected, .complete {{
    color: #176544;
    border: 1px solid #9ad7b6;
    background: #e4f7ec;
}}
.low {{
    color: #185c91;
    border: 1px solid #9bcdf4;
    background: #e8f5ff;
}}
.high {{
    color: #87451c;
    border: 1px solid #efc08a;
    background: #fff1dc;
}}
.waiting, .stabilizing, .not-detected {{
    color: #76570a;
    border: 1px solid #ead58b;
    background: #fff8dc;
}}
.detected, .complete, .not-detected {{
    padding: 14px;
    border-radius: 14px;
    font-weight: 700;
}}
div[data-testid="stMetric"] {{
    background: {theme["surface"]};
    padding: 18px;
    border: 1px solid {theme["border"]};
    border-radius: 16px;
}}
div[data-testid="stMetricLabel"] {{
    color: {theme["muted"]} !important;
}}
div[data-testid="stMetricValue"] {{
    color: {theme["text"]} !important;
}}
div[data-testid="stButton"] button,
.stButton > button {{
    min-height: 44px;
    white-space: normal;
    font-weight: 700;
    border-radius: 12px;
    border: 1px solid {theme["border"]};
}}
div[data-testid="stButton"] button[kind="primary"],
.stButton > button[kind="primary"] {{
    background: {theme["primary"]};
    color: #ffffff;
    border-color: {theme["primary"]};
}}
div[data-testid="stButton"] button p {{
    font-size: 0.95rem;
    line-height: 1.25;
}}
div[data-testid="stAlert"] {{
    border-radius: 12px;
}}
input, textarea, div[data-baseweb="select"] {{
    border-radius: 10px !important;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border-radius: 12px;
    overflow: hidden;
}}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        transition: none !important;
        animation: none !important;
    }}
}}

/* Modern, friendly dashboard polish */
.stApp {{
    transition: background-color 0.25s ease;
}}

.block-container {{
    padding-left: clamp(1rem, 3vw, 2.5rem);
    padding-right: clamp(1rem, 3vw, 2.5rem);
}}

.main-title {{
    line-height: 1.15;
    margin-bottom: 0.35rem;
}}

.subtitle {{
    line-height: 1.6;
    opacity: 0.88;
}}

.system-ready {{
    border: 1px solid {{theme["border"]}};
    border-radius: 999px;
    padding: 0.55rem 0.8rem;
    text-align: center;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    background: {{theme["surface"]}};
    color: {{theme["text"]}};
}}

.stButton > button {{
    min-height: 2.8rem;
    border-radius: 12px;
    font-weight: 650;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px {{theme["border"]}};
}}

.stButton > button:focus-visible {{
    outline: 3px solid {{theme["accent"]}};
    outline-offset: 3px;
}}

.stSlider, .stSelectbox, .stToggle {{
    padding-top: 0.15rem;
    padding-bottom: 0.15rem;
}}

.stApp [data-testid="stMetric"] {{
    background: {{theme["surface"]}};
    border: 1px solid {{theme["border"]}};
    border-radius: 18px;
    padding: 1.1rem;
    box-shadow: 0 4px 14px {{theme["border"]}};
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.stApp [data-testid="stMetric"]:hover {{
    transform: translateY(-3px);
    box-shadow: 0 9px 22px {{theme["border"]}};
}}

.stApp [data-testid="stMetricLabel"] {{
    font-size: 0.9rem;
    font-weight: 650;
    letter-spacing: 0.02em;
}}

.stApp [data-testid="stMetricValue"] {{
    font-size: clamp(1.35rem, 2.5vw, 2rem);
    font-weight: 800;
    letter-spacing: -0.035em;
}}

@media (prefers-reduced-motion: reduce) {{
    .stApp [data-testid="stMetric"] {{
        transition: none;
    }}
    .stApp [data-testid="stMetric"]:hover {{
        transform: none;
    }}
}}

.bpm-card {{
    background: linear-gradient(
        145deg,
        {{theme["bpm_bg"]}},
        {{theme["surface"]}}
    );
    border: 1px solid {{theme["border"]}};
    border-radius: 22px;
    padding: clamp(1.2rem, 3vw, 2rem);
    text-align: center;
    box-shadow: 0 8px 24px {{theme["border"]}};
    margin-bottom: 1rem;
}}

.bpm-number {{
    font-size: clamp(3.5rem, 8vw, 5.5rem);
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -0.055em;
    color: {{theme["bpm"]}};
    font-variant-numeric: tabular-nums;
}}

.bpm-unit {{
    margin-top: 0.65rem;
    margin-bottom: 1rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    color: {{theme["muted"]}};
}}

.badge {{
    display: inline-block;
    border-radius: 999px;
    padding: 0.4rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 700;
    line-height: 1.4;
}}

.stApp h2, .stApp h3 {{
    letter-spacing: -0.025em;
}}

.stApp hr {{
    border-color: {{theme["border"]}};
    opacity: 0.8;
}}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    line-height: 1.6;
}}

@media (prefers-reduced-motion: reduce) {{
    .stApp, .stButton > button {{
        transition: none !important;
    }}
    .stButton > button:hover {{
        transform: none;
    }}
}}

</style>
""",
    unsafe_allow_html=True,
)



# ============================================================
# ============================================================
# AUTHENTICATION GATE
# ============================================================

import database

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if st.session_state.authenticated_user is None:

    # Premium login-page styling. Applied only before login.
    st.markdown("""
    <style>
    /* Login page background */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 15% 15%, #123b50 0%, transparent 42%),
            radial-gradient(ellipse at 90% 85%, #064451 0%, transparent 38%),
            #06141f !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        display: none !important;
    }

    .block-container {
        max-width: 1120px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }

    /* Brand heading */
    .hr-brand {
        color: #f0fbff;
        font-size: clamp(34px, 4vw, 54px);
        font-weight: 800;
        letter-spacing: -1.8px;
        line-height: 1.1;
        margin: 12px 0;
    }

    .hr-brand span {
        color: #39e6d0;
    }

    .hr-tagline {
        color: #b1cbd8;
        font-size: 17px;
        line-height: 1.7;
        max-width: 470px;
    }

    .hr-eyebrow {
        display: inline-block;
        color: #54f0dc;
        border: 1px solid rgba(84, 240, 220, 0.35);
        background: rgba(30, 190, 180, 0.09);
        border-radius: 30px;
        padding: 7px 13px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.6px;
    }

    .hr-heart {
        color: #46ead5;
        font-size: 54px;
        margin: 12px 0 6px;
        text-shadow: 0 0 25px rgba(57, 230, 208, 0.35);
    }

    .hr-status {
        color: #91b5c5;
        font-size: 13px;
        margin-top: 12px;
    }

    /* Login form surface */
    div[data-testid="stForm"] {
        background: rgba(13, 39, 54, 0.78);
        border: 1px solid rgba(87, 159, 181, 0.30);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stTabs"] button {
        color: #b6d2df !important;
        font-weight: 600 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #52edda !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #39e6d0 !important;
    }

    .stTextInput label {
        color: #d8edf5 !important;
        font-weight: 600 !important;
    }

    .stTextInput input {
        background: #0a202d !important;
        color: #f0fbff !important;
        border: 1px solid #285165 !important;
        border-radius: 10px !important;
        min-height: 44px;
    }

    .stTextInput input:focus {
        border-color: #39e6d0 !important;
        box-shadow: 0 0 0 1px #39e6d0 !important;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(100deg, #0fa6b7, #39d9c5) !important;
        color: #05202a !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        min-height: 46px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 6px 24px rgba(57, 230, 208, 0.20);
        transform: translateY(-1px);
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    .hr-footer {
        color: #7898a8;
        font-size: 12px;
        text-align: center;
        margin-top: 24px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1.5rem !important;
        }

        .hr-brand {
            font-size: 36px;
        }

        div[data-testid="stForm"] {
            padding: 16px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Two-column welcome and authentication layout.
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("""
        <div style="padding-top: 32px;">
            <div class="hr-eyebrow">AI-POWERED WELLNESS</div>
            <div class="hr-heart">♥</div>
            <div class="hr-brand">Heart Rate<br><span>AI.</span></div>
            <div class="hr-tagline">
                Your personal wellness space, powered by intelligent
                heart-rate monitoring and real-time signal analysis.
            </div>
            <div style="margin-top: 28px; max-width: 440px;">
                <svg viewBox="0 0 440 100"
                     style="width:100%; height:auto;"
                     xmlns="http://www.w3.org/2000/svg"
                     role="img" aria-label="Heartbeat waveform">
                    <defs>
                        <linearGradient id="hrline" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%" stop-color="#0fa6b7"/>
                            <stop offset="100%" stop-color="#55f0d9"/>
                        </linearGradient>
                    </defs>
                    <path d="M0 53 L70 53 L92 53 L108 39
                             L124 69 L145 12 L169 83
                             L188 53 L225 53 L246 53
                             L261 42 L277 62 L294 53
                             L440 53"
                          fill="none" stroke="url(#hrline)"
                          stroke-width="3" stroke-linecap="round"
                          stroke-linejoin="round"/>
                    <path d="M0 53 L440 53"
                          fill="none" stroke="#245064"
                          stroke-width="1" stroke-dasharray="4 8"/>
                </svg>
            </div>
            <div class="hr-status">●  Intelligent monitoring · Personal health insights</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div style="padding-top: 20px;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="color:#f0fbff;font-size:27px;'
            'font-weight:750;margin-bottom:5px;">Welcome back</div>'
            '<div style="color:#a9c6d4;font-size:14px;'
            'margin-bottom:22px;">Sign in to continue to your wellness space.</div>',
            unsafe_allow_html=True,
        )

        login_tab, register_tab = st.tabs(["Sign In", "Create Account"])

        with login_tab:
            with st.form("login_form"):
                login_username = st.text_input(
                    "Username",
                    key="login_username",
                    placeholder="Enter your username",
                )
                login_password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                    placeholder="Enter your password",
                )
                login_submitted = st.form_submit_button(
                    "Sign In  →",
                    type="primary",
                    use_container_width=True,
                )

            if login_submitted:
                if not login_username.strip() or not login_password:
                    st.warning("Please enter your username and password.")
                else:
                    user = database.authenticate_user(
                        login_username.strip(), login_password
                    )
                    if user:
                        st.session_state.authenticated_user = user
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

            st.caption(
                "Use your registered account. Administrator access continues "
                "to use your existing account role."
            )

        with register_tab:
            with st.form("register_form"):
                register_name = st.text_input(
                    "Full name",
                    key="register_name",
                    placeholder="Enter your full name",
                )
                register_username = st.text_input(
                    "Choose a username",
                    key="register_username",
                    placeholder="Create a username",
                )
                register_password = st.text_input(
                    "Create a password (at least 8 characters)",
                    type="password",
                    key="register_password",
                    placeholder="Create a strong password",
                )
                register_confirm = st.text_input(
                    "Confirm password",
                    type="password",
                    key="register_confirm",
                    placeholder="Re-enter your password",
                )
                register_submitted = st.form_submit_button(
                    "Create Account  →",
                    type="primary",
                    use_container_width=True,
                )

            if register_submitted:
                if not all([
                    register_name.strip(),
                    register_username.strip(),
                    register_password,
                    register_confirm,
                ]):
                    st.warning("Please fill in all fields.")
                elif len(register_password) < 8:
                    st.warning("Your password must be at least 8 characters.")
                elif register_password != register_confirm:
                    st.error("The passwords do not match.")
                else:
                    success, message = database.register_user(
                        register_name.strip(),
                        register_username.strip(),
                        register_password,
                    )
                    if success:
                        st.success(
                            message + " You can now sign in using the Sign In tab."
                        )
                    else:
                        st.error(message)

        st.markdown(
            '<div class="hr-footer">'
            'HEART RATE AI · INTELLIGENCE FOR YOUR WELLNESS'
            '</div>',
            unsafe_allow_html=True,
        )

    st.stop()

current_user = st.session_state.authenticated_user
st.session_state.user_id = current_user["id"]
st.session_state.user_role = current_user["role"]
st.session_state.user_name = current_user["full_name"]


# SESSION STATE
# ============================================================

defaults = {

    "running": False,

    "camera": None,
    "detector": None,

    "signal_buffer": [],
    "timestamp_buffer": [],
    "rgb_buffer": [],

    "candidate_bpms": [],

    "raw_bpm": None,
    "locked_bpm": None,
    "bpm_locked": False,

    "actual_fps": None,

    "all_bpm": [],
    "current_bpm": None,

    "session_start": None,

    "last_bpm_time": 0.0,
    "last_graph_time": 0.0,

    "last_face_time": 0.0,
    "face_was_lost": False,

    "iris_colour_history": {"LEFT IRIS": [], "RIGHT IRIS": []},

    "history": [],
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# RESET MEASUREMENT
# ============================================================

def reset_measurement():

    st.session_state.signal_buffer = []

    st.session_state.timestamp_buffer = []

    st.session_state.rgb_buffer = []

    st.session_state.candidate_bpms = []

    st.session_state.raw_bpm = None

    st.session_state.locked_bpm = None

    st.session_state.bpm_locked = False

    st.session_state.actual_fps = None

    st.session_state.current_bpm = None

    st.session_state.all_bpm = []

    st.session_state.history = []

    st.session_state.iris_colour_history = {"LEFT IRIS": [], "RIGHT IRIS": []}

    st.session_state.last_bpm_time = 0.0

    st.session_state.last_graph_time = 0.0

    st.session_state.last_face_time = (
        time.perf_counter()
    )

    st.session_state.face_was_lost = False

    st.session_state.session_start = None


# ============================================================
# CAMERA CLEANUP
# ============================================================

def close_camera():

    camera = st.session_state.camera

    if camera is None:
        return

    try:

        if hasattr(camera, "close"):

            camera.close()

        elif hasattr(camera, "release"):

            camera.release()

        elif hasattr(camera, "cap"):

            if camera.cap is not None:

                camera.cap.release()

    except Exception:

        pass

    st.session_state.camera = None


# ============================================================
# DETECTOR CLEANUP
# ============================================================

def close_detector():

    detector = st.session_state.detector

    if detector is None:
        return

    try:

        detector.close()

    except Exception:

        pass

    st.session_state.detector = None


# ============================================================
# ACTUAL FPS
# ============================================================

def calculate_actual_fps(timestamps):

    if timestamps is None:

        return None

    if len(timestamps) < 30:

        return None

    times = np.asarray(
        timestamps,
        dtype=np.float64
    )

    differences = np.diff(times)

    if len(differences) == 0:

        return None

    differences = differences[
        (differences > 0.001)
        &
        (differences < 0.2)
    ]

    if len(differences) < 10:

        return None

    median_interval = np.median(
        differences
    )

    if median_interval <= 0:

        return None

    fps = 1.0 / median_interval

    if fps < 5 or fps > 120:

        return None

    return float(fps)


# ============================================================
# FOREHEAD ROI
# ============================================================

def get_forehead_roi(frame, face_landmarks):
    """Locate both eyes, then place a wider pulse ROI above the eyebrows."""

    if frame is None or face_landmarks is None:
        return None

    h, w = frame.shape[:2]
    landmarks = face_landmarks.landmark

    try:
        # Eye contour landmarks from MediaPipe Face Mesh.
        eye_ids = [33, 133, 159, 145, 362, 263, 386, 374]
        eye_points = np.asarray([
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in eye_ids
        ], dtype=np.int32)

        ex, ey, ew, eh = cv2.boundingRect(eye_points)

        # Keep the forehead ROI clearly separated from the eye box.
        eye_center_x = ex + ew // 2
        roi_width = max(int(ew * 1.15), 50)
        roi_height = max(int(eh * 2.0), 24)
        gap = max(12, int(eh * 0.6))

        x = eye_center_x - roi_width // 2
        y = ey - gap - roi_height

        x = max(0, x)
        y = max(0, y)
        x2 = min(w, x + roi_width)
        y2 = min(h, y + roi_height)

        # If the forehead ROI would be too short, reject it rather than
        # accidentally sampling the eyes.
        if x2 <= x or y2 <= y or (y2 - y) < int(roi_height * 0.65):
            return None

        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            return None

        return roi, x, y, x2, y2, (ex, ey, ex + ew, ey + eh)

    except (IndexError, AttributeError, ValueError, TypeError):
        return None


# ============================================================
# GREEN SIGNAL
# ============================================================

def extract_green_value(roi):

    if roi is None:

        return None

    if roi.size == 0:

        return None

    green_channel = (
        roi[:, :, 1]
    )

    value = float(
        np.mean(
            green_channel
        )
    )

    if not np.isfinite(
        value
    ):

        return None

    return value


# ============================================================
# RGB SIGNAL (EXPERIMENTAL)
# ============================================================

def extract_rgb_values(roi):
    """Return mean red, green, and blue values from the forehead ROI."""

    if roi is None or roi.size == 0:
        return None

    # OpenCV frames use BGR channel order.
    means = np.mean(roi, axis=(0, 1))

    if len(means) < 3 or not np.isfinite(means[:3]).all():
        return None

    blue, green, red = map(float, means[:3])

    return {
        "red": red,
        "green": green,
        "blue": blue,
    }


# ============================================================
# VALID BPM
# ============================================================

def is_valid_bpm(bpm):

    if bpm is None:

        return False

    try:

        bpm = float(
            bpm
        )

    except Exception:

        return False

    if not np.isfinite(
        bpm
    ):

        return False

    if bpm < MIN_VALID_BPM:

        return False

    if bpm > MAX_VALID_BPM:

        return False

    return True


# ============================================================
# ADD BPM CANDIDATE
# ============================================================

def add_bpm_candidate(
    raw_bpm
):

    if not is_valid_bpm(
        raw_bpm
    ):

        return False

    raw_bpm = float(
        raw_bpm
    )

    candidates = (
        st.session_state.candidate_bpms
    )

    # ========================================================
    # JUMP REJECTION
    # ========================================================

    if len(candidates) >= 3:

        recent = candidates[
            -LOCK_WINDOW:
        ]

        recent_median = float(
            np.median(
                recent
            )
        )

        difference = abs(
            raw_bpm
            -
            recent_median
        )

        if difference > MAX_BPM_JUMP:

            print(
                "Rejected BPM jump: "
                f"{recent_median:.1f} "
                f"-> {raw_bpm:.1f}"
            )

            return False

    candidates.append(
        raw_bpm
    )

    if len(candidates) > 20:

        candidates.pop(0)

    return True


# ============================================================
# CHECK BPM LOCK
# ============================================================

def check_bpm_lock():

    candidates = (
        st.session_state.candidate_bpms
    )

    if len(candidates) < LOCK_WINDOW:

        return None

    recent = np.asarray(
        candidates[
            -LOCK_WINDOW:
        ],
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(recent)
    ):

        return None

    median_bpm = float(
        np.median(
            recent
        )
    )

    deviation = float(
        np.std(
            recent
        )
    )

    bpm_range = float(
        np.ptp(
            recent
        )
    )

    print(
        "LOCK CHECK | "
        f"Values="
        f"{np.round(recent, 1).tolist()} | "
        f"Median={median_bpm:.1f} | "
        f"STD={deviation:.2f} | "
        f"Range={bpm_range:.1f}"
    )

    if deviation <= LOCK_STD_THRESHOLD:

        return round(
            median_bpm,
            1
        )

    return None


# ============================================================
# SIGNAL QUALITY
# ============================================================

def calculate_quality():

    signal = (
        st.session_state.signal_buffer
    )

    n = len(signal)

    if n == 0:

        return 0

    if n < 60:

        return 30

    if n < 120:

        return 50

    if n < MIN_SAMPLES:

        return 70

    arr = np.asarray(
        signal,
        dtype=np.float64
    )

    if len(arr) < 20:

        return 50

    # Remove overall trend before judging signal variation

    x = np.arange(
        len(arr),
        dtype=np.float64
    )

    try:

        coefficients = np.polyfit(
            x,
            arr,
            1
        )

        trend = np.polyval(
            coefficients,
            x
        )

        clean = (
            arr
            -
            trend
        )

    except Exception:

        clean = (
            arr
            -
            np.mean(arr)
        )

    std = float(
        np.std(clean)
    )

    if std < 0.05:

        return 35

    diff = np.diff(
        clean
    )

    if len(diff) == 0:

        return 40

    motion = float(
        np.std(diff)
    )

    relative_motion = (
        motion
        /
        (std + 1e-8)
    )

    if relative_motion < 0.30:

        return 95

    if relative_motion < 0.50:

        return 85

    if relative_motion < 0.75:

        return 70

    if relative_motion < 1.0:

        return 55

    return 35


# ============================================================
# BPM CARD
# ============================================================

def get_bpm_status(bpm):
    """
    Return display category for BPM.
    """

    if bpm is None:
        return "WAITING"

    bpm = float(bpm)

    if bpm < 60:
        return "LOW"

    if bpm <= 100:
        return "NORMAL"

    return "HIGH"


def render_bpm(target_box):
    """
    Render the Current Heart Rate card.
    """

    # --------------------------------------------------------
    # LOCKED BPM
    # --------------------------------------------------------

    if (
        st.session_state.bpm_locked
        and
        st.session_state.locked_bpm is not None
    ):

        bpm = float(
            st.session_state.locked_bpm
        )

        status = get_bpm_status(
            bpm
        )

        bpm_text = f"{bpm:.1f}"

        badge_class = (
            f"badge {status.lower()}"
        )

        badge_text = (
            f"🔒 LOCKED • {status}"
        )

    # --------------------------------------------------------
    # RAW / STABILIZING BPM
    # --------------------------------------------------------

    elif (
        st.session_state.raw_bpm
        is not None
    ):

        bpm = float(
            st.session_state.raw_bpm
        )

        status = get_bpm_status(
            bpm
        )

        bpm_text = f"{bpm:.1f}"

        badge_class = (
            "badge waiting"
        )

        badge_text = (
            f"● STABILIZING • {status}"
        )

    # --------------------------------------------------------
    # NO BPM YET
    # --------------------------------------------------------

    else:

        bpm_text = "--"

        badge_class = (
            "badge waiting"
        )

        if st.session_state.running:

            badge_text = (
                "● MEASURING"
            )

        else:

            badge_text = (
                "● READY"
            )

    # --------------------------------------------------------
    # IMPORTANT:
    # HTML is intentionally constructed without indentation.
    # This prevents Streamlit from displaying <div> as text.
    # --------------------------------------------------------

    html = (
        '<div class="bpm-card">'
        f'<div class="bpm-number">{bpm_text}</div>'
        '<div class="bpm-unit">'
        'BEATS PER MINUTE'
        '</div>'
        f'<span class="{badge_class}">'
        f'{badge_text}'
        '</span>'
        '</div>'
    )

    target_box.markdown(
        html,
        unsafe_allow_html=True
    )

    if st.session_state.bpm_locked:
        wellness_note = (
            "Your estimate is ready. Sit comfortably and use this "
            "reading as general information, not a medical diagnosis."
        )
    elif st.session_state.running:
        wellness_note = (
            "Stay relaxed, keep your face centred, and use steady "
            "lighting while your estimate stabilises."
        )
    else:
        wellness_note = (
            "Ready when you are. Sit comfortably and look towards "
            "the camera for a contactless estimate."
        )

    target_box.markdown(
        f'<div style="margin-top:10px;padding:12px 15px;'
        f'border:1px solid {theme["border"]};'
        f'border-radius:14px;background:{theme["surface"]};'
        f'color:{theme["muted"]};font-size:0.92rem;line-height:1.5;">'
        f'{wellness_note}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### My Account")
    st.write(f"**{st.session_state.user_name}**")
    st.caption(f"Role: {st.session_state.user_role.title()}")

    if st.button("Log out", key="logout_button", use_container_width=True):
        st.session_state.running = False
        close_camera()
        close_detector()
        st.session_state.authenticated_user = None
        st.session_state.pop("user_id", None)
        st.session_state.pop("user_role", None)
        st.session_state.pop("user_name", None)
        st.rerun()

    st.divider()

    st.markdown("### Appearance")

    st.selectbox(
        "Choose a theme",
        options=list(THEMES.keys()),
        key="selected_theme",
        help="Change the dashboard colour palette."
    )

    st.markdown(
        "## Heart Rate AI"
    )

    st.caption(
        "Contactless heart-rate estimation "
        "using facial video and rPPG."
    )

    st.divider()

    st.markdown(
        "### Measurement Settings"
    )

    required_samples = st.slider(
        "Required samples",
        MIN_SAMPLES,
        MAX_SAMPLES,
        MIN_SAMPLES,
        20,
        help=(
            "BPM estimation begins after this "
            "number of samples."
        )
    )

    show_roi = st.toggle(
        "Show forehead ROI",
        True
    )

    show_signal = st.toggle(
        "Show live signal",
        True
    )

    st.divider()

    st.markdown(
        "### Stability Settings"
    )

    st.caption(
        f"Candidates required: {LOCK_WINDOW}"
    )

    st.caption(
        f"Lock deviation: ≤ "
        f"{LOCK_STD_THRESHOLD:.1f} BPM"
    )

    st.caption(
        f"Jump rejection: "
        f">{MAX_BPM_JUMP:.0f} BPM"
    )

    st.divider()

    st.info(
        """
**For better readings**

• Keep your head still  
• Look directly at the camera  
• Keep hands away from your face  
• Use steady front lighting  
• Avoid strong backlighting  
• Wait until the BPM is locked
"""
    )




# ============================================================
# ADMIN DASHBOARD
# ============================================================

if st.session_state.user_role == "admin":
    st.title("Heart Rate AI — Admin Dashboard")
    st.caption("Administration · Registered users · Measurement records")

    users_tab, measurements_tab = st.tabs(
        ["Registered Users", "All Measurements"]
    )

    with users_tab:
        try:
            users = database.get_all_users()
            if users:
                users_df = pd.DataFrame(users)
                st.metric("Registered accounts", len(users_df))
                st.dataframe(
                    users_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No registered users found.")
        except Exception as exc:
            st.error(f"Could not load registered users: {exc}")

    with measurements_tab:
        try:
            measurements = database.get_all_measurements(limit=500)
            if measurements:
                measurements_df = pd.DataFrame(measurements)
                st.metric("Records shown", len(measurements_df))
                st.dataframe(
                    measurements_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No measurement records found.")
        except Exception as exc:
            st.error(f"Could not load measurement records: {exc}")

    st.stop()


# ============================================================

# ============================================================
# PERSONALIZED USER OVERVIEW
# ============================================================

# This section is reached only by regular users because the
# admin dashboard above ends with st.stop().

st.markdown("""

/* ==========================================================
   HEART RATE AI — USER DASHBOARD UI POLISH
   Presentation only; measurement logic remains unchanged.
   ========================================================== */

<style>
/* Main page spacing */
.main .block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}

/* Cleaner typography */
h1, h2, h3 {
    letter-spacing: -0.035em;
    font-weight: 750 !important;
}

/* Wellness and diagnostic metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        rgba(14, 165, 183, 0.13),
        rgba(56, 217, 197, 0.07)
    );
    border: 1px solid rgba(14, 165, 183, 0.22);
    border-radius: 17px;
    padding: 19px 20px;
    min-height: 100px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(14, 165, 183, 0.48);
    transform: translateY(-2px);
}

div[data-testid="stMetricLabel"] {
    font-weight: 600;
}

/* Buttons: readable text and consistent sizing */
.stButton > button,
.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 11px !important;
    min-height: 43px;
    font-weight: 650 !important;
    transition: all 0.2s ease;
}

.stButton > button p,
.stDownloadButton > button p,
div[data-testid="stFormSubmitButton"] > button p {
    font-weight: 650 !important;
}

/* Input controls */
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    border-radius: 10px;
}

/* Informational banners */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Expanders */
div[data-testid="stExpander"] {
    border-radius: 13px;
    overflow: hidden;
}

/* Data tables */
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(100, 130, 150, 0.18);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    letter-spacing: -0.02em;
}

/* User welcome header */
.user-overview-title {
    font-size: clamp(1.7rem, 2.7vw, 2.25rem);
    font-weight: 800;
    letter-spacing: -0.045em;
    line-height: 1.2;
    margin-bottom: 0.35rem;
}

.user-overview-subtitle {
    font-size: 1rem;
    opacity: 0.78;
    margin-bottom: 1.2rem;
}

/* Responsive spacing */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    div[data-testid="stMetric"] {
        padding: 14px;
        min-height: 85px;
    }
}
</style>

""", unsafe_allow_html=True)

display_name = (
    str(st.session_state.get("user_name", "there")).strip()
    or "there"
)

st.markdown(
    f'<div class="user-overview-title">'
    f'Welcome, {display_name} 👋</div>'
    f'<div class="user-overview-subtitle">'
    f'Your personal heart health monitoring space.</div>',
    unsafe_allow_html=True,
)

# Retrieve only the authenticated user's saved measurements.
try:
    overview_records = database.get_user_measurements(
        st.session_state.user_id,
        limit=100,
    )
except Exception as exc:
    overview_records = []
    st.warning(f"Unable to load your saved measurement summary: {exc}")

overview_df = pd.DataFrame(overview_records)

def find_column(dataframe, candidates):
    """Find a matching database column without assuming its casing."""
    if dataframe.empty:
        return None

    lookup = {str(column).strip().lower(): column
              for column in dataframe.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None

bpm_column = find_column(
    overview_df,
    ["bpm", "heart_rate", "heart rate", "heart_rate_bpm",
     "estimated_bpm", "pulse", "value"],
)

quality_column = find_column(
    overview_df,
    ["signal_quality", "quality", "signal quality",
     "signal_quality_percent", "quality_percent"],
)

date_column = find_column(
    overview_df,
    ["created_at", "measured_at", "measurement_date",
     "timestamp", "date", "created", "recorded_at"],
)

latest_bpm = "—"
latest_quality = "—"
latest_date = "No measurements yet"

if not overview_df.empty:
    # Database history is expected to be returned newest first.
    # Use the first record as the latest saved measurement.
    latest_record = overview_df.iloc[0]

    if bpm_column is not None:
        try:
            value = pd.to_numeric(
                latest_record[bpm_column], errors="coerce"
            )
            if pd.notna(value):
                latest_bpm = f"{float(value):.1f} BPM"
        except (TypeError, ValueError):
            pass

    if quality_column is not None:
        try:
            value = pd.to_numeric(
                latest_record[quality_column], errors="coerce"
            )
            if pd.notna(value):
                latest_quality = f"{float(value):g}%"
        except (TypeError, ValueError):
            pass

    if date_column is not None:
        date_value = latest_record[date_column]
        if pd.notna(date_value):
            latest_date = str(date_value)

st.markdown("#### Your wellness at a glance")

card1, card2, card3 = st.columns(3)

with card1:
    st.metric(
        "Latest Heart Rate",
        latest_bpm,
        help="Your latest saved measurement, if available.",
    )

with card2:
    st.metric(
        "Saved Measurements",
        len(overview_df),
        help="Number of saved records returned for your account.",
    )

with card3:
    st.metric(
        "Latest Signal Quality",
        latest_quality,
        help="Quality of the latest saved measurement, if recorded.",
    )

with st.expander("View latest measurement details"):
    st.write(f"**Last recorded:** {latest_date}")

    if overview_df.empty:
        st.info(
            "Your saved measurements will appear here after "
            "you complete and save your first measurement."
        )
    else:
        st.caption(
            "This overview displays saved records associated "
            "with your account. It is not a medical diagnosis."
        )

st.divider()


# WELCOME BANNER
# ============================================================

st.markdown(
    f"""
    <div style="
        background: linear-gradient(120deg, {theme['surface']}, {theme['bpm_bg']});
        border: 1px solid {theme['border']};
        border-radius: 22px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.2rem;
    ">
        <div style="
            color: {theme['primary']};
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        ">YOUR PERSONAL WELLNESS SPACE</div>
        <div style="
            color: {theme['text']};
            font-size: clamp(1.5rem, 3vw, 2.2rem);
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.55rem;
        ">Let's check your pulse.</div>
        <div style="
            color: {theme['muted']};
            font-size: 1rem;
            line-height: 1.65;
            max-width: 760px;
        ">Get comfortable, look towards your camera, and let Heart Rate AI
        analyse subtle changes in your facial video to estimate your heart rate.</div>
    </div>
    """,
    unsafe_allow_html=True
)

step1, step2, step3 = st.columns(3, gap="medium")

welcome_steps = [
    ("01", "Get comfortable", "Face the camera in steady, front-facing light."),
    ("02", "Start measuring", "Stay still while the app collects a video signal."),
    ("03", "View your estimate", "Wait for the reading to stabilise before reviewing it."),
]

for column, (number, title, description) in zip(
    (step1, step2, step3), welcome_steps
):
    with column:
        st.markdown(
            f"""
            <div style="
                background: {theme['surface']};
                border: 1px solid {theme['border']};
                border-radius: 16px;
                padding: 1rem;
                min-height: 120px;
                margin-bottom: 1rem;
            ">
                <div style="
                    color: {theme['accent']};
                    font-size: 0.78rem;
                    font-weight: 800;
                    letter-spacing: 0.08em;
                    margin-bottom: 0.45rem;
                ">STEP {number}</div>
                <div style="
                    color: {theme['text']};
                    font-size: 1rem;
                    font-weight: 750;
                    margin-bottom: 0.4rem;
                ">{title}</div>
                <div style="
                    color: {theme['muted']};
                    font-size: 0.88rem;
                    line-height: 1.5;
                ">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = (
    st.columns(
        [5, 1]
    )
)

with header_left:

    st.markdown(
        '<div class="main-title">'
        'Heart Rate AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Real-Time Contactless Cardiovascular Monitoring'
        '</div>',
        unsafe_allow_html=True
    )


with header_right:

    st.markdown(
        '<div class="system-ready">'
        '● SYSTEM READY'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# CONTROL BUTTONS
# ============================================================

button1, button2, button3, _ = (
    st.columns(
        [1, 1, 1, 3]
    )
)


with button1:

    start = st.button(
        "Start",
        type="primary",
        width="stretch"
    )


with button2:

    stop = st.button(
        "Stop",
        width="stretch"
    )


with button3:

    reset = st.button(
        "New Measurement",
        width="stretch"
    )


# ============================================================
# BUTTON ACTIONS
# ============================================================

if reset:

    st.session_state.running = False

    close_camera()

    close_detector()

    reset_measurement()


if stop:

    st.session_state.running = False

    close_camera()

    close_detector()


if (
    start
    and
    not st.session_state.running
):

    close_camera()

    close_detector()

    reset_measurement()
    st.session_state.measurement_saved_for_run = False

    camera = Camera()

    detector = FaceDetector()

    try:

        opened = camera.open()

    except Exception as e:

        opened = False

        st.error(
            f"Camera error: {e}"
        )

    if opened:

        st.session_state.camera = (
            camera
        )

        st.session_state.detector = (
            detector
        )

        st.session_state.running = True

        st.session_state.session_start = (
            time.time()
        )

        st.session_state.last_face_time = (
            time.perf_counter()
        )

    else:

        st.error(
            "Could not open camera."
        )

        try:

            detector.close()

        except Exception:

            pass


# ============================================================
# MAIN DASHBOARD LAYOUT
# ============================================================

left, right = st.columns(
    [1.7, 1],
    gap="medium"
)


# ============================================================
# CAMERA COLUMN
# ============================================================

with left:

    st.markdown(
        "### Live Camera"
    )

    camera_box = st.empty()

    camera_status = st.empty()


# ============================================================
# BPM COLUMN
# ============================================================

with right:

    st.markdown(
        "### Current Heart Rate"
    )

    bpm_box = st.empty()

    measurement_status_box = (
        st.empty()
    )

    quality_box = st.empty()

    progress_text = st.empty()

    progress_bar = st.empty()


render_bpm(
    bpm_box
)


# ============================================================
# SIGNAL GRAPH
# ============================================================

st.divider()

st.markdown(
    "### Live Pulse Signal"
)

graph_box = st.empty()


# ============================================================
# DIAGNOSTICS
# ============================================================

st.markdown(
    "### Measurement Diagnostics"
)

d1, d2, d3, d4 = st.columns(4)

fps_box = d1.empty()

samples_box = d2.empty()

candidate_box = d3.empty()

raw_box = d4.empty()


# ============================================================
# SESSION STATISTICS
# ============================================================

st.markdown(
    "### Session Statistics"
)

m1, m2, m3, m4 = (
    st.columns(4)
)

avg_box = m1.empty()

min_box = m2.empty()

max_box = m3.empty()

time_box = m4.empty()


# ============================================================
# HISTORY
# ============================================================

st.divider()

st.markdown(
    "### 📝 Measurement History"
)

history_box = st.empty()


# ============================================================
# UPDATE STATISTICS
# ============================================================

def update_statistics():

    values = (
        st.session_state.all_bpm
    )

    if values:

        avg_box.metric(
            "Average BPM",
            f"{np.mean(values):.1f}"
        )

        min_box.metric(
            "Minimum BPM",
            f"{np.min(values):.1f}"
        )

        max_box.metric(
            "Maximum BPM",
            f"{np.max(values):.1f}"
        )

    else:

        avg_box.metric(
            "Average BPM",
            "—"
        )

        min_box.metric(
            "Minimum BPM",
            "—"
        )

        max_box.metric(
            "Maximum BPM",
            "—"
        )

    if (
        st.session_state.session_start
        is not None
    ):

        duration = int(
            time.time()
            -
            st.session_state.session_start
        )

    else:

        duration = 0

    time_box.metric(
        "Session",
        f"{duration} sec"
    )


# ============================================================
# UPDATE QUALITY / PROGRESS
# ============================================================

def update_quality():

    count = len(
        st.session_state.signal_buffer
    )

    quality = (
        calculate_quality()
    )

    if quality >= 80:

        label = "GOOD"

    elif quality >= 50:

        label = "FAIR"

    else:

        label = "LOW"

    quality_box.markdown(
        f"**Signal Quality:** "
        f"{label} • {quality}%"
    )

    # Once minimum samples are collected, progress represents
    # stabilization instead of raw sample collection.

    if count < required_samples:

        progress = min(
            count / required_samples,
            1.0
        )

        progress_text.caption(
            f"Signal collection: "
            f"{count}/{required_samples} samples"
        )

    elif not st.session_state.bpm_locked:

        candidate_count = len(
            st.session_state.candidate_bpms
        )

        progress = min(
            candidate_count
            /
            LOCK_WINDOW,
            1.0
        )

        progress_text.caption(
            f"BPM stabilization: "
            f"{candidate_count}/{LOCK_WINDOW} "
            f"accepted estimates"
        )

    else:

        progress = 1.0

        progress_text.caption(
            "✓ Measurement complete"
        )

    progress_bar.progress(
        float(progress)
    )


# ============================================================
# UPDATE GRAPH
# ============================================================

def update_graph():

    if not show_signal:
        graph_box.info(
            "Live signal display is paused. Enable it in the sidebar to view your pulse signal."
        )
        return

    signal = st.session_state.signal_buffer

    if len(signal) < 2:
        graph_box.info(
            "Your pulse signal will appear here once the camera starts collecting data."
        )
        return

    recent_signal = signal[-MAX_SAMPLES:]

    df = pd.DataFrame({
        "Green Intensity": recent_signal
    })

    chart = st.line_chart(
        df,
        height=300,
        width="stretch",
        color=theme["accent"],
        x_label="Recent samples",
        y_label="Green intensity",
    )

# ============================================================
# UPDATE HISTORY
# ============================================================

def update_history():

    try:
        saved_history = database.get_user_measurements(
            st.session_state.user_id,
            limit=100,
        )
    except Exception as exc:
        history_box.error(f"Could not load saved history: {exc}")
        return

    if not saved_history:
        history_box.info(
            "No saved measurements yet. Complete a measurement "
            "and it will appear here for future visits."
        )
        return

    df = pd.DataFrame(saved_history)

    history_box.markdown(
        f"**Your measurement history** · "
        f"Showing {len(df)} saved measurements"
    )

    history_box.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# UPDATE DIAGNOSTICS
# ============================================================

def update_diagnostics():

    fps = (
        st.session_state.actual_fps
    )

    if fps is None:

        fps_value = "—"

    else:

        fps_value = (
            f"{fps:.1f}"
        )

    fps_box.metric(
        "Actual FPS",
        fps_value
    )

    samples_box.metric(
        "Samples",
        len(
            st.session_state.signal_buffer
        )
    )

    candidate_box.metric(
        "Stable Candidates",
        (
            f"{min(len(st.session_state.candidate_bpms), LOCK_WINDOW)}"
            f"/{LOCK_WINDOW}"
        )
    )

    raw = (
        st.session_state.raw_bpm
    )

    if raw is None:

        raw_value = "—"

    else:

        raw_value = (
            f"{raw:.1f}"
        )

    raw_box.metric(
        "Raw BPM",
        raw_value
    )


# ============================================================
# INITIAL DISPLAY
# ============================================================

update_statistics()

update_quality()

update_graph()

update_history()

update_diagnostics()


if not st.session_state.running:

    if st.session_state.bpm_locked:

        camera_box.success(
            "Measurement complete. "
            "Click New Measurement to measure again."
        )

    else:

        camera_box.info(
            "Camera stopped. "
            "Click Start to begin."
        )


# ============================================================
# LIVE MEASUREMENT
# ============================================================

if st.session_state.running:

    camera = (
        st.session_state.camera
    )

    detector = (
        st.session_state.detector
    )

    while st.session_state.running:

        # ====================================================
        # READ CAMERA
        # ====================================================

        success, frame = (
            camera.read()
        )

        if not success:

            time.sleep(
                0.02
            )

            continue

        current_time = (
            time.perf_counter()
        )

        frame = cv2.flip(
            frame,
            1
        )

        # ====================================================
        # FACE DETECTION
        # ====================================================

        try:

            face_landmarks = (
                detector.detect(
                    frame
                )
            )

        except Exception as e:

            face_landmarks = None

            print(
                "Face detection error:",
                e
            )

        roi_found = False

        green_value = None

        # ====================================================
        # FOREHEAD ROI
        # ====================================================

        if face_landmarks is not None:

            roi_result = (
                get_forehead_roi(
                    frame,
                    face_landmarks
                )
            )

            if roi_result is not None:

                (
                    roi,
                    x,
                    y,
                    x2,
                    y2,
                    eye_box
                ) = roi_result

                roi_found = True

                st.session_state.last_face_time = (
                    current_time
                )

                st.session_state.face_was_lost = (
                    False
                )

                # ============================================
                # IRIS DETECTION AND APPROXIMATE COLOUR ESTIMATION
                # ============================================
                landmarks = face_landmarks.landmark
                frame_h, frame_w = frame.shape[:2]

                iris_groups = {
                    "LEFT IRIS": [474, 475, 476, 477],
                    "RIGHT IRIS": [469, 470, 471, 472],
                }

                # Put labels in separate rows above the eyes.
                label_rows = {
                    "LEFT IRIS": 28,
                    "RIGHT IRIS": 48,
                }

                for eye_name, indices in iris_groups.items():
                    try:
                        points = np.asarray([
                            (
                                int(landmarks[idx].x * frame_w),
                                int(landmarks[idx].y * frame_h)
                            )
                            for idx in indices
                        ], dtype=np.int32)

                        cx = int(np.mean(points[:, 0]))
                        cy = int(np.mean(points[:, 1]))

                        # Use a conservative iris sample to reduce pupil/skin contamination.
                        distances = np.linalg.norm(
                            points - np.array([cx, cy]), axis=1
                        )
                        radius = max(2, int(np.max(distances) * 0.65))

                        x1 = max(0, cx - radius)
                        y1 = max(0, cy - radius)
                        x2 = min(frame_w, cx + radius + 1)
                        y2 = min(frame_h, cy + radius + 1)

                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        yy, xx = np.ogrid[y1:y2, x1:x2]
                        circle_mask = (
                            (xx - cx) ** 2 + (yy - cy) ** 2
                            <= radius ** 2
                        )

                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        pixels = hsv[circle_mask]

                        # Exclude dark pupil pixels, reflections and weak colour.
                        pixels = pixels[
                            (pixels[:, 2] >= 45) &
                            (pixels[:, 2] <= 220) &
                            (pixels[:, 1] >= 30)
                        ]

                        colour = "UNCLEAR"
                        if len(pixels) >= 5:
                            hue = float(np.median(pixels[:, 0]))
                            sat = float(np.median(pixels[:, 1]))

                            if sat < 35:
                                colour = "UNCLEAR"
                            elif hue < 10 or hue >= 170:
                                colour = "BROWN/AMBER"
                            elif hue < 20:
                                colour = "BROWN"
                            elif hue < 35:
                                colour = "HAZEL/BROWN"
                            elif hue < 85:
                                colour = "GREENISH"
                            elif hue < 130:
                                colour = "BLUISH"
                            else:
                                colour = "UNCLEAR"

                        # Stabilise each eye independently over recent valid frames.
                        history = st.session_state.iris_colour_history[eye_name]
                        if colour != "UNCLEAR":
                            history.append(colour)
                            del history[:-15]

                        if history:
                            counts = {name: history.count(name) for name in set(history)}
                            colour = max(counts, key=counts.get)
                        else:
                            colour = "UNCLEAR"

                        if show_roi:
                            cv2.circle(
                                frame, (cx, cy), radius,
                                (255, 255, 0), 2
                            )

                            # Update each eye's label, but draw the panel only once.
                            display_colours = st.session_state.setdefault(
                                "iris_display_colours",
                                {"LEFT IRIS": "DETECTING", "RIGHT IRIS": "DETECTING"}
                            )
                            display_colours[eye_name] = colour

                            if eye_name == "RIGHT IRIS":
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                scale = 0.46
                                thickness = 1
                                rows = [
                                    f"LEFT IRIS: {display_colours['LEFT IRIS']}",
                                    f"RIGHT IRIS: {display_colours['RIGHT IRIS']}"
                                ]
                                sizes = [
                                    cv2.getTextSize(row, font, scale, thickness)[0]
                                    for row in rows
                                ]
                                panel_w = max(size[0] for size in sizes) + 18
                                px = max(8, frame_w - panel_w - 12)
                                py = 12
                                cv2.rectangle(
                                    frame, (px, py), (px + panel_w, py + 58),
                                    (20, 20, 20), -1
                                )
                                cv2.rectangle(
                                    frame, (px, py), (px + panel_w, py + 58),
                                    (255, 255, 0), 1
                                )
                                for i, row in enumerate(rows):
                                    cv2.putText(
                                        frame, row, (px + 8, py + 23 + i * 22),
                                        font, scale, (255, 255, 0),
                                        thickness, cv2.LINE_AA
                                    )

                    except (IndexError, ValueError, cv2.error):
                        continue

                # DRAW FOREHEAD ROI
                # ============================================

                if show_roi:

                    ex, ey, ex2, ey2 = eye_box
                    cv2.rectangle(
                        frame,
                        (ex, ey),
                        (ex2, ey2),
                        (255, 0, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        "EYES DETECTED",
                        (ex, min(frame.shape[0] - 10, ey2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )

                    # Wider, centred forehead DISPLAY box only.
                    frame_h, frame_w = frame.shape[:2]
                    eye_width = max(1, ex2 - ex)
                    forehead_left = max(0, ex - int(0.12 * eye_width))
                    forehead_right = min(frame_w - 1, ex2 + int(0.12 * eye_width))
                    forehead_top = max(0, ey - int(0.85 * eye_width))
                    forehead_bottom = max(forehead_top + 1, ey - 8)

                    cv2.rectangle(
                        frame,
                        (forehead_left, forehead_top),
                        (forehead_right, forehead_bottom),
                        (0, 255, 0),
                        2
                    )

                    forehead_label = "FOREHEAD ROI"
                    (label_w, label_h), _ = cv2.getTextSize(
                        forehead_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                    )
                    forehead_label_x = max(
                        5, min(frame_w - label_w - 5,
                               (forehead_left + forehead_right - label_w) // 2)
                    )
                    forehead_label_y = max(22, forehead_top - 8)

                    cv2.putText(
                        frame,
                        forehead_label,
                        (forehead_label_x, forehead_label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                # ============================================
                # GREEN CHANNEL
                # ============================================

                green_value = (
                    extract_green_value(
                        roi
                    )
                )

                rgb_value = extract_rgb_values(roi)

                # ============================================
                # COLLECT SIGNAL ONLY UNTIL BPM LOCKS
                # ============================================

                if (
                    green_value is not None
                    and
                    not st.session_state.bpm_locked
                ):

                    st.session_state.signal_buffer.append(
                        green_value
                    )

                    st.session_state.rgb_buffer.append(rgb_value)

                    st.session_state.timestamp_buffer.append(
                        current_time
                    )

                    if (
                        len(
                            st.session_state.signal_buffer
                        )
                        >
                        MAX_SAMPLES
                    ):

                        st.session_state.signal_buffer.pop(
                            0
                        )

                        st.session_state.timestamp_buffer.pop(0)

                        if st.session_state.rgb_buffer:
                            st.session_state.rgb_buffer.pop(0)

        # ====================================================
        # FACE LOST HANDLING
        # ====================================================

        if (
            not roi_found
            and
            not st.session_state.bpm_locked
        ):

            lost_duration = (
                current_time
                -
                st.session_state.last_face_time
            )

            if (
                lost_duration
                >
                FACE_LOST_RESET_SECONDS
                and
                not st.session_state.face_was_lost
            ):

                print(
                    "Face lost too long. "
                    "Signal measurement reset."
                )

                st.session_state.signal_buffer = []

                st.session_state.timestamp_buffer = []

                st.session_state.rgb_buffer = []

                st.session_state.candidate_bpms = []

                st.session_state.raw_bpm = None

                st.session_state.actual_fps = None

                st.session_state.last_bpm_time = 0.0

                st.session_state.face_was_lost = (
                    True
                )

        # ====================================================
        # ACTUAL TIMESTAMP FPS
        # ====================================================

        if not st.session_state.bpm_locked:

            st.session_state.actual_fps = (
                calculate_actual_fps(
                    st.session_state.timestamp_buffer
                )
            )

        # ====================================================
        # DRAW CAMERA INFORMATION
        # ====================================================

        sample_count = len(
            st.session_state.signal_buffer
        )

        cv2.putText(
            frame,
            (
                f"Samples: "
                f"{sample_count}/"
                f"{required_samples}"
            ),
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Smooth displayed parameters only; keep raw measurements unchanged.
        if "display_fps_smooth" not in st.session_state:
            st.session_state.display_fps_smooth = None
        if "display_green_smooth" not in st.session_state:
            st.session_state.display_green_smooth = None

        actual_fps = st.session_state.actual_fps
        if actual_fps is not None:
            previous_fps = st.session_state.display_fps_smooth
            st.session_state.display_fps_smooth = (
                actual_fps if previous_fps is None
                else 0.15 * actual_fps + 0.85 * previous_fps
            )
            actual_fps = st.session_state.display_fps_smooth

        if green_value is not None:
            previous_green = st.session_state.display_green_smooth
            st.session_state.display_green_smooth = (
                green_value if previous_green is None
                else 0.15 * green_value + 0.85 * previous_green
            )
            green_value = st.session_state.display_green_smooth

        if actual_fps is not None:

            cv2.putText(
                frame,
                (
                    f"FPS: "
                    f"{actual_fps:.1f}"
                ),
                (25, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 0),
                2
            )

        if green_value is not None:

            cv2.putText(
                frame,
                (
                    f"Green: "
                    f"{green_value:.2f}"
                ),
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        # ====================================================
        # BPM ESTIMATION
        # ====================================================

        now = time.time()

        if (
            not st.session_state.bpm_locked
            and
            len(
                st.session_state.signal_buffer
            )
            >=
            required_samples
            and
            st.session_state.actual_fps
            is not None
            and
            (
                now
                -
                st.session_state.last_bpm_time
            )
            >=
            BPM_UPDATE_SECONDS
        ):

            # ================================================
            # ESTIMATE RAW BPM
            # ================================================

            try:

                estimated_bpm = (
                    estimate_bpm(
                        st.session_state.signal_buffer,
                        st.session_state.actual_fps
                    )
                )

                # Experimental CHROM comparison; green remains the live estimate.
                chrom_bpm = estimate_chrom_bpm(
                    st.session_state.rgb_buffer,
                    st.session_state.actual_fps
                )

                if chrom_bpm is not None:
                    print(
                        f"EXPERIMENTAL CHROM BPM={chrom_bpm:.1f} | "
                        f"GREEN BPM={estimated_bpm}"
                    )

            except Exception as e:

                print(
                    "estimate_bpm ERROR:",
                    repr(e)
                )

                estimated_bpm = None

            # ================================================
            # DIAGNOSTICS
            # ================================================

            arr = np.asarray(
                st.session_state.signal_buffer,
                dtype=np.float64
            )

            if estimated_bpm is None:

                st.session_state.raw_bpm = (
                    None
                )

                print(
                    "Estimator returned None | "
                    f"FPS="
                    f"{st.session_state.actual_fps:.2f} | "
                    f"Samples={len(arr)} | "
                    f"Mean={np.mean(arr):.2f} | "
                    f"STD={np.std(arr):.4f} | "
                    f"Range={np.ptp(arr):.4f}"
                )

            else:

                raw_bpm = float(
                    estimated_bpm
                )

                st.session_state.raw_bpm = (
                    raw_bpm
                )

                print(
                    f"Raw BPM={raw_bpm:.1f} | "
                    f"FPS="
                    f"{st.session_state.actual_fps:.2f} | "
                    f"Samples={len(arr)} | "
                    f"STD={np.std(arr):.4f} | "
                    f"Range={np.ptp(arr):.4f}"
                )

                # ============================================
                # ADD CANDIDATE
                # ============================================

                accepted = (
                    add_bpm_candidate(
                        raw_bpm
                    )
                )

                if accepted:

                    print(
                        "Accepted BPM | "
                        f"Candidates="
                        f"{len(st.session_state.candidate_bpms)}"
                    )

                    # ========================================
                    # CHECK LOCK
                    # ========================================

                    possible_lock = (
                        check_bpm_lock()
                    )

                    if possible_lock is not None:

                        st.session_state.locked_bpm = (
                            possible_lock
                        )

                        st.session_state.current_bpm = (
                            possible_lock
                        )

                        st.session_state.bpm_locked = (
                            True
                        )

                        # Store one final BPM measurement
                        st.session_state.all_bpm.append(
                            possible_lock
                        )

                        st.session_state.history.append(
                            {
                                "Time":
                                    time.strftime(
                                        "%H:%M:%S"
                                    ),

                                "BPM":
                                    round(
                                        possible_lock,
                                        1
                                    ),

                                "Signal Quality":
                                    f"{calculate_quality()}%",

                                "Status":
                                    (
                                        "LOW"
                                        if possible_lock < 60
                                        else
                                        "NORMAL"
                                        if possible_lock <= 100
                                        else
                                        "HIGH"
                                    )
                            }
                        )

                        # Persist the completed measurement for this account.
                        quality = f"{calculate_quality()}%"
                        status = (
                            "LOW" if possible_lock < 60 else
                            "NORMAL" if possible_lock <= 100 else
                            "HIGH"
                        )
                        if not st.session_state.get("measurement_saved_for_run", False):
                            database.save_measurement(
                                st.session_state.user_id,
                                possible_lock,
                                quality,
                                status,
                            )
                            st.session_state.measurement_saved_for_run = True

                        print()

                        print(
                            "=" * 55
                        )

                        print(
                            "MEASUREMENT COMPLETE"
                        )

                        print(
                            "FINAL LOCKED BPM: "
                            f"{possible_lock:.1f}"
                        )

                        print(
                            "=" * 55
                        )

                        print()

            st.session_state.last_bpm_time = (
                now
            )

        # ====================================================
        # CAMERA OVERLAY - BPM STATE
        # ====================================================

        if st.session_state.bpm_locked:

            locked = (
                st.session_state.locked_bpm
            )

            cv2.putText(
                frame,
                (
                    f"BPM: "
                    f"{locked:.1f}"
                ),
                (25, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "MEASUREMENT COMPLETE",
                (25, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        elif (
            len(
                st.session_state.signal_buffer
            )
            <
            required_samples
        ):

            cv2.putText(
                frame,
                "BPM: Collecting signal...",
                (25, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        elif len(
            st.session_state.candidate_bpms
        ) == 0:

            cv2.putText(
                frame,
                "BPM: Analyzing...",
                (25, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        else:

            candidate_progress = min(
                len(
                    st.session_state.candidate_bpms
                ),
                LOCK_WINDOW
            )

            cv2.putText(
                frame,
                (
                    "BPM: Stabilizing "
                    f"{candidate_progress}/"
                    f"{LOCK_WINDOW}"
                ),
                (25, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # ====================================================
        # DISPLAY CAMERA
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        camera_box.image(
            rgb_frame,
            channels="RGB",
            width="stretch"
        )

        # ====================================================
        # CAMERA STATUS
        # ====================================================

        if st.session_state.bpm_locked:

            camera_status.markdown(
                '<div class="complete">'
                '✓ All set! Measurement complete • '
                'Your heart-rate estimate is ready'
                '</div>',
                unsafe_allow_html=True
            )

        elif roi_found:

            camera_status.markdown(
                '<div class="detected">'
                '✓ Face detected • '
                'Collecting your pulse signal'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            camera_status.markdown(
                '<div class="not-detected">'
                'Face not detected yet • '
                'Centre your face in the camera view'
                '</div>',
                unsafe_allow_html=True
            )

        # ====================================================
        # MEASUREMENT STATUS
        # ====================================================

        if st.session_state.bpm_locked:

            measurement_status_box.markdown(
                '<div class="complete">'
                '✓ YOU DID IT!<br>'
                'Your heart-rate estimate is ready to view.'
                '</div>',
                unsafe_allow_html=True
            )

        elif len(
            st.session_state.signal_buffer
        ) < required_samples:

            measurement_status_box.markdown(
                '<div class="stabilizing">'
                'Getting your pulse signal ready... Stay relaxed.'
                '</div>',
                unsafe_allow_html=True
            )

        elif len(
            st.session_state.candidate_bpms
        ) < LOCK_WINDOW:

            count = len(
                st.session_state.candidate_bpms
            )

            measurement_status_box.markdown(
                (
                    '<div class="stabilizing">'
                    'Steadying your reading... '
                    f'{count}/{LOCK_WINDOW}'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

        # ====================================================
        # UPDATE BPM CARD
        # ====================================================

        render_bpm(
            bpm_box
        )

        update_quality()

        update_statistics()

        update_diagnostics()

        # ====================================================
        # GRAPH / HISTORY
        # ====================================================

        if (
            now
            -
            st.session_state.last_graph_time
            >=
            GRAPH_UPDATE_SECONDS
        ):

            update_graph()

            update_history()

            st.session_state.last_graph_time = (
                now
            )

        # ====================================================
        # CPU CONTROL
        # ====================================================

        time.sleep(
            0.02
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Heart Rate AI • Contactless rPPG Monitoring • "
    "Educational and Research Prototype"
)