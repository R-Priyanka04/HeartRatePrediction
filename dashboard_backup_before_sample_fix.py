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
    page_title="PulseVision",
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
    "Light": {
        "page": "#f4f7fb", "sidebar": "#ffffff", "surface": "#ffffff",
        "text": "#202940", "muted": "#65718a", "primary": "#536dfe",
        "accent": "#8b9cff", "border": "#dfe5f0", "bpm_bg": "#edf0ff",
        "bpm": "#4058db", "info_bg": "#e8efff", "info_text": "#304b91",
    },
    "Dark": {
        "page": "#000000", "sidebar": "#080808", "surface": "#111111",
        "text": "#FFFFFF", "muted": "#C0C0C0", "primary": "#20DCC8",
        "accent": "#40F5DF", "border": "#303030", "bpm_bg": "#151515",
        "bpm": "#40F5DF", "info_bg": "#102725", "info_text": "#BFFFF5",
    },
}

if st.session_state.get("selected_theme") not in THEMES:
    st.session_state.selected_theme = "Light"

selected_theme = st.session_state.selected_theme
theme = THEMES[selected_theme]

st.markdown(
    f"""
<style>
:root {{
    color-scheme: {"dark" if selected_theme == "Dark" else "light"};
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


/* Heart Rate AI friendly interactions */
.stButton > button {{
    min-height: 48px;
    padding: 0.65rem 1.15rem;
    border-radius: 14px !important;
    font-weight: 700 !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(65, 120, 150, .16);
}}
.stButton > button:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="combobox"]:focus-visible {{
    outline: 3px solid #ffbf69 !important;
    outline-offset: 3px !important;
}}
.stTextInput input,
.stNumberInput input,
.stSelectbox [role="combobox"] {{
    min-height: 44px;
    border-radius: 12px !important;
}}
[data-testid="stMetric"] {{
    border-radius: 20px;
    padding: 20px;
    transition: transform .2s ease, box-shadow .2s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(50, 90, 120, .10);
}}
.stApp p,
.stApp li {{
    line-height: 1.65;
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
        background: #000000 !important;
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
            <div class="hr-brand">Pulse<span>Vision</span></div>
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
            'PULSEVISION · INTELLIGENCE FOR YOUR WELLNESS'
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

    st.markdown("### Appearance")

    st.selectbox(
        "Choose appearance",
        options=list(THEMES.keys()),
        key="selected_theme",
        help="Change the dashboard colour palette."
    )

    # Show navigation and measurement controls to regular users only.
    if st.session_state.user_role != "admin":
        st.divider()

        st.radio(
            "NAVIGATION",
            options=[
                "Dashboard",
                "Live Measurement",
                "Measurement History",
                "Settings",
                "Wellness Arcade",
            ],
            key="user_navigation",
            label_visibility="visible",
        )

        st.divider()

        st.markdown("### Measurement Settings")

        required_samples = st.slider(
            "Required samples",
            MIN_SAMPLES,
            MAX_SAMPLES,
            MIN_SAMPLES,
            20,
            help="BPM estimation begins after this number of samples.",
        )

        show_roi = st.toggle("Show forehead ROI", True)
        show_signal = st.toggle("Show live signal", True)

        st.divider()

        st.info(
            """
**For better readings**

• Keep your head still  
• Face the camera in steady lighting  
• Avoid sudden movements while measuring
"""
        )

        st.divider()

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






# ============================================================
selected_page = st.session_state.get(
    "user_navigation", "Dashboard"
)


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

st.markdown("""
<style>
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}

h1, h2, h3 {
    letter-spacing: -0.03em;
    font-weight: 750;
}

div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        rgba(14, 165, 183, 0.13),
        rgba(56, 217, 197, 0.07)
    );
    border: 1px solid rgba(14, 165, 183, 0.25);
    border-radius: 17px;
    padding: 18px 20px;
}

.stButton > button {
    border-radius: 10px !important;
    min-height: 42px;
    font-weight: 650 !important;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stExpander"] {
    border-radius: 12px;
    overflow: hidden;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(100, 130, 150, 0.18);
}

.user-overview-title {
    font-size: clamp(1.7rem, 2.7vw, 2.25rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.2;
    margin-bottom: 0.35rem;
}

.user-overview-subtitle {
    font-size: 1rem;
    opacity: 0.8;
    margin-bottom: 1.2rem;
}

@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    div[data-testid="stMetric"] {
        padding: 14px;
    }
}
</style>
""", unsafe_allow_html=True)

# NAVIGATION: HISTORY AND SETTINGS
# ============================================================

if st.session_state.user_role != "admin":
    if selected_page == "Measurement History":
        st.title("�� Measurement History")
        st.caption("Your saved heart-rate measurements")

        try:
            history_records = database.get_user_measurements(
                st.session_state.user_id,
                limit=100,
            )

            if history_records:
                history_df = pd.DataFrame(history_records)
                st.metric("Saved measurements", len(history_df))
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No saved measurements yet. "
                    "Complete a measurement and save it to see it here."
                )
        except Exception as exc:
            st.error(f"Could not load measurement history: {exc}")

        st.stop()

    elif selected_page == "Settings":
        st.title("⚙️ Settings")
        st.caption("Personalise your Heart Rate AI experience")

        st.subheader("Appearance")
        st.write(
            "Use the **Choose appearance** selector in the sidebar "
            "to change the dashboard colours."
        )

        st.subheader("Measurement preferences")
        st.write(
            "Adjust **Required samples**, **Show forehead ROI**, "
            "and **Show live signal** in the sidebar."
        )

        st.subheader("Account")
        st.write(f"Signed in as: **{st.session_state.user_name}**")
        st.write(f"Role: **{st.session_state.user_role.title()}**")
        st.caption(
            "Use the Log out button in the sidebar to end your session."
        )

        st.stop()



# ============================================================

# ============================================================
# WELLNESS ARCADE
# ============================================================

if selected_page == "Wellness Arcade":
    import random
    import time

    st.title("🎮 Wellness Arcade")
    st.caption("Eight little ways to relax, play and refresh your mind.")

    if "arcade_points" not in st.session_state:
        st.session_state.arcade_points = 0
    if "arcade_best_scores" not in st.session_state:
        st.session_state.arcade_best_scores = {}
    if "arcade_achievements" not in st.session_state:
        st.session_state.arcade_achievements = []

    def arcade_award(game, score):
        best = st.session_state.arcade_best_scores
        best[game] = max(best.get(game, 0), score)
        st.session_state.arcade_points += max(0, score)

        if score > 0:
            badge = f"Played: {game}"
            if badge not in st.session_state.arcade_achievements:
                st.session_state.arcade_achievements.append(badge)

    def arcade_memory_pick(index):
        matched = set(st.session_state.get("memory_matched", []))
        selected = list(st.session_state.get("memory_selected", []))
        cards = st.session_state.memory_cards

        if index in matched or index in selected:
            return

        # Clear the previous unmatched pair when another card is picked.
        if len(selected) >= 2:
            selected = []

        selected.append(index)

        if len(selected) == 2:
            if cards[selected[0]] == cards[selected[1]]:
                matched.update(selected)
                st.session_state.memory_matched = sorted(matched)
                arcade_award("Memory Match", 10)
            else:
                st.session_state.memory_misses = (
                    st.session_state.get("memory_misses", 0) + 1
                )

        st.session_state.memory_selected = selected

    def arcade_bubble_pop():
        if st.session_state.get("arcade_bubble_active", False):
            elapsed = time.time() - st.session_state.arcade_bubble_start
            if elapsed < 30:
                st.session_state.arcade_bubble_score += 1
                arcade_award("Bubble Pop", 1)

    st.markdown("""
    <style>
    .arcade-banner {
        padding: 22px;
        border-radius: 20px;
        margin-bottom: 18px;
        background: linear-gradient(120deg,#dff7ff,#eee6ff);
        color: #17354b;
    }
    .arcade-banner h3 { margin-top: 0; }
    div[data-testid="stMetric"] {
        border-radius: 16px;
        padding: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="arcade-banner">'
        '<h3>✨ Welcome to your Wellness Arcade</h3>'
        '<p>Take a short break, challenge your brain, and have fun.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)
    a.metric("⭐ Total points", st.session_state.arcade_points)
    b.metric("🎮 Games played", len(st.session_state.arcade_achievements))
    c.metric("🏆 Personal bests", len(st.session_state.arcade_best_scores))

    games = st.tabs([
        "🫁 Breathe & Relax",
        "🫧 Bubble Pop",
        "🧠 Memory Match",
        "⚡ Reaction Test",
        "🔢 Number Sequence",
        "🌸 Zen Garden",
        "❤️ Health Quiz",
        "🎨 Zen Doodle",
        "🏆 Achievements",
    ])

    # --------------------------------------------------------
    # 1. BREATHE & RELAX
    # --------------------------------------------------------
    with games[0]:
        st.subheader("🫁 Breathe & Relax")
        st.write(
            "Follow the circle and breathe comfortably. "
            "Do not hold your breath; stop if you feel uncomfortable."
        )

        if st.button("▶ Start breathing session", key="arcade_breathe_start"):
            st.session_state.arcade_breath_start = time.time()
            st.session_state.arcade_breath_active = True

        if st.session_state.get("arcade_breath_active", False):
            elapsed = int(time.time() - st.session_state.arcade_breath_start)
            phase = elapsed % 8
            instruction = (
                "🌬️ Breathe in gently" if phase < 4
                else "🌿 Breathe out slowly"
            )
            st.markdown("""
            <style>
            @keyframes breathePulse {
                0%,100% { transform:scale(.72); opacity:.75; }
                50% { transform:scale(1.08); opacity:1; }
            }
            .breathing-orb {
                width:170px;height:170px;margin:25px auto;
                border-radius:50%;
                background:radial-gradient(circle at 35% 30%,
                #b9f3ff,#52b9ed,#7775e8);
                box-shadow:0 0 38px rgba(82,185,237,.35);
                animation:breathePulse 8s ease-in-out infinite;
            }
            </style>
            <div class="breathing-orb"></div>
            """, unsafe_allow_html=True)
            st.info(f"{instruction} · Session: {elapsed} seconds")
            if st.button("Finish breathing session", key="arcade_breathe_finish"):
                st.session_state.arcade_breath_active = False
                arcade_award("Breathe & Relax", max(1, elapsed // 5))
                st.success(f"Well done! You completed {elapsed} seconds.")
                st.rerun()
            st.caption("The animation runs continuously; instructions update when the page reruns.")
        else:
            st.info("Start whenever you feel ready.")

    # --------------------------------------------------------
    # 2. BUBBLE POP
    # --------------------------------------------------------
    with games[1]:
        st.subheader("🫧 Bubble Pop Relaxation")
        st.write("Pop bubbles for 30 seconds. Each tap earns a point.")

        for key, value in [
            ("arcade_bubble_score", 0),
            ("arcade_bubble_active", False),
            ("arcade_bubble_best", 0),
        ]:
            if key not in st.session_state:
                st.session_state[key] = value

        if st.button("▶ Start / Restart 30-second game", key="arcade_bubble_start_button"):
            st.session_state.arcade_bubble_score = 0
            st.session_state.arcade_bubble_start = time.time()
            st.session_state.arcade_bubble_active = True
            st.rerun()

        if st.session_state.arcade_bubble_active:
            elapsed = int(time.time() - st.session_state.arcade_bubble_start)
            remaining = max(0, 30 - elapsed)
            if remaining == 0:
                st.session_state.arcade_bubble_active = False
                st.session_state.arcade_bubble_best = max(
                    st.session_state.arcade_bubble_best,
                    st.session_state.arcade_bubble_score,
                )
                st.success(f"Time's up! Score: {st.session_state.arcade_bubble_score}")
            else:
                x, y = st.columns(2)
                x.metric("⭐ Score", st.session_state.arcade_bubble_score)
                y.metric("⏱️ Seconds left", remaining)
                cols = st.columns(4)
                for i in range(12):
                    with cols[i % 4]:
                        st.button(
                            random.choice(["🫧", "🔵", "🟣", "💠"]),
                            key=f"arcade_bubble_{i}",
                            use_container_width=True,
                            on_click=arcade_bubble_pop,
                        )
                st.caption("The timer is checked when the page reruns.")
        st.metric("🏆 Personal best this session", st.session_state.arcade_bubble_best)
        if st.button("🔄 Reset bubble score", key="arcade_bubble_reset"):
            st.session_state.arcade_bubble_score = 0
            st.session_state.arcade_bubble_active = False
            st.rerun()

    # --------------------------------------------------------
    # 3. MEMORY MATCH
    # --------------------------------------------------------
    with games[2]:
        st.subheader("🧠 Memory Match")
        st.write("Find all six matching pairs.")

        if "memory_cards" not in st.session_state:
            st.session_state.memory_cards = random.sample(
                ["🌸", "⭐", "💎", "🌈", "🍀", "🦋"] * 2, 12
            )
            st.session_state.memory_matched = []
            st.session_state.memory_selected = []
            st.session_state.memory_misses = 0

        if st.button("🔀 New memory board", key="memory_new_board"):
            st.session_state.memory_cards = random.sample(
                ["🌸", "⭐", "💎", "🌈", "🍀", "🦋"] * 2, 12
            )
            st.session_state.memory_matched = []
            st.session_state.memory_selected = []
            st.session_state.memory_misses = 0
            st.rerun()

        matched = set(st.session_state.memory_matched)
        selected = set(st.session_state.memory_selected)
        cols = st.columns(4)
        for i, symbol in enumerate(st.session_state.memory_cards):
            with cols[i % 4]:
                label = symbol if i in matched or i in selected else "❔"
                st.button(
                    label,
                    key=f"memory_card_{i}",
                    disabled=(i in matched),
                    use_container_width=True,
                    on_click=arcade_memory_pick,
                    args=(i,),
                )

        st.write(f"Pairs found: {len(matched) // 2} / 6")
        st.write(f"Misses: {st.session_state.memory_misses}")
        if len(matched) == 12:
            st.success("🎉 You matched every pair!")
            arcade_award("Memory Match Complete", 25)

    # --------------------------------------------------------
    # 4. REACTION TIME
    # --------------------------------------------------------
    with games[3]:
        st.subheader("⚡ Reaction Time Challenge")
        st.write("Start the test, then press the test button. This simple version measures server-side response time, so it is only an approximation.")

        if "reaction_ready_at" not in st.session_state:
            st.session_state.reaction_ready_at = None
        if "reaction_result" not in st.session_state:
            st.session_state.reaction_result = None

        if st.button("🎯 Start reaction test", key="reaction_start"):
            st.session_state.reaction_ready_at = time.time() + random.uniform(2, 5)
            st.session_state.reaction_result = None
            st.info("Wait a few seconds, then press the test button.")

        if st.session_state.reaction_ready_at is not None:
            if st.button("🟢 TAP NOW!", key="reaction_tap"):
                now = time.time()
                if now < st.session_state.reaction_ready_at:
                    st.session_state.reaction_result = "Too early! Try again."
                else:
                    ms = int((now - st.session_state.reaction_ready_at) * 1000)
                    st.session_state.reaction_result = f"Approximate response: {ms} ms"
                    arcade_award("Reaction Test", max(1, 1000 - ms))
                st.session_state.reaction_ready_at = None
                st.rerun()

        if st.session_state.reaction_result:
            st.info(st.session_state.reaction_result)

    # --------------------------------------------------------
    # 5. NUMBER SEQUENCE
    # --------------------------------------------------------
    with games[4]:
        st.subheader("🔢 Number Sequence")
        st.write("Memorise the digits, hide them, then enter them from memory.")

        if "sequence_level" not in st.session_state:
            st.session_state.sequence_level = 3
        if "sequence_value" not in st.session_state:
            st.session_state.sequence_value = "".join(
                str(random.randint(0, 9))
                for _ in range(st.session_state.sequence_level)
            )
        if "sequence_hidden" not in st.session_state:
            st.session_state.sequence_hidden = False
        if "sequence_feedback" not in st.session_state:
            st.session_state.sequence_feedback = ""

        st.metric("🧠 Level", st.session_state.sequence_level)
        st.progress(st.session_state.sequence_level / 12)

        if not st.session_state.sequence_hidden:
            st.info("Memorise this sequence carefully!")
            st.markdown(
                "<div style='text-align:center;font-size:2.5rem;"
                "font-weight:700;letter-spacing:0.5rem;padding:1rem;"
                "border-radius:16px;background:rgba(100,150,220,0.15)'>"
                + " ".join(st.session_state.sequence_value)
                + "</div>",
                unsafe_allow_html=True,
            )
            if st.button("🙈 Hide sequence & start", key="sequence_hide"):
                st.session_state.sequence_hidden = True
                st.session_state.sequence_feedback = ""
                st.rerun()
        else:
            st.success("The sequence is hidden. Enter it from memory!")
            answer = st.text_input(
                "Enter the digits in the correct order",
                key="sequence_answer_hidden",
                max_chars=12,
            )
            if st.button("✅ Check answer", key="sequence_check_hidden"):
                if answer.strip() == st.session_state.sequence_value:
                    completed_level = st.session_state.sequence_level
                    arcade_award("Number Sequence", completed_level * 5)
                    st.session_state.sequence_level = min(12, completed_level + 1)
                    st.session_state.sequence_value = "".join(
                        str(random.randint(0, 9))
                        for _ in range(st.session_state.sequence_level)
                    )
                    st.session_state.sequence_hidden = False
                    st.session_state.sequence_feedback = (
                        f"🎉 Correct! You completed level {completed_level}!"
                    )
                else:
                    st.session_state.sequence_feedback = (
                        "❌ Not quite! Try again, or reveal the sequence to practise."
                    )
                st.rerun()

            if st.button("👀 Reveal & practise again", key="sequence_reveal"):
                st.session_state.sequence_hidden = False
                st.session_state.sequence_feedback = ""
                st.rerun()

        if st.session_state.sequence_feedback:
            st.success(st.session_state.sequence_feedback)

        if st.button("🔄 Reset sequence game", key="sequence_reset_new"):
            st.session_state.sequence_level = 3
            st.session_state.sequence_value = "".join(
                str(random.randint(0, 9)) for _ in range(3)
            )
            st.session_state.sequence_hidden = False
            st.session_state.sequence_feedback = ""
            st.rerun()

    # 6. ZEN GARDEN
    # --------------------------------------------------------
    with games[5]:
        st.subheader("🌸 Zen Garden")
        st.write("Create a tiny peaceful garden by adding decorations.")
        if "zen_garden_items" not in st.session_state:
            st.session_state.zen_garden_items = []

        left, middle, right = st.columns(3)
        if left.button("🌸 Add flower", key="zen_flower"):
            st.session_state.zen_garden_items.append("🌸")
            arcade_award("Zen Garden", 1)
        if middle.button("🌳 Add tree", key="zen_tree"):
            st.session_state.zen_garden_items.append("🌳")
            arcade_award("Zen Garden", 1)
        if right.button("🪨 Add stone", key="zen_stone"):
            st.session_state.zen_garden_items.append("🪨")
            arcade_award("Zen Garden", 1)

        if st.session_state.zen_garden_items:
            st.markdown(
                '<div class="arcade-banner" style="font-size:30px;'
                'line-height:1.8;text-align:center;">'
                + " ".join(st.session_state.zen_garden_items)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Your garden is waiting for its first flower.")
        if st.button("🧹 Clear garden", key="zen_clear"):
            st.session_state.zen_garden_items = []
            st.rerun()

    # --------------------------------------------------------
    # 7. HEART HEALTH QUIZ
    # --------------------------------------------------------
    with games[6]:
        st.subheader("❤️ Heart Health Quiz")
        questions = [
            ("Which habit supports heart health?", ["Regular physical activity", "Smoking", "Avoiding all sleep"], 0),
            ("Which is a useful everyday habit?", ["Never drink water", "Eat a balanced diet", "Skip every meal"], 1),
            ("What can help manage stress?", ["Slow comfortable breathing", "More stress", "Never take breaks"], 0),
            ("Which is generally recommended for adults?", ["Avoid all movement", "Regular activity suited to you", "Never rest"], 1),
            ("What should you do if exercise causes chest pain?", ["Ignore it", "Stop and seek medical advice", "Exercise harder"], 1),
        ]
        if "health_quiz_index" not in st.session_state:
            st.session_state.health_quiz_index = 0
        if "health_quiz_score" not in st.session_state:
            st.session_state.health_quiz_score = 0
        q, choices, correct = questions[st.session_state.health_quiz_index]
        st.write(f"Question {st.session_state.health_quiz_index + 1} of {len(questions)}")
        st.markdown("### " + q)
        selected_answer = st.radio(
            "Choose your answer",
            choices,
            key=f"health_quiz_answer_{st.session_state.health_quiz_index}",
        )
        if st.button("Submit answer", key=f"health_quiz_submit_{st.session_state.health_quiz_index}"):
            if choices.index(selected_answer) == correct:
                st.session_state.health_quiz_score += 1
                st.success("Correct! +1 point")
                arcade_award("Heart Health Quiz", 1)
            else:
                st.info("Good try! Read the options and keep learning.")
            if st.session_state.health_quiz_index < len(questions) - 1:
                st.session_state.health_quiz_index += 1
            else:
                st.session_state.health_quiz_index = 0
                st.session_state.health_quiz_finished = True
            st.rerun()

        st.metric("Quiz points", st.session_state.health_quiz_score)
        if st.session_state.get("health_quiz_finished", False):
            st.success(f"Quiz complete! Final score: {st.session_state.health_quiz_score}/{len(questions)}")
            if st.button("Play quiz again", key="health_quiz_again"):
                st.session_state.health_quiz_score = 0
                st.session_state.health_quiz_index = 0
                st.session_state.health_quiz_finished = False
                st.rerun()

        st.caption("Educational game only; it does not provide a medical diagnosis.")

    # --------------------------------------------------------
    # 8. ZEN DOODLE
    # --------------------------------------------------------
    with games[7]:
        st.subheader("🎨 Zen Doodle")
        st.write("Build a colourful emoji pattern. Choose a symbol and add it to your canvas.")
        if "doodle_marks" not in st.session_state:
            st.session_state.doodle_marks = []

        symbol = st.selectbox(
            "Choose your drawing symbol",
            ["🌸", "⭐", "💙", "🌈", "🍀", "🦋", "✨", "🌙"],
            key="doodle_symbol",
        )
        if st.button("✍️ Add to canvas", key="doodle_add"):
            st.session_state.doodle_marks.append(symbol)
        if st.session_state.doodle_marks:
            st.markdown(
                '<div class="arcade-banner" style="font-size:30px;'
                'line-height:2;text-align:center;">'
                + " ".join(st.session_state.doodle_marks)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Choose a symbol and start your pattern.")
        if st.button("🧽 Clear canvas", key="doodle_clear"):
            st.session_state.doodle_marks = []
            st.rerun()

    # --------------------------------------------------------
    # ACHIEVEMENTS
    # --------------------------------------------------------
    with games[8]:
        st.subheader("🏆 Achievements")
        st.write("Your scores and achievements are kept for this session.")
        if st.session_state.arcade_best_scores:
            st.dataframe(
                [
                    {"Game": game, "Best score": score}
                    for game, score in st.session_state.arcade_best_scores.items()
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Play a game to start earning scores.")

        if st.session_state.arcade_achievements:
            st.markdown("### 🎖️ Badges")
            for badge in st.session_state.arcade_achievements:
                st.write("🏅 " + badge)
        else:
            st.info("Your first badge is waiting!")

        if st.button("Reset arcade scores", key="arcade_reset_all"):
            for key in [
                "arcade_points",
                "arcade_best_scores",
                "arcade_achievements",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

    st.stop()


# PERSONALIZED USER OVERVIEW
if selected_page == "Dashboard":
    # ============================================================

    # This section is reached only by regular users because the
    # admin dashboard above ends with st.stop().



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

    st.markdown(
        """
        <div style="padding:18px 22px; margin:18px 0;
                    border-radius:18px;
                    background:{theme["surface"]};
                    color:{theme["text"]};
                    border:1px solid {theme["border"]};
            <h3 style="margin:0 0 6px 0;">Your heart, your health!</h3>
            <p style="margin:0;">Ready for a quick, contactless heart-rate estimate?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def go_to_live_measurement():
        st.session_state.user_navigation = "Live Measurement"

    st.button(
        "❤️ Start My Measurement",
        key="home_start_measurement",
        type="primary",
        use_container_width=True,
        on_click=go_to_live_measurement,
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

    # Premium wellness dashboard
    st.markdown("""
    <style>

    .wellness-hero {
        position: relative;
        overflow: hidden;
        padding: 34px 36px;
        border-radius: 26px;
        margin: 14px 0 26px;
        border: 1px solid rgba(255,255,255,.16);
        background: var(--dashboard-surface, #111827);
        color: var(--dashboard-text, #ffffff) !important;
        box-shadow: 0 16px 38px rgba(0,0,0,.18);
        transition: transform .25s ease, box-shadow .25s ease;
    }
    .wellness-hero:hover {
        transform: translateY(-3px);
        box-shadow: 0 22px 44px rgba(22,50,65,.22);
    }
    .wellness-hero::after {
        content: "♥";
        position: absolute;
        right: 36px;
        top: 5px;
        font-size: 150px;
        line-height: 1.2;
        color: rgba(255,255,255,.055);
        pointer-events: none;
    }
    .wellness-hero h2,
    .wellness-hero p,
    .wellness-hero .hero-number,
    .wellness-hero .hero-label {
        position: relative;
        z-index: 1;
        color: var(--dashboard-text, #ffffff) !important;
        opacity: 1 !important;
    }
    .wellness-hero h2 {
        margin: 15px 0 8px;
        font-size: clamp(22px, 2.3vw, 29px);
        font-weight: 750;
        letter-spacing: -.5px;
    }
    .wellness-hero p {
        margin: 0;
        color: var(--dashboard-muted, #b6c3d6) !important;
        font-size: 14px;
        line-height: 1.7;
    }
    .wellness-hero .hero-number {
        margin: 12px 0 5px;
        font-size: clamp(40px, 4.2vw, 54px);
        font-weight: 850;
        line-height: 1.15;
        letter-spacing: -1.7px;
        text-shadow: 0 0 25px rgba(105,255,220,.18);
    }
    .wellness-hero .hero-label {
        font-size: 11px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 1.7px;
        color: var(--dashboard-primary, #4f8cff) !important;
    }

    .section-heading {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -.45px;
        margin: 28px 0 14px;
    }

    .quick-card {
        box-sizing: border-box;
        padding: 23px 22px;
        min-height: 158px;
        height: 100%;
        border: 1px solid var(--dashboard-border, rgba(120,160,180,.3));
        border-radius: 20px;
        background: var(--dashboard-surface, rgba(255,255,255,.65));
        box-shadow: 0 5px 18px rgba(30,60,80,.045);
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    }
    .quick-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 13px 28px rgba(30,60,80,.12);
        border-color: var(--dashboard-accent, #43a9a5);
    }
    .quick-card .quick-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 14px;
        font-size: 24px;
        margin-bottom: 15px;
        background: var(--dashboard-icon-bg, rgba(70,160,170,.13));
    }
    .quick-card h4 {
        margin: 0 0 9px;
        font-size: 17px;
        font-weight: 750;
        color: var(--dashboard-text, inherit);
    }
    .quick-card p {
        font-size: 13px;
        line-height: 1.65;
        color: var(--dashboard-muted, inherit);
        margin: 0;
    }

    [data-testid="stMetric"] {
        padding: 19px 20px;
        border-radius: 19px;
        border: 1px solid var(--dashboard-border, rgba(120,160,180,.25));
        background: var(--dashboard-surface, rgba(255,255,255,.55));
        box-shadow: 0 5px 18px rgba(30,60,80,.035);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 11px 24px rgba(30,60,80,.09);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-weight: 800;
        letter-spacing: -.6px;
    }
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid var(--dashboard-border, rgba(120,160,180,.25));
    }

    @media (max-width: 760px) {
        .wellness-hero {
            padding: 25px 22px;
            border-radius: 20px;
        }
        .wellness-hero::after {
            right: 10px;
            font-size: 100px;
        }
        .section-heading {
            font-size: 20px;
        }
        .quick-card {
            min-height: 130px;
            padding: 18px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Your wellness at a glance</div>',
        unsafe_allow_html=True,
    )

    # Featured heart-rate card
    st.markdown(
        f"""
        <div class="wellness-hero">
            <div class="hero-label">♥ Latest saved heart rate</div>
            <div class="hero-number">{latest_bpm}</div>
            <h2>Your health, at a glance</h2>
            <p>Keep track of your measurements and signal quality over time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary metrics
    card1, card2, card3 = st.columns(3)

    with card1:
        st.metric(
            "❤️ Latest Heart Rate",
            latest_bpm,
            help="Your latest saved measurement, if available.",
        )

    with card2:
        st.metric(
            "📊 Saved Measurements",
            len(overview_df),
            help="Saved measurement records for your account.",
        )

    with card3:
        st.metric(
            "✨ Signal Quality",
            latest_quality,
            help="Quality of your latest saved measurement, if recorded.",
        )

    st.markdown(
        '<div class="section-heading">Explore your dashboard</div>',
        unsafe_allow_html=True,
    )

    quick1, quick2, quick3 = st.columns(3)

    with quick1:
        st.markdown(
            """
            <div class="quick-card">
                <div class="quick-icon">📹</div>
                <h4>Live Measurement</h4>
                <p>Measure your heart rate using facial video.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with quick2:
        st.markdown(
            """
            <div class="quick-card">
                <div class="quick-icon">📈</div>
                <h4>Measurement History</h4>
                <p>Review your saved heart-rate measurements.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with quick3:
        st.markdown(
            """
            <div class="quick-card">
                <div class="quick-icon">⚙️</div>
                <h4>Personal Settings</h4>
                <p>Adjust measurement preferences and appearance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-heading">Recent measurements</div>',
        unsafe_allow_html=True,
    )

    if overview_df.empty:
        st.info(
            "💙 No saved measurements yet. Complete your first "
            "measurement and save it to see your history here."
        )
    else:
        # Show a compact preview of the user's saved records.
        preview = overview_df.head(5).copy()
        if date_column is not None:
            preview = preview.rename(columns={date_column: "Date"})
        if bpm_column is not None:
            preview = preview.rename(columns={bpm_column: "Heart Rate"})
        if quality_column is not None:
            preview = preview.rename(columns={quality_column: "Signal Quality"})

        visible_columns = [
            c for c in ("Date", "Heart Rate", "Signal Quality")
            if c in preview.columns
        ]

        if visible_columns:
            st.dataframe(
                preview[visible_columns],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Your saved measurements are available below.")
            st.dataframe(preview, use_container_width=True, hide_index=True)

    with st.expander("View latest measurement details"):
        st.write(f"**Last recorded:** {latest_date}")
        st.caption(
            "This overview displays saved records associated with your "
            "account. Heart-rate estimates are informational and are not "
            "a medical diagnosis."
        )

    st.divider()
    st.stop()

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

