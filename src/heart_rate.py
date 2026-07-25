import cv2
import numpy as np
import time
from collections import deque


# ==========================================================
# SETTINGS
# ==========================================================

MAX_SIGNAL_SAMPLES = 300
BPM_HISTORY_SIZE = 10

green_signal = deque(maxlen=MAX_SIGNAL_SAMPLES)
timestamps = deque(maxlen=MAX_SIGNAL_SAMPLES)
bpm_history = deque(maxlen=BPM_HISTORY_SIZE)


# ==========================================================
# EXTRACT FOREHEAD ROI
# ==========================================================

def extract_forehead(frame, face_landmarks):
    """
    Extract forehead region from MediaPipe face landmarks.
    """

    if frame is None or face_landmarks is None:
        return np.array([])

    h, w, _ = frame.shape

    # Forehead landmarks
    landmark_ids = [10, 338, 297, 332, 284]

    points = []

    for idx in landmark_ids:

        lm = face_landmarks.landmark[idx]

        x = int(lm.x * w)
        y = int(lm.y * h)

        points.append((x, y))

    points = np.array(points, dtype=np.int32)

    x, y, rw, rh = cv2.boundingRect(points)

    # Use central part of forehead
    x += int(rw * 0.20)
    y += int(rh * 0.15)

    rw = int(rw * 0.60)
    rh = int(rh * 0.60)

    # Image boundary protection
    x = max(0, x)
    y = max(0, y)

    x2 = min(w, x + rw)
    y2 = min(h, y + rh)

    if x2 <= x or y2 <= y:
        return np.array([])

    roi = frame[y:y2, x:x2]

    return roi


# ==========================================================
# GREEN SIGNAL
# ==========================================================

def update_signal(roi):
    """
    Extract average green-channel intensity from forehead.
    """

    if roi is None:
        return None

    if roi.size == 0:
        return None

    # Green channel
    green_channel = roi[:, :, 1]

    value = float(np.mean(green_channel))

    if not np.isfinite(value):
        return None

    green_signal.append(value)
    timestamps.append(time.time())

    return value


# ==========================================================
# SIGNAL QUALITY
# ==========================================================

def get_signal_quality():
    """
    Estimate whether enough useful variation exists
    in the green-channel signal.

    Returns:
        quality : POOR / FAIR / GOOD
        score   : 0-100
    """

    if len(green_signal) < 30:
        return "POOR", 0

    signal = np.asarray(
        green_signal,
        dtype=np.float64
    )

    if not np.all(np.isfinite(signal)):
        return "POOR", 0

    variation = np.std(signal)

    # Very little variation
    if variation < 0.15:
        return "POOR", 20

    # Moderate signal
    elif variation < 0.5:
        return "FAIR", 55

    # Strong variation
    else:
        return "GOOD", 90


# ==========================================================
# ROI QUALITY
# ==========================================================

def check_roi_quality(roi):
    """
    Check whether forehead ROI is large enough
    and has usable brightness.
    """

    if roi is None or roi.size == 0:
        return False, "No forehead detected"

    height, width = roi.shape[:2]

    # ROI too small
    if width < 20 or height < 10:
        return False, "Move closer to camera"

    brightness = float(
        np.mean(
            cv2.cvtColor(
                roi,
                cv2.COLOR_BGR2GRAY
            )
        )
    )

    if brightness < 40:
        return False, "Lighting too dark"

    if brightness > 235:
        return False, "Lighting too bright"

    return True, "Good"


# ==========================================================
# BPM SMOOTHING
# ==========================================================

def smooth_bpm(bpm):
    """
    Stabilize BPM using median of recent readings.
    """

    if bpm is None:
        return None

    if not np.isfinite(bpm):
        return None

    if bpm < 40 or bpm > 180:
        return None

    bpm_history.append(float(bpm))

    smoothed = np.median(
        np.asarray(bpm_history)
    )

    return round(
        float(smoothed),
        1
    )


# ==========================================================
# BPM STATUS
# ==========================================================

def get_bpm_status(bpm):
    """
    Basic display classification.

    This is only an informational software label,
    not a medical diagnosis.
    """

    if bpm is None:
        return "CALCULATING"

    if bpm < 60:
        return "LOW"

    if bpm <= 100:
        return "NORMAL"

    return "HIGH"


# ==========================================================
# MEASUREMENT PROGRESS
# ==========================================================

def get_measurement_progress(required_samples=180):
    """
    Return signal collection progress from 0 to 100.
    """

    if required_samples <= 0:
        return 100

    progress = (
        len(green_signal)
        / required_samples
    ) * 100

    return min(
        100,
        int(progress)
    )


# ==========================================================
# RESET
# ==========================================================

def reset_buffers():
    """
    Clear all measurements for a new session.
    """

    green_signal.clear()
    timestamps.clear()
    bpm_history.clear()


# ==========================================================
# SAMPLE COUNT
# ==========================================================

def get_sample_count():
    return len(green_signal)