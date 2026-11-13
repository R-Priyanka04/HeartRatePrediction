
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
red_signal = deque(maxlen=MAX_SIGNAL_SAMPLES)
blue_signal = deque(maxlen=MAX_SIGNAL_SAMPLES)
timestamps = deque(maxlen=MAX_SIGNAL_SAMPLES)
bpm_history = deque(maxlen=BPM_HISTORY_SIZE)


# ==========================================================
# EXTRACT FOREHEAD ROI
# ==========================================================

def extract_forehead(frame, face_landmarks):
    """
    Extract forehead ROI using the shared MediaPipe ROI function.
    """

    if frame is None or face_landmarks is None:
        return np.array([])

    # Use the same ROI calculation as the standalone ROI test.
    from src.eye_roi import get_landmark_forehead_box

    box = get_landmark_forehead_box(frame, face_landmarks)

    if box is None:
        return np.array([])

    x, y, width, height = box

    if width <= 0 or height <= 0:
        return np.array([])

    return frame[y:y + height, x:x + width].copy()


# ==========================================================
# EXTRACT RGB SIGNALS
# ==========================================================

def update_rgb_signal(roi):
    """
    Extract mean red, green, and blue channel intensities.

    OpenCV stores colour channels in BGR order.
    Returns a dictionary of channel means or None.
    """

    if roi is None or roi.size == 0:
        return None

    if roi.ndim != 3 or roi.shape[2] < 3:
        return None

    means = np.mean(roi[:, :, :3], axis=(0, 1))

    # Reject invalid measurements.
    if not np.all(np.isfinite(means)):
        return None

    blue_value = float(means[0])
    green_value = float(means[1])
    red_value = float(means[2])

    sample_time = time.time()

    # Keep channel buffers and timestamps synchronized.
    blue_signal.append(blue_value)
    green_signal.append(green_value)
    red_signal.append(red_value)
    timestamps.append(sample_time)

    return {
        "red": red_value,
        "green": green_value,
        "blue": blue_value,
        "timestamp": sample_time,
    }


def update_signal(roi):
    """
    Backward-compatible green-channel update.

    Existing dashboard code can continue calling update_signal().
    """

    sample = update_rgb_signal(roi)

    if sample is None:
        return None

    return sample["green"]


# ==========================================================
# SIGNAL ACCESS
# ==========================================================

def get_rgb_signals():
    """Return copies of the current RGB signals and timestamps."""

    return {
        "red": np.asarray(red_signal, dtype=np.float64),
        "green": np.asarray(green_signal, dtype=np.float64),
        "blue": np.asarray(blue_signal, dtype=np.float64),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
    }


# ==========================================================
# SIGNAL QUALITY
# ==========================================================

def get_signal_quality():
    """
    Basic variation-based signal indicator.
    This is not a validated physiological quality metric.
    """

    if len(green_signal) < 30:
        return "POOR", 0

    signal = np.asarray(green_signal, dtype=np.float64)

    if not np.all(np.isfinite(signal)):
        return "POOR", 0

    variation = float(np.std(signal))

    if variation < 0.15:
        return "POOR", 20
    elif variation < 0.5:
        return "FAIR", 55
    else:
        return "GOOD", 90


# ==========================================================
# ROI QUALITY
# ==========================================================

def check_roi_quality(roi):
    """Check ROI dimensions and basic brightness."""

    if roi is None or roi.size == 0:
        return False, "No forehead detected"

    height, width = roi.shape[:2]

    if width < 20 or height < 10:
        return False, "Move closer to camera"

    brightness = float(
        np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
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
    """Stabilize valid BPM readings using the median."""

    if bpm is None or not np.isfinite(bpm):
        return None

    if bpm < 40 or bpm > 180:
        return None

    bpm_history.append(float(bpm))

    return round(float(np.median(bpm_history)), 1)


# ==========================================================
# BPM STATUS
# ==========================================================

def get_bpm_status(bpm):
    """Informational software label, not a medical diagnosis."""

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
    """Return collection progress from 0 to 100."""

    if required_samples <= 0:
        return 100

    progress = len(green_signal) / required_samples * 100

    return min(100, int(progress))


# ==========================================================
# RESET
# ==========================================================

def reset_buffers():
    """Clear all signal buffers for a new measurement."""

    green_signal.clear()
    red_signal.clear()
    blue_signal.clear()
    timestamps.clear()
    bpm_history.clear()


# ==========================================================
# SAMPLE COUNT
# ==========================================================

def get_sample_count():
    return len(green_signal)
