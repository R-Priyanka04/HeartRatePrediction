import numpy as np

from scipy.signal import (
    butter,
    filtfilt,
    detrend,
    welch
)


# ============================================================
# HEART RATE RANGE
# ============================================================

MIN_BPM = 45.0
MAX_BPM = 180.0

MIN_HZ = MIN_BPM / 60.0
MAX_HZ = MAX_BPM / 60.0


# ============================================================
# PREPROCESS SIGNAL
# ============================================================

def preprocess_signal(signal):
    """
    Clean the raw green-channel rPPG signal.

    Steps:
        1. Convert to numpy
        2. Remove invalid values
        3. Detrend lighting drift
        4. Normalize
    """

    if signal is None:
        return None

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    if len(signal) < 60:
        return None

    if not np.all(
        np.isfinite(signal)
    ):
        return None

    # --------------------------------------------------------
    # Remove slow illumination trend
    # --------------------------------------------------------

    try:

        signal = detrend(
            signal,
            type="linear"
        )

    except Exception:

        signal = (
            signal
            -
            np.mean(signal)
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    std = np.std(signal)

    if std < 1e-6:
        return None

    signal = (
        signal
        -
        np.mean(signal)
    ) / std

    return signal


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(signal, fps):
    """
    Keep frequencies corresponding approximately to
    45-180 BPM.
    """

    if signal is None:
        return None

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    if fps is None:
        return None

    fps = float(fps)

    if fps <= 0:
        return None

    nyquist = fps / 2.0

    low = MIN_HZ / nyquist
    high = MAX_HZ / nyquist

    low = max(
        0.001,
        low
    )

    high = min(
        0.99,
        high
    )

    if low >= high:
        return None

    try:

        # Order 3 is sufficient for rPPG
        b, a = butter(
            3,
            [low, high],
            btype="bandpass"
        )

        filtered = filtfilt(
            b,
            a,
            signal
        )

    except Exception as e:

        print(
            "Bandpass filter error:",
            repr(e)
        )

        return None

    if not np.all(
        np.isfinite(filtered)
    ):
        return None

    return filtered


# ============================================================
# BPM ESTIMATION
# ============================================================

def estimate_bpm(signal, fps):
    """
    Estimate BPM from forehead green-channel intensity.

    Pipeline:

        Raw green signal
              ↓
        Detrending
              ↓
        Normalization
              ↓
        Band-pass filtering
              ↓
        Welch power spectrum
              ↓
        Dominant frequency
              ↓
             BPM
    """

    if signal is None:
        return None

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Need enough samples
    # --------------------------------------------------------

    if len(signal) < 120:
        return None

    if fps is None:
        return None

    fps = float(fps)

    if fps < 5 or fps > 120:
        return None

    # --------------------------------------------------------
    # 1. PREPROCESS
    # --------------------------------------------------------

    processed = preprocess_signal(
        signal
    )

    if processed is None:

        print(
            "BPM DEBUG: preprocessing failed"
        )

        return None

    # --------------------------------------------------------
    # 2. BANDPASS FILTER
    # --------------------------------------------------------

    filtered = bandpass_filter(
        processed,
        fps
    )

    if filtered is None:

        print(
            "BPM DEBUG: bandpass failed"
        )

        return None

    # --------------------------------------------------------
    # Check filtered signal
    # --------------------------------------------------------

    filtered_std = float(
        np.std(filtered)
    )

    if filtered_std < 1e-8:

        print(
            "BPM DEBUG: filtered signal too weak"
        )

        return None

    # --------------------------------------------------------
    # 3. WELCH POWER SPECTRUM
    # --------------------------------------------------------

    try:

        nperseg = min(
            len(filtered),
            256
        )

        frequencies, power = welch(
            filtered,
            fs=fps,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=4096,
            detrend=False,
            scaling="density"
        )

    except Exception as e:

        print(
            "BPM DEBUG: Welch failed:",
            repr(e)
        )

        return None

    # --------------------------------------------------------
    # 4. PHYSIOLOGICAL FREQUENCY RANGE
    # --------------------------------------------------------

    valid_mask = (
        (frequencies >= MIN_HZ)
        &
        (frequencies <= MAX_HZ)
    )

    valid_frequencies = (
        frequencies[valid_mask]
    )

    valid_power = (
        power[valid_mask]
    )

    if len(valid_frequencies) == 0:

        print(
            "BPM DEBUG: no valid frequencies"
        )

        return None

    if not np.any(
        np.isfinite(valid_power)
    ):

        print(
            "BPM DEBUG: invalid PSD"
        )

        return None

    valid_power = np.nan_to_num(
        valid_power,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if np.max(valid_power) <= 0:

        print(
            "BPM DEBUG: no spectral power"
        )

        return None

    # --------------------------------------------------------
    # 5. FIND DOMINANT FREQUENCY
    # --------------------------------------------------------

    peak_index = int(
        np.argmax(valid_power)
    )

    peak_frequency = float(
        valid_frequencies[
            peak_index
        ]
    )

    peak_power = float(
        valid_power[
            peak_index
        ]
    )

    # --------------------------------------------------------
    # 6. SPECTRAL CONFIDENCE
    # --------------------------------------------------------

    total_power = float(
        np.sum(valid_power)
    )

    if total_power <= 0:
        return None

    confidence = (
        peak_power
        /
        total_power
    )

    # IMPORTANT:
    #
    # Do NOT reject the BPM here merely because the confidence
    # is low.
    #
    # With Welch + nfft=4096, power is distributed across many
    # frequency bins. A threshold such as 0.01 can therefore
    # reject usable signals.
    #
    # We print confidence for debugging instead.

    # --------------------------------------------------------
    # 7. FREQUENCY -> BPM
    # --------------------------------------------------------

    bpm = (
        peak_frequency
        *
        60.0
    )

    # --------------------------------------------------------
    # 8. SANITY CHECK
    # --------------------------------------------------------

    if not np.isfinite(bpm):
        return None

    if bpm < MIN_BPM:
        return None

    if bpm > MAX_BPM:
        return None

    print(
        f"BPM DEBUG | "
        f"Peak={peak_frequency:.3f} Hz | "
        f"BPM={bpm:.1f} | "
        f"Confidence={confidence:.4f}"
    )

    return round(
        float(bpm),
        1
    )


# ============================================================
# BPM SMOOTHING
# ============================================================

def smooth_bpm(
    new_bpm,
    history,
    max_history=10
):
    """
    Median smoothing for BPM.
    """

    if new_bpm is None:
        return None

    if history is None:
        return None

    new_bpm = float(
        new_bpm
    )

    if not np.isfinite(
        new_bpm
    ):
        return None

    if (
        new_bpm < MIN_BPM
        or
        new_bpm > MAX_BPM
    ):
        return None

    # --------------------------------------------------------
    # Reject extreme sudden jumps
    # --------------------------------------------------------

    if len(history) >= 3:

        previous = float(
            np.median(
                history[-5:]
            )
        )

        difference = abs(
            new_bpm
            -
            previous
        )

        if difference > 30:

            return round(
                previous,
                1
            )

    history.append(
        new_bpm
    )

    if len(history) > max_history:

        history.pop(0)

    if len(history) < 3:
        return None

    result = np.median(
        history
    )

    return round(
        float(result),
        1
    )


# ============================================================
# BPM STABILITY
# ============================================================

def is_bpm_stable(
    history,
    max_deviation=8.0
):
    """
    Check whether recent BPM estimates agree.
    """

    if history is None:
        return False

    if len(history) < 5:
        return False

    recent = np.asarray(
        history[-5:],
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(recent)
    ):
        return False

    deviation = float(
        np.std(recent)
    )

    return (
        deviation
        <=
        max_deviation
    )


# ============================================================
# SIGNAL QUALITY
# ============================================================

def calculate_signal_quality(
    signal
):
    """
    Estimate simple signal quality from 0-100.

    This is intended for UI feedback rather than medical
    interpretation.
    """

    if signal is None:
        return 0.0

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    if len(signal) < 30:
        return 0.0

    if not np.all(
        np.isfinite(signal)
    ):
        return 0.0

    # Remove overall trend before evaluating motion
    try:

        clean = detrend(
            signal
        )

    except Exception:

        clean = (
            signal
            -
            np.mean(signal)
        )

    std = float(
        np.std(clean)
    )

    if std < 0.01:
        return 10.0

    differences = np.diff(
        clean
    )

    if len(differences) == 0:
        return 0.0

    motion = float(
        np.std(differences)
    )

    # Relative motion is more meaningful than absolute
    # green-channel intensity.

    relative_motion = (
        motion
        /
        (std + 1e-8)
    )

    if relative_motion < 0.30:

        quality = 95.0

    elif relative_motion < 0.50:

        quality = 85.0

    elif relative_motion < 0.75:

        quality = 70.0

    elif relative_motion < 1.00:

        quality = 50.0

    else:

        quality = 25.0

    return quality