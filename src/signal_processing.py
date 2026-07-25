import numpy as np
from scipy.signal import butter, filtfilt, detrend, windows


# --------------------------------------------------
# Band-pass filter
# --------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=3):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    # Safety checks
    low = max(low, 0.001)
    high = min(high, 0.99)

    if low >= high:
        return None, None

    try:
        b, a = butter(
            order,
            [low, high],
            btype="band"
        )

        return b, a

    except Exception:
        return None, None


# --------------------------------------------------
# Filter pulse signal
# --------------------------------------------------

def bandpass_filter(signal, fs):

    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < 30:
        return signal

    # Heart-rate frequency:
    # 0.7 Hz = 42 BPM
    # 3.0 Hz = 180 BPM

    b, a = butter_bandpass(
        0.7,
        3.0,
        fs,
        order=3
    )

    if b is None:
        return signal

    try:

        filtered = filtfilt(
            b,
            a,
            signal
        )

        return filtered

    except Exception:

        return signal


# --------------------------------------------------
# Estimate BPM using FFT
# --------------------------------------------------

def estimate_bpm(signal, fs):

    if signal is None:
        return None

    if fs is None or fs <= 0:
        return None

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    # Need enough samples
    if len(signal) < 180:
        return None

    # Remove NaN / infinity
    if not np.all(np.isfinite(signal)):
        return None

    # ----------------------------------------------
    # Remove slow brightness drift
    # ----------------------------------------------

    signal = detrend(signal)

    # ----------------------------------------------
    # Band-pass filter
    # ----------------------------------------------

    signal = bandpass_filter(
        signal,
        fs
    )

    # ----------------------------------------------
    # Window signal
    # ----------------------------------------------

    window = windows.hann(
        len(signal)
    )

    signal = signal * window

    # ----------------------------------------------
    # FFT
    # ----------------------------------------------

    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / fs
    )

    magnitude = np.abs(fft_values)

    # ----------------------------------------------
    # Keep human heart-rate frequencies
    # ----------------------------------------------

    mask = (
        (frequencies >= 0.7)
        &
        (frequencies <= 3.0)
    )

    valid_frequencies = frequencies[mask]
    valid_magnitude = magnitude[mask]

    if len(valid_frequencies) == 0:
        return None

    # Find strongest frequency
    peak_index = np.argmax(
        valid_magnitude
    )

    peak_frequency = valid_frequencies[
        peak_index
    ]

    # Hz -> BPM
    bpm = peak_frequency * 60.0

    # Final sanity check
    if bpm < 40 or bpm > 180:
        return None

    return round(
        float(bpm),
        1
    )