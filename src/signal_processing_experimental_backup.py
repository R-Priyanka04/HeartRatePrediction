import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch

MIN_BPM = 45.0
MAX_BPM = 180.0
MIN_HZ = MIN_BPM / 60.0
MAX_HZ = MAX_BPM / 60.0


def estimate_bpm_experimental(signal, fps, use_guarded_rule=False):
    """
    Experimental green-channel BPM estimator.

    use_guarded_rule=False:
        Select the strongest spectral peak (baseline).

    use_guarded_rule=True:
        If the strongest peak is below 80 BPM, switch to
        the strongest eligible candidate in 115-145 BPM,
        provided its relative power is at least 0.20.

    This is experimental and has not been validated for live use.
    """

    if signal is None or fps is None:
        return None

    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < 120 or not np.all(np.isfinite(signal)):
        return None

    fps = float(fps)
    if not 5 <= fps <= 120:
        return None

    try:
        processed = detrend(signal, type="linear")
        std = np.std(processed)
        if std < 1e-6:
            return None
        processed = (processed - np.mean(processed)) / std

        nyquist = fps / 2.0
        low = max(0.001, MIN_HZ / nyquist)
        high = min(0.99, MAX_HZ / nyquist)

        if low >= high:
            return None

        b, a = butter(3, [low, high], btype="bandpass")
        filtered = filtfilt(b, a, processed)

        if np.std(filtered) < 1e-8:
            return None

        nperseg = min(len(filtered), 256)
        frequencies, power = welch(
            filtered,
            fs=fps,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=4096,
            detrend=False,
            scaling="density",
        )
    except (ValueError, TypeError, FloatingPointError):
        return None

    valid = (
        (frequencies >= MIN_HZ)
        & (frequencies <= MAX_HZ)
    )
    frequencies = frequencies[valid]
    power = np.nan_to_num(power[valid], nan=0.0, posinf=0.0, neginf=0.0)

    if len(power) == 0 or np.max(power) <= 0:
        return None

    strongest_idx = int(np.argmax(power))
    selected_idx = strongest_idx

    if use_guarded_rule:
        strongest_bpm = frequencies[strongest_idx] * 60.0

        if strongest_bpm < 80.0:
            total_power = float(np.sum(power))

            if total_power > 0:
                relative_power = power / total_power
                eligible = np.where(
                    (frequencies * 60.0 >= 115.0)
                    & (frequencies * 60.0 <= 145.0)
                    & (relative_power >= 0.20)
                )[0]

                if len(eligible):
                    selected_idx = int(
                        eligible[np.argmax(power[eligible])]
                    )

    bpm = float(frequencies[selected_idx] * 60.0)

    if not np.isfinite(bpm) or not MIN_BPM <= bpm <= MAX_BPM:
        return None

    return round(bpm, 1)
