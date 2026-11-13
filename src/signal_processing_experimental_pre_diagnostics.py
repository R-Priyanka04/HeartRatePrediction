
import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch

MIN_BPM = 45.0
MAX_BPM = 180.0
MIN_HZ = MIN_BPM / 60.0
MAX_HZ = MAX_BPM / 60.0


def estimate_bpm_experimental(
    signal,
    fps,
    use_guarded_rule=False,
    return_details=False,
):
    """
    Experimental green-channel BPM estimator.

    Baseline:
        Select the strongest eligible spectral bin.

    Guarded experimental rule:
        If the strongest peak is below 80 BPM, look for a candidate
        near twice that BPM. Select it only if its power is at least
        10% of the strongest peak's power.

    This rule is experimental and is not validated for live use.
    """

    def result(bpm, details):
        if return_details:
            return bpm, details
        return bpm

    if signal is None or fps is None:
        return result(None, {"reason": "missing signal or fps"})

    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < 120 or not np.all(np.isfinite(signal)):
        return result(None, {"reason": "invalid or insufficient samples"})

    try:
        fps = float(fps)
    except (ValueError, TypeError):
        return result(None, {"reason": "invalid fps"})

    if not 5 <= fps <= 120:
        return result(None, {"reason": "fps out of range"})

    try:
        processed = detrend(signal, type="linear")
        std = np.std(processed)

        if std < 1e-6:
            return result(None, {"reason": "near-constant signal"})

        processed = (processed - np.mean(processed)) / std

        nyquist = fps / 2.0
        low = max(0.001, MIN_HZ / nyquist)
        high = min(0.99, MAX_HZ / nyquist)

        if low >= high:
            return result(None, {"reason": "invalid filter band"})

        b, a = butter(3, [low, high], btype="bandpass")
        filtered = filtfilt(b, a, processed)

        if np.std(filtered) < 1e-8:
            return result(None, {"reason": "filtered signal too weak"})

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

    except (ValueError, TypeError, FloatingPointError) as exc:
        return result(None, {"reason": f"spectral processing failed: {exc}"})

    valid = (frequencies >= MIN_HZ) & (frequencies <= MAX_HZ)
    frequencies = frequencies[valid]
    power = np.nan_to_num(
        power[valid], nan=0.0, posinf=0.0, neginf=0.0
    )

    if len(power) == 0 or np.max(power) <= 0:
        return result(None, {"reason": "no usable spectral power"})

    strongest_idx = int(np.argmax(power))
    strongest_bpm = float(frequencies[strongest_idx] * 60.0)
    strongest_power = float(power[strongest_idx])

    selected_idx = strongest_idx
    candidate_bpm = None
    candidate_power_ratio = None
    correction_applied = False

    if use_guarded_rule and strongest_bpm < 80.0:
        target_bpm = 2.0 * strongest_bpm

        # Search near 2x the strongest peak, allowing spectral resolution
        # and modest physiological variation.
        tolerance_bpm = max(6.0, target_bpm * 0.08)

        candidate_mask = (
            (frequencies * 60.0 >= target_bpm - tolerance_bpm)
            & (frequencies * 60.0 <= target_bpm + tolerance_bpm)
            & (frequencies * 60.0 >= 90.0)
            & (frequencies * 60.0 <= 160.0)
        )
        candidates = np.where(candidate_mask)[0]

        if len(candidates) > 0 and strongest_power > 0:
            candidate_idx = int(
                candidates[np.argmax(power[candidates])]
            )
            candidate_power = float(power[candidate_idx])
            candidate_bpm = float(frequencies[candidate_idx] * 60.0)
            candidate_power_ratio = candidate_power / strongest_power

            # Experimental threshold; validate before considering deployment.
            if candidate_power_ratio >= 0.10:
                selected_idx = candidate_idx
                correction_applied = True

    bpm = float(frequencies[selected_idx] * 60.0)

    if not np.isfinite(bpm) or not MIN_BPM <= bpm <= MAX_BPM:
        return result(None, {"reason": "estimated bpm out of range"})

    details = {
        "strongest_bpm": round(strongest_bpm, 1),
        "selected_bpm": round(bpm, 1),
        "candidate_bpm": (
            round(candidate_bpm, 1) if candidate_bpm is not None else None
        ),
        "candidate_power_ratio": (
            round(candidate_power_ratio, 3)
            if candidate_power_ratio is not None else None
        ),
        "correction_applied": correction_applied,
    }

    return result(round(bpm, 1), details)

