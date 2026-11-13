

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch, detrend, find_peaks


CSV_FILE = "forehead_v2_diagnostics.csv"

START_SECONDS = 8
END_SECONDS = 55
WINDOW_SECONDS = 20
STEP_SECONDS = 2

# Experimental screening settings, not medically validated.
MIN_PEAK_RATIO = 1.5
MAX_BPM_CHANGE = 15.0


def spectral_analysis(signal, fs):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 10 or not np.all(np.isfinite(signal)):
        return None

    signal = detrend(signal)

    sos = butter(
        3,
        [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError:
        return None

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=len(filtered),
        noverlap=0,
        window="hann",
    )

    valid = (frequencies >= 0.7) & (frequencies <= 3.0)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) < 3:
        return None

    local_peaks, _ = find_peaks(power)
    local_peaks = local_peaks[valid[local_peaks]]

    if len(local_peaks) == 0:
        local_peaks = np.array([
            valid_indices[np.argmax(power[valid])]
        ])

    ranked = local_peaks[np.argsort(power[local_peaks])[::-1]]
    best = ranked[0]

    separated = [
        p for p in ranked[1:]
        if abs(frequencies[p] - frequencies[best]) >= 0.15
    ]

    second = separated[0] if separated else None

    if second is None:
        ratio = np.nan
        competing_bpm = np.nan
    else:
        ratio = power[best] / max(power[second], 1e-15)
        competing_bpm = frequencies[second] * 60

    bpm = frequencies[best] * 60

    spectral_ok = bool(
        second is not None
        and np.isfinite(ratio)
        and ratio >= MIN_PEAK_RATIO
    )

    return {
        "bpm_candidate": bpm,
        "competing_peak_bpm": competing_bpm,
        "separated_peak_ratio": ratio,
        "spectral_ok": spectral_ok,
    }


def main():
    df = pd.read_csv(CSV_FILE)

    required = [
        "accepted",
        "elapsed_seconds",
        "mean_green",
        "x1", "y1", "x2", "y2",
        "width", "height",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df = df[
        (df["accepted"] == True)
        & (df["elapsed_seconds"] >= START_SECONDS)
        & (df["elapsed_seconds"] <= END_SECONDS)
    ].dropna(subset=required[1:]).sort_values("elapsed_seconds").copy()

    if len(df) < 100:
        raise ValueError("Not enough valid samples in the CSV.")

    t = df["elapsed_seconds"].to_numpy(dtype=float)

    keep = np.concatenate(([True], np.diff(t) > 0))
    df = df.iloc[np.flatnonzero(keep)].copy()
    t = df["elapsed_seconds"].to_numpy(dtype=float)

    if len(t) < 100:
        raise ValueError("Not enough unique timestamps.")

    fs_original = 1.0 / np.median(np.diff(t))
    fs = min(fs_original, 30.0)

    if fs <= 6.0:
        raise ValueError(f"Sampling rate is too low: {fs:.2f} Hz")

    uniform_t = np.arange(t[0], t[-1], 1.0 / fs)

    def resample(column):
        return np.interp(
            uniform_t,
            t,
            df[column].to_numpy(dtype=float),
        )

    green = resample("mean_green")
    x1, y1 = resample("x1"), resample("y1")
    x2, y2 = resample("x2"), resample("y2")
    width, height = resample("width"), resample("height")

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    movement = np.sqrt(
        np.diff(center_x) ** 2 + np.diff(center_y) ** 2
    )

    rows = []

    first_start = max(START_SECONDS, uniform_t[0])
    last_start = min(
        END_SECONDS - WINDOW_SECONDS,
        uniform_t[-1] - WINDOW_SECONDS,
    )

    for start in np.arange(
        first_start,
        last_start + 1e-9,
        STEP_SECONDS,
    ):
        end = start + WINDOW_SECONDS
        mask = (uniform_t >= start) & (uniform_t < end)

        if mask.sum() < int(fs * WINDOW_SECONDS * 0.95):
            continue

        result = spectral_analysis(green[mask], fs)
        if result is None:
            continue

        move_mask = (
            (uniform_t[1:] >= start)
            & (uniform_t[1:] < end)
        )
        window_movement = movement[move_mask]

        rows.append({
            "start_seconds": start,
            "end_seconds": end,
            "midpoint_seconds": start + WINDOW_SECONDS / 2,
            **result,
            "roi_width_sd": np.std(width[mask]),
            "roi_height_sd": np.std(height[mask]),
            "roi_center_movement_median": (
                np.median(window_movement)
                if len(window_movement) else np.nan
            ),
            "roi_center_movement_p95": (
                np.percentile(window_movement, 95)
                if len(window_movement) else np.nan
            ),
            "roi_center_movement_max": (
                np.max(window_movement)
                if len(window_movement) else np.nan
            ),
        })

    results = pd.DataFrame(rows)

    if results.empty:
        raise ValueError("No analysis windows were produced.")

    # Compare each candidate to the preceding window's candidate.
    # This is a screening flag, not proof of a physiological error.
    results["previous_bpm_candidate"] = (
        results["bpm_candidate"].shift(1)
    )

    results["bpm_change"] = (
        results["bpm_candidate"]
        - results["previous_bpm_candidate"]
    ).abs()

    # First window has no previous window to compare against.
    results["temporal_ok"] = (
        results["bpm_change"].isna()
        | (results["bpm_change"] <= MAX_BPM_CHANGE)
    )

    results["combined_ok"] = (
        results["spectral_ok"] & results["temporal_ok"]
    )

    results["reported_bpm"] = results["bpm_candidate"].where(
        results["combined_ok"],
        np.nan,
    )

    results["screening_status"] = np.select(
        [
            ~results["spectral_ok"] & ~results["temporal_ok"],
            ~results["spectral_ok"],
            ~results["temporal_ok"],
        ],
        [
            "Spectral and temporal flags",
            "Spectral ambiguity",
            "Abrupt change",
        ],
        default="Passed experimental checks",
    )

    results.to_csv("improved_signal_diagnostics.csv", index=False)

    print("\n--- EXPERIMENTAL SIGNAL QUALITY DIAGNOSTICS ---")
    print(f"Sampling rate: {fs:.2f} Hz")
    print(f"Window: {WINDOW_SECONDS}s; step: {STEP_SECONDS}s")
    print(f"Minimum peak ratio: {MIN_PEAK_RATIO}")
    print(f"Maximum change threshold: {MAX_BPM_CHANGE} BPM")
    print("These are experimental screening rules, not validated confidence scores.")

    cols = [
        "start_seconds",
        "bpm_candidate",
        "competing_peak_bpm",
        "separated_peak_ratio",
        "bpm_change",
        "spectral_ok",
        "temporal_ok",
        "reported_bpm",
        "screening_status",
    ]

    print(results[cols].round(2).to_string(index=False))

    print(
        f"\nWindows passing both checks: "
        f"{int(results['combined_ok'].sum())}/{len(results)}"
    )
    print("Saved: improved_signal_diagnostics.csv")

    # Graph 1: all candidates versus candidates passing both screens.
    plt.figure(figsize=(12, 6))
    plt.plot(
        results["midpoint_seconds"],
        results["bpm_candidate"],
        marker="o",
        label="BPM candidate",
    )
    plt.plot(
        results["midpoint_seconds"],
        results["reported_bpm"],
        marker="x",
        linestyle="--",
        label="Candidate passing both screens",
    )
    plt.xlabel("Window midpoint (seconds)")
    plt.ylabel("BPM estimate")
    plt.title("Green Signal: Spectral and Temporal Screening")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("improved_signal_diagnostics.png", dpi=150)
    plt.close()

    print("Saved: improved_signal_diagnostics.png")


if __name__ == "__main__":
    main()

