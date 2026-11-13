
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

# Experimental screening threshold.
# This is NOT a clinically validated confidence threshold.
MIN_PEAK_RATIO = 1.5


def spectral_analysis(signal, fs):
    """Estimate a pulse-frequency candidate and screen spectral ambiguity."""

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
    indices = np.where(valid)[0]

    if len(indices) < 3:
        return None

    # Find local spectral maxima.
    local_peaks, _ = find_peaks(power)
    local_peaks = local_peaks[valid[local_peaks]]

    if len(local_peaks) == 0:
        local_peaks = np.array([
            indices[np.argmax(power[valid])]
        ])

    ranked = local_peaks[np.argsort(power[local_peaks])[::-1]]
    best = ranked[0]

    # Ignore nearby peaks when looking for a competing frequency.
    separated = [
        p for p in ranked[1:]
        if abs(frequencies[p] - frequencies[best]) >= 0.15
    ]

    second = separated[0] if separated else None

    if second is not None:
        ratio = power[best] / max(power[second], 1e-15)
        competing_bpm = frequencies[second] * 60
    else:
        ratio = np.nan
        competing_bpm = np.nan

    bpm = frequencies[best] * 60

    # Experimental screening rule:
    # accept only if a distinct competing peak exists and
    # the strongest peak is sufficiently stronger.
    quality_ok = bool(
        second is not None
        and np.isfinite(ratio)
        and ratio >= MIN_PEAK_RATIO
    )

    return {
        "bpm": bpm,
        "reported_bpm": bpm if quality_ok else np.nan,
        "quality_ok": quality_ok,
        "competing_bpm": competing_bpm,
        "separated_peak_ratio": ratio,
        "frequencies": frequencies,
        "power": power,
        "filtered": filtered,
    }


def main():
    df = pd.read_csv(CSV_FILE)

    required_columns = [
        "accepted",
        "elapsed_seconds",
        "mean_green",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )

    # Keep accepted samples within the requested interval.
    df = df[
        (df["accepted"] == True)
        & (df["elapsed_seconds"] >= START_SECONDS)
        & (df["elapsed_seconds"] <= END_SECONDS)
    ].dropna(
        subset=[
            "elapsed_seconds",
            "mean_green",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
        ]
    ).sort_values("elapsed_seconds").copy()

    if len(df) < 100:
        raise ValueError(
            "Not enough valid samples. Check the CSV recording."
        )

    t = df["elapsed_seconds"].to_numpy(dtype=float)

    # Remove duplicate or non-increasing timestamps.
    keep = np.concatenate(([True], np.diff(t) > 0))
    df = df.iloc[np.flatnonzero(keep)].copy()

    t = df["elapsed_seconds"].to_numpy(dtype=float)

    if len(t) < 100:
        raise ValueError(
            "Not enough unique timestamps for analysis."
        )

    fs_original = 1.0 / np.median(np.diff(t))
    fs = min(fs_original, 30.0)

    if fs <= 6.0:
        raise ValueError(
            f"Sampling rate {fs:.2f} Hz is too low for this filter."
        )

    # Resample the measurements onto a uniform time grid.
    uniform_t = np.arange(t[0], t[-1], 1.0 / fs)

    green = np.interp(
        uniform_t,
        t,
        df["mean_green"].to_numpy(dtype=float),
    )

    x1 = np.interp(
        uniform_t, t, df["x1"].to_numpy(dtype=float)
    )
    y1 = np.interp(
        uniform_t, t, df["y1"].to_numpy(dtype=float)
    )
    x2 = np.interp(
        uniform_t, t, df["x2"].to_numpy(dtype=float)
    )
    y2 = np.interp(
        uniform_t, t, df["y2"].to_numpy(dtype=float)
    )
    width = np.interp(
        uniform_t, t, df["width"].to_numpy(dtype=float)
    )
    height = np.interp(
        uniform_t, t, df["height"].to_numpy(dtype=float)
    )

    # Track the ROI centre's frame-to-frame movement.
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    movement = np.sqrt(
        np.diff(center_x) ** 2 + np.diff(center_y) ** 2
    )

    rows = []

    start = max(START_SECONDS, uniform_t[0])
    final_start = min(
        END_SECONDS - WINDOW_SECONDS,
        uniform_t[-1] - WINDOW_SECONDS,
    )

    if final_start < start:
        raise ValueError(
            "Recording is too short for the selected window length."
        )

    starts = np.arange(
        start,
        final_start + 1e-9,
        STEP_SECONDS,
    )

    for window_start in starts:
        window_end = window_start + WINDOW_SECONDS

        mask = (
            (uniform_t >= window_start)
            & (uniform_t < window_end)
        )

        if mask.sum() < int(fs * WINDOW_SECONDS * 0.95):
            continue

        result = spectral_analysis(green[mask], fs)

        if result is None:
            continue

        move_mask = (
            (uniform_t[1:] >= window_start)
            & (uniform_t[1:] < window_end)
        )

        window_movement = movement[move_mask]

        row = {
            "start_seconds": window_start,
            "end_seconds": window_end,
            "bpm_candidate": result["bpm"],
            "reported_bpm": result["reported_bpm"],
            "quality_ok": result["quality_ok"],
            "competing_peak_bpm": result["competing_bpm"],
            "separated_peak_ratio": result["separated_peak_ratio"],
            "roi_width_sd": np.std(width[mask]),
            "roi_height_sd": np.std(height[mask]),
            "roi_center_movement_median": (
                np.median(window_movement)
                if len(window_movement)
                else np.nan
            ),
            "roi_center_movement_p95": (
                np.percentile(window_movement, 95)
                if len(window_movement)
                else np.nan
            ),
            "roi_center_movement_max": (
                np.max(window_movement)
                if len(window_movement)
                else np.nan
            ),
        }

        rows.append(row)

    if not rows:
        raise ValueError(
            "No analysis windows were produced."
        )

    results = pd.DataFrame(rows)

    results.to_csv(
        "improved_signal_diagnostics.csv",
        index=False,
    )

    print("\n--- EXPERIMENTAL SIGNAL QUALITY DIAGNOSTICS ---")
    print(f"Sampling rate: {fs:.2f} Hz")
    print(f"Window length: {WINDOW_SECONDS} seconds")
    print(f"Step size: {STEP_SECONDS} seconds")
    print(f"Peak-ratio screening threshold: {MIN_PEAK_RATIO}")
    print(
        "Quality flags are experimental, not validated "
        "confidence scores."
    )

    display_columns = [
        "start_seconds",
        "end_seconds",
        "bpm_candidate",
        "reported_bpm",
        "quality_ok",
        "competing_peak_bpm",
        "separated_peak_ratio",
    ]

    print(
        results[display_columns].round(2).to_string(index=False)
    )

    passed = int(results["quality_ok"].sum())
    total = len(results)

    print(f"\nWindows passing spectral screening: {passed}/{total}")
    print(
        f"Windows flagged for ambiguity: {total - passed}/{total}"
    )

    print("\nSaved: improved_signal_diagnostics.csv")

    # Plot raw BPM candidates and screened estimates separately.
    midpoint = (
        results["start_seconds"] + WINDOW_SECONDS / 2
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        midpoint,
        results["bpm_candidate"],
        marker="o",
        label="BPM candidate",
    )

    plt.plot(
        midpoint,
        results["reported_bpm"],
        marker="x",
        linestyle="--",
        label="Candidate passing spectral screen",
    )

    plt.xlabel("Window midpoint (seconds)")
    plt.ylabel("BPM estimate")
    plt.title("Green Signal: BPM Candidates and Quality Screening")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "improved_signal_diagnostics.png",
        dpi=150,
    )
    plt.close()

    print("Saved: improved_signal_diagnostics.png")


if __name__ == "__main__":
    main()

