
import os
import numpy as np
import pandas as pd
from scipy.signal import detrend, windows

FILES = [
    "rgb_session_1.csv",
    "rgb_session_2.csv",
    "rgb_session_3.csv",
]

TARGET_FPS = 20.0
WINDOW_SECONDS = 20
STEP_SECONDS = 5
MIN_BPM = 42
MAX_BPM = 180


def load_and_resample(filename):
    df = pd.read_csv(filename)

    required = {"time_seconds", "red", "green", "blue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[list(required)].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("time_seconds")
    df = df.drop_duplicates(subset="time_seconds")

    t = df["time_seconds"].to_numpy(dtype=float)
    rgb = df[["red", "green", "blue"]].to_numpy(dtype=float)

    if len(t) < 100 or t[-1] <= t[0]:
        raise ValueError(f"Not enough valid data in {filename}")

    # Shift time to zero and interpolate onto a common 20 Hz grid.
    t = t - t[0]
    uniform_t = np.arange(0, t[-1], 1.0 / TARGET_FPS)

    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, channel])
        for channel in range(3)
    ])

    return uniform_t, uniform_rgb


def make_signals(rgb):
    red, green, blue = rgb.T

    # Green-channel signal
    green_signal = green.copy()

    # Normalized green
    green_mean = np.mean(green)
    normalized_green = (
        (green - green_mean) / green_mean
        if abs(green_mean) > 1e-12
        else green - green_mean
    )

    # CHROM: normalize each colour channel within the window.
    means = np.mean(rgb, axis=0)
    means = np.where(np.abs(means) < 1e-12, 1e-12, means)
    r, g, b = (rgb / means).T

    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b

    std_y = np.std(y)
    alpha = np.std(x) / std_y if std_y > 1e-12 else 0.0
    chrom_signal = x - alpha * y

    return {
        "Green": green_signal,
        "Normalized_Green": normalized_green,
        "CHROM": chrom_signal,
    }


def estimate_bpm(signal):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 10 or not np.all(np.isfinite(signal)):
        return np.nan

    signal = detrend(signal, type="linear")
    signal = signal * windows.hann(len(signal), sym=False)

    if np.std(signal) < 1e-10:
        return np.nan

    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / TARGET_FPS)

    valid = (
        (frequencies >= MIN_BPM / 60.0)
        & (frequencies <= MAX_BPM / 60.0)
    )

    if not np.any(valid):
        return np.nan

    valid_indices = np.flatnonzero(valid)
    peak_index = valid_indices[np.argmax(spectrum[valid])]
    return float(frequencies[peak_index] * 60.0)


def main():
    all_results = []

    for filename in FILES:
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Cannot find {filename}. Run this script "
                "from your HeartRatePrediction project folder."
            )

        t, rgb = load_and_resample(filename)
        duration = t[-1] - t[0]

        print(f"\n{filename}")
        print(f"Resampled duration: {duration:.2f} seconds")
        print(f"Common sampling rate: {TARGET_FPS:.1f} Hz")

        window_samples = int(WINDOW_SECONDS * TARGET_FPS)
        step_samples = int(STEP_SECONDS * TARGET_FPS)

        session_rows = []

        for start in range(
            0, len(rgb) - window_samples + 1, step_samples
        ):
            end = start + window_samples
            segment = rgb[start:end]
            signals = make_signals(segment)

            row = {
                "session": filename,
                "window_start_seconds": round(
                    float(t[start]), 2
                ),
            }

            for method, signal in signals.items():
                row[method + "_BPM"] = estimate_bpm(signal)

            session_rows.append(row)

        if not session_rows:
            print("No complete analysis windows available.")
            continue

        all_results.extend(session_rows)
        session_df = pd.DataFrame(session_rows)

        print(session_df.to_string(index=False))

    results = pd.DataFrame(all_results)

    if results.empty:
        raise RuntimeError("No results were generated.")

    results.to_csv("repeatability_results.csv", index=False)

    summary_rows = []
    for session, group in results.groupby("session"):
        for method in ["Green", "Normalized_Green", "CHROM"]:
            values = group[method + "_BPM"].dropna()

            summary_rows.append({
                "session": session,
                "method": method,
                "valid_windows": len(values),
                "mean_BPM": values.mean() if len(values) else np.nan,
                "median_BPM": values.median() if len(values) else np.nan,
                "std_BPM": (
                    values.std(ddof=1) if len(values) > 1
                    else np.nan
                ),
                "min_BPM": values.min() if len(values) else np.nan,
                "max_BPM": values.max() if len(values) else np.nan,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("repeatability_summary.csv", index=False)

    print("\n\n=== REPEATABILITY SUMMARY ===")
    print(summary.round(2).to_string(index=False))

    print("\nSaved:")
    print("  repeatability_results.csv")
    print("  repeatability_summary.csv")
    print(
        "\nReminder: consistency is not the same as accuracy. "
        "Without a reference pulse measurement, we cannot "
        "determine which BPM estimate is correct."
    )


if __name__ == "__main__":
    main()

