

import os
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks, windows

FILES = [
    "rgb_session_1.csv",
    "rgb_session_2.csv",
    "rgb_session_3.csv",
]

TARGET_FPS = 20.0
WINDOW_SECONDS = 20
STEP_SECONDS = 5

# Wider diagnostic range than the previous 42–180 BPM search.
MIN_BPM = 30
MAX_BPM = 180


def load_rgb(filename):
    df = pd.read_csv(filename)

    required = ["time_seconds", "red", "green", "blue"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{filename}: missing columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df[required].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    df = df.sort_values("time_seconds")
    df = df.drop_duplicates(subset="time_seconds")

    # copy=True fixes the read-only array error.
    t = df["time_seconds"].to_numpy(dtype=float, copy=True)
    rgb = df[["red", "green", "blue"]].to_numpy(
        dtype=float, copy=True
    )

    if len(t) < 100 or t[-1] <= t[0]:
        raise ValueError(f"Not enough valid data in {filename}")

    # Shift timestamps to start at zero.
    t -= t[0]

    # Resample every session to a common 20 Hz grid.
    uniform_t = np.arange(
        0, t[-1], 1.0 / TARGET_FPS
    )

    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, channel])
        for channel in range(3)
    ])

    return uniform_t, uniform_rgb


def get_signals(rgb):
    red, green, blue = rgb.T

    # CHROM uses colour channels normalised by their window means.
    means = np.mean(rgb, axis=0)
    means = np.where(
        np.abs(means) < 1e-12, 1e-12, means
    )

    r, g, b = (rgb / means).T

    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b

    alpha = np.std(x) / max(np.std(y), 1e-12)
    chrom = x - alpha * y

    return {
        "Green": green,
        "CHROM": chrom,
    }


def top_peaks(signal, count=5):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 10 or not np.all(np.isfinite(signal)):
        return []

    signal = detrend(signal, type="linear")
    signal = signal * windows.hann(
        len(signal), sym=False
    )

    if np.std(signal) < 1e-10:
        return []

    power = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(
        len(signal), d=1.0 / TARGET_FPS
    )
    bpm = frequencies * 60.0

    valid = (
        (bpm >= MIN_BPM)
        & (bpm <= MAX_BPM)
    )

    valid_indices = np.flatnonzero(valid)

    if len(valid_indices) == 0:
        return []

    # Identify local peaks inside the permitted BPM range.
    peaks, _ = find_peaks(power)
    peaks = [p for p in peaks if valid[p]]

    # Also include the strongest bin, even if it is not a
    # local maximum or lies at the boundary of the search range.
    strongest = valid_indices[
        np.argmax(power[valid])
    ]

    candidates = sorted(
        set(peaks + [int(strongest)]),
        key=lambda p: power[p],
        reverse=True,
    )[:count]

    max_power = max(
        float(power[candidates[0]]), 1e-12
    )

    return [
        (
            round(float(bpm[p]), 1),
            round(float(power[p] / max_power), 3),
        )
        for p in candidates
    ]


def main():
    all_rows = []

    window_samples = int(
        WINDOW_SECONDS * TARGET_FPS
    )
    step_samples = int(
        STEP_SECONDS * TARGET_FPS
    )

    for filename in FILES:
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Cannot find {filename}. "
                "Run this script from your HeartRatePrediction folder."
            )

        t, rgb = load_rgb(filename)

        print("\n" + "=" * 68)
        print(filename)
        print(f"Duration: {t[-1]:.2f} seconds")
        print(f"Sampling rate: {TARGET_FPS:.1f} Hz")
        print("Window start | Green peaks BPM (relative power)")
        print("             | CHROM peaks BPM (relative power)")

        for start in range(
            0,
            len(rgb) - window_samples + 1,
            step_samples,
        ):
            end = start + window_samples
            segment = rgb[start:end]
            signals = get_signals(segment)

            green_peaks = top_peaks(
                signals["Green"]
            )
            chrom_peaks = top_peaks(
                signals["CHROM"]
            )

            start_time = float(t[start])

            print(f"\n{start_time:5.1f} s")
            print("  Green:", green_peaks)
            print("  CHROM:", chrom_peaks)

            all_rows.append({
                "session": filename,
                "window_start_seconds": round(
                    start_time, 2
                ),
                "green_peaks": str(green_peaks),
                "chrom_peaks": str(chrom_peaks),
            })

    results = pd.DataFrame(all_rows)
    results.to_csv(
        "repeatability_spectral_peaks.csv",
        index=False,
    )

    print("\nSaved repeatability_spectral_peaks.csv")
    print(
        "\nRelative power is normalised to the strongest "
        "peak within each window. It is not a probability "
        "or a confidence score."
    )


if __name__ == "__main__":
    main()

