
import os
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks, windows

FILES = [
    "rgb_session_1.csv",
    "rgb_session_2.csv",
    "rgb_session_3.csv",
]

FPS = 20.0
WINDOW_SECONDS = 20
STEP_SECONDS = 5
MIN_BPM = 30
MAX_BPM = 180


def load_rgb(filename):
    df = pd.read_csv(filename)

    required = ["time_seconds", "red", "green", "blue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename}: missing {missing}")

    df = df[required].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    df = df.sort_values("time_seconds")
    df = df.drop_duplicates(subset="time_seconds")

    t = df["time_seconds"].to_numpy(
        dtype=float, copy=True
    )
    rgb = df[["red", "green", "blue"]].to_numpy(
        dtype=float, copy=True
    )

    if len(t) < 100 or t[-1] <= t[0]:
        raise ValueError(f"Insufficient data: {filename}")

    t -= t[0]
    uniform_t = np.arange(0, t[-1], 1.0 / FPS)

    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, c])
        for c in range(3)
    ])

    return uniform_t, uniform_rgb


def candidate_peaks(signal, count=5):
    signal = detrend(np.asarray(signal, dtype=float))
    signal *= windows.hann(len(signal), sym=False)

    if np.std(signal) < 1e-10:
        return []

    power = np.abs(np.fft.rfft(signal)) ** 2
    bpm = np.fft.rfftfreq(
        len(signal), d=1.0 / FPS
    ) * 60.0

    valid = (bpm >= MIN_BPM) & (bpm <= MAX_BPM)
    if not np.any(valid):
        return []

    peaks, _ = find_peaks(power)
    peaks = [p for p in peaks if valid[p]]

    valid_indices = np.flatnonzero(valid)
    strongest = valid_indices[np.argmax(power[valid])]

    candidates = sorted(
        set(peaks + [int(strongest)]),
        key=lambda p: power[p],
        reverse=True,
    )[:count]

    max_power = max(float(power[candidates[0]]), 1e-12)

    return [
        {
            "bpm": round(float(bpm[p]), 1),
            "relative_power": round(
                float(power[p] / max_power), 3
            ),
        }
        for p in candidates
    ]


def main():
    rows = []
    window_n = int(WINDOW_SECONDS * FPS)
    step_n = int(STEP_SECONDS * FPS)

    for filename in FILES:
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        t, rgb = load_rgb(filename)
        previous_strongest = None

        print("\n" + "=" * 65)
        print(filename)

        for start in range(
            0, len(rgb) - window_n + 1, step_n
        ):
            green = rgb[start:start + window_n, 1]
            candidates = candidate_peaks(green)

            if not candidates:
                print(f"{t[start]:5.1f}s: no usable spectral candidates")
                continue

            strongest = candidates[0]["bpm"]
            jump = (
                abs(strongest - previous_strongest)
                if previous_strongest is not None
                else np.nan
            )

            row = {
                "session": filename,
                "window_start_seconds": round(float(t[start]), 2),
                "strongest_BPM": strongest,
                "change_from_previous_BPM": jump,
                "candidates": str(candidates),
            }
            rows.append(row)

            jump_text = (
                f"{jump:.1f} BPM"
                if np.isfinite(jump)
                else "first window"
            )

            print(
                f"\n{t[start]:5.1f}s | strongest={strongest:.1f} BPM"
                f" | change={jump_text}"
            )
            print("Candidates:", candidates)

            previous_strongest = strongest

    results = pd.DataFrame(rows)
    results.to_csv("candidate_peak_tracking.csv", index=False)

    print("\nSaved candidate_peak_tracking.csv")
    print(
        "Note: continuity is diagnostic only. It does not establish "
        "which frequency corresponds to the true pulse."
    )


if __name__ == "__main__":
    main()

