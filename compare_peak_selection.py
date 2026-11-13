
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
MIN_BPM = 40
MAX_BPM = 180

# Experimental parameters; these are not validated thresholds.
CONTINUITY_WEIGHT = 0.04
MIN_RELATIVE_POWER = 0.50


def load_rgb(filename):
    df = pd.read_csv(filename)

    required = ["time_seconds", "red", "green", "blue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

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
        raise ValueError(f"Insufficient data in {filename}")

    t -= t[0]
    uniform_t = np.arange(0, t[-1], 1.0 / FPS)

    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, c])
        for c in range(3)
    ])

    return uniform_t, uniform_rgb


def get_candidates(green):
    signal = detrend(np.asarray(green, dtype=float))
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

    indices = sorted(
        set(peaks + [int(strongest)]),
        key=lambda p: power[p],
        reverse=True,
    )[:5]

    max_power = max(float(power[indices[0]]), 1e-12)

    return [
        {
            "bpm": float(bpm[p]),
            "relative_power": float(power[p] / max_power),
        }
        for p in indices
    ]


def main():
    rows = []
    window_n = int(WINDOW_SECONDS * FPS)
    step_n = int(STEP_SECONDS * FPS)

    for filename in FILES:
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        t, rgb = load_rgb(filename)
        previous_selected = None

        print("\n" + "=" * 65)
        print(filename)

        for start in range(
            0, len(rgb) - window_n + 1, step_n
        ):
            candidates = get_candidates(
                rgb[start:start + window_n, 1]
            )

            if not candidates:
                print(f"{t[start]:.1f}s: no candidates")
                rows.append({
                    "session": filename,
                    "window_start_seconds": float(t[start]),
                    "strongest_BPM": np.nan,
                    "continuity_BPM": np.nan,
                    "status": "NO CANDIDATES",
                })
                continue

            strongest = candidates[0]["bpm"]

            # Only consider candidates with substantial power
            # relative to the strongest peak.
            eligible = [
                c for c in candidates
                if c["relative_power"] >= MIN_RELATIVE_POWER
            ]

            if previous_selected is None:
                selected = candidates[0]
            elif eligible:
                # Log power rewards spectral strength.
                # The penalty discourages large jumps.
                selected = max(
                    eligible,
                    key=lambda c: (
                        np.log(max(c["relative_power"], 1e-12))
                        - CONTINUITY_WEIGHT
                        * abs(c["bpm"] - previous_selected)
                    ),
                )
            else:
                selected = None

            if selected is None:
                selected_bpm = np.nan
                status = "REJECTED: no sufficiently strong candidate"
            else:
                selected_bpm = selected["bpm"]

                powers = sorted(
                    [c["relative_power"] for c in candidates],
                    reverse=True,
                )
                ratio = (
                    powers[0] / max(powers[1], 1e-12)
                    if len(powers) > 1
                    else np.inf
                )

                if ratio < 1.2:
                    status = "AMBIGUOUS: competing peaks"
                else:
                    status = "CANDIDATE: not reference-validated"

                previous_selected = selected_bpm

            print(
                f"{t[start]:5.1f}s | strongest={strongest:5.1f}"
                f" | continuity={selected_bpm!s:>5}"
                f" | {status}"
            )

            rows.append({
                "session": filename,
                "window_start_seconds": round(float(t[start]), 2),
                "strongest_BPM": strongest,
                "continuity_BPM": selected_bpm,
                "status": status,
                "candidates": str(candidates),
            })

    results = pd.DataFrame(rows)
    results.to_csv("peak_selection_comparison.csv", index=False)

    print("\nSaved peak_selection_comparison.csv")
    print(
        "\nContinuity-based estimates are experimental only. "
        "They must not be treated as accurate heart rates "
        "without comparison against a reference device."
    )


if __name__ == "__main__":
    main()

