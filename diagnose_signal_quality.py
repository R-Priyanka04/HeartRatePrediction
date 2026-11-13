
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import detrend, windows

FILES = [
    "rgb_session_1.csv",
    "rgb_session_2.csv",
    "rgb_session_3.csv",
]

FPS = 20.0
WINDOW_SECONDS = 20
STEP_SECONDS = 5

# Candidate pulse-frequency range for diagnostic purposes only.
MIN_BPM = 40
MAX_BPM = 180

# These are exploratory quality flags, not validated thresholds.
MAX_JUMP_PERCENTILE = 99.0
MAX_DRIFT_RATIO = 0.8
MIN_PEAK_RATIO = 1.5


def load_recording(filename):
    df = pd.read_csv(filename)

    required = ["time_seconds", "red", "green", "blue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{filename}: missing columns {missing}"
        )

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
        raise ValueError(f"Not enough valid data: {filename}")

    t -= t[0]
    uniform_t = np.arange(0, t[-1], 1.0 / FPS)

    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, c])
        for c in range(3)
    ])

    return uniform_t, uniform_rgb


def analyze_window(rgb):
    green = rgb[:, 1].astype(float)

    # Robustly centre the signal.
    centre = np.median(green)
    scale = max(abs(centre), 1e-9)
    normalized = (green - centre) / scale

    # First differences help reveal abrupt changes.
    differences = np.diff(normalized)
    jump_size = (
        float(np.percentile(np.abs(differences), MAX_JUMP_PERCENTILE))
        if len(differences)
        else 0.0
    )

    # Compare slow baseline variation with detrended variation.
    time = np.arange(len(green)) / FPS
    linear_fit = np.polyval(
        np.polyfit(time, normalized, 1), time
    )
    detrended_signal = detrend(normalized)

    raw_variation = np.std(normalized)
    residual_variation = np.std(detrended_signal)

    drift_ratio = (
        np.std(linear_fit) / max(raw_variation, 1e-12)
    )

    # Inspect spectral peaks after linear detrending.
    signal = detrended_signal * windows.hann(
        len(detrended_signal), sym=False
    )

    power = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(
        len(signal), d=1.0 / FPS
    )
    bpm = frequencies * 60.0

    pulse_mask = (bpm >= MIN_BPM) & (bpm <= MAX_BPM)
    pulse_power = power[pulse_mask]
    pulse_bpm = bpm[pulse_mask]

    if len(pulse_power) == 0 or np.max(pulse_power) <= 1e-12:
        peak_bpm = np.nan
        peak_ratio = 0.0
    else:
        peak_index = int(np.argmax(pulse_power))
        peak_bpm = float(pulse_bpm[peak_index])

        sorted_power = np.sort(pulse_power)
        second_peak = (
            sorted_power[-2] if len(sorted_power) > 1 else 0.0
        )
        peak_ratio = float(
            pulse_power[peak_index] / max(second_peak, 1e-12)
        )

    return {
        "green_jump_percentile": jump_size,
        "linear_drift_ratio": float(drift_ratio),
        "green_residual_std": float(residual_variation),
        "candidate_peak_BPM": peak_bpm,
        "strongest_to_second_bin_ratio": peak_ratio,
    }


def main():
    all_rows = []

    for filename in FILES:
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Missing {filename}. Run this script in your project folder."
            )

        t, rgb = load_recording(filename)

        window_samples = int(WINDOW_SECONDS * FPS)
        step_samples = int(STEP_SECONDS * FPS)

        print("\n" + "=" * 72)
        print(filename)

        for start in range(
            0, len(rgb) - window_samples + 1, step_samples
        ):
            segment = rgb[start:start + window_samples]
            metrics = analyze_window(segment)

            row = {
                "session": filename,
                "window_start_seconds": round(float(t[start]), 2),
                **metrics,
            }

            flags = []

            # These are exploratory indicators, not clinical criteria.
            if metrics["green_jump_percentile"] > 0.03:
                flags.append("possible abrupt change")

            if metrics["linear_drift_ratio"] > MAX_DRIFT_RATIO:
                flags.append("possible baseline drift")

            if metrics["strongest_to_second_bin_ratio"] < MIN_PEAK_RATIO:
                flags.append("competing spectral bins")

            row["flags"] = "; ".join(flags) if flags else "no rule triggered"
            all_rows.append(row)

    results = pd.DataFrame(all_rows)

    results.to_csv(
        "signal_quality_diagnostics.csv", index=False
    )

    print("\n=== WINDOW DIAGNOSTICS ===")
    print(results.round(4).to_string(index=False))

    print("\n=== FLAG COUNTS ===")
    print(
        results.groupby("session")["flags"]
        .apply(
            lambda series: series.ne("no rule triggered").sum()
        )
        .to_string()
    )

    # Plot the candidate peak and diagnostic flags by window.
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True
    )

    for ax, filename in zip(axes, FILES):
        subset = results[results["session"] == filename]

        ax.plot(
            subset["window_start_seconds"],
            subset["candidate_peak_BPM"],
            marker="o",
        )
        ax.set_ylabel("Candidate BPM")
        ax.set_title(filename)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Window start (seconds)")
    fig.tight_layout()
    fig.savefig(
        "signal_quality_diagnostics.png", dpi=150
    )
    plt.close(fig)

    print("\nSaved:")
    print("  signal_quality_diagnostics.csv")
    print("  signal_quality_diagnostics.png")
    print(
        "\nImportant: thresholds are exploratory. "
        "Flags do not prove a window is unusable, and an unflagged "
        "window is not necessarily accurate."
    )


if __name__ == "__main__":
    main()

