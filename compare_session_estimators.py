
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend, welch, windows

FILES = [
    "rgb_session_1.csv",
    "rgb_session_2.csv",
    "rgb_session_3.csv",
]

TARGET_FPS = 20.0
WINDOW_LENGTHS = [20, 30]
STEP_SECONDS = 5
MIN_BPM = 42.0
MAX_BPM = 180.0


def load_recording(filename):
    df = pd.read_csv(filename)

    required = ["time_seconds", "red", "green", "blue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

    df = df[required].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values("time_seconds")
    df = df.drop_duplicates("time_seconds")

    t = df["time_seconds"].to_numpy(dtype=float)
    rgb = df[["red", "green", "blue"]].to_numpy(dtype=float)

    if len(t) < 100 or t[-1] <= t[0]:
        raise ValueError(f"Insufficient valid data in {filename}")

    t = t - t[0]
    uniform_t = np.arange(0, t[-1], 1.0 / TARGET_FPS)

    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, i])
        for i in range(3)
    ])

    return uniform_t, uniform_rgb


def prepare_green(signal):
    signal = detrend(np.asarray(signal, dtype=float))

    if not np.all(np.isfinite(signal)) or np.std(signal) < 1e-8:
        return None

    sos = butter(
        3,
        [MIN_BPM / 60.0, MAX_BPM / 60.0],
        btype="bandpass",
        fs=TARGET_FPS,
        output="sos",
    )

    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError:
        return None

    if not np.all(np.isfinite(filtered)) or np.std(filtered) < 1e-8:
        return None

    return filtered


def estimate_peak(signal, method):
    n = len(signal)

    # Apply the same window to both estimators.
    weighted = signal * windows.hann(n, sym=False)

    if method == "FFT":
        spectrum = np.abs(np.fft.rfft(weighted)) ** 2
        frequencies = np.fft.rfftfreq(
            n, d=1.0 / TARGET_FPS
        )

    elif method == "Welch":
        frequencies, spectrum = welch(
            signal,
            fs=TARGET_FPS,
            window="hann",
            nperseg=n,
            noverlap=0,
            nfft=max(n, 4096),
            detrend="constant",
            scaling="density",
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    valid = (
        (frequencies >= MIN_BPM / 60.0)
        & (frequencies <= MAX_BPM / 60.0)
    )

    if not np.any(valid):
        return np.nan

    valid_frequencies = frequencies[valid]
    valid_power = spectrum[valid]

    if not np.all(np.isfinite(valid_power)) or np.max(valid_power) <= 0:
        return np.nan

    peak = np.argmax(valid_power)
    return float(valid_frequencies[peak] * 60.0)


def main():
    rows = []

    for filename in FILES:
        t, rgb = load_recording(filename)
        print(f"\n{filename} | duration: {t[-1]:.1f}s")

        for window_seconds in WINDOW_LENGTHS:
            window_samples = int(window_seconds * TARGET_FPS)
            step_samples = int(STEP_SECONDS * TARGET_FPS)

            for start in range(
                0, len(rgb) - window_samples + 1, step_samples
            ):
                end = start + window_samples
                segment = rgb[start:end]

                green = segment[:, 1]
                filtered = prepare_green(green)

                if filtered is None:
                    continue

                for method in ["FFT", "Welch"]:
                    bpm = estimate_peak(filtered, method)

                    rows.append({
                        "session": filename,
                        "window_seconds": window_seconds,
                        "start_seconds": round(t[start], 2),
                        "method": method,
                        "bpm": bpm,
                    })

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError("No estimates produced.")

    results.to_csv("estimator_comparison.csv", index=False)

    valid = results.dropna(subset=["bpm"])

    summary = (
        valid.groupby(["session", "window_seconds", "method"])["bpm"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    summary.to_csv("estimator_comparison_summary.csv", index=False)

    print("\n=== ESTIMATOR COMPARISON ===")
    print(summary.round(2).to_string(index=False))

    print("\nSaved:")
    print("  estimator_comparison.csv")
    print("  estimator_comparison_summary.csv")
    print(
        "\nInterpretation: lower variability indicates greater "
        "repeatability only; it does not establish accuracy."
    )


if __name__ == "__main__":
    main()

