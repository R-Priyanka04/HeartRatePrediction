
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch, detrend


CSV_FILE = "forehead_v2_diagnostics.csv"
WINDOW_LENGTHS = [16, 20, 24]
STEP_SECONDS = 2
MIN_BPM = 42
MAX_BPM = 180


def analyse_green(signal, fs):
    signal = detrend(signal)

    sos = butter(
        3,
        [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, signal)

    # Use the full available window for spectral estimation.
    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=len(filtered),
        noverlap=0,
        window="hann",
        detrend="constant",
    )

    valid = (
        (frequencies >= MIN_BPM / 60)
        & (frequencies <= MAX_BPM / 60)
    )

    indices = np.where(valid)[0]

    if len(indices) < 2:
        return None, None, None

    ranked = indices[np.argsort(power[indices])[::-1]]

    peak = ranked[0]
    second = ranked[1]

    ratio = power[peak] / max(power[second], 1e-15)

    return (
        frequencies[peak] * 60,
        frequencies[second] * 60,
        ratio,
    )


def main():
    df = pd.read_csv(CSV_FILE)

    df = df[
        (df["accepted"] == True)
        & (df["elapsed_seconds"] >= 8)
    ].dropna(
        subset=["elapsed_seconds", "mean_green"]
    ).sort_values("elapsed_seconds").copy()

    t = df["elapsed_seconds"].to_numpy(dtype=float)
    green = df["mean_green"].to_numpy(dtype=float)

    if len(t) < 100:
        raise ValueError("Not enough valid samples.")

    # Remove duplicate timestamps.
    keep = np.concatenate(([True], np.diff(t) > 0))
    t = t[keep]
    green = green[keep]

    fs_original = 1.0 / np.median(np.diff(t))
    fs = min(fs_original, 30.0)

    # Resample timestamps to a uniform grid before spectral analysis.
    uniform_t = np.arange(t[0], t[-1], 1.0 / fs)
    uniform_green = np.interp(uniform_t, t, green)

    # Remove the overall brightness baseline.
    normalized_green = (
        uniform_green / np.mean(uniform_green)
    ) - 1.0

    results = []

    print("\n--- GREEN LONG-WINDOW EXPERIMENT ---")
    print(f"Original median sampling rate: {fs_original:.2f} Hz")
    print(f"Uniform sampling rate: {fs:.2f} Hz")
    print("Experimental estimates only; no reference HR available.")

    for length in WINDOW_LENGTHS:
        estimates = []

        print(f"\n{length}-second windows")
        print("-" * 55)
        print(
            f"{'Window':<17}{'Peak BPM':>11}"
            f"{'Second BPM':>13}{'Ratio':>9}"
        )

        starts = np.arange(
            uniform_t[0],
            uniform_t[-1] - length + 0.001,
            STEP_SECONDS,
        )

        for start in starts:
            mask = (
                (uniform_t >= start)
                & (uniform_t < start + length)
            )

            segment = normalized_green[mask]

            if len(segment) < int(fs * length * 0.95):
                continue

            bpm, second, ratio = analyse_green(segment, fs)

            if bpm is None:
                continue

            midpoint = start + length / 2

            estimates.append((midpoint, bpm))

            results.append({
                "window_seconds": length,
                "start_seconds": start,
                "midpoint_seconds": midpoint,
                "bpm": bpm,
                "second_bpm": second,
                "peak_bin_ratio": ratio,
            })

            print(
                f"{start:.1f}-{start + length:.1f}s".ljust(17)
                + f"{bpm:>11.1f}"
                + f"{second:>13.1f}"
                + f"{ratio:>9.2f}"
            )

        if estimates:
            values = np.array([item[1] for item in estimates])

            print(
                f"Summary: mean={np.mean(values):.1f} BPM, "
                f"SD={np.std(values):.1f}, "
                f"range={np.min(values):.1f}-{np.max(values):.1f}"
            )

    if not results:
        print("No valid windows found.")
        return

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        "green_long_window_results.csv",
        index=False,
    )

    plt.figure(figsize=(11, 6))

    for length in WINDOW_LENGTHS:
        subset = result_df[
            result_df["window_seconds"] == length
        ]

        plt.plot(
            subset["midpoint_seconds"],
            subset["bpm"],
            marker="o",
            label=f"{length}s windows",
        )

    plt.xlabel("Window midpoint (seconds)")
    plt.ylabel("Estimated BPM")
    plt.title("Green Signal: Long-Window Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "green_long_window_comparison.png",
        dpi=150,
    )
    plt.close()

    print("\nSaved: green_long_window_results.csv")
    print("Saved: green_long_window_comparison.png")
    print(
        "Note: the peak-bin ratio compares adjacent frequency bins "
        "in some cases; it is not a validated confidence score."
    )


if __name__ == "__main__":
    main()

