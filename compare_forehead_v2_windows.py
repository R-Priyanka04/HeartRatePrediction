
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch, detrend


CSV_FILE = "forehead_v2_diagnostics.csv"
MIN_BPM = 42
MAX_BPM = 180

WINDOW_LENGTHS = [8, 12, 16]
STEP_SECONDS = 2


def estimate_bpm(signal, fs):
    signal = detrend(signal)

    sos = butter(
        3,
        [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, signal)

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=min(len(filtered), int(fs * 8)),
    )

    valid = (
        (frequencies >= MIN_BPM / 60)
        & (frequencies <= MAX_BPM / 60)
    )

    indices = np.where(valid)[0]

    if len(indices) < 2:
        return None, None, None

    order = indices[np.argsort(power[indices])[::-1]]
    peak = order[0]
    second = order[1]

    ratio = power[peak] / max(power[second], 1e-15)

    return (
        frequencies[peak] * 60,
        frequencies[second] * 60,
        ratio,
    )


def main():
    df = pd.read_csv(CSV_FILE)

    # Keep accepted frames after the initial movement.
    df = df[
        (df["accepted"] == True)
        & (df["elapsed_seconds"] >= 8)
    ].dropna(
        subset=[
            "elapsed_seconds",
            "mean_red",
            "mean_green",
            "mean_blue",
        ]
    ).copy()

    df = df.sort_values("elapsed_seconds")

    t = df["elapsed_seconds"].to_numpy(dtype=float)

    if len(t) < 100:
        raise ValueError("Not enough valid samples.")

    # Remove duplicate timestamps.
    unique = np.concatenate(([True], np.diff(t) > 0))
    df = df.iloc[np.flatnonzero(unique)].copy()
    t = df["elapsed_seconds"].to_numpy(dtype=float)

    # Resample RGB means onto a uniform time grid.
    fs_original = 1.0 / np.median(np.diff(t))
    fs = min(fs_original, 30.0)

    uniform_t = np.arange(t[0], t[-1], 1.0 / fs)

    red = np.interp(
        uniform_t, t, df["mean_red"].to_numpy(dtype=float)
    )
    green = np.interp(
        uniform_t, t, df["mean_green"].to_numpy(dtype=float)
    )
    blue = np.interp(
        uniform_t, t, df["mean_blue"].to_numpy(dtype=float)
    )

    # Normalize each colour channel.
    rn = red / np.mean(red) - 1.0
    gn = green / np.mean(green) - 1.0
    bn = blue / np.mean(blue) - 1.0

    # Simplified CHROM-like signal.
    x = 3.0 * rn - 2.0 * gn
    y = 1.5 * rn + gn - 1.5 * bn
    chrom = x - (
        np.std(x) / max(np.std(y), 1e-12)
    ) * y

    signals = {
        "Green": gn,
        "CHROM-like": chrom,
    }

    results = []

    print("\n--- FOREHEAD V2 WINDOW COMPARISON ---")
    print(f"Original median sampling rate: {fs_original:.2f} Hz")
    print(f"Uniform sampling rate: {fs:.2f} Hz")
    print(f"Analyzed duration: {uniform_t[-1] - uniform_t[0]:.2f} s")
    print("All estimates are experimental, not validated.\n")

    for method, signal in signals.items():
        print(f"\n{method}")
        print("-" * 66)
        print(
            f"{'Window':<14}{'Peak BPM':>12}"
            f"{'Second BPM':>14}{'Ratio':>10}"
        )

        for length in WINDOW_LENGTHS:
            starts = np.arange(
                uniform_t[0],
                uniform_t[-1] - length + 0.001,
                STEP_SECONDS,
            )

            estimates = []

            for start in starts:
                mask = (
                    (uniform_t >= start)
                    & (uniform_t < start + length)
                )

                if np.count_nonzero(mask) < int(fs * 6):
                    continue

                bpm, second, ratio = estimate_bpm(
                    signal[mask], fs
                )

                if bpm is None:
                    continue

                estimates.append((start, bpm, second, ratio))

                results.append({
                    "method": method,
                    "window_seconds": length,
                    "start_seconds": start,
                    "midpoint_seconds": start + length / 2,
                    "bpm": bpm,
                    "second_bpm": second,
                    "power_ratio": ratio,
                })

                print(
                    f"{start:.1f}-{start + length:.1f}s".ljust(14)
                    + f"{bpm:>12.1f}"
                    + f"{second:>14.1f}"
                    + f"{ratio:>10.2f}"
                )

            if estimates:
                values = np.array([item[1] for item in estimates])

                print(
                    f"  {length}s summary: "
                    f"mean={np.mean(values):.1f}, "
                    f"SD={np.std(values):.1f}, "
                    f"range={np.min(values):.1f}-{np.max(values):.1f} BPM"
                )

    if not results:
        print("No valid windows found.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "forehead_v2_window_results.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True
    )

    for ax, method in zip(axes, signals.keys()):
        subset = results_df[
            results_df["method"] == method
        ]

        for length in WINDOW_LENGTHS:
            part = subset[
                subset["window_seconds"] == length
            ]

            if not part.empty:
                ax.plot(
                    part["midpoint_seconds"],
                    part["bpm"],
                    marker="o",
                    label=f"{length}s windows",
                )

        ax.set_title(method)
        ax.set_ylabel("Estimated BPM")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Window midpoint (seconds)")
    fig.tight_layout()
    fig.savefig(
        "forehead_v2_window_comparison.png",
        dpi=150,
    )
    plt.close(fig)

    print("\nSaved detailed results:")
    print("forehead_v2_window_results.csv")
    print("Saved graph:")
    print("forehead_v2_window_comparison.png")
    print(
        "\nConsistency is not proof of accuracy. "
        "An independent reference is needed for validation."
    )


if __name__ == "__main__":
    main()

