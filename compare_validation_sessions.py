
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch, detrend


FILES = [
    "validation_rgb_1.csv",
    "validation_rgb_2.csv",
    "validation_rgb_3.csv",
]

WINDOWS = [
    (5, 25),
    (15, 35),
    (25, 45),
    (35, 55),
]

MIN_BPM = 42
MAX_BPM = 180


def load_csv(filename):
    """Load timestamps and RGB forehead measurements."""
    with open(filename, newline="") as file:
        rows = list(csv.DictReader(file))

    if len(rows) < 100:
        raise ValueError(f"Not enough samples in {filename}")

    t = np.array(
        [float(row["time_seconds"]) for row in rows]
    )
    red = np.array([float(row["red"]) for row in rows])
    green = np.array([float(row["green"]) for row in rows])
    blue = np.array([float(row["blue"]) for row in rows])

    # Sort by timestamp if necessary.
    order = np.argsort(t)

    return (
        t[order],
        red[order],
        green[order],
        blue[order],
    )


def build_signals(red, green, blue):
    """
    Build experimental green and CHROM-like signals.

    The CHROM-like scaling factor is calculated separately
    for each analysis window, not from the full recording.
    """
    rn = red / max(np.mean(red), 1e-8) - 1
    gn = green / max(np.mean(green), 1e-8) - 1
    bn = blue / max(np.mean(blue), 1e-8) - 1

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    alpha = np.std(x) / max(np.std(y), 1e-12)
    chrom = x - alpha * y

    return {
        "Green": gn,
        "CHROM-like": chrom,
    }


def estimate_bpm(signal, fs):
    """Estimate the dominant frequency and peak separation."""
    signal = detrend(signal)

    # Bandpass range: 42 to 180 BPM.
    sos = butter(
        3,
        [MIN_BPM / 60, MAX_BPM / 60],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, signal)

    # Use the complete analysis window where possible.
    nperseg = min(len(filtered), int(fs * 20))

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )

    valid = (
        (frequencies >= MIN_BPM / 60)
        & (frequencies <= MAX_BPM / 60)
    )

    indices = np.where(valid)[0]

    if len(indices) == 0:
        return None, None

    peak_index = indices[np.argmax(power[indices])]
    peak_bpm = frequencies[peak_index] * 60

    # Compare with peaks at least 6 BPM away, so adjacent
    # frequency bins are not automatically treated as
    # separate competing peaks.
    competing = indices[
        np.abs(frequencies[indices] * 60 - peak_bpm) >= 6
    ]

    if len(competing) == 0:
        ratio = float("nan")
    else:
        second_index = competing[
            np.argmax(power[competing])
        ]
        ratio = power[peak_index] / max(
            power[second_index], 1e-15
        )

    return peak_bpm, ratio


def main():
    all_results = []

    print("THREE-SESSION VALIDATION")
    print("Experimental estimates only; no reference pulse is available.")
    print("=" * 82)

    # Analyze each recording independently.
    for filename in FILES:
        if not os.path.isfile(filename):
            print(f"\nERROR: Missing file: {filename}")
            print("Check that all three CSV files are in this folder.")
            return

        t, red, green, blue = load_csv(filename)

        duration = t[-1] - t[0]
        fs = 1 / np.median(np.diff(t))

        print(
            f"\n{filename}: {len(t)} samples, "
            f"{duration:.2f} seconds, {fs:.2f} Hz"
        )

        for start, end in WINDOWS:
            mask = (t >= start) & (t < end)

            if np.count_nonzero(mask) < int(fs * 15):
                print(f"  {start}-{end}s: insufficient samples")
                continue

            segment_t = t[mask]
            segment_fs = 1 / np.median(np.diff(segment_t))

            # Recalculate normalization and CHROM scaling
            # independently for every analysis window.
            signals = build_signals(
                red[mask],
                green[mask],
                blue[mask],
            )

            for method, signal in signals.items():
                bpm, ratio = estimate_bpm(signal, segment_fs)

                if bpm is None:
                    continue

                all_results.append({
                    "file": filename,
                    "method": method,
                    "start": start,
                    "end": end,
                    "bpm": bpm,
                    "ratio": ratio,
                })

                ratio_text = (
                    f"{ratio:.2f}"
                    if np.isfinite(ratio)
                    else "N/A"
                )

                print(
                    f"  {start:02d}-{end:02d}s | "
                    f"{method:<11} | "
                    f"{bpm:6.1f} BPM | "
                    f"peak ratio: {ratio_text}"
                )

    print("\nSUMMARY BY SESSION AND METHOD")
    print("=" * 82)

    for filename in FILES:
        for method in ("Green", "CHROM-like"):
            values = [
                result["bpm"]
                for result in all_results
                if result["file"] == filename
                and result["method"] == method
            ]

            if values:
                print(
                    f"{filename:<22} {method:<11} "
                    f"mean={np.mean(values):5.1f} BPM | "
                    f"SD={np.std(values):4.1f} BPM | "
                    f"range={min(values):.1f}-{max(values):.1f}"
                )

    # Plot the estimates from each session.
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 10), sharex=True
    )

    for ax, filename in zip(axes, FILES):
        for method in ("Green", "CHROM-like"):
            items = [
                result for result in all_results
                if result["file"] == filename
                and result["method"] == method
            ]

            if items:
                midpoints = [
                    (result["start"] + result["end"]) / 2
                    for result in items
                ]
                bpms = [
                    result["bpm"] for result in items
                ]

                ax.plot(
                    midpoints,
                    bpms,
                    marker="o",
                    label=method,
                )

        ax.set_title(filename)
        ax.set_ylabel("Estimated BPM")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Window midpoint (seconds)")
    fig.tight_layout()

    output_plot = "validation_sessions_comparison.png"
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)

    print(f"\nSaved graph: {output_plot}")
    print(
        "\nReminder: repeatability and peak separation do not "
        "establish actual heart-rate accuracy."
    )


if __name__ == "__main__":
    main()
