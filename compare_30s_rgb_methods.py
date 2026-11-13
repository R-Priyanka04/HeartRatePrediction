

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch, detrend


CSV_FILE = "forehead_v2_diagnostics.csv"

START = 10.0
END = 40.0

LOW_HZ = 0.7
HIGH_HZ = 3.0


def calculate_spectrum(signal, fs):
    """Filter a signal and return its heart-rate spectrum."""
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 100:
        raise ValueError("Signal window is too short.")

    signal = detrend(signal)

    sos = butter(
        3,
        [LOW_HZ, HIGH_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, signal)

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=len(filtered),
        noverlap=0,
        window="hann",
    )

    valid = (
        (frequencies >= LOW_HZ)
        & (frequencies <= HIGH_HZ)
    )

    bpm = frequencies[valid] * 60
    power = power[valid]

    if len(power) == 0 or np.max(power) <= 0:
        raise ValueError("No usable spectral power in the heart-rate band.")

    return bpm, power


def print_top_peaks(name, bpm, power, count=5):
    """Print the strongest frequency bins for one method."""
    top_indices = np.argsort(power)[::-1][:count]

    print(f"\n{name}: strongest frequency bins")

    for index in top_indices:
        print(
            f"  {bpm[index]:6.2f} BPM | "
            f"power={power[index]:.6g} | "
            f"relative={power[index] / np.max(power):.3f}"
        )


def main():
    df = pd.read_csv(CSV_FILE)

    required = [
        "elapsed_seconds",
        "accepted",
        "mean_red",
        "mean_green",
        "mean_blue",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df = df[
        (df["accepted"] == True)
        & (df["elapsed_seconds"] >= START)
        & (df["elapsed_seconds"] <= END)
    ].dropna(
        subset=[
            "elapsed_seconds",
            "mean_red",
            "mean_green",
            "mean_blue",
        ]
    ).sort_values("elapsed_seconds")

    t = df["elapsed_seconds"].to_numpy(dtype=float)

    rgb = df[
        ["mean_red", "mean_green", "mean_blue"]
    ].to_numpy(dtype=float)

    # Remove duplicate or non-increasing timestamps.
    keep = np.concatenate(([True], np.diff(t) > 0))
    t = t[keep]
    rgb = rgb[keep]

    if len(t) < 100:
        raise ValueError("Not enough valid RGB samples in this interval.")

    fs = 1.0 / np.median(np.diff(t))

    # Resample onto a uniform time grid.
    uniform_t = np.arange(START, END, 1.0 / fs)

    rgb_uniform = np.column_stack([
        np.interp(uniform_t, t, rgb[:, channel])
        for channel in range(3)
    ])

    red, green, blue = rgb_uniform.T

    # Method 1: Green-channel baseline.
    green_signal = green / (np.mean(green) + 1e-12) - 1

    # Method 2: Green divided by total RGB intensity.
    rgb_sum = red + green + blue

    normalized_green = green / (rgb_sum + 1e-12)
    normalized_green -= np.mean(normalized_green)

    # Method 3: CHROM-style color combination.
    normalized_rgb = (
        rgb_uniform / (np.mean(rgb_uniform, axis=0) + 1e-12)
    ) - 1

    r, g, b = normalized_rgb.T

    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b

    chrom = x - (
        np.std(x) / (np.std(y) + 1e-12)
    ) * y

    signals = {
        "Green": green_signal,
        "Normalized green": normalized_green,
        "CHROM": chrom,
    }

    print("\n--- 30-SECOND RGB SPECTRAL COMPARISON ---")
    print(f"Analysis interval: {START:.1f}-{END:.1f} seconds")
    print(f"Estimated sampling rate: {fs:.2f} Hz")
    print(f"Samples analysed: {len(uniform_t)}")

    plt.figure(figsize=(12, 7))
    results = []

    for name, signal in signals.items():
        bpm, power = calculate_spectrum(signal, fs)

        peak_index = int(np.argmax(power))
        peak_bpm = float(bpm[peak_index])

        print_top_peaks(name, bpm, power)

        # Normalize each method independently for plotting only.
        relative_power = power / (np.max(power) + 1e-12)

        plt.plot(
            bpm,
            relative_power,
            marker="o",
            markersize=3,
            label=f"{name} (peak {peak_bpm:.1f} BPM)",
        )

        results.append({
            "method": name,
            "peak_bpm": peak_bpm,
            "peak_power": float(power[peak_index]),
            "sampling_rate_hz": fs,
            "window_start_seconds": START,
            "window_end_seconds": END,
        })

    plt.xlabel("Frequency (BPM)")
    plt.ylabel("Normalized power within each method")
    plt.title("30-second RGB Method Spectra: 10–40 seconds")
    plt.ylim(0, 1.05)
    plt.xlim(LOW_HZ * 60, HIGH_HZ * 60)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_file = "rgb_spectra_30s.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()

    results_file = "rgb_spectra_30s_summary.csv"
    pd.DataFrame(results).to_csv(results_file, index=False)

    print(f"\nSaved graph: {plot_file}")
    print(f"Saved summary: {results_file}")
    print(
        "\nThese are experimental spectral estimates, "
        "not validated heart-rate measurements."
    )
    print("No reference heart rate was available for this recording.")


if __name__ == "__main__":
    main()

