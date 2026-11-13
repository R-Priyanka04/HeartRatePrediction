
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend, welch


CSV_FILE = "forehead_v2_diagnostics.csv"
WINDOWS = [(10, 30), (28, 48)]


def main():
    df = pd.read_csv(CSV_FILE)

    df = df[
        (df["accepted"] == True)
        & df["elapsed_seconds"].notna()
        & df["mean_green"].notna()
    ].sort_values("elapsed_seconds")

    df = df.drop_duplicates(
        subset="elapsed_seconds",
        keep="first",
    )

    t = df["elapsed_seconds"].to_numpy(float)
    green = df["mean_green"].to_numpy(float)

    if len(t) < 100:
        raise ValueError("Not enough valid samples.")

    fs = 1.0 / np.median(np.diff(t))
    uniform_t = np.arange(t[0], t[-1], 1.0 / fs)
    uniform_green = np.interp(uniform_t, t, green)

    sos = butter(
        3,
        [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 8),
    )

    for row, (start, end) in enumerate(WINDOWS):
        mask = (
            (uniform_t >= start)
            & (uniform_t < end)
        )

        signal = uniform_green[mask]
        times = uniform_t[mask]

        if len(signal) < 100:
            print(f"Skipping {start}-{end}s: insufficient samples.")
            continue

        filtered = sosfiltfilt(
            sos,
            detrend(signal),
        )

        frequencies, power = welch(
            filtered,
            fs=fs,
            nperseg=len(filtered),
            window="hann",
        )

        valid = (
            (frequencies >= 0.7)
            & (frequencies <= 3.0)
        )

        valid_freq = frequencies[valid]
        valid_power = power[valid]

        peak_indices = np.argsort(valid_power)[::-1][:5]

        print(f"\nWindow {start}-{end} seconds")
        print("Top five frequency bins (BPM, relative power):")

        for idx in peak_indices:
            print(
                f"{valid_freq[idx] * 60:.2f} BPM, "
                f"{valid_power[idx]:.6g}"
            )

        axes[row, 0].plot(times, signal)
        axes[row, 0].set_title(
            f"Raw green channel: {start}-{end}s"
        )
        axes[row, 0].set_xlabel("Time (seconds)")
        axes[row, 0].set_ylabel("Mean green value")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(
            valid_freq * 60,
            valid_power,
        )
        axes[row, 1].set_title(
            f"Filtered signal spectrum: {start}-{end}s"
        )
        axes[row, 1].set_xlabel("Frequency (BPM)")
        axes[row, 1].set_ylabel("Power")
        axes[row, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        "green_signal_spectrum_comparison.png",
        dpi=150,
    )
    plt.close(fig)

    print(
        "\nSaved: green_signal_spectrum_comparison.png"
    )


if __name__ == "__main__":
    main()

