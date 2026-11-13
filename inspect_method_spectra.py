
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch, detrend


CSV_FILE = "forehead_v2_diagnostics.csv"
START = 28
END = 48


def main():
    df = pd.read_csv(CSV_FILE)

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

    t = df["elapsed_seconds"].to_numpy(float)
    rgb = df[["mean_red", "mean_green", "mean_blue"]].to_numpy(float)

    keep = np.concatenate(([True], np.diff(t) > 0))
    t, rgb = t[keep], rgb[keep]

    if len(t) < 100:
        raise ValueError("Not enough valid samples.")

    fs = 1.0 / np.median(np.diff(t))
    uniform_t = np.arange(t[0], t[-1], 1.0 / fs)

    rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, i])
        for i in range(3)
    ])

    red, green, blue = rgb.T

    signals = {
        "Green": green / (np.mean(green) + 1e-8) - 1,
    }

    normalized_green = green / (
        red + green + blue + 1e-8
    )
    signals["Normalized green"] = (
        normalized_green - np.mean(normalized_green)
    )

    normalized_rgb = rgb / (
        np.mean(rgb, axis=0) + 1e-8
    ) - 1

    r, g, b = normalized_rgb.T
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b

    signals["CHROM"] = x - (
        np.std(x) / (np.std(y) + 1e-8)
    ) * y

    sos = butter(
        3,
        [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for ax, (name, signal) in zip(axes, signals.items()):
        filtered = sosfiltfilt(sos, detrend(signal))

        freq, power = welch(
            filtered,
            fs=fs,
            nperseg=len(filtered),
            window="hann",
        )

        valid = (freq >= 0.7) & (freq <= 3.0)
        bpm_axis = freq[valid] * 60
        power_axis = power[valid]

        ax.plot(bpm_axis, power_axis)
        ax.set_title(f"{name} spectrum ({START}-{END}s)")
        ax.set_xlabel("Frequency (BPM)")
        ax.set_ylabel("Power")
        ax.grid(True, alpha=0.3)

        top = np.argsort(power_axis)[::-1][:5]

        print(f"\n{name}: strongest frequency bins")
        for idx in top:
            print(
                f"{bpm_axis[idx]:.2f} BPM, "
                f"power={power_axis[idx]:.6g}"
            )

    fig.tight_layout()
    fig.savefig("method_spectra_28_48.png", dpi=150)
    plt.close(fig)

    print("\nSaved: method_spectra_28_48.png")
    print("Diagnostic only; no reference heart rate is available.")


if __name__ == "__main__":
    main()

