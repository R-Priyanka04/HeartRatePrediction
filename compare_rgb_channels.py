
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_FILE = "forehead_v2_diagnostics.csv"
START = 41.5
END = 45.5


def main():
    df = pd.read_csv(CSV_FILE)

    required = [
        "elapsed_seconds",
        "accepted",
        "mean_red",
        "mean_green",
        "mean_blue",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[
        (df["accepted"] == True)
        & (df["elapsed_seconds"] >= START)
        & (df["elapsed_seconds"] <= END)
    ].dropna(subset=required).sort_values("elapsed_seconds")

    if len(df) < 10:
        raise ValueError("Not enough samples in this interval.")

    t = df["elapsed_seconds"].to_numpy(float)

    channels = {
        "Red": df["mean_red"].to_numpy(float),
        "Green": df["mean_green"].to_numpy(float),
        "Blue": df["mean_blue"].to_numpy(float),
    }

    # Normalize each channel to its own mean so changes can
    # be compared despite different baseline intensity values.
    normalized = {
        name: values / np.mean(values)
        for name, values in channels.items()
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for name, values in channels.items():
        axes[0].plot(t, values, label=name)

    axes[0].set_title("Raw RGB Channel Values")
    axes[0].set_ylabel("Mean channel value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, values in normalized.items():
        axes[1].plot(t, values, label=name)

    axes[1].set_title("RGB Channels Normalized to Their Window Means")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Relative intensity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("rgb_channel_comparison.png", dpi=150)
    plt.close(fig)

    # Compare mean channel values before and during the drop.
    before = (t >= 42.0) & (t < 42.7)
    during = (t >= 42.8) & (t <= 43.9)

    print("\nMean RGB values before and during the change:")

    for name, values in channels.items():
        before_mean = np.mean(values[before])
        during_mean = np.mean(values[during])
        change_pct = (during_mean / before_mean - 1) * 100

        print(
            f"{name}: before={before_mean:.3f}, "
            f"during={during_mean:.3f}, "
            f"change={change_pct:.2f}%"
        )

    print("\nSaved: rgb_channel_comparison.png")


if __name__ == "__main__":
    main()

