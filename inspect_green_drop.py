
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

    df = df[
        (df["accepted"] == True)
        & (df["elapsed_seconds"] >= START)
        & (df["elapsed_seconds"] <= END)
    ].dropna(
        subset=[
            "elapsed_seconds",
            "mean_green",
            "x1", "y1", "x2", "y2",
            "width", "height",
        ]
    ).sort_values("elapsed_seconds")

    if len(df) < 10:
        raise ValueError("Not enough samples in the selected interval.")

    t = df["elapsed_seconds"].to_numpy(float)
    green = df["mean_green"].to_numpy(float)

    center_x = (df["x1"].to_numpy(float) + df["x2"].to_numpy(float)) / 2
    center_y = (df["y1"].to_numpy(float) + df["y2"].to_numpy(float)) / 2

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, green, marker=".", markersize=3)
    axes[0].set_ylabel("Mean green")
    axes[0].set_title("Green-channel change around 43 seconds")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, center_x, label="ROI center X")
    axes[1].plot(t, center_y, label="ROI center Y")
    axes[1].set_ylabel("Center coordinate (pixels)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, df["width"], label="ROI width")
    axes[2].plot(t, df["height"], label="ROI height")
    axes[2].set_ylabel("ROI dimensions (pixels)")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("green_drop_roi_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Samples inspected: {len(df)}")
    print(f"Green range: {green.min():.3f} to {green.max():.3f}")
    print(f"ROI center X range: {center_x.min():.1f} to {center_x.max():.1f}")
    print(f"ROI center Y range: {center_y.min():.1f} to {center_y.max():.1f}")
    print(f"ROI width range: {df['width'].min()} to {df['width'].max()}")
    print(f"ROI height range: {df['height'].min()} to {df['height'].max()}")
    print("Saved: green_drop_roi_comparison.png")


if __name__ == "__main__":
    main()

