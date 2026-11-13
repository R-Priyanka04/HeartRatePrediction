
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = [
    "rgb_session_1.csv",
    "rgb_session_2.csv",
    "rgb_session_3.csv",
]

for filename in FILES:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing file: {filename}")

    df = pd.read_csv(filename)

    required = ["time_seconds", "red", "green", "blue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{filename} is missing columns {missing}"
        )

    df = df[required].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    df = df.sort_values("time_seconds")
    df = df.drop_duplicates("time_seconds")

    t = df["time_seconds"].to_numpy(dtype=float, copy=True)
    rgb = df[["red", "green", "blue"]].to_numpy(dtype=float)

    t -= t[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    channels = [
        ("Red", rgb[:, 0]),
        ("Green", rgb[:, 1]),
        ("Blue", rgb[:, 2]),
    ]

    for ax, (name, signal) in zip(axes, channels):
        ax.plot(t, signal, linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"Raw forehead RGB signals — {filename}")
    fig.tight_layout()

    output = filename.replace(".csv", "_rgb_plot.png")
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print(f"Saved: {output}")

print("\nAll plots saved.")

