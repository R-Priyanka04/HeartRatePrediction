
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = [
    "validation_rgb_1.csv",
    "validation_rgb_2.csv",
    "validation_rgb_3.csv",
]


def load_file(filename):
    with open(filename, newline="") as f:
        rows = list(csv.DictReader(f))

    t = np.array([float(r["time_seconds"]) for r in rows])
    red = np.array([float(r["red"]) for r in rows])
    green = np.array([float(r["green"]) for r in rows])
    blue = np.array([float(r["blue"]) for r in rows])
    width = np.array([float(r["roi_width"]) for r in rows])
    height = np.array([float(r["roi_height"]) for r in rows])

    return t, red, green, blue, width, height


def describe(name, values):
    print(
        f"  {name:<12} mean={np.mean(values):8.2f} | "
        f"SD={np.std(values):7.2f} | "
        f"min={np.min(values):7.2f} | "
        f"max={np.max(values):7.2f}"
    )


def main():
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for row_index, filename in enumerate(FILES):
        if not os.path.isfile(filename):
            print(f"Missing file: {filename}")
            return

        t, red, green, blue, width, height = load_file(filename)

        # Ignore the first 4 seconds while the camera settles.
        mask = t >= 4
        t = t[mask]
        red = red[mask]
        green = green[mask]
        blue = blue[mask]
        width = width[mask]
        height = height[mask]

        print(f"\n{filename}")
        print("-" * 65)

        describe("Red", red)
        describe("Green", green)
        describe("Blue", blue)
        describe("ROI width", width)
        describe("ROI height", height)

        # Flag large changes relative to typical frame-to-frame changes.
        for name, values in [
            ("Red", red),
            ("Green", green),
            ("Blue", blue),
            ("ROI width", width),
            ("ROI height", height),
        ]:
            changes = np.abs(np.diff(values))
            p95 = np.percentile(changes, 95)
            maximum = np.max(changes)

            print(
                f"  {name:<12} change P95={p95:.2f} | "
                f"maximum={maximum:.2f}"
            )

        # Plot RGB channel intensities.
        ax = axes[row_index, 0]
        ax.plot(t, red, label="Red")
        ax.plot(t, green, label="Green")
        ax.plot(t, blue, label="Blue")
        ax.set_title(f"{filename}: RGB")
        ax.set_ylabel("Mean intensity")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot ROI dimensions.
        ax = axes[row_index, 1]
        ax.plot(t, width, label="ROI width")
        ax.plot(t, height, label="ROI height")
        ax.set_title(f"{filename}: ROI dimensions")
        ax.set_ylabel("Pixels")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")

    fig.tight_layout()
    fig.savefig("validation_roi_diagnostics.png", dpi=150)
    plt.close(fig)

    print("\nSaved graph: validation_roi_diagnostics.png")


if __name__ == "__main__":
    main()

