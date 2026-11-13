
import csv
import os
import numpy as np

FILES = [
    "validation_rgb_1.csv",
    "validation_rgb_2.csv",
    "validation_rgb_3.csv",
]

def main():
    for filename in FILES:
        if not os.path.exists(filename):
            print(f"Missing file: {filename}")
            return

        with open(filename, newline="") as f:
            rows = list(csv.DictReader(f))

        t = np.array([float(r["time_seconds"]) for r in rows])
        width = np.array([float(r["roi_width"]) for r in rows])
        height = np.array([float(r["roi_height"]) for r in rows])

        # Ignore initial camera/landmark settling.
        keep = t >= 4
        t = t[keep]
        width = width[keep]
        height = height[keep]

        dw = np.abs(np.diff(width))
        dh = np.abs(np.diff(height))

        # Flag abrupt changes; these are diagnostic thresholds,
        # not proven failure thresholds.
        width_limit = max(8.0, np.percentile(dw, 95) * 2)
        height_limit = max(5.0, np.percentile(dh, 95) * 2)

        width_jumps = np.where(dw > width_limit)[0]
        height_jumps = np.where(dh > height_limit)[0]

        print(f"\n{filename}")
        print("-" * 60)
        print(
            f"Width: mean={np.mean(width):.1f}px, "
            f"SD={np.std(width):.1f}px, "
            f"range={np.min(width):.0f}-{np.max(width):.0f}px"
        )
        print(
            f"Height: mean={np.mean(height):.1f}px, "
            f"SD={np.std(height):.1f}px, "
            f"range={np.min(height):.0f}-{np.max(height):.0f}px"
        )
        print(
            f"Width jump threshold: {width_limit:.1f}px; "
            f"count: {len(width_jumps)}"
        )
        print(
            f"Height jump threshold: {height_limit:.1f}px; "
            f"count: {len(height_jumps)}"
        )

        for label, indices in [
            ("Width", width_jumps),
            ("Height", height_jumps),
        ]:
            for i in indices:
                print(
                    f"  {label} jump near {t[i + 1]:.2f}s: "
                    f"{(dw if label == 'Width' else dh)[i]:.1f}px"
                )

if __name__ == "__main__":
    main()

