
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
            continue

        with open(filename, newline="") as f:
            rows = list(csv.DictReader(f))

        t = np.array([
            float(row["time_seconds"]) for row in rows
        ])

        print(f"\n{filename}")
        print("-" * 60)
        print("Samples:", len(t))

        if len(t) < 2:
            print("Not enough samples.")
            continue

        dt = np.diff(t)
        fs = 1 / np.median(dt)

        print(f"Duration: {t[-1] - t[0]:.3f} seconds")
        print(f"Median sample interval: {np.median(dt):.6f} seconds")
        print(f"Estimated sampling rate: {fs:.3f} Hz")
        print(f"Minimum interval: {np.min(dt):.6f} seconds")
        print(f"Maximum interval: {np.max(dt):.6f} seconds")
        print(f"Interval SD: {np.std(dt):.6f} seconds")
        print(f"Intervals <= 0: {np.sum(dt <= 0)}")

        expected_dt = 1 / fs
        deviation = np.abs(dt - expected_dt) / expected_dt

        print(
            "Intervals deviating more than 10%:",
            int(np.sum(deviation > 0.10)),
            f"of {len(dt)}",
        )

        gaps = np.where(dt > 1.5 * expected_dt)[0]

        print("Large timestamp gaps:", len(gaps))

        for i in gaps[:10]:
            print(
                f"  Gap near {t[i + 1]:.3f}s: "
                f"{dt[i]:.4f} seconds"
            )


if __name__ == "__main__":
    main()
