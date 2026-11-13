
import csv
import os
import numpy as np

FILES = [
    "validation_rgb_1.csv",
    "validation_rgb_2.csv",
    "validation_rgb_3.csv",
]

# Intervals around the previously detected ROI changes.
INTERVALS = [
    (35, 42),
    (44, 51),
]


def load_csv(filename):
    with open(filename, newline="") as f:
        rows = list(csv.DictReader(f))

    data = {}
    for key in [
        "time_seconds", "red", "green", "blue",
        "roi_width", "roi_height",
    ]:
        data[key] = np.array([float(row[key]) for row in rows])

    return data


def report_interval(data, start, end):
    t = data["time_seconds"]
    mask = (t >= start) & (t <= end)

    if np.count_nonzero(mask) < 2:
        print("  Not enough samples.")
        return

    print(f"  Interval: {start}-{end}s")

    for key in [
        "red", "green", "blue", "roi_width", "roi_height"
    ]:
        values = data[key][mask]
        changes = np.abs(np.diff(values))

        print(
            f"    {key:<10} "
            f"mean={np.mean(values):7.2f} "
            f"SD={np.std(values):6.2f} "
            f"range={np.min(values):.1f}-{np.max(values):.1f} "
            f"max_step={np.max(changes):.2f}"
        )


def main():
    for filename in FILES:
        if not os.path.exists(filename):
            print(f"Missing file: {filename}")
            return

        data = load_csv(filename)

        print(f"\n{filename}")
        print("=" * 72)

        for start, end in INTERVALS:
            report_interval(data, start, end)


if __name__ == "__main__":
    main()

