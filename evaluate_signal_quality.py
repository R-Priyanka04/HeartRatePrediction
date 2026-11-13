
import numpy as np
import pandas as pd

WINDOW_FILE = "green_long_window_results.csv"
ROI_FILE = "forehead_v2_diagnostics.csv"

# Experimental thresholds, not clinically validated.
MAX_BPM_JUMP = 10.0
MIN_PEAK_BIN_RATIO = 1.20


def main():
    windows = pd.read_csv(WINDOW_FILE)
    roi = pd.read_csv(ROI_FILE)

    windows = windows.sort_values(
        ["window_seconds", "midpoint_seconds"]
    ).copy()

    # Calculate ROI stability over each window.
    roi = roi.sort_values("elapsed_seconds")

    roi_rows = []

    for _, row in windows.iterrows():
        start = row["start_seconds"]
        end = start + row["window_seconds"]

        segment = roi[
            (roi["elapsed_seconds"] >= start)
            & (roi["elapsed_seconds"] < end)
            & (roi["accepted"] == True)
        ]

        if len(segment) < 2:
            roi_rows.append({
                "roi_width_sd": np.nan,
                "roi_height_sd": np.nan,
                "roi_center_motion": np.nan,
            })
            continue

        center_x = (
            segment["x1"].to_numpy(dtype=float)
            + segment["x2"].to_numpy(dtype=float)
        ) / 2

        center_y = (
            segment["y1"].to_numpy(dtype=float)
            + segment["y2"].to_numpy(dtype=float)
        ) / 2

        # Median frame-to-frame center movement in pixels.
        movement = np.sqrt(
            np.diff(center_x) ** 2
            + np.diff(center_y) ** 2
        )

        roi_rows.append({
            "roi_width_sd": segment["width"].std(),
            "roi_height_sd": segment["height"].std(),
            "roi_center_motion": (
                float(np.median(movement))
                if len(movement) else 0.0
            ),
        })

    roi_metrics = pd.DataFrame(roi_rows, index=windows.index)
    windows = pd.concat([windows, roi_metrics], axis=1)

    # Compare estimates with nearby, overlapping windows
    # of the same duration.
    windows["bpm_change"] = (
        windows.groupby("window_seconds")["bpm"]
        .diff()
        .abs()
    )

    windows["peak_ratio_ok"] = (
        windows["peak_bin_ratio"] >= MIN_PEAK_BIN_RATIO
    )

    windows["bpm_stable"] = (
        windows["bpm_change"].isna()
        | (windows["bpm_change"] <= MAX_BPM_JUMP)
    )

    # These are experimental screening rules, not validated
    # estimates of probability or clinical signal quality.
    windows["screening_label"] = np.where(
        windows["peak_ratio_ok"]
        & windows["bpm_stable"],
        "CANDIDATE",
        "UNCERTAIN",
    )

    output = "signal_quality_screening.csv"
    windows.to_csv(output, index=False)

    print("\n--- EXPERIMENTAL SIGNAL QUALITY SCREEN ---")
    print("These labels are not validated confidence scores.\n")

    for length in sorted(windows["window_seconds"].unique()):
        part = windows[windows["window_seconds"] == length]

        candidates = (part["screening_label"] == "CANDIDATE").sum()
        uncertain = (part["screening_label"] == "UNCERTAIN").sum()

        print(f"{int(length)}-second windows:")
        print(f"  Candidate: {candidates}")
        print(f"  Uncertain: {uncertain}")

    print("\nFirst 15 screened windows:")
    print(
        windows[
            [
                "window_seconds",
                "start_seconds",
                "bpm",
                "second_bpm",
                "peak_bin_ratio",
                "bpm_change",
                "roi_center_motion",
                "screening_label",
            ]
        ].head(15).round(2).to_string(index=False)
    )

    print(f"\nDetailed results saved to: {output}")


if __name__ == "__main__":
    main()

