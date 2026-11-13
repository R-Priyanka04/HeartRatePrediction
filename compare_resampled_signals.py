
import csv
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, detrend

FILES = [
    "validation_rgb_1.csv",
    "validation_rgb_2.csv",
    "validation_rgb_3.csv",
]

WINDOWS = [(5, 25), (15, 35), (25, 45), (35, 55)]
TARGET_FS = 30.0
MIN_BPM = 42
MAX_BPM = 180


def load_csv(filename):
    with open(filename, newline="") as f:
        rows = list(csv.DictReader(f))

    return {
        key: np.array([float(row[key]) for row in rows])
        for key in ["time_seconds", "red", "green", "blue"]
    }


def build_signals(r, g, b):
    rn = r / max(np.mean(r), 1e-8) - 1
    gn = g / max(np.mean(g), 1e-8) - 1
    bn = b / max(np.mean(b), 1e-8) - 1

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    alpha = np.std(x) / max(np.std(y), 1e-12)

    return {
        "Green": gn,
        "CHROM-like": x - alpha * y,
    }


def estimate_bpm(signal, fs):
    if len(signal) < int(fs * 8):
        return None

    signal = detrend(signal)

    sos = butter(
        3,
        [MIN_BPM / 60, MAX_BPM / 60],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, signal)

    nperseg = min(len(filtered), int(fs * 10))

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )

    valid = (
        (frequencies >= MIN_BPM / 60)
        & (frequencies <= MAX_BPM / 60)
    )

    indices = np.where(valid)[0]

    if len(indices) == 0:
        return None

    peak = indices[np.argmax(power[indices])]
    return frequencies[peak] * 60


def main():
    for filename in FILES:
        data = load_csv(filename)
        t = data["time_seconds"]

        if np.any(np.diff(t) <= 0):
            print(f"\n{filename}: invalid timestamp ordering.")
            continue

        regular_t = np.arange(t[0], t[-1], 1 / TARGET_FS)

        print(f"\n{filename}")
        print("=" * 70)
        print(f"Original samples: {len(t)}")
        print(f"Regular-grid samples: {len(regular_t)}")
        print(f"Target sampling rate: {TARGET_FS:.1f} Hz")

        original_signals = build_signals(
            data["red"], data["green"], data["blue"]
        )

        resampled = {
            key: np.interp(regular_t, t, data[key])
            for key in ["red", "green", "blue"]
        }

        resampled_signals = build_signals(
            resampled["red"],
            resampled["green"],
            resampled["blue"],
        )

        for start, end in WINDOWS:
            original_mask = (
                (t >= start) & (t < end)
            )
            regular_mask = (
                (regular_t >= start) & (regular_t < end)
            )

            print(f"\nWindow {start}-{end}s")

            for method in ["Green", "CHROM-like"]:
                original_bpm = estimate_bpm(
                    original_signals[method][original_mask],
                    TARGET_FS,
                )

                regular_bpm = estimate_bpm(
                    resampled_signals[method][regular_mask],
                    TARGET_FS,
                )

                a = (
                    f"{original_bpm:.1f}"
                    if original_bpm is not None else "N/A"
                )
                b = (
                    f"{regular_bpm:.1f}"
                    if regular_bpm is not None else "N/A"
                )

                print(
                    f"  {method:<12} "
                    f"Original timestamps: {a:>6} BPM | "
                    f"Resampled: {b:>6} BPM"
                )

    print(
        "\nExperimental comparison only. The original-timestamp "
        "version uses the median sampling rate as an approximation. "
        "Neither method is validated against a reference pulse."
    )


if __name__ == "__main__":
    main()

