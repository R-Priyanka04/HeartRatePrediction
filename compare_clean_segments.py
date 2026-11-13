
import csv
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, detrend

FILES = [
    "validation_rgb_1.csv",
    "validation_rgb_2.csv",
    "validation_rgb_3.csv",
]

WINDOWS = [(5, 25), (15, 35), (25, 45), (35, 55)]
MIN_BPM = 42
MAX_BPM = 180


def load_recording(filename):
    with open(filename, newline="") as f:
        rows = list(csv.DictReader(f))

    data = {}
    for key in [
        "time_seconds", "red", "green", "blue",
        "roi_width", "roi_height",
    ]:
        data[key] = np.array([float(r[key]) for r in rows])

    return data


def make_signals(data):
    r, g, b = data["red"], data["green"], data["blue"]

    rn = r / max(np.mean(r), 1e-8) - 1
    gn = g / max(np.mean(g), 1e-8) - 1
    bn = b / max(np.mean(b), 1e-8) - 1

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    alpha = np.std(x) / max(np.std(y), 1e-12)
    return {"Green": gn, "CHROM-like": x - alpha * y}


def estimate(signal, fs):
    signal = detrend(signal)

    sos = butter(
        3, [MIN_BPM / 60, MAX_BPM / 60],
        btype="bandpass", fs=fs, output="sos",
    )
    filtered = sosfiltfilt(sos, signal)

    nperseg = min(len(filtered), int(fs * 20))
    f, power = welch(
        filtered, fs=fs, nperseg=nperseg,
        noverlap=nperseg // 2,
    )

    valid = (f >= MIN_BPM / 60) & (f <= MAX_BPM / 60)
    idx = np.where(valid)[0]

    if len(idx) == 0:
        return None

    return f[idx[np.argmax(power[idx])]] * 60


def main():
    for filename in FILES:
        data = load_recording(filename)
        t = data["time_seconds"]
        fs = 1 / np.median(np.diff(t))

        print(f"\n{filename}")
        print("=" * 65)

        for start, end in WINDOWS:
            base = (t >= start) & (t < end)

            # Flag abrupt ROI changes between adjacent samples.
            dw = np.abs(np.diff(data["roi_width"]))
            dh = np.abs(np.diff(data["roi_height"]))

            bad_steps = (
                (dw > 8) | (dh > 5)
            )

            # Mark samples on both sides of a flagged change.
            bad_samples = np.zeros(len(t), dtype=bool)
            positions = np.where(bad_steps)[0]
            bad_samples[positions] = True
            bad_samples[positions + 1] = True

            clean = base & ~bad_samples
            total = np.count_nonzero(base)
            retained = np.count_nonzero(clean)

            print(
                f"\nWindow {start}-{end}s: "
                f"retained {retained}/{total} samples"
            )

            if retained < int(fs * 10):
                print("  Not enough clean samples for comparison.")
                continue

            # This deliberately removes samples and may introduce
            # gaps; interpret results as a diagnostic only.
            segment = {
                key: values[clean]
                for key, values in data.items()
            }
            signals = make_signals(segment)

            # Actual timestamps may have gaps after removal.
            # Do not treat this as a final pulse estimate.
            for method, signal in signals.items():
                bpm = estimate(signal, fs)
                print(
                    f"  {method}: "
                    f"{bpm:.1f} BPM (exploratory only)"
                    if bpm is not None
                    else f"  {method}: unavailable"
                )

    print(
        "\nImportant: removing samples creates time gaps. "
        "These BPM values are diagnostic only, not valid final estimates."
    )


if __name__ == "__main__":
    main()

