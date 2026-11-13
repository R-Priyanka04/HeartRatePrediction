
import csv
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, detrend

CSV_FILE = "forehead_stability.csv"
MIN_BPM = 42
MAX_BPM = 180


def estimate(signal, fs):
    signal = detrend(signal)

    sos = butter(
        3,
        [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = sosfiltfilt(sos, signal)

    nperseg = min(len(filtered), int(fs * 8))
    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=nperseg,
    )

    valid = (
        (frequencies >= MIN_BPM / 60)
        & (frequencies <= MAX_BPM / 60)
    )
    indices = np.where(valid)[0]

    if len(indices) < 2:
        return None, None, None

    order = indices[np.argsort(power[indices])[::-1]]
    best = order[0]
    second = order[1]

    bpm = frequencies[best] * 60
    second_bpm = frequencies[second] * 60
    ratio = power[best] / max(power[second], 1e-15)

    return bpm, second_bpm, ratio


def main():
    rows = []

    with open(CSV_FILE, newline="") as file:
        rows = list(csv.DictReader(file))

    t = np.array([float(r["time_seconds"]) for r in rows])
    red = np.array([float(r["red"]) for r in rows])
    green = np.array([float(r["green"]) for r in rows])
    blue = np.array([float(r["blue"]) for r in rows])

    fs = 1 / np.median(np.diff(t))

    # Normalize colour channels.
    rn = red / np.mean(red) - 1
    gn = green / np.mean(green) - 1
    bn = blue / np.mean(blue) - 1

    # Simplified CHROM-like components.
    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn
    chrom = x - (np.std(x) / max(np.std(y), 1e-12)) * y

    methods = {
        "Green": gn,
        "CHROM X": x,
        "CHROM Y": y,
        "CHROM-like": chrom,
    }

    # Identical windows and analysis settings for each method.
    windows = [
        (4, 14),
        (7, 17),
        (10, 20),
        (13, 23),
        (14, min(24.5, t[-1])),
    ]

    print(f"Sampling rate: {fs:.2f} Hz")
    print("\nAll estimates are experimental, not validated.")
    print("-" * 76)
    print(
        f"{'Window':<15}{'Method':<14}"
        f"{'Peak BPM':>11}{'2nd BPM':>11}{'Power ratio':>15}"
    )
    print("-" * 76)

    for start, end in windows:
        mask = (t >= start) & (t <= end)

        if np.count_nonzero(mask) < int(fs * 8):
            continue

        segment_fs = 1 / np.median(np.diff(t[mask]))

        for name, signal in methods.items():
            bpm, second_bpm, ratio = estimate(
                signal[mask], segment_fs
            )

            if bpm is None:
                print(f"{start}-{end}s {name}: unavailable")
            else:
                print(
                    f"{start}-{end}s".ljust(15)
                    + f"{name:<14}"
                    + f"{bpm:>11.1f}"
                    + f"{second_bpm:>11.1f}"
                    + f"{ratio:>15.2f}"
                )

        print()

    print("Power ratio compares the two strongest frequency bins.")
    print("Nearby bins can represent the same broad peak.")
    print("A high ratio does not prove the BPM is accurate.")


if __name__ == "__main__":
    main()

