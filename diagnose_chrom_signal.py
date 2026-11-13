
import csv
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, detrend

CSV_FILE = "forehead_stability.csv"


def main():
    rows = []

    with open(CSV_FILE, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    time_values = np.array(
        [float(r["time_seconds"]) for r in rows]
    )
    red = np.array([float(r["red"]) for r in rows])
    green = np.array([float(r["green"]) for r in rows])
    blue = np.array([float(r["blue"]) for r in rows])

    fs = 1 / np.median(np.diff(time_values))

    # Remove the initial camera-settling period.
    mask = time_values >= 4
    t = time_values[mask]
    r = red[mask]
    g = green[mask]
    b = blue[mask]

    # Normalize the channels before constructing CHROM-like signals.
    rn = r / np.mean(r) - 1
    gn = g / np.mean(g) - 1
    bn = b / np.mean(b) - 1

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    alpha = np.std(x) / max(np.std(y), 1e-12)
    chrom = x - alpha * y

    signals = {
        "Green raw": g,
        "Green normalized": gn,
        "CHROM component X": x,
        "CHROM component Y": y,
        "CHROM-like": chrom,
    }

    print(f"Sampling rate: {fs:.2f} Hz")
    print(f"Samples after 4 seconds: {len(t)}")

    sos = butter(
        3, [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    for name, signal in signals.items():
        signal = detrend(signal)
        filtered = sosfiltfilt(sos, signal)

        f, power = welch(
            filtered,
            fs=fs,
            nperseg=min(len(filtered), int(fs * 8)),
        )

        valid = (f >= 0.7) & (f <= 3.0)
        valid_indices = np.where(valid)[0]

        peak_indices = valid_indices[
            np.argsort(power[valid_indices])[-3:][::-1]
        ]

        print(f"\n{name}")
        print(f"Raw standard deviation: {np.std(signal):.8f}")
        print(f"Filtered standard deviation: {np.std(filtered):.8f}")
        print("Top spectral peaks:")

        for idx in peak_indices:
            print(
                f"  {f[idx] * 60:.1f} BPM "
                f"(power={power[idx]:.8e})"
            )

    print("\nDiagnostic complete. No files were modified.")


if __name__ == "__main__":
    main()

