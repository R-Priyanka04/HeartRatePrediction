
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch, detrend

FILE = "validation_rgb.csv"
MIN_BPM = 42
MAX_BPM = 180


def estimate(signal, fs):
    signal = detrend(signal)

    sos = butter(
        3, [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = sosfiltfilt(sos, signal)

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=min(len(filtered), int(fs * 12)),
    )

    valid = (frequencies >= MIN_BPM / 60) & (
        frequencies <= MAX_BPM / 60
    )
    indices = np.where(valid)[0]

    if len(indices) == 0:
        return None, None

    order = indices[np.argsort(power[indices])[::-1]]
    peak = order[0]
    second = order[1] if len(order) > 1 else peak

    ratio = power[peak] / max(power[second], 1e-15)

    return frequencies[peak] * 60, ratio


def main():
    with open(FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    t = np.array([float(r["time_seconds"]) for r in rows])
    red = np.array([float(r["red"]) for r in rows])
    green = np.array([float(r["green"]) for r in rows])
    blue = np.array([float(r["blue"]) for r in rows])

    fs = 1 / np.median(np.diff(t))

    rn = red / np.mean(red) - 1
    gn = green / np.mean(green) - 1
    bn = blue / np.mean(blue) - 1

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn
    chrom = x - (np.std(x) / max(np.std(y), 1e-12)) * y

    signals = {
        "Green": gn,
        "CHROM-like": chrom,
    }

    windows = [(5, 25), (15, 35), (25, 45), (35, 55), (10, 50)]
    plot_data = {}

    print(f"Samples: {len(t)}")
    print(f"Duration: {t[-1] - t[0]:.2f} seconds")
    print(f"Sampling rate: {fs:.2f} Hz")
    print("\nExperimental estimates only; not validated.")
    print("-" * 65)

    for name, signal in signals.items():
        print(f"\n{name}")
        estimates = []

        for start, end in windows:
            mask = (t >= start) & (t < end)

            if np.count_nonzero(mask) < int(fs * 10):
                continue

            segment_fs = 1 / np.median(np.diff(t[mask]))
            bpm, ratio = estimate(signal[mask], segment_fs)

            if bpm is not None:
                estimates.append((start, end, bpm, ratio))
                print(
                    f"{start:2d}-{end:2d}s: "
                    f"{bpm:5.1f} BPM, peak ratio={ratio:.2f}"
                )

        values = np.array([item[2] for item in estimates])

        if len(values):
            print(
                f"Mean: {np.mean(values):.1f} BPM; "
                f"SD: {np.std(values):.1f} BPM; "
                f"range: {np.min(values):.1f}-{np.max(values):.1f}"
            )

        plot_data[name] = estimates

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, estimates in plot_data.items():
        if estimates:
            midpoints = [(a + b) / 2 for a, b, _, _ in estimates]
            bpms = [bpm for _, _, bpm, _ in estimates]
            ax.plot(midpoints, bpms, marker="o", label=name)

    ax.set_title("Validation Recording: Experimental BPM Estimates")
    ax.set_xlabel("Window midpoint (seconds)")
    ax.set_ylabel("Estimated BPM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("validation_analysis.png", dpi=150)
    plt.close(fig)

    print("\nSaved graph: validation_analysis.png")


if __name__ == "__main__":
    main()

