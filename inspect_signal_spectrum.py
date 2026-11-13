
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch, detrend

FILES = [
    "validation_rgb_1.csv",
    "validation_rgb_2.csv",
    "validation_rgb_3.csv",
]

WINDOW = (5, 25)
MIN_BPM = 42
MAX_BPM = 180


def load_csv(filename):
    with open(filename, newline="") as f:
        rows = list(csv.DictReader(f))

    t = np.array([float(r["time_seconds"]) for r in rows])
    r = np.array([float(row["red"]) for row in rows])
    g = np.array([float(row["green"]) for row in rows])
    b = np.array([float(row["blue"]) for row in rows])

    return t, r, g, b


def make_signals(r, g, b):
    rn = r / max(np.mean(r), 1e-8) - 1
    gn = g / max(np.mean(g), 1e-8) - 1
    bn = b / max(np.mean(b), 1e-8) - 1

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    alpha = np.std(x) / max(np.std(y), 1e-12)
    chrom = x - alpha * y

    return {
        "Green": gn,
        "CHROM X": x,
        "CHROM Y": y,
        "CHROM-like": chrom,
    }


def main():
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    for ax, filename in zip(axes, FILES):
        t, r, g, b = load_csv(filename)
        mask = (t >= WINDOW[0]) & (t < WINDOW[1])

        tw = t[mask]
        fs = 1 / np.median(np.diff(tw))

        signals = make_signals(r[mask], g[mask], b[mask])

        print(f"\n{filename}: {fs:.2f} Hz")

        for name, signal in signals.items():
            filtered = detrend(signal)

            sos = butter(
                3, [MIN_BPM / 60, MAX_BPM / 60],
                btype="bandpass",
                fs=fs,
                output="sos",
            )
            filtered = sosfiltfilt(sos, filtered)

            frequencies, power = welch(
                filtered,
                fs=fs,
                nperseg=min(len(filtered), int(fs * 20)),
                noverlap=min(len(filtered), int(fs * 20)) // 2,
            )

            valid = (
                (frequencies >= MIN_BPM / 60)
                & (frequencies <= MAX_BPM / 60)
            )
            indices = np.where(valid)[0]
            order = indices[np.argsort(power[indices])[::-1]]

            peaks = []
            for idx in order:
                bpm = frequencies[idx] * 60

                # Avoid listing adjacent bins as separate peaks.
                if all(abs(bpm - old[0]) >= 6 for old in peaks):
                    peaks.append((bpm, power[idx]))

                if len(peaks) == 3:
                    break

            print(f"  {name}: " + ", ".join(
                f"{bpm:.1f} BPM" for bpm, _ in peaks
            ))

            ax.plot(
                frequencies[valid] * 60,
                power[valid],
                label=name,
            )

        ax.set_title(filename)
        ax.set_xlabel("Frequency (BPM)")
        ax.set_ylabel("Spectral power")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig("signal_spectrum_comparison.png", dpi=150)
    plt.close(fig)

    print("\nSaved graph: signal_spectrum_comparison.png")


if __name__ == "__main__":
    main()

