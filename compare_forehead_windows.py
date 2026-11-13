
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch, detrend

CSV_FILE = "forehead_stability.csv"
MIN_BPM = 42
MAX_BPM = 180


def analyse(signal, fs):
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
        nperseg=min(len(filtered), int(fs * 8)),
    )

    valid = (frequencies >= MIN_BPM / 60) & (
        frequencies <= MAX_BPM / 60
    )
    indices = np.where(valid)[0]
    order = indices[np.argsort(power[indices])[::-1]]

    if len(order) < 2:
        return None, None, None

    peak = order[0]
    second = order[1]
    ratio = power[peak] / max(power[second], 1e-15)

    return frequencies[peak] * 60, frequencies[second] * 60, ratio


def main():
    with open(CSV_FILE, newline="") as f:
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

    # Exclude initial settling period.
    keep = t >= 4
    t = t[keep]
    signals = {
        "Green": gn[keep],
        "CHROM-like": chrom[keep],
    }

    # Test multiple window lengths, moving forward in 2-second steps.
    window_lengths = [8, 12, 16]
    step = 2

    all_results = []

    print(f"Sampling rate: {fs:.2f} Hz")
    print("Experimental results only; not accuracy measurements.\n")

    for name, signal in signals.items():
        print(f"\n{name}")
        print("-" * 56)
        print(f"{'Window':<18}{'Peak BPM':>12}{'Second BPM':>14}{'Ratio':>10}")

        for length in window_lengths:
            starts = np.arange(t[0], t[-1] - length + 0.001, step)
            estimates = []

            for start in starts:
                mask = (t >= start) & (t < start + length)
                if np.count_nonzero(mask) < int(fs * 6):
                    continue

                bpm, second, ratio = analyse(signal[mask], fs)

                if bpm is not None:
                    estimates.append((start, bpm, second, ratio))
                    all_results.append(
                        (name, length, start, bpm, second, ratio)
                    )
                    print(
                        f"{start:.0f}-{start+length:.0f}s".ljust(18)
                        + f"{bpm:>12.1f}"
                        + f"{second:>14.1f}"
                        + f"{ratio:>10.2f}"
                    )

            if estimates:
                values = np.array([e[1] for e in estimates])
                print(
                    f"  {length}s windows: mean={np.mean(values):.1f}, "
                    f"SD={np.std(values):.1f} BPM, "
                    f"range={np.min(values):.1f}-{np.max(values):.1f}"
                )

    if not all_results:
        print("No valid windows found.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for ax, name in zip(axes, ["Green", "CHROM-like"]):
        for length in window_lengths:
            items = [
                r for r in all_results
                if r[0] == name and r[1] == length
            ]
            if items:
                ax.plot(
                    [r[2] + length / 2 for r in items],
                    [r[3] for r in items],
                    marker="o",
                    label=f"{length}s windows",
                )

        ax.set_title(name)
        ax.set_ylabel("Estimated BPM")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Window midpoint (seconds)")
    fig.tight_layout()
    fig.savefig("forehead_window_comparison.png", dpi=150)
    plt.close(fig)

    print("\nSaved graph: forehead_window_comparison.png")


if __name__ == "__main__":
    main()

