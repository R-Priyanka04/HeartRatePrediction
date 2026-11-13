import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend


def chrom_signal(rgb):
    rgb = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b

    return x - (np.std(x) / (np.std(y) + 1e-8)) * y


def process_signal(rgb, fs):
    pulse = detrend(chrom_signal(rgb))
    sos = butter(
        3, [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, pulse)


def estimate_bpm(pulse, fs):
    windowed = pulse * np.hanning(len(pulse))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(pulse), d=1.0 / fs)

    valid = (frequencies >= 0.7) & (frequencies <= 3.0)
    if not np.any(valid):
        return None

    indices = np.flatnonzero(valid)
    peak = indices[np.argmax(spectrum[valid])]
    return float(frequencies[peak] * 60.0)


def main():
    data = np.genfromtxt(
        "rgb_recording.csv",
        delimiter=",",
        names=True,
    )

    t = np.asarray(data["time_seconds"], dtype=float)
    rgb = np.column_stack([
        data["red"], data["green"], data["blue"]
    ]).astype(float)

    valid = np.isfinite(t) & np.all(np.isfinite(rgb), axis=1)
    t, rgb = t[valid], rgb[valid]

    order = np.argsort(t)
    t, rgb = t[order], rgb[order]

    # Remove duplicate timestamps, if any.
    t, unique_indices = np.unique(t, return_index=True)
    rgb = rgb[unique_indices]

    if len(t) < 300:
        print("Not enough samples for analysis.")
        return

    elapsed = t[-1] - t[0]
    fs = (len(t) - 1) / elapsed
    print("Samples:", len(t))
    print("Duration:", round(float(elapsed), 2), "seconds")
    print("Approximate sampling rate:", round(float(fs), 2), "Hz")

    # Resample to a uniform timeline before digital filtering.
    uniform_t = np.arange(0, elapsed, 1.0 / fs)
    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t - t[0], rgb[:, i])
        for i in range(3)
    ])

    pulse = process_signal(uniform_rgb, fs)

    print("\n10-second window estimates (5-second step):")
    estimates = []

    for start in np.arange(0, uniform_t[-1] - 10 + 0.001, 5):
        mask = (
            (uniform_t >= start)
            & (uniform_t < start + 10)
        )
        segment = pulse[mask]

        if len(segment) < int(fs * 9):
            continue

        bpm = estimate_bpm(segment, fs)
        if bpm is not None:
            estimates.append(bpm)
            print(f"{start:5.1f}-{start + 10:5.1f} s: {bpm:.1f} BPM")

    if estimates:
        print("\nSummary of window estimates:")
        print(f"Mean: {np.mean(estimates):.1f} BPM")
        print(f"Standard deviation: {np.std(estimates):.1f} BPM")
        print(f"Range: {min(estimates):.1f}-{max(estimates):.1f} BPM")
        print("These estimates are experimental, not medically validated.")

    plt.figure(figsize=(11, 5))
    plt.plot(uniform_t, pulse)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Filtered CHROM amplitude")
    plt.title("Offline CHROM Analysis")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chrom_csv_analysis.png", dpi=150)
    plt.close()

    print("\nSaved graph: chrom_csv_analysis.png")


if __name__ == "__main__":
    main()
