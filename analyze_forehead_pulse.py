
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch, detrend

CSV_FILE = "forehead_stability.csv"
MIN_BPM = 42
MAX_BPM = 180


def load_data(filename):
    rows = []

    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "time": float(row["time_seconds"]),
                "red": float(row["red"]),
                "green": float(row["green"]),
                "blue": float(row["blue"]),
            })

    if len(rows) < 100:
        raise ValueError("Not enough samples for analysis.")

    data = {
        key: np.array([r[key] for r in rows], dtype=float)
        for key in ("time", "red", "green", "blue")
    }

    return data


def estimate_bpm(signal, fs):
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

    if not np.any(valid):
        return None, filtered, frequencies, power

    valid_indices = np.where(valid)[0]
    peak_index = valid_indices[np.argmax(power[valid])]

    bpm = frequencies[peak_index] * 60

    return bpm, filtered, frequencies, power


def main():
    data = load_data(CSV_FILE)

    time_values = data["time"]
    duration = time_values[-1] - time_values[0]

    # Use timestamps to estimate the sampling rate.
    fs = 1.0 / np.median(np.diff(time_values))

    print(f"Samples: {len(time_values)}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Estimated sampling rate: {fs:.2f} Hz")

    # Remove the first 4 seconds to avoid initial settling.
    # This is a hypothesis to test, not a guarantee of better accuracy.
    segments = [
        ("After 4 seconds", 4, duration),
        ("Middle segment", 5, min(15, duration)),
        ("Later segment", 15, duration),
    ]

    results = []

    for name, start, end in segments:
        mask = (time_values >= start) & (time_values <= end)

        if np.count_nonzero(mask) < int(fs * 8):
            print(f"\n{name}: skipped (segment too short)")
            continue

        green = data["green"][mask]
        red = data["red"][mask]
        blue = data["blue"][mask]

        segment_time = time_values[mask]
        segment_fs = 1.0 / np.median(np.diff(segment_time))

        # Green baseline.
        green_bpm, green_filtered, frequencies, power = (
            estimate_bpm(green, segment_fs)
        )

        # CHROM-like colour projection for comparison.
        # This is a simplified experimental projection, not a full
        # reproduction of the paper's complete rPPG pipeline.
        red_norm = red / max(np.mean(red), 1e-8) - 1
        green_norm = green / max(np.mean(green), 1e-8) - 1
        blue_norm = blue / max(np.mean(blue), 1e-8) - 1

        x = 3 * red_norm - 2 * green_norm
        y = 1.5 * red_norm + green_norm - 1.5 * blue_norm

        alpha = (
            np.std(x) / max(np.std(y), 1e-8)
        )

        chrom = x - alpha * y

        chrom_bpm, chrom_filtered, _, _ = estimate_bpm(
            chrom, segment_fs
        )

        print(f"\n{name} ({start}-{end:.1f} seconds)")
        print(
            f"Green estimate: "
            f"{green_bpm:.1f} BPM"
            if green_bpm is not None
            else "Green estimate: unavailable"
        )
        print(
            f"CHROM-like estimate: "
            f"{chrom_bpm:.1f} BPM"
            if chrom_bpm is not None
            else "CHROM-like estimate: unavailable"
        )

        results.append((
            name,
            segment_time,
            green_filtered,
            chrom_filtered,
        ))

    if not results:
        print("No sufficiently long segments were available.")
        return

    fig, axes = plt.subplots(
        len(results), 1,
        figsize=(11, 3 * len(results)),
        squeeze=False,
    )

    for ax, result in zip(axes[:, 0], results):
        name, segment_time, green_filtered, chrom_filtered = result

        ax.plot(
            segment_time,
            green_filtered,
            label="Filtered green",
        )
        ax.plot(
            segment_time,
            chrom_filtered,
            label="Filtered CHROM-like",
            alpha=0.8,
        )
        ax.set_title(name)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Filtered signal")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig("forehead_pulse_analysis.png", dpi=150)
    plt.close(fig)

    print("\nSaved graph: forehead_pulse_analysis.png")


if __name__ == "__main__":
    main()

