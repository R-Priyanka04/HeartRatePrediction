import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, detrend

FILES = ["rgb_recording_1.csv", "rgb_recording_2.csv"]
LOW_HZ = 0.7
HIGH_HZ = 3.0


def estimate_bpm(signal, fs):
    signal = detrend(signal)
    sos = butter(
        3, [LOW_HZ, HIGH_HZ],
        btype="bandpass", fs=fs, output="sos"
    )
    signal = sosfiltfilt(sos, signal)

    freq, power = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=min(len(signal), int(fs * 5)),
        noverlap=min(int(fs * 2.5), min(len(signal), int(fs * 5)) - 1)
    )

    valid = (freq >= LOW_HZ) & (freq <= HIGH_HZ)
    return float(freq[valid][np.argmax(power[valid])] * 60)


def main():
    for filename in FILES:
        data = np.genfromtxt(filename, delimiter=",", names=True)
        t = data["time_seconds"]
        rgb = np.column_stack([data["red"], data["green"], data["blue"]])

        valid = np.isfinite(t) & np.all(np.isfinite(rgb), axis=1)
        t, rgb = t[valid], rgb[valid]
        order = np.argsort(t)
        t, rgb = t[order], rgb[order]
        t, idx = np.unique(t, return_index=True)
        rgb = rgb[idx]
        t = t - t[0]

        fs = (len(t) - 1) / t[-1]
        uniform_t = np.arange(0, t[-1], 1 / fs)
        rgb = np.column_stack([
            np.interp(uniform_t, t, rgb[:, i]) for i in range(3)
        ])

        red, green, blue = rgb.T

        # Baseline: green channel
        green_signal = green / (np.mean(green) + 1e-8) - 1

        # Normalized green: reduce common brightness variation
        normalized_green = green / (
            red + green + blue + 1e-8
        )
        normalized_green -= np.mean(normalized_green)

        # CHROM pulse signal
        normalized = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1
        r, g, b = normalized.T
        x = 3 * r - 2 * g
        y = 1.5 * r + g - 1.5 * b
        chrom = x - (
            np.std(x) / (np.std(y) + 1e-8)
        ) * y

        signals = {
            "Green": green_signal,
            "Normalized green": normalized_green,
            "CHROM": chrom,
        }

        print(f"\n=== {filename} ===")
        print("Window | Green | Normalized green | CHROM")

        estimates = {name: [] for name in signals}

        for start in np.arange(0, uniform_t[-1] - 10 + 0.001, 5):
            mask = (uniform_t >= start) & (uniform_t < start + 10)
            if np.count_nonzero(mask) < int(fs * 9):
                continue

            values = []
            for name, signal in signals.items():
                bpm = estimate_bpm(signal[mask], fs)
                estimates[name].append(bpm)
                values.append(bpm)

            print(
                f"{start:2.0f}-{start+10:2.0f}s | "
                f"{values[0]:5.1f} | {values[1]:16.1f} | {values[2]:5.1f}"
            )

        print("\nWindow-to-window standard deviation (BPM):")
        for name, values in estimates.items():
            print(f"{name}: {np.std(values):.1f}")

        print("Experimental only; no reference heart rate available.")


if __name__ == "__main__":
    main()
