import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend, welch

FILES = ["rgb_recording_1.csv", "rgb_recording_2.csv"]


def load_file(filename):
    data = np.genfromtxt(filename, delimiter=",", names=True)
    t = np.asarray(data["time_seconds"], dtype=float)
    rgb = np.column_stack([
        data["red"], data["green"], data["blue"]
    ])

    valid = np.isfinite(t) & np.all(np.isfinite(rgb), axis=1)
    t, rgb = t[valid], rgb[valid]
    order = np.argsort(t)
    t, rgb = t[order], rgb[order]
    t, idx = np.unique(t, return_index=True)
    rgb = rgb[idx]
    t -= t[0]

    fs = (len(t) - 1) / t[-1]
    uniform_t = np.arange(0, t[-1], 1 / fs)
    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, i])
        for i in range(3)
    ])
    return uniform_t, uniform_rgb, fs


def chrom(rgb):
    normalized = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1
    r, g, b = normalized.T
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    return x - np.std(x) / (np.std(y) + 1e-8) * y


def filtered_pulse(rgb, fs, remove_trend):
    pulse = chrom(rgb)
    if remove_trend:
        pulse = detrend(pulse)

    sos = butter(
        3, [0.7, 3.0], btype="bandpass",
        fs=fs, output="sos"
    )
    return sosfiltfilt(sos, pulse)


def bpm_fft(pulse, fs):
    spectrum = np.abs(
        np.fft.rfft(pulse * np.hanning(len(pulse)))
    )
    frequencies = np.fft.rfftfreq(len(pulse), d=1 / fs)
    valid = (frequencies >= 0.7) & (frequencies <= 3.0)
    return float(frequencies[valid][np.argmax(spectrum[valid])] * 60)


def bpm_welch(pulse, fs):
    frequencies, power = welch(
        pulse,
        fs=fs,
        window="hann",
        nperseg=min(len(pulse), int(fs * 5)),
        noverlap=min(int(fs * 2.5), min(len(pulse), int(fs * 5)) - 1),
        detrend="constant",
    )
    valid = (frequencies >= 0.7) & (frequencies <= 3.0)
    return float(frequencies[valid][np.argmax(power[valid])] * 60)


def main():
    for filename in FILES:
        t, rgb, fs = load_file(filename)
        methods = {
            "CHROM + FFT": [],
            "CHROM + Welch": [],
            "Detrended CHROM + Welch": [],
        }

        print(f"\n=== {filename} ===")
        print("Window | FFT | Welch | Detrended + Welch")

        for start in np.arange(0, t[-1] - 10 + 0.001, 5):
            mask = (t >= start) & (t < start + 10)
            segment = rgb[mask]
            if len(segment) < int(fs * 9):
                continue

            pulse1 = filtered_pulse(segment, fs, False)
            pulse2 = filtered_pulse(segment, fs, True)

            values = [
                bpm_fft(pulse1, fs),
                bpm_welch(pulse1, fs),
                bpm_welch(pulse2, fs),
            ]

            for key, value in zip(methods, values):
                methods[key].append(value)

            print(
                f"{start:2.0f}-{start+10:2.0f}s | "
                f"{values[0]:5.1f} | {values[1]:5.1f} | {values[2]:5.1f}"
            )

        print("\nSummary (mean and standard deviation):")
        for name, values in methods.items():
            print(
                f"{name}: mean={np.mean(values):.1f} BPM, "
                f"SD={np.std(values):.1f} BPM"
            )

        print("Experimental results only; accuracy is not validated.")


if __name__ == "__main__":
    main()
