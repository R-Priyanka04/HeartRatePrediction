import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend, welch

FILES = ["rgb_recording_1.csv", "rgb_recording_2.csv"]
WINDOW_SECONDS = 10
STEP_SECONDS = 5


def load_recording(filename):
    data = np.genfromtxt(filename, delimiter=",", names=True)

    t = np.asarray(data["time_seconds"], dtype=float)
    rgb = np.column_stack([
        data["red"], data["green"], data["blue"]
    ])

    valid = np.isfinite(t) & np.all(np.isfinite(rgb), axis=1)
    t, rgb = t[valid], rgb[valid]

    order = np.argsort(t)
    t, rgb = t[order], rgb[order]
    t, indices = np.unique(t, return_index=True)
    rgb = rgb[indices]
    t -= t[0]

    fs = (len(t) - 1) / t[-1]
    uniform_t = np.arange(0, t[-1], 1 / fs)

    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, i])
        for i in range(3)
    ])

    return uniform_t, uniform_rgb, fs


def chrom_signal(rgb):
    normalized = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1
    r, g, b = normalized.T

    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b

    return x - np.std(x) / (np.std(y) + 1e-8) * y


def prepare_pulse(rgb, fs):
    pulse = detrend(chrom_signal(rgb))

    sos = butter(
        3, [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, pulse)


def get_peak(frequencies, power):
    valid = (frequencies >= 0.7) & (frequencies <= 3.0)
    frequencies = frequencies[valid]
    power = power[valid]

    if len(power) == 0:
        return None

    index = np.argmax(power)
    return float(frequencies[index] * 60)


def main():
    for filename in FILES:
        t, rgb, fs = load_recording(filename)
        estimates_fft = []
        estimates_welch = []

        print(f"\n{'=' * 48}")
        print(filename)
        print(f"Sampling rate: {fs:.2f} Hz")
        print("Window | FFT BPM | Welch BPM")

        for start in np.arange(
            0, t[-1] - WINDOW_SECONDS + 0.001, STEP_SECONDS
        ):
            mask = (
                (t >= start)
                & (t < start + WINDOW_SECONDS)
            )
            segment = rgb[mask]

            if len(segment) < int(fs * 9):
                continue

            pulse = prepare_pulse(segment, fs)
            windowed = pulse * np.hanning(len(pulse))

            # Current approach: FFT magnitude peak.
            spectrum_fft = np.abs(np.fft.rfft(windowed))
            frequencies_fft = np.fft.rfftfreq(
                len(windowed), d=1 / fs
            )
            bpm_fft = get_peak(frequencies_fft, spectrum_fft)

            # Alternative: Welch power spectral density.
            frequencies_w, power_w = welch(
                pulse,
                fs=fs,
                window="hann",
                nperseg=min(len(pulse), int(fs * 5)),
                noverlap=min(
                    int(fs * 2.5),
                    min(len(pulse), int(fs * 5)) - 1,
                ),
                detrend="constant",
            )
            bpm_welch = get_peak(frequencies_w, power_w)

            if bpm_fft is not None and bpm_welch is not None:
                estimates_fft.append(bpm_fft)
                estimates_welch.append(bpm_welch)
                print(
                    f"{start:4.0f}-{start + WINDOW_SECONDS:4.0f}s"
                    f" | {bpm_fft:7.1f} | {bpm_welch:9.1f}"
                )

        for name, estimates in [
            ("FFT", estimates_fft),
            ("Welch", estimates_welch),
        ]:
            if estimates:
                print(
                    f"{name} summary: mean={np.mean(estimates):.1f} BPM, "
                    f"SD={np.std(estimates):.1f} BPM"
                )

        print(
            "These are experimental estimates; lower variability "
            "does not establish greater accuracy."
        )


if __name__ == "__main__":
    main()
