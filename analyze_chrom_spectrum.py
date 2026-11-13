import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend, find_peaks

data = np.genfromtxt(
    "rgb_recording.csv",
    delimiter=",",
    names=True,
)

t = np.asarray(data["time_seconds"], dtype=float)
rgb = np.column_stack([
    data["red"], data["green"], data["blue"]
])

valid = np.isfinite(t) & np.all(np.isfinite(rgb), axis=1)
t, rgb = t[valid], rgb[valid]

order = np.argsort(t)
t, rgb = t[order], rgb[order]

t, unique_indices = np.unique(t, return_index=True)
rgb = rgb[unique_indices]
t = t - t[0]

fs = (len(t) - 1) / t[-1]
uniform_t = np.arange(0, t[-1], 1.0 / fs)

uniform_rgb = np.column_stack([
    np.interp(uniform_t, t, rgb[:, i])
    for i in range(3)
])


def chrom_signal(values):
    values = values / (np.mean(values, axis=0) + 1e-8) - 1.0
    r, g, b = values.T

    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b

    return x - np.std(x) / (np.std(y) + 1e-8) * y


def get_spectrum(values):
    pulse = detrend(chrom_signal(values))

    sos = butter(
        3, [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    pulse = sosfiltfilt(sos, pulse)

    window = np.hanning(len(pulse))
    spectrum = np.abs(np.fft.rfft(pulse * window))
    frequencies = np.fft.rfftfreq(len(pulse), d=1.0 / fs)

    valid_freq = (frequencies >= 0.7) & (frequencies <= 3.0)
    return frequencies[valid_freq], spectrum[valid_freq]


segments = [
    ("5–15 seconds", 5, 15),
    ("10–20 seconds", 10, 20),
    ("15–25 seconds", 15, 25),
]

fig, axes = plt.subplots(
    len(segments), 1, figsize=(10, 10), sharex=True
)

print(f"Sampling rate: {fs:.2f} Hz")

for ax, (name, start, end) in zip(axes, segments):
    mask = (uniform_t >= start) & (uniform_t < end)
    segment = uniform_rgb[mask]

    if len(segment) < int(fs * 9):
        print(name, "- insufficient data")
        continue

    frequencies, spectrum = get_spectrum(segment)
    peaks, _ = find_peaks(spectrum)

    # Show the three strongest local peaks.
    strongest = peaks[
        np.argsort(spectrum[peaks])[-3:]
    ][::-1]

    print(f"\n{name}:")
    if len(strongest) == 0:
        print("No distinct local peaks detected.")
    else:
        for index in strongest:
            bpm = frequencies[index] * 60
            print(
                f"  {bpm:.1f} BPM, "
                f"relative peak amplitude {spectrum[index]:.4f}"
            )

    ax.plot(frequencies * 60, spectrum)
    ax.set_ylabel("FFT amplitude")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Frequency expressed as BPM")
plt.tight_layout()
plt.savefig("chrom_spectrum_analysis.png", dpi=150)
plt.close()

print("\nSaved graph: chrom_spectrum_analysis.png")
