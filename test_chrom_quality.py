import numpy as np
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
t, unique = np.unique(t, return_index=True)
rgb = rgb[unique]
t -= t[0]

fs = (len(t) - 1) / t[-1]
uniform_t = np.arange(0, t[-1], 1 / fs)
rgb = np.column_stack([
    np.interp(uniform_t, t, rgb[:, i])
    for i in range(3)
])


def chrom(values):
    values = values / (np.mean(values, axis=0) + 1e-8) - 1
    r, g, b = values.T
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    return x - np.std(x) / (np.std(y) + 1e-8) * y


def analyze(segment):
    pulse = detrend(chrom(segment))
    sos = butter(
        3, [0.7, 3.0], btype="bandpass",
        fs=fs, output="sos"
    )
    pulse = sosfiltfilt(sos, pulse)

    spectrum = np.abs(
        np.fft.rfft(pulse * np.hanning(len(pulse)))
    )
    frequencies = np.fft.rfftfreq(len(pulse), d=1 / fs)

    valid_freq = (frequencies >= 0.7) & (frequencies <= 3.0)
    spectrum = spectrum[valid_freq]
    frequencies = frequencies[valid_freq]

    peaks, _ = find_peaks(spectrum)

    if len(peaks) == 0:
        return None

    peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    best = peaks[0]
    bpm = frequencies[best] * 60

    if len(peaks) > 1:
        second = peaks[1]
        ratio = spectrum[best] / (spectrum[second] + 1e-12)
    else:
        ratio = float("inf")

    # Exploratory ratio only; not a validated confidence score.
    confidence = "LOW" if ratio < 1.5 else "BETTER"
    return bpm, ratio, confidence


print(f"Sampling rate: {fs:.2f} Hz")
print("\nWindow-by-window quality analysis:")

for start in np.arange(0, uniform_t[-1] - 10, 5):
    mask = (uniform_t >= start) & (uniform_t < start + 10)
    segment = rgb[mask]

    result = analyze(segment)
    if result is None:
        print(f"{start:.0f}-{start+10:.0f}s: no distinct peak")
        continue

    bpm, ratio, confidence = result
    print(
        f"{start:.0f}-{start+10:.0f}s: "
        f"{bpm:.1f} BPM | peak ratio={ratio:.2f} | "
        f"{confidence} relative peak separation"
    )

print("\nPeak separation is experimental and not medically validated.")
