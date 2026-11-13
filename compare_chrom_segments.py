import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend

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
t, idx = np.unique(t, return_index=True)
rgb = rgb[idx]
t = t - t[0]

fs = (len(t) - 1) / t[-1]
uniform_t = np.arange(0, t[-1], 1.0 / fs)
uniform_rgb = np.column_stack([
    np.interp(uniform_t, t, rgb[:, i])
    for i in range(3)
])


def chrom_signal(values):
    normalized = values / (np.mean(values, axis=0) + 1e-8) - 1.0
    r, g, b = normalized.T
    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b
    return x - np.std(x) / (np.std(y) + 1e-8) * y


def estimate_bpm(values):
    if len(values) < int(fs * 8):
        return None

    pulse = detrend(chrom_signal(values))
    sos = butter(
        3, [0.7, 3.0], btype="bandpass",
        fs=fs, output="sos"
    )
    pulse = sosfiltfilt(sos, pulse)

    spectrum = np.abs(
        np.fft.rfft(pulse * np.hanning(len(pulse)))
    )
    frequencies = np.fft.rfftfreq(len(pulse), d=1.0 / fs)
    valid_freq = (frequencies >= 0.7) & (frequencies <= 3.0)

    indices = np.flatnonzero(valid_freq)
    peak = indices[np.argmax(spectrum[valid_freq])]
    return frequencies[peak] * 60.0


segments = [
    ("First 5 seconds", 0, 5),
    ("5–15 seconds", 5, 15),
    ("15–25 seconds", 15, 25),
    ("After 5 seconds", 5, float(uniform_t[-1])),
    ("Full recording", 0, float(uniform_t[-1])),
]

print(f"Sampling rate: {fs:.2f} Hz")

for name, start, end in segments:
    mask = (uniform_t >= start) & (uniform_t < end)
    bpm = estimate_bpm(uniform_rgb[mask])

    if bpm is None:
        print(f"{name}: insufficient data")
    else:
        print(f"{name}: {bpm:.1f} BPM (experimental)")
