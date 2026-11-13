import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, detrend

data = np.genfromtxt("rgb_recording.csv", delimiter=",", names=True)
t = data["time_seconds"]
rgb = np.column_stack([data["red"], data["green"], data["blue"]])

valid = np.isfinite(t) & np.all(np.isfinite(rgb), axis=1)
t, rgb = t[valid], rgb[valid]
order = np.argsort(t)
t, rgb = t[order], rgb[order]
t, idx = np.unique(t, return_index=True)
rgb = rgb[idx]
t -= t[0]

fs = (len(t) - 1) / t[-1]

def get_bpm(signal):
    signal = detrend(signal)
    sos = butter(3, [0.7, 3.0], btype="bandpass",
                 fs=fs, output="sos")
    signal = sosfiltfilt(sos, signal)

    nperseg = min(len(signal), int(fs * 5))
    freq, power = welch(
        signal, fs=fs, window="hann",
        nperseg=nperseg,
        noverlap=min(int(fs * 2.5), nperseg - 1)
    )
    band = (freq >= 0.7) & (freq <= 3.0)
    return freq[band][np.argmax(power[band])] * 60

normalized = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1
r, g, b = normalized.T
x = 3 * r - 2 * g
y = 1.5 * r + g - 1.5 * b
chrom = x - np.std(x) / (np.std(y) + 1e-8) * y

for start, end in [(0, 15), (15, 29), (0, 29)]:
    mask = (t >= start) & (t < end)
    green = rgb[mask, 1]
    pulse_chrom = chrom[mask]

    print(
        f"{start}-{end}s | "
        f"Green: {get_bpm(green):.1f} BPM | "
        f"CHROM: {get_bpm(pulse_chrom):.1f} BPM"
    )

print("Experimental estimates only; accuracy is not validated.")

