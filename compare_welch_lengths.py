import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, detrend

data = np.genfromtxt("rgb_recording.csv", delimiter=",", names=True)
t = data["time_seconds"]
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

normalized = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1
r, g, b = normalized.T
x = 3 * r - 2 * g
y = 1.5 * r + g - 1.5 * b
chrom = x - np.std(x) / (np.std(y) + 1e-8) * y

sos = butter(3, [0.7, 3.0], btype="bandpass",
             fs=fs, output="sos")
chrom = sosfiltfilt(sos, detrend(chrom))

for start, end in [(5, 15), (10, 20), (15, 25)]:
    segment = chrom[(t >= start) & (t < end)]
    segment = detrend(segment)

    print(f"\nWindow {start}-{end}s")

    for seconds in [5, 10, 15]:
        nperseg = min(len(segment), int(fs * seconds))
        if nperseg < 16:
            continue

        freq, power = welch(
            segment,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2
        )

        band = (freq >= 0.7) & (freq <= 3.0)
        f = freq[band]
        p = power[band]
        order = np.argsort(p)[::-1]
        top = order[:3]

        print(
            f"Welch segment {seconds}s: "
            + ", ".join(
                f"{f[i] * 60:.1f} BPM"
                for i in top
            )
        )

print("\nExperimental only; no reference heart rate available.")

