import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, detrend

data = np.genfromtxt("rgb_recording.csv", delimiter=",", names=True)
t = data["time_seconds"]
green = data["green"]

valid = np.isfinite(t) & np.isfinite(green)
t, green = t[valid], green[valid]

order = np.argsort(t)
t, green = t[order], green[order]
t, idx = np.unique(t, return_index=True)
green = green[idx]
t -= t[0]

fs = (len(t) - 1) / t[-1]

for start, end in [(0, 15), (15, 29), (0, 29)]:
    mask = (t >= start) & (t < end)
    x = detrend(green[mask])

    sos = butter(
        3, [0.7, 3.0], btype="bandpass",
        fs=fs, output="sos"
    )
    x = sosfiltfilt(sos, x)

    nperseg = min(len(x), int(fs * 5))
    freq, power = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=min(int(fs * 2.5), nperseg - 1),
    )

    band = (freq >= 0.7) & (freq <= 3.0)
    bpm = freq[band][np.argmax(power[band])] * 60

    print(f"{start}-{end}s: {bpm:.1f} BPM ({len(x)} samples)")

print("Experimental estimates only; accuracy is not validated.")
