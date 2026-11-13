
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend


CSV_FILE = "forehead_v2_diagnostics.csv"
OUTPUT_FILE = "forehead_signal_spectrum.png"

df = pd.read_csv(CSV_FILE)

# Analyze a 30-second segment after initial movement.
df = df[
    (df["elapsed_seconds"] >= 10) &
    (df["elapsed_seconds"] < 40) &
    (df["accepted"] == True)
].copy()

if len(df) < 300:
    raise ValueError(
        f"Not enough samples in selected segment: {len(df)}"
    )

times = df["elapsed_seconds"].to_numpy(dtype=float)
green = df["mean_green"].to_numpy(dtype=float)

# Estimate sampling rate from timestamps.
intervals = np.diff(times)
fs = 1.0 / np.median(intervals)

# Detrend and band-pass filter the green intensity.
signal = detrend(green)

sos = butter(
    3,
    [0.7, 3.0],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered = sosfiltfilt(sos, signal)

# Frequency spectrum.
window = np.hanning(len(filtered))
spectrum = np.abs(np.fft.rfft(filtered * window))
frequencies = np.fft.rfftfreq(len(filtered), d=1.0 / fs)

valid = (frequencies >= 0.7) & (frequencies <= 3.0)

if not np.any(valid):
    raise ValueError("No frequencies in the selected pulse range.")

candidate_indices = np.where(valid)[0]
peak_index = candidate_indices[
    np.argmax(spectrum[valid])
]

peak_frequency = frequencies[peak_index]
estimated_bpm = peak_frequency * 60

# Report the three strongest peaks in the pulse range.
ranked = candidate_indices[
    np.argsort(spectrum[valid])[::-1][:3]
]

print("\n--- FOREHEAD SIGNAL ANALYSIS ---")
print(f"Samples: {len(green)}")
print(f"Segment duration: {times[-1] - times[0]:.2f} seconds")
print(f"Estimated sampling rate: {fs:.2f} Hz")
print(f"Green mean: {np.mean(green):.2f}")
print(f"Green standard deviation: {np.std(green):.2f}")
print(f"Strongest pulse-range frequency: {peak_frequency:.3f} Hz")
print(f"Corresponding BPM candidate: {estimated_bpm:.1f}")

print("\nThree strongest spectral peaks:")
for idx in ranked:
    print(
        f"{frequencies[idx]:.3f} Hz "
        f"({frequencies[idx] * 60:.1f} BPM)"
    )

plt.figure(figsize=(10, 5))
plt.plot(
    frequencies[valid],
    spectrum[valid],
    linewidth=1.2
)
plt.axvline(
    peak_frequency,
    linestyle="--",
    label=f"Peak: {estimated_bpm:.1f} BPM"
)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Forehead Green-Channel Pulse Spectrum")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
plt.close()

print(f"\nSpectrum graph saved to: {OUTPUT_FILE}")
print("This is an unvalidated BPM candidate, not a confirmed heart rate.")

