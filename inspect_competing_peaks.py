
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend, welch, find_peaks

FILE = "rgb_session_2.csv"
START = 40.0
WINDOW = 20.0
FPS = 20.0


def main():
    df = pd.read_csv(FILE)
    df = df[["time_seconds", "green"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    df = df.sort_values("time_seconds").drop_duplicates("time_seconds")

    t = df["time_seconds"].to_numpy(dtype=float)
    green = df["green"].to_numpy(dtype=float)
    t = t - t[0]

    uniform_t = np.arange(0, t[-1], 1.0 / FPS)
    uniform_green = np.interp(uniform_t, t, green)

    mask = (
        (uniform_t >= START)
        & (uniform_t < START + WINDOW)
    )
    segment = uniform_green[mask]

    if len(segment) < int(FPS * WINDOW * 0.95):
        raise ValueError("Not enough samples in selected window.")

    signal = detrend(segment)

    sos = butter(
        3, [0.7, 3.0],
        btype="bandpass",
        fs=FPS,
        output="sos",
    )
    signal = sosfiltfilt(sos, signal)

    # FFT spectrum
    weighted = signal * np.hanning(len(signal))
    fft_power = np.abs(np.fft.rfft(weighted)) ** 2
    fft_freq = np.fft.rfftfreq(len(signal), d=1.0 / FPS)

    # Welch spectrum with one full-window segment
    welch_freq, welch_power = welch(
        signal,
        fs=FPS,
        window="hann",
        nperseg=len(signal),
        noverlap=0,
        nfft=max(len(signal), 4096),
        detrend="constant",
        scaling="density",
    )

    def top_peaks(freq, power, method):
        valid = (freq >= 0.7) & (freq <= 3.0)
        f, p = freq[valid], power[valid]
        peaks, _ = find_peaks(p)

        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(p))])

        order = peaks[np.argsort(p[peaks])[::-1]][:5]

        print(f"\n{method} top peaks:")
        for idx in order:
            print(
                f"  {f[idx] * 60:6.2f} BPM"
                f" | relative power: {p[idx] / np.max(p):.3f}"
            )

    top_peaks(fft_freq, fft_power, "FFT")
    top_peaks(welch_freq, welch_power, "Welch")

    plt.figure(figsize=(10, 5))

    for freq, power, label in [
        (fft_freq, fft_power, "FFT"),
        (welch_freq, welch_power, "Welch"),
    ]:
        valid = (freq >= 0.7) & (freq <= 3.0)
        f, p = freq[valid], power[valid]
        p = p / (np.max(p) + 1e-12)
        plt.plot(f * 60, p, label=label)

    plt.xlabel("Frequency (BPM)")
    plt.ylabel("Relative spectral power")
    plt.title("Session 2: Competing Peaks, 40–60 seconds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("session2_competing_peaks.png", dpi=150)
    plt.close()

    print("\nSaved: session2_competing_peaks.png")


if __name__ == "__main__":
    main()

