import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, detrend

FILES = ["rgb_recording_1.csv", "rgb_recording_2.csv"]


def estimate_bpm(signal, fs):
    signal = detrend(signal)
    sos = butter(
        3, [0.7, 3.0], btype="bandpass",
        fs=fs, output="sos"
    )
    signal = sosfiltfilt(sos, signal)

    freq, power = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=min(len(signal), int(fs * 10)),
        noverlap=min(
            int(fs * 5),
            min(len(signal), int(fs * 10)) - 1
        ),
    )

    valid = (freq >= 0.7) & (freq <= 3.0)
    return float(freq[valid][np.argmax(power[valid])] * 60)


def main():
    for filename in FILES:
        data = np.genfromtxt(filename, delimiter=",", names=True)
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
        uniform_t = np.arange(0, t[-1], 1 / fs)
        rgb = np.column_stack([
            np.interp(uniform_t, t, rgb[:, i]) for i in range(3)
        ])

        green = rgb[:, 1]
        print(f"\n=== {filename} ===")

        for window_length in [10, 15, 20]:
            estimates = []
            step = 5

            for start in np.arange(
                0, uniform_t[-1] - window_length + 0.001, step
            ):
                mask = (
                    (uniform_t >= start)
                    & (uniform_t < start + window_length)
                )
                if np.count_nonzero(mask) < int(fs * (window_length - 1)):
                    continue

                bpm = estimate_bpm(green[mask], fs)
                estimates.append(bpm)

                print(
                    f"{window_length:2d}s window "
                    f"{start:2.0f}-{start+window_length:2.0f}s: "
                    f"{bpm:.1f} BPM"
                )

            if estimates:
                print(
                    f"{window_length}s summary: "
                    f"mean={np.mean(estimates):.1f}, "
                    f"SD={np.std(estimates):.1f} BPM"
                )


if __name__ == "__main__":
    main()
