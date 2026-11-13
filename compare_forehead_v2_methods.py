
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch, detrend


CSV_FILE = "forehead_v2_diagnostics.csv"
START_SECONDS = 10
END_SECONDS = 40
MIN_BPM = 42
MAX_BPM = 180


def bandpass(signal, fs):
    signal = detrend(signal)

    sos = butter(
        3,
        [0.7, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(sos, signal)


def estimate_bpm(signal, fs):
    filtered = bandpass(signal, fs)

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=min(len(filtered), int(fs * 8)),
        noverlap=None,
    )

    valid = (
        (frequencies >= MIN_BPM / 60)
        & (frequencies <= MAX_BPM / 60)
    )

    indices = np.where(valid)[0]

    if len(indices) < 2:
        return None, None, None, frequencies, power

    order = indices[np.argsort(power[indices])[::-1]]
    best = order[0]
    second = order[1]

    bpm = frequencies[best] * 60
    second_bpm = frequencies[second] * 60
    ratio = power[best] / max(power[second], 1e-15)

    return bpm, second_bpm, ratio, frequencies, power


def main():
    df = pd.read_csv(CSV_FILE)

    df = df[
        (df["elapsed_seconds"] >= START_SECONDS)
        & (df["elapsed_seconds"] <= END_SECONDS)
        & (df["accepted"] == True)
    ].copy()

    df = df.dropna(
        subset=["elapsed_seconds", "mean_red",
                "mean_green", "mean_blue"]
    )

    if len(df) < 100:
        raise ValueError(
            f"Not enough valid samples: {len(df)}"
        )

    # Use timestamps to create a uniformly sampled signal.
    # This avoids treating irregular camera frame intervals
    # as if every sample were equally spaced.
    t = df["elapsed_seconds"].to_numpy(dtype=float)
    order = np.argsort(t)
    t = t[order]
    df = df.iloc[order]

    unique = np.concatenate(([True], np.diff(t) > 0))
    t = t[unique]
    df = df.iloc[np.flatnonzero(unique)]

    fs_original = 1.0 / np.median(np.diff(t))
    fs = min(fs_original, 30.0)

    uniform_t = np.arange(t[0], t[-1], 1.0 / fs)

    red = np.interp(
        uniform_t, t, df["mean_red"].to_numpy(dtype=float)
    )
    green = np.interp(
        uniform_t, t, df["mean_green"].to_numpy(dtype=float)
    )
    blue = np.interp(
        uniform_t, t, df["mean_blue"].to_numpy(dtype=float)
    )

    # Normalize each channel relative to its own mean.
    rn = red / np.mean(red) - 1.0
    gn = green / np.mean(green) - 1.0
    bn = blue / np.mean(blue) - 1.0

    # CHROM components and simplified CHROM signal.
    x = 3.0 * rn - 2.0 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    chrom = x - (
        np.std(x) / max(np.std(y), 1e-12)
    ) * y

    methods = {
        "Green": gn,
        "CHROM X": x,
        "CHROM Y": y,
        "CHROM-like": chrom,
    }

    print("\n--- FOREHEAD V2: GREEN VS CHROM ---")
    print(f"Selected window: {t[0]:.2f}-{t[-1]:.2f} seconds")
    print(f"Original median sampling rate: {fs_original:.2f} Hz")
    print(f"Uniform analysis sampling rate: {fs:.2f} Hz")
    print(f"Uniform samples: {len(uniform_t)}")
    print("\nExperimental estimates; no reference HR is available.")
    print("-" * 75)
    print(
        f"{'Method':<14}{'Peak BPM':>12}"
        f"{'2nd BPM':>12}{'Power ratio':>15}"
    )
    print("-" * 75)

    plt.figure(figsize=(10, 5))

    for name, signal in methods.items():
        bpm, second, ratio, frequencies, power = estimate_bpm(
            signal, fs
        )

        if bpm is None:
            print(f"{name:<14} unavailable")
            continue

        print(
            f"{name:<14}{bpm:>12.1f}"
            f"{second:>12.1f}{ratio:>15.2f}"
        )

        valid = (
            (frequencies >= MIN_BPM / 60)
            & (frequencies <= MAX_BPM / 60)
        )

        plt.plot(
            frequencies[valid] * 60,
            power[valid],
            label=f"{name}: {bpm:.1f} BPM",
        )

    plt.xlabel("Frequency (BPM)")
    plt.ylabel("Welch power")
    plt.title("Green vs CHROM — Same Forehead Recording")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "forehead_v2_green_chrom_comparison.png",
        dpi=150,
    )
    plt.close()

    print("\nGraph saved to:")
    print("forehead_v2_green_chrom_comparison.png")
    print("\nA stronger peak indicates spectral prominence,")
    print("not necessarily a more accurate heart rate.")


if __name__ == "__main__":
    main()


