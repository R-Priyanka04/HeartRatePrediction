
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch, detrend


CSV_FILE = "forehead_v2_diagnostics.csv"

LOW_HZ = 0.7
HIGH_HZ = 3.0

WINDOW_SECONDS = 20
STEP_SECONDS = 2


def estimate_bpm(signal, fs):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 20 or not np.all(np.isfinite(signal)):
        return np.nan, None, None

    signal = detrend(signal)

    sos = butter(
        3,
        [LOW_HZ, HIGH_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError:
        return np.nan, None, None

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=len(filtered),
        noverlap=0,
        window="hann",
    )

    valid = (
        (frequencies >= LOW_HZ)
        & (frequencies <= HIGH_HZ)
    )

    if not np.any(valid):
        return np.nan, frequencies, power

    valid_indices = np.flatnonzero(valid)
    best = valid_indices[np.argmax(power[valid])]

    return (
        float(frequencies[best] * 60),
        frequencies,
        power,
    )


def main():
    df = pd.read_csv(CSV_FILE)

    required = [
        "elapsed_seconds",
        "accepted",
        "mean_red",
        "mean_green",
        "mean_blue",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[
        (df["accepted"] == True)
        & (df["elapsed_seconds"] >= 8)
        & (df["elapsed_seconds"] <= 55)
    ].dropna(subset=required).sort_values("elapsed_seconds")

    t = df["elapsed_seconds"].to_numpy(float)
    rgb = df[
        ["mean_red", "mean_green", "mean_blue"]
    ].to_numpy(float)

    keep = np.concatenate(([True], np.diff(t) > 0))
    t, rgb = t[keep], rgb[keep]

    if len(t) < 100:
        raise ValueError("Not enough valid RGB samples.")

    fs = 1.0 / np.median(np.diff(t))
    uniform_t = np.arange(t[0], t[-1], 1.0 / fs)

    rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, i])
        for i in range(3)
    ])

    red, green, blue = rgb.T

    # Method 1: green-channel baseline
    green_signal = green / (np.mean(green) + 1e-8) - 1

    # Method 2: normalized green
    normalized_green = green / (
        red + green + blue + 1e-8
    )
    normalized_green -= np.mean(normalized_green)

    # Method 3: CHROM-style color combination
    normalized_rgb = (
        rgb / (np.mean(rgb, axis=0) + 1e-8)
    ) - 1

    r, g, b = normalized_rgb.T

    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b

    chrom = x - (
        np.std(x) / (np.std(y) + 1e-8)
    ) * y

    signals = {
        "Green": green_signal,
        "Normalized green": normalized_green,
        "CHROM": chrom,
    }

    rows = []

    for start in np.arange(
        max(8, uniform_t[0]),
        min(55 - WINDOW_SECONDS, uniform_t[-1] - WINDOW_SECONDS)
        + 1e-9,
        STEP_SECONDS,
    ):
        end = start + WINDOW_SECONDS
        mask = (
            (uniform_t >= start)
            & (uniform_t < end)
        )

        if mask.sum() < int(fs * WINDOW_SECONDS * 0.95):
            continue

        row = {
            "start_seconds": start,
            "end_seconds": end,
            "midpoint_seconds": (start + end) / 2,
        }

        for name, signal in signals.items():
            bpm, _, _ = estimate_bpm(signal[mask], fs)
            row[f"{name}_bpm"] = bpm

        rows.append(row)

    results = pd.DataFrame(rows)

    if results.empty:
        raise ValueError("No complete analysis windows found.")

    results.to_csv(
        "forehead_rgb_method_comparison.csv",
        index=False,
    )

    print("\n--- RGB METHOD COMPARISON ---")
    print(f"Estimated sampling rate: {fs:.2f} Hz")
    print(
        results[
            [
                "start_seconds",
                "Green_bpm",
                "Normalized green_bpm",
                "CHROM_bpm",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nWindow-to-window standard deviation (BPM):")
    for name in signals:
        values = results[f"{name}_bpm"].dropna()
        print(f"{name}: {values.std():.2f}")

    print(
        "\nThese are experimental estimates. "
        "No reference heart rate is available."
    )

    plt.figure(figsize=(12, 6))

    for name in signals:
        plt.plot(
            results["midpoint_seconds"],
            results[f"{name}_bpm"],
            marker="o",
            label=name,
        )

    plt.xlabel("Window midpoint (seconds)")
    plt.ylabel("BPM candidate")
    plt.title("Green vs Normalized Green vs CHROM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "forehead_rgb_method_comparison.png",
        dpi=150,
    )
    plt.close()

    print("Saved: forehead_rgb_method_comparison.csv")
    print("Saved: forehead_rgb_method_comparison.png")


if __name__ == "__main__":
    main()

