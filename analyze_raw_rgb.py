import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.genfromtxt(
    "rgb_recording.csv",
    delimiter=",",
    names=True,
)

t = np.asarray(data["time_seconds"], dtype=float)
red = np.asarray(data["red"], dtype=float)
green = np.asarray(data["green"], dtype=float)
blue = np.asarray(data["blue"], dtype=float)

valid = (
    np.isfinite(t)
    & np.isfinite(red)
    & np.isfinite(green)
    & np.isfinite(blue)
)

t, red, green, blue = (
    t[valid], red[valid], green[valid], blue[valid]
)

order = np.argsort(t)
t, red, green, blue = (
    t[order], red[order], green[order], blue[order]
)

print("Samples:", len(t))
print("Duration:", round(float(t[-1] - t[0]), 2), "seconds")

for name, values in (
    ("RED", red),
    ("GREEN", green),
    ("BLUE", blue),
):
    differences = np.abs(np.diff(values))

    print(f"\n{name}")
    print("Mean:", round(float(np.mean(values)), 2))
    print("Standard deviation:", round(float(np.std(values)), 3))
    print("Maximum consecutive change:", round(float(np.max(differences)), 2))
    print("95th percentile change:", round(float(np.percentile(differences, 95)), 2))

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(t, red)
axes[0].set_ylabel("Red intensity")
axes[0].set_title("Raw Forehead RGB Signals")

axes[1].plot(t, green)
axes[1].set_ylabel("Green intensity")

axes[2].plot(t, blue)
axes[2].set_ylabel("Blue intensity")
axes[2].set_xlabel("Time (seconds)")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("raw_rgb_analysis.png", dpi=150)
plt.close()

print("\nSaved graph: raw_rgb_analysis.png")
