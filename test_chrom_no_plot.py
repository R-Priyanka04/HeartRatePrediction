import cv2
import time
import numpy as np
import matplotlib

# Use a non-interactive backend; do not open a plot window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend

from src.camera import Camera
from src.face_detection import FaceDetector
from src.heart_rate import extract_forehead


def chrom_signal(rgb):
    rgb = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b

    alpha = np.std(x) / (np.std(y) + 1e-8)
    return x - alpha * y


def main():
    camera = Camera(camera_index=0)
    detector = FaceDetector()
    timestamps, rgb_values = [], []

    try:
        if not camera.open():
            print("ERROR: Could not open camera.")
            return

        print("Collecting RGB signals for 30 seconds.")
        start = time.time()

        while time.time() - start < 30:
            success, frame = camera.read()
            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            landmarks = detector.detect(frame)

            if landmarks is not None:
                roi = extract_forehead(frame, landmarks)

                if roi is not None and roi.size > 0 and roi.ndim == 3:
                    b, g, r = np.mean(roi[:, :, :3], axis=(0, 1))
                    values = [r, g, b]

                    if np.all(np.isfinite(values)):
                        timestamps.append(time.time())
                        rgb_values.append(values)

    finally:
        # Close camera and detector before plotting or saving.
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    print("CAPTURE_AND_CLEANUP_FINISHED")
    print("Samples:", len(timestamps))

    if len(timestamps) < 300:
        print("Not enough samples for analysis.")
        return

    t = np.asarray(timestamps, dtype=float)
    t -= t[0]
    rgb = np.asarray(rgb_values, dtype=float)
    fs = (len(t) - 1) / t[-1]

    uniform_t = np.arange(0, t[-1], 1.0 / fs)
    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, i])
        for i in range(3)
    ])

    pulse = detrend(chrom_signal(uniform_rgb))
    sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fs, output="sos")
    pulse = sosfiltfilt(sos, pulse)

    window = np.hanning(len(pulse))
    spectrum = np.abs(np.fft.rfft(pulse * window))
    frequencies = np.fft.rfftfreq(len(pulse), d=1.0 / fs)
    valid = (frequencies >= 0.7) & (frequencies <= 3.0)

    if np.any(valid):
        indices = np.flatnonzero(valid)
        peak = indices[np.argmax(spectrum[valid])]
        print("Experimental BPM:", round(float(frequencies[peak] * 60), 1))

    plt.figure(figsize=(10, 5))
    plt.plot(uniform_t, pulse)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Filtered CHROM amplitude")
    plt.title("Experimental CHROM Signal")
    plt.tight_layout()
    plt.savefig("chrom_signal_no_plot.png", dpi=150)
    plt.close()

    print("PLOT_SAVED_WITHOUT_DISPLAY")
    print("DIAGNOSTIC_FINISHED")


if __name__ == "__main__":
    main()
