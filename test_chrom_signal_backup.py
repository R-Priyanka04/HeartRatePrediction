import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend

from src.camera import Camera
from src.face_detection import FaceDetector
from src.heart_rate import extract_forehead


def chrom_signal(rgb):
    # Normalize RGB channels to reduce baseline differences.
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

        print("Collecting RGB data for 30 seconds.")
        print("Keep your face still and lighting steady.")

        start = time.time()

        while time.time() - start < 30:
            success, frame = camera.read()
            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            landmarks = detector.detect(frame)

            if landmarks is not None:
                roi = extract_forehead(frame, landmarks)

                if roi is not None and roi.size > 0:
                    # OpenCV order is B, G, R.
                    b, g, r = np.mean(roi[:, :, :3], axis=(0, 1))
                    values = [r, g, b]

                    if np.all(np.isfinite(values)):
                        timestamps.append(time.time())
                        rgb_values.append(values)

            cv2.imshow("CHROM Signal Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if len(timestamps) < 300:
            print("Not enough samples. Please repeat the test.")
            return

        t = np.asarray(timestamps)
        t = t - t[0]
        rgb = np.asarray(rgb_values)

        fs = (len(t) - 1) / (t[-1] - t[0])
        print("\nSamples:", len(t))
        print("Duration:", round(float(t[-1]), 2), "seconds")
        print("Sampling rate:", round(float(fs), 2), "Hz")

        # Resample onto a uniform time grid for digital filtering.
        uniform_t = np.arange(t[0], t[-1], 1.0 / fs)
        uniform_rgb = np.column_stack([
            np.interp(uniform_t, t, rgb[:, i])
            for i in range(3)
        ])

        raw = chrom_signal(uniform_rgb)
        raw = detrend(raw)

        # Retain approximate pulse frequencies: 42–180 BPM.
        sos = butter(
            3, [0.7, 3.0], btype="bandpass",
            fs=fs, output="sos"
        )
        pulse = sosfiltfilt(sos, raw)

        # Estimate dominant frequency from a Hann-windowed spectrum.
        window = np.hanning(len(pulse))
        spectrum = np.abs(np.fft.rfft(pulse * window))
        frequencies = np.fft.rfftfreq(len(pulse), d=1.0 / fs)

        valid = (frequencies >= 0.7) & (frequencies <= 3.0)
        if np.any(valid):
            candidates = np.flatnonzero(valid)
            peak_index = candidates[np.argmax(spectrum[valid])]
            peak_hz = frequencies[peak_index]
            estimated_bpm = peak_hz * 60.0

            print("Experimental CHROM BPM:", round(float(estimated_bpm), 1))
            print("IMPORTANT: This is an unvalidated estimate, not a medical reading.")

        plt.figure(figsize=(11, 6))
        plt.plot(uniform_t, pulse)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Filtered CHROM amplitude")
        plt.title("CHROM Pulse Signal (Experimental)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("chrom_signal.png", dpi=150)
        print("Saved plot to chrom_signal.png")
        plt.show()

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
