import cv2
import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend

from src.camera import Camera
from src.face_detection import FaceDetector
from src.heart_rate import extract_forehead


def chrom_signal(rgb):
    rgb = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b

    return x - (np.std(x) / (np.std(y) + 1e-8)) * y


def estimate_bpm(rgb, fs):
    pulse = detrend(chrom_signal(rgb))

    sos = butter(
        3, [0.7, 3.0],
        btype="bandpass", fs=fs, output="sos"
    )
    pulse = sosfiltfilt(sos, pulse)

    pulse = pulse * np.hanning(len(pulse))
    spectrum = np.abs(np.fft.rfft(pulse))
    frequencies = np.fft.rfftfreq(len(pulse), d=1.0 / fs)

    valid = (frequencies >= 0.7) & (frequencies <= 3.0)
    if not np.any(valid):
        return None

    indices = np.flatnonzero(valid)
    peak = indices[np.argmax(spectrum[valid])]

    return float(frequencies[peak] * 60)


def main():
    camera = Camera(camera_index=0)
    detector = FaceDetector()
    timestamps, rgb_values = [], []

    try:
        if not camera.open():
            print("ERROR: Could not open camera.")
            return

        print("Collecting 30 seconds of RGB data.")
        print("Keep your face still and lighting steady.")

        start = time.time()

        while time.time() - start < 30:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            landmarks = detector.detect(frame)

            if landmarks is not None:
                roi = extract_forehead(frame, landmarks)

                if roi is not None and roi.size > 0:
                    b, g, r = np.mean(
                        roi[:, :, :3], axis=(0, 1)
                    )
                    values = [r, g, b]

                    if np.all(np.isfinite(values)):
                        timestamps.append(time.time())
                        rgb_values.append(values)

            cv2.imshow("CHROM Window Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    if len(timestamps) < 300:
        print("Not enough samples.")
        return

    t = np.asarray(timestamps, dtype=float)
    t -= t[0]
    rgb = np.asarray(rgb_values, dtype=float)

    # Resample to a uniform timeline before digital filtering.
    fs = (len(t) - 1) / t[-1]
    uniform_t = np.arange(0, t[-1], 1.0 / fs)
    uniform_rgb = np.column_stack([
        np.interp(uniform_t, t, rgb[:, i])
        for i in range(3)
    ])

    print(f"\nDuration: {t[-1]:.2f} s")
    print(f"Sampling rate: {fs:.2f} Hz")
    print("\n10-second window estimates:")

    results = []

    for start_time in np.arange(0, uniform_t[-1] - 10 + 0.001, 5):
        mask = (
            (uniform_t >= start_time)
            & (uniform_t < start_time + 10)
        )
        segment = uniform_rgb[mask]

        if len(segment) < int(fs * 9):
            continue

        bpm = estimate_bpm(segment, fs)
        if bpm is not None:
            results.append(bpm)
            print(
                f"{start_time:5.1f}-{start_time + 10:5.1f} s: "
                f"{bpm:.1f} BPM"
            )

    if results:
        print("\nSummary:")
        print(f"Mean estimate: {np.mean(results):.1f} BPM")
        print(f"Standard deviation: {np.std(results):.1f} BPM")
        print(f"Range: {min(results):.1f}-{max(results):.1f} BPM")
        print(
            "These are experimental estimates, not validated "
            "heart-rate measurements."
        )


if __name__ == "__main__":
    main()
