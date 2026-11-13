import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from src.camera import Camera
from src.face_detection import FaceDetector
from src.heart_rate import extract_forehead


def main():
    camera = Camera(camera_index=0)
    detector = FaceDetector()

    # Independent test buffers: these are not limited to 300 samples.
    timestamps = []
    red_values = []
    green_values = []
    blue_values = []

    try:
        if not camera.open():
            print("ERROR: Could not open camera.")
            return

        print("Collecting forehead RGB signals for 20 seconds.")
        print("Keep your face centered and lighting steady.")
        print("Press Q only if you want to stop early.")

        start = time.time()
        last_print = start

        while time.time() - start < 20:
            success, frame = camera.read()

            if not success or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            landmarks = detector.detect(frame)

            if landmarks is not None:
                roi = extract_forehead(frame, landmarks)

                if roi is not None and roi.size > 0:
                    if roi.ndim == 3 and roi.shape[2] >= 3:
                        # OpenCV image channel order is B, G, R.
                        means = np.mean(
                            roi[:, :, :3],
                            axis=(0, 1)
                        )

                        if np.all(np.isfinite(means)):
                            blue, green, red = map(float, means)

                            timestamps.append(time.time())
                            red_values.append(red)
                            green_values.append(green)
                            blue_values.append(blue)

                            now = time.time()
                            if now - last_print >= 5:
                                print(
                                    f"R: {red:.2f} "
                                    f"G: {green:.2f} "
                                    f"B: {blue:.2f}"
                                )
                                last_print = now

                            cv2.imshow("Forehead ROI", roi)

            cv2.imshow("RGB Signal Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Test stopped manually.")
                break

        print("\n--- RGB TEST RESULTS ---")
        print("Samples collected:", len(timestamps))

        if len(timestamps) >= 2:
            t = np.asarray(timestamps, dtype=float)
            t = t - t[0]

            red = np.asarray(red_values, dtype=float)
            green = np.asarray(green_values, dtype=float)
            blue = np.asarray(blue_values, dtype=float)

            elapsed = float(t[-1])
            print("Observed duration:", round(elapsed, 2), "seconds")

            if elapsed > 0:
                print(
                    "Approximate sampling rate:",
                    round((len(t) - 1) / elapsed, 2),
                    "samples/second"
                )

            for name, values in (
                ("RED", red),
                ("GREEN", green),
                ("BLUE", blue),
            ):
                print(
                    f"{name} mean: {np.mean(values):.2f}, "
                    f"standard deviation: {np.std(values):.3f}, "
                    f"minimum: {np.min(values):.2f}, "
                    f"maximum: {np.max(values):.2f}"
                )

            # Plot the raw RGB signals.
            plt.figure(figsize=(11, 6))

            plt.plot(t, red, label="RED")
            plt.plot(t, green, label="GREEN")
            plt.plot(t, blue, label="BLUE")

            plt.xlabel("Time (seconds)")
            plt.ylabel("Mean channel intensity")
            plt.title("Forehead RGB Signals - 20 Second Test")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        else:
            print("Not enough valid forehead samples were collected.")

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
