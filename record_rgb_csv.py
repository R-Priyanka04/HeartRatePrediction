import csv
import time
import cv2
import numpy as np

from src.camera import Camera
from src.face_detection import FaceDetector
from src.heart_rate import extract_forehead


def main():
    camera = Camera(camera_index=0)
    detector = FaceDetector()

    rows = []
    start = None

    try:
        if not camera.open():
            print("ERROR: Could not open camera.")
            return

        print("Recording forehead RGB signals for 30 seconds.")
        print("Keep your face still and lighting steady.")
        print("Press Q to stop early.")

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
                    # OpenCV channel order is Blue, Green, Red.
                    blue, green, red = np.mean(
                        roi[:, :, :3], axis=(0, 1)
                    )

                    values = [red, green, blue]

                    if np.all(np.isfinite(values)):
                        rows.append([
                            time.time() - start,
                            float(red),
                            float(green),
                            float(blue),
                        ])

            cv2.imshow("RGB CSV Recorder", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Recording stopped early.")
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    if len(rows) < 100:
        print("Not enough samples. Please run the recording again.")
        return

    filename = "rgb_recording.csv"

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_seconds",
            "red",
            "green",
            "blue",
        ])
        writer.writerows(rows)

    times = np.array([row[0] for row in rows])
    duration = times[-1] - times[0]
    fps = (len(times) - 1) / duration if duration > 0 else 0

    print("\nRecording saved:", filename)
    print("Samples:", len(rows))
    print("Duration:", round(float(duration), 2), "seconds")
    print("Approximate sampling rate:", round(float(fps), 2), "Hz")


if __name__ == "__main__":
    main()
