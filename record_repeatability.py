
import csv
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.face_detection import FaceDetector
from src.heart_rate import extract_forehead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Recording session number: 1, 2, or 3",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Recording duration in seconds (default: 60)",
    )
    args = parser.parse_args()

    filename = Path(f"rgb_session_{args.session}.csv")

    if filename.exists():
        print(f"ERROR: {filename} already exists.")
        print("Keeping the existing file unchanged.")
        print("Choose another session number or rename the old file.")
        return

    camera = Camera(camera_index=0)
    detector = FaceDetector()

    rows = []
    start = None

    try:
        if not camera.open():
            print("ERROR: Could not open camera.")
            return

        print(f"Session {args.session}: recording for {args.duration} seconds.")
        print("Sit still, face the camera, and keep lighting steady.")
        print("Press Q to stop early.")

        start = time.time()

        while time.time() - start < args.duration:
            success, frame = camera.read()

            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            landmarks = detector.detect(frame)

            if landmarks is not None:
                roi = extract_forehead(frame, landmarks)

                if roi is not None and roi.size > 0:
                    # OpenCV image channels are Blue, Green, Red.
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

            elapsed = time.time() - start
            cv2.putText(
                frame,
                f"Session {args.session} | {elapsed:.1f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Repeatability Recorder", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Recording stopped early.")
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    if len(rows) < 100:
        print(f"Not enough samples ({len(rows)}). No file saved.")
        return

    with filename.open("w", newline="") as file:
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

    print(f"\nSaved: {filename}")
    print("Samples:", len(rows))
    print("Duration:", round(float(duration), 2), "seconds")
    print("Approximate sampling rate:", round(float(fps), 2), "Hz")


if __name__ == "__main__":
    main()

