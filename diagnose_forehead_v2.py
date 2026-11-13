
import cv2
import csv
import time
from datetime import datetime

from src.camera import Camera
from src.face_detection import FaceDetector
from src.forehead_tracker_v2 import ForeheadTracker


CSV_FILE = "forehead_v2_diagnostics.csv"


def main():
    camera = Camera()
    face_detector = FaceDetector()
    tracker = ForeheadTracker()

    accepted_count = 0
    rejected_count = 0
    start_time = time.time()

    fields = [
        "timestamp", "elapsed_seconds", "face_detected",
        "accepted", "reason", "x1", "y1", "x2", "y2",
        "width", "height", "mean_blue", "mean_green", "mean_red"
    ]

    try:
        camera.open()

        with open(CSV_FILE, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()

            print("Recording forehead tracker diagnostics.")
            print("Press Q in the camera window to stop.")

            while True:
                success, frame = camera.read()

                if not success or frame is None:
                    print("Could not read a camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                elapsed = time.time() - start_time

                landmarks = face_detector.detect(frame)
                face_detected = landmarks is not None

                result = tracker.extract(frame, landmarks)

                accepted = bool(result.get("valid", False))
                reason = result.get("reason", "")

                box = result.get("box")
                width = result.get("width", 0)
                height = result.get("height", 0)

                x1 = y1 = x2 = y2 = ""
                mean_blue = mean_green = mean_red = ""

                if box is not None:
                    x1, y1, x2, y2 = box

                if accepted:
                    accepted_count += 1
                    roi = result.get("roi")

                    if roi is not None and roi.size > 0:
                        means = roi.mean(axis=(0, 1))
                        mean_blue = round(float(means[0]), 3)
                        mean_green = round(float(means[1]), 3)
                        mean_red = round(float(means[2]), 3)

                    if box is not None:
                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2
                        )
                else:
                    rejected_count += 1

                    if box is not None:
                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 0, 255),
                            2
                        )

                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_seconds": round(elapsed, 4),
                    "face_detected": face_detected,
                    "accepted": accepted,
                    "reason": reason,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "mean_blue": mean_blue,
                    "mean_green": mean_green,
                    "mean_red": mean_red
                })

                status = "ACCEPTED" if accepted else "REJECTED"

                cv2.putText(
                    frame,
                    f"{status}: {reason}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if accepted else (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"ROI: {width} x {height}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Accepted: {accepted_count}  Rejected: {rejected_count}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Forehead Tracker V2 Diagnostics", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nRecording stopped.")

    finally:
        camera.release()
        face_detector.close()
        cv2.destroyAllWindows()

    total = accepted_count + rejected_count
    percentage = (
        100 * accepted_count / total if total else 0
    )

    print("\n--- Diagnostic Summary ---")
    print(f"Accepted frames: {accepted_count}")
    print(f"Rejected frames: {rejected_count}")
    print(f"Acceptance rate: {percentage:.2f}%")
    print(f"CSV saved to: {CSV_FILE}")


if __name__ == "__main__":
    main()

