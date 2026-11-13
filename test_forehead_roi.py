
import cv2
import numpy as np
import time

from src.camera import Camera
from src.face_detection import FaceDetector
from src.eye_roi import get_landmark_forehead_box


def main():
    camera = Camera(camera_index=0)
    face_detector = FaceDetector()

    measurements = []
    start_time = time.time()
    last_report = start_time

    try:
        if not camera.open():
            print("ERROR: Could not open camera.")
            return

        print("Forehead ROI stability test started.")
        print("Keep your head still for 15 seconds.")
        print("Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            landmarks = face_detector.detect(frame)

            if landmarks is not None:
                box = get_landmark_forehead_box(
                    frame, landmarks
                )

                if box is not None:
                    x, y, w, h = box
                    roi = frame[y:y + h, x:x + w]

                    if roi.size > 0:
                        measurements.append(box)

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            "Forehead ROI stability test",
                            (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2
                        )

                        cv2.imshow("Forehead ROI", roi)

            elapsed = time.time() - start_time
            now = time.time()

            if now - last_report >= 5:
                last_report = now

                if measurements:
                    recent = np.asarray(measurements[-150:])

                    print("\n--- ROI stability report ---")
                    print("Samples:", len(recent))
                    print("Mean (x, y, w, h):",
                          np.round(recent.mean(axis=0), 2))
                    print("Std  (x, y, w, h):",
                          np.round(recent.std(axis=0), 2))
                    print("Elapsed seconds:", round(elapsed, 1))

            cv2.imshow("Forehead ROI Stability Test", frame)

            if elapsed >= 15:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Test interrupted.")

    finally:
        if measurements:
            data = np.asarray(measurements)

            print("\n=== FINAL ROI STABILITY RESULTS ===")
            print("Total samples:", len(data))
            print("Mean (x, y, w, h):",
                  np.round(data.mean(axis=0), 2))
            print("Std (x, y, w, h):",
                  np.round(data.std(axis=0), 2))
            print("Min (x, y, w, h):", data.min(axis=0))
            print("Max (x, y, w, h):", data.max(axis=0))

        camera.release()
        face_detector.close()
        cv2.destroyAllWindows()
        print("Test finished.")


if __name__ == "__main__":
    main()
