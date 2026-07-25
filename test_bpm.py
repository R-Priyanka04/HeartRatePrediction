import cv2
import numpy as np

from src.camera import Camera
from src.face_detection import FaceDetector
from src.signal_processing import estimate_bpm


MIN_SAMPLES = 180
MAX_SAMPLES = 300

signal_buffer = []
bpm_history = []


def main():

    print("=" * 60)
    print("HEART RATE PREDICTION - LIVE BPM TEST")
    print("=" * 60)

    camera = Camera()
    detector = FaceDetector()

    # -------------------------------
    # Open camera
    # -------------------------------

    if not camera.open():
        print("Could not open camera.")
        detector.close()
        return

    print("Camera ready.")
    print("Look directly at the camera.")
    print("Keep your head still.")
    print("Wait for BPM calculation.")
    print("Press Q to quit.")

    while True:

        success, frame = camera.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        # -------------------------------
        # Face detection
        # -------------------------------

        face_landmarks = detector.detect(frame)

        bpm = None

        if face_landmarks is not None:

            h, w, _ = frame.shape

            # -------------------------------
            # Forehead landmarks
            # -------------------------------

            ids = [10, 338, 297, 332, 284]

            points = []

            for idx in ids:

                landmark = face_landmarks.landmark[idx]

                px = int(landmark.x * w)
                py = int(landmark.y * h)

                points.append((px, py))

            points = np.array(points)

            x, y, rw, rh = cv2.boundingRect(points)

            # Make ROI smaller
            x += int(rw * 0.20)
            y += int(rh * 0.15)

            rw = int(rw * 0.60)
            rh = int(rh * 0.60)

            x = max(0, x)
            y = max(0, y)

            x2 = min(w, x + rw)
            y2 = min(h, y + rh)

            roi = frame[y:y2, x:x2]

            # -------------------------------
            # Draw forehead ROI
            # -------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "FOREHEAD ROI",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # -------------------------------
            # Green channel extraction
            # -------------------------------

            if roi.size > 0:

                green_value = float(
                    np.mean(roi[:, :, 1])
                )

                signal_buffer.append(green_value)

                if len(signal_buffer) > MAX_SAMPLES:
                    signal_buffer.pop(0)

                # -------------------------------
                # FPS
                # -------------------------------

                fps = camera.get_fps()

                if fps is None or fps <= 0:
                    fps = 30.0

                # -------------------------------
                # BPM calculation
                # -------------------------------

                if len(signal_buffer) >= MIN_SAMPLES:

                    estimated_bpm = estimate_bpm(
                        signal_buffer,
                        fps
                    )

                    if estimated_bpm is not None:

                        bpm_history.append(
                            estimated_bpm
                        )

                        if len(bpm_history) > 10:
                            bpm_history.pop(0)

                        bpm = float(
                            np.median(bpm_history)
                        )

                # -------------------------------
                # Display sample count
                # -------------------------------

                cv2.putText(
                    frame,
                    f"Samples: {len(signal_buffer)}/{MIN_SAMPLES}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # -------------------------------
                # Display green value
                # -------------------------------

                cv2.putText(
                    frame,
                    f"Green: {green_value:.2f}",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # -------------------------------
                # Display BPM
                # -------------------------------

                if bpm is None:

                    cv2.putText(
                        frame,
                        "BPM: Calculating...",
                        (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2
                    )

                else:

                    cv2.putText(
                        frame,
                        f"BPM: {bpm:.1f}",
                        (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        3
                    )

                    print(
                        f"Samples: {len(signal_buffer)} | "
                        f"Green: {green_value:.2f} | "
                        f"FPS: {fps:.1f} | "
                        f"BPM: {bpm:.1f}"
                    )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        # -------------------------------
        # Show window
        # -------------------------------

        cv2.imshow(
            "Heart Rate Prediction - BPM Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # -------------------------------
    # Cleanup
    # -------------------------------

    camera.release()
    detector.close()

    cv2.destroyAllWindows()

    print("BPM test finished.")


if __name__ == "__main__":
    main()