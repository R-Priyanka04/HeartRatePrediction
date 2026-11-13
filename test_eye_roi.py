
import cv2

from src.camera import Camera
from src.face_detection import FaceDetector
from src.eye_roi import get_landmark_forehead_box


def main():
    camera = Camera(camera_index=0)
    face_detector = FaceDetector()

    try:
        if not camera.open():
            print("ERROR: Could not open camera.")
            return

        print("MediaPipe forehead ROI test started. Press Q to quit.")

        frame_count = 0

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            landmarks = face_detector.detect(frame)

            if landmarks is not None:
                box = get_landmark_forehead_box(frame, landmarks)

                if box is not None:
                    x, y, w, h = box

                    # Draw the forehead region.
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "MediaPipe Forehead ROI",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    roi = frame[y:y + h, x:x + w]

                    if roi.size > 0:
                        cv2.imshow("Forehead ROI", roi)

                    frame_count += 1

                    if frame_count % 30 == 0:
                        print(
                            f"Forehead ROI: {(x, y, w, h)}"
                        )
                        print(f"ROI shape: {roi.shape}")

            else:
                cv2.putText(
                    frame,
                    "Face not detected",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("MediaPipe Forehead Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Test interrupted.")

    finally:
        camera.release()
        face_detector.close()
        cv2.destroyAllWindows()
        print("Test finished.")


if __name__ == "__main__":
    main()
