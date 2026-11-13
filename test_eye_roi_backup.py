
import cv2

from src.camera import Camera
from src.face_detection import FaceDetector
from src.eye_roi import (
    EyeROILocalizer,
    get_landmark_forehead_box,
    draw_eye_and_forehead_roi
)


def main():
    camera = Camera(camera_index=0)
    localizer = EyeROILocalizer()
    face_detector = FaceDetector()

    try:
        if not camera.open():
            print("ERROR: Could not open camera.")
            return

        print("Camera started. Press Q to quit.")
        frame_count = 0

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            # Detect the face first.
            landmarks = face_detector.detect(frame)

            # Validate eye detections against those landmarks.
            result = localizer.get_forehead_roi(
                frame, landmarks
            )

            reference_box = None

            if landmarks is not None:
                reference_box = get_landmark_forehead_box(
                    frame, landmarks
                )

            if result is not None:
                roi, forehead_box, eye_box = result

                draw_eye_and_forehead_roi(
                    frame,
                    eye_box,
                    forehead_box,
                    reference_box
                )

                frame_count += 1

                if frame_count % 30 == 0:
                    print("-" * 40)
                    print("Eye box:", eye_box)
                    print("Paper ROI:", forehead_box)
                    print("Reference ROI:", reference_box)
                    print("ROI shape:", roi.shape)

                cv2.imshow("Paper Forehead ROI", roi)

            else:
                message = (
                    "Face/eye pair not validated - face camera"
                )

                cv2.putText(
                    frame,
                    message,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Eye-based ROI test", frame)

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
