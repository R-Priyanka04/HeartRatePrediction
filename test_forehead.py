import cv2

from src.camera import Camera
from src.face_detection import FaceDetector
from src.heart_rate import extract_forehead, update_signal, green_signal


def main():

    print("=" * 55)
    print("HEART RATE PREDICTION - FOREHEAD ROI TEST")
    print("=" * 55)

    camera = Camera(
        camera_index=0,
        width=640,
        height=480
    )

    detector = FaceDetector()

    if not camera.start():
        print("Could not start camera.")
        return

    print("Camera ready.")
    print("Look directly at the camera.")
    print("Keep your head reasonably still.")
    print("Press Q to quit.")

    while True:

        success, frame = camera.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        landmarks = detector.detect(frame)

        if landmarks is not None:

            roi, coords = extract_forehead(
                frame,
                landmarks
            )

            if roi is not None and coords is not None:

                x1, y1, x2, y2 = coords

                # Draw forehead measurement box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                value = update_signal(roi)

                cv2.putText(
                    frame,
                    "FOREHEAD ROI",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                if value is not None:

                    cv2.putText(
                        frame,
                        f"Green: {value:.2f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

                cv2.putText(
                    frame,
                    f"Samples: {len(green_signal)}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                "FACE DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE - LOOK AT CAMERA",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Heart Rate Prediction - Forehead Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    detector.close()
    camera.stop()


if __name__ == "__main__":
    main()