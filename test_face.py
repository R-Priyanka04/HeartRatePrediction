import cv2

from src.camera import Camera
from src.face_detection import FaceDetector


def main():

    print("=" * 50)
    print("HEART RATE PREDICTION - FACE DETECTION TEST")
    print("=" * 50)

    camera = Camera(
        camera_index=0,
        width=640,
        height=480
    )

    detector = FaceDetector()

    if not camera.start():
        print("Could not start camera.")
        return

    print("Face detector started.")
    print("Look directly at the camera.")
    print("Press Q to quit.")

    while True:

        success, frame = camera.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        landmarks = detector.detect(frame)

        if landmarks is not None:

            detector.draw_landmarks(
                frame,
                landmarks
            )

            cv2.putText(
                frame,
                "FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Heart Rate Prediction - Face Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    camera.stop()


if __name__ == "__main__":
    main()