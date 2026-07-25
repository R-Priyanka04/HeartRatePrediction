import cv2
from src.camera import Camera


def main():
    print("=" * 50)
    print("HEART RATE PREDICTION - CAMERA TEST")
    print("=" * 50)

    camera = Camera(
        camera_index=0,
        width=640,
        height=480
    )

    if not camera.start():
        print("Camera test failed.")
        return

    print(f"Camera FPS: {camera.get_fps():.2f}")
    print("Press Q to close the camera.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to read camera frame.")
            break

        # Mirror camera like a selfie
        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            "MacBook Camera Test",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Heart Rate Prediction - Camera Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.stop()


if __name__ == "__main__":
    main()