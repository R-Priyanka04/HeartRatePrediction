import cv2


def test_camera():

    print("Opening MacBook camera...")

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():
        print("ERROR: Camera could not be opened.")
        return

    print("Camera opened successfully.")

    # Give camera time to initialise
    for _ in range(30):
        cap.read()

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to read camera frame.")
            continue

        # Mirror image
        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            "CAMERA TEST - Press Q to exit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "MacBook Camera Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()