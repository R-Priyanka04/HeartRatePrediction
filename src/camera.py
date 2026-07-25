import cv2
import time


class Camera:
    """
    Camera manager for the Heart Rate Prediction System.
    Designed for macOS / MacBook camera.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    # ------------------------------------------------
    # Open camera
    # ------------------------------------------------
    def open(self):

        print("Opening MacBook camera...")

        self.cap = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_AVFOUNDATION
        )

        if not self.cap.isOpened():
            print("Could not open camera.")
            self.cap = None
            return False

        # Camera resolution
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        # Give camera time to initialize
        time.sleep(1)

        # Warm-up frames
        for _ in range(10):
            self.cap.read()

        print("Camera opened successfully.")

        return True

    # ------------------------------------------------
    # Read frame
    # ------------------------------------------------
    def read(self):

        if self.cap is None:
            return False, None

        if not self.cap.isOpened():
            return False, None

        success, frame = self.cap.read()

        if not success:
            return False, None

        return True, frame

    # ------------------------------------------------
    # Get FPS
    # ------------------------------------------------
    def get_fps(self):

        if self.cap is None:
            return 30.0

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps is None or fps <= 0:
            return 30.0

        return float(fps)

    # ------------------------------------------------
    # Release camera
    # ------------------------------------------------
    def release(self):

        if self.cap is not None:

            if self.cap.isOpened():
                self.cap.release()

            self.cap = None

        print("Camera closed.")

    # ------------------------------------------------
    # Check camera
    # ------------------------------------------------
    def is_opened(self):

        if self.cap is None:
            return False

        return self.cap.isOpened()