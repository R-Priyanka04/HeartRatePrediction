import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        """
        Detect face landmarks from a camera frame.
        """

        if frame is None:
            return None

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]

        return None

    def draw_landmarks(self, frame, face_landmarks):
        """
        Draw MediaPipe face landmarks.
        """

        if face_landmarks is None:
            return frame

        height, width, _ = frame.shape

        for landmark in face_landmarks.landmark:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

        return frame

    def close(self):
        """
        Release MediaPipe resources.
        """

        self.face_mesh.close()