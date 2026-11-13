
import cv2
import numpy as np


class EyeROILocalizer:
    """
    Detect a plausible pair of eyes using Haar detection,
    validate the detections against MediaPipe landmarks,
    and calculate the paper-based forehead ROI.
    """

    # MediaPipe Face Mesh eye-corner landmarks.
    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]

    def __init__(self):
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_eye.xml"
        )

        self.eye_detector = cv2.CascadeClassifier(
            cascade_path
        )

        if self.eye_detector.empty():
            raise RuntimeError(
                "Could not load OpenCV Haar eye detector."
            )

        self.previous_eye_box = None
        self.smoothing = 0.70

    def _get_landmark_eye_box(self, frame, landmarks):
        """Estimate the eye region from MediaPipe eye corners."""

        if landmarks is None:
            return None

        image_h, image_w = frame.shape[:2]
        indices = self.LEFT_EYE + self.RIGHT_EYE

        points = []

        for index in indices:
            landmark = landmarks.landmark[index]
            points.append((
                int(landmark.x * image_w),
                int(landmark.y * image_h)
            ))

        points = np.asarray(points)

        x1 = int(points[:, 0].min())
        y1 = int(points[:, 1].min())
        x2 = int(points[:, 0].max())
        y2 = int(points[:, 1].max())

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None

        # Include a small vertical margin around the eye corners.
        margin_y = max(5, int(height * 0.8))
        y1 = max(0, y1 - margin_y)
        y2 = min(image_h, y2 + margin_y)

        return x1, y1, x2 - x1, y2 - y1

    def detect_eye_region(self, frame, landmarks=None):
        """Detect eyes and reject Haar boxes far from the landmarks."""

        if landmarks is None:
            self.previous_eye_box = None
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self.eye_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(20, 10)
        )

        landmark_box = self._get_landmark_eye_box(
            frame, landmarks
        )

        if landmark_box is None or len(detections) < 2:
            self.previous_eye_box = None
            return None

        lx, ly, lw, lh = landmark_box

        candidates = []

        for x, y, w, h in detections:
            cx = x + w / 2
            cy = y + h / 2

            # Haar detection center must lie near the landmark eye region.
            if not (
                lx - 0.15 * lw <= cx <= lx + 1.15 * lw
                and ly - 0.50 * lh <= cy <= ly + 1.50 * lh
            ):
                continue

            candidates.append(
                (int(x), int(y), int(w), int(h))
            )

        if len(candidates) < 2:
            self.previous_eye_box = None
            return None

        valid_pairs = []

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i]
                b = candidates[j]

                if a[0] > b[0]:
                    a, b = b, a

                ax, ay, aw, ah = a
                bx, by, bw, bh = b

                center_a_y = ay + ah / 2
                center_b_y = by + bh / 2

                size_ratio = max(aw, bw) / max(
                    1, min(aw, bw)
                )

                if size_ratio > 1.8:
                    continue

                if abs(center_a_y - center_b_y) > max(
                    12, 0.6 * max(ah, bh)
                ):
                    continue

                center_distance = (
                    bx + bw / 2 - (ax + aw / 2)
                )

                average_width = (aw + bw) / 2

                if not (
                    0.8 * average_width
                    <= center_distance
                    <= 4.0 * average_width
                ):
                    continue

                x1 = min(ax, bx)
                y1 = min(ay, by)
                x2 = max(ax + aw, bx + bw)
                y2 = max(ay + ah, by + bh)

                width = x2 - x1
                height = y2 - y1

                if height <= 0:
                    continue

                aspect_ratio = width / height

                if not 1.5 <= aspect_ratio <= 6.0:
                    continue

                # Prefer a box whose center is close to the
                # MediaPipe eye-region center.
                haar_cx = (x1 + x2) / 2
                haar_cy = (y1 + y2) / 2
                landmark_cx = lx + lw / 2
                landmark_cy = ly + lh / 2

                distance = np.sqrt(
                    ((haar_cx - landmark_cx) / max(lw, 1)) ** 2
                    + ((haar_cy - landmark_cy) / max(lh, 1)) ** 2
                )

                score = distance + 0.2 * abs(
                    np.log(size_ratio)
                )

                valid_pairs.append(
                    (score, (x1, y1, width, height))
                )

        if not valid_pairs:
            self.previous_eye_box = None
            return None

        valid_pairs.sort(key=lambda item: item[0])
        current = np.asarray(
            valid_pairs[0][1], dtype=float
        )

        # Smooth the coordinates across successive frames.
        if self.previous_eye_box is not None:
            previous = np.asarray(
                self.previous_eye_box, dtype=float
            )

            current = (
                self.smoothing * previous
                + (1.0 - self.smoothing) * current
            )

        eye_box = tuple(
            int(round(value)) for value in current
        )

        self.previous_eye_box = eye_box
        return eye_box

    def get_forehead_roi(self, frame, landmarks=None):
        """Calculate the forehead ROI using the paper's formula."""

        eye_box = self.detect_eye_region(frame, landmarks)

        if eye_box is None:
            return None

        x1, y1, w1, h1 = eye_box
        image_h, image_w = frame.shape[:2]

        # Original formula from the reference paper.
        x2 = int(x1 + 0.125 * w1)
        y2 = int(y1 - 1.5 * h1)
        w2 = int(0.75 * w1)
        h2 = int(h1)

        x_start = max(0, x2)
        y_start = max(0, y2)
        x_end = min(image_w, x2 + w2)
        y_end = min(image_h, y2 + h2)

        if x_end <= x_start or y_end <= y_start:
            return None

        roi = frame[y_start:y_end, x_start:x_end]

        if roi.size == 0:
            return None

        forehead_box = (
            x_start,
            y_start,
            x_end - x_start,
            y_end - y_start
        )

        return roi, forehead_box, eye_box


def get_landmark_forehead_box(frame, face_landmarks):
    """Return the existing MediaPipe forehead reference box."""

    image_h, image_w = frame.shape[:2]

    indices = [10, 338, 297, 332, 284]
    points = []

    for index in indices:
        landmark = face_landmarks.landmark[index]
        points.append((
            int(landmark.x * image_w),
            int(landmark.y * image_h)
        ))

    points = np.asarray(points)

    x = int(points[:, 0].min())
    y = int(points[:, 1].min())
    x2 = int(points[:, 0].max())
    y2 = int(points[:, 1].max())

    width = x2 - x
    height = y2 - y

    x += int(width * 0.20)
    y += int(height * 0.15)

    width = int(width * 0.60)
    height = int(height * 0.60)

    x = max(0, x)
    y = max(0, y)

    width = min(width, image_w - x)
    height = min(height, image_h - y)

    if width <= 0 or height <= 0:
        return None

    return x, y, width, height


def draw_eye_and_forehead_roi(
    frame,
    eye_box,
    forehead_box,
    reference_box=None
):
    """Draw the detected eye and forehead regions."""

    if eye_box is not None:
        x, y, w, h = eye_box

        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (255, 0, 0), 2
        )

        cv2.putText(
            frame, "EYE REGION",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 1
        )

    if forehead_box is not None:
        x, y, w, h = forehead_box

        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

        cv2.putText(
            frame, "PAPER ROI",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1
        )

    if reference_box is not None:
        x, y, w, h = reference_box

        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 255), 2
        )

        cv2.putText(
            frame, "LANDMARK REFERENCE",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 1
        )

    return frame
