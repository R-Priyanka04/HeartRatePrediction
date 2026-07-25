import cv2
import numpy as np
from collections import deque


# Store green-channel measurements
green_signal = deque(maxlen=300)


def extract_forehead(frame, face_landmarks):
    """
    Extract a forehead region using MediaPipe face landmarks.

    Returns:
        roi    -> cropped forehead image
        coords -> (x1, y1, x2, y2)
    """

    if frame is None or face_landmarks is None:
        return None, None

    height, width, _ = frame.shape

    # Forehead-related MediaPipe landmarks
    landmark_ids = [10, 338, 297, 332, 284]

    points = []

    for landmark_id in landmark_ids:

        landmark = face_landmarks.landmark[landmark_id]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append([x, y])

    points = np.array(points, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(points)

    # Expand the small landmark region into a useful forehead ROI
    center_x = x + w // 2
    center_y = y + h // 2

    roi_width = max(int(w * 0.75), 50)
    roi_height = max(int(h * 2.0), 35)

    x1 = center_x - roi_width // 2
    x2 = center_x + roi_width // 2

    y1 = center_y - roi_height // 2
    y2 = center_y + roi_height // 2

    # Keep ROI inside frame
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None, None

    return roi, (x1, y1, x2, y2)


def get_green_value(roi):
    """
    Calculate average green-channel intensity.
    """

    if roi is None or roi.size == 0:
        return None

    green_channel = roi[:, :, 1]

    value = float(np.mean(green_channel))

    return value


def update_signal(roi):
    """
    Extract and store the current green-channel value.
    """

    value = get_green_value(roi)

    if value is None:
        return None

    green_signal.append(value)

    return value


def reset_signal():
    """
    Clear all collected pulse samples.
    """

    green_signal.clear()