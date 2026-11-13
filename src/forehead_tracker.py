
import cv2
import numpy as np


class ForeheadTracker:
    """
    Experimental forehead ROI tracker using MediaPipe Face Mesh.

    Returns a forehead region, its dimensions, and basic
    tracking-quality information.
    """

    # Face Mesh landmark indices used to estimate the upper
    # face boundary and approximate forehead region.
    FOREHEAD_POINTS = [
        10, 109, 67, 103, 54, 21,
        338, 297, 332, 284, 251
    ]

    def __init__(
        self,
        min_width=60,
        min_height=15,
        max_dimension_change=0.20,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_dimension_change = max_dimension_change

        self.previous_box = None

    def reset(self):
        """Reset tracking history."""
        self.previous_box = None

    def extract(self, frame, face_landmarks):
        """
        Extract the forehead ROI.

        Returns a dictionary containing:
          valid: whether the ROI passed basic checks
          roi: cropped BGR image or None
          box: (x1, y1, x2, y2) or None
          width, height: ROI dimensions
          reason: status message
        """

        result = {
            "valid": False,
            "roi": None,
            "box": None,
            "width": 0,
            "height": 0,
            "reason": "No face detected",
        }

        if frame is None or face_landmarks is None:
            self.reset()
            return result

        frame_height, frame_width = frame.shape[:2]

        points = face_landmarks.landmark

        if len(points) <= max(self.FOREHEAD_POINTS):
            result["reason"] = "Insufficient face landmarks"
            self.reset()
            return result

        # Reject landmarks that are outside the frame or
        # clearly invalid.
        selected = []

        for index in self.FOREHEAD_POINTS:
            landmark = points[index]

            if not np.isfinite(landmark.x) or not np.isfinite(landmark.y):
                result["reason"] = "Invalid landmarks"
                self.reset()
                return result

            if not (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1):
                result["reason"] = "Landmarks outside frame"
                self.reset()
                return result

            selected.append([
                landmark.x * frame_width,
                landmark.y * frame_height,
            ])

        points_xy = np.array(selected, dtype=np.float32)

        # Bound the upper-face landmark region.
        x_min = float(np.min(points_xy[:, 0]))
        x_max = float(np.max(points_xy[:, 0]))
        y_min = float(np.min(points_xy[:, 1]))
        y_max = float(np.max(points_xy[:, 1]))

        face_region_width = x_max - x_min
        face_region_height = y_max - y_min

        if face_region_width <= 0 or face_region_height <= 0:
            result["reason"] = "Invalid region geometry"
            self.reset()
            return result

        # Select the upper portion of the landmark bounding box
        # as an approximate forehead region.
        x1 = int(x_min + 0.20 * face_region_width)
        x2 = int(x_max - 0.20 * face_region_width)
        y1 = int(y_min)
        y2 = int(y_min + 0.65 * face_region_height)

        # Keep the crop within the image.
        x1 = max(0, min(x1, frame_width - 1))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height - 1))
        y2 = max(0, min(y2, frame_height))

        width = x2 - x1
        height = y2 - y1

        result["box"] = (x1, y1, x2, y2)
        result["width"] = width
        result["height"] = height

        if width < self.min_width or height < self.min_height:
            result["reason"] = "Forehead ROI too small"
            self.reset()
            return result

        # Reject a sudden change in dimensions. This is a
        # basic heuristic, not a validated quality threshold.
        if self.previous_box is not None:
            old_x1, old_y1, old_x2, old_y2 = self.previous_box

            old_width = old_x2 - old_x1
            old_height = old_y2 - old_y1

            width_change = abs(width - old_width) / max(old_width, 1)
            height_change = abs(height - old_height) / max(old_height, 1)

            if (
                width_change > self.max_dimension_change
                or height_change > self.max_dimension_change
            ):
                result["reason"] = "Sudden ROI size change"
                self.previous_box = (x1, y1, x2, y2)
                return result

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            result["reason"] = "Empty forehead ROI"
            self.reset()
            return result

        self.previous_box = (x1, y1, x2, y2)

        result["valid"] = True
        result["roi"] = roi
        result["reason"] = "ROI accepted"

        return result

