
import cv2
import numpy as np


class ForeheadTracker:
    """
    Experimental forehead tracker with position and
    dimension stability checks.

    This is a heuristic tracker, not a validated
    heart-rate signal-quality model.
    """

    FOREHEAD_POINTS = [
        10, 109, 67, 103, 54, 21,
        338, 297, 332, 284, 251
    ]

    def __init__(
        self,
        min_width=60,
        min_height=15,
        max_dimension_change=0.35,
        max_position_change=0.35,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_dimension_change = max_dimension_change
        self.max_position_change = max_position_change

        # Store the last accepted ROI
        self.previous_box = None

    def reset(self):
        """Reset tracking history."""
        self.previous_box = None

    def extract(self, frame, face_landmarks):
        """
        Return:
            valid, roi, box, width, height, reason

        Box coordinates use (x1, y1, x2, y2).
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
        landmarks = face_landmarks.landmark

        if len(landmarks) <= max(self.FOREHEAD_POINTS):
            result["reason"] = "Insufficient landmarks"
            self.reset()
            return result

        selected = []

        for index in self.FOREHEAD_POINTS:
            lm = landmarks[index]

            if not np.isfinite(lm.x) or not np.isfinite(lm.y):
                result["reason"] = "Invalid landmarks"
                self.reset()
                return result

            if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1):
                result["reason"] = "Landmarks outside frame"
                self.reset()
                return result

            selected.append([
                lm.x * frame_width,
                lm.y * frame_height,
            ])

        points = np.asarray(selected, dtype=np.float32)

        x_min = float(np.min(points[:, 0]))
        x_max = float(np.max(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))

        region_width = x_max - x_min
        region_height = y_max - y_min

        if region_width <= 0 or region_height <= 0:
            result["reason"] = "Invalid region geometry"
            self.reset()
            return result

        # Estimate a central forehead region
        x1 = int(x_min + 0.20 * region_width)
        x2 = int(x_max - 0.20 * region_width)
        y1 = int(y_min)
        y2 = int(y_min + 0.65 * region_height)

        x1 = max(0, min(x1, frame_width - 1))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height - 1))
        y2 = max(0, min(y2, frame_height))

        width = x2 - x1
        height = y2 - y1
        current_box = (x1, y1, x2, y2)

        result["box"] = current_box
        result["width"] = width
        result["height"] = height

        if width < self.min_width or height < self.min_height:
            result["reason"] = "Forehead ROI too small"
            return result

        # Compare with the last accepted ROI
        if self.previous_box is not None:
            px1, py1, px2, py2 = self.previous_box

            previous_width = px2 - px1
            previous_height = py2 - py1

            # Check changes in ROI dimensions
            width_change = (
                abs(width - previous_width)
                / max(previous_width, 1)
            )

            height_change = (
                abs(height - previous_height)
                / max(previous_height, 1)
            )

            if (
                width_change > self.max_dimension_change
                or height_change > self.max_dimension_change
            ):
                result["reason"] = "ROI size changed suddenly"
                return result

            # Compare the ROI center positions
            old_cx = (px1 + px2) / 2.0
            old_cy = (py1 + py2) / 2.0

            new_cx = (x1 + x2) / 2.0
            new_cy = (y1 + y2) / 2.0

            shift_x = abs(new_cx - old_cx) / max(previous_width, 1)
            shift_y = abs(new_cy - old_cy) / max(previous_height, 1)

            if (
                shift_x > self.max_position_change
                or shift_y > self.max_position_change
            ):
                result["reason"] = "ROI position changed suddenly"
                return result

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            result["reason"] = "Empty forehead ROI"
            return result

        # Update history only when the ROI passes all checks
        self.previous_box = current_box

        result["valid"] = True
        result["roi"] = roi
        result["reason"] = "ROI accepted"

        return result

