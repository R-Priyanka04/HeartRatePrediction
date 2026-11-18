"""
Validation recorder using the SAME forehead ROI geometry as dashboard.py.

This is a validation-tool update only. It does not change the live estimator,
dashboard UI, database, or saved measurement history.
"""
import csv
import time
import cv2
import mediapipe as mp
import numpy as np

OUTPUT_FILE = "validation_rgb_live_roi.csv"
DURATION = 60
EYE_IDS = [33, 133, 159, 145, 362, 263, 386, 374]


def get_forehead_roi(frame, face_landmarks):
    """ROI geometry copied from the current dashboard.py implementation."""
    if frame is None or face_landmarks is None:
        return None

    height, width = frame.shape[:2]
    landmarks = face_landmarks.landmark

    try:
        eye_points = np.asarray([
            (int(landmarks[i].x * width), int(landmarks[i].y * height))
            for i in EYE_IDS
        ], dtype=np.int32)

        ex, ey, ew, eh = cv2.boundingRect(eye_points)
        eye_center_x = ex + ew // 2
        roi_width = max(int(ew * 1.15), 50)
        roi_height = max(int(eh * 2.0), 24)
        gap = max(12, int(eh * 0.6))

        x = max(0, eye_center_x - roi_width // 2)
        y = max(0, ey - gap - roi_height)
        x2 = min(width, x + roi_width)
        y2 = min(height, y + roi_height)

        if x2 <= x or y2 <= y or (y2 - y) < int(roi_height * 0.65):
            return None

        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            return None

        return roi, x, y, x2, y2, ex, ey, ew, eh

    except (IndexError, AttributeError, ValueError, TypeError):
        return None


def main():
    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    rows = []
    start = time.perf_counter()

    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as detector:
            print(f"Recording for up to {DURATION} seconds.")
            print("Use the same camera position, lighting and posture as the live app.")
            print("Keep still and keep the forehead ROI inside the green rectangle.")
            print("Press Q to stop early.")

            while time.perf_counter() - start < DURATION:
                ok, frame = camera.read()
                if not ok:
                    continue

                elapsed = time.perf_counter() - start
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb_frame)
                display = frame.copy()

                if result.multi_face_landmarks:
                    roi_result = get_forehead_roi(
                        frame, result.multi_face_landmarks[0]
                    )
                    if roi_result is not None:
                        roi, x1, y1, x2, y2, ex, ey, ew, eh = roi_result
                        means = np.mean(roi, axis=(0, 1))
                        blue, green, red = map(float, means[:3])
                        rows.append([
                            elapsed, red, green, blue,
                            x2 - x1, y2 - y1,
                            ew, eh, ex + ew / 2, ey + eh / 2,
                            x1, y1
                        ])
                        cv2.rectangle(
                            display, (x1, y1), (x2, y2), (0, 255, 0), 2
                        )

                cv2.putText(
                    display, f"Time: {elapsed:.1f}/{DURATION}s",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2
                )
                cv2.imshow("Heart Rate Validation - Live App ROI", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(rows) < 2:
        print("Not enough valid face/ROI samples were collected.")
        return

    times = np.asarray([r[0] for r in rows], dtype=float)
    intervals = np.diff(times)
    valid_intervals = intervals[(intervals > 0.001) & (intervals < 0.5)]
    fps = (1.0 / np.median(valid_intervals)) if len(valid_intervals) else float("nan")

    print(f"\nCollected {len(rows)} ROI samples.")
    print(f"Duration: {times[-1] - times[0]:.2f} seconds")
    print(f"Approximate sampling rate: {fps:.2f} Hz")

    reference = input(
        "\nEnter the pulse-oximeter BPM observed during this recording "
        "(or press Enter to leave it blank): "
    ).strip()

    reference_bpm = ""
    if reference:
        try:
            value = float(reference)
            if not np.isfinite(value) or not 30 <= value <= 220:
                raise ValueError
            reference_bpm = value
        except ValueError:
            print("Reference must be a number between 30 and 220. Saving it blank.")
            reference_bpm = ""

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_seconds", "red", "green", "blue",
            "roi_width", "roi_height",
            "eye_width", "eye_height", "eye_center_x", "eye_center_y",
            "roi_x", "roi_y", "reference_bpm"
        ])
        for row in rows:
            writer.writerow(row + [reference_bpm])

    print(f"\nSaved: {OUTPUT_FILE}")
    if reference_bpm != "":
        print(f"Reference BPM saved: {reference_bpm}")
        print("For stronger validation, repeat at least 5 times and note the oximeter value during each capture.")
    else:
        print("No reference BPM was entered; this recording can compare signal methods, not accuracy.")


if __name__ == "__main__":
    main()
