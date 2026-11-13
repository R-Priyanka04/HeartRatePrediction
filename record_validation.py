
import cv2
import mediapipe as mp
import csv
import time
import numpy as np

OUTPUT_FILE = "validation_rgb.csv"
DURATION = 60

mp_face_mesh = mp.solutions.face_mesh
FOREHEAD_LANDMARKS = [10, 109, 67, 103, 54, 21, 338, 297, 332, 284, 251]


def main():
    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    rows = []
    start = time.time()

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as detector:

            print("Recording for 60 seconds.")
            print("Sit still, look at the camera, and keep lighting steady.")
            print("Press Q to stop early.")

            while time.time() - start < DURATION:
                ok, frame = camera.read()
                if not ok:
                    continue

                elapsed = time.time() - start
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)

                display = frame.copy()

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    height, width = frame.shape[:2]

                    points = np.array([
                        (int(landmarks[i].x * width),
                         int(landmarks[i].y * height))
                        for i in FOREHEAD_LANDMARKS
                    ])

                    x, y, w, h = cv2.boundingRect(points)

                    x1 = max(0, x + int(w * 0.20))
                    x2 = min(width, x + int(w * 0.80))
                    y1 = max(0, y)
                    y2 = min(height, y + int(h * 0.65))

                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]

                        if roi.size:
                            b, g, r = cv2.mean(roi)[:3]
                            rows.append([
                                elapsed, r, g, b,
                                x2 - x1, y2 - y1
                            ])

                            cv2.rectangle(
                                display, (x1, y1), (x2, y2),
                                (0, 255, 0), 2
                            )

                cv2.putText(
                    display,
                    f"Time: {elapsed:.1f}/{DURATION}s",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2
                )

                cv2.imshow("Heart Rate Validation Recording", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_seconds", "red", "green", "blue",
            "roi_width", "roi_height"
        ])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} samples to {OUTPUT_FILE}")

    if len(rows) >= 2:
        times = np.array([row[0] for row in rows])
        print(f"Recorded duration: {times[-1] - times[0]:.2f}s")
        print(
            "Approximate sampling rate: "
            f"{1 / np.median(np.diff(times)):.2f} Hz"
        )


if __name__ == "__main__":
    main()

