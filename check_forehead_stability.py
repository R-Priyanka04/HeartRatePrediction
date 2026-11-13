
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

# --------------------------------------------------
# Forehead RGB stability experiment
# This script does not modify the production pipeline.
# --------------------------------------------------

CAMERA_INDEX = 0
DURATION_SECONDS = 30
OUTPUT_CSV = "forehead_stability.csv"
OUTPUT_PLOT = "forehead_stability.png"

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Approximate forehead landmarks.
# These points are inside the upper forehead region.
FOREHEAD_LANDMARKS = [10, 109, 67, 103, 54, 21, 338, 297, 332, 284, 251]

def main():
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    timestamps = []
    red_values = []
    green_values = []
    blue_values = []
    roi_widths = []
    roi_heights = []

    start_time = time.time()

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            print("Camera started.")
            print("Look directly at the camera.")
            print("Keep your face still and lighting unchanged.")
            print("Recording for 30 seconds. Press Q to stop early.")

            while time.time() - start_time < DURATION_SECONDS:
                ok, frame = camera.read()

                if not ok:
                    print("WARNING: Could not read camera frame.")
                    continue

                elapsed = time.time() - start_time

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                results = face_mesh.process(rgb_frame)

                display = frame.copy()

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    height, width = frame.shape[:2]

                    points = []
                    for index in FOREHEAD_LANDMARKS:
                        landmark = landmarks[index]
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                    points = np.array(points, dtype=np.int32)

                    # Use a bounding rectangle around the forehead points.
                    x, y, w, h = cv2.boundingRect(points)

                    # Use the upper central portion of this rectangle
                    # to reduce the chance of including eyebrows/eyes.
                    x1 = max(0, x + int(w * 0.20))
                    x2 = min(width, x + int(w * 0.80))
                    y1 = max(0, y)
                    y2 = min(height, y + int(h * 0.65))

                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]

                        if roi.size > 0:
                            # OpenCV frames are BGR.
                            b, g, r = cv2.mean(roi)[:3]

                            timestamps.append(elapsed)
                            red_values.append(r)
                            green_values.append(g)
                            blue_values.append(b)
                            roi_widths.append(x2 - x1)
                            roi_heights.append(y2 - y1)

                            cv2.rectangle(
                                display,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2,
                            )

                            cv2.putText(
                                display,
                                f"R:{r:.1f} G:{g:.1f} B:{b:.1f}",
                                (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

                remaining = max(
                    0, int(DURATION_SECONDS - elapsed)
                )
                cv2.putText(
                    display,
                    f"Recording: {remaining}s",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow("Forehead Stability Test", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Recording stopped early.")
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(timestamps) < 30:
        print("ERROR: Too few valid forehead samples.")
        print("Check camera access and face detection.")
        return

    # Save the measurements.
    with open(OUTPUT_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_seconds",
            "red",
            "green",
            "blue",
            "roi_width",
            "roi_height",
        ])

        writer.writerows(zip(
            timestamps,
            red_values,
            green_values,
            blue_values,
            roi_widths,
            roi_heights,
        ))

    print(f"\nValid samples: {len(timestamps)}")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.2f}s")
    print(f"Saved data: {OUTPUT_CSV}")

    # Report abrupt changes and variability.
    channels = {
        "RED": np.array(red_values),
        "GREEN": np.array(green_values),
        "BLUE": np.array(blue_values),
    }

    for name, values in channels.items():
        differences = np.abs(np.diff(values))

        print(f"\n{name}")
        print(f"Mean: {np.mean(values):.2f}")
        print(f"Standard deviation: {np.std(values):.2f}")
        print(
            "95th percentile consecutive change: "
            f"{np.percentile(differences, 95):.2f}"
        )

    print("\nROI dimensions:")
    print(
        f"Width range: {min(roi_widths)}–{max(roi_widths)} pixels"
    )
    print(
        f"Height range: {min(roi_heights)}–{max(roi_heights)} pixels"
    )

    # Save a non-interactive plot.
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(timestamps, red_values)
    axes[0].set_ylabel("Red")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timestamps, green_values)
    axes[1].set_ylabel("Green")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(timestamps, blue_values)
    axes[2].set_ylabel("Blue")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(timestamps, roi_widths, label="ROI width")
    axes[3].plot(timestamps, roi_heights, label="ROI height")
    axes[3].set_ylabel("ROI size (pixels)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Forehead RGB and ROI Stability")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=150)
    plt.close(fig)

    print(f"Saved graph: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()

