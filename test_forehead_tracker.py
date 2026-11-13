
import cv2
import time

from src.camera import Camera
from src.face_detection import FaceDetector
from src.forehead_tracker_v2 import ForeheadTracker


def main():
    camera = Camera()
    face_detector = FaceDetector()
    tracker = ForeheadTracker()

    accepted_frames = 0
    rejected_frames = 0
    rejection_reasons = {}
    start_time = time.time()

    try:
        camera.open()

        print("Camera opened successfully.")
        print("Keep your face centered and well-lit.")
        print("Move your head slightly after staying still.")
        print("Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("Unable to read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            face_landmarks = face_detector.detect(frame)

            if face_landmarks is None:
                rejected_frames += 1
                reason = "Face not detected"
                rejection_reasons[reason] = (
                    rejection_reasons.get(reason, 0) + 1
                )

                cv2.putText(
                    display_frame,
                    reason,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

            else:
                result = tracker.extract(frame, face_landmarks)

                if result["valid"]:
                    accepted_frames += 1

                    x1, y1, x2, y2 = result["box"]
                    roi = result["roi"]

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        display_frame,
                        "ROI ACCEPTED",
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        display_frame,
                        (
                            f"Size: {result['width']} x "
                            f"{result['height']} px"
                        ),
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        display_frame,
                        f"Position: ({x1}, {y1}) to ({x2}, {y2})",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        1,
                    )

                    if roi is not None and roi.size > 0:
                        blue, green, red = cv2.mean(roi)[:3]

                        cv2.putText(
                            display_frame,
                            (
                                f"R: {red:.1f}  "
                                f"G: {green:.1f}  "
                                f"B: {blue:.1f}"
                            ),
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                else:
                    rejected_frames += 1

                    reason = result["reason"]
                    rejection_reasons[reason] = (
                        rejection_reasons.get(reason, 0) + 1
                    )

                    if result["box"] is not None:
                        x1, y1, x2, y2 = result["box"]

                        cv2.rectangle(
                            display_frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2,
                        )

                    cv2.putText(
                        display_frame,
                        f"REJECTED: {reason}"[:75],
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 0, 255),
                        2,
                    )

            elapsed = time.time() - start_time
            total = accepted_frames + rejected_frames

            acceptance_rate = (
                100.0 * accepted_frames / total
                if total else 0.0
            )

            cv2.putText(
                display_frame,
                f"Accepted: {accepted_frames}",
                (20, display_frame.shape[0] - 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display_frame,
                f"Rejected: {rejected_frames}",
                (20, display_frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Acceptance: {acceptance_rate:.1f}%",
                (20, display_frame.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                display_frame,
                f"Time: {elapsed:.1f}s",
                (20, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Forehead Tracker V2 Test", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as error:
        print(f"Test error: {error}")
        raise

    finally:
        try:
            camera.release()
        except Exception as error:
            print(f"Camera cleanup: {error}")

        try:
            face_detector.close()
        except Exception as error:
            print(f"Face detector cleanup: {error}")

        cv2.destroyAllWindows()

        total = accepted_frames + rejected_frames

        print("\n--- FOREHEAD TRACKER V2 RESULTS ---")
        print(f"Accepted frames: {accepted_frames}")
        print(f"Rejected frames: {rejected_frames}")

        if total:
            print(
                f"Acceptance rate: "
                f"{100.0 * accepted_frames / total:.2f}%"
            )

        print(f"Duration: {time.time() - start_time:.1f} seconds")

        if rejection_reasons:
            print("\nRejection reasons:")

            for reason, count in rejection_reasons.items():
                print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
