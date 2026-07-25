import cv2
import numpy as np
import time

from src.camera import Camera
from src.face_detection import FaceDetector
from src.signal_processing import estimate_bpm


# ============================================================
# SETTINGS
# ============================================================

MIN_SAMPLES = 180
MAX_SAMPLES = 300

WINDOW_NAME = "Heart Rate Prediction - BPM Lock Test"

# Calculate BPM roughly once per second
ESTIMATION_INTERVAL = 1.0

# Number of recent candidate BPM values used to decide stability
LOCK_WINDOW = 6

# Maximum standard deviation allowed before locking
LOCK_STD_THRESHOLD = 4.0

# Maximum difference between a new BPM and recent median
MAX_BPM_JUMP = 20.0

# Valid range used by this test program
MIN_VALID_BPM = 45.0
MAX_VALID_BPM = 180.0

# If face is lost for this long, signal collection is reset
FACE_LOST_RESET_SECONDS = 2.0


# ============================================================
# RESET MEASUREMENT
# ============================================================

def reset_measurement():

    signal_buffer = []
    timestamp_buffer = []
    candidate_bpms = []

    return (
        signal_buffer,
        timestamp_buffer,
        candidate_bpms
    )


# ============================================================
# ACTUAL FPS
# ============================================================

def calculate_actual_fps(timestamps):

    if timestamps is None:
        return None

    if len(timestamps) < 30:
        return None

    times = np.asarray(
        timestamps,
        dtype=np.float64
    )

    differences = np.diff(times)

    if len(differences) == 0:
        return None

    # Remove unrealistic frame intervals
    differences = differences[
        (differences > 0.001)
        &
        (differences < 0.2)
    ]

    if len(differences) < 10:
        return None

    median_interval = np.median(
        differences
    )

    if median_interval <= 0:
        return None

    fps = 1.0 / median_interval

    if fps < 5 or fps > 120:
        return None

    return float(fps)


# ============================================================
# FOREHEAD ROI
# ============================================================

def get_forehead_roi(
    frame,
    face_landmarks
):

    if frame is None:
        return None

    if face_landmarks is None:
        return None

    h, w, _ = frame.shape

    landmark_ids = [
        10,
        338,
        297,
        332,
        284
    ]

    points = []

    try:

        for idx in landmark_ids:

            landmark = (
                face_landmarks.landmark[idx]
            )

            px = int(
                landmark.x * w
            )

            py = int(
                landmark.y * h
            )

            points.append(
                (px, py)
            )

    except Exception:

        return None

    if len(points) == 0:
        return None

    points = np.asarray(
        points,
        dtype=np.int32
    )

    x, y, rw, rh = cv2.boundingRect(
        points
    )

    # --------------------------------------------------------
    # Create a slightly larger forehead ROI
    # --------------------------------------------------------

    center_x = x + rw // 2

    roi_width = max(
        int(rw * 0.80),
        30
    )

    roi_height = max(
        int(rh * 0.90),
        15
    )

    x = center_x - roi_width // 2

    rw = roi_width
    rh = roi_height

    # Move ROI slightly below hairline
    y = y + int(
        rh * 0.10
    )

    # Boundary checks
    x = max(
        0,
        x
    )

    y = max(
        0,
        y
    )

    x2 = min(
        w,
        x + rw
    )

    y2 = min(
        h,
        y + rh
    )

    if x2 <= x:
        return None

    if y2 <= y:
        return None

    roi = frame[
        y:y2,
        x:x2
    ]

    if roi.size == 0:
        return None

    return (
        roi,
        x,
        y,
        x2,
        y2
    )


# ============================================================
# GREEN CHANNEL
# ============================================================

def extract_green_value(roi):

    if roi is None:
        return None

    if roi.size == 0:
        return None

    green_channel = roi[:, :, 1]

    green_value = float(
        np.mean(
            green_channel
        )
    )

    if not np.isfinite(
        green_value
    ):
        return None

    return green_value


# ============================================================
# VALIDATE RAW BPM
# ============================================================

def is_valid_bpm(raw_bpm):

    if raw_bpm is None:
        return False

    try:

        bpm = float(
            raw_bpm
        )

    except Exception:

        return False

    if not np.isfinite(
        bpm
    ):
        return False

    if bpm < MIN_VALID_BPM:
        return False

    if bpm > MAX_VALID_BPM:
        return False

    return True


# ============================================================
# ADD BPM CANDIDATE
# ============================================================

def add_bpm_candidate(
    raw_bpm,
    candidate_bpms
):

    if not is_valid_bpm(
        raw_bpm
    ):
        return False

    raw_bpm = float(
        raw_bpm
    )

    # --------------------------------------------------------
    # Compare new value against recent median
    # --------------------------------------------------------

    if len(candidate_bpms) >= 3:

        recent = candidate_bpms[
            -LOCK_WINDOW:
        ]

        recent_median = float(
            np.median(
                recent
            )
        )

        difference = abs(
            raw_bpm
            -
            recent_median
        )

        # Reject sudden jumps
        if difference > MAX_BPM_JUMP:

            print(
                f"Rejected BPM jump: "
                f"{recent_median:.1f} -> "
                f"{raw_bpm:.1f}"
            )

            return False

    candidate_bpms.append(
        raw_bpm
    )

    # Keep a limited recent history
    if len(candidate_bpms) > 20:

        candidate_bpms.pop(0)

    return True


# ============================================================
# CHECK WHETHER BPM CAN BE LOCKED
# ============================================================

def check_bpm_lock(
    candidate_bpms
):

    if len(candidate_bpms) < LOCK_WINDOW:
        return None

    recent = np.asarray(
        candidate_bpms[
            -LOCK_WINDOW:
        ],
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(recent)
    ):
        return None

    median_bpm = float(
        np.median(
            recent
        )
    )

    deviation = float(
        np.std(
            recent
        )
    )

    bpm_range = float(
        np.ptp(
            recent
        )
    )

    print(
        "LOCK CHECK | "
        f"Values={np.round(recent, 1).tolist()} | "
        f"Median={median_bpm:.1f} | "
        f"STD={deviation:.2f} | "
        f"Range={bpm_range:.1f}"
    )

    # --------------------------------------------------------
    # Stability test
    # --------------------------------------------------------

    if deviation <= LOCK_STD_THRESHOLD:

        return round(
            median_bpm,
            1
        )

    return None


# ============================================================
# DRAW USER INTERFACE
# ============================================================

def draw_information(
    frame,
    sample_count,
    actual_fps,
    green_value,
    raw_bpm,
    candidate_count,
    bpm_locked,
    locked_bpm
):

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Samples: {sample_count}/{MIN_SAMPLES}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if actual_fps is None:

        fps_text = (
            "Actual FPS: calculating..."
        )

    else:

        fps_text = (
            f"Actual FPS: {actual_fps:.1f}"
        )

    cv2.putText(
        frame,
        fps_text,
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 200, 0),
        2
    )

    # --------------------------------------------------------
    # Green signal
    # --------------------------------------------------------

    if green_value is None:

        green_text = (
            "Green: --"
        )

    else:

        green_text = (
            f"Green: {green_value:.2f}"
        )

    cv2.putText(
        frame,
        green_text,
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Raw BPM
    # --------------------------------------------------------

    if raw_bpm is None:

        raw_text = (
            "Raw BPM: --"
        )

    else:

        raw_text = (
            f"Raw BPM: {raw_bpm:.1f}"
        )

    cv2.putText(
        frame,
        raw_text,
        (25, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (220, 220, 220),
        2
    )

    # --------------------------------------------------------
    # Main BPM display
    # --------------------------------------------------------

    if bpm_locked:

        bpm_text = (
            f"BPM: {locked_bpm:.1f}"
        )

        bpm_color = (
            0,
            255,
            0
        )

    elif sample_count < MIN_SAMPLES:

        bpm_text = (
            f"BPM: Collecting "
            f"{sample_count}/{MIN_SAMPLES}"
        )

        bpm_color = (
            0,
            255,
            255
        )

    elif candidate_count == 0:

        bpm_text = (
            "BPM: Analyzing signal..."
        )

        bpm_color = (
            0,
            255,
            255
        )

    else:

        progress = min(
            candidate_count,
            LOCK_WINDOW
        )

        bpm_text = (
            f"BPM: Stabilizing "
            f"{progress}/{LOCK_WINDOW}"
        )

        bpm_color = (
            0,
            255,
            255
        )

    cv2.putText(
        frame,
        bpm_text,
        (25, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        bpm_color,
        2
    )

    # --------------------------------------------------------
    # Locked message
    # --------------------------------------------------------

    if bpm_locked:

        cv2.putText(
            frame,
            "MEASUREMENT COMPLETE",
            (25, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "BPM LOCKED - Press R to measure again",
            (25, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1
        )


# ============================================================
# SAFE CAMERA CLEANUP
# ============================================================

def cleanup_camera(
    camera
):

    try:

        if hasattr(
            camera,
            "close"
        ):

            camera.close()

            return

        if hasattr(
            camera,
            "release"
        ):

            camera.release()

            return

        if hasattr(
            camera,
            "cap"
        ):

            if camera.cap is not None:

                camera.cap.release()

    except Exception as e:

        print(
            "Camera cleanup warning:",
            e
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print(
        "=" * 65
    )

    print(
        "HEART RATE PREDICTION - "
        "BPM LOCK TEST"
    )

    print(
        "=" * 65
    )

    print()

    print(
        "Instructions:"
    )

    print(
        "- Look directly at the camera"
    )

    print(
        "- Keep your head still"
    )

    print(
        "- Keep your hands away from your face"
    )

    print(
        "- Use steady lighting"
    )

    print(
        "- Avoid strong backlighting"
    )

    print(
        "- Wait while BPM stabilizes"
    )

    print()

    print(
        "Controls:"
    )

    print(
        "Q = Quit"
    )

    print(
        "R = Reset measurement"
    )

    print()

    # ========================================================
    # CREATE CAMERA / DETECTOR
    # ========================================================

    camera = Camera()

    detector = FaceDetector()

    # ========================================================
    # BUFFERS
    # ========================================================

    (
        signal_buffer,
        timestamp_buffer,
        candidate_bpms
    ) = reset_measurement()

    # ========================================================
    # STATE VARIABLES
    # ========================================================

    raw_bpm = None

    locked_bpm = None

    bpm_locked = False

    actual_fps = None

    last_estimation_time = 0

    last_face_time = time.perf_counter()

    face_was_lost = False

    # ========================================================
    # OPEN CAMERA
    # ========================================================

    print(
        "Opening MacBook camera..."
    )

    try:

        opened = camera.open()

    except Exception as e:

        print(
            "Camera open error:",
            repr(e)
        )

        try:
            detector.close()
        except Exception:
            pass

        return

    if opened is False:

        print(
            "Could not open camera."
        )

        try:
            detector.close()
        except Exception:
            pass

        return

    print(
        "Camera opened successfully."
    )

    print()

    print(
        "Starting measurement..."
    )

    # ========================================================
    # CAMERA LOOP
    # ========================================================

    try:

        while True:

            success, frame = (
                camera.read()
            )

            if not success:

                key = (
                    cv2.waitKey(1)
                    &
                    0xFF
                )

                if key == ord("q"):
                    break

                continue

            current_time = (
                time.perf_counter()
            )

            frame = cv2.flip(
                frame,
                1
            )

            # =================================================
            # FACE DETECTION
            # =================================================

            face_landmarks = (
                detector.detect(
                    frame
                )
            )

            green_value = None

            forehead_found = False

            # =================================================
            # FOREHEAD
            # =================================================

            if face_landmarks is not None:

                result = (
                    get_forehead_roi(
                        frame,
                        face_landmarks
                    )
                )

                if result is not None:

                    (
                        roi,
                        x,
                        y,
                        x2,
                        y2
                    ) = result

                    forehead_found = True

                    last_face_time = (
                        current_time
                    )

                    face_was_lost = False

                    # -----------------------------------------
                    # Draw ROI
                    # -----------------------------------------

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "FOREHEAD ROI",
                        (
                            x,
                            max(
                                25,
                                y - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    # -----------------------------------------
                    # Only continue collecting if not locked
                    # -----------------------------------------

                    if not bpm_locked:

                        green_value = (
                            extract_green_value(
                                roi
                            )
                        )

                        if green_value is not None:

                            signal_buffer.append(
                                green_value
                            )

                            timestamp_buffer.append(
                                current_time
                            )

                            # ---------------------------------
                            # Maintain sliding window
                            # ---------------------------------

                            if (
                                len(signal_buffer)
                                >
                                MAX_SAMPLES
                            ):

                                signal_buffer.pop(
                                    0
                                )

                                timestamp_buffer.pop(
                                    0
                                )

                    else:

                        # Still display current green value
                        green_value = (
                            extract_green_value(
                                roi
                            )
                        )

            # =================================================
            # FACE LOST RESET
            # =================================================

            if (
                not forehead_found
                and
                not bpm_locked
            ):

                lost_duration = (
                    current_time
                    -
                    last_face_time
                )

                if (
                    lost_duration
                    >
                    FACE_LOST_RESET_SECONDS
                    and
                    not face_was_lost
                ):

                    print(
                        "Face lost too long. "
                        "Resetting measurement."
                    )

                    (
                        signal_buffer,
                        timestamp_buffer,
                        candidate_bpms
                    ) = reset_measurement()

                    raw_bpm = None

                    actual_fps = None

                    last_estimation_time = 0

                    face_was_lost = True

            # =================================================
            # ACTUAL FPS
            # =================================================

            if not bpm_locked:

                actual_fps = (
                    calculate_actual_fps(
                        timestamp_buffer
                    )
                )

            # =================================================
            # BPM ESTIMATION
            # =================================================

            if (
                not bpm_locked
                and
                len(signal_buffer)
                >=
                MIN_SAMPLES
                and
                actual_fps is not None
            ):

                elapsed = (
                    current_time
                    -
                    last_estimation_time
                )

                if (
                    elapsed
                    >=
                    ESTIMATION_INTERVAL
                ):

                    last_estimation_time = (
                        current_time
                    )

                    # -----------------------------------------
                    # Calculate BPM
                    # -----------------------------------------

                    try:

                        estimated = (
                            estimate_bpm(
                                signal_buffer,
                                actual_fps
                            )
                        )

                    except Exception as e:

                        print(
                            "estimate_bpm ERROR:",
                            repr(e)
                        )

                        estimated = None

                    # -----------------------------------------
                    # Signal diagnostics
                    # -----------------------------------------

                    arr = np.asarray(
                        signal_buffer,
                        dtype=np.float64
                    )

                    if estimated is None:

                        raw_bpm = None

                        print(
                            "Estimator returned None | "
                            f"FPS={actual_fps:.2f} | "
                            f"Samples={len(arr)} | "
                            f"Mean={np.mean(arr):.2f} | "
                            f"STD={np.std(arr):.4f} | "
                            f"Range={np.ptp(arr):.4f}"
                        )

                    else:

                        raw_bpm = float(
                            estimated
                        )

                        print(
                            f"Raw BPM={raw_bpm:.1f} | "
                            f"FPS={actual_fps:.2f} | "
                            f"Samples={len(arr)} | "
                            f"STD={np.std(arr):.4f} | "
                            f"Range={np.ptp(arr):.4f}"
                        )

                        # =====================================
                        # ADD BPM CANDIDATE
                        # =====================================

                        accepted = (
                            add_bpm_candidate(
                                raw_bpm,
                                candidate_bpms
                            )
                        )

                        if accepted:

                            print(
                                "Accepted BPM | "
                                f"Candidates="
                                f"{len(candidate_bpms)}"
                            )

                            # =================================
                            # CHECK LOCK
                            # =================================

                            possible_lock = (
                                check_bpm_lock(
                                    candidate_bpms
                                )
                            )

                            if (
                                possible_lock
                                is not None
                            ):

                                locked_bpm = (
                                    possible_lock
                                )

                                bpm_locked = True

                                print()
                                print(
                                    "=" * 50
                                )

                                print(
                                    "MEASUREMENT COMPLETE"
                                )

                                print(
                                    f"FINAL LOCKED BPM: "
                                    f"{locked_bpm:.1f}"
                                )

                                print(
                                    "=" * 50
                                )

                                print()

            # =================================================
            # DRAW INFORMATION
            # =================================================

            draw_information(
                frame,
                len(signal_buffer),
                actual_fps,
                green_value,
                raw_bpm,
                len(candidate_bpms),
                bpm_locked,
                locked_bpm
            )

            # =================================================
            # FACE / SIGNAL STATUS
            # =================================================

            frame_height = (
                frame.shape[0]
            )

            if bpm_locked:

                status_text = (
                    "MEASUREMENT COMPLETE"
                )

                status_color = (
                    0,
                    255,
                    0
                )

            elif forehead_found:

                status_text = (
                    "SIGNAL COLLECTING"
                )

                status_color = (
                    0,
                    255,
                    0
                )

            else:

                status_text = (
                    "FACE NOT DETECTED"
                )

                status_color = (
                    0,
                    0,
                    255
                )

            cv2.putText(
                frame,
                status_text,
                (
                    25,
                    frame_height - 45
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                status_color,
                2
            )

            # =================================================
            # CONTROLS
            # =================================================

            cv2.putText(
                frame,
                "Q: Quit    R: New Measurement",
                (
                    25,
                    frame_height - 18
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 220),
                1
            )

            # =================================================
            # SHOW WINDOW
            # =================================================

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # =================================================
            # KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # -------------------------------------------------
            # QUIT
            # -------------------------------------------------

            if key == ord("q"):

                print()

                print(
                    "Stopping measurement..."
                )

                break

            # -------------------------------------------------
            # RESET / NEW MEASUREMENT
            # -------------------------------------------------

            elif key == ord("r"):

                print()

                print(
                    "=" * 50
                )

                print(
                    "NEW MEASUREMENT"
                )

                print(
                    "=" * 50
                )

                (
                    signal_buffer,
                    timestamp_buffer,
                    candidate_bpms
                ) = reset_measurement()

                raw_bpm = None

                locked_bpm = None

                bpm_locked = False

                actual_fps = None

                last_estimation_time = 0

                last_face_time = (
                    current_time
                )

                face_was_lost = False

    # ========================================================
    # CTRL+C
    # ========================================================

    except KeyboardInterrupt:

        print()

        print(
            "Measurement interrupted."
        )

    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        print()

        print(
            "Unexpected error:",
            repr(e)
        )

        import traceback

        traceback.print_exc()

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        cv2.destroyAllWindows()

        try:

            detector.close()

        except Exception:

            pass

        cleanup_camera(
            camera
        )

        print()

        print(
            "Camera closed."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()