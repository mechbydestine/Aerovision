import cv2
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from djitellopy import Tello


# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "hand_landmarker.task"
MAX_NUM_HANDS = 1

MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3

MIRROR_VIEW = False

GESTURE_WINDOW_SECONDS = 0.7
COMMAND_COOLDOWN = 1.5

latest_result = None
gesture_votes = []
last_triggered_gesture = None
last_trigger_time = 0.0


# =========================================================
# CALLBACK
# =========================================================
def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


# =========================================================
# GESTURE HELPERS
# =========================================================
def finger_is_extended(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def thumb_is_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]

    thumb_extended = thumb_tip.y < thumb_ip.y < thumb_mcp.y
    other_fingers_closed = (
        landmarks[8].y > landmarks[6].y and
        landmarks[12].y > landmarks[10].y and
        landmarks[16].y > landmarks[14].y and
        landmarks[20].y > landmarks[18].y
    )

    return thumb_extended and other_fingers_closed and thumb_tip.y < index_mcp.y


def hand_is_open(landmarks):
    return (
        finger_is_extended(landmarks, 8, 6) and
        finger_is_extended(landmarks, 12, 10) and
        finger_is_extended(landmarks, 16, 14) and
        finger_is_extended(landmarks, 20, 18)
    )


def hand_is_fist(landmarks):
    return (
        landmarks[8].y > landmarks[6].y and
        landmarks[12].y > landmarks[10].y and
        landmarks[16].y > landmarks[14].y and
        landmarks[20].y > landmarks[18].y
    )


def hand_is_index_up(landmarks):
    return (
        finger_is_extended(landmarks, 8, 6) and
        landmarks[12].y > landmarks[10].y and
        landmarks[16].y > landmarks[14].y and
        landmarks[20].y > landmarks[18].y
    )


def hand_is_peace(landmarks):
    return (
        finger_is_extended(landmarks, 8, 6) and
        finger_is_extended(landmarks, 12, 10) and
        landmarks[16].y > landmarks[14].y and
        landmarks[20].y > landmarks[18].y
    )


def recognize_gesture(landmarks):
    if hand_is_fist(landmarks):
        return "FIST"
    if hand_is_open(landmarks):
        return "OPEN_PALM"
    if hand_is_peace(landmarks):
        return "PEACE"
    if hand_is_index_up(landmarks):
        return "INDEX_UP"
    if thumb_is_up(landmarks):
        return "THUMB_UP"
    return "UNKNOWN"


# =========================================================
# TELLO COMMANDS
# =========================================================
def execute_command(tello, gesture_name):
    if gesture_name == "OPEN_PALM":
        print("[COMMAND] TAKEOFF")
        tello.takeoff()

    elif gesture_name == "FIST":
        print("[COMMAND] LAND")
        tello.land()

    elif gesture_name == "INDEX_UP":
        print("[COMMAND] FORWARD")
        tello.move_forward(20)

    elif gesture_name == "THUMB_UP":
        print("[COMMAND] UP")
        tello.move_up(20)

    elif gesture_name == "PEACE":
        print("[COMMAND] PHOTO")
        frame = tello.get_frame_read().frame
        filename = f"tello_photo_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")


def maybe_trigger_stable_command(tello, gesture_name):
    global gesture_votes, last_triggered_gesture, last_trigger_time

    now = time.time()
    gesture_votes = [(g, t) for (g, t) in gesture_votes if now - t <= GESTURE_WINDOW_SECONDS]

    if gesture_name not in ("UNKNOWN", "NO_HAND"):
        gesture_votes.append((gesture_name, now))

    if not gesture_votes:
        return

    counts = {}
    for g, _ in gesture_votes:
        counts[g] = counts.get(g, 0) + 1

    best_gesture = max(counts, key=counts.get)
    best_count = counts[best_gesture]

    if best_count >= 4:
        if best_gesture != last_triggered_gesture or (now - last_trigger_time) > COMMAND_COOLDOWN:
            execute_command(tello, best_gesture)
            last_triggered_gesture = best_gesture
            last_trigger_time = now


# =========================================================
# DRAWING
# =========================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


def draw_hand_manual(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = []

    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    for start_idx, end_idx in HAND_CONNECTIONS:
        x1, y1 = points[start_idx]
        x2, y2 = points[end_idx]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)


def draw_landmarks_and_label(frame, result):
    annotated = frame.copy()

    if result is None:
        cv2.putText(
            annotated,
            "No result yet",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        return annotated, "NO_HAND"

    if not result.hand_landmarks:
        cv2.putText(
            annotated,
            "Hand not detected",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        return annotated, "NO_HAND"

    for hand_landmarks in result.hand_landmarks:
        draw_hand_manual(annotated, hand_landmarks)

        gesture_name = recognize_gesture(hand_landmarks)

        h, w, _ = annotated.shape
        wrist = hand_landmarks[0]
        x = int(wrist.x * w)
        y = int(wrist.y * h) - 20

        cv2.putText(
            annotated,
            gesture_name,
            (x, max(30, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        return annotated, gesture_name

    return annotated, "NO_HAND"


# =========================================================
# MAIN
# =========================================================
def main():
    global latest_result

    tello = Tello()

    print("Connecting to Tello...")
    tello.connect()
    print("Battery:", tello.get_battery(), "%")

    tello.streamoff()
    tello.streamon()
    frame_read = tello.get_frame_read()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        result_callback=result_callback
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        print("Gesture system started.")
        print("Q = quit")
        print("OPEN_PALM -> TAKEOFF")
        print("FIST      -> LAND")
        print("INDEX_UP  -> FORWARD")
        print("THUMB_UP  -> UP")
        print("PEACE     -> PHOTO")

        while True:
            frame = frame_read.frame

            if frame is None or frame.size == 0:
                continue

            # Tello frame for display should stay BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (960, 720))
            display_frame = frame.copy()

            if MIRROR_VIEW:
                display_frame = cv2.flip(display_frame, 1)

            # Convert ONLY for MediaPipe input
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            annotated_frame, gesture_name = draw_landmarks_and_label(display_frame, latest_result)

            if gesture_name != "NO_HAND":
                maybe_trigger_stable_command(tello, gesture_name)

            # Debug text
            cv2.putText(
                annotated_frame,
                f"Battery: {tello.get_battery()}%",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated_frame,
                f"Gesture: {gesture_name}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated_frame,
                "Q = quit",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Tello Gesture Control", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    tello.streamoff()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()