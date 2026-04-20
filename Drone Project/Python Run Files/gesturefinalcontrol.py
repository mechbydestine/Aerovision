import cv2
import mediapipe as mp
import time
from djitellopy import Tello

MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def finger_up(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y


def thumb_up(landmarks):
    # thumb tip above thumb IP joint
    return landmarks[4].y < landmarks[3].y


def thumb_down(landmarks):
    return landmarks[4].y > landmarks[3].y


def draw_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def classify_gesture(landmarks):
    index_up = finger_up(landmarks, 8, 6)
    middle_up = finger_up(landmarks, 12, 10)
    ring_up = finger_up(landmarks, 16, 14)
    pinky_up = finger_up(landmarks, 20, 18)
    thumb_is_up = thumb_up(landmarks)
    thumb_is_down = thumb_down(landmarks)

    fingers_count = sum([index_up, middle_up, ring_up, pinky_up])

    # Takeoff: open hand (4 fingers up)
    if fingers_count == 4:
        return "TAKEOFF"

    # Land: fist
    if fingers_count == 0 and not thumb_is_up:
        return "LAND"

    # Forward: index only
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "FORWARD"

    # Backward: index + middle
    if index_up and middle_up and not ring_up and not pinky_up:
        return "BACKWARD"

    # Start/stop recording: index + middle + ring
    if index_up and middle_up and ring_up and not pinky_up:
        return "RECORD"

    # Photo: peace sign
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PHOTO"

    # Rotate left: pinky only
    if pinky_up and not index_up and not middle_up and not ring_up:
        return "ROTATE_LEFT"

    # Rotate right: thumb + pinky look
    if pinky_up and not index_up and not middle_up and not ring_up and thumb_is_up:
        return "ROTATE_RIGHT"

    # Up / down
    if thumb_is_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "UP"

    if thumb_is_down and not index_up and not middle_up and not ring_up and not pinky_up:
        return "DOWN"

    return "NONE"


def save_photo(frame):
    filename = f"tello_photo_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Photo saved: {filename}")


def start_recording(frame):
    h, w, _ = frame.shape
    filename = f"tello_video_{int(time.time())}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
    print(f"Recording started: {filename}")
    return writer, filename


def move_drone(tello, gesture, speed=30, move_time=0.35):
    if gesture == "FORWARD":
        print("Forward")
        tello.send_rc_control(0, speed, 0, 0)
    elif gesture == "BACKWARD":
        print("Backward")
        tello.send_rc_control(0, -speed, 0, 0)
    elif gesture == "UP":
        print("Up")
        tello.send_rc_control(0, 0, speed, 0)
    elif gesture == "DOWN":
        print("Down")
        tello.send_rc_control(0, 0, -speed, 0)
    elif gesture == "ROTATE_LEFT":
        print("Rotate left")
        tello.send_rc_control(0, 0, 0, -speed)
    elif gesture == "ROTATE_RIGHT":
        print("Rotate right")
        tello.send_rc_control(0, 0, 0, speed)
    else:
        return

    time.sleep(move_time)
    tello.send_rc_control(0, 0, 0, 0)  # hover again


# -----------------------------
# Connect Tello
# -----------------------------
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

tello.streamon()
frame_read = tello.get_frame_read()

# -----------------------------
# Open laptop webcam for gestures
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Laptop webcam failed to open.")
    raise SystemExit

# -----------------------------
# MediaPipe Hand Landmarker
# -----------------------------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

is_flying = False
recording = False
video_writer = None
last_action_time = 0
cooldown = 2.0

print("""
GESTURES:
Open hand = takeoff + hover
Fist = land
Index only = forward
Peace sign = photo
Index + middle = backward
Index + middle + ring = record on/off
Thumb up = up
Thumb down = down
Pinky only = rotate left
Thumb + pinky = rotate right
ESC = quit
""")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, gesture_frame = cap.read()
        if not ok:
            print("Failed to read webcam frame")
            break

        gesture_frame = cv2.flip(gesture_frame, 1)
        rgb = cv2.cvtColor(gesture_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        gesture = "NO HAND"
        if result.hand_landmarks:
            draw_landmarks(gesture_frame, result.hand_landmarks)
            landmarks = result.hand_landmarks[0]
            gesture = classify_gesture(landmarks)

        drone_frame = frame_read.frame.copy()

        # Record from drone camera if recording
        if recording and video_writer is not None:
            video_writer.write(drone_frame)

        now = time.time()
        if now - last_action_time > cooldown:
            if gesture == "TAKEOFF" and not is_flying:
                print("Takeoff")
                tello.takeoff()
                tello.send_rc_control(0, 0, 0, 0)  # hover
                is_flying = True
                last_action_time = now

            elif gesture == "LAND" and is_flying:
                print("Land")
                tello.land()
                is_flying = False
                last_action_time = now

            elif gesture == "PHOTO":
                save_photo(drone_frame)
                last_action_time = now

            elif gesture == "RECORD":
                if not recording:
                    video_writer, _ = start_recording(drone_frame)
                    recording = True
                else:
                    recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print("Recording stopped")
                last_action_time = now

            elif is_flying and gesture in {
                "FORWARD", "BACKWARD", "UP", "DOWN",
                "ROTATE_LEFT", "ROTATE_RIGHT"
            }:
                move_drone(tello, gesture)
                last_action_time = now

        cv2.putText(
            gesture_frame,
            f"Gesture: {gesture}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            gesture_frame,
            f"Flying: {is_flying}  Recording: {recording}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Laptop Gesture Camera", gesture_frame)
        cv2.imshow("Tello Camera", drone_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()
tello.streamoff()

if is_flying:
    tello.land()