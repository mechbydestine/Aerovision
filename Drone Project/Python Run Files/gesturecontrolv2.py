"""
Gesture-to-drone control using MediaPipe Tasks API and djitellopy.
Requires hand_landmarker.task in the same folder.
ONE finger = takeoff
FIST = land
"""

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


def draw_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera failed to open")
    raise SystemExit

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

is_flying = False
last_action_time = 0
cooldown = 2.5

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        gesture = "NO HAND"

        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks)
            landmarks = result.hand_landmarks[0]

            index_up = finger_up(landmarks, 8, 6)
            middle_up = finger_up(landmarks, 12, 10)
            ring_up = finger_up(landmarks, 16, 14)
            pinky_up = finger_up(landmarks, 20, 18)
            count = sum([index_up, middle_up, ring_up, pinky_up])

            if index_up and not middle_up and not ring_up and not pinky_up:
                gesture = "ONE"
            elif count == 0:
                gesture = "FIST"
            else:
                gesture = "OTHER"

        now = time.time()
        if now - last_action_time > cooldown:
            if gesture == "ONE" and not is_flying:
                print("TAKEOFF")
                tello.takeoff()
                is_flying = True
                last_action_time = now
            elif gesture == "FIST" and is_flying:
                print("LAND")
                tello.land()
                is_flying = False
                last_action_time = now

        cv2.putText(frame, f"Gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Flying: {is_flying}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "ONE finger = takeoff | FIST = land", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Gesture Drone Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

if is_flying:
    tello.land()
