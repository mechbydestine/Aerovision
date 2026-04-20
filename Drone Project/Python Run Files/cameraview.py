from djitellopy import Tello
import cv2

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

tello.streamon()
frame_read = tello.get_frame_read()

while True:
    frame = frame_read.frame
    cv2.imshow("Tello Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break