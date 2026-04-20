from djitellopy import Tello
import cv2
import time

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

tello.streamon()
frame_read = tello.get_frame_read()

speed = 35
move_time = 0.3
is_flying = False

print("""
CONTROLS:
t = takeoff
l = land
w/s = forward/back
a/d = left/right
r/f = up/down
q/e = rotate
p = take photo
ESC = quit
""")

while True:
    frame = frame_read.frame
    cv2.imshow("Tello Control", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('t'):
        if not is_flying:
            tello.takeoff()
            is_flying = True
            print("Takeoff")

    elif key == ord('l'):
        if is_flying:
            tello.land()
            is_flying = False
            print("Landing")

    elif key == ord('w') and is_flying:
        print("Forward")
        tello.send_rc_control(0, speed, 0, 0)
        time.sleep(move_time)
        tello.send_rc_control(0, 0, 0, 0)

    elif key == ord('s') and is_flying:
        print("Backward")
        tello.send_rc_control(0, -speed, 0, 0)
        time.sleep(move_time)
        tello.send_rc_control(0, 0, 0, 0)

    elif key == ord('a') and is_flying:
        print("Left")
        tello.send_rc_control(-speed, 0, 0, 0)
        time.sleep(move_time)
        tello.send_rc_control(0, 0, 0, 0)

    elif key == ord('d') and is_flying:
        print("Right")
        tello.send_rc_control(speed, 0, 0, 0)
        time.sleep(move_time)
        tello.send_rc_control(0, 0, 0, 0)

    elif key == ord('r') and is_flying:
        print("Up")
        tello.send_rc_control(0, 0, speed, 0)
        time.sleep(move_time)
        tello.send_rc_control(0, 0, 0, 0)

    elif key == ord('f') and is_flying:
        print("Down")
        tello.send_rc_control(0, 0, -speed, 0)
        time.sleep(move_time)
        tello.send_rc_control(0, 0, 0, 0)

    elif key == ord('q') and is_flying:
        print("Rotate left")
        tello.send_rc_control(0, 0, 0, -speed)
        time.sleep(move_time)
        tello.send_rc_control(0, 0, 0, 0)

    elif key == ord('e') and is_flying:
        print("Rotate right")
        tello.send_rc_control(0, 0, 0, speed)
        time.sleep(move_time)
        tello.send_rc_control(0, 0, 0, 0)

    elif key == ord('p'):
        filename = f"tello_photo_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Photo saved as {filename}")

    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
tello.streamoff()

if is_flying:
    tello.land()