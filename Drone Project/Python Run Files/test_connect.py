from djitellopy import Tello
import time

tello = Tello()

print("Connecting...")
tello.connect()

print("Battery:", tello.get_battery())

tello.takeoff()

time.sleep(5)

tello.land()

print("Done.")