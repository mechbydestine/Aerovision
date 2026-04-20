# AeroVision

##  Repairing and Reprogramming a Gesture-Controlled Imaging Drone
**2026 CODE Club Engineering Challenge @pgcc**

AeroVision restores and reprograms a broken Ryze Tello Drone by DJI into a Python-controlled aerial imaging drone with live camera view, keyboard flight control, photo capture, and video recording. The final target is a gesture-controlled interface using computer vision from a connected camera, plus a lighter custom 3D-printed frame to reduce heat and improve portability.

The original idea was to build a tiny autonomous spider-style drone that could hover, take pictures, record video, and respond to hand gestures. Because the timeline was only three weeks, I pivoted to repairing an old drone first and turning it into a programmable prototype that could realistically be finished on time.

## Project Thumbnail

## What I repaired
- Replaced battery setup with new **1100mAh 3.8V** batteries
- Added a **USB-C charging dock**
- Bought replacement **brushed motors** to keep cost low and soldered it into the original frame
- Verified that the drone could fly correctly again before coding
  
Pic

What works now
- Python connection to the Tello over Wi-Fi
- Live camera window in OpenCV
- Keyboard flight control
- Photo capture from the live feed
- Video recording from the live feed
- Gesture recognition through computer vision
- Hand gesture mapping to flight commands
Currently work in progress
- Lightweight custom frame redesign in Fusion 360
- Better airflow and weight reduction to address overheating

## Design process
I modeled the drone in Fusion 360 using real measurements from the drone and compared them against the published dimensions to keep the frame accurate. I created two versions: the first included raised leg features but added too much weight and was scrapped, while the second version focused on a smaller, lighter body with more open areas for airflow.

## Biggest challenges
1. **Overheating** after several minutes of operation or near the end of battery life, which caused delayed or laggy movement on the next startup.
2. **Scope control** because building a fully custom drone from scratch was too expensive and difficult for the deadline.
3. **3D printing fit issues** because the second frame was accidentally scaled too large before printing, which made the propeller arms too long to swap in safely.
4. **Software debugging** while moving from basic control scripts to gesture recognition.

## I learned that...
- It is better to get a working prototype first, then expand features.
- Mechanical design decisions directly affect flight performance and heat.
- Small scaling mistakes in CAD can completely change motor and propeller geometry.
- Python, OpenCV, and drone SDK control can turn a consumer drone into a programmable robotics platform.

## If I had more time
- Finish gesture-controlled flight
- Resize and reprint the V2 frame accurately
- Tune the thermal design and test airflow changes
- Add a final lightweight shell inspired by the spider-drone concept art

## Demo Video and Presentation!
