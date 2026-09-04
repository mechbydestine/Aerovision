# Hardware notes

The physical side of the project. The drone didn't fly when I got it, so this is what was broken, what I replaced, and the two frames I modeled in Fusion 360.

<img width="360" alt="project thumbnail" src="media/build-collage.png" />

## What I repaired
- Replaced battery setup with new **1100mAh 3.8V** batteries
- Added a **USB-C charging dock**
- Bought replacement **brushed motors** to keep cost low and soldered it into the original frame
- Verified that the drone could fly correctly again before coding

Fixing the drone before writing any code is the call I'd make again. With three weeks I didn't want to be debugging a gesture classifier and a drone that might not fly at the same time.

## The numbers I worked off

These are the published Ryze Tello specs I checked my CAD against:

| | |
| --- | --- |
| Dimensions | 98 x 92.5 x 41 mm |
| Take-off weight | about 80 g with props and battery |
| Battery | 1100 mAh, 3.8 V |
| Camera | 5 MP photos, 720p stream |
| Control link | wifi, UDP port 8889 for commands and 11111 for video |

## Design process
I modeled the drone in Fusion 360 using real measurements from the drone and compared them against the published dimensions to keep the frame accurate. I created two versions: the first included raised leg features but added too much weight and was scrapped, while the second version focused on a smaller, lighter body with more open areas for airflow.

| Version | File | How it went |
| --- | --- | --- |
| v1 spider | [frame-v1-spider.stl](../hardware/stl/frame-v1-spider.stl) | raised leg features, too heavy, scrapped |
| v2 spider | [frame-v2-spider.stl](../hardware/stl/frame-v2-spider.stl) | same idea, trimmed down |
| v2 base | [frame-v2-base.stl](../hardware/stl/frame-v2-base.stl) | smaller body with open areas for airflow, current design |
| v2 top | [frame-v2-top.stl](../hardware/stl/frame-v2-top.stl) | matching top plate |

<img width="46%" alt="v2 base" src="../hardware/renders/frame-v2-base.png" /> <img width="46%" alt="v2 top" src="../hardware/renders/frame-v2-top.png" />

## Why I redesigned the frame

The drone overheats after several minutes of operation or near the end of battery life, and that caused delayed or laggy movement on the next startup. The v2 frame goes after that directly with less material, a smaller body, and open sections so air can actually move over the boards instead of sitting inside a closed shell.

## The print that didn't fit

The second frame was accidentally scaled too large before printing, which made the propeller arms too long to swap in safely. The parts printed clean, they just couldn't go on the drone. I caught it by measuring the print against the real airframe, so now I do that check before anything goes to the printer.

## Still to do
- Resize and reprint the V2 frame accurately
- Tune the thermal design and test airflow changes
- Add a final lightweight shell inspired by the spider-drone concept art
