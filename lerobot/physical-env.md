# Simulation environment

MuJoCo model of an SO-101 arm working above a cork mat, observed by a single
wrist camera.

## Physical scene

| Part | Dimensions |
| --- | --- |
| Cork mat | 250 mm deep x 300 mm wide, 3 mm thick, near edge 72 mm in front of the arm's base foot |
| Platform under the arm | 237 mm wide x 167 mm deep, 12 mm high, front edge flush with the mat |

The arm faces the **short** side of the mat, so its 300 mm run left to right.

## Arm

Six joints, restricted to the travel the follower's own calibration reports
rather than the MJCF limits:

| Joint | Range |
| --- | --- |
| shoulder_pan | 167° |
| shoulder_lift | 187° |
| elbow_flex | 165° |
| wrist_flex | 156° |
| wrist_roll | 320° |
| gripper | 98° |

One servo tick is **0.088°** (4096 steps per turn) and LeRobot truncates anything
finer.

Start pose puts the tip 117 mm above the mat, so the camera has the mat in
frame.

## Camera

A single standard camera mounted on the follower's wrist, in the same position
as on the SO-101 follower arm described in the NVIDIA Isaac sim-to-real guide:
640 x 480 at 30 fps, 70° vertical field of view. There is no screen capture,
the camera frame is the only image observation.

## Observation

- one 640 x 480 wrist-camera frame
- six joint values in the follower's normalised units

## Per-episode randomisation

These quantities vary so nothing can be learned against a fixed value:

| Quantity | Range |
| --- | --- |
| Mat friction | 0.15 to 0.55 |
| Mat distance | 238 to 258 mm |
| Mat thickness | 2.5 to 3.5 mm |
| Platform height | 9 to 15 mm |
| Actuator stiffness | 350 to 1000 |
| Camera position / tilt / fovy | ±10 mm, ±0.05 rad, 64 to 76° |
| Light and shades | cork 0.62 to 0.86, shell 0.55 to 0.95 |

Control rate is drawn from 10, 5.7, 5, 3 or 2 Hz, because the real rollout ran at
5.7 Hz against a nominal 10. In 35% of episodes the policy is handed the previous
frame instead of the current one, since on the bench the camera read and the
inference both finish before the action lands.

Physics runs at a 2 ms timestep; an episode lasts 60 seconds.
