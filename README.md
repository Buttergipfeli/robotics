# SO-101 sim-to-real: ball on a toilet roll

MuJoCo simulation, scripted expert, ACT training and real-robot rollout for a
LeRobot SO-101 arm. The task: pick a stress ball off a cork mat and place it on
top of a toilet paper roll, observed by a single wrist camera. Policies are
trained on simulated demonstrations only and transferred to the real arm.

## Layout

| Path | Content |
| --- | --- |
| `lerobot/sim/` | scene, environment, scripted expert, episode recorder, training and evaluation |
| `lerobot/real/` | calibration, teleoperation, camera checks, policy rollout on the real arm |
| `lerobot/runpod/` | cloud training launcher and log viewer |
| `installation-guide.md` | Python and package setup |
| `lerobot/physical-env.md` | scene, task, randomisation and the real-to-sim joint mapping |
| `lerobot/runpod-training.md` | cloud training setup and workflow |
| `lerobot/training-progress.md` | training runs and results |

## Setup

1. Follow `installation-guide.md`.
2. Clone the SO-ARM100 repository into `lerobot/SO-ARM100/`. The scene
   includes its SO-101 model; it is not part of this repository.
3. Copy `lerobot/real/env.example` to `lerobot/real/.env` and
   `lerobot/runpod/env.example` to `lerobot/runpod/.env`, then fill in your
   serial ports, camera index and RunPod key. Both `.env` files are gitignored.

## Workflow

```bash
./.venv/bin/python3 lerobot/sim/test_expert.py 20
./.venv/bin/python3 lerobot/sim/record_episodes.py 300
./.venv/bin/python3 lerobot/runpod/launch_training.py
./.venv/bin/python3 lerobot/sim/eval_policy.py 50
./.venv/bin/python3 lerobot/real/rollout_policy.py
```

`train_policy.py` runs the same training locally. `watch_expert.py` and
`watch_policy.py` open a live viewer (use `mjpython` on macOS).

Datasets, checkpoints, calibration files and camera captures are local
artefacts and not committed.

## Safety

`rollout_policy.py` drives a real robot arm with a learned policy. The policy
can and will command unexpected motions: it can drive joints against their
mechanical stops, collide with objects and pinch fingers. In this project a
policy pushed a joint into its stop hard enough to damage a servo gear.

Test every checkpoint in simulation first, keep clear of the arm, keep the
power supply within reach and be ready to cut it. `RANGE_MARGIN` in
`lerobot/real/joint_mapping.py` keeps commands away from the calibrated
limits, but it is no substitute for supervision.

Everything here is provided as is, without warranty of any kind. You run it
on your own hardware at your own risk; the authors accept no liability for
damage to robots, equipment, property or persons. See sections 7 and 8 of the
`LICENSE`.

## License

Copyright 2026 Andre Kocher. Licensed under the Apache License, Version 2.0,
see `LICENSE`.

This project uses third-party software, each under its own license. Direct
dependencies (pinned in `requirements.txt`):

| Package | License |
| --- | --- |
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause |
| [opencv-python-headless](https://github.com/opencv/opencv-python) | Apache-2.0 |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | BSD-3-Clause |
| [runpod-python](https://github.com/runpod/runpod-python) | MIT |

LeRobot in turn pulls in, among others, torchvision (BSD-3-Clause), torchcodec
(BSD-3-Clause), Hugging Face Hub (Apache-2.0), datasets (Apache-2.0) and PyAV
(BSD-3-Clause). The complete resolved set is whatever `pip install -r
requirements.txt` installs; `pip-licenses` prints it with licenses on demand.

The SO-101 model comes from [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache-2.0); it is cloned separately and not redistributed here.
