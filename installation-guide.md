# Installation guide

## Python via `uv`

```bash
uv python install 3.13
```

```bash
uv venv --python 3.13 --seed
```

```bash
source ./.venv/bin/activate
```

## Packages

All direct dependencies are pinned in `requirements.txt` to the versions this
project was developed and tested with:

```bash
pip install -r requirements.txt
```

## SO-101 model

The scene includes the SO-101 model from the SO-ARM100 repository. Clone it
into `lerobot/SO-ARM100/`, the path the scene expects:

```bash
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git lerobot/SO-ARM100
```

## Check the setup

```bash
./.venv/bin/python3 lerobot/sim/test_expert.py 3
```

Interactive viewers need `mjpython` on macOS:

```bash
./.venv/bin/mjpython lerobot/sim/watch_expert.py
```
