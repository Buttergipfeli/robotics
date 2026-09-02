# Pyhton Installation Guide

## Install Python via `uv`

Install the python version 3.13:

```bash
uv python install 3.13
```

After that set up venv with pip in it:

```bash
uv venv --python 3.13 --seed
```

Activate venv:

```bash
source ./.venv/bin/activate
```

## Install packages

### mujoco

```bash
pip install mujoco
```

### lerobot

Clone the lerobot repository

```bash
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git
```

## Start the MuJoCo viewer

```bash
python -m mujoco.viewer --mjcf=SO-ARM100/Simulation/SO101/scene.xml
```
