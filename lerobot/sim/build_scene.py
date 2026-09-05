from pathlib import Path

import mujoco

from task_objects import StressBall, ToiletRoll
from wrist_camera import WristCamera
from sim_env import So101PickBallEnv

SIM_DIR = Path(__file__).parent
SHOW_CAMERA = False
REST_QPOS = [0.0, -1.746, 1.586, 1.219, -1.571, 0.0]

def build_model():
    spec = mujoco.MjSpec.from_file(str(SIM_DIR / "my_scene.xml"))
    WristCamera(show_view_marker=SHOW_CAMERA, show_mount=SHOW_CAMERA).attach(spec.body("gripper"))
    ball = StressBall()
    ball.attach(spec)
    ToiletRoll().attach(spec)
    spec.add_key(name="rest", qpos=REST_QPOS + ball.qpos0(), ctrl=REST_QPOS)
    return spec.compile()

if __name__ == "__main__":
    env = So101PickBallEnv(build_model())
    env.reset()
    env.launch()
