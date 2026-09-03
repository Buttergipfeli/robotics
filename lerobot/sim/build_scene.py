from pathlib import Path

import mujoco
import mujoco.viewer

from wrist_camera import WristCamera

SIM_DIR = Path(__file__).parent
SHOW_CAMERA = True
REST_QPOS = [0.0, -1.746, 1.586, 1.219, -1.571, 0.0]

def build_model():
    spec = mujoco.MjSpec.from_file(str(SIM_DIR / "my_scene.xml"))
    WristCamera(show_view_marker=SHOW_CAMERA, show_mount=SHOW_CAMERA).attach(spec.body("gripper"))
    spec.add_key(name="rest", qpos=REST_QPOS, ctrl=REST_QPOS)
    return spec.compile()

if __name__ == "__main__":
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.viewer.launch(model, data)
