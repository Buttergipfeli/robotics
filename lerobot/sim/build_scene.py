from pathlib import Path

import mujoco
import mujoco.viewer

from wrist_camera import WristCamera

SIM_DIR = Path(__file__).parent
SHOW_CAMERA = True

def build_model():
    spec = mujoco.MjSpec.from_file(str(SIM_DIR / "my_scene.xml"))
    WristCamera(show_view_marker=SHOW_CAMERA, show_mount=SHOW_CAMERA).attach(spec.body("gripper"))
    return spec.compile()

if __name__ == "__main__":
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)
