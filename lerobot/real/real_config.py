from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

FOLLOWER_PORT = "/dev/tty.usbmodemXXXX"
LEADER_PORT = "/dev/tty.usbmodemYYYY"
FOLLOWER_ID = "so101_follower_main"
LEADER_ID = "so101_leader_main"

CAMERA_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

CONTROL_HZ = 10
TELEOP_HZ = 50
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REST_ACTION = [0.0, -100.0, 93.8, 73.5, -58.0, 0.0]

MAX_RELATIVE_TARGET = 20.0


def make_follower(with_camera=True, max_relative_target=MAX_RELATIVE_TARGET):
    cameras = {}
    if with_camera:
        cameras["wrist"] = OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            fps=CAM_FPS,
            width=CAM_WIDTH,
            height=CAM_HEIGHT,
        )
    config = SOFollowerRobotConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        use_degrees=False,
        cameras=cameras,
        max_relative_target=max_relative_target,
    )
    return SOFollower(config)


def make_leader():
    config = SOLeaderTeleopConfig(
        port=LEADER_PORT,
        id=LEADER_ID,
        use_degrees=False,
    )
    return SOLeader(config)
