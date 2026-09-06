import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from build_scene import build_model
from expert import ScriptedExpert
from sim_env import So101PickBallEnv

SIM_DIR = Path(__file__).parent

REPO_ID = "local/so101_ball_in_roll"
DATA_ROOT = SIM_DIR / "data" / "so101_ball_in_roll"
ROBOT_TYPE = "so101_follower"
TASK = "Pick up the ball and place it on the toilet roll"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IMAGE_NAMES = ["height", "width", "channels"]
IMAGE_CHANNELS = 3
VALUE_DTYPE = "float32"

EXPERT_NOISE_STD = 0.5
DEFAULT_TARGET_EPISODES = 50
DEFAULT_SEED = 0


def build_features(env):
    return {
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (env.CAM_HEIGHT, env.CAM_WIDTH, IMAGE_CHANNELS),
            "names": IMAGE_NAMES,
        },
        "observation.state": {
            "dtype": VALUE_DTYPE,
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": VALUE_DTYPE,
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
    }


def main(target_episodes=DEFAULT_TARGET_EPISODES, seed=DEFAULT_SEED):
    env = So101PickBallEnv(build_model(), seed=seed)
    expert = ScriptedExpert(env, noise_std=EXPERT_NOISE_STD, seed=seed)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=env.CONTROL_HZ,
        root=str(DATA_ROOT),
        robot_type=ROBOT_TYPE,
        features=build_features(env),
    )

    saved = 0
    attempts = 0
    while saved < target_episodes:
        attempts += 1
        obs = env.reset()
        expert.reset()
        while True:
            action = expert.act()
            dataset.add_frame({
                "observation.images.wrist": obs["image"],
                "observation.state": obs["state"].astype(VALUE_DTYPE),
                "action": action.astype(VALUE_DTYPE),
                "task": TASK,
            })
            obs, done, info = env.step(action)
            if done or expert.failed:
                break

        if info.get("success"):
            dataset.save_episode()
            saved += 1
            print(f"saved {saved}/{target_episodes} (attempts: {attempts})")
        else:
            dataset.clear_episode_buffer()
            print(f"discarded (attempts: {attempts})")


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET_EPISODES
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEED
    main(target_episodes=target, seed=seed)
