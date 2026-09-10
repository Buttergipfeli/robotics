import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from real_config import CONTROL_HZ, JOINT_NAMES, REST_ACTION, make_follower

CHECKPOINT = Path(__file__).parents[1] / "sim" / "train" / "act_ball" / "checkpoints" / "last" / "pretrained_model"
DEVICE = "mps"
EPISODE_SECONDS = 30
REST_SECONDS = 3.0


def obs_to_raw(obs):
    image = torch.from_numpy(obs["wrist"].copy()).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(np.array([obs[f"{j}.pos"] for j in JOINT_NAMES], dtype=np.float32))
    return {
        "observation.images.wrist": image,
        "observation.state": state,
    }


def vector_to_action(vec):
    return {f"{name}.pos": float(v) for name, v in zip(JOINT_NAMES, vec)}


def move_to_rest(robot):
    obs = robot.get_observation()
    current = np.array([obs[f"{j}.pos"] for j in JOINT_NAMES], dtype=np.float64)
    target = np.array(REST_ACTION)
    steps = int(REST_SECONDS * CONTROL_HZ)
    for i in range(1, steps + 1):
        blend = current + (target - current) * i / steps
        robot.send_action(vector_to_action(blend))
        time.sleep(1 / CONTROL_HZ)


def main(episodes=1):
    policy = ACTPolicy.from_pretrained(str(CHECKPOINT))
    policy.to(DEVICE)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(CHECKPOINT),
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )

    robot = make_follower(with_camera=True)
    robot.connect()
    try:
        move_to_rest(robot)
        for ep in range(episodes):
            input(f"Episode {ep}: place ball and roll on the mat, then press Enter...")
            policy.reset()
            for _ in range(EPISODE_SECONDS * CONTROL_HZ):
                start = time.perf_counter()
                obs = robot.get_observation()
                batch = preprocessor(obs_to_raw(obs))
                with torch.inference_mode():
                    action = policy.select_action(batch)
                action = postprocessor(action)
                robot.send_action(vector_to_action(action.squeeze(0).numpy()))
                time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - start)))
            move_to_rest(robot)
    except KeyboardInterrupt:
        print("\nRollout aborted, returning to rest.")
    finally:
        move_to_rest(robot)
        robot.disconnect()


if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(episodes=episodes)
