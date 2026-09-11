import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from joint_mapping import JointMapping
from real_config import CONTROL_HZ, JOINT_NAMES, REST_ACTION, make_follower

CHECKPOINTS_DIR = Path(__file__).parents[1] / "sim" / "train" / "act_ball" / "checkpoints"
DEFAULT_CHECKPOINT = "last"
DEVICE = "mps"
EPISODE_SECONDS = 30
REST_SECONDS = 3.0


def obs_to_raw(obs, mapping):
    image = torch.from_numpy(obs["wrist"].copy()).permute(2, 0, 1).float() / 255.0
    state = mapping.real_to_sim([obs[f"{j}.pos"] for j in JOINT_NAMES])
    return {
        "observation.images.wrist": image,
        "observation.state": torch.from_numpy(state.astype(np.float32)),
    }


def vector_to_action(vec):
    return {f"{name}.pos": float(v) for name, v in zip(JOINT_NAMES, vec)}


def move_to_rest(robot, mapping):
    obs = robot.get_observation()
    current = np.array([obs[f"{j}.pos"] for j in JOINT_NAMES], dtype=np.float64)
    target = mapping.sim_to_real(REST_ACTION)
    steps = int(REST_SECONDS * CONTROL_HZ)
    for i in range(1, steps + 1):
        blend = current + (target - current) * i / steps
        robot.send_action(vector_to_action(blend))
        time.sleep(1 / CONTROL_HZ)


def checkpoint_dir(name=DEFAULT_CHECKPOINT):
    if name.isdigit():
        name = f"{int(name):06d}"
    path = (Path(name) if "/" in name else CHECKPOINTS_DIR / name) / "pretrained_model"
    if not path.exists():
        sys.exit(f"Checkpoint not found: {path}")
    return path


def main(episodes=1, checkpoint=DEFAULT_CHECKPOINT):
    path = checkpoint_dir(checkpoint)
    print(f"Checkpoint {path.parent.resolve().name}")
    policy = ACTPolicy.from_pretrained(str(path))
    policy.to(DEVICE)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )

    robot = make_follower(with_camera=True)
    robot.connect()
    mapping = JointMapping(robot.bus.calibration)
    try:
        move_to_rest(robot, mapping)
        for ep in range(episodes):
            input(f"Episode {ep}: place ball and roll on the mat, then press Enter...")
            policy.reset()
            for _ in range(EPISODE_SECONDS * CONTROL_HZ):
                start = time.perf_counter()
                obs = robot.get_observation()
                batch = preprocessor(obs_to_raw(obs, mapping))
                with torch.inference_mode():
                    action = policy.select_action(batch)
                action = postprocessor(action)
                robot.send_action(vector_to_action(mapping.sim_to_real(action.squeeze(0).numpy())))
                time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - start)))
            move_to_rest(robot, mapping)
    except KeyboardInterrupt:
        print("\nRollout aborted, returning to rest.")
    finally:
        move_to_rest(robot, mapping)
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, nargs="?", default=1)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    main(episodes=args.episodes, checkpoint=args.checkpoint)
