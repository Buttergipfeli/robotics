import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from build_scene import build_model
from sim_env import So101PickBallEnv

SIM_DIR = Path(__file__).parent

CHECKPOINTS_DIR = SIM_DIR / "train" / "act_ball" / "checkpoints"
DEFAULT_CHECKPOINT = "last"
DEVICE = "mps"
DEFAULT_EPISODES = 20
DEFAULT_SEED = 100


def obs_to_raw(obs):
    image = torch.from_numpy(obs["image"].copy()).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(obs["state"].astype(np.float32))
    return {
        "observation.images.wrist": image,
        "observation.state": state,
    }


def checkpoint_dir(name=DEFAULT_CHECKPOINT):
    if name.isdigit():
        name = f"{int(name):06d}"
    path = (Path(name) if "/" in name else CHECKPOINTS_DIR / name) / "pretrained_model"
    if not path.exists():
        sys.exit(f"Checkpoint not found: {path}")
    return path


def load_policy(checkpoint=DEFAULT_CHECKPOINT):
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
    return policy, preprocessor, postprocessor


def main(episodes=DEFAULT_EPISODES, seed=DEFAULT_SEED, checkpoint=DEFAULT_CHECKPOINT):
    policy, preprocessor, postprocessor = load_policy(checkpoint)

    env = So101PickBallEnv(build_model(), seed=seed)

    successes = 0
    for ep in range(episodes):
        obs = env.reset()
        policy.reset()
        info = {}
        while True:
            batch = preprocessor(obs_to_raw(obs))
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocessor(action)
            obs, done, info = env.step(action.squeeze(0).numpy())
            if done:
                break
        success = bool(info.get("success"))
        successes += success
        result = "SUCCESS" if success else f"FAIL ({'timeout' if info.get('timeout') else 'ball_lost'})"
        print(f"Episode {ep}: {result}, steps={env.t}")

    print(f"\nSuccess rate: {successes}/{episodes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, nargs="?", default=DEFAULT_EPISODES)
    parser.add_argument("seed", type=int, nargs="?", default=DEFAULT_SEED)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    main(episodes=args.episodes, seed=args.seed, checkpoint=args.checkpoint)
