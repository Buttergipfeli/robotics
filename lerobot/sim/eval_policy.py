import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from build_scene import build_model
from sim_env import So101PickBallEnv

SIM_DIR = Path(__file__).parent

CHECKPOINT = SIM_DIR / "train" / "act_ball" / "checkpoints" / "last" / "pretrained_model"
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


def main(episodes=DEFAULT_EPISODES, seed=DEFAULT_SEED):
    policy = ACTPolicy.from_pretrained(str(CHECKPOINT))
    policy.to(DEVICE)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(CHECKPOINT)
    )

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
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EPISODES
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEED
    main(episodes=episodes, seed=seed)
