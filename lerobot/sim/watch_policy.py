import argparse
import time

import mujoco
import mujoco.viewer
import torch

from build_scene import build_model
from eval_policy import DEFAULT_CHECKPOINT, load_policy, obs_to_raw
from sim_env import So101PickBallEnv


def main(episodes=10, seed=100, checkpoint=DEFAULT_CHECKPOINT):
    policy, preprocessor, postprocessor = load_policy(checkpoint)

    env = So101PickBallEnv(build_model(), seed=seed)
    tick = 1.0 / env.CONTROL_HZ

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(episodes):
            if not viewer.is_running():
                break
            obs = env.reset()
            policy.reset()
            viewer.sync()
            time.sleep(1.0)
            while viewer.is_running():
                start = time.time()
                batch = preprocessor(obs_to_raw(obs))
                with torch.inference_mode():
                    action = policy.select_action(batch)
                action = postprocessor(action)
                obs, done, info = env.step(action.squeeze(0).numpy())
                viewer.sync()
                if done:
                    if info.get("success"):
                        result = "SUCCESS"
                    elif info.get("timeout"):
                        result = "FAIL (timeout)"
                    else:
                        result = "FAIL (ball_lost)"
                    print(f"Episode {ep}: {result}")
                    time.sleep(1.5)
                    break
                time.sleep(max(0.0, tick - (time.time() - start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, nargs="?", default=10)
    parser.add_argument("seed", type=int, nargs="?", default=100)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    main(episodes=args.episodes, seed=args.seed, checkpoint=args.checkpoint)
