import sys
import time

import mujoco
import mujoco.viewer
import torch

from build_scene import build_model
from eval_policy import load_policy, obs_to_raw
from sim_env import So101PickBallEnv


def main(episodes=10, seed=100):
    policy, preprocessor, postprocessor = load_policy()

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
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(episodes=episodes, seed=seed)
