import time

import mujoco
import mujoco.viewer

from build_scene import build_model
from expert import ScriptedExpert
from sim_env import So101PickBallEnv


def main(episodes=10, seed=0):
    env = So101PickBallEnv(build_model(), seed=seed)
    expert = ScriptedExpert(env, noise_std=0.5, seed=seed)
    tick = 1.0 / env.CONTROL_HZ

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(episodes):
            if not viewer.is_running():
                break
            env.reset()
            expert.reset()
            viewer.sync()
            time.sleep(1.0)
            while viewer.is_running():
                start = time.time()
                obs, done, info = env.step(expert.act())
                viewer.sync()
                if done or expert.failed:
                    result = "SUCCESS" if info.get("success") else f"FAIL ({expert.phase})"
                    print(f"Episode {ep}: {result}")
                    time.sleep(1.5)
                    break
                time.sleep(max(0.0, tick - (time.time() - start)))


if __name__ == "__main__":
    main()
