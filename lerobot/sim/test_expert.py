import numpy as np

from build_scene import build_model
from expert import ScriptedExpert
from sim_env import So101PickBallEnv


def main(episodes=10, seed=0):
    env = So101PickBallEnv(build_model(), seed=seed)
    expert = ScriptedExpert(env, noise_std=0.5, seed=seed)

    successes = 0
    for ep in range(episodes):
        env.reset()
        expert.reset()
        info = {}
        while True:
            action = expert.act()
            obs, done, info = env.step(action)
            if done or expert.failed:
                break
            if expert.done() and env.t > expert.RELEASE_WAIT:
                pass
        result = "SUCCESS" if info.get("success") else f"FAIL (phase={expert.phase}, {info})"
        successes += bool(info.get("success"))
        print(f"Episode {ep}: {result}, steps={env.t}")

    print(f"\nSuccess rate: {successes}/{episodes}")


if __name__ == "__main__":
    import sys

    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(episodes=episodes, seed=seed)
