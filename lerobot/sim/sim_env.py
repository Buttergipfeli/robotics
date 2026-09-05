import numpy as np
import mujoco
import mujoco.viewer

class So101PickBallEnv:
    CONTROL_HZ = 10
    EPISODE_SECONDS = 30

    MAT_X = (0.1366, 0.3866)
    MAT_Y = (-0.15, 0.15)
    BALL_RADIUS = 0.0334
    BALL_Z = 0.0244
    ROLL_RADIUS = 0.0605
    MIN_BALL_ROLL_DIST = 0.115

    def __init__(self, model, seed=None):
        self.model = model
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.n_substeps = int(1 / (self.CONTROL_HZ * self.model.opt.timestep))

    def launch(self):
        mujoco.viewer.launch(self.model, self.data)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.time = 0.0
        self.t = 0

        margin = self.ROLL_RADIUS + 0.01
        roll_xy = self.rng.uniform(
            [self.MAT_X[0] + margin, self.MAT_Y[0] + margin],
            [self.MAT_X[1] - margin, self.MAT_Y[1] - margin],
        )
        self.model.body("toilet_roll").pos[:2] = roll_xy
        #return self.get_observation()
