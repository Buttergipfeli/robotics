import numpy as np
import mujoco
import mujoco.viewer

class So101PickBallEnv:
    CONTROL_HZ = 10
    EPISODE_SECONDS = 30

    MAT_X = (0.1366, 0.3866)
    MAT_Y = (-0.15, 0.15)
    BALL_RADIUS = 0.025
    BASKET_RADIUS = 0.05
    MIN_BALL_BASKET_DIST = 0.10
    BALL_Z = 0.016

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
        self.data.t = 0.0

        margin = self.BASKET_RADIUS + 0.01
        basket_xy = self.rng.uniform(
            [self.MAT_X[0] + margin, self.MAT_Y[0] + margin],
            [self.MAT_X[1] - margin, self.MAT_Y[1] - margin],
        )
        self.model.body("basket").pos[:2] = basket_xy
        #return self.get_observation()
