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
    ARM_REST_XY = (0.157, 0.0)

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
        while True:
            roll_xy = self.rng.uniform(
                [self.MAT_X[0] + margin, self.MAT_Y[0] + margin],
                [self.MAT_X[1] - margin, self.MAT_Y[1] - margin],
            )
            if np.linalg.norm(roll_xy - self.ARM_REST_XY) >= self.ROLL_RADIUS + 0.06:
                break
        self.model.body("toilet_roll").pos[:2] = roll_xy

        while True:
            ball_xy = self.rng.uniform(
                [self.MAT_X[0] + self.BALL_RADIUS, self.MAT_Y[0] + self.BALL_RADIUS],
                [self.MAT_X[1] - self.BALL_RADIUS, self.MAT_Y[1] - self.BALL_RADIUS],
            )
            if (
                np.linalg.norm(ball_xy - roll_xy) >= self.MIN_BALL_ROLL_DIST
                and np.linalg.norm(ball_xy - self.ARM_REST_XY) >= 0.09
            ):
                break


        self.data.qpos[6:9] = [*ball_xy, self.BALL_Z]
        self.data.qpos[9:13] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)
