import numpy as np
import mujoco
import mujoco.viewer

class So101PickBallEnv:
    CONTROL_HZ = 10
    EPISODE_SECONDS = 30
    CAM_WIDTH = 640
    CAM_HEIGHT = 480

    MAT_X = (0.1366, 0.3866)
    MAT_Y = (-0.15, 0.15)
    MAT_EDGE_MARGIN = 0.01

    BALL_RADIUS = 0.0334
    BALL_Z = 0.0244
    ROLL_BODY = "toilet_roll"
    ROLL_RADIUS = 0.0605
    ROLL_HEIGHT = 0.10

    ARM_REST_XY = (0.157, 0.0)
    MIN_BALL_ROLL_DIST = 0.115
    MIN_BALL_ARM_DIST = 0.09
    MIN_ROLL_ARM_CLEARANCE = 0.06

    ARM_QPOS = slice(0, 6)
    BALL_POS = slice(6, 9)
    BALL_QUAT = slice(9, 13)
    BALL_LIN_VEL = slice(6, 9)
    IDENTITY_QUAT = (1, 0, 0, 0)

    SUCCESS_MAX_HORIZONTAL_DIST = 0.015
    SUCCESS_MAX_SPEED = 0.05
    BALL_LOST_MAX_Z = 0.05

    def __init__(self, model, seed=None):
        self.model = model
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.n_substeps = int(1 / (self.CONTROL_HZ * self.model.opt.timestep))
        self.max_steps = self.CONTROL_HZ * self.EPISODE_SECONDS
        self.renderer = mujoco.Renderer(self.model, height=self.CAM_HEIGHT, width=self.CAM_WIDTH)

    def launch(self):
        mujoco.viewer.launch(self.model, self.data)

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.t += 1

        obs = self.get_observation()
        success = self.is_success()
        timeout = self.t >= self.max_steps
        ball_lost = self.is_ball_lost()
        done = success or timeout or ball_lost

        return obs, done, {"success": success, "timeout": timeout, "ball_lost": ball_lost}

    def is_success(self):
        ball = self.data.qpos[self.BALL_POS]
        roll = self.model.body(self.ROLL_BODY).pos
        horizontal = np.linalg.norm(ball[:2] - roll[:2])
        seated = (
            horizontal < self.SUCCESS_MAX_HORIZONTAL_DIST
            and ball[2] > roll[2] + self.ROLL_HEIGHT
        )
        still = np.linalg.norm(self.data.qvel[self.BALL_LIN_VEL]) < self.SUCCESS_MAX_SPEED
        return seated and still

    def is_ball_lost(self):
        x, y, z = self.data.qpos[self.BALL_POS]
        on_mat = self.MAT_X[0] < x < self.MAT_X[1] and self.MAT_Y[0] < y < self.MAT_Y[1]
        return z < self.BALL_LOST_MAX_Z and not on_mat

    def get_observation(self):
        self.renderer.update_scene(self.data, camera="wrist")
        image = self.renderer.render()
        joints = self.data.qpos[self.ARM_QPOS].copy()
        return {"image": image, "state": joints}

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.time = 0.0
        self.t = 0

        margin = self.ROLL_RADIUS + self.MAT_EDGE_MARGIN
        while True:
            roll_xy = self.rng.uniform(
                [self.MAT_X[0] + margin, self.MAT_Y[0] + margin],
                [self.MAT_X[1] - margin, self.MAT_Y[1] - margin],
            )
            if np.linalg.norm(roll_xy - self.ARM_REST_XY) >= self.ROLL_RADIUS + self.MIN_ROLL_ARM_CLEARANCE:
                break
        self.model.body(self.ROLL_BODY).pos[:2] = roll_xy

        while True:
            ball_xy = self.rng.uniform(
                [self.MAT_X[0] + self.BALL_RADIUS, self.MAT_Y[0] + self.BALL_RADIUS],
                [self.MAT_X[1] - self.BALL_RADIUS, self.MAT_Y[1] - self.BALL_RADIUS],
            )
            if (
                np.linalg.norm(ball_xy - roll_xy) >= self.MIN_BALL_ROLL_DIST
                and np.linalg.norm(ball_xy - self.ARM_REST_XY) >= self.MIN_BALL_ARM_DIST
            ):
                break

        self.data.qpos[self.BALL_POS] = [*ball_xy, self.BALL_Z]
        self.data.qpos[self.BALL_QUAT] = self.IDENTITY_QUAT

        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()
