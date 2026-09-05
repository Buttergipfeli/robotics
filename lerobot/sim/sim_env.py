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

    GRIPPER_IDX = 5

    GRASP_ASSIST_RADIUS = 0.05
    GRIP_CLOSE_CTRL = 0.7
    GRIP_OPEN_CTRL = 0.9
    TCP_SITE = "gripperframe"

    def __init__(self, model, seed=None):
        self.model = model
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.n_substeps = int(1 / (self.CONTROL_HZ * self.model.opt.timestep))
        self.max_steps = self.CONTROL_HZ * self.EPISODE_SECONDS
        self.renderer = mujoco.Renderer(self.model, height=self.CAM_HEIGHT, width=self.CAM_WIDTH)

        ranges = np.array([self.model.joint(i).range for i in range(6)])
        self.joint_mid = (ranges[:, 0] + ranges[:, 1]) / 2
        self.joint_half = (ranges[:, 1] - ranges[:, 0]) / 2

        self.tcp_site_id = self.model.site(self.TCP_SITE).id
        self.grip_body_id = self.model.body("gripper").id
        self.ball_body_id = self.model.body("stress_ball").id
        self.grasp_eq_id = self.model.equality("grasp_assist").id

    def launch(self):
        mujoco.viewer.launch(self.model, self.data)

    def step(self, action):
        self.data.ctrl[:] = self.norm_to_rad(action)
        self._update_grasp_assist()
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
        if self.data.eq_active[self.grasp_eq_id]:
            return False
        ball = self.data.qpos[self.BALL_POS]
        roll = self.model.body(self.ROLL_BODY).pos
        horizontal = np.linalg.norm(ball[:2] - roll[:2])
        seated = (
            horizontal < self.SUCCESS_MAX_HORIZONTAL_DIST
            and ball[2] > roll[2] + self.ROLL_HEIGHT
        )
        still = np.linalg.norm(self.data.qvel[self.BALL_LIN_VEL]) < self.SUCCESS_MAX_SPEED
        return seated and still

    def _update_grasp_assist(self):
        active = bool(self.data.eq_active[self.grasp_eq_id])
        grip_cmd = self.data.ctrl[self.GRIPPER_IDX]
        if not active and grip_cmd < self.GRIP_CLOSE_CTRL:
            tcp = self.data.site_xpos[self.tcp_site_id]
            ball = self.data.qpos[self.BALL_POS]
            if np.linalg.norm(ball - tcp) < self.GRASP_ASSIST_RADIUS:
                self._activate_grasp_weld()
        elif active and grip_cmd > self.GRIP_OPEN_CTRL:
            self.data.eq_active[self.grasp_eq_id] = 0

    def _activate_grasp_weld(self):
        grip_rot = self.data.xmat[self.grip_body_id].reshape(3, 3)
        rel_pos = grip_rot.T @ (self.data.xpos[self.ball_body_id] - self.data.xpos[self.grip_body_id])
        grip_quat = self.data.xquat[self.grip_body_id]
        grip_quat_inv = np.array([grip_quat[0], -grip_quat[1], -grip_quat[2], -grip_quat[3]])
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, grip_quat_inv, self.data.xquat[self.ball_body_id])
        eq_data = self.model.eq_data[self.grasp_eq_id]
        eq_data[:] = 0
        eq_data[3:6] = rel_pos
        eq_data[6:10] = rel_quat
        eq_data[10] = 1.0
        self.data.eq_active[self.grasp_eq_id] = 1

    def is_ball_lost(self):
        x, y, z = self.data.qpos[self.BALL_POS]
        on_mat = self.MAT_X[0] < x < self.MAT_X[1] and self.MAT_Y[0] < y < self.MAT_Y[1]
        return z < self.BALL_LOST_MAX_Z and not on_mat

    def get_observation(self):
        self.renderer.update_scene(self.data, camera="wrist")
        image = self.renderer.render()
        joints = self.rad_to_norm(self.data.qpos[self.ARM_QPOS])
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

    def rad_to_norm(self, rad):
        rad = np.asarray(rad, dtype=np.float64)
        norm = (rad - self.joint_mid) / self.joint_half * 100.0
        gripper_lo = self.joint_mid[self.GRIPPER_IDX] - self.joint_half[self.GRIPPER_IDX]
        norm[self.GRIPPER_IDX] = (rad[self.GRIPPER_IDX] - gripper_lo) / (2 * self.joint_half[self.GRIPPER_IDX]) * 100.0
        return norm

    def norm_to_rad(self, norm):
        norm = np.asarray(norm, dtype=np.float64)
        rad = self.joint_mid + norm / 100.0 * self.joint_half
        gripper_lo = self.joint_mid[self.GRIPPER_IDX] - self.joint_half[self.GRIPPER_IDX]
        rad[self.GRIPPER_IDX] = gripper_lo + norm[self.GRIPPER_IDX] / 100.0 * (2 * self.joint_half[self.GRIPPER_IDX])
        return rad
