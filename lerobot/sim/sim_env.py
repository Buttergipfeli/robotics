import numpy as np
import mujoco
import mujoco.viewer

class So101PickBallEnv:
    CONTROL_HZ = 10
    EPISODE_SECONDS = 30
    CAM_WIDTH = 640
    CAM_HEIGHT = 480

    MAT_BODY = "cork_mat"
    MAT_EDGE_MARGIN = 0.01

    BALL_RADIUS = 0.0334
    BALL_Z = 0.0244
    ROLL_BODY = "toilet_roll"
    ROLL_RADIUS = 0.0605
    ROLL_HEIGHT = 0.096

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
    SUCCESS_HOLD_TICKS = 10
    BALL_LOST_MAX_Z = 0.05

    GRIPPER_IDX = 5

    GRASP_ASSIST_RADIUS = 0.05
    GRIP_CLOSE_CTRL = 0.9
    GRIP_OPEN_CTRL = 1.0
    TCP_SITE = "gripperframe"
    BALL_HOLD_POS = (0.020, 0.0, -0.090)

    MAT_SHIFT = 0.01
    BALL_FRICTION_RANGE = (0.8, 2.5)
    BALL_MASS_SCALE_RANGE = (0.9, 1.1)
    CAM_POS_JITTER = 0.008
    CAM_TILT_JITTER = 0.05
    CAM_FOVY_RANGE = (64.0, 76.0)
    LIGHT_POS_JITTER = 0.3
    LIGHT_DIFFUSE_SCALE_RANGE = (0.6, 1.2)
    MAT_SHADE_SCALE_RANGE = (0.8, 1.15)
    PAPER_SHADE_SCALE_RANGE = (0.85, 1.02)

    def __init__(self, model, seed=None, randomize=True):
        self.model = model
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
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

        self.mat_body_id = self.model.body(self.MAT_BODY).id
        self.mat_geom_id = next(
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.mat_body_id
        )
        self.mat_half = self.model.geom_size[self.mat_geom_id][:2].copy()
        self.ball_geom_id = self.model.geom("stress_ball").id
        self.cam_id = self.model.camera("wrist").id
        self.paper_geom_ids = [
            g for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("roll_paper")
        ]

        self.base_mat_pos = self.model.body_pos[self.mat_body_id].copy()
        self.base_mat_rgba = self.model.geom_rgba[self.mat_geom_id].copy()
        self.base_paper_rgba = self.model.geom_rgba[self.paper_geom_ids[0]].copy()
        self.base_ball_mass = float(self.model.body_mass[self.ball_body_id])
        self.base_ball_inertia = self.model.body_inertia[self.ball_body_id].copy()
        self.base_cam_pos = self.model.cam_pos[self.cam_id].copy()
        self.base_cam_quat = self.model.cam_quat[self.cam_id].copy()
        self.base_light_pos = self.model.light_pos[0].copy() if self.model.nlight else None
        self.base_light_diffuse = self.model.light_diffuse[0].copy() if self.model.nlight else None

    def launch(self):
        mujoco.viewer.launch(self.model, self.data)

    def step(self, action):
        self.data.ctrl[:] = self.norm_to_rad(action)
        self._update_grasp_assist()
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.t += 1

        obs = self.get_observation()
        if self.is_success():
            self.success_streak += 1
        else:
            self.success_streak = 0
        success = self.success_streak >= self.SUCCESS_HOLD_TICKS
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
            grip_rot = self.data.xmat[self.grip_body_id].reshape(3, 3)
            hold_world = self.data.xpos[self.grip_body_id] + grip_rot @ np.array(self.BALL_HOLD_POS)
            ball = self.data.qpos[self.BALL_POS]
            if np.linalg.norm(ball - hold_world) < self.GRASP_ASSIST_RADIUS:
                self._activate_grasp_weld()
        elif active and grip_cmd > self.GRIP_OPEN_CTRL:
            self.data.eq_active[self.grasp_eq_id] = 0

    def _activate_grasp_weld(self):
        eq_data = self.model.eq_data[self.grasp_eq_id]
        eq_data[:] = 0
        eq_data[3:6] = self.BALL_HOLD_POS
        eq_data[6:10] = [1, 0, 0, 0]
        eq_data[10] = 1.0
        self.data.eq_active[self.grasp_eq_id] = 1

    def is_ball_lost(self):
        x, y, z = self.data.qpos[self.BALL_POS]
        mat_x, mat_y = self._mat_bounds()
        on_mat = mat_x[0] < x < mat_x[1] and mat_y[0] < y < mat_y[1]
        return z < self.BALL_LOST_MAX_Z and not on_mat

    def _mat_bounds(self):
        pos = self.model.body_pos[self.mat_body_id]
        return (
            (pos[0] - self.mat_half[0], pos[0] + self.mat_half[0]),
            (pos[1] - self.mat_half[1], pos[1] + self.mat_half[1]),
        )

    def _randomize_domain(self):
        rng = self.rng
        model = self.model

        model.body_pos[self.mat_body_id][:2] = self.base_mat_pos[:2] + rng.uniform(
            -self.MAT_SHIFT, self.MAT_SHIFT, 2
        )
        model.geom_friction[self.ball_geom_id][0] = rng.uniform(*self.BALL_FRICTION_RANGE)
        mass_scale = rng.uniform(*self.BALL_MASS_SCALE_RANGE)
        model.body_mass[self.ball_body_id] = self.base_ball_mass * mass_scale
        model.body_inertia[self.ball_body_id] = self.base_ball_inertia * mass_scale

        model.cam_pos[self.cam_id] = self.base_cam_pos + rng.uniform(
            -self.CAM_POS_JITTER, self.CAM_POS_JITTER, 3
        )
        quat = self.base_cam_quat.copy()
        for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
            tilt = np.zeros(4)
            mujoco.mju_axisAngle2Quat(
                tilt, np.array(axis), rng.uniform(-self.CAM_TILT_JITTER, self.CAM_TILT_JITTER)
            )
            rotated = np.zeros(4)
            mujoco.mju_mulQuat(rotated, quat, tilt)
            quat = rotated
        model.cam_quat[self.cam_id] = quat
        model.cam_fovy[self.cam_id] = rng.uniform(*self.CAM_FOVY_RANGE)

        mat_shade = rng.uniform(*self.MAT_SHADE_SCALE_RANGE)
        model.geom_rgba[self.mat_geom_id][:3] = np.clip(self.base_mat_rgba[:3] * mat_shade, 0, 1)
        paper_shade = rng.uniform(*self.PAPER_SHADE_SCALE_RANGE)
        for g in self.paper_geom_ids:
            model.geom_rgba[g][:3] = np.clip(self.base_paper_rgba[:3] * paper_shade, 0, 1)

        if self.model.nlight:
            model.light_pos[0][:2] = self.base_light_pos[:2] + rng.uniform(
                -self.LIGHT_POS_JITTER, self.LIGHT_POS_JITTER, 2
            )
            model.light_diffuse[0] = self.base_light_diffuse * rng.uniform(
                *self.LIGHT_DIFFUSE_SCALE_RANGE
            )

    def get_observation(self):
        self.renderer.update_scene(self.data, camera="wrist")
        image = self.renderer.render()
        joints = self.rad_to_norm(self.data.qpos[self.ARM_QPOS])
        return {"image": image, "state": joints}

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.time = 0.0
        self.t = 0
        self.success_streak = 0

        if self.randomize:
            self._randomize_domain()
        mat_x, mat_y = self._mat_bounds()

        margin = self.ROLL_RADIUS + self.MAT_EDGE_MARGIN
        while True:
            roll_xy = self.rng.uniform(
                [mat_x[0] + margin, mat_y[0] + margin],
                [mat_x[1] - margin, mat_y[1] - margin],
            )
            if np.linalg.norm(roll_xy - self.ARM_REST_XY) >= self.ROLL_RADIUS + self.MIN_ROLL_ARM_CLEARANCE:
                break
        self.model.body(self.ROLL_BODY).pos[:2] = roll_xy

        while True:
            ball_xy = self.rng.uniform(
                [mat_x[0] + self.BALL_RADIUS, mat_y[0] + self.BALL_RADIUS],
                [mat_x[1] - self.BALL_RADIUS, mat_y[1] - self.BALL_RADIUS],
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
