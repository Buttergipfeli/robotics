import numpy as np
import mujoco


class ScriptedExpert:
    TCP_SITE = "gripperframe"
    GRIPPER_BODY = "gripper"
    IK_DOFS = [0, 1, 2, 3]
    WRIST_ROLL_IDX = 4
    WRIST_ROLL_REST = -1.571
    GRIPPER_IDX = 5

    GRIPPER_OPEN = 1.3
    GRIPPER_CLOSED = -0.05

    HOVER_HEIGHT = 0.10
    GRASP_TCP_OFFSET_Z = 0.005
    LIFT_TCP_Z = 0.20
    RELEASE_CLEARANCE = 0.025
    RELEASE_POS_TOL = 0.01
    MAX_RETRIES = 2

    POS_TOL = 0.015
    MAX_DELTA = 0.12
    MAX_DELTA_FINE = 0.05
    FINE_PHASES = ("descend", "grasp", "lift")
    PHASE_TIMEOUT = 60
    GRASP_WAIT = 10
    GRIPPER_MAX_DELTA = 0.15
    GRASP_LATERAL_OFFSET = -0.02
    RELEASE_WAIT = 6

    AXIS_LEN = 0.08
    AXIS_WEIGHT = 0.1
    IK_ITERS = 80
    IK_DAMPING = 1e-4

    def __init__(self, env, noise_std=0.5, seed=None):
        self.env = env
        self.model = env.model
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.ik_data = mujoco.MjData(self.model)
        self.site_id = self.model.site(self.TCP_SITE).id
        self.grip_body_id = self.model.body(self.GRIPPER_BODY).id
        ranges = np.array([self.model.joint(i).range for i in self.IK_DOFS])
        self.ik_lo = ranges[:, 0] + 0.02
        self.ik_hi = ranges[:, 1] - 0.02
        self.reset()

    def reset(self):
        self.phase = "raise"
        self.phase_ticks = 0
        self.wait = 0
        self.failed = False
        self.cmd = None
        self.grasp_target = None
        self.release_target = None
        self.retries = 0
        self.best_dist = np.inf
        self.stall = 0

    def act(self):
        env = self.env
        if self.cmd is None:
            self.cmd = env.data.qpos[:6].copy()

        ball = env.data.qpos[env.BALL_POS].copy()
        roll = env.model.body(env.ROLL_BODY).pos.copy()
        roll_top_z = roll[2] + env.ROLL_HEIGHT
        tcp = env.data.site_xpos[self.site_id]

        self.phase_ticks += 1
        if self.phase_ticks > self.PHASE_TIMEOUT:
            self.failed = True

        if self.phase == "raise":
            target = np.array([0.20, 0.0, 0.20])
            gripper = self.GRIPPER_OPEN
            if tcp[2] > 0.17:
                self._next("approach")
        elif self.phase == "approach":
            target = np.array([ball[0], ball[1], ball[2] + self.HOVER_HEIGHT])
            gripper = self.GRIPPER_OPEN
            if np.linalg.norm(tcp - target) < self.POS_TOL:
                rot = env.data.xmat[self.grip_body_id].reshape(3, 3)
                jaw_dir = rot[:2, 0] / (np.linalg.norm(rot[:2, 0]) + 1e-9)
                self.grasp_target = np.array([
                    ball[0] + jaw_dir[0] * self.GRASP_LATERAL_OFFSET,
                    ball[1] + jaw_dir[1] * self.GRASP_LATERAL_OFFSET,
                    ball[2] + self.GRASP_TCP_OFFSET_Z,
                ])
                self._next("descend")
        elif self.phase == "descend":
            target = self.grasp_target
            gripper = self.GRIPPER_OPEN
            dist = np.linalg.norm(tcp - target)
            if dist < self.POS_TOL:
                self._next("grasp")
            else:
                if dist < self.best_dist - 0.002:
                    self.best_dist = dist
                    self.stall = 0
                else:
                    self.stall += 1
                if self.stall >= 10:
                    if np.linalg.norm(tcp - ball) < 0.045:
                        self._next("grasp")
                    elif self.retries < self.MAX_RETRIES:
                        self.retries += 1
                        self._next("approach")
                    else:
                        self.failed = True
        elif self.phase == "grasp":
            target = self.grasp_target
            gripper = self.GRIPPER_CLOSED
            self.wait += 1
            if self.wait >= self.GRASP_WAIT:
                if env.data.eq_active[env.grasp_eq_id]:
                    self._next("lift")
                elif self.retries < self.MAX_RETRIES:
                    self.retries += 1
                    self._next("approach")
                else:
                    self.failed = True
        elif self.phase == "lift":
            target = np.array([self.grasp_target[0], self.grasp_target[1], self.LIFT_TCP_Z])
            gripper = self.GRIPPER_CLOSED
            if ball[2] > 0.12:
                self._next("move")
            elif self.phase_ticks > 15 and ball[2] < 0.06:
                self.failed = True
        elif self.phase == "move":
            desired_ball = np.array([roll[0], roll[1], roll_top_z + env.BALL_RADIUS + self.RELEASE_CLEARANCE])
            target = tcp + (desired_ball - ball)
            gripper = self.GRIPPER_CLOSED
            if np.linalg.norm(ball[:2] - desired_ball[:2]) < self.RELEASE_POS_TOL:
                self.release_target = tcp + (desired_ball - ball)
                self._next("release")
        else:
            target = self.release_target
            gripper = self.GRIPPER_OPEN
            self.wait += 1

        arm_target = self._ik(target)
        full_target = np.append(arm_target, gripper)
        max_delta = self.MAX_DELTA_FINE if self.phase in self.FINE_PHASES else self.MAX_DELTA
        delta = np.clip(full_target - self.cmd, -max_delta, max_delta)
        delta[self.GRIPPER_IDX] = np.clip(
            full_target[self.GRIPPER_IDX] - self.cmd[self.GRIPPER_IDX],
            -self.GRIPPER_MAX_DELTA,
            self.GRIPPER_MAX_DELTA,
        )
        self.cmd = self.cmd + delta

        action = env.rad_to_norm(self.cmd)
        noise = self.rng.normal(0, self.noise_std, 6)
        noise[self.GRIPPER_IDX] = 0
        return action + noise

    def done(self):
        return self.phase == "release" and self.wait >= self.RELEASE_WAIT

    def _next(self, phase):
        self.phase = phase
        self.phase_ticks = 0
        self.wait = 0
        self.best_dist = np.inf
        self.stall = 0

    def _ik(self, target):
        pan = float(np.arctan2(target[1], target[0]))
        seeds = [
            self.env.data.qpos[:5].copy(),
            np.array([pan, -0.6, 1.0, 0.8, self.WRIST_ROLL_REST]),
            np.array([pan, 0.2, 1.4, 0.2, self.WRIST_ROLL_REST]),
        ]
        best = None
        best_err = np.inf
        for seed in seeds:
            q, err = self._ik_from(seed, target)
            if err < best_err:
                best, best_err = q, err
            if best_err < 0.002:
                break
        return best

    def _ik_from(self, seed, target):
        d = self.ik_data
        d.qpos[:5] = seed
        d.qpos[self.WRIST_ROLL_IDX] = self.WRIST_ROLL_REST
        jac1 = np.zeros((3, self.model.nv))
        jac2 = np.zeros((3, self.model.nv))
        for _ in range(self.IK_ITERS):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            site = d.site_xpos[self.site_id]
            rot = d.xmat[self.grip_body_id].reshape(3, 3)
            p2 = site + rot[:, 2] * self.AXIS_LEN
            err = np.concatenate([
                target - site,
                (target + [0, 0, self.AXIS_LEN] - p2) * self.AXIS_WEIGHT,
            ])
            if np.linalg.norm(err[:3]) < 0.002:
                break
            mujoco.mj_jacSite(self.model, d, jac1, None, self.site_id)
            mujoco.mj_jac(self.model, d, jac2, None, p2, self.grip_body_id)
            full_jac = np.vstack([
                jac1[:, self.IK_DOFS],
                jac2[:, self.IK_DOFS] * self.AXIS_WEIGHT,
            ])
            dq = full_jac.T @ np.linalg.solve(
                full_jac @ full_jac.T + self.IK_DAMPING * np.eye(6), err
            )
            new_q = d.qpos[self.IK_DOFS] + np.clip(dq, -0.2, 0.2)
            d.qpos[self.IK_DOFS] = np.clip(new_q, self.ik_lo, self.ik_hi)
        mujoco.mj_kinematics(self.model, d)
        pos_err = float(np.linalg.norm(target - d.site_xpos[self.site_id]))
        arm = d.qpos[:5].copy()
        arm[self.WRIST_ROLL_IDX] = self.WRIST_ROLL_REST
        return arm, pos_err
