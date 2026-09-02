import math
from dataclasses import dataclass

import mujoco


@dataclass
class WristCamera:
    wrist_surface_y: float = 0.024
    mount_hole_front_z: float = -0.03
    mount_height: float = 0.023
    tilt_deg: float = 30.0
    plate_size: float = 0.045
    plate_thickness: float = 0.004
    lens_length: float = 0.015
    lens_radius: float = 0.008
    fovy: float = 70.0
    show_mount: bool = True
    show_view_marker: bool = False

    def attach(self, gripper):
        tilt = math.radians(-self.tilt_deg)
        quat = [math.cos(tilt / 2), math.sin(tilt / 2), 0, 0]

        riser_top = [
            0,
            self.wrist_surface_y + self.mount_height,
            self.mount_hole_front_z,
        ]
        plate_up = [0, math.cos(tilt), math.sin(tilt)]
        plate_pos = [
            riser_top[i] + plate_up[i] * self.plate_size / 2 for i in range(3)
        ]
        view_dir = [0, math.sin(tilt), -math.cos(tilt)]

        def along_view(dist):
            return [plate_pos[i] + view_dir[i] * dist for i in range(3)]

        lens_pos = along_view(self.plate_thickness / 2 + self.lens_length / 2)
        cam_pos = along_view(self.plate_thickness / 2 + self.lens_length + 0.001)

        if self.show_mount:
            self._add_mount_geoms(gripper, quat, plate_pos, lens_pos)
        if self.show_view_marker:
            self._add_view_marker(gripper, quat, cam_pos, view_dir)
        gripper.add_camera(
            name="wrist",
            pos=cam_pos,
            quat=quat,
            fovy=self.fovy,
        )

    def _add_view_marker(self, gripper, quat, cam_pos, view_dir):
        def along(dist):
            return [cam_pos[i] + view_dir[i] * dist for i in range(3)]

        gripper.add_geom(
            name="wrist_cam_marker_ring",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.lens_radius + 0.001, 0.0005],
            pos=cam_pos,
            quat=quat,
            rgba=[1, 0.15, 0.15, 1],
            contype=0,
            conaffinity=0,
        )
        gripper.add_geom(
            name="wrist_cam_marker_dot",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.002],
            pos=along(0.001),
            rgba=[1, 1, 1, 1],
            contype=0,
            conaffinity=0,
        )
        gripper.add_geom(
            name="wrist_cam_marker_ray",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.0012, 0.03],
            pos=along(0.031),
            quat=quat,
            rgba=[1, 0.15, 0.15, 0.6],
            contype=0,
            conaffinity=0,
        )
        gripper.add_geom(
            name="wrist_cam_marker_tip",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.003],
            pos=along(0.061),
            rgba=[1, 0.15, 0.15, 1],
            contype=0,
            conaffinity=0,
        )

    def _add_mount_geoms(self, gripper, quat, plate_pos, lens_pos):
        gripper.add_geom(
            name="wrist_cam_mount",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.plate_size / 2, self.mount_height / 2, self.plate_thickness / 2],
            pos=[0, self.wrist_surface_y + self.mount_height / 2, self.mount_hole_front_z],
            rgba=[0.1, 0.1, 0.1, 1],
            contype=0,
            conaffinity=0,
        )
        gripper.add_geom(
            name="wrist_cam_plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.plate_size / 2, self.plate_size / 2, self.plate_thickness / 2],
            pos=plate_pos,
            quat=quat,
            rgba=[0.1, 0.1, 0.1, 1],
            contype=0,
            conaffinity=0,
        )
        gripper.add_geom(
            name="wrist_cam_lens",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.lens_radius, self.lens_length / 2],
            pos=lens_pos,
            quat=quat,
            rgba=[0.05, 0.05, 0.05, 1],
            contype=0,
            conaffinity=0,
        )
