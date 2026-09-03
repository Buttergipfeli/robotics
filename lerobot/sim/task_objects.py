import math
from dataclasses import dataclass

import mujoco


@dataclass
class StressBall:
    diameter: float = 0.05
    mass: float = 0.05
    rgba: tuple = (0.15, 0.35, 0.95, 1.0)
    pos: tuple = (0.22, 0.06, 0.016)

    def attach(self, spec):
        body = spec.worldbody.add_body(name="stress_ball", pos=list(self.pos))
        body.add_freejoint()
        body.add_geom(
            name="stress_ball",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.diameter / 2, 0, 0],
            mass=self.mass,
            rgba=list(self.rgba),
            condim=4,
        )

    def qpos0(self):
        return [*self.pos, 1, 0, 0, 0]


@dataclass
class Basket:
    diameter: float = 0.15
    height: float = 0.10
    wall_thickness: float = 0.004
    floor_thickness: float = 0.008
    segments: int = 16
    rgba: tuple = (0.92, 0.92, 0.92, 1.0)
    pos: tuple = (0.28, -0.07, -0.009)

    def attach(self, spec):
        body = spec.worldbody.add_body(name="basket", pos=list(self.pos))
        radius = self.diameter / 2
        body.add_geom(
            name="basket_floor",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, self.floor_thickness / 2, 0],
            pos=[0, 0, self.floor_thickness / 2],
            rgba=list(self.rgba),
        )
        wall_radius = radius - self.wall_thickness / 2
        segment_width = 2 * wall_radius * math.tan(math.pi / self.segments) * 1.05
        for i in range(self.segments):
            angle = 2 * math.pi * i / self.segments
            body.add_geom(
                name=f"basket_wall_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.wall_thickness / 2, segment_width / 2, self.height / 2],
                pos=[
                    wall_radius * math.cos(angle),
                    wall_radius * math.sin(angle),
                    self.height / 2,
                ],
                quat=[math.cos(angle / 2), 0, 0, math.sin(angle / 2)],
                rgba=list(self.rgba),
            )
