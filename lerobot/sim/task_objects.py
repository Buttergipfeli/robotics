import math
from dataclasses import dataclass

import mujoco


@dataclass
class StressBall:
    diameter: float = 0.0668
    mass: float = 0.045
    rgb: tuple = (0.66, 0.86, 1.0)
    emission: float = 0.2
    specular: float = 0.2
    texture_size: int = 64
    pos: tuple = (0.22, 0.06, 0.0244)

    def attach(self, spec):
        texture = spec.add_texture()
        texture.name = "stress_ball"
        texture.type = mujoco.mjtTexture.mjTEXTURE_CUBE
        texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
        texture.rgb1 = list(self.rgb)
        texture.width = self.texture_size
        texture.height = self.texture_size
        material = spec.add_material()
        material.name = "stress_ball"
        material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "stress_ball"
        material.emission = self.emission
        material.specular = self.specular

        body = spec.worldbody.add_body(name="stress_ball", pos=list(self.pos))
        body.add_freejoint()
        body.add_geom(
            name="stress_ball",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.diameter / 2, 0, 0],
            mass=self.mass,
            material="stress_ball",
            condim=6,
            friction=[2.0, 0.05, 0.005],
            solref=[0.04, 1.0],
        )

    def qpos0(self):
        return [*self.pos, 1, 0, 0, 0]


@dataclass
class ToiletRoll:
    outer_diameter: float = 0.121
    core_diameter: float = 0.0446
    height: float = 0.096
    core_thickness: float = 0.001
    segments: int = 20
    paper_rgba: tuple = (0.93, 0.93, 0.95, 1.0)
    core_rgba: tuple = (0.45, 0.40, 0.35, 1.0)
    pos: tuple = (0.28, -0.07, -0.009)

    def attach(self, spec):
        body = spec.worldbody.add_body(name="toilet_roll", pos=list(self.pos))
        outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2

        ring_radius = (outer_radius + core_radius) / 2
        ring_thickness = outer_radius - core_radius
        segment_width = 2 * outer_radius * math.tan(math.pi / self.segments) * 1.05
        for i in range(self.segments):
            angle = 2 * math.pi * i / self.segments
            body.add_geom(
                name=f"roll_paper_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[ring_thickness / 2, segment_width / 2, self.height / 2],
                pos=[
                    ring_radius * math.cos(angle),
                    ring_radius * math.sin(angle),
                    self.height / 2,
                ],
                quat=[math.cos(angle / 2), 0, 0, math.sin(angle / 2)],
                rgba=list(self.paper_rgba),
            )

        core_ring_radius = core_radius - self.core_thickness / 2
        core_segment_width = 2 * core_radius * math.tan(math.pi / self.segments) * 1.05
        for i in range(self.segments):
            angle = 2 * math.pi * i / self.segments
            body.add_geom(
                name=f"roll_core_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.core_thickness / 2, core_segment_width / 2, self.height / 2],
                pos=[
                    core_ring_radius * math.cos(angle),
                    core_ring_radius * math.sin(angle),
                    self.height / 2,
                ],
                quat=[math.cos(angle / 2), 0, 0, math.sin(angle / 2)],
                rgba=list(self.core_rgba),
            )
