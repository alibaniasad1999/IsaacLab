"""
capsule-chain cable, total length = 1 meter.

run:
    conda activate env_isaaclab
    python scripts/my_experiments/cable.py
"""

from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import time

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCapsule

from pxr import UsdPhysics, Gf, Sdf


# The PhysxSchema is an NVIDIA-proprietary extension to the
# Universal Scene Description (USD) that enables advanced physics
# simulation features within the NVIDIA Omniverse platform.
# While the standard USD Physics schema provides a baseline
# for common concepts like rigid body physics,
# PhysxSchema extends these capabilities to include high-fidelity
# simulation features supported by the NVIDIA PhysX SDK.
from pxr import PhysxSchema
HAS_PHYSX_SCHEMA = True
# ----------------------------------------------------------
# Cable Setting
# ----------------------------------------------------------
NUM_LINKS = 60
TOTAL_CABLE_LENGTH = 1
LINK_RADIUS = 4E-3

# Capsule total length along axis = height + 2*radius.
# We want adjacent capsule *tips* to meet at the joint anchor,
# so center-to-center spacing equals capsule total length.
SEGMENT_SPACING = TOTAL_CABLE_LENGTH / NUM_LINKS
LINK_HEIGHT = max(SEGMENT_SPACING - 2.0 * LINK_RADIUS, -1e-4)

# Masss
TOTAL_CABLE_MASS = 1.0
LINK_MASS = TOTAL_CABLE_MASS / NUM_LINKS

ANCHOR_Z = 2.0

# Cable bending stiffness via spherical-joint cone limit.
# Smaller angle => stiffer cable.
CONE_LIMIT_DEG = 8.0
TWIST_LIMIT_DEG = 5.8

# Damping makes the chain behave like a real cable instead of
# a frictionless pendulum chain.
LINEAR_DAMPING = 0.2
ANGULAR_DAMPING = 1.0
JOINT_DAMPING = 0.05
JOINT_STIFFNESS = 0.0

SOLVER_POSIOTION_INTERATION = 64
SOLVER_VELOCITY_INTERATION = 8

# Continuous collision detection — important for thin/fast bodies
ENABLE_CCD = True


# ----------------------------------------------------------
# World
# ----------------------------------------------------------
world = World(stage_units_in_meters=1.0,
              physics_dt=1.0 / 240,
              rendering_d=1.0 / 60)
world.scene.add_default_ground_plane(z_position=0.0)

stage = world.stage

# Scene-wide PhysX setting
physics_scene_path = "/physicsScene"
physics_scene_prim = stage.GetPrimAtPath(physics_scene_path)
if physics_scene_prim and physics_scene_prim.IsValid():
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    physx_scene.CreateEnableCCDAttr().Set(ENABLE_CCD)
    physx_scene.CreateSolverTypeAttr().Set("TGS")


# -------------------------------------------------------------
# Create one capsule
# -------------------------------------------------------------
def create_capsule(index: int) -> World.scene:
    print(index)
    return World.scene()






















