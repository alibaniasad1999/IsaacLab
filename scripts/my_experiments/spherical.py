"""
Step 2:
Two DynamicCapsules connected with a Spherical Joint.

Run:

    ./python.sh spherical.py

or with IsaacLab:

    ./isaaclab.sh -p scripts/my_experiments/spherical.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import time

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCapsule

# Import USD physics after SimulationApp starts
from pxr import UsdPhysics, Gf, Sdf


# ------------------------------------------------------------
# Create world
# ------------------------------------------------------------
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane(z_position=0.0)


# ------------------------------------------------------------
# Capsule settings
# ------------------------------------------------------------
CAPSULE_HEIGHT = 0.6
CAPSULE_RADIUS = 0.08
CAPSULE_MASS = 0.2


# ------------------------------------------------------------
# Create capsule 1
# ------------------------------------------------------------
capsule_1 = world.scene.add(
    DynamicCapsule(
        prim_path="/World/capsule1",
        name="capsule1",
        position=np.array([0.0, 0.0, 2.5]),
        radius=CAPSULE_RADIUS,
        height=CAPSULE_HEIGHT,
        color=np.array([0.1, 0.4, 1.0]),
        mass=CAPSULE_MASS,
    )
)


# ------------------------------------------------------------
# Create capsule 2
# Put it below capsule 1
# ------------------------------------------------------------
capsule_2 = world.scene.add(
    DynamicCapsule(
        prim_path="/World/capsule2",
        name="capsule2",
        position=np.array([0.0, 0.0, 1.9]),
        radius=CAPSULE_RADIUS,
        height=CAPSULE_HEIGHT,
        color=np.array([0.8, 0.3, 1.0]),
        mass=CAPSULE_MASS,
    )
)


# ------------------------------------------------------------
# Add Spherical Joint between capsule1 and capsule2
# ------------------------------------------------------------
stage = world.stage

joint = UsdPhysics.SphericalJoint.Define(
    stage,
    "/World/spherical_joint_capsule1_capsule2"
)

# Connect body 0 to capsule1
joint.CreateBody0Rel().SetTargets([
    Sdf.Path("/World/capsule1")
])

# Connect body 1 to capsule2
joint.CreateBody1Rel().SetTargets([
    Sdf.Path("/World/capsule2")
])

# Joint point on capsule1:
# bottom of capsule1
joint.CreateLocalPos0Attr().Set(
    Gf.Vec3f(0.0, 0.0, -CAPSULE_HEIGHT / 2.0)
)

# Joint point on capsule2:
# top of capsule2
joint.CreateLocalPos1Attr().Set(
    Gf.Vec3f(0.0, 0.0, CAPSULE_HEIGHT / 2.0)
)

# Neighboring connected bodies should not collide with each other
joint.CreateCollisionEnabledAttr().Set(False)

print("Spherical joint created between capsule1 and capsule2.")


# ------------------------------------------------------------
# Start simulation
# ------------------------------------------------------------
world.reset()

try:
    while simulation_app.is_running():
        world.step(render=True)

        pos_1, quat_1 = capsule_1.get_world_pose()
        pos_2, quat_2 = capsule_2.get_world_pose()

        print(
            f"capsule1 z: {pos_1[2]:.3f} | "
            f"capsule2 z: {pos_2[2]:.3f} | "
            f"time: {time.time():.2f}"
        )

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    simulation_app.close()
