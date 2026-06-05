"""
Step 3:
Make a 5-link rope from capsules.

Structure:
world fixed point
    |
fixed joint
    |
capsule_0
    |
spherical joint
    |
capsule_1
    |
spherical joint
    |
capsule_2
    |
spherical joint
    |
capsule_3
    |
spherical joint
    |
capsule_4
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import time

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCapsule

# Import USD/PhysX after SimulationApp starts
from pxr import UsdPhysics, Gf, Sdf

# Optional PhysX tuning
try:
    from pxr import PhysxSchema
    HAS_PHYSX_SCHEMA = True
except Exception:
    HAS_PHYSX_SCHEMA = False


# -------------------------------------------------------------
# Create World
# -------------------------------------------------------------
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane(z_position=0.0)

stage = world.stage


# -------------------------------------------------------------
# Cable settings
# -------------------------------------------------------------
NUM_LINKS = 5

# Your requested size:
# 5 cm long, 0.5 cm radius
CAPSULE_HEIGHT = 0.05      # 5 cm
CAPSULE_RADIUS = 0.05     # 0.5 cm

# Start with this. If jittery, try 0.02 or 0.05
CAPSULE_MASS = 0.01        # kg

# Joint bending limit
JOINT_LIMIT_DEG = 30.0

# Solver tuning
SOLVER_POSITION_ITERATIONS = 32
SOLVER_VELOCITY_ITERATIONS = 8

# Rope top anchor height
ANCHOR_Z = 1.2


# -------------------------------------------------------------
# Helper: create one dynamic capsule
# -------------------------------------------------------------
def create_capsule(index):
    """
    Creates one vertical capsule.

    capsule_0 is highest.
    capsule_4 is lowest.
    """

    center_z = ANCHOR_Z - (index + 0.5) * CAPSULE_HEIGHT

    capsule = world.scene.add(
        DynamicCapsule(
            prim_path=f"/World/capsule_{index}",
            name=f"capsule_{index}",
            position=np.array([0.0, 0.0, center_z]),
            radius=CAPSULE_RADIUS,
            height=CAPSULE_HEIGHT,
            color=np.array([0.1 + 0.12 * index, 0.4, 1.0]),
            mass=CAPSULE_MASS,
        )
    )

    prim = stage.GetPrimAtPath(f"/World/capsule_{index}")

    # Optional: improve solver iterations for each rigid body
    if HAS_PHYSX_SCHEMA:
        try:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rb.CreateSolverPositionIterationCountAttr().Set(
                SOLVER_POSITION_ITERATIONS
            )
            physx_rb.CreateSolverVelocityIterationCountAttr().Set(
                SOLVER_VELOCITY_ITERATIONS
            )
        except Exception as e:
            print(f"Could not set solver iterations for capsule_{index}: {e}")

    return capsule


# -------------------------------------------------------------
# Helper: create spherical joint between two neighboring capsules
# -------------------------------------------------------------
def create_spherical_joint(index):
    """
    Connects capsule_index bottom
    to capsule_(index+1) top.
    """

    joint = UsdPhysics.SphericalJoint.Define(
        stage,
        f"/World/spherical_joint_{index}_{index + 1}"
    )

    # Body 0 = upper capsule
    joint.CreateBody0Rel().SetTargets([
        Sdf.Path(f"/World/capsule_{index}")
    ])

    # Body 1 = lower capsule
    joint.CreateBody1Rel().SetTargets([
        Sdf.Path(f"/World/capsule_{index + 1}")
    ])

    # Bottom of upper capsule
    joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(0.0, 0.0, -CAPSULE_HEIGHT / 2.0)
    )

    # Top of lower capsule
    joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(0.0, 0.0, CAPSULE_HEIGHT / 2.0)
    )

    # Do not collide neighboring connected capsules
    joint.CreateCollisionEnabledAttr().Set(False)

    # Cone limit: controls max bending
    joint.CreateAxisAttr("Z")
    joint.CreateConeAngle0LimitAttr().Set(JOINT_LIMIT_DEG)
    joint.CreateConeAngle1LimitAttr().Set(JOINT_LIMIT_DEG)

    print(f"Created spherical joint between capsule_{index} and capsule_{index + 1}")


# -------------------------------------------------------------
# Helper: fix first capsule to the world
# -------------------------------------------------------------
def create_fixed_joint_to_world():
    """
    Pins the top of capsule_0 to a fixed world point.
    """

    fixed_joint = UsdPhysics.FixedJoint.Define(
        stage,
        "/World/fixed_joint_capsule_0_to_world"
    )

    # Body0 empty = world
    # Body1 = capsule_0
    fixed_joint.CreateBody1Rel().SetTargets([
        Sdf.Path("/World/capsule_0")
    ])

    # World anchor point
    fixed_joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(0.0, 0.0, ANCHOR_Z)
    )

    # Top of capsule_0 in capsule_0 local frame
    fixed_joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(0.0, 0.0, CAPSULE_HEIGHT / 2.0)
    )

    print("Created fixed joint between capsule_0 and world.")


# -------------------------------------------------------------
# Build the 5-link rope
# -------------------------------------------------------------
capsules = []

for i in range(NUM_LINKS):
    capsules.append(create_capsule(i))

# Pin first capsule to the world
create_fixed_joint_to_world()

# Connect each neighboring pair
for i in range(NUM_LINKS - 1):
    create_spherical_joint(i)


# -------------------------------------------------------------
# Start simulation
# -------------------------------------------------------------
world.reset()

# Give the last capsule a small sideways velocity so you can see swing
try:
    capsules[-1].set_linear_velocity(np.array([0.25, 0.0, 0.0]))
except Exception:
    try:
        last_prim = stage.GetPrimAtPath(f"/World/capsule_{NUM_LINKS - 1}")
        rb_api = UsdPhysics.RigidBodyAPI(last_prim)
        rb_api.CreateVelocityAttr().Set(Gf.Vec3f(0.25, 0.0, 0.0))
    except Exception as e:
        print("Could not set initial velocity:", e)

print("")
print("5-link rope created.")
print("Expected result: capsule_0 stays fixed, lower capsules hang and swing.")
print("If it jitters: increase mass a little, lower joint limit, or increase solver iterations.")
print("")


try:
    while simulation_app.is_running():
        world.step(render=True)

        pos_top, _ = capsules[0].get_world_pose()
        pos_last, _ = capsules[-1].get_world_pose()

        print(
            f"top z: {pos_top[2]:.3f} | "
            f"last link xyz: "
            f"{pos_last[0]:.3f}, {pos_last[1]:.3f}, {pos_last[2]:.3f} | "
            f"time: {time.time():.2f}"
        )

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    simulation_app.close()
