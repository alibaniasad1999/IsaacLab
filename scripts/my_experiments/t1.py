"""
Capsule-chain cable, total length = 1 meter.

Run:

    ./isaaclab.sh -p scripts/my_experiments/t1.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import time

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCapsule

from pxr import UsdPhysics, Gf, Sdf, UsdGeom

try:
    from pxr import PhysxSchema
    HAS_PHYSX_SCHEMA = True
except Exception:
    HAS_PHYSX_SCHEMA = False


# -------------------------------------------------------------
# Cable settings
# -------------------------------------------------------------
NUM_LINKS = 60
TOTAL_CABLE_LENGTH = 1.0          # meters
LINK_RADIUS = 0.004               # 4 mm — thin cable

# Capsule total length along axis = height + 2*radius.
# We want adjacent capsule *tips* to meet at the joint anchor,
# so center-to-center spacing equals capsule total length.
SEGMENT_SPACING = TOTAL_CABLE_LENGTH / NUM_LINKS
LINK_HEIGHT = max(SEGMENT_SPACING - 2.0 * LINK_RADIUS, 1e-4)

# Mass
TOTAL_CABLE_MASS = 0.1            # kg (100 g for a 1 m thin cable)
LINK_MASS = TOTAL_CABLE_MASS / NUM_LINKS

ANCHOR_Z = 1.5                    # top fixed point height

# Cable bending stiffness via spherical-joint cone limit.
# Smaller angle => stiffer cable.
CONE_LIMIT_DEG = 8.0
TWIST_LIMIT_DEG = 5.0

# Damping makes the chain behave like a real cable instead of
# a frictionless pendulum chain.
LINEAR_DAMPING = 0.2
ANGULAR_DAMPING = 1.0
JOINT_DAMPING = 0.05
JOINT_STIFFNESS = 0.0             # purely damped joint, limit handles stiffness

SOLVER_POSITION_ITERATIONS = 64
SOLVER_VELOCITY_ITERATIONS = 8

# Continuous collision detection — important for thin/fast bodies
ENABLE_CCD = True


# -------------------------------------------------------------
# World
# -------------------------------------------------------------
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane(z_position=0.0)

stage = world.stage

# Scene-wide PhysX settings
if HAS_PHYSX_SCHEMA:
    physics_scene_path = "/physicsScene"
    physics_scene_prim = stage.GetPrimAtPath(physics_scene_path)
    if physics_scene_prim and physics_scene_prim.IsValid():
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
        physx_scene.CreateEnableCCDAttr().Set(ENABLE_CCD)
        physx_scene.CreateSolverTypeAttr().Set("TGS")


# -------------------------------------------------------------
# Create one capsule
# -------------------------------------------------------------
def create_capsule(index):
    # Top capsule's top tip at ANCHOR_Z, then stack downward by SEGMENT_SPACING.
    center_z = ANCHOR_Z - LINK_HEIGHT / 2.0 - LINK_RADIUS - index * SEGMENT_SPACING

    capsule = world.scene.add(
        DynamicCapsule(
            prim_path=f"/World/capsule_{index}",
            name=f"capsule_{index}",
            position=np.array([0.0, 0.0, center_z]),
            radius=LINK_RADIUS,
            height=LINK_HEIGHT,
            color=np.array([0.05, 0.05, 0.05]),
            mass=LINK_MASS,
        )
    )

    prim = stage.GetPrimAtPath(f"/World/capsule_{index}")

    # Damping + CCD + solver iterations on each rigid body
    if HAS_PHYSX_SCHEMA:
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.CreateSolverPositionIterationCountAttr().Set(SOLVER_POSITION_ITERATIONS)
        physx_rb.CreateSolverVelocityIterationCountAttr().Set(SOLVER_VELOCITY_ITERATIONS)
        physx_rb.CreateLinearDampingAttr().Set(LINEAR_DAMPING)
        physx_rb.CreateAngularDampingAttr().Set(ANGULAR_DAMPING)
        physx_rb.CreateEnableCCDAttr().Set(ENABLE_CCD)
        # Sleep thresholds — let small motions still register
        physx_rb.CreateSleepThresholdAttr().Set(1e-5)
        physx_rb.CreateStabilizationThresholdAttr().Set(1e-6)

    return capsule


# -------------------------------------------------------------
# D6 joint (3 rotational DOFs with cone+twist limits, all
# translation locked) between capsule_i and capsule_i+1.
# Spherical with cone limit only constrains the swing axis;
# a D6 joint gives a true bend-stiffness + twist-stiffness cable.
# -------------------------------------------------------------
def create_link_joint(index):
    joint_path = f"/World/cable_joint_{index}_{index + 1}"
    joint = UsdPhysics.Joint.Define(stage, joint_path)

    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/capsule_{index}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/capsule_{index + 1}")])

    # Bottom tip of upper capsule
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    # Top tip of lower capsule
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2.0 + LINK_RADIUS)))

    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)

    prim = joint.GetPrim()

    # Lock all 3 translational DOFs
    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr().Set(1.0)   # low > high => locked
        limit.CreateHighAttr().Set(-1.0)

    # Cone-limit the two swing axes
    for axis in ("rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr().Set(-CONE_LIMIT_DEG)
        limit.CreateHighAttr().Set(CONE_LIMIT_DEG)

    # Twist limit
    twist_limit = UsdPhysics.LimitAPI.Apply(prim, "rotX")
    twist_limit.CreateLowAttr().Set(-TWIST_LIMIT_DEG)
    twist_limit.CreateHighAttr().Set(TWIST_LIMIT_DEG)

    # Damping on each rotational DOF — gives the cable its viscous feel
    for axis in ("rotX", "rotY", "rotZ"):
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set("force")
        drive.CreateDampingAttr().Set(JOINT_DAMPING)
        drive.CreateStiffnessAttr().Set(JOINT_STIFFNESS)
        drive.CreateMaxForceAttr().Set(1e6)


# -------------------------------------------------------------
# Fixed joint: pin capsule_0 to world
# -------------------------------------------------------------
def create_fixed_joint_to_world():
    fixed_joint = UsdPhysics.FixedJoint.Define(
        stage,
        "/World/fixed_joint_capsule_0_to_world"
    )

    # Body0 empty = world
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/capsule_0")])

    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, ANCHOR_Z))
    # Top tip of capsule_0
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2.0 + LINK_RADIUS)))


# -------------------------------------------------------------
# Build cable
# -------------------------------------------------------------
print("Creating capsules...")
capsules = [create_capsule(i) for i in range(NUM_LINKS)]

print("Creating fixed joint...")
create_fixed_joint_to_world()

print("Creating link joints...")
for i in range(NUM_LINKS - 1):
    create_link_joint(i)

print("")
print("Cable created.")
print(f"  links            : {NUM_LINKS}")
print(f"  total length     : {TOTAL_CABLE_LENGTH} m")
print(f"  segment spacing  : {SEGMENT_SPACING * 1000:.2f} mm")
print(f"  capsule height   : {LINK_HEIGHT * 1000:.2f} mm")
print(f"  capsule radius   : {LINK_RADIUS * 1000:.2f} mm")
print(f"  link mass        : {LINK_MASS * 1000:.3f} g")
print(f"  cone limit       : {CONE_LIMIT_DEG} deg")
print("")


# -------------------------------------------------------------
# Start simulation
# -------------------------------------------------------------
world.reset()

# Small sideways nudge at the bottom link so the cable swings
try:
    capsules[-1].set_linear_velocity(np.array([0.3, 0.0, 0.0]))
except Exception as e:
    print("Could not set initial velocity:", e)


try:
    step_count = 0
    while simulation_app.is_running():
        world.step(render=True)
        step_count += 1

        if step_count % 120 == 0:
            pos_top, _ = capsules[0].get_world_pose()
            pos_last, _ = capsules[-1].get_world_pose()
            print(
                f"top z: {pos_top[2]:.4f} | "
                f"tip xyz: {pos_last[0]:+.4f}, {pos_last[1]:+.4f}, {pos_last[2]:.4f} | "
                f"t: {time.time():.1f}"
            )

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    simulation_app.close()
