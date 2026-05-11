"""
Flexible cable connected to rigid connectors in IsaacSim.

A capsule-chain cable (1 meter) suspended from a fixed rigid connector
(top) with a weighted rigid connector (bottom), plus an obstacle to
demonstrate realistic contact and flexibility.

run:
    conda activate env_isaaclab
    python scripts/my_experiments/cable.py
"""

from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import time

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCapsule, DynamicCuboid

from pxr import UsdPhysics, Gf, Sdf, PhysxSchema

# ----------------------------------------------------------
# Cable Settings
# ----------------------------------------------------------
NUM_LINKS = 60
TOTAL_CABLE_LENGTH = 1.0
LINK_RADIUS = 4E-3

# Capsule total length along axis = height + 2*radius.
# We want adjacent capsule *tips* to meet at the joint anchor,
# so center-to-center spacing equals capsule total length.
SEGMENT_SPACING = TOTAL_CABLE_LENGTH / NUM_LINKS
LINK_HEIGHT = max(SEGMENT_SPACING - 2.0 * LINK_RADIUS, 1e-4)

# Mass
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

SOLVER_POSITION_ITERATIONS = 64
SOLVER_VELOCITY_ITERATIONS = 8

# Continuous collision detection — important for thin/fast bodies
ENABLE_CCD = True

# ----------------------------------------------------------
# Rigid Connector Settings
# ----------------------------------------------------------
TOP_CONNECTOR_SIZE = 0.06       # 6 cm cube
TOP_CONNECTOR_MASS = 5.0
TOP_CONNECTOR_COLOR = np.array([0.2, 0.4, 0.8])   # blue

BOTTOM_CONNECTOR_SIZE = 0.04    # 4 cm cube
BOTTOM_CONNECTOR_MASS = 0.5
BOTTOM_CONNECTOR_COLOR = np.array([0.8, 0.2, 0.2])  # red

# ----------------------------------------------------------
# Obstacle Settings
# ----------------------------------------------------------
OBSTACLE_SIZE = 0.08
OBSTACLE_POSITION = np.array([0.15, 0.0, ANCHOR_Z - TOTAL_CABLE_LENGTH * 0.4])
OBSTACLE_MASS = 50.0
OBSTACLE_COLOR = np.array([0.3, 0.7, 0.3])  # green


# ----------------------------------------------------------
# World
# ----------------------------------------------------------
world = World(stage_units_in_meters=1.0,
              physics_dt=1.0 / 240.0,
              rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane(z_position=0.0)

stage = world.stage

# Scene-wide PhysX settings
physics_scene_path = "/physicsScene"
physics_scene_prim = stage.GetPrimAtPath(physics_scene_path)
if physics_scene_prim and physics_scene_prim.IsValid():
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    physx_scene.CreateEnableCCDAttr().Set(ENABLE_CCD)
    physx_scene.CreateSolverTypeAttr().Set("TGS")


# -------------------------------------------------------------
# Rigid connectors
# -------------------------------------------------------------
def create_top_connector() -> DynamicCuboid:
    """Fixed rigid connector block at the top anchor point."""
    connector = world.scene.add(
        DynamicCuboid(
            prim_path="/World/top_connector",
            name="top_connector",
            position=np.array([0.0, 0.0, ANCHOR_Z + TOP_CONNECTOR_SIZE / 2]),
            size=TOP_CONNECTOR_SIZE,
            mass=TOP_CONNECTOR_MASS,
            color=TOP_CONNECTOR_COLOR,
        )
    )

    # Pin the connector to the world so it stays in place
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/fix_top_connector")
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/top_connector")])
    fixed_joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(0.0, 0.0, ANCHOR_Z + TOP_CONNECTOR_SIZE / 2)
    )
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    return connector


def create_bottom_connector() -> DynamicCuboid:
    """Weighted rigid connector at the cable's free end."""
    last_center_z = (ANCHOR_Z
                     - (NUM_LINKS - 1) * SEGMENT_SPACING
                     - LINK_RADIUS - LINK_HEIGHT / 2)
    bottom_z = last_center_z - LINK_HEIGHT / 2 - LINK_RADIUS - BOTTOM_CONNECTOR_SIZE / 2

    connector = world.scene.add(
        DynamicCuboid(
            prim_path="/World/bottom_connector",
            name="bottom_connector",
            position=np.array([0.0, 0.0, bottom_z]),
            size=BOTTOM_CONNECTOR_SIZE,
            mass=BOTTOM_CONNECTOR_MASS,
            color=BOTTOM_CONNECTOR_COLOR,
        )
    )

    return connector


def create_obstacle() -> DynamicCuboid:
    """Fixed obstacle for the cable to interact with."""
    obstacle = world.scene.add(
        DynamicCuboid(
            prim_path="/World/obstacle",
            name="obstacle",
            position=OBSTACLE_POSITION,
            size=OBSTACLE_SIZE,
            mass=OBSTACLE_MASS,
            color=OBSTACLE_COLOR,
        )
    )

    # Pin obstacle to world
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/fix_obstacle")
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/obstacle")])
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*OBSTACLE_POSITION))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    return obstacle


# -------------------------------------------------------------
# Create one capsule link
# -------------------------------------------------------------
def create_capsule(index: int) -> DynamicCapsule:
    # Top capsule's top tip at ANCHOR_Z, then stack downward by SEGMENT_SPACING.
    center_z = ANCHOR_Z - index * SEGMENT_SPACING - LINK_RADIUS - LINK_HEIGHT / 2
    capsule = world.scene.add(
        DynamicCapsule(
            prim_path=f"/World/capsule_{index}",
            name=f"capsule_{index}",
            position=np.array([0.0, 0.0, center_z]),
            radius=LINK_RADIUS,
            height=LINK_HEIGHT,
            color=np.array([0.05, 0.05, 0.05]),
            mass=LINK_MASS
        )
    )

    prim = stage.GetPrimAtPath(f"/World/capsule_{index}")

    # Damping + CCD + solver iterations on each rigid body
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
# D6 joint between adjacent capsules
# (3 rotational DOFs with cone+twist limits, translation locked)
# -------------------------------------------------------------
def create_link_joint(index: int):
    joint_path = f"/World/cable_joint_{index}_{index+1}"
    joint = UsdPhysics.Joint.Define(stage, joint_path)

    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/capsule_{index}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/capsule_{index+1}")])

    # Bottom tip of upper capsule → top tip of lower capsule
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2.0 + LINK_RADIUS)))

    # Disable collision between joined capsules — they overlap at the anchor point
    joint.CreateCollisionEnabledAttr().Set(False)
    # Use maximal-coordinate solver — more stable for long chains
    joint.CreateExcludeFromArticulationAttr().Set(True)

    # Get the underlying USD prim to apply LimitAPI and DriveAPI schemas
    prim = joint.GetPrim()

    # Lock all 3 translational DOFs
    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        # low > high => locked
        limit.CreateLowAttr().Set(1.0)
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
# Attach cable ends to connectors
# -------------------------------------------------------------
def attach_cable_to_top_connector():
    """D6 joint from top connector to first capsule."""
    joint = UsdPhysics.Joint.Define(stage, "/World/joint_top_connector_to_cable")

    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/top_connector")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/capsule_0")])

    # Bottom face of connector → top tip of first capsule
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -TOP_CONNECTOR_SIZE / 2))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2.0 + LINK_RADIUS)))

    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)

    prim = joint.GetPrim()

    # Lock translation
    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr().Set(1.0)
        limit.CreateHighAttr().Set(-1.0)

    # Allow limited bending at the connector
    for axis in ("rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr().Set(-CONE_LIMIT_DEG)
        limit.CreateHighAttr().Set(CONE_LIMIT_DEG)

    twist_limit = UsdPhysics.LimitAPI.Apply(prim, "rotX")
    twist_limit.CreateLowAttr().Set(-TWIST_LIMIT_DEG)
    twist_limit.CreateHighAttr().Set(TWIST_LIMIT_DEG)


def attach_cable_to_bottom_connector():
    """Fixed joint from last capsule to bottom connector."""
    joint = UsdPhysics.FixedJoint.Define(
        stage, "/World/joint_cable_to_bottom_connector"
    )

    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/capsule_{NUM_LINKS - 1}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/bottom_connector")])

    # Bottom tip of last capsule → top face of bottom connector
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, BOTTOM_CONNECTOR_SIZE / 2))

    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)


# -------------------------------------------------------------
# Build scene
# -------------------------------------------------------------
print("Creating connectors...")
top_connector = create_top_connector()
bottom_connector = create_bottom_connector()

print("Creating obstacle...")
obstacle = create_obstacle()

print("Creating capsules...")
capsules = [create_capsule(i) for i in range(NUM_LINKS)]

print("Attaching cable to top connector...")
attach_cable_to_top_connector()

print("Creating link joints...")
for i in range(NUM_LINKS - 1):
    create_link_joint(i)

print("Attaching cable to bottom connector...")
attach_cable_to_bottom_connector()

print("")
print("Cable scene created.")
print(f"  links            : {NUM_LINKS}")
print(f"  total length     : {TOTAL_CABLE_LENGTH} m")
print(f"  segment spacing  : {SEGMENT_SPACING * 1000:.2f} mm")
print(f"  capsule height   : {LINK_HEIGHT * 1000:.2f} mm")
print(f"  capsule radius   : {LINK_RADIUS * 1000:.2f} mm")
print(f"  link mass        : {LINK_MASS * 1000:.3f} g")
print(f"  cone limit       : {CONE_LIMIT_DEG} deg")
print(f"  top connector    : {TOP_CONNECTOR_SIZE * 100:.0f} cm cube, {TOP_CONNECTOR_MASS} kg (fixed)")
print(f"  bottom connector : {BOTTOM_CONNECTOR_SIZE * 100:.0f} cm cube, {BOTTOM_CONNECTOR_MASS} kg (free)")
print(f"  obstacle at      : {OBSTACLE_POSITION}")
print("")


# -------------------------------------------------------------
# Start simulation
# -------------------------------------------------------------
world.reset()

# Nudge the bottom connector sideways toward the obstacle
try:
    bottom_connector.set_linear_velocity(np.array([1.5, 0.0, 0.0]))
except Exception as e:
    print("Could not set initial velocity:", e)


try:
    step_count = 0
    while simulation_app.is_running():
        world.step(render=True)
        step_count += 1

        if step_count % 120 == 0:
            pos_top, _ = capsules[0].get_world_pose()
            pos_bot, _ = bottom_connector.get_world_pose()
            print(
                f"top z: {pos_top[2]:.4f} | "
                f"connector xyz: {pos_bot[0]:+.4f}, "
                f"{pos_bot[1]:+.4f}, {pos_bot[2]:.4f} | "
                f"t: {time.time():.1f}"
            )

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    simulation_app.close()
