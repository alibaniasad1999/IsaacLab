"""
Flexible cable connected to rigid connectors in IsaacSim.
Automatically records 10 seconds of simulation and saves key frames for report.

run:
    conda activate env_isaaclab
    python scripts/my_experiments/cable_v1.py
"""

from pathlib import Path
from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import time
import cv2

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCapsule, DynamicCuboid

from pxr import UsdPhysics, Gf, Sdf, PhysxSchema
import omni.replicator.core as rep

# ----------------------------------------------------------
# Cable Settings
# ----------------------------------------------------------
NUM_LINKS = 60
TOTAL_CABLE_LENGTH = 1.0
LINK_RADIUS = 4E-3

SEGMENT_SPACING = TOTAL_CABLE_LENGTH / NUM_LINKS
LINK_HEIGHT = max(SEGMENT_SPACING - 2.0 * LINK_RADIUS, 1e-4)

# Mass
TOTAL_CABLE_MASS = 1.0
LINK_MASS = TOTAL_CABLE_MASS / NUM_LINKS

ANCHOR_Z = 2.0

# Cable bending stiffness
CONE_LIMIT_DEG = 8.0
TWIST_LIMIT_DEG = 5.8

# Damping
LINEAR_DAMPING = 0.2
ANGULAR_DAMPING = 1.0
JOINT_DAMPING = 0.05
JOINT_STIFFNESS = 0.0

SOLVER_POSITION_ITERATIONS = 64
SOLVER_VELOCITY_ITERATIONS = 8

ENABLE_CCD = True

# ----------------------------------------------------------
# Rigid Connector Settings
# ----------------------------------------------------------
TOP_CONNECTOR_SIZE = 0.06
TOP_CONNECTOR_MASS = 5.0
TOP_CONNECTOR_COLOR = np.array([0.2, 0.4, 0.8])

BOTTOM_CONNECTOR_SIZE = 0.04
BOTTOM_CONNECTOR_MASS = 0.5
BOTTOM_CONNECTOR_COLOR = np.array([0.8, 0.2, 0.2])

# ----------------------------------------------------------
# Obstacle Settings
# ----------------------------------------------------------
OBSTACLE_SIZE = 0.08
OBSTACLE_POSITION = np.array([0.15, 0.0, ANCHOR_Z - TOTAL_CABLE_LENGTH * 0.4])
OBSTACLE_MASS = 50.0
OBSTACLE_COLOR = np.array([0.3, 0.7, 0.3])

# ----------------------------------------------------------
# Recording Settings
# ----------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "cable_output"
OUTPUT_DIR.mkdir(exist_ok=True)

PHYSICS_DT = 1.0 / 240.0
RENDER_DT = 1.0 / 60.0

RECORD_SECONDS = 10
VIDEO_FPS = 30
STEPS_PER_VIDEO_FRAME = int(1.0 / (PHYSICS_DT * VIDEO_FPS))  # = 8
TOTAL_RECORD_STEPS = int(RECORD_SECONDS / PHYSICS_DT)         # = 2400
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_PATH = OUTPUT_DIR / "cable_simulation.mp4"

# Key frames to save as PNG for the report (in seconds)
KEY_FRAME_TIMES = [0.0, 2.0, 5.0, 10.0]
KEY_FRAME_STEPS = [int(t / PHYSICS_DT) for t in KEY_FRAME_TIMES]

WARMUP_STEPS = 30  # let renderer initialize before capturing


# ----------------------------------------------------------
# World
# ----------------------------------------------------------
world = World(stage_units_in_meters=1.0,
              physics_dt=PHYSICS_DT,
              rendering_dt=RENDER_DT)
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

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/fix_obstacle")
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/obstacle")])
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*OBSTACLE_POSITION))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    return obstacle


# -------------------------------------------------------------
# Create one capsule link
# -------------------------------------------------------------
def create_capsule(index: int) -> DynamicCapsule:
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

    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateSolverPositionIterationCountAttr().Set(SOLVER_POSITION_ITERATIONS)
    physx_rb.CreateSolverVelocityIterationCountAttr().Set(SOLVER_VELOCITY_ITERATIONS)
    physx_rb.CreateLinearDampingAttr().Set(LINEAR_DAMPING)
    physx_rb.CreateAngularDampingAttr().Set(ANGULAR_DAMPING)
    physx_rb.CreateEnableCCDAttr().Set(ENABLE_CCD)
    physx_rb.CreateSleepThresholdAttr().Set(1e-5)
    physx_rb.CreateStabilizationThresholdAttr().Set(1e-6)

    return capsule


# -------------------------------------------------------------
# D6 joint between adjacent capsules
# -------------------------------------------------------------
def create_link_joint(index: int):
    joint_path = f"/World/cable_joint_{index}_{index+1}"
    joint = UsdPhysics.Joint.Define(stage, joint_path)

    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/capsule_{index}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/capsule_{index+1}")])

    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2.0 + LINK_RADIUS)))

    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)

    prim = joint.GetPrim()

    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr().Set(1.0)
        limit.CreateHighAttr().Set(-1.0)

    for axis in ("rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr().Set(-CONE_LIMIT_DEG)
        limit.CreateHighAttr().Set(CONE_LIMIT_DEG)

    twist_limit = UsdPhysics.LimitAPI.Apply(prim, "rotX")
    twist_limit.CreateLowAttr().Set(-TWIST_LIMIT_DEG)
    twist_limit.CreateHighAttr().Set(TWIST_LIMIT_DEG)

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
    joint = UsdPhysics.Joint.Define(stage, "/World/joint_top_connector_to_cable")

    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/top_connector")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/capsule_0")])

    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -TOP_CONNECTOR_SIZE / 2))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2.0 + LINK_RADIUS)))

    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)

    prim = joint.GetPrim()

    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr().Set(1.0)
        limit.CreateHighAttr().Set(-1.0)

    for axis in ("rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr().Set(-CONE_LIMIT_DEG)
        limit.CreateHighAttr().Set(CONE_LIMIT_DEG)

    twist_limit = UsdPhysics.LimitAPI.Apply(prim, "rotX")
    twist_limit.CreateLowAttr().Set(-TWIST_LIMIT_DEG)
    twist_limit.CreateHighAttr().Set(TWIST_LIMIT_DEG)


def attach_cable_to_bottom_connector():
    joint = UsdPhysics.FixedJoint.Define(
        stage, "/World/joint_cable_to_bottom_connector"
    )

    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/capsule_{NUM_LINKS - 1}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/bottom_connector")])

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
# Camera setup for recording
# -------------------------------------------------------------
print("Setting up recording camera...")

# Set viewport camera to a good angle
try:
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:
    from omni.isaac.core.utils.viewports import set_camera_view

set_camera_view(
    eye=np.array([1.0, 0.8, 2.2]),
    target=np.array([0.0, 0.0, 1.5]),
)

# Create render product from viewport camera
render_product = rep.create.render_product("/OmniverseKit_Persp", (VIDEO_WIDTH, VIDEO_HEIGHT))
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])


# -------------------------------------------------------------
# Start simulation + recording
# -------------------------------------------------------------
world.reset()

try:
    bottom_connector.set_linear_velocity(np.array([1.5, 0.0, 0.0]))
except Exception as e:
    print("Could not set initial velocity:", e)

# Warm up renderer so annotator produces valid frames
print("Warming up renderer...")
for _ in range(WARMUP_STEPS):
    world.step(render=True)

# Start video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

print(f"Recording {RECORD_SECONDS} seconds of simulation...")
print(f"  Output video : {VIDEO_PATH}")
print(f"  Key frames   : {OUTPUT_DIR}")
print("")

step_count = 0
frames_written = 0
recording_done = False

try:
    while simulation_app.is_running():
        world.step(render=True)
        step_count += 1

        # --- Recording phase ---
        if not recording_done and step_count <= TOTAL_RECORD_STEPS:
            # Capture every N-th step to match VIDEO_FPS
            if step_count % STEPS_PER_VIDEO_FRAME == 0:
                data = rgb_annotator.get_data()
                if data is not None and data.size > 0:
                    frame_rgb = data[:, :, :3]  # drop alpha channel
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                    frames_written += 1

            # Save key frames as PNG for the report
            if step_count in KEY_FRAME_STEPS:
                data = rgb_annotator.get_data()
                if data is not None and data.size > 0:
                    t = step_count * PHYSICS_DT
                    frame_bgr = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR)
                    path = OUTPUT_DIR / f"frame_t{t:.0f}s.png"
                    cv2.imwrite(str(path), frame_bgr)
                    print(f"  Saved key frame: {path.name}  (t = {t:.1f}s)")

            # Finish recording
            if step_count >= TOTAL_RECORD_STEPS:
                video_writer.release()
                recording_done = True
                print("")
                print(f"Recording complete: {frames_written} frames written")
                print(f"  Video: {VIDEO_PATH}")
                print("Simulation continues — close window or Ctrl+C to exit.")
                print("")

        # --- Telemetry ---
        if step_count % 480 == 0:  # every 2 seconds
            sim_time = step_count * PHYSICS_DT
            pos_top, _ = capsules[0].get_world_pose()
            pos_bot, _ = bottom_connector.get_world_pose()
            status = "REC" if not recording_done else "   "
            print(
                f"[{status}] t={sim_time:5.1f}s | "
                f"top z: {pos_top[2]:.4f} | "
                f"connector: {pos_bot[0]:+.4f}, "
                f"{pos_bot[1]:+.4f}, {pos_bot[2]:.4f}"
            )

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    if not recording_done:
        video_writer.release()
        print(f"Partial recording saved: {frames_written} frames")
    simulation_app.close()

print("Done.")
