"""
Flexible cable with rigid connectors in Isaac Sim.

Aligned with Govoni et al. 2025 (arXiv:2504.13659):

  1. Material-driven parameters: JOINT_STIFFNESS and JOINT_DAMPING are derived
     from Young's modulus + geometry + damping ratio via beam theory
     (K_bend = EI / L_segment).
  2. Cone limit widened (30deg). The soft EI/L spring does the bending work;
     the cone limit is a safety net.
  3. Twist DOF freed (Govoni: "Twisting springs are excluded").
     No twist limit, no rotX drive.
  4. Two experiment modes:
       * "hanging_kick"    -- top fixed, bottom kicked, obstacle present.
       * "both_ends_fixed" -- Govoni-style stability test: top fixed, bottom
                             kinematic, 5 mm step displacement after settling.
  5. Per-step CSV logging of capsule positions + stability monitor that flags
     divergence and writes summary.json. Designed to be driven by
     govoni_sweep.py via environment variables.

Run (defaults: hanging_kick mode, 200 links, soft rubber):
    conda activate env_isaaclab
    python scripts/cable_simulation/cable.py

Run a Govoni-style stability test (no GUI, no video, short):
    CABLE_MODE=both_ends_fixed CABLE_HEADLESS=1 CABLE_RECORD=0 \
        CABLE_NUM_LINKS=10 CABLE_E=1002.6e6 CABLE_PHYSICS_DT=5e-6 \
        CABLE_MAX_TIME=1.0 python cable.py
"""

from pathlib import Path
import os
import math
import json

# ---------------------------------------------------------------
# SimulationApp must be created BEFORE importing any isaacsim modules
# ---------------------------------------------------------------
from isaacsim.simulation_app import SimulationApp

HEADLESS     = os.environ.get("CABLE_HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": HEADLESS})

# ---------------------------------------------------------------
# Imports (after SimulationApp is up)
# ---------------------------------------------------------------
import numpy as np
import csv
import subprocess
import cv2

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCapsule, DynamicCuboid
from pxr import UsdPhysics, Gf, Sdf, PhysxSchema
import omni.replicator.core as rep


# ===============================================================
# 1. CONFIGURATION  (env vars override defaults -- used by sweep)
# ===============================================================

# ---- Cable geometry ----
NUM_LINKS            = int(  os.environ.get("CABLE_NUM_LINKS",   200))
TOTAL_CABLE_LENGTH   = float(os.environ.get("CABLE_LENGTH",      1.0))
LINK_RADIUS          = float(os.environ.get("CABLE_RADIUS",      1.5e-3))  # 1.5 mm
TOTAL_CABLE_MASS     = float(os.environ.get("CABLE_MASS",        1.0))

ANCHOR_Z             = 2.0   # height of top anchor above ground

# ---- Material properties (default: PUR / polyurethane robot cable) ----
YOUNG_MODULUS        = float(os.environ.get("CABLE_E",           30e6))     # Pa  (PUR ~ 30 MPa)
POISSON_RATIO        = float(os.environ.get("CABLE_NU",          0.45))     # PUR
DAMPING_RATIO        = float(os.environ.get("CABLE_ZETA",        0.2))      # fraction of critical damping

# ---- Translational MSD springs (axial elasticity) ----
# k_s = EA / L_seg  (Govoni Eq. 1).  Set CABLE_AXIAL=0 to lock translations.
ENABLE_AXIAL_SPRING  = os.environ.get("CABLE_AXIAL", "1") == "1"
AXIAL_DAMPING_RATIO  = float(os.environ.get("CABLE_AXIAL_ZETA",  0.3))

# ---- Legacy mode (v1 hand-tuned parameters for comparison) ----
LEGACY_MODE          = os.environ.get("CABLE_LEGACY", "0") == "1"

# ---- Joint limits ----
CONE_LIMIT_DEG       = 8.0 if LEGACY_MODE else 30.0
TWIST_LIMIT_DEG      = 5.8 if LEGACY_MODE else None  # None = free twist

# ---- PhysX rigid-body drag (acts like air resistance, separate from EI/L) ----
LINEAR_DAMPING       = 0.05
ANGULAR_DAMPING      = 0.10

# ---- Solver ----
SOLVER_POSITION_ITERATIONS = 64
SOLVER_VELOCITY_ITERATIONS = 8
ENABLE_CCD                 = True

# ---- Time stepping ----
PHYSICS_DT           = float(os.environ.get("CABLE_PHYSICS_DT",  1.0/240.0))
RENDER_DT            = float(os.environ.get("CABLE_RENDER_DT",   1.0/60.0))

# ---- Experiment mode ----
EXPERIMENT_MODE      = os.environ.get("CABLE_MODE", "hanging_kick")
assert EXPERIMENT_MODE in ("hanging_kick", "both_ends_fixed"), \
    f"Unknown CABLE_MODE: {EXPERIMENT_MODE}"

STEP_DISPLACEMENT_M  = float(os.environ.get("CABLE_STEP_DISP",   5e-3))   # 5 mm -- matches Govoni
SETTLE_SECONDS       = float(os.environ.get("CABLE_SETTLE",      0.5))    # settle before step
INITIAL_KICK_VEL     = np.array([
    float(os.environ.get("CABLE_KICK_VX", 1.5)),
    float(os.environ.get("CABLE_KICK_VY", 0.0)),
    float(os.environ.get("CABLE_KICK_VZ", 0.0)),
])

# ---- Stability monitor ----
DIVERGENCE_OMEGA_DEG_S = 1.0e4   # any joint above this deg/s => flagged unstable
STABILITY_CHECK_EVERY  = 4       # render steps between checks

# ---- Output / logging ----
SCRIPT_DIR  = Path(__file__).parent
OUTPUT_BASE = SCRIPT_DIR / "cable_output"
OUTPUT_DIR  = Path(os.environ.get("CABLE_OUTPUT_DIR", str(OUTPUT_BASE / EXPERIMENT_MODE)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH        = OUTPUT_DIR / "trajectory.csv"
SUMMARY_PATH    = OUTPUT_DIR / "summary.json"
LOG_CAPSULES    = sorted({2, 4, 8, NUM_LINKS // 4, NUM_LINKS // 2,
                          3 * NUM_LINKS // 4, NUM_LINKS - 1})
LOG_CAPSULES    = [i for i in LOG_CAPSULES if 0 <= i < NUM_LINKS]

# ---- Recording (off by default in sweep) ----
RECORD_VIDEO    = os.environ.get("CABLE_RECORD", "1") == "1"
MAX_SIM_TIME    = float(os.environ.get("CABLE_MAX_TIME", 10.0))   # seconds
VIDEO_FPS       = 60
VIDEO_WIDTH     = 1920
VIDEO_HEIGHT    = 1080
VIDEO_PATH      = OUTPUT_DIR / "cable_simulation.mp4"
KEY_FRAME_TIMES = [0.0, 2.0, 5.0, 10.0]
WARMUP_STEPS    = 10


# ===============================================================
# 2. DERIVED PARAMETERS
# ===============================================================
SEGMENT_SPACING   = TOTAL_CABLE_LENGTH / NUM_LINKS
LINK_HEIGHT       = max(SEGMENT_SPACING - 2.0 * LINK_RADIUS, 1e-4)
LINK_MASS         = TOTAL_CABLE_MASS / NUM_LINKS

# Beam-theory bending stiffness:  k_b = EI / L_segment
LINK_AREA_MOMENT  = math.pi * LINK_RADIUS**4 / 4.0          # I = pi r^4 / 4  [m^4]
EI                = YOUNG_MODULUS * LINK_AREA_MOMENT        # flexural rigidity [N.m^2]
K_BEND_RAD        = EI / SEGMENT_SPACING                    # [N.m/rad]
JOINT_STIFFNESS   = K_BEND_RAD * math.pi / 180.0            # [N.m/deg] for USD DriveAPI

# Bending damping:  C = zeta . 2.sqrt(K . I_rot)
LINK_ROT_INERTIA  = (1.0/3.0) * LINK_MASS * SEGMENT_SPACING**2   # slender rod about end
C_CRIT_RAD        = 2.0 * math.sqrt(max(K_BEND_RAD * LINK_ROT_INERTIA, 1e-30))
JOINT_DAMPING     = DAMPING_RATIO * C_CRIT_RAD * math.pi / 180.0  # [N.m.s/deg]

# Axial (translational) MSD springs:  k_s = EA / L_seg  (Govoni Eq. 1)
CROSS_SECTION_AREA = math.pi * LINK_RADIUS**2                     # A = pi r^2  [m^2]
K_AXIAL            = YOUNG_MODULUS * CROSS_SECTION_AREA / SEGMENT_SPACING  # [N/m]
C_AXIAL_CRIT       = 2.0 * math.sqrt(max(K_AXIAL * LINK_MASS, 1e-30))
C_AXIAL            = AXIAL_DAMPING_RATIO * C_AXIAL_CRIT           # [N.s/m]

# Legacy mode: override with v1 hand-tuned values
if LEGACY_MODE:
    JOINT_STIFFNESS    = 0.0     # v1: no bending spring
    JOINT_DAMPING      = 0.05    # v1: hand-picked, ~70,000x critical
    LINEAR_DAMPING     = 0.2     # v1 values
    ANGULAR_DAMPING    = 1.0
    ENABLE_AXIAL_SPRING = False  # v1: locked translations


# ===============================================================
# 3. WORLD SETUP
# ===============================================================
world = World(stage_units_in_meters=1.0,
              physics_dt=PHYSICS_DT,
              rendering_dt=RENDER_DT)
world.scene.add_default_ground_plane(z_position=0.0)
stage = world.stage

# Scene-wide PhysX
physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
if physics_scene_prim and physics_scene_prim.IsValid():
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    physx_scene.CreateEnableCCDAttr().Set(ENABLE_CCD)
    physx_scene.CreateSolverTypeAttr().Set("TGS")


# ===============================================================
# 4. BUILD HELPERS
# ===============================================================
def create_top_connector() -> DynamicCuboid:
    SIZE, MASS = 0.03, 2.0
    connector = world.scene.add(
        DynamicCuboid(
            prim_path="/World/top_connector",
            name="top_connector",
            position=np.array([0.0, 0.0, ANCHOR_Z + SIZE / 2]),
            size=SIZE,
            mass=MASS,
            color=np.array([0.2, 0.4, 0.8]),
        )
    )
    # Fix to world
    fj = UsdPhysics.FixedJoint.Define(stage, "/World/fix_top_connector")
    fj.CreateBody1Rel().SetTargets([Sdf.Path("/World/top_connector")])
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, ANCHOR_Z + SIZE / 2))
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    return connector


def create_bottom_connector(make_kinematic: bool) -> DynamicCuboid:
    SIZE, MASS = 0.02, 0.2
    last_center_z = (ANCHOR_Z
                     - (NUM_LINKS - 1) * SEGMENT_SPACING
                     - LINK_RADIUS - LINK_HEIGHT / 2)
    bottom_z = last_center_z - LINK_HEIGHT / 2 - LINK_RADIUS - SIZE / 2
    connector = world.scene.add(
        DynamicCuboid(
            prim_path="/World/bottom_connector",
            name="bottom_connector",
            position=np.array([0.0, 0.0, bottom_z]),
            size=SIZE,
            mass=MASS,
            color=np.array([0.8, 0.2, 0.2]),
        )
    )
    if make_kinematic:
        # Both-ends-fixed mode: bottom is kinematic so we can apply a clean
        # 5 mm step displacement without physics fighting it.
        prim = stage.GetPrimAtPath("/World/bottom_connector")
        rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb_api.CreateKinematicEnabledAttr().Set(True)
    return connector


def create_obstacle() -> DynamicCuboid:
    SIZE, MASS = 0.04, 50.0
    pos = np.array([0.12, 0.0, ANCHOR_Z - TOTAL_CABLE_LENGTH * 0.4])
    obstacle = world.scene.add(
        DynamicCuboid(
            prim_path="/World/obstacle",
            name="obstacle",
            position=pos,
            size=SIZE,
            mass=MASS,
            color=np.array([0.3, 0.7, 0.3]),
        )
    )
    fj = UsdPhysics.FixedJoint.Define(stage, "/World/fix_obstacle")
    fj.CreateBody1Rel().SetTargets([Sdf.Path("/World/obstacle")])
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos))
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    return obstacle


def create_capsule(index: int) -> DynamicCapsule:
    center_z = (ANCHOR_Z
                - index * SEGMENT_SPACING
                - LINK_RADIUS - LINK_HEIGHT / 2)
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
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateSolverPositionIterationCountAttr().Set(SOLVER_POSITION_ITERATIONS)
    physx_rb.CreateSolverVelocityIterationCountAttr().Set(SOLVER_VELOCITY_ITERATIONS)
    physx_rb.CreateLinearDampingAttr().Set(LINEAR_DAMPING)
    physx_rb.CreateAngularDampingAttr().Set(ANGULAR_DAMPING)
    physx_rb.CreateEnableCCDAttr().Set(ENABLE_CCD)
    # CCD = Continuous Collision Detection.
    # Without CCD, PhysX only checks for collisions at discrete time steps.
    # Fast-moving thin bodies (like our 1.5 mm capsules) can "tunnel" through
    # obstacles between steps if they move further than their own size in one
    # step. CCD sweeps the body's trajectory and catches collisions that would
    # otherwise be missed. Costs ~1.5-2x in compute but essential for cables.

    physx_rb.CreateSleepThresholdAttr().Set(1e-5)
    # Sleep threshold (units: kinetic energy, Joules).
    # When a body's kinetic energy stays below this value for several
    # consecutive steps, PhysX puts it to "sleep" -- skips it entirely in the
    # solver until something wakes it up. Saves CPU on settled bodies.
    # Default is ~5e-3. Our value (1e-5) is 500x smaller, meaning capsules
    # only sleep when they are very nearly motionless. This preserves the
    # smooth final settling of the cable; a higher threshold would freeze
    # capsules in a not-quite-equilibrium pose.

    physx_rb.CreateStabilizationThresholdAttr().Set(1e-6)
    # Stabilization threshold (units: kinetic energy, Joules).
    # When a body's kinetic energy drops below this -- but it's not yet
    # asleep -- PhysX applies extra damping to kill numerical jitter caused
    # by accumulated floating-point error. Set smaller than the sleep
    # threshold (1e-6 < 1e-5) so that stabilization is triggered only for
    # the very-slow regime; faster bodies run with normal solver dynamics.
    # Effect: a settled cable rests cleanly instead of visibly trembling.
    return capsule


def create_link_joint(index: int):
    """D6 joint between capsule_index and capsule_{index+1}.
    Changes vs v1:
      - Cone limit on rotY/rotZ widened (configured globally).
      - Twist (rotX) is FREE -- no limit, no drive (matches Govoni).
      - rotY/rotZ have soft EI/L spring + viscous damper from JOINT_STIFFNESS
        and JOINT_DAMPING.
    """
    joint_path = f"/World/link_joint_{index}"
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    # sdf -> Scene Description Foundations
    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/capsule_{index}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/capsule_{index + 1}")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)

    prim = joint.GetPrim()

    # Translational DOFs: soft MSD springs (k_s = EA/L) or locked
    if ENABLE_AXIAL_SPRING:
        for axis in ("transX", "transY", "transZ"):
            drive = UsdPhysics.DriveAPI.Apply(prim, axis)
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(K_AXIAL)
            drive.CreateDampingAttr().Set(C_AXIAL)
            drive.CreateMaxForceAttr().Set(1e8)
    else:
        for axis in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, axis)
            lim.CreateLowAttr().Set(1.0)
            lim.CreateHighAttr().Set(-1.0)   # inverted => axis locked

    # Bending = the two SWING axes (rotY, rotZ): soft EI/L spring + damper.
    #
    # PhysX caveat: a HARD cone limit on both swings forms a "pyramid". When
    # the twist axis is FREE (current model, per Govoni) that pyramid pairs
    # with an unconstrained twist and PhysX rejects it as "double pyramid
    # mode not supported". So in the current model we rely on the spring
    # drive alone (no hard swing limit) -- which is the correct Govoni MSD
    # approach anyway. Legacy mode keeps the hard cone limit AND a hard twist
    # limit (the v1 configuration, which PhysX accepts).
    for axis in ("rotY", "rotZ"):
        if LEGACY_MODE:
            lim = UsdPhysics.LimitAPI.Apply(prim, axis)
            lim.CreateLowAttr().Set(-CONE_LIMIT_DEG)
            lim.CreateHighAttr().Set(+CONE_LIMIT_DEG)

        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set("force")
        drive.CreateDampingAttr().Set(JOINT_DAMPING)
        drive.CreateStiffnessAttr().Set(JOINT_STIFFNESS)
        drive.CreateMaxForceAttr().Set(1e6)

    # Twist (rotX): free in current model (Govoni excludes twist),
    # limited only in legacy mode.
    if TWIST_LIMIT_DEG is not None:
        lim = UsdPhysics.LimitAPI.Apply(prim, "rotX")
        lim.CreateLowAttr().Set(-TWIST_LIMIT_DEG)
        lim.CreateHighAttr().Set(+TWIST_LIMIT_DEG)


def attach_cable_to_top_connector():
    joint = UsdPhysics.Joint.Define(stage, "/World/joint_top_connector_to_cable")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/top_connector")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/capsule_0")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -0.015))   # bottom of top connector
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)
    prim = joint.GetPrim()

    for axis in ("transX", "transY", "transZ"):
        lim = UsdPhysics.LimitAPI.Apply(prim, axis)
        lim.CreateLowAttr().Set(1.0)
        lim.CreateHighAttr().Set(-1.0)

    # Bending: soft spring drive (same as link joints). Hard cone limit only
    # in legacy mode -- pairing a hard swing pyramid with a free twist
    # triggers PhysX "double pyramid mode not supported".
    for axis in ("rotY", "rotZ"):
        if LEGACY_MODE:
            lim = UsdPhysics.LimitAPI.Apply(prim, axis)
            lim.CreateLowAttr().Set(-CONE_LIMIT_DEG)
            lim.CreateHighAttr().Set(+CONE_LIMIT_DEG)

        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set("force")
        drive.CreateDampingAttr().Set(JOINT_DAMPING)
        drive.CreateStiffnessAttr().Set(JOINT_STIFFNESS)
        drive.CreateMaxForceAttr().Set(1e6)

    # Twist (rotX): free in current model, limited only in legacy mode.
    if TWIST_LIMIT_DEG is not None:
        lim = UsdPhysics.LimitAPI.Apply(prim, "rotX")
        lim.CreateLowAttr().Set(-TWIST_LIMIT_DEG)
        lim.CreateHighAttr().Set(+TWIST_LIMIT_DEG)


def attach_cable_to_bottom_connector():
    """Single fixed joint between last capsule and bottom connector.
    Works for both modes -- in both_ends_fixed mode the bottom connector is
    kinematic, so this fixed joint effectively pins the cable end to a
    scripted position.
    """
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/joint_cable_to_bottom_connector")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/capsule_{NUM_LINKS - 1}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/bottom_connector")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +0.01))   # top of bottom connector
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)


# ===============================================================
# 5. BUILD SCENE
# ===============================================================
_mode_label = "LEGACY (v1 hand-tuned)" if LEGACY_MODE else "material-driven"
print("=" * 70)
print(f"Cable simulation  --  mode = {EXPERIMENT_MODE}  [{_mode_label}]")
print("=" * 70)
print(f"  links              : {NUM_LINKS}")
print(f"  segment length     : {SEGMENT_SPACING*1000:.3f} mm")
print(f"  capsule radius     : {LINK_RADIUS*1000:.2f} mm")
print(f"  link mass          : {LINK_MASS*1000:.3f} g")
_twist = f"{TWIST_LIMIT_DEG}deg" if TWIST_LIMIT_DEG is not None else "FREE"
print(f"  cone limit         : {CONE_LIMIT_DEG}deg  (twist: {_twist})")
print(f"  physics dt         : {PHYSICS_DT*1e6:.3f} us   ({1.0/PHYSICS_DT:.1f} Hz)")
print(f"  render  dt         : {RENDER_DT*1e3:.3f} ms   ({1.0/RENDER_DT:.1f} Hz)")
print("  ---- Material -----")
print(f"  Young's modulus E  : {YOUNG_MODULUS/1e6:.3f} MPa")
print(f"  Poisson ratio  nu   : {POISSON_RATIO}")
print(f"  Damping ratio  zeta   : {DAMPING_RATIO}")
print("  ---- Derived ------")
print(f"  I (area moment)    : {LINK_AREA_MOMENT:.3e} m^4")
print(f"  EI (flexural rig.) : {EI:.3e} N.m^2")
print(f"  K_bend per rad     : {K_BEND_RAD:.3e} N.m/rad")
print(f"  JOINT_STIFFNESS    : {JOINT_STIFFNESS:.3e} N.m/deg")
print(f"  I_rot per link     : {LINK_ROT_INERTIA:.3e} kg.m^2")
print(f"  C_critical per rad : {C_CRIT_RAD:.3e} N.m.s/rad")
print(f"  JOINT_DAMPING      : {JOINT_DAMPING:.3e} N.m.s/deg")
print("  ---- Axial MSD ----")
print(f"  axial springs      : {'ON' if ENABLE_AXIAL_SPRING else 'OFF (locked)'}")
if ENABLE_AXIAL_SPRING:
    print(f"  cross-section A    : {CROSS_SECTION_AREA:.3e} m^2")
    print(f"  K_axial (EA/L)     : {K_AXIAL:.3e} N/m")
    print(f"  C_axial            : {C_AXIAL:.3e} N.s/m")
print("=" * 70)

print("Creating top connector...")
top_connector = create_top_connector()

print("Creating bottom connector...")
make_kin = (EXPERIMENT_MODE == "both_ends_fixed")
bottom_connector = create_bottom_connector(make_kinematic=make_kin)

use_obstacle = (EXPERIMENT_MODE == "hanging_kick")
if use_obstacle:
    print("Creating obstacle...")
    obstacle = create_obstacle()

print(f"Creating {NUM_LINKS} capsules...")
capsules = [create_capsule(i) for i in range(NUM_LINKS)]

print("Attaching cable to top connector...")
attach_cable_to_top_connector()

print("Creating link joints...")
for i in range(NUM_LINKS - 1):
    create_link_joint(i)

print("Attaching cable to bottom connector...")
attach_cable_to_bottom_connector()
print("Scene built.\n")


# ===============================================================
# 6. CAMERA / RECORDING SETUP (skipped if recording is off)
# ===============================================================
rgb_annotator = None
if RECORD_VIDEO and not HEADLESS:
    print("Setting up viewport camera + recording...")
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        from omni.isaac.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array([1.5, 1.0, 2.5]),
                    target=np.array([0.0, 0.0, 1.5]))
    render_product = rep.create.render_product(
        "/OmniverseKit_Persp", (VIDEO_WIDTH, VIDEO_HEIGHT))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])


# ===============================================================
# 7. RESET + INITIAL PERTURBATION
# ===============================================================
world.reset()

# Warm-up render so the annotator produces valid frames
if rgb_annotator is not None:
    print("Warming up renderer...")
    for _ in range(WARMUP_STEPS):
        world.step(render=True)

if EXPERIMENT_MODE == "hanging_kick":
    try:
        bottom_connector.set_linear_velocity(INITIAL_KICK_VEL)
        print(f"Applied initial velocity: {INITIAL_KICK_VEL}")
    except Exception as e:
        print(f"Could not set initial velocity: {e}")

elif EXPERIMENT_MODE == "both_ends_fixed":
    # Let cable settle under gravity for SETTLE_SECONDS
    n_settle = int(SETTLE_SECONDS / RENDER_DT)
    print(f"Settling cable for {SETTLE_SECONDS}s ({n_settle} render steps)...")
    for _ in range(n_settle):
        world.step(render=(rgb_annotator is not None))
    # Apply 5 mm step displacement in +x to bottom (kinematic) connector
    pos, ori = bottom_connector.get_world_pose()
    new_pos = pos.copy()
    new_pos[0] += STEP_DISPLACEMENT_M
    bottom_connector.set_world_pose(new_pos, ori)
    print(f"Applied {STEP_DISPLACEMENT_M*1000:.1f} mm step in +x to bottom connector.")


# ===============================================================
# 8. CSV LOGGING SETUP
# ===============================================================
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
header = ["t"] + [f"cap{i}_x" for i in LOG_CAPSULES] \
              + [f"cap{i}_y" for i in LOG_CAPSULES] \
              + [f"cap{i}_z" for i in LOG_CAPSULES]
csv_writer.writerow(header)
print(f"CSV log --> {CSV_PATH}")
print(f"  capsules logged: {LOG_CAPSULES}")


# ===============================================================
# 9. SIMULATION LOOP
# ===============================================================
FRAMES_DIR = OUTPUT_DIR / "frames"
if RECORD_VIDEO and rgb_annotator is not None:
    FRAMES_DIR.mkdir(exist_ok=True)

step_count         = 0
frames_written     = 0
recording_done     = not RECORD_VIDEO       # if recording off, treat as already done
instability_at     = None
max_omega_seen     = 0.0
key_frame_steps    = [int(t / RENDER_DT) for t in KEY_FRAME_TIMES]
total_record_steps = int(MAX_SIM_TIME / RENDER_DT)
total_steps        = total_record_steps

print(f"\nSimulating up to t = {MAX_SIM_TIME}s ({total_steps} render steps)...\n")

try:
    while simulation_app.is_running() and step_count < total_steps:
        world.step(render=(rgb_annotator is not None) or (not HEADLESS))
        step_count += 1
        sim_time = step_count * RENDER_DT

        # --- CSV log (every render step) ---
        row = [sim_time]
        xs, ys, zs = [], [], []
        for idx in LOG_CAPSULES:
            p, _ = capsules[idx].get_world_pose()
            xs.append(float(p[0])); ys.append(float(p[1])); zs.append(float(p[2]))
        csv_writer.writerow(row + xs + ys + zs)

        # --- Stability monitor (sample, don't read all 200) ---
        if step_count % STABILITY_CHECK_EVERY == 0 and instability_at is None:
            stride = max(NUM_LINKS // 10, 1)
            for idx in range(0, NUM_LINKS, stride):
                w = capsules[idx].get_angular_velocity()      # rad/s
                wmag_deg = float(np.linalg.norm(w)) * 180.0 / math.pi
                if wmag_deg > max_omega_seen:
                    max_omega_seen = wmag_deg
                if wmag_deg > DIVERGENCE_OMEGA_DEG_S:
                    instability_at = sim_time
                    print(f"  *** INSTABILITY  t={sim_time:.4f}s "
                          f"capsule {idx} |omega|={wmag_deg:.1e} deg/s ***")
                    break

        # --- Recording (video frames + key frames) ---
        if not recording_done and rgb_annotator is not None:
            data = rgb_annotator.get_data()
            if data is not None and data.size > 0:
                frame_bgr = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(FRAMES_DIR / f"frame_{frames_written:05d}.png"), frame_bgr)
                frames_written += 1

            if step_count in key_frame_steps and data is not None and data.size > 0:
                kf_path = OUTPUT_DIR / f"frame_t{sim_time:.0f}s.png"
                cv2.imwrite(str(kf_path),
                            cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR))
                print(f"  saved key frame: {kf_path.name}  (t={sim_time:.1f}s)")

            if step_count >= total_record_steps:
                recording_done = True
                print(f"\nCapture complete: {frames_written} frames")
                if frames_written > 0:
                    print("Stitching video with ffmpeg...")
                    cmd = ["ffmpeg", "-y",
                           "-framerate", str(VIDEO_FPS),
                           "-i", str(FRAMES_DIR / "frame_%05d.png"),
                           "-c:v", "libx264",
                           "-pix_fmt", "yuv420p",
                           "-crf", "18",
                           str(VIDEO_PATH)]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode == 0:
                        print(f"Video saved: {VIDEO_PATH}")
                    else:
                        print(f"ffmpeg failed: {r.stderr[:300]}")

        # --- Periodic telemetry ---
        if step_count % 120 == 0:
            pos_top, _ = capsules[0].get_world_pose()
            pos_bot, _ = bottom_connector.get_world_pose()
            tag = "UNSTABLE" if instability_at is not None else "  OK   "
            print(f"[{tag}] t={sim_time:5.2f}s "
                  f"max|omega|={max_omega_seen:.2e} deg/s "
                  f"bot=({pos_bot[0]:+.4f},{pos_bot[1]:+.4f},{pos_bot[2]:+.4f})")

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    csv_file.close()
    print(f"\nCSV closed: {CSV_PATH}  ({step_count} rows)")

    # ============================================================
    # 10. SUMMARY JSON  (consumed by govoni_sweep.py)
    # ============================================================
    summary = {
        "legacy_mode":          LEGACY_MODE,
        "experiment_mode":      EXPERIMENT_MODE,
        "num_links":            NUM_LINKS,
        "total_cable_length_m": TOTAL_CABLE_LENGTH,
        "young_modulus_pa":     YOUNG_MODULUS,
        "damping_ratio":        DAMPING_RATIO,
        "physics_dt_s":         PHYSICS_DT,
        "render_dt_s":          RENDER_DT,
        "joint_stiffness_Nm_per_deg":  JOINT_STIFFNESS,
        "joint_damping_Nms_per_deg":   JOINT_DAMPING,
        "axial_spring_enabled":        ENABLE_AXIAL_SPRING,
        "axial_stiffness_N_per_m":     K_AXIAL if ENABLE_AXIAL_SPRING else 0.0,
        "axial_damping_Ns_per_m":      C_AXIAL if ENABLE_AXIAL_SPRING else 0.0,
        "cone_limit_deg":       CONE_LIMIT_DEG,
        "total_sim_time_s":     step_count * RENDER_DT,
        "stable":               (instability_at is None),
        "instability_at_s":     instability_at,
        "max_omega_deg_per_s":  max_omega_seen,
        "csv_path":             str(CSV_PATH),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written: {SUMMARY_PATH}")
    print(f"  stable           : {summary['stable']}")
    if instability_at is not None:
        print(f"  instability_at_s : {instability_at:.4f}")
    print(f"  max |omega| (deg/s)  : {max_omega_seen:.3e}")

    simulation_app.close()

print("Done.")
