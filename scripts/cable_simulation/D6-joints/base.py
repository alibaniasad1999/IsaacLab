"""
Flexible cable with rigid connectors in Isaac Sim -- D6-joints (capsule-chain) model.

This is the GUI variant of the capsule-chain / D6-joint cable. It differs from
scripts/cable_simulation/base/cable.py in three deliberate ways:

  1. ALL physical/numerical data are HARDCODED CONSTANTS in this file. Nothing
     is imported from cable_config and nothing is read from environment
     variables for the data -- the numbers below ARE the configuration. (The
     single exception is the optional end-time, see point 3.)

  2. NO video recording. The ffmpeg / replicator rgb-annotator recording path
     used by the base script is removed entirely. This script only visualises
     live (local GUI or livestream) and writes a per-step CSV + summary.json.

  3. AUTO-START, no end time by default. The script builds the scene, starts
     physics automatically, and you just watch the cable move in the viewer /
     livestream (the toolbar Play/Stop buttons still pause/resume it). It runs
     forever (until you close the window / Stop), UNLESS you provide an end
     time via the CABLE_MAX_TIME environment variable, e.g.

         CABLE_MAX_TIME=10 ./isaaclab.sh -p scripts/cable_simulation/D6-joints/base.py

     With no CABLE_MAX_TIME the simulation has no end time.

Model recap (same physics as the base D6 script):
  * Material-driven parameters: JOINT_STIFFNESS and JOINT_DAMPING are derived
    from Young's modulus + geometry + damping ratio via beam theory
    (K_bend = EI / L_segment).
  * Bending is a soft EI/L spring drive on the two swing axes (capsule axis is
    Z, so the swing/bending axes are rotX and rotY).
  * Twist DOF (rotZ = about the cable axis) is free, with a small viscous damper.
  * Axial DOFs are LOCKED (inextensible cable).
  * Hanging-kick experiment: top fixed, bottom kicked, obstacle present.

Run (local GUI; then press Play):
    ./isaaclab.sh -p scripts/cable_simulation/D6-joints/base.py

Run (remote / headless server, view via WebRTC livestream client):
    ./isaaclab.sh -p scripts/cable_simulation/D6-joints/base.py --livestream 2
"""

from pathlib import Path
import os
import math
import json
import time

# ---------------------------------------------------------------
# Launch Isaac Sim via Isaac Lab's AppLauncher. AppLauncher parses the standard
# Isaac Lab CLI flags (notably --livestream 1/2 for WebRTC streaming and
# --headless), so this same script works on a local desktop window OR streamed
# from a remote/headless server. The app MUST be created before importing any
# isaacsim / pxr modules.
# ---------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Flexible cable (D6-joints capsule chain) simulation -- GUI/livestream.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------
# Imports (after the app is up)
# ---------------------------------------------------------------
import numpy as np
import csv

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCapsule, DynamicCuboid
from isaacsim.core.prims import RigidPrim as RigidPrimView
from pxr import UsdPhysics, Gf, Sdf, PhysxSchema
import omni.timeline


# ===============================================================
# 1. CONFIGURATION  (ALL HARDCODED CONSTANTS -- no env vars, no cable_config)
# ===============================================================

# ---- Physical cable (was cable_config) ----
TOTAL_CABLE_LENGTH   = 1.0          # [m]   total cable length
REAL_RADIUS          = 1.5e-3       # [m]   physical cable radius (1.5 mm)
YOUNG_MODULUS        = 40e6         # [Pa]  flexible TPU ~40 MPa
POISSON_RATIO        = 0.48         # near-incompressible elastomer
DENSITY              = 1150.0       # [kg/m^3] TPU
CABLE_VOLUME         = math.pi * REAL_RADIUS ** 2 * TOTAL_CABLE_LENGTH
CABLE_MASS           = DENSITY * CABLE_VOLUME     # ~8 g for a 1 m x 1.5 mm cable

# ---- Cable geometry ----
NUM_LINKS            = 200
LINK_RADIUS          = REAL_RADIUS   # capsule chain simulates the real 1.5 mm radius
ANCHOR_Z             = 2.0           # height of top anchor above ground

# ---- Material damping ----
# Structural damping ratio. Flexible TPU is LIGHTLY damped (loss factor
# tan(delta) ~ 0.05-0.1), so the realistic critical-damping fraction is small.
DAMPING_RATIO        = 0.05          # fraction of critical damping

# ---- Cable mass: derived from density x volume (NOT hardcoded directly) ----
TOTAL_CABLE_MASS     = CABLE_MASS

# ---- Translational MSD springs (axial elasticity): OFF (inextensible) ----
# Locked translations = inextensible cable. A 1 m x 3 mm TPU cable stretches
# ~0.3 mm under its own weight -- utterly negligible -- and locking the axial
# DOFs lets the sim run cleanly at the default rate (~100-1000x faster).
ENABLE_AXIAL_SPRING  = False
AXIAL_DAMPING_RATIO  = 0.05

# ---- Joint limits ----
CONE_LIMIT_DEG       = 30.0
TWIST_LIMIT_DEG      = None          # None = free twist

# ---- PhysX rigid-body drag (acts like air resistance, separate from EI/L) ----
LINEAR_DAMPING       = 0.05
ANGULAR_DAMPING      = 0.5

# ---- Solver ----
SOLVER_POSITION_ITERATIONS = 32
SOLVER_VELOCITY_ITERATIONS = 4       # TGS caps at 4
ENABLE_CCD                 = True

# ---- Time stepping ----
PHYSICS_DT           = 1.0 / 480.0
RENDER_DT            = 1.0 / 60.0

# ---- Max angular velocity clamp [deg/s] ----
MAX_ANGULAR_VEL_DEG  = 2.0e4

# ---- Experiment (hanging kick: top fixed, bottom kicked, obstacle present) ----
INITIAL_KICK_VEL     = np.array([1.5, 0.0, 0.0])

# ---- Stability monitor ----
DIVERGENCE_OMEGA_DEG_S = 1.0e8
DIVERGENCE_POS_M       = 10.0
STABILITY_CHECK_EVERY  = 4           # render steps between checks

# ---- Output / logging ----
SCRIPT_DIR  = Path(__file__).parent
OUTPUT_DIR  = SCRIPT_DIR / "cable_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH        = OUTPUT_DIR / "trajectory.csv"
SUMMARY_PATH    = OUTPUT_DIR / "summary.json"
LOG_CAPSULES    = sorted({2, 4, 8, NUM_LINKS // 4, NUM_LINKS // 2,
                          3 * NUM_LINKS // 4, NUM_LINKS - 1})
LOG_CAPSULES    = [i for i in LOG_CAPSULES if 0 <= i < NUM_LINKS]

# ---- End time ----
# THE ONLY env var consulted: the optional simulation end time. If unset there
# is NO end time -- the sim runs until you close the GUI / press Stop. If set,
# the loop stops after that many seconds of SIMULATED time (counted only while
# you have the timeline playing).
_max_time_env   = os.environ.get("CABLE_MAX_TIME", "").strip()
MAX_SIM_TIME    = float(_max_time_env) if _max_time_env else None


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
JOINT_STIFFNESS   = K_BEND_RAD * math.pi / 180.0           # [N.m/deg] for USD DriveAPI

# Bending damping:  C = zeta . 2.sqrt(K . I_rot)
LINK_ROT_INERTIA  = (1.0/3.0) * LINK_MASS * SEGMENT_SPACING**2   # slender rod about end
C_CRIT_RAD        = 2.0 * math.sqrt(max(K_BEND_RAD * LINK_ROT_INERTIA, 1e-30))
JOINT_DAMPING     = DAMPING_RATIO * C_CRIT_RAD * math.pi / 180.0  # [N.m.s/deg]

# Axial (translational) MSD springs:  k_s = EA / L_seg
CROSS_SECTION_AREA = math.pi * LINK_RADIUS**2                     # A = pi r^2  [m^2]
K_AXIAL            = YOUNG_MODULUS * CROSS_SECTION_AREA / SEGMENT_SPACING  # [N/m]
C_AXIAL_CRIT       = 2.0 * math.sqrt(max(K_AXIAL * LINK_MASS, 1e-30))
C_AXIAL            = AXIAL_DAMPING_RATIO * C_AXIAL_CRIT           # [N.s/m]


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
    # 10 g cable clip: keeps the jointed-body mass ratio sane (~250:1 against a
    # link). Welded to the world, so its inertia plays no dynamic role.
    SIZE, MASS = 0.03, 0.01
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


def create_bottom_connector() -> DynamicCuboid:
    # 5 g end clip: same mass-ratio reasoning as the top connector. Also makes
    # the "kick" inject a physically sensible momentum for an 8 g cable.
    SIZE, MASS = 0.02, 0.005
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
    # CCD = Continuous Collision Detection. Sweeps the body's trajectory so the
    # thin 1.5 mm capsules don't tunnel through obstacles between steps.

    physx_rb.CreateSleepThresholdAttr().Set(1e-5)
    # Sleep threshold (kinetic energy, Joules). Capsules only sleep when very
    # nearly motionless, preserving the smooth final settling of the cable.

    physx_rb.CreateMaxAngularVelocityAttr().Set(MAX_ANGULAR_VEL_DEG * math.pi / 180.0)
    # Max angular velocity (this PhysX attribute takes rad/s). The near-massless
    # links can spin up to absurd rates from solver round-off; clamping at
    # ~2e4 deg/s removes the divergence path without touching physical motion.

    physx_rb.CreateStabilizationThresholdAttr().Set(1e-6)
    # Stabilization threshold (kinetic energy, Joules). Set below the sleep
    # threshold so a settled cable rests cleanly instead of visibly trembling.
    return capsule


def create_link_joint(index: int):
    """D6 joint between capsule_index and capsule_{index+1}.

    Axis convention: the capsule (and cable) axis is local Z, so
      - rotX, rotY = the two SWING (bending) axes -> soft EI/L spring+damper
      - rotZ       = TWIST about the cable axis  -> free (damper only)
    """
    joint_path = f"/World/link_joint_{index}"
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/capsule_{index}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/capsule_{index + 1}")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2.0 + LINK_RADIUS)))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)

    prim = joint.GetPrim()

    # Translational DOFs: soft MSD springs (k_s = EA/L) or locked.
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

    # Bending = the two SWING axes (rotX, rotY): soft EI/L spring + damper.
    # No hard cone limit -- pairing a hard swing pyramid with a free twist
    # triggers PhysX "double pyramid mode not supported".
    for axis in ("rotX", "rotY"):
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set("force")
        drive.CreateDampingAttr().Set(JOINT_DAMPING)
        drive.CreateStiffnessAttr().Set(JOINT_STIFFNESS)
        drive.CreateMaxForceAttr().Set(1e6)

    # Twist (rotZ, about the cable axis): no spring (free, by design), but a
    # small viscous damper so solver noise doesn't accumulate as spin.
    drive = UsdPhysics.DriveAPI.Apply(prim, "rotZ")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(JOINT_DAMPING)
    drive.CreateMaxForceAttr().Set(1e6)
    if TWIST_LIMIT_DEG is not None:
        lim = UsdPhysics.LimitAPI.Apply(prim, "rotZ")
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

    # Bending (rotX, rotY): soft spring drive (same as link joints).
    for axis in ("rotX", "rotY"):
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set("force")
        drive.CreateDampingAttr().Set(JOINT_DAMPING)
        drive.CreateStiffnessAttr().Set(JOINT_STIFFNESS)
        drive.CreateMaxForceAttr().Set(1e6)

    # Twist (rotZ): damping-only drive, same rationale as the link joints.
    drive = UsdPhysics.DriveAPI.Apply(prim, "rotZ")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(JOINT_DAMPING)
    drive.CreateMaxForceAttr().Set(1e6)
    if TWIST_LIMIT_DEG is not None:
        lim = UsdPhysics.LimitAPI.Apply(prim, "rotZ")
        lim.CreateLowAttr().Set(-TWIST_LIMIT_DEG)
        lim.CreateHighAttr().Set(+TWIST_LIMIT_DEG)


def attach_cable_to_bottom_connector():
    """Single fixed joint between last capsule and bottom connector."""
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
print("=" * 70)
print("Cable simulation  --  D6-joints  [GUI, material-driven, hardcoded]")
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
print(f"  density  rho        : {DENSITY:.1f} kg/m^3")
print(f"  cable mass (rho.V)  : {TOTAL_CABLE_MASS*1000:.2f} g")
print(f"  Damping ratio  zeta   : {DAMPING_RATIO}")
print("  ---- Derived ------")
print(f"  EI (flexural rig.) : {EI:.3e} N.m^2")
print(f"  JOINT_STIFFNESS    : {JOINT_STIFFNESS:.3e} N.m/deg")
print(f"  JOINT_DAMPING      : {JOINT_DAMPING:.3e} N.m.s/deg")
print(f"  axial springs      : {'ON' if ENABLE_AXIAL_SPRING else 'OFF (locked)'}")
print(f"  end time           : {'none (run until GUI stop)' if MAX_SIM_TIME is None else f'{MAX_SIM_TIME} s'}")
print("=" * 70)

print("Creating top connector...")
top_connector = create_top_connector()

print("Creating bottom connector...")
bottom_connector = create_bottom_connector()

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

# Batched view over all capsules: one PhysX tensor read per log step.
capsule_view = world.scene.add(
    RigidPrimView(
        prim_paths_expr=[f"/World/capsule_{i}" for i in range(NUM_LINKS)],
        name="capsule_view",
        reset_xform_properties=False,
    )
)
print("Scene built.\n")


# ===============================================================
# 6. CAMERA SETUP
# ===============================================================
try:
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:
    from omni.isaac.core.utils.viewports import set_camera_view
set_camera_view(eye=np.array([1.5, 1.0, 2.5]),
                target=np.array([0.0, 0.0, 1.5]))


# ===============================================================
# 7. RESET
# ===============================================================
world.reset()


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
# 9. START SIMULATION (auto-start)
# ===============================================================
# world.reset() already started the timeline playing. Make sure it is playing
# so physics advances as soon as the loop begins -- you just watch it in the
# viewer / livestream, no Play button needed. (The toolbar Play/Stop buttons
# still work to pause/resume while the script runs.)
timeline = omni.timeline.get_timeline_interface()
timeline.play()

# Render a few frames so the viewport/livestream shows the scene before the
# physics step loop takes over.
for _ in range(10):
    simulation_app.update()

print("\n" + "=" * 70)
print("  Simulation started automatically.")
if MAX_SIM_TIME is None:
    print("  No end time set -- runs until you Stop / close the window.")
    print("  (Set CABLE_MAX_TIME=<seconds> to give it an end time.)")
else:
    print(f"  Will stop after {MAX_SIM_TIME} s of simulated time.")
print("=" * 70 + "\n")

# Apply the initial kick now that physics is live (hanging-kick experiment).
try:
    bottom_connector.set_linear_velocity(INITIAL_KICK_VEL)
    print(f"Applied initial velocity: {INITIAL_KICK_VEL}")
except Exception as e:
    print(f"Could not set initial velocity: {e}")


# ===============================================================
# 10. SIMULATION LOOP
# ===============================================================
step_count       = 0
instability_at   = None
max_omega_seen   = 0.0
MONITOR_INDICES  = list(range(0, NUM_LINKS, max(NUM_LINKS // 10, 1)))

if MAX_SIM_TIME is None:
    print("\nSimulating with NO end time (press Stop / close window to end)...\n")
else:
    print(f"\nSimulating up to t = {MAX_SIM_TIME}s...\n")
wall_t0 = time.perf_counter()

try:
    while simulation_app.is_running():
        # Honor the GUI Stop/Play button: if the user pauses via the toolbar,
        # keep rendering the stream but don't advance physics.
        if not timeline.is_playing():
            simulation_app.update()
            continue

        # End-time check (only if the user supplied CABLE_MAX_TIME).
        if MAX_SIM_TIME is not None and step_count * RENDER_DT >= MAX_SIM_TIME:
            print(f"\nReached end time t = {MAX_SIM_TIME}s.")
            break

        world.step(render=True)
        step_count += 1
        sim_time = step_count * RENDER_DT

        # --- CSV log (every render step, one batched tensor read) ---
        log_pos, _ = capsule_view.get_world_poses(indices=LOG_CAPSULES, usd=False)
        log_pos = np.asarray(log_pos)
        csv_writer.writerow([sim_time]
                            + [float(v) for v in log_pos[:, 0]]
                            + [float(v) for v in log_pos[:, 1]]
                            + [float(v) for v in log_pos[:, 2]])

        # --- Stability monitor (batched, sampled every Nth capsule) ---
        if step_count % STABILITY_CHECK_EVERY == 0 and instability_at is None:
            if not np.all(np.isfinite(log_pos)) \
                    or np.max(np.abs(log_pos)) > DIVERGENCE_POS_M:
                instability_at = sim_time
                print(f"  *** INSTABILITY  t={sim_time:.4f}s "
                      f"position diverged (|p|max="
                      f"{np.max(np.abs(log_pos)):.2e} m) ***")
            vels = np.asarray(capsule_view.get_velocities(indices=MONITOR_INDICES))
            wmag_deg = np.linalg.norm(vels[:, 3:6], axis=1) * 180.0 / math.pi
            worst = int(np.argmax(wmag_deg))
            if wmag_deg[worst] > max_omega_seen:
                max_omega_seen = float(wmag_deg[worst])
            if wmag_deg[worst] > DIVERGENCE_OMEGA_DEG_S and instability_at is None:
                instability_at = sim_time
                print(f"  *** INSTABILITY  t={sim_time:.4f}s "
                      f"capsule {MONITOR_INDICES[worst]} "
                      f"|omega|={wmag_deg[worst]:.1e} deg/s ***")

        # --- Periodic telemetry ---
        if step_count % 120 == 0:
            pos_bot, _ = bottom_connector.get_world_pose()
            tag = "UNSTABLE" if instability_at is not None else "  OK   "
            rtf = sim_time / max(time.perf_counter() - wall_t0, 1e-9)
            print(f"[{tag}] t={sim_time:5.2f}s "
                  f"max|omega|={max_omega_seen:.2e} deg/s "
                  f"bot=({pos_bot[0]:+.4f},{pos_bot[1]:+.4f},{pos_bot[2]:+.4f}) "
                  f"rtf={rtf:.2f}x")

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    wall_elapsed = time.perf_counter() - wall_t0
    csv_file.close()
    print(f"\nCSV closed: {CSV_PATH}  ({step_count} rows)")

    # ============================================================
    # 11. SUMMARY JSON
    # ============================================================
    summary = {
        "method":               "D6-joints",
        "num_links":            NUM_LINKS,
        "total_cable_length_m": TOTAL_CABLE_LENGTH,
        "total_cable_mass_kg":  TOTAL_CABLE_MASS,
        "young_modulus_pa":     YOUNG_MODULUS,
        "poisson_ratio":        POISSON_RATIO,
        "density_kg_m3":        DENSITY,
        "damping_ratio":        DAMPING_RATIO,
        "physics_dt_s":         PHYSICS_DT,
        "render_dt_s":          RENDER_DT,
        "joint_stiffness_Nm_per_deg":  JOINT_STIFFNESS,
        "joint_damping_Nms_per_deg":   JOINT_DAMPING,
        "axial_spring_enabled":        ENABLE_AXIAL_SPRING,
        "cone_limit_deg":       CONE_LIMIT_DEG,
        "solver_position_iterations": SOLVER_POSITION_ITERATIONS,
        "solver_velocity_iterations": SOLVER_VELOCITY_ITERATIONS,
        "max_sim_time_s":       MAX_SIM_TIME,
        "total_sim_time_s":     step_count * RENDER_DT,
        "wall_clock_s":         wall_elapsed,
        "realtime_factor":      (step_count * RENDER_DT) / max(wall_elapsed, 1e-9),
        "stable":               (instability_at is None),
        "instability_at_s":     instability_at,
        "max_omega_deg_per_s":  max_omega_seen,
        "csv_path":             str(CSV_PATH),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written: {SUMMARY_PATH}")
    print(f"  wall clock       : {wall_elapsed:.1f} s "
          f"({summary['realtime_factor']:.2f}x realtime)")
    print(f"  stable           : {summary['stable']}")
    if instability_at is not None:
        print(f"  instability_at_s : {instability_at:.4f}")
    print(f"  max |omega| (deg/s)  : {max_omega_seen:.3e}")

    simulation_app.close()

print("Done.")
