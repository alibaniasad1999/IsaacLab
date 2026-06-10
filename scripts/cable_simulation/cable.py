"""
Flexible cable with rigid connectors in Isaac Sim (base capsule-chain model).

Design:

  1. Material-driven parameters: JOINT_STIFFNESS and JOINT_DAMPING are derived
     from Young's modulus + geometry + damping ratio via beam theory
     (K_bend = EI / L_segment).
  2. Bending is provided by a soft EI/L spring drive on the two swing axes.
     The capsule axis is Z, so the swing (bending) axes are rotX and rotY.
  3. Twist DOF (rotZ = about the cable axis) is free (no limit, no drive).
  4. Axial DOFs are LOCKED by default (inextensible cable). Optional axial
     elasticity via translational MSD springs (k_s = EA / L_segment) with
     CABLE_AXIAL=1 -- but note this forces a tiny physics dt (see stability
     guard) and makes the sim 100-1000x slower.
  5. Two experiment modes:
       * "hanging_kick"    -- top fixed, bottom kicked, obstacle present.
       * "both_ends_fixed" -- stability test: top fixed, bottom kinematic,
                             5 mm step displacement after a short settle.
  6. Per-step CSV logging of capsule positions + stability monitor that flags
     divergence and writes summary.json.

Run (defaults: hanging_kick mode, 200 links):
    conda activate env_isaaclab
    python scripts/cable_simulation/cable.py

Run a step-displacement stability test (no GUI, no video, short):
    CABLE_MODE=both_ends_fixed CABLE_HEADLESS=1 CABLE_RECORD=0 \
        CABLE_NUM_LINKS=10 CABLE_E=1e9 CABLE_PHYSICS_DT=5e-6 \
        CABLE_MAX_TIME=1.0 python cable.py
"""

from pathlib import Path
import os
import math
import json
import time

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
from isaacsim.core.prims import RigidPrim as RigidPrimView
from pxr import UsdPhysics, Gf, Sdf, PhysxSchema
import omni.replicator.core as rep


# ===============================================================
# 1. CONFIGURATION  (env vars override defaults -- used by sweep)
# ===============================================================

# ---- Cable geometry ----
NUM_LINKS            = int(  os.environ.get("CABLE_NUM_LINKS",   200))
TOTAL_CABLE_LENGTH   = float(os.environ.get("CABLE_LENGTH",      1.0))
LINK_RADIUS          = float(os.environ.get("CABLE_RADIUS",      1.5e-3))  # 1.5 mm

ANCHOR_Z             = 2.0   # height of top anchor above ground

# ---- Material properties (default: flexible TPU / polyurethane robot cable) ----
# Flexible thermoplastic polyurethane cable jacket (Shore ~85-95A):
#   E ~ 30-50 MPa, nu ~ 0.48 (near-incompressible elastomer), rho ~ 1150 kg/m^3.
YOUNG_MODULUS        = float(os.environ.get("CABLE_E",           40e6))     # Pa  (flexible TPU ~ 40 MPa)
POISSON_RATIO        = float(os.environ.get("CABLE_NU",          0.48))     # near-incompressible elastomer
DENSITY              = float(os.environ.get("CABLE_DENSITY",     1150.0))   # kg/m^3 (TPU)
# Structural damping ratio. Flexible TPU is LIGHTLY damped (loss factor
# tan(delta) ~ 0.05-0.1), so the realistic critical-damping fraction is small.
# (The old value 0.2 made the cable swing as if through honey.)
DAMPING_RATIO        = float(os.environ.get("CABLE_ZETA",        0.05))     # fraction of critical damping

# ---- Cable mass: derived from density x volume (NOT hardcoded) ----
# A 1 m x 3 mm PUR cable physically weighs ~8 g; the old hardcoded 1.0 kg was
# ~120x too heavy and dominated the (unrealistic) sag/swing dynamics.
_CABLE_VOLUME        = math.pi * LINK_RADIUS**2 * TOTAL_CABLE_LENGTH
TOTAL_CABLE_MASS     = float(os.environ.get("CABLE_MASS", DENSITY * _CABLE_VOLUME))

# ---- Translational MSD springs (axial elasticity) ----
# k_s = EA / L_seg  (axial stiffness).  Set CABLE_AXIAL=1 to enable.
#
# DEFAULT IS OFF (locked translations = inextensible cable). Why:
#   * Physical: a 1 m x 3 mm TPU cable stretches ~0.3 mm under its own
#     weight (strain = m g / (EA) ~ 3e-4) -- utterly negligible.
#   * Performance: the axial springs are the stiffest element in the system
#     and force the stability guard below to clamp physics_dt to ~10-200 us
#     (i.e. 5,000-75,000 physics steps per simulated second). With locked
#     translations the sim runs cleanly at the default 240 Hz -- a
#     ~100-1000x speedup for visually/dynamically identical results.
ENABLE_AXIAL_SPRING  = os.environ.get("CABLE_AXIAL", "0") == "1"
AXIAL_DAMPING_RATIO  = float(os.environ.get("CABLE_AXIAL_ZETA",  0.05))

# ---- Legacy mode (v1 hand-tuned parameters for comparison) ----
LEGACY_MODE          = os.environ.get("CABLE_LEGACY", "0") == "1"

# ---- Joint limits ----
CONE_LIMIT_DEG       = 8.0 if LEGACY_MODE else 30.0
TWIST_LIMIT_DEG      = 5.8 if LEGACY_MODE else None  # None = free twist

# ---- PhysX rigid-body drag (acts like air resistance, separate from EI/L) ----
# Angular damping 0.5: for an 8 g cable air drag dominates rotational motion,
# and numerically it bleeds off the spin that contact impulses inject into
# the near-massless (0.04 g) links. 0.1 let contact-induced spin spikes
# persist for seconds.
LINEAR_DAMPING       = float(os.environ.get("CABLE_LIN_DAMP", 0.05))
ANGULAR_DAMPING      = float(os.environ.get("CABLE_ANG_DAMP", 0.5))

# ---- Solver ----
# 32 TGS position iterations hold a 200-link chain together at 240 Hz with
# locked axial DOFs; 64 was tuned for the (much harder) stiff-spring setup.
SOLVER_POSITION_ITERATIONS = int(os.environ.get("CABLE_POS_ITERS", 32))
SOLVER_VELOCITY_ITERATIONS = int(os.environ.get("CABLE_VEL_ITERS", 4))  # TGS caps at 4
ENABLE_CCD                 = os.environ.get("CABLE_CCD", "1") == "1"

# ---- Time stepping ----
# 480 Hz: the bending springs of the very light links live at ~1.5 kHz, so
# 240 Hz leaves the rotational modes under-resolved and they slowly pump
# energy into the chain (positions eventually diverge). 480 Hz + the angular
# velocity clamp below keeps the chain stable while remaining ~12x faster
# than the old stiff-axial-spring configuration.
PHYSICS_DT           = float(os.environ.get("CABLE_PHYSICS_DT",  1.0/480.0))
RENDER_DT            = float(os.environ.get("CABLE_RENDER_DT",   1.0/60.0))

# ---- Max angular velocity clamp [deg/s] ----
# Numerical safety valve for the near-massless links: physical swing/whip
# motion of this cable stays below ~2e4 deg/s, anything above is solver
# noise. Clamping prevents the noise from feeding back into positions.
MAX_ANGULAR_VEL_DEG  = float(os.environ.get("CABLE_MAX_OMEGA", 2.0e4))

# ---- Experiment mode ----
EXPERIMENT_MODE      = os.environ.get("CABLE_MODE", "hanging_kick")
assert EXPERIMENT_MODE in ("hanging_kick", "both_ends_fixed"), \
    f"Unknown CABLE_MODE: {EXPERIMENT_MODE}"

STEP_DISPLACEMENT_M  = float(os.environ.get("CABLE_STEP_DISP",   5e-3))   # 5 mm step input
SETTLE_SECONDS       = float(os.environ.get("CABLE_SETTLE",      0.5))    # settle before step
INITIAL_KICK_VEL     = np.array([
    float(os.environ.get("CABLE_KICK_VX", 1.5)),
    float(os.environ.get("CABLE_KICK_VY", 0.0)),
    float(os.environ.get("CABLE_KICK_VZ", 0.0)),
])

# ---- Stability monitor ----
# The PRIMARY stability verdict is positional: the sim is "unstable" when a
# capsule leaves a sane bounding box or goes NaN -- that is what a user sees
# as the cable exploding. Angular velocity is only a secondary tripwire set
# at true-explosion level: transient spin spikes of the near-massless links
# during contacts (1e5-1e6 deg/s for a few ms) are sub-grid noise that does
# not move positions and must not fail the run.
DIVERGENCE_OMEGA_DEG_S = float(os.environ.get("CABLE_DIV_OMEGA", 1.0e8))
DIVERGENCE_POS_M       = float(os.environ.get("CABLE_DIV_POS",   10.0))
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

# Axial (translational) MSD springs:  k_s = EA / L_seg
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


# ---------------------------------------------------------------
# STABILITY GUARD  (prevents "Illegal BroadPhaseUpdateData" / explosion)
# ---------------------------------------------------------------
# A semi-implicit spring integrator is only stable when the physics timestep
# resolves the stiffest spring in the system. The axial MSD springs
# (k_s = EA/L) are by far the stiffest: single-DOF natural frequency is
# omega_n = sqrt(k_s / m_link). In a 1D chain the highest mode reaches
# ~2*omega_n, so stability requires dt < 2/(2*omega_n) = 1/omega_n. We target
# half of that bound (factor 0.5 -> ~2x margin) and clamp PHYSICS_DT down if
# the user's value is too coarse. Without this, a coarse dt (e.g. the 1/240
# default) makes the cable explode and PhysX reports
# "Illegal BroadPhaseUpdateData".
if ENABLE_AXIAL_SPRING:
    _omega_axial = math.sqrt(K_AXIAL / max(LINK_MASS, 1e-30))   # single-DOF [rad/s]
    _omega_chain = 2.0 * _omega_axial                           # highest chain mode
    DT_STABLE    = 0.5 * (2.0 / _omega_chain)                   # ~2x safety margin
    if PHYSICS_DT > DT_STABLE:
        print(f"[stability] axial spring omega_n = {_omega_axial:.1f} rad/s "
              f"({_omega_axial/(2*math.pi):.1f} Hz); requested physics_dt="
              f"{PHYSICS_DT*1e6:.1f} us is too coarse.")
        print(f"[stability] clamping physics_dt -> {DT_STABLE*1e6:.1f} us "
              f"({1.0/DT_STABLE:.0f} Hz) to keep the spring chain stable.")
        print(f"[stability] (set CABLE_AXIAL=0 to lock axial DOFs and run "
              f"fast at a coarse dt instead.)")
        PHYSICS_DT = DT_STABLE


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
    # MASS: keep the jointed-body mass ratio sane. PhysX maximal-coordinate
    # joints degrade badly when the two bodies differ by more than ~100x in
    # mass; the old 2.0 kg block against a 0.04 g capsule (50,000:1) made the
    # first joint pop and inject energy into the chain. 10 g (a cable clip)
    # is physical AND solvable (~250:1). The connector is welded to the world
    # anyway, so its inertia plays no dynamic role.
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


def create_bottom_connector(make_kinematic: bool) -> DynamicCuboid:
    # 5 g end clip: same mass-ratio reasoning as the top connector (the old
    # 0.2 kg was 5,000:1 against a link). Also makes the "kick" inject a
    # physically sensible momentum for an 8 g cable.
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

    physx_rb.CreateMaxAngularVelocityAttr().Set(MAX_ANGULAR_VEL_DEG * math.pi / 180.0)
    # Max angular velocity (NOTE: this PhysX attribute takes rad/s -- verified
    # empirically: a clamp of 2e4 capped the sim at ~1.04e6 deg/s = 1.8e4
    # rad/s). The links weigh ~0.04 g with rotational
    # inertia ~3e-10 kg.m^2; solver round-off can spin such bodies up to
    # absurd rates, and those spikes couple back into joint positions until
    # the cable explodes. Real motion of this cable never exceeds ~2e4 deg/s
    # (a whip-crack), so clamping there removes the divergence path without
    # touching physical behavior.

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

    Axis convention: the capsule (and cable) axis is the local Z axis, so
      - rotX, rotY = the two SWING (bending) axes -> soft EI/L spring+damper
      - rotZ       = TWIST about the cable axis  -> free (limited in legacy)
    (Earlier versions sprung rotY/rotZ and freed rotX, which left one
    bending plane with zero stiffness and put the EI spring on twist.)
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

    # Bending = the two SWING axes (rotX, rotY): soft EI/L spring + damper.
    #
    # PhysX caveat: a HARD cone limit on both swings forms a "pyramid". When
    # the twist axis is FREE (current model) that pyramid pairs
    # with an unconstrained twist and PhysX rejects it as "double pyramid
    # mode not supported". So in the current model we rely on the spring
    # drive alone (no hard swing limit) -- the correct MSD spring approach.
    # Legacy mode keeps the hard cone limit AND a hard twist
    # limit (the v1 configuration, which PhysX accepts).
    for axis in ("rotX", "rotY"):
        if LEGACY_MODE:
            lim = UsdPhysics.LimitAPI.Apply(prim, axis)
            lim.CreateLowAttr().Set(-CONE_LIMIT_DEG)
            lim.CreateHighAttr().Set(+CONE_LIMIT_DEG)

        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set("force")
        drive.CreateDampingAttr().Set(JOINT_DAMPING)
        drive.CreateStiffnessAttr().Set(JOINT_STIFFNESS)
        drive.CreateMaxForceAttr().Set(1e6)

    # Twist (rotZ, about the cable axis): no spring (free, by design), but a
    # small viscous damper. The twist inertia of a 1.5 mm capsule is ~5e-11
    # kg.m^2; totally undamped, this DOF is where solver noise accumulates
    # (observed: capsule spinning at 1e6 deg/s about its own axis). Physically
    # this damping is the cable's internal torsional friction.
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

    # Bending (rotX, rotY): soft spring drive (same as link joints). Hard
    # cone limit only in legacy mode -- pairing a hard swing pyramid with a
    # free twist triggers PhysX "double pyramid mode not supported".
    for axis in ("rotX", "rotY"):
        if LEGACY_MODE:
            lim = UsdPhysics.LimitAPI.Apply(prim, axis)
            lim.CreateLowAttr().Set(-CONE_LIMIT_DEG)
            lim.CreateHighAttr().Set(+CONE_LIMIT_DEG)

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
print(f"  density  rho        : {DENSITY:.1f} kg/m^3")
print(f"  cable mass (rho.V)  : {TOTAL_CABLE_MASS*1000:.2f} g")
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

# Batched view over all capsules: one PhysX tensor read per log step instead
# of one USD pose query per capsule (the explicit path list preserves index
# order; a regex would sort capsule_10 before capsule_2).
capsule_view = world.scene.add(
    RigidPrimView(
        prim_paths_expr=[f"/World/capsule_{i}" for i in range(NUM_LINKS)],
        name="capsule_view",
        reset_xform_properties=False,
    )
)
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
def start_ffmpeg(width: int, height: int) -> subprocess.Popen:
    """Stream raw RGB frames straight into ffmpeg -- no PNG round-trip."""
    cmd = ["ffmpeg", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}",
           "-framerate", str(VIDEO_FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           str(VIDEO_PATH)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


step_count         = 0
frames_written     = 0
recording_done     = not RECORD_VIDEO       # if recording off, treat as already done
ffmpeg_proc        = None
instability_at     = None
max_omega_seen     = 0.0
key_frame_steps    = [int(t / RENDER_DT) for t in KEY_FRAME_TIMES]
total_record_steps = int(MAX_SIM_TIME / RENDER_DT)
total_steps        = total_record_steps
MONITOR_INDICES    = list(range(0, NUM_LINKS, max(NUM_LINKS // 10, 1)))

print(f"\nSimulating up to t = {MAX_SIM_TIME}s ({total_steps} render steps)...\n")
wall_t0 = time.perf_counter()

try:
    while simulation_app.is_running() and step_count < total_steps:
        world.step(render=(rgb_annotator is not None) or (not HEADLESS))
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
            # Position divergence (the clamped omega can't reveal it)
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

        # --- Recording (video frames + key frames) ---
        if not recording_done and rgb_annotator is not None:
            data = rgb_annotator.get_data()
            if data is not None and data.size > 0:
                rgb = np.ascontiguousarray(data[:, :, :3], dtype=np.uint8)
                if ffmpeg_proc is None:
                    h, w = rgb.shape[:2]
                    ffmpeg_proc = start_ffmpeg(w, h)
                ffmpeg_proc.stdin.write(rgb.tobytes())
                frames_written += 1

                if step_count in key_frame_steps:
                    kf_path = OUTPUT_DIR / f"frame_t{sim_time:.0f}s.png"
                    cv2.imwrite(str(kf_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    print(f"  saved key frame: {kf_path.name}  (t={sim_time:.1f}s)")

            if step_count >= total_record_steps:
                recording_done = True
                print(f"\nCapture complete: {frames_written} frames")

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

    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print(f"Video saved: {VIDEO_PATH}  ({frames_written} frames)")

    # ============================================================
    # 10. SUMMARY JSON
    # ============================================================
    summary = {
        "legacy_mode":          LEGACY_MODE,
        "experiment_mode":      EXPERIMENT_MODE,
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
        "axial_stiffness_N_per_m":     K_AXIAL if ENABLE_AXIAL_SPRING else 0.0,
        "axial_damping_Ns_per_m":      C_AXIAL if ENABLE_AXIAL_SPRING else 0.0,
        "cone_limit_deg":       CONE_LIMIT_DEG,
        "solver_position_iterations": SOLVER_POSITION_ITERATIONS,
        "solver_velocity_iterations": SOLVER_VELOCITY_ITERATIONS,
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
