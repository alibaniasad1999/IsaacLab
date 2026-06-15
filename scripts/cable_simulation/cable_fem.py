"""
FEM (volumetric soft-body) cable in Isaac Sim -- a true deformable, NOT rigid.

This is the deformable-body counterpart to cable.py (the capsule-chain model).
Instead of a chain of rigid capsules connected by D6 joints, the cable here is
ONE continuous tetrahedral FEM body simulated on the GPU by PhysX. It bends,
sags, stretches and -- the point of this script -- collides with a rigid
obstacle and does NOT pass through it.

WHY THE CABLE IS FATTENED (and how realism is preserved)
---------------------------------------------------------
A real robot cable is ~1.5 mm radius. PhysX FEM voxelizes the body into hexes;
a rod needs >=3 hexes ACROSS its diameter or it behaves as a rigid stick (it
cannot represent a bending gradient through the cross-section). Reaching that on
a 1.5 mm rod would need a voxel resolution where PxTetMaker fails (~res>200).

The fix (verified on this machine): simulate a FAT rod (R_SIM = 15 mm) at a
modest resolution, but re-scale the *material* so the fat rod behaves like the
thin one:

  * Bending stiffness  EI ~ E * r^4.  To keep EI of the thin cable we'd need
    E_sim = E_real * (r_real / R_sim)^4. That exponent (4) reproduces EI exactly
    but leaves the rod axially "rubber-band" soft. Exponent 3 is the practical
    default: bending only ~10x too stiff but axial behaviour stays cable-like
    (it doesn't visibly stretch under its own weight or under end tension --
    important for the two-robot use case). Override with CABLE_E_EXP.

  * Mass.  Fattening multiplies cross-section by (R_sim/r_real)^2 ~ 100x. To
    keep the real ~8 g/m cable weight (so gravity sag/swing is physical) we
    scale density by (r_real / R_sim)^2.

So the *thing you see* is a thick soft tube, but its weight-per-length and its
bend-vs-gravity response match a real TPU cable. Set CABLE_E_EXP=4 for exact EI
(softer axially). NOTE: R_SIM ~ 15 mm is the FEM thinness FLOOR -- thinner makes
PhysX either truncate the simulation mesh or fail to cook the tets (see the
SIM_RADIUS block below). For a genuinely thin 1.5 mm cable, use cable.py.

MATERIAL (cable.py settings: flexible TPU / polyurethane robot-cable jacket)
  E = 40 MPa, nu = 0.48, rho = 1150 kg/m^3, real radius 1.5 mm.
  (nu=0.48 is near-incompressible; co-rotational FEM near nu->0.5 can suffer
   volumetric locking -- if the solve gets twitchy set CABLE_NU=0.45.)

EXPERIMENT
  The LEFT end is welded to the world; the RIGHT end carries a FREE cube (an
  orange loose connector) that gravity pulls. The cable swings/sags down, lands
  on a fixed rigid bar and DRAPES over it -- demonstrating penetration-free
  deformable<->rigid contact while you watch the free end move. Set
  CABLE_RIGHT_FIXED=1 to weld both ends (static "connect two robots" config),
  or CABLE_ENDS=one for a single-ended cantilever (no right cube).

Run (GUI + recorded video, default):
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
    python scripts/cable_simulation/cable_fem.py

Quick headless stability check (no GUI, no video):
    CABLE_HEADLESS=1 CABLE_RECORD=0 CABLE_MAX_TIME=3 \
        python scripts/cable_simulation/cable_fem.py
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

HEADLESS       = os.environ.get("CABLE_HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": HEADLESS})

# ---------------------------------------------------------------
# Imports (after SimulationApp is up)
# ---------------------------------------------------------------
import numpy as np
import torch
import csv
import subprocess

try:
    import cv2
except ImportError:
    cv2 = None

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import DeformablePrim
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema, Usd
from omni.physx.scripts import deformableUtils, physicsUtils
import omni.replicator.core as rep

# Shared physical-cable parameters (length, real radius, E, nu, density) live in
# cable_config.py so all three cable scripts model the SAME physical cable. The
# FEM-specific fattening (SIM_RADIUS) and rescaling are still done below.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cable_config import (TOTAL_CABLE_LENGTH, REAL_RADIUS, YOUNG_MODULUS,
                          POISSON_RATIO, DENSITY)


# ===============================================================
# 1. CONFIGURATION  (env vars override defaults)
# ===============================================================

# ---- Real cable geometry (the physical target -- from cable_config) ----
# NOTE: REAL_RADIUS is ONLY the scaling reference -- it does NOT set the visible
# thickness (that's SIM_RADIUS). For very small REAL_RADIUS the (r/R)^3 / (r/R)^2
# rescaling below would drive E_SIM and DENSITY_SIM toward zero (a degenerate,
# collapsing FEM body); the floors in section 2 keep the body well-conditioned.

# ---- FAITHFUL mode: a fair comparison BASE ----------------------------------
# CABLE_FAITHFUL=1 (DEFAULT) configures cable_fem.py to reproduce the cable_config
# physical target EXACTLY -- same EI and same mass the capsule (cable.py) and
# Cosserat-rod (cable_warp.py) models reproduce -- so the three methods can later
# be compared apples-to-apples. It does this by:
#   * E_SCALE_EXP = 4  -> EI_sim == EI_real EXACTLY (not 32x too stiff)
#   * a low E floor so that exact (soft) E is not clamped
#   * heavy vertex + elasticity damping -> the volumetric "jelly" WOBBLE is bled
#     out WITHOUT changing the equilibrium stiffness (damping shifts dynamics, not
#     EI), so it bends like the real floppy TPU cable and settles smoothly
#   * a clean GRAVITY-BENDING scene (cantilever, NO obstacle/contact) -- because a
#     contact bar needs a FIRM cable and the exact-EI cable is soft and squashes
#     through it; bending under gravity is the fair, reproducible comparison test.
# A startup FIDELITY block prints EI_sim/EI_real and mass_sim/mass_real (~1.00).
#
# CABLE_FAITHFUL=0 restores the firm contact/drag DEMO (exp 2.5, moving bar, etc.).
FAITHFUL = os.environ.get("CABLE_FAITHFUL", "1") == "1"

# ---- Simulated (fattened) rod -- see docstring ----
# The cable MUST be fattened. PhysX FEM has a hard floor on thinness:
#   * the voxelizer only covers the FULL length when the diameter spans >=3
#     voxels, i.e. resolution >= 3*L/(2*R); below that the sim mesh TRUNCATES
#     (e.g. R=10 mm at res 100 gives a body that only reaches x=0.70 m -- the
#     far end then has no sim nodes and any attachment there grabs nothing);
#   * but PxTetMaker FAILS to cook above res ~130 (MEASURED on this GPU: res 130
#     cooks, res 150 -> "createVoxelTetrahedronMesh failed").
# Together these pin the thinnest reliable rod at R ~ 12 mm (24 mm dia, res 130,
# 3 voxels across) -- the DEFAULT below. Thinner is NOT possible on this FEM path
# (10 mm needs res 150 -> cook fails). For a true 1.5 mm cable use the Cosserat
# rod (cable_warp.py) or capsule chain (cable.py); run why_fem_cant_be_thin.py for
# the full explanation. The material is rescaled (below) so this rod still
# bends/weighs like the real 1.5 mm cable -- and a thinner sim rod needs LESS
# fattening, so 12 mm is also stiffer (E~221 kPa vs 126 at 15 mm) = less jelly.
SIM_RADIUS         = float(os.environ.get("CABLE_SIM_RADIUS", 12e-3))  # 12 mm = FEM floor
E_SCALE_EXP        = float(os.environ.get("CABLE_E_EXP", 4.0 if FAITHFUL else 2.5))
# 4 = EXACT EI (faithful base, default); 2.5 = firm contact demo; LOWER = stiffer.
if SIM_RADIUS < 12e-3:
    print(f"[warn] CABLE_SIM_RADIUS={SIM_RADIUS*1000:.1f} mm is below the FEM floor "
          f"(~12 mm): the sim mesh will truncate or cooking will fail (10 mm -> "
          f"PxTetMaker fails). Use cable_warp.py / cable.py for a genuinely thin cable; "
          f"see why_fem_cant_be_thin.py.")

# ---- Material (flexible TPU -- E/nu/density from cable_config) ----
FRICTION      = float(os.environ.get("CABLE_FRICTION", 0.4))    # dynamic friction on the bar

# ---- Scene geometry ----
ANCHOR_Z   = float(os.environ.get("CABLE_Z0",     1.5))   # height of the end connectors
                                                          # (1.5 m gives the free end
                                                          #  room to swing without hitting
                                                          #  the floor)
ENDS       = os.environ.get("CABLE_ENDS", "one" if FAITHFUL else "both")  # "both" | "one"
assert ENDS in ("both", "one"), f"CABLE_ENDS must be both|one, got {ENDS}"
# FAITHFUL default = "one" = a clean CANTILEVER (left fixed, right free, no end
# cube): the tip droops purely under self-weight, a direct, reproducible EI test.

# End conditions. By default the LEFT end is welded to the world and the RIGHT
# end carries a FREE cube (a loose connector) that gravity pulls -- so you can
# watch the cable swing/drape dynamically. Set CABLE_RIGHT_FIXED=1 to weld both
# ends (the static "connect two robots" config).
LEFT_FIXED   = os.environ.get("CABLE_LEFT_FIXED",  "1") == "1"
RIGHT_FIXED  = os.environ.get("CABLE_RIGHT_FIXED", "0") == "1"
END_CUBE_MASS = float(os.environ.get("CABLE_END_MASS", 0.01))  # kg, the free end cube
                                                               # (~cable weight; heavier
                                                               #  rubber-bands the soft rod)
# Drag on the free cube. Dropping the end from horizontal injects a lot of
# energy; with no damping the underdamped pendulum pumps it into a violent 3D
# whip that slams the cube through the bar. This bleeds the swing so the cable
# settles into a clean drape (still clearly moving -- a few visible swings).
END_CUBE_DAMP = float(os.environ.get("CABLE_END_DAMP", 1.0))

# Build the cable STRAIGHT (rest = straight, which is the physical rest state).
# A curved/pre-sagged rest mesh empirically breaks PhysxAutoAttachment at the
# ends (the cable then falls free) -- so pre-sag is OFF by default and warned.
PRE_SAG    = float(os.environ.get("CABLE_PRESAG", 0.0))   # mid-span dip of rest arc (keep 0)
if PRE_SAG > 0:
    print("[warn] CABLE_PRESAG>0 builds a curved rest mesh, which can break the "
          "end auto-attachments (cable falls). Use 0 unless you know why.")

# FAITHFUL base has NO obstacle (the exact-EI cable is soft and would squash
# through a contact bar; gravity bending is the fair test). The contact demo is
# CABLE_FAITHFUL=0 (or force it back on with CABLE_OBSTACLE=1).
USE_OBSTACLE = os.environ.get("CABLE_OBSTACLE", "0" if FAITHFUL else "1") == "1"
OB_RADIUS    = float(os.environ.get("CABLE_OB_RADIUS", 0.05))   # rigid bar radius
# Bar top sits OB_DEPTH below the anchor line: deep enough to clear the straight
# cable's underside at t=0 (no pre-penetration, needs OB_DEPTH > SIM_RADIUS) yet
# high enough to intrude into the cable's natural gravity sag, so the cable is
# forced to rest ON the bar -- the penetration-free contact demo.
OB_DEPTH     = float(os.environ.get("CABLE_OB_DEPTH", 0.08))
# Bar position along the cable (x) and across it (y). x defaults to mid-span.
OB_X         = float(os.environ.get("CABLE_OB_X", TOTAL_CABLE_LENGTH / 2.0))
OB_Y         = float(os.environ.get("CABLE_OB_Y", 0.0))
# Make the bar GRABBABLE so you can move it with the mouse during the sim and
# watch the cable react. A static collider's pose is frozen at sim start (moving
# it does nothing -- the cable keeps hitting the OLD spot), and a KINEMATIC body
# ignores the mouse (viewport drag applies forces, which only move DYNAMIC
# bodies). So the bar is a DYNAMIC rigid body with gravity OFF + heavy damping:
# it stays where you put it, the light cable can't shove it, and you can
# Shift+Left-click-drag it live. Set CABLE_OB_MOVABLE=0 for a frozen static bar.
OB_MOVABLE = os.environ.get("CABLE_OB_MOVABLE",
                            os.environ.get("CABLE_OB_KINEMATIC", "1")) == "1"
# A gravity-OFF bar has nothing holding it but inertia: under the cable's sustained
# weight a light one slowly drifts away (and the swing knocks it aside, so the cable
# falls through). Only MASS resists that, so the bar is heavy -- the mouse grab-spring
# still drags a heavy body fine; it's high DAMPING that makes a grab feel stuck, so
# damping is kept low. (100 kg / damp 8 is the lightest+least-damped combo verified
# penetration-free against the energetic first swing.) Rotation is locked (below) so
# it can't tumble. Lower OB_MASS for an easier drag (may let the cable nudge it);
# raise OB_DAMP if it ever drifts.
OB_MASS    = float(os.environ.get("CABLE_OB_MASS", 100.0))  # kg; cable can't move it
OB_DAMP    = float(os.environ.get("CABLE_OB_DAMP", 8.0))    # low -> stays draggable
# Collision cushion on the bar so a SOFT cable rests on it without tunnelling
# through (see make_obstacle). 0 = PhysX default (fine for a firm cable).
OB_CONTACT_OFFSET = float(os.environ.get("CABLE_OB_CONTACT_OFFSET", 0.0))
OB_REST_OFFSET    = float(os.environ.get("CABLE_OB_REST_OFFSET", 0.005))

# ---- Code-driven bar motion (a deterministic alternative to mouse-dragging) ----
# Set CABLE_OB_MOVE_SPEED!=0 to have the SCRIPT drive the bar at a constant speed
# (e.g. 0.05 = 5 cm/s; NEGATIVE = opposite direction, e.g. -0.05 along z = lower
# the bar) so you can watch -- and the telemetry can VERIFY -- whether the cable
# rides along with the bar or slips off and falls. Code-driving forces the bar
# KINEMATIC (its pose is set exactly each step: the cable can't push it, it can't
# drift, and it can't be mouse-grabbed). It travels along OB_MOVE_AXIS, starting
# after OB_MOVE_START s (so the cable drapes first), bouncing within +/-OB_MOVE_RANGE
# of its start (range=0 = one-way travel, which is what carries the cable off).
#
# To move "until the cable is no longer on the bar": one-way (range 0). The clean
# way is to LOWER the bar (axis z, NEGATIVE speed): the cable separates straight
# off the top surface, no rim to scrape. Sliding sideways (axis y) also drops the
# cable but the cable scrapes the bar's flat END as it falls off the edge.
# DEFAULT: the bar moves ON ITS OWN -- it lowers at 10 cm/s (axis z, negative) and
# descends to the floor, so the cable rides it down, lifts off (~t=5s) and STAYS off.
# This is the verified clean "carry the cable off the cylinder" demo, so a plain
# `python cable_fem.py` shows motion with no env vars. (>~15 cm/s starts to poke the
# cable into the bar, so 10 is the quick-but-clean default.)
# Set CABLE_OB_MOVE_SPEED=0 to turn motion OFF and get the mouse-draggable bar back.
OB_MOVE_SPEED = float(os.environ.get("CABLE_OB_MOVE_SPEED",
                                     0.0 if FAITHFUL else -0.1))    # m/s; !=0 enables,
                                                                    # NEGATIVE = down/-axis
# (no bar motion in the FAITHFUL base -- there's no bar)
OB_MOVE_AXIS  = os.environ.get("CABLE_OB_MOVE_AXIS", "z").lower()    # x | y | z
OB_MOVE_RANGE = float(os.environ.get("CABLE_OB_MOVE_RANGE", 0.0))    # m; 0 = one-way
OB_MOVE_START = float(os.environ.get("CABLE_OB_MOVE_START", 1.0))    # s settle before moving
assert OB_MOVE_AXIS in ("x", "y", "z"), f"CABLE_OB_MOVE_AXIS must be x|y|z, got {OB_MOVE_AXIS}"
OB_CODE_DRIVEN = OB_MOVE_SPEED != 0.0

# ---- FEM discretisation / solver ----
# Voxel resolution = number of voxels along the LONGEST mesh extent (the cable
# length). We auto-pick it so the diameter spans ~TARGET_ACROSS hexes (a bend
# gradient needs >=3): res ~ TARGET_ACROSS * L / (2*SIM_RADIUS). PxTetMaker FAILS
# to cook above ~130 on this GPU (MEASURED: 130 cooks, 150 fails), so we cap at
# 130 -- which is what bounds how thin the cable can be (12 mm radius -> 125 ->
# 3.0 hexes across; 10 mm would need 150 -> cook fails). Override with CABLE_FEM_RES.
TARGET_ACROSS  = float(os.environ.get("CABLE_HEX_ACROSS", 3.0))
_auto_res      = int(math.ceil(TARGET_ACROSS * TOTAL_CABLE_LENGTH / (2.0 * SIM_RADIUS)))
FEM_RESOLUTION = int(os.environ.get("CABLE_FEM_RES", min(max(_auto_res, 24), 130)))
# 80 position iterations keep the soft cable from compressing into the rigid bar
# during the energetic free-end swing (50 was enough for a STATIC bar, but the
# default movable/dynamic bar resolves contact a touch softer, so the swing poked
# in at 50 -> bumped to 80); paired with the 240 Hz dt below.
POS_ITERS       = int(os.environ.get("CABLE_POS_ITERS", 80))
SELF_COLLISION  = os.environ.get("CABLE_SELF_COLL", "0") == "1"
# Vertex velocity damping bleeds the soft body's wobble. 0.05 settles a firm rod;
# the FAITHFUL exact-EI rod is much softer so it needs much heavier damping (0.5)
# to bleed the jelly jiggle -- damping changes the DYNAMICS, not the equilibrium EI,
# so the cable still bends by exactly the config amount but stops wobbling.
VERTEX_DAMPING  = float(os.environ.get("CABLE_VTX_DAMP", 0.5 if FAITHFUL else 0.05))
# Material elasticity damping (Rayleigh-style internal damping): further kills the
# volumetric jelly without changing EI. Only in faithful mode by default.
ELAST_DAMPING   = float(os.environ.get("CABLE_ELAST_DAMP", 0.3 if FAITHFUL else 0.0))

# ---- Visual cable mesh tessellation (cosmetic; sim mesh is the voxel tets) ----
MESH_SEGMENTS = int(os.environ.get("CABLE_MESH_SEG",   16))   # around circumference
MESH_STACKS   = int(os.environ.get("CABLE_MESH_STACK", 80))   # along the length

# ---- Time stepping ----
PHYSICS_DT = float(os.environ.get("CABLE_PHYSICS_DT", 1.0/240.0))
RENDER_DT  = float(os.environ.get("CABLE_RENDER_DT",  1.0/60.0))

# ---- Interactive / recording / run length ----
# In the GUI we run OPEN-ENDED (no time limit, no recording) so you can grab the
# orange cube with the mouse: Shift + Left-click-drag it. The cable is attached,
# so it follows; release and it relaxes back.
INTERACTIVE  = os.environ.get("CABLE_INTERACTIVE", "0" if HEADLESS else "1") == "1"
RECORD_VIDEO = (os.environ.get("CABLE_RECORD", "1") == "1") and not INTERACTIVE
MAX_SIM_TIME = float(os.environ.get("CABLE_MAX_TIME", 8.0))   # seconds (ignored if INTERACTIVE)
# Grabbable free cube = dynamic + gravity OFF, so it stays put (easy to grab),
# the attached cable follows it, and it relaxes back on release. (Dynamic bodies
# transmit motion through the attachment; KINEMATIC ones do NOT -- verified.)
# DEFAULT OFF: with grab on, the right cube floats with gravity disabled at z=Z0
# and -- since the left end is welded at the same height -- the cable just hangs
# nearly straight between two fixed-height points and looks like "no gravity, it
# stays up there". Leaving grab OFF lets the right end FALL, so you actually see
# the cable sag, swing and drape over the bar. Re-enable mouse-grab with CABLE_GRAB=1.
GRAB_CUBE    = os.environ.get("CABLE_GRAB", "0") == "1"
VIDEO_FPS    = 60
VIDEO_WIDTH  = 1920
VIDEO_HEIGHT = 1080

# ---- Output ----
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = Path(os.environ.get("CABLE_OUTPUT_DIR",
                                 str(SCRIPT_DIR / "cable_output" / "fem")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH     = OUTPUT_DIR / "trajectory.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
VIDEO_PATH   = OUTPUT_DIR / "cable_fem.mp4"
KEY_FRAME_TIMES = [0.0, 1.0, 3.0, 6.0]
WARMUP_STEPS = 10


# ===============================================================
# 2. DERIVED PARAMETERS
# ===============================================================
_ratio        = REAL_RADIUS / SIM_RADIUS                 # < 1

# Raw rescaled material so the fat sim rod bends/weighs like the thin real cable.
_E_SIM_raw    = YOUNG_MODULUS * (_ratio ** E_SCALE_EXP)
_RHO_SIM_raw  = DENSITY * (_ratio ** 2)

# Well-conditioning floors. A FEM body needs a minimum stiffness and density or
# it becomes a degenerate, collapsing/jittering blob (a "physically correct"
# sub-millimetre cable rescales to ~tens of Pa and ~0.1 kg/m^3 -- unusable).
# Below these the floor engages and the sim no longer tracks REAL_RADIUS exactly;
# the defaults reproduce the verified-good 15 mm config (40 kPa, 11.5 kg/m^3).
# FAITHFUL mode lowers the E floor to ~2 kPa so the EXACT-EI (soft) modulus is not
# clamped -- exp=4 at 12 mm gives ~9.8 kPa, which must pass through to hit EI_real.
# (Verified stable at 9.8 kPa with the heavy damping above.) Non-faithful keeps the
# 40 kPa floor for the firm contact demo.
E_SIM_MIN     = float(os.environ.get("CABLE_E_SIM_MIN", 2.0e3 if FAITHFUL else 4.0e4))  # Pa
RHO_SIM_MIN   = float(os.environ.get("CABLE_RHO_SIM_MIN", 11.5))    # kg/m^3
E_SIM         = max(_E_SIM_raw,   E_SIM_MIN)
DENSITY_SIM   = max(_RHO_SIM_raw, RHO_SIM_MIN)
if E_SIM > _E_SIM_raw or DENSITY_SIM > _RHO_SIM_raw:
    print(f"[info] material floor engaged for REAL_RADIUS={REAL_RADIUS*1e3:.3f} mm: "
          f"E_sim {_E_SIM_raw:.3g}->{E_SIM:.3g} Pa, "
          f"rho_sim {_RHO_SIM_raw:.3g}->{DENSITY_SIM:.3g} kg/m^3 "
          f"(raise CABLE_RADIUS or lower the floors to track the real cable exactly).")

_real_vol     = math.pi * REAL_RADIUS**2 * TOTAL_CABLE_LENGTH
REAL_MASS     = DENSITY * _real_vol                      # what a real cable weighs
_sim_vol      = math.pi * SIM_RADIUS**2 * TOTAL_CABLE_LENGTH
SIM_MASS      = DENSITY_SIM * _sim_vol                   # should ~equal REAL_MASS

# Real flexural rigidity (for reporting) and the sim's effective one
EI_REAL = YOUNG_MODULUS * math.pi * REAL_RADIUS**4 / 4.0
EI_SIM  = E_SIM         * math.pi * SIM_RADIUS**4  / 4.0

# Obstacle placement (straight cable): underside at t=0 sits at ANCHOR_Z-SIM_RADIUS.
OB_TOP      = ANCHOR_Z - OB_DEPTH
OB_CENTER_Z = OB_TOP - OB_RADIUS
OB_LENGTH   = 0.5   # bar length along Y
_cable_bottom_init = ANCHOR_Z - SIM_RADIUS
if USE_OBSTACLE and OB_TOP >= _cable_bottom_init:
    print(f"[warn] bar top {OB_TOP:.3f} is above the cable underside "
          f"{_cable_bottom_init:.3f} at t=0 -> initial interpenetration. "
          f"Increase CABLE_OB_DEPTH.")


# ===============================================================
# 3. WORLD + GPU DYNAMICS  (deformables are GPU-only)
# ===============================================================
# device="cuda:0" + torch backend puts the whole physics pipeline on the GPU.
# This is MANDATORY for FEM deformables: PhysX soft bodies only run on the GPU,
# and the physics-tensor reads (nodal positions) are only implemented for the
# GPU simulation view (the CPU view raises "getSimNodalPositions not implemented").
world = World(stage_units_in_meters=1.0,
              physics_dt=PHYSICS_DT,
              rendering_dt=RENDER_DT,
              backend="torch",
              device="cuda:0")
world.scene.add_default_ground_plane(z_position=0.0)
stage = world.stage

physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
if physics_scene_prim and physics_scene_prim.IsValid():
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    physx_scene.CreateSolverTypeAttr().Set("TGS")
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)     # REQUIRED for FEM
    physx_scene.CreateBroadphaseTypeAttr().Set("GPU")


# ===============================================================
# 4. BUILD HELPERS
# ===============================================================
def centerline_z(x: float) -> float:
    """Initial cable centreline height -- a shallow parabola (0 dip at the
    ends, PRE_SAG dip at mid-span). A mild pre-sag gives the both-ends-pinned
    cable real slack to drape; a perfectly straight taut span barely sags."""
    t = x / TOTAL_CABLE_LENGTH
    return ANCHOR_Z - PRE_SAG * 4.0 * t * (1.0 - t)


def build_cable_mesh() -> UsdGeom.Mesh:
    """Triangulated solid cylinder along +X, pre-sagged, radius SIM_RADIUS.
    The surface mesh feeds PhysX cooking; the simulated tets come from the
    voxelisation at FEM_RESOLUTION, so a modest tessellation is fine."""
    L, R = TOTAL_CABLE_LENGTH, SIM_RADIUS
    nseg, nst = MESH_SEGMENTS, MESH_STACKS

    points = []
    for i in range(nst + 1):
        x  = L * i / nst
        zc = centerline_z(x)
        for j in range(nseg):
            th = 2.0 * math.pi * j / nseg
            points.append(Gf.Vec3f(x, R * math.cos(th), zc + R * math.sin(th)))
    c0 = len(points); points.append(Gf.Vec3f(0.0, 0.0, centerline_z(0.0)))
    c1 = len(points); points.append(Gf.Vec3f(L,   0.0, centerline_z(L)))

    counts, idx = [], []

    def tri(a, b, c):
        counts.append(3); idx.extend((a, b, c))

    for i in range(nst):
        for j in range(nseg):
            jn = (j + 1) % nseg
            a, b = i * nseg + j,       i * nseg + jn
            d, e = (i + 1) * nseg + j, (i + 1) * nseg + jn
            tri(a, b, e); tri(a, e, d)
    for j in range(nseg):              # caps
        jn = (j + 1) % nseg
        tri(c0, jn, j)
        tri(c1, nst * nseg + j, nst * nseg + jn)

    mesh = UsdGeom.Mesh.Define(stage, "/World/FemCable/mesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.05, 0.05)])
    return mesh


def make_deformable(mesh: UsdGeom.Mesh):
    """Turn the cylinder mesh into a PhysX FEM soft body + bind TPU material."""
    ok = deformableUtils.add_physx_deformable_body(
        stage,
        mesh.GetPath(),
        collision_simplification=True,
        simulation_hexahedral_resolution=FEM_RESOLUTION,
        solver_position_iteration_count=POS_ITERS,
        self_collision=SELF_COLLISION,
        vertex_velocity_damping=VERTEX_DAMPING,
    )
    if not ok:
        raise RuntimeError("add_physx_deformable_body failed (cooking error -- "
                           "try a lower CABLE_FEM_RES or a fatter CABLE_SIM_RADIUS)")

    mat_path = "/World/FemCable/material"
    deformableUtils.add_deformable_body_material(
        stage, mat_path,
        youngs_modulus=E_SIM,
        poissons_ratio=POISSON_RATIO,
        density=DENSITY_SIM,
        dynamic_friction=FRICTION,
        elasticity_damping=ELAST_DAMPING,    # internal damping -> kills jelly wobble
    )
    physicsUtils.add_physics_material_to_prim(stage, mesh.GetPrim(), mat_path)


def make_anchor(name: str, x: float, fixed: bool, mass: float, grab: bool = False):
    """Rigid cube overlapping a cable end, used as an attachment target.

    fixed=True  -> welded to the world (pins that end).
    grab=True   -> dynamic + gravity OFF: stays put, Shift+drag it with the mouse
                   and the attached cable follows; release and it relaxes back.
    otherwise   -> a FREE dynamic cube: gravity drags it around (it falls/swings)."""
    SIZE = max(4.0 * SIM_RADIUS, 0.04)
    pos  = np.array([x, 0.0, centerline_z(x)])
    path = f"/World/{name}"
    color = np.array([0.2, 0.4, 0.8]) if fixed else np.array([0.95, 0.55, 0.05])
    cube = world.scene.add(DynamicCuboid(
        prim_path=path, name=name, position=pos, size=SIZE,
        mass=mass, color=color))
    if fixed:
        fj = UsdPhysics.FixedJoint.Define(stage, f"/World/fix_{name}")
        fj.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in pos]))
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    else:
        # Dynamic cube: drag so motion settles instead of building a 3D whip.
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path))
        rb.CreateLinearDampingAttr().Set(END_CUBE_DAMP)
        rb.CreateAngularDampingAttr().Set(END_CUBE_DAMP)
        if grab:
            rb.CreateDisableGravityAttr().Set(True)   # stays put, grabbable
    return path, cube


def attach_cable(mesh_path: str, anchor_path: str, name: str):
    """PhysX auto-attachment between the deformable mesh and a rigid anchor.

    IMPORTANT: actor0 MUST be the mesh prim that carries PhysxDeformableBodyAPI
    (/World/FemCable/mesh), not the parent Xform -- otherwise the auto-attach
    finds no deformable nodes and silently pins nothing."""
    att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, Sdf.Path(f"/World/{name}"))
    att.CreateActor0Rel().SetTargets([Sdf.Path(mesh_path)])
    att.CreateActor1Rel().SetTargets([Sdf.Path(anchor_path)])
    PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())


def make_obstacle():
    """Rigid bar (cylinder along Y) the cable drapes over, placed at (OB_X, OB_Y,
    OB_CENTER_Z).

    OB_MOVABLE=1 (default): a DYNAMIC rigid body with gravity OFF and heavy
    damping. It hangs in place (no gravity) and is too massive for the light cable
    to push, but you can Shift+Left-click-drag it in the viewport and the cable
    reacts live. (A static collider's pose is frozen at sim start, and a kinematic
    body ignores the mouse -- only dynamic bodies respond to the drag force.)

    OB_MOVABLE=0: a plain STATIC collider (immovable, pose baked at start)."""
    path = "/World/obstacle"
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(OB_RADIUS)
    cyl.CreateHeightAttr(OB_LENGTH)
    cyl.CreateAxisAttr("Y")
    cyl.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.7, 0.3)])
    UsdGeom.XformCommonAPI(cyl).SetTranslate(
        Gf.Vec3d(OB_X, OB_Y, OB_CENTER_Z))
    prim = cyl.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    # Contact offset = a collision "cushion": PhysX starts generating contacts when
    # the cable is still OB_CONTACT_OFFSET away from the bar, so a SOFT cable gets
    # caught before it can sink/tunnel through the thin bar. Rest offset keeps a
    # small standoff at equilibrium. This is what lets a floppy (cable-like) rod
    # rest on the bar without the firm-rod hack. 0 = leave PhysX defaults.
    if OB_CONTACT_OFFSET > 0:
        col.CreateContactOffsetAttr().Set(OB_CONTACT_OFFSET)
        col.CreateRestOffsetAttr().Set(OB_REST_OFFSET)
    if OB_CODE_DRIVEN:
        # Kinematic: the loop sets its pose exactly each step. Infinitely "heavy"
        # to the cable (can't be pushed) and never drifts -- ideal for a clean test.
        UsdPhysics.RigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
    elif OB_MOVABLE:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(OB_MASS)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateDisableGravityAttr().Set(True)       # hangs in place, grabbable
        rb.CreateLinearDampingAttr().Set(OB_DAMP)     # high enough that it stops
        rb.CreateAngularDampingAttr().Set(OB_DAMP)    # dead on release (no drift/spin),
        # Lock rotation entirely so a grab can't tumble the bar -- it stays a clean
        # horizontal cylinder, only translating where you drag it.
        rb.CreateLockedRotAxisAttr().Set(7)           # bit-flags x|y|z = 1|2|4
    return path


# ===============================================================
# 5. BUILD SCENE
# ===============================================================
print("=" * 70)
print(f"FEM deformable cable  --  {'FAITHFUL BASE' if FAITHFUL else 'CONTACT DEMO'}"
      f"  ends={ENDS}  obstacle={USE_OBSTACLE}")
print("=" * 70)
if FAITHFUL:
    print("  *** FAITHFUL mode: reproduces the cable_config target for a FAIR compare.")
    print(f"  *** FIDELITY vs config:  EI_sim/EI_real = {EI_SIM/EI_REAL:.3f}   "
          f"mass_sim/mass_real = {SIM_MASS/REAL_MASS:.3f}")
    print(f"  *** (target 1.000/1.000; the cable bends + weighs like the real TPU cable;")
    print(f"  ***  jelly wobble damped via vtx={VERTEX_DAMPING:g} elast={ELAST_DAMPING:g}.")
    print(f"  ***  set CABLE_FAITHFUL=0 for the firm moving-bar contact demo.)")
    print("=" * 70)
print(f"  real radius        : {REAL_RADIUS*1000:.2f} mm")
print(f"  sim  radius (fat)  : {SIM_RADIUS*1000:.2f} mm  (ratio r/R = {_ratio:.3f})")
print(f"  length             : {TOTAL_CABLE_LENGTH:.3f} m")
print(f"  FEM resolution     : {FEM_RESOLUTION}  (hexes across diameter ~ "
      f"{2*SIM_RADIUS/(TOTAL_CABLE_LENGTH/FEM_RESOLUTION):.1f})")
print(f"  physics dt         : {PHYSICS_DT*1e6:.1f} us  ({1.0/PHYSICS_DT:.0f} Hz)")
print("  ---- Material (real -> sim) ----")
print(f"  Young's modulus E  : {YOUNG_MODULUS/1e6:.2f} MPa  ->  {E_SIM/1e3:.2f} kPa "
      f"(exp {E_SCALE_EXP:g})")
print(f"  Poisson ratio  nu  : {POISSON_RATIO}")
print(f"  density  rho       : {DENSITY:.1f}  ->  {DENSITY_SIM:.2f} kg/m^3")
print(f"  cable mass         : real {REAL_MASS*1000:.2f} g  /  sim {SIM_MASS*1000:.2f} g")
print(f"  EI (flexural rig.) : real {EI_REAL:.3e}  /  sim {EI_SIM:.3e} N.m^2 "
      f"(x{EI_SIM/EI_REAL:.1f})")
print("  ---- Ends ----")
print(f"  left end           : {'FIXED (welded)' if LEFT_FIXED else 'FREE cube'}")
if ENDS == "both":
    _rl = ("FIXED (welded)" if RIGHT_FIXED
           else "GRABBABLE (dynamic, gravity off -- Shift+drag)" if GRAB_CUBE
           else "FREE cube (falls)")
    print(f"  right end          : {_rl}  (cube {END_CUBE_MASS*1000:.0f} g)")
if USE_OBSTACLE:
    print("  ---- Obstacle ----")
    print(f"  bar radius         : {OB_RADIUS*1000:.0f} mm")
    print(f"  bar top z          : {OB_TOP:.3f} m  (cable bottom starts at "
          f"{_cable_bottom_init:.3f} m)")
print("=" * 70)

print("Building cable mesh...")
cable_mesh = build_cable_mesh()
MESH_PATH  = str(cable_mesh.GetPath())

print("Creating FEM deformable body (cooking tets, may take a few seconds)...")
make_deformable(cable_mesh)

print(f"Creating anchors + attachments  (left {'FIXED' if LEFT_FIXED else 'FREE'}"
      f"{', right ' + ('FIXED' if RIGHT_FIXED else 'FREE') if ENDS == 'both' else ''})...")
free_cubes = []   # dynamic end cubes to watch in telemetry
a0, c0 = make_anchor("anchor_left", 0.0,
                     fixed=LEFT_FIXED,
                     mass=0.01 if LEFT_FIXED else END_CUBE_MASS)
attach_cable(MESH_PATH, a0, "attach_left")
if not LEFT_FIXED:
    free_cubes.append(("left", c0))
if ENDS == "both":
    a1, c1 = make_anchor("anchor_right", TOTAL_CABLE_LENGTH,
                         fixed=RIGHT_FIXED,
                         mass=0.01 if RIGHT_FIXED else END_CUBE_MASS,
                         grab=GRAB_CUBE and not RIGHT_FIXED)
    attach_cable(MESH_PATH, a1, "attach_right")
    if not RIGHT_FIXED:
        free_cubes.append(("right", c1))

obstacle_view = None
if USE_OBSTACLE:
    print("Creating obstacle bar...")
    OB_PATH = make_obstacle()
    if OB_CODE_DRIVEN:
        # A view to set the kinematic bar's pose each step.
        from isaacsim.core.prims import RigidPrim
        obstacle_view = RigidPrim(prim_paths_expr=OB_PATH, name="obstacle_view")
        world.scene.add(obstacle_view)
        _dir = {"x": ("+X", "-X"), "y": ("+Y", "-Y"), "z": ("UP", "DOWN")}[OB_MOVE_AXIS]
        _way = _dir[0] if OB_MOVE_SPEED > 0 else _dir[1]
        _mode = (f"one-way {_way}" if OB_MOVE_RANGE == 0
                 else f"bounce +/-{OB_MOVE_RANGE*100:.0f} cm on {OB_MOVE_AXIS}")
        print("  " + "*" * 60)
        print(f"  *** BAR MOVES BY CODE: {abs(OB_MOVE_SPEED)*100:.0f} cm/s {_mode}, "
              f"from t={OB_MOVE_START:.1f}s")
        print(f"  *** (set CABLE_OB_MOVE_SPEED=0 to stop it / use mouse-drag)")
        print("  " + "*" * 60)
    elif OB_MOVABLE:
        print("  bar is MOUSE-DRAGGABLE (Shift+Left-drag). Not moving by code.")

# View for reading deformed nodal positions each step (min-z, contact monitor).
deformable_view = DeformablePrim(prim_paths_expr=MESH_PATH, name="cable_view")
world.scene.add(deformable_view)

print("Scene built.\n")


# ===============================================================
# 6. CAMERA / RECORDING SETUP
# ===============================================================
rgb_annotator = None
if not HEADLESS:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        from omni.isaac.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array([2.2, 2.2, 1.7]),
                    target=np.array([TOTAL_CABLE_LENGTH / 2.0, 0.0, ANCHOR_Z - 0.4]))
if RECORD_VIDEO and not HEADLESS:
    print("Setting up recording...")
    render_product = rep.create.render_product("/OmniverseKit_Persp",
                                                (VIDEO_WIDTH, VIDEO_HEIGHT))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])


# ===============================================================
# 7. RESET + WARM UP
# ===============================================================
world.reset()
if rgb_annotator is not None:
    print("Warming up renderer...")
    for _ in range(WARMUP_STEPS):
        world.step(render=True)


# ===============================================================
# 8. CSV LOGGING SETUP
# ===============================================================
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["t", "min_z", "mid_z", "max_x",
                     "min_dist_to_bar", "on_bar", "bar_x", "bar_y", "bar_z"])
print(f"CSV log --> {CSV_PATH}")

# The bar is a FINITE cylinder: axis along Y, length OB_LENGTH, radius OB_RADIUS,
# centred at (cx, cy, cz). To judge contact correctly for ANY bar motion (incl.
# sliding it sideways out from under the cable) we measure each cable node's
# distance to the finite axis SEGMENT in full 3D -- not just the x/z radial, which
# wrongly ignores the bar ending in Y.
_BAR_CX, _BAR_CY, _BAR_CZ = OB_X, OB_Y, OB_CENTER_Z
# A cable node resting on the bar sits ~OB_RADIUS from the axis; call it "on the
# bar" within this band, and "penetrating" if a node gets meaningfully inside.
ON_BAR_BAND   = OB_RADIUS + 0.025
PEN_THRESHOLD = OB_RADIUS - 0.003


def bar_axis_dist(nodes, cx, cy, cz):
    """Smallest distance from any cable node to the bar's finite axis segment.
    Accounts for the bar's Y length: a node beyond the bar's ends picks up the
    Y overhang, so when the bar slides out from under the cable this distance
    grows and the cable correctly reads as OFF the bar."""
    dx = nodes[:, 0] - cx
    dz = nodes[:, 2] - cz
    dy = np.maximum(np.abs(nodes[:, 1] - cy) - OB_LENGTH / 2.0, 0.0)  # 0 if within span
    return float(np.min(np.sqrt(dx * dx + dy * dy + dz * dz)))


def arc_length(nodes, rest_x, nb=60):
    """Approximate the cable's CURRENT centreline arc length: bin nodes by their
    REST x into nb slices, take each slice centroid, sum the gaps. Compared to the
    rest length L this gives the axial STRETCH -- the key fidelity check a real
    (inextensible) cable must pass and that soft exact-EI FEM fails."""
    order = np.argsort(rest_x)
    xs = rest_x[order]
    pts = nodes[order]
    edges = np.linspace(xs[0], xs[-1], nb + 1)
    cents = []
    for i in range(nb):
        m = (xs >= edges[i]) & (xs <= edges[i + 1])
        if m.any():
            cents.append(pts[m].mean(axis=0))
    cents = np.asarray(cents)
    if len(cents) < 2:
        return float("nan")
    return float(np.sum(np.linalg.norm(np.diff(cents, axis=0), axis=1)))


def read_nodes():
    """(N,3) world nodal positions of the simulation tet mesh, or None."""
    try:
        p = deformable_view.get_simulation_mesh_nodal_positions()
        if hasattr(p, "detach"):          # torch tensor (cuda backend)
            p = p.detach().cpu().numpy()
        p = np.asarray(p)
        return p[0] if p.ndim == 3 else p
    except Exception:
        return None


# ===============================================================
# 9. SIMULATION LOOP
# ===============================================================
def start_ffmpeg(width: int, height: int) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-framerate", str(VIDEO_FPS),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           str(VIDEO_PATH)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


step_count     = 0
frames_written = 0
ffmpeg_proc    = None
instability_at = None
min_z_overall  = ANCHOR_Z
min_dist_overall = float("inf")
contact_seen   = False           # was the cable ever resting on the bar?
left_bar_at    = None            # time the cable came OFF the bar (after contact)
_rest_x        = None            # rest x of each node (for the stretch metric)
arc_len_last   = float("nan")    # latest centreline arc length (= L if no stretch)
key_frame_steps = [int(t / RENDER_DT) for t in KEY_FRAME_TIMES]
total_steps     = int(MAX_SIM_TIME / RENDER_DT)

# --- Code-driven bar trajectory: constant-speed triangle wave along one axis ---
_OB_AXIS_I = {"x": 0, "y": 1, "z": 2}[OB_MOVE_AXIS]
_OB_BASE   = np.array([OB_X, OB_Y, OB_CENTER_Z], dtype=np.float32)


_OB_FLOOR_Z = OB_RADIUS + 0.02   # lowest the bar CENTRE may go (stays above ground)


def bar_position(sim_t: float) -> np.ndarray:
    """Where the code-driven bar should be at time sim_t (world position)."""
    pos = _OB_BASE.copy()
    if sim_t <= OB_MOVE_START:
        return pos
    d = OB_MOVE_SPEED * (sim_t - OB_MOVE_START)      # signed distance travelled
    if OB_MOVE_RANGE > 0.0:                           # bounce within +/-range
        m = abs(d) % (4.0 * OB_MOVE_RANGE)
        if   m <= OB_MOVE_RANGE:       off = m
        elif m <= 3.0 * OB_MOVE_RANGE: off = 2.0 * OB_MOVE_RANGE - m
        else:                          off = m - 4.0 * OB_MOVE_RANGE
    else:                                             # one-way travel (signed)
        off = d
    pos[_OB_AXIS_I] += off
    # Clamp the bar to stay above the ground. For the default downward sweep this
    # makes it descend all the way to the floor and stop there -- well below the
    # cable, so the cable comes off and STAYS off (it can't swing back onto a bar
    # that has parked at the floor).
    pos[2] = max(float(pos[2]), _OB_FLOOR_Z)
    return pos


def drive_bar(sim_t: float) -> np.ndarray:
    """Set the bar's pose for this step and return the bar centre used."""
    if not OB_CODE_DRIVEN:
        return _OB_BASE
    p = bar_position(sim_t)
    if obstacle_view is not None:
        pos = torch.tensor(p, device="cuda:0", dtype=torch.float32).unsqueeze(0)
        obstacle_view.set_world_poses(positions=pos)
    return p

if INTERACTIVE:
    print("\n" + "=" * 70)
    print("INTERACTIVE -- window stays open.")
    if FAITHFUL:
        print("  FAITHFUL BASE: the cantilever cable droops under its own weight by")
        print("  exactly the config's bending stiffness, then settles smoothly (the")
        print("  jelly wobble is damped out). This is the fair comparison base.")
        print("  (CABLE_FAITHFUL=0 for the firm moving-bar contact demo.)")
    else:
        if GRAB_CUBE:
            print("  Shift + Left-click-drag the ORANGE cube -> the cable follows it.")
            print("  Release -> it relaxes back.  (cube gravity is OFF so it stays put.)")
        else:
            print("  Gravity pulls the free ORANGE end down: watch the cable sag, swing")
            print("  and drape over the green bar. (Set CABLE_GRAB=1 to instead disable")
            print("  the end-cube's gravity and Shift+drag it with the mouse.)")
        if USE_OBSTACLE and OB_MOVABLE:
            print("  Shift + Left-click-drag the GREEN bar -> move it and watch the")
            print("  cable react live. (It has no gravity and stays where you drop it.)")
        print("  (FEM cable: expect soft/jelly behaviour and more sag than the Warp")
        print("  rod -- that's the model's nature.)")
    print("  Close the window to quit.")
    print("=" * 70 + "\n")
else:
    print(f"\nSimulating up to t = {MAX_SIM_TIME}s ({total_steps} render steps)...\n")
wall_t0 = time.perf_counter()

try:
    while simulation_app.is_running() and (INTERACTIVE or step_count < total_steps):
        sim_time = (step_count + 1) * RENDER_DT
        _bar_now = drive_bar(sim_time)   # move the kinematic bar BEFORE stepping;
        world.step(render=(rgb_annotator is not None) or (not HEADLESS))
        step_count += 1

        nodes = read_nodes()
        if nodes is not None and nodes.size:
            if _rest_x is None:                 # capture rest geometry once
                _rest_x = nodes[:, 0].copy()
            arc_len_last = arc_length(nodes, _rest_x)
            min_z = float(np.min(nodes[:, 2]))
            mid_z = float(np.mean(nodes[:, 2]))
            max_x = float(np.max(nodes[:, 0]))
            d_bar = (bar_axis_dist(nodes, _bar_now[0], _bar_now[1], _bar_now[2])
                     if USE_OBSTACLE else float("inf"))
            on_bar = bool(USE_OBSTACLE and d_bar <= ON_BAR_BAND)
            csv_writer.writerow([f"{sim_time:.4f}", f"{min_z:.5f}",
                                 f"{mid_z:.5f}", f"{max_x:.5f}",
                                 f"{d_bar:.5f}" if np.isfinite(d_bar) else "",
                                 int(on_bar),
                                 f"{_bar_now[0]:.4f}", f"{_bar_now[1]:.4f}",
                                 f"{_bar_now[2]:.4f}"])
            min_z_overall = min(min_z_overall, min_z)
            if USE_OBSTACLE and np.isfinite(d_bar):
                min_dist_overall = min(min_dist_overall, d_bar)
                if on_bar:
                    contact_seen = True
                # First time the cable LEAVES the bar after it was resting on it
                # (the bar slid out / dropped away) -- the cable is now unsupported.
                elif contact_seen and left_bar_at is None:
                    left_bar_at = sim_time
                    print(f"  >>> cable LEFT the bar at t={sim_time:.2f}s "
                          f"(bar at x={_bar_now[0]:.2f} y={_bar_now[1]:.2f} "
                          f"z={_bar_now[2]:.2f}, dist={d_bar*100:.1f} cm) <<<")
            # Stability: a draping cable never leaves a sane box / goes NaN.
            if (not np.all(np.isfinite(nodes))
                    or np.max(np.abs(nodes)) > 10.0) and instability_at is None:
                instability_at = sim_time
                print(f"  *** INSTABILITY t={sim_time:.3f}s "
                      f"|p|max={np.max(np.abs(nodes)):.2e} ***")

        # Recording
        if rgb_annotator is not None:
            data = rgb_annotator.get_data()
            if data is not None and data.size > 0:
                rgb = np.ascontiguousarray(data[:, :, :3], dtype=np.uint8)
                if ffmpeg_proc is None:
                    h, w = rgb.shape[:2]
                    ffmpeg_proc = start_ffmpeg(w, h)
                ffmpeg_proc.stdin.write(rgb.tobytes())
                frames_written += 1
                if step_count in key_frame_steps and cv2 is not None:
                    kf = OUTPUT_DIR / f"frame_t{sim_time:.0f}s.png"
                    cv2.imwrite(str(kf), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    print(f"  saved key frame: {kf.name}  (t={sim_time:.1f}s)")

        if step_count % 60 == 0:
            rtf = sim_time / max(time.perf_counter() - wall_t0, 1e-9)
            tag = "UNSTABLE" if instability_at else "  OK   "
            on_now = USE_OBSTACLE and np.isfinite(d_bar) and d_bar <= ON_BAR_BAND
            barpos = (f"  bar=({_bar_now[0]:.2f},{_bar_now[1]:.2f},{_bar_now[2]:.2f})"
                      if OB_CODE_DRIVEN else "")
            cube_str = ""
            for side, cube in free_cubes:
                cp, _ = cube.get_world_pose()
                cube_str += (f"  {side}cube=({cp[0]:+.2f},{cp[1]:+.2f},{cp[2]:+.2f})")
            print(f"[{tag}] t={sim_time:5.2f}s  min_z={min_z_overall:.3f} m  "
                  f"on_bar={'Y' if on_now else 'n'}"
                  f"{' (LEFT @%.1fs)' % left_bar_at if left_bar_at else ''}"
                  f"{barpos}{cube_str}  rtf={rtf:.2f}x")

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

    summary = {
        "model":                "FEM_deformable",
        "ends":                 ENDS,
        "obstacle":             USE_OBSTACLE,
        "length_m":             TOTAL_CABLE_LENGTH,
        "real_radius_m":        REAL_RADIUS,
        "sim_radius_m":         SIM_RADIUS,
        "e_scale_exp":          E_SCALE_EXP,
        "young_modulus_real_pa": YOUNG_MODULUS,
        "young_modulus_sim_pa":  E_SIM,
        "poisson_ratio":        POISSON_RATIO,
        "density_real":         DENSITY,
        "density_sim":          DENSITY_SIM,
        "real_mass_kg":         REAL_MASS,
        "sim_mass_kg":          SIM_MASS,
        "EI_real":              EI_REAL,
        "EI_sim":               EI_SIM,
        # fidelity vs the config target (both ~1.0 in FAITHFUL mode = fair base)
        "faithful_mode":        FAITHFUL,
        "EI_ratio_sim_over_real":   EI_SIM / EI_REAL,
        "mass_ratio_sim_over_real": SIM_MASS / REAL_MASS,
        # axial stretch: arc length / rest length (1.0 = inextensible like a real
        # cable; exact-EI FEM is axially soft so this blows up under gravity)
        "arc_length_final_m":   arc_len_last,
        "stretch_pct":          (None if not np.isfinite(arc_len_last)
                                 else (arc_len_last / TOTAL_CABLE_LENGTH - 1.0) * 100.0),
        "fem_resolution":       FEM_RESOLUTION,
        "physics_dt_s":         PHYSICS_DT,
        "obstacle_top_z":       OB_TOP if USE_OBSTACLE else None,
        "obstacle_radius_m":    OB_RADIUS if USE_OBSTACLE else None,
        "min_z_reached":        min_z_overall,
        "min_dist_to_bar_m":    (min_dist_overall if USE_OBSTACLE
                                 and np.isfinite(min_dist_overall) else None),
        "contact_with_bar":     contact_seen,
        # penetration = a node got meaningfully inside the bar radius
        "penetrated_bar":       bool(USE_OBSTACLE and np.isfinite(min_dist_overall)
                                     and min_dist_overall < PEN_THRESHOLD),
        # code-driven motion: did the cable come OFF the bar, and when?
        "bar_code_driven":      OB_CODE_DRIVEN,
        "cable_left_bar":       left_bar_at is not None,
        "cable_left_bar_at_s":  left_bar_at,
        "total_sim_time_s":     step_count * RENDER_DT,
        "wall_clock_s":         wall_elapsed,
        "realtime_factor":      (step_count * RENDER_DT) / max(wall_elapsed, 1e-9),
        "stable":               instability_at is None,
        "instability_at_s":     instability_at,
        "csv_path":             str(CSV_PATH),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written: {SUMMARY_PATH}")
    print(f"  stable         : {summary['stable']}")
    print(f"  min_z reached  : {min_z_overall:.3f} m")
    if FAITHFUL:
        _st = summary['stretch_pct']
        print(f"  FIDELITY       : EI x{EI_SIM/EI_REAL:.3f}  mass x{SIM_MASS/REAL_MASS:.3f}"
              + (f"  STRETCH {_st:+.0f}%" if _st is not None else ""))
        if _st is not None and abs(_st) > 5:
            print(f"  -> EI matches the config, but the cable STRETCHED {_st:.0f}% "
                  f"(a real cable is ~inextensible). FEM ties bending to axial via one")
            print(f"     E, so exact-EI is axially soft. For a faithful base use "
                  f"cable_warp.py (Cosserat rod decouples bend vs stretch).")
    if USE_OBSTACLE:
        print(f"  bar top z      : {OB_TOP:.3f} m")
        print(f"  contact w/ bar : {contact_seen}")
        print(f"  penetrated bar : {summary['penetrated_bar']}")
        if OB_CODE_DRIVEN:
            print(f"  cable left bar : {summary['cable_left_bar']}"
                  + (f" at t={left_bar_at:.2f}s" if left_bar_at else ""))
    print(f"  realtime factor: {summary['realtime_factor']:.2f}x")

    simulation_app.close()

print("Done.")
