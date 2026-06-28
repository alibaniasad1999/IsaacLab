"""
cable_learn.py  --  a CAPSULE-CHAIN cable, built from scratch to LEARN from.
============================================================================

This is the *teaching* version of cable.py. It models the same physical idea
with the same Isaac Sim / USD API, but stripped down to the bare minimum so you
can read every line and understand WHY it is there. Once this makes sense, the
big cable.py (axial springs, twist tuning, obstacles, scaling) will read easily.

------------------------------------------------------------------------------
THE IDEA (read this first)
------------------------------------------------------------------------------
A real cable is continuous and floppy -- hard to simulate directly. The trick:
approximate it by a CHAIN of short RIGID capsules, joined by joints that bend.

    anchor
      O==O==O==O==O==O==O==O      each "==" is a rigid capsule
          ^  ^  ^                 each "O" is a joint that lets neighbours rotate

Four physical ingredients, one per section below:

  1. CAPSULES  -- the rigid pieces. More pieces = smoother cable, but slower.
  2. JOINTS    -- a D6 joint between each pair. We LOCK stretching (a cable
                  barely stretches) and ALLOW bending.
  3. STIFFNESS -- a small spring on the bending axes = "how floppy". Physically
                  EI / L (E=material stiffness, I=cross-section, L=segment).
  4. GRAVITY + GROUND -- pull the cable down; the floor stops it -> it DRAPES.

------------------------------------------------------------------------------
THE EXPERIMENT (matches your real-cable photo/video setup)
------------------------------------------------------------------------------
  * The FIRST capsule is the ANCHOR (welded to the world, never moves).
  * One chosen capsule is the PULLED point: we move it (kinematically) to a
    known position -- like you pulling a point of the real cable by a known
    amount.
  * Gravity + a ground plane make the rest of the cable sag and rest on the
    floor.
  * We let it settle, then LOG every capsule's (x, z) to a CSV in METRES --
    the SAME format extract_cable_profile.py compares against. So sim vs real
    is a direct overlay.

------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------
  conda activate env_isaaclab

  # watch it in the GUI:
  python scripts/cable_simulation/base/cable_learn.py

  # headless, just produce the CSV:
  CABLE_HEADLESS=1 python scripts/cable_simulation/base/cable_learn.py

Knobs (environment variables, all optional -- defaults are sensible):
  CABLE_LENGTH   total cable length [m]            (default 1.0)
  CABLE_LINKS    number of capsules                (default 40)
  CABLE_E        Young's modulus [Pa] = stiffness  (default 40e6, TPU)
  CABLE_PULL_X   x to pull the pulled-point to [m] (default 0.4)
  CABLE_PULL_Z   z (height) to pull it to    [m]   (default 0.3)
  CABLE_SETTLE   seconds to let it settle          (default 4.0)
  CABLE_OUT      output csv path                   (default cable_learn.csv)
"""

import os
import math
import csv

# --------------------------------------------------------------------------
# STEP 0: start Isaac Sim.
# The SimulationApp MUST be created before importing any isaacsim.* modules --
# it boots the underlying Omniverse/USD runtime that those imports need.
# --------------------------------------------------------------------------
from isaacsim.simulation_app import SimulationApp

HEADLESS = os.environ.get("CABLE_HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": HEADLESS})

import numpy as np
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCapsule
from pxr import UsdPhysics, Gf, Sdf


# ==========================================================================
# PARAMETERS  --  the few numbers that define the cable and the experiment.
# ==========================================================================
LENGTH   = float(os.environ.get("CABLE_LENGTH", 1.0))    # cable length [m]
N_LINKS  = int(os.environ.get("CABLE_LINKS", 40))        # number of capsules
RADIUS   = 0.0015                                         # cable radius [m] (1.5 mm)
E        = float(os.environ.get("CABLE_E", 40e6))        # Young's modulus [Pa]
DENSITY  = 1150.0                                         # TPU density [kg/m^3]

PULL_X   = float(os.environ.get("CABLE_PULL_X", 0.4))    # pulled point target x [m]
PULL_Z   = float(os.environ.get("CABLE_PULL_Z", 0.3))    # pulled point target z [m]
SETTLE_S = float(os.environ.get("CABLE_SETTLE", 4.0))    # settle time [s]
OUT_CSV  = os.environ.get("CABLE_OUT", "cable_learn.csv")

# --- geometry derived from the above -------------------------------------
# Each capsule covers an equal slice of the cable. A capsule's drawn length is
# its cylindrical HEIGHT plus its two hemispherical caps (each = RADIUS).
SEG_LEN  = LENGTH / N_LINKS                # length of one segment [m]
CAP_H    = max(SEG_LEN - 2 * RADIUS, 1e-4)  # cylinder height (caps add 2*RADIUS)
ANCHOR_Z = 1.0                              # height the anchor hangs from [m]

# --- mass per capsule: density * volume of a capsule (cylinder + sphere) ---
VOL_CAP  = math.pi * RADIUS**2 * CAP_H + (4.0 / 3.0) * math.pi * RADIUS**3
LINK_M   = DENSITY * VOL_CAP               # [kg]

# --- BENDING STIFFNESS: the one "how floppy" knob, from beam theory --------
# A beam resists bending with stiffness EI, where I is the area moment of a
# circular cross-section, I = pi/4 * r^4. Spread over one segment of length
# SEG_LEN, the rotational spring constant is K = EI / SEG_LEN  [N.m/rad].
# USD's DriveAPI wants stiffness in N.m/DEGREE, so we convert (* pi/180).
I_AREA   = math.pi / 4.0 * RADIUS**4
K_BEND   = E * I_AREA / SEG_LEN                       # [N.m/rad]
STIFFNESS = K_BEND * math.pi / 180.0                 # [N.m/deg] for DriveAPI
DAMPING   = 0.1 * STIFFNESS                           # gentle damping (settles)

# physics timestep: small enough to stay stable for a stiff thin cable.
PHYSICS_DT = 1.0 / 240.0

print(f"[cable_learn] {N_LINKS} capsules, seg {SEG_LEN*100:.1f} cm, "
      f"mass/seg {LINK_M*1000:.2f} g, K_bend {K_BEND:.2e} N.m/rad")


# ==========================================================================
# BUILD THE SCENE
# ==========================================================================
world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT)
world.scene.add_default_ground_plane()        # the FLOOR the cable rests on
stage = world.stage


def make_capsule(i: int) -> DynamicCapsule:
    """Create capsule i, laid out horizontally along +x from the anchor.

    We START the cable as a straight horizontal line at z=ANCHOR_Z. Gravity
    and the pulled point then deform it into its real draped shape. The capsule
    is drawn along its LOCAL Z axis, so we rotate it 90 deg about Y to lie
    along world +x (orientation is quaternion [w, x, y, z]).
    """
    x = (i + 0.5) * SEG_LEN                    # centre of segment i
    capsule = world.scene.add(
        DynamicCapsule(
            prim_path=f"/World/cap_{i}",
            name=f"cap_{i}",
            position=np.array([x, 0.0, ANCHOR_Z]),
            orientation=np.array([0.70710678, 0.0, 0.70710678, 0.0]),  # +90 about Y
            radius=RADIUS,
            height=CAP_H,
            color=np.array([0.05, 0.05, 0.05]),
            mass=LINK_M,
        )
    )
    return capsule


def make_bending_joint(i: int):
    """A D6 joint between capsule i and capsule i+1.

    A D6 joint has 6 DOF: 3 translations + 3 rotations. For a cable we want:
      * NO translation  -> LOCK transX/Y/Z (an inextensible cable).
      * bending          -> a soft spring on the two SWING axes (rotX, rotY).
      * twist (rotZ)     -> leave free (a cable spins freely about its own axis).

    The two bodies are connected at the point where the capsules touch: the
    +x end of capsule i and the -x end of capsule i+1. In each capsule's LOCAL
    frame (capsule drawn along local Z) that contact point is at local +Z and
    -Z respectively, half a segment away from the centre.
    """
    joint = UsdPhysics.Joint.Define(stage, f"/World/joint_{i}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/cap_{i}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/cap_{i + 1}")])
    # local attach points: half a segment along each capsule's own axis (local Z)
    half = SEG_LEN / 2.0
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, +half))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -half))
    joint.CreateCollisionEnabledAttr().Set(False)  # neighbours don't self-collide
    prim = joint.GetPrim()

    # LOCK the three translation axes: set low > high, which USD reads as "locked".
    for axis in ("transX", "transY", "transZ"):
        lim = UsdPhysics.LimitAPI.Apply(prim, axis)
        lim.CreateLowAttr().Set(1.0)
        lim.CreateHighAttr().Set(-1.0)   # low>high => axis is locked

    # BENDING: a soft spring+damper on the two swing axes. This is the cable's
    # stiffness -- low E = floppy, high E = wire-like.
    for axis in ("rotX", "rotY"):
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(STIFFNESS)
        drive.CreateDampingAttr().Set(DAMPING)
        drive.CreateMaxForceAttr().Set(1e6)
    # rotZ (twist) is left free -- no drive, no limit.


def weld_anchor():
    """Fix capsule 0 to the world so the cable hangs from a fixed anchor."""
    fj = UsdPhysics.FixedJoint.Define(stage, "/World/anchor")
    fj.CreateBody1Rel().SetTargets([Sdf.Path("/World/cap_0")])


# --- build everything -----------------------------------------------------
capsules = [make_capsule(i) for i in range(N_LINKS)]
for i in range(N_LINKS - 1):
    make_bending_joint(i)
weld_anchor()

# The PULLED point: the last capsule. We make it KINEMATIC (not pushed by
# physics; we set its pose directly) so we can place it like you pulling the
# cable to a known spot. Everything BETWEEN anchor and pull is free to sag.
PULLED = capsules[-1]


# ==========================================================================
# RUN: place the pulled point, let it settle, log the profile.
# ==========================================================================
def set_pulled_pose():
    """Move the pulled capsule to (PULL_X, 0, PULL_Z), kept kinematic."""
    PULLED.set_world_pose(
        position=np.array([PULL_X, 0.0, PULL_Z]),
        orientation=np.array([0.70710678, 0.0, 0.70710678, 0.0]),
    )


def read_profile():
    """Return (x_m, z_m) arrays of every capsule centre, ordered along cable."""
    xs, zs = [], []
    for cap in capsules:
        pos, _ = cap.get_world_pose()
        xs.append(float(pos[0]))
        zs.append(float(pos[2]))   # z = height (y stays ~0: it's a 2-D plane)
    return np.array(xs), np.array(zs)


def write_csv(path, xs, zs):
    # Shift so the anchor is the origin (x=0) -- matches extract_cable_profile.
    x0, z0 = xs[0], zs[0]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_m", "z_m"])
        for x, z in zip(xs, zs):
            w.writerow([f"{x - x0:.6f}", f"{z - z0:.6f}"])
    print(f"[cable_learn] wrote {len(xs)} points -> {path}")


def main():
    world.reset()                 # initialise physics with everything above
    # Make the last capsule kinematic AFTER reset (so the handle is positioned
    # by us, not by gravity), then place it at the pull target.
    PULLED.set_collision_enabled(True)
    set_pulled_pose()

    n_steps = int(SETTLE_S / PHYSICS_DT)
    print(f"[cable_learn] settling for {SETTLE_S}s ({n_steps} steps)...")
    for step in range(n_steps):
        set_pulled_pose()         # hold the pulled point in place each step
        world.step(render=not HEADLESS)

    xs, zs = read_profile()
    write_csv(OUT_CSV, xs, zs)

    # arc length is a quick sanity check: should be ~ LENGTH (cable is ~inextensible)
    arc = float(np.sum(np.hypot(np.diff(xs), np.diff(zs))))
    print(f"[cable_learn] settled. arc length {arc:.3f} m (cable is {LENGTH} m)")

    if not HEADLESS:
        print("[cable_learn] holding the GUI open; Ctrl-C to quit.")
        while simulation_app.is_running():
            set_pulled_pose()
            world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
