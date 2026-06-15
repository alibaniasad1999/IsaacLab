"""
Two Franka robots connected by a flexible cable -- method comparison test.

Two Franka Panda arms face each other and hold the two ends of a 0.6 m
flexible cable. After a short settle, the LEADER arm swings its base joint
(panda_joint1) sinusoidally while the FOLLOWER holds its pose, so the cable
is dragged side to side. The SAME experiment runs with ALL THREE cable models:

  CABLE_METHOD=capsule      rigid capsule chain + D6 joints      (cable.py)
  CABLE_METHOD=deformable   FEM deformable cylinder, fattened    (cable_fem.py)
  CABLE_METHOD=warp         1-D Cosserat/XPBD elastic rod (Warp) (cable_warp.py)

All physical targets come from cable_config.py (E, radius, density). The capsule
and FEM cables are real PhysX bodies jointed/attached to the panda_hand TCPs; the
Warp rod is a 1-D rod whose two end nodes are pinned to the two TCPs every frame
(it sags/bends between them but exerts no force back on the arms). FEM fattens its
rod (12 mm) and rescales E/density so it still bends/weighs like the 1.5 mm cable.

All three write the same trajectory.csv / summary.json layout (incl. wall-clock
realtime factor) into cable_output/two_robots_<method>/, so the three methods can
be compared directly.

Run (GUI):
    conda activate env_isaaclab
    CABLE_METHOD=capsule    python scripts/cable_simulation/cable_two_robots.py
    CABLE_METHOD=deformable python scripts/cable_simulation/cable_two_robots.py
    CABLE_METHOD=warp       python scripts/cable_simulation/cable_two_robots.py

Headless sweep:
    CABLE_HEADLESS=1 CABLE_MAX_TIME=10 CABLE_METHOD=warp \
        python scripts/cable_simulation/cable_two_robots.py
"""

import argparse
import os
import json
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Two robots connected by a cable.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

HEADLESS = os.environ.get("CABLE_HEADLESS", "0") == "1"
if HEADLESS:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Imports after SimulationApp --
import numpy as np
import torch

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from isaacsim.core.prims import RigidPrim as RigidPrimView
from pxr import UsdPhysics, PhysxSchema, Gf, Sdf, UsdGeom


# ===============================================================
# 1. CONFIGURATION
# ===============================================================
CABLE_METHOD       = os.environ.get("CABLE_METHOD", "capsule")
assert CABLE_METHOD in ("capsule", "deformable", "warp"), \
    f"Unknown CABLE_METHOD: {CABLE_METHOD} (capsule | deformable | warp)"

# -- Cable geometry / material (same TPU cable as cable.py) --
CABLE_LENGTH       = float(os.environ.get("CABLE_LENGTH",   0.6))      # m
CABLE_RADIUS       = float(os.environ.get("CABLE_RADIUS",   1.5e-3))   # m
YOUNG_MODULUS      = float(os.environ.get("CABLE_E",        40e6))     # Pa
DENSITY            = float(os.environ.get("CABLE_DENSITY",  1150.0))   # kg/m^3
DAMPING_RATIO      = float(os.environ.get("CABLE_ZETA",     0.05))
POISSON_RATIO      = float(os.environ.get("CABLE_NU",       0.3))      # FEM only
ELASTICITY_DAMPING = float(os.environ.get("CABLE_EDAMP",    0.005))    # FEM only

# -- Capsule-chain parameters --
NUM_LINKS          = int(os.environ.get("CABLE_NUM_LINKS",  60))
POS_ITERS          = int(os.environ.get("CABLE_POS_ITERS",  32))
VEL_ITERS          = int(os.environ.get("CABLE_VEL_ITERS",  4))  # TGS caps at 4
LINEAR_DAMPING     = 0.05
ANGULAR_DAMPING    = 0.10

# -- FEM parameters --
# FEM MUST fatten the rod: PhysX voxelises it and needs >=3 hexes across the
# diameter to bend (else it's a rigid stick / truncates), which a 1.5 mm rod can't
# reach within the cooker's resolution cap. So simulate a FAT rod (FEM_SIM_RADIUS)
# and rescale E by (r/R)^exp and density by (r/R)^2 so it still bends/weighs like
# the real 1.5 mm cable (same trick as cable_fem.py).
FEM_SIM_RADIUS     = float(os.environ.get("CABLE_FEM_SIM_RADIUS", 12e-3))   # fat rod
# 4 = EXACT EI (floppy, bends like a real cable -- default here since there's no
# contact, just hanging between grippers); 2.5 = firm (was too rigid at the base).
FEM_E_EXP          = float(os.environ.get("CABLE_FEM_E_EXP", 4.0))
FEM_E_MIN          = float(os.environ.get("CABLE_FEM_E_MIN", 5.0e3))        # Pa floor (low=soft)
_fem_ratio         = CABLE_RADIUS / FEM_SIM_RADIUS                          # < 1
FEM_E              = max(YOUNG_MODULUS * _fem_ratio ** FEM_E_EXP, FEM_E_MIN)  # Pa
FEM_DENSITY        = max(DENSITY * _fem_ratio ** 2, 11.5)                   # kg/m^3 (floored)
# Resolution so the fat diameter spans ~3 hexes: res ~ 3*L/(2*R), capped at 130.
HEX_RESOLUTION     = int(os.environ.get(
    "CABLE_HEX_RES",
    min(max(int(math.ceil(3.0 * CABLE_LENGTH / (2.0 * FEM_SIM_RADIUS))), 16), 130)))
FEM_SOLVER_ITERS   = int(os.environ.get("CABLE_SOLVER_ITERS", 60))
VERTEX_DAMPING     = float(os.environ.get("CABLE_VDAMP",     0.05))
# Smaller grab region = more PIN-like (the cable can droop right at the gripper)
# instead of CLAMP-like (a big rigid grabbed chunk that leaves the hand straight
# then curves -- the "rigid at the base" look). 0.025 droops naturally yet holds.
ATTACH_OVERLAP     = float(os.environ.get("CABLE_ATTACH_OVERLAP", 0.025))  # grab radius near hand
# The FEM rod is simulated FAT but rendered THIN so it LOOKS like the capsule/warp
# cables (same trick as cable_fem.py): hide the fat sim mesh and draw a thin tube
# along its deformed centreline. Set CABLE_THIN_VISUAL=0 to see the true fat rod.
FEM_THIN_VISUAL    = os.environ.get("CABLE_THIN_VISUAL", "1") == "1"
VIS_RADIUS         = float(os.environ.get("CABLE_VIS_RADIUS", CABLE_RADIUS))  # thin look
THIN_NODES         = int(os.environ.get("CABLE_VIS_NODES", 40))   # centreline samples

# -- Robot / trajectory --
TRAJ_AMPLITUDE_RAD = float(os.environ.get("CABLE_TRAJ_AMP",  0.3))   # joint1 swing
TRAJ_FREQUENCY_HZ  = float(os.environ.get("CABLE_TRAJ_FREQ", 0.5))
SETTLE_SECONDS     = float(os.environ.get("CABLE_SETTLE",    1.0))

# Franka panda_hand world pose at the FRANKA_PANDA_HIGH_PD_CFG default init
# configuration with the base at the origin (measured once via sim.reset():
# pos=(0.0896, -0.0010, 0.9288), hand z axis pointing straight down). The
# TCP sits a further 0.1034 m along the hand z, i.e. 0.1034 m BELOW the hand.
HAND_LOCAL_POS     = np.array([float(os.environ.get("CABLE_HAND_X", 0.0896)),
                               float(os.environ.get("CABLE_HAND_Y", -0.0010)),
                               float(os.environ.get("CABLE_HAND_Z", 0.9288))])
TCP_OFFSET_IN_HAND = np.array([0.0, 0.0, 0.1034])   # panda TCP in hand frame
HAND_TCP_DROP      = 0.1034   # hand z axis points (mostly) down at init pose

# -- Time stepping --
PHYSICS_DT         = float(os.environ.get("CABLE_PHYSICS_DT", 1.0/240.0))
RENDER_DT          = float(os.environ.get("CABLE_RENDER_DT",  1.0/60.0))
MAX_SIM_TIME       = float(os.environ.get("CABLE_MAX_TIME",   10.0))

# -- Stability monitor (true divergence reaches 1e8+; physical whips ~1e4) --
DIVERGENCE_OMEGA_DEG_S = float(os.environ.get("CABLE_DIV_OMEGA", 1.0e6))
DIVERGENCE_VEL         = 1.0e4   # deformable: vertex velocity threshold m/s

# -- Output --
SCRIPT_DIR  = Path(__file__).parent
OUTPUT_DIR  = Path(os.environ.get(
    "CABLE_OUTPUT_DIR",
    str(SCRIPT_DIR / "cable_output" / f"two_robots_{CABLE_METHOD}")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH     = OUTPUT_DIR / "trajectory.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


# ===============================================================
# 2. DERIVED PARAMETERS
# ===============================================================
SEGMENT_SPACING  = CABLE_LENGTH / NUM_LINKS
LINK_HEIGHT      = max(SEGMENT_SPACING - 2.0 * CABLE_RADIUS, 1e-4)
CABLE_MASS       = DENSITY * math.pi * CABLE_RADIUS**2 * CABLE_LENGTH
LINK_MASS        = CABLE_MASS / NUM_LINKS

# Beam-theory bending spring (same derivation as cable.py)
AREA_MOMENT      = math.pi * CABLE_RADIUS**4 / 4.0
EI               = YOUNG_MODULUS * AREA_MOMENT
K_BEND_RAD       = EI / SEGMENT_SPACING
JOINT_STIFFNESS  = K_BEND_RAD * math.pi / 180.0           # N.m/deg
LINK_ROT_INERTIA = (1.0/3.0) * LINK_MASS * SEGMENT_SPACING**2
C_CRIT_RAD       = 2.0 * math.sqrt(max(K_BEND_RAD * LINK_ROT_INERTIA, 1e-30))
JOINT_DAMPING    = DAMPING_RATIO * C_CRIT_RAD * math.pi / 180.0   # N.m.s/deg

# Robot base placement: hands face each other along x, TCPs CABLE_LENGTH
# apart. Leader base at -BASE_X (yaw 0), follower at +BASE_X (yaw 180 deg).
TCP_LOCAL = HAND_LOCAL_POS + np.array([0.0, 0.0, -HAND_TCP_DROP])  # in base frame
BASE_X    = CABLE_LENGTH / 2.0 + TCP_LOCAL[0]
ANCHOR_LEAD = np.array([-CABLE_LENGTH / 2.0,  TCP_LOCAL[1], TCP_LOCAL[2]])
ANCHOR_FOLW = np.array([+CABLE_LENGTH / 2.0, -TCP_LOCAL[1], TCP_LOCAL[2]])
CABLE_MID   = 0.5 * (ANCHOR_LEAD + ANCHOR_FOLW)


def quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion (w, x, y, z)."""
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def to_np(x) -> np.ndarray:
    """Torch tensor (possibly CUDA) or array-like -> numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ===============================================================
# 3. SCENE
# ===============================================================
def spawn_robots():
    leader_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Leader")
    leader_cfg.init_state = leader_cfg.init_state.replace(
        pos=(-BASE_X, 0.0, 0.0))
    follower_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Follower")
    follower_cfg.init_state = follower_cfg.init_state.replace(
        pos=(+BASE_X, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0))   # yaw 180 deg
    return Articulation(leader_cfg), Articulation(follower_cfg)


def build_capsule_cable(stage):
    """Rigid capsule chain along +x between the two TCP anchors."""
    capsule_cfg = sim_utils.CapsuleCfg(
        radius=CABLE_RADIUS,
        height=LINK_HEIGHT,
        axis="X",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=LINEAR_DAMPING,
            angular_damping=ANGULAR_DAMPING,
            solver_position_iteration_count=POS_ITERS,
            solver_velocity_iteration_count=VEL_ITERS,
            sleep_threshold=1e-5,
            stabilization_threshold=1e-6,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=LINK_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
    )
    paths = []
    for i in range(NUM_LINKS):
        cx = ANCHOR_LEAD[0] + (i + 0.5) * SEGMENT_SPACING
        path = f"/World/Cable/capsule_{i}"
        capsule_cfg.func(path, capsule_cfg,
                         translation=(cx, CABLE_MID[1], CABLE_MID[2]))
        paths.append(path)

    # Link joints: capsule axis is X, so twist = rotX (free) and the two
    # bending axes are rotY / rotZ (EI/L spring + damper). Translations
    # locked (inextensible cable -- see cable.py for why).
    half = LINK_HEIGHT / 2.0 + CABLE_RADIUS
    for i in range(NUM_LINKS - 1):
        joint = UsdPhysics.Joint.Define(stage, f"/World/Cable/link_joint_{i}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(paths[i])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(paths[i + 1])])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(+half, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-half, 0.0, 0.0))
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        prim = joint.GetPrim()
        for axis in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, axis)
            lim.CreateLowAttr().Set(1.0)
            lim.CreateHighAttr().Set(-1.0)      # inverted => locked
        for axis in ("rotY", "rotZ"):
            drive = UsdPhysics.DriveAPI.Apply(prim, axis)
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(JOINT_STIFFNESS)
            drive.CreateDampingAttr().Set(JOINT_DAMPING)
            drive.CreateMaxForceAttr().Set(1e6)

    # End joints: ball joint (translations locked, rotations free) between
    # each hand TCP and the matching cable end. Rotations are left free so
    # the differing hand/capsule frame orientations don't fight each other.
    for name, hand_path, cap_path, cap_local in (
        ("leader",   "/World/Leader/panda_hand",   paths[0],  Gf.Vec3f(-half, 0, 0)),
        ("follower", "/World/Follower/panda_hand", paths[-1], Gf.Vec3f(+half, 0, 0)),
    ):
        joint = UsdPhysics.Joint.Define(stage, f"/World/Cable/joint_{name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(cap_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*TCP_OFFSET_IN_HAND))
        joint.CreateLocalPos1Attr().Set(cap_local)
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        prim = joint.GetPrim()
        for axis in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, axis)
            lim.CreateLowAttr().Set(1.0)
            lim.CreateHighAttr().Set(-1.0)

    return paths


def build_deformable_cable(stage):
    """FEM cylinder between the TCP anchors, auto-attached to both hands."""
    cable_cfg = DeformableObjectCfg(
        prim_path="/World/Cable",
        spawn=sim_utils.MeshCylinderCfg(
            radius=FEM_SIM_RADIUS,          # FAT rod (see FEM params); rescaled below
            height=CABLE_LENGTH,
            axis="X",
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                self_collision=False,
                solver_position_iteration_count=FEM_SOLVER_ITERS,
                vertex_velocity_damping=VERTEX_DAMPING,
                simulation_hexahedral_resolution=HEX_RESOLUTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.8)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=FEM_E,       # rescaled so fat rod bends like 1.5 mm cable
                poissons_ratio=POISSON_RATIO,
                density=FEM_DENSITY,        # rescaled so it keeps the real ~5 g mass
                elasticity_damping=ELASTICITY_DAMPING,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=tuple(CABLE_MID),
        ),
    )
    cable = DeformableObject(cfg=cable_cfg)

    # Auto-attachments: grab cable vertices near each hand's collision shapes.
    # NOTE: actor0 must be the prim carrying the deformable-body API (the
    # mesh prim under the cfg path), not the root Xform -- otherwise the
    # attachment silently grabs nothing and the cable falls off the hands.
    from pxr import Usd
    mesh_path = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Cable")):
        if prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
            mesh_path = prim.GetPath()
            break
    if mesh_path is None:
        raise RuntimeError("No PhysxDeformableBodyAPI prim under /World/Cable")

    for name, hand_path in (("leader",   "/World/Leader/panda_hand"),
                            ("follower", "/World/Follower/panda_hand")):
        att = PhysxSchema.PhysxPhysicsAttachment.Define(
            stage, f"{mesh_path}/attachment_{name}")
        att.GetActor0Rel().SetTargets([mesh_path])
        att.GetActor1Rel().SetTargets([Sdf.Path(hand_path)])
        auto = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
        auto.CreateEnableDeformableVertexAttachmentsAttr().Set(True)
        auto.CreateDeformableVertexOverlapOffsetAttr().Set(ATTACH_OVERLAP)

    return cable


# ===============================================================
# 3c. WARP COSSERAT ROD  (1-D XPBD elastic rod -- the cable_warp.py model)
# ===============================================================
# A thin, inextensible elastic rod (no FEM fattening, no PhysX). Its two end nodes
# are pinned to the two gripper TCPs every frame; it sags/bends under gravity in
# between. Rendered as a thin swept tube. Same kernels as cable_warp.py.
ROD_NODES    = int(os.environ.get("ROD_NODES", 48))
ROD_RADIUS   = float(os.environ.get("ROD_RADIUS", CABLE_RADIUS))      # thin, real radius
ROD_BEND     = float(os.environ.get("ROD_BEND_COMP", 1.0e-3))        # bending compliance
ROD_STRETCH  = float(os.environ.get("ROD_STRETCH_COMP", 1.0e-9))     # ~0 => inextensible
ROD_SUBSTEPS = int(os.environ.get("ROD_SUBSTEPS", 16))
ROD_ITERS    = int(os.environ.get("ROD_ITERS", 12))
ROD_DAMP     = float(os.environ.get("ROD_DAMPING", 0.02))
ROD_VIS_SEG  = 10


def _build_warp_rod_class():
    """Import warp lazily (only for the warp method) and return a WarpRod class."""
    import warp as wp
    wp.init()
    dev = "cuda:0" if wp.is_cuda_available() else "cpu"

    @wp.kernel
    def predict(x: wp.array(dtype=wp.vec3), x_prev: wp.array(dtype=wp.vec3),
                v: wp.array(dtype=wp.vec3), invm: wp.array(dtype=float),
                g: float, dt: float):
        i = wp.tid()
        x_prev[i] = x[i]
        if invm[i] > 0.0:
            vel = v[i] + wp.vec3(0.0, 0.0, g) * dt
            v[i] = vel
            x[i] = x[i] + vel * dt

    @wp.kernel
    def set_pins(x: wp.array(dtype=wp.vec3), a: wp.vec3, h: wp.vec3, n: int):
        i = wp.tid()
        if i == 0:
            x[0] = a
        if i == n - 1:
            x[n - 1] = h

    @wp.kernel
    def solve_rod(x: wp.array(dtype=wp.vec3), invm: wp.array(dtype=float),
                  rest: wp.array(dtype=float), n: int, iters: int,
                  a_stretch: float, a_bend: float):
        if wp.tid() != 0:
            return
        for _it in range(iters):
            for s in range(n - 1):
                xa = x[s]; xb = x[s + 1]
                wa = invm[s]; wb = invm[s + 1]
                d = xa - xb
                ln = wp.length(d)
                if ln > 1.0e-9 and (wa + wb) > 0.0:
                    nrm = d / ln
                    C = ln - rest[s]
                    dl = -C / (wa + wb + a_stretch)
                    x[s] = xa + wa * dl * nrm
                    x[s + 1] = xb - wb * dl * nrm
            for i in range(1, n - 1):
                xa = x[i - 1]; xb = x[i]; xc = x[i + 1]
                wa = invm[i - 1]; wb = invm[i]; wc = invm[i + 1]
                mid = 0.5 * (xa + xc)
                d = xb - mid
                ln = wp.length(d)
                denom = wb + 0.25 * wa + 0.25 * wc + a_bend
                if ln > 1.0e-9 and denom > 0.0:
                    nrm = d / ln
                    dl = -ln / denom
                    x[i - 1] = xa - 0.5 * wa * dl * nrm
                    x[i] = xb + wb * dl * nrm
                    x[i + 1] = xc - 0.5 * wc * dl * nrm

    @wp.kernel
    def finalize(x: wp.array(dtype=wp.vec3), x_prev: wp.array(dtype=wp.vec3),
                 v: wp.array(dtype=wp.vec3), invm: wp.array(dtype=float),
                 dt: float, damp: float):
        i = wp.tid()
        if invm[i] > 0.0:
            v[i] = (x[i] - x_prev[i]) / dt * (1.0 - damp)

    class WarpRod:
        def __init__(self, stage, end_a, end_b):
            self.n = ROD_NODES
            init = np.linspace(end_a, end_b, self.n).astype(np.float32)
            self.x = wp.array(init, dtype=wp.vec3, device=dev)
            self.xprev = wp.zeros(self.n, dtype=wp.vec3, device=dev)
            self.v = wp.zeros(self.n, dtype=wp.vec3, device=dev)
            m_node = CABLE_MASS / self.n
            invm = np.full(self.n, 1.0 / m_node, dtype=np.float32)
            invm[0] = 0.0; invm[-1] = 0.0           # both ends pinned to grippers
            self.invm = wp.array(invm, dtype=float, device=dev)
            seg = CABLE_LENGTH / (self.n - 1)
            self.rest = wp.array(np.full(self.n - 1, seg, dtype=np.float32),
                                 dtype=float, device=dev)
            self._build_tube(stage)

        def _build_tube(self, stage):
            self.tube = UsdGeom.Mesh.Define(stage, "/World/Cable")
            counts, idx = [], []
            for i in range(self.n - 1):
                for j in range(ROD_VIS_SEG):
                    jn = (j + 1) % ROD_VIS_SEG
                    a, b = i * ROD_VIS_SEG + j, i * ROD_VIS_SEG + jn
                    d, e = (i + 1) * ROD_VIS_SEG + j, (i + 1) * ROD_VIS_SEG + jn
                    counts.append(4); idx.extend((a, b, e, d))
            self.tube.CreateFaceVertexCountsAttr(counts)
            self.tube.CreateFaceVertexIndicesAttr(idx)
            self.tube.CreatePointsAttr([Gf.Vec3f(0, 0, 0)] * (self.n * ROD_VIS_SEG))
            self.tube.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.3, 0.1)])
            self._thetas = [2.0 * math.pi * j / ROD_VIS_SEG for j in range(ROD_VIS_SEG)]

        def step(self, end_a, end_b, dt):
            a = wp.vec3(float(end_a[0]), float(end_a[1]), float(end_a[2]))
            h = wp.vec3(float(end_b[0]), float(end_b[1]), float(end_b[2]))
            sdt = dt / ROD_SUBSTEPS
            a_s = ROD_STRETCH / (sdt * sdt)
            a_b = ROD_BEND / (sdt * sdt)
            for _ in range(ROD_SUBSTEPS):
                wp.launch(predict, dim=self.n,
                          inputs=[self.x, self.xprev, self.v, self.invm, -9.81, sdt], device=dev)
                wp.launch(set_pins, dim=self.n, inputs=[self.x, a, h, self.n], device=dev)
                wp.launch(solve_rod, dim=1,
                          inputs=[self.x, self.invm, self.rest, self.n, ROD_ITERS, a_s, a_b], device=dev)
                wp.launch(set_pins, dim=self.n, inputs=[self.x, a, h, self.n], device=dev)
                wp.launch(finalize, dim=self.n,
                          inputs=[self.x, self.xprev, self.v, self.invm, sdt, ROD_DAMP], device=dev)
            P = self.x.numpy()
            self._update_tube(P)
            return P

        def _update_tube(self, P):
            tang = np.gradient(P, axis=0)
            tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
            ref = np.array([0.0, 0.0, 1.0])
            if abs(float(tang[0] @ ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            nrm = ref - tang[0] * float(tang[0] @ ref)
            nrm /= np.linalg.norm(nrm) + 1e-9
            pts = []
            for i in range(self.n):
                t = tang[i]
                nrm = nrm - t * float(t @ nrm)
                ln = np.linalg.norm(nrm)
                if ln < 1e-6:
                    r2 = np.array([0., 1., 0.]) if abs(t[2]) > 0.9 else np.array([0., 0., 1.])
                    nrm = r2 - t * float(t @ r2); ln = np.linalg.norm(nrm)
                nrm /= ln
                bn = np.cross(t, nrm)
                ci = P[i]
                for th in self._thetas:
                    p = ci + ROD_RADIUS * (math.cos(th) * nrm + math.sin(th) * bn)
                    pts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
            self.tube.GetPointsAttr().Set(pts)

    return WarpRod


# ===============================================================
# 3d. THIN VISUAL TUBE  (render the fat FEM rod as a thin cable)
# ===============================================================
_TUBE_SEG    = 10
_tube_thetas = [2.0 * math.pi * j / _TUBE_SEG for j in range(_TUBE_SEG)]


def make_thin_tube(stage, path, n_nodes, color):
    tube = UsdGeom.Mesh.Define(stage, path)
    counts, idx = [], []
    for i in range(n_nodes - 1):
        for j in range(_TUBE_SEG):
            jn = (j + 1) % _TUBE_SEG
            a, b = i * _TUBE_SEG + j, i * _TUBE_SEG + jn
            d, e = (i + 1) * _TUBE_SEG + j, (i + 1) * _TUBE_SEG + jn
            counts.append(4); idx.extend((a, b, e, d))
    tube.CreateFaceVertexCountsAttr(counts)
    tube.CreateFaceVertexIndicesAttr(idx)
    tube.CreatePointsAttr([Gf.Vec3f(0, 0, 0)] * (n_nodes * _TUBE_SEG))
    tube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return tube


def sweep_thin_tube(tube, P, radius):
    """Sweep a thin circle along centreline P (M,3) with a parallel-transport frame."""
    n = len(P)
    tang = np.gradient(P, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(tang[0] @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    nrm = ref - tang[0] * float(tang[0] @ ref)
    nrm /= np.linalg.norm(nrm) + 1e-9
    pts = []
    for i in range(n):
        t = tang[i]
        nrm = nrm - t * float(t @ nrm)
        ln = np.linalg.norm(nrm)
        if ln < 1e-6:
            r2 = np.array([0., 1., 0.]) if abs(t[2]) > 0.9 else np.array([0., 0., 1.])
            nrm = r2 - t * float(t @ r2); ln = np.linalg.norm(nrm)
        nrm /= ln
        bn = np.cross(t, nrm)
        ci = P[i]
        for th in _tube_thetas:
            p = ci + radius * (math.cos(th) * nrm + math.sin(th) * bn)
            pts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
    tube.GetPointsAttr().Set(pts)


def fem_centerline(nodal_pos, rest_x, nb):
    """Ordered centreline (nb,3) of the FEM rod: bin its nodes by their REST x and
    average each slice (carry the last centroid forward for any empty slice)."""
    order = np.argsort(rest_x)
    xs = rest_x[order]
    pts = nodal_pos[order]
    edges = np.linspace(xs[0], xs[-1], nb + 1)
    cents = np.zeros((nb, 3), dtype=np.float64)
    last = pts[0]
    for i in range(nb):
        m = (xs >= edges[i]) & (xs <= edges[i + 1])
        if m.any():
            last = pts[m].mean(axis=0)
        cents[i] = last
    return cents


# ===============================================================
# 4. MAIN
# ===============================================================
def main():
    print("=" * 70)
    print(f"Two robots + cable  --  method = {CABLE_METHOD}")
    print("=" * 70)
    print(f"  cable length    : {CABLE_LENGTH} m   radius {CABLE_RADIUS*1e3:.1f} mm")
    print(f"  E / rho         : {YOUNG_MODULUS/1e6:.0f} MPa / {DENSITY:.0f} kg/m^3")
    print(f"  cable mass      : {CABLE_MASS*1e3:.1f} g")
    if CABLE_METHOD == "capsule":
        print(f"  links           : {NUM_LINKS}  (k_bend={JOINT_STIFFNESS:.2e} N.m/deg)")
    elif CABLE_METHOD == "deformable":
        print(f"  hex resolution  : {HEX_RESOLUTION}  (fat R={FEM_SIM_RADIUS*1e3:.0f} mm, "
              f"E_sim={FEM_E/1e3:.0f} kPa)")
    else:
        print(f"  warp rod nodes  : {ROD_NODES}  (thin R={ROD_RADIUS*1e3:.1f} mm)")
    print(f"  base separation : {2*BASE_X:.3f} m  (TCPs {CABLE_LENGTH} m apart)")
    print(f"  trajectory      : joint1 +/-{TRAJ_AMPLITUDE_RAD} rad @ {TRAJ_FREQUENCY_HZ} Hz")
    print(f"  physics dt      : {PHYSICS_DT*1e3:.2f} ms   max time {MAX_SIM_TIME}s")
    print(f"  output          : {OUTPUT_DIR}")

    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=max(1, round(RENDER_DT / PHYSICS_DT)),
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.6, 1.6, 1.4], target=[0.0, 0.0, 0.6])
    stage = omni.usd.get_context().get_stage()

    # Scene
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)

    leader, follower = spawn_robots()

    cable = capsule_paths = warp_rod = None
    if CABLE_METHOD == "capsule":
        capsule_paths = build_capsule_cable(stage)
    elif CABLE_METHOD == "deformable":
        cable = build_deformable_cable(stage)
    # warp rod is built AFTER reset (it's not a PhysX body; it needs the live TCPs)

    sim.reset()
    leader.update(PHYSICS_DT)
    follower.update(PHYSICS_DT)

    hand_idx = leader.body_names.index("panda_hand")
    j1_idx   = leader.joint_names.index("panda_joint1")

    def tcp_of(robot):
        hp = robot.data.body_pos_w[0, hand_idx].cpu().numpy()
        hq = robot.data.body_quat_w[0, hand_idx].cpu().numpy()
        return hp + quat_rotate(hq, TCP_OFFSET_IN_HAND)

    # Sanity check: how far is the assumed anchor from the actual TCP?
    for tag, robot, anchor in (("leader", leader, ANCHOR_LEAD),
                               ("follower", follower, ANCHOR_FOLW)):
        err = float(np.linalg.norm(tcp_of(robot) - anchor))
        print(f"  {tag} TCP at {np.round(tcp_of(robot), 4)}  anchor error {err*1e3:.1f} mm")

    # Cable mid-point tracking
    capsule_view = monitor_idx = None
    if CABLE_METHOD == "capsule":
        capsule_view = RigidPrimView(prim_paths_expr=capsule_paths,
                                     name="capsule_view",
                                     reset_xform_properties=False)
        capsule_view.initialize()
        mid_index = NUM_LINKS // 2
        monitor_idx = list(range(0, NUM_LINKS, max(NUM_LINKS // 10, 1)))
    elif CABLE_METHOD == "deformable":
        nodal_pos = cable.data.nodal_pos_w
        d = (nodal_pos[0] - torch.tensor(CABLE_MID, dtype=nodal_pos.dtype,
                                         device=nodal_pos.device)).norm(dim=-1)
        mid_index = int(d.argmin().item())
        # Render the FAT sim rod as a THIN tube so it looks like the other cables.
        fem_thin_tube = fem_rest_x = None
        if FEM_THIN_VISUAL:
            from pxr import Usd
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Cable")):
                if prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
                    UsdGeom.Imageable(prim).MakeInvisible()   # hide fat rod (render only)
                    break
            fem_rest_x = nodal_pos[0][:, 0].cpu().numpy()     # rest x of each node
            fem_thin_tube = make_thin_tube(stage, "/World/CableThin",
                                           THIN_NODES, (0.1, 0.3, 0.8))
            print(f"  thin visual ON: rod sim R={FEM_SIM_RADIUS*1e3:.0f} mm rendered "
                  f"as {VIS_RADIUS*1e3:.1f} mm tube (CABLE_THIN_VISUAL=0 to disable)")
    else:   # warp -- build the rod now, pinned to the two live TCPs
        WarpRod = _build_warp_rod_class()
        warp_rod = WarpRod(stage, tcp_of(leader), tcp_of(follower))
        mid_index = ROD_NODES // 2
        warp_P = np.linspace(tcp_of(leader), tcp_of(follower), ROD_NODES)

    # Joint targets = the LIVE pose right after reset (the ready pose the arms are
    # actually in), so the high-PD controller holds them THERE. Using
    # default_joint_pos drove the arms to a different folded pose (they collapsed).
    leader_targets   = leader.data.joint_pos.clone()
    follower_targets = follower.data.joint_pos.clone()
    q1_default       = float(leader_targets[0, j1_idx].item())
    print(f"  hold target captured (leader hand z should stay ~{tcp_of(leader)[2]:.2f} m)")

    # CSV
    csv_file = open(CSV_PATH, "w", newline="")
    csv_file.write("t,leader_q1_target,"
                   "leader_ee_x,leader_ee_y,leader_ee_z,"
                   "follower_ee_x,follower_ee_y,follower_ee_z,"
                   "mid_x,mid_y,mid_z,cable_span,cable_sag\n")

    render_every = max(1, round(RENDER_DT / PHYSICS_DT))
    sim_time, step_count = 0.0, 0
    stable, instability_at, max_speed = True, None, 0.0

    print(f"\nSimulating up to t = {MAX_SIM_TIME}s ...")
    wall_t0 = time.perf_counter()

    while simulation_app.is_running() and sim_time < MAX_SIM_TIME:
        # -- Leader trajectory (joint1 sine after settling) --
        if sim_time >= SETTLE_SECONDS:
            phase = 2.0 * math.pi * TRAJ_FREQUENCY_HZ * (sim_time - SETTLE_SECONDS)
            leader_targets[0, j1_idx] = q1_default + TRAJ_AMPLITUDE_RAD * math.sin(phase)
        leader.set_joint_position_target(leader_targets)
        follower.set_joint_position_target(follower_targets)
        leader.write_data_to_sim()
        follower.write_data_to_sim()

        sim.step()
        sim_time += PHYSICS_DT
        step_count += 1
        leader.update(PHYSICS_DT)
        follower.update(PHYSICS_DT)
        if cable is not None:
            cable.update(PHYSICS_DT)

        # -- Log + stability at render rate --
        if step_count % render_every == 0:
            lp = leader.data.body_pos_w[0, hand_idx].cpu().numpy()
            lq = leader.data.body_quat_w[0, hand_idx].cpu().numpy()
            fp = follower.data.body_pos_w[0, hand_idx].cpu().numpy()
            fq = follower.data.body_quat_w[0, hand_idx].cpu().numpy()
            tcp_l = lp + quat_rotate(lq, TCP_OFFSET_IN_HAND)
            tcp_f = fp + quat_rotate(fq, TCP_OFFSET_IN_HAND)
            span  = float(np.linalg.norm(tcp_f - tcp_l))

            if CABLE_METHOD == "capsule":
                mid_pos, _ = capsule_view.get_world_poses(indices=[mid_index],
                                                          usd=False)
                mid = to_np(mid_pos)[0]
                vels = to_np(capsule_view.get_velocities(indices=monitor_idx))
                speed = float(np.max(np.linalg.norm(vels[:, 3:6], axis=1))) \
                    * 180.0 / math.pi                                  # deg/s
                threshold = DIVERGENCE_OMEGA_DEG_S
            elif CABLE_METHOD == "deformable":
                nodes_np = cable.data.nodal_pos_w[0].cpu().numpy()
                mid = nodes_np[mid_index]
                vel = cable.data.nodal_state_w[0, :, 3:6]
                speed = float(vel.norm(dim=-1).max().item())           # m/s
                threshold = DIVERGENCE_VEL
                if fem_thin_tube is not None:                          # redraw thin tube
                    P = fem_centerline(nodes_np, fem_rest_x, THIN_NODES)
                    sweep_thin_tube(fem_thin_tube, P, VIS_RADIUS)
            else:   # warp -- advance the rod pinned to the two TCPs, render the tube
                warp_P = warp_rod.step(tcp_l, tcp_f, RENDER_DT)
                mid = warp_P[mid_index]
                ok = bool(np.all(np.isfinite(warp_P))) and float(np.max(np.abs(warp_P))) < 50.0
                speed = 0.0 if ok else 1.0e9
                threshold = 1.0

            max_speed = max(max_speed, speed)
            if stable and speed > threshold:
                stable, instability_at = False, sim_time
                print(f"  *** UNSTABLE at t={sim_time:.3f}s "
                      f"(speed metric {speed:.1e}) ***")

            sag = float(mid[2] - 0.5 * (tcp_l[2] + tcp_f[2]))
            csv_file.write(
                f"{sim_time:.6f},{leader_targets[0, j1_idx].item():.6f},"
                f"{tcp_l[0]:.6f},{tcp_l[1]:.6f},{tcp_l[2]:.6f},"
                f"{tcp_f[0]:.6f},{tcp_f[1]:.6f},{tcp_f[2]:.6f},"
                f"{mid[0]:.6f},{mid[1]:.6f},{mid[2]:.6f},"
                f"{span:.6f},{sag:.6f}\n")

        # -- Progress --
        if step_count % (render_every * 120) == 0:
            rtf = sim_time / max(time.perf_counter() - wall_t0, 1e-9)
            print(f"  t={sim_time:5.2f}s  "
                  f"{'STABLE' if stable else 'UNSTABLE'}  "
                  f"max speed metric={max_speed:.2e}  rtf={rtf:.2f}x")

    wall_elapsed = time.perf_counter() - wall_t0
    csv_file.close()

    print("=" * 70)
    print(f"Done: t={sim_time:.2f}s  wall={wall_elapsed:.1f}s "
          f"({sim_time / max(wall_elapsed, 1e-9):.2f}x realtime)  stable={stable}")

    summary = {
        "method":            CABLE_METHOD,
        "test":              "two_robots",
        "cable_length_m":    CABLE_LENGTH,
        "cable_radius_m":    CABLE_RADIUS,
        "cable_mass_kg":     CABLE_MASS,
        "young_modulus_pa":  YOUNG_MODULUS,
        "density_kg_m3":     DENSITY,
        "num_links":         NUM_LINKS if CABLE_METHOD == "capsule" else None,
        "hex_resolution":    HEX_RESOLUTION if CABLE_METHOD == "deformable" else None,
        "rod_nodes":         ROD_NODES if CABLE_METHOD == "warp" else None,
        "traj_amplitude_rad": TRAJ_AMPLITUDE_RAD,
        "traj_frequency_hz": TRAJ_FREQUENCY_HZ,
        "settle_seconds":    SETTLE_SECONDS,
        "physics_dt_s":      PHYSICS_DT,
        "total_sim_time_s":  sim_time,
        "wall_clock_s":      wall_elapsed,
        "realtime_factor":   sim_time / max(wall_elapsed, 1e-9),
        "stable":            stable,
        "instability_at_s":  instability_at,
        "max_speed_metric":  max_speed,
        "csv_path":          str(CSV_PATH),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang at 100% CPU on this machine; all
    # outputs are on disk by now, so force an exit if it doesn't return.
    import threading
    threading.Timer(20.0, lambda: os._exit(0)).start()
    simulation_app.close()
    os._exit(0)
