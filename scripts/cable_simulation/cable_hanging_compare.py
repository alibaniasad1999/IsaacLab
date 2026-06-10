"""
BOTH cable models side by side in ONE scene -- direct hanging comparison.

Same setting as cable.py (top end fixed, cable hanging down), but the two
methods run together so they are perfectly comparable:

  y = -0.3 :  rigid capsule chain  (cable.py model: D6 joints, EI/L drives)
  y = +0.3 :  FEM deformable cylinder (cable_deformable.py model)

Each cable carries the SAME end object (a 5 g cube fixed to the bottom end)
and receives the SAME horizontal force pulse on that object:

    t in [FORCE_START, FORCE_END):  F = (CABLE_FORCE_X, 0, 0) on both cubes

Why the FEM cable no longer behaves like a rigid bar
----------------------------------------------------
A 1 m x 3 mm rod meshed with hex elements is hopeless: even at resolution 48
the elements are ~14x longer than wide and SHEAR-LOCK, which makes the FEM
cable bend orders of magnitude too stiffly (it swings like a stick). The fix
used here is the standard equivalent-rod trick: simulate a FATTER rod with
identical bending stiffness EI and identical mass per length:

    R_sim > r_real,   E_sim   = E * (r_real / R_sim)^p    (p = CABLE_FEM_EXP)
                      rho_sim = rho * (r_real / R_sim)^2  (same kg/m)
    p = 4 matches EI exactly (but EA gets far too soft -> rubber-banding);
    p = 3 (default) trades 10x bending stiffness for sane axial stiffness.

Two ingredients are BOTH required, otherwise the FEM cable acts rigid:
  1. R_sim = 6 mm fat rod (above) so elements can be near-cubic, AND
  2. voxel resolution high enough for >= 3 elements ACROSS the diameter
     (default 250 -> 4 mm voxels on a 12 mm-wide rod). With one element
     across the width there is no bending strain gradient to resolve and
     no value of E makes it floppy.
(Trade-off: axial stiffness EA ends up (r/R)^2 ~ 16x too soft -- a few mm
of extra stretch under these gram-scale loads.)

Outputs -> cable_output/hanging_compare/:
    trajectory.csv   t, tip + midpoint positions of both cables
    summary.json     stability + timing
    comparison.png   tip/mid trajectories overlaid

Run (GUI):
    conda activate env_isaaclab
    python scripts/cable_simulation/cable_hanging_compare.py

Headless:
    CABLE_HEADLESS=1 CABLE_MAX_TIME=8 \
        python scripts/cable_simulation/cable_hanging_compare.py
"""

import argparse
import os
import json
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capsule vs FEM cable, one scene.")
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
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext

from isaacsim.core.prims import RigidPrim as RigidPrimView
from pxr import UsdPhysics, PhysxSchema, Gf, Sdf


# ===============================================================
# 1. CONFIGURATION  (same cable as cable.py)
# ===============================================================
CABLE_LENGTH   = float(os.environ.get("CABLE_LENGTH",   1.0))      # m
CABLE_RADIUS   = float(os.environ.get("CABLE_RADIUS",   1.5e-3))   # m (real)
YOUNG_MODULUS  = float(os.environ.get("CABLE_E",        40e6))     # Pa (TPU)
DENSITY        = float(os.environ.get("CABLE_DENSITY",  1150.0))   # kg/m^3
DAMPING_RATIO  = float(os.environ.get("CABLE_ZETA",     0.05))

ANCHOR_Z       = 2.0
Y_CAPSULE      = -0.3      # capsule chain lane
Y_FEM          = +0.3      # FEM cable lane

# -- End object: small cube fixed to the bottom end of each cable --
# (4 cm so it stays visible next to the fat FEM rod; mass is what matters.)
END_CUBE_SIZE  = float(os.environ.get("CABLE_END_SIZE", 0.04))
END_CUBE_MASS  = float(os.environ.get("CABLE_END_MASS", 0.005))    # 5 g clip

# -- Force pulse on both end cubes --
FORCE_X        = float(os.environ.get("CABLE_FORCE_X",  0.05))     # N
FORCE_START    = float(os.environ.get("CABLE_FORCE_T0", 1.0))      # s
FORCE_END      = float(os.environ.get("CABLE_FORCE_T1", 2.0))      # s

# -- Capsule chain (cable.py parameters) --
NUM_LINKS      = int(os.environ.get("CABLE_NUM_LINKS",  200))
POS_ITERS      = int(os.environ.get("CABLE_POS_ITERS",  32))
VEL_ITERS      = int(os.environ.get("CABLE_VEL_ITERS",  4))
LIN_DAMP       = float(os.environ.get("CABLE_LIN_DAMP", 0.05))
ANG_DAMP       = float(os.environ.get("CABLE_ANG_DAMP", 0.5))
MAX_OMEGA_DEG  = float(os.environ.get("CABLE_MAX_OMEGA", 2.0e4))   # deg/s

# -- FEM cable: equivalent fat rod (see module docstring) --
FEM_SIM_RADIUS = float(os.environ.get("CABLE_FEM_RADIUS", 15e-3))  # m
_scale         = CABLE_RADIUS / FEM_SIM_RADIUS
# E scaling exponent: 4 matches EI exactly but leaves EA (r/R)^2 = 100x too
# soft -- the rod then stretches like a rubber band (observed: tip dropped
# 0.35 m during the swing). Exponent 3 is the hanging-cable compromise:
# EI ends up (R/r) = 10x stiff (still drapes -- bending boundary layer
# sqrt(EI/T) ~ 13 cm), EA only 10x soft (~mm-scale stretch).
FEM_SCALE_EXP  = float(os.environ.get("CABLE_FEM_EXP",   3.0))
FEM_E          = YOUNG_MODULUS * _scale**FEM_SCALE_EXP
FEM_DENSITY    = DENSITY * _scale**2              # same mass / length
FEM_NU         = float(os.environ.get("CABLE_NU",       0.3))
FEM_EDAMP      = float(os.environ.get("CABLE_EDAMP",    0.005))
# Voxel resolution along the LONGEST axis (the 1 m length). Bending fidelity
# requires >= 3 hex elements ACROSS the rod diameter, i.e.
#   res >= 3 * L / (2 * R_sim)  =  3 * 1.0 / 0.030  =  100  for R_sim = 15 mm.
# Anything coarser leaves a single element across the width, which cannot
# represent a bending strain gradient -> the cable behaves like a rigid bar
# no matter what E is. (PxTetMaker fails above ~res 200 -- the voxel grid
# grows cubically -- which is why the rod is fattened to 15 mm instead of
# pushing the resolution.)
HEX_RESOLUTION = int(os.environ.get("CABLE_HEX_RES",    100))
FEM_ITERS      = int(os.environ.get("CABLE_SOLVER_ITERS", 32))
VERTEX_DAMPING = float(os.environ.get("CABLE_VDAMP",    0.005))
ATTACH_OVERLAP = 0.01

# -- Time stepping (cable.py rationale: light links need 480 Hz) --
PHYSICS_DT     = float(os.environ.get("CABLE_PHYSICS_DT", 1.0/480.0))
RENDER_DT      = float(os.environ.get("CABLE_RENDER_DT",  1.0/60.0))
MAX_SIM_TIME   = float(os.environ.get("CABLE_MAX_TIME",   8.0))

DIVERGENCE_POS_M = 10.0

# -- Output --
SCRIPT_DIR  = Path(__file__).parent
OUTPUT_DIR  = Path(os.environ.get("CABLE_OUTPUT_DIR",
              str(SCRIPT_DIR / "cable_output" / "hanging_compare")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH     = OUTPUT_DIR / "trajectory.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
PLOT_PATH    = OUTPUT_DIR / "comparison.png"

# -- Derived capsule-chain parameters (same formulas as cable.py) --
SEG          = CABLE_LENGTH / NUM_LINKS
LINK_HEIGHT  = max(SEG - 2.0 * CABLE_RADIUS, 1e-4)
LINK_MASS    = DENSITY * math.pi * CABLE_RADIUS**2 * CABLE_LENGTH / NUM_LINKS
EI           = YOUNG_MODULUS * math.pi * CABLE_RADIUS**4 / 4.0
K_BEND_RAD   = EI / SEG
K_DRIVE      = K_BEND_RAD * math.pi / 180.0                       # N.m/deg
I_ROT        = (1.0/3.0) * LINK_MASS * SEG**2
C_DRIVE      = DAMPING_RATIO * 2.0 * math.sqrt(K_BEND_RAD * I_ROT) * math.pi / 180.0


def to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ===============================================================
# 2. CAPSULE CHAIN  (vertical, hanging at y = Y_CAPSULE)
# ===============================================================
def build_capsule_cable(stage):
    half = LINK_HEIGHT / 2.0 + CABLE_RADIUS
    capsule_cfg = sim_utils.CapsuleCfg(
        radius=CABLE_RADIUS,
        height=LINK_HEIGHT,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=LIN_DAMP,
            angular_damping=ANG_DAMP,
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
        cz = ANCHOR_Z - i * SEG - CABLE_RADIUS - LINK_HEIGHT / 2.0
        path = f"/World/CapCable/capsule_{i}"
        capsule_cfg.func(path, capsule_cfg, translation=(0.0, Y_CAPSULE, cz))
        # Max angular velocity: PhysX takes rad/s here (cable.py, verified).
        prim = stage.GetPrimAtPath(path)
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.CreateMaxAngularVelocityAttr().Set(MAX_OMEGA_DEG * math.pi / 180.0)
        paths.append(path)

    def make_bend_twist(prim):
        """Bending springs on rotX/rotY, damping-only twist on rotZ."""
        for axis in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, axis)
            lim.CreateLowAttr().Set(1.0)
            lim.CreateHighAttr().Set(-1.0)        # inverted => locked
        for axis in ("rotX", "rotY"):
            drive = UsdPhysics.DriveAPI.Apply(prim, axis)
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(K_DRIVE)
            drive.CreateDampingAttr().Set(C_DRIVE)
            drive.CreateMaxForceAttr().Set(1e6)
        drive = UsdPhysics.DriveAPI.Apply(prim, "rotZ")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(C_DRIVE)
        drive.CreateMaxForceAttr().Set(1e6)

    # Top end -> world (single-body joint = anchored to the world frame)
    joint = UsdPhysics.Joint.Define(stage, "/World/CapCable/anchor_joint")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(paths[0])])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, Y_CAPSULE, ANCHOR_Z))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +(LINK_HEIGHT / 2 + CABLE_RADIUS)))
    joint.CreateExcludeFromArticulationAttr().Set(True)
    make_bend_twist(joint.GetPrim())

    # Link joints
    for i in range(NUM_LINKS - 1):
        joint = UsdPhysics.Joint.Define(stage, f"/World/CapCable/link_joint_{i}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(paths[i])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(paths[i + 1])])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -half))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +half))
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        make_bend_twist(joint.GetPrim())

    # End cube fixed to the last capsule
    cube_z = ANCHOR_Z - CABLE_LENGTH - END_CUBE_SIZE / 2.0
    cube_cfg = sim_utils.CuboidCfg(
        size=(END_CUBE_SIZE,) * 3,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=LIN_DAMP, angular_damping=ANG_DAMP,
            solver_position_iteration_count=POS_ITERS,
            solver_velocity_iteration_count=VEL_ITERS,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=END_CUBE_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
    )
    cube_cfg.func("/World/CapCable/end_cube", cube_cfg,
                  translation=(0.0, Y_CAPSULE, cube_z))
    fj = UsdPhysics.FixedJoint.Define(stage, "/World/CapCable/end_joint")
    fj.CreateBody0Rel().SetTargets([Sdf.Path(paths[-1])])
    fj.CreateBody1Rel().SetTargets([Sdf.Path("/World/CapCable/end_cube")])
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -half))
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, +END_CUBE_SIZE / 2.0))
    fj.CreateCollisionEnabledAttr().Set(False)
    fj.CreateExcludeFromArticulationAttr().Set(True)

    return paths


# ===============================================================
# 3. FEM CABLE  (equivalent-EI fat rod at y = Y_FEM)
# ===============================================================
def build_fem_cable(stage):
    cable_cfg = DeformableObjectCfg(
        prim_path="/World/FemCable",
        spawn=sim_utils.MeshCylinderCfg(
            radius=FEM_SIM_RADIUS,
            height=CABLE_LENGTH,
            axis="Z",
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                self_collision=False,
                solver_position_iteration_count=FEM_ITERS,
                vertex_velocity_damping=VERTEX_DAMPING,
                simulation_hexahedral_resolution=HEX_RESOLUTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.8)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=FEM_E,
                poissons_ratio=FEM_NU,
                density=FEM_DENSITY,
                elasticity_damping=FEM_EDAMP,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, Y_FEM, ANCHOR_Z - CABLE_LENGTH / 2.0),
        ),
    )
    cable = DeformableObject(cfg=cable_cfg)

    # End cube overlapping the cylinder bottom, auto-attached
    cube_z = ANCHOR_Z - CABLE_LENGTH - END_CUBE_SIZE / 2.0 + 0.005
    cube_cfg = sim_utils.CuboidCfg(
        size=(END_CUBE_SIZE,) * 3,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=LIN_DAMP, angular_damping=ANG_DAMP,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=END_CUBE_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
    )
    cube_cfg.func("/World/FemEndCube", cube_cfg, translation=(0.0, Y_FEM, cube_z))

    # The deformable-body API sits on the MESH prim spawned somewhere under
    # the cfg prim path (not on the root Xform). Targeting the root makes the
    # auto-attachment silently grab nothing and the cube falls off, so find
    # the actual deformable prim first.
    from pxr import Usd
    mesh_path = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/FemCable")):
        if prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
            mesh_path = prim.GetPath()
            break
    if mesh_path is None:
        raise RuntimeError("No PhysxDeformableBodyAPI prim under /World/FemCable")
    print(f"  FEM deformable mesh prim: {mesh_path}")

    att = PhysxSchema.PhysxPhysicsAttachment.Define(
        stage, f"{mesh_path}/attachment_end")
    att.GetActor0Rel().SetTargets([mesh_path])
    att.GetActor1Rel().SetTargets([Sdf.Path("/World/FemEndCube")])
    auto = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
    auto.CreateEnableDeformableVertexAttachmentsAttr().Set(True)
    auto.CreateDeformableVertexOverlapOffsetAttr().Set(ATTACH_OVERLAP)

    return cable


# ===============================================================
# 4. MAIN
# ===============================================================
def main():
    print("=" * 70)
    print("Hanging comparison -- capsule chain vs FEM, one scene")
    print("=" * 70)
    print(f"  cable: L={CABLE_LENGTH} m  r={CABLE_RADIUS*1e3:.1f} mm  "
          f"E={YOUNG_MODULUS/1e6:.0f} MPa  rho={DENSITY:.0f}")
    print(f"  capsule chain: {NUM_LINKS} links  (k={K_DRIVE:.2e} N.m/deg)")
    print(f"  FEM rod: R_sim={FEM_SIM_RADIUS*1e3:.1f} mm  "
          f"E_sim={FEM_E/1e3:.2f} kPa  rho_sim={FEM_DENSITY:.1f} kg/m^3  "
          f"hex={HEX_RESOLUTION}")
    print(f"  end object: {END_CUBE_MASS*1e3:.0f} g cube on both cables")
    print(f"  force pulse: {FORCE_X} N in +x on both cubes, "
          f"t in [{FORCE_START}, {FORCE_END}) s")
    print(f"  dt={PHYSICS_DT*1e3:.2f} ms   max time {MAX_SIM_TIME} s")
    print(f"  output: {OUTPUT_DIR}")

    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=max(1, round(RENDER_DT / PHYSICS_DT)),
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.2, 0.0, 1.8], target=[0.0, 0.0, 1.4])
    stage = omni.usd.get_context().get_stage()

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)

    capsule_paths = build_capsule_cable(stage)
    fem_cable = build_fem_cable(stage)

    sim.reset()
    fem_cable.reset()

    # Fix FEM top vertices kinematically (same as cable_deformable.py)
    nodal_kin = fem_cable.data.nodal_kinematic_target.clone()
    nodal_pos0 = fem_cable.data.default_nodal_state_w[..., :3]
    z_vals = nodal_pos0[0, :, 2]
    top_idx = torch.where(z_vals >= z_vals.max() - 0.01)[0]
    nodal_kin[..., :3] = nodal_pos0
    nodal_kin[..., 3] = 1.0
    nodal_kin[0, top_idx, 3] = 0.0
    fem_cable.write_nodal_kinematic_target_to_sim(nodal_kin)
    # FEM tip vertex (closest to the bottom end centerline)
    fem_tip_idx = int(z_vals.argmin().item())
    d_mid = (nodal_pos0[0] - torch.tensor(
        [0.0, Y_FEM, ANCHOR_Z - CABLE_LENGTH / 2.0],
        dtype=nodal_pos0.dtype, device=nodal_pos0.device)).norm(dim=-1)
    fem_mid_idx = int(d_mid.argmin().item())
    print(f"  FEM: {nodal_pos0.shape[1]} vertices, {len(top_idx)} fixed at top")

    # Views: all capsules (for mid/stability) + the two end cubes (force+log)
    capsule_view = RigidPrimView(prim_paths_expr=capsule_paths,
                                 name="capsule_view",
                                 reset_xform_properties=False)
    capsule_view.initialize()
    cube_view = RigidPrimView(
        prim_paths_expr=["/World/CapCable/end_cube", "/World/FemEndCube"],
        name="cube_view", reset_xform_properties=False)
    cube_view.initialize()
    mid_index = NUM_LINKS // 2

    force_vec = torch.tensor([[FORCE_X, 0.0, 0.0],
                              [FORCE_X, 0.0, 0.0]], dtype=torch.float32)

    csv_file = open(CSV_PATH, "w", newline="")
    csv_file.write("t,force_x,"
                   "cap_tip_x,cap_tip_y,cap_tip_z,"
                   "fem_tip_x,fem_tip_y,fem_tip_z,"
                   "cap_mid_x,cap_mid_y,cap_mid_z,"
                   "fem_mid_x,fem_mid_y,fem_mid_z\n")

    render_every = max(1, round(RENDER_DT / PHYSICS_DT))
    sim_time, step_count = 0.0, 0
    stable, instability_at = True, None

    print(f"\nSimulating {MAX_SIM_TIME}s ...")
    wall_t0 = time.perf_counter()

    while simulation_app.is_running() and sim_time < MAX_SIM_TIME:
        in_pulse = FORCE_START <= sim_time < FORCE_END
        if in_pulse:
            cube_view.apply_forces(force_vec, is_global=True)

        sim.step()
        sim_time += PHYSICS_DT
        step_count += 1
        fem_cable.update(PHYSICS_DT)

        if step_count % render_every == 0:
            cube_pos, _ = cube_view.get_world_poses(usd=False)
            cube_pos = to_np(cube_pos)
            cap_mid_pos, _ = capsule_view.get_world_poses(indices=[mid_index],
                                                          usd=False)
            cap_mid = to_np(cap_mid_pos)[0]
            fem_pos = fem_cable.data.nodal_pos_w[0]
            fem_tip = to_np(fem_pos[fem_tip_idx])
            fem_mid = to_np(fem_pos[fem_mid_idx])
            cap_tip = cube_pos[0]
            fem_cube = cube_pos[1]

            csv_file.write(
                f"{sim_time:.6f},{FORCE_X if in_pulse else 0.0:.4f},"
                f"{cap_tip[0]:.6f},{cap_tip[1]:.6f},{cap_tip[2]:.6f},"
                f"{fem_cube[0]:.6f},{fem_cube[1]:.6f},{fem_cube[2]:.6f},"
                f"{cap_mid[0]:.6f},{cap_mid[1]:.6f},{cap_mid[2]:.6f},"
                f"{fem_mid[0]:.6f},{fem_mid[1]:.6f},{fem_mid[2]:.6f}\n")

            all_pts = np.concatenate([cube_pos.ravel(), cap_mid, fem_tip])
            if stable and (not np.all(np.isfinite(all_pts))
                           or np.max(np.abs(all_pts)) > DIVERGENCE_POS_M):
                stable, instability_at = False, sim_time
                print(f"  *** position divergence at t={sim_time:.3f}s ***")

        if step_count % (render_every * 120) == 0:
            rtf = sim_time / max(time.perf_counter() - wall_t0, 1e-9)
            print(f"  t={sim_time:5.2f}s  {'STABLE' if stable else 'UNSTABLE'}"
                  f"  rtf={rtf:.2f}x")

    wall_elapsed = time.perf_counter() - wall_t0
    csv_file.close()
    print(f"\nDone: t={sim_time:.2f}s  wall={wall_elapsed:.1f}s "
          f"({sim_time/max(wall_elapsed,1e-9):.2f}x realtime)  stable={stable}")

    summary = {
        "test":               "hanging_compare_one_scene",
        "cable_length_m":     CABLE_LENGTH,
        "cable_radius_m":     CABLE_RADIUS,
        "young_modulus_pa":   YOUNG_MODULUS,
        "density_kg_m3":      DENSITY,
        "num_links":          NUM_LINKS,
        "fem_sim_radius_m":   FEM_SIM_RADIUS,
        "fem_E_pa":           FEM_E,
        "fem_density_kg_m3":  FEM_DENSITY,
        "hex_resolution":     HEX_RESOLUTION,
        "end_cube_mass_kg":   END_CUBE_MASS,
        "force_x_n":          FORCE_X,
        "force_window_s":     [FORCE_START, FORCE_END],
        "physics_dt_s":       PHYSICS_DT,
        "total_sim_time_s":   sim_time,
        "wall_clock_s":       wall_elapsed,
        "realtime_factor":    sim_time / max(wall_elapsed, 1e-9),
        "stable":             stable,
        "instability_at_s":   instability_at,
        "csv_path":           str(CSV_PATH),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {SUMMARY_PATH}")

    make_plot()


def make_plot():
    import csv as csv_mod
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(CSV_PATH) as f:
        reader = csv_mod.reader(f)
        header = next(reader)
        rows = np.array([[float(v) for v in r] for r in reader if r])
    col = {n: i for i, n in enumerate(header)}
    t = rows[:, col["t"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Hanging cable + end mass + force pulse: capsule chain vs FEM")

    ax = axes[0]
    ax.plot(t, rows[:, col["cap_tip_x"]], "r", label="capsule tip x")
    ax.plot(t, rows[:, col["fem_tip_x"]], "b", label="FEM tip x")
    ax.plot(t, rows[:, col["force_x"]] * 10, "k:", alpha=0.5,
            label="force pulse (x10 N)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("x [m]")
    ax.set_title("End-object swing (force direction)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, rows[:, col["cap_tip_z"]], "r", label="capsule tip z")
    ax.plot(t, rows[:, col["fem_tip_z"]], "b", label="FEM tip z")
    ax.set_xlabel("t [s]"); ax.set_ylabel("z [m]")
    ax.set_title("End-object height")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(rows[:, col["cap_tip_x"]], rows[:, col["cap_tip_z"]], "r",
            label="capsule")
    ax.plot(rows[:, col["fem_tip_x"]], rows[:, col["fem_tip_z"]], "b",
            label="FEM")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.set_title("End-object path (x-z)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.axis("equal")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=140)
    print(f"Plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang at 100% CPU on this machine; all
    # outputs are on disk by now, so force an exit if it doesn't return.
    import threading
    threading.Timer(20.0, lambda: os._exit(0)).start()
    simulation_app.close()
    os._exit(0)
