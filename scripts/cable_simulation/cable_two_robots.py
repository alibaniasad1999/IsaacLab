"""
Two Franka robots connected by a flexible cable -- method comparison test.

Two Franka Panda arms face each other and hold the two ends of a 0.6 m
flexible cable. After a short settle, the LEADER arm swings its base joint
(panda_joint1) sinusoidally while the FOLLOWER holds its pose, so the cable
is dragged side to side. The same experiment runs with either cable model:

  CABLE_METHOD=capsule      rigid capsule chain + D6 joints (cable.py model)
  CABLE_METHOD=deformable   FEM deformable cylinder (cable_deformable.py model)

Both write the same trajectory.csv / summary.json layout (incl. wall-clock
realtime factor) into cable_output/two_robots_<method>/, so the two methods
can be compared directly with compare_methods.py.

Run (GUI):
    conda activate env_isaaclab
    CABLE_METHOD=capsule    python scripts/cable_simulation/cable_two_robots.py
    CABLE_METHOD=deformable python scripts/cable_simulation/cable_two_robots.py

Headless sweep:
    CABLE_HEADLESS=1 CABLE_MAX_TIME=10 CABLE_METHOD=capsule \
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
from pxr import UsdPhysics, PhysxSchema, Gf, Sdf


# ===============================================================
# 1. CONFIGURATION
# ===============================================================
CABLE_METHOD       = os.environ.get("CABLE_METHOD", "capsule")
assert CABLE_METHOD in ("capsule", "deformable"), \
    f"Unknown CABLE_METHOD: {CABLE_METHOD}"

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
HEX_RESOLUTION     = int(os.environ.get("CABLE_HEX_RES",     24))
FEM_SOLVER_ITERS   = int(os.environ.get("CABLE_SOLVER_ITERS", 32))
VERTEX_DAMPING     = float(os.environ.get("CABLE_VDAMP",     0.005))
ATTACH_OVERLAP     = 0.02   # m -- radius for grabbing cable vertices near hand

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
            radius=CABLE_RADIUS,
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
                youngs_modulus=YOUNG_MODULUS,
                poissons_ratio=POISSON_RATIO,
                density=DENSITY,
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
    else:
        print(f"  hex resolution  : {HEX_RESOLUTION}")
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

    if CABLE_METHOD == "capsule":
        capsule_paths = build_capsule_cable(stage)
        cable = None
    else:
        cable = build_deformable_cable(stage)
        capsule_paths = None

    sim.reset()
    leader.update(PHYSICS_DT)
    follower.update(PHYSICS_DT)

    hand_idx = leader.body_names.index("panda_hand")
    j1_idx   = leader.joint_names.index("panda_joint1")

    # Sanity check: how far is the assumed anchor from the actual TCP?
    for tag, robot, anchor in (("leader", leader, ANCHOR_LEAD),
                               ("follower", follower, ANCHOR_FOLW)):
        hp = robot.data.body_pos_w[0, hand_idx].cpu().numpy()
        hq = robot.data.body_quat_w[0, hand_idx].cpu().numpy()
        tcp = hp + quat_rotate(hq, TCP_OFFSET_IN_HAND)
        err = float(np.linalg.norm(tcp - anchor))
        print(f"  {tag} TCP at {np.round(tcp, 4)}  anchor error {err*1e3:.1f} mm")

    # Cable mid-point tracking
    if CABLE_METHOD == "capsule":
        capsule_view = RigidPrimView(prim_paths_expr=capsule_paths,
                                     name="capsule_view",
                                     reset_xform_properties=False)
        capsule_view.initialize()
        mid_index = NUM_LINKS // 2
        monitor_idx = list(range(0, NUM_LINKS, max(NUM_LINKS // 10, 1)))
    else:
        nodal_pos = cable.data.nodal_pos_w
        d = (nodal_pos[0] - torch.tensor(CABLE_MID, dtype=nodal_pos.dtype,
                                         device=nodal_pos.device)).norm(dim=-1)
        mid_index = int(d.argmin().item())

    # Joint targets
    leader_targets   = leader.data.default_joint_pos.clone()
    follower_targets = follower.data.default_joint_pos.clone()
    q1_default       = float(leader_targets[0, j1_idx].item())

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
            else:
                mid = cable.data.nodal_pos_w[0, mid_index].cpu().numpy()
                vel = cable.data.nodal_state_w[0, :, 3:6]
                speed = float(vel.norm(dim=-1).max().item())           # m/s
                threshold = DIVERGENCE_VEL

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
