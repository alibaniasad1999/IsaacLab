"""
Flexible cable simulation using Isaac Sim's deformable body (FEM) API.

This is the second method for cable simulation, to be compared against
the rigid capsule-chain approach in cable.py.

The cable is a single deformable cylinder mesh with material properties
(Young's modulus, Poisson's ratio, damping) set to match the same PUR
(polyurethane) cable used in cable.py.

The top end is kinematically constrained (fixed to world), the bottom
is free. Same two experiment modes as cable.py:
  - "hanging_kick": top fixed, bottom gets initial velocity
  - "both_ends_fixed": both ends fixed, step displacement after settling

Run:
    conda activate env_isaaclab
    python scripts/cable_simulation/cable_deformable.py

Headless / sweep mode:
    CABLE_HEADLESS=1 CABLE_RECORD=0 CABLE_MAX_TIME=2.0 \
        python scripts/cable_simulation/cable_deformable.py

Compare with capsule-chain:
    python scripts/cable_simulation/cable_deformable.py
    python scripts/cable_simulation/cable.py
"""

import argparse
import os
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deformable cable simulation in Isaac Sim.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

HEADLESS = os.environ.get("CABLE_HEADLESS", "0") == "1"
if HEADLESS:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Imports after SimulationApp --
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext


# ===============================================================
# 1. CONFIGURATION
# ===============================================================

# -- Cable geometry (match cable.py defaults) --
TOTAL_CABLE_LENGTH = float(os.environ.get("CABLE_LENGTH",     1.0))     # m
CABLE_RADIUS       = float(os.environ.get("CABLE_RADIUS",     1.5e-3))  # m
CABLE_DIAMETER     = 2.0 * CABLE_RADIUS

# -- Material: flexible TPU robot cable. Defaults match cable.py (E=40 MPa,
# rho=1150) so the two methods are directly comparable. --
YOUNG_MODULUS      = float(os.environ.get("CABLE_E",          40e6))    # Pa
# NOTE: A near-incompressible Poisson ratio (TPU's true ~0.48) causes severe
# VOLUMETRIC LOCKING in coarse hex FEM, which artificially stiffens bending.
# 0.3 keeps the cable visibly floppy; raise it back toward 0.48 only if you
# also raise CABLE_HEX_RES enough to avoid locking.
POISSON_RATIO      = float(os.environ.get("CABLE_NU",         0.3))
DENSITY            = float(os.environ.get("CABLE_DENSITY",    1150.0))  # kg/m^3 (TPU)
ELASTICITY_DAMPING = float(os.environ.get("CABLE_EDAMP",      0.005))
DAMPING_SCALE      = float(os.environ.get("CABLE_DSCALE",     1.0))

# -- FEM resolution --
# WARNING: at the cable's TRUE radius (1.5 mm) the voxel mesher places a
# single hex element across the cross-section for any affordable resolution
# (3 elements across a 3 mm rod would need res ~1000 over 1 m). One element
# across the width cannot represent a bending strain gradient, so this thin
# FEM rod is INHERENTLY bend-stiff -- it will swing like a bar regardless of
# E. For a FEM cable that actually drapes, see cable_hanging_compare.py,
# which simulates a fatter rod with EI- and mass-equivalent scaled material
# (R_sim = 6 mm, E*(r/R)^4, rho*(r/R)^2, res 250).
HEX_RESOLUTION     = int(os.environ.get("CABLE_HEX_RES",     24))

# -- Solver --
PHYSICS_DT         = float(os.environ.get("CABLE_PHYSICS_DT", 1.0/240.0))
RENDER_DT          = float(os.environ.get("CABLE_RENDER_DT",  1.0/60.0))
SOLVER_ITERATIONS  = int(os.environ.get("CABLE_SOLVER_ITERS", 32))
VERTEX_DAMPING     = float(os.environ.get("CABLE_VDAMP",      0.005))

# -- Experiment mode --
EXPERIMENT_MODE    = os.environ.get("CABLE_MODE", "hanging_kick")
assert EXPERIMENT_MODE in ("hanging_kick", "both_ends_fixed"), \
    f"Unknown CABLE_MODE: {EXPERIMENT_MODE}"

STEP_DISPLACEMENT  = float(os.environ.get("CABLE_STEP_DISP",  5e-3))
SETTLE_SECONDS     = float(os.environ.get("CABLE_SETTLE",     0.5))
KICK_VELOCITY      = float(os.environ.get("CABLE_KICK_VX",    1.5))

# -- Output --
SCRIPT_DIR         = Path(__file__).parent
OUTPUT_DIR         = Path(os.environ.get("CABLE_OUTPUT_DIR",
                     str(SCRIPT_DIR / "cable_output" / f"deformable_{EXPERIMENT_MODE}")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH       = OUTPUT_DIR / "summary.json"
CSV_PATH           = OUTPUT_DIR / "trajectory.csv"

MAX_SIM_TIME       = float(os.environ.get("CABLE_MAX_TIME", 10.0))

# -- Stability monitor --
DIVERGENCE_VEL     = 1.0e4   # m/s -- any vertex above this => unstable
STABILITY_CHECK_EVERY = 4

ANCHOR_Z           = 2.0


# ===============================================================
# 2. SCENE DESIGN
# ===============================================================
def design_scene():
    """Build the scene: ground, light, deformable cable."""
    # Ground
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Light
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Origin for the cable
    sim_utils.create_prim("/World/Origin", "Xform",
                          translation=(0.0, 0.0, ANCHOR_Z))

    # Deformable cable as a cylinder mesh
    cable_cfg = DeformableObjectCfg(
        prim_path="/World/Origin/Cable",
        spawn=sim_utils.MeshCylinderCfg(
            radius=CABLE_RADIUS,
            height=TOTAL_CABLE_LENGTH,
            axis="Z",
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                self_collision=False,
                solver_position_iteration_count=SOLVER_ITERATIONS,
                vertex_velocity_damping=VERTEX_DAMPING,
                simulation_hexahedral_resolution=HEX_RESOLUTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.3, 0.8),
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=YOUNG_MODULUS,
                poissons_ratio=POISSON_RATIO,
                density=DENSITY,
                elasticity_damping=ELASTICITY_DAMPING,
                damping_scale=DAMPING_SCALE,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -TOTAL_CABLE_LENGTH / 2.0),
        ),
        debug_vis=True,
    )
    cable_object = DeformableObject(cfg=cable_cfg)

    # Obstacle for hanging_kick mode
    if EXPERIMENT_MODE == "hanging_kick":
        obs_cfg = sim_utils.MeshCuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.7, 0.3),
            ),
        )
        obs_cfg.func(
            "/World/Obstacle", obs_cfg,
            translation=(0.12, 0.0, ANCHOR_Z - TOTAL_CABLE_LENGTH * 0.4),
        )

    return cable_object


# ===============================================================
# 3. HELPER FUNCTIONS
# ===============================================================
def get_top_vertex_indices(nodal_pos, n_top=None, z_threshold=None):
    """Find vertex indices near the top of the cable."""
    z_vals = nodal_pos[0, :, 2]
    if z_threshold is None:
        z_threshold = z_vals.max() - 0.01
    mask = z_vals >= z_threshold
    indices = torch.where(mask)[0]
    return indices


def get_bottom_vertex_indices(nodal_pos, z_threshold=None):
    """Find vertex indices near the bottom of the cable."""
    z_vals = nodal_pos[0, :, 2]
    if z_threshold is None:
        z_threshold = z_vals.min() + 0.01
    mask = z_vals <= z_threshold
    indices = torch.where(mask)[0]
    return indices


def get_tracked_vertex_indices(nodal_pos, count=7):
    """Pick evenly spaced vertices along the cable for logging."""
    z_vals = nodal_pos[0, :, 2]
    z_min, z_max = z_vals.min().item(), z_vals.max().item()
    targets = np.linspace(z_min, z_max, count)
    indices = []
    for zt in targets:
        idx = (z_vals - zt).abs().argmin().item()
        if idx not in indices:
            indices.append(idx)
    return indices


# ===============================================================
# 4. SIMULATION LOOP
# ===============================================================
def run_simulator(sim, cable):
    """Main simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0
    render_steps_per_physics = max(1, round(RENDER_DT / sim_dt))

    # Get initial nodal positions
    nodal_state = cable.data.default_nodal_state_w.clone()
    nodal_kinematic_target = cable.data.nodal_kinematic_target.clone()
    nodal_pos = nodal_state[..., :3]

    # Identify vertex groups
    top_indices = get_top_vertex_indices(nodal_pos)
    bottom_indices = get_bottom_vertex_indices(nodal_pos)
    tracked_indices = get_tracked_vertex_indices(nodal_pos)

    print(f"  top vertices:    {len(top_indices)}")
    print(f"  bottom vertices: {len(bottom_indices)}")
    print(f"  tracked vertices: {len(tracked_indices)}")

    # Fix top vertices (kinematic constraint)
    # 0 = constrained, 1 = free
    nodal_kinematic_target[..., :3] = nodal_pos
    nodal_kinematic_target[..., 3] = 1.0  # all free by default
    nodal_kinematic_target[0, top_indices, 3] = 0.0  # fix top
    cable.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

    # Fix bottom too if both_ends_fixed
    if EXPERIMENT_MODE == "both_ends_fixed":
        nodal_kinematic_target[0, bottom_indices, 3] = 0.0
        cable.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

    # Stability tracking
    stable = True
    instability_at_s = None
    max_vel_seen = 0.0

    # CSV logging
    csv_file = open(CSV_PATH, "w", newline="")
    csv_header = ["t"]
    for vi in tracked_indices:
        csv_header.extend([f"v{vi}_x", f"v{vi}_y", f"v{vi}_z"])
    csv_file.write(",".join(csv_header) + "\n")

    kick_applied = False
    step_applied = False

    print(f"\nStarting simulation ({EXPERIMENT_MODE}, max {MAX_SIM_TIME}s)...")
    print("=" * 60)
    wall_t0 = time.perf_counter()

    while simulation_app.is_running() and sim_time < MAX_SIM_TIME:
        # -- Apply kick (hanging_kick mode, at t=0) --
        if EXPERIMENT_MODE == "hanging_kick" and not kick_applied:
            current_state = cable.data.nodal_state_w.clone()
            current_state[0, bottom_indices, 3] = KICK_VELOCITY  # vx
            cable.write_nodal_state_to_sim(current_state)
            kick_applied = True
            print(f"  t={sim_time:.3f}s: kick applied (vx={KICK_VELOCITY} m/s)")

        # -- Apply step displacement (both_ends_fixed, after settling) --
        if (EXPERIMENT_MODE == "both_ends_fixed"
                and not step_applied
                and sim_time >= SETTLE_SECONDS):
            nodal_kinematic_target[0, bottom_indices, 0] += STEP_DISPLACEMENT
            cable.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
            step_applied = True
            print(f"  t={sim_time:.3f}s: step displacement applied "
                  f"({STEP_DISPLACEMENT*1000:.1f} mm)")

        # -- Step physics --
        cable.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        step_count += 1
        cable.update(sim_dt)

        # -- Log at render rate --
        if step_count % render_steps_per_physics == 0:
            pos = cable.data.nodal_pos_w  # (num_instances, num_vertices, 3)

            # CSV row
            row = [f"{sim_time:.6f}"]
            for vi in tracked_indices:
                p = pos[0, vi]
                row.extend([f"{p[0].item():.6f}",
                            f"{p[1].item():.6f}",
                            f"{p[2].item():.6f}"])
            csv_file.write(",".join(row) + "\n")

        # -- Stability check --
        if step_count % (render_steps_per_physics * STABILITY_CHECK_EVERY) == 0:
            vel = cable.data.nodal_state_w[0, :, 3:6]
            vel_mag = vel.norm(dim=-1).max().item()
            max_vel_seen = max(max_vel_seen, vel_mag)

            if vel_mag > DIVERGENCE_VEL and stable:
                stable = False
                instability_at_s = sim_time
                print(f"  *** UNSTABLE at t={sim_time:.3f}s "
                      f"(max |v|={vel_mag:.1e} m/s) ***")

        # -- Progress --
        if step_count % (render_steps_per_physics * 60) == 0:
            rtf = sim_time / max(time.perf_counter() - wall_t0, 1e-9)
            print(f"  t={sim_time:.1f}s  max|v|={max_vel_seen:.2e} m/s  "
                  f"{'STABLE' if stable else 'UNSTABLE'}  rtf={rtf:.2f}x")

    wall_elapsed = time.perf_counter() - wall_t0
    csv_file.close()
    print("=" * 60)
    print(f"Simulation complete: {sim_time:.2f}s, {step_count} steps")
    print(f"  wall clock: {wall_elapsed:.1f} s "
          f"({sim_time / max(wall_elapsed, 1e-9):.2f}x realtime)")
    print(f"  stable: {stable}")
    print(f"  max |v|: {max_vel_seen:.3e} m/s")
    print(f"  CSV: {CSV_PATH}")

    # -- Summary JSON --
    summary = {
        "method":               "deformable_body",
        "experiment_mode":      EXPERIMENT_MODE,
        "total_cable_length_m": TOTAL_CABLE_LENGTH,
        "cable_radius_m":       CABLE_RADIUS,
        "young_modulus_pa":     YOUNG_MODULUS,
        "poissons_ratio":       POISSON_RATIO,
        "density_kg_m3":        DENSITY,
        "elasticity_damping":   ELASTICITY_DAMPING,
        "damping_scale":        DAMPING_SCALE,
        "hex_resolution":       HEX_RESOLUTION,
        "solver_iterations":    SOLVER_ITERATIONS,
        "vertex_damping":       VERTEX_DAMPING,
        "physics_dt_s":         PHYSICS_DT,
        "render_dt_s":          RENDER_DT,
        "total_sim_time_s":     sim_time,
        "wall_clock_s":         wall_elapsed,
        "realtime_factor":      sim_time / max(wall_elapsed, 1e-9),
        "stable":               stable,
        "instability_at_s":     instability_at_s,
        "max_vertex_vel_m_s":   max_vel_seen,
        "csv_path":             str(CSV_PATH),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {SUMMARY_PATH}")


# ===============================================================
# 5. MAIN
# ===============================================================
def main():
    print("=" * 60)
    print(f"Deformable cable -- mode = {EXPERIMENT_MODE}")
    print("=" * 60)
    print(f"  length:    {TOTAL_CABLE_LENGTH} m")
    print(f"  radius:    {CABLE_RADIUS*1000:.2f} mm")
    print(f"  E:         {YOUNG_MODULUS/1e6:.1f} MPa (PUR)")
    print(f"  nu:        {POISSON_RATIO}")
    print(f"  density:   {DENSITY} kg/m^3")
    print(f"  hex res:   {HEX_RESOLUTION}")
    print(f"  solver it: {SOLVER_ITERATIONS}")
    print(f"  dt:        {PHYSICS_DT*1e6:.1f} us ({1.0/PHYSICS_DT:.0f} Hz)")
    print(f"  output:    {OUTPUT_DIR}")

    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=max(1, round(RENDER_DT / PHYSICS_DT)),
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 2.5], target=[0.0, 0.0, ANCHOR_Z - 0.5])

    cable = design_scene()

    sim.reset()
    cable.reset()
    print("[INFO] Setup complete.")

    run_simulator(sim, cable)


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang at 100% CPU on this machine; all
    # outputs are on disk by now, so force an exit if it doesn't return.
    import threading
    threading.Timer(20.0, lambda: os._exit(0)).start()
    simulation_app.close()
    os._exit(0)
