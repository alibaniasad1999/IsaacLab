"""
Elastic-ROD cable in NVIDIA Warp (paper-style), shown + driven inside Isaac Sim.

Why this instead of the PhysX FEM deformable: a cable is a 1-D elastic ROD, not
a 3-D blob of jelly. The FEM "deformable" is volumetric -> it looks like gelatin,
stretches like a slinky, and its collision mesh doesn't match what you see. The
OGC SIGGRAPH-2025 paper does cloth AND rods, but its rods are COSSERAT rods in
NVIDIA Warp -- a 1-D model. This script is that idea: a position-based elastic
rod (inextensible stretch + bending stiffness + gravity), solved with XPBD in
Warp, so it behaves like a real rope/cable connecting two points.

  * STEADY end  -> pinned to a fixed anchor (blue).
  * MOVING end  -> a handle cube (orange) you move with the mouse. The rod's end
                   node is GLUED to the handle's position every frame (in code),
                   so it never detaches or snaps back -- you move the handle and
                   the cable deforms (arcs/swings) to follow.
  * The rod is a 1-D centreline rendered as a THIN tube (any radius -- no FEM
    fattening limit), and collision is EXACT node-vs-primitive (ground + an
    optional sphere obstacle), so there is no "collision with empty space".

HOW TO MOVE THE HANDLE (mouse):
  The handle is KINEMATIC, so during play you can select it, press W (Move tool)
  and DRAG the gizmo axis -- the cable follows. (Shift+Left-drag also applies.)

Run (interactive GUI):
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
    python scripts/cable_simulation/cable_warp.py

Headless self-test (scripts the handle, checks the rod follows + stays
inextensible + stable):
    CABLE_HEADLESS=1 CABLE_SELFTEST=1 python scripts/cable_simulation/cable_warp.py
"""

import os
import math

from isaacsim.simulation_app import SimulationApp
HEADLESS = os.environ.get("CABLE_HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": HEADLESS})

import sys
import numpy as np
import warp as wp
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema, UsdShade

# Shared physical-cable parameters (length, mass) live in cable_config.py so all
# three cable scripts model the SAME physical cable. The rod keeps its own visual
# radius (a 1-D rod has no FEM fattening limit) and node count.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cable_config import TOTAL_CABLE_LENGTH, CABLE_MASS

wp.init()
DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"


# ===============================================================
# 1. CONFIG
# ===============================================================
N            = int(os.environ.get("ROD_NODES", 48))          # rod nodes
LENGTH       = TOTAL_CABLE_LENGTH                             # rest length (m), inextensible
VIS_RADIUS   = float(os.environ.get("ROD_RADIUS", 5e-3))     # visual + collision radius (m)
TOTAL_MASS   = CABLE_MASS                                     # kg (physical cable mass)
BEND_COMP    = float(os.environ.get("ROD_BEND_COMP", 4.0e-3))  # bending compliance: bigger =
                                                               # floppier (swings/drapes like a
                                                               # rope, not a stiff stick)
STRETCH_COMP = float(os.environ.get("ROD_STRETCH_COMP", 1.0e-9))  # ~0 => inextensible
SUBSTEPS     = int(os.environ.get("ROD_SUBSTEPS", 16))
ITERS        = int(os.environ.get("ROD_ITERS", 12))
VEL_DAMPING  = float(os.environ.get("ROD_DAMPING", 0.02))
GRAVITY      = float(os.environ.get("ROD_GRAVITY", -9.81))

ANCHOR_POS = np.array([-0.4, 0.0, 1.5])    # STEADY end
HANDLE_POS = np.array([+0.4, 0.0, 1.5])    # MOVING end (start)

USE_SPHERE   = os.environ.get("ROD_OBSTACLE", "1") == "1"
SPHERE_POS   = np.array([0.0, 0.0, 1.05])
SPHERE_R     = float(os.environ.get("ROD_SPHERE_R", 0.12))
GROUND_Z     = 0.0

RENDER_DT  = 1.0 / 60.0
SELFTEST   = os.environ.get("CABLE_SELFTEST", "0") == "1"
MAX_TIME   = float(os.environ.get("CABLE_MAX_TIME", 6.0))
VIS_SEG    = 10   # tube cross-section segments

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Video recording (GUI render mode). CABLE_RECORD=1 -> record a finite MAX_TIME
# clip with the handle scripted on a sweep (so the cable visibly deforms), written
# to CABLE_OUTPUT_DIR/cable_warp.mp4.
import subprocess
RECORD     = os.environ.get("CABLE_RECORD", "0") == "1" and not HEADLESS
OUTPUT_DIR = os.environ.get("CABLE_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "cable_output", "warp"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_PATH = os.path.join(OUTPUT_DIR, "cable_warp.mp4")


def _warp_ffmpeg(w, h):
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-framerate", "60", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", VIDEO_PATH]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ===============================================================
# 2. WARP XPBD ROD KERNELS
# ===============================================================
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
def set_pins(x: wp.array(dtype=wp.vec3), anchor: wp.vec3, handle: wp.vec3,
             n: int, pin_end: int):
    i = wp.tid()
    if i == 0:
        x[0] = anchor
    if i == n - 1 and pin_end == 1:
        x[n - 1] = handle


@wp.kernel
def solve_rod(x: wp.array(dtype=wp.vec3), invm: wp.array(dtype=float),
              rest: wp.array(dtype=float), n: int, iters: int,
              a_stretch: float, a_bend: float):
    # single-thread sequential Gauss-Seidel projection (N is small)
    if wp.tid() != 0:
        return
    for _it in range(iters):
        # stretch (inextensible)
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
        # bending (resist curvature: pull node toward midpoint of neighbours)
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


@wp.kernel
def collide(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
            invm: wp.array(dtype=float), ground_z: float, rad: float,
            sc: wp.vec3, sr: float, use_sphere: int):
    i = wp.tid()
    if invm[i] <= 0.0:
        return
    p = x[i]
    px = p[0]; py = p[1]; pz = p[2]
    if pz < ground_z + rad:
        pz = ground_z + rad
        v[i] = wp.vec3(v[i][0], v[i][1], 0.0)
    p = wp.vec3(px, py, pz)
    if use_sphere == 1:
        d = p - sc
        dist = wp.length(d)
        mind = sr + rad
        if dist < mind and dist > 1.0e-9:
            p = sc + d / dist * mind
    x[i] = p


# ===============================================================
# 3. ROD STATE
# ===============================================================
_seg_rest = LENGTH / (N - 1)
_init = np.linspace(ANCHOR_POS, HANDLE_POS, N).astype(np.float32)
x_wp      = wp.array(_init, dtype=wp.vec3, device=DEVICE)
xprev_wp  = wp.zeros(N, dtype=wp.vec3, device=DEVICE)
v_wp      = wp.zeros(N, dtype=wp.vec3, device=DEVICE)
m_node    = TOTAL_MASS / N
invm_np   = np.full(N, 1.0 / m_node, dtype=np.float32)
invm_np[0] = 0.0          # steady (anchor) end always pinned
invm_np[-1] = 0.0         # moving end pinned to the handle...
if RECORD:
    invm_np[-1] = 1.0 / m_node   # ...EXCEPT when recording: free it so it FALLS on the sphere
invm_wp   = wp.array(invm_np, dtype=float, device=DEVICE)
rest_wp   = wp.array(np.full(N - 1, _seg_rest, dtype=np.float32), dtype=float, device=DEVICE)


def step_rod(anchor, handle, pin_end=1):
    a = wp.vec3(float(anchor[0]), float(anchor[1]), float(anchor[2]))
    h = wp.vec3(float(handle[0]), float(handle[1]), float(handle[2]))
    sc = wp.vec3(float(SPHERE_POS[0]), float(SPHERE_POS[1]), float(SPHERE_POS[2]))
    dt = RENDER_DT / SUBSTEPS
    a_s = STRETCH_COMP / (dt * dt)
    a_b = BEND_COMP / (dt * dt)
    for _ in range(SUBSTEPS):
        wp.launch(predict, dim=N, inputs=[x_wp, xprev_wp, v_wp, invm_wp, GRAVITY, dt], device=DEVICE)
        wp.launch(set_pins, dim=N, inputs=[x_wp, a, h, N, pin_end], device=DEVICE)
        wp.launch(solve_rod, dim=1, inputs=[x_wp, invm_wp, rest_wp, N, ITERS, a_s, a_b], device=DEVICE)
        wp.launch(set_pins, dim=N, inputs=[x_wp, a, h, N, pin_end], device=DEVICE)
        wp.launch(finalize, dim=N, inputs=[x_wp, xprev_wp, v_wp, invm_wp, dt, VEL_DAMPING], device=DEVICE)
        wp.launch(collide, dim=N, inputs=[x_wp, v_wp, invm_wp, GROUND_Z, VIS_RADIUS,
                                          sc, SPHERE_R, 1 if USE_SPHERE else 0], device=DEVICE)
    return x_wp.numpy()


# ===============================================================
# 4. ISAAC SCENE (render + the movable handle)
# ===============================================================
world = World(stage_units_in_meters=1.0, physics_dt=RENDER_DT, rendering_dt=RENDER_DT)
world.scene.add_default_ground_plane(z_position=GROUND_Z)
stage = world.stage


def cube(name, pos, size, color):
    path = f"/World/{name}"
    obj = world.scene.add(DynamicCuboid(prim_path=path, name=name, position=np.array(pos),
                                        size=size, color=np.array(color), mass=0.05))
    rb = UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
    rb.CreateKinematicEnabledAttr().Set(True)   # ends are scripted/movable, not physics-driven
    return path, obj


anchor_path, anchor_cube = cube("anchor", ANCHOR_POS, 0.05, [0.2, 0.4, 0.8])     # steady
handle_path, handle_cube = cube("handle", HANDLE_POS, 0.06, [0.95, 0.55, 0.05])  # MOVE THIS
if RECORD:
    # Cable-only recording: keep the blue ANCHOR end fixed (and visible), DELETE the
    # orange handle cube (hide it) and free that end so the cable FALLS onto the sphere.
    UsdGeom.Imageable(stage.GetPrimAtPath(handle_path)).MakeInvisible()

if USE_SPHERE:
    sph = UsdGeom.Sphere.Define(stage, "/World/obstacle")
    sph.CreateRadiusAttr(SPHERE_R)
    sph.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.7, 0.3)])
    UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*[float(v) for v in SPHERE_POS]))

# ---- thin tube mesh for the rod (updated every frame) ----
tube = UsdGeom.Mesh.Define(stage, "/World/cable")
_counts, _idx = [], []
for i in range(N - 1):
    for j in range(VIS_SEG):
        jn = (j + 1) % VIS_SEG
        a, b = i * VIS_SEG + j, i * VIS_SEG + jn
        d, e = (i + 1) * VIS_SEG + j, (i + 1) * VIS_SEG + jn
        _counts.append(4); _idx.extend((a, b, e, d))
tube.CreateFaceVertexCountsAttr(_counts)
tube.CreateFaceVertexIndicesAttr(_idx)
tube.CreatePointsAttr([Gf.Vec3f(0, 0, 0)] * (N * VIS_SEG))
tube.CreateDisplayColorAttr([Gf.Vec3f(0.04, 0.04, 0.05)])
_mat = UsdShade.Material.Define(stage, "/World/cable/mat")
_sh = UsdShade.Shader.Define(stage, "/World/cable/mat/sh")
_sh.CreateIdAttr("UsdPreviewSurface")
_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.04, 0.04, 0.05))
_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
_mat.CreateSurfaceOutput().ConnectToSource(_sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI(tube.GetPrim()).Bind(_mat)

_thetas = [2.0 * math.pi * j / VIS_SEG for j in range(VIS_SEG)]


def update_tube(P):
    """Sweep a thin circle along the rod centreline P (N,3) with a
    parallel-transport frame, write the tube points."""
    tang = np.gradient(P, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(tang[0] @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    nrm = ref - tang[0] * float(tang[0] @ ref)
    nrm /= np.linalg.norm(nrm) + 1e-9
    pts = []
    for i in range(N):
        t = tang[i]
        nrm = nrm - t * float(t @ nrm)
        ln = np.linalg.norm(nrm)
        if ln < 1e-6:
            r2 = np.array([0.0, 1.0, 0.0]) if abs(t[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
            nrm = r2 - t * float(t @ r2); ln = np.linalg.norm(nrm)
        nrm /= ln
        b = np.cross(t, nrm)
        ci = P[i]
        for th in _thetas:
            p = ci + VIS_RADIUS * (math.cos(th) * nrm + math.sin(th) * b)
            pts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
    tube.GetPointsAttr().Set(pts)


# ===============================================================
# 5. RUN
# ===============================================================
if not HEADLESS:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        from omni.isaac.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array([1.6, 1.8, 2.0]), target=np.array([0.0, 0.0, 1.2]))

rgb_annotator = None
ffmpeg_proc = None
if not HEADLESS and (RECORD or os.environ.get("CABLE_SHOT", "0") == "1"):
    import omni.replicator.core as rep
    rp = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([rp])

world.reset()
update_tube(x_wp.numpy())

print("=" * 70)
print(f"Warp elastic-rod cable  N={N}  L={LENGTH} m  vis_radius={VIS_RADIUS*1000:.1f} mm  device={DEVICE}")
print(f"  steady end @ {ANCHOR_POS}   moving handle @ {HANDLE_POS}")
print("=" * 70)
if not HEADLESS:
    print("MOVE THE CABLE: select the ORANGE handle cube, press W, drag the gizmo")
    print("axis (or Shift+Left-drag).  The steady blue end stays put. Close to quit.\n")


def handle_world_pos():
    p, _ = handle_cube.get_world_pose()
    p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
    return p


step = 0
total = int(MAX_TIME / RENDER_DT)
import time as _t
t0 = _t.perf_counter()
try:
    # When recording (or self-testing) run a FINITE clip and SCRIPT the handle so
    # the cable visibly deforms; otherwise (interactive GUI) loop until the user quits.
    while simulation_app.is_running() and ((not HEADLESS and not RECORD) or step < total):
        if SELFTEST:
            ang = 0.6 * math.sin(2.0 * math.pi * 0.25 * step * RENDER_DT)
            hp = np.array([0.4 + 0.3 * math.sin(ang), 0.3 * math.sin(2 * ang),
                           1.5 - 0.4 * abs(math.sin(ang))])
            handle_cube.set_world_pose(hp, None)
            handle = hp
        elif RECORD:
            handle = HANDLE_POS          # ignored (end is free in record mode)
        else:
            handle = handle_world_pos()  # interactive: follow the mouse-moved cube

        # Record mode: one fixed (blue) anchor, free far end FALLS and drapes on the
        # sphere. The rod is floppy (BEND_COMP) so it falls/curls like a real rope.
        P = step_rod(ANCHOR_POS, handle, pin_end=(0 if RECORD else 1))
        update_tube(P)
        world.step(render=(not HEADLESS))
        step += 1

        if rgb_annotator is not None and RECORD:
            d = rgb_annotator.get_data()
            if d is not None and getattr(d, "size", 0) > 0:
                rgb = np.ascontiguousarray(d[:, :, :3], dtype=np.uint8)
                if ffmpeg_proc is None:
                    ffmpeg_proc = _warp_ffmpeg(rgb.shape[1], rgb.shape[0])
                ffmpeg_proc.stdin.write(rgb.tobytes())

        if step % 60 == 0:
            seglen = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
            end_err = float(np.linalg.norm(P[-1] - handle))
            rtf = (step * RENDER_DT) / max(_t.perf_counter() - t0, 1e-9)
            print(f"t={step*RENDER_DT:5.2f}s  rod_len={seglen:.3f} (rest {LENGTH})  "
                  f"end_follow_err={end_err*1000:.2f} mm  min_z={P[:,2].min():.3f}  rtf={rtf:.2f}x")
except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print(f"Video saved: {VIDEO_PATH}")
    print("Done.")
    simulation_app.close()
