# FEM cable thickness — why it can't be thin (and what actually worked)

This documents the one real limitation in [`cable_fem.py`](cable_fem.py): the
simulated cable **cannot be made as thin as a real 1.5 mm robot cable**. The
floor is **~15 mm radius (30 mm diameter)**. This is not a tuning choice — it
falls out of how PhysX builds a deformable (FEM) body, and below it the cable
either silently breaks or fails to start. Everything here was measured on this
machine (RTX 3080, Isaac Sim 5.1); the numbers are from real runs.

For a genuinely thin (1.5 mm) cable, use the capsule-chain model
[`cable.py`](cable.py) — that's what it's for. FEM trades thinness for true
volumetric contact.

---

## TL;DR

| Sim radius | Diameter | Resolution used | Result |
|-----------:|---------:|----------------:|--------|
| **15 mm**  | 30 mm    | 100 (≈3 voxels across) | ✅ works — cooks, spans the full 1 m, both attachments hold |
| 12.5 mm    | 25 mm    | ~120 (≈2.4 across)     | ⚠️ borderline — at/just under the cooking ceiling |
| 10 mm      | 20 mm    | 150 (auto, ≈3 across)  | ❌ **cooking fails** (`createVoxelTetrahedronMesh failed`) |
| 10 mm      | 20 mm    | 100 (forced)           | ❌ cooks, but **sim mesh truncates to x ≤ 0.70 m** → far end has no nodes, attachment grabs nothing, cube falls off |
| 8 mm       | 16 mm    | 180 (auto, ≈3 across)  | ❌ **cooking fails** |

**Working thinness today: 15 mm radius.** The material is then rescaled so this
fat rod still *bends and weighs* like the real 1.5 mm cable (see below), but it
*looks* ~30 mm thick.

---

## Why the cable is fattened at all

A real robot cable is ~1.5 mm radius. A PhysX FEM body is built by **voxelizing**
the mesh into hexahedra and tetrahedralizing those. A rod only bends correctly
when its **diameter spans ≥ 3 voxels** — with fewer, there is no room for a
strain gradient across the cross-section, so the rod behaves like a rigid stick.

To get 3 voxels across a 1.5 mm-diameter rod over a 1 m length you would need a
voxel resolution in the thousands, which the cooker cannot build. So instead we
simulate a **fat** rod (15 mm) at a modest resolution and rescale the material:

- **Bending:** `E_sim = E_real · (r_real / R_sim)^exp` (exp = 3 by default) so the
  fat rod's flexural rigidity `EI` is close to the thin cable's.
- **Mass:** `ρ_sim = ρ_real · (r_real / R_sim)²` so the fat rod weighs the real
  ~8 g/m (gravity sag/swing stays physical).

The visible tube is thick, but its weight-per-length and bend-vs-gravity
response match a real TPU cable.

---

## The two limits that collide

Going thinner means a smaller diameter, which (to keep ≥3 voxels across) needs a
**higher resolution**. But resolution is bounded from above. The two constraints
pull against each other:

**Limit 1 — full-length coverage needs enough voxels across the diameter.**
The voxel size is `L / resolution`. To get the target number of voxels across
the diameter:

```
resolution  ≥  TARGET_ACROSS · L / (2 · R)        (TARGET_ACROSS = 3, L = 1 m)
```

If the resolution is too low for the chosen radius, the voxelizer does **not**
fail loudly — it quietly produces a body that **doesn't cover the whole cable**.
Measured: a 10 mm-radius rod at resolution 100 produced a simulation mesh whose
nodes only reached **x = 0.70 m** of the 1 m cable. The far 30 cm then has no
simulation nodes, so the end attachment there grabs nothing and the end falls
free. (You see it as the free cube dropping straight to the floor.)

**Limit 2 — PxTetMaker fails to cook above resolution ≈ 120.**
Measured: resolution 150 and 180 both abort with

```
[Error] [omni.physx.cooking.plugin] PxTetMaker::createVoxelTetrahedronMesh failed
[Error] [omni.physx.cooking.plugin] Creating tetrahedral meshes for deformable simulation failed: /World/FemCable/mesh
```

and the body never initializes.

**Putting them together:**

```
3·L/(2·R) ≤ resolution ≤ 120
⇒ R ≥ 3·L / (2·120) = 3·1.0 / 240 ≈ 0.0125 m = 12.5 mm
```

So the theoretical floor is ~12.5 mm radius; **15 mm is the validated safe
value** (resolution 100, exactly 3 voxels across, comfortably under the cooking
ceiling).

---

## How it was tested

Each case is one headless run of `cable_fem.py` with env overrides. The two
things checked:

1. **Did cooking succeed?** Count the failure lines:
   ```bash
   grep -c "createVoxelTetrahedronMesh failed" <logfile>
   # 0 = cooked OK,  >0 = failed
   ```
2. **Did the sim mesh span the full length?** Look at `max_x` in the first row
   of the output CSV (`cable_output/fem/trajectory.csv`). The cable runs x = 0→1 m,
   so a healthy body reports `max_x ≈ 1.00` from frame 1; a truncated body reports
   `max_x ≈ 0.70` and never changes (it's a property of the mesh, not the motion).

The commands run (all headless, no video, short):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab

# 15 mm @ res 100  -> works, max_x ~ 1.00
CABLE_HEADLESS=1 CABLE_RECORD=0 CABLE_MAX_TIME=3 \
    python scripts/cable_simulation/cable_fem.py

# 10 mm, auto resolution (=150) -> cooking fails
CABLE_HEADLESS=1 CABLE_RECORD=0 CABLE_SIM_RADIUS=0.010 CABLE_MAX_TIME=2 \
    python scripts/cable_simulation/cable_fem.py

# 10 mm forced to res 100 -> cooks, but max_x truncates to ~0.70
CABLE_HEADLESS=1 CABLE_RECORD=0 CABLE_SIM_RADIUS=0.010 CABLE_FEM_RES=100 \
    CABLE_MAX_TIME=3 python scripts/cable_simulation/cable_fem.py

# 8 mm, auto resolution (=180) -> cooking fails
CABLE_HEADLESS=1 CABLE_RECORD=0 CABLE_SIM_RADIUS=0.008 CABLE_MAX_TIME=2 \
    python scripts/cable_simulation/cable_fem.py
```

A separate probe confirmed the coverage claim directly by reading the cooked
rest mesh (`get_simulation_mesh_rest_points`): at 15 mm / res 100 the body has
**2001 nodes spanning x = 0 → 1.0 m, 101 layers ~10 mm apart, 3 across** — the
full cable; at 10 mm the x-extent fell short of 1.0 m.

`cable_fem.py` encodes these limits so you don't trip over them silently:
- the resolution is auto-picked as `clamp(⌈3·L/(2·R)⌉, 24, 120)`;
- setting `CABLE_SIM_RADIUS` below 13 mm prints a warning that the mesh will
  truncate or cooking will fail.

---

## What this means in practice

- **Use 15 mm (the default).** It's the thinnest reliably-simulatable FEM rod
  here. The material rescaling makes it bend and weigh like the real cable.
- **Don't chase 1.5 mm in FEM.** It is not achievable on this path; the rod
  would be rigid even if it cooked.
- **Need a truly thin cable?** Use [`cable.py`](cable.py) (capsule chain) — it
  models the real 1.5 mm geometry directly.
- **Need the thin *look* with FEM contact?** A possible (not-yet-built) option
  is to keep the 15 mm body for simulation/collision and skin a thin (~1.5 mm)
  visual tube to it — physics stays 15 mm, appearance ~1.5 mm. Ask if you want
  this.
