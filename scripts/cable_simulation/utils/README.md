# Cable profile extraction from a photo / video

Measure a real cable's 2-D shape (height `z` vs horizontal `x`, in **metres**)
from a phone **photo** or **video**, using SAM segmentation. The profile is
saved as a CSV for later analysis.

> **Important — what this does and does NOT do.**
> This extracts the cable's **shape**. It does **not** identify the material
> stiffness. A static cable hung between two fixed points under gravity has a
> shape set almost entirely by its length and the span (inextensibility): a
> stiff cable and a floppy cable of the same size hang **almost identically**,
> so the static shape cannot tell them apart. Identifying stiffness needs a
> **dynamic** experiment (e.g. film the cable oscillating and measure its
> frequency). The profile here is the clean input kept for those methods.

---

## TL;DR — run it

```bash
cd scripts/cable_simulation/utils

# extract the profile from a photo:
./run_image.sh media/IMG_0501.JPG
```

Results land in **`results/IMG_0501/`** (`media/` stays inputs-only):

```
results/IMG_0501/
  profile.csv   cable shape:  x_m, z_m   (metres)
  profile.png   the extracted shape
```

---

## Setup (once)

```bash
pip install ultralytics opencv-contrib-python scipy matplotlib imageio
```

* `ultralytics` — SAM segmentation (best model auto-downloads on first run into
  `models/`, then reused; CUDA / Apple-MPS / CPU chosen automatically).

Use your project python, e.g.:

```bash
PYTHON=/Users/ali/Workspace/PhD/IsaacLab/.isaacenv/bin/python \
  ./run_image.sh media/IMG_0501.JPG
```

---

## The interactive window

When you run it, a window opens. All actions happen **in the window** (the
terminal is only used to type the scale distance):

| key / click | action |
|-------------|--------|
| `r` / `e`   | rotate the image +90 / -90 (make the cable upright), then `ENTER` |
| left-click  | the two **scale** points (then type the distance in the terminal) |
| `b` + drag  | draw a **box** around the cable (best way to exclude background) |
| left-click  | a point **on** the cable (positive) |
| right-click | a point on the **wrong** thing (negative) — use sparingly (<3) |
| `ENTER`     | run / re-run SAM |
| `a`         | **accept** the mask and continue |
| `u`         | undo the last box / click |

**Tip:** press `b` and box the cable first, then one left-click on it, then
`ENTER`. The mask is forced to be **one continuous piece** (far same-colour
clutter is dropped).

---

## The scale reference (pixels → metres)

A photo has no scale. You click **two points** whose real distance you measured
with a tape (a "ruler" in the scene — they need **not** be on the cable), and
type that distance. Everything is then in metres. The cable **length** comes
out of this automatically (the profile's arc length).

---

## Taking the photo (accuracy)

1. **Square-on** — phone perpendicular to the plane the cable hangs in. Tilt
   bends straight lines and corrupts the metres.
2. The cable should sag in **one vertical plane**.
3. The **scale reference** must lie in the **same plane** as the cable.
4. Whole cable visible, plain background, no motion blur.

Run `python extract_cable_profile.py --help-photo` for the full checklist.

---

## Video (profile over time)

```bash
python extract_cable_profile.py video --video media/IMG_0507.MOV
```

You set the scale + segment the cable on **frame 1** (same window as photo,
plus `r`/`e` rotation applied to the whole video); SAM then tracks it through
every frame. Outputs (named after the video):

```
IMG_0507_profile.csv   profile per frame  (t, p*_x, p*_z)
IMG_0507_result.mp4    overlay video (mask + centre-line per frame)
IMG_0507_result.gif    same as a GIF (needs imageio)
```

This per-frame profile is the right input for a **dynamics-based** stiffness
identification later (track a point's oscillation → frequency → EI).

---

## Running the tools individually

```bash
# photo -> profile
python extract_cable_profile.py photo --image media/IMG_0501.JPG \
    --out results/IMG_0501/profile.csv

# compare a profile to an Isaac-Sim run
python extract_cable_profile.py compare --real results/IMG_0501/profile.csv \
    --sim ../base/slides_output/fem/cable_only/trajectory.csv --plot cmp.png
```

Useful knobs (also env vars for `run_image.sh`: `ROTATE`, `SMOOTH`, `FIT`):

| flag | meaning |
|------|---------|
| `--rotate 90` | starting rotation (still tunable live with `r`/`e`) |
| `--smooth 2`  | smoother centre-line (higher = smoother, default 1) |
| `--fit catenary\|spline\|auto` | curve reconstruction (default auto) |

---

## Files

| file | what it is |
|------|------------|
| `run_image.sh` | one-command profile extraction for a photo |
| `extract_cable_profile.py` | SAM segmentation + metric profile (photo/video/compare) |
| `cable_segment.py` | standalone SAM cable segmentation (shared engine) |
| `media/` | your input photos / videos (git-ignored) |
| `models/` | SAM weights, auto-downloaded (git-ignored) |
| `results/` | per-image output folders (git-ignored) |

---

## Guessed config: Apple woven USB-C cable

There is **no published Young's modulus** for an Apple cable — it's a composite
(copper + shielding + insulation + woven polyester jacket), and bending is
governed by the whole-cable `EI`, not a single material `E`. The values below
are **educated guesses** for the 1 m woven USB-C charge cable, meant as a
starting point for the simulation. **Tune them** until the simulated droop/sag
matches a photo of your real cable (the woven jacket makes it noticeably
**stiffer** than the plain-TPU default).

How these were guessed (vs the TPU default `E=40 MPa`, `r=1.5 mm`, `~8 g`):
the woven cable is **thicker** (~2 mm radius), **heavier** (~30 g/m, more copper
+ fabric), and **stiffer** (woven jacket → effective `E ~ 80 MPa`, roughly 2×).

| property | value | note |
|----------|-------|------|
| length `L` | 1.0 m | the standard cable |
| radius `r` | 2.0 mm | thicker than TPU baseline |
| Young's modulus `E` | 80 MPa | **guess**: ~2× stiffer than TPU (woven jacket) |
| Poisson ratio `ν` | 0.45 | typical polymer/elastomer composite |
| density `ρ` | 2390 kg/m³ | high — the copper core makes effective bulk density large |
| mass | ~30 g | matches `ρ·π·r²·L` for a 1 m, 2 mm-radius cable |

Run the sim with these as environment overrides (no need to edit
`cable_config.py`):

```bash
CABLE_LENGTH=1.0 \
CABLE_RADIUS=0.002 \
CABLE_E=80e6 \
CABLE_NU=0.45 \
CABLE_DENSITY=2390 \
CABLE_MASS=0.030 \
python ../base/cable.py
```

Or paste into `cable_config.py`:

```python
TOTAL_CABLE_LENGTH = 1.0       # m
REAL_RADIUS        = 0.002     # m   (2 mm)
YOUNG_MODULUS      = 80e6      # Pa  (GUESS: stiff woven jacket)
POISSON_RATIO      = 0.45
DENSITY            = 2390.0    # kg/m^3  (high: copper core)
CABLE_MASS         = 0.030     # kg  (~30 g, override the derived value)
```

> ⚠️ These are **guesses, not measurements.** For the real stiffness, measure it
> (cantilever droop test, or the dynamics/oscillation method noted above) —
> `EI = μ g s⁴ / (8 δ)` from a horizontal cantilever of free length `s`, tip
> droop `δ`, mass-per-length `μ`. Then `E = EI / (π r⁴ / 4)`.
