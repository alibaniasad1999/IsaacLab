# Cable measurement & identification from a photo

Measure a real cable's 2-D shape from a phone **photo**, then identify its
material properties (bending stiffness, Young's modulus) so you can reproduce it
in the Isaac-Sim cable model.

The cable **length is measured from the image** — you do not need to type it.

---

## TL;DR — run it

```bash
cd scripts/cable_simulation/utils

# everything in one command (segment -> profile -> identify):
./run_image.sh media/IMG_0501.JPG 8.1 1.5
#                  ^image          ^mass[g] ^radius[mm]
```

Results land in **`results/IMG_0501/`** (not in `media/`, which stays
inputs-only):

```
results/IMG_0501/
  profile.csv          cable shape:  x_m, z_m   (metres)
  profile.png          the extracted shape
  profile_fit.png      the elastica fit drawn over your data
  identification.txt   Gamma / EI / Young's modulus + cable_config values
```

---

## What you provide

| Input | Needed for | How |
|-------|------------|-----|
| **photo** | everything | side-on phone photo of the hanging cable |
| **a scale reference** | metres (not pixels) | in the photo, click two points whose real distance you measured with a tape |
| **mass (g)** | absolute stiffness `EI`, Young's modulus `E` | weigh the cable on a kitchen scale |
| **radius (mm)** | Young's modulus `E` | measure / from the spec sheet |
| ~~length~~ | — | **measured automatically from the image** |

Without mass/radius you still get the dimensionless stiffness `Gamma` (how
floppy the cable is); with them you get absolute `EI`, `E`, density.

---

## Setup (once)

```bash
pip install ultralytics opencv-contrib-python scipy matplotlib imageio
```

* `ultralytics`  — SAM segmentation (best model auto-downloads on first run
  into `models/`, then reused; runs on CUDA / Apple-MPS / CPU automatically).
* the rest — image I/O, curve fitting, plotting, GIF.

Use your project python, e.g.:

```bash
PYTHON=/Users/ali/Workspace/PhD/IsaacLab/.isaacenv/bin/python \
  ./run_image.sh media/IMG_0501.JPG 8.1 1.5
```

---

## The interactive window (STEP 1)

When `run_image.sh` segments the cable, a window opens. All actions happen
**in the window** (the terminal is only used to type the scale distance):

| key / click | action |
|-------------|--------|
| `r` / `e`   | rotate the image +90 / -90 (make the cable upright), then `ENTER` |
| left-click  | click the two **scale** points (then type the distance in the terminal) |
| `b` + drag  | draw a **box** around the cable (best way to exclude the background) |
| left-click  | a point **on** the cable (positive) |
| right-click | a point on the **wrong** thing (negative) — use sparingly (<3) |
| `ENTER`     | run / re-run SAM |
| `a`         | **accept** the mask and continue |
| `u`         | undo the last box / click |

**Tip for a clean result:** press `b` and box the cable first, then one
left-click on it, then `ENTER`. The mask is forced to be **one continuous
piece** (far same-colour clutter is dropped).

---

## Taking the photo (accuracy)

1. **Square-on** — phone perpendicular to the plane the cable hangs in. Tilt
   bends straight lines and corrupts the metres.
2. The cable should sag in **one vertical plane** (the two ends pinned, sagging
   between them) — like the simulation.
3. The **scale reference** (the two points you measure) must lie in the **same
   plane** as the cable (not closer to the lens).
4. Whole cable visible, plain background, no motion blur.
5. Measure **one real distance** with a tape before you shoot (you type it in).

Run `python extract_cable_profile.py --help-photo` for the full checklist.

---

## Reading the result (`identification.txt`)

```
hanging-shape stiffness   Gamma = mu g L^3 / EI = 12        well-identified
Young's modulus   E    = 1.67e+03 MPa
----- matching cable_config.py values -----
YOUNG_MODULUS = 1.666e+09  # Pa
DENSITY       = 1145.9      # kg/m^3
...
```

* **Gamma** = gravity-sag ÷ bending-resistance. Small = stiff, large = floppy.
* If it says **`well-identified`** the number is trustworthy. If it says
  **`LOWER bound (too floppy)`**, the cable hangs like an ideal rope and its
  bending stiffness is too small to measure from this shape — re-hang it from a
  **shorter span** (less sag) so bending matters.
* Always check **`profile_fit.png`**: the blue fitted curve should hug the red
  measured points. A large `elastica fit error` (mm) means don't trust it.

The printed `cable_config.py` block / `CABLE_E=... python ../base/cable.py`
command lets you drop the identified cable straight into the Isaac-Sim model.

---

## Running the tools individually

`run_image.sh` just chains these — run them yourself for more control:

```bash
# 1) segment + extract the metric profile
python extract_cable_profile.py photo --image media/IMG_0501.JPG \
    --out results/IMG_0501/profile.csv

# 2) identify the cable (length auto-measured from the profile)
python identify_cable.py --profile results/IMG_0501/profile.csv \
    --mass-g 8.1 --radius-mm 1.5 --overlay

# 3) (optional) compare the profile to an Isaac-Sim run
python extract_cable_profile.py compare --real results/IMG_0501/profile.csv \
    --sim ../base/slides_output/fem/cable_only/trajectory.csv --plot cmp.png
```

Useful knobs (also work as env vars for `run_image.sh`: `ROTATE`, `SMOOTH`,
`FIT`):

| flag | meaning |
|------|---------|
| `--rotate 90` | starting rotation (still tunable live with `r`/`e`) |
| `--smooth 2`  | smoother centre-line (higher = smoother, default 1) |
| `--fit catenary\|spline\|auto` | curve reconstruction (default auto) |
| `--length-m 1.0` | override the image-measured length (rarely needed) |

---

## Files

| file | what it is |
|------|------------|
| `run_image.sh` | the one-command workflow (segment → profile → identify) |
| `extract_cable_profile.py` | SAM segmentation + metric profile (photo/video/compare) |
| `cable_segment.py` | standalone SAM cable segmentation (shared engine) |
| `identify_cable.py` | elastica fit → Gamma / EI / Young's modulus |
| `media/` | your input photos / videos (git-ignored) |
| `models/` | SAM weights, auto-downloaded (git-ignored) |
| `results/` | per-image output folders (git-ignored) |
```
