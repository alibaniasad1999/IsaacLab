"""
extract_cable_profile.py
========================

Extract the 2-D height profile (height z vs horizontal x, in METRES) of a
*real* cable from a phone PHOTO or VIDEO, so it can be overlaid -- on the same
axes -- on the Isaac-Sim cable trajectories produced by the scripts in
``scripts/cable_simulation/base/`` (``trajectory.csv``).

----------------------------------------------------------------------------
THE PHYSICAL SETUP THIS CODE ASSUMES
----------------------------------------------------------------------------
You record a cable from the SIDE with a phone (2-D, no depth):

    * One end of the cable is ANCHORED at a fixed point.
    * You PULL some other point of the cable by a KNOWN amount (e.g. you lift
      a point to a known height above the floor).
    * The rest of the cable drapes / hangs / partly rests on the floor.

The cable can be ANY shape and ANY length -- we do NOT assume the 1 m sim
length. You give the code a few real-world MEASUREMENTS (metres) you took with
a tape, plus a few POINTS (clicked, or tracked from video).

----------------------------------------------------------------------------
THREE INPUT MODES
----------------------------------------------------------------------------
photo   : click points on a still image.                 -> one cable profile
video   : draw several boxes along the cable in frame 1; CSRT tracks each box
          through the video.  Every frame the box CENTRES give the cable shape,
          so you get the profile EVOLVING over time (compare to the sim's
          time evolution).  Also dumps a representative still frame.
compare : overlay a real profile on a sim trajectory.csv and report the error.

----------------------------------------------------------------------------
HOW THE CABLE CURVE IS RECONSTRUCTED  (--fit)
----------------------------------------------------------------------------
From a handful of points we have to reconstruct the continuous cable. Two ways:

  catenary  (default): a cable hanging under gravity forms a CATENARY,
            z = a*cosh((x - x0)/a) + c. Fitting this gives the physically
            correct shape from VERY FEW points, and the fit RESIDUAL tells you
            how "ideal" your real cable is. Auto-falls back to spline if the
            cable clearly is not a pure hang (e.g. it rests on the floor) and
            the residual is poor.
  spline    : a smooth arc-length spline through the points -- free-form, makes
            no physics assumption, handles floor contact / S-shapes. Does not
            infer anything between points.

----------------------------------------------------------------------------
SCALE & FRAME  (matched to the sim)
----------------------------------------------------------------------------
A single 2-D image has no scale: pixels are not metres. You give ONE known
real distance (two reference points + the tape-measured metres between them)
-> metres-per-pixel. The sim reports metres in a vertical x-z plane (y~0):
    x = horizontal,  z = height (up).
We reproduce that exactly: origin = the ANCHOR end you click, +x toward the
pulled end, +z up. Both curves then share x=0 at the anchor.

Run ``python extract_cable_profile.py --help-photo`` for the shooting checklist.

----------------------------------------------------------------------------
EXAMPLES
----------------------------------------------------------------------------
  # still photo
  python extract_cable_profile.py photo --image cable.jpg --out real.csv

  # video: 6 tracked boxes along the cable, catenary fit
  python extract_cable_profile.py video --video cable.mp4 --boxes 6 \
      --out real_over_time.csv --frame-out keyframe.png

  # compare to a sim run
  python extract_cable_profile.py compare --real real.csv \
      --sim ../base/slides_output/fem/cable_only/trajectory.csv --plot cmp.png
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# LOGGER  -- so you can always SEE what the program is doing and never feel
# "stuck". Every step prints; long loops print a percentage. Set LOGLEVEL=DEBUG
# for extra detail.
# --------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cable")


def _progress(done: int, total: int, what: str, every: float = 1.0,
              _state={}):
    """Log a percentage at most once per `every` seconds (so it's readable)."""
    now = time.time()
    last = _state.get(what, 0.0)
    if done >= total or now - last >= every:
        _state[what] = now
        pct = 100.0 * done / max(total, 1)
        log.info("  %s: %d/%d  (%.0f%%)", what, done, total, pct)

# --- matplotlib (clicking + plotting) -------------------------------------
try:
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "MacOSX")
                   if sys.platform == "darwin" else "TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
except Exception as exc:  # pragma: no cover - environment dependent
    matplotlib = plt = imread = None
    _MPL_ERR = exc
else:
    _MPL_ERR = None


# ===========================================================================
# Photo / video checklist
# ===========================================================================
PHOTO_CHECKLIST = """
SHOOTING CHECKLIST  (read first -- this is where accuracy is won or lost)
-----------------------------------------------------------------------------
1. SQUARE-ON. Point the phone PERPENDICULAR to the plane the cable hangs in.
   Shooting up / down / at an angle bends straight lines -> wrong metres.
   Stand at the cable's mid-height and step back + zoom rather than up close.

2. FLAT PLANE. Pull the cable in ONE vertical plane (like the sim); don't let
   it swing toward / away from the camera.

3. SCALE REFERENCE IN THE SAME PLANE. The two points whose real distance you
   measure (pulled height above floor, end-to-end span, or a ruler) must be at
   the SAME distance from the camera as the cable.

4. WHOLE CABLE VISIBLE, good contrast, plain background, steady phone (no blur).

5. MEASURE one real distance with a tape BEFORE you leave (you'll type it in).

VIDEO ONLY:
6. Keep the phone STILL on a tripod / surface for the whole clip -- the metric
   frame is set from frame 1 and assumed fixed. If the camera moves, metres
   drift. Start recording BEFORE you pull, so frame 1 shows a clear cable.
-----------------------------------------------------------------------------
""".strip()


# ===========================================================================
# Data structures
# ===========================================================================
@dataclass
class CableProfile:
    """One cable centre-line in metres, in the sim's x-z frame."""

    x_m: np.ndarray
    z_m: np.ndarray
    source: str = ""
    fit: str = ""

    def arc_length(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x_m), np.diff(self.z_m))))

    def write_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x_m", "z_m"])
            for x, z in zip(self.x_m, self.z_m):
                w.writerow([f"{x:.6f}", f"{z:.6f}"])

    @classmethod
    def read_csv(cls, path: str) -> "CableProfile":
        xs, zs = [], []
        with open(path, newline="") as fh:
            r = csv.DictReader(fh)
            cols = {c.lower(): c for c in (r.fieldnames or [])}
            if "x_m" not in cols or "z_m" not in cols:
                raise ValueError(
                    f"{path}: expected columns x_m,z_m (got {r.fieldnames}). "
                    "For a sim trajectory.csv use the 'compare' command.")
            for row in r:
                xs.append(float(row[cols["x_m"]]))
                zs.append(float(row[cols["z_m"]]))
        return cls(np.asarray(xs), np.asarray(zs), source=path)


@dataclass
class TimeSeriesProfile:
    """A cable profile per video frame -> shape evolving over time."""

    times: list[float] = field(default_factory=list)
    frames: list[CableProfile] = field(default_factory=list)

    def write_csv(self, path: str) -> None:
        """Wide CSV: t, p0_x..pN_x, p0_z..pN_z -- same flavour as the sim CSV
        so the 'compare' command and existing tooling can read a frame."""
        n = len(self.frames[0].x_m)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            head = (["t"]
                    + [f"p{i}_x" for i in range(n)]
                    + [f"p{i}_z" for i in range(n)])
            w.writerow(head)
            for t, fr in zip(self.times, self.frames):
                w.writerow([f"{t:.4f}"]
                           + [f"{v:.6f}" for v in fr.x_m]
                           + [f"{v:.6f}" for v in fr.z_m])


# ===========================================================================
# Curve fitting  (catenary / spline)
# ===========================================================================
def _fit_spline(px, pz, n_samples, smooth=1.0):
    """Smoothing spline along the cable (arc-length parameterised).

    Uses a parametric SMOOTHING spline (splprep with s>0) so it irons out
    pixel-level jitter from the mask while still following the real cable
    bends -- not an interpolating spline that chases every wiggle.

    `smooth` scales the smoothing strength: the spline is allowed an RMS
    deviation of ~`smooth` data-units (metres) from the points, so 1.0 here
    means a couple of millimetres of jitter is smoothed away (the points are
    in metres). Set higher for a smoother line, lower to hug the points.
    """
    pts = np.column_stack([px, pz]).astype(float)
    keep = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
    pts = pts[keep]
    if len(pts) < 2:
        raise ValueError("Need >=2 distinct points to fit a curve.")
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    t = np.concatenate([[0.0], np.cumsum(seg)])
    t /= t[-1]
    tt = np.linspace(0, 1, n_samples)
    try:
        from scipy.interpolate import splprep, splev
        n = len(pts)
        # scale the data so 'smooth' is in pixel-ish jitter units regardless of
        # the metre scale: target RMS dev ~ smooth * (median point spacing).
        spacing = float(np.median(seg)) if len(seg) else 1.0
        s = (smooth * spacing) ** 2 * n          # splprep's s is sum of sq res
        k = min(3, n - 1)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], u=t, s=s, k=k)
        sx, sz = splev(tt, tck)
        return np.asarray(sx), np.asarray(sz)
    except Exception:
        return np.interp(tt, t, pts[:, 0]), np.interp(tt, t, pts[:, 1])


def _fit_catenary(px, pz, n_samples):
    """Fit z = a*cosh((x-x0)/a) + c. Returns (x, z, rms_residual_m).

    Requires SciPy. Raises if it cannot fit so the caller can fall back.
    """
    from scipy.optimize import curve_fit

    px = np.asarray(px, float)
    pz = np.asarray(pz, float)
    order = np.argsort(px)
    px, pz = px[order], pz[order]
    span = max(px.max() - px.min(), 1e-6)

    def cat(x, a, x0, c):
        return a * np.cosh((x - x0) / a) + c

    # a~span is a sane start; x0 at the lowest point; c lifts to data.
    p0 = [span, px[np.argmin(pz)], pz.min() - span]
    popt, _ = curve_fit(cat, px, pz, p0=p0, maxfev=20000)
    resid = float(np.sqrt(np.mean((cat(px, *popt) - pz) ** 2)))
    xs = np.linspace(px.min(), px.max(), n_samples)
    return xs, cat(xs, *popt), resid


def fit_profile(px, pz, fit: str, n_samples: int,
                smooth: float = 1.0) -> CableProfile:
    """Reconstruct a continuous profile from points using --fit policy."""
    px = np.asarray(px, float)
    pz = np.asarray(pz, float)
    if fit == "spline":
        sx, sz = _fit_spline(px, pz, n_samples, smooth=smooth)
        return CableProfile(sx, sz, fit="spline")
    if fit == "catenary":
        sx, sz, _ = _fit_catenary(px, pz, n_samples)
        return CableProfile(sx, sz, fit="catenary")
    # auto: try catenary, fall back to smoothing spline if poor / it fails.
    try:
        sx, sz, resid = _fit_catenary(px, pz, n_samples)
        span = max(px.max() - px.min(), 1e-6)
        if resid < 0.05 * span:           # within 5% of horizontal span
            return CableProfile(sx, sz, fit=f"catenary(rms={resid*1000:.0f}mm)")
        print(f"  catenary residual {resid*1000:.0f} mm is large "
              f"(>5% of {span:.2f} m span) -> using smoothing spline instead.")
    except Exception as exc:
        print(f"  catenary fit failed ({exc}) -> using smoothing spline.")
    sx, sz = _fit_spline(px, pz, n_samples, smooth=smooth)
    return CableProfile(sx, sz, fit="spline")


# ===========================================================================
# Pixels -> metres
# ---------------------------------------------------------------------------
# The SCALE reference is just two points in space whose REAL distance you
# measured (like a ruler held in the scene) -- it is used ONLY to convert
# pixels to metres. It is NOT an anchor and has NOTHING to do with the cable's
# position. The output origin is simply the cable's own left-most point, so
# x>=0 grows to the right and z is height (image +y is down, so we flip it).
# ===========================================================================
def pixels_to_metres(cable_px, scale_p0, scale_p1, scale_real_m):
    """Convert cable pixel points to metres using ONLY the scale reference.

    Returns (x_m, z_m, m_per_px). The scale points set the metres-per-pixel
    ratio; the cable's left-most point becomes the origin (x=0). No anchor,
    no cable endpoint is used for scaling.
    """
    ref_pix = float(np.hypot(*(np.asarray(scale_p1) - np.asarray(scale_p0))))
    if ref_pix <= 0:
        raise ValueError("Scale reference points coincide (0 px apart).")
    m_per_px = scale_real_m / ref_pix

    cable_px = np.asarray(cable_px, float)
    x = cable_px[:, 0] * m_per_px
    z = -cable_px[:, 1] * m_per_px       # flip image-y (down) to height (up)
    # origin at the cable's own extremes (independent of the scale ruler).
    x -= x.min()
    z -= z.min()
    return x, z, m_per_px


# ===========================================================================
# Image rotation (phone-editor style) + edge snapping
# ===========================================================================
def rotate_image(img, deg: int):
    """Rotate an image by a multiple of 90 deg, like the phone photo editor.

    deg is one of 0/90/180/270 (positive = clockwise, matching the phone's
    rotate button). Returns the rotated image array.
    """
    deg = int(deg) % 360
    if deg == 0:
        return img
    if deg not in (90, 180, 270):
        raise SystemExit(f"--rotate must be 0/90/180/270, got {deg}.")
    # np.rot90 turns COUNTER-clockwise; k=3 gives one clockwise 90, etc.
    k = {90: 3, 180: 2, 270: 1}[deg]
    log.info("rotating image %d deg (clockwise)", deg)
    return np.rot90(img, k=k)


def rotate_interactive(fig, ax, img, start_deg: int = 0):
    """Let the user rotate the image LIVE in the matplotlib window.

    Press 'r' to rotate +90 deg (clockwise), 'e' for -90, until the cable looks
    upright, then press ENTER (or close-button / right-click is not used here)
    to lock it in. All clicking afterwards happens on this final orientation.
    Returns the rotated image array.
    """
    state = {"deg": int(start_deg) % 360, "done": False}

    def redraw():
        ax.clear()
        ax.imshow(rotate_image(img, state["deg"]))
        ax.set_title(f"ROTATE: press 'r' (+90) / 'e' (-90) to make the cable "
                     f"upright, then ENTER.   [now {state['deg']} deg]")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "r":
            state["deg"] = (state["deg"] + 90) % 360
            log.info("rotate -> %d deg", state["deg"])
            redraw()
        elif event.key == "e":
            state["deg"] = (state["deg"] - 90) % 360
            log.info("rotate -> %d deg", state["deg"])
            redraw()
        elif event.key in ("enter", "return"):
            state["done"] = True

    redraw()
    log.info("ROTATE the image in the window: 'r'=+90, 'e'=-90, ENTER=accept")
    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    # block here, processing GUI events, until the user presses ENTER.
    while not state["done"] and plt.fignum_exists(fig.number):
        plt.pause(0.05)
    fig.canvas.mpl_disconnect(cid)

    final = rotate_image(img, state["deg"])
    log.info("locked rotation at %d deg", state["deg"])
    # expose the chosen angle so callers (video) can rotate every frame the
    # same way; the photo caller simply ignores this attribute.
    rotate_interactive.last_deg = int(state["deg"])
    # leave the rotated image shown for the clicking steps.
    ax.clear()
    ax.imshow(final)
    fig.canvas.draw_idle()
    return final


def _to_gray(img):
    """Return a float grayscale [0..1] view of an RGB/RGBA/gray image."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a / 255.0
    if a.ndim == 2:
        return a
    if a.shape[-1] >= 3:
        return a[..., :3].mean(axis=-1)
    return a[..., 0]


def snap_to_edge(gray, x, y, win: int = 12):
    """Move a clicked (x, y) onto the strongest nearby cable edge.

    The cable is a thin line that contrasts with the background. In a small
    window around the click we look for the pixel whose local gradient
    (edge strength) is largest and snap there. You click roughly; this refines
    onto the true cable centre-line. Returns the snapped (x, y).
    """
    h, w = gray.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - win), min(w, xi + win + 1)
    y0, y1 = max(0, yi - win), min(h, yi + win + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return x, y
    # edge strength = gradient magnitude (Sobel-like via numpy gradient).
    gy, gx = np.gradient(patch)
    mag = np.hypot(gx, gy)
    # prefer edges, but bias toward the click so we don't jump far away.
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    cy, cx = yi - y0, xi - x0
    dist = np.hypot(xx - cx, yy - cy)
    score = mag * np.exp(-(dist ** 2) / (2 * (win / 2.0) ** 2))
    j, i = np.unravel_index(np.argmax(score), score.shape)
    return float(x0 + i), float(y0 + j)


# ===========================================================================
# CABLE TRACING with SAM (Segment Anything)
# ---------------------------------------------------------------------------
# The actual segmentation lives in cable_segment.py (one source of truth): you
# click point(s) ON the cable (and optionally a BOX around it / negative clicks
# on the wrong stuff); SAM returns the exact cable mask, then we keep only the
# single CONTINUOUS connected piece that contains your clicks. Here we just
# thin that mask to a 1-px centre-line and WALK it end-to-end to get an ordered
# profile (click order does NOT matter; the order is found from the mask).
#
# Install: pip install ultralytics    (best model auto-downloads to
# cable_segment.py's ./models, runs on CUDA / Apple MPS / CPU automatically).
# ===========================================================================
from cable_segment import segment_cable  # noqa: E402  (one source of truth)


def centerline_from_mask(mask, smooth=9, log_fn=log.info):
    """Clean centre-line of a thin cable mask by MEAN-OF-MASK per slice.

    The old skeleton-walk zig-zagged and jumped across spurs. Instead, we slice
    the mask along its LONG axis and, in each slice, take the AVERAGE position
    of the cable pixels -> exactly one smooth, monotone point per slice. No
    spurs, no back-jumps. Then a light moving-average smooths it.

    Axis is AUTO-DETECTED: if the cable's bounding box is wider than tall we
    slice by columns (one point per x); otherwise by rows (one point per y).
    Returns an ordered (N,2) array of (x, y) pixel points along the cable.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return np.column_stack([xs, ys]).astype(float)

    wide = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    pts = []
    if wide:                                   # slice by columns -> mean y per x
        for x in range(xs.min(), xs.max() + 1):
            col = ys[xs == x]
            if len(col):
                pts.append((x, col.mean()))
    else:                                      # slice by rows -> mean x per y
        for y in range(ys.min(), ys.max() + 1):
            row = xs[ys == y]
            if len(row):
                pts.append((row.mean(), y))
    pts = np.array(pts, float)
    log_fn("  centre-line: %d slice points (axis=%s)",
           len(pts), "x" if wide else "y")

    # light moving-average smoothing along the cable (odd window).
    if smooth and len(pts) > smooth:
        k = smooth | 1
        pad = k // 2
        ker = np.ones(k) / k
        sx = np.convolve(np.pad(pts[:, 0], pad, mode="edge"), ker, "valid")
        sy = np.convolve(np.pad(pts[:, 1], pad, mode="edge"), ker, "valid")
        pts = np.column_stack([sx, sy])
    return pts


def trace_cable_from_hints(img, pos_px, neg_px=None, box=None, log_fn=log.info):
    """Full pipeline: prompts -> SAM cable mask -> ordered centre-line.

    pos_px : points ON the cable (positive). neg_px : points NOT the cable.
    box    : optional (x0,y0,x1,y1) to constrain SAM to the cable region.
    Returns (ordered_points (N,2), mask (HxW uint8)) so the caller can show the
    mask overlay + the clean centre-line. The mask is already the single
    continuous connected cable piece (cable_segment handles connectivity).
    """
    rgb = np.asarray(img)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgb = rgb[..., :3]

    mask = segment_cable(rgb, pos_px, neg_xy=neg_px, box=box)
    ordered = centerline_from_mask(mask, log_fn=log_fn)
    if len(ordered) < 5:
        raise SystemExit(
            "Segmentation produced too few centre-line points. Click more "
            "points ON the cable, or draw a tighter BOX around it.")
    log_fn("traced cable: %d ordered centre-line points", len(ordered))
    return ordered, mask


# ===========================================================================
# On-window instructions  -- so you never have to guess WHERE to act. The
# banner is drawn ON the matplotlib image; when terminal input is needed it
# says "GO TO TERMINAL" explicitly.
# ===========================================================================
def _banner(ax, lines, terminal=False):
    """Draw an instruction box on top of the image. Returns the text artist
    so the caller can remove it before the next step."""
    txt = "\n".join(lines)
    color = "yellow" if terminal else "white"
    edge = "red" if terminal else "black"
    art = ax.text(
        0.5, 0.985, txt, transform=ax.transAxes,
        ha="center", va="top", fontsize=11, family="monospace",
        color=color, zorder=1000,
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor=("darkred" if terminal else "black"),
                  edgecolor=edge, alpha=0.85, linewidth=2))
    ax.figure.canvas.draw_idle()
    plt.pause(0.01)
    return art


def _show_terminal_banner(ax, what):
    """Flash a clear 'look at the terminal now' message on the window."""
    art = _banner(ax, [">>> GO TO THE TERMINAL <<<", what], terminal=True)
    return art


# ===========================================================================
# Interactive clicking (photo)
# ===========================================================================
def _click(ax, n, prompt, color, marker="x", gray=None, snap=False,
           instruction=None):
    """Collect clicks; if snap and gray are given, refine each onto the edge.

    `instruction` is a list of lines drawn ON the image so the user knows what
    to click without looking at the terminal.
    """
    log.info(prompt)
    banner = None
    if instruction:
        banner = _banner(ax, instruction)
    pts = plt.ginput(n=(n if n is not None else -1), timeout=0,
                     show_clicks=True)
    if banner is not None:
        banner.remove()
        ax.figure.canvas.draw_idle()
    arr = np.array(pts, dtype=float)
    if n is not None and len(arr) != n:
        raise SystemExit(f"Expected {n} clicks, got {len(arr)}. Aborting.")
    if snap and gray is not None and len(arr):
        snapped = np.array([snap_to_edge(gray, p[0], p[1]) for p in arr])
        moved = np.hypot(*(snapped - arr).T).mean() if len(arr) else 0.0
        log.info("  snapped %d point(s) to edge (avg move %.1f px)",
                 len(arr), moved)
        arr = snapped
    if len(arr):
        ax.plot(arr[:, 0], arr[:, 1], marker, color=color, ms=8)
        plt.draw()
    log.info("  got %d point(s)", len(arr))
    return arr


def _ask_scale_distance():
    while True:
        try:
            v = float(input(
                "  -> type the REAL distance between A and B, in METRES: "))
            if v > 0:
                return v
        except ValueError:
            pass
        print("     please enter a positive number, e.g. 0.35")


def _segment_interactive(fig, ax, img):
    """Interactive SAM segmentation: box + positive/negative clicks, live.

    LEFT-click = on the cable, RIGHT-click = on the wrong thing (negative),
    'b' = draw a box around the cable, 'u' = undo, ENTER = run/re-run SAM,
    'a' = accept the current mask. Returns (ordered_centreline (N,2), mask).
    Uses cable_segment.segment_cable, which keeps ONE continuous cable piece.
    """
    from matplotlib.widgets import RectangleSelector

    st = {"pos": [], "neg": [], "box": None, "mask": None, "cable": None,
          "boxmode": False, "accepted": False}
    base = np.asarray(img)

    def redraw():
        ax.clear()
        ax.imshow(base)
        if st["mask"] is not None:
            ov = np.zeros((*st["mask"].shape, 4))
            ov[st["mask"] > 0] = [0.1, 0.9, 0.3, 0.4]
            ax.imshow(ov)
        if st["cable"] is not None:
            ax.plot(st["cable"][:, 0], st["cable"][:, 1], "-",
                    color="deepskyblue", lw=1.5)
        if st["box"] is not None:
            x0, y0, x1, y1 = st["box"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor="yellow", lw=2))
        for (x, y) in st["pos"]:
            ax.plot(x, y, "o", color="lime", ms=9, mec="black")
        for (x, y) in st["neg"]:
            ax.plot(x, y, "X", color="red", ms=10, mec="black")
        tip = "  [BOX MODE: drag]" if st["boxmode"] else ""
        ax.set_title("LEFT=cable RIGHT=not-cable b=box u=undo ENTER=run "
                     "a=accept" + tip)
        ax.set_axis_off()
        fig.canvas.draw_idle()

    def on_box(eclick, erelease):
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        st["box"] = (x0, y0, x1, y1)
        st["boxmode"] = False
        selector.set_active(False)
        log.info("box set (%.0f,%.0f)-(%.0f,%.0f)", x0, y0, x1, y1)
        redraw()

    selector = RectangleSelector(ax, on_box, useblit=True, button=[1],
                                 interactive=False)
    selector.set_active(False)

    def on_click(event):
        if st["boxmode"] or event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:
            st["pos"].append((event.xdata, event.ydata))
        elif event.button == 3:
            st["neg"].append((event.xdata, event.ydata))
        redraw()

    def run():
        if not st["pos"] and st["box"] is None:
            log.warning("click a point ON the cable (or draw a box) first.")
            return
        try:
            st["cable"], st["mask"] = trace_cable_from_hints(
                base, st["pos"], neg_px=st["neg"], box=st["box"])
        except Exception as exc:
            log.error("segmentation failed: %s", exc)
        redraw()

    def on_key(event):
        if event.key in ("enter", "return"):
            run()
        elif event.key == "b":
            st["boxmode"] = not st["boxmode"]
            selector.set_active(st["boxmode"])
            log.info("box mode %s", "ON -- drag a box" if st["boxmode"]
                     else "off")
            redraw()
        elif event.key == "u":
            if st["box"] is not None:
                st["box"] = None
            elif st["neg"]:
                st["neg"].pop()
            elif st["pos"]:
                st["pos"].pop()
            redraw()
        elif event.key == "a":
            if st["cable"] is None:
                log.warning("run SAM first (ENTER) before accepting.")
            else:
                st["accepted"] = True       # stop the wait loop; keep fig open

    cid1 = fig.canvas.mpl_connect("button_press_event", on_click)
    cid2 = fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    log.info("SEGMENT: press 'b' + drag a box, LEFT-click the cable, ENTER to "
             "run, then 'a' to accept.")
    while not st["accepted"] and plt.fignum_exists(fig.number):
        plt.pause(0.05)
    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    if st["cable"] is None:
        raise SystemExit("No cable segmented (window closed). Aborting.")
    return st["cable"], st["mask"]


def cmd_photo(args):
    if _MPL_ERR is not None:
        raise SystemExit(f"matplotlib required for 'photo': {_MPL_ERR}")
    if not os.path.exists(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    log.info("loading image: %s", args.image)
    img = imread(args.image)
    log.info("image size: %d x %d px", img.shape[1], img.shape[0])

    fig, ax = plt.subplots(figsize=(11, 8))

    # phone-editor style 90-deg rotation, done LIVE in the matplotlib window:
    # press 'r' to rotate 90 deg until the cable looks upright, ENTER to start.
    img = rotate_interactive(fig, ax, img, start_deg=args.rotate)
    gray = _to_gray(img)
    snap = not args.no_snap
    log.info("edge snapping: %s", "ON" if snap else "OFF")

    log.info("=== STEP 1/3  SCALE REFERENCE ===")
    scale = _click(
        ax, 2, "click point A then point B (known distance)", "yellow", "o",
        instruction=["STEP 1/3  SCALE REFERENCE  (just for pixel size)",
                     "Click TWO points whose REAL distance you measured --",
                     "like a RULER in the scene. These need NOT be on the",
                     "cable; they only set metres-per-pixel."])
    # this next input happens in the terminal -- tell the user on the window.
    tb = _show_terminal_banner(
        ax, "type the real distance between A and B (in metres)")
    scale_real_m = _ask_scale_distance()
    tb.remove()
    ax.figure.canvas.draw_idle()

    log.info("=== STEP 2/3  SEGMENT THE CABLE (SAM) ===")
    if not args.no_trace:
        cable, mask = _segment_interactive(fig, ax, img)
    else:
        # No SAM: just use clicked points, optionally edge-snapped.
        hints = _click(
            ax, None, "clicking points...", "yellow", "+",
            instruction=["STEP 2/3  CLICK ALONG THE CABLE (--no-trace)",
                         "Click points along the cable; ENTER when done."])
        if len(hints) < 2:
            raise SystemExit("Click at least 2 points on the cable.")
        cable = hints
        mask = None
        if not args.no_snap:
            cable = np.array([snap_to_edge(gray, p[0], p[1]) for p in hints])
            log.info("--no-trace: snapped your %d clicks to the edge.",
                     len(cable))

    # Result view: translucent mask overlay + ONE clean centre-line.
    ax.clear()
    ax.imshow(img)
    if mask is not None:
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0] = [0.1, 0.9, 0.3, 0.35]     # green, translucent
        ax.imshow(overlay)                            # the segmented cable
    ax.plot(cable[:, 0], cable[:, 1], "-", color="deepskyblue", lw=1.5,
            label="cable centre-line")
    ax.legend(loc="upper right")
    ax.set_axis_off()
    _banner(ax, [f"STEP 3/3  SEGMENTED CABLE  ({len(cable)} points)",
                 "Green = SAM mask,  blue = centre-line.",
                 "Close the window to continue to scaling + CSV."])
    plt.draw()
    plt.pause(0.1)

    log.info("converting %d traced points to metres + fitting (%s)...",
             len(cable), args.fit)
    x_m, z_m, mpp = pixels_to_metres(cable, scale[0], scale[1], scale_real_m)
    prof = fit_profile(x_m, z_m, args.fit, args.samples, smooth=args.smooth)
    prof.source = args.image

    csv_path, plot_path = _out_paths(args.image, args.out)
    prof.write_csv(csv_path)
    log.info("DONE -> %s", csv_path)
    _report(prof, scale_real_m, mpp)

    _preview(prof, save_path=plot_path)
    log.info("close the windows to finish.")
    plt.show()


# ===========================================================================
# Video tracking (CSRT, several boxes)
# ===========================================================================
def _load_cv2():
    try:
        import cv2
    except ImportError:
        raise SystemExit(
            "OpenCV (with contrib) is required for 'video'. Install:\n"
            "    pip install opencv-contrib-python")
    return cv2


def cmd_video(args):
    """SAM-segment the cable in frame 1 (you click), track it through the whole
    video, then write a per-frame CSV + a results overlay video and GIF.

    Workflow: on frame 1 you (1) draw a scale box, (2) click on the cable and
    optionally a box around it. Every frame is then SAM-segmented inside a box
    that follows the cable, so no CSRT drift. The results video shows the photo
    with the segmented centre-line drawn on each frame.
    """
    cv2 = _load_cv2()
    if _MPL_ERR is not None:
        raise SystemExit(f"matplotlib required for 'video': {_MPL_ERR}")
    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / max(1, args.stride)
    ok, frame0_bgr = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame.")
    frame0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)

    # --- ROTATE frame 1 (live, like photo). The chosen angle is applied to
    # EVERY frame below, so the whole video -- pixels AND the x/z frame -- is
    # rotated consistently (e.g. make a vertical hang lie horizontal).
    fig, ax = plt.subplots(figsize=(11, 8))
    frame0 = rotate_interactive(fig, ax, frame0, start_deg=args.rotate)
    rot_deg = getattr(rotate_interactive, "last_deg", int(args.rotate))
    H, W = frame0.shape[:2]
    log.info("video rotated %d deg to match frame 1.", rot_deg)

    # STEP 1: scale reference (two clicks + a typed distance), same as photo.
    log.info("=== FRAME 1, STEP 1/2  SCALE REFERENCE ===")
    scale = _click(
        ax, 2, "click point A then point B (known distance)", "yellow", "o",
        instruction=["FRAME 1 - STEP 1/2  SCALE REFERENCE",
                     "Click TWO points whose REAL distance you measured",
                     "(a ruler in the scene). They need NOT be on the cable."])
    tb = _show_terminal_banner(
        ax, "type the real distance between A and B (in metres)")
    scale_real_m = args.scale_m if args.scale_m else _ask_scale_distance()
    tb.remove()
    ax.figure.canvas.draw_idle()
    scale_p0, scale_p1 = scale[0], scale[1]

    # STEP 2: segment the cable in frame 1 (box + clicks), same as photo.
    log.info("=== FRAME 1, STEP 2/2  SEGMENT THE CABLE ===")
    _, mask0 = _segment_interactive(fig, ax, frame0)
    plt.close(fig)

    # box that FOLLOWS the cable: start from frame-1 mask bbox, padded.
    def mask_bbox(m, pad=60):
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return None
        return (max(0, xs.min() - pad), max(0, ys.min() - pad),
                min(W, xs.max() + pad), min(H, ys.max() + pad))

    box = mask_bbox(mask0)
    series = TimeSeriesProfile()

    # results video writer (named after the input video).
    csv_path, plot_path = _out_paths(args.video, args.out)
    vid_path = os.path.splitext(args.video)[0] + "_result.mp4"
    gif_frames = []
    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, fps), (W, H))

    def process(frame_rgb, idx, mask=None):
        nonlocal box
        if mask is None:
            try:
                mask = segment_cable(frame_rgb, [], box=box)
            except Exception as exc:
                log.warning("frame %d: segmentation failed (%s)", idx, exc)
                return False
        # centre-line + metric profile
        cl = centerline_from_mask(mask, log_fn=lambda *a, **k: None)
        if len(cl) < 5:
            return False
        x_m, z_m, _ = pixels_to_metres(cl, scale_p0, scale_p1, scale_real_m)
        series.times.append(idx / (fps * max(1, args.stride)))
        series.frames.append(fit_profile(x_m, z_m, args.fit, args.samples,
                                          smooth=args.smooth))
        # draw the overlay frame for the results video.
        vis = frame_rgb.copy()
        vis[mask > 0] = (0.5 * vis[mask > 0] +
                         0.5 * np.array([40, 230, 80])).astype(np.uint8)
        pts = cl.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, (0, 120, 255), 3)
        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        gif_frames.append(vis[::4, ::4])           # downsampled for the gif
        box = mask_bbox(mask) or box               # follow the cable
        return True

    log.info("=== TRACKING THROUGH THE VIDEO ===")
    process(frame0, 0, mask=mask0)                 # frame 1 (reuse its mask)
    idx, total = 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % max(1, args.stride) == 0:
            rgb_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            rgb_fr = rotate_image(rgb_fr, rot_deg)   # same rotation as frame 1
            process(rgb_fr, idx)
            _progress(idx, total or idx, "frames")
        idx += 1
    cap.release()
    writer.release()

    if not series.frames:
        raise SystemExit("No frames segmented.")

    # --- write outputs (all named after the video) -------------------------
    series.write_csv(csv_path)
    log.info("CSV (profile per frame) -> %s", csv_path)
    log.info("results video           -> %s", vid_path)
    _save_gif(gif_frames, os.path.splitext(args.video)[0] + "_result.gif", fps)

    last = series.frames[-1]
    last.source = args.video
    last.write_csv(os.path.splitext(args.video)[0] + "_last_profile.csv")
    _report(last, scale_real_m, None)
    _preview(last, extra=series, save_path=plot_path)
    log.info("tracked %d frames over %.2f s.", len(series.frames),
             series.times[-1] if series.times else 0.0)
    plt.show()


def _save_gif(frames, path, fps):
    """Save the overlay frames as an animated GIF (best-effort)."""
    if not frames:
        return
    try:
        import imageio
        imageio.mimsave(path, frames, fps=max(1, int(fps)))
        log.info("results gif             -> %s", path)
    except Exception as exc:
        log.info("(gif skipped: %s -- pip install imageio for a GIF)", exc)


# ===========================================================================
# Output naming  -- results are named after the input image/video.
# ===========================================================================
def _out_paths(input_path, out_override=None):
    """Derive result paths (CSV + plot).

    If --out is given, both the CSV and the plot go THERE (the plot next to the
    CSV), so you can direct all results into a results/ folder. Otherwise they
    land next to the input image.
    e.g. --out results/IMG_0501/profile.csv -> .../profile.png alongside it.
    """
    if out_override:
        csv_path = out_override
        plot_path = os.path.splitext(out_override)[0] + ".png"
    else:
        base = os.path.splitext(input_path)[0]
        csv_path = base + "_profile.csv"
        plot_path = base + "_profile.png"
    # make sure the destination folder exists.
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    return csv_path, plot_path


# ===========================================================================
# Reporting / preview
# ===========================================================================
def _report(prof, scale_real_m, mpp):
    print("\n----------------------------------------------------------")
    print(f"  fit             : {prof.fit}")
    print(f"  scale reference : {scale_real_m:.4f} m"
          + (f"  ({mpp*1000:.3f} mm/px)" if mpp else ""))
    print(f"  points          : {len(prof.x_m)}")
    print(f"  x range         : {prof.x_m.min():.3f} .. {prof.x_m.max():.3f} m")
    print(f"  z range         : {prof.z_m.min():.3f} .. {prof.z_m.max():.3f} m")
    print(f"  arc length      : {prof.arc_length():.3f} m")
    print("----------------------------------------------------------")


def _preview(prof, extra: "TimeSeriesProfile | None" = None, save_path=None):
    if _MPL_ERR is not None:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    if extra is not None:
        for fr in extra.frames[::max(1, len(extra.frames) // 8)]:
            ax.plot(fr.x_m, fr.z_m, "-", color="grey", alpha=0.3, lw=1)
    ax.plot(prof.x_m, prof.z_m, "-", color="cyan", lw=2, label="profile")
    ax.scatter([0], [0], color="lime", zorder=5, label="origin")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("z  height [m]")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Extracted cable  ({prof.fit}, arc {prof.arc_length():.3f} m)")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("saved profile plot -> %s", save_path)
    return fig


# ===========================================================================
# Sim trajectory parsing + comparison
# ===========================================================================
def load_sim_profile(path, frame="last"):
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if len(rows) < 2:
        raise SystemExit(f"{path}: need a header + at least one data row.")
    header, data = rows[0], rows[1:]
    x_cols, z_cols, order = {}, {}, []
    for i, col in enumerate(header):
        c = col.strip().lower()
        if c.endswith("_x"):
            node = c[:-2]
            x_cols[node] = i
            if node not in order:
                order.append(node)
        elif c.endswith("_z"):
            z_cols[c[:-2]] = i
    nodes = [n for n in order if n in x_cols and n in z_cols]
    if not nodes:
        raise SystemExit(f"{path}: no *_x/*_z node columns. Header: {header[:8]}")
    if frame == "last":
        row = data[-1]
    elif frame == "first":
        row = data[0]
    else:
        row = data[int(frame)]
    xs = np.array([float(row[x_cols[n]]) for n in nodes])
    zs = np.array([float(row[z_cols[n]]) for n in nodes])
    return CableProfile(xs, zs, source=f"{path} [frame={frame}]")


def _resample_common_x(real, sim, n=200):
    def mono(c):
        o = np.argsort(c.x_m)
        return c.x_m[o], c.z_m[o]
    rx, rz = mono(real)
    sx, sz = mono(sim)
    lo, hi = max(rx.min(), sx.min()), min(rx.max(), sx.max())
    if hi <= lo:
        raise SystemExit(
            "Real and sim curves do not overlap in x -- check both use the "
            "anchor as origin and the same pull direction.")
    g = np.linspace(lo, hi, n)
    return g, np.interp(g, rx, rz), np.interp(g, sx, sz)


def cmd_compare(args):
    real = CableProfile.read_csv(args.real)
    sim = load_sim_profile(args.sim, frame=args.sim_frame)
    g, rz, sz = _resample_common_x(real, sim)
    err = rz - sz
    rms, mx = float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err)))

    print("\n=== REAL vs SIM ===")
    print(f"  real : {real.source}")
    print(f"  sim  : {sim.source}")
    print(f"  real arc length : {real.arc_length():.3f} m")
    print(f"  sim  arc length : {sim.arc_length():.3f} m")
    print(f"  overlap x       : {g.min():.3f} .. {g.max():.3f} m")
    print(f"  RMS height error: {rms*1000:.1f} mm")
    print(f"  max height error: {mx*1000:.1f} mm")

    if args.plot:
        if _MPL_ERR is not None:
            print(f"(no plot: matplotlib unavailable: {_MPL_ERR})")
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(real.x_m, real.z_m, "-o", ms=3, color="crimson", label="real")
        ax.plot(sim.x_m, sim.z_m, "-s", ms=3, color="royalblue", label="sim")
        ax.fill_between(g, rz, sz, color="grey", alpha=0.25,
                        label=f"error (RMS {rms*1000:.0f} mm)")
        ax.set_xlabel("x  [m]")
        ax.set_ylabel("z  height [m]")
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Real cable vs Isaac-Sim cable")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"  overlay plot    : {args.plot}")


# ===========================================================================
# CLI
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Extract a real cable's 2-D height profile from a photo or "
                    "video and compare it to an Isaac-Sim trajectory.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--help-photo", action="store_true",
                   help="print the shooting checklist and exit")
    sub = p.add_subparsers(dest="cmd")

    def add_fit(sp):
        sp.add_argument("--fit", choices=["auto", "catenary", "spline"],
                        default="auto",
                        help="curve reconstruction (default auto: catenary, "
                             "fall back to spline if residual is poor)")
        sp.add_argument("--samples", type=int, default=500,
                        help="points sampled on the fitted curve (denser=smoother)")
        sp.add_argument("--smooth", type=float, default=1.0,
                        help="spline smoothing strength (higher=smoother, "
                             "lower=hugs the points)")

    pp = sub.add_parser(
        "photo",
        help="SAM segments the cable (box + clicks) -> metric profile")
    pp.add_argument("--image", required=True)
    pp.add_argument("--out", default=None,
                    help="CSV path (default: <image>_profile.csv next to image)")
    pp.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="STARTING rotation (clockwise deg); you can still "
                         "fine-tune live with 'r'/'e' in the window")
    pp.add_argument("--no-trace", action="store_true",
                    help="skip image-processing trace; use your clicks directly")
    pp.add_argument("--no-snap", action="store_true",
                    help="(only with --no-trace) don't snap clicks to the edge")
    add_fit(pp)

    pv = sub.add_parser(
        "video",
        help="click the cable in frame 1; SAM tracks it -> CSV + results video")
    pv.add_argument("--video", required=True)
    pv.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="STARTING rotation for frame 1 (you can fine-tune live "
                         "with 'r'/'e'); applied to the WHOLE video")
    pv.add_argument("--out", default=None,
                    help="CSV path (default: <video>_profile.csv next to video)")
    pv.add_argument("--scale-m", type=float, default=None,
                    help="real width of the scale box in m (skip the prompt)")
    pv.add_argument("--stride", type=int, default=2,
                    help="process every Nth frame (higher=faster, default 2)")
    add_fit(pv)

    pc = sub.add_parser("compare", help="overlay real vs sim, report error")
    pc.add_argument("--real", required=True)
    pc.add_argument("--sim", required=True)
    pc.add_argument("--sim-frame", default="last")
    pc.add_argument("--plot", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.help_photo:
        print(PHOTO_CHECKLIST)
        return
    if args.cmd == "photo":
        cmd_photo(args)
    elif args.cmd == "video":
        cmd_video(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
