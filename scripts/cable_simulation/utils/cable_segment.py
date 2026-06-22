"""
cable_segment.py  --  standalone cable segmentation with SAM.
=============================================================

ONE job, done well: load an image, you click on the cable, SAM returns ONLY
the cable mask (rejecting the rest of the environment). No scaling, no length,
no CSV, no env variables, no command-line args to remember.

    python cable_segment.py                 # uses the default image below
    python cable_segment.py path/to/img.jpg # or pass an image

INTERACTION (everything happens in the image window):
    LEFT  click  -> a point ON the cable        (positive: "this IS cable")
    RIGHT click  -> a point on the WRONG thing  (negative: "NOT cable")
    'u'          -> undo the last click
    ENTER        -> run / re-run SAM with the current clicks
    's'          -> save the mask + overlay PNG next to the image, then done
    'q' / close  -> quit

Use negatives to fix SAM grabbing the environment: left-click the cable, run,
then RIGHT-click whatever extra stuff it grabbed and run again. Repeat until
only the cable is green.

IMPORTING (it is small + functional, so other code can reuse it):
    from cable_segment import segment_cable, load_sam
    mask = segment_cable(rgb_uint8, pos_xy=[(x,y)], neg_xy=[(x,y)])

MODEL: the best SAM model is auto-downloaded ONCE into ./models/ next to this
file, so running from any folder reuses it (never re-downloads). CUDA is used
if present, else Apple-Silicon MPS, else CPU -- chosen automatically.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cable_segment")

# --- default image (so you can just run the file with no args) -------------
DEFAULT_IMAGE = "media/IMG_0501.JPG"

# --- model lives NEXT TO THIS SCRIPT, so it works from any working dir ------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
# best general SAM model in ultralytics. (sam2.1_l = largest / highest quality)
MODEL_NAME = "sam2.1_l.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


# ===========================================================================
# Model loading  (auto device, fixed location, no env vars)
# ===========================================================================
_MODEL = {"obj": None}
_WEIGHTS_URL = ("https://github.com/ultralytics/assets/releases/download/"
                "v8.4.0/" + MODEL_NAME)


def _download_weights(dest):
    """Download the SAM weights to `dest`, handling macOS SSL cert issues.

    Stock macOS Python often lacks CA certs (CERTIFICATE_VERIFY_FAILED). We use
    certifi's bundle if present; otherwise fall back to the system 'curl',
    which uses the OS keychain. Downloads to a .part file then renames, so an
    interrupted download never leaves a corrupt model in place.
    """
    import shutil
    import urllib.request

    tmp = dest + ".part"
    log.info("downloading %s once -> %s", MODEL_NAME, dest)
    log.info("  (this is a large file; it happens only the first time)")

    ctx = None
    try:
        import ssl
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = None

    try:
        if ctx is not None:
            with urllib.request.urlopen(_WEIGHTS_URL, context=ctx) as r, \
                    open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        else:
            raise RuntimeError("no certifi; using curl")
    except Exception as exc:
        log.info("  urllib download failed (%s); trying curl...", exc)
        rc = os.system(f'curl -L --fail -o "{tmp}" "{_WEIGHTS_URL}"')
        if rc != 0 or not os.path.exists(tmp):
            raise SystemExit(
                "Could not download the SAM weights. Either:\n"
                "  pip install certifi   (fixes macOS SSL), or\n"
                f"  download manually:\n    {_WEIGHTS_URL}\n"
                f"  and place it at:\n    {dest}")
    os.replace(tmp, dest)
    log.info("  download complete (%.0f MB)",
             os.path.getsize(dest) / 1e6)


def _pick_device() -> str:
    """cuda if available, else Apple MPS, else cpu -- decided automatically."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam():
    """Load the SAM model once (cached). Stored in ./models next to this file.

    ultralytics downloads weights into the CURRENT WORKING DIR by default, which
    means re-downloads when you run from elsewhere. We force the download/load
    to MODEL_PATH so it is downloaded exactly once and reused everywhere.
    """
    if _MODEL["obj"] is not None:
        return _MODEL["obj"]
    try:
        from ultralytics import SAM
    except ImportError:
        raise SystemExit("Install ultralytics:  pip install ultralytics")

    os.makedirs(MODEL_DIR, exist_ok=True)
    # If the weights are not there yet, fetch them ONCE into MODEL_DIR. We do
    # the download ourselves (with proper SSL certs) because ultralytics'
    # downloader hits CERTIFICATE_VERIFY_FAILED on stock macOS Python.
    if not os.path.exists(MODEL_PATH):
        _download_weights(MODEL_PATH)

    device = _pick_device()
    log.info("loading SAM '%s' on %s", MODEL_NAME, device.upper())
    model = SAM(MODEL_PATH)
    model.to(device)
    _MODEL["obj"] = model
    return model


# ===========================================================================
# Segmentation core
# ===========================================================================
def _cable_score(mask01, pos_xy):
    """How CABLE-LIKE a mask is: long & thin and covering the positive clicks.

    A cable is elongated, so bbox diagonal is large relative to area; a fat
    environment blob scores low. Masks missing all positive clicks score -inf.
    """
    H, W = mask01.shape
    m = (mask01 > 0.5)
    area = int(m.sum())
    if area < 20:
        return -np.inf
    hits = sum(int(m[min(H - 1, int(round(y))), min(W - 1, int(round(x)))])
               for (x, y) in pos_xy)
    if pos_xy and hits == 0:
        return -np.inf
    ys, xs = np.where(m)
    diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    elong = diag / np.sqrt(area + 1e-6)          # thin+long -> high
    fill_pen = 1.0 if area < 0.5 * H * W else 0.3  # punish giant fills
    return hits * 10.0 + elong * 5.0 * fill_pen


def segment_cable(rgb, pos_xy, neg_xy=None, box=None):
    """Return a binary cable mask (uint8 0/255) for the given image + prompts.

    rgb     : HxWx3 uint8 image (RGB).
    pos_xy  : list of (x, y) points ON the cable    (positive prompts).
    neg_xy  : list of (x, y) points NOT the cable   (negative prompts), or None.
    box     : optional (x0, y0, x1, y1) bounding box. SAM only segments INSIDE
              it -- the strongest way to exclude the surrounding environment.
              Far more effective than piling on negative clicks.

    NOTE: too many NEGATIVE points over-constrain SAM and make it return an
    empty mask. Prefer a BOX to exclude the environment; keep negatives few.
    """
    model = load_sam()
    pos_xy = [(float(x), float(y)) for (x, y) in pos_xy]
    neg_xy = [(float(x), float(y)) for (x, y) in (neg_xy or [])]
    if not pos_xy and box is None:
        raise ValueError("Need at least one positive (cable) click or a box.")

    if len(neg_xy) > 3:
        log.warning("%d negative points -- SAM2 often returns NOTHING with "
                    "many negatives. Use a BOX (key 'b') instead; keeping the "
                    "last 3.", len(neg_xy))
        neg_xy = neg_xy[-3:]

    kwargs = {"verbose": False}
    if box is not None:
        kwargs["bboxes"] = [[float(v) for v in box]]
    if pos_xy or neg_xy:
        kwargs["points"] = [pos_xy + neg_xy]
        kwargs["labels"] = [[1] * len(pos_xy) + [0] * len(neg_xy)]

    log.info("SAM: %d positive, %d negative, box=%s",
             len(pos_xy), len(neg_xy), box is not None)
    res = model.predict(rgb, **kwargs)
    r = res[0]
    if r.masks is None or len(r.masks.data) == 0:
        raise RuntimeError(
            "SAM returned no mask. Likely too many negative clicks -- press "
            "'u' to undo some, or draw a BOX (key 'b') around the cable.")

    masks = r.masks.data.cpu().numpy()               # (n, H, W)
    # If several masks come back, keep the most cable-like (long & thin).
    ref = pos_xy if pos_xy else [( (box[0] + box[2]) / 2,
                                   (box[1] + box[3]) / 2 )]
    scores = [_cable_score(m, ref) for m in masks]
    best = int(np.argmax(scores))
    if not np.isfinite(scores[best]):
        best = int(np.argmax([m.sum() for m in masks]))
    mask = (masks[best] > 0.5).astype(np.uint8) * 255
    log.info("  mask: candidate %d/%d, %d px",
             best + 1, len(masks), int((mask > 0).sum()))

    # The cable is ONE continuous object -> keep only the connected piece that
    # contains your clicks; drop far-away separate blobs of the same colour.
    seed = pos_xy if pos_xy else list(ref)
    mask = _keep_connected(mask, seed)
    return mask


def _keep_connected(mask, seed_xy, bridge_px=7):
    """Keep only the single connected cable piece that contains the seed clicks.

    1. 'Close' the mask by ~bridge_px so a few-pixel break in YOUR cable does
       not split it (but genuinely far-apart objects stay separate).
    2. Label connected components; keep the one(s) the seed points fall in.
    3. If the seeds land on no component (e.g. just off the cable), keep the
       largest component instead.
    The returned mask is intersected back with the ORIGINAL so bridging does
    not fatten the cable -- it only decides connectivity.
    """
    import cv2

    H, W = mask.shape
    orig = (mask > 0).astype(np.uint8)
    if orig.sum() == 0:
        return mask

    # gentle close to bridge tiny gaps before measuring connectivity.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * bridge_px + 1, 2 * bridge_px + 1))
    closed = cv2.morphologyEx(orig, cv2.MORPH_CLOSE, k)

    n, labels = cv2.connectedComponents(closed)
    keep = set()
    for (x, y) in seed_xy:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < H and 0 <= xi < W:
            lbl = labels[yi, xi]
            if lbl != 0:
                keep.add(int(lbl))
    if not keep:
        sizes = [(int((labels == i).sum()), i) for i in range(1, n)]
        if sizes:
            keep.add(max(sizes)[1])

    comp = np.isin(labels, list(keep)).astype(np.uint8)
    # Return the CLOSED component (so the cable is ONE continuous piece with the
    # tiny gaps filled), but only where it stays near the original cable -- so
    # the bridging fills hairline breaks without fattening the whole cable. We
    # allow the closed pixels within `bridge_px` of the original mask.
    near = cv2.dilate(orig, k)
    out = (comp & near) * 255
    dropped = int(orig.sum()) - int(((out > 0) & (orig > 0)).sum())
    n_final, _ = cv2.connectedComponents((out > 0).astype(np.uint8))
    log.info("  connectivity: 1 cable piece (%d sub-parts bridged), "
             "dropped %d disconnected px", max(n_final - 1, 1), dropped)
    return out.astype(np.uint8)


# ===========================================================================
# Interactive window
# ===========================================================================
def run_interactive(image_path):
    import matplotlib
    matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")
    img = imread(image_path)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = img[..., :3]
    log.info("image: %s  (%dx%d)", image_path, rgb.shape[1], rgb.shape[0])

    state = {"pos": [], "neg": [], "box": None, "mask": None,
             "boxmode": False, "done": False}

    fig, ax = plt.subplots(figsize=(11, 8))

    def redraw():
        ax.clear()
        ax.imshow(rgb)
        if state["mask"] is not None:
            ov = np.zeros((*state["mask"].shape, 4))
            ov[state["mask"] > 0] = [0.1, 0.9, 0.3, 0.4]      # green cable
            ax.imshow(ov)
        if state["box"] is not None:
            x0, y0, x1, y1 = state["box"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor="yellow", lw=2))
        for (x, y) in state["pos"]:
            ax.plot(x, y, "o", color="lime", ms=9, mec="black")
        for (x, y) in state["neg"]:
            ax.plot(x, y, "X", color="red", ms=10, mec="black")
        mode = "  [BOX MODE: drag a box]" if state["boxmode"] else ""
        ax.set_title("LEFT=cable RIGHT=not-cable b=box u=undo "
                     "ENTER=run s=save q=quit" + mode)
        ax.set_axis_off()
        fig.canvas.draw_idle()

    def on_box(eclick, erelease):
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        state["box"] = (x0, y0, x1, y1)
        state["boxmode"] = False
        selector.set_active(False)
        log.info("  box set: (%.0f, %.0f)-(%.0f, %.0f)", x0, y0, x1, y1)
        redraw()

    from matplotlib.widgets import RectangleSelector
    selector = RectangleSelector(ax, on_box, useblit=True, button=[1],
                                 interactive=False)
    selector.set_active(False)

    def on_click(event):
        if state["boxmode"] or event.inaxes != ax or event.xdata is None:
            return                                  # box mode handled by selector
        if event.button == 1:                       # left = positive
            state["pos"].append((event.xdata, event.ydata))
            log.info("  + cable point (%.0f, %.0f)", event.xdata, event.ydata)
        elif event.button == 3:                     # right = negative
            state["neg"].append((event.xdata, event.ydata))
            log.info("  - not-cable point (%.0f, %.0f)",
                     event.xdata, event.ydata)
        redraw()

    def run_sam():
        if not state["pos"] and state["box"] is None:
            log.warning("click a point ON the cable (or draw a box) first.")
            return
        try:
            state["mask"] = segment_cable(rgb, state["pos"], state["neg"],
                                          box=state["box"])
        except Exception as exc:
            log.error("segmentation failed: %s", exc)
        redraw()

    def save():
        if state["mask"] is None:
            log.warning("nothing to save yet -- press ENTER to run SAM first.")
            return
        import matplotlib.pyplot as _plt
        base = os.path.splitext(image_path)[0]
        mask_path = base + "_cable_mask.png"
        over_path = base + "_cable_overlay.png"
        _plt.imsave(mask_path, state["mask"], cmap="gray")
        fig.savefig(over_path, dpi=150, bbox_inches="tight")
        log.info("saved: %s", mask_path)
        log.info("saved: %s", over_path)
        state["done"] = True

    def on_key(event):
        if event.key in ("enter", "return"):
            run_sam()
        elif event.key == "b":                       # toggle box-draw mode
            state["boxmode"] = not state["boxmode"]
            selector.set_active(state["boxmode"])
            if state["boxmode"]:
                log.info("  BOX MODE ON -- drag a box around the cable")
            else:
                log.info("  box mode off")
            redraw()
        elif event.key == "u":
            if state["box"] is not None:
                state["box"] = None
            elif state["neg"]:
                state["neg"].pop()
            elif state["pos"]:
                state["pos"].pop()
            log.info("  undo")
            redraw()
        elif event.key == "s":
            save()
            if state["done"]:
                plt.close(fig)
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    log.info("READY. Best workflow for a thin cable in a busy scene:")
    log.info("  1) press 'b', drag a tight BOX around the cable")
    log.info("  2) LEFT-click once or twice ON the cable")
    log.info("  3) press ENTER to segment, 's' to save")
    log.info("Avoid many right-clicks: SAM2 returns nothing with >3 negatives.")
    plt.show()
    return state["mask"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    image_path = argv[0] if argv else os.path.join(HERE, DEFAULT_IMAGE)
    run_interactive(image_path)


if __name__ == "__main__":
    main()
