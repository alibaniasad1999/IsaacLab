"""
CYLINDER demo with a FLOPPY (cable-like) cable.
===============================================

Same soft, config-faithful cable as cable_fem_base.py (it bends by exactly the
real TPU stiffness -- floppy, NOT the rigid stick the old contact demo had), but
now draped OVER a green CYLINDER.

How it stays floppy AND doesn't tunnel through the thin bar: the bar is given a
collision CONTACT OFFSET (a cushion), so PhysX catches the soft cable before it
can sink through. Verified: soft cable (E~10 kPa) rests on the bar, no penetration,
drapes its overhang down to ~0.9 m, stable.

Scene: a CANTILEVER (left end fixed, no end weight) laid over the bar -- the bar
holds the middle so the overhanging half droops down like a real cable without a
hanging weight stretching it to the floor. The bar is STATIONARY and you can
Shift + Left-click-drag it in the GUI to watch the cable react.

This is a behaviour/contact DEMO. The cable is still the exact-EI FEM cable, so the
summary reports some axial STRETCH (the inherent FEM limit; see
why_fem_cant_be_thin.py). For a fair comparison BASE use cable_fem_base.py.

Run:
    python scripts/cable_simulation/cable_fem_contact.py

Useful knobs (all the cable_fem.py CABLE_* vars work):
    CABLE_OB_MOVE_SPEED=-0.05  make the bar AUTO-LOWER (cable rides it off)
    CABLE_OB_X=0.65            move the bar along the cable
    CABLE_OB_CONTACT_OFFSET=0.03  bigger cushion if a soft cable still pokes in
    CABLE_E_EXP=3.5            a touch firmer cable (less droop) if you want
"""
import os
import runpy
from pathlib import Path

# Use the floppy, config-faithful cable (exact EI, damped jelly) -- the same body
# as cable_fem_base.py -- then add the cylinder back on top of it.
os.environ["CABLE_FAITHFUL"] = "1"
os.environ.setdefault("CABLE_OBSTACLE", "1")             # bring the cylinder back
os.environ.setdefault("CABLE_OB_MOVE_SPEED", "0")        # stationary + mouse-draggable
os.environ.setdefault("CABLE_OB_CONTACT_OFFSET", "0.02")  # cushion: soft cable won't tunnel
os.environ.setdefault("CABLE_POS_ITERS", "120")          # enough iters for clean soft contact
# Separate output folder so it doesn't clobber the faithful base's results.
os.environ.setdefault("CABLE_OUTPUT_DIR",
                      str(Path(__file__).parent / "cable_output" / "fem_contact"))
runpy.run_path(str(Path(__file__).parent / "cable_fem.py"), run_name="__main__")
