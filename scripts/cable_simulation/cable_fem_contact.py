"""
CYLINDER CONTACT demo  --  the firm cable draping on / moved by the green bar.
=============================================================================

This is the "watch the cable interact with the cylinder" version. It is a thin
launcher that runs cable_fem.py with FAITHFUL mode OFF:

  * a FIRM cable (E_SCALE_EXP = 2.5) so it rests on the bar without squashing
    through it (the faithful exact-EI cable is too soft and penetrates -- that is
    why contact needs its own, firmer script)
  * the green CYLINDER obstacle is ON
  * the bar MOVES by code (it lowers, the cable rides it down and drops off); in
    the GUI you can Shift + Left-click-drag it to move it by hand

NOTE: this is a behaviour/contact DEMO, not a faithful comparison base -- the firm
cable is ~32x too stiff in bending. For a fair comparison base use cable_fem_base.py.

Run:
    python scripts/cable_simulation/cable_fem_contact.py

Useful knobs (all the cable_fem.py CABLE_* vars work):
    CABLE_OB_MOVE_SPEED=0     stop the auto-motion and mouse-drag the bar instead
    CABLE_OB_MOVE_SPEED=-0.05 lower the bar more slowly
    CABLE_OB_MOVE_AXIS=x|y|z  direction the bar travels
"""
import os
import runpy
from pathlib import Path

# Lock this launcher to the firm contact-demo mode, then run the shared engine.
os.environ["CABLE_FAITHFUL"] = "0"
# Separate output folder so it doesn't clobber the faithful base's results.
os.environ.setdefault("CABLE_OUTPUT_DIR",
                      str(Path(__file__).parent / "cable_output" / "fem_contact"))
runpy.run_path(str(Path(__file__).parent / "cable_fem.py"), run_name="__main__")
