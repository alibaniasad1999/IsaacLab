"""
FAITHFUL comparison BASE  --  the cable that behaves like cable_config.py.
=========================================================================

This is the "behave like the config, not jelly" cable, with NO cylinder. It is a
thin launcher that runs cable_fem.py in its FAITHFUL mode:

  * E_SCALE_EXP = 4  -> EI_sim == EI_real EXACTLY (bends like the real TPU cable)
  * mass == real, and the volumetric "jelly" wobble is damped out
  * a clean CANTILEVER scene (no obstacle) -- a reproducible gravity-bending test

At startup it prints a FIDELITY line (EI/mass ratios ~1.00) and the summary logs
faithful_mode / EI_ratio / mass_ratio / stretch_pct, so you can confirm the cable
matches the config. Use THIS script as the apples-to-apples base when you compare
the three cable models (cable.py, this, cable_warp.py).

HONEST CAVEAT: the summary will report ~50% STRETCH. That is the inherent FEM
limit -- one isotropic E can't be both bending-soft and axially-stiff (see
why_fem_cant_be_thin.py, section 3b). For a TRULY faithful cable use cable_warp.py.

Run:
    python scripts/cable_simulation/cable_fem_base.py

All the usual cable_fem.py CABLE_* env knobs still work.
For the cylinder/contact version instead, run cable_fem_contact.py.
"""
import os
import runpy
from pathlib import Path

# Lock this launcher to the faithful base mode, then run the shared engine.
os.environ["CABLE_FAITHFUL"] = "1"
# Separate output folder so it doesn't clobber the contact demo's results.
os.environ.setdefault("CABLE_OUTPUT_DIR",
                      str(Path(__file__).parent / "cable_output" / "fem_base"))
runpy.run_path(str(Path(__file__).parent / "cable_fem.py"), run_name="__main__")
