"""
Shared physical-cable parameters for ALL cable simulations in this folder.

The three scripts here model the SAME physical cable with three different
numerical methods:

  * cable.py      -- rigid capsule-chain + D6 joints (beam-theory bending)
  * cable_fem.py  -- volumetric FEM deformable (PhysX soft body)
  * cable_warp.py -- 1-D Cosserat/XPBD elastic rod (NVIDIA Warp)

Each method re-discretises and re-scales these numbers in its own way (the FEM
script fattens the rod and rescales E/density; the warp script uses a thin
visual radius; the capsule chain derives joint stiffness from beam theory) --
but they must all START from the same physical target so the cable behaves
consistently across methods. That single source of truth lives HERE.

DO NOT hardcode these values in the individual scripts. Import them:

    from cable_config import (TOTAL_CABLE_LENGTH, REAL_RADIUS, YOUNG_MODULUS,
                              POISSON_RATIO, DENSITY, CABLE_MASS)

Every value is still overridable per-run via the same CABLE_* environment
variables the scripts used before, so existing run commands keep working.

Physical target: a flexible TPU / polyurethane robot-cable jacket (Shore ~85-95A)
  E ~ 40 MPa, nu ~ 0.48 (near-incompressible elastomer), rho ~ 1150 kg/m^3,
  radius 1.5 mm, length 1 m  ->  it physically weighs ~8 g.
"""

import os
import math

# ---------------------------------------------------------------
# Geometry (the physical cable)
# ---------------------------------------------------------------
# Total cable length [m].
TOTAL_CABLE_LENGTH = float(os.environ.get("CABLE_LENGTH", 1.0))

# Real (physical) cable radius [m]. 1.5 mm. In cable.py this is the actual
# simulated capsule radius; in cable_fem.py it is the scaling REFERENCE the fat
# sim rod is rescaled toward (the FEM body itself uses its own SIM_RADIUS);
# in cable_warp.py the visual radius is independent (ROD_RADIUS) but this is the
# physical value the rod represents.
REAL_RADIUS = float(os.environ.get("CABLE_RADIUS", 1.5e-3))

# ---------------------------------------------------------------
# Material (flexible TPU)
# ---------------------------------------------------------------
YOUNG_MODULUS = float(os.environ.get("CABLE_E",       40e6))     # Pa  (TPU ~40 MPa)
POISSON_RATIO = float(os.environ.get("CABLE_NU",      0.48))     # near-incompressible
DENSITY       = float(os.environ.get("CABLE_DENSITY", 1150.0))   # kg/m^3 (TPU)

# ---------------------------------------------------------------
# Derived: physical cable mass (rho * volume), NOT hardcoded.
# A 1 m x 1.5 mm-radius TPU cable weighs ~8 g.  Override with CABLE_MASS.
# ---------------------------------------------------------------
CABLE_VOLUME = math.pi * REAL_RADIUS ** 2 * TOTAL_CABLE_LENGTH
CABLE_MASS   = float(os.environ.get("CABLE_MASS", DENSITY * CABLE_VOLUME))
