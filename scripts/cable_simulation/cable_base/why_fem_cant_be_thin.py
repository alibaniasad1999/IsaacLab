"""
WHY THE FEM CABLE CANNOT BE BOTH THIN AND NON-JELLY  (a report)
================================================================

This is a standalone explainer (no Isaac Sim needed -- pure Python). Run it:

    python scripts/cable_simulation/why_fem_cant_be_thin.py

It computes, from the SAME physical-cable numbers the simulations use
(cable_config.py), the hard limits of the PhysX volumetric-FEM cable in
cable_fem.py, and shows why a genuinely thin (1.5 mm), non-jelly cable is
impossible with that method -- and which method to use instead.

SHORT VERSION
-------------
cable_fem.py models the cable as a VOLUMETRIC (voxel/tetrahedral) FEM soft body.
PhysX voxelizes the rod into hexes along its length. To bend like a rod (not a
rigid stick) a cross-section needs >= 3 hexes across its DIAMETER. The number of
hexes across is  2*R*res/L, so you need

        res  >=  3 * L / (2 * R).

For the real cable (R = 1.5 mm, L = 1 m) that is res ~ 1000. But PhysX's
PxTetMaker refuses to cook above res ~130 on this GPU (MEASURED: res 130 cooks,
res 150 -> "createVoxelTetrahedronMesh failed"). So the thinnest rod that still
voxelizes is

        R_min  =  3 * L / (2 * res_max)  ~  12 mm   (24 mm diameter).

That is ~8x THICKER than the real cable. To fake the real bending/weight on that
fat rod you rescale Young's modulus and density DOWN by (R_real/R_sim)^~2.5 and
^2 -- which makes a soft, low-resolution block: that softness + the near-
incompressible Poisson ratio + only ~3 elements across the section is exactly
what looks like "jelly". Stiffen it and the bending rigidity EI blows past the
real cable (it stops behaving like a cable). You cannot have all three of
{THIN, correct EI, NON-JELLY} in voxel FEM -- pick two.

THE FIX: use a 1-D rod model, which never voxelizes, so thinness is free:
  * cable_warp.py -- Cosserat / XPBD elastic rod (thin mm radius, exact EI, no
                     volumetric jelly).  <-- this is the thin, non-jelly cable.
  * cable.py      -- capsule chain + D6 joints (thin capsules, beam bending).
"""

import math

# Pull the physical target from the shared config (same numbers the sims use).
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from cable_config import (TOTAL_CABLE_LENGTH as L, REAL_RADIUS as R_REAL,
                              YOUNG_MODULUS as E_REAL, POISSON_RATIO as NU,
                              DENSITY as RHO)
except Exception:                       # fallback if run outside the repo
    L, R_REAL, E_REAL, NU, RHO = 1.0, 1.5e-3, 40e6, 0.48, 1150.0

# Empirically-MEASURED PhysX cooking limit on this machine (PxTetMaker):
RES_COOK_OK   = 130       # res 130 -> cooks ; res 150 -> fails
RES_COOK_FAIL = 150       # res 150 -> createVoxelTetrahedronMesh failed
R_SIM_FLOOR   = 12e-3     # the SHIPPED cable_fem.py default = thinnest that cooks
HEX_ACROSS    = 3         # min voxels across a diameter to bend (not be a rigid stick)
E_EXP         = 2.5       # the (R_real/R_sim) exponent cable_fem.py uses for E


def res_needed(radius):           # resolution to get HEX_ACROSS hexes across 2*radius
    return HEX_ACROSS * L / (2.0 * radius)


def thinnest_radius(res_max):     # thinnest rod that still voxelizes at res_max
    return HEX_ACROSS * L / (2.0 * res_max)


def EI(E, r):                     # flexural rigidity of a circular rod
    return E * math.pi * r ** 4 / 4.0


def hr(title=""):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def main():
    R_FLOOR = R_SIM_FLOOR                          # 12 mm (shipped default)
    res_at_floor = res_needed(R_FLOOR)             # ~125 (<= cooking cap 130)
    res_for_real = res_needed(R_REAL)              # ~1000

    hr("WHY THE FEM CABLE CAN'T BE THIN AND NON-JELLY")
    print(f"Physical target (cable_config.py):")
    print(f"  length L           = {L*100:.0f} cm")
    print(f"  real radius R      = {R_REAL*1000:.2f} mm   (diameter {2*R_REAL*1000:.1f} mm)")
    print(f"  Young's modulus E  = {E_REAL/1e6:.0f} MPa")
    print(f"  Poisson ratio nu   = {NU}   (near-incompressible elastomer)")
    print(f"  density rho        = {RHO:.0f} kg/m^3")
    print(f"  real EI            = {EI(E_REAL, R_REAL):.3e} N.m^2")

    hr("1) THINNESS IS CAPPED BY THE VOXELIZER + THE COOKER")
    print("PhysX FEM voxelizes the rod into hexes along its LENGTH. To bend like")
    print(f"a rod (not a rigid stick) a section needs >= {HEX_ACROSS} hexes across its")
    print("diameter. hexes_across = 2*R*res/L, so:  res >= 3*L/(2*R).")
    print()
    print(f"  * to voxelize the REAL {R_REAL*1000:.1f} mm cable you'd need")
    print(f"      res >= 3*L/(2*R) = {res_for_real:.0f}")
    print(f"  * but PxTetMaker COOKING FAILS above res ~{RES_COOK_OK} on this GPU")
    print(f"      (MEASURED: res {RES_COOK_OK} cooks, res {RES_COOK_FAIL} -> "
          f"'createVoxelTetrahedronMesh failed')")
    print()
    print(f"  => required res ({res_for_real:.0f}) is ~{res_for_real/RES_COOK_OK:.0f}x "
          f"OVER the cooking cap.")
    print(f"  => thinnest rod that still cooks (the SHIPPED default):")
    print(f"       R_min = {R_FLOOR*1000:.0f} mm ({2*R_FLOOR*1000:.0f} mm diameter), "
          f"res = 3*L/(2*R) = {res_at_floor:.0f}  (<= cap {RES_COOK_OK}, cooks)")
    print(f"     -> that is ~{R_FLOOR/R_REAL:.0f}x THICKER than the real cable. "
          f"This is the wall.")

    hr("2) FATTENING FORCES SOFTNESS -> 'JELLY'")
    R_SIM = R_FLOOR
    ratio = R_REAL / R_SIM
    E_sim = E_REAL * ratio ** E_EXP
    rho_sim = RHO * ratio ** 2
    print(f"At the {R_SIM*1000:.0f} mm floor the section is fattened by "
          f"(R_sim/R_real)^2 = {(1/ratio)**2:.0f}x in area.")
    print("To keep the REAL bending & weight on that fat rod, cable_fem.py rescales")
    print("the material DOWN:")
    print(f"  E_sim   = E * (R_real/R_sim)^{E_EXP} = {E_REAL/1e6:.0f} MPa * "
          f"{ratio**E_EXP:.4f} = {E_sim/1e3:.0f} kPa")
    print(f"  rho_sim = rho * (R_real/R_sim)^2     = {rho_sim:.1f} kg/m^3")
    print()
    print("A ~%.0f kPa block with only ~%d hexes across, at nu=%.2f (near-"
          % (E_sim/1e3, HEX_ACROSS, NU))
    print("incompressible -> volumetric locking/jiggle), is INHERENTLY wobbly.")
    print("That wobble is the 'jelly'. Three independent causes, all from voxel FEM:")
    print("  (a) only ~3 elements across the section -> poor bending representation")
    print("  (b) low-res co-rotational FEM has spurious soft modes")
    print("  (c) nu near 0.5 -> volumetric locking at low resolution")

    hr("3) YOU CAN'T JUST STIFFEN IT -- EI BLOWS UP")
    for label, r, e in [("sim @ floor (firm)", R_SIM, E_sim),
                        ("sim if E x4 stiffer", R_SIM, E_sim * 4)]:
        print(f"  {label:22s}: EI = {EI(e, r):.3e}  "
              f"(x{EI(e, r)/EI(E_REAL, R_REAL):.0f} the real cable)")
    print("Stiffen E to kill the jelly and the rod's bending rigidity races past")
    print("the real cable's -> it stops behaving like a cable (becomes a stick).")

    hr("3b) THE DEEPER WALL: BENDING vs AXIAL ARE TIED BY ONE E (-> STRETCH)")
    print("Even ignoring thickness, FEM can't be a FAITHFUL cable. A real cable is")
    print("bending-SOFT but axially-STIFF (it sags yet barely stretches). Those are")
    print("EI ~ E*r^4 and EA ~ E*r^2 -- with a SINGLE E you cannot set them apart.")
    print()
    EA_real = E_REAL * math.pi * R_REAL ** 2
    # exact-EI rescale (exp=4): E_sim = E*(r/R)^4  -> EA_sim/EA_real = (r/R)^2
    ratio_thin = R_REAL / R_SIM
    EA_sim = (E_REAL * ratio_thin ** 4) * math.pi * R_SIM ** 2
    print(f"  match bending exactly (exp=4): EI_sim/EI_real = 1.00  (good)")
    print(f"     ...but then  EA_sim/EA_real = (r/R)^2 = {EA_sim/EA_real:.3f}")
    print(f"     -> axial stiffness is only {EA_sim/EA_real*100:.1f}% of real")
    print(f"     -> MEASURED on this machine: the exact-EI cable STRETCHED ~50% under")
    print(f"        its own weight (arc length 1.5 m for a 1.0 m cable). A real cable")
    print(f"        stretches ~0%. So 'faithful bending' FEM is NOT a faithful cable.")
    print(f"  match axial instead (firm, exp<=2.5): no stretch, but EI is 20-100x too")
    print(f"     high (section 3) -> too stiff to bend like the real cable.")
    print("  => you get faithful BENDING or faithful AXIAL, never both. A 1-D rod")
    print("     model sets bend/stretch/twist stiffness INDEPENDENTLY, so it gets all.")

    hr("THE TRADE-OFF TRIANGLE  (voxel FEM gives any TWO, never all three)")
    print("        THIN (1.5 mm)")
    print("            /\\")
    print("           /  \\")
    print("          /    \\")
    print("  correct EI --- NON-JELLY")
    print(" cable_fem.py sits on the [correct-EI <-> non-jelly] edge with a FAT rod.")

    hr("WHAT TO USE INSTEAD (these never voxelize -> thinness is free)")
    print("  * cable_warp.py -- Cosserat/XPBD elastic rod: native mm radius, exact")
    print("                     EI/torsion, NO volumetric jelly.  <-- THIN + NON-JELLY")
    print("  * cable.py      -- capsule chain + D6 joints: thin capsules, beam bending")
    print()
    print("cable_fem.py is the right tool ONLY when you specifically need a")
    print("VOLUMETRIC contact response (a fat soft body squashing/draping); it is the")
    print("wrong tool for a thin, stiff cable. Best the FEM path can do here:")
    print(f"  R_sim = {R_SIM*1000:.0f} mm (the cooking floor), E_sim ~ {E_sim/1e3:.0f} kPa.")
    print("=" * 72)


if __name__ == "__main__":
    main()
