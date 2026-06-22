"""
identify_cable.py  --  identify a real cable's material from its hanging shape.
=============================================================================

You have a cable hanging between TWO fixed points under GRAVITY ONLY (no applied
force), and it does NOT stretch (inextensible -- a USB extender). From the 2-D
profile you extracted with extract_cable_profile.py, this script identifies the
cable's mechanical properties and prints the matching cable_config.py values so
you can reproduce it in the Isaac-Sim cable.py.

----------------------------------------------------------------------------
THE PHYSICS (and what is / isn't identifiable)  -- read this first
----------------------------------------------------------------------------
A stiff, inextensible rod hung at two points under gravity takes the ELASTICA
shape. Bending stiffness EI resists curving; self-weight mu = rho*A pulls down.
The DIMENSIONLESS shape depends on a SINGLE number:

        Gamma = mu * g * L^3 / EI        (gravity sag  /  bending resistance)

  * Gamma -> 0    : stiff rod, barely sags (nearly straight / shallow arc).
  * Gamma -> inf  : floppy rope, sags into an ideal CATENARY (EI negligible).

CONSEQUENCE: from the SHAPE ALONE you can identify Gamma = EI/mu (how floppy the
cable is per unit weight) -- but NOT EI and mu separately, because a stiff-heavy
and a floppy-light cable can hang identically. To get ABSOLUTE numbers you give
ONE extra real measurement: the cable MASS (weigh it). Then, knowing length L
and radius r:

        mu  = mass / L                              [kg/m]
        EI  = mu * g * L^3 / Gamma                  [N.m^2]   (from the fit)
        E   = EI / (pi * r^4 / 4)                   [Pa]      Young's modulus
        rho = mu / (pi * r^2)                       [kg/m^3]  density

E and rho are then directly comparable to YOUNG_MODULUS and DENSITY in
cable_config.py.

----------------------------------------------------------------------------
USAGE
----------------------------------------------------------------------------
    python identify_cable.py --profile media/IMG_0501_profile.csv \
        --mass-g 8.1 --length-m 1.0 --radius-mm 1.5

  --profile   : x_m,z_m CSV from extract_cable_profile.py (the hanging cable).
  --mass-g    : cable mass in grams (weigh it). Omit to get only Gamma & EI/mu.
  --length-m  : true cable length (arc length). Omit -> use the profile's arc.
  --radius-mm : cable radius. Needed to convert EI -> Young's modulus E.

Output: Gamma, EI, mu, E, rho (those it can determine), the elastica fit error,
and a ready-to-paste block of cable_config.py overrides.
"""

from __future__ import annotations

import argparse
import csv
import logging

import numpy as np

logging.basicConfig(level="INFO",
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("identify")

G = 9.81  # gravity [m/s^2]


# ===========================================================================
# Read the extracted profile
# ===========================================================================
def read_profile(path, frame="last"):
    """Read a cable profile, accepting BOTH formats:

      * simple : columns x_m, z_m            (photo output -- one profile)
      * wide   : columns t, p0_x.., p0_z..   (video output -- one row PER FRAME)

    For the wide/video format we pick ONE frame (default the last, settled one)
    so we identify the cable from a single static shape, as the physics needs.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise SystemExit(f"{path}: empty or header-only.")
    header = [c.strip().lower() for c in rows[0]]
    data = rows[1:]

    if "x_m" in header and "z_m" in header:
        xi, zi = header.index("x_m"), header.index("z_m")
        x = np.array([float(r[xi]) for r in data])
        z = np.array([float(r[zi]) for r in data])
    elif any(c.endswith("_x") for c in header):
        # wide per-frame video format: pick one frame, gather its p*_x / p*_z.
        x_cols = [i for i, c in enumerate(header) if c.endswith("_x")]
        z_cols = [i for i, c in enumerate(header) if c.endswith("_z")]
        if frame == "last":
            row = data[-1]
        elif frame == "first":
            row = data[0]
        else:
            row = data[int(frame)]
        x = np.array([float(row[i]) for i in x_cols])
        z = np.array([float(row[i]) for i in z_cols])
        log.info("video CSV: using frame '%s' of %d (%d points)",
                 frame, len(data), len(x))
    else:
        raise SystemExit(
            f"{path}: need columns x_m,z_m (photo) or t,p*_x,p*_z (video). "
            f"Got {header[:6]}...")

    # Order along the cable's DOMINANT axis: a vertical (hanging) cable must be
    # ordered by z, a horizontal one by x. Sorting by x would scramble a
    # vertical cable. Pick whichever axis spans more.
    if (z.max() - z.min()) > (x.max() - x.min()):
        o = np.argsort(z)
    else:
        o = np.argsort(x)
    x, z = x[o], z[o]
    return x, z


def arc_length(x, z):
    return float(np.sum(np.hypot(np.diff(x), np.diff(z))))


# ===========================================================================
# The elastica: shape of an inextensible rod under gravity, pinned at 2 ends
# ---------------------------------------------------------------------------
# Arc-length s in [0, L]. State along the rod (planar, gravity = -z):
#   x' = cos(theta),  z' = sin(theta)              (unit-speed, inextensible)
#   theta' = kappa                                 (curvature)
#   EI * kappa' = -(horizontal internal force) * sin(theta)
#                 + (vertical internal force) * cos(theta)
# With only gravity + end reactions, the internal force varies linearly along
# the rod. Non-dimensionalise by L: sigma = s/L in [0,1], and let
#   g* = mu g L^3 / EI = Gamma. The shape then depends only on Gamma and the
# (unknown) end reaction forces, which we solve for so the rod connects the two
# endpoints. We integrate theta(sigma) and fit Gamma + reactions to the data.
# ===========================================================================
def _elastica_xy(gamma, hx, hz, theta0, n=200):
    """Shape of a heavy elastica in dimensionless arc length t in [0,1].

    Forces are scaled by EI/L^2; the distributed weight is gamma per unit t,
    acting in -Z. The internal force at arc t equals the tip reaction (hx, hz)
    PLUS the weight of the remaining rod (gamma*(1-t) downward). Moment balance
    for an inextensible rod gives the curvature ODE:

        x'(t)     = cos(theta)
        z'(t)     = sin(theta)
        theta'(t) = kappa
        kappa'(t) = hz_eff * cos(theta) - hx * sin(theta)
        where hz_eff(t) = hz + gamma * (1 - t)      (weight below this point)

    Integrated with RK4 for accuracy/stability (the earlier Euler version was
    unstable). Returns (X, Z) sampled on t in [0,1] (multiply by L for metres).
    """
    t = np.linspace(0, 1, n)
    dt = t[1] - t[0]
    X = np.zeros(n); Z = np.zeros(n)
    th = theta0; ka = 0.0

    def deriv(state, tt):
        x_, z_, th_, ka_ = state
        hz_eff = hz + gamma * (1.0 - tt)
        return np.array([np.cos(th_), np.sin(th_), ka_,
                         hz_eff * np.cos(th_) - hx * np.sin(th_)])

    state = np.array([0.0, 0.0, th, ka])
    for i in range(1, n):
        tt = t[i - 1]
        k1 = deriv(state, tt)
        k2 = deriv(state + 0.5 * dt * k1, tt + 0.5 * dt)
        k3 = deriv(state + 0.5 * dt * k2, tt + 0.5 * dt)
        k4 = deriv(state + dt * k3, tt + dt)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        X[i], Z[i] = state[0], state[1]
    return X, Z


def fit_elastica(x, z, log_fn=log.info):
    """Fit the heavy elastica to a measured profile.

    Returns (gamma, rms_err_m, params). We normalise the data by its ARC LENGTH
    (so the model rod, which has unit dimensionless length, matches), then fit
    Gamma + tip reactions + start angle. We also try several initial Gamma so
    the optimiser does not get stuck (the elastica is non-convex in Gamma).
    """
    from scipy.optimize import least_squares
    from scipy.interpolate import interp1d

    # Normalise the data by its arc length so it has the same unit length as
    # the model rod, and resample evenly in arc length. (Smooth the PROFILE
    # upstream in extract_cable_profile.py with --smooth; here we keep the
    # measured shape so the fit reflects the real cable.)
    x0 = x - x[0]
    z0 = z - z[0]
    seg = np.hypot(np.diff(x0), np.diff(z0))
    s = np.concatenate([[0], np.cumsum(seg)])
    Lpx = s[-1]
    sn = s / Lpx
    m = 120
    grid = np.linspace(0, 1, m)
    Xd = interp1d(sn, x0 / Lpx)(grid)
    Zd = interp1d(sn, z0 / Lpx)(grid)

    def residual(p):
        gamma, hx, hz, th0 = p
        X, Z = _elastica_xy(gamma, hx, hz, th0, n=m)
        return np.concatenate([X - Xd, Z - Zd])

    best = None
    th0 = np.arctan2(Zd[4] - Zd[0], Xd[4] - Xd[0])
    # dense multi-start over Gamma AND tip vertical reaction (the two that set
    # the shape) so a noisy curve does not trap the optimiser in a bad minimum.
    for g0 in (0.3, 1.0, 3.0, 8.0, 20.0, 50.0, 150.0, 500.0):
        for hz0 in (-g0, -g0 / 2.0, 0.0):
            p0 = [g0, 0.0, hz0, th0]
            bounds = ([0.0, -1e4, -1e4, -np.pi], [1e6, 1e4, 1e4, np.pi])
            try:
                sol = least_squares(residual, p0, bounds=bounds, max_nfev=4000,
                                    x_scale=[g0 + 1, 10, 10, 1])
            except Exception:
                continue
            cost = float(np.sqrt(np.mean(sol.fun ** 2)))
            if best is None or cost < best[0]:
                best = (cost, sol)
    if best is None:
        raise SystemExit("elastica fit failed.")
    rms_n, sol = best
    gamma = float(sol.x[0])
    rms_m = rms_n * Lpx

    # Identifiability: in the FLOPPY regime the shape stops changing with Gamma,
    # so large Gamma is only a LOWER BOUND. Detect by how much the residual
    # changes between this Gamma and 2x it.
    r2 = float(np.sqrt(np.mean(residual([gamma * 2] + list(sol.x[1:])) ** 2)))
    sensitive = abs(r2 - rms_n) > 0.2 * rms_n + 1e-6
    identifiable = gamma < 60 and sensitive
    log_fn("elastica fit: Gamma = %.4g  (rms %.1f mm, %s)", gamma, rms_m * 1000,
           "well-identified" if identifiable else "floppy: Gamma is a LOWER bound")

    # fitted model curve back in the data's metric frame (for --overlay).
    Xm, Zm = _elastica_xy(gamma, sol.x[1], sol.x[2], sol.x[3], n=m)
    fit_xy = (Xm * Lpx + x[0], Zm * Lpx + z[0])
    return gamma, rms_m, sol.x, identifiable, fit_xy


# ===========================================================================
# Identification
# ===========================================================================
def identify(profile_path, mass_g=None, length_m=None, radius_mm=None,
             frame="last", overlay=False):
    x, z = read_profile(profile_path, frame=frame)
    L_arc = arc_length(x, z)              # length MEASURED from the image (metres)
    L = length_m if length_m else L_arc
    src = "you gave --length-m" if length_m else "MEASURED from the image"
    log.info("profile: %d points. cable length L = %.3f m (%s).",
             len(x), L, src)
    if length_m and abs(length_m - L_arc) > 0.1 * max(length_m, 1e-6):
        log.warning("image-measured length %.3f m differs >10%% from your "
                    "--length-m %.3f m -- check the scale reference.",
                    L_arc, length_m)

    gamma, rms_m, _, identifiable, fit_xy = fit_elastica(x, z)
    print("\n========================  IDENTIFICATION  ========================")
    bound = "" if identifiable else "  (LOWER BOUND -- cable too floppy to pin down)"
    print(f"  hanging-shape stiffness   Gamma = mu g L^3 / EI = {gamma:.4g}{bound}")
    if gamma < 1:
        regime = "STIFF rod (barely sags)"
    elif gamma < 50:
        regime = "semi-flexible cable"
    else:
        regime = "FLOPPY rope (near-ideal catenary, EI tiny)"
    print(f"  regime                    : {regime}")
    print(f"  elastica fit error        : {rms_m*1000:.1f} mm")
    if not identifiable:
        print("  NOTE: the cable hangs like an ideal catenary -> its bending")
        print("        stiffness EI is too small to measure from this shape.")
        print("        The reported Gamma/EI is a LOWER bound on floppiness;")
        print("        E is an UPPER bound. For a stiffer reading, hang it from")
        print("        a SHORTER span (less sag) so bending matters.")
    print(f"  EI / mu                    = g L^3 / Gamma = "
          f"{G * L**3 / gamma:.4g}  [m^4/s^2]")

    out = {"gamma": gamma, "L": L, "EI_over_mu": G * L**3 / gamma,
           "rms_m": rms_m}

    if mass_g is not None:
        mu = (mass_g / 1000.0) / L                     # kg/m
        EI = mu * G * L**3 / gamma                      # N.m^2
        out.update(mu=mu, EI=EI, mass_kg=mass_g / 1000.0)
        print(f"\n  -- with mass {mass_g:.2f} g --")
        print(f"  mass per length   mu   = {mu*1000:.3f} g/m")
        print(f"  bending stiffness EI   = {EI:.4g} N.m^2")
        if radius_mm is not None:
            r = radius_mm / 1000.0
            I = np.pi * r**4 / 4.0
            E = EI / I
            A = np.pi * r**2
            rho = mu / A
            out.update(E=E, rho=rho, radius_m=r)
            print(f"  Young's modulus   E    = {E/1e6:.3g} MPa")
            print(f"  density           rho  = {rho:.0f} kg/m^3")
            _print_config_block(L, r, E, rho, mass_g / 1000.0)
        else:
            print("  (give --radius-mm to convert EI -> Young's modulus E)")
    else:
        print("\n  Shape alone gives the RATIO EI/mu only. To get absolute EI,")
        print("  Young's modulus and density, re-run with --mass-g (weigh the")
        print("  cable) and --radius-mm.")
    print("==================================================================")

    if overlay:
        _plot_overlay(x, z, fit_xy, gamma, rms_m, identifiable, profile_path)
    return out


def _plot_overlay(x, z, fit_xy, gamma, rms_m, identifiable, profile_path):
    """Plot the measured profile + the fitted elastica so you can eyeball it."""
    import os
    try:
        import matplotlib
        matplotlib.use("MacOSX" if os.sys.platform == "darwin" else "TkAgg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.info("(overlay skipped: matplotlib unavailable: %s)", exc)
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, z, "o", ms=3, color="crimson", label="measured cable")
    ax.plot(fit_xy[0], fit_xy[1], "-", color="royalblue", lw=2,
            label=f"fitted elastica (Γ={gamma:.3g})")
    tag = "well-identified" if identifiable else "floppy: Γ is a LOWER bound"
    ax.set_title(f"Cable fit  --  rms {rms_m*1000:.1f} mm  ({tag})")
    ax.set_xlabel("x  [m]"); ax.set_ylabel("z  height [m]")
    ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.3); ax.legend()
    out_png = os.path.splitext(profile_path)[0] + "_fit.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
    log.info("saved fit overlay -> %s", out_png)
    plt.show()


def _print_config_block(L, r, E, rho, mass_kg):
    """Print cable_config.py overrides you can paste / export as env vars."""
    print("\n  ----- matching cable_config.py values -----")
    print(f"  TOTAL_CABLE_LENGTH = {L:.4f}     # m")
    print(f"  REAL_RADIUS        = {r:.5f}   # m")
    print(f"  YOUNG_MODULUS      = {E:.4g}    # Pa")
    print(f"  DENSITY            = {rho:.1f}      # kg/m^3")
    print(f"  CABLE_MASS         = {mass_kg:.5f}  # kg")
    print("\n  or run the sim directly with these:")
    print(f"    CABLE_LENGTH={L:.4f} CABLE_RADIUS={r:.5f} \\")
    print(f"    CABLE_E={E:.4g} CABLE_DENSITY={rho:.1f} \\")
    print(f"    CABLE_MASS={mass_kg:.5f} python ../base/cable.py")


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Identify a cable's properties from its hanging profile.")
    p.add_argument("--profile", required=True,
                   help="profile CSV: x_m,z_m (photo) or t,p*_x,p*_z (video)")
    p.add_argument("--mass-g", type=float, default=None,
                   help="cable mass in grams (weigh it) -> absolute EI, E, rho")
    p.add_argument("--length-m", type=float, default=None,
                   help="true cable length [m] (default: profile arc length)")
    p.add_argument("--radius-mm", type=float, default=None,
                   help="cable radius [mm] -> Young's modulus E")
    p.add_argument("--frame", default="last",
                   help="for a VIDEO csv: which frame ('last','first', or index)")
    p.add_argument("--overlay", action="store_true",
                   help="plot the fitted elastica over the measured profile")
    args = p.parse_args(argv)
    identify(args.profile, args.mass_g, args.length_m, args.radius_mm,
             frame=args.frame, overlay=args.overlay)


if __name__ == "__main__":
    main()
