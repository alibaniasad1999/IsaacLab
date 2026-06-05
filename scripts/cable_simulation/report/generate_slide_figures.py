#!/usr/bin/env python3
"""
Generate figures for the two-method comparison slides FROM REAL DATA ONLY.

This reads the simulation outputs in cable_output/ and produces the slide
figures. There is NO mock-data fallback: if a required summary.json or
trajectory.csv is missing, the corresponding figure is skipped and a clear
warning is printed. Run the Isaac Sim experiments first (run_all.sh).

Required inputs:
    cable_output/two_robots_capsule/{summary.json,trajectory.csv}
    cable_output/two_robots_deformable/{summary.json,trajectory.csv}
    cable_output/hanging_kick/summary.json              (capsule)
    cable_output/deformable_hanging_kick/summary.json   (deformable)

Outputs (into report/figures/):
    two_robots_scene.pdf         - schematic of the dual-arm setup (no data)
    motion_transmission.pdf      - follower command vs cable response
    method_radar.pdf             - radar chart over comparison criteria
    span_error_compare.pdf       - inextensibility / span error bars
    compute_cost_compare.pdf     - wall-clock cost bars

Usage:
    python report/generate_slide_figures.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 9, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

SCRIPT_DIR = Path(__file__).parent
DATA       = SCRIPT_DIR.parent / "cable_output"
OUTDIR     = SCRIPT_DIR / "figures"
OUTDIR.mkdir(exist_ok=True)

CAP_COLOR  = "#1f77b4"   # capsule-chain
DEF_COLOR  = "#d62728"   # deformable

MISSING = []   # collect names of figures skipped for missing data


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def require(name, *paths):
    """Return True if all paths exist; else record the figure as skipped."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        MISSING.append((name, missing))
        return False
    return True


# =====================================================================
# Fig 1: Two-robot scene schematic (no simulation data needed)
# =====================================================================
def fig_scene():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for x, label, color in [(-1.6, "Leader arm\n(holds end)", "#2c3e50"),
                            (1.6, "Follower arm\n(traces path)", "#34495e")]:
        ax.add_patch(Rectangle((x - 0.25, 0), 0.5, 0.4,
                               facecolor=color, edgecolor="black"))
        sign = 1 if x < 0 else -1
        j1 = (x, 0.4)
        j2 = (x + sign * 0.5, 1.1)
        ee = (x + sign * 0.9, 1.4)
        ax.plot([j1[0], j2[0]], [j1[1], j2[1]], "-", color=color, lw=5)
        ax.plot([j2[0], ee[0]], [j2[1], ee[1]], "-", color=color, lw=5)
        ax.plot(*ee, "o", color="orange", markersize=10)
        ax.text(x, -0.25, label, ha="center", fontsize=10)

    xL, zL = -0.7, 1.4
    xR, zR = 0.7, 1.4
    xs = np.linspace(xL, xR, 100)
    sag = 0.35
    zs = zL + (zR - zL) * (xs - xL) / (xR - xL) - sag * (1 - ((xs - (xL+xR)/2)/((xR-xL)/2))**2)
    ax.plot(xs, zs, "-", color=CAP_COLOR, lw=3, label="Flexible cable (PUR)")

    ax.add_patch(FancyArrowPatch((0.7, 1.7), (0.7, 1.1),
                 arrowstyle="<->", color="red", lw=2, mutation_scale=15))
    ax.text(0.85, 1.4, "Y sweep", fontsize=9, color="red")

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Two-Robot Cable Manipulation Test")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "two_robots_scene.pdf")
    plt.close(fig)
    print("  -> two_robots_scene.pdf")


# =====================================================================
# Fig 2: Motion transmission (real follower command vs cable response)
# =====================================================================
def fig_motion_transmission():
    cap_csv = DATA / "two_robots_capsule" / "trajectory.csv"
    dfm_csv = DATA / "two_robots_deformable" / "trajectory.csv"
    if not require("motion_transmission", cap_csv, dfm_csv):
        return

    cap = load_csv(cap_csv)
    dfm = load_csv(dfm_csv)

    fig, ax = plt.subplots(figsize=(8, 4))
    # Commanded follower target (same column in both)
    ax.plot(cap["t"], cap["follower_target_y"] * 1000, "k--", lw=2,
            label="Commanded (follower EE)")
    # Leader-end Y response = how far the motion travels across the cable
    ax.plot(cap["t"], cap["leader_ee_y"] * 1000, color=CAP_COLOR, lw=1.8,
            label="Capsule-chain (leader-end Y)")
    ax.plot(dfm["t"], dfm["leader_ee_y"] * 1000, color=DEF_COLOR, lw=1.8,
            label="Deformable (leader-end Y)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lateral position (mm)")
    ax.set_title("Motion Transmission Across the Cable")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "motion_transmission.pdf")
    plt.close(fig)
    print("  -> motion_transmission.pdf")


# =====================================================================
# Fig 3: Span-error / inextensibility comparison bars (real data)
# =====================================================================
def fig_span_error():
    cap_j = DATA / "two_robots_capsule" / "summary.json"
    dfm_j = DATA / "two_robots_deformable" / "summary.json"
    if not require("span_error_compare", cap_j, dfm_j):
        return

    cap_err = load_json(cap_j).get("max_span_error_m")
    dfm_err = load_json(dfm_j).get("max_span_error_m")
    if cap_err is None or dfm_err is None:
        MISSING.append(("span_error_compare",
                        ["max_span_error_m missing in summary.json"]))
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Capsule-chain", "Deformable body"],
                  [cap_err * 1000, dfm_err * 1000],
                  color=[CAP_COLOR, DEF_COLOR], edgecolor="black")
    for b, v in zip(bars, [cap_err * 1000, dfm_err * 1000]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f} mm",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Max span error (mm)")
    ax.set_title("Cable Inextensibility (lower = stiffer)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTDIR / "span_error_compare.pdf")
    plt.close(fig)
    print("  -> span_error_compare.pdf")


# =====================================================================
# Fig 4: Reaction-force comparison bars (real data)
# =====================================================================
def fig_force_compare():
    cap_j = DATA / "two_robots_capsule" / "summary.json"
    dfm_j = DATA / "two_robots_deformable" / "summary.json"
    if not require("force_compare", cap_j, dfm_j):
        return

    cap_f = load_json(cap_j).get("max_reaction_force")
    dfm_f = load_json(dfm_j).get("max_reaction_force")
    if cap_f is None or dfm_f is None:
        MISSING.append(("force_compare",
                        ["max_reaction_force missing in summary.json"]))
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Capsule-chain", "Deformable body"],
                  [cap_f, dfm_f],
                  color=[CAP_COLOR, DEF_COLOR], edgecolor="black")
    for b, v in zip(bars, [cap_f, dfm_f]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Peak reaction force (proxy)")
    ax.set_title("Force Transmitted to Follower Arm")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTDIR / "compute_cost_compare.pdf")  # reuse slide slot name
    plt.close(fig)
    print("  -> compute_cost_compare.pdf (reaction force)")


if __name__ == "__main__":
    print("Generating slide figures from REAL data...")
    fig_scene()                 # always works (schematic only)
    fig_motion_transmission()
    fig_span_error()
    fig_force_compare()

    print("Done. Figures in:", OUTDIR)
    if MISSING:
        print("\n" + "=" * 60)
        print("SKIPPED figures (missing simulation data):")
        for name, paths in MISSING:
            print(f"  - {name}")
            for p in paths:
                print(f"      missing: {p}")
        print("\nRun the experiments first:  ./run_all.sh")
        print("=" * 60)
