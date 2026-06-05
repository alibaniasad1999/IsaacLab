#!/usr/bin/env python3
"""Generate all figures for cable_report.pdf from simulation data."""

import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

DATA    = Path(__file__).parent.parent / "cable_output"
SWEEP   = DATA / "govoni_sweep"
KICK    = DATA / "hanging_kick"
OUTDIR  = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)


# =====================================================================
# Fig 1: Hanging-kick cable tip trajectory (x, z vs time)
# =====================================================================
def fig_tip_trajectory():
    df = pd.read_csv(KICK / "trajectory.csv")
    tip_x = "cap199_x"
    tip_z = "cap199_z"
    mid_z = "cap100_z"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax1.plot(df["t"], df[tip_x], label="Tip (link 199)", color="#d62728", linewidth=1.2)
    ax1.set_ylabel("X position (m)")
    ax1.set_title("Cable tip lateral displacement")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["t"], df[tip_z], label="Tip (link 199)", color="#d62728", linewidth=1.2)
    ax2.plot(df["t"], df[mid_z], label="Midpoint (link 100)", color="#1f77b4", linewidth=1.2)
    ax2.set_ylabel("Z position (m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Cable vertical position")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "tip_trajectory.pdf")
    plt.close(fig)
    print("  -> tip_trajectory.pdf")


# =====================================================================
# Fig 2: Cable shape snapshots (z vs link index at different times)
# =====================================================================
def fig_cable_shape_snapshots():
    df = pd.read_csv(KICK / "trajectory.csv")
    z_cols = [c for c in df.columns if c.endswith("_z")]
    x_cols = [c for c in df.columns if c.endswith("_x")]
    link_indices = [int(c.replace("cap", "").replace("_z", "")) for c in z_cols]
    sorted_order = np.argsort(link_indices)
    link_indices = [link_indices[i] for i in sorted_order]
    z_cols = [z_cols[i] for i in sorted_order]
    x_cols = [x_cols[i] for i in sorted_order]

    times = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(times)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for t_target, color in zip(times, colors):
        idx = (df["t"] - t_target).abs().idxmin()
        zs = [df.loc[idx, c] for c in z_cols]
        xs = [df.loc[idx, c] for c in x_cols]
        t_actual = df.loc[idx, "t"]
        ax.plot(xs, zs, "o-", color=color, markersize=3, linewidth=1.5,
                label=f"t = {t_actual:.1f} s")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Z position (m)")
    ax.set_title("Cable shape at selected time instants (hanging kick)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUTDIR / "cable_shapes.pdf")
    plt.close(fig)
    print("  -> cable_shapes.pdf")


# =====================================================================
# Fig 3: Govoni sweep — midpoint Z displacement over time (all 5 runs)
# =====================================================================
def fig_govoni_midpoint():
    labels = {
        "run_1":  "Row 1: E=12.6 MPa, i=10",
        "run_2a": "Row 2a: E=526 MPa, i=10",
        "run_2b": "Row 2b: E=1002 MPa, i=6",
        "run_3":  "Row 3: E=1002.6 MPa, i=10 (paper: unstable)",
        "run_4":  "Row 4: E=1002.6 MPa, i=10",
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    linestyles = ["-", "-", "-", "--", ":"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (run_id, label), color, ls in zip(labels.items(), colors, linestyles):
        csv_path = SWEEP / run_id / "trajectory.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        z_cols = sorted([c for c in df.columns if c.endswith("_z")])
        mid_col = z_cols[len(z_cols) // 2]
        # Normalize to displacement from initial
        z0 = df[mid_col].iloc[0]
        lw = 2.0 if run_id == "run_3" else 1.2
        ax.plot(df["t"], df[mid_col] - z0, color=color, linestyle=ls,
                linewidth=lw, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Midpoint Z displacement (m)")
    ax.set_title("Govoni sweep: midpoint vertical displacement after step input")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTDIR / "govoni_midpoint.pdf")
    plt.close(fig)
    print("  -> govoni_midpoint.pdf")


# =====================================================================
# Fig 4: Derived stiffness vs Young's modulus (parametric analysis)
# =====================================================================
def fig_stiffness_landscape():
    E_range = np.logspace(6, 10, 200)  # 1 MPa to 10 GPa
    r = 1.5e-3
    I_area = math.pi * r**4 / 4
    N_values = [6, 10, 50, 200]
    L_total = 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for N, color in zip(N_values, colors):
        L_seg = L_total / N
        K_bend = E_range * I_area / L_seg
        K_deg = K_bend * math.pi / 180
        ax1.loglog(E_range / 1e6, K_deg, color=color, linewidth=1.5,
                   label=f"N = {N}")

    # Mark the Govoni test points
    for E_mpa, marker, ms in [(12.6, "o", 8), (526, "s", 8), (1002.6, "D", 8)]:
        L_seg = L_total / 10
        K = E_mpa * 1e6 * I_area / L_seg * math.pi / 180
        ax1.plot(E_mpa, K, marker, color="black", markersize=ms, zorder=5)

    ax1.set_xlabel("Young's modulus E (MPa)")
    ax1.set_ylabel("Joint stiffness K (N$\\cdot$m/deg)")
    ax1.set_title("Bending stiffness vs. material")
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    # Material regions
    materials = [
        (1, 100, "Rubber", "#2ca02c"),
        (100, 3000, "Plastics", "#ff7f0e"),
        (3000, 10000, "Metals", "#d62728"),
    ]
    for lo, hi, name, color in materials:
        ax1.axvspan(lo, hi, alpha=0.08, color=color)
        ax1.text(math.sqrt(lo * hi), ax1.get_ylim()[0] * 3, name,
                 ha="center", fontsize=8, color=color, fontweight="bold")

    # Right panel: natural frequency vs E for different N
    for N, color in zip(N_values, colors):
        L_seg = L_total / N
        m_seg = 1.0 / N
        K_bend = E_range * I_area / L_seg
        I_rot = (1/3) * m_seg * L_seg**2
        omega_n = np.sqrt(K_bend / I_rot)
        f_n = omega_n / (2 * math.pi)
        ax2.loglog(E_range / 1e6, f_n, color=color, linewidth=1.5,
                   label=f"N = {N}")

    ax2.axhline(120, color="gray", linestyle="--", linewidth=1,
                label="Physics rate (240 Hz / 2)")
    ax2.set_xlabel("Young's modulus E (MPa)")
    ax2.set_ylabel("Natural frequency $f_n$ (Hz)")
    ax2.set_title("Joint natural frequency vs. material")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUTDIR / "stiffness_landscape.pdf")
    plt.close(fig)
    print("  -> stiffness_landscape.pdf")


# =====================================================================
# Fig 5: Max angular velocity bar chart (Govoni sweep)
# =====================================================================
def fig_max_omega_bar():
    rows = ["1", "2a", "2b", "3", "4"]
    labels = [
        "Row 1\nE=12.6\ni=10",
        "Row 2a\nE=526\ni=10",
        "Row 2b\nE=1002\ni=6",
        "Row 3\nE=1002.6\ni=10",
        "Row 4\nE=1002.6\ni=10",
    ]
    max_omegas = []
    for r in rows:
        with open(SWEEP / f"run_{r}" / "summary.json") as f:
            s = json.load(f)
        max_omegas.append(s["max_omega_deg_per_s"])

    paper_stable = [True, True, True, False, True]
    colors = ["#2ca02c" if s else "#d62728" for s in paper_stable]
    # All ours are stable, use edge color to distinguish
    bar_colors = ["#4CAF50", "#4CAF50", "#4CAF50", "#4CAF50", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(rows)), max_omegas, color=bar_colors,
                  edgecolor=colors, linewidth=2.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Max angular velocity (deg/s)")
    ax.set_title("Peak angular velocity across Govoni configurations\n"
                 "(green border = paper stable, red border = paper unstable)")
    ax.axhline(1e4, color="red", linestyle="--", linewidth=1,
               label="Divergence threshold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate row 3
    ax.annotate("Paper diverges here\nOurs: stable",
                xy=(3, max_omegas[3]), xytext=(3.5, max_omegas[3] + 150),
                fontsize=9, fontweight="bold", color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))

    fig.tight_layout()
    fig.savefig(OUTDIR / "max_omega_bar.pdf")
    plt.close(fig)
    print("  -> max_omega_bar.pdf")


# =====================================================================
# Fig 6: Tip x-z phase portrait (hanging kick — shows energy dissipation)
# =====================================================================
def fig_phase_portrait():
    df = pd.read_csv(KICK / "trajectory.csv")
    tip_x = df["cap199_x"].values
    tip_z = df["cap199_z"].values
    t = df["t"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(tip_x, tip_z, c=t, cmap="coolwarm", s=2, zorder=3)
    ax.plot(tip_x[0], tip_z[0], "go", markersize=10, label="Start", zorder=4)
    ax.plot(tip_x[-1], tip_z[-1], "rs", markersize=10, label="End", zorder=4)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time (s)")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Z position (m)")
    ax.set_title("Cable tip trajectory in X-Z plane\n(color = time, showing energy dissipation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "phase_portrait.pdf")
    plt.close(fig)
    print("  -> phase_portrait.pdf")


# =====================================================================
# Fig 7: Damping ratio sensitivity analysis (derived parameters)
# =====================================================================
def fig_damping_sensitivity():
    E = 12.6e6
    r = 1.5e-3
    L_seg = 1.0 / 200
    m_seg = 1.0 / 200
    I_area = math.pi * r**4 / 4
    K = E * I_area / L_seg
    I_rot = (1/3) * m_seg * L_seg**2

    zeta_range = np.linspace(0.01, 2.0, 200)
    C_crit = 2 * math.sqrt(K * I_rot)
    C_vals = zeta_range * C_crit * math.pi / 180  # N.m.s/deg

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(zeta_range, C_vals, "b-", linewidth=2)
    ax.axvline(0.2, color="green", linestyle="--", linewidth=1.5,
               label=f"Our default ($\\zeta = 0.2$)")
    ax.axvline(1.0, color="orange", linestyle="--", linewidth=1.5,
               label="Critical damping ($\\zeta = 1.0$)")
    ax.axhline(0.05, color="red", linestyle=":", linewidth=1.5,
               label="v1 hand-tuned value (0.05 N·m·s/deg)")

    ax.set_xlabel("Damping ratio $\\zeta$")
    ax.set_ylabel("Joint damping C (N$\\cdot$m$\\cdot$s/deg)")
    ax.set_title("Joint damping vs. damping ratio\n(E = 12.6 MPa rubber, N = 200)")
    ax.set_ylim(None, 1e-1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "damping_sensitivity.pdf")
    plt.close(fig)
    print("  -> damping_sensitivity.pdf")


if __name__ == "__main__":
    print("Generating figures...")
    fig_tip_trajectory()
    fig_cable_shape_snapshots()
    fig_govoni_midpoint()
    fig_stiffness_landscape()
    fig_max_omega_bar()
    fig_phase_portrait()
    fig_damping_sensitivity()
    print("Done. Figures in:", OUTDIR)
