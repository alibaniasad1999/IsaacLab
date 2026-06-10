"""
Compare the two cable simulation methods (rigid capsule chain vs FEM
deformable body) on the two-robots experiment.

Reads the outputs written by cable_two_robots.py:
    cable_output/two_robots_capsule/{trajectory.csv, summary.json}
    cable_output/two_robots_deformable/{trajectory.csv, summary.json}

and produces, in cable_output/comparison/:
    comparison.png    -- overlaid trajectories + speed bar chart
    comparison.json   -- numeric metrics side by side

If the single-cable outputs (hanging_kick / deformable_hanging_kick) exist,
their stability + timing are included in the JSON/table too.

Run (no Isaac Sim needed):
    python scripts/cable_simulation/compare_methods.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
OUT_BASE   = SCRIPT_DIR / "cable_output"
CMP_DIR    = OUT_BASE / "comparison"
CMP_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {
    "capsule":    OUT_BASE / "two_robots_capsule",
    "deformable": OUT_BASE / "two_robots_deformable",
}
COLORS = {"capsule": "tab:red", "deformable": "tab:blue"}


def load_run(run_dir: Path):
    summary_path = run_dir / "summary.json"
    csv_path = run_dir / "trajectory.csv"
    if not summary_path.exists() or not csv_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader if row])
    data = {name: rows[:, i] for i, name in enumerate(header)}
    return {"summary": summary, "data": data}


def main():
    runs = {}
    for method, run_dir in METHODS.items():
        run = load_run(run_dir)
        if run is None:
            print(f"[skip] no output for '{method}' in {run_dir} "
                  f"(run cable_two_robots.py with CABLE_METHOD={method})")
        else:
            runs[method] = run

    if not runs:
        print("Nothing to compare yet.")
        return

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Two robots + cable: capsule chain vs FEM deformable")

    ax = axes[0, 0]
    for m, run in runs.items():
        d = run["data"]
        ax.plot(d["t"], d["mid_y"], color=COLORS[m], label=f"{m} mid y")
        ax.plot(d["t"], d["leader_ee_y"], color=COLORS[m], ls="--", alpha=0.5,
                label=f"{m} leader EE y")
    ax.set_xlabel("t [s]"); ax.set_ylabel("y [m]")
    ax.set_title("Lateral motion (leader swing -> cable midpoint)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for m, run in runs.items():
        d = run["data"]
        ax.plot(d["t"], d["cable_sag"] * 1e3, color=COLORS[m], label=m)
    ax.set_xlabel("t [s]"); ax.set_ylabel("sag [mm]")
    ax.set_title("Cable midpoint sag below TCP line")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for m, run in runs.items():
        d = run["data"]
        ax.plot(d["t"], d["cable_span"], color=COLORS[m], label=m)
    ax.set_xlabel("t [s]"); ax.set_ylabel("TCP span [m]")
    ax.set_title("End-to-end span (follower pull-in)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    names, rtfs, bar_colors = [], [], []
    for m, run in runs.items():
        names.append(m)
        rtfs.append(run["summary"].get("realtime_factor", 0.0))
        bar_colors.append(COLORS[m])
    bars = ax.bar(names, rtfs, color=bar_colors)
    ax.bar_label(bars, fmt="%.2fx")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_ylabel("realtime factor (sim s / wall s)")
    ax.set_title("Simulation speed")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    plot_path = CMP_DIR / "comparison.png"
    fig.savefig(plot_path, dpi=140)
    print(f"Plot:    {plot_path}")

    # ---------------- Metrics ----------------
    metrics = {}
    for m, run in runs.items():
        s, d = run["summary"], run["data"]
        metrics[m] = {
            "stable":            s.get("stable"),
            "sim_time_s":        s.get("total_sim_time_s"),
            "wall_clock_s":      s.get("wall_clock_s"),
            "realtime_factor":   s.get("realtime_factor"),
            "mean_sag_mm":       float(np.mean(d["cable_sag"])) * 1e3,
            "max_abs_mid_y_m":   float(np.max(np.abs(d["mid_y"]))),
            "span_min_m":        float(np.min(d["cable_span"])),
            "span_max_m":        float(np.max(d["cable_span"])),
        }

    # Cross-method trajectory difference (common time grid)
    if len(runs) == 2:
        (m1, r1), (m2, r2) = runs.items()
        t1, t2 = r1["data"]["t"], r2["data"]["t"]
        t_end = min(t1[-1], t2[-1])
        t = np.linspace(0.0, t_end, 500)
        diffs = {}
        for col in ("mid_y", "mid_z", "cable_span", "cable_sag"):
            a = np.interp(t, t1, r1["data"][col])
            b = np.interp(t, t2, r2["data"][col])
            diffs[f"rms_diff_{col}_mm"] = float(np.sqrt(np.mean((a - b) ** 2))) * 1e3
        metrics["cross_method"] = diffs
        if metrics[m2]["wall_clock_s"] and metrics[m1]["wall_clock_s"]:
            metrics["cross_method"]["speed_ratio"] = (
                metrics[m1]["realtime_factor"] / metrics[m2]["realtime_factor"])

    # Include single-cable (hanging) runs if present, timing/stability only
    for label, sub in (("hanging_capsule", "hanging_kick"),
                       ("hanging_deformable", "deformable_hanging_kick")):
        p = OUT_BASE / sub / "summary.json"
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            metrics[label] = {
                "stable":          s.get("stable"),
                "sim_time_s":      s.get("total_sim_time_s"),
                "wall_clock_s":    s.get("wall_clock_s"),
                "realtime_factor": s.get("realtime_factor"),
            }

    json_path = CMP_DIR / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {json_path}\n")

    # ---------------- Table ----------------
    rows = [(k, v.get("stable"), v.get("realtime_factor"),
             v.get("mean_sag_mm"), v.get("max_abs_mid_y_m"))
            for k, v in metrics.items() if k != "cross_method"]
    print(f"{'run':22s} {'stable':>7s} {'rtf':>8s} {'sag[mm]':>9s} {'|mid_y|max[m]':>14s}")
    for name, stable, rtf, sag, midy in rows:
        rtf_s = f"{rtf:.2f}x" if rtf is not None else "-"
        sag_s = f"{sag:.1f}" if sag is not None else "-"
        my_s  = f"{midy:.3f}" if midy is not None else "-"
        print(f"{name:22s} {str(stable):>7s} {rtf_s:>8s} {sag_s:>9s} {my_s:>14s}")
    if "cross_method" in metrics:
        print("\ncross-method RMS differences:")
        for k, v in metrics["cross_method"].items():
            unit = "" if k == "speed_ratio" else " mm"
            print(f"  {k:28s}: {v:.2f}{unit}")


if __name__ == "__main__":
    main()
