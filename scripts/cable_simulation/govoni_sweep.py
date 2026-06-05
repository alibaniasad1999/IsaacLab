#!/usr/bin/env python3
"""
Govoni Table 1 sweep driver  —  replicates the parameter sweep from
Govoni et al. 2025 (arXiv:2504.13659, Table 1) using cable.py.

For each row in Govoni Table 1, this script:
  1. sets the env vars consumed by cable.py,
  2. launches cable.py as a subprocess (headless, no video),
  3. reads back the summary.json the subprocess writes,
  4. assembles a comparison table.

Final output:
    cable_output/govoni_sweep/comparison.csv
    cable_output/govoni_sweep/comparison.md
    cable_output/govoni_sweep/run_<N>/summary.json    (one per row)

Usage:
    python govoni_sweep.py
    python govoni_sweep.py --rows 1,3       # subset of rows
    python govoni_sweep.py --dry-run        # print commands, don't run
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
CABLE_V2    = SCRIPT_DIR / "cable.py"
OUTPUT_ROOT = SCRIPT_DIR / "cable_output" / "govoni_sweep"

# ----------------------------------------------------------------
# Govoni et al. 2025, Table 1 (page 4 of arXiv:2504.13659)
#
#   No. | E (MPa)  | mi disc. | Δt (s)     | Stability | Sec→inst.
#   ----|----------|----------|------------|-----------|----------
#    1  | 12.6     |   i=10   | 5.0e-6     | Stable    | ∞
#    2a | 526.0    |   i=10   | 5.0e-6     | Stable    | ∞
#    2b | 1002.0   |   i=6    | 5.0e-6     | Stable    | ∞
#    3  | 1002.6   |   i=10   | 5.0e-6     | Unstable  | 0.4
#    4  | 1002.6   |   i=10   | 1.0e-7     | Stable    | ∞
#
# "i" is mass discretization in their MSD model (number of point masses).
# We map this directly to NUM_LINKS in our capsule chain.
#
# Caveat: their Δt of 5e-6 s (200 kHz) is impractical in Isaac Sim with
# 200+ rigid bodies. We instead run at the user's working physics dt
# (1/240 s = ~4.17e-3) using PhysX TGS with 64 position iterations, which
# is the regime cable.py is actually intended for. The point of the
# sweep is to see whether *our* solver+model can handle the stiffness
# regimes where their explicit MSD integrator fails — not to literally
# match their Δt.
# ----------------------------------------------------------------

GOVONI_TABLE_1 = [
    # (row_id, E_MPa, i_discretization, dt_govoni_s, paper_stability, paper_sec_to_inst)
    ("1",  12.6,   10, 5.0e-6,  "Stable",   None),
    ("2a", 526.0,  10, 5.0e-6,  "Stable",   None),
    ("2b", 1002.0,  6, 5.0e-6,  "Stable",   None),
    ("3",  1002.6, 10, 5.0e-6,  "Unstable", 0.4),
    ("4",  1002.6, 10, 1.0e-7,  "Stable",   None),
]

# Common simulation settings (kept identical across rows for a fair sweep)
COMMON_ENV = {
    "CABLE_MODE":      "both_ends_fixed",   # Govoni-style: fix ends, displace one
    "CABLE_HEADLESS":  "1",                 # no GUI
    "CABLE_RECORD":    "0",                 # no video
    "CABLE_LENGTH":    "0.5",               # 50 cm cable
    "CABLE_RADIUS":    "1.5e-3",            # 1.5 mm
    "CABLE_MASS":      "0.05",              # 50 g (typical thin rubber cord)
    "CABLE_NU":        "0.5",               # rubber Poisson ratio
    "CABLE_ZETA":      "0.2",               # damping ratio
    "CABLE_PHYSICS_DT": str(1.0 / 240.0),   # our solver's working Δt
    "CABLE_RENDER_DT":  str(1.0 / 60.0),
    "CABLE_SETTLE":    "0.5",               # settle 0.5 s before applying step
    "CABLE_STEP_DISP": "5.0e-3",            # 5 mm step (matches Govoni)
    "CABLE_MAX_TIME":  "2.0",               # 2 s of simulation per run
}


def run_one(row_id: str, E_MPa: float, i_disc: int,
            dt_govoni: float, dry_run: bool) -> dict | None:
    """Run cable.py for one Table 1 row and return its summary.json."""
    run_dir = OUTPUT_ROOT / f"run_{row_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["CABLE_E"]          = f"{E_MPa * 1e6:.6e}"   # MPa → Pa
    env["CABLE_NUM_LINKS"]  = str(i_disc)
    env["CABLE_OUTPUT_DIR"] = str(run_dir)

    cmd = [sys.executable, str(CABLE_V2)]
    print("─" * 72)
    print(f"  Row {row_id}:  E={E_MPa} MPa,  i={i_disc},  Govoni Δt={dt_govoni:.1e}s")
    print(f"           our Δt={COMMON_ENV['CABLE_PHYSICS_DT']}s,  output={run_dir}")
    print("─" * 72)

    if dry_run:
        print("  [dry-run] would execute:", " ".join(cmd))
        print("  [dry-run] with overrides:")
        for k, v in env.items():
            if k.startswith("CABLE_"):
                print(f"    {k} = {v}")
        return None

    try:
        result = subprocess.run(
            cmd, env=env,
            capture_output=True, text=True,
            timeout=600,   # 10 min hard cap per run
        )
    except subprocess.TimeoutExpired:
        print("  ⚠ TIMEOUT after 600 s — marking as failed")
        return {"row": row_id, "stable": None, "instability_at_s": None,
                "error": "timeout"}
    except FileNotFoundError as e:
        print(f"  ⚠ subprocess failed: {e}")
        return {"row": row_id, "stable": None, "instability_at_s": None,
                "error": str(e)}

    # tail stdout for the operator
    if result.stdout:
        tail = "\n".join(result.stdout.splitlines()[-8:])
        print("  stdout tail:\n" + "\n".join("    " + l for l in tail.splitlines()))
    if result.returncode != 0:
        print(f"  ⚠ exit code {result.returncode}")
        if result.stderr:
            print("  stderr tail:")
            for l in result.stderr.splitlines()[-6:]:
                print("    " + l)

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"  ⚠ no summary.json at {summary_path}")
        return {"row": row_id, "stable": None, "instability_at_s": None,
                "error": "missing_summary"}
    with open(summary_path) as f:
        s = json.load(f)
    s["row"] = row_id
    return s


def build_comparison_table(results: list[dict]) -> None:
    """Write comparison.csv and comparison.md alongside the run dirs."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "comparison.csv"
    md_path  = OUTPUT_ROOT / "comparison.md"

    # Map row_id → (paper stability, paper sec-to-inst)
    paper = {r[0]: (r[4], r[5]) for r in GOVONI_TABLE_1}

    headers = [
        "Row", "E (MPa)", "i (links)",
        "Paper stable?", "Paper t_unstable (s)",
        "Ours stable?",  "Ours t_unstable (s)",
        "Ours max |ω| (deg/s)",
        "Matches paper?",
    ]

    rows_out = []
    for s in results:
        rid = s.get("row", "?")
        E_mpa = s.get("young_modulus_pa", 0.0) / 1e6
        i_lk  = s.get("num_links", "?")
        paper_stab, paper_t = paper.get(rid, ("?", None))
        our_stab  = s.get("stable")
        our_t     = s.get("instability_at_s")
        max_w     = s.get("max_omega_deg_per_s")
        # comparison verdict
        if our_stab is None:
            verdict = "n/a (run failed)"
        else:
            paper_is_stable = (paper_stab == "Stable")
            if our_stab == paper_is_stable:
                verdict = "✓ matches"
            elif our_stab and not paper_is_stable:
                verdict = "★ ours stable (paper diverges)"
            else:
                verdict = "✗ ours unstable (paper stable)"
        rows_out.append([
            rid,
            f"{E_mpa:.1f}",
            str(i_lk),
            paper_stab,
            "—" if paper_t is None else f"{paper_t:.2f}",
            "—" if our_stab is None else ("Stable" if our_stab else "Unstable"),
            "—" if our_t is None else f"{our_t:.3f}",
            "—" if max_w is None else f"{max_w:.2e}",
            verdict,
        ])

    # CSV
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows_out)

    # Markdown
    md_lines = []
    md_lines.append("# Govoni Table 1 — Reproduction Results\n")
    md_lines.append("Source paper: Govoni et al. 2025, arXiv:2504.13659.")
    md_lines.append("Sweep driver: `govoni_sweep.py` → `cable.py`.\n")
    md_lines.append("Our solver: PhysX TGS, 64 position iterations, Δt = 1/240 s.")
    md_lines.append("Their solver: explicit MSD integration at Δt = 5e-6 s (or 1e-7 s for row 4).\n")
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows_out:
        md_lines.append("| " + " | ".join(r) + " |")
    md_lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print()
    print("=" * 72)
    print("Comparison written:")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    print("=" * 72)
    print()
    # Also print the markdown table to stdout for convenience
    for line in md_lines:
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="",
                    help="Comma-separated row IDs to run, e.g. '1,3'. "
                         "Default: all rows.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would run, don't actually run.")
    args = ap.parse_args()

    if not CABLE_V2.exists():
        print(f"ERROR: cannot find cable.py at {CABLE_V2}")
        sys.exit(1)

    rows_filter = {x.strip() for x in args.rows.split(",") if x.strip()}
    table = GOVONI_TABLE_1 if not rows_filter else \
            [r for r in GOVONI_TABLE_1 if r[0] in rows_filter]
    if not table:
        print(f"ERROR: no rows match --rows={args.rows}")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(table)} configuration(s) from Govoni Table 1")
    print(f"Output root: {OUTPUT_ROOT}\n")

    results = []
    for (rid, E_mpa, i_disc, dt_g, _, _) in table:
        s = run_one(rid, E_mpa, i_disc, dt_g, args.dry_run)
        if s is not None:
            results.append(s)

    if not args.dry_run and results:
        build_comparison_table(results)


if __name__ == "__main__":
    main()
