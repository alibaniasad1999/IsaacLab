#!/usr/bin/env python3
"""
Comparison sweep — runs each Govoni Table 1 config TWICE:
  1. Current method (material-driven K = EI/L, derived damping)
  2. Legacy method  (v1: K = 0, damping = 0.05, cone = 8°, twist = 5.8°)

Produces:
    cable_output/comparison_sweep/comparison.csv
    cable_output/comparison_sweep/comparison.md
    cable_output/comparison_sweep/current_run_<N>/summary.json
    cable_output/comparison_sweep/legacy_run_<N>/summary.json

Usage:
    python comparison_sweep.py
    python comparison_sweep.py --dry-run
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
CABLE_PY    = SCRIPT_DIR / "cable.py"
OUTPUT_ROOT = SCRIPT_DIR / "cable_output" / "comparison_sweep"

GOVONI_TABLE_1 = [
    ("1",  12.6,   10, "Stable",   None),
    ("2a", 526.0,  10, "Stable",   None),
    ("2b", 1002.0,  6, "Stable",   None),
    ("3",  1002.6, 10, "Unstable", 0.4),
    ("4",  1002.6, 10, "Stable",   None),
]

COMMON_ENV = {
    "CABLE_MODE":       "both_ends_fixed",
    "CABLE_HEADLESS":   "1",
    "CABLE_RECORD":     "0",
    "CABLE_LENGTH":     "0.5",
    "CABLE_RADIUS":     "1.5e-3",
    "CABLE_MASS":       "0.05",
    "CABLE_NU":         "0.5",
    "CABLE_ZETA":       "0.2",
    "CABLE_PHYSICS_DT": str(1.0 / 240.0),
    "CABLE_RENDER_DT":  str(1.0 / 60.0),
    "CABLE_SETTLE":     "0.5",
    "CABLE_STEP_DISP":  "5.0e-3",
    "CABLE_MAX_TIME":   "2.0",
}


def run_one(row_id: str, E_MPa: float, i_disc: int,
            legacy: bool, dry_run: bool) -> dict | None:
    tag = "legacy" if legacy else "current"
    run_dir = OUTPUT_ROOT / f"{tag}_run_{row_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["CABLE_E"]          = f"{E_MPa * 1e6:.6e}"
    env["CABLE_NUM_LINKS"]  = str(i_disc)
    env["CABLE_OUTPUT_DIR"] = str(run_dir)
    env["CABLE_LEGACY"]     = "1" if legacy else "0"

    cmd = [sys.executable, str(CABLE_PY)]
    print(f"  [{tag:>7}] Row {row_id}: E={E_MPa} MPa, i={i_disc}")

    if dry_run:
        print(f"           [dry-run] would execute: {' '.join(cmd)}")
        return None

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"           TIMEOUT")
        return {"row": row_id, "method": tag, "stable": None, "error": "timeout"}
    except FileNotFoundError as e:
        print(f"           FAILED: {e}")
        return {"row": row_id, "method": tag, "stable": None, "error": str(e)}

    if result.returncode != 0:
        print(f"           exit code {result.returncode}")

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"row": row_id, "method": tag, "stable": None, "error": "missing_summary"}
    with open(summary_path) as f:
        s = json.load(f)
    s["row"] = row_id
    s["method"] = tag
    return s


def build_table(results: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    paper = {r[0]: (r[3], r[4]) for r in GOVONI_TABLE_1}

    headers = [
        "Row", "E (MPa)", "i",
        "Paper", "Paper t_inst",
        "Current stable?", "Current max|ω|",
        "Legacy stable?", "Legacy max|ω|",
    ]

    current = {r["row"]: r for r in results if r.get("method") == "current"}
    legacy  = {r["row"]: r for r in results if r.get("method") == "legacy"}

    rows_out = []
    for rid, E_mpa, i_lk, paper_stab, paper_t in GOVONI_TABLE_1:
        c = current.get(rid, {})
        l = legacy.get(rid, {})

        def fmt_stable(s): return "—" if s is None else ("Stable" if s else "Unstable")
        def fmt_omega(s): return "—" if s is None else f"{s:.0f}"

        rows_out.append([
            rid,
            f"{E_mpa:.1f}",
            str(i_lk),
            paper_stab,
            "—" if paper_t is None else f"{paper_t:.2f}",
            fmt_stable(c.get("stable")),
            fmt_omega(c.get("max_omega_deg_per_s")),
            fmt_stable(l.get("stable")),
            fmt_omega(l.get("max_omega_deg_per_s")),
        ])

    with open(OUTPUT_ROOT / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows_out)

    md_lines = [
        "# Current vs Legacy — Govoni Table 1 Comparison\n",
        "**Current**: material-driven (K = EI/L, derived damping, cone 30°, twist free)",
        "**Legacy**: v1 hand-tuned (K = 0, damping = 0.05, cone 8°, twist 5.8°)\n",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows_out:
        md_lines.append("| " + " | ".join(r) + " |")
    md_lines.append("")

    with open(OUTPUT_ROOT / "comparison.md", "w") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 72)
    print("Comparison table written:")
    print(f"  {OUTPUT_ROOT / 'comparison.csv'}")
    print(f"  {OUTPUT_ROOT / 'comparison.md'}")
    print("=" * 72 + "\n")
    for line in md_lines:
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rows", default="")
    args = ap.parse_args()

    rows_filter = {x.strip() for x in args.rows.split(",") if x.strip()}
    table = GOVONI_TABLE_1 if not rows_filter else \
            [r for r in GOVONI_TABLE_1 if r[0] in rows_filter]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(table)} configs × 2 methods = {len(table)*2} simulations")
    print(f"Output: {OUTPUT_ROOT}\n")

    results = []
    for rid, E_mpa, i_disc, _, _ in table:
        print(f"── Row {rid} ──")
        for legacy in (False, True):
            s = run_one(rid, E_mpa, i_disc, legacy, args.dry_run)
            if s is not None:
                results.append(s)
        print()

    if not args.dry_run and results:
        build_table(results)


if __name__ == "__main__":
    main()
