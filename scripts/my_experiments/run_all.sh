#!/bin/bash
# =============================================================================
# run_all.sh — Run all cable experiments and generate the report PDF.
#
# Prerequisites:
#   - Isaac Sim conda environment (env_isaaclab) activated
#   - Python 3 with matplotlib, pandas, numpy installed
#   - pdflatex installed
#
# Usage:
#   cd scripts/my_experiments
#   chmod +x run_all.sh
#   ./run_all.sh
#
# What it does:
#   1. Runs the hanging-kick demo (200 links, rubber, 10s, with video)
#   2. Runs the Govoni Table 1 sweep (5 configs, headless)
#   3. Generates all report figures from the simulation data
#   4. Compiles the LaTeX report
# =============================================================================

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Cable Simulation — Full Run"
echo "=============================================="
echo ""

# ------------------------------------------------------------------
# Step 1: Hanging-kick experiment (visual demo, 200 links, 10s)
# ------------------------------------------------------------------
echo "[1/4] Running hanging-kick experiment (200 links, 10s, with video)..."
echo "      This produces: cable_output/hanging_kick/"
echo ""

python cable.py

echo ""
echo "      Done. Output:"
echo "        - cable_output/hanging_kick/cable_simulation.mp4"
echo "        - cable_output/hanging_kick/frame_t*.png"
echo "        - cable_output/hanging_kick/trajectory.csv"
echo "        - cable_output/hanging_kick/summary.json"
echo ""

# ------------------------------------------------------------------
# Step 2: Govoni Table 1 sweep (5 configs, headless, ~10-30 min)
# ------------------------------------------------------------------
echo "[2/4] Running Govoni Table 1 sweep (5 configurations, headless)..."
echo "      This produces: cable_output/govoni_sweep/"
echo ""

python govoni_sweep.py

echo ""
echo "      Done. Output:"
echo "        - cable_output/govoni_sweep/comparison.md"
echo "        - cable_output/govoni_sweep/comparison.csv"
echo "        - cable_output/govoni_sweep/run_*/summary.json"
echo ""

# ------------------------------------------------------------------
# Step 3: Generate report figures from simulation data
# ------------------------------------------------------------------
echo "[3/4] Generating report figures..."
echo ""

python3 report/generate_plots.py

echo ""

# ------------------------------------------------------------------
# Step 4: Compile LaTeX report
# ------------------------------------------------------------------
echo "[4/4] Compiling report PDF..."
echo ""

cd report
pdflatex -interaction=nonstopmode cable_report.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode cable_report.tex > /dev/null 2>&1
cd ..

echo "      Done."
echo ""
echo "=============================================="
echo "  All done!"
echo "=============================================="
echo ""
echo "  Report:  report/cable_report.pdf"
echo "  Video:   cable_output/hanging_kick/cable_simulation.mp4"
echo "  Sweep:   cable_output/govoni_sweep/comparison.md"
echo ""
echo "  To view sweep results:"
echo "    cat cable_output/govoni_sweep/comparison.md"
echo ""
