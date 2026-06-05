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
#   1. Hanging-kick demo (200 links, rubber, 10s, with video)
#   2. Govoni Table 1 sweep (5 configs, current method only)
#   3. Current vs Legacy comparison sweep (5 configs × 2 methods = 10 runs)
#   4. Generate all report figures
#   5. Compile the LaTeX report
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Cable Simulation — Full Run"
echo "=============================================="
echo ""

# ------------------------------------------------------------------
# Step 1: Hanging-kick experiment
# ------------------------------------------------------------------
echo "[1/5] Running hanging-kick experiment (200 links, 10s, with video)..."
python cable.py
echo "      Done → cable_output/hanging_kick/"
echo ""

# ------------------------------------------------------------------
# Step 2: Govoni Table 1 sweep (current method)
# ------------------------------------------------------------------
echo "[2/5] Running Govoni Table 1 sweep (5 configs, headless)..."
python govoni_sweep.py
echo "      Done → cable_output/govoni_sweep/"
echo ""

# ------------------------------------------------------------------
# Step 3: Current vs Legacy comparison sweep
# ------------------------------------------------------------------
echo "[3/5] Running current vs legacy comparison (10 runs, headless)..."
python comparison_sweep.py
echo "      Done → cable_output/comparison_sweep/"
echo ""

# ------------------------------------------------------------------
# Step 4: Generate report figures
# ------------------------------------------------------------------
echo "[4/5] Generating report figures..."
python3 report/generate_plots.py
echo ""

# ------------------------------------------------------------------
# Step 5: Compile LaTeX report
# ------------------------------------------------------------------
echo "[5/5] Compiling report PDF..."
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
echo "  Report:     report/cable_report.pdf"
echo "  Video:      cable_output/hanging_kick/cable_simulation.mp4"
echo "  Govoni:     cable_output/govoni_sweep/comparison.md"
echo "  Legacy vs:  cable_output/comparison_sweep/comparison.md"
echo ""
