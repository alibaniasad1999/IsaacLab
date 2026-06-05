#!/bin/bash
# =============================================================================
# run_all.sh -- Run all cable experiments and generate the report PDF.
#
# Prerequisites:
#   - Isaac Sim conda environment (env_isaaclab) activated
#   - Python 3 with matplotlib, pandas, numpy installed
#   - pdflatex installed
#
# Usage:
#   cd scripts/cable_simulation
#   chmod +x run_all.sh
#   ./run_all.sh
#
# What it does:
#   1. Capsule-chain hanging-kick (200 links, 10s, video)
#   2. Capsule-chain Govoni Table 1 sweep
#   3. Deformable body hanging-kick
#   4. Deformable body stability test
#   5. Two-robot manipulation test (both methods)
#   6. Method comparison table
#   7. Generate report + slide figures
#   8. Compile report + slides PDF
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Cable Simulation -- Full Run"
echo "=============================================="
echo ""

# ------------------------------------------------------------------
# Step 0: Archive any old results and start clean
# ------------------------------------------------------------------
echo "[0/8] Clearing old results..."
if [ -d cable_output ]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    mv cable_output "cable_output_archived_${STAMP}"
    echo "      Old data archived -> cable_output_archived_${STAMP}/"
fi
mkdir -p cable_output
# Also clear stale figures so nothing old leaks into the PDFs
rm -f report/figures/*.pdf
echo "      Fresh cable_output/ and cleared report/figures/"
echo ""

# ------------------------------------------------------------------
# Step 1: Capsule-chain hanging-kick (visual demo)
# ------------------------------------------------------------------
echo "[1/8] Capsule-chain: hanging-kick (200 links, 10s, video)..."
python cable.py
echo "      Done -> cable_output/hanging_kick/"
echo ""

# ------------------------------------------------------------------
# Step 2: Capsule-chain Govoni sweep
# ------------------------------------------------------------------
echo "[2/8] Capsule-chain: Govoni Table 1 sweep (5 configs)..."
python govoni_sweep.py
echo "      Done -> cable_output/govoni_sweep/"
echo ""

# ------------------------------------------------------------------
# Step 3: Deformable body hanging-kick
# ------------------------------------------------------------------
echo "[3/8] Deformable body: hanging-kick (10s)..."
python cable_deformable.py
echo "      Done -> cable_output/deformable_hanging_kick/"
echo ""

# ------------------------------------------------------------------
# Step 4: Deformable body stability test
# ------------------------------------------------------------------
echo "[4/8] Deformable body: both-ends-fixed stability test..."
CABLE_MODE=both_ends_fixed CABLE_HEADLESS=1 CABLE_RECORD=0 \
    CABLE_MAX_TIME=2.0 \
    CABLE_OUTPUT_DIR="$SCRIPT_DIR/cable_output/deformable_both_ends_fixed" \
    python cable_deformable.py
echo "      Done -> cable_output/deformable_both_ends_fixed/"
echo ""

# ------------------------------------------------------------------
# Step 5: Two-robot manipulation test (both methods)
# ------------------------------------------------------------------
echo "[5/8] Two-robot test: capsule-chain..."
CABLE_METHOD=capsule python cable_two_robots.py
echo "      Two-robot test: deformable..."
CABLE_METHOD=deformable python cable_two_robots.py
echo "      Done -> cable_output/two_robots_*/"
echo ""

# ------------------------------------------------------------------
# Step 6: Build method-comparison table
# ------------------------------------------------------------------
echo "[6/8] Building method comparison table..."
python3 compare_methods.py
echo "      Done -> cable_output/method_comparison/"
echo ""

# ------------------------------------------------------------------
# Step 7: Generate figures (report + slides)
# ------------------------------------------------------------------
echo "[7/8] Generating report + slide figures..."
python3 report/generate_plots.py
python3 report/generate_slide_figures.py
echo ""

# ------------------------------------------------------------------
# Step 8: Compile LaTeX report + slides
# ------------------------------------------------------------------
echo "[8/8] Compiling report + slides PDF..."
cd report
pdflatex -interaction=nonstopmode cable_report.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode cable_report.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode cable_slides.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode cable_slides.tex > /dev/null 2>&1
cd ..
echo "      Done."

echo ""
echo "=============================================="
echo "  All done!"
echo "=============================================="
echo ""
echo "  Report:            report/cable_report.pdf"
echo "  Capsule video:     cable_output/hanging_kick/cable_simulation.mp4"
echo "  Govoni sweep:      cable_output/govoni_sweep/comparison.md"
echo "  Deformable kick:   cable_output/deformable_hanging_kick/summary.json"
echo "  Deformable fixed:  cable_output/deformable_both_ends_fixed/summary.json"
echo ""
