#!/bin/bash
# =============================================================================
# run_all.sh -- Run the cable experiments and build the comparison slides.
#
# Scope: two cable models on the same PUR material, compared head-to-head.
#   - Base model:       rigid capsule-chain  (cable.py)
#   - Deformable model: FEM deformable body  (cable_deformable.py)
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
#   1. Capsule-chain (base) hanging-kick   (200 links, 10s, video)
#   2. Deformable body hanging-kick
#   3. Deformable body both-ends-fixed stability test
#   4. Two-robot manipulation test (both models)
#   5. Method comparison table
#   6. Generate slide figures
#   7. Compile comparison slides PDF
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
echo "[0/7] Clearing old results..."
if [ -d cable_output ]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    mv cable_output "cable_output_archived_${STAMP}"
    echo "      Old data archived -> cable_output_archived_${STAMP}/"
fi
mkdir -p cable_output
rm -f report/figures/*.pdf
echo "      Fresh cable_output/ and cleared report/figures/"
echo ""

# ------------------------------------------------------------------
# Step 1: Capsule-chain (base model) hanging-kick
# ------------------------------------------------------------------
echo "[1/7] Capsule-chain (base): hanging-kick (200 links, 10s, video)..."
python cable.py
echo "      Done -> cable_output/hanging_kick/"
echo ""

# ------------------------------------------------------------------
# Step 2: Deformable body hanging-kick
# ------------------------------------------------------------------
echo "[2/7] Deformable body: hanging-kick (10s)..."
python cable_deformable.py
echo "      Done -> cable_output/deformable_hanging_kick/"
echo ""

# ------------------------------------------------------------------
# Step 3: Deformable body stability test
# ------------------------------------------------------------------
echo "[3/7] Deformable body: both-ends-fixed stability test..."
CABLE_MODE=both_ends_fixed CABLE_HEADLESS=1 CABLE_RECORD=0 \
    CABLE_MAX_TIME=2.0 \
    CABLE_OUTPUT_DIR="$SCRIPT_DIR/cable_output/deformable_both_ends_fixed" \
    python cable_deformable.py
echo "      Done -> cable_output/deformable_both_ends_fixed/"
echo ""

# ------------------------------------------------------------------
# Step 4: Two-robot manipulation test (both models)
# ------------------------------------------------------------------
echo "[4/7] Two-robot test: capsule-chain..."
CABLE_METHOD=capsule python cable_two_robots.py
echo "      Two-robot test: deformable..."
CABLE_METHOD=deformable python cable_two_robots.py
echo "      Done -> cable_output/two_robots_*/"
echo ""

# ------------------------------------------------------------------
# Step 5: Build method-comparison table
# ------------------------------------------------------------------
echo "[5/7] Building method comparison table..."
python3 compare_methods.py
echo "      Done -> cable_output/method_comparison/"
echo ""

# ------------------------------------------------------------------
# Step 6: Generate slide figures
# ------------------------------------------------------------------
echo "[6/7] Generating slide figures..."
python3 report/generate_slide_figures.py
echo ""

# ------------------------------------------------------------------
# Step 7: Compile comparison slides PDF
# ------------------------------------------------------------------
echo "[7/7] Compiling comparison slides PDF..."
cd report
pdflatex -interaction=nonstopmode cable_slides.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode cable_slides.tex > /dev/null 2>&1
cd ..
echo "      Done."

echo ""
echo "=============================================="
echo "  All done!"
echo "=============================================="
echo ""
echo "  Slides:            report/cable_slides.pdf"
echo "  Capsule video:     cable_output/hanging_kick/cable_simulation.mp4"
echo "  Deformable kick:   cable_output/deformable_hanging_kick/summary.json"
echo "  Deformable fixed:  cable_output/deformable_both_ends_fixed/summary.json"
echo "  Comparison table:  cable_output/method_comparison/comparison.md"
echo ""
