#!/usr/bin/env bash
# ============================================================================
# run_image.sh  --  one-shot cable workflow for a single IMAGE.
#
#   segment the cable (SAM)  ->  metric profile (CSV + plot)  ->  identify the
#   cable material (Gamma / EI / Young's modulus).
#
# All results go into  results/<image-name>/  so media/ stays inputs-only.
#
# USAGE
#   ./run_image.sh <image> [mass_g] [radius_mm] [length_m]
#
#   <image>     path to the photo (e.g. media/IMG_0501.JPG)
#   mass_g      cable mass in grams   (optional -> full identification: E, EI)
#   radius_mm   cable radius in mm    (optional -> Young's modulus E)
#   length_m    true cable length    (OPTIONAL -- by default the length is
#                                     MEASURED from the image; only pass this to
#                                     override the image-measured length)
#
# EXAMPLES
#   ./run_image.sh media/IMG_0501.JPG                 # length auto from image
#   ./run_image.sh media/IMG_0501.JPG 8.1 1.5         # + mass & radius -> E
#   ./run_image.sh media/IMG_0501.JPG 8.1 1.5 1.0     # also force length=1.0 m
# ============================================================================
set -euo pipefail

# --- resolve paths relative to THIS script so it works from any folder ------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"            # override with PYTHON=/path/to/python

IMAGE="${1:-}"
MASS_G="${2:-}"
RADIUS_MM="${3:-}"
LENGTH_M="${4:-}"          # optional override; default = length measured from image

if [[ -z "$IMAGE" ]]; then
  echo "usage: $0 <image> [mass_g] [radius_mm] [length_m]" >&2
  echo "  (length is measured from the image unless you pass length_m)" >&2
  exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "image not found: $IMAGE" >&2
  exit 1
fi

# --- results folder named after the image -----------------------------------
NAME="$(basename "${IMAGE%.*}")"            # IMG_0501
OUTDIR="$HERE/results/$NAME"
mkdir -p "$OUTDIR"
PROFILE="$OUTDIR/profile.csv"

echo "=================================================================="
echo "  image    : $IMAGE"
echo "  results  : $OUTDIR"
echo "=================================================================="

# --- STEP 1: segment + extract the metric profile ---------------------------
echo
echo ">>> STEP 1/2  segment the cable and extract its profile"
EXTRACT_ARGS=( photo --image "$IMAGE" --out "$PROFILE" )
# pass through any extra knobs you set as env vars (rotation, smoothing, fit).
[[ -n "${ROTATE:-}"  ]] && EXTRACT_ARGS+=( --rotate "$ROTATE" )
[[ -n "${SMOOTH:-}"  ]] && EXTRACT_ARGS+=( --smooth "$SMOOTH" )
[[ -n "${FIT:-}"     ]] && EXTRACT_ARGS+=( --fit "$FIT" )
"$PY" "$HERE/extract_cable_profile.py" "${EXTRACT_ARGS[@]}"

if [[ ! -f "$PROFILE" ]]; then
  echo "no profile produced -- aborting." >&2
  exit 1
fi

# --- STEP 2: identify the cable from its profile ----------------------------
echo
echo ">>> STEP 2/2  identify the cable material from the profile"
ID_ARGS=( --profile "$PROFILE" --overlay )
[[ -n "$MASS_G"    ]] && ID_ARGS+=( --mass-g   "$MASS_G" )
[[ -n "$LENGTH_M"  ]] && ID_ARGS+=( --length-m "$LENGTH_M" )
[[ -n "$RADIUS_MM" ]] && ID_ARGS+=( --radius-mm "$RADIUS_MM" )
# save the identification text report too.
"$PY" "$HERE/identify_cable.py" "${ID_ARGS[@]}" | tee "$OUTDIR/identification.txt"

echo
echo "=================================================================="
echo "  DONE. results in: $OUTDIR"
echo "    profile.csv         (x_m, z_m)"
echo "    profile.png         (extracted shape)"
echo "    profile_fit.png     (elastica fit overlay)"
echo "    identification.txt  (Gamma / EI / Young's modulus)"
echo "=================================================================="
