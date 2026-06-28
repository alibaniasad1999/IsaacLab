#!/usr/bin/env bash
# ============================================================================
# run_image.sh  --  extract a cable's 2-D profile from a single IMAGE.
#
#   segment the cable (SAM)  ->  metric profile (CSV + plot)
#
# The profile (x_m, z_m, in metres) is saved for later analysis. NOTE: a single
# STATIC hanging shape does NOT identify material stiffness (a stiff and a
# floppy cable of the same size hang almost identically) -- the profile is kept
# for other methods (e.g. dynamics / oscillation). See README.md.
#
# All results go into  results/<image-name>/  so media/ stays inputs-only.
#
# USAGE
#   ./run_image.sh <image>
#
# EXAMPLES
#   ./run_image.sh media/IMG_0501.JPG
#   ROTATE=90 SMOOTH=2 ./run_image.sh media/IMG_0501.JPG
# ============================================================================
set -euo pipefail

# --- resolve paths relative to THIS script so it works from any folder ------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"            # override with PYTHON=/path/to/python

IMAGE="${1:-}"
if [[ -z "$IMAGE" ]]; then
  echo "usage: $0 <image>" >&2
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

# --- segment + extract the metric profile -----------------------------------
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

echo
echo "=================================================================="
echo "  DONE. results in: $OUTDIR"
echo "    profile.csv   (x_m, z_m -- the cable shape in metres)"
echo "    profile.png   (extracted shape)"
echo "=================================================================="
