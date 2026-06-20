#!/usr/bin/env bash
# =============================================================================
# run_all_and_capture.sh
# Runs all 6 cable simulations (3 methods x 2 scenarios), records a 10 s video
# and a t=5 s frame of each, lays them out under slides_output/, then builds the
# Beamer slide deck.
#
#   3 methods : capsule (cable.py) | warp-Cosserat (cable_warp.py) | FEM (cable_fem.py)
#   2 scenes  : cable only         | cable connected to a Franka robot (cable_two_robots.py)
#
# Requirements: env_isaaclab conda env, a DISPLAY (GUI render is needed to record),
# ffmpeg, pdflatex.  Run from anywhere:
#     bash scripts/cable_simulation/run_all_and_capture.sh            # all 6
#     bash scripts/cable_simulation/run_all_and_capture.sh cosserat   # just that method (both scenes)
#     bash scripts/cable_simulation/run_all_and_capture.sh capsule/robot   # just one clip
#     bash scripts/cable_simulation/run_all_and_capture.sh fem        # any substring of the folder name
# The optional argument FILTERS which sims run (matches the output-folder name);
# the slides are still rebuilt with whatever frames exist.
# =============================================================================
set -u
FILTER="${1:-}"     # optional: only run sims whose folder name contains this

# ---- paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/slides_output"   # captured sim media (frames, csv, json, videos/)
SLIDES="$SCRIPT_DIR/slides"       # the LaTeX deck (cable_slides.tex) + its built PDF
PER_SIM_SECONDS=10            # length of each recorded clip
JOB_TIMEOUT=900              # hard cap per sim (startup + run + shutdown), seconds

# ---- environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export DISPLAY="${DISPLAY:-:1}"     # GUI render target (recording needs a display)

mkdir -p "$SLIDES"

# Common env that makes every script render a window, record, and stop at 10 s.
COMMON="CABLE_HEADLESS=0 CABLE_RECORD=1 CABLE_INTERACTIVE=0 CABLE_MAX_TIME=$PER_SIM_SECONDS"

# ---- the 6 jobs:  out_folder | python script | extra env | video_name ----
# frame.png + logs stay in the per-method out_folder (the slides read frame.png
# from there); the final clip is renamed to <video_name>.mp4 and collected into
# slides_output/videos/ so all six are in one flat folder, ready to upload.
JOBS=(
  "capsule/cable_only|cable.py|CABLE_MODE=hanging_kick|1_capsule_cable"
  "capsule/robot|cable_two_robots.py|CABLE_METHOD=capsule|1_capsule_robot"
  "cosserat_warp/cable_only|cable_warp.py|ROD_OBSTACLE=1|2_cosserat_cable"
  "cosserat_warp/robot|cable_two_robots.py|CABLE_METHOD=warp|2_cosserat_robot"
  "fem/cable_only|cable_fem_contact.py||3_fem_cable"
  "fem/robot|cable_two_robots.py|CABLE_METHOD=deformable|3_fem_robot"
)

VIDEOS="$OUT/videos"   # flat folder collecting all six final clips

run_one () {
  local folder="$1" script="$2" extra="$3" vidname="$4"
  local dir="$OUT/$folder"
  mkdir -p "$dir" "$VIDEOS"
  echo "============================================================"
  echo ">>> $folder   ($script  $extra)"
  echo "============================================================"
  # Each script writes its mp4 + logs into CABLE_OUTPUT_DIR=$dir.
  env $COMMON $extra CABLE_OUTPUT_DIR="$dir" \
      timeout "$JOB_TIMEOUT" python "$SCRIPT_DIR/$script" \
      > "$dir/run.log" 2>&1
  echo "    (exit $?) log: $dir/run.log"

  # Find whatever .mp4 the script produced.
  local mp4
  mp4="$(find "$dir" -maxdepth 1 -name '*.mp4' -printf '%T@ %p\n' 2>/dev/null \
         | sort -nr | head -1 | cut -d' ' -f2-)"
  if [ -n "$mp4" ] && [ -f "$mp4" ]; then
    # Grab a key frame at t=5 s (fall back to the last frame for short clips).
    ffmpeg -y -ss 5 -i "$mp4" -frames:v 1 "$dir/frame.png" >/dev/null 2>&1 \
      || ffmpeg -y -sseof -1 -i "$mp4" -frames:v 1 "$dir/frame.png" >/dev/null 2>&1
    # Collect the clip into the flat videos/ folder under a clear name.
    mv -f "$mp4" "$VIDEOS/$vidname.mp4"
    echo "    OK -> $VIDEOS/$vidname.mp4  +  $dir/frame.png"
  else
    echo "    !! no .mp4 produced (see run.log). Slides will show a placeholder."
  fi
}

_ran=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r folder script extra vidname <<< "$job"
  if [ -n "$FILTER" ] && [[ "$folder" != *"$FILTER"* ]]; then
    continue        # skip sims that don't match the filter
  fi
  run_one "$folder" "$script" "$extra" "$vidname"
  _ran=$((_ran+1))
done
if [ "$_ran" -eq 0 ]; then
  echo "!! no sims matched filter '$FILTER'. Folders: capsule/cable_only capsule/robot"
  echo "   cosserat_warp/cable_only cosserat_warp/robot fem/cable_only fem/robot"
fi

# ---- build the slides ----
echo "============================================================"
echo ">>> Building slides"
echo "============================================================"
# Single source of truth: slides/cable_slides.tex. Build it in place (frames
# resolve via ../slides_output/<method>/...); PDF + aux land in slides/.
cd "$SLIDES"
pdflatex -interaction=nonstopmode cable_slides.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode cable_slides.tex >/dev/null 2>&1

echo
echo "DONE."
echo "  Slides : $SLIDES/cable_slides.pdf"
echo "  Videos : $VIDEOS/  (all six clips, ready to upload)"
echo "  Frames : $OUT/{capsule,cosserat_warp,fem}/*/frame.png"
