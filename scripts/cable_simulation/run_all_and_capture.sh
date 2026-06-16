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
#     bash scripts/cable_simulation/run_all_and_capture.sh approach2  # just method 2 (both scenes)
#     bash scripts/cable_simulation/run_all_and_capture.sh approach2_robot   # just one clip
#     bash scripts/cable_simulation/run_all_and_capture.sh warp       # any substring of the folder name
# The optional argument FILTERS which sims run (matches the output-folder name);
# the slides are still rebuilt with whatever frames exist.
# =============================================================================
set -u
FILTER="${1:-}"     # optional: only run sims whose folder name contains this

# ---- paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/slides_output"
SLIDES="$OUT/slides"
PER_SIM_SECONDS=10            # length of each recorded clip
JOB_TIMEOUT=900              # hard cap per sim (startup + run + shutdown), seconds

# ---- environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export DISPLAY="${DISPLAY:-:1}"     # GUI render target (recording needs a display)

mkdir -p "$SLIDES"

# Common env that makes every script render a window, record, and stop at 10 s.
COMMON="CABLE_HEADLESS=0 CABLE_RECORD=1 CABLE_INTERACTIVE=0 CABLE_MAX_TIME=$PER_SIM_SECONDS"

# ---- the 6 jobs:  out_folder | python script | extra env ----
JOBS=(
  "approach1_cable_only|cable.py|CABLE_MODE=hanging_kick"
  "approach1_robot|cable_two_robots.py|CABLE_METHOD=capsule"
  "approach2_cable_only|cable_warp.py|ROD_OBSTACLE=1"
  "approach2_robot|cable_two_robots.py|CABLE_METHOD=warp"
  "approach3_cable_only|cable_fem_contact.py|"
  "approach3_robot|cable_two_robots.py|CABLE_METHOD=deformable"
)

run_one () {
  local folder="$1" script="$2" extra="$3"
  local dir="$OUT/$folder"
  mkdir -p "$dir"
  echo "============================================================"
  echo ">>> $folder   ($script  $extra)"
  echo "============================================================"
  # Each script writes its mp4 + logs into CABLE_OUTPUT_DIR=$dir.
  env $COMMON $extra CABLE_OUTPUT_DIR="$dir" \
      timeout "$JOB_TIMEOUT" python "$SCRIPT_DIR/$script" \
      > "$dir/run.log" 2>&1
  echo "    (exit $?) log: $dir/run.log"

  # Find whatever .mp4 the script produced and normalise it to video.mp4.
  local mp4
  mp4="$(find "$dir" -maxdepth 1 -name '*.mp4' -printf '%T@ %p\n' 2>/dev/null \
         | sort -nr | head -1 | cut -d' ' -f2-)"
  if [ -n "$mp4" ] && [ -f "$mp4" ]; then
    [ "$(basename "$mp4")" != "video.mp4" ] && mv -f "$mp4" "$dir/video.mp4"
    # Grab a key frame at t=5 s (fall back to the last frame for short clips).
    ffmpeg -y -ss 5 -i "$dir/video.mp4" -frames:v 1 "$dir/frame.png" \
        >/dev/null 2>&1 \
      || ffmpeg -y -sseof -1 -i "$dir/video.mp4" -frames:v 1 "$dir/frame.png" \
        >/dev/null 2>&1
    echo "    OK -> $dir/video.mp4  +  frame.png"
  else
    echo "    !! no .mp4 produced (see run.log). Slides will show a placeholder."
  fi
}

_ran=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r folder script extra <<< "$job"
  if [ -n "$FILTER" ] && [[ "$folder" != *"$FILTER"* ]]; then
    continue        # skip sims that don't match the filter
  fi
  run_one "$folder" "$script" "$extra"
  _ran=$((_ran+1))
done
if [ "$_ran" -eq 0 ]; then
  echo "!! no sims matched filter '$FILTER'. Folders: approach1_cable_only approach1_robot"
  echo "   approach2_cable_only approach2_robot approach3_cable_only approach3_robot"
fi

# ---- build the slides ----
echo "============================================================"
echo ">>> Building slides"
echo "============================================================"
cp -f "$SCRIPT_DIR/cable_slides.tex" "$SLIDES/cable_slides.tex"
cd "$SLIDES"
pdflatex -interaction=nonstopmode cable_slides.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode cable_slides.tex >/dev/null 2>&1

echo
echo "DONE."
echo "  Slides : $SLIDES/cable_slides.pdf"
echo "  Videos : $OUT/approach*/video.mp4"
echo "  Frames : $OUT/approach*/frame.png"
