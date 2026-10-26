#!/usr/bin/env bash
# The recovery tolerance, with enough repeats to tell it from noise.
#
# Every sweep before this ran one pass per layout, and one pass cannot carry a
# claim here: the same layout under the same configuration has given 1 mm and
# 3 mm of wall contact, and 0.27 and 0.43 of progress. The runs are not
# deterministic -- ROS timing varies -- so repeating a configuration samples
# the distribution rather than repeating a trajectory.
#
# Four layouts rather than eight, chosen as the tightest that still clear the
# 0.30 m passable band the surveyed environment sets, so the file manoeuvre is
# exercised hard and the geometry is still something the drivers could have
# met. Narrower coverage, trustworthy numbers.
set -u
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WS"
OUT="${OUT:-repeat_runs}"
LAYOUTS="${LAYOUTS:-layout_02 layout_05 layout_07 layout_08}"
DIRS="${DIRS:-ccw cw}"
REPEATS="${REPEATS:-3}"
DUR="${DUR:-170}"
# Which parameter the two arms differ in. Everything else is held fixed, which
# is the whole point: a comparison where two things moved answers nothing, and
# this sweep exists because one that bundled a measurement change with a layout
# change could not be read.
ARM_FLAG="${ARM_FLAG:-escape_tolerance}"
ARM_ON="${ARM_ON:-0.025}"
ARM_OFF="${ARM_OFF:-0.0}"
# Held fixed in *both* arms, so a second change can be carried without
# becoming a second variable. e.g. EXTRA_ARGS="sbs_stagger:=0.40".
EXTRA_ARGS="${EXTRA_ARGS:-}"
mkdir -p "$OUT"

for r in $(seq 1 "$REPEATS"); do
  for L in $LAYOUTS; do
    for arm in fix nofix; do
      [ "$arm" = fix ] && VAL="$ARM_ON" || VAL="$ARM_OFF"
      for d in $DIRS; do
        tag="${L}_${arm}_${d}_r${r}"
        [ -s "$OUT/${tag}.csv" ] && continue
        timeout "$DUR" ./in_container.sh ros2 launch sbs_bringup joint_planning.launch.py \
          direction:="$d" objects:="layouts/${L}.yaml" rviz:=false metrics:=true \
          "${ARM_FLAG}:=$VAL" $EXTRA_ARGS metrics_file:="/tmp/${tag}.csv" \
          > "$OUT/${tag}.log" 2>&1 < /dev/null
        sleep 4
        docker cp "sbs_devel_container:/tmp/${tag}.csv" "$OUT/" >/dev/null 2>&1
        rows=$(grep -vc '^#' "$OUT/${tag}.csv" 2>/dev/null || echo 0)
        echo "${tag}: rows ${rows} died $(grep -c 'process has died' "$OUT/${tag}.log")"
      done
    done
  done
  echo "--- repeat ${r} of ${REPEATS} complete"
done
echo "sweep complete"
