#!/usr/bin/env bash
# Our runs in the two conditions the drivers were recorded in, for the
# distribution comparison.
#
# The corpus holds them separately -- a clear aisle and an aisle with cones --
# and the behaviours differ enough that pooling them makes a bimodal
# population look like error. So we reproduce both: `objects:=default` is the
# surveyed cones the drivers actually met, `objects:=none` the same route with
# nothing in it.
set -u
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WS"
OUT="${OUT:-condition_runs}"
REPEATS="${REPEATS:-3}"
DUR="${DUR:-170}"
# Extra launch arguments, so a configuration can be measured against the
# drivers without editing this file. e.g. EXTRA="sbs_stagger:=0.60".
EXTRA="${EXTRA:-}"
mkdir -p "$OUT"

for r in $(seq 1 "$REPEATS"); do
  for spec in "obstacle default" "free none"; do
    set -- $spec
    tag_c=$1; objects=$2
    for d in ccw cw; do
      tag="${tag_c}_${d}_r${r}"
      [ -s "$OUT/${tag}.csv" ] && continue
      timeout "$DUR" ./in_container.sh ros2 launch sbs_bringup joint_planning.launch.py \
        direction:="$d" objects:="$objects" rviz:=false metrics:=true $EXTRA \
        metrics_file:="/tmp/${tag}.csv" > "$OUT/${tag}.log" 2>&1 < /dev/null
      sleep 4
      docker cp "sbs_devel_container:/tmp/${tag}.csv" "$OUT/" >/dev/null 2>&1
      rows=$(grep -vc '^#' "$OUT/${tag}.csv" 2>/dev/null || echo 0)
      echo "${tag}: rows ${rows} died $(grep -c 'process has died' "$OUT/${tag}.log")"
    done
  done
done
echo "conditions complete"
