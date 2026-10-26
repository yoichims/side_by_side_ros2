#!/usr/bin/env bash
# Sweep the cone layouts, both ways: with the lane profile regenerated for the
# layout, and with the profile left as it is.
#
# The lane offsets and the side-by-side flag are derived offline from the map *and*
# one particular set of objects. Regenerating per layout is the best case --
# the planner is told where the cones are before it starts. Leaving the profile
# alone is the honest case, and the one that matters if cones are ever moved
# between runs. The difference between the two arms is what that offline
# knowledge is worth, which is the thing worth measuring before deciding to
# move the free-space reasoning to runtime.
set -u
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WS"
OUT="${OUT:-layout_runs}"
DIRS="${DIRS:-ccw cw}"
DUR="${DUR:-170}"
ENVDIR=environments/experiment
mkdir -p "$OUT"

# Keep the shipped profile so the sweep can restore it.
for d in ccw cw; do cp "$ENVDIR/centerlines/${d}_offsets.csv" "$OUT/.${d}_offsets.orig"; done
restore() { for d in ccw cw; do cp "$OUT/.${d}_offsets.orig" "$ENVDIR/centerlines/${d}_offsets.csv"; done; }
trap restore EXIT

for layout in layouts/layout_*.yaml; do
  name=$(basename "$layout" .yaml)
  for arm in derived stale; do
    if [ "$arm" = derived ]; then
      for d in $DIRS; do
        python3 src/sbs_planner/tools/derive_lane_offsets.py \
          "$ENVDIR/centerlines/${d}_centerline.csv" "$ENVDIR/map.yaml" \
          --objects "$layout" --write >/dev/null 2>&1
      done
    else
      restore
    fi
    for d in $DIRS; do
      tag="${name}_${arm}_${d}"
      timeout "$DUR" ./in_container.sh ros2 launch sbs_bringup joint_planning.launch.py \
        direction:="$d" objects:="$layout" rviz:=false metrics:=true \
        metrics_file:="/tmp/${tag}.csv" > "$OUT/${tag}.log" 2>&1
      sleep 4
      docker cp "sbs_devel_container:/tmp/${tag}.csv" "$OUT/" >/dev/null 2>&1
      rows=$(grep -vc '^#' "$OUT/${tag}.csv" 2>/dev/null || echo 0)
      echo "${tag}: rows ${rows} died $(grep -c 'process has died' "$OUT/${tag}.log")"
    done
  done
done
