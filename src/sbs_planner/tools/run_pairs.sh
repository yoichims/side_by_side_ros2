#!/usr/bin/env bash
# Both arms on the pair-cone layouts: the runtime corridor measurement on, and
# the planner as it was before it.
#
# Everything the earlier sweeps measured was on single-cone layouts, which
# understated every obstacle: an obstacle here is a *pair* of cones spanning
# about 0.83 m, and the 0.23 m between them takes no vehicle. So the control
# has to be re-run too -- the obstacles changed, not just the planner.
#
# Neither arm touches the lane profile, so there is no trap to restore and no
# `derive_lane_offsets` in the loop. Both run against the shipped profile,
# which is what happens when cones move and nobody regenerates -- the honest
# case, and the one [[derived-vs-stale-is-a-null]] showed is the only one the
# offsets can distinguish anyway.
set -u
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WS"
OUT="${OUT:-pair_runs}"
DIRS="${DIRS:-ccw cw}"
DUR="${DUR:-170}"
mkdir -p "$OUT"

for layout in layouts/layout_*.yaml; do
  name=$(basename "$layout" .yaml)
  for arm in base room; do
    if [ "$arm" = base ]; then
      FLAGS="runtime_room:=false w_centre:=0.0"
    else
      FLAGS="runtime_room:=true w_centre:=0.0"
    fi
    for d in $DIRS; do
      tag="${name}_${arm}_${d}"
      timeout "$DUR" ./in_container.sh ros2 launch sbs_bringup joint_planning.launch.py \
        direction:="$d" objects:="$layout" rviz:=false metrics:=true $FLAGS \
        metrics_file:="/tmp/${tag}.csv" > "$OUT/${tag}.log" 2>&1 < /dev/null
      sleep 4
      docker cp "sbs_devel_container:/tmp/${tag}.csv" "$OUT/" >/dev/null 2>&1
      rows=$(grep -vc '^#' "$OUT/${tag}.csv" 2>/dev/null || echo 0)
      echo "${tag}: rows ${rows} died $(grep -c 'process has died' "$OUT/${tag}.log")"
    done
  done
done
echo "sweep complete"
