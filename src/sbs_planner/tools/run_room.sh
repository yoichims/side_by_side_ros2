#!/usr/bin/env bash
# Runtime corridor measurement, on, against the shipped profile.
#
# The control arm already exists: layout_runs/*_stale_* was run with the same
# shipped profile and without `runtime_room` or `w_centre`, so this sweep is
# the treatment arm of that comparison and nothing here touches the profile.
# That also means no trap and no restore -- the previous sweep hung after its
# last run with those in place.
set -u
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WS"
OUT="${OUT:-room_runs}"
DIRS="${DIRS:-ccw cw}"
DUR="${DUR:-170}"
mkdir -p "$OUT"

for layout in layouts/layout_*.yaml; do
  name=$(basename "$layout" .yaml)
  for d in $DIRS; do
    tag="${name}_room_${d}"
    timeout "$DUR" ./in_container.sh ros2 launch sbs_bringup joint_planning.launch.py \
      direction:="$d" objects:="$layout" rviz:=false metrics:=true \
      metrics_file:="/tmp/${tag}.csv" > "$OUT/${tag}.log" 2>&1 < /dev/null
    sleep 4
    docker cp "sbs_devel_container:/tmp/${tag}.csv" "$OUT/" >/dev/null 2>&1
    rows=$(grep -vc '^#' "$OUT/${tag}.csv" 2>/dev/null || echo 0)
    echo "${tag}: rows ${rows} died $(grep -c 'process has died' "$OUT/${tag}.log")"
  done
done
echo "sweep complete"
