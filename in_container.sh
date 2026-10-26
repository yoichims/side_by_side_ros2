#!/usr/bin/env bash
#
# Run one command inside the development container with the environment
# already sourced.
#
#   ./in_container.sh ros2 launch sbs_bringup joint_planning.launch.py rviz:=false
#   ./in_container.sh python3 -m pytest src/sbs_planner/test -q
#   ./in_container.sh ros2 topic echo /v1/objects --once
#   ./in_container.sh                       # no command: prints what is sourced
#
# join_container.sh gives an interactive shell, which is the wrong shape for a
# single check: sourcing ROS and the overlay by hand every time is enough
# friction that it gets skipped, and skipping it is expensive. The host cannot
# help -- it runs Ubuntu 26.04 with Python 3.13, while the container is Humble
# on 3.10, so there is no host rclpy and installing a *newer* ROS 2 would mean
# testing against a different distribution from the one the vehicles run.
#
# Nothing here is interactive: no -t, so output pipes and greps cleanly and it
# works from scripts. Use join_container.sh when you want a shell.

set -u

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
source utils

if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "container ${CONTAINER_NAME} does not exist; run ./run_container.sh" >&2
  exit 1
fi
if [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != true ]; then
  echo "container ${CONTAINER_NAME} is not running; run ./run_container.sh" >&2
  exit 1
fi

# The overlay may not exist yet on a fresh workspace, so sourcing it is
# optional; ROS itself is not.
# No `set -u` in here: ROS's own setup.bash reads unbound variables
# (AMENT_TRACE_SETUP_FILES among them) and aborts under it.
PREAMBLE="source /opt/ros/humble/setup.bash
if [ -f ${SBS_WS_CONTAINER}/install/setup.bash ]; then
  source ${SBS_WS_CONTAINER}/install/setup.bash
fi
cd ${SBS_WS_CONTAINER}"

if [ "$#" -eq 0 ]; then
  exec docker exec "${CONTAINER_NAME}" bash -lc "${PREAMBLE}
    echo \"ROS_DISTRO=\${ROS_DISTRO}  \$(python3 -V)  cwd=\$(pwd)\"
    echo \"packages: \$(ros2 pkg list 2>/dev/null | wc -l) available, workspace holds:\"
    ls -1 src 2>/dev/null | sed 's/^/  /'"
fi

# "$@" is expanded on this side and quoted for the remote shell, so arguments
# containing spaces survive.
#
# Reaped from this side when this script goes away.
#
# `docker exec` does not propagate the client's death: when this script is
# killed -- a `timeout`, a Ctrl-C, a closed terminal -- the container-side
# process keeps running, detached and invisible. A `ros2 launch` leaked that
# way keeps publishing on the same ROS domain, so the next run shares
# /vN/odom and /vN/cmd_vel with every earlier one, and the two vehicles are
# driven by several planners at once. Fifty-three orphaned nodes accumulated
# this way before it was noticed, and every run measured against them was
# meaningless.
#
# The run is tagged with a unique token in its environment, which every child
# inherits, so the whole tree can be found again by reading /proc/*/environ --
# process groups and parentage are both unreliable here, since launch spawns
# its nodes freely and an orphan is reparented to init. Killing on stdin EOF
# was tried first and is wrong: a script or a `timeout` calls this with stdin
# already closed, so the guard fired before the command had run.
TOKEN="sbs_$$_$(date +%s)"

reap() {
  docker exec "${CONTAINER_NAME}" bash -c '
    tagged() {
      for p in /proc/[0-9]*; do
        if grep -qz "SBS_RUN_TOKEN=$1" "$p/environ" 2>/dev/null; then
          echo "${p#/proc/}"
        fi
      done
    }
    pids=$(tagged "$1")
    [ -n "$pids" ] || exit 0
    kill -INT $pids 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      sleep 0.5
      [ -n "$(tagged "$1")" ] || exit 0
    done
    kill -KILL $(tagged "$1") 2>/dev/null' _ "${TOKEN}" >/dev/null 2>&1
}

# The token reaches the container in the environment, never on a command line,
# so the reaper cannot match itself.
trap 'reap; exit 130' INT
trap 'reap; exit 143' TERM
trap 'reap' EXIT

# "$@" is expanded on this side and quoted for the remote shell, so arguments
# containing spaces survive.
docker exec -e "SBS_RUN_TOKEN=${TOKEN}" "${CONTAINER_NAME}" bash -lc "${PREAMBLE}
exec \"\$@\"" _ "$@" &
child=$!
wait "${child}"
exit $?
