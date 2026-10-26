#!/usr/bin/env bash
#
# Build the workspace and source the overlay.
#
#   source ./compile_and_source.sh                     build everything
#   source ./compile_and_source.sh --packages-select sbs_planner
#   source ./compile_and_source.sh --packages-up-to sbs_bringup
#
# Any extra arguments are passed straight through to colcon build.
#
# SOURCE this script, do not execute it. Running ./compile_and_source.sh builds
# correctly, but the overlay is then sourced in a subshell that immediately
# exits, so the shell you are typing in still sees the old install/ -- or none
# at all. The script says so rather than letting that pass silently.
#
# To run a single command in the container with the environment already
# sourced -- a test, a launch, a topic echo -- use ./in_container.sh instead of
# sourcing this by hand in an interactive shell.
#
# --symlink-install is on by default. It matters here: the utility weights in
# src/sbs_motion/config/util.yaml, the Frenet sampling in frene.yaml, the launch
# files and the RViz config are all installed as symlinks, so editing one and
# relaunching picks up the change with no rebuild. That is most of the loop when
# tuning the planner's parameters.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo -e "\e[33m[WARN]: run this with 'source', otherwise the overlay is sourced\e[0m"
  echo -e "\e[33m        in a subshell and your shell keeps the old environment:\e[0m"
  echo -e "\e[33m          source ./compile_and_source.sh\e[0m"
  echo
fi

_sbs_ws="${SBS_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ ! -d ${_sbs_ws}/src ]]; then
  echo -e "\e[31m[ERROR]: no src/ under ${_sbs_ws}; is this the workspace root?\e[0m"
else
  (
    cd "${_sbs_ws}" || exit 1
    colcon build --symlink-install --parallel-workers "$(nproc)" "$@"
  )

  if [[ -f ${_sbs_ws}/install/setup.bash ]]; then
    # shellcheck disable=SC1091
    source "${_sbs_ws}/install/setup.bash"
    echo "sourced ${_sbs_ws}/install/setup.bash"
  else
    echo -e "\e[31m[ERROR]: build produced no install/setup.bash -- not sourced.\e[0m"
  fi
fi

unset _sbs_ws
