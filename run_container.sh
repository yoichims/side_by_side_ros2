#!/usr/bin/env bash
#
# Starts the side-by-side development container.
#
# The repository itself is the colcon workspace: it is mounted at
# /home/user/sbs_ws, so src/ is already the package tree and environments/ and
# article.pdf come along with it.
#
# Adapted from a generic ROS 2 container script. The graphics and GPU
# detection is kept as it is -- this runs on several machines with different
# GPUs and different display servers, and each path probes at runtime and falls
# back rather than assuming. What is gone is the sensor rig: camera, USB serial
# and the external bag disk. This project simulates its vehicles and its laser,
# so there is no hardware to pass through.

set -u

source utils

IMAGE_TAG=$(cat .env | grep IMAGE_TAG | cut -d = -f 2)

# Prefer the Wayland backend for Qt apps (QT_QPA_PLATFORM=wayland;xcb).
# NOT the default, and NOT usable by RViz: rviz2 renders through OGRE 1.12
# built with the GLX backend, which hard-requires an X display and aborts with
# "Couldn't open X display" under a native Wayland surface. Useful for rqt and
# other pure-Qt tools only. The Wayland socket itself is always exposed on a
# Wayland session regardless of this flag.
PREFER_WAYLAND_QT=0

while [[ ${#} -gt 0 ]]; do
  key="${1}"

  case ${key} in
    -d|--distro)
      IMAGE_TAG="${2}"
      shift
      shift
      ;;
    -w|--wayland)
      PREFER_WAYLAND_QT=1
      shift
      ;;
    -h|--help)
      echo "Usage: ./run_container.sh [-d DISTRO] [-w]"
      echo "  -d, --distro   ROS 2 distribution tag (default: from .env)"
      echo "  -w, --wayland  let Qt apps use Wayland (RViz cannot; see above)"
      echo
      echo "Environment:"
      echo "  SBS_NO_NVIDIA=1  skip the NVIDIA path even where it would work"
      echo "  SBS_GPU_HELP=1   explain how to enable NVIDIA passthrough"
      exit 0
      ;;
    *)
      echo -e "\e[33mIgnoring unknown option ${key}.\e[0m"
      exit 1
      ;;
  esac
done

# --- Graphical session detection -----------------------------------------
# The host may be running X11 or Wayland. X11 access is set up either way:
# on Wayland, DISPLAY points at XWayland, and rviz2 can ONLY use that path.
SESSION_TYPE="${XDG_SESSION_TYPE:-}"
if [[ -z ${SESSION_TYPE} ]]; then
  if [[ -n ${WAYLAND_DISPLAY:-} ]]; then SESSION_TYPE=wayland
  elif [[ -n ${DISPLAY:-} ]]; then SESSION_TYPE=x11
  else SESSION_TYPE=none; fi
fi

DOCKER_ARGS_GUI=""
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

if [[ -n ${DISPLAY:-} ]]; then
  touch ${XAUTH}
  xauth nlist ${DISPLAY} 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f ${XAUTH} nmerge - 2>/dev/null
  DOCKER_ARGS_GUI+="-e DISPLAY "
  DOCKER_ARGS_GUI+="-e QT_X11_NO_MITSHM=1 "
  DOCKER_ARGS_GUI+="-e XAUTHORITY=${XAUTH} "
  DOCKER_ARGS_GUI+="-v ${XAUTH}:${XAUTH}:rw "
  DOCKER_ARGS_GUI+="-v ${XSOCK}:${XSOCK}:rw "
else
  echo -e "\e[33m[WARN]: DISPLAY is unset -- no X11 path. RViz will not start.\e[0m"
  echo -e "\e[33m         Launch with rviz:=false, or run headless.\e[0m"
fi

# --- NVIDIA passthrough ---------------------------------------------------
# nvidia-smi only proves the kernel driver is installed. Handing Docker
# --gpus=all additionally requires the NVIDIA Container Toolkit on the host:
# recent Docker resolves --gpus through CDI and, with no toolkit present,
# refuses to create the container at all with
#     failed to discover GPU vendor from CDI: no known GPU vendor found
# which takes the whole session down, GUI included. So probe for the toolkit
# rather than for the driver. Machines that have it keep full NVIDIA
# rendering; machines that do not still start and render on whatever GPU
# /dev/dri exposes below (the Intel iGPU on a hybrid laptop). Set
# SBS_NO_NVIDIA=1 to skip the NVIDIA path even where it would work.
nvidia_passthrough_available() {
  # Legacy runtime hook (nvidia-container-toolkit <= 1.13 style).
  command -v nvidia-container-cli &> /dev/null && return 0
  command -v nvidia-container-runtime &> /dev/null && return 0
  # Runtime registered with the daemon via /etc/docker/daemon.json.
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' && return 0
  # CDI spec produced by `nvidia-ctk cdi generate`.
  grep -rqs 'nvidia.com/gpu' /etc/cdi /var/run/cdi 2>/dev/null && return 0
  return 1
}

if [[ ${SBS_NO_NVIDIA:-0} -ne 1 ]] && command -v nvidia-smi &> /dev/null; then
  gpu_list=$(nvidia-smi -L 2>/dev/null)
  if [[ -z ${gpu_list} ]]; then
    echo -e "\e[33m[WARN]: nvidia-smi found no GPU.\e[0m"
  elif nvidia_passthrough_available; then
    DOCKER_ARGS_GUI+="-e NVIDIA_DRIVER_CAPABILITIES=all "
    DOCKER_ARGS_GUI+="-e __NV_PRIME_RENDER_OFFLOAD=1 "
    DOCKER_ARGS_GUI+="-e __GLX_VENDOR_LIBRARY_NAME=nvidia "
    DOCKER_ARGS_GUI+="--gpus=all "
    echo "GPU: NVIDIA passthrough enabled (--gpus=all)."
  else
    echo "GPU: /dev/dri rendering (NVIDIA Container Toolkit not installed)."
    if [[ ${SBS_GPU_HELP:-0} -eq 1 ]]; then
      # The toolkit is not in Ubuntu's own repositories, so telling anyone to
      # apt-get install it produces "Unable to locate package". Only offer the
      # install when the package is actually reachable.
      if apt-cache policy nvidia-container-toolkit 2>/dev/null | grep -q Candidate; then
        echo "     To enable the NVIDIA path:"
        echo "       sudo apt-get install -y nvidia-container-toolkit"
        echo "       sudo nvidia-ctk runtime configure --runtime=docker"
        echo "       sudo systemctl restart docker"
      else
        echo "     It is not in Ubuntu's repositories; NVIDIA's has to be added"
        echo "     first. See \"NVIDIA rendering\" in readme.md for the steps."
      fi
    fi
  fi
fi

# RViz needs the render nodes for hardware OpenGL. This must sit OUTSIDE the
# NVIDIA branch: on an Intel-only machine nothing would be passed and RViz
# would fall back to software rendering. The render group is added by host gid
# because it is not a fixed number.
DOCKER_ARGS_GPU=""
if [[ -d /dev/dri ]]; then
  shopt -s nullglob
  dri_devices=(/dev/dri/*)
  shopt -u nullglob
  for dev in "${dri_devices[@]}"; do
    [[ -c ${dev} ]] && DOCKER_ARGS_GPU+="--device=${dev} "
  done
  # Both groups by host gid: the render nodes are root:render and the card
  # nodes are root:video, and neither gid is fixed across distributions, so
  # the image's own "video" membership cannot be relied on.
  for grp in render video; do
    grp_gid=$(getent group ${grp} | cut -d: -f3)
    [[ -n ${grp_gid} ]] && DOCKER_ARGS_GPU+="--group-add ${grp_gid} "
  done
else
  echo -e "\e[33m[WARN]: No /dev/dri; RViz will use software rendering.\e[0m"
fi

# --- Wayland socket -------------------------------------------------------
# Exposed whenever the host session is Wayland, so pure-Qt tools can use it.
# QT_QPA_PLATFORM is pinned to xcb unless --wayland is given, because
# rviz2 aborts on a native Wayland surface (OGRE GLX backend, see above).
DOCKER_ARGS_WAYLAND=""
if [[ ${SESSION_TYPE} == wayland ]]; then
  WL_SOCK="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/${WAYLAND_DISPLAY:-wayland-0}"
  if [[ -S ${WL_SOCK} ]]; then
    WL_IN="/run/user/$(id -u)/${WAYLAND_DISPLAY:-wayland-0}"
    DOCKER_ARGS_WAYLAND="-v ${WL_SOCK}:${WL_IN}:rw "
    DOCKER_ARGS_WAYLAND+="-e XDG_RUNTIME_DIR=/run/user/$(id -u) "
    DOCKER_ARGS_WAYLAND+="-e WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0} "
    DOCKER_ARGS_WAYLAND+="-e XDG_SESSION_TYPE=wayland "
    # Qt5 picks its platform plugin from XDG_SESSION_TYPE. With qtwayland5
    # installed that means it would choose the Wayland plugin -- and rviz2 then
    # dies in OGRE with "Invalid parentWindowHandle (wrong server or screen)",
    # because its GLX render system cannot draw into a Wayland surface.
    # Pin xcb so RViz keeps working; --wayland overrides it deliberately.
    DOCKER_ARGS_WAYLAND+="-e QT_QPA_PLATFORM=xcb "
    if [[ ${PREFER_WAYLAND_QT} -eq 1 ]]; then
      DOCKER_ARGS_WAYLAND+="-e QT_QPA_PLATFORM='wayland;xcb' "
      echo -e "\e[33m[WARN]: --wayland sets QT_QPA_PLATFORM=wayland;xcb for Qt apps.\e[0m"
      echo -e "\e[33m         rviz2 CANNOT run this way (OGRE uses GLX). Start it with:\e[0m"
      echo -e "\e[33m           QT_QPA_PLATFORM=xcb rviz2\e[0m"
    fi
  else
    echo -e "\e[33m[WARN]: Wayland session but no socket at ${WL_SOCK}.\e[0m"
  fi
fi

# --- Fast DDS transport profile, container-wide ---------------------------
#
# coi_6th_floor is 4000x4000 cells, so the map is a ~16 MB OccupancyGrid. On
# the stock 213 KB socket buffer a sample that large arrives as many fragments
# and losing any one of them discards the whole map -- silently. The symptom is
# a map server that publishes and a simulated laser that never does.
#
# both.launch.py sets this for the nodes it starts, but a launch argument
# cannot reach a process you start yourself, so a second terminal running
# `ros2 topic echo` or its own RViz needed the export by hand every time.
# Setting it here covers every process in the container, joined shells
# included, because `docker exec` inherits the container's environment.
#
# The path is the SOURCE tree rather than install/, so it resolves before the
# workspace has ever been built.
DDS_PROFILE=${SBS_WS_CONTAINER}/src/sbs_bringup/config/fastdds_large_messages.xml

echo "Workspace: $(pwd) -> ${SBS_WS_CONTAINER}"
echo "Starting container using ${IMAGE_NAME}:${IMAGE_TAG}"

exec docker run -it --rm --name=${CONTAINER_NAME} \
  --net=host \
  -e SBS_WS=${SBS_WS_CONTAINER} \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=${DDS_PROFILE} \
  ${DOCKER_ARGS_GUI} \
  ${DOCKER_ARGS_WAYLAND} \
  ${DOCKER_ARGS_GPU} \
  -v "$(pwd)":${SBS_WS_CONTAINER}:rw \
  ${IMAGE_NAME}:${IMAGE_TAG} \
  bash
