ARG ROS2_DISTRO=humble
FROM ros:${ROS2_DISTRO}-ros-core
ARG ROS2_DISTRO

ENV DEBIAN_FRONTEND=noninteractive

ARG APT_MIRROR="us"
RUN sed --in-place --regexp-extended "s|(//)(archive\.ubuntu)|\1${APT_MIRROR}.\2|" /etc/apt/sources.list

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    iproute2 \
    iputils-ping \
    nano \
    emacs \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    sudo \
    # For analysing the recorded human driving data and the trajectories the
    # planner produces.
  && python3 -m pip install \
    "numpy<2.0.0" \
    pandas \
    scipy \
    matplotlib \
  && ( [ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || rosdep init ) \
  && rosdep update \
  && rm -rf /var/lib/apt/lists/*

# Build and runtime dependencies of the nine packages in src/.
#
#   grid_map      the utility grid sbs_motion scores over, and occmap's output
#   laser_filters the shadow filter between the simulated laser and uni_density
#   twist_mux     replaces the ROS 1 yocs_cmd_vel_mux nodelet, which has no port
#   nav2_*        map server, plus the lifecycle manager that activates it
#   laser_geometry scan -> cloud projection in scan_accumulator
#
# Deliberately absent: velodyne, v4l2_camera, rqt_image_view and the IMU
# plugin. The sensor drivers and logging were dropped from this repository, and
# the simulation has no camera or IMU.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libeigen3-dev \
    libpcl-dev \
    libyaml-cpp-dev \
    ros-${ROS2_DISTRO}-grid-map \
    ros-${ROS2_DISTRO}-laser-filters \
    ros-${ROS2_DISTRO}-laser-geometry \
    ros-${ROS2_DISTRO}-nav2-lifecycle-manager \
    ros-${ROS2_DISTRO}-nav2-map-server \
    ros-${ROS2_DISTRO}-nav2-rviz-plugins \
    ros-${ROS2_DISTRO}-pcl-conversions \
    ros-${ROS2_DISTRO}-pcl-msgs \
    ros-${ROS2_DISTRO}-pcl-ros \
    ros-${ROS2_DISTRO}-ros2bag \
    ros-${ROS2_DISTRO}-rosbag2 \
    ros-${ROS2_DISTRO}-rosbag2-storage-default-plugins \
    ros-${ROS2_DISTRO}-rosbag2-storage-mcap \
    ros-${ROS2_DISTRO}-rviz2 \
    ros-${ROS2_DISTRO}-tf2 \
    ros-${ROS2_DISTRO}-tf2-eigen \
    ros-${ROS2_DISTRO}-tf2-geometry-msgs \
    ros-${ROS2_DISTRO}-tf2-sensor-msgs \
    ros-${ROS2_DISTRO}-tf2-tools \
    ros-${ROS2_DISTRO}-twist-mux \
  && rm -rf /var/lib/apt/lists/*

# Qt's Wayland platform plugin, so pure-Qt tools can run as native Wayland
# clients instead of going through XWayland. RViz cannot: its OGRE render
# system uses GLX and hard-requires an X display, which is why
# run_container.sh pins QT_QPA_PLATFORM=xcb unless told otherwise.
# mesa-utils is here for diagnosing the GL path.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    qtwayland5 \
    libwayland-egl1 \
    mesa-utils \
  && rm -rf /var/lib/apt/lists/*

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid ${GROUP_ID} user \
  && adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} user \
  && usermod -a -G sudo,video,dialout user \
  && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# The DRI render nodes are root:render on the host, and run_container.sh passes
# that group by numeric gid because it is not fixed across distributions. This
# image has no render group at all, so the gid arrives unnamed and every shell
# greets you with "groups: cannot find name for group ID 990". The access was
# always fine -- an unnamed supplementary gid still grants it -- but the noise
# is not. The Makefile reads the host's gid and passes it here, which is the
# same thing it already does for USER_ID and GROUP_ID.
ARG RENDER_GID=""
RUN if [ -n "${RENDER_GID}" ]; then \
      if getent group "${RENDER_GID}" > /dev/null; then \
        echo "gid ${RENDER_GID} is already $(getent group ${RENDER_GID} | cut -d: -f1)"; \
      else \
        groupadd -g "${RENDER_GID}" render; \
      fi; \
      usermod -a -G "${RENDER_GID}" user; \
    else \
      echo "No RENDER_GID given; hardware GL still works, the gid is just unnamed."; \
    fi

RUN mkdir -p /run/user/${USER_ID} \
  && chown ${USER_ID}:${GROUP_ID} /run/user/${USER_ID} \
  && chmod 700 /run/user/${USER_ID}

USER user
# The workspace overlay is sourced only once it exists, so a shell in a
# freshly cloned tree that has not been built yet still starts cleanly.
RUN echo "source /opt/ros/${ROS2_DISTRO}/setup.bash" >> /home/user/.bashrc \
  && echo "[ -f /home/user/sbs_ws/install/setup.bash ] && source /home/user/sbs_ws/install/setup.bash" >> /home/user/.bashrc

WORKDIR /home/user/sbs_ws
