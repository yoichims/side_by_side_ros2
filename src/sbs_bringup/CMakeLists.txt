cmake_minimum_required(VERSION 3.8)
project(sbs_bringup)

# Dependencies live in package.xml alone; see simple_simulator/CMakeLists.txt.
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

install(PROGRAMS
  scripts/publish_partner_obstacle.py
  scripts/publish_footprint.py
  scripts/publish_world_objects.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE launch config)
