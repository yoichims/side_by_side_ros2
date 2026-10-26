cmake_minimum_required(VERSION 3.8)
project(simple_simulator)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Dependencies are declared once, in package.xml. `ament_auto_find_build_dependencies`
# reads them from there and finds each one, so a dependency added to the
# manifest does not also have to be added here -- the two lists could drift,
# and the failure when they did was a link error a long way from the cause.
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Not in package.xml because it is genuinely optional: the simulator builds
# and runs without it, single-threaded.
find_package(OpenMP)

ament_auto_add_executable(odo_simulator src/odo_simulator.cpp)

ament_auto_add_executable(utm_simulator src/utm_simulator.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(utm_simulator OpenMP::OpenMP_CXX)
endif()

ament_auto_package(INSTALL_TO_SHARE launch)
