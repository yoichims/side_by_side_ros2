cmake_minimum_required(VERSION 3.8)
project(sbs_msgs)

# geometry_msgs and std_msgs are declared in package.xml and found from there;
# `rosidl_generate_interfaces` still has to be told which of them the
# interfaces reference, which is what DEPENDENCIES is for.
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CostMatrix.msg"
  "msg/RolloutSet.msg"
  "msg/SafetyStatus.msg"
  "msg/SubgoalList.msg"
  "srv/SendGoals.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_auto_package()
