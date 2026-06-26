cmake_minimum_required(VERSION 3.8)
project(haptic_device)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)

# ========== HAPTIC DEVICE NODE ==========
add_executable(haptic_device_node src/haptic_device_node.cpp)
target_compile_options(haptic_device_node PRIVATE -g -O0 -W -Dlinux -D_cplusplus)
ament_target_dependencies(haptic_device_node
  rclcpp std_msgs geometry_msgs tf2)
target_link_libraries(haptic_device_node HDU HD rt stdc++ m)

# ========== INSTALL ==========
install(TARGETS
  haptic_device_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
