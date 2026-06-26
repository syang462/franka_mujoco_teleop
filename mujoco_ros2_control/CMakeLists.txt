cmake_minimum_required(VERSION 3.8)
project(mujoco_ros2_control)

# Find ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Find GLFW
find_package(glfw3 REQUIRED)
find_package(Eigen3 REQUIRED)

# Find MuJoCo
set(MUJOCO_ROOT /home/ao/.mujoco/mujoco-3.9.0)
find_path(MUJOCO_INCLUDE_DIR
    NAMES mujoco/mujoco.h
    PATHS ${MUJOCO_ROOT}/include
    REQUIRED
)

# Create executable
add_executable(ros_control src/ros_control.cpp)

# Link non-ROS2 library (MuJoCo)
target_link_libraries(ros_control
    ${MUJOCO_ROOT}/lib/libmujoco.so
    glfw
    Eigen3::Eigen
)

# Link ROS2 libraries (automatically handles everything)
ament_target_dependencies(ros_control rclcpp std_msgs geometry_msgs)

target_include_directories(ros_control PUBLIC
    ${MUJOCO_INCLUDE_DIR}
)

set_target_properties(ros_control PROPERTIES
    INSTALL_RPATH "${MUJOCO_ROOT}/lib"
    BUILD_RPATH "${MUJOCO_ROOT}/lib"
)

# Install
install(TARGETS ros_control
    DESTINATION lib/${PROJECT_NAME}
)

# ROS2 specific
ament_package()
