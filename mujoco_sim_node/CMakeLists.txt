cmake_minimum_required(VERSION 3.8)
project(mujoco_sim_node)

# Find ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(franka_msgs REQUIRED)

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
add_executable(mujoco_node src/mujoco_node.cpp)

# Link non-ROS2 library (MuJoCo)
target_link_libraries(mujoco_node
    ${MUJOCO_ROOT}/lib/libmujoco.so
    glfw
    Eigen3::Eigen
)

# Link ROS2 libraries (automatically handles everything)
ament_target_dependencies(mujoco_node rclcpp std_msgs geometry_msgs sensor_msgs rclcpp_action franka_msgs) 

target_include_directories(mujoco_node PUBLIC
    ${MUJOCO_INCLUDE_DIR}
)

set_target_properties(mujoco_node PROPERTIES
    INSTALL_RPATH "${MUJOCO_ROOT}/lib"
    BUILD_RPATH "${MUJOCO_ROOT}/lib"
)

# Install
install(TARGETS mujoco_node
    DESTINATION lib/${PROJECT_NAME}
)

# ROS2 specific
ament_package()
