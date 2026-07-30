cmake_minimum_required(VERSION 3.8)
project(camera_viewer)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(camera_viewer src/camera_viewer.cpp)

ament_target_dependencies(camera_viewer rclcpp sensor_msgs cv_bridge)

target_link_libraries(camera_viewer ${OpenCV_LIBS})

install(TARGETS camera_viewer DESTINATION lib/${PROJECT_NAME})

ament_package()
