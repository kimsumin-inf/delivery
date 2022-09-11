cmake_minimum_required(VERSION 2.8.3)
project(delivery)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  darknet_ros_msgs
  data_transfer_msg
  roscpp
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)

catkin_package(INCLUDE_DIRS include  CATKIN_DEPENDS cv_bridge darknet_ros_msgs data_transfer_msg roscpp sensor_msgs std_msgs)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/delivery_node.cpp src/delivery.cpp include/delivery/delivery.h)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})