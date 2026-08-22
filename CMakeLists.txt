# cmake_minimum_required(VERSION 3.0.2)  # original
cmake_minimum_required(VERSION 3.10...3.22)
project(aqua_slam)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -w")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-DCOMPILEDWITHC11)
set(CMAKE_CXX_STANDARD 17)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages


#set(OpenCV_DIR /home/da/project/Thirdparty/opencv3.2/share/OpenCV)
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /home/da/project/Thirdparty/opencv3.2)
# find_package(OpenCV 3.2 REQUIRED)  # original
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# original --------------------------------------------------------------------
# find_package(catkin REQUIRED COMPONENTS
#         geometry_msgs
#         image_transport
#         message_filters
#         nav_msgs
# #        octomap_msgs
# #        pose_graph_msgs
#         roscpp
#         sensor_msgs
#         tf
#         message_generation
#         rospy
#         std_msgs
#         waterlinked_a50_ros_driver
#         )
# -----------------------------------------------------------------------------

find_package(Pangolin REQUIRED)
set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/eigen3.3.4)
find_package(Eigen3 3.3 REQUIRED)
# set(VTK_DIR "/usr/lib/x86_64-linux-gnu/cmake/vtk-9.1")  # temporary: disabled (pulls Java via PCL visualization)
# find_package(VTK REQUIRED)  # temporary: disabled
find_package(PCL REQUIRED COMPONENTS common io filters)  # temporary: removed visualization (VTK/Java not needed for node)
# find_package(PCL REQUIRED COMPONENTS common io visualization)  # original
find_package(fmt REQUIRED)
find_package(Boost REQUIRED COMPONENTS log_setup log)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_LOG_DYN_LINK")

# ROS 2 -----------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(std_srvs REQUIRED)

find_package(pcl_conversions REQUIRED)
# Optional UW_SLAM bridge message for SuperPoint match injection
find_package(uw_slam_bridge QUIET)
if(uw_slam_bridge_FOUND)
  message(STATUS "UW_SLAM: found uw_slam_bridge, enabling StereoMatches injection")
  add_definitions(-DUW_SLAM_HAS_STEREO_MATCHES=1)
  list(APPEND dependencies uw_slam_bridge)
else()
  message(STATUS "UW_SLAM: uw_slam_bridge not found; building without StereoMatches msg")
endif()
# find_package(pcl_ros REQUIRED)
# -----------------------------------------------------------------------------


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   SetInfof.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need

# original --------------------------------------------------------------------
# find_package(catkin REQUIRED COMPONENTS
# catkin_package(
#         INCLUDE_DIRS include
#         LIBRARIES lib
#         CATKIN_DEPENDS geometry_msgs image_transport message_filters nav_msgs octomap_msgs roscpp sensor_msgs tf message_runtime waterlinked_a50_ros_driver
#         #          DEPENDS system_lib
# )
# -----------------------------------------------------------------------------

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CameraModels
        ${PROJECT_SOURCE_DIR}/include/cv_bridge_slam
        ${PROJECT_SOURCE_DIR}/Thirdparty/octomap/include
        ${EIGEN3_INCLUDE_DIR}
        #${PROJECT_SOURCE_DIR}/Thirdparty/include/eigen3
        ${Pangolin_INCLUDE_DIRS}
        # ${VTK_INCLUDE_DIRS}  # new
        # /usr/include/vtk-9.1  # temporary: disabled
        ${PCL_INCLUDE_DIRS}
        /home/da/Matlab/extern/include #matlab head
        # ${catkin_INCLUDE_DIRS}  # original

)

# ROS 2 -----------------------------------------------------------------------
include_directories(
  include
#   ${rclcpp_INCLUDE_DIRS}

#   ${image_transport_INCLUDE_DIRS}
#   ${cv_bridge_INCLUDE_DIRS}

#   ${sensor_msgs_INCLUDE_DIRS}
#   ${nav_msgs_INCLUDE_DIRS}
#   ${geometry_msgs_INCLUDE_DIRS}
)
# -----------------------------------------------------------------------------

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/dvl1.cpp
# )

link_directories(
        ${PCL_LIBRARY_DIRS}
        /home/da/Matlab/bin/glnxa64
)
add_definitions(${PCL_DEFINITIONS})

add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/g2o)
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2)
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/octomap)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} )

# save all variables to cmake_variables.txt
get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)
FILE(WRITE "cmake_variables.txt" "") # Empty file
foreach (_variableName ${_variableNames})
    FILE(APPEND "cmake_variables.txt" "${_variableName}=${${_variableName}}\n")
    # FILE(APPEND "cmake_variables.txt" "${_variableName}\n")
endforeach ()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED  # original
# add_library(aqua_slam SHARED 
        ${PROJECT_SOURCE_DIR}/include/System.h
        ${PROJECT_SOURCE_DIR}/include/Tracking.h
        ${PROJECT_SOURCE_DIR}/include/LocalMapping.h
        ${PROJECT_SOURCE_DIR}/include/LoopClosing.h
        ${PROJECT_SOURCE_DIR}/include/ORBextractor.h
        ${PROJECT_SOURCE_DIR}/include/ORBmatcher.h
        ${PROJECT_SOURCE_DIR}/include/FrameDrawer.h
        ${PROJECT_SOURCE_DIR}/include/Converter.h
        ${PROJECT_SOURCE_DIR}/include/MapPoint.h
        ${PROJECT_SOURCE_DIR}/include/KeyFrame.h
        ${PROJECT_SOURCE_DIR}/include/Atlas.h
        ${PROJECT_SOURCE_DIR}/include/Map.h
#        ${PROJECT_SOURCE_DIR}/include/MapDrawer.h
        ${PROJECT_SOURCE_DIR}/include/PnPsolver.h
        ${PROJECT_SOURCE_DIR}/include/Frame.h
        ${PROJECT_SOURCE_DIR}/include/KeyFrameDatabase.h
        ${PROJECT_SOURCE_DIR}/include/Sim3Solver.h
        ${PROJECT_SOURCE_DIR}/include/Initializer.h
#        ${PROJECT_SOURCE_DIR}/include/Viewer.h
        ${PROJECT_SOURCE_DIR}/include/CameraModels/GeometricCamera.h
        ${PROJECT_SOURCE_DIR}/include/CameraModels/Pinhole.h
        ${PROJECT_SOURCE_DIR}/include/CameraModels/KannalaBrandt8.h
        ${PROJECT_SOURCE_DIR}/include/MLPnPsolver.h
        ${PROJECT_SOURCE_DIR}/include/TwoViewReconstruction.h
        # ${PROJECT_SOURCE_DIR}/include/RosHandling.h  # original
        ${PROJECT_SOURCE_DIR}/include/RosHandling.h
        ${PROJECT_SOURCE_DIR}/include/DenseMapper.h
        ${PROJECT_SOURCE_DIR}/include/LKTracker.h
        ${PROJECT_SOURCE_DIR}/src/System.cc
        ${PROJECT_SOURCE_DIR}/src/Tracking.cc
        ${PROJECT_SOURCE_DIR}/src/LocalMapping.cc
        ${PROJECT_SOURCE_DIR}/src/LoopClosing.cc
        ${PROJECT_SOURCE_DIR}/src/ORBextractor.cc
        ${PROJECT_SOURCE_DIR}/src/ORBmatcher.cc
        ${PROJECT_SOURCE_DIR}/src/FrameDrawer.cc
        ${PROJECT_SOURCE_DIR}/src/Converter.cc
        ${PROJECT_SOURCE_DIR}/src/MapPoint.cc
        ${PROJECT_SOURCE_DIR}/src/KeyFrame.cc
        ${PROJECT_SOURCE_DIR}/src/Atlas.cc
        ${PROJECT_SOURCE_DIR}/src/Map.cc
#        ${PROJECT_SOURCE_DIR}/src/MapDrawer.cc
        ${PROJECT_SOURCE_DIR}/src/DvlGyroOptimizer.cpp
        ${PROJECT_SOURCE_DIR}/src/PnPsolver.cc
        ${PROJECT_SOURCE_DIR}/src/Frame.cc
        ${PROJECT_SOURCE_DIR}/src/KeyFrameDatabase.cc
        ${PROJECT_SOURCE_DIR}/src/Sim3Solver.cc
        ${PROJECT_SOURCE_DIR}/src/Initializer.cc
#        ${PROJECT_SOURCE_DIR}/src/Viewer.cc
        ${PROJECT_SOURCE_DIR}/src/CameraModels/Pinhole.cpp
        ${PROJECT_SOURCE_DIR}/src/CameraModels/KannalaBrandt8.cpp
        ${PROJECT_SOURCE_DIR}/src/MLPnPsolver.cpp
        ${PROJECT_SOURCE_DIR}/src/TwoViewReconstruction.cc
        # ${PROJECT_SOURCE_DIR}/src/RosHandling.cpp  # original
        ${PROJECT_SOURCE_DIR}/src/RosHandling.cpp
        ${PROJECT_SOURCE_DIR}/src/DenseMapper.cpp
        # ${PROJECT_SOURCE_DIR}/src/cv_bridge.cpp  # original
        ${PROJECT_SOURCE_DIR}/src/rgb_colors.cpp
        ${PROJECT_SOURCE_DIR}/src/LKTracker.cpp src/FrameKLT.cpp src/FrameKLT.h src/Integrator.cpp src/Integrator.h)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp

  image_transport
  cv_bridge

  sensor_msgs
  nav_msgs
  geometry_msgs
  octomap_msgs
  visualization_msgs

  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs

  std_srvs

  pcl_conversions
)

# add_library(${PROJECT_NAME}_Optmization SHARED  # original
add_library(${PROJECT_NAME}_optimization SHARED
# add_library(aqua_slam_optimization SHARED 
        ${PROJECT_SOURCE_DIR}/include/Optimizer.h
        ${PROJECT_SOURCE_DIR}/include/ImuTypes.h
        ${PROJECT_SOURCE_DIR}/include/DVLGroPreIntegration.h
        ${PROJECT_SOURCE_DIR}/include/G2oTypes.h
        ${PROJECT_SOURCE_DIR}/include/OptimizableTypes.h
        ${PROJECT_SOURCE_DIR}/src/Optimizer.cc
        ${PROJECT_SOURCE_DIR}/src/ImuTypes.cc
        ${PROJECT_SOURCE_DIR}/src/DVLGroPreIntegration.cpp
        ${PROJECT_SOURCE_DIR}/src/G2oTypes.cc
        ${PROJECT_SOURCE_DIR}/src/OptimizableTypes.cpp)
ament_target_dependencies(${PROJECT_NAME}_optimization
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_srvs
  image_transport
  cv_bridge
  visualization_msgs
  octomap_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  pcl_conversions
)
# target_link_libraries(${PROJECT_NAME}_Optmization  # original
target_link_libraries(${PROJECT_NAME}_optimization
# target_link_libraries(aqua_slam_node_optimization
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES}
        # ${VTK_LIBRARIES}  # temporary: disabled
        fmt::fmt-header-only
        ${Sophus_LIBRARIES}
        DBoW2
        g2o

        # ${catkin_LIBRARIES}  # original
        )
target_link_libraries(${PROJECT_NAME}
        ${PROJECT_NAME}_optimization
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES}
        # ${VTK_LIBRARIES}  # temporary: disabled
        fmt::fmt-header-only
        DBoW2
        octomap
        # ${catkin_LIBRARIES}  # original
        -lboost_serialization
        -lboost_system
        -lboost_filesystem
        -lboost_iostreams
        -lcrypto
        )

# ROS 2 -----------------------------------------------------------------------
add_executable(${PROJECT_NAME}_node src/node.cpp)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp

  image_transport
  cv_bridge

  sensor_msgs
  nav_msgs
  geometry_msgs
  octomap_msgs
  visualization_msgs

  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs

  std_srvs

  pcl_conversions
#   pcl_ros
)
if(uw_slam_bridge_FOUND)
  ament_target_dependencies(${PROJECT_NAME}_node uw_slam_bridge)
endif()

# target_link_libraries(stereo_dvl2 ${Boost_LIBRARIES} ${PROJECT_NAME} ${PROJECT_NAME}_Optmization)  // original
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
  ${PROJECT_NAME}_optimization
  ${Boost_LIBRARIES}
)
# -----------------------------------------------------------------------------

#add_executable(g2o_BA src/g2o_BA.cpp)
#target_link_libraries(g2o_BA ${PROJECT_NAME} ${PROJECT_NAME}_Optmization)
#target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/dvl1_node.cpp)
# add_executable(stereo_dvl2 ${PROJECT_SOURCE_DIR}/src/ros_stereo_DVL_tighly.cc)  # original
# target_link_libraries(stereo_dvl2 ${Boost_LIBRARIES} ${PROJECT_NAME} ${PROJECT_NAME}_Optmization)  # original
# target_compile_features(stereo_dvl2 PUBLIC cxx_std_17)  # original
# add_dependencies(stereo_dvl2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})  # original

#add_executable(align src/test/align_trajectory.cpp)
#target_link_libraries(align ${PROJECT_NAME} ${PROJECT_NAME}_Optmization)
#
#add_executable(dense src/test/dense_reconstruct.cpp)
#target_link_libraries(dense ${PROJECT_NAME} ${PROJECT_NAME}_Optmization)
#add_executable(dense_ros src/test/dense_reconstruct_ros.cpp)
#target_link_libraries(dense_ros ${PROJECT_NAME} ${PROJECT_NAME}_Optmization)
#
#add_executable(dense_viewer src/test/dense_viewer.cpp)
#target_link_libraries(dense_viewer ${PROJECT_NAME} ${PROJECT_NAME}_Optmization)
#
#add_executable(boost_test ${PROJECT_SOURCE_DIR}/src/test/boost_test.cpp)
#target_link_libraries(boost_test -lboost_serialization
#        -lboost_system)
#
#add_executable(tf_test ${PROJECT_SOURCE_DIR}/src/test/tf_test.cpp)

#add_executable(lk_tracker ${PROJECT_SOURCE_DIR}/src/lkTracker.cpp)
##target_link_libraries(lk_tracker ${OpenCV_LIBS} ${catkin_LIBRARIES})
#target_link_libraries(lk_tracker ${Boost_LIBRARIES} ${PROJECT_NAME})
#target_compile_features(lk_tracker PUBLIC cxx_std_17)
#
#add_executable(eigen_test ${PROJECT_SOURCE_DIR}/src/test/eigen_test.cpp)
#target_link_libraries(eigen_test ${catkin_LIBRARIES})
#target_compile_features(eigen_test PUBLIC cxx_std_17)
#
#add_executable(dvL_model_test ${PROJECT_SOURCE_DIR}/src/dvl_model/dvl.cpp)
#target_link_libraries(dvL_model_test ${catkin_LIBRARIES} ${PROJECT_NAME} -lmat -lmx)
#target_compile_features(dvL_model_test PUBLIC cxx_std_17)
#
#add_executable(matlab_write ${PROJECT_SOURCE_DIR}/src/dvl_model/matwrite.cpp)
#target_link_libraries(matlab_write -lmat -lmx)
#target_compile_features(matlab_write PUBLIC cxx_std_17)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# original --------------------------------------------------------------------
# install(TARGETS stereo_dvl2
#         # RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}  # original
#         )
# -----------------------------------------------------------------------------

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_dvl1.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

# ROS 2 -----------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_optimization
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES
  ${PROJECT_SOURCE_DIR}/lib/liboctomap.so
  ${PROJECT_SOURCE_DIR}/lib/liboctomath.so
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
  DESTINATION lib
)

install(DIRECTORY launch data urdf Vocabulary
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dvl_converter.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dvl_converter
)

install(PROGRAMS
  scripts/sim_dvl_converter.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sim_dvl_converter
)

install(PROGRAMS
  scripts/rov_keyboard_teleop.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME rov_keyboard_teleop
)

ament_package()
# -----------------------------------------------------------------------------
