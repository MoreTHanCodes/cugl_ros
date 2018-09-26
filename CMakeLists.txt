cmake_minimum_required(VERSION 2.8.3)
project(cugl_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find universal dependencies
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED
NO_MODULE
PATHS /usr/local
NO_DEFAULT_PATH)

# set CUDA NVCC flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_30,code=sm_30")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_35,code=sm_35")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_37,code=sm_37")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_52,code=sm_52")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_60,code=sm_60")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_61,code=sm_61")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  pcl_conversions
  pcl_ros
)

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
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
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
#   Service1.srv
#   Service2.srv
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
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
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
# catkin_package(
#   INCLUDE_DIRS include ${OpenCV_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR}
#   LIBRARIES ${PROJECT_NAME}_common ${PROJECT_NAME}_visualization ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCV_LIBRARIES}
#   CATKIN_DEPENDS roscpp std_msgs
#   DEPENDS system_lib CUDA EIGEN3 GLUT OPENGL OpenCV
# )
catkin_package(
  INCLUDE_DIRS include ${OpenCV_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR}
  LIBRARIES ${PROJECT_NAME}_common ${PROJECT_NAME}_visualization ${PROJECT_NAME}_dynamic_fusion ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCV_LIBRARIES}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs pcl_conversions pcl_ros
  DEPENDS CUDA EIGEN3 GLUT OPENGL OpenCV
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${GLUT_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
)

set(BUILD_COMMON TRUE)
set(BUILD_VISUALIZATION TRUE)
set(BUILD_APPS TRUE)
set(BUILD_ROS TRUE)

#################
## CUGL COMMON ##
#################
if(BUILD_COMMON)
  set(SUBSYS_NAME common)
  set(SUBSYS_DEPS)

  set(incs
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/helper_cuda.h"
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/helper_cuda_gl.h"
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/helper_gl.h"
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/helper_string.h"
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/containers/device_array_handle.h"
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/containers/device_array.h"
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/containers/texture_bridge.h"
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/geometry/affine3d.h"
  )

  set(impl_incs
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/containers/impl/device_array.hpp"
  )

  set(srcs
    "src/${SUBSYS_NAME}/containers/texture_bridge.cpp"
  )

  set(LIB_NAME "${PROJECT_NAME}_${SUBSYS_NAME}")

  CUDA_ADD_LIBRARY(${LIB_NAME} SHARED
    ${srcs}
  )

  target_link_libraries(${LIB_NAME}
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${SUBSYS_DEPS}
  )

#  install(TARGETS ${LIB_NAME}
#    LIBRARY DESTINATION lib)
#
#  install(FILES ${incs} ${impl_incs}
#    DESTINATION include/cugl/${SUBSYS_NAME})

endif(BUILD_COMMON)


########################
## CUGL VISUALIZATION ##
########################

if(BUILD_VISUALIZATION)
  set(SUBSYS_NAME visualization)
  set(SUBSYS_DEPS ${PROJECT_NAME}_common)

  set(incs
    "include/${PROJECT_NAME}/${SUBSYS_NAME}/visualizer_base.h"
  )

  set(impl_incs)

  set(srcs
    "src/${SUBSYS_NAME}/visualizer_base.cpp"
  )

  set(LIB_NAME "${PROJECT_NAME}_${SUBSYS_NAME}")

  add_library(${LIB_NAME} SHARED
    ${srcs}
  )

  target_link_libraries(${LIB_NAME}
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${SUBSYS_DEPS}
  )

#  install(TARGETS ${LIB_NAME}
#    LIBRARY DESTINATION lib)
#
#  install(FILES ${incs}
#    DESTINATION include/cugl/${SUBSYS_NAME})

endif(BUILD_VISUALIZATION)

###############
## CUGL APPS ##
###############
if(BUILD_APPS)
  set(SUBSYS_NAME dynamic_fusion)
  set(SUBSYS_DEPS ${PROJECT_NAME}_common ${PROJECT_NAME}_visualization)

  set(APP_NAME "${PROJECT_NAME}_${SUBSYS_NAME}_node")

  set(incs
    "include/${PROJECT_NAME}/apps/${SUBSYS_NAME}/cuda_internal.h"
    "include/${PROJECT_NAME}/apps/${SUBSYS_NAME}/deform_energy_terms.h"
    "include/${PROJECT_NAME}/apps/${SUBSYS_NAME}/deform_voxel_hierarchy.h"
    "include/${PROJECT_NAME}/apps/${SUBSYS_NAME}/dynfu_app.h"
    "include/${PROJECT_NAME}/apps/${SUBSYS_NAME}/dynfu_configure.h"
    "include/${PROJECT_NAME}/apps/${SUBSYS_NAME}/pcg_solver.h"
    "include/${PROJECT_NAME}/apps/${SUBSYS_NAME}/stream_capture.h"
  )

  set(impl_incs)

  set(srcs
    "src/apps/${SUBSYS_NAME}/deform_energy_terms.cpp"
    "src/apps/${SUBSYS_NAME}/deform_voxel_hierarchy.cpp"
    "src/apps/${SUBSYS_NAME}/dynfu_app.cpp"
    "src/apps/${SUBSYS_NAME}/pcg_solver.cpp"
    "src/apps/${SUBSYS_NAME}/stream_capture.cpp"
  )

  set(cuda_srcs
    "src/apps/${SUBSYS_NAME}/cuda/deform_energy_terms.cu"
    "src/apps/${SUBSYS_NAME}/cuda/deform_voxel_hierarchy.cu"
    "src/apps/${SUBSYS_NAME}/cuda/dynfu_core.cu"
    "src/apps/${SUBSYS_NAME}/cuda/imgproc.cu"
    "src/apps/${SUBSYS_NAME}/cuda/pcg_solver.cu"
  )

  CUDA_COMPILE(cuda_objs ${cuda_srcs})

  add_executable(${APP_NAME}
    ${srcs}
    ${cuda_srcs}
    ${cuda_objs}
  )

  target_link_libraries(${APP_NAME}
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${SUBSYS_DEPS}
    realsense
    m
  )

  # add_library(${LIB_NAME} SHARED
  #   ${srcs}
  #   ${cuda_srcs}
  #   ${cuda_objs}
  # )

  # target_link_libraries(${LIB_NAME}
  #   ${CUDA_LIBRARIES}
  #   ${OPENGL_LIBRARIES}
  #   ${OpenCV_LIBRARIES}
  #   ${SUBSYS_DEPS}
  #   realsense
  #   m
  # )

#  install(TARGETS ${LIB_NAME}
#    LIBRARY DESTINATION lib)
#
#  install(FILES ${incs}
#    DESTINATION include/cugl/apps/dynamic_fusion)

endif(BUILD_APPS)

##############
## CUGL ROS ##
##############
if(BUILD_ROS)
  #  set(SUBSYS_NAME dynfu_node)
  #  set(SUBSYS_DEPS ${PROJECT_NAME}_common ${PROJECT_NAME}_visualization ${PROJECT_NAME}_dynamic_fusion)
  #
  #  set(APP_NAME "${PROJECT_NAME}_${SUBSYS_NAME}")
  #
  #  set(incs)
  #
  #  set(impl_incs)
  #
  #  set(srcs
  #    "src/ros/dynfu_node.cpp"
  #  )
  #
  #  add_executable(${APP_NAME}
  #    ${srcs}
  #  )
  #
  #  target_link_libraries(${APP_NAME}
  #    ${catkin_LIBRARIES}
  #    ${SUBSYS_DEPS}
  #  )

#  install(TARGETS ${LIB_NAME}
#    LIBRARY DESTINATION lib)
#
#  install(FILES ${incs}
#    DESTINATION include/cugl/apps/dynamic_fusion)

endif(BUILD_ROS)





## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/cugl_ros.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/cugl_ros_node.cpp)

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
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
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
# catkin_add_gtest(${PROJECT_NAME}-test test/test_cugl_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
