cmake_minimum_required( VERSION 2.8 )
include_directories(common)
find_package(sdtrack REQUIRED)
find_package(ba REQUIRED)
find_package(HAL REQUIRED)
find_package(PbMsgs REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Pangolin REQUIRED)
find_package(SceneGraph REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${SDTRACK_INCLUDE_DIRS}
                    ${BA_INCLUDE_DIRS}
                    ${HAL_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS}
                    ${SceneGraph_INCLUDE_DIRS} 
                    ${Ceres_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR})

def_executable(sd_ceres_tracker
  SOURCES sd_ceres_tracker.cpp
  DEPENDS
  sdtrack
  LINK_LIBS
  ${SDRACK_LIBRARIES}
  ${BA_LIBRARIES}
  ${HAL_LIBRARIES}
  ${PbMsgs_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  ${MINIGLOG_LIBRARIES}
  ${SceneGraph_LIBRARIES}
  ${Pangolin_LIBRARIES}
  ${CERES_LIBRARIES}
  )

def_executable(sd_ceres_calib
  SOURCES sd_ceres_calib.cpp
  DEPENDS
  sdtrack
  LINK_LIBS
  ${SDRACK_LIBRARIES}
  ${BA_LIBRARIES}
  ${HAL_LIBRARIES}
  ${PbMsgs_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  ${MINIGLOG_LIBRARIES}
  ${SceneGraph_LIBRARIES}
  ${Pangolin_LIBRARIES}
  ${CERES_LIBRARIES}
  )
