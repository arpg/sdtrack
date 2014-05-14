cmake_minimum_required( VERSION 2.8 )
include_directories(common)
find_package(sdtrack REQUIRED)
find_package(ba REQUIRED)
find_package(HAL REQUIRED)
find_package(PbMsgs REQUIRED)
find_package(Pangolin REQUIRED)
find_package(SceneGraph REQUIRED)

include_directories(${SDTRACK_INCLUDE_DIRS}
                    ${BA_INCLUDE_DIRS}
                    ${HAL_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS}
                    ${SceneGraph_INCLUDE_DIRS} 
                    ${CMAKE_CURRENT_SOURCE_DIR})
link_libraries(${SDRACK_LIBRARIES}
               ${BA_LIBRARIES}
               ${HAL_LIBRARIES}
               ${SceneGraph_LIBRARIES}
               ${Pangolin_LIBRARIES})

def_executable(sd_vtracker
  SOURCES sd_vtracker.cpp 
  DEPENDS
  ba
  sdtrack
  )
