project(sdtrack)
cmake_minimum_required(VERSION 2.8)

if( ANDROID )
  set(REAL_TYPE float CACHE STRING "Floating point type")
else()
  set(REAL_TYPE double CACHE STRING "Floating point type")
endif()
#add_definitions(-DCHECK_NANS)
# Add to module path, so we can find our cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)

include(def_library)
include(export_package)
include(def_executable)

string( TOLOWER ${PROJECT_NAME} LIBRARY_NAME )

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOS_OBJECT_USE_OBJC=0")
endif()

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -Wall -Wextra -Wno-unused-parameter")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wno-unused-parameter")

find_package(Calibu 0.1 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV2 REQUIRED)
find_package(MINIGLOG REQUIRED)
find_package(CVARS REQUIRED)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SDTRACKERConfig.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/SDTRACKERConfig.h)

set(PROJ_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV2_INCLUDE_DIRS}
  ${Calibu_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIR}
  ${MINIGLOG_INCLUDE_DIRS}
  ${CVARS_INCLUDE_DIR}
  )

set(PROJ_LIBRARIES
  ${Sophus_LIBRARIES}
  ${Calibu_LIBRARY}
  ${OpenCV2_LIBRARIES}
  ${CVARS_LIBRARY}
  )

include_directories(${PROJ_INCLUDE_DIRS})
set(INC_PREFIX ${CMAKE_SOURCE_DIR}/include/sdtrack) 
set(SDTRACKER_HDRS
    ${INC_PREFIX}/utils.h
    ${INC_PREFIX}/track.h
    ${INC_PREFIX}/semi_dense_tracker.h
    ${INC_PREFIX}//options.h
    ${INC_PREFIX}/keypoint.h)

set(SDTRACKER_SRCS
    ${CMAKE_SOURCE_DIR}/src/semi_dense_tracker.cpp)

def_library(${LIBRARY_NAME}
  SOURCES ${SDTRACKER_HDRS} ${SDTRACKER_SRCS}
  LINK_LIBS ${PROJ_LIBRARIES} 
  PACKAGE sdtrack 
  )

# Create the SDTRACKERConfig.cmake file for the build tree.
set(EXPORT_SDTRACKER_INC
  ${PROJ_INCLUDE_DIRS}
  )

export_package(sdtrack
  TARGETS ${LIBRARY_NAME}
  LIBRARY ${LIBRARY_NAME}
  LIBRARIES ${LIBRARY_NAME}
  INCLUDE_DIRS ${EXPORT_SDTRACKER_INC}
  )

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sdtrackerConfig.h"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${LIBRARY_NAME}
  )

# For the examples to find this and only this version of sdtracker
set(SDTRACKER_DIR ${CMAKE_CURRENT_BINARY_DIR})

option(BUILD_APPLICATIONS "Build Applications" OFF)

if( BUILD_APPLICATIONS )
  add_subdirectory(applications)
endif()
