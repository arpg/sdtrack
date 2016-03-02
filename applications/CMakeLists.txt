cmake_minimum_required( VERSION 2.8 )
include_directories(common)
add_subdirectory(self_cal)
add_subdirectory(vtracker) 
add_subdirectory(vitracker)
add_subdirectory(vibatch) 
 

find_package(Ceres QUIET)


#only build this if we have ceres
#if(Ceres_FOUND)
#  add_subdirectory(ceres_tracker)
#endif()


