# Compute paths
get_filename_component( PACKAGE_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH )
SET( BA_INCLUDE_DIRS ";/usr/include/eigen3;/home/smcguire/research/Calibu/src/../include;/home/smcguire/research/Calibu/build/src/include;/usr/include;/usr/include/opencv2;/usr/include/opencv2/core;/usr/include/opencv2/imgproc;/usr/include/opencv2/features2d;/usr/include/opencv2/nonfree;/usr/include/opencv2/flann;/usr/include/opencv2/calib3d;/usr/include/opencv2/objdetect;/usr/include/opencv2/legacy;/usr/include/opencv2/contrib;/usr/include/opencv2/highgui;/usr/include/opencv2/ml;/usr/include/opencv2/video;/usr/include/opencv2/gpu;/home/smcguire/research/Calibu/build/src/include/;/usr/include/eigen3;/home/smcguire/research/Sophus;/usr/include/eigen3;/home/smcguire/research/Sophus;/usr/include/eigen3" )
SET( BA_INCLUDE_DIR  ";/usr/include/eigen3;/home/smcguire/research/Calibu/src/../include;/home/smcguire/research/Calibu/build/src/include;/usr/include;/usr/include/opencv2;/usr/include/opencv2/core;/usr/include/opencv2/imgproc;/usr/include/opencv2/features2d;/usr/include/opencv2/nonfree;/usr/include/opencv2/flann;/usr/include/opencv2/calib3d;/usr/include/opencv2/objdetect;/usr/include/opencv2/legacy;/usr/include/opencv2/contrib;/usr/include/opencv2/highgui;/usr/include/opencv2/ml;/usr/include/opencv2/video;/usr/include/opencv2/gpu;/home/smcguire/research/Calibu/build/src/include/;/usr/include/eigen3;/home/smcguire/research/Sophus;/usr/include/eigen3;/home/smcguire/research/Sophus;/usr/include/eigen3" )

# Library dependencies (contains definitions for IMPORTED targets)
if(NOT TARGET "" AND NOT BA_BINARY_DIR)
  include( "${PACKAGE_CMAKE_DIR}/BATargets.cmake" )
endif()

SET(BA_LIBRARIES ba)
SET(BA_LIBRARY ba)
SET(BA_INCLUDE_DIRS /home/smcguire/research/ba/src/../include;/usr/include/eigen3;/home/smcguire/research/Calibu/src/../include;/home/smcguire/research/Calibu/build/src/include;/usr/include;/usr/include/opencv2;/usr/include/opencv2/core;/usr/include/opencv2/imgproc;/usr/include/opencv2/features2d;/usr/include/opencv2/nonfree;/usr/include/opencv2/flann;/usr/include/opencv2/calib3d;/usr/include/opencv2/objdetect;/usr/include/opencv2/legacy;/usr/include/opencv2/contrib;/usr/include/opencv2/highgui;/usr/include/opencv2/ml;/usr/include/opencv2/video;/usr/include/opencv2/gpu;/home/smcguire/research/Calibu/build/src/include/;/usr/include/eigen3;/home/smcguire/research/Sophus;/usr/include/eigen3;/home/smcguire/research/Sophus;/usr/include/eigen3)
SET(BA_LINK_DIRS )
