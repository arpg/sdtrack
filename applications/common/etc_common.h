#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sdtrack/track.h>
#include <sdtrack/utils.h>
#include <sdtrack/semi_dense_tracker.h>

#include <CVars/CVarVectorIO.h>

////////////////////////////////////////////////////////////////////////////
// Overloading Eigen for CVars
namespace CVarUtils {
inline std::ostream& operator<<(std::ostream& Stream, Eigen::Vector3d& Mat) {
  unsigned int nRows = Mat.rows();
  unsigned int nCols = Mat.cols();

  Stream << "[ ";

  for (unsigned int ii = 0; ii < nRows - 1; ii++) {
    for (unsigned int jj = 0; jj < nCols - 1; jj++) {
      Stream << Mat(ii, jj);
      Stream << ", ";
    }
    Stream << Mat(ii, nCols - 1);
    Stream << "; ";
  }
  for (unsigned int jj = 0; jj < nCols - 1; jj++) {
    Stream << Mat(nRows - 1, jj);
    Stream << ", ";
  }
  Stream << Mat(nRows - 1, nCols - 1);
  Stream << " ]";

  return Stream;
}

////////////////////////////////////////////////////////////////////////////
inline std::istream& operator>>(std::istream& Stream, Eigen::Vector3d& Mat) {

  unsigned int nRows = Mat.rows();
  unsigned int nCols = Mat.cols();
  char str[256];

  Stream.getline(str, 255, '[');
  if (Stream.gcount() > 1) {
    return Stream;
  }
  for (unsigned int ii = 0; ii < nRows - 1; ii++) {
    for (unsigned int jj = 0; jj < nCols - 1; jj++) {
      Stream.getline(str, 255, ',');
      Mat(ii, jj) = std::strtod(str, NULL);
    }
    Stream.getline(str, 255, ';');
    Mat(ii, nCols - 1) = std::strtod(str, NULL);
  }
  for (unsigned int jj = 0; jj < nCols - 1; jj++) {
    Stream.getline(str, 255, ',');
    Mat(nRows - 1, jj) = std::strtod(str, NULL);
  }
  Stream.getline(str, 255, ']');
  Mat(nRows - 1, nCols - 1) = std::strtod(str, NULL);
  return Stream;
}

inline std::ostream& operator<<(std::ostream& Stream, Eigen::MatrixXd& Mat) {
  int nRows = Mat.rows();
  int nCols = Mat.cols();

  if (nRows && nCols) {
    Stream << "[ ";

    for (int ii = 0; ii < nRows - 1; ii++) {
      for (int jj = 0; jj < nCols - 1; jj++) {
        Stream << Mat(ii, jj);
        Stream << ", ";
      }
      Stream << Mat(ii, nCols - 1);
      Stream << "; ";
    }
    for (int jj = 0; jj < nCols - 1; jj++) {
      Stream << Mat(nRows - 1, jj);
      Stream << ", ";
    }
    Stream << Mat(nRows - 1, nCols - 1);
    Stream << " ]";
  }

  return Stream;
}

////////////////////////////////////////////////////////////////////////////
inline std::istream& operator>>(std::istream& Stream, Eigen::MatrixXd& Mat) {

  int nRows = Mat.rows();
  int nCols = Mat.cols();

  if (nRows && nCols) {
    char str[256];
    Stream.getline(str, 255, '[');
    if (Stream.gcount() > 1) {
      return Stream;
    }
    for (int ii = 0; ii < nRows - 1; ii++) {
      for (int jj = 0; jj < nCols - 1; jj++) {
        Stream.getline(str, 255, ',');
        Mat(ii, jj) = std::strtod(str, NULL);
      }
      Stream.getline(str, 255, ';');
      Mat(ii, nCols - 1) = std::strtod(str, NULL);
    }
    for (int jj = 0; jj < nCols - 1; jj++) {
      Stream.getline(str, 255, ',');
      Mat(nRows - 1, jj) = std::strtod(str, NULL);
    }
    Stream.getline(str, 255, ']');
    Mat(nRows - 1, nCols - 1) = std::strtod(str, NULL);
  }
  return Stream;
}

inline std::ostream& operator<<(std::ostream& Stream,
                                std::vector<Eigen::MatrixXd>& vEigen) {

  if (vEigen.empty()) {
    Stream << "[ ]";
    return Stream;
  }

  Stream << "[ " << vEigen[0];
  for (size_t i = 1; i < vEigen.size(); i++) {
    Stream << " " << vEigen[i];
  }
  Stream << " ]";

  return Stream;
}

inline std::istream& operator>>(std::istream& Stream,
                                std::vector<Eigen::MatrixXd>& vEigen) {

  std::stringstream sBuf;
  Eigen::MatrixXd Mat;

  vEigen.clear();

  char str[256];

  // strip first bracket
  Stream.getline(str, 255, '[');
  if (Stream.gcount() > 1) {
    return Stream;
  }

  // get first element
  Stream.getline(str, 255, '[');
  Stream.getline(str, 255, ']');

  while (Stream.gcount() > 1) {
    sBuf << "[";
    sBuf << str;
    sBuf << "]";
    sBuf >> Mat;
    vEigen.push_back(Mat);

    // get next element
    Stream.getline(str, 255, '[');
    Stream.getline(str, 255, ']');
  }
  return Stream;
}

}

namespace sdtrackUtils {
inline std::ostream& operator<<(std::ostream& Stream,
                                Sophus::SE3d const& pose) {

  Stream << "[ " <<
            pose.translation().transpose() <<
            " | " <<
            pose.rotationMatrix().eulerAngles(0,1,2).transpose() <<
            " ]";

  return Stream;
}


}

namespace sdtrack {
struct TrackerPose {
  TrackerPose() { opt_id.resize(3); }

  std::list<std::shared_ptr<DenseTrack>> tracks;
  Sophus::SE3t t_wp;
  Eigen::Vector3t v_w;
  Eigen::Vector6t b;
  std::vector<uint32_t> opt_id;
  std::vector<Sophus::SE3t> calib_t_wp;
  Eigen::VectorXd cam_params;
  double time;
  uint32_t longest_track;
};

inline void GetBaPoseRange(
    const std::vector<std::shared_ptr<sdtrack::TrackerPose>>& poses,
    const uint32_t num_active_poses, uint32_t& start_pose,
    uint32_t& start_active_pose) {
  start_active_pose =
      poses.size() > num_active_poses ? poses.size() - num_active_poses : 0;

  start_pose = start_active_pose;

  // Go through all the poses in the acitve window and determine
  // the index of the reference frame for the longest track.
  // (so the oldest covisibile landmark will determine the start_pose)
  for (uint32_t ii = start_active_pose; ii < poses.size(); ++ii) {
    std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
    start_pose = std::min(ii - (pose->longest_track - 1), start_pose);
  }
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 6, 1> log_decoupled(
    const Sophus::SE3Group<Scalar>& a) {
  Eigen::Matrix<Scalar, 6, 1> res;
  res.template head<3>() = a.translation();
  res.template tail<3>() = a.so3().log();
  return res;
}

///////////////////////////////////////////////////////////////////////////////
// Convenience functions to convert a pose between vision and robotics
// coordinate conventions.
// Poses used in BA need to be in the vision coordinate convention

inline Sophus::SE3d VisionToRobotics(Sophus::SE3d T_v){
  Sophus::SE3t M_vr;
  M_vr.so3() = calibu::RdfRobotics.inverse();
  return T_v * M_vr;
}
inline Sophus::SE3d RoboticsToVision(Sophus::SE3d T_r){
  Sophus::SE3t M_rv;
  M_rv.so3() = calibu::RdfRobotics;
  return T_r * M_rv;
}

// Converts a realtive pose from vision frame to robotics
inline Sophus::SE3d RelVision2Robotics(Sophus::SE3d T_v){
  Sophus::SO3t M_vr;
  // Vision to robotics RDF
  M_vr = calibu::RdfRobotics;

  T_v.so3() = M_vr*T_v.so3()*M_vr.inverse();
  T_v.translation() = M_vr*T_v.translation();
  return T_v;
}

// Converts a relative pose from robotics frame to vision
inline Sophus::SE3d RelRobotics2Vision(Sophus::SE3d T_r){
  Sophus::SO3t M_rv;
  // Robotics to vision RDF
  M_rv = calibu::RdfRobotics.inverse();

  T_r.so3() = M_rv*T_r.so3()*M_rv.inverse();
  T_r.translation() = M_rv*T_r.translation();
  return T_r;
}

}
