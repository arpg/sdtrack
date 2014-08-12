#pragma once

#include <sdtrack/track.h>
#include <sdtrack/utils.h>
#include <sdtrack/semi_dense_tracker.h>

#include <CVars/CVarVectorIO.h>
#include <Eigen/Core>

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

namespace sdtrack {
struct TrackerPose {
  TrackerPose() { opt_id.resize(2); }

  std::list<std::shared_ptr<DenseTrack>> tracks;
  Sophus::SE3t t_wp;
  Eigen::Vector3t v_w;
  Eigen::Vector6t b;
  std::vector<uint32_t> opt_id;
  double time;
  uint32_t longest_track;
};

inline void GetBaPoseRange(
    const std::vector<std::shared_ptr<sdtrack::TrackerPose>>& poses,
    const uint32_t num_active_poses, uint32_t& start_pose,
    uint32_t& start_active_pose) {
  start_active_pose =
      poses.size() > num_active_poses ? poses.size() - num_active_poses : 0;
  start_pose = poses.size();
  for (uint32_t ii = start_active_pose; ii < poses.size(); ++ii) {
    std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
    // std::cerr << "Start id: " << start_pose << " pose longest track " <<
    //              pose->longest_track << " for pose id " << ii << std::endl;
    start_pose = std::min(ii - (pose->longest_track - 1), start_pose);
  }

  /*
  std::cerr << "Num poses: " << poses.size() << " start pose " <<
               start_pose << " start active pose " << start_active_pose <<
               std::endl;*/
}
}
