#pragma once
#include <vector>
#include <calibu/cam/camera_crtp.h>
#include <Eigen/Eigenvalues>
#include <ba/BundleAdjuster.h>
#include <ba/InterpolationBuffer.h>
#include <sdtrack/semi_dense_tracker.h>
#include "math_types.h"
#include "etc_common.h"
#include <mutex>

#define LM_DIM 3

namespace Sophus {
typedef SE3Group<Scalar> SE3t;
}

namespace Eigen {
typedef Matrix<Scalar, 2, 1> Vector2t;
}

namespace sdtrack {
struct CalibrationWindow {
  uint32_t start_index = UINT_MAX;
  uint32_t end_index = UINT_MAX;
  double score = DBL_MAX;
  double kl_divergence = 0;
  Eigen::MatrixXd covariance;
  Eigen::VectorXd mean;
  uint32_t num_measurements;
};

class OnlineCalibrator {
 public:
  OnlineCalibrator();
  void Init(
      std::mutex* ba_mutex, calibu::Rig<Scalar>* rig, uint32_t num_windows,
      uint32_t window_length, Eigen::VectorXd covariance_weights,
      double imu_time_offset_in = 0,
      ba::InterpolationBufferT<ba::ImuMeasurementT<double>, double>* buffer =
          nullptr);
  void TestJacobian(Eigen::Vector2t pix, Sophus::SE3t t_ba, Scalar rho);

  template <bool UseImu>
  void AnalyzePriorityQueue(
      std::vector<std::shared_ptr<TrackerPose>>& poses,
      std::list<std::shared_ptr<DenseTrack>>* current_tracks,
      CalibrationWindow& overal_window, uint32_t num_iterations = 1,
      bool apply_results = false);

  template <bool UseImu>
  void AddCalibrationWindowToBa(
      std::vector<std::shared_ptr<TrackerPose>>& poses,
      CalibrationWindow& window);

  bool AnalyzeCalibrationWindow(CalibrationWindow& new_window);

  template <bool UseImu>
  void AnalyzeCalibrationWindow(
      std::vector<std::shared_ptr<TrackerPose>>& poses,
      std::list<std::shared_ptr<DenseTrack>>* current_tracks,
      uint32_t start_pose, uint32_t end_pose, CalibrationWindow& window,
      uint32_t num_iterations = 1, bool apply_results = false);
  const std::vector<CalibrationWindow>& windows() { return windows_; }
  void ClearQueue() { windows_.clear(); }
  double GetWindowScore(const CalibrationWindow& window);

  double ComputeKlDivergence(const CalibrationWindow& window0,
                             const CalibrationWindow& window1);
  void SetPriorityQueueDistribution(const Eigen::MatrixXd& covariance,
                                    const Eigen::VectorXd& mean);

  double ComputeHotellingScore(const CalibrationWindow& window0,
                               const CalibrationWindow& window1);

  double ComputeBhattacharyyaDistance(const CalibrationWindow &window0,
                                      const CalibrationWindow &window1);
  double ComputeYao1965(const CalibrationWindow &window0,
                        const CalibrationWindow &window1);
  double ComputeNelVanDerMerwe1986(const CalibrationWindow &window0,
                                   const CalibrationWindow &window1);

  void SetBaDebugLevel(int level);

  uint32_t NumWindows() { return windows_.size(); }
  uint32_t queue_length() { return queue_length_; }
  bool needs_update() { return needs_update_; }
private:
  bool needs_update_ = false;
  std::vector<CalibrationWindow> windows_;
  uint32_t queue_length_ = 5;
  uint32_t window_length_ = 10;
  calibu::Rig<Scalar>* rig_;
  Eigen::VectorXd covariance_weights_;
  CalibrationWindow total_window_;
  ba::BundleAdjuster<double, 1, 6, 5> selfcal_ba;
  ba::BundleAdjuster<double, 1, 15, 5, false> vi_selfcal_ba;
  ba::InterpolationBufferT<ba::ImuMeasurementT<double>, double>* imu_buffer;
  uint32_t ba_id_ = 2;
  double imu_time_offset;
  std::mutex* ba_mutex_;

  template <bool>
  struct Proxy {};
};

template <>
struct OnlineCalibrator::Proxy<true> {
  Proxy(OnlineCalibrator* owner_ptr) : owner(owner_ptr) {}
  OnlineCalibrator* owner;
  decltype(vi_selfcal_ba) & GetBa() const { return owner->vi_selfcal_ba; }
};

template <>
struct OnlineCalibrator::Proxy<false> {
  Proxy(OnlineCalibrator* owner_ptr) : owner(owner_ptr) {}
  OnlineCalibrator* owner;
  decltype(selfcal_ba) & GetBa() const { return owner->selfcal_ba; }
};
}
