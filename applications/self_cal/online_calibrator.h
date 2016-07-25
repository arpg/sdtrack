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
#include <thread>


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

struct PriorityQueueParams {
  bool use_imu;
  bool do_tvs;
  std::vector<std::shared_ptr<TrackerPose>> poses;
//  std::list<std::shared_ptr<DenseTrack>> current_tracks;
  uint32_t current_tracks_size;
  CalibrationWindow* overal_window;
  uint32_t num_iterations = 1;
  bool apply_results = false;
  bool rotation_only_Tvs = false;
  void (*callback)(bool);
};

class OnlineCalibrator {
 public:
  OnlineCalibrator();
  void Init(std::mutex* ba_mutex, std::mutex* oc_mutex, calibu::Rig<Scalar>* rig, uint32_t num_windows,
      uint32_t window_length, Eigen::VectorXd covariance_weights,
      double imu_time_offset_in = 0,
      ba::InterpolationBufferT<ba::ImuMeasurementT<double>, double>* buffer =
          nullptr);
  void TestJacobian(Eigen::Vector2t pix, Sophus::SE3t t_ba, Scalar rho);

  template <bool UseImu, bool DoTvs, bool DoAsync>
  bool AnalyzePriorityQueue(
      std::vector<std::shared_ptr<TrackerPose>>& poses,
      uint32_t current_tracks_size,
      CalibrationWindow& overal_window, uint32_t num_iterations = 1,
      bool apply_results = false, bool rotation_only_Tvs = false);

  template <bool UseImu, bool DoTvs, bool PriorityQueue>
  void AddCalibrationWindowToBa(
      std::vector<std::shared_ptr<TrackerPose>>& poses,
      CalibrationWindow& window, int ba_id);

  bool AnalyzeCalibrationWindow(CalibrationWindow& new_window);

  template <bool UseImu, bool DoTvs>
  bool AnalyzeCalibrationWindow(
      std::vector<std::shared_ptr<TrackerPose>>& poses,
      std::list<std::shared_ptr<DenseTrack>>* current_tracks,
      uint32_t start_pose, uint32_t end_pose, CalibrationWindow& window,
      uint32_t num_iterations = 1, bool apply_results = false,
      bool rotation_only_Tvs = false);
  const std::vector<CalibrationWindow>& windows() { return windows_; }
  void ClearQueue() { windows_.clear(); }
  double GetWindowScore(const CalibrationWindow& window, bool rotation_only_Tvs);
  double GetWindowScore(const CalibrationWindow& window);
  std::shared_ptr<PriorityQueueParams> PriorityQueueParameters();
  void DoPriorityQueueThread();


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
  void NotifyConditionVariable(){
    oc_condition_var.notify_all();
  }

  uint32_t NumWindows() { return windows_.size(); }
  uint32_t queue_length() { return queue_length_; }
  bool needs_update() {
    std::lock_guard<std::mutex>oc_lck(*oc_mutex_);
    return needs_update_;
  }
  void SetDebugLevel(const int level){
    debug_level = level;
  }
  std::shared_ptr<std::thread> pq_thread;


private:
  bool needs_update_ = false;
  std::vector<CalibrationWindow> windows_;
  uint32_t queue_length_ = 5;
  uint32_t window_length_ = 10;
  int debug_level = 0;
  int debug_level_threshold = 0;
  calibu::Rig<Scalar>* rig_;
  Eigen::VectorXd covariance_weights_;
  CalibrationWindow total_window_;

  // CANDIDATE WINDOW BA
  // Visual BA with camera parameters
  // Lm size: 1, Pose size: 6, Calibration size: 5
  ba::BundleAdjuster<double, 1, 6, 5> selfcal_ba;
  // VI BA with camera parameters
  ba::BundleAdjuster<double, 1, 15, 5, false> vi_selfcal_ba;
  //VI BA with camera + IMU parameters
  ba::BundleAdjuster<double, 1, 15, 5, true> vi_tvs_selfcal_ba;
  //VI BA with IMU parameters
  ba::BundleAdjuster<double, 1, 15, 0, true> vi_only_tvs_selfcal_ba;


  // PRIORITY QUEUE BA (there need to be separete instances since the PQ
  // runs async)
  // Visual BA with camera parameters for PQ
  // Lm size: 1, Pose size: 6, Calibration size: 5
  ba::BundleAdjuster<double, 1, 6, 5> pq_selfcal_ba;
  // VI BA with camera parameters for PQ
  ba::BundleAdjuster<double, 1, 15, 5, false> pq_vi_selfcal_ba;
  //VI BA with camera + IMU parameters for PQ
  ba::BundleAdjuster<double, 1, 15, 5, true> pq_vi_tvs_selfcal_ba;
  //VI BA with IMU parameters for PQ
  ba::BundleAdjuster<double, 1, 15, 0, true> pq_vi_only_tvs_selfcal_ba;
  ba::InterpolationBufferT<ba::ImuMeasurementT<double>, double>* imu_buffer;
  uint32_t candidate_ba_id_ = 2;
  uint32_t pq_ba_id_ = 3;
  double imu_time_offset;
  std::mutex* ba_mutex_;
  std::mutex* oc_mutex_;
  std::mutex  oc_prioriy_queue_mutex_;
  std::condition_variable oc_condition_var;
  std::shared_ptr<PriorityQueueParams> pq_params_;

  template <bool, bool, bool>
  struct Proxy {};
};

//Candidate window specializations
template <>
struct OnlineCalibrator::Proxy<true, false, false> {
  Proxy(OnlineCalibrator* owner_ptr) : owner(owner_ptr) {}
  OnlineCalibrator* owner;
  decltype(vi_selfcal_ba) & GetBa() const { return owner->vi_selfcal_ba; }
};

template <>
struct OnlineCalibrator::Proxy<false, false, false> {
  Proxy(OnlineCalibrator* owner_ptr) : owner(owner_ptr) {}
  OnlineCalibrator* owner;
  decltype(selfcal_ba) & GetBa() const { return owner->selfcal_ba; }
};

template <>
struct OnlineCalibrator::Proxy<true, true, false> {
  Proxy(OnlineCalibrator* owner_ptr) : owner(owner_ptr) {}
  OnlineCalibrator* owner;
  decltype(vi_only_tvs_selfcal_ba) & GetBa() const { return owner->vi_only_tvs_selfcal_ba; }
};

// Priority Queue Specializations
template <>
struct OnlineCalibrator::Proxy<true, false, true> {
  Proxy(OnlineCalibrator* owner_ptr) : owner(owner_ptr) {}
  OnlineCalibrator* owner;
  decltype(pq_vi_selfcal_ba) & GetBa() const { return owner->pq_vi_selfcal_ba; }
};

template <>
struct OnlineCalibrator::Proxy<false, false, true> {
  Proxy(OnlineCalibrator* owner_ptr) : owner(owner_ptr) {}
  OnlineCalibrator* owner;
  decltype(pq_selfcal_ba) & GetBa() const { return owner->pq_selfcal_ba; }
};

template <>
struct OnlineCalibrator::Proxy<true, true, true> {
  Proxy(OnlineCalibrator* owner_ptr) : owner(owner_ptr) {}
  OnlineCalibrator* owner;
  decltype(pq_vi_only_tvs_selfcal_ba) & GetBa() const { return owner->pq_vi_only_tvs_selfcal_ba; }
};

//template <>
//struct OnlineCalibrator::Proxy<false, true> {
//  Proxy(OnlineCalibrator* owner_ptr) : owner(owner_ptr) {}
//  OnlineCalibrator* owner;
//  decltype(vi_only_tvs_selfcal_ba) & GetBa() const { return owner->vi_only_tvs_selfcal_ba; }
//};
}
