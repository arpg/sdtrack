#pragma once
#include <vector>
#include <calibu/cam/camera_crtp_interop.h>
#include <Eigen/Eigenvalues>
#include <ba/BundleAdjuster.h>
#include <sdtrack/semi_dense_tracker.h>
#include "math_types.h"
#include "etc_common.h"

#define LM_DIM 3

namespace Sophus {
  typedef SE3Group<Scalar> SE3t;
}

namespace Eigen {
  typedef Matrix<Scalar,2,1> Vector2t;
}

namespace sdtrack
{
  struct CalibrationWindow
  {
    uint32_t start_index = UINT_MAX;
    uint32_t end_index = UINT_MAX;
    double score = DBL_MAX;
    double kl_divergence = 0;
    Eigen::MatrixXd covariance;
    Eigen::VectorXd mean;
  };

  class OnlineCalibrator
  {
public:
    OnlineCalibrator();
    void Init(calibu::Rig<Scalar>* rig, uint32_t num_windows, uint32_t window_length,
              Eigen::VectorXd covariance_weights);
    void TestJacobian(Eigen::Vector2t pix, Sophus::SE3t t_ba, Scalar rho);


    void AnalyzePriorityQueue(
        std::vector<std::shared_ptr<TrackerPose>> &poses,
        std::list<std::shared_ptr<DenseTrack>> *current_tracks,
        CalibrationWindow& overal_window,
        uint32_t num_iterations = 1,
        bool apply_results = false);

    void AddCalibrationWindowToBa(
        std::vector<std::shared_ptr<TrackerPose>>& poses,
        const CalibrationWindow& window);

    bool AnalyzeCalibrationWindow(CalibrationWindow& new_window);

    void AnalyzeCalibrationWindow(
        std::vector<std::shared_ptr<TrackerPose>>& poses,
        std::list<std::shared_ptr<DenseTrack>>* current_tracks,
        uint32_t start_pose, uint32_t end_pose,
        CalibrationWindow& window,
        uint32_t num_iterations = 1,
        bool apply_results = false);
    const std::vector<CalibrationWindow>& windows() { return windows_; }
    double GetWindowScore(const CalibrationWindow& window);

    double ComputeKlDivergence(const CalibrationWindow &window0,
                               const CalibrationWindow &window1);
    void SetPriorityQueueDistribution(const Eigen::MatrixXd& covariance,
                              const Eigen::VectorXd& mean);
  private:
    std::vector<CalibrationWindow> windows_;
    uint32_t num_windows_ = 5;
    uint32_t window_length_ = 10;
    calibu::Rig<Scalar>* rig_;
    Eigen::VectorXd covariance_weights_;
    CalibrationWindow total_window_;
    ba::BundleAdjuster<double, 1, 6, 5> selfcal_ba;
    uint32_t ba_id_ = 1;
  };
}
