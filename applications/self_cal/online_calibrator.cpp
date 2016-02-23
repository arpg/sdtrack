#include <glog/logging.h>
#include "online_calibrator.h"
#include "ftest.h"

using namespace sdtrack;

OnlineCalibrator::OnlineCalibrator()
{

}

void OnlineCalibrator::Init(std::mutex* ba_mutex,
                            calibu::Rig<Scalar> *rig,
                            uint32_t num_windows,
                            uint32_t window_length,
                            Eigen::VectorXd covariance_weights,
                            double imu_time_offset_in,
                            ba::InterpolationBufferT<
                            ba::ImuMeasurementT<double>, double>* buffer)
{
  ba_mutex_ = ba_mutex;
  imu_time_offset = imu_time_offset_in;
  imu_buffer = buffer;
  queue_length_ = num_windows;
  window_length_ = window_length;
  rig_ = rig;
  covariance_weights_ = covariance_weights;
  // Take the square root as we must pre/post multiply by the values.
  for (int ii = 0; ii < covariance_weights_.rows(); ++ii) {
    covariance_weights_[ii] = sqrt(covariance_weights_[ii]);
  }
  total_window_.mean = Eigen::VectorXd(covariance_weights.rows());
  total_window_.mean.setZero();
  total_window_.covariance = Eigen::MatrixXd(covariance_weights.rows(),
                                             covariance_weights.rows());
  total_window_.covariance.setIdentity();
  // Initial distribution: total uncertainty.
  total_window_.covariance *= 1e6;
  windows_.clear();
  windows_.reserve(num_windows);

  selfcal_ba.debug_level_threshold = -1;
  vi_selfcal_ba.debug_level_threshold = -1;
  vi_tvs_selfcal_ba.debug_level_threshold = -1;
  vi_only_tvs_selfcal_ba.debug_level_threshold = -1;
}

void OnlineCalibrator::TestJacobian(Eigen::Vector2t pix,
                                    Sophus::SE3t t_ba,
                                    Scalar rho)
{
  std::shared_ptr<calibu::CameraInterface<Scalar>> cam = rig_->cameras_[0];
  Eigen::VectorXd params = cam->GetParams();
  const double eps = 1e-6;
  Eigen::Matrix<Scalar, 2, Eigen::Dynamic> jacobian_fd(2, cam->NumParams());
  // Test the transfer jacobian.
  for (uint32_t ii = 0 ; ii < cam->NumParams()  ; ++ii) {
    const double old_param = params[ii];
    // modify the parameters and transfer again.
    params[ii] = old_param + eps;
    const Eigen::Vector3t ray_plus = cam->Unproject(pix);
    const Eigen::Vector2t pix_plus = cam->Transfer3d(t_ba, ray_plus, rho);

    params[ii] = old_param - eps;
    const Eigen::Vector3t ray_minus = cam->Unproject(pix);
    const Eigen::Vector2t pix_minus = cam->Transfer3d(t_ba, ray_minus, rho);

    jacobian_fd.col(ii) = (pix_plus - pix_minus) / (2 * eps);
    params[ii] = old_param;
  }

  // Now compare the two jacobians.
  auto jacobian = cam->dTransfer_dparams(t_ba, pix, rho);
  std::cerr << "jacobian:\n" << jacobian << "\n jacobian_fd:\n" <<
               jacobian_fd << "\n error: " << (jacobian - jacobian_fd).norm() <<
               std::endl;
}

template<bool UseImu, bool DoTvs>
void OnlineCalibrator::AnalyzePriorityQueue(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    CalibrationWindow &overal_window,
    uint32_t num_iterations, bool apply_results)
{
  Eigen::VectorXd cam_params_backup = rig_->cameras_[0]->GetParams();
  Proxy<UseImu, DoTvs> ba_proxy(this);
  auto& ba = ba_proxy.GetBa();

  ba::Options<double> options;
  options.write_reduced_camera_matrix = false;
  options.projection_outlier_threshold = 1.0;
  options.use_dogleg = true;
  options.use_triangular_matrices = true;
  /// ZZZZZZ FIGURE OUT WHY USING SPARSE HERE FAILS SOMETIMES
  options.use_sparse_solver = false;
  options.calculate_calibration_marginals = true;
  options.error_change_threshold = 1e-6;

  ba.Init(options, poses.size(),
                  current_tracks->size() * poses.size());
  ba.AddCamera(rig_->cameras_[0]);

  // Add all the windows to ba.
  {
    std::lock_guard<std::mutex> lock(*ba_mutex_);
    for (CalibrationWindow& window : windows_) {
      if (window.start_index < poses.size() &&
          window.end_index < poses.size()) {
        AddCalibrationWindowToBa<UseImu, DoTvs>(poses, window);
      }
    }
  }

  ba.Solve(num_iterations);


  // Obtain the mean from the BA.
  if(DoTvs && UseImu){
    overal_window.mean = ba.rig()->cameras_[0]->Pose().log();
  }else{
    overal_window.mean = ba.rig()->cameras_[0]->GetParams();
  }

  overal_window.covariance =
      ba.GetSolutionSummary().calibration_marginals;

  // At this point the BA rig t_wc_ does not update the external one, so we
  // have to manually do it.
  {
    std::lock_guard<std::mutex> lock(*ba_mutex_);
    if (apply_results) {
      rig_->cameras_[0]->Pose() = ba.rig()->cameras_[0]->Pose();
      std::cerr << "new PQ t_wc\n:" << rig_->cameras_[0]->Pose().matrix()
                << std::endl;
    } else {
      rig_->cameras_[0]->SetParams(cam_params_backup);
    }
  }

  needs_update_ = false;
}

template<bool UseImu, bool DoTvs>
void OnlineCalibrator::AddCalibrationWindowToBa(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    CalibrationWindow &window)
{
  window.num_measurements = 0;
  Proxy<UseImu, DoTvs> ba_proxy(this);
  auto& ba = ba_proxy.GetBa();
  const uint32_t start_active_pose = window.start_index;

  // First find the longest track id in the base frame. We will use this
  // pose to remove the scale nullspace.
  uint32_t max_keypoints = 0;
  std::shared_ptr<DenseTrack> longest_track = nullptr;
  std::shared_ptr<TrackerPose> pose = poses[window.start_index];
  for (std::shared_ptr<DenseTrack> track : pose->tracks) {
    if (track->keypoints.size() > max_keypoints) {
      max_keypoints = track->keypoints.size();
      longest_track = track;
    }
  }

  // First add all the poses and landmarks to ba.
  for (uint32_t ii = window.start_index; ii < window.end_index ; ++ii) {
    std::shared_ptr<TrackerPose> pose = poses[ii];
    // If not using an IMU, the first pose of the window is inactive. Otherwise
    // all poses are active and we do a manual regularization below.
    pose->opt_id[ba_id_] = ba.AddPose(
          pose->t_wp, Eigen::VectorXt(), pose->v_w, pose->b,
          (UseImu ? ii >= start_active_pose : ii > start_active_pose),
          pose->time);

    /// ZZZZZZZZ: This is problematic. What if track size was zero but the pose
    /// had projection residuals? we don't want to regularize in this case
    if (pose->tracks.size() == 0 && !UseImu) {
      ba.RegularizePose(pose->opt_id[ba_id_], true, false, false, true);
    }

    if (UseImu && ii > start_active_pose && ii > 0) {
      std::vector<ba::ImuMeasurementT<Scalar>> meas =
          imu_buffer->GetRange(poses[ii - 1]->time, pose->time);

      /*std::cerr << "Adding OC imu residual between poses " << ii - 1 << " with "
                 " time " << poses[ii - 1]->time <<  " and " << ii <<
                 " with time " << pose->time << " with " << meas.size() <<
                 " measurements" << std::endl;*/


      ba.AddImuResidual(poses[ii - 1]->opt_id[ba_id_],
          pose->opt_id[ba_id_], meas);

      // If using an IMU and this is the first pose in the window, regularize
      // the translation and gravity rotation so there are no nullspaces
      if (ii == start_active_pose) {
        /*std::cerr << "OC regularizing gravity and trans and bias of pose " <<
                     pose->opt_id[ba_id_] << std::endl;*/
        ba.RegularizePose(pose->opt_id[ba_id_], true, true, false, false);
      }
    }

    for (std::shared_ptr<DenseTrack> track: pose->tracks) {
      if (track->num_good_tracked_frames == 1 || track->is_outlier) {
        track->external_id[ba_id_] = UINT_MAX;
        continue;
      }

      Eigen::Vector4d ray;
      ray.head<3>() = track->ref_keypoint.ray;
      ray[3] = track->ref_keypoint.rho;
      ray = sdtrack::MultHomogeneous(pose->t_wp  * rig_->cameras_[0]->Pose(), ray);
      bool active = longest_track == nullptr ? true :
        (UseImu ? true : track->id != longest_track->id);
      track->external_id[ba_id_] =
          ba.AddLandmark(ray, pose->opt_id[ba_id_], 0, active);
      // std::cerr << "Adding lm with opt_id " << track->external_id << " and "
      //              " x_r " << ray.transpose() << " and x_orig: " <<
      //              x_r_orig_->transpose() << std::endl;
    }
  }

  // Now add all reprojections to ba)
  for (uint32_t ii = window.start_index ; ii < window.end_index ; ++ii) {
    std::shared_ptr<TrackerPose> pose = poses[ii];
    for (std::shared_ptr<DenseTrack> track : pose->tracks) {
      if (track->external_id[ba_id_] == UINT_MAX) {
        continue;
      }
      // Limit the number of measurements we add here for a track, as they
      // could exceed the size of this calibration window.
      for (size_t jj = 0; jj < track->keypoints.size() &&
           jj < (window.end_index - ii) ; ++jj) {
        if (track->keypoints[jj][0].tracked) {
          const Eigen::Vector2d& z = track->keypoints[jj][0].kp;
          ba.AddProjectionResidual(
                z, pose->opt_id[ba_id_] + jj, track->external_id[ba_id_], 0);
          window.num_measurements++;
        }
      }
    }
  }
}

bool OnlineCalibrator::AnalyzeCalibrationWindow(
    CalibrationWindow &new_window)
{
  std::cerr << "Analyzing window with score " << new_window.score <<
               " start: " << new_window.start_index << " end: " <<
               new_window.end_index << std::endl;

  // Do not add a degenerate window.
  if (new_window.score == 0) {
    return false;
  }

  // Go through all the windows and see if this one beats the one with the
  // highest score. We only consider windows with at most 1 overlap.

  uint32_t max_id = UINT_MAX;
  uint32_t overlap_id = UINT_MAX;
  double max_score = 0;
  uint32_t num_overlaps = 0;
  for (size_t ii = 0; ii < windows_.size() ; ++ii){
    CalibrationWindow& window = windows_[ii];
    // std::cerr << "\t comparing with window " << ii << " with score : " <<
    //              windows_[ii].score << " start " <<  windows_[ii].start_index <<
    //              " end " << windows_[ii].end_index << std::endl;
    if (!((new_window.start_index < window.start_index && new_window.end_index <
           window.start_index) || (new_window.start_index > window.end_index &&
                                   new_window.end_index > window.end_index) || window.score == DBL_MAX)) {
      num_overlaps++;
      std::cerr << "Overlap detected between " << window.start_index << ", " <<
                   window.end_index << " and " << new_window.start_index <<
                   ", " << new_window.end_index << std::endl;

      overlap_id = ii;
    }

    if (num_overlaps > 1) {
      std::cerr << "Num overlaps: " << num_overlaps << " rejecting window. " <<
                   std::endl;
      // Then this window intersects with more than one other window.
      // We cannot continue.
      max_id = UINT_MAX;
      break;
    }

    if (window.score >= max_score) {
      max_score = window.score;
      max_id = ii;
    }
  }

  // If we haven't got a full queue, insert the window if it doesn't overlap.
  if (windows_.size() < queue_length_) {
    if (num_overlaps == 0) {
      windows_.push_back(new_window);
      std::cerr << "Pushing back non overlapping window into position " <<
                   windows_.size() << std::endl;
      needs_update_ = true;
      return true;
    } else {
      std::cerr << "Rejecting window as queue is incomplete and window "
                   "overlaps" << std::endl;
      return false;
    }
  } else {
    // If we overlap, consider this segment if overlaps are allowed.
    if (num_overlaps == 1) {
      max_id = overlap_id;
      max_score = windows_[max_id].score;
      std::cerr << "Overlapping with 1 segment. Using overlapping score " <<
                   max_score << " and id " << max_id << std::endl;
    }

    // Calculate the margin by which this candidate window beats what's in the
    // priority queue.
    const double margin = (max_score - new_window.score) / max_score;
    std::cerr << "Max score: " << max_score<< " margin: " << margin << std::endl;

    // Replace it if it beats a non-overlapping window.
    if (max_id != UINT_MAX && margin > 0.05) {
      const CalibrationWindow& old_window = windows_[max_id];
      std::cerr << "Replaced window at idx " << max_id << " with score " <<
                   old_window.score << " start: " << old_window.start_index <<
                   " end: " << old_window.end_index << " with score " <<
                   new_window.score << " start: " << new_window.start_index <<
                   " end: " << new_window.end_index << std::endl;
      windows_[max_id] = new_window;
      needs_update_ = true;
      return true;
    }
    return false;
  }
}

//bool OnlineCalibrator::AnalyzeCalibrationWindow(CalibrationWindow &new_window)
//{
//  std::cerr << "Analyzing window with score " << new_window.score <<
//               " start: " << new_window.start_index << " end: " <<
//               new_window.end_index << std::endl;
//  std::cerr << "\tComputing kl divergence for candidate window" << std::endl;
//  new_window.kl_divergence = ComputeKlDivergence(total_window_,
//                                                 new_window);
//  std::cerr << "\t KL divergence for new window: " <<
//               new_window.kl_divergence << std::endl;

//  if (windows_.size() < num_windows_) {
//    windows_.push_back(new_window);
//    return true;
//  }

//  // Go through all the windows and see if this one beats the one with the
//  // highest score. We only consider windows with at most 1 overlap.

//  uint32_t min_id = UINT_MAX;
//  double max_score = 0;
//  uint32_t num_overlaps = 0;
//  for (size_t ii = 0; ii < windows_.size() ; ++ii){
//    CalibrationWindow& window = windows_[ii];
//    if (!((new_window.start_index < window.start_index && new_window.end_index <
//        window.start_index) || (new_window.start_index > window.end_index &&
//        new_window.end_index > window.end_index) || window.score == DBL_MAX)) {
//      num_overlaps++;
//    }

//    if (num_overlaps > 1) {
//      std::cerr << "Num overlaps: " << num_overlaps << " rejecting window. " <<
//                   std::endl;
//      // Then this window intersects with more than one other window.
//      // We cannot continue.
//      min_id = UINT_MAX;
//      break;
//    }

//    if (window.kl_divergence >= max_score) {
//      max_score = window.kl_divergence;
//      min_id = ii;
//    }
//  }

//  // Calculate the margin by which this candidate window beats what's in the
//  // priority queue.
//  const double margin = (new_window.kl_divergence - max_score) / max_score;

//  // Replace it if it beats a non-overlapping window.
//  if (min_id != UINT_MAX && margin > 0.05) {
//    const CalibrationWindow& old_window = windows_[min_id];
//    std::cerr << "Replaced window at idx " << min_id << " with kl " <<
//                 old_window.kl_divergence << " start: " <<
//                 old_window.start_index << " end: " << old_window.end_index <<
//                 " with kl " << new_window.kl_divergence << " start: " <<
//                 new_window.start_index << " end: " << new_window.end_index <<
//                 std::endl;
//    std::cerr << "\t KL divergence for old window: " <<
//                 windows_[min_id].kl_divergence << std::endl;
//    windows_[min_id] = new_window;
//    return true;
//  }
//  return false;
//}

double OnlineCalibrator::ComputeBhattacharyyaDistance(
    const CalibrationWindow& window0,
    const CalibrationWindow& window1)
{
  const double n0 = window0.num_measurements;
  const double n1 = window1.num_measurements;
  if (n0 == 0 || n1 == 0 ) {
    return 0;
  }

  const Eigen::MatrixXd cov_pooled =
      ((n0 - 1) * window0.covariance + (n1) * window1.covariance) /
      (n0 + n1 - 2);
  const Eigen::VectorXd mean_diff = window0.mean - window1.mean;
  return 1.0 / 8.0 * mean_diff.transpose() * cov_pooled.inverse() * mean_diff;
}

///
/// \brief One solution two the multivariate Behrens–Fisher problem.
/// \param window0
/// \param window1
/// \return
///
double OnlineCalibrator::ComputeYao1965(
    const CalibrationWindow& window0,
    const CalibrationWindow& window1)
{
  const double p = window0.covariance.rows();
  const double n0 = window0.num_measurements;
  const double n1 = window1.num_measurements;
  const Eigen::MatrixXd& s0 = window0.covariance;
  const Eigen::MatrixXd& s1 = window1.covariance;

  double p_score = 1.0;
  double f = 0, t2 = 0;
  //// ZZZ IMPLEMENT CONDITION NUMBER INSTEAD OF RANK HERE
  if (n0 == 0 || n1 == 0 || p == 0 || s0.fullPivLu().rank() != p ||
      s1.fullPivLu().rank() != p) {
    p_score = 1.0;
  } else {
    const Eigen::MatrixXd s_inv = (s0 + s1).inverse();
    const Eigen::VectorXd xd = window0.mean - window1.mean;
    t2 = xd.transpose() * s_inv * xd;
    const double v_denom = t2;
    const double v = 1.0 /
        ((1.0 / n0) *
        powi((double)(xd.transpose() * s_inv * s0 * s_inv * xd) / v_denom, 2) +
        (1.0 / n1) *
        powi((double)(xd.transpose() * s_inv * s1 * s_inv * xd) / v_denom, 2));

    f = t2 / ((v * p) / (v - p + 1));
    p_score = compute_p_score(f, p, v - p + 1);

  }
  if (p_score < 0.5 || isnan(p_score) || isinf(p_score) || p_score == 1.0) {
    std::cerr << "computing p score for f " << f << " p: " << p << " n0 " <<
                 n0 << " n1 " << n1 << " with t^2 " << t2 <<
                 " p_score: " << p_score << std::endl;
    std::cerr << "with cov0:\n" << window0.covariance << std::endl;
    std::cerr << "with mean0:\n" << window0.mean.transpose() << std::endl;
    std::cerr << "with cov1:\n" << window1.covariance << std::endl;
    std::cerr << "with mean1:\n" << window1.mean.transpose() << std::endl;
  }
  return p_score;
}

double OnlineCalibrator::ComputeNelVanDerMerwe1986(
    const CalibrationWindow& window0,
    const CalibrationWindow& window1)
{
  const double p = window0.covariance.rows();
  const double n0 = window0.num_measurements;
  const double n1 = window1.num_measurements;
  const Eigen::MatrixXd& s0 = (1.0 / n0) * (1.0 / (n0 - 1.0)) *
      window0.covariance;
  const Eigen::MatrixXd& s1 = (1.0 / n1) * (1.0 / (n1 - 1.0)) *
      window1.covariance;

  //// ZZZ IMPLEMENT CONDITION NUMBER INSTEAD OF RANK HERE
  if (n0 == 0 || n1 == 0 || p == 0 || s0.fullPivLu().rank() != p ||
      s1.fullPivLu().rank() != p) {
    return 1.0;
  }

  const Eigen::MatrixXd s = (s0 + s1);
  const Eigen::MatrixXd s_inv = s.inverse();
  const Eigen::MatrixXd s_2 = s * s;
  const Eigen::MatrixXd s0_2 = s0 * s0;
  const Eigen::MatrixXd s1_2 = s1 * s1;
  const Eigen::VectorXd xd = window0.mean - window1.mean;
  const double v = (s_2.trace() + powi(s.trace(), 2)) /
      ((1.0 / n0 * (s0_2.trace() + powi(s0.trace(), 2))) +
        (1.0 / n1 * (s1_2.trace() + powi(s1.trace(), 2))));
  const double t2 = xd.transpose() * s_inv * xd;
  const double f = t2 / ((v * p) / (v - p + 1));
  const double p_score = compute_p_score(f, p, v - p + 1);

  if (p_score < 0.1 || isnan(p_score) || isinf(p_score)) {
    std::cerr << "computing p score for f " << f << " p: " << p << " n0 " <<
                 n0 << " n1 " << n1 << " with t^2 " << t2 <<
                 " p_score: " << p_score << std::endl;
    std::cerr << "with cov0:\n" << window0.covariance << std::endl;
    std::cerr << "with mean0:\n" << window0.mean.transpose() << std::endl;
    std::cerr << "with cov1:\n" << window1.covariance << std::endl;
    std::cerr << "with mean1:\n" << window1.mean.transpose() << std::endl;
  }
  return p_score;
}

void OnlineCalibrator::SetBaDebugLevel(int level)
{
   selfcal_ba.debug_level_threshold = level;
   vi_selfcal_ba.debug_level_threshold = level;
   vi_tvs_selfcal_ba.debug_level_threshold = level;
   vi_only_tvs_selfcal_ba.debug_level_threshold = level;
}

double OnlineCalibrator::ComputeHotellingScore(
    const CalibrationWindow& window0,
    const CalibrationWindow& window1)
{
  const double p = window0.covariance.rows();
  const double n0 = window0.num_measurements;
  const double n1 = window1.num_measurements;
  if (n0 == 0 || n1 == 0 || p == 0) {
    return 0;
  }

  const Eigen::MatrixXd cov_pooled =
      ((n0 - 1) * window0.covariance + (n1) * window1.covariance) /
      (n0 + n1 - 2);
  const Eigen::VectorXd mean_diff = window0.mean - window1.mean;
  const double t_squared = mean_diff.transpose() *
      (cov_pooled * (1.0 / n0 + 1.0 / n1)).inverse() * mean_diff;
  const double f_val = (n0 + n1 - p - 1.0) / (p * (n0 + n1 - 2.0)) * t_squared;
  const double p_score = compute_p_score(f_val, p, n0 + n1 - p - 1);
  return p_score;
}

double OnlineCalibrator::ComputeKlDivergence(
    const CalibrationWindow& window0,
    const CalibrationWindow& window1)
{
  const Eigen::MatrixXd cov0 =  covariance_weights_.asDiagonal() *
      window0.covariance * covariance_weights_.asDiagonal();
  const Eigen::MatrixXd cov1 =  covariance_weights_.asDiagonal() *
      window1.covariance * covariance_weights_.asDiagonal();
  const Eigen::VectorXd mean0 =
      covariance_weights_.array() * window0.mean.array();
  const Eigen::VectorXd mean1 =
      covariance_weights_.array() * window1.mean.array();
  // std::cerr << "\tComputing KL divergence between\n " << cov0 <<
  //              "\t \n and \n\t" << cov1 << std::endl;
  const Eigen::VectorXd mean1_sub_mean0 = mean1 - mean0;
  // std::cerr << "\tMean diff. is " << mean1_sub_mean0.transpose() << std::endl;
  // std::cerr << "\tCov inverse is\n" << cov1.inverse() << std::endl;
  const Eigen::MatrixXd cov1_inv = cov1.inverse();
  return (cov1_inv * cov0).trace() + mean1_sub_mean0.transpose() *
      cov1_inv * mean1_sub_mean0 - mean1_sub_mean0.rows() -
      log(cov0.determinant() / cov1.determinant());
}

void OnlineCalibrator::SetPriorityQueueDistribution(
    const Eigen::MatrixXd &covariance, const Eigen::VectorXd &mean)
{
  total_window_.covariance = covariance;
  total_window_.mean = mean;

  // Now calculate the KL divergence of all of the windows to the batch
  // solution.
  /*
  for (int ii = 0; ii < windows_.size() ; ++ii) {
    windows_[ii].kl_divergence = ComputeKlDivergence(total_window_,
                                                     windows_[ii]);
    std::cerr << "\t KL divergence for window " << ii << " is " <<
                 windows_[ii].kl_divergence << std::endl;

  }*/
}

double OnlineCalibrator::GetWindowScore(const CalibrationWindow& window)
{

  if (window.covariance.fullPivLu().rank() == covariance_weights_.rows()) {
    // First transform the covariance given the weights.
    return
        (covariance_weights_.asDiagonal() * window.covariance *
         covariance_weights_.asDiagonal()).determinant();
  } else {
    return 0;
  }
}

template<bool UseImu, bool DoTvs>
void OnlineCalibrator::AnalyzeCalibrationWindow(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    uint32_t start_pose, uint32_t end_pose,
    CalibrationWindow &window,
    uint32_t num_iterations, bool apply_results)
{
  Eigen::VectorXd cam_params_backup = rig_->cameras_[0]->GetParams();

  std::cerr << "Analyzing calibration window with imu = " << UseImu <<
               " from " << start_pose << " to " << end_pose << std::endl;
  Proxy<UseImu, DoTvs> ba_proxy(this);
  auto& ba = ba_proxy.GetBa();

  window.start_index = start_pose;
  window.end_index = end_pose;

  ba::Options<double> options;
  options.write_reduced_camera_matrix = false;
  options.projection_outlier_threshold = 1.0;
  options.trust_region_size = 10;
  options.use_dogleg = true;
  options.use_sparse_solver = false;//(end_pose - start_pose) > window_length_ * 10;
  /// ZZZZ WHY IS THIS NEEDED? SPARSE IS BROKEN WITH TRIANGULAR IT SEEMS.
  options.use_triangular_matrices = !options.use_sparse_solver;
  options.calculate_calibration_marginals = true;
  options.error_change_threshold = 1e-6;
  options.trust_region_size = 100;

  // Do a bundle adjustment on the current set
  if (current_tracks && poses.size() > 1) {
    // If using the IMU, we have to do a special regularization step for each
    // candidate window, and so we disable the automatic one as it will only
    // do so for the one root pose.
    if (UseImu) {
      options.enable_auto_regularization = false;
    }

    ba.Init(options, poses.size(),
                    current_tracks->size() * poses.size());

    {
      ba.AddCamera(rig_->cameras_[0]);
      std::lock_guard<std::mutex> lock(*ba_mutex_);
      AddCalibrationWindowToBa<UseImu, DoTvs>(poses, window);
    }

    Sophus::SE3t M_vr;
    M_vr.so3() = calibu::RdfRobotics.inverse();

    if(DoTvs){
      Sophus::SE3t t_vs = ba.rig()->cameras_[0]->Pose();
      t_vs = t_vs * M_vr;
      std::cerr << "PRE BA Tvs is:\n" << t_vs.matrix() << std::endl;
    }

    // Optimize the poses and calibration parameters
    ba.Solve(num_iterations);

    const ba::SolutionSummary<double>& summary =
        ba.GetSolutionSummary();

    // Obtain the mean from the BA
    if(DoTvs){
      Eigen::MatrixXd imu_parameters = ba.rig()->cameras_[0]->Pose().log();
      window.mean = imu_parameters;

      Sophus::SE3t t_vs = ba.rig()->cameras_[0]->Pose();
      t_vs = t_vs * M_vr;
      std::cerr << "POST BA Tvs is:\n" << t_vs.matrix() << std::endl;
    }else{
      window.mean = ba.rig()->cameras_[0]->GetParams();
    }

    window.covariance = summary.calibration_marginals;
    window.score = GetWindowScore(window);

    if (apply_results) {
      std::lock_guard<std::mutex> lock(*ba_mutex_);
      if(DoTvs){
        rig_->cameras_[0]->Pose() = ba.rig()->cameras_[0]->Pose();
        std::cerr << "new selfcal_rig t_wc\n:"
                  << rig_->cameras_[0]->Pose().matrix() << std::endl;
      }

      //Sophus::SE3d t_ba;
      std::shared_ptr<TrackerPose> last_pose = poses.back();
      // Get the pose of the last pose. This is used to calculate the relative
      // transform from the pose to the current pose.
      last_pose->t_wp = ba.GetPose(last_pose->opt_id[ba_id_]).t_wp;
      // std::cerr << "last pose t_wp: " << std::endl << last_pose->t_wp.matrix() <<
      //              std::endl;

      // Read out the pose and landmark values.
      for (uint32_t ii = start_pose ; ii < poses.size() ; ++ii) {
        std::shared_ptr<TrackerPose> pose = poses[ii];
        const ba::PoseT<double>& ba_pose =
            ba.GetPose(pose->opt_id[ba_id_]);
        // std::cerr << "Pose " << pose->opt_id << " t_wp " << std::endl <<
        //              pose->t_wp.matrix() << std::endl << " after opt: " <<
        //              std::endl << ba_pose.t_wp.matrix() << std::endl;

        pose->t_wp = ba_pose.t_wp;

        //t_ba = last_pose->t_wp.inverse() * pose->t_wp;
        for (std::shared_ptr<DenseTrack> track: pose->tracks) {
          if (track->external_id[ba_id_] == UINT_MAX) {
            continue;
          }

          track->needs_backprojection = true;
        }
      }
    } else {
      // TODO: check why this needs to be set back to the original camera
      // parameters. Should not have changed.
      std::lock_guard<std::mutex> lock(*ba_mutex_);
      rig_->cameras_[0]->SetParams(cam_params_backup);
    }
  }
}

template void OnlineCalibrator::AnalyzePriorityQueue<false, false>(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    CalibrationWindow& overal_window, uint32_t num_iterations = 1,
    bool apply_results = false);

template void OnlineCalibrator::AnalyzePriorityQueue<true, true>(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    CalibrationWindow& overal_window, uint32_t num_iterations = 1,
    bool apply_results = false);

template void OnlineCalibrator::AnalyzePriorityQueue<true, false>(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    CalibrationWindow& overal_window, uint32_t num_iterations = 1,
    bool apply_results = false);

template void OnlineCalibrator::AddCalibrationWindowToBa<false, false>(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    CalibrationWindow& window);

template void OnlineCalibrator::AddCalibrationWindowToBa<true, true>(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    CalibrationWindow& window);

template void OnlineCalibrator::AddCalibrationWindowToBa<true, false>(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    CalibrationWindow& window);

//template void OnlineCalibrator::AddCalibrationWindowToBa<false, true>(
//    std::vector<std::shared_ptr<TrackerPose>>& poses,
//    CalibrationWindow& window);

template void OnlineCalibrator::AnalyzeCalibrationWindow<false, false>(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    uint32_t start_pose, uint32_t end_pose, CalibrationWindow& window,
    uint32_t num_iterations = 1, bool apply_results = false);

template void OnlineCalibrator::AnalyzeCalibrationWindow<true, true>(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    uint32_t start_pose, uint32_t end_pose, CalibrationWindow& window,
    uint32_t num_iterations = 1, bool apply_results = false);

template void OnlineCalibrator::AnalyzeCalibrationWindow<true, false>(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    uint32_t start_pose, uint32_t end_pose, CalibrationWindow& window,
    uint32_t num_iterations = 1, bool apply_results = false);
