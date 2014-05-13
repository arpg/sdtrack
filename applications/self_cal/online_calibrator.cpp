#include <online_calibrator.h>

using namespace sdtrack;

OnlineCalibrator::OnlineCalibrator()
{

}

void OnlineCalibrator::Init(calibu::Rig<Scalar> *rig,
                            uint32_t num_windows,
                            uint32_t window_length,
                            Eigen::VectorXd covariance_weights)
{
  num_windows_ = num_windows;
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

  ba::debug_level_threshold = 0;
}

void OnlineCalibrator::TestJacobian(Eigen::Vector2t pix,
                                    Sophus::SE3t t_ba,
                                    Scalar rho)
{
  calibu::CameraInterface<Scalar>* cam = rig_->cameras_[0];
  Scalar* params = cam->GetParams();
  const double eps = 1e-6;
  Eigen::Matrix<Scalar, 2, Eigen::Dynamic> jacobian_fd(2, cam->NumParams());
  // Test the transfer jacobian.
  for (int ii = 0 ; ii < cam->NumParams()  ; ++ii) {
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

void OnlineCalibrator::AnalyzePriorityQueue(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    CalibrationWindow &overal_window,
    uint32_t num_iterations, bool apply_results)
{
  // Backup the calibration params, in case we are not applying them.
  Eigen::VectorXd params_backup(rig_->cameras_[0]->NumParams());
  for (int ii = 0; ii < params_backup.rows() ; ++ii) {
    params_backup[ii] = rig_->cameras_[0]->GetParams()[ii];
  }

  ba::Options<double> options;
  options.write_reduced_camera_matrix = false;
  options.projection_outlier_threshold = 1.0;
  options.trust_region_size = 10;
  options.use_dogleg = true;
  options.use_triangular_matrices = true;
  options.use_sparse_solver = false;
  options.calculate_calibration_marginals = true;
  options.error_change_threshold = 1e-6;
  options.trust_region_size = 100;

  selfcal_ba.Init(options, poses.size(),
                  current_tracks->size() * poses.size());
  selfcal_ba.AddCamera(rig_->cameras_[0], rig_->t_wc_[0]);

  // Add all the windows to ba.
  for (const CalibrationWindow& window : windows()) {
    if (window.start_index < poses.size() && window.end_index < poses.size()) {
      AddCalibrationWindowToBa(poses, window);
    }
  }

  selfcal_ba.Solve(num_iterations);

  // Obtain the mean from the BA.
  double* params = selfcal_ba.rig().cameras_[0]->GetParams();
  overal_window.mean =
      Eigen::VectorXd(selfcal_ba.rig().cameras_[0]->NumParams());
  for (int ii = 0; ii < overal_window.mean.rows() ; ++ii) {
    overal_window.mean[ii] = params[ii];
  }
  overal_window.covariance =
      selfcal_ba.GetSolutionSummary().calibration_marginals;

  // If we are not applying results, reset them.
  if (!apply_results) {
    for (int ii = 0; ii < params_backup.rows() ; ++ii) {
      // Replace the parameters with the backup.
      rig_->cameras_[0]->GetParams()[ii] = params_backup[ii];
    }
  }
}

void OnlineCalibrator::AddCalibrationWindowToBa(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    const CalibrationWindow &window)
{
  const uint32_t start_active_pose = window.start_index;

  // First find the longest track id in the base frame. We will use this
  // pose to remove the scale nullspace.
  uint32_t max_keypoints = 0;
  std::shared_ptr<DenseTrack> longest_track = nullptr;
  std::shared_ptr<TrackerPose> pose = poses[window.start_index];
  if (pose->tracks.size() == 0) {
    return;
  }
  for (std::shared_ptr<DenseTrack> track : pose->tracks) {
    if (track->keypoints.size() > max_keypoints) {
      max_keypoints = track->keypoints.size();
      longest_track = track;
    }
  }

  // First add all the poses and landmarks to ba.
  for (uint32_t ii = window.start_index; ii < window.end_index ; ++ii) {
    std::shared_ptr<TrackerPose> pose = poses[ii];
    pose->opt_id = selfcal_ba.AddPose(pose->t_wp, ii > start_active_pose);
    // std::cerr << "Adding pose with opt_id " << pose->opt_id << " and t_wp " <<
    //              pose->t_wp.matrix() << std::endl;

    for (std::shared_ptr<DenseTrack> track: pose->tracks) {
      if (track->num_good_tracked_frames == 1 || track->is_outlier) {
        track->external_id = UINT_MAX;
        continue;
      }

      Eigen::Vector4d ray;
      ray.head<3>() = track->ref_keypoint.ray;
      ray[3] = track->ref_keypoint.rho;
      ray = MultHomogeneous(pose->t_wp  * rig_->t_wc_[0], ray);
      bool active = track->id != longest_track->id;
      track->external_id = selfcal_ba.AddLandmark(ray, pose->opt_id, 0, active);
      // std::cerr << "Adding lm with opt_id " << track->external_id << " and "
      //              " x_r " << ray.transpose() << " and x_orig: " <<
      //              x_r_orig_->transpose() << std::endl;
    }
  }

  // Now add all reprojections to ba)
  for (uint32_t ii = window.start_index ; ii < window.end_index ; ++ii) {
    std::shared_ptr<TrackerPose> pose = poses[ii];
    for (std::shared_ptr<DenseTrack> track : pose->tracks) {
      if (track->external_id == UINT_MAX) {
        continue;
      }
      // Limit the number of measurements we add here for a track, as they
      // could exceed the size of this calibration window.
      for (size_t jj = 0; jj < track->keypoints.size() &&
           jj < (window.end_index - ii) ; ++jj) {
        if (track->keypoints_tracked[jj]) {
          const Eigen::Vector2d& z = track->keypoints[jj];
          selfcal_ba.AddProjectionResidual(
                z, pose->opt_id + jj, track->external_id, 0);
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
  if (windows_.size() < num_windows_) {
    if (num_overlaps == 0) {
      windows_.push_back(new_window);
      std::cerr << "Pushing back non overlapping window into position " <<
                   windows_.size() << std::endl;
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
  // First transform the covariance given the weights.
  return
      (covariance_weights_.asDiagonal() * window.covariance *
       covariance_weights_.asDiagonal()).determinant();
}



void OnlineCalibrator::AnalyzeCalibrationWindow(
    std::vector<std::shared_ptr<TrackerPose>>& poses,
    std::list<std::shared_ptr<DenseTrack>>* current_tracks,
    uint32_t start_pose, uint32_t end_pose,
    CalibrationWindow &window,
    uint32_t num_iterations, bool apply_results)
{
  window.start_index = start_pose;
  window.end_index = end_pose;

  // Backup the calibration params, in case we are not applying them.
  Eigen::VectorXd params_backup(rig_->cameras_[0]->NumParams());
  for (int ii = 0; ii < params_backup.rows() ; ++ii) {
    params_backup[ii] = rig_->cameras_[0]->GetParams()[ii];
  }

  ba::Options<double> options;
  options.write_reduced_camera_matrix = false;
  options.projection_outlier_threshold = 1.0;
  options.trust_region_size = 10;
  options.use_dogleg = true;
  options.use_sparse_solver = (end_pose - start_pose) > window_length_ * 10;
  /// ZZZZ WHY IS THIS NEEDED? SPARSE IS BROKEN WITH TRIANGULAR IT SEEMS.
  options.use_triangular_matrices = !options.use_sparse_solver;
  options.calculate_calibration_marginals = true;
  options.error_change_threshold = 1e-6;
  options.trust_region_size = 100;

  // Do a bundle adjustment on the current set
  if (current_tracks && poses.size() > 1) {
    selfcal_ba.Init(options, poses.size(),
                    current_tracks->size() * poses.size());
    selfcal_ba.AddCamera(rig_->cameras_[0], rig_->t_wc_[0]);


    AddCalibrationWindowToBa(poses, window);

    // Optimize the poses
    selfcal_ba.Solve(num_iterations);

    const ba::SolutionSummary<double>& summary =
        selfcal_ba.GetSolutionSummary();
    // Obtain the mean from the BA.
    double* params = selfcal_ba.rig().cameras_[0]->GetParams();
    window.mean = Eigen::VectorXd(selfcal_ba.rig().cameras_[0]->NumParams());
    for (int ii = 0; ii < window.mean.rows() ; ++ii) {
      window.mean[ii] = params[ii];
    }
    window.covariance = summary.calibration_marginals;
    window.score = GetWindowScore(window);

    if (apply_results) {
      Sophus::SE3d t_ba;
      std::shared_ptr<TrackerPose> last_pose = poses.back();
      // Get the pose of the last pose. This is used to calculate the relative
      // transform from the pose to the current pose.
      last_pose->t_wp = selfcal_ba.GetPose(last_pose->opt_id).t_wp;
      // std::cerr << "last pose t_wp: " << std::endl << last_pose->t_wp.matrix() <<
      //              std::endl;

      // Read out the pose and landmark values.
      for (uint32_t ii = start_pose ; ii < poses.size() ; ++ii) {
        std::shared_ptr<TrackerPose> pose = poses[ii];
        const ba::PoseT<double>& ba_pose = selfcal_ba.GetPose(pose->opt_id);
        // std::cerr << "Pose " << pose->opt_id << " t_wp " << std::endl <<
        //              pose->t_wp.matrix() << std::endl << " after opt: " <<
        //              std::endl << ba_pose.t_wp.matrix() << std::endl;

        pose->t_wp = ba_pose.t_wp;

        t_ba = last_pose->t_wp.inverse() * pose->t_wp;
        for (std::shared_ptr<DenseTrack> track: pose->tracks) {
          if (track->external_id == UINT_MAX) {
            continue;
          }
          // Set the t_ab on this track.
          //std::cerr << "Changing track t_ba from " << std::endl <<
          //             track->t_ba.matrix() << std::endl << " to " <<
          //             t_ba.matrix() << std::endl;
          track->t_ba = t_ba;

          // Get the landmark location in the world frame.
          const Eigen::Vector4d& x_w =
              selfcal_ba.GetLandmark(track->external_id);
          Eigen::Vector4d prev_ray;
          prev_ray.head<3>() = track->ref_keypoint.ray;
          prev_ray[3] = track->ref_keypoint.rho;
          // Make the ray relative to the pose.
          Eigen::Vector4d x_r = MultHomogeneous(
                (pose->t_wp * rig_->t_wc_[0]).inverse(), x_w);
          // Normalize the xyz component of the ray to compare to the original
          // ray.
          x_r /= x_r.head<3>().norm();
          // std::cerr << "lm " << track->external_id << " x_r " <<
          //              prev_ray.transpose() << " after opt: " <<
          //              x_r.transpose() << std::endl;
          // Update the inverse depth on the track
          track->ref_keypoint.rho = x_r[3];
        }
      }
    } else {
      std::cerr << "Resetting parameters. " << std::endl;
      for (int ii = 0; ii < params_backup.rows() ; ++ii) {
        // Replace the parameters with the backup.
        rig_->cameras_[0]->GetParams()[ii] = params_backup[ii];
      }
    }
  }
}
