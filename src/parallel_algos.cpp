#include <glog/logging.h>
#include <sdtrack/parallel_algos.h>

using namespace sdtrack;

OptimizeTrack::OptimizeTrack(SemiDenseTracker &tracker_ref, const PyramidLevelOptimizationOptions &opt, std::vector<std::shared_ptr<DenseTrack> > &track_vec, OptimizationStats &opt_stats, uint32_t lvl, const std::vector<std::vector<cv::Mat> > &pyr, int debug_level) :
  tracker(tracker_ref),
  options(opt),
  stats(opt_stats),
  tracks(track_vec),
  level(lvl),
  image_pyrmaid(pyr),
  g_sdtrack_debug(debug_level)
{
  u.setZero();
  r_p.setZero();
  residual = 0;
}


OptimizeTrack::OptimizeTrack(const OptimizeTrack &other, tbb::split):
  tracker(other.tracker),
  options(other.options),
  stats(other.stats),
  tracks(other.tracks),
  level(other.level),
  image_pyrmaid(other.image_pyrmaid),
  g_sdtrack_debug(other.g_sdtrack_debug)

{
  u.setZero();
  r_p.setZero();
  residual = 0;
}


void OptimizeTrack::join(OptimizeTrack& other) {
  u += other.u;
  r_p += other.r_p;
  residual += other.residual;
}

void OptimizeTrack::operator()(const tbb::blocked_range<int> &r) {
  Eigen::Matrix<double, 6, 1> w;
  double v;
  double r_l;


  Eigen::Matrix<double, 2, 6> dp_dx;
  Eigen::Matrix<double, 2, 4> dp_dray;
  Eigen::Vector4t ray;
  Eigen::Matrix<double, 2, 4> dprojection_dray;
  std::vector<Eigen::Matrix<double, 1, 6>> di_dx;
  std::vector<double> di_dray;
  std::vector<double> res;
  di_dx.clear();
  di_dray.clear();
  res.clear();
  Eigen::Matrix<double, 1, 6> mean_di_dx;
  double mean_di_dray;
  Eigen::Matrix<double, 1, 6> final_di_dx;
  double final_di_dray;
  // std::vector<Eigen::Vector2t> valid_projections;
  // std::vector<unsigned int> valid_rays;

  uint32_t track_id = 0;
  uint32_t residual_id = 0;
  uint32_t num_inliers = 0;


  double track_residual;
  uint32_t residual_count = 0;
  uint32_t residual_offset = 0;

  // First project all tracks into this frame and form
  // the localization step.
  stats.jacobian_time = 0;
  stats.transfer_time = 0;
  stats.schur_time = 0;
  stats.solve_time = 0;
  stats.lm_time = 0;
  double schur_time;

  // for (std::shared_ptr<DenseTrack>& track : tracks) {
  for ( int ii = r.begin(); ii != r.end(); ii++ ) {
    std::shared_ptr<DenseTrack>& track = tracks[ii];
    // If we are only optimizing tracks from a single camera, skip track if
    // it wasn't initialized in the specified camera.
    if (options.only_optimize_camera_id != -1 && track->ref_cam_id !=
        options.only_optimize_camera_id) {
      continue;
    }

    const Sophus::SE3d& t_vc = tracker.camera_rig_->cameras_[track->ref_cam_id]->Pose();

    track->opt_id = UINT_MAX;
    track->residual_used = false;
    // If we are not solving for landmarks, there is no point including
    // uninitialized landmarks in the camera pose estimation
    if (options.optimize_landmarks == 0 &&
        track->keypoints.size() < MIN_OBS_FOR_CAM_LOCALIZATION) {
      continue;
    }

    track_residual = 0;

    // Project into the image and form the problem.
    DenseKeypoint& ref_kp = track->ref_keypoint;
    Patch& ref_patch = ref_kp.patch_pyramid[level];

    // Prepare the w matrix. We will add to it as we go through the rays.
    w.setZero();
    // Same for the v matrix
    v = 0;
    // Same for the RHS subtraction term
    r_l = 0;

    track->residual_offset = residual_offset;
    residual_offset++;

    for (uint32_t cam_id = 0 ; cam_id < tracker.num_cameras_ ; ++cam_id) {
      const Sophus::SE3d t_cv = tracker.camera_rig_->cameras_[cam_id]->Pose().inverse();
      const Eigen::Matrix4d t_cv_mat = t_cv.matrix();
      const Sophus::SE3d track_t_va =
          tracker.t_ba_ * track->t_ba * t_vc;
      const Sophus::SE3d track_t_ba = t_cv * track_t_va;
      const Eigen::Matrix4d track_t_ba_matrix = track_t_ba.matrix();

      PatchTransfer& transfer = track->transfer[cam_id];
      transfer.tracked_pixels = 0;
      transfer.rmse = 0;

      const double transfer_time = Tic();
      if (options.transfer_patches) {
        tracker.TransferPatch(track, level, cam_id, track_t_ba,
                              tracker.camera_rig_->cameras_[cam_id],
                              transfer, true);
      }
      stats.transfer_time += Toc(transfer_time);

      // Do not use this patch if less than half of its pixels reprojcet.
      if (transfer.valid_projections.size() < ref_patch.rays.size() / 2) {
        continue;
      }

      const double jacobian_time = Tic();
      di_dx.resize(transfer.valid_rays.size());
      di_dray.resize(transfer.valid_rays.size());
      res.resize(transfer.valid_rays.size());
      mean_di_dray = 0;
      mean_di_dx.setZero();
      double ncc_num = 0, ncc_den_a = 0, ncc_den_b = 0;
      for (size_t kk = 0; kk < transfer.valid_rays.size() ; ++kk) {
        const size_t ii = transfer.valid_rays[kk];
        // First transfer this pixel over to our current image.
        const Eigen::Vector2t& pix = transfer.valid_projections[kk];

        // need 2x6 transfer residual
        ray.head<3>() = ref_patch.rays[ii];
        ray[3] = ref_kp.rho;
        const Eigen::Vector4t ray_v = MultHomogeneous(track_t_va, ray);

        dprojection_dray = transfer.dprojections[kk];
        dprojection_dray *= tracker.pyramid_coord_ratio_[level][0];

        Eigen::Matrix<double, 1, 2> di_dp;
        const double val_pix = transfer.projected_values[ii];
        tracker.GetImageDerivative(image_pyrmaid[cam_id][level], pix,
                                   di_dp, val_pix);

        // need 2x4 transfer w.r.t. reference ray
        dp_dray = dprojection_dray * track_t_ba_matrix;
        di_dray[kk] = di_dp * dp_dray.col(3);

        //      for (unsigned int jj = 0; jj < 6; ++jj) {
        //        dp_dx.block<2,1>(0,jj) =
        //            //dprojection_dray * Sophus::SE3d::generator(jj) * ray;
        //            dprojection_dray * generators_[jj] * ray_v;
        //      }

        if (options.optimize_pose) {
          dprojection_dray *= t_cv_mat;
          dp_dx.col(0) = dprojection_dray.col(0) * ray_v[3];
          dp_dx.col(1) = dprojection_dray.col(1) * ray_v[3];
          dp_dx.col(2) = dprojection_dray.col(2) * ray_v[3];

          dp_dx.col(3) = dprojection_dray.col(2) * ray_v[1] -
              dprojection_dray.col(1) * ray_v[2];

          dp_dx.col(4) = dprojection_dray.col(0) * ray_v[2] -
              dprojection_dray.col(2) * ray_v[0];

          dp_dx.col(5) = dprojection_dray.col(1) * ray_v[0] -
              dprojection_dray.col(0) * ray_v[1];

          di_dx[kk] = di_dp * dp_dx;
        }

        // Insert the residual.
        const Scalar c_huber =
            1.2107 * tracker.pyramid_error_thresholds_[level];
        const double mean_s_ref = ref_patch.values[ii] - ref_patch.mean;
        const double mean_s_proj = val_pix - transfer.mean_value;
        res[kk] = mean_s_proj - mean_s_ref;
        bool inlier = true;
        if (tracker.tracker_options_.use_robust_norm_) {
          const double weight_sqrt = //sqrt(1.0 / ref_patch.statistics[ii][1]);
              sqrt(fabs(res[kk]) > c_huber ? c_huber / fabs(res[kk]) : 1.0);
          // LOG(g_sdtrack_debug) << "Weight for " << res[kk] << " at level " << level <<
          //              " is " << weight_sqrt * weight_sqrt << std::endl;
          res[kk] *= weight_sqrt;
          di_dx[kk] *= weight_sqrt;
          di_dray[kk] *= weight_sqrt;
          if (weight_sqrt != 1) {
            inlier = false;
          }
        }
        const double res_sqr = res[kk] * res[kk];

        if (inlier) {
          transfer.rmse += res_sqr;
          ncc_num += mean_s_ref * mean_s_proj;
          ncc_den_a += mean_s_ref * mean_s_ref;
          ncc_den_b += mean_s_proj * mean_s_proj;
          num_inliers++;
        }

        mean_di_dray += di_dray[kk];
        mean_di_dx += di_dx[kk];

        transfer.residuals[ii] = res[kk];
        residual_count++;
        track_residual += res_sqr;

        transfer.tracked_pixels++;
      }

      mean_di_dray /= transfer.valid_rays.size();
      mean_di_dx /= transfer.valid_rays.size();
      stats.jacobian_time += Toc(jacobian_time);

      schur_time = Tic();
      for (size_t kk = 0; kk < transfer.valid_rays.size() ; ++kk) {
        if (options.optimize_pose) {
          final_di_dx = di_dx[kk] - mean_di_dx;
          // Update u by adding j_p' * j_p
          u += final_di_dx.transpose() * final_di_dx;
          // Update rp by adding j_p' * r
          r_p += final_di_dx.transpose() * res[kk];
        }

        if (options.optimize_landmarks) {
          final_di_dray = di_dray[kk] - mean_di_dray;
          const double di_dray_id = final_di_dray;
          // Add the contribution of this ray to the w and v matrices.
          if (options.optimize_pose) {
            w += final_di_dx.transpose() * di_dray_id;
          }

          v += di_dray_id * di_dray_id;
          // Add contribution for the subraction term on the rhs.
          r_l += di_dray_id * res[kk];
        }
        residual_id++;
      }

      // Compute the track RMSE and NCC scores.
      transfer.rmse = transfer.tracked_pixels == 0 ?
            1e9 : sqrt(transfer.rmse / num_inliers);
      const double denom = sqrt(ncc_den_a * ncc_den_b);
      transfer.ncc = denom == 0 ? 0 : ncc_num / denom;
    }

    bool omit_track = false;
    if (track->id == tracker.longest_track_id_ &&
        track->keypoints.size() <= 2 &&
        options.optimize_landmarks && options.optimize_pose &&
        tracker.num_cameras_ == 1) {
      LOG(WARNING) << "omitting longest track id " <<
                              tracker.longest_track_id_ << std::endl;
      omit_track = true;
    }

    // If this landmark is the longest track, we omit it to fix scale.
    if (options.optimize_landmarks && !omit_track) {
      track->opt_id = track_id;
      double regularizer = options.optimize_landmarks && options.optimize_pose ?
            1e3 : 0;//level >= 2 ? 1e3 : level == 1 ? 1e2 : 1e1;

      v += regularizer;
      if (v < 1e-6) {
        v = 1e-6;
      }

      if (std::isnan(v) || std::isinf(v)) {
        LOG(WARNING) << "v is bad: " << v << std::endl;
      }

      // LOG(g_sdtrack_debug) << "v: " << v << std::endl;
      const double v_inv = 1.0 / v;
      track->v_inv_vec = v_inv;
      track->r_l_vec = r_l;

      if (options.optimize_pose) {
        track->w_vec = w;
        // Subtract the contribution of these residuals from u and r_p
        u -= w * v_inv * w.transpose();
        r_p -= w * v_inv * r_l;
      }
      track_id++;
    } else {
      track->opt_id = UINT_MAX;
    }

    // Add to the overal residual here, as we're sure the track will be
    // included in the optimization.
    residual += track_residual;
    track->residual_used = true;

    stats.schur_time += Toc(schur_time);
    // LOG(g_sdtrack_debug) << "track rmse for level " << level << " : " << track->rmse <<
    //              std::endl;
  }
}

ParallelExtractKeypoints::ParallelExtractKeypoints(
    SemiDenseTracker &tracker_ref,
    const cv::Mat &img_ref,
    const std::vector<cv::Rect> &bounds_ref):
  tracker(tracker_ref),
  image(img_ref),
  bounds(bounds_ref)
  {}

ParallelExtractKeypoints::ParallelExtractKeypoints(
    const ParallelExtractKeypoints &other,
    tbb::split) :
  tracker(other.tracker),
  image(other.image),
  bounds(other.bounds)
  {}

void ParallelExtractKeypoints::join(ParallelExtractKeypoints &other)
{
  keypoints.insert(keypoints.end(), other.keypoints.begin(),
                   other.keypoints.end());
}

void ParallelExtractKeypoints::operator()(const tbb::blocked_range<int> &r)
{
  std::vector<cv::KeyPoint> cell_kp;
  for ( int ii = r.begin(); ii != r.end(); ii++ ) {
    cell_kp.clear();
    const cv::Rect bound = bounds[ii];
    cv::Mat roi(image, bound);
    tracker.detector_->detect(roi, cell_kp);

    for (cv::KeyPoint& kp : cell_kp) {
      kp.pt.x += bound.x;
      kp.pt.y += bound.y;
      if(kp.pt.y < 450)
        keypoints.push_back(kp);
    }
  }
}


Parallel2dAlignment::Parallel2dAlignment(
    SemiDenseTracker &tracker_ref,
    const AlignmentOptions &options_ref,
    const std::vector<std::vector<cv::Mat>> &pyr,
    std::vector<std::shared_ptr<DenseTrack> > &tracks_v,
    Sophus::SE3d tcv,
    uint32_t lvl,
    uint32_t cam) :
  tracker(tracker_ref),
  options(options_ref),
  image_pyramid(pyr),
  tracks(tracks_v),
  t_cv(tcv),
  level(lvl),
  cam_id(cam) {}

void Parallel2dAlignment::operator()(const tbb::blocked_range<int> &r) const
{
  Eigen::LDLT<Eigen::Matrix2d> solver;
  Eigen::Matrix2d jtj;
  Eigen::Vector2d jtr, delta_pix, delta_pix_0th_level;
  double res_total;
  Eigen::Matrix<double, 1, 2> di_dp;
  double ncc_num = 0, ncc_den_a = 0, ncc_den_b = 0;

  for (int ii = r.begin(); ii != r.end(); ii++) {
    std::shared_ptr<DenseTrack>& track = tracks[ii];
    const Sophus::SE3d& t_vc = tracker.camera_rig_->cameras_[track->ref_cam_id]->Pose();
    // If we are only optimizing tracks from a single camera, skip track if
    // it wasn't initialized in the specified camera.
    if (options.only_optimize_camera_id != -1 && track->ref_cam_id !=
        options.only_optimize_camera_id) {
      continue;
    }

    // Don't align newly created tracks.
    if (track->keypoints.size() < 2) {
      continue;
    }

    // 2D Alignment optimization loop.
    while (true) {
      jtj.setZero();
      jtr.setZero();
      res_total = 0;
      ncc_num = 0;
      ncc_den_a = 0;
      ncc_den_b = 0;

      // Set up the 2d alignment problem.
      PatchTransfer& transfer = track->transfer[cam_id];

      if (transfer.level != level) {
        // We have to re-transfer this track.
        const Sophus::SE3d track_t_ba =
            t_cv * tracker.t_ba_ * track->t_ba * t_vc;

        tracker.TransferPatch(track, level, cam_id,  track_t_ba,
                              tracker.camera_rig_->cameras_[cam_id],
                              transfer, false);
      }
      // uint32_t level = transfer.level;
      DenseKeypoint& ref_kp = track->ref_keypoint;
      Patch& ref_patch = ref_kp.patch_pyramid[level];
      bool out_of_bounds = false;
      for (size_t kk = 0; kk < transfer.valid_rays.size() ; ++kk) {
        const size_t ii = transfer.valid_rays[kk];
        const Eigen::Vector2t& pix = transfer.valid_projections[kk];
        if (!tracker.IsReprojectionValid(pix, image_pyramid[cam_id][level])) {
          out_of_bounds = true;
          break;
        }
        // Get the residual at this point.
        const double val_pix = transfer.projected_values[ii];
        //GetSubPix(image_pyrmaid[level], pix[0], pix[1]);
        const double mean_s_ref = ref_patch.values[ii] - ref_patch.mean;
        const double mean_s_proj = val_pix - transfer.mean_value;
        const double res = mean_s_proj - mean_s_ref;

        // Also tet the jacobian.
        tracker.GetImageDerivative(
              image_pyramid[cam_id][level], pix, di_dp, val_pix);

        jtj += di_dp.transpose() * di_dp;
        jtr += di_dp.transpose() * res;
        res_total += fabs(res);
      }

      if (out_of_bounds) {
        break;
      }

      // Calculate the update to this patch.
      solver.compute(jtj);
      delta_pix = solver.solve(jtr);
      delta_pix_0th_level[0] =
          delta_pix[0] / tracker.pyramid_coord_ratio_[level][0];
      delta_pix_0th_level[1] =
          delta_pix[1] / tracker.pyramid_coord_ratio_[level][1];

      out_of_bounds = false;
      transfer.mean_value = 0;
      for (size_t kk = 0; kk < transfer.valid_rays.size() ; ++kk) {
        const size_t ii = transfer.valid_rays[kk];
        transfer.valid_projections[kk] -= delta_pix;
        transfer.projections[ii] -= delta_pix_0th_level;
        const Eigen::Vector2t& pix = transfer.valid_projections[kk];
        if (!tracker.IsReprojectionValid(
              pix, image_pyramid[cam_id][level])) {
          out_of_bounds = true;
          break;
        }
        transfer.projected_values[ii] =
            tracker.GetSubPix(image_pyramid[cam_id][level], pix[0], pix[1]);
        transfer.mean_value += transfer.projected_values[ii];
      }

      if (out_of_bounds) {
        break;
      }
      if (transfer.valid_rays.size() != 0) {
        transfer.mean_value /= transfer.valid_rays.size();
      }

      double post_res_total = 0;
      for (size_t kk = 0; kk < transfer.valid_rays.size() ; ++kk) {
        const size_t ii = transfer.valid_rays[kk];
        const double mean_s_ref = ref_patch.values[ii] - ref_patch.mean;
        const double mean_s_proj = transfer.projected_values[ii] -
            transfer.mean_value;
        ncc_num += mean_s_ref * mean_s_proj;
        ncc_den_a += mean_s_ref * mean_s_ref;
        ncc_den_b += mean_s_proj * mean_s_proj;
        const double res = mean_s_proj - mean_s_ref;
        transfer.residuals[ii] = mean_s_proj - mean_s_ref;
        post_res_total += fabs(res);
      }

      const double denom = sqrt(ncc_den_a * ncc_den_b);
      const double prev_ncc = transfer.ncc;
      transfer.ncc = denom == 0 ? 0 : ncc_num / denom;

      // If the residual increases, quit and roll back the changes.
      if (post_res_total >= res_total) {
        // roll back the changes
        transfer.ncc = prev_ncc;
        break;
      } else {
        track->offset_2d[cam_id] -= delta_pix_0th_level;
        if (options.apply_to_kp) {
          track->keypoints.back()[cam_id].kp -= delta_pix_0th_level;
        }
      }

      if (delta_pix.norm() < 0.01) {
        break;
      }

    }
  }
}

