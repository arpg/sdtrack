#include <sdtrack/semi_dense_tracker.h>
#define CENTER_WEIGHT 500
#define MIN_OBS_FOR_CAM_LOCALIZATION 3
using namespace sdtrack;


void SemiDenseTracker::Initialize(const KeypointOptions &keypoint_options,
                                  const TrackerOptions &tracker_options,
                                  calibu::Rig<Scalar> *rig)
{
  // rig_ = rig;
  // const calibu::CameraModelGeneric<Scalar>& cam = rig->cameras[0].camera;
  // camera_rig_->AddCamera(calibu::CreateFromOldCamera<Scalar>(cam),
  //                       rig->cameras[0].T_wc);
  camera_rig_ = rig;

  keypoint_options_ = keypoint_options;
  tracker_options_ = tracker_options;
  next_track_id_ = 0;
  switch (tracker_options_.detector_type)
  {
    case TrackerOptions::Detector_FAST:
      detector_ = new cv::FastFeatureDetector(
            keypoint_options.fast_threshold,
            keypoint_options.fast_nonmax_suppression);
      break;
    case TrackerOptions::Detector_GFTT:
      detector_ = new cv::GoodFeaturesToTrackDetector(
            keypoint_options.max_num_features /
            powi(tracker_options.feature_cells, 2),
            keypoint_options.gftt_absolute_strength_threshold,
            keypoint_options.gftt_min_distance_between_features,
            keypoint_options.gftt_feature_block_size,
            keypoint_options.gftt_use_harris);
      break;

    case TrackerOptions::Detector_SURF:
      detector_ = new cv::SurfFeatureDetector(1000);
      break;
  }

  uint32_t patch_dim = tracker_options_.patch_dim;
  double robust_norm_thresh = tracker_options_.robust_norm_threshold_;
  pyramid_patch_dims_.resize(tracker_options_.pyramid_levels);
  pyramid_patch_corner_dims_.resize(pyramid_patch_dims_.size());
  pyramid_patch_interp_factors_.resize(pyramid_patch_dims_.size());
  pyramid_error_thresholds_.resize(pyramid_patch_dims_.size());
  // Calculate the pyramid dimensions for each patch. This is used to initialize
  // new patches.
  for (uint32_t ii = 0 ; ii < tracker_options_.pyramid_levels ; ++ii) {
    std::cerr << "Level " << ii << " patch dim is " << patch_dim << std::endl;
    pyramid_error_thresholds_[ii] = robust_norm_thresh;
    robust_norm_thresh *= 2;
    pyramid_patch_dims_[ii] = patch_dim;
    // The array coordinates of the four corners of the patch.
    pyramid_patch_corner_dims_[ii] = {0,                                  // tl
                                      patch_dim - 1,                      // tr
                                      patch_dim * patch_dim - patch_dim,  // bl
                                      patch_dim * patch_dim - 1};         // br

    // For each cell, we also need to get the interpolation factors.
    pyramid_patch_interp_factors_[ii].reserve(powi(patch_dim, 2));
    const double factor = powi(patch_dim - 1, 2);
    for (double yy = 0; yy < patch_dim ; ++yy) {
      for (double xx = 0; xx < patch_dim ; ++xx) {
        pyramid_patch_interp_factors_[ii].push_back({
          ( (patch_dim - 1 - xx) * (patch_dim - 1 - yy) ) / factor,   // tl
          ( xx * (patch_dim - 1 - yy) ) / factor,   // tr
          ( (patch_dim - 1 - xx) * yy ) / factor,   // bl
          ( xx * yy ) / factor   // br
        });
      }
    }
    // patch_dim = (patch_dim + 1) / 2;
  }

  for (int ii = 0; ii < 6 ; ++ii) {
    generators_[ii] = Sophus::SE3d::generator(ii);
  }

  // Inititalize the feature cells.
  feature_cells_.resize(tracker_options_.feature_cells,
                        tracker_options_.feature_cells);
  feature_cells_.setZero();
  feature_cell_rho_.resize(tracker_options_.feature_cells,
                           tracker_options_.feature_cells);
  feature_cell_rho_.setZero();

  mask_.AddImage(rig->cameras_[0]->ImageSize()[0],
      rig->cameras_[0]->ImageSize()[1]);

  pyramid_coord_ratio_.resize(tracker_options_.pyramid_levels);
}

void SemiDenseTracker::ExtractKeypoints(const cv::Mat &image,
                                        std::vector<cv::KeyPoint> &keypoints)
{
  std::vector<cv::KeyPoint> cell_kp;
  keypoints.clear();
  keypoints.reserve(keypoint_options_.max_num_features);
  uint32_t cell_width = image.cols / tracker_options_.feature_cells;
  uint32_t cell_height = image.rows / tracker_options_.feature_cells;
  uint32_t cells_hit = 0;
  const double time = Tic();
  for (uint32_t ii = 0  ; ii < tracker_options_.feature_cells ; ++ii) {
    for (uint32_t jj = 0  ; jj < tracker_options_.feature_cells ; ++jj) {
      if (feature_cells_(jj, ii) >= lm_per_cell_) {
        continue;
      }

      const cv::Rect bounds(ii * cell_width, jj * cell_height,
                            cell_width, cell_height);
      cv::Mat roi(image, bounds);
      detector_->detect(roi, cell_kp);

      cells_hit++;

      //      std::cerr << "Detected " << cell_kp.size() << " in " << bounds.x <<
      //                   ", " << bounds.y << ", " << bounds.width << ", " <<
      //                   bounds.height << std::endl;

      // Shift the keypoints.
      for (cv::KeyPoint& kp : cell_kp) {
        kp.pt.x += bounds.x;
        kp.pt.y += bounds.y;
        keypoints.push_back(kp);
      }
    }
  }

  // std::cerr << "total " << keypoints.size() << " keypoints " << std::endl;
  //  detector_->detect(image, keypoints);
  HarrisScore(image.data, image.cols, image.rows,
              tracker_options_.patch_dim, keypoints);
  std::cerr << "extract feature detection for " << keypoints.size() <<
               " and "  << cells_hit << " cells " <<  " keypoints took " <<
               Toc(time) << " seconds." << std::endl;
}

bool SemiDenseTracker::IsKeypointValid(const cv::KeyPoint &kp,
                                       uint32_t image_width,
                                       uint32_t image_height)
{
  // Only for the car dataset.
  //  if (kp.pt.x < 410 && kp.pt.y > 300) {
  //    return false;
  //  }

  if (kp.response < tracker_options_.harris_score_threshold) {
    return false;
  }

  uint32_t margin =
      tracker_options_.patch_dim * tracker_options_.pyramid_levels;
  // Key keypoint also can't be closer than a certain amount to the edge.
  if (kp.pt.x < margin || kp.pt.y < margin ||
      kp.pt.x > image_width - margin || kp.pt.y > image_height - margin)
  {
    return false;
  }

  // Make sure this keypoint isn't already ina a patch somewhere.
  if (mask_.GetMask(0, kp.pt.x, kp.pt.y))
  {
    return false;
  }

  return true;
}

bool SemiDenseTracker::IsReprojectionValid(const Eigen::Vector2t &pix,
                                           const cv::Mat& image)
{
  if (pix[0] <= 2 || pix[0] > (image.cols - 2) ) {
    return false;
  }

  if (pix[1] <= 2 || pix[1] > (image.rows - 2) ) {
    return false;
  }

  return true;
}

void SemiDenseTracker::BackProjectTrack(std::shared_ptr<DenseTrack> track,
                                        bool initialize_pixel_vals)
{
  DenseKeypoint& kp = track->ref_keypoint;
  // Unproject the center pixel for this track.
  kp.ray =
      camera_rig_->cameras_[0]->Unproject(kp.center_px).normalized() *
      tracker_options_.default_ray_depth;

  //std::cerr << "Initializing keypoint at " << kp.pt.x << ", " <<
  //               kp.pt.y << " with response: " << kp.response << std::endl;


  for (uint32_t ii = 0 ; ii < tracker_options_.pyramid_levels ; ++ii) {

    Patch& patch = kp.patch_pyramid[ii];
    // Transform the patch to lower levels
    patch.center[0] = kp.center_px[0] * pyramid_coord_ratio_[ii][0];
    patch.center[1] = kp.center_px[1] * pyramid_coord_ratio_[ii][1];

    // std::cerr << "\tCenter at level " << ii << " " << patch.center[0] <<
    //              ", " << patch.center[1] << std::endl;

    uint32_t array_dim = 0;
    double mean_value = 0;
    const double extent = (patch.dim - 1) / 2.0;
    const double x_max = patch.center[0] + extent;
    const double y_max = patch.center[1] + extent;

    for (double yy = patch.center[1] - extent; yy <= y_max ; ++yy) {
      for (double xx = patch.center[0] - extent; xx <= x_max ; ++xx) {
        // Calculate this pixel in the 0th level image
        Eigen::Vector2t px_level0(xx / pyramid_coord_ratio_[ii][0],
            yy / pyramid_coord_ratio_[ii][1]);
        patch.rays[array_dim] =
            camera_rig_->cameras_[0]->Unproject(px_level0).normalized() *
            tracker_options_.default_ray_depth;
        if (initialize_pixel_vals) {
          const double val = GetSubPix(image_pyrmaid_[ii], xx, yy);
          patch.values[array_dim] = val;
          patch.projected_values[array_dim] = patch.values[array_dim];
          patch.projections[array_dim] = px_level0;
          mean_value += patch.values[array_dim];
        }
        array_dim++;
      }
    }

    // Calculate the mean, and subtract from all values.
    if (initialize_pixel_vals) {
      mean_value /= patch.values.size();
      patch.mean = mean_value;
    }

    if (array_dim != patch.rays.size()) {
      std::cerr << "Not enough rays!" << std::endl;
      throw 0;
    }
  }
}

uint32_t SemiDenseTracker::StartNewTracks(std::vector<cv::Mat> &image_pyrmaid,
                                      std::vector<cv::KeyPoint> &cv_keypoints,
                                      uint32_t num_to_start)
{
  // Initialize the random inverse depth generator.
  const double range = tracker_options_.default_ray_depth * 0.1;
  std::uniform_real_distribution<double> distribution(
        tracker_options_.default_ray_depth - range,
        tracker_options_.default_ray_depth + range);

  uint32_t num_started = 0;
  // const CameraInterface& cam = *camera_rig_->cameras[0];
  previous_keypoints_.clear();
  previous_keypoints_.reserve(cv_keypoints.size());

  std::sort(cv_keypoints.begin(), cv_keypoints.end(),
            [](const cv::KeyPoint& a, const cv::KeyPoint& b)
  { return a.response > b.response; } );


  // Clear the new tracks array, which will be populated below.
  new_tracks_.clear();

  for (cv::KeyPoint& kp : cv_keypoints)
  {
    if (num_to_start == 0) {
      break;
    }

    // Figure out which feature cell this particular reprojection falls into,
    // and increment that cell
    const uint32_t addressx =
        (kp.pt.x / image_pyrmaid[0].cols) * tracker_options_.feature_cells;
    const uint32_t addressy =
        (kp.pt.y / image_pyrmaid[0].rows) * tracker_options_.feature_cells;
    if (feature_cells_(addressy, addressx) >= lm_per_cell_) {
      continue;
    }

    if (!IsKeypointValid(kp, image_pyrmaid[0].cols,
                         image_pyrmaid[0].rows)) {
      continue;
    }

    mask_.SetMask(0, kp.pt.x, kp.pt.y);
    feature_cells_(addressy, addressx)++;

    // Otherwise extract pyramid for this keypoint, and also backproject all
    // pixels as the rays will not change.
    std::shared_ptr<DenseTrack> new_track(new DenseTrack(
                                            tracker_options_.pyramid_levels, pyramid_patch_dims_));
    new_track->id = next_track_id_++;
    current_tracks_.push_back(new_track);
    new_tracks_.push_back(new_track);
    DenseKeypoint& new_kp = new_track->ref_keypoint;
    // Get the default rho from this cell, if available:
    //    if (feature_cell_rho_(addressy, addressx) != 0) {
    //      new_kp.rho = feature_cell_rho_(addressy, addressx);
    //    } else {

    // const double rand_0_1 =
    //     static_cast <double>(rand()) / static_cast<double>(RAND_MAX);
    // const double variance = tracker_options_.default_rho * 0.1;
    // This makes the rho vary by +- the amount specified by variance.
    // new_kp.rho = tracker_options_.default_rho; /* + rand_0_1 * 2 * variance -
    //    variance;*/
    new_kp.rho = distribution(generator_);
    //    }
    new_kp.response = kp.response;
    new_kp.center_px = Eigen::Vector2t(kp.pt.x, kp.pt.y);

    new_track->keypoints.push_back(new_kp.center_px);
    new_track->keypoint_external_data.push_back(UINT_MAX);
    new_track->keypoints_tracked.push_back(true);
    new_track->num_good_tracked_frames++;
    new_track->rmse = 0;

    BackProjectTrack(new_track, true);

    num_to_start--;
    num_started++;
  }
  return num_started;
}

double SemiDenseTracker::EvaluateTrackResiduals(uint32_t level,
    const std::vector<cv::Mat>& image_pyrmaid,
    std::list<std::shared_ptr<DenseTrack>>& tracks,
    bool transfer_jacobians,
    bool optimized_tracks_only)
{
  feature_cells_.setZero();
  feature_cell_rho_.setZero();

  const Sophus::SE3d t_vc = camera_rig_->t_wc_[0];
  const Sophus::SE3d t_cv = camera_rig_->t_wc_[0].inverse();

  double residual = 0;
  uint32_t residual_count = 0;
  for (std::shared_ptr<DenseTrack>& track : tracks) {
    if (optimized_tracks_only && !track->residual_used) {
      continue;
    }

    PatchTransfer& transfer = track->transfer;
    const Sophus::SE3d track_t_ba = t_cv * t_ba_ * track->t_ba * t_vc;
    DenseKeypoint& ref_kp = track->ref_keypoint;
    Patch& ref_patch = ref_kp.patch_pyramid[level];

    uint32_t num_inliers = 0;

    TransferPatch(track, level, track_t_ba, camera_rig_->cameras_[0], transfer,
        transfer_jacobians);

    track->tracked_pixels = 0;
    track->pixels_attempted = transfer.pixels_attempted;
    track->rmse = 0;
    double ncc_num = 0, ncc_den_a = 0, ncc_den_b = 0;

    if (transfer.valid_projections.size() < ref_patch.rays.size() / 2) {
      continue;
    }


    for (size_t kk = 0; kk < transfer.valid_rays.size() ; ++kk)
    {
      const size_t ii = transfer.valid_rays[kk];
      // First transfer this pixel over to our current image.
      const double val_pix = ref_patch.projected_values[ii];

      const Scalar c_huber = 1.2107 * pyramid_error_thresholds_[level];
      const double mean_s_ref = ref_patch.values[ii] - ref_patch.mean;
      const double mean_s_proj = val_pix - transfer.mean_value;
      double res = mean_s_proj - mean_s_ref;
      bool inlier = true;
      if (tracker_options_.use_robust_norm_) {
        const double weight_sqrt = sqrt(1.0 / ref_patch.statistics[ii][1]);
            // sqrt(fabs(res) > c_huber ? c_huber / fabs(res) : 1.0);
         std::cerr << "Weight for " << res << " at level " << level << " is " <<
                      weight_sqrt * weight_sqrt << std::endl;
        res *= weight_sqrt;
        if (weight_sqrt != 1) {
          inlier = false;
        }
      }
      const double res_sqr = res * res;

      if (inlier) {
        track->rmse += res_sqr;
        ncc_num += mean_s_ref * mean_s_proj;
        ncc_den_a += mean_s_ref * mean_s_ref;
        ncc_den_b += mean_s_proj * mean_s_proj;
        num_inliers++;
      }

      ref_patch.residuals[ii] = res;
      residual_count++;
      residual += res_sqr;

      track->tracked_pixels++;
    }

    // Reproject the center, but only if this landmark's position was optimized
    // and not in inverse depth mode.
#if (LM_DIM == 3)
    if (!track->inverse_depth_ray && track->opt_id != UINT_MAX) {
      const Eigen::Vector2t center_pix = transfer.center_projection;
      const double center_weight = CENTER_WEIGHT;
      // Form the residual between the center pix and the original pixel loc.
      const Eigen::Vector2t c_res = (center_pix - ref_kp.center_px);
      track->center_error = c_res.norm();
      residual += c_res.transpose() * center_weight * c_res;
    }
#endif

    // Compute the track RMSE and NCC scores.
    track->rmse = track->tracked_pixels == 0 ?
          1e9 : sqrt(track->rmse / num_inliers);
    const double denom = sqrt(ncc_den_a * ncc_den_b);
    track->ncc = denom == 0 ? 0 : ncc_num / denom;
  }

  return sqrt(residual);
}

void SemiDenseTracker::ReprojectTrackCenters()
{
  const Sophus::SE3d t_vc = camera_rig_->t_wc_[0];
  const Sophus::SE3d t_cv = camera_rig_->t_wc_[0].inverse();

  average_track_length = 0;
  const calibu::CameraInterface<Scalar>& cam = *camera_rig_->cameras_[0];
  for (std::shared_ptr<DenseTrack>& track : current_tracks_)
  {
    const Sophus::SE3d track_t_ba = t_cv * t_ba_ * track->t_ba * t_vc;
    const DenseKeypoint& ref_kp = track->ref_keypoint;
    // Transfer the center ray. This is used for 2d tracking.
    const Eigen::Vector2t center_pix =
        cam.Transfer3d(track_t_ba, ref_kp.ray + ref_kp.ray_delta, ref_kp.rho);
    if (IsReprojectionValid(center_pix, image_pyrmaid_[0])) {
      track->keypoints.back() = center_pix;
      mask_.SetMask(0, center_pix[0], center_pix[1]);

      // Figure out which feature cell this particular reprojection falls into,
      // and increment that cell.
      const uint32_t addressx =
          (center_pix[0] / image_pyrmaid_[0].cols) *
          tracker_options_.feature_cells;
      const uint32_t addressy =
          (center_pix[1] / image_pyrmaid_[0].rows) *
          tracker_options_.feature_cells;
      if (addressy > feature_cells_.rows() ||
          addressx > feature_cells_.cols()) {
        std::cerr << "Out of bounds feature cell access at : " << addressy <<
                     ", " << addressx << std::endl;
      }
      feature_cells_(addressy, addressx)++;

      if (track->keypoints.size() > 0) {
        average_track_length +=
            (track->keypoints.back() - track->keypoints.front()).norm();
      }
    } else {
      track->is_outlier = true;
    }
  }

  if (current_tracks_.size() > 0) {
    average_track_length /= current_tracks_.size();
  }
}

void SemiDenseTracker::TransformTrackTabs(const Sophus::SE3d &t_cb)
{
  // Multiply the t_ba of all tracks by the current delta.
  for (std::shared_ptr<DenseTrack>& track : current_tracks_)
  {
    track->t_ba = t_cb * track->t_ba;
  }
}

void SemiDenseTracker::OptimizeTracks(int level, bool optimize_landmarks)
{
  const double time = Tic();
  OptimizationStats stats;
  OptimizationOptions options;
  if (level == -1) {
    // Initialize the dense matrices that will be used to solve the problem.
    for (int ii = tracker_options_.pyramid_levels - 1 ; ii >= 0 ; ii--) {
      options.optimize_landmarks = !(ii > 1 && average_track_length > 10);
      uint32_t iterations = 0;
      do {
        Sophus::SE3d t_ba_old_ = t_ba_;
        OptimizePyramidLevel(ii, image_pyrmaid_, current_tracks_,
                             options, stats);

        double post_error = EvaluateTrackResiduals(
              ii, image_pyrmaid_, current_tracks_, false, true);

        // std::cerr << "deltap: " << stats.delta_pose_norm << " deltal: " <<
        //              stats.delta_lm_norm << " prev rmse: " <<
        //              stats.pre_solve_error << " post rmse: " <<
        //              post_error << std::endl;

        if (post_error > stats.pre_solve_error) {
           // std::cerr << "Exiting due to " << post_error << " > " <<
           //              stats.pre_solve_error << " rolling back. " <<
           //              std::endl;

          // Roll back the changes.
          t_ba_ = t_ba_old_;
          for (std::shared_ptr<DenseTrack> track : current_tracks_) {
           // Only roll back tracks that were in the optimization.
           if (track->opt_id == UINT_MAX) {
             continue;
           }

           if (track->inverse_depth_ray) {
             track->ref_keypoint.rho = track->ref_keypoint.old_rho;
           } else {
             track->ref_keypoint.ray_delta =
                 track->ref_keypoint.old_ray_delta;
           }
          }
          break;
        }
        const double change = post_error == 0 ? 0 :
          fabs(stats.pre_solve_error - post_error) / post_error;
        if (change < 0.01) {
           // std::cerr << "Exiting due to change% = " << change <<
           //              " prev error " << stats.pre_solve_error <<
           //              " post : " << post_error << std::endl;
          break;
        }

         // std::cerr << "prev: " << stats.pre_solve_error << " post: " <<
         //             post_error << std::endl;
        post_error = stats.pre_solve_error;
        iterations++;
      } while (stats.delta_pose_norm > 1e-4 || stats.delta_lm_norm > 1e-4);

       // std::cerr << "Optimized level " << ii << " for " << iterations <<
       //              " iterations with optimize_lm = " <<
       //             options.optimize_landmarks << " logest track: " <<
       //               longest_track_id_ << std::endl;

    }
  } else {
    options.optimize_landmarks = optimize_landmarks;
    OptimizePyramidLevel(level, image_pyrmaid_, current_tracks_,
                         options, stats);
    ///zzzzzz evaluate residuals at all levels so we can see
    for (uint32_t ii = 0 ; ii < tracker_options_.pyramid_levels ; ++ii)  {
      if (ii != level) {
        std::cerr << "post rmse: " << ii << " " << EvaluateTrackResiduals(
                       ii, image_pyrmaid_, current_tracks_, false, true) <<
                     std::endl;
      }
    }
    double post_error = EvaluateTrackResiduals(
          level, image_pyrmaid_, current_tracks_, false, true);
    std::cerr << "post rmse: " << post_error << " " << "pre : " <<
                 stats.pre_solve_error << std::endl;
  }

  // Reproject patch centers. This will add to the keypoints vector in each
  // patch which is used to pass on center values to an outside 2d BA.
  ReprojectTrackCenters();


  // Print pre-post errors
  std::cerr << "Level " << level << " solve took " << Toc(time) << "s" <<
               " with delta_p_norm: " << stats.delta_pose_norm << " and delta "
               "lm norm: " << stats.delta_lm_norm << std::endl;
}

void SemiDenseTracker::PruneTracks()
{
  if (image_pyrmaid_.size() == 0) {
    return;
  }

  num_successful_tracks_ = 0;
  // OptimizeTracks();

  std::list<std::shared_ptr<DenseTrack>>::iterator iter =
      current_tracks_.begin();
  while (iter != current_tracks_.end())
  {
    std::shared_ptr<DenseTrack> track = *iter;
#if (LM_DIM == 3)
    // Update the track rho based on the new 3d point.
    const double ratio =
        (track->ref_keypoint.ray + track->ref_keypoint.ray_delta).norm() /
        track->ref_keypoint.ray.norm();
    track->ref_keypoint.rho /= ratio;
#endif

    const double percent_tracked =
        ((double)track->tracked_pixels / (double)track->pixels_attempted);
    // std::cerr << "rmse for track " << track->rmse << " tracked: " <<
    //             percent_tracked << std::endl;
    if (track->ncc > tracker_options_.dense_ncc_threshold &&
        percent_tracked == 1.0 /*&& track->center_error < 1.0*/) {
      track->tracked = true;
      track->keypoints_tracked.back() = true;
      track->num_good_tracked_frames++;

      // Update the error statistics for each pixel of the patch
      for (Patch& patch : track->ref_keypoint.patch_pyramid) {
        for (int jj = 0; jj < patch.residuals.size(); ++jj) {
          // Now update the statistics for this particular ray.
          const uint32_t n = track->keypoints.size();
          const double res = fabs(patch.residuals[jj]);
          if (n == 1) {
            patch.statistics[jj][0] = res;
            patch.statistics[jj][1] = 1.0;
          } else {
            const double delta = res - patch.statistics[jj][0];
            patch.statistics[jj][0] += delta / n;
            patch.statistics[jj][1] += delta * (res - patch.statistics[jj][0]);
          }
        }

      }

      num_successful_tracks_++;
      ++iter;
    } else {
      // std::cerr << "deleting track with opt_id " << (*iter)->opt_id <<
      //              std::endl;
      iter = current_tracks_.erase(iter);
    }
  }
}

void SemiDenseTracker::PruneOutliers()
{
  num_successful_tracks_ = 0;
  std::list<std::shared_ptr<DenseTrack>>::iterator iter =
      current_tracks_.begin();
  while (iter != current_tracks_.end())
  {
    std::shared_ptr<DenseTrack> track = *iter;
    if (track->is_outlier) {
      iter = current_tracks_.erase(iter);
    } else {
      num_successful_tracks_ ++;
      ++iter;
    }
  }
}


void SemiDenseTracker::TransferPatch(std::shared_ptr<DenseTrack> track,
                                     uint32_t level,
                                     const Sophus::SE3d &t_ba,
                                     calibu::CameraInterface<Scalar> *cam,
                                     PatchTransfer& result,
                                     bool transfer_jacobians)
{
  Eigen::Vector4t ray;
  DenseKeypoint& ref_kp = track->ref_keypoint;
  Patch& ref_patch = ref_kp.patch_pyramid[level];
  result.valid_projections.clear();
  result.valid_rays.clear();
  result.dprojections.clear();
  result.valid_projections.reserve(ref_patch.rays.size());
  result.valid_rays.reserve(ref_patch.rays.size());
  result.dprojections.reserve(ref_patch.rays.size());
  result.pixels_attempted = 0;
  ref_patch.projected_values.clear();
  result.mean_value = 0;

  // If we are doing simplified (4 corner) patch transfer, transfer the four
  // corners.
  bool corners_project = true;
  Eigen::Vector2t corner_projections[4];
  Eigen::Matrix<double, 2, 4> corner_dprojections[4];
  for (int ii = 0 ; ii < 4 ; ++ii) {
    const Eigen::Vector3t& corner_ray =
        ref_patch.rays[pyramid_patch_corner_dims_[level][ii]] +
        ref_kp.ray_delta;
    corner_projections[ii] = cam->Transfer3d(t_ba, corner_ray, ref_kp.rho);

    if (transfer_jacobians) {
      ray.head<3>() = corner_ray;
      ray[3] = ref_kp.rho;
      const Eigen::Vector4t ray_b = MultHomogeneous(t_ba, ray);
      corner_dprojections[ii] =
          cam->dTransfer3d_dray(Sophus::SE3d(), ray_b.head<3>(), ray_b[3]);
    }

    // std::cerr << "Corner 0 at index " << ii << " reprojected to " <<
    //               corner_projections[ii].transpose() << std::endl;

    // if (!IsReprojectionValid(corner_projections[ii], image_pyrmaid_[0])) {
    //   corners_project = false;
    // std::cerr << "Corner " << ii << " didn't project." << std::endl;
    //   break;
    // }
  }

  if (!corners_project) {
    return;
  }

  // Transfer the center ray.
  const Eigen::Vector3t center_ray = ref_kp.ray + ref_kp.ray_delta;
  // Reproject the center and form a residual.
  result.center_projection =
      cam->Transfer3d(Sophus::SE3d(), center_ray, ref_kp.rho);

  // If the user requested jacobians, get the center ray jacobian in the ref
  // frame.
  if (transfer_jacobians) {
    result.center_dprojection =
        cam->dTransfer3d_dray(
          Sophus::SE3d(), center_ray, ref_kp.rho).topLeftCorner<2, 3>();
  }

  // First project the entire patch and see if it falls within the bounds of
  // the image.
  for (size_t ii = 0; ii < ref_patch.rays.size() ; ++ii) {
    const double tl_factor = pyramid_patch_interp_factors_[level][ii][0];
    const double tr_factor = pyramid_patch_interp_factors_[level][ii][1];
    const double bl_factor = pyramid_patch_interp_factors_[level][ii][2];
    const double br_factor = pyramid_patch_interp_factors_[level][ii][3];

    // First transfer this pixel over to our current image.
    Eigen::Vector2t pix;// = cam->Transfer3d(t_ba, ref_patch.rays[ii],
    //                    ref_kp.rho);

    if (corners_project) {
      // std::cerr << "prev pix: " << pix.transpose() << std::endl;
      // linearly interpolate this
      pix =
          tl_factor * corner_projections[0] +
          tr_factor * corner_projections[1] +
          bl_factor * corner_projections[2] +
          br_factor * corner_projections[3];
      // std::cerr << "post pix: " << pix.transpose() << "\n" << std::endl;
    }

    result.pixels_attempted++;
    // Check bounds
    ref_patch.projections[ii] = pix;
    pix[0] *= pyramid_coord_ratio_[level][0];
    pix[1] *= pyramid_coord_ratio_[level][1];

    if (!IsReprojectionValid(pix, image_pyrmaid_[level])) {
      // std::cerr << "Reprojection at " << pix.transpose() <<
      //              " not valid. " << std::endl;
      ref_patch.projected_values[ii] = 0;
      ref_patch.residuals[ii] = 0;
      continue;
    } else {
      // ray.head<3>() = ref_patch.rays[ii] + ref_kp.ray_delta;
      // ray[3] = ref_kp.rho;
      // const Eigen::Vector4t ray_b = Sophus::MultHomogeneous(t_ba, ray);

      // Eigen::Matrix<double, 2, 4> dprojection_dray =
      //     cam->dTransfer3d_dray(Sophus::SE3d(), ray_b.head<3>(), ray_b[3]);

      if (corners_project && transfer_jacobians) {
        // std::cerr << "Prev dprojection: " << std::endl << dprojection_dray <<
        //              std::endl;
        result.dprojections.push_back(
              tl_factor * corner_dprojections[0] +
            tr_factor * corner_dprojections[1] +
            bl_factor * corner_dprojections[2] +
            br_factor * corner_dprojections[3]
            );
        //  std::cerr << "new dprojection: " << std::endl <<
        //               result.dprojections.back() << "\n" <<  std::endl;
      }

      result.valid_projections.push_back(pix);
      result.valid_rays.push_back(ii);
      if (std::isnan(pix[0]) || std::isnan(pix[1])) {
        std::cerr << "Pixel at: " << pix.transpose() <<
                     " center ray: " << ref_kp.ray + ref_kp.ray_delta << " rho: " <<
                     ref_kp.rho << " inv depth: " << track->inverse_depth_ray <<
                     std::endl;
      }
      const double val = GetSubPix(image_pyrmaid_[level], pix[0], pix[1]);
      ref_patch.projected_values[ii] = val;
      result.mean_value += val;
    }
  }

  // Calculate the mean value
  if (result.valid_rays.size()) {
    result.mean_value /= result.valid_rays.size();
  } else {
    result.mean_value = 0;
  }
  ref_patch.projected_mean = result.mean_value;
}

void SemiDenseTracker::StartNewLandmarks()
{
  // Figure out the number of landmarks per cell that's required.
  lm_per_cell_ = tracker_options_.num_active_tracks /
      powi(tracker_options_.feature_cells, 2);

  std::vector<cv::KeyPoint> cv_keypoints;
  // Extract features and descriptors from this image
  ExtractKeypoints(image_pyrmaid_[0], cv_keypoints);


  // Afterwards, we must spawn a number of new tracks
  const int num_new_tracks =
      std::max(0, (int)tracker_options_.num_active_tracks -
               (int)num_successful_tracks_);

  const uint32_t started =
      StartNewTracks(image_pyrmaid_, cv_keypoints, num_new_tracks);

  std::cerr << "Tracked: " << num_successful_tracks_ << " started " <<
               started << " out of " << num_new_tracks << " new tracks with " <<
               cv_keypoints.size() << " keypoints " << std::endl;
}

void SemiDenseTracker::OptimizePyramidLevel(uint32_t level,
    const std::vector<cv::Mat>& image_pyrmaid,
    std::list<std::shared_ptr<DenseTrack>> &tracks,
    const OptimizationOptions& options,
    OptimizationStats& stats)
{
  static Eigen::Matrix<double, 6, 6> u;
  static Eigen::Matrix<double, 6, 1> r_p;
  u.setZero();
  r_p.setZero();
  Eigen::Matrix<double, 6, LM_DIM> w;
  static std::vector<Eigen::Matrix<double, 6, LM_DIM>> w_vec;
  Eigen::Matrix<double, LM_DIM, LM_DIM> v;
  static std::vector<Eigen::Matrix<double, LM_DIM, LM_DIM>> v_inv_vec;
  Eigen::Matrix<double, LM_DIM, 1> r_l;
  Eigen::VectorXd r_l_vec(tracks.size() * LM_DIM);

  v_inv_vec.resize(tracks.size());
  w_vec.resize(v_inv_vec.size());

  Eigen::Matrix<double, 2, 6> dp_dx;
  Eigen::Matrix<double, 2, 4> dp_dray;
  Eigen::Vector4t ray;
  Eigen::Matrix<double, 2, 4> dprojection_dray;
  static std::vector<Eigen::Matrix<double, 1, 6>> di_dx;
  static std::vector<Eigen::Matrix<double, 1, LM_DIM>> di_dray;
  static std::vector<double> res;
  di_dx.clear();
  di_dray.clear();
  res.clear();
  Eigen::Matrix<double, 1, 6> mean_di_dx;
  Eigen::Matrix<double, 1, LM_DIM> mean_di_dray;
  Eigen::Matrix<double, 1, 6> final_di_dx;
  Eigen::Matrix<double, 1, LM_DIM> final_di_dray;
  // std::vector<Eigen::Vector2t> valid_projections;
  // std::vector<unsigned int> valid_rays;

  uint32_t track_id = 0;
  uint32_t residual_id = 0;
  uint32_t num_inliers = 0;


  double track_residual;
  double residual = 0;
  uint32_t residual_count = 0;
  uint32_t residual_offset = 0;
  const Sophus::SE3d t_vc = camera_rig_->t_wc_[0];
  const Sophus::SE3d t_cv = camera_rig_->t_wc_[0].inverse();
  const Eigen::Matrix4d t_cv_mat = t_cv.matrix();

  // First project all tracks into this frame and form
  // the localization step.
  for (std::shared_ptr<DenseTrack>& track : tracks)
  {
    track->opt_id = UINT_MAX;
    track->residual_used = false;
    // If we are not solving for landmarks, there is no point including
    // uninitialized landmarks in the camera pose estimation
    if (options.optimize_landmarks == 0 &&
        track->keypoints.size() < MIN_OBS_FOR_CAM_LOCALIZATION) {
      continue;
    }
    track_residual = 0;
    // If there are only two keypoints, we want to optimize this landmark in
    // inverse depth, as the epipolar line can be directly optimize to yield the
    // minimum appearance error.
    track->inverse_depth_ray = track->keypoints.size() == 2;
    PatchTransfer& transfer = track->transfer;

    const Sophus::SE3d track_t_ba = t_cv * t_ba_ * track->t_ba * t_vc;
    const Eigen::Matrix4d track_t_ba_matrix = track_t_ba.matrix();
    // Project into the image and form the problem.
    DenseKeypoint& ref_kp = track->ref_keypoint;
    Patch& ref_patch = ref_kp.patch_pyramid[level];

    track->tracked_pixels = 0;
    track->pixels_attempted = transfer.pixels_attempted;
    track->rmse = 0;

    // Prepare the w matrix. We will add to it as we go through the rays.
    w.setZero();
    // Same for the v matrix
    v.setZero();
    // Same for the RHS subtraction term
    r_l.setZero();

    if (options.transfer_patchs) {
      TransferPatch(track, level, track_t_ba, camera_rig_->cameras_[0], transfer,
          true);
    }

    // Do not use this patch if less than half of its pixels reprojcet.
    if (transfer.valid_projections.size() < ref_patch.rays.size() / 2) {
      // std::cerr << "Rejecting track due to lack of projections." << std::endl;
      continue;
    }


    track->residual_offset = residual_offset;
    residual_offset += track->inverse_depth_ray ? 1 : 3;

    di_dx.resize(transfer.valid_rays.size());
    di_dray.resize(transfer.valid_rays.size());
    res.resize(transfer.valid_rays.size());
    mean_di_dray.setZero();
    mean_di_dx.setZero();
    double ncc_num = 0, ncc_den_a = 0, ncc_den_b = 0;
    for (size_t kk = 0; kk < transfer.valid_rays.size() ; ++kk)
    {
      const size_t ii = transfer.valid_rays[kk];
      // First transfer this pixel over to our current image.
      const Eigen::Vector2t pix = transfer.valid_projections[kk];

      // need 2x6 transfer residual
      ray.head<3>() = ref_patch.rays[ii] + ref_kp.ray_delta;
      ray[3] = ref_kp.rho;
      const Eigen::Vector4t ray_v = MultHomogeneous(
            t_ba_ * track->t_ba * t_vc, ray);

      dprojection_dray = transfer.dprojections[kk];

      /// ZZZZZZZZZZZZZZ
      /// this needs to be a proper pre/post transform from the top level pixel
      /// space to the pyramid level pixel space
      dprojection_dray *= pyramid_coord_ratio_[level][0];

      for (unsigned int jj = 0; jj < 6; ++jj) {
        dp_dx.block<2,1>(0,jj) =
            //dprojection_dray * Sophus::SE3d::generator(jj) * ray;
            dprojection_dray * t_cv_mat * generators_[jj] * ray_v;
      }

      // need 2x4 transfer w.r.t. reference ray
      dp_dray = dprojection_dray * track_t_ba_matrix;

      double eps = 1e-9;
      const double val_pix = ref_patch.projected_values[ii];
      const double valx_pix = GetSubPix(image_pyrmaid[level],
                                        pix[0] + eps, pix[1]);
      const double valy_pix = GetSubPix(image_pyrmaid[level],
                                        pix[0], pix[1] + eps);
      Eigen::Matrix<double, 1, 2> di_dp;
      di_dp[0] = (valx_pix - val_pix)/(eps);
      di_dp[1] = (valy_pix - val_pix)/(eps);

      di_dx[kk] = di_dp * dp_dx;
#if (LM_DIM == 1)
      di_dray[kk] = di_dp * dp_dray.col(3);
#else
      if (track->inverse_depth_ray) {
        di_dray[kk](0) = di_dp * dp_dray.col(3);
      } else {
        di_dray[kk] = di_dp * dp_dray.topLeftCorner<2,3>();
      }
#endif

      // Insert the residual.
      const Scalar c_huber = 1.2107 * pyramid_error_thresholds_[level];
      const double mean_s_ref = ref_patch.values[ii] - ref_patch.mean;
      const double mean_s_proj = val_pix - transfer.mean_value;
      res[kk] = mean_s_proj - mean_s_ref;
      bool inlier = true;
      if (tracker_options_.use_robust_norm_) {
        const double weight_sqrt = sqrt(1.0 / ref_patch.statistics[ii][1]);
            // sqrt(fabs(res[kk]) > c_huber ? c_huber / fabs(res[kk]) : 1.0);
         std::cerr << "Weight for " << res[kk] << " at level " << level <<
                      " is " << weight_sqrt * weight_sqrt << std::endl;
        res[kk] *= weight_sqrt;
        di_dx[kk] *= weight_sqrt;
        di_dray[kk] *= weight_sqrt;
        if (weight_sqrt != 1) {
          inlier = false;
        }
      }
      const double res_sqr = res[kk] * res[kk];

      if (inlier) {
        track->rmse += res_sqr;
        ncc_num += mean_s_ref * mean_s_proj;
        ncc_den_a += mean_s_ref * mean_s_ref;
        ncc_den_b += mean_s_proj * mean_s_proj;
        num_inliers++;
      }

      mean_di_dray += di_dray[kk];
      mean_di_dx += di_dx[kk];

      ref_patch.residuals[ii] = res[kk];
      residual_count++;
      track_residual += res_sqr;

      track->tracked_pixels++;
    }

    mean_di_dray /= transfer.valid_rays.size();
    mean_di_dx /= transfer.valid_rays.size();

    for (size_t kk = 0; kk < transfer.valid_rays.size() ; ++kk)
    {
      final_di_dray = di_dray[kk] - mean_di_dray;
      final_di_dx = di_dx[kk] - mean_di_dx;

      if (options.optimize_landmarks) {
        if (track->inverse_depth_ray) {
          const double di_dray_id = final_di_dray(0);
          // Add the contribution of this ray to the w and v matrices.
          w.col(0) += final_di_dx.transpose() * di_dray_id;
          v(0) += di_dray_id * di_dray_id;

          // Add contribution for the subraction term on the rhs.
          r_l(0) += di_dray_id * res[kk];
        } else {
          // Add the contribution of this ray to the w and v matrices.
          w += final_di_dx.transpose() * final_di_dray;
          v += final_di_dray.transpose() * final_di_dray;

          // Add contribution for the subraction term on the rhs.
          r_l += final_di_dray.transpose() * res[kk];
        }
      }

      // Update u by adding j_p' * j_p
      u += final_di_dx.transpose() * final_di_dx;
      // Update rp by adding j_p' * r
      r_p += final_di_dx.transpose() * res[kk];

      residual_id++;
    }

    // If this landmark is the longest track, we omit it to fix scale.
    if (options.optimize_landmarks && track->id != longest_track_id_) {
      track->opt_id = track_id;
      double regularizer = level >= 2 ? 1e3 : level == 1 ? 1e2 : 1e1;
      if (track->inverse_depth_ray) {
        v(0) += regularizer;
        const double v_inv = 1.0 / v(0);
        v_inv_vec[track_id](0) = v_inv;
        w_vec[track_id].col(0) = w.col(0);
        r_l_vec(track->residual_offset) = r_l(0);
        // Subtract the contribution of these residuals from u and r_p
        u -= w.col(0) * v_inv * w.col(0).transpose();
        r_p -= w.col(0) * v_inv * r_l(0);
      } else {
#if (LM_DIM == 3)
        const double center_weight = CENTER_WEIGHT;
        // Reproject the center and form a residual.
        const Eigen::Vector2t& center_pix = transfer.center_projection;
        // Form the residual between the center pix and the original pixel loc.
        const Eigen::Vector2t c_res = (center_pix - ref_kp.center_px);
        track->center_error = c_res.norm();
        // std::cerr << " Center error for " << track_id << ": " <<
        //              (center_pix - ref_kp.center_px).transpose() << std::endl;

        const Eigen::Matrix<double, 2, 3>& dp_dray3d =
            transfer.center_dprojection;

        v += dp_dray3d.transpose() * center_weight * dp_dray3d;
        // Add contribution for the subraction term on the rhs.
        r_l += dp_dray3d.transpose() * center_weight * c_res;

        const Eigen::Vector3t eivals =
            v.selfadjointView<Eigen::Lower>().eigenvalues();
        const double cond = eivals.maxCoeff() / eivals.minCoeff();
        // Skip this track if it's not observable.
        if (fabs(cond) > 1e6) {
          track->opt_id = UINT_MAX;
          continue;
        }

        residual += c_res.transpose() * center_weight * c_res;
        v_inv_vec[track_id] = v.inverse();
#endif
        w_vec[track_id] = w;
        r_l_vec.segment<LM_DIM>(track->residual_offset) = r_l;
        // Subtract the contribution of these residuals from u and r_p
        u -= w * v_inv_vec[track_id] * w.transpose();
        r_p -= w * v_inv_vec[track_id] * r_l;
      }

      track_id++;
    }

    // Add to the overal residual here, as we're sure the track will be
    // included in the optimization.
    residual += track_residual;
    track->residual_used = true;

    // Compute the track RMSE and NCC scores.
    track->rmse = track->tracked_pixels == 0 ?
          1e9 : sqrt(track->rmse / num_inliers);
    const double denom = sqrt(ncc_den_a * ncc_den_b);
    track->ncc = denom == 0 ? 0 : ncc_num / denom;
    // std::cerr << "track rmse for level " << level << " : " << track->rmse <<
    //              std::endl;
  }


  // Solve for the pose update
  Eigen::LDLT<Eigen::Matrix<Scalar, 6, 6>> solver;
  solver.compute(u);
  Eigen::Matrix<double, 6, 1> delta_p = solver.solve(r_p);
  // std::cerr << "delta_p: " << delta_p.transpose() << std::endl;
  t_ba_ = Sophus::SE3d::exp(-delta_p) * t_ba_;
  // std::cerr << "t_ba = " << std::endl << t_ba_.matrix() <<
  //              std::endl;

  stats.delta_lm_norm = 0;
  uint32_t delta_lm_count = 0;
  // Now back-substitute all the keypoints
  if (options.optimize_landmarks) {
    for (std::shared_ptr<DenseTrack>& track : tracks)
    {
      if (track->opt_id != UINT_MAX)
      {
        delta_lm_count++;
        if (track->inverse_depth_ray) {
          double delta_ray = v_inv_vec[track->opt_id](0) *
              (r_l_vec(track->residual_offset) -
               w_vec[track->opt_id].col(0).transpose() * delta_p);
          track->ref_keypoint.old_rho = track->ref_keypoint.rho;
          track->ref_keypoint.rho -= delta_ray;
          stats.delta_lm_norm += fabs(delta_ray);
          if (std::isnan(delta_ray) || std::isinf(delta_ray)) {
            std::cerr << "delta_ray " << track->id << ": " << delta_ray <<
                         "vinv:" << v_inv_vec[track->opt_id](0) << " r_l " <<
                         r_l_vec(track->residual_offset) << " w: " <<
                         w_vec[track->opt_id].col(0).transpose() << "dp : " <<
                         delta_p.transpose() << std::endl;
          }

          if (track->ref_keypoint.rho < 0) {
            /*std::cerr << "rho negative: " << track->ref_keypoint.rho <<
                         "inverse depth: " << track->inverse_depth_ray <<
                         " from : " << track->ref_keypoint.old_rho <<
                         std::endl;*/
            track->ref_keypoint.rho = 1e-3;
          }
        } else {
          Eigen::Matrix<double, 3, 1> delta_lm =
              v_inv_vec[track->opt_id] *
              (r_l_vec.segment<3>(track->residual_offset) -
               w_vec[track->opt_id].transpose() * delta_p);
          // Store the old ray delta in case we need to roll back.
          track->ref_keypoint.old_ray_delta = track->ref_keypoint.ray_delta;
          track->ref_keypoint.ray_delta -= delta_lm;
          stats.delta_lm_norm += delta_lm.norm();

          if (std::isnan(delta_lm[0]) || std::isinf(delta_lm[0]) ||
              std::isnan(delta_lm[1]) || std::isinf(delta_lm[1]) ||
              std::isnan(delta_lm[2]) || std::isinf(delta_lm[2])) {
            std::cerr << "delta_ray " << track->id << ": " << delta_lm.transpose() <<
                         "vinv\n:" << v_inv_vec[track->opt_id] << "\n r_l " <<
                         r_l_vec.segment<3>(track->residual_offset).transpose() <<
                         " w: \n" << w_vec[track->opt_id] <<
                         "\ndp : " << delta_p.transpose() << std::endl;
          }
        }
      }
    }
  }

  // set the optimization stats.
  stats.pre_solve_error = sqrt(residual);
  stats.delta_pose_norm = delta_p.norm();
  stats.delta_lm_norm = delta_lm_count == 0 ? 0 :
                                              stats.delta_lm_norm / delta_lm_count;
}


double SemiDenseTracker::GetSubPix(const cv::Mat &image, double x, double y)
{
  return Interpolate(x, y, image.data, image.cols, image.rows);
}

void SemiDenseTracker::AddImage(const cv::Mat &image,
                                const Sophus::SE3d &t_ba_guess)
{
  // If there were any outliers (externally marked), now is the time to prune
  // them.
  PruneOutliers();

  mask_.Clear();
  t_ba_ = t_ba_guess;
  // Create the image pyramid for the incoming image
  cv::buildPyramid(image, image_pyrmaid_, tracker_options_.pyramid_levels);

  for (uint32_t ii = 0 ; ii < tracker_options_.pyramid_levels ; ++ii) {
    pyramid_coord_ratio_[ii][0] =
        (double)(image_pyrmaid_[ii].cols) /
        (double)(image_pyrmaid_[0].cols);
    pyramid_coord_ratio_[ii][1] =
        (double)(image_pyrmaid_[ii].rows) /
        (double)(image_pyrmaid_[0].rows);

    // std::cerr << "image_pyramid[" << ii << "]: " <<
    //              pyramid_coord_ratio_[ii].transpose() << "image size: " <<
    //              image_pyrmaid_[ii].cols << ", " << image_pyrmaid_[ii].rows <<
    //              std::endl;
  }

  if (last_image_was_keyframe_) {
    uint32_t max_length = 0;
    for (std::shared_ptr<DenseTrack>& track : current_tracks_)
    {
#if (LM_DIM == 3)
      track->ref_keypoint.ray_delta.setZero();
#endif
      track->keypoints.push_back(Eigen::Vector2t());
      track->keypoint_external_data.push_back(UINT_MAX);
      track->keypoints_tracked.push_back(false);

      if (track->keypoints.size() > max_length) {
        longest_track_id_ = track->id;
        max_length = track->keypoints.size();
      }
    }
  }

  last_image_was_keyframe_ = false;
}
