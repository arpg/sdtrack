#include <glog/logging.h>
#include <sdtrack/semi_dense_tracker.h>
#include <sdtrack/parallel_algos.h>
#include <CVars/CVar.h>

using namespace sdtrack;

int& g_sdtrack_debug =
    CVarUtils::CreateCVar<int>("debug.sdtrack", google::INFO + 1, "");

void SemiDenseTracker::Initialize(const KeypointOptions& keypoint_options,
                                  const TrackerOptions& tracker_options,
                                  calibu::Rig<Scalar>* rig) {
  // rig_ = rig;
  // const calibu::CameraModelGeneric<Scalar>& cam = rig->cameras[0].camera;
  // camera_rig_->AddCamera(calibu::CreateFromOldCamera<Scalar>(cam),
  //                       rig->cameras[0].T_wc);
  t_ba_ = Sophus::SE3d();
  camera_rig_ = rig;
  num_cameras_ = camera_rig_->cameras_.size();
  image_pyramid_.resize(num_cameras_);

  keypoint_options_ = keypoint_options;
  tracker_options_ = tracker_options;
  next_track_id_ = 0;
  switch (tracker_options_.detector_type) {
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
      //detector_ = new cv::SurfFeatureDetector(1000);
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
    LOG(INFO) << "Level " << ii << " patch dim is " << patch_dim;
    pyramid_error_thresholds_[ii] = robust_norm_thresh;
    robust_norm_thresh *= 2;
    pyramid_patch_dims_[ii] = patch_dim;
    // The array coordinates of the four corners of the patch.
    pyramid_patch_corner_dims_[ii] = {0,                                  // tl
                                      patch_dim - 1,                      // tr
                                      patch_dim* patch_dim - patch_dim,   // bl
                                      patch_dim* patch_dim - 1
    };         // br

    // For each cell, we also need to get the interpolation factors.
    pyramid_patch_interp_factors_[ii].reserve(powi(patch_dim, 2));
    const double factor = powi(patch_dim - 1, 2);
    for (double yy = 0; yy < patch_dim ; ++yy) {
      for (double xx = 0; xx < patch_dim ; ++xx) {
        pyramid_patch_interp_factors_[ii].push_back({
            ((patch_dim - 1 - xx) * (patch_dim - 1 - yy)) / factor,     // tl
                (xx * (patch_dim - 1 - yy)) / factor,     // tr
                ((patch_dim - 1 - xx) * yy) / factor,     // bl
                (xx * yy) / factor     // br
                });
      }
    }
    // patch_dim = (patch_dim + 1) / 2;
  }

  for (int ii = 0; ii < 6 ; ++ii) {
    generators_[ii] = Sophus::SE3d::generator(ii);
  }

  feature_cells_.resize(num_cameras_);
  active_feature_cells_.resize(num_cameras_);
  for (size_t cam_id = 0; cam_id < num_cameras_; ++cam_id) {
    // Inititalize the feature cells.
    feature_cells_[cam_id].resize(tracker_options_.feature_cells,
                                  tracker_options_.feature_cells);
    feature_cells_[cam_id].setZero();
    active_feature_cells_[cam_id] = tracker_options_.feature_cells *
        tracker_options_.feature_cells;
  }

  for (size_t ii = 0; ii < rig->cameras_.size(); ++ii) {
    mask_.AddImage(rig->cameras_[ii]->Width(),
                   rig->cameras_[ii]->Height());
  }

  pyramid_coord_ratio_.resize(tracker_options_.pyramid_levels);
  current_tracks_.clear();
  new_tracks_.clear();
  previous_keypoints_.clear();
}

void SemiDenseTracker::ExtractKeypoints(const cv::Mat& image,
                                        std::vector<cv::KeyPoint>& keypoints,
                                        uint32_t cam_id) {
  const double req_lm_per_cell = (double)tracker_options_.num_active_tracks /
      active_feature_cells_[cam_id];
  std::vector<cv::KeyPoint> cell_kp;
  keypoints.clear();
  keypoints.reserve(keypoint_options_.max_num_features);
  uint32_t cell_width = image.cols / tracker_options_.feature_cells;
  uint32_t cell_height = image.rows / tracker_options_.feature_cells;
  uint32_t cells_hit = 0;
  double time = Tic();

  std::vector<cv::Rect> bounds_vec;
  bounds_vec.reserve(powi(tracker_options_.feature_cells,2));

  for (uint32_t ii = 0  ; ii < tracker_options_.feature_cells ; ++ii) {
    for (uint32_t jj = 0  ; jj < tracker_options_.feature_cells ; ++jj) {
      const auto feature_cell = feature_cells_[cam_id](jj, ii);
      if (feature_cell >= req_lm_per_cell || feature_cell == kUnusedCell) {
        continue;
      }

      const cv::Rect bounds(ii * cell_width, jj * cell_height,
                            cell_width, cell_height);
      bounds_vec.push_back(bounds);
    }
  }


  ParallelExtractKeypoints extractor(*this, image, bounds_vec);

  tbb::parallel_reduce(tbb::blocked_range<int>(0, bounds_vec.size()),
                       extractor);

  keypoints = extractor.keypoints;

  if (tracker_options_.do_corner_subpixel_refinement) {
    std::vector<cv::Point2f> subpixel_centers(keypoints.size());
    for (uint32_t ii = 0 ; ii < keypoints.size() ; ++ii) {
      subpixel_centers[ii] = keypoints[ii].pt;
    }
    cv::TermCriteria criteria(cv::TermCriteria::COUNT, 10, 0);
    cv::cornerSubPix(image, subpixel_centers,
                     cv::Size(tracker_options_.patch_dim,
                              tracker_options_.patch_dim), cv::Size(-1, -1),
                     criteria);
    for (uint32_t ii = 0 ; ii < keypoints.size() ; ++ii) {
      // LOG(INFO) << "kp " << ii << " refined from " << keypoints[ii].pt.x << ", "
      //           << keypoints[ii].pt.y << " to " << subpixel_centers[ii].x <<
      //              ", " << subpixel_centers[ii].y << std::endl;
      keypoints[ii].pt = subpixel_centers[ii];
    }
  }

  HarrisScore(image.data, image.cols, image.rows,
              tracker_options_.patch_dim, keypoints);
  LOG(INFO) << "extract feature detection for " << keypoints.size() <<
      " and "  << cells_hit << " cells " <<  " keypoints took " <<
      Toc(time) << " seconds." << std::endl;
}

bool SemiDenseTracker::IsKeypointValid(const cv::KeyPoint& kp,
                                       uint32_t image_width,
                                       uint32_t image_height,
                                       uint32_t cam_id) {
  // Only for the car dataset.
  //  if (kp.pt.y > 400) {
  //    return false;
  //  }

  if (kp.response < 200 || (kp.angle / kp.response) > 3.0
      /*tracker_options_.harris_score_threshold*/) {
    return false;
  }

  uint32_t margin =
      (tracker_options_.patch_dim + 1) * tracker_options_.pyramid_levels;
  // Key keypoint also can't be closer than a certain amount to the edge.
  if (kp.pt.x < margin || kp.pt.y < margin ||
      kp.pt.x > image_width - margin || kp.pt.y > image_height - margin) {
    return false;
  }

  // Make sure this keypoint isn't already ina a patch somewhere.
  if (mask_.GetMask(cam_id, kp.pt.x, kp.pt.y)) {
    return false;
  }

  return true;
}

bool SemiDenseTracker::IsReprojectionValid(const Eigen::Vector2t& pix,
                                           const cv::Mat& image) {
  if (pix[0] <= 2 || pix[0] > (image.cols - 2)) {
    return false;
  }

  if (pix[1] <= 2 || pix[1] > (image.rows - 2)) {
    return false;
  }

  return true;
}

void SemiDenseTracker::BackProjectTrack(std::shared_ptr<DenseTrack> track,
                                        bool initialize_pixel_vals) {
  const uint32_t cam_id = track->ref_cam_id;
  DenseKeypoint& kp = track->ref_keypoint;
  // Unproject the center pixel for this track.
  kp.ray =
      camera_rig_->cameras_[cam_id]->Unproject(kp.center_px).normalized() *
      tracker_options_.default_ray_depth;

  //LOG(INFO) << "Initializing keypoint at " << kp.pt.x << ", " <<
  //               kp.pt.y << " with response: " << kp.response << std::endl;


  for (uint32_t ii = 0 ; ii < tracker_options_.pyramid_levels ; ++ii) {

    Patch& patch = kp.patch_pyramid[ii];
    // Transform the patch to lower levels
    patch.center[0] = kp.center_px[0] * pyramid_coord_ratio_[ii][0];
    patch.center[1] = kp.center_px[1] * pyramid_coord_ratio_[ii][1];

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
            camera_rig_->cameras_[cam_id]->Unproject(px_level0).normalized() *
            tracker_options_.default_ray_depth;
        if (initialize_pixel_vals) {
          const double val = GetSubPix(image_pyramid_[cam_id][ii], xx, yy);
          patch.values[array_dim] = val;
          track->transfer[cam_id].projected_values[array_dim] =
              patch.values[array_dim];
          track->transfer[cam_id].projections[array_dim] = px_level0;
          mean_value += patch.values[array_dim];
        }
        array_dim++;
      }
    }

    //    LOG(INFO) << "\tCenter at level " << ii << " " << patch.center[0] <<
    //                 ", " << patch.center[1] << " x: " <<
    //                 patch.center[0] - extent << " to " << x_max << " y: " <<
    //                 patch.center[1] - extent << " to " << y_max << std::endl;

    // Calculate the mean, and subtract from all values.
    if (initialize_pixel_vals) {
      mean_value /= patch.values.size();
      patch.mean = mean_value;
    }

    if (array_dim != patch.rays.size()) {
      LOG(FATAL) << "Not enough rays!" << std::endl;
    }
  }
}

uint32_t SemiDenseTracker::StartNewTracks(
    std::vector<cv::Mat>& image_pyrmaid,
    std::vector<cv::KeyPoint>& cv_keypoints,
    uint32_t num_to_start,
    uint32_t cam_id) {
  const double req_lm_per_cell = (double)tracker_options_.num_active_tracks /
      active_feature_cells_[cam_id];

  // Initialize the random inverse depth generator.
  const double range = tracker_options_.default_rho * 0.1;
  std::uniform_real_distribution<double> distribution(
      tracker_options_.default_rho - range,
      tracker_options_.default_rho + range);

  uint32_t num_started = 0;
  // const CameraInterface& cam = *camera_rig_->cameras[0];
  previous_keypoints_.clear();
  previous_keypoints_.reserve(cv_keypoints.size());

  std::sort(cv_keypoints.begin(), cv_keypoints.end(),
            [](const cv::KeyPoint & a, const cv::KeyPoint & b) {
              return a.response > b.response;
            });


  for (cv::KeyPoint& kp : cv_keypoints) {
    if (num_to_start == 0) {
      break;
    }

    // Figure out which feature cell this particular reprojection falls into,
    // and increment that cell
    const uint32_t addressx =
        (kp.pt.x / image_pyrmaid[0].cols) * tracker_options_.feature_cells;
    const uint32_t addressy =
        (kp.pt.y / image_pyrmaid[0].rows) * tracker_options_.feature_cells;

    const auto feature_cell = feature_cells_[cam_id](addressy, addressx);
    if (feature_cell >= req_lm_per_cell || feature_cell == kUnusedCell) {
      continue;
    }

    if (!IsKeypointValid(kp, image_pyrmaid[0].cols, image_pyrmaid[0].rows,
                         cam_id)) {
      continue;
    }

    mask_.SetMask(cam_id, kp.pt.x, kp.pt.y);
    if (feature_cells_[cam_id](addressy, addressx) != kUnusedCell) {
      feature_cells_[cam_id](addressy, addressx)++;
    }


    // Otherwise extract pyramid for this keypoint, and also backproject all
    // pixels as the rays will not change.
    std::shared_ptr<DenseTrack> new_track(
        new DenseTrack(tracker_options_.pyramid_levels, pyramid_patch_dims_,
                       num_cameras_));
    new_track->id = next_track_id_++;
    current_tracks_.push_back(new_track);
    new_tracks_.push_back(new_track);
    DenseKeypoint& new_kp = new_track->ref_keypoint;

    new_kp.response = kp.response;
    new_kp.response2 = kp.angle;
    new_kp.center_px = Eigen::Vector2t(kp.pt.x, kp.pt.y);

    new_track->ref_cam_id = cam_id;
    new_track->keypoints.emplace_back(num_cameras_);

    for (uint32_t ii = 0; ii < num_cameras_; ++ii) {
      new_track->keypoints.back()[ii].kp.setZero();
      new_track->keypoints.back()[ii].tracked = 0;
    }

    Keypoint& track_kp = new_track->keypoints.back()[cam_id];
    track_kp.kp = new_kp.center_px;
    track_kp.tracked = true;


    new_track->num_good_tracked_frames++;

    // inherit the depth of this track from the closeset track.
    bool seeded_from_closest_track = false;
    if (tracker_options_.use_closest_track_to_seed_rho) {
      std::shared_ptr<DenseTrack> closest_track = nullptr;
      double min_distance = DBL_MAX;
      for (std::shared_ptr<DenseTrack> track : current_tracks_) {
        if (!track->keypoints.back()[cam_id].tracked) {
          continue;
        }

        const double dist =
            (track->keypoints.back()[cam_id].kp - new_kp.center_px).norm();
        if (dist < min_distance && track->keypoints.size() > 2) {
          min_distance = dist;
          closest_track = track;
        }
      }

      if (closest_track != nullptr && min_distance < 100) {
        new_kp.rho = closest_track->ref_keypoint.rho;
        seeded_from_closest_track = true;
      }
    }

    if (!seeded_from_closest_track && tracker_options_.use_random_rho_seeding) {
      new_kp.rho = distribution(generator_);
    } else {
      new_kp.rho = tracker_options_.default_rho;
    }

    BackProjectTrack(new_track, true);

    num_to_start--;
    num_started++;
  }
  return num_started;
}

double SemiDenseTracker::EvaluateTrackResiduals(
    uint32_t level,
    const std::vector<std::vector<cv::Mat>>& image_pyrmaid,
    std::list<std::shared_ptr<DenseTrack>>& tracks,
    bool transfer_jacobians,
    bool optimized_tracks_only) {

  double residual = 0;
  uint32_t residual_count = 0;
  for (uint32_t cam_id = 0 ; cam_id < num_cameras_ ; ++cam_id) {
    const Sophus::SE3d t_cv = camera_rig_->cameras_[cam_id]->Pose().inverse();
    for (std::shared_ptr<DenseTrack>& track : tracks) {
      const Sophus::SE3d& t_vc = camera_rig_->cameras_[track->ref_cam_id]->Pose();
      if (optimized_tracks_only && !track->residual_used) {
        continue;
      }

      PatchTransfer& transfer = track->transfer[cam_id];
      const Sophus::SE3d track_t_ba = t_cv * t_ba_ * track->t_ba * t_vc;
      DenseKeypoint& ref_kp = track->ref_keypoint;
      Patch& ref_patch = ref_kp.patch_pyramid[level];

      uint32_t num_inliers = 0;

      TransferPatch(
          track, level, cam_id, track_t_ba,  camera_rig_->cameras_[cam_id],
          transfer, transfer_jacobians);

      transfer.tracked_pixels = 0;
      transfer.rmse = 0;
      double ncc_num = 0, ncc_den_a = 0, ncc_den_b = 0;

      if (transfer.valid_projections.size() < ref_patch.rays.size() / 2) {
        continue;
      }


      for (size_t kk = 0; kk < transfer.valid_rays.size() ; ++kk) {
        const size_t ii = transfer.valid_rays[kk];
        // First transfer this pixel over to our current image.
        const double val_pix = transfer.projected_values[ii];

        const Scalar c_huber = 1.2107 * pyramid_error_thresholds_[level];
        const double mean_s_ref = ref_patch.values[ii] - ref_patch.mean;
        const double mean_s_proj = val_pix - transfer.mean_value;
        double res = mean_s_proj - mean_s_ref;
        bool inlier = true;
        if (tracker_options_.use_robust_norm_) {
          const double weight_sqrt = //sqrt(1.0 / ref_patch.statistics[ii][1]);
              sqrt(fabs(res) > c_huber ? c_huber / fabs(res) : 1.0);
          // LOG(INFO) << "Weight for " << res << " at level " << level << " is " <<
          //              weight_sqrt * weight_sqrt << std::endl;
          res *= weight_sqrt;
          if (weight_sqrt != 1) {
            inlier = false;
          }
        }
        const double res_sqr = res * res;

        if (inlier) {
          transfer.rmse += res_sqr;
          ncc_num += mean_s_ref * mean_s_proj;
          ncc_den_a += mean_s_ref * mean_s_ref;
          ncc_den_b += mean_s_proj * mean_s_proj;
          num_inliers++;
        }

        transfer.residuals[ii] = res;
        residual_count++;
        residual += res_sqr;

        transfer.tracked_pixels++;
      }

      // Compute the track RMSE and NCC scores.
      transfer.rmse = transfer.tracked_pixels == 0 ?
          1e9 : sqrt(transfer.rmse / num_inliers);
      const double denom = sqrt(ncc_den_a * ncc_den_b);
      transfer.ncc = denom == 0 ? 0 : ncc_num / denom;
    }
  }

  return sqrt(residual);
}

void SemiDenseTracker::ReprojectTrackCenters() {
  average_track_length_ = 0;
  tracks_suitable_for_cam_localization = 0;
  for (uint32_t cam_id = 0; cam_id < num_cameras_ ; ++cam_id) {
    const calibu::CameraInterface<Scalar>& cam = *camera_rig_->cameras_[cam_id];
    const Sophus::SE3d t_cv = camera_rig_->cameras_[cam_id]->Pose().inverse();

    for (std::shared_ptr<DenseTrack>& track : current_tracks_) {
      const Sophus::SE3d& t_vc = camera_rig_->cameras_[track->ref_cam_id]->Pose();
      const Sophus::SE3d track_t_ba = t_cv * t_ba_ * track->t_ba * t_vc;
      const DenseKeypoint& ref_kp = track->ref_keypoint;
      // Transfer the center ray. This is used for 2d tracking.
      const Eigen::Vector2t center_pix =
          cam.Transfer3d(track_t_ba, ref_kp.ray, ref_kp.rho) +
          track->offset_2d[cam_id];
      if (IsReprojectionValid(center_pix, image_pyramid_[cam_id][0])) {
        track->keypoints.back()[cam_id].kp = center_pix;
        mask_.SetMask(cam_id, center_pix[0], center_pix[1]);

        // Figure out which feature cell this particular reprojection falls into,
        // and increment that cell.
        const uint32_t addressx =
            (center_pix[0] / image_pyramid_[cam_id][0].cols) *
            tracker_options_.feature_cells;
        const uint32_t addressy =
            (center_pix[1] / image_pyramid_[cam_id][0].rows) *
            tracker_options_.feature_cells;
        if (addressy > feature_cells_[cam_id].rows() ||
            addressx > feature_cells_[cam_id].cols()) {
          LOG(INFO) << "Out of bounds feature cell access at : " << addressy <<
              ", " << addressx << std::endl;
        }
        if (feature_cells_[cam_id](addressy, addressx) != kUnusedCell) {
          feature_cells_[cam_id](addressy, addressx)++;
        }

        if (track->keypoints.size() > 1) {
          average_track_length_ +=
              (track->keypoints.back()[cam_id].kp -
               track->keypoints.front()[cam_id].kp).norm();
        }

        if (track->keypoints.size() >= MIN_OBS_FOR_CAM_LOCALIZATION) {
          tracks_suitable_for_cam_localization++;
        }

      } else {
        // invalidate this latest keypoint.
        track->keypoints.back()[cam_id].tracked = false;
        track->transfer[cam_id].tracked_pixels = 0;
      }
    }
  }

  if (current_tracks_.size() > 0) {
    average_track_length_ /= current_tracks_.size();
  }
}

void SemiDenseTracker::TransformTrackTabs(const Sophus::SE3d& t_cb) {
  // Multiply the t_ba of all tracks by the current delta.
  for (std::shared_ptr<DenseTrack>& track : current_tracks_) {
    track->t_ba = t_cb * track->t_ba;
  }
}

void SemiDenseTracker::OptimizeTracks(uint32_t level, bool optimize_landmarks,
                                      bool optimize_pose, bool trust_guess)
{
  OptimizationOptions options;
  options.optimize_landmarks = optimize_landmarks;
  options.optimize_pose = optimize_pose;
  options.trust_guess = trust_guess;
  OptimizeTracks(options, level);
}


void SemiDenseTracker::OptimizeTracks(const OptimizationOptions &options,
                                      uint32_t level) {
  const double time = Tic();
  OptimizationStats stats;
  PyramidLevelOptimizationOptions level_options;
  bool roll_back = false;
  Sophus::SE3d t_ba_old_;
  int last_level = level;
  // Level -1 means that we will optimize the entire pyramid.
  if (level == static_cast<uint32_t>(-1)) {
    bool optimized_pose = false;
    double time = Tic();
    for (int ii = tracker_options_.pyramid_levels - 1 ; ii >= 0 ; ii--) {
      last_level = ii;

      if (options.trust_guess) {
        level_options.optimize_landmarks = true;
        level_options.optimize_pose = false;
      } else {
        if (average_track_length_ > 10 &&
            tracks_suitable_for_cam_localization >
            tracker_options_.num_active_tracks) {
          if (optimized_pose == false) {
            level_options.optimize_landmarks = false;
            level_options.optimize_pose = true;
            if (last_level < 3) {
              ii = tracker_options_.pyramid_levels;
              optimized_pose = true;
            }
          } else {
            level_options.optimize_landmarks = true;
            level_options.optimize_pose = !level_options.optimize_landmarks;
          }
        } else {
          level_options.optimize_landmarks = true;
          level_options.optimize_pose = true;
        }
      }

      LOG(INFO)
          << "Auto optim. level " << last_level << " with pose : " <<
             level_options.optimize_pose << " and lm : " <<
             level_options.optimize_landmarks << " with av track " <<
             average_track_length_ << std::endl;

      uint32_t iterations = 0;
      // Continuously iterate this pyramid level until we meet a stop
      // condition.
      bool pose_exit;
      bool landmark_exit;

      do {
        t_ba_old_ = t_ba_;
        OptimizePyramidLevel(last_level, image_pyramid_, current_tracks_,
                             level_options, stats);

        double post_error = EvaluateTrackResiduals(
            last_level, image_pyramid_, current_tracks_, false, true);

        // Exit if the error increased. (This also forces a roll-back).
        if (post_error > stats.pre_solve_error) {
          roll_back = true;
          break;
        }
        const double change = post_error == 0 ? 0 :
            fabs(stats.pre_solve_error - post_error) / post_error;

        // Exit if the change in the params was less than a threshold.
        if (change < 0.01) {
          break;
        }

        post_error = stats.pre_solve_error;
        iterations++;

        pose_exit = false;
        landmark_exit = false;
        if (stats.delta_pose_norm <= 1e-4 && options.optimize_pose) {
          pose_exit = true;
        }

        if (stats.delta_lm_norm <= 1e-4 * current_tracks_.size() &&
            options.optimize_landmarks) {
          landmark_exit = true;
        }

      } while (!pose_exit && !landmark_exit);
    }

    std::cerr << "Pyramid optimization took " << Toc(time) << " seconds." << std::endl;

    time = Tic();
    // Do final 2d alignment of tracks.
    AlignmentOptions alignment_options;
    alignment_options.apply_to_kp = false;
    alignment_options.only_optimize_camera_id = options.only_optimize_camera_id;
    Do2dAlignment(alignment_options, GetImagePyramid(), GetCurrentTracks(), 0);

    std::cerr << "2D alignment optimization took " << Toc(time) << " seconds." << std::endl;
  } else {
    // The user has specified the pyramid level they want optimized.
    level_options.optimize_landmarks = options.optimize_landmarks;
    level_options.optimize_pose = options.optimize_pose;
    level_options.only_optimize_camera_id = options.only_optimize_camera_id;
    t_ba_old_ = t_ba_;
    OptimizePyramidLevel(level, image_pyramid_, current_tracks_,
                         level_options, stats);
    ///zzzzzz evaluate residuals at all levels so we can see
    for (uint32_t ii = 0 ; ii < tracker_options_.pyramid_levels ; ++ii)  {
      if (ii != level) {
        LOG(INFO) << "post rmse: " << ii << " " <<
                                EvaluateTrackResiduals(ii, image_pyramid_,
                                                       current_tracks_, false,
                                                       true) <<
                                std::endl;
      }
    }
    double post_error = EvaluateTrackResiduals(
        level, image_pyramid_, current_tracks_, false, true);
    LOG(INFO) << "post rmse: " << post_error << " " << "pre : " <<
        stats.pre_solve_error << std::endl;
    if (post_error > stats.pre_solve_error) {
      LOG(INFO) << "Exiting due to " << post_error << " > " <<
          stats.pre_solve_error << " rolling back. " <<
          std::endl;

      roll_back = true;
    }
  }

  // If a roll back is required, undo the changes.
  if (roll_back) {
    // Roll back the changes.
    t_ba_ = t_ba_old_;
    for (std::shared_ptr<DenseTrack> track : current_tracks_) {
      // Only roll back tracks that were in the optimization.
      if (track->opt_id == UINT_MAX) {
        continue;
      }
      track->ref_keypoint.rho = track->ref_keypoint.old_rho;
    }
    EvaluateTrackResiduals(
        last_level, image_pyramid_, current_tracks_, false, true);
  }

  // Reproject patch centers. This will update the keypoints vector in each
  // patch which is used to pass on center values to an outside 2d BA.
  ReprojectTrackCenters();

  // Print pre-post errors
  LOG(INFO) << "Level " << level << " solve took " << Toc(time) <<
                          "s" << " with delta_p_norm: " <<
                          stats.delta_pose_norm << " and delta lm norm: " <<
                          stats.delta_lm_norm << std::endl;
}

void SemiDenseTracker::PruneTracks(int only_prune_camera) {
  if (image_pyramid_.size() == 0) {
    return;
  }

  num_successful_tracks_ = 0;
  // OptimizeTracks();

  std::list<std::shared_ptr<DenseTrack>>::iterator iter =
      current_tracks_.begin();
  while (iter != current_tracks_.end()) {
    std::shared_ptr<DenseTrack> track = *iter;

    uint32_t num_successful_cams = 0;
    for (uint32_t cam_id = 0; cam_id < num_cameras_ ; ++cam_id) {
      if (only_prune_camera != -1 && (int)track->ref_cam_id != only_prune_camera) {
        continue;
      }

      PatchTransfer& transfer = track->transfer[cam_id];
      const double dim_ratio = transfer.dimension / tracker_options_.patch_dim;
      const double percent_tracked =
          ((double)transfer.tracked_pixels /
           (double)transfer.pixels_attempted);

      if (transfer.ncc > tracker_options_.dense_ncc_threshold &&
          percent_tracked == 1.0 && !(track->transfer[cam_id].level == 0 &&
                                      (dim_ratio > 2.0 || dim_ratio < 0.5))) {
        track->keypoints.back()[cam_id].tracked = true;
        num_successful_cams++;
      } else {
        track->keypoints.back()[cam_id].tracked = false;
      }
    }

    if (num_successful_cams == 0) {
      track->tracked = false;
      iter = current_tracks_.erase(iter);
    } else {
      track->num_good_tracked_frames++;
      track->tracked = true;
      num_successful_tracks_++;
      ++iter;
    }
  }
}

void SemiDenseTracker::PruneOutliers() {
  num_successful_tracks_ = 0;
  std::list<std::shared_ptr<DenseTrack>>::iterator iter =
      current_tracks_.begin();
  while (iter != current_tracks_.end()) {
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
                                     uint32_t cam_id,
                                     const Sophus::SE3d& t_ba,
                                     std::shared_ptr<calibu::CameraInterface<Scalar>> cam,
                                     PatchTransfer& result,
                                     bool transfer_jacobians,
                                     bool use_approximation) {
  result.level = level;
  Eigen::Vector4t ray;
  DenseKeypoint& ref_kp = track->ref_keypoint;
  Patch& ref_patch = ref_kp.patch_pyramid[level];
  result.valid_projections.clear();
  result.valid_rays.clear();
  result.dprojections.clear();
  result.valid_projections.reserve(ref_patch.rays.size());
  result.valid_rays.reserve(ref_patch.rays.size());
  result.dprojections.reserve(ref_patch.rays.size());

  result.projected_values.resize(ref_patch.rays.size());
  result.residuals.resize(ref_patch.rays.size());
  result.pixels_attempted = 0;

  result.mean_value = 0;

  if (track->needs_backprojection) {
    BackProjectTrack(track);
    track->needs_backprojection = false;
  }

  // If we are doing simplified (4 corner) patch transfer, transfer the four
  // corners.
  bool corners_project = true;
  Eigen::Vector2t corner_projections[4];
  Eigen::Matrix<double, 2, 4> corner_dprojections[4];
  for (int ii = 0 ; ii < 4 ; ++ii) {
    const Eigen::Vector3t& corner_ray =
        ref_patch.rays[pyramid_patch_corner_dims_[level][ii]];
    corner_projections[ii] = cam->Transfer3d(t_ba, corner_ray, ref_kp.rho) +
        track->offset_2d[cam_id];

    if (transfer_jacobians) {
      ray.head<3>() = corner_ray;
      ray[3] = ref_kp.rho;
      const Eigen::Vector4t ray_b = MultHomogeneous(t_ba, ray);
      corner_dprojections[ii] =
          cam->dTransfer3d_dray(Sophus::SE3d(), ray_b.head<3>(), ray_b[3]);
    }
  }

  if (!corners_project) {
    return;
  }

  // Get the new dimension in pixels.
  result.dimension = (corner_projections[1] - corner_projections[0]).norm();


  // Transfer the center ray.
  const Eigen::Vector3t center_ray = ref_kp.ray;
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
    Eigen::Vector2t pix;
    if (corners_project) {
      if (!use_approximation) {
        pix = cam->Transfer3d(t_ba, ref_patch.rays[ii],
                              ref_kp.rho) + track->offset_2d[cam_id];
      } else {
        pix =
            tl_factor * corner_projections[0] +
            tr_factor * corner_projections[1] +
            bl_factor * corner_projections[2] +
            br_factor * corner_projections[3];
      }
    }

    result.pixels_attempted++;
    // Check bounds
    result.projections[ii] = pix;
    pix[0] *= pyramid_coord_ratio_[level][0];
    pix[1] *= pyramid_coord_ratio_[level][1];

    if (!IsReprojectionValid(pix, image_pyramid_[cam_id][level])) {
      result.projected_values[ii] = 0;
      result.residuals[ii] = 0;
      continue;
    } else {
      if (corners_project && transfer_jacobians) {
        ray.head<3>() = ref_patch.rays[ii];
        ray[3] = ref_kp.rho;
        const Eigen::Vector4t ray_b = MultHomogeneous(t_ba, ray);
        if (!use_approximation) {
          result.dprojections.push_back(
              cam->dTransfer3d_dray(
                  Sophus::SE3d(), ray_b.head<3>(), ray_b[3]));
        } else {
          result.dprojections.push_back(
              tl_factor * corner_dprojections[0] +
              tr_factor * corner_dprojections[1] +
              bl_factor * corner_dprojections[2] +
              br_factor * corner_dprojections[3]
                                        );
        }
      }

      result.valid_projections.push_back(pix);
      result.valid_rays.push_back(ii);
      if (std::isnan(pix[0]) || std::isnan(pix[1])) {
        LOG(INFO) << "Pixel at: " << pix.transpose() <<
            " center ray: " << ref_kp.ray << " rho: " <<
            ref_kp.rho << std::endl;
      }
      const double val = GetSubPix(image_pyramid_[cam_id][level],
                                   pix[0], pix[1]);
      result.projected_values[ii] = val;
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

void SemiDenseTracker::StartNewLandmarks(int only_start_in_camera) {
  // Afterwards, we must spawn a number of new tracks
  const int num_new_tracks =
      std::max(0, (int)tracker_options_.num_active_tracks -
               (int)num_successful_tracks_);

  // Clear the new tracks array, which will be populated below.
  new_tracks_.clear();

  // Start tracks in every camera.
  for (uint32_t cam_id = 0; cam_id < num_cameras_; ++cam_id) {
    if (only_start_in_camera != -1 && (int)cam_id != only_start_in_camera) {
      continue;
    }

    std::vector<cv::KeyPoint> cv_keypoints;
    // Extract features and descriptors from this image
    ExtractKeypoints(image_pyramid_[cam_id][0], cv_keypoints, cam_id);

    const uint32_t started =
        StartNewTracks(image_pyramid_[cam_id], cv_keypoints, num_new_tracks,
                       cam_id);

    LOG(INFO) << "Tracked: " << num_successful_tracks_ << " started " <<
        started << " out of " << num_new_tracks <<
        " new tracks with " << cv_keypoints.size() <<
        " keypoints in cam " << cam_id <<  std::endl;
  }
}

void SemiDenseTracker::GetImageDerivative(
    const cv::Mat& image, const Eigen::Vector2d& pix,
    Eigen::Matrix<double, 1, 2>& di_dppix, double val_pix) {
  double eps = 1e-9;
  const double valx_pix = GetSubPix(image, pix[0] + eps, pix[1]);
  const double valy_pix = GetSubPix(image, pix[0], pix[1] + eps);
  di_dppix[0] = (valx_pix - val_pix) / (eps);
  di_dppix[1] = (valy_pix - val_pix) / (eps);
}

void SemiDenseTracker::Do2dTracking(
    std::list<std::shared_ptr<DenseTrack>>& tracks) {
  AlignmentOptions alignment_options;
  alignment_options.apply_to_kp = false;
  for (int level = tracker_options_.pyramid_levels - 1 ; level >= 0 ; --level) {
    Do2dAlignment(alignment_options, GetImagePyramid(), GetCurrentTracks(),
                  level);
  }
  ReprojectTrackCenters();
}

void SemiDenseTracker::Do2dAlignment(
    const AlignmentOptions &options,
    const std::vector<std::vector<cv::Mat>>& image_pyrmaid,
    std::list<std::shared_ptr<DenseTrack>>& tracks,
    uint32_t level) {
  for (uint32_t cam_id = 0; cam_id < num_cameras_; ++cam_id) {
    const Sophus::SE3d t_cv = camera_rig_->cameras_[cam_id]->Pose().inverse();

    std::vector<std::shared_ptr<DenseTrack>> track_vec{
      std::begin(tracks), std::end(tracks)};
    Parallel2dAlignment alignment(*this, options, image_pyrmaid,
                                  track_vec, t_cv, level, cam_id);

    tbb::parallel_for(tbb::blocked_range<int>(0, track_vec.size()),
                      alignment);
  }
}

void SemiDenseTracker::OptimizePyramidLevel(
    uint32_t level,
    const std::vector<std::vector<cv::Mat>>& image_pyrmaid,
    std::list<std::shared_ptr<DenseTrack>>& tracks,
    const PyramidLevelOptimizationOptions& options,
    OptimizationStats& stats) {

  static Eigen::Matrix<double, 6, 6> u;
  static Eigen::Matrix<double, 6, 1> r_p;


  std::vector<std::shared_ptr<DenseTrack>> track_vec{
    std::begin(tracks), std::end(tracks)};

  OptimizeTrack optimizer(*this, options, track_vec, stats, level,
                          image_pyrmaid, g_sdtrack_debug);

  tbb::parallel_reduce(tbb::blocked_range<int>(0, track_vec.size()),
                       optimizer);

  u = optimizer.u;
  r_p = optimizer.r_p;

  // Solve for the pose update
  const double solve_time = Tic();
  Eigen::Matrix<double, 6, 1> delta_p;
  if (options.optimize_pose) {
    Eigen::LDLT<Eigen::Matrix<Scalar, 6, 6>> solver;
    solver.compute(u);
    delta_p = solver.solve(r_p);
    t_ba_ = Sophus::SE3d::exp(-delta_p * tracker_options_.gn_scaling) * t_ba_;
  }
  stats.solve_time += Toc(solve_time);

  const double lm_time = Tic();
  stats.delta_lm_norm = 0;
  uint32_t delta_lm_count = 0;

  // Now back-substitute all the tracks.
  if (options.optimize_landmarks) {
    for (std::shared_ptr<DenseTrack>& track : tracks) {
      if (track->opt_id != UINT_MAX) {
        delta_lm_count++;
        double delta_ray;
        if (options.optimize_pose) {
          delta_ray = track->v_inv_vec /*v_inv_vec[track->opt_id]*/ *
              (track->r_l_vec /*r_l_vec(track->residual_offset)*/ -
               track->w_vec.transpose() /*w_vec[track->opt_id].transpose()*/ * delta_p);
        } else {
          delta_ray = track->v_inv_vec /*v_inv_vec[track->opt_id]*/ *
              (track->r_l_vec /*r_l_vec(track->residual_offset)*/);
        }

        delta_ray *= tracker_options_.gn_scaling;
        track->ref_keypoint.old_rho = track->ref_keypoint.rho;
        track->ref_keypoint.rho -= delta_ray;
        stats.delta_lm_norm += fabs(delta_ray);

        if (std::isnan(delta_ray) || std::isinf(delta_ray)) {
          LOG(INFO) << "delta_ray " << track->id << ": " << delta_ray <<
              "vinv:" << track->v_inv_vec/*v_inv_vec[track->opt_id]*/ << " r_l " <<
              track->r_l_vec /*r_l_vec(track->residual_offset)*/ << " w: " <<
              track->w_vec.transpose() /*w_vec[track->opt_id].transpose()*/ << "dp : " <<
              delta_p.transpose() << std::endl;
        }

        if (track->ref_keypoint.rho < 0) {
          track->ref_keypoint.rho = track->ref_keypoint.old_rho / 2;
        }
      }
    }
  }
  stats.lm_time += Toc(lm_time);

  // set the optimization stats.
  stats.pre_solve_error = sqrt(optimizer.residual);
  stats.delta_pose_norm = delta_p.norm();
  stats.delta_lm_norm =
      delta_lm_count == 0 ? 0 : stats.delta_lm_norm / delta_lm_count;
}


double SemiDenseTracker::GetSubPix(const cv::Mat& image, double x, double y) {
  return Interpolate(x, y, image.data, image.cols, image.rows);
}

void SemiDenseTracker::AddImage(const std::vector<cv::Mat>& images,
                                const Sophus::SE3d& t_ba_guess) {
  // If there were any outliers (externally marked), now is the time to prune
  // them.
  PruneOutliers();

  // Clear out all 2d offsets for new tracks
  for (uint32_t cam_id = 0; cam_id < num_cameras_ ; ++cam_id) {
    active_feature_cells_[cam_id] = 0;
    // Clear out the feature cells that are not marked as unused.
    for (int row = 0; row < feature_cells_[cam_id].rows(); ++row) {
      for (int col = 0; col < feature_cells_[cam_id].cols(); ++col) {
        if (feature_cells_[cam_id](row, col) != kUnusedCell) {
          feature_cells_[cam_id](row, col) = 0;
          active_feature_cells_[cam_id]++;
        }
      }
    }

    for (std::shared_ptr<DenseTrack>& track : current_tracks_) {
      track->offset_2d[cam_id].setZero();
      track->transfer[cam_id].level = UNINITIALIZED_TRANSFER;
      track->transfer[cam_id].ncc = 0;
    }
  }

  mask_.Clear();
  t_ba_ = t_ba_guess;
  // Create the image pyramid for the incoming image
  for (uint32_t cam_id = 0 ; cam_id < num_cameras_ ; ++cam_id) {
    image_pyramid_[cam_id].resize(tracker_options_.pyramid_levels);
    image_pyramid_[cam_id][0] = images[cam_id];
    for (uint32_t ii = 1 ; ii < tracker_options_.pyramid_levels ; ++ii) {
      cv::resize(image_pyramid_[cam_id][ii - 1],
                 image_pyramid_[cam_id][ii], cv::Size(0, 0), 0.5, 0.5);
    }
  }

  for (uint32_t ii = 0 ; ii < tracker_options_.pyramid_levels ; ++ii) {
    pyramid_coord_ratio_[ii][0] =
        (double)(image_pyramid_[0][ii].cols) /
        (double)(image_pyramid_[0][0].cols);
    pyramid_coord_ratio_[ii][1] =
        (double)(image_pyramid_[0][ii].rows) /
        (double)(image_pyramid_[0][0].rows);
  }

  if (last_image_was_keyframe_) {
    uint32_t max_length = 0;
    for (std::shared_ptr<DenseTrack>& track : current_tracks_) {
      track->keypoints.emplace_back(num_cameras_);

      if (track->keypoints.size() > max_length) {
        longest_track_id_ = track->id;
        max_length = track->keypoints.size();
      }
    }
  }

  last_image_was_keyframe_ = false;
}
