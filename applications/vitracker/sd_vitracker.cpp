// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.
#undef NDEBUG
#include <assert.h>
#include <Eigen/Eigen>
#include <miniglog/logging.h>
#include "GetPot"
#include <unistd.h>

#include "etc_common.h"
#include <HAL/Camera/CameraDevice.h>
#include <calibu/utils/Xml.h>
#include <sdtrack/TicToc.h>
#include <HAL/IMU/IMUDevice.h>
#include <HAL/Messages/Matrix.h>
#include <SceneGraph/SceneGraph.h>
#include <pangolin/pangolin.h>
#include <ba/BundleAdjuster.h>
#include <ba/InterpolationBuffer.h>
#include <sdtrack/utils.h>
#include "math_types.h"
#include "gui_common.h"
#include "CVars/CVar.h"
#include "chi2inv.h"
#include "vitrack-cvars.h"
#include <thread>
#ifdef CHECK_NANS
#include <xmmintrin.h>
#endif

#define POSES_TO_INIT 10
#include <sdtrack/semi_dense_tracker.h>



uint32_t keyframe_tracks = UINT_MAX;
double start_time = 0;
uint32_t frame_count = 0;
Sophus::SE3d last_t_ba, prev_delta_t_ba, prev_t_ba;

const int window_width = 1024;
const int window_height = 764;
std::string g_usage = "SD VITRACKER. Example usage:\n"
    "-cam file:[loop=1]///Path/To/Dataset/[left,right]*pgm "
    "-imu join:///path/to/imu -cmod cameras.xml";
bool is_keyframe = true, is_prev_keyframe = true;
bool include_new_landmarks = true;
bool optimize_landmarks = true;
bool optimize_pose = true;
bool is_running = false;
bool follow_camera = false;
bool is_stepping = false;
bool has_gps = false;
bool is_manual_mode = false;
bool do_bundle_adjustment = true;
bool do_start_new_landmarks = true;
bool use_system_time = false;
int image_width;
int image_height;

calibu::Rig<Scalar> old_rig;
calibu::Rig<Scalar> rig;
hal::Camera camera_device;
hal::IMU imu_device;

sdtrack::SemiDenseTracker tracker;
TrackerGuiVars gui_vars;
std::shared_ptr<GetPot> cl;

std::list<std::shared_ptr<sdtrack::DenseTrack>>* current_tracks = nullptr;
int last_optimization_level = 0;
std::shared_ptr<hal::Image> camera_img;
std::vector<std::vector<std::shared_ptr<SceneGraph::ImageView>>> patches;
std::vector<std::shared_ptr<sdtrack::TrackerPose>> poses;
std::vector<std::unique_ptr<SceneGraph::GLAxis> > axes;
SceneGraph::AxisAlignedBoundingBox aabb;

// Gps structures
std::vector<std::shared_ptr<sdtrack::TrackerPose>> gps_poses;
std::shared_ptr<std::thread> gps_thread;


// Inertial stuff.
std::mutex aac_mutex;
std::shared_ptr<std::thread> aac_thread;
ba::BundleAdjuster<double, 1, 6, 0> bundle_adjuster;
ba::BundleAdjuster<double, 1, 15, 0> vi_bundle_adjuster;
ba::BundleAdjuster<double, 1, 15, 0> aac_bundle_adjuster;
ba::InterpolationBufferT<ba::ImuMeasurementT<Scalar>, Scalar> imu_buffer;
std::vector<uint32_t> ba_imu_residual_ids, aac_imu_residual_ids;
int orig_num_aac_poses = num_aac_poses;
double prev_cond_error;
int imu_cond_start_pose_id = -1;
int imu_cond_residual_id = -1;

// Plotters.
std::vector<Eigen::VectorXd> plot_data;
std::vector<pangolin::DataLog> plot_logs;
std::vector<pangolin::Plotter*> plot_views;

// State variables
std::vector<cv::KeyPoint> keypoints;

void ImuCallback(const hal::ImuMsg& ref) {
  const double timestamp = use_system_time ? ref.system_time() :
                                             ref.device_time();
  Eigen::VectorXd a, w;
  hal::ReadVector(ref.accel(), &a);
  hal::ReadVector(ref.gyro(), &w);
  // std::cerr << "Added accel: " << a.transpose() << " and gyro " <<
  //             w.transpose() << " at time " << timestamp << std::endl;
  imu_buffer.AddElement(ba::ImuMeasurementT<Scalar>(w, a, timestamp));
}

template <typename BaType>
void DoBundleAdjustment(BaType& ba, bool use_imu, uint32_t& num_active_poses,
                        bool initialize_lm, bool do_adaptive_conditioning,
                        uint32_t id, std::vector<uint32_t>& imu_residual_ids)
{
  if (initialize_lm) {
    use_imu = false;
  }

  if (reset_outliers) {
    for (std::shared_ptr<sdtrack::TrackerPose> pose : poses) {
      for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
        track->is_outlier = false;
      }
    }
    reset_outliers = false;
  }

  bundle_adjuster.debug_level_threshold = ba_debug_level;
  vi_bundle_adjuster.debug_level_threshold = vi_ba_debug_level;
  aac_bundle_adjuster.debug_level_threshold = aac_ba_debug_level;

  imu_residual_ids.clear();
  ba::Options<double> options;
  options.gyro_sigma = gyro_sigma;
  options.accel_sigma = accel_sigma;
  options.accel_bias_sigma = accel_bias_sigma;
  options.gyro_bias_sigma = gyro_bias_sigma;
  options.use_dogleg = use_dogleg;
  options.use_sparse_solver = true;
  options.param_change_threshold = 1e-10;
  options.error_change_threshold = 1e-3;
  options.use_robust_norm_for_proj_residuals =
      use_robust_norm_for_proj && !initialize_lm;
  options.projection_outlier_threshold = outlier_threshold;
  options.regularize_biases_in_batch = poses.size() < POSES_TO_INIT ||
      regularize_biases_in_batch;
  options.calculate_inertial_covariance_once = calculate_covariance_once;
  uint32_t num_outliers = 0;
  Sophus::SE3d t_ba;
  // Find the earliest pose touched by the current tracks.
  uint32_t start_active_pose, start_pose_id;

  uint32_t end_pose_id;
  {
    std::lock_guard<std::mutex> lock(aac_mutex);
    end_pose_id = poses.size() - 1;

    GetBaPoseRange(poses, num_active_poses, start_pose_id, start_active_pose);

    if (start_pose_id == end_pose_id) {
      return;
    }

    // Add an extra pose to conditon the IMU
    if (use_imu && use_imu_measurements && start_active_pose == start_pose_id &&
        start_pose_id != 0) {
      start_pose_id--;
      std::cerr << "expanding sp from " << start_pose_id - 1 << " to " << start_pose_id << std::endl;
    }
  }

  bool all_poses_active = start_active_pose == start_pose_id;


  // Do a bundle adjustment on the current set
  if (current_tracks && end_pose_id) {

    if (!do_adaptive_conditioning) {
      gui_vars.timer.Tic("ba_pre");
    }

    {
      std::lock_guard<std::mutex> lock(aac_mutex);
      if (use_imu) {
        ba.SetGravity(gravity_vector);
      }
      ba.Init(options, end_pose_id + 1,
              current_tracks->size() * (end_pose_id + 1));
      for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
        ba.AddCamera(rig.cameras_[cam_id]);
      }

      // First add all the poses and landmarks to ba.
      for (uint32_t ii = start_pose_id ; ii <= end_pose_id ; ++ii) {
        std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
        const bool is_active = ii >= start_active_pose && !initialize_lm;
        pose->opt_id[id] = ba.AddPose(
              pose->t_wp, Eigen::VectorXt(), pose->v_w, pose->b,
              is_active, pose->time);
        if (ii == start_active_pose && use_imu && all_poses_active) {
          ba.RegularizePose(pose->opt_id[id], true, true, false, false);
        }

        if (use_imu && ii >= start_active_pose && ii > 0) {
          std::vector<ba::ImuMeasurementT<Scalar>> meas =
              imu_buffer.GetRange(poses[ii - 1]->time, pose->time);
          /*std::cerr << "Adding imu residual between poses " << ii - 1 << " with "
                     " time " << poses[ii - 1]->time <<  " and " << ii <<
                     " with time " << pose->time << " with " << meas.size() <<
                     " measurements" << std::endl;
                     */
          imu_residual_ids.push_back(
                ba.AddImuResidual(poses[ii - 1]->opt_id[id],
                pose->opt_id[id], meas));
          // Store the conditioning edge of the IMU.
          if (do_adaptive_conditioning) {
            if (imu_cond_start_pose_id == -1 &&
                !ba.GetPose(poses[ii - 1]->opt_id[id]).is_active &&
                ba.GetPose(pose->opt_id[id]).is_active) {
              // std::cerr << "Setting cond pose id to " << ii - 1 << std::endl;
              imu_cond_start_pose_id = ii - 1;
              imu_cond_residual_id = imu_residual_ids.back();
              // std::cerr << "Setting cond residual id to " <<
              //              imu_cond_residual_id << std::endl;
            } else if (imu_cond_start_pose_id == ii - 1) {
              imu_cond_residual_id = imu_residual_ids.back();
              // std::cerr << "Setting cond residual id to " <<
              //              imu_cond_residual_id << std::endl;
            }
          }
        }

        if (!use_only_imu) {
          for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
            const bool constrains_active =
                track->keypoints.size() + ii >= start_active_pose;
            if (track->num_good_tracked_frames <= 1 || track->is_outlier ||
                !constrains_active) {
              /*
            std::cerr << "ignoring track " << track->id << " with " <<
                         track->keypoints.size() << "keypoints with ngf " <<
                         track->num_good_tracked_frames << " outlier: " <<
                         track->is_outlier << " constraints " << constrains_active <<
                         std::endl;
                         */
              track->external_id[id] = UINT_MAX;
              continue;
            }

            Eigen::Vector4d ray;
            ray.head<3>() = track->ref_keypoint.ray;
            ray[3] = track->ref_keypoint.rho;
            ray = sdtrack::MultHomogeneous(
                  pose->t_wp * rig.cameras_[track->ref_cam_id]->Pose(), ray);
            bool active = track->id != tracker.longest_track_id() ||
                !all_poses_active || use_imu || initialize_lm;
            if (!active) {
              std::cerr << "Landmark " << track->id << " inactive. outlier = " <<
                           track->is_outlier << " length: " <<
                           track->keypoints.size() << std::endl;
            }
            track->external_id[id] =
                ba.AddLandmark(ray, pose->opt_id[id], track->ref_cam_id, active);
          }
        }
      }

      if (!use_only_imu) {
        // Now add all reprojections to ba)
        for (uint32_t ii = start_pose_id ; ii <= end_pose_id ; ++ii) {
          std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
          for (std::shared_ptr<sdtrack::DenseTrack> track : pose->tracks) {
            if (track->external_id[id] == UINT_MAX) {
              continue;
            }
            for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
              for (size_t jj = 0; jj < track->keypoints.size() ; ++jj) {
                if (track->keypoints[jj][cam_id].tracked) {
                  const Eigen::Vector2d& z = track->keypoints[jj][cam_id].kp;
                  if (ba.GetNumPoses() > (pose->opt_id[id] + jj)) {
                    const uint32_t res_id =
                        ba.AddProjectionResidual(
                          z, pose->opt_id[id] + jj,
                          track->external_id[id], cam_id, 2.0);
                  }
                }
              }
            }
          }
        }
      }
    }

    if (!do_adaptive_conditioning) {
      gui_vars.timer.Toc("ba_pre");
    }

    // Optimize the poses
    if (!do_adaptive_conditioning) {
      gui_vars.timer.Tic("ba_solve");
    }

    ba.Solve(num_ba_iterations);

    if (!do_adaptive_conditioning) {
      gui_vars.timer.Toc("ba_solve");
    }

    if (!do_adaptive_conditioning) {
      gui_vars.timer.Tic("ba_post");
    }

    {
      std::lock_guard<std::mutex> lock(aac_mutex);

      uint32_t last_pose_id =
          is_keyframe ? poses.size() - 1 : poses.size() - 2;
      std::shared_ptr<sdtrack::TrackerPose> last_pose = is_keyframe ?
            poses.back() : poses[poses.size() - 2];

      if (last_pose_id <= end_pose_id) {
        // Get the pose of the last pose. This is used to calculate the relative
        // transform from the pose to the current pose.
        last_pose->t_wp = ba.GetPose(last_pose->opt_id[id]).t_wp;
      }
      // std::cerr << "last pose t_wp: " << std::endl << last_pose->t_wp.matrix() <<
      //              std::endl;

      // Read out the pose and landmark values.
      for (uint32_t ii = start_pose_id ; ii <= end_pose_id ; ++ii) {
        std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
        const ba::PoseT<double>& ba_pose = ba.GetPose(pose->opt_id[id]);

        if (!initialize_lm) {
          pose->t_wp = ba_pose.t_wp;
          if (use_imu) {
            pose->v_w = ba_pose.v_w;
            pose->b = ba_pose.b;
          }
        }

        if (!use_only_imu) {
          // Here the last pose is actually t_wb and the current pose t_wa.
          last_t_ba = t_ba;
          t_ba = last_pose->t_wp.inverse() * pose->t_wp;
          for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
            if (track->external_id[id] == UINT_MAX) {
              continue;
            }

            if (!initialize_lm) {
              track->t_ba = t_ba;
            }

            // Get the landmark location in the world frame.
            const Eigen::Vector4d& x_w =
                ba.GetLandmark(track->external_id[id]);
            double ratio = ba.LandmarkOutlierRatio(track->external_id[id]);

            if (do_outlier_rejection && poses.size() > POSES_TO_INIT &&
                !initialize_lm /*&& (do_adaptive_conditioning || !do_async_ba)*/) {
              if (ratio > 0.3 && track->tracked == false &&
                  (end_pose_id >= min_poses_for_imu - 1 || !use_imu)) {
                /*
              std::cerr << "Rejecting landmark with outliers : ";
              for (int id: landmark.proj_residuals) {
                typename BaType::ProjectionResidual res =
                    ba.GetProjectionResidual(id);
                std::cerr << res.residual.transpose() << "(" << res.residual.norm() <<
                             "), ";
              }
              std::cerr << std::endl;
              */
                num_outliers++;
                track->is_outlier = true;
              } else {
                track->is_outlier = false;
              }
            }

            // Make the ray relative to the pose.
            Eigen::Vector4d x_r =
                sdtrack::MultHomogeneous(
                  (pose->t_wp * rig.cameras_[track->ref_cam_id]->Pose()).inverse(), x_w);
            // Normalize the xyz component of the ray to compare to the original
            // ray.
            x_r /= x_r.head<3>().norm();
            /*
          if (track->keypoints.size() >= min_lm_measurements_for_drawing) {
            std::cerr << "Setting rho for track " << track->id << " with " <<
                         track->keypoints.size() << " kps from " <<
                         track->ref_keypoint.rho << " to " << x_r[3] << std::endl;
          }
          */
            track->ref_keypoint.rho = x_r[3];
          }
        }
      }

      if (follow_camera) {
        FollowCamera(gui_vars, poses.back()->t_wp);
      }
    }

    if (!do_adaptive_conditioning) {
      gui_vars.timer.Toc("ba_post");
    }
  }
  const ba::SolutionSummary<Scalar>& summary = ba.GetSolutionSummary();
  // std::cerr << "Rejected " << num_outliers << " outliers." << std::endl;

  if (use_imu && imu_cond_start_pose_id != -1 && do_adaptive_conditioning) {
    const uint32_t cond_dims =
        summary.num_cond_inertial_residuals * BaType::kPoseDim +
        summary.num_cond_proj_residuals * 2;
    const uint32_t active_dims = summary.num_inertial_residuals +
        summary.num_proj_residuals - cond_dims;
    const Scalar cond_error = summary.cond_inertial_error +
        summary.cond_proj_error;
    const Scalar active_error =
        summary.inertial_error + summary.proj_error_ - cond_error;

    const double cond_inertial_error =
        ba.GetImuResidual(
          imu_cond_residual_id).mahalanobis_distance;

    if (prev_cond_error == -1) {
      prev_cond_error = DBL_MAX;
    }

    const Scalar cond_chi2_dist = chi2inv(adaptive_threshold, cond_dims);
    const Scalar cond_v_chi2_dist =
        chi2inv(adaptive_threshold, summary.num_cond_proj_residuals * 2);
    const Scalar cond_i_chi2_dist =
        chi2inv(adaptive_threshold, BaType::kPoseDim);
    const Scalar active_chi2_dist = chi2inv(adaptive_threshold, active_dims);
    plot_logs[0].Log(cond_i_chi2_dist, cond_inertial_error);
    plot_logs[2].Log(cond_v_chi2_dist, summary.cond_proj_error);
    // plot_logs[2].Log(cond_chi2_dist, cond_error);
    // plot_logs[2].Log(poses[start_active_pose]->v_w.norm(),
    //                  poses.back()->v_w.norm());

    /*
    std::cerr << "chi2inv(" << adaptive_threshold << ", " << cond_dims <<
                 "): " << cond_chi2_dist << " vs. " << cond_error <<
                 std::endl;

    std::cerr << "v_chi2inv(" << adaptive_threshold << ", " <<
                 summary.num_cond_proj_residuals * 2 << "): " <<
                 cond_v_chi2_dist << " vs. " <<
                 summary.cond_proj_error << std::endl;

    std::cerr << "i_chi2inv(" << adaptive_threshold << ", " <<
                 BaType::kPoseDim << "):" << cond_i_chi2_dist << " vs. " <<
                 cond_inertial_error << std::endl;

    std::cerr << "ec/Xc: " << cond_error / cond_chi2_dist << " ea/Xa: " <<
                 active_error / active_chi2_dist << std::endl;

    std::cerr << summary.num_cond_proj_residuals * 2 << " cond proj residuals "
                 " with dist: " << summary.cond_proj_error << " vs. " <<
                 summary.num_proj_residuals * 2 <<
                 " total proj residuals with dist: " <<
                 summary.proj_error_ << " and " <<
                 summary.num_cond_inertial_residuals * BaType::kPoseDim <<
                 " total cond imu residuals with dist: " <<
                 summary.cond_inertial_error <<
                 " vs. " << summary.num_inertial_residuals *
                 BaType::kPoseDim << " total imu residuals with dist : " <<
                 summary.inertial_error << std::endl;
    */
    // if (do_adaptive_conditioning) {
    if (num_active_poses > end_pose_id) {
      num_active_poses = orig_num_aac_poses;
      std::cerr << "Reached batch solution. resetting number of poses to " <<
                   num_ba_poses << std::endl;
    }

    if (cond_error == 0 || cond_dims == 0) {
      // status = OptStatus_NoChange;
    } else {
      const double cond_total_error =
          (cond_inertial_error + summary.cond_proj_error);
      const double inertial_ratio = cond_inertial_error / cond_i_chi2_dist;
      const double visual_ratio = summary.cond_proj_error / cond_v_chi2_dist;
      if ((inertial_ratio > 1.0 || visual_ratio > 1.0) &&
          (cond_total_error <= prev_cond_error) &&
          (((prev_cond_error - cond_total_error) / prev_cond_error) > 0.00001)) {
        num_active_poses += 30;//(start_active_pose - start_pose);
        // std::cerr << "INCREASING WINDOW SIZE TO " << num_active_poses <<
        //              std::endl;
      } else /*if (ratio < 0.3)*/ {
        num_active_poses = orig_num_aac_poses;
        // std::cerr << "RESETTING WINDOW SIZE TO " << num_active_poses <<
        //              std::endl;
      }
      prev_cond_error = cond_total_error;
    }
    // }
    plot_logs[1].Log(num_active_poses, poses.size());
    Eigen::VectorXd data_to_save(6);
    data_to_save << num_active_poses, poses.size(), cond_i_chi2_dist,
        cond_inertial_error, cond_v_chi2_dist, summary.cond_proj_error;

    plot_data.push_back(data_to_save);
  }
}

void UpdateCurrentPose()
{
  std::shared_ptr<sdtrack::TrackerPose> new_pose = poses.back();
  if (poses.size() > 1) {
    new_pose->t_wp = poses[poses.size() - 2]->t_wp * tracker.t_ba().inverse();
  }

  // Also use the current tracks to update the index of the earliest covisible
  // pose.
  size_t max_track_length = 0;
  for (std::shared_ptr<sdtrack::DenseTrack>& track : tracker.GetCurrentTracks()) {
    max_track_length = std::max(track->keypoints.size(), max_track_length);
  }
  new_pose->longest_track = max_track_length;
  std::cerr << "Setting longest track for pose " << poses.size() << " to " <<
               new_pose->longest_track << std::endl;
}

void DoGps()
{

}

void DoAAC()
{
  while (true) {
    if (poses.size() > 10 && do_async_ba) {
//      DoBundleAdjustment(bundle_adjuster, false, num_aac_poses, true, false,
//                          1, aac_imu_residual_ids);
      uint32_t num_poses = poses.size();
      orig_num_aac_poses = num_aac_poses;
      while (true) {
        if (poses.size() > min_poses_for_imu && use_imu_measurements) {
          DoBundleAdjustment(aac_bundle_adjuster, true, num_aac_poses,
                             false, do_adaptive, 1, aac_imu_residual_ids);
        }

        if (num_aac_poses == orig_num_aac_poses || !do_adaptive) {
          break;
        }
      }

      // std::cerr << "Resetting conditioning edge. " << std::endl;
      imu_cond_start_pose_id = -1;
      prev_cond_error = -1;
    }
    usleep(1000);
  }
}

void DoBA()
{
//  DoBundleAdjustment(bundle_adjuster, false, num_ba_poses, true, false,
//                     0, ba_imu_residual_ids);
  if (poses.size() > min_poses_for_imu && use_imu_measurements) {
    DoBundleAdjustment(vi_bundle_adjuster, true, num_ba_poses, false, false,
                       0, ba_imu_residual_ids);
  } else {
    DoBundleAdjustment(bundle_adjuster, false, num_ba_poses,
                       false, false, 0, ba_imu_residual_ids);
  }
}

void BaAndStartNewLandmarks()
{
  gui_vars.timer.Tic("ba");
  if (!is_keyframe) {
    gui_vars.timer.Tic("snl");
    gui_vars.timer.Toc("snl");

    gui_vars.timer.Tic("ba_pre");
    gui_vars.timer.Toc("ba_pre");

    gui_vars.timer.Tic("ba_post");
    gui_vars.timer.Toc("ba_post");

    gui_vars.timer.Tic("ba_solve");
    gui_vars.timer.Toc("ba_solve");

    gui_vars.timer.Toc("ba");
    return;
  }
  uint32_t keyframe_id = poses.size();

  double ba_time = sdtrack::Tic();
  if (do_bundle_adjustment) {
    DoBA();
  }
  ba_time = sdtrack::Toc(ba_time);
  gui_vars.timer.Toc("ba");

  gui_vars.timer.Tic("snl");
  if (do_start_new_landmarks) {
    tracker.StartNewLandmarks(0);
  }
  gui_vars.timer.Toc("snl");

  std::shared_ptr<sdtrack::TrackerPose> new_pose = poses.back();
  // Update the tracks on this new pose.
  new_pose->tracks = tracker.GetNewTracks();

  if (!do_bundle_adjustment) {
    tracker.TransformTrackTabs(tracker.t_ba());
  }



  std::cerr << "Timings ba: " << ba_time << std::endl;
}

void ProcessImage(std::vector<cv::Mat>& images, double timestamp)
{
  std::cerr << "Processing image with timestamp " << timestamp << std::endl;
#ifdef CHECK_NANS
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() &
                         ~(_MM_MASK_INVALID | _MM_MASK_OVERFLOW |
                           _MM_MASK_DIV_ZERO));
#endif

  if (frame_count == 0) {
    start_time = sdtrack::Tic();
  }

  frame_count++;
  //  if (poses.size() > 100) {
  //    exit(EXIT_SUCCESS);
  //  }

  Sophus::SE3d guess;
  // If this is a keyframe, set it as one on the tracker.
  prev_delta_t_ba = tracker.t_ba() * prev_t_ba.inverse();

  if (is_prev_keyframe) {
    prev_t_ba = Sophus::SE3d();
  } else {
    prev_t_ba = tracker.t_ba();
  }

  // Add a pose to the poses array
  if (is_prev_keyframe) {
    std::shared_ptr<sdtrack::TrackerPose> new_pose(new sdtrack::TrackerPose);
    if (poses.size() > 0) {
      new_pose->t_wp = poses.back()->t_wp * last_t_ba.inverse();
      new_pose->v_w = poses.back()->v_w;
      new_pose->b = poses.back()->b;
    } else {
      if (imu_buffer.elements.size() > 0) {
        Eigen::Vector3t down = -imu_buffer.elements.front().a.normalized();
        std::cerr << "Down vector based on first imu meas: " <<
                     down.transpose() << std::endl;

        // compute path transformation
        Eigen::Vector3t forward(1.0, 0.0, 0.0);
        Eigen::Vector3t right = down.cross(forward);
        right.normalize();
        forward = right.cross(down);
        forward.normalize();

        Eigen::Matrix4t base = Eigen::Matrix4t::Identity();
        base.block<1, 3>(0, 0) = forward;
        base.block<1, 3>(1, 0) = right;
        base.block<1, 3>(2, 0) = down;
        new_pose->t_wp = Sophus::SE3t(base);
      }
      // Set the initial velocity and bias. The initial pose is initialized to
      // align the gravity plane
      new_pose->v_w.setZero();
      new_pose->b.setZero();
      // corridor
      // new_pose->b << 0.00209809 , 0.00167743, -7.46213e-05 ,
      //     0.151629 ,0.0224114, 0.826392;

      // gw_block
      // new_pose->b << 0.00288919,  0.0023673, 0.00714931 ,
      //     -0.156199,   0.258919,   0.422379;

      // gw_car_block
      // new_pose->b << 0.00217338, -0.00122939,  0.00220202,
      //     -0.175229,  -0.0731785,    0.548693;

    }
    {
      std::unique_lock<std::mutex>(aac_mutex);
      poses.push_back(new_pose);
    }
    axes.push_back(std::unique_ptr<SceneGraph::GLAxis>(
                      new SceneGraph::GLAxis(0.5)));
    gui_vars.scene_graph.AddChild(axes.back().get());
  }

  // Set the timestamp of the latest pose to this image's timestamp.
  poses.back()->time = timestamp + imu_time_offset;

  guess = prev_delta_t_ba * prev_t_ba;
  if(guess.translation() == Eigen::Vector3d(0,0,0) &&
     poses.size() > 1) {
    guess.translation() = Eigen::Vector3d(0, 0, 0.001);
  }

  if (use_imu_measurements &&
      use_imu_for_guess && poses.size() >= min_poses_for_imu) {
    std::shared_ptr<sdtrack::TrackerPose> pose1 = poses[poses.size() - 2];
    std::shared_ptr<sdtrack::TrackerPose> pose2 = poses.back();
    std::vector<ba::ImuPoseT<Scalar>> imu_poses;
    ba::PoseT<Scalar> start_pose;
    start_pose.t_wp = pose1->t_wp;
    start_pose.b = pose1->b;
    start_pose.v_w = pose1->v_w;
    start_pose.time = pose1->time;
    // Integrate the measurements since the last frame.
    std::vector<ba::ImuMeasurementT<Scalar> > meas =
        imu_buffer.GetRange(pose1->time, pose2->time);
    decltype(vi_bundle_adjuster)::ImuResidual::IntegrateResidual(
          start_pose, meas, start_pose.b.head<3>(), start_pose.b.tail<3>(),
          vi_bundle_adjuster.GetImuCalibration().g_vec, imu_poses);

    if (imu_poses.size() > 1) {
      // std::cerr << "Prev guess t_ab is\n" << guess.matrix3x4() << std::endl;
      ba::ImuPoseT<Scalar>& last_pose = imu_poses.back();
      //      guess.so3() = last_pose.t_wp.so3().inverse() *
      //          imu_poses.front().t_wp.so3();
      guess = last_pose.t_wp.inverse() *
          imu_poses.front().t_wp;
      pose2->t_wp = last_pose.t_wp;
      pose2->v_w = last_pose.v_w;
      poses.back()->t_wp = pose2->t_wp;
      poses.back()->v_w = pose2->v_w;
      poses.back()->b = pose2->b;
      // std::cerr << "Imu guess t_ab is\n" << guess.matrix3x4() << std::endl;
    }
  }

  gui_vars.timer.Tic("track");
  {
    std::lock_guard<std::mutex> lock(aac_mutex);

    tracker.AddImage(images, guess);
    gui_vars.timer.Tic("evaluate");
    tracker.EvaluateTrackResiduals(0, tracker.GetImagePyramid(),
                                   tracker.GetCurrentTracks());
    gui_vars.timer.Toc("evaluate");

    if (!is_manual_mode) {
      tracker.OptimizeTracks(-1, optimize_landmarks, optimize_pose);
      tracker.PruneTracks();
    }
    // Update the pose t_ab based on the result from the tracker.
    UpdateCurrentPose();
    if (follow_camera) {
      FollowCamera(gui_vars, poses.back()->t_wp);
    }
  }
  gui_vars.timer.Toc("track");

  if (do_keyframing) {
    const double track_ratio = (double)tracker.num_successful_tracks() /
        (double)keyframe_tracks;
    const double total_trans = tracker.t_ba().translation().norm();
    const double total_rot = tracker.t_ba().so3().log().norm();

    double average_depth = 0;
    if (current_tracks == nullptr || current_tracks->size() == 0) {
      average_depth = 1;
    } else {
      for (std::shared_ptr<sdtrack::DenseTrack>& track : *current_tracks) {
          average_depth += (1.0 / track->ref_keypoint.rho);
      }
      average_depth /= current_tracks->size();
    }


    bool keyframe_condition = track_ratio < 0.7 ||
        total_trans > 0.2 || total_rot > 0.1
        /*|| tracker.num_successful_tracks() < 64*/;

    std::cerr << "\tRatio: " << track_ratio << " trans: " << total_trans <<
                 "av: depth: " << average_depth << " rot: " <<
                 total_rot << std::endl;

    {
      std::lock_guard<std::mutex> lock(aac_mutex);
      if (keyframe_tracks != 0) {
        if (keyframe_condition) {
          is_keyframe = true;
        } else {
          is_keyframe = false;
        }


        // If this is a keyframe, set it as one on the tracker.
        prev_delta_t_ba = tracker.t_ba() * prev_t_ba.inverse();

        if (is_keyframe) {
          tracker.AddKeyframe();
        }
        is_prev_keyframe = is_keyframe;
      }
    }
  } else {
    std::lock_guard<std::mutex> lock(aac_mutex);
    tracker.AddKeyframe();
  }

  std::cerr << "Num successful : " << tracker.num_successful_tracks() <<
               " keyframe tracks: " << keyframe_tracks << std::endl;

  if (!is_manual_mode) {
    BaAndStartNewLandmarks();
  }

  if (is_keyframe) {
    std::cerr << "KEYFRAME." << std::endl;
    keyframe_tracks = tracker.GetCurrentTracks().size();
    std::cerr << "New keyframe tracks: " << keyframe_tracks << std::endl;
  } else {
    std::cerr << "NOT KEYFRAME." << std::endl;
  }

  current_tracks = &tracker.GetCurrentTracks();

#ifdef CHECK_NANS
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() |
                         (_MM_MASK_INVALID | _MM_MASK_OVERFLOW |
                          _MM_MASK_DIV_ZERO));
#endif

  std::cerr << "FRAME : " << frame_count << " KEYFRAME: " << poses.size() <<
               " FPS: " << frame_count / sdtrack::Toc(start_time) << std::endl;
}

void DrawImageData(uint32_t cam_id)
{
  if (cam_id == 0) {
    gui_vars.handler->track_centers.clear();
  }

  SceneGraph::AxisAlignedBoundingBox aabb;
  for (uint32_t ii = 0; ii < poses.size() ; ++ii) {
    axes[ii]->SetPose(poses[ii]->t_wp.matrix());
    aabb.Insert(poses[ii]->t_wp.translation());
  }
  gui_vars.grid.set_bounds(aabb);

  // Draw the tracks
  for (std::shared_ptr<sdtrack::DenseTrack>& track : *current_tracks) {
    Eigen::Vector2d center;
    if (track->keypoints.back()[cam_id].tracked ||
        track->keypoints.size() <= 2) {
      DrawTrackData(track, image_width, image_height, center,
                    gui_vars.handler->selected_track == track, cam_id);
    }
    if (cam_id == 0) {
      gui_vars.handler->track_centers.push_back(
            std::pair<Eigen::Vector2d, std::shared_ptr<sdtrack::DenseTrack>>(
              center, track));
    }
  }

  // Populate the first column with the reference from the selected track.
  if (gui_vars.handler->selected_track != nullptr) {
    DrawTrackPatches(gui_vars.handler->selected_track, gui_vars.patches);
  }

  for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
    gui_vars.camera_view[cam_id]->RenderChildren();
  }
}

void Run()
{
  std::vector<pangolin::GlTexture> gl_tex;

  // pangolin::Timer timer;
  bool capture_success = false;
  std::shared_ptr<hal::ImageArray> images = hal::ImageArray::Create();
  camera_device.Capture(*images);

  while(!pangolin::ShouldQuit()) {
    gui_vars.timer.Tic();

    capture_success = false;
    const bool go = is_stepping;
    if (!is_running) {
      is_stepping = false;
    }
    // usleep(20000);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor4f(1.0f,1.0f,1.0f,1.0f);

    if (go) {
      capture_success = camera_device.Capture(*images);
    }

    if (capture_success) {
      double timestamp = use_system_time ? images->Ref().system_time() :
                                           images->Ref().device_time();


      // Wait until we have enough measurements to interpolate this frame's
      // timestamp
      const double start_time = sdtrack::Tic();
      while (imu_buffer.end_time < timestamp &&
             sdtrack::Toc(start_time) < 0.1) {
        usleep(10);
      }

      gl_tex.resize(images->Size());

      for (uint32_t cam_id = 0 ; cam_id < (unsigned int) images->Size() ; ++cam_id) {
        if (!gl_tex[cam_id].tid) {
          camera_img = images->at(cam_id);
          GLint internal_format = (camera_img->Format() == GL_LUMINANCE ?
                                     GL_LUMINANCE : GL_RGBA);
          // Only initialise now we know format.
          gl_tex[cam_id].Reinitialise(
                camera_img->Width(), camera_img->Height(), internal_format,
                false, 0, camera_img->Format(), camera_img->Type(), 0);
        }
      }

      camera_img = images->at(0);
      image_width = camera_img->Width();
      image_height = camera_img->Height();
      gui_vars.handler->image_height = image_height;
      gui_vars.handler->image_width = image_width;

      std::vector<cv::Mat> cvmat_images;
      for (int ii = 0; ii < (unsigned int) images->Size() ; ++ii) {
        cvmat_images.push_back(images->at(ii)->Mat());
      }
      ProcessImage(cvmat_images, timestamp);
    }

    if (camera_img && camera_img->data()) {
      for (uint32_t cam_id = 0 ; cam_id < rig.cameras_.size() &&
           cam_id < (unsigned int) images->Size(); ++cam_id) {
        camera_img = images->at(cam_id);
        gui_vars.camera_view[cam_id]->ActivateAndScissor();
        gl_tex[cam_id].Upload(camera_img->data(), camera_img->Format(),
                      camera_img->Type());
        gl_tex[cam_id].RenderToViewportFlipY();
        DrawImageData(cam_id);
      }
      // gui_vars.camera_view->RenderChildren();

      gui_vars.grid_view->ActivateAndScissor(gui_vars.gl_render3d);
      const ba::ImuCalibrationT<Scalar>& imu =
          vi_bundle_adjuster.GetImuCalibration();
      std::vector<ba::ImuPoseT<Scalar>> imu_poses;

      glLineWidth(1.0f);
      // Draw the inertial residual
      for (uint32_t id : ba_imu_residual_ids) {
        const ba::ImuResidualT<Scalar>& res = vi_bundle_adjuster.GetImuResidual(id);
        const ba::PoseT<Scalar>& pose = vi_bundle_adjuster.GetPose(res.pose1_id);
        std::vector<ba::ImuMeasurementT<Scalar> > meas =
            imu_buffer.GetRange(res.measurements.front().time,
                                res.measurements.back().time +
                                imu_extra_integration_time);
        res.IntegrateResidual(pose, meas, pose.b.head<3>(), pose.b.tail<3>(),
                              imu.g_vec, imu_poses);
        // std::cerr << "integrating residual with " << res.measurements.size() <<
        //              " measurements " << std::endl;
        if (pose.is_active) {
          glColor3f(1.0, 0.0, 1.0);
        } else {
          glColor3f(1.0, 0.2, 0.5);
        }

        for (size_t ii = 1 ; ii < imu_poses.size() ; ++ii) {
          ba::ImuPoseT<Scalar>& prev_imu_pose = imu_poses[ii - 1];
          ba::ImuPoseT<Scalar>& imu_pose = imu_poses[ii];
          pangolin::glDrawLine(prev_imu_pose.t_wp.translation()[0],
              prev_imu_pose.t_wp.translation()[1],
              prev_imu_pose.t_wp.translation()[2],
              imu_pose.t_wp.translation()[0],
              imu_pose.t_wp.translation()[1],
              imu_pose.t_wp.translation()[2]);
        }
      }

      if (draw_landmarks) {
        DrawLandmarks(min_lm_measurements_for_drawing, poses, rig,
                      gui_vars.handler, selected_track_id);
      }
      // gui_vars.grid_view->RenderChildren();
    }
    gui_vars.timer.Toc();
    if (go) {
      gui_vars.timer_view.Update(20, gui_vars.timer.GetNames(3),
                                 gui_vars.timer.GetTimes(3));
    }
    pangolin::FinishFrame();
  }
}

void InitTracker()
{
  patch_size = 9;
  sdtrack::KeypointOptions keypoint_options;
  keypoint_options.gftt_feature_block_size = patch_size;
  keypoint_options.max_num_features = num_features * 2;
  keypoint_options.gftt_min_distance_between_features = 3;
  keypoint_options.gftt_absolute_strength_threshold = 0.005;
  sdtrack::TrackerOptions tracker_options;
  tracker_options.pyramid_levels = pyramid_levels;
  tracker_options.detector_type = sdtrack::TrackerOptions::Detector_GFTT;
  tracker_options.num_active_tracks = num_features;
  tracker_options.use_robust_norm_ = false;
  tracker_options.robust_norm_threshold_ = 30;
  tracker_options.patch_dim = patch_size;
  tracker_options.default_rho = 1.0/5.0;
  tracker_options.feature_cells = feature_cells;
  tracker_options.iteration_exponent = 2;
  tracker_options.center_weight = tracker_center_weight;
  tracker_options.dense_ncc_threshold = ncc_threshold;
  tracker_options.harris_score_threshold = 2e6;
  tracker_options.gn_scaling = 1.0;
  tracker.Initialize(keypoint_options, tracker_options, &rig);
  /*for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
    for (int ii = 0 ; ii < 3;  ++ii) {
      for (int jj = 0 ; jj < feature_cells ; ++jj) {
        tracker.feature_cells()[cam_id](ii, jj) =
            sdtrack::SemiDenseTracker::kUnusedCell;
      }
    }
  }*/

}

bool LoadCameras()
{
  //LoadCameraAndRig(*cl, camera_device, old_rig);
  //rig.Clear();
  LoadCameraAndRig(*cl, camera_device, rig);

  // Load the imu
  std::string imu_str = cl->follow("","-imu");
  if (!imu_str.empty()) {
    try {
      imu_device = hal::IMU(imu_str);
    } catch (hal::DeviceException& e) {
      LOG(ERROR) << "Error loading imu device: " << e.what()
                 << " ... proceeding without.";
    }
    imu_device.RegisterIMUDataCallback(&ImuCallback);
  }
  // Capture an image so we have some IMU data.
  std::shared_ptr<hal::ImageArray> images = hal::ImageArray::Create();
  while (imu_buffer.elements.size() == 0) {
    camera_device.Capture(*images);
  }

  if (!use_system_time) {
    imu_time_offset = imu_buffer.elements.back().time -
        images->Ref().device_time();
    std::cerr << "Setting initial time offset to " << imu_time_offset <<
                 std:: endl;
  }

  return true;
}

void InitGui()
{
  InitTrackerGui(gui_vars, window_width, window_height, image_width,
                 image_height, rig.cameras_.size());

  pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT,
        [&]() {
    is_stepping = true;
  });

  pangolin::RegisterKeyPressCallback('r', [&]() {
    for (uint32_t ii = 0; ii < plot_views.size(); ++ii) {
      plot_views[ii]->Keyboard(*plot_views[ii], 'a', 0, 0, true);
    }
  });

  pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_CTRL + 'r',
        [&]() {
    camera_img.reset();
    is_keyframe = true;
    is_prev_keyframe = true;
    is_running = false;
    InitTracker();
    poses.clear();
    ba_imu_residual_ids.clear();
    aac_imu_residual_ids.clear();
    imu_buffer.Clear();
    gui_vars.scene_graph.Clear();
    gui_vars.scene_graph.AddChild(&gui_vars.grid);
    axes.clear();
    LoadCameras();
  });

  pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_CTRL + 's',
        [&]() {
    // write all the poses to a file.
    std::ofstream pose_file("poses.txt", std::ios_base::trunc);
    std::ofstream log_file("logs.txt", std::ios_base::trunc);
    Sophus::SE3d last_pose = poses.front()->t_wp;
    double total_dist = 0;
    int count = 0;

    for (Eigen::VectorXd& data: plot_data) {
      log_file << data.transpose().format(sdtrack::kLongCsvFmt) << std::endl;
    }

    for (auto pose : poses) {
      pose_file << pose->t_wp.translation().transpose().format(
                     sdtrack::kLongCsvFmt) << std::endl;
      total_dist += (pose->t_wp.translation() - last_pose.translation()).norm();
      last_pose = pose->t_wp;
      std::cerr << "b for pose " << count++ << " is " << pose->b.transpose() <<
                   " v is " << pose->v_w.transpose() <<  std::endl;
    }
    const double error = (poses.back()->t_wp.translation() -
                          poses.front()->t_wp.translation()).norm();
    std::cerr << "Total distance travelled: " << total_dist << " error: " <<
                 error << " percentage error: " << error / total_dist * 100 <<
                 std::endl;

    std::ofstream lm_file("landmarks.txt", std::ios_base::trunc);
    for (std::shared_ptr<sdtrack::TrackerPose> pose : poses) {
      for (std::shared_ptr<sdtrack::DenseTrack> track : pose->tracks) {
        if (track->num_good_tracked_frames < min_lm_measurements_for_drawing) {
          continue;
        }
        Eigen::Vector4d ray;
        ray.head<3>() = track->ref_keypoint.ray;
        ray[3] = track->ref_keypoint.rho;
        ray = sdtrack::MultHomogeneous(pose->t_wp * rig.cameras_[0]->Pose(), ray);
        ray /= ray[3];
        lm_file << ray.transpose().format(sdtrack::kLongCsvFmt) << ", " <<
                   track->keypoints.size() << " , " <<
                   track->is_outlier << std::endl;
      }
    }

    for (uint32_t ii = 0; ii < plot_logs.size() ; ++ii) {
      pangolin::DataLog& log = plot_logs[ii];
      char filename[100];
      sprintf(filename, "log_%d.txt" , ii);
      // log.Save(filename);
    }
  });

  pangolin::RegisterKeyPressCallback(' ', [&]() {
    is_running = !is_running;
  });

  pangolin::RegisterKeyPressCallback('f', [&]() {
    follow_camera = !follow_camera;
  });

  pangolin::RegisterKeyPressCallback('b', [&]() {
    // last_optimization_level = 0;
    // tracker.OptimizeTracks();
    DoBA();
  });

  pangolin::RegisterKeyPressCallback('k', [&]() {
    sdtrack::AlignmentOptions options;
    options.apply_to_kp = true;
    tracker.Do2dAlignment(options,
                          tracker.GetImagePyramid(),
                          tracker.GetCurrentTracks(), last_optimization_level);
  });

  pangolin::RegisterKeyPressCallback('B', [&]() {
    do_bundle_adjustment = !do_bundle_adjustment;
    std::cerr << "Do BA:" << do_bundle_adjustment << std::endl;
  });

  pangolin::RegisterKeyPressCallback('i', [&]() {
    include_new_landmarks = !include_new_landmarks;
    std::cerr << "include new lms:" << include_new_landmarks << std::endl;
  });

  pangolin::RegisterKeyPressCallback('S', [&]() {
    do_start_new_landmarks = !do_start_new_landmarks;
    std::cerr << "Do SNL:" << do_start_new_landmarks << std::endl;
  });

  pangolin::RegisterKeyPressCallback('2', [&]() {
    last_optimization_level = 2;
    tracker.OptimizeTracks(last_optimization_level,
                           optimize_landmarks, optimize_pose);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('3', [&]() {
    last_optimization_level = 3;
    tracker.OptimizeTracks(last_optimization_level,
                           optimize_landmarks, optimize_pose);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('1', [&]() {
    last_optimization_level = 1;
    tracker.OptimizeTracks(last_optimization_level,
                           optimize_landmarks,
                           optimize_pose);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('0', [&]() {
    last_optimization_level = 0;
    tracker.OptimizeTracks(last_optimization_level,
                           optimize_landmarks,
                           optimize_pose);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('9', [&]() {
    last_optimization_level = 0;
    tracker.OptimizeTracks(-1, optimize_landmarks,
                           optimize_pose);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('p', [&]() {
    tracker.PruneTracks();
    // Update the pose t_ab based on the result from the tracker.
    UpdateCurrentPose();
    BaAndStartNewLandmarks();
  });

  pangolin::RegisterKeyPressCallback('l', [&]() {
    optimize_landmarks = !optimize_landmarks;
    std::cerr << "optimize landmarks: " << optimize_landmarks << std::endl;
  });

  pangolin::RegisterKeyPressCallback('c', [&]() {
    optimize_pose = !optimize_pose;
    std::cerr << "optimize pose: " << optimize_pose << std::endl;
  });

  pangolin::RegisterKeyPressCallback('m', [&]() {
    is_manual_mode = !is_manual_mode;
    std::cerr << "Manual mode:" << is_manual_mode << std::endl;
  });

  // Initialize the plotters.
  plot_views.resize(3);
  plot_logs.resize(3);
  double bottom = 0;
  for (size_t ii = 0; ii < plot_views.size(); ++ii) {
    plot_views[ii] = new pangolin::Plotter(&plot_logs[ii]);
    plot_views[ii]->SetBounds(bottom, bottom + 0.1, 0.6, 1.0);
    plot_views[ii]->ToggleTracking();
    bottom += 0.1;
    pangolin::DisplayBase().AddDisplay(*plot_views[ii]);
  }
}

int main(int argc, char** argv) {
  srand(0);
  cl = std::shared_ptr<GetPot>(new GetPot(argc, argv));
  if (cl->search("--help")) {
    LOG(INFO) << g_usage;
    exit(-1);
  }

  if (cl->search("-use_system_time")) {
    use_system_time = true;
  }

  if (cl->search("-posys")) {
    has_gps = true;
  }

  if (cl->search("-startnow")) {
    is_running = true;
    is_stepping = true;
  }

  LOG(INFO) << "Initializing camera...";
  LoadCameras();

  // Set the initial gravity from the first bit of IMU data.
  if (imu_buffer.elements.size() == 0) {
    LOG(ERROR) << "No initial IMU measurements were found.";
  }

  //////////////////////////
  /// ZZZZZZZZZZZZZZZZZZ: Get rid of this. Only valid for ICRA test rig
  imu_time_offset = 0;//-0.0697;
  if (cl->search("-ts")) {
    imu_time_offset = cl->follow(0.0, "-ts");
    LOG(INFO) << "Setting time offset to " << imu_time_offset << std::endl;
  }

  InitTracker();

  InitGui();

  aac_thread = std::shared_ptr<std::thread>(new std::thread(&DoAAC));

  gps_thread = std::shared_ptr<std::thread>(new std::thread(&DoGps));

  Run();

  return 0;
}
