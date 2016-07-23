// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.
#undef NDEBUG
#include <assert.h>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include "GetPot"
#include <unistd.h>
#include <iomanip>
#include <type_traits>

#include "etc_common.h"
#include <HAL/Camera/CameraDevice.h>
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
#include <thread>
#include "selfcal-cvars.h"
#include "chi2inv.h"
#include "sophus/so2.hpp"

#ifdef CHECK_NANS
#include <xmmintrin.h>
#endif

#include <sdtrack/semi_dense_tracker.h>

#include "online_calibrator.h"

#define POSES_TO_INIT 30

using sdtrackUtils::operator<<;
using sdtrack::log_decoupled;
using sdtrack::UnrotatePose;

uint32_t keyframe_tracks = UINT_MAX;
double start_time = 0;
uint32_t frame_count = 0;
Sophus::SE3d last_t_ba; // Last relative transform
Sophus::SE3d prev_delta_t_ba; //
Sophus::SE3d prev_t_ba; //

int debug_level_threshold = 0;

bool compare_self_cal_with_batch = false;
bool unknown_cam_calibration = false;
bool unknown_imu_calibration = true;


const int window_width = 640 * 1.5;
const int window_height = 480 * 1.5;
std::string g_usage = "SD SELFCAL. Example usage:\n"
                      "-cam file:[loop=1]///Path/To/Dataset/[left,right]*pgm -cmod cameras.xml";

bool is_keyframe = true;
bool is_prev_keyframe = true;
bool optimize_landmarks = true;
bool optimize_pose = true;
bool follow_camera = false;
bool is_running = false;
bool is_stepping = false;
bool is_manual_mode = false;
bool do_bundle_adjustment = true;
bool do_start_new_landmarks = true;
bool use_system_time = false;
bool analysed_imu_calib = false;
bool analysed_cam_calib = false;
double aac_time;
double aac_calls;
int image_width;
int image_height;
calibu::Rig<Scalar> rig;
calibu::Rig<Scalar> selfcal_rig;
calibu::Rig<Scalar> aac_rig;
hal::Camera camera_device;
bool has_imu = false;
hal::IMU imu_device;
sdtrack::SemiDenseTracker tracker;
uint32_t keyframe_id;

// ground truth Tvs for comparison purposes
Eigen::Vector6d gt_Tvs;
// ground truth camera parameters for comparison purposes
Eigen::VectorXd gt_cam_params;

bool has_gt = false;

enum CalibrationType
{
  Camera,
  IMU,
  Batch
};

struct Metrics{
  double batch_time = 0, ba_time = 0, analyze_time = 0, queue_time = 0, snl_time = 0, aac_time = 0;
  double batch_calls = 0, ba_calls = 0, analyze_calls = 0, queue_calls = 0, snl_calls = 0, aac_calls =0;
  double num_change_detections = 0;
  double num_windows_analysed = 0;
};

Metrics global_metrics;

struct Calibration {
  sdtrack::OnlineCalibrator online_calibrator;
  double last_window_kl_divergence = 0;
  double last_added_window_kl_divergence = 0;
  uint32_t unknown_calibration_start_pose = 0;

  // This is the overall priority queue window. It's start and end pose
  // do not have meaning since it's usually used to hold the mean and
  // covariance of the whole priority queue. It is also used in the initial
  // batch mode.
  sdtrack::CalibrationWindow pq_window;

  // This is the sliding window that is tested against each window in the
  // priority queue to see if it should be added or swapped in.
  sdtrack::CalibrationWindow candidate_window;

  sdtrack::CalibrationWindow current_window;
  uint32_t num_change_detected = 0;
  uint32_t num_change_needed = 3;
  uint32_t num_self_cal_segments = 5;
  uint32_t self_cal_segment_length = 10;
  bool plot_graphs = false;
  // Flag for doing self_cal specifically
  bool do_self_cal = true;
  bool unknown_calibration = false;
  CalibrationType type;
};

//std::vector<std::shared_ptr<Calibration>> calibrations;
std::map<CalibrationType, std::shared_ptr<Calibration>> calibrations;

std::vector<pangolin::DataLog> plot_logs;
std::vector<pangolin::Plotter*> plot_views;
std::vector<pangolin::Plotter*> analysis_views;
std::vector<pangolin::DataLog> analysis_logs;

TrackerGuiVars gui_vars;
pangolin::View* params_plot_view;
pangolin::View* imu_plot_view;
pangolin::View* analysis_plot_view;
std::shared_ptr<GetPot> cl;

// TrackCenterMap current_track_centers;
std::list<std::shared_ptr<sdtrack::DenseTrack>>* current_tracks = nullptr;
int last_optimization_level = 0;
// std::shared_ptr<sdtrack::DenseTrack> selected_track = nullptr;
std::shared_ptr<hal::Image> camera_img;
std::vector<std::vector<std::shared_ptr<SceneGraph::ImageView>>> patches;
std::vector<std::shared_ptr<sdtrack::TrackerPose>> poses;
std::vector<std::unique_ptr<SceneGraph::GLAxis>> axes;
std::shared_ptr<SceneGraph::GLPrimitives<>> line_strip;

// Inertial stuff
ba::BundleAdjuster<double, 1, 6, 0> bundle_adjuster;
ba::BundleAdjuster<double, 1, 15, 0> vi_bundle_adjuster;
ba::BundleAdjuster<double, 1, 15, 0> aac_bundle_adjuster;
ba::InterpolationBufferT<ba::ImuMeasurementT<Scalar>, Scalar> imu_buffer;
std::vector<uint32_t> ba_imu_residual_ids, aac_imu_residual_ids;
int orig_num_aac_poses = num_aac_poses;
double prev_cond_error;
int imu_cond_start_pose_id = -1;
int imu_cond_residual_id = -1;
std::shared_ptr<std::thread> aac_thread;
std::mutex aac_mutex;
std::mutex online_calibrator_mutex;

sdtrack::CalibrationWindow global_pq_window;

double total_last_frame_proj_norm = 0;

// State variables
std::vector<cv::KeyPoint> keypoints;
Sophus::SE3d guess; // Relative transform guess from the previous pose to the
// next.


///////////////////////////////////////////////////////////////////////////
Eigen::VectorXd CompareWithGt(bool cam, bool tvs,
                              Eigen::VectorXd params){
  Eigen::VectorXd result(params.rows());
  if(tvs && has_gt){
    for(uint32_t ii = 0; ii < params.rows(); ++ii){
      result[ii] = fabs((params[ii] - gt_Tvs[ii]));
    }
  }
  else if(cam && has_gt){
    for(uint32_t ii = 0; ii < params.rows(); ++ii){
      result[ii] = fabs((params[ii] - gt_cam_params[ii]));
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////
double ErrorScore(bool cam, bool tvs,
                  Eigen::VectorXd params){
  double result = 0;
  if(tvs && has_gt){
    result = (params - gt_Tvs).norm();
  }
  else if(cam && has_gt){
    result = (params - gt_Tvs).norm();
  }

  return result;
}


///////////////////////////////////////////////////////////////////////////
double GetTotalMeasured(Metrics m){
  return m.ba_time + m.analyze_time + m.batch_time + m.snl_time + m.queue_time
      + m.aac_time;
}

///////////////////////////////////////////////////////////////////////////
std::shared_ptr<Calibration> GetCalibration(CalibrationType type){
  return calibrations[type];
}


// Checks if self-cal is possible for a given online calibrator
///////////////////////////////////////////////////////////////////////////
bool SelfCalActive(std::shared_ptr<Calibration> calib){
  bool is_active = false;
  switch(calib->type){
  case Camera:
    is_active = calib->do_self_cal;
    break;
  case IMU:
    is_active = has_imu && use_imu_measurements && calib->do_self_cal;
    break;
  case Batch:
    // Always false since the batch calibrator should ouly be used for
    // the inital joint batch estimation and not online self-cal
    is_active = false;
  }

  return is_active;
}

///////////////////////////////////////////////////////////////////////////
void ImuCallback(const hal::ImuMsg& ref) {

  const double timestamp = use_system_time ? ref.system_time() :
                                             ref.device_time();

  //  std::cerr << std::setprecision(6) << std::fixed <<
  //              "imu system time: " << ref.system_time() <<
  //              " imu device time: " << ref.device_time() ;

  Eigen::VectorXd a, w;
  hal::ReadVector(ref.accel(), &a);
  hal::ReadVector(ref.gyro(), &w);

  imu_buffer.AddElement(ba::ImuMeasurementT<Scalar>(w, a, timestamp));
}

///////////////////////////////////////////////////////////////////////////
void CheckParameterChange(std::shared_ptr<Calibration> calib){
  if (calib->last_window_kl_divergence < 0.2 &&
      calib->last_window_kl_divergence != 0 &&
      (calib->online_calibrator.NumWindows() ==
       calib->online_calibrator.queue_length()) &&
      !calib->unknown_calibration) {
    calib->num_change_detected++;


    if (calib->num_change_detected >
        calib->num_change_needed) {
      VLOG(1) << "PARAM CHANGE DETECTED" ;
      calib->unknown_calibration = true;
      //TODO: Check this, seems like it should be num_change_needed *
      // self_cal_segment_length
      calib->unknown_calibration_start_pose
          = poses.size() - calib->num_change_needed;
      VLOG(1) << "Unknown cam calibration = true with start pose " <<
                 calib->unknown_calibration_start_pose
                 ;
      calib->online_calibrator.ClearQueue();
    }
  } else {
    // num_change_needed *consecutive* change detections are required to trigger
    // a parameter change, so zero out the number of change detections if a
    // window is not significantly different.
    calib->num_change_detected = 0;
  }
}

///////////////////////////////////////////////////////////////////////////
template <typename BaType>
void DoBundleAdjustment(BaType& ba, bool use_imu,
                        bool do_adaptive_conditioning,
                        uint32_t& num_active_poses, uint32_t id,
                        std::vector<uint32_t>& imu_residual_ids,
                        calibu::Rig<Scalar>& ba_rig)
{
  std::lock_guard<std::mutex> lock(aac_mutex);

  std::vector<uint32_t> last_frame_proj_residual_ids;
  if (reset_outliers) {
    for (std::shared_ptr<sdtrack::TrackerPose> pose : poses) {
      for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
        track->is_outlier = false;
      }
    }
    reset_outliers = false;
  }

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
  options.use_robust_norm_for_proj_residuals = use_robust_norm_for_proj;
  options.projection_outlier_threshold = outlier_threshold;
  options.use_per_pose_cam_params = true;
  options.regularize_biases_in_batch = poses.size() < POSES_TO_INIT ||
      regularize_biases_in_batch;


  uint32_t num_outliers = 0;
  Sophus::SE3d t_ba;
  uint32_t start_active_pose, start_pose_id;

  uint32_t end_pose_id;
  {
    //std::lock_guard<std::mutex> lock(aac_mutex);
    end_pose_id = poses.size() - 1;

    GetBaPoseRange(poses, num_active_poses, start_pose_id, start_active_pose);

    if (start_pose_id == end_pose_id) {
      return;
    }

    if(do_adaptive_conditioning){
      VLOG(3) << "Doing AAC with " << end_pose_id - start_pose_id + 1 << " poses"
              << " and " << end_pose_id - start_active_pose + 1 << " active poses" ;
    }

    // Add an extra pose to conditon the IMU
    // This will happen when the optimization windows is the same as the active window
    // so no landmarks were visible in the current window and also in past poses
    // expanding the window by 1 pose will include the IMU conditioning residual
    if (use_imu && use_imu_measurements && start_active_pose == start_pose_id &&
        start_pose_id != 0) {
      start_pose_id--;
      VLOG(3) << "expanding start pose from " << start_pose_id - 1 << " to " << start_pose_id ;
    }
  }

  bool all_poses_active = start_active_pose == start_pose_id;


  // Do a bundle adjustment on the current set
  if (current_tracks && end_pose_id) {
    {
      //std::lock_guard<std::mutex> lock(aac_mutex);
      if (use_imu) {
        ba.SetGravity(gravity_vector);
      }

      ba.Init(options, end_pose_id + 1, current_tracks->size() *
              (end_pose_id + 1));
      for (uint32_t cam_id = 0; cam_id < ba_rig.cameras_.size(); ++cam_id) {
        ba.AddCamera(ba_rig.cameras_[cam_id]);
      }

      // First add all the poses and landmarks to ba.
      for (uint32_t ii = start_pose_id ; ii <= end_pose_id ; ++ii) {
        std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];

        // Add pose to BA
        pose->opt_id[id] = ba.AddPose(
              pose->t_wp, pose->cam_params, pose->v_w, pose->b,
              ii >= start_active_pose , pose->time);

        if (ii == start_active_pose && use_imu && all_poses_active) {
          // Regularize the IMU nullspace: translation and the rotation about
          // the gravity vector.
          ba.RegularizePose(pose->opt_id[id], true, true, false, false);
        }

        if (use_imu && ii >= start_active_pose && ii > 0) {
          std::vector<ba::ImuMeasurementT<Scalar>> meas =
              imu_buffer.GetRange(poses[ii - 1]->time, pose->time);

          /*StreamMessage(selfcal_debug_level) << "Adding imu residual between poses " << ii - 1 << std::setprecision(15) <<
                       " with time " << poses[ii - 1]->time << " active: " <<
                       ba.GetPose(poses[ii - 1]->opt_id[id]).is_active <<
                       " and " << ii << " with time " << pose->time <<
                       " active: " <<
                       ba.GetPose(poses[ii]->opt_id[id]).is_active <<
                       " with " << meas.size() << " measurements" << std::endl;*/

          imu_residual_ids.push_back(
                ba.AddImuResidual(poses[ii - 1]->opt_id[id],
                pose->opt_id[id], meas));
          if (do_adaptive_conditioning) {
            if (imu_cond_start_pose_id == -1 &&
                !ba.GetPose(poses[ii - 1]->opt_id[id]).is_active &&
                ba.GetPose(pose->opt_id[id]).is_active) {
              VLOG(3) << "Setting cond pose id to " << ii - 1;
              imu_cond_start_pose_id = ii - 1;
              imu_cond_residual_id = imu_residual_ids.back();
              VLOG(3) << "Setting cond residual id to " <<
                         imu_cond_residual_id;
            } else if (imu_cond_start_pose_id == (int)(ii - 1)) {
              imu_cond_residual_id = imu_residual_ids.back();
              VLOG(3) << "Setting cond residual id to " <<
                         imu_cond_residual_id;
            }
          }
        }

        // Add landmarks to ba
        for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
          // Check if this landmark was seen in any poses that are in the
          // active window
          const bool constraints_active =
              track->keypoints.size() + ii > start_active_pose;
          if (track->num_good_tracked_frames <= 1 || track->is_outlier ||
              !constraints_active) {
            // Don't add this landmark to ba if there are no active constraints
            // associated to it
            track->external_id[id] = UINT_MAX;
            continue;
          }
          Eigen::Vector4d ray;
          ray.head<3>() = track->ref_keypoint.ray;
          ray[3] = track->ref_keypoint.rho;
          ray = sdtrack::MultHomogeneous(pose->t_wp  * ba_rig.cameras_[0]->Pose(), ray);

          // Disable the longest track id to remove ambiguity, if no imu is present
          // and all poses are active
          bool active = track->id != tracker.longest_track_id() ||
              !all_poses_active || use_imu;
          if (!active) {
            VLOG(3) << "Landmark " << track->id << " inactive. ";
          }
          track->external_id[id] =
              ba.AddLandmark(ray, pose->opt_id[id], 0, active);
        }
      }

      // Now add all reprojections to ba
      for (uint32_t ii = start_pose_id ; ii <= end_pose_id ; ++ii) {
        std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
        uint32_t total_proj_res = 0;
        for (std::shared_ptr<sdtrack::DenseTrack> track : pose->tracks) {
          if (track->external_id[id] == UINT_MAX) {
            continue;
          }
          for (uint32_t cam_id = 0; cam_id < ba_rig.cameras_.size(); ++cam_id) {
            for (size_t jj = 0; jj < track->keypoints.size() ; ++jj) {
              if (track->keypoints[jj][cam_id].tracked) {
                const Eigen::Vector2d& z = track->keypoints[jj][cam_id].kp;
                if (ba.GetNumPoses() > (pose->opt_id[id] + jj)) {
                  const uint32_t res_id =
                      ba.AddProjectionResidual(
                        z, pose->opt_id[id] + jj,
                        track->external_id[id], cam_id, 2.0);

                  // Store reprojection constraint ids for the last frame.
                  if ((ii + jj) == end_pose_id) {
                    last_frame_proj_residual_ids.push_back(res_id);
                  }
                  total_proj_res++;
                }
              }
            }
          }
        }

        if (!do_adaptive_conditioning) {
          // StreamMessage(selfcal_debug_level) << "Total proj res for pose: " << ii << ": " <<
          //             total_proj_res << std::endl;
        }
      }
    }

    // Optimize the poses
    ba.Solve(num_ba_iterations);


    {
      //std::lock_guard<std::mutex> lock(aac_mutex);

      total_last_frame_proj_norm = 0;

      ////ZZZZZZZZZZ THIS IS NOT THREAD SAFE
      // Calculate the average reprojection error.
      for (uint32_t id : last_frame_proj_residual_ids) {
        const auto& res = ba.GetProjectionResidual(id);
        total_last_frame_proj_norm += res.z.norm();
      }
      total_last_frame_proj_norm /= last_frame_proj_residual_ids.size();

      uint32_t last_pose_id =
          is_keyframe ? poses.size() - 1 : poses.size() - 2;
      std::shared_ptr<sdtrack::TrackerPose> last_pose = poses[last_pose_id];

      if (last_pose_id <= end_pose_id) {
        // Get the pose of the last pose. This is used to calculate the relative
        // transform from the pose to the current pose.
        last_pose->t_wp = ba.GetPose(last_pose->opt_id[id]).t_wp;
      }

      // Read out the pose and landmark values.
      for (uint32_t ii = start_pose_id ; ii <= end_pose_id ; ++ii) {
        std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
        const ba::PoseT<double>& ba_pose =
            ba.GetPose(pose->opt_id[id]);

        pose->t_wp = ba_pose.t_wp;
        if (use_imu) {
          pose->v_w = ba_pose.v_w;
          pose->b = ba_pose.b;
        }

        // Here the last pose is actually t_wb and the current pose t_wa.
        last_t_ba = t_ba;
        t_ba = last_pose->t_wp.inverse() * pose->t_wp;
        for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
          if (track->external_id[id] == UINT_MAX) {
            continue;
          }
          track->t_ba = t_ba;

          // Get the landmark location in the world frame.
          const Eigen::Vector4d& x_w =
              ba.GetLandmark(track->external_id[id]);
          double ratio =
              ba.LandmarkOutlierRatio(track->external_id[id]);


          if (do_outlier_rejection && !GetCalibration(Camera)->unknown_calibration &&
              poses.size() > POSES_TO_INIT) {
            if (ratio > 0.3 && track->tracked == false &&
                (end_pose_id >= min_poses_for_imu - 1 || !use_imu)) {
              //            if (ratio > 0.3 &&
              //                ((track->keypoints.size() == num_ba_poses - 1) ||
              //                 track->tracked == false)) {
              num_outliers++;
              track->is_outlier = true;
            } else {
              track->is_outlier = false;
            }
          }

          Eigen::Vector4d prev_ray;
          prev_ray.head<3>() = track->ref_keypoint.ray;
          prev_ray[3] = track->ref_keypoint.rho;
          // Make the ray relative to the pose.
          Eigen::Vector4d x_r =
              sdtrack::MultHomogeneous(
                (pose->t_wp * ba_rig.cameras_[0]->Pose()).inverse(), x_w);
          // Normalize the xyz component of the ray to compare to the original
          // ray.
          x_r /= x_r.head<3>().norm();
          track->ref_keypoint.rho = x_r[3];
        }
      }

      if (follow_camera) {
        FollowCamera(gui_vars, poses.back()->t_wp);
      }
    }
  }
  if (!do_adaptive_conditioning) {
    VLOG(2) << "Rejected " << num_outliers << " outliers." ;
  }

  const ba::SolutionSummary<Scalar>& summary = ba.GetSolutionSummary();
  // StreamMessage(selfcal_debug_level) << "Rejected " << num_outliers << " outliers." << std::endl;

  if (use_imu && imu_cond_start_pose_id != -1 && do_adaptive_conditioning) {
    const uint32_t cond_dims =
        summary.num_cond_inertial_residuals * BaType::kPoseDim +
        summary.num_cond_proj_residuals * 2;
    const Scalar cond_error = summary.cond_inertial_error +
        summary.cond_proj_error;

    const double cond_inertial_error =
        ba.GetImuResidual(
          imu_cond_residual_id).mahalanobis_distance;

    if (prev_cond_error == -1) {
      prev_cond_error = DBL_MAX;
    }

    const Scalar cond_v_chi2_dist =
        chi2inv(adaptive_threshold, summary.num_cond_proj_residuals * 2);
    const Scalar cond_i_chi2_dist =
        chi2inv(adaptive_threshold, BaType::kPoseDim);

    if (num_active_poses > end_pose_id) {
      num_active_poses = orig_num_aac_poses;
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
        num_active_poses += 30;
      } else {
        num_active_poses = orig_num_aac_poses;

      }
      prev_cond_error = cond_total_error;
    }
  }
}

///////////////////////////////////////////////////////////////////////////
void UpdateCurrentPose()
{
  std::shared_ptr<sdtrack::TrackerPose> current_pose = poses.back();
  if (poses.size() > 1) {
    current_pose->t_wp = poses[poses.size() - 2]->t_wp * tracker.t_ba().inverse();
  }

  // Also use the current tracks to update the index of the earliest covisible
  // pose.
  size_t max_track_length = 0;
  for (std::shared_ptr<sdtrack::DenseTrack>& track : tracker.GetCurrentTracks()) {
    max_track_length = std::max(track->keypoints.size(), max_track_length);
  }
  current_pose->longest_track = max_track_length;
  VLOG(3) << "Setting longest track for pose " << poses.size() << " to " <<
             current_pose->longest_track ;
}



///////////////////////////////////////////////////////////////////////////
void DoSerialAAC()
{
  aac_time = 0;

  if (has_imu && use_imu_measurements &&
      poses.size() > 10 && do_serial_ac) {
    orig_num_aac_poses = num_aac_poses;
    while (true) {
      if (poses.size() > min_poses_for_imu &&
          use_imu_measurements && has_imu) {
        {
          // Get the lastest parameters from the rig
          std::lock_guard<std::mutex> lock(aac_mutex);
          //aac_rig.cameras_[0]->SetParams(rig.cameras_[0]->GetParams());
          Eigen::VectorXd rig_cam_params = rig.cameras_[0]->GetParams();
          aac_rig.cameras_[0]->SetParams(rig_cam_params);

          if(do_only_rotation_imu_self_cal){
            Sophus::SE3t new_imu_params = rig.cameras_[0]->Pose();
            new_imu_params.translation() = aac_rig.cameras_[0]->Pose().
                translation();
            aac_rig.cameras_[0]->SetPose(new_imu_params);
          }else{
            aac_rig.cameras_[0]->SetPose(rig.cameras_[0]->Pose());
          }


        }
        aac_time = sdtrack::Tic();
        aac_calls++;
        VLOG(3) << "AAC active poses: " <<
                   num_aac_poses ;
        DoBundleAdjustment(aac_bundle_adjuster, true, do_adaptive,
                           num_aac_poses, 1, aac_imu_residual_ids,
                           aac_rig);
        aac_time = sdtrack::Toc(aac_time);
        global_metrics.aac_calls += aac_calls;
        global_metrics.aac_time += aac_time;

      }

      if ((int)num_aac_poses == orig_num_aac_poses || !do_adaptive) {
        // If the adaptive window did not have to increase, or
        // if adaptive mode had been tured off, exit the inner loop
        break;
      }

      usleep(10);
    }

    imu_cond_start_pose_id = -1;
    prev_cond_error = -1;
  }

}

///////////////////////////////////////////////////////////////////////////
void DoAAC()
{
  aac_time = 0;
  while (true) {
    if (has_imu && use_imu_measurements &&
        poses.size() > 10 && do_async_ba &&
        !GetCalibration(IMU)->unknown_calibration) {
      orig_num_aac_poses = num_aac_poses;
      while (true) {
        if (poses.size() > min_poses_for_imu &&
            use_imu_measurements && has_imu) {
          {
            // Get the latest parameters from the rig
            std::lock_guard<std::mutex> lock(aac_mutex);
            aac_rig.cameras_[0]->SetParams(rig.cameras_[0]->GetParams());

            if(do_only_rotation_imu_self_cal){
              Sophus::SE3t new_imu_params = rig.cameras_[0]->Pose();
              new_imu_params.translation() = aac_rig.cameras_[0]->Pose().
                  translation();
              aac_rig.cameras_[0]->SetPose(new_imu_params);
            }else{
              aac_rig.cameras_[0]->SetPose(rig.cameras_[0]->Pose());
            }
          }
          aac_time = sdtrack::Tic();
          aac_calls++;
          DoBundleAdjustment(aac_bundle_adjuster, true, do_adaptive,
                             num_aac_poses, 1, aac_imu_residual_ids,
                             aac_rig);
          aac_time = sdtrack::Toc(aac_time);
          global_metrics.aac_calls += aac_calls;
          global_metrics.aac_time += aac_time;
        }

        if ((int)num_aac_poses == orig_num_aac_poses || !do_adaptive) {
          // If the adaptive window did not have to increase, or
          // if adaptive mode had been tured off, exit the inner loop
          break;
        }

        usleep(100);
      }

      imu_cond_start_pose_id = -1;
      prev_cond_error = -1;
    }
    usleep(1000);
  }
}

///////////////////////////////////////////////////////////////////////////
void ApplyPriorityQueueResults(bool apply_results){

  VLOG(1) << "PQ analysis finished. Applying results to rig. apply results = "
          << apply_results;

  std::shared_ptr<Calibration> cam_calib = GetCalibration(Camera);
  std::shared_ptr<Calibration> imu_calib = GetCalibration(IMU);


  // Apply the results from selfcal_rig over to the actual rig
  if (apply_results) {

    std::unique_lock<std::mutex>(aac_mutex);
    std::unique_lock<std::mutex>(online_calibrator_mutex);
    Eigen::VectorXd new_params = selfcal_rig.cameras_[0]->GetParams();
    rig.cameras_[0]->SetParams(new_params);
    if(do_only_rotation_imu_self_cal){
      Sophus::SE3t new_imu_params = selfcal_rig.cameras_[0]->Pose();
      // Preserve the rig's translation
      new_imu_params.translation() = rig.cameras_[0]->Pose().translation();
      rig.cameras_[0]->SetPose(new_imu_params);
    }else{
      rig.cameras_[0]->SetPose(selfcal_rig.cameras_[0]->Pose());
    }

    // compare the new pq mean to the gt, if available
    if(has_gt && analysed_imu_calib){
      Eigen::VectorXd calib_comparison =
          CompareWithGt(false, true,
                        log_decoupled(UnrotatePose(rig.cameras_[0]->Pose())));

      VLOG(1) << "PQ error: " << calib_comparison.transpose();
      VLOG(1) << "PQ error score: " << calib_comparison.norm();

      std::ofstream("imu_pq_error.txt", std::ios_base::app) <<
                                                               keyframe_id << ", " <<
                                                               calib_comparison.transpose().format(sdtrack::kLongCsvFmt)
                                                            << std::endl;

    }

    // Write this to the imu candidate window file
    std::ofstream("imu_pq.txt", std::ios_base::app) << keyframe_id << ", " <<
                                                       imu_calib->candidate_window.covariance.diagonal().transpose().format(
                                                         sdtrack::kLongCsvFmt) << ", " <<
                                                       imu_calib->pq_window.mean.transpose().format(
                                                         sdtrack::kLongCsvFmt) << ", " <<
                                                       selfcal_rig.cameras_[0]->Pose().translation().transpose().format(
                                                         sdtrack::kLongCsvFmt) << ", " <<
                                                       selfcal_rig.cameras_[0]->Pose().rotationMatrix().eulerAngles(0,1,2).transpose().format(
                                                         sdtrack::kLongCsvFmt) << std::endl;

    // Set the correct cam params for all the poses that were
    // created with the previous parameters
    // If we are not doint cam self-cal then it will just set the
    // same parameters
    for (size_t ii = cam_calib->unknown_calibration_start_pose;
         ii < poses.size(); ++ii) {
      for (std::shared_ptr<sdtrack::DenseTrack> track : poses[ii]->tracks) {
        poses[ii]->cam_params = new_params;
        track->ref_keypoint.ray =
            rig.cameras_[0]->Unproject(
              track->ref_keypoint.center_px).normalized();
        track->needs_backprojection = true;
      }
    }

    VLOG(1) << "Setting new PQ params from selfcal_rig to rig: "
               ;
    VLOG(1) << "new rig cam params: " <<
               rig.cameras_[0]->GetParams().transpose()
               ;

    VLOG(1) << "new rig Tvs params: \n" << sdtrack::UnrotatePose
               (rig.cameras_[0]->Pose());
  }

  if(analysed_cam_calib){
    {
      std::lock_guard<std::mutex> lock(aac_mutex);
      VLOG(1) << "Analyzed camera priority queue with mean " <<
                 cam_calib->pq_window.mean.transpose() << " and cov\n " <<
                 cam_calib->pq_window.covariance ;
    }
    cam_calib->online_calibrator.SetPriorityQueueDistribution(
          cam_calib->pq_window.covariance,
          cam_calib->pq_window.mean);
  }

  if(analysed_imu_calib){
    VLOG(1) << "Analyzed IMU priority queue with mean " <<
               imu_calib->pq_window.mean.transpose() << " and cov\n " <<
               imu_calib->pq_window.covariance ;
    imu_calib->online_calibrator.SetPriorityQueueDistribution(
          imu_calib->pq_window.covariance,
          imu_calib->pq_window.mean);
  }
}

///////////////////////////////////////////////////////////////////////////
void  BaAndStartNewLandmarks()
{

  if (!is_keyframe) {
    return;
  }

  std::shared_ptr<Calibration> cam_calib = GetCalibration(Camera);
  cam_calib->do_self_cal = do_cam_self_cal;
  cam_calib->self_cal_segment_length = min_poses_for_camera;


  std::shared_ptr<Calibration> imu_calib = GetCalibration(IMU);
  imu_calib->do_self_cal = do_imu_self_cal;
  imu_calib->self_cal_segment_length = min_poses_for_imu;

  std::shared_ptr<Calibration> batch_calib = GetCalibration(Batch);

  keyframe_id = poses.size();
  double batch_time = 0, ba_time = 0, analyze_time = 0, queue_time = 0, snl_time = 0;
  const uint32_t batch_end = poses.size();

  bool have_unknown_calib = false;
  for (auto const &calib : calibrations){
    calib.second->online_calibrator.SetDebugLevel(selfcal_debug_level);
    if(SelfCalActive(calib.second) && calib.second->unknown_calibration){
      have_unknown_calib = true;
      VLOG(1) << "Have unknown calibration." <<
                 std::endl;
      break;
    }
  }

  batch_time = sdtrack::Tic();

  if(have_unknown_calib){
    global_metrics.batch_calls++;

    // If we have an unknown calibration, do batch optimization until we
    // converge on the calibration parameters
    double score = 0;
    int num_params = 0;
    bool use_batch_candidate = true;

    // If we are optimizing over camera and imu parameters, but know
    // nothing about either, start in joint batch mode
    if(cam_calib->unknown_calibration && imu_calib->unknown_calibration &&
       SelfCalActive(imu_calib) && SelfCalActive(cam_calib) &&
       ((batch_end - batch_calib->unknown_calibration_start_pose)
        > batch_calib->self_cal_segment_length)){
      if(poses.size() > min_poses_for_imu) {
        VLOG(1) << "Performing batch joint optimization for the camera "
                << "and IMU calibration (visual + inertial)" <<
                   std::endl;
        batch_calib->online_calibrator.
            AnalyzeCalibrationWindow<true, true>(
              poses, current_tracks, batch_calib->unknown_calibration_start_pose,
              batch_end, batch_calib->pq_window, num_selfcal_ba_iterations, true,
              do_only_rotation_imu_self_cal);
        score = batch_calib->online_calibrator.GetWindowScore(
              batch_calib->pq_window);
        global_pq_window = batch_calib->pq_window;
        num_params = selfcal_rig.cameras_[0]->GetParams().rows() +
            Sophus::SE3t::DoF;
      }
    }
    // If we have no idea about *only* the camera calibration, do cam batch mode.
    else if (cam_calib->unknown_calibration && SelfCalActive(cam_calib)
             && ((batch_end - cam_calib->unknown_calibration_start_pose)
                 > cam_calib->self_cal_segment_length)) {
      // Only visual
      VLOG(1) << "Performing batch optimization for the camera "
              << "calibration (visual only)";
      cam_calib->online_calibrator
          .AnalyzeCalibrationWindow<false, false>(
            poses, current_tracks, cam_calib->unknown_calibration_start_pose,
            batch_end, cam_calib->pq_window, num_selfcal_ba_iterations, true);
      score = cam_calib->online_calibrator.GetWindowScore(cam_calib->pq_window);
      global_pq_window = cam_calib->pq_window;

      //TODO: Check if we still need the 'current_window'
      if(cam_calib->pq_window.mean.rows() != 0){
        cam_calib->current_window = cam_calib->pq_window;
      }
      num_params = selfcal_rig.cameras_[0]->GetParams().rows();

    }
    // If we have no idea about *only* the imu calibration, do imu batch mode.
    else if (imu_calib->unknown_calibration && SelfCalActive(imu_calib)
             && ((batch_end - imu_calib->unknown_calibration_start_pose
                  ) > imu_calib->self_cal_segment_length)) {
      if (poses.size() > min_poses_for_imu) {
        VLOG(1) << "Performing batch optimization for the IMU "
                << "calibration";

        // while doing batch, initially just optimize over the rotation of the imu-camera
        // calibration.
        use_batch_candidate= imu_calib->online_calibrator
            .AnalyzeCalibrationWindow<true, true>(
              poses, current_tracks, imu_calib->unknown_calibration_start_pose,
              batch_end, imu_calib->pq_window, num_selfcal_ba_iterations,
              true,
              do_only_rotation_imu_self_cal);


        if(use_batch_candidate){
          score = imu_calib->online_calibrator.
              GetWindowScore(imu_calib->pq_window);
          global_pq_window = imu_calib->pq_window;

          if(imu_calib->pq_window.mean.rows() != 0){
            imu_calib->current_window = imu_calib->pq_window;
          }
        }
        num_params = Sophus::SE3t::DoF;
      }
    }


    if (global_pq_window.covariance.fullPivLu().rank() ==
        num_params && num_params != 0 && use_batch_candidate /*&& score < 1e7*/) {
      // window is good enough for us to use the calibraiton
      // parameters in the actual rig:

      // Copy over the new parameters

      {
        std::unique_lock<std::mutex>(online_calibrator_mutex);
        std::unique_lock<std::mutex>(aac_mutex);
        rig.cameras_[0]->SetParams(selfcal_rig.cameras_[0]->GetParams());
        rig.cameras_[0]->SetPose(selfcal_rig.cameras_[0]->Pose());

      }

      VLOG(1) << "Setting new batch params from selfcal_rig to rig: "
                 ;
      VLOG(1) << "new rig cam params: " <<
                 rig.cameras_[0]->GetParams().transpose()
                 ;

      VLOG(1)  << "new rig Tvs params: " <<
                  UnrotatePose(rig.cameras_[0]->Pose());

      {
        // We need to backproject all the tracks associated
        // to the poses that we did not know anything about the
        // calibration.

        std::unique_lock<std::mutex>(aac_mutex);
        for (uint32_t ii = cam_calib->unknown_calibration_start_pose ;
             ii < poses.size() ; ++ii) {
          // Set the per-pose camera parameters. If the params change, we need
          // to know the right calibration for each pose.
          poses[ii]->cam_params = rig.cameras_[0]->GetParams();
          for (std::shared_ptr<sdtrack::DenseTrack> track: poses[ii]->tracks) {
            if (track->external_id[0] == UINT_MAX) {
              continue;
            }
            track->ref_keypoint.ray =
                rig.cameras_[0]->Unproject(
                  track->ref_keypoint.center_px).normalized();
            track->needs_backprojection = true;
          }
        }
      }

    }

    if(num_params > 0 && use_batch_candidate){

      // Write this to the batch file.
      std::ofstream("batch.txt", std::ios_base::app) << keyframe_id << ", " <<
                                                        global_pq_window.covariance.diagonal().transpose().format(
                                                          sdtrack::kLongCsvFmt) << ", " << score << ", " <<
                                                        global_pq_window.mean.transpose().format(sdtrack::kLongCsvFmt) ;

      VLOG(1) << "Batch means are: " << global_pq_window.mean.transpose();
      VLOG(1) << "Batch sigmas are:\n" << global_pq_window.covariance;
      VLOG(1) << "Batch score: "<< score;

      // If the determinant is smaller than a heuristic, switch to cam self_cal.
      if ((SelfCalActive(cam_calib)) &&
          ((score < 1e7 && score != 0 && !std::isnan(score) && !std::isinf(score)) ||
           ((batch_end - cam_calib->unknown_calibration_start_pose)
            > cam_calib->self_cal_segment_length * 2))) {
        VLOG(1) << "Determinant small enough, or we have enough poses: switching to cam self-cal";
        ;
        cam_calib->unknown_calibration = false;
      }

      // If the determinant is smaller than a heuristic, switch to imu self_cal.
      //TODO: Check if this heuristic is applicable to imu calibraiton
      if ((SelfCalActive(imu_calib)) &&
          ((score < 1e7 && score != 0 && !std::isnan(score) && !std::isinf(score)) ||
           ((batch_end - imu_calib->unknown_calibration_start_pose)
            > imu_calib->self_cal_segment_length * 2))) {
        VLOG(1) << "Determinant small enough, or we have enough poses.";
        if(has_gt){
          Eigen::VectorXd error = CompareWithGt(false, true,
                                                log_decoupled(UnrotatePose(rig.cameras_[0]->Pose())));
          VLOG(1) << "Tvs after batch Error: " << error.transpose();
          VLOG(1) << "Tvs after batch Error Norm: " << error.norm();

        }
        if(do_only_rotation_imu_self_cal){
          // we were optimizing only over rotation. try to converge on
          // an initial estimate for the translation as well
          VLOG(1) << "Enabling translation optimization.";
          do_only_rotation_imu_self_cal = false;
        }else{
          // we optimized over rotation and translation.
          // the calibraiton optimization can now be handed over to the
          // priority queue.
          VLOG(1) << "Rotation and Translation estimates found. Passing over to imu self-cal.";
          imu_calib->unknown_calibration = false;
        }
      }
    }else{
      if(use_batch_candidate){
        // The number of poses in the optimization has not yet reached the
        // minimum
        VLOG(1) << "Not enough poses to estimate parameters. "
                << "Num poses since unknown cam calib: "
                << batch_end - cam_calib->unknown_calibration_start_pose
                << " min poses for camera: "
                << cam_calib->self_cal_segment_length
                << " Num poses since unknown imu calib: "
                << batch_end - imu_calib->unknown_calibration_start_pose
                << " min poses for imu: " << imu_calib->self_cal_segment_length
                   ;
      }else{
        VLOG(1) << "Enougth poses for batch estimate, but the estimate was bad so"
                << " not using";
      }
    }

  }// END Batch Mode
  batch_time = sdtrack::Toc(batch_time);


  if (do_bundle_adjustment) {
    ba_time = sdtrack::Tic();

    bool has_unknown_calibration = false;
    for(auto const &calib : calibrations){
      if(calib.second->unknown_calibration && SelfCalActive(calib.second)){
        has_unknown_calibration = true;
        break;
      }
    }

    uint32_t ba_size = num_ba_poses;

    if(has_unknown_calibration){

      // If we still haven't converged on all of the calibration parameters
      // include all the poses with unknown calibration in the pose graph ba
      ba_size = std::max(
            (SelfCalActive(imu_calib) && imu_calib->unknown_calibration) ?
              batch_end - imu_calib->unknown_calibration_start_pose : num_ba_poses,
            (SelfCalActive(cam_calib) && cam_calib->unknown_calibration) ?
              batch_end - cam_calib->unknown_calibration_start_pose : num_ba_poses);
    }

    VLOG(3) << "ba_size: " << ba_size;

    // Do sliding window bundle adjustment on just poses and landmarks, using the lastest
    // calibration parameters. This optimizes the pose graph and landmarks.
    if (has_imu && use_imu_measurements &&
        poses.size() > min_poses_for_imu && !imu_calib->unknown_calibration) {
      VLOG(2) << "doing VI BA.";
      global_metrics.ba_calls++;
      DoBundleAdjustment(vi_bundle_adjuster, true, false, ba_size, 0,
                         ba_imu_residual_ids, rig);
    } else {
      VLOG(2) << "doing visual BA.";
      global_metrics.ba_calls++;
      DoBundleAdjustment(bundle_adjuster, false, false, ba_size, 0,
                         ba_imu_residual_ids, rig);
    }
    ba_time = sdtrack::Toc(ba_time);

    // Check if we need to do self-cal on camera or imu parameters
    // Even though the batch estimation might still be ongoing, this will
    // analyse candidate windows to see if they are eligible to be put in the
    // priority queue.
    bool should_do_self_cal = false;
    for(auto const &calib : calibrations){
      should_do_self_cal = (SelfCalActive(calib.second)) &&
          (batch_end - calib.second->unknown_calibration_start_pose >
           calib.second->self_cal_segment_length);
      if(should_do_self_cal)
        break;
    }

    if (should_do_self_cal) {
      analyze_time = sdtrack::Tic();
      global_metrics.analyze_calls++;


      // Check if camera parameters self-cal is active and if we have
      // enough poses to run the optimizaiton
      if(SelfCalActive(cam_calib) &&
         (batch_end - cam_calib->unknown_calibration_start_pose) >
         cam_calib->self_cal_segment_length){
        global_metrics.num_windows_analysed++;
        uint32_t start_pose =
            std::max(0, (int)poses.size() - (int)cam_calib->self_cal_segment_length);
        uint32_t end_pose = poses.size();
        VLOG(1) << "Analyzing calibration window for camera parameters (visual) "
                << "from pose " << start_pose << " to pose ";
        cam_calib->online_calibrator.AnalyzeCalibrationWindow<false, false>(
              poses, current_tracks, start_pose, end_pose,
              cam_calib->candidate_window, num_selfcal_ba_iterations);

        // Analyse the candidate window and add to the priority queue if it's good enough
        cam_calib->online_calibrator.AnalyzeCalibrationWindow(
              cam_calib->candidate_window);
      }

      // Check if IMU parameters self-cal is active and if we have
      // enough poses to run the optimizaiton
      if(SelfCalActive(imu_calib) &&
         (batch_end - imu_calib->unknown_calibration_start_pose) >
         imu_calib->self_cal_segment_length){
        uint32_t start_pose =
            std::max(0, (int)poses.size() - (int)imu_calib->self_cal_segment_length);
        uint32_t end_pose = poses.size();
        VLOG(1) << "Analyzing calibration window for IMU parameters from pose " <<
                   start_pose << " to pose " << end_pose << " (visual + imu)";
        // Produce an estimate for the calibration params. for this calibration
        // window
        bool use_candidate =
            imu_calib->online_calibrator.AnalyzeCalibrationWindow<true, true>(
              poses, current_tracks, start_pose, end_pose,
              imu_calib->candidate_window, num_selfcal_ba_iterations, false,
              do_only_rotation_imu_self_cal);


        // Analyse the candidate window and add to the priority queue
        // if it's good enough
        if(use_candidate && imu_calib->online_calibrator.AnalyzeCalibrationWindow(
             imu_calib->candidate_window)){
          // calibration window was added to the priority queue
          // compare to the gt, if available
          Eigen::VectorXd tvs_comparison = CompareWithGt(false, true,
                                                         imu_calib->candidate_window.mean);
          VLOG(1) << "Window error : " << tvs_comparison.transpose();
          VLOG(1) << "Window error score: " << tvs_comparison.norm();
        }


        // Write this to the imu candidate window file.
        std::ofstream("imu_candidate.txt", std::ios_base::app)
            << keyframe_id << ", " << end_pose - start_pose << ", " <<
               imu_calib->candidate_window.mean.transpose().format(
                 sdtrack::kLongCsvFmt) << ", " <<
               imu_calib->candidate_window.covariance.diagonal().transpose().format(
                 sdtrack::kLongCsvFmt) << ", " <<
               imu_calib->candidate_window.score << std::endl;

      }

      //----------CHANGE DETECTION----------------//
      if(SelfCalActive(cam_calib)){
        cam_calib->last_window_kl_divergence =
            cam_calib->online_calibrator.ComputeYao1965(
              cam_calib->pq_window,
              cam_calib->candidate_window);
      }

      if(SelfCalActive(imu_calib)){
        // Set this to zero since we dont't yet know how to calculate the mean
        // for Tvs optimization
        //TODO: Implement 6DOF transform mean and remove this line
        // this effectively removes the change detection, as a change will
        // never be triggered.
        imu_calib->last_window_kl_divergence = 0;

        //        imu_calib->last_window_kl_divergence =
        //            imu_calib->online_calibrator.ComputeYao1965(
        //              imu_calib->pq_window,
        //              imu_calib->candidate_window);
      }

      VLOG(2) << "KL divergence for last cam  " <<
                 cam_calib->last_window_kl_divergence
              << " num window changes: " <<
                 (int)cam_calib->num_change_detected;

      for(auto const &calib : calibrations){
        if(SelfCalActive(calib.second)){

          if(calib.second->candidate_window.mean.rows() != 0){
            calib.second->current_window = calib.second->candidate_window;
          }

          // Treat cases where the KL divergence was not computable
          if (isnan(calib.second->last_window_kl_divergence) ||
              isinf(calib.second->last_window_kl_divergence)) {
            calib.second->last_window_kl_divergence = 0;
          }

          // Check if there has been a change in calibration parameters.
          // If so, clear the priority queue.
          CheckParameterChange(calib.second);

        }
      }

      //----------------------END CHANGE DETECTION--------------------------//

      analyze_time = sdtrack::Toc(analyze_time);

      // If the priority queue was modified, calculate the new results for it.
      // This is when a window is swapped out or added to the priority queue
      bool queue_needs_update = false;
      for(auto const &calib : calibrations){
        queue_needs_update = calib.second->online_calibrator.needs_update() /*&&
            !calib.second->unknown_calibration*/ && SelfCalActive(calib.second);

        if(queue_needs_update){
          VLOG(1) << "PQ for sensor: " <<  calib.first << " needs update.";
          break;
        }
      }

      if (queue_needs_update) {
        VLOG(1) << "PQ modified, need to calculate"
                << " new mean" ;

        analysed_cam_calib = false;
        analysed_imu_calib = false;

        queue_time = sdtrack::Tic();
        global_metrics.queue_calls++;
        for(auto const &calib : calibrations){

          if(calib.second->online_calibrator.needs_update() &&
             !calib.second->unknown_calibration){
            // This is just for plotting the KL divergence
            calib.second->last_added_window_kl_divergence =
                calib.second->last_window_kl_divergence;
          }

        }



        if (SelfCalActive(imu_calib)
            && imu_calib->online_calibrator.needs_update()) {
          analysed_imu_calib = true;

          // async priority queue, just set the parameters.

          std::shared_ptr<sdtrack::PriorityQueueParams> params =
              imu_calib->online_calibrator.PriorityQueueParameters();
          params->poses = poses;
          params->current_tracks = *current_tracks;
          params->num_iterations = num_selfcal_ba_iterations;
          // only apply resuslts if the calibration is already known,
          // otherwise the batch estimation will take care of estimating
          // the calibraiton parameters.
          params->apply_results = !imu_calib->unknown_calibration;
          params->rotation_only_Tvs = do_only_rotation_imu_self_cal;
          params->overal_window = &(imu_calib->pq_window);


          // notify the online calibrator that there is a queue to be analysed
          VLOG(1) << "Notifying online calibrator that PQ needs updating...";
          imu_calib->online_calibrator.NotifyConditionVariable();

          //          imu_calib->online_calibrator.AnalyzePriorityQueue<true, true>(
          //                poses, current_tracks, imu_calib->pq_window, num_selfcal_ba_iterations,
          //                !imu_calib->unknown_calibration, do_only_rotation_imu_self_cal);
        }

        if (SelfCalActive(cam_calib) && cam_calib->online_calibrator.needs_update()){
          analysed_cam_calib = true;
          VLOG(1) << "Analyzing Cam params PQ (visual)..." ;
          cam_calib->online_calibrator.AnalyzePriorityQueue<false, false>(
                poses, current_tracks, cam_calib->pq_window, num_selfcal_ba_iterations,
                !cam_calib->unknown_calibration);
        }





        //        const double cam_score =
        //            cam_calib->online_calibrator.GetWindowScore(cam_calib->pq_window);
        //        const double imu_score =
        //            imu_calib->online_calibrator.GetWindowScore(imu_calib->pq_window);

        //        // Write this to the pq file.
        //        std::ofstream("cam_pq.txt", std::ios_base::app) << keyframe_id << ", " <<
        //          cam_calib->pq_window.covariance.diagonal().transpose().format(
        //            sdtrack::kLongCsvFmt) << ", " << cam_score << ", " <<
        //          cam_calib->pq_window.mean.transpose().format(sdtrack::kLongCsvFmt) << ", " <<
        //          cam_calib->last_window_kl_divergence << std::endl;

        //        std::ofstream("imu_pq.txt", std::ios_base::app) << keyframe_id << ", " <<
        //          imu_calib->pq_window.covariance.diagonal().transpose().format(
        //            sdtrack::kLongCsvFmt) << ", " << imu_score << ", " <<
        //          imu_calib->pq_window.mean.transpose().format(sdtrack::kLongCsvFmt) << ", " <<
        //          imu_calib->last_window_kl_divergence << std::endl;

        if (compare_self_cal_with_batch && !cam_calib->unknown_calibration
            && SelfCalActive(cam_calib)) {
          /* // Also analyze the full batch solution.
          sdtrack::CalibrationWindow batch_window;
          if (ImuSelfCalActive()) {
            camera_online_calib.AnalyzeCalibrationWindow<true, true>(
                  poses, current_tracks, 0, poses.size(), batch_window, 50);
          } else {
            camera_online_calib.AnalyzeCalibrationWindow<false, false>(
                  poses, current_tracks, 0, poses.size(), batch_window, 50);
          }

          const double batch_score =
              camera_online_calib.GetWindowScore(batch_window);

          // Write this to the batch file.
          std::ofstream("batch.txt", std::ios_base::app) << keyframe_id << ", " <<
            batch_window.covariance.diagonal().transpose().format(
            sdtrack::kLongCsvFmt) << ", " << batch_score << ", " <<
            batch_window.mean.transpose().format(sdtrack::kLongCsvFmt) <<
            std::endl;

          StreamMessage(selfcal_debug_level) << "Batch means are: " << batch_window.mean.transpose() <<
                       std::endl;
          StreamMessage(selfcal_debug_level) << "Batch sigmas are:\n" <<
                       batch_window.covariance << std::endl;
          StreamMessage(selfcal_debug_level) << "Batch score: " << batch_score << std::endl;*/
        }

        queue_time = sdtrack::Toc(queue_time);
      }// END PQ UPDATE
    }

    /* if ((do_cam_self_cal || ImuSelfCalActive()) && current_window.mean.rows() != 0) {
      std::ofstream("sigmas.txt", std::ios_base::app) << keyframe_id << ", " <<
        current_window.covariance.diagonal(). transpose().format(
          sdtrack::kLongCsvFmt) << ", " <<
        current_window.mean.transpose().format(sdtrack::kLongCsvFmt) <<
        ", " << last_window_kl_divergence << ", " << current_window.score <<
        std::endl;
    }*/
  }// END BUNDLE ADJUSTMENT

  if (do_start_new_landmarks) {
    snl_time = sdtrack::Tic();
    tracker.StartNewLandmarks();
    global_metrics.snl_calls++;
    snl_time = sdtrack::Toc(snl_time);
  }

  VLOG(4) << "Timings batch: " << batch_time << " ba: " << ba_time <<
             " analyze: " << analyze_time << " queue: " << queue_time <<
             " snl: " << snl_time ;

  global_metrics.batch_time += batch_time;
  global_metrics.ba_time += ba_time;
  global_metrics.analyze_time += analyze_time;
  global_metrics.queue_time += queue_time;
  global_metrics.snl_time += snl_time;
  double total_time = GetTotalMeasured(global_metrics);

  VLOG(4) << "Global timings ("<< total_time << ") -> batch: "
          << global_metrics.batch_time << "(" << global_metrics.batch_time/total_time*100
          << "%)"
          << " ba: " << global_metrics.ba_time << " (" << global_metrics.ba_time/total_time*100 << "%)"
                                                                                                   " analyze: " << global_metrics.analyze_time << " (" << global_metrics.analyze_time/total_time*100 << "%)"
          << " queue: " <<  global_metrics.queue_time << " (" << global_metrics.queue_time/total_time*100 << "%)" <<
             " snl: " << global_metrics.snl_time << " (" << global_metrics.snl_time/total_time*100 << "%)" <<
             " aac: " << global_metrics.aac_time << " (" << global_metrics.aac_time/total_time*100 << "%)" ;
  VLOG(4) << "Global time/call -> batch: " << (global_metrics.batch_calls > 0 ? global_metrics.batch_time/global_metrics.batch_calls:0)
          << " ba: " << global_metrics.ba_time/global_metrics.ba_calls <<
             " analyze: " << (global_metrics.analyze_calls > 0 ? global_metrics.analyze_time/global_metrics.analyze_calls : 0)
          << " queue: " << (global_metrics.queue_calls > 0 ? global_metrics.queue_time/global_metrics.queue_calls : 0) <<
             " snl: " <<  global_metrics.snl_time/global_metrics.snl_calls <<
             " aac: " <<  (global_metrics.aac_calls > 0 ? global_metrics.aac_time/global_metrics.aac_calls : 0) ;

  std::ofstream("timings.txt", std::ios_base::app) << keyframe_id << ", " <<
                                                      batch_time << ", " << ba_time << ", " << analyze_time << ", " <<
                                                      queue_time << ", " << snl_time ;

  std::shared_ptr<sdtrack::TrackerPose> new_pose = poses.back();
  // Update the tracks on this new pose.
  new_pose->tracks = tracker.GetNewTracks();

  if (!do_bundle_adjustment) {
    tracker.TransformTrackTabs(tracker.t_ba());
  }
}

///////////////////////////////////////////////////////////////////////////
void ProcessImage(std::vector<cv::Mat>& images, double timestamp)
{
  bundle_adjuster.debug_level_threshold = ba_debug_level;
  vi_bundle_adjuster.debug_level_threshold = vi_ba_debug_level;
  aac_bundle_adjuster.debug_level_threshold = aac_ba_debug_level;

  // Set the desired debug levels for all the online calibrators ba instances
  for(auto &calib : calibrations){
    calib.second->online_calibrator.SetBaDebugLevel(selfcal_ba_debug_level);
  }


#ifdef CHECK_NANS
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() &
                         ~(_MM_MASK_INVALID | _MM_MASK_OVERFLOW |
                           _MM_MASK_DIV_ZERO));
#endif

  if (frame_count == 0) {
    start_time = sdtrack::Tic();
  }
  frame_count++;

  // ???
  prev_delta_t_ba = tracker.t_ba() * prev_t_ba.inverse();

  // If this is a keyframe, set it as one on the tracker.
  if (is_prev_keyframe) {
    prev_t_ba = Sophus::SE3d();
  } else {
    prev_t_ba = tracker.t_ba();
  }

  // Add a pose to the poses array
  if (is_prev_keyframe) {
    std::shared_ptr<sdtrack::TrackerPose> new_pose(new sdtrack::TrackerPose);
    if (poses.size() > 0) {
      // Use information form the previous pose to initialize the new one
      new_pose->t_wp = poses.back()->t_wp * last_t_ba.inverse();
      if (use_imu_measurements && has_imu) {
        new_pose->v_w = poses.back()->v_w;
        new_pose->b = poses.back()->b;
      }
    } else {
      // First pose, align roll and pich to IMU, velocity and bias to zero
      // The initial pose is aligned the gravity direction
      if (has_imu && use_imu_measurements && imu_buffer.elements.size() > 0) {
        Eigen::Vector3t down = -imu_buffer.elements.front().a.normalized();
        VLOG(1) << "Down vector based on first imu meas: " <<
                   down.transpose() ;

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
      // Set the initial velocity and bias.
      new_pose->v_w.setZero();
      new_pose->b.setZero();
      //       new_pose->b << 0.00209809 , 0.00167743, -7.46213e-05 ,
      //           0.151629 ,0.0224114, 0.826392;
    }

    // Add new pose to global poses array.
    {
      std::unique_lock<std::mutex>(aac_mutex);
      new_pose->cam_params = rig.cameras_[0]->GetParams();
      poses.push_back(new_pose);
    }

    // Add pose to GUI
    axes.push_back(std::unique_ptr<SceneGraph::GLAxis>(
                     new SceneGraph::GLAxis(0.2)));
    gui_vars.scene_graph.AddChild(axes.back().get());
  }

  // Set the timestamp of the latest pose to this image's timestamp.
  poses.back()->time = timestamp + imu_time_offset;

  if(poses.size() > 1 && poses.back()->time <= poses[poses.size()-2]->time){
    LOG(ERROR) << "Pose timestamp same or decreasing: pose " << poses.size()-1 << " time: "
               << poses.back()->time <<
                  " <= pose " << poses.size()-2 << " time: " << poses[poses.size()-2]->time
                  ;
    exit(EXIT_FAILURE);
  }

  double track_ratio = (double)tracker.num_successful_tracks() / (double)num_features;
  if (track_ratio > 0.3) {
    guess = prev_delta_t_ba * prev_t_ba;
  } else {
    VLOG(2) << "Do not have good number of tracks "<<
               ", using Identity for guess."
               " Ratio: " << track_ratio
               ;
    guess = Sophus::SE3d();
  }


  // Perturb pose if the guess translation is zero.
  if(guess.translation() == Eigen::Vector3d(0,0,0) &&
     poses.size() > 1) {
    guess.translation() = Eigen::Vector3d(0, 0, 0.001);
  }

  if (has_imu &&
      use_imu_measurements &&
      use_imu_for_guess && poses.size() > min_poses_for_imu) {
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
      VLOG(2) << "Using IMU integration for guess." ;
      ba::ImuPoseT<Scalar>& last_pose = imu_poses.back();
      guess = last_pose.t_wp.inverse() *
          imu_poses.front().t_wp;
      pose2->t_wp = last_pose.t_wp;
      pose2->v_w = last_pose.v_w;
      poses.back()->t_wp = pose2->t_wp;
      poses.back()->v_w = pose2->v_w;
      poses.back()->b = pose2->b;
    }
  }

  //StreamMessage(selfcal_debug_level) << "Guess:\n " << guess.matrix() << std::endl;

  bool tracking_failed = false;
  {
    std::lock_guard<std::mutex> lock(aac_mutex);

    tracker.AddImage(images, guess);
    tracker.EvaluateTrackResiduals(0, tracker.GetImagePyramid(),
                                   tracker.GetCurrentTracks());

    if (!is_manual_mode) {
      tracker.OptimizeTracks(-1, optimize_landmarks, optimize_pose);
    }
    tracker.PruneTracks();

    if ((tracker.num_successful_tracks() < 10) &&
        has_imu && use_imu_measurements) {
      VLOG(2) << "Tracking failed. " <<
                 tracker.num_successful_tracks() <<
                 " successful tracks. Using guess." ;
      tracking_failed = true;
      tracker.set_t_ba(guess);
    }else if(tracker.num_successful_tracks() < 10){
      VLOG(2) << "Tracking failed. But no IMU data"
              << " so using tracker guess anyway." <<
                 std::endl;
    }

    // Update the pose based on the result from the tracker.
    UpdateCurrentPose();

    if (follow_camera) {
      FollowCamera(gui_vars, poses.back()->t_wp);
    }
  }

  if (do_keyframing) {
    const double track_ratio = (double)tracker.num_successful_tracks() /
        (double)keyframe_tracks;
    const double total_trans = tracker.t_ba().translation().norm();
    const double total_rot = tracker.t_ba().so3().log().norm();

    bool keyframe_condition = track_ratio < 0.8 || total_trans > 0.2 ||
        total_rot > 0.1 /*|| tracker.num_successful_tracks() < 64*/;

    VLOG(3) << "\tRatio: " << track_ratio << " trans: " << total_trans <<
               " rot: " << total_rot ;

    {
      std::lock_guard<std::mutex> lock(aac_mutex);

      if (keyframe_tracks != 0) {
        if (keyframe_condition) {
          is_keyframe = true;
        } else {
          is_keyframe = false;
        }
      }

      VLOG(2) << "is keyframe: " << is_keyframe ;

      prev_delta_t_ba = tracker.t_ba() * prev_t_ba.inverse();

      // If this is a keyframe, set it as one on the tracker.
      if (is_keyframe) {
        tracker.AddKeyframe();
      }
      is_prev_keyframe = is_keyframe;
    }
  } else {
    std::lock_guard<std::mutex> lock(aac_mutex);
    tracker.AddKeyframe();
  }

  VLOG(2) << "Num successful tracks: " << tracker.num_successful_tracks() <<
             " keyframe tracks: " << keyframe_tracks ;

  if (!is_manual_mode) {
    BaAndStartNewLandmarks();
  }

  // Check to see if any online calibrator should be plotted
  for(auto const &calib : calibrations){
    if (calib.second->plot_graphs && calib.second->do_self_cal){

      uint32_t num_params = 0;
      switch (calib.second->type){
      case Camera :
        num_params = rig.cameras_[0]->NumParams();
        if(calib.second->candidate_window.mean.rows() == 0){
          calib.second->candidate_window.mean = rig.cameras_[0]->GetParams();
        }

        for (size_t ii = 0; ii < num_params; ++ii) {
          if(has_gt){
            double gt_value = gt_cam_params[ii];
            plot_logs[ii].Log(rig.cameras_[0]->GetParams()[ii],
                calib.second->candidate_window.mean[ii], gt_value);
          }else{
            plot_logs[ii].Log(rig.cameras_[0]->GetParams()[ii],
                calib.second->candidate_window.mean[ii]);
          }
        }
        break;

      case IMU :
        num_params = Sophus::SE3t::DoF;
        if(calib.second->candidate_window.mean.rows() == 0){
          calib.second->candidate_window.mean = rig.cameras_[0]->Pose().log();
        }

        for (size_t ii = 0; ii < num_params; ++ii) {
          double pq_value = log_decoupled(UnrotatePose(rig.cameras_[0]->Pose())
              )[ii];
          double candidate_value = calib.second->candidate_window.mean[ii];
          if(has_gt){
            // plot the ground truth value for comparison
            double gt_value = gt_Tvs[ii];
            plot_logs[ii].Log(pq_value, candidate_value, gt_value);
          }else{
            plot_logs[ii].Log(pq_value, candidate_value);
          }
        }
        break;
      }

      analysis_logs[0].Log(calib.second->last_window_kl_divergence,
                           calib.second->last_added_window_kl_divergence);
      analysis_logs[1].Log(tracker.num_successful_tracks());
      analysis_logs[2].Log(total_last_frame_proj_norm);


      // Currently only plotting one self calibrator is supported.
      break;
    }
  }

  //  if (draw_plots) {
  //    const bool imu_plots_needed = ImuSelfCalActive();
  //    const uint32_t num_cam_params = do_cam_self_cal ?
  //          rig.cameras_[0]->NumParams(): 0;
  //    if (candidate_window.mean.rows() == 0) {
  //      if(do_cam_self_cal){
  //        candidate_window.mean = rig.cameras_[0]->GetParams();
  //      }else if (ImuSelfCalActive()){
  //        candidate_window.mean = rig.cameras_[0]->Pose().log();
  //      }
  //    }

  //    for (size_t ii = 0; ii < num_cam_params; ++ii) {
  //      plot_logs[ii].Log(rig.cameras_[0]->GetParams()[ii],
  ////         old_rig.cameras[0].camera.GenericParams()[ii],
  //          candidate_window.mean[ii]);
  //    }

  //    if (imu_plots_needed) {

  //      Eigen::Vector6d error =
  //          (rig.cameras_[0]->Pose().inverse() * rig.cameras_[0]->Pose()).log();

  //      for (size_t ii = num_cam_params; ii < num_cam_params + 6; ++ii) {
  //        plot_logs[ii].Log(rig.cameras_[0]->Pose().log()[ii - num_cam_params],
  //            candidate_window.mean[ii]);
  //      }
  ////      Eigen::Vector6d error =
  ////          (rig.t_wc_[0].inverse() * old_rig.cameras[0].T_wc).log();
  ////      for (size_t ii = num_cam_params; ii < num_cam_params + 6; ii++) {
  ////        plot_logs[ii].Log(error[ii - num_cam_params]);

  //    }

  //    analysis_logs[0].Log(last_window_kl_divergence,
  //                         last_added_window_kl_divergence);
  //    analysis_logs[1].Log(tracker.num_successful_tracks());
  //    analysis_logs[2].Log(total_last_frame_proj_norm);
  //  }

  if (is_keyframe) {
    VLOG(1) << "KEYFRAME." ;
    keyframe_tracks = tracker.GetCurrentTracks().size();
    VLOG(1) << "New keyframe tracks: " << keyframe_tracks ;
  } else {
    VLOG(1) << "NOT KEYFRAME." << std::endl;
  }

  current_tracks = &tracker.GetCurrentTracks();

#ifdef CHECK_NANS
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() |
                         (_MM_MASK_INVALID | _MM_MASK_OVERFLOW |
                          _MM_MASK_DIV_ZERO));
#endif

  //  StreamMessage(selfcal_debug_level) << "FRAME : " << frame_count << " KEYFRAME: " << poses.size() <<
  //                                        " FPS: " << frame_count / sdtrack::Toc(start_time) << std::endl;
}

void DrawImageData(uint32_t cam_id)
{
  if (cam_id == 0) {
    gui_vars.handler->track_centers.clear();
  }

  SceneGraph::AxisAlignedBoundingBox aabb;
  line_strip->Clear();
  for (uint32_t ii = 0; ii < poses.size() ; ++ii) {
    axes[ii]->SetPose(poses[ii]->t_wp.matrix());
    aabb.Insert(poses[ii]->t_wp.translation());
    Eigen::Vector3f vertex = poses[ii]->t_wp.translation().cast<float>();
    line_strip->AddVertex(vertex);
  }
  gui_vars.grid.set_bounds(aabb);

  // Draw the tracks
  for (std::shared_ptr<sdtrack::DenseTrack>& track : *current_tracks) {
    Eigen::Vector2d center;
    if (track->keypoints.back()[cam_id].tracked ||
        track->keypoints.size() <= 2 ) {
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

  for (size_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
    gui_vars.camera_view[cam_id]->RenderChildren();
  }
}

///////////////////////////////////////////////////////////////////////////
void Run()
{
  std::vector<pangolin::GlTexture> gl_tex;

  // pangolin::Timer timer;
  bool capture_success = false;
  std::shared_ptr<hal::ImageArray> images = hal::ImageArray::Create();
  camera_device.Capture(*images);
  while(!pangolin::ShouldQuit()) {
    capture_success = false;
    const bool go = is_stepping;
    if (!is_running) {
      is_stepping = false;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor4f(1.0f,1.0f,1.0f,1.0f);

    if (go) {

      if (has_imu && use_imu_measurements &&
          imu_buffer.elements.size() == 0) {
        // Capture an image so we have some IMU data.
        std::shared_ptr<hal::ImageArray> img = hal::ImageArray::Create();
        while (imu_buffer.elements.size() == 0) {
          camera_device.Capture(*img);
        }
      }

      capture_success = camera_device.Capture(*images);
    }

    if (capture_success) {
      double timestamp = use_system_time ? images->Ref().system_time() :
                                           images->Ref().device_time();

      //      std::cerr << std::setprecision(6) << std::fixed <<
      //                   "image device time: " << images->Ref().device_time()
      //                << std::endl << " image system time: " << images->Ref().system_time()
      //                << std::endl;

      // Wait until we have enough measurements to interpolate this frame's
      // timestamp
      if (has_imu && use_imu_measurements) {
        const double start_time = sdtrack::Tic();
        while (imu_buffer.end_time < timestamp &&
               sdtrack::Toc(start_time) < 0.1) {
          usleep(10);
        }
      }

      gl_tex.resize(images->Size());

      for (int cam_id = 0 ; cam_id < images->Size() ; ++cam_id) {
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
      for (int ii = 0; ii < images->Size() ; ++ii) {
        cvmat_images.push_back(images->at(ii)->Mat());
      }


      ProcessImage(cvmat_images, timestamp);

    }

    if (camera_img && camera_img->data()) {
      for (size_t cam_id = 0 ; cam_id < rig.cameras_.size() &&
           cam_id < (uint32_t)images->Size(); ++cam_id) {
        camera_img = images->at(cam_id);
        gui_vars.camera_view[cam_id]->ActivateAndScissor();
        gl_tex[cam_id].Upload(camera_img->data(), camera_img->Format(),
                              camera_img->Type());
        gl_tex[cam_id].RenderToViewportFlipY();
        DrawImageData(cam_id);
      }

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
    }
    pangolin::FinishFrame();
  }
}

///////////////////////////////////////////////////////////////////////////
void InitGui() {

  InitTrackerGui(gui_vars, window_width, window_height , image_width,
                 image_height, rig.cameras_.size());
  line_strip.reset(new SceneGraph::GLPrimitives<>);
  gui_vars.scene_graph.AddChild(line_strip.get());

  pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT,
        [&]() {
    is_stepping = true;
  });

  pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_CTRL + 's',
        [&]() {
    // write all the poses to a file.
    std::ofstream pose_file("poses.txt", std::ios_base::trunc);
    for (auto pose : poses) {
      pose_file << pose->t_wp.translation().transpose().format(
                     sdtrack::kLongCsvFmt) << std::endl;
    }

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
        lm_file << ray.transpose().format(sdtrack::kLongCsvFmt) << std::endl;
      }
    }
  });

  pangolin::RegisterKeyPressCallback('r', [&]() {
    for (uint32_t ii = 0; ii < plot_views.size(); ++ii) {
      plot_views[ii]->Keyboard(*plot_views[ii], 'a', 0, 0, true);
    }

    for (uint32_t ii = 0; ii < analysis_views.size(); ++ii) {
      analysis_views[ii]->Keyboard(*analysis_views[ii], 'a', 0, 0, true);
    }
  });

  pangolin::RegisterKeyPressCallback(' ', [&]() {
    is_running = !is_running;
  });

  pangolin::RegisterKeyPressCallback('f', [&]() {
    follow_camera = !follow_camera;
  });

  pangolin::RegisterKeyPressCallback('c', [&]() {
    do_cam_self_cal = !do_cam_self_cal;
  });

  pangolin::RegisterKeyPressCallback('u', [&]() {
    GetCalibration(Camera)->unknown_calibration = true;
    unknown_cam_calibration = true;
    GetCalibration(Camera)->unknown_calibration_start_pose = -2;
    VLOG(1) << "Unknown camera calibration = true with start pose " <<
               GetCalibration(Camera)->unknown_calibration_start_pose
               ;
    GetCalibration(Camera)->online_calibrator.ClearQueue();
  });

  pangolin::RegisterKeyPressCallback('b', [&]() {
    last_optimization_level = 0;
    tracker.OptimizeTracks();
  });

  pangolin::RegisterKeyPressCallback('B', [&]() {
    do_bundle_adjustment = !do_bundle_adjustment;
    VLOG(1) << "Do BA:" << do_bundle_adjustment ;
  });

  pangolin::RegisterKeyPressCallback('k', [&]() {
    is_keyframe = !is_keyframe;
    VLOG(1) << "is_keyframe:" << is_keyframe ;
  });

  pangolin::RegisterKeyPressCallback('S', [&]() {
    do_start_new_landmarks = !do_start_new_landmarks;
    VLOG(1) << "Do SNL:" << do_start_new_landmarks ;
  });

  pangolin::RegisterKeyPressCallback('2', [&]() {
    last_optimization_level = 2;
    tracker.OptimizeTracks(last_optimization_level, optimize_landmarks,
                           optimize_pose);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('3', [&]() {
    last_optimization_level = 3;
    tracker.OptimizeTracks(last_optimization_level, optimize_landmarks,
                           optimize_pose);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('1', [&]() {
    last_optimization_level = 1;
    tracker.OptimizeTracks(last_optimization_level, optimize_landmarks,
                           optimize_pose);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('0', [&]() {
    last_optimization_level = 0;
    tracker.OptimizeTracks(last_optimization_level, optimize_landmarks,
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
    VLOG(1) << "optimize landmarks: " << optimize_landmarks ;
  });

  pangolin::RegisterKeyPressCallback('c', [&]() {
    optimize_pose = !optimize_pose;
    VLOG(1) << "optimize pose: " << optimize_pose ;
  });

  pangolin::RegisterKeyPressCallback('m', [&]() {
    is_manual_mode = !is_manual_mode;
    VLOG(1) << "Manual mode:" << is_manual_mode ;
  });

  // set up the plotters.
  if (do_cam_self_cal || do_imu_self_cal) {
    params_plot_view = &pangolin::Display("plot").SetLayout(
          pangolin::LayoutEqualVertical);
    pangolin::Display("multi").AddDisplay(*params_plot_view);

    const bool imu_plots_needed = has_imu && use_imu_measurements &&
        do_imu_self_cal;

    const uint32_t num_cam_params = do_cam_self_cal ?
          rig.cameras_[0]->NumParams() : 0;
    const uint32_t num_imu_params = imu_plots_needed ? Sophus::SE3t::DoF : 0;
    const uint32_t num_plots = num_cam_params + num_imu_params;

    plot_views.resize(num_plots);
    plot_logs.resize(num_plots);

    if(do_cam_self_cal){
      plot_logs[0].SetLabels({"fx - p.q.", "fx - candidate seg."});
      plot_logs[1].SetLabels({"fy - p.q.", "fy - candidate seg."});
      plot_logs[2].SetLabels({"cx - p.q.", "cx - candidate seg."});
      plot_logs[3].SetLabels({"cy - p.q.", "cy - candidate seg."});
      if (num_plots > 4){
        plot_logs[4].SetLabels({"w - p.q.", "w - candidate seg."});
      }
    }


    for (size_t ii = 0; ii < num_cam_params; ++ii) {
      plot_views[ii] = new pangolin::Plotter(&plot_logs[ii]);
      params_plot_view->AddDisplay(*plot_views[ii]);
      double param = rig.cameras_[0]->GetParams()[ii];
      pangolin::XYRange range(0, 500, param - param * 0.5,
                              param + param * 0.5);
      plot_views[ii]->SetDefaultView(range);
      plot_views[ii]->SetViewSmooth(range);
      plot_views[ii]->ToggleTracking();
    }


    // Add the t_vs displays.
    if (imu_plots_needed) {
      uint32_t index = num_cam_params;
      plot_logs[index++].SetLabels({"x - p.q.", "x - candidate seg."});
      plot_logs[index++].SetLabels({"y - p.q.", "y - candidate seg."});
      plot_logs[index++].SetLabels({"z - p.q.", "z - candidate seg."});
      plot_logs[index++].SetLabels({"p - p.q.", "p - candidate seg."});
      plot_logs[index++].SetLabels({"q - p.q.", "q - candidate seg."});
      plot_logs[index].SetLabels({"r - p.q.", "r - candidate seg."});

      imu_plot_view = &pangolin::Display("imu_plot").SetLayout(
            pangolin::LayoutEqualVertical);
      pangolin::Display("multi").AddDisplay(*imu_plot_view);

      for (size_t ii = num_cam_params; ii < 6 + num_cam_params; ii++) {
        plot_views[ii] = new pangolin::Plotter(&plot_logs[ii]);
        imu_plot_view->AddDisplay(*plot_views[ii]);
        pangolin::XYRange range(0, 500, -5, 5);
        plot_views[ii]->SetDefaultView(range);
        plot_views[ii]->SetViewSmooth(range);
        plot_views[ii]->ToggleTracking();
      }
    }

    analysis_plot_view = &pangolin::Display("analysis_plot").SetLayout(
          pangolin::LayoutEqualVertical);
    pangolin::Display("multi").AddDisplay(*analysis_plot_view);

    analysis_views.resize(3);
    analysis_logs.resize(3);

    analysis_logs[0].SetLabels({"p-value (candidate seg.)",
                                "p-value (last p.q. window)"});
    analysis_logs[1].SetLabels({"num. successful tracks"});
    analysis_logs[2].SetLabels({"last frame mean reproj. error"});

    for (size_t ii = 0; ii < analysis_views.size(); ++ii) {
      analysis_views[ii] = new pangolin::Plotter(&analysis_logs[ii]);
      analysis_plot_view->AddDisplay(*analysis_views[ii]);
      analysis_views[ii]->ToggleTracking();
    }
  }
}
#include <calibu/cam/camera_crtp.h>
#include <calibu/cam/camera_crtp_impl.h>
#include <calibu/cam/camera_models_crtp.h>

///////////////////////////////////////////////////////////////////////////
bool LoadCameras(GetPot& cl)
{
  LoadCameraAndRig(cl, camera_device, rig);

  for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {

    // Try to use shallow copy constructor
    //std::remove_reference<decltype(*rig.cameras_[0])>::type cam;

    // Create a new camera for each rig, copying over the pose and params
    // from the rig that was read in from the XML.
    std::shared_ptr<calibu::CameraInterface<Scalar>> cam_ptr(
          new calibu::FovCamera<Scalar>());

    cam_ptr->SetPose(rig.cameras_[0]->Pose());
    cam_ptr->SetParams(rig.cameras_[0]->GetParams());
    selfcal_rig.AddCamera(cam_ptr);

    cam_ptr.reset(new calibu::FovCamera<Scalar>());
    cam_ptr->SetPose(rig.cameras_[0]->Pose());
    cam_ptr->SetParams(rig.cameras_[0]->GetParams());

    aac_rig.AddCamera(cam_ptr);
  }

  // Load the imu
  std::string imu_str = cl.follow("","-imu");
  if (!imu_str.empty()) {
    try {
      imu_device = hal::IMU(imu_str);
    } catch (hal::DeviceException& e) {
      LOG(ERROR) << "Error loading imu device: " << e.what()
                 << " ... proceeding without.";
    }
    has_imu = true;
    imu_device.RegisterIMUDataCallback(&ImuCallback);
  }

  // If we require self-calibration from an unknown initial calibration, then
  // perturb the values (camera calibraiton parameters only)
  if (unknown_cam_calibration) {
    Eigen::VectorXd params = rig.cameras_[0]->GetParams();
    if(has_gt){
      gt_cam_params = params;
    }
    // fov in rads.
    const double fov_rads = 90 * M_PI / 180.0;
    const double f_x =
        0.5 * rig.cameras_[0]->Height() / tan(fov_rads / 2);
    VLOG(1) << "Changing fx from " << params[0] << " to " << f_x ;
    VLOG(1) << "Changing fy from " << params[1] << " to " << f_x ;
    params[0] = f_x;
    params[1] = f_x;
    params[2] = rig.cameras_[0]->Width() / 2;
    params[3] = rig.cameras_[0]->Height() / 2;
    if (params.rows() > 4) {
      params[4] = 1.0;
    }

    rig.cameras_[0]->SetParams(params);
    selfcal_rig.cameras_[0]->SetParams(params);
    aac_rig.cameras_[0]->SetParams(params);

    // Add a marker in the batch file for this initial, unknown calibration.
    Eigen::VectorXd initial_covariance(params.rows());
    initial_covariance.setOnes();
    std::ofstream("batch.txt", std::ios_base::app) << 0 << ", " <<
                                                      initial_covariance.transpose().format(sdtrack::kLongCsvFmt) <<
                                                      ", " << 0 << ", " << params.transpose().format(sdtrack::kLongCsvFmt) <<
                                                      std::endl;
  }

  if (has_imu && unknown_imu_calibration && use_imu_measurements) {

    Sophus::SE3t Tvs = rig.cameras_[0]->Pose();

    // If IMU calibration is unknown, perturb the IMU rotation
    VLOG(1) << "IMU XML Calibration log: " <<
               log_decoupled(UnrotatePose(Tvs)).transpose();


    if(has_gt){
      gt_Tvs = log_decoupled(UnrotatePose(Tvs));
      VLOG(1) << "gt_Tvs: " << gt_Tvs.transpose();
    }

    Sophus::SO3d old_rot = UnrotatePose(Tvs).so3();
    Sophus::SO3d new_rot = Sophus::SO3d::exp(
          (Eigen::Vector3d() << 0.1, 0.2, 0.3).finished());
    VLOG(1) << "Changing rotation from: [ " << old_rot.matrix().eulerAngles
               (0,1,2).transpose() << " ] to [ "<<  new_rot.matrix().eulerAngles
               (0,1,2).transpose() << " ]";

    Tvs.so3() = new_rot;
    Tvs = sdtrack::RotatePose(Tvs);

    // Set translation to a perturbed initial value, close to zero.
    Eigen::Vector3d new_translation(Eigen::Vector3d::Zero());
    // experimental initialization of the translation component.
    new_translation << 0.01, 0.01, 0.01;
    VLOG(1) << "Changing translation from: [ " << Tvs.translation().transpose()
            << " ] to [ "<<  new_translation.transpose() << " ]";
    Tvs.translation() = new_translation;

    rig.cameras_[0]->SetPose(Tvs);
    VLOG(1) << "Unknown IMU calibration, using:"
            << sdtrack::UnrotatePose(rig.cameras_[0]->Pose());

    if(has_gt){
      // write initial pq error to log file
      Eigen::VectorXd calib_comparison =
          CompareWithGt(false, true,
                        log_decoupled(UnrotatePose(rig.cameras_[0]->Pose())));
      VLOG(1) << "initial tvs error: " << calib_comparison.transpose();
      VLOG(1) << "initial tvs error norm: " << calib_comparison.norm();
    }

    for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
      selfcal_rig.cameras_[cam_id]->SetPose(rig.cameras_[cam_id]->Pose());
      aac_rig.cameras_[cam_id]->SetPose(aac_rig.cameras_[cam_id]->Pose());
    }

  }else if(has_imu && use_imu_measurements){
    VLOG(1) << "Using IMU calibration: "
            << sdtrack::UnrotatePose(rig.cameras_[0]->Pose());
  }

  return true;
}



///////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  //    Sophus::SO3t R_1(M_PI, M_PI_4, M_PI_2);
  //    Sophus::SO3t R_2(M_PI_2, M_PI, M_PI_4);

  //    Eigen::Vector3d lie_R2= Sophus::SO3d::log(R_2);


  //    if((R_1.matrix()*R_2.matrix()).isApprox(
  //      Sophus::SO3d::exp(R_1.Adj()*lie_R2).matrix() * R_1.matrix())){
  //      StreamMessage(selfcal_debug_level) << "the adjoint works as expected..." << std::endl;

  //    }else{
  //      StreamMessage(selfcal_debug_level) << "it did not work..." << std::endl;
  //    }

  //    Sophus::SE3t T(Sophus::SO3t(M_PI, M_PI_4, M_PI_2),
  //                   Eigen::Vector3d::Zero());

  //    Eigen::AngleAxisd aaZ(M_PI_2, Eigen::Vector3d::UnitZ());
  //    Eigen::AngleAxisd aaY(M_PI_4, Eigen::Vector3d::UnitY());
  //    Eigen::AngleAxisd aaX(M_PI, Eigen::Vector3d::UnitX());
  //    Eigen::Quaterniond initial_rotation = aaX * aaY * aaZ;
  //    Sophus::SE3d pose(initial_rotation,
  //                      Eigen::Vector3d::Zero());

  //    StreamMessage(selfcal_debug_level) << "pose: " << std::endl << pose.matrix()
  //              << std::endl;
  //    StreamMessage(selfcal_debug_level) << "pose euler angles: \n" << pose.rotationMatrix().eulerAngles(0,1,2) << std::endl;


  //    StreamMessage(selfcal_debug_level) << "T: " << std::endl << T.matrix() << std::endl;
  //    StreamMessage(selfcal_debug_level) << "T euler angles: \n" << T.rotationMatrix().eulerAngles(0,1,2) << std::endl;

  //  exit(1);

  google::InitGoogleLogging(argv[0]);

  // Clear the log files.
  {
    std::ofstream sigmas_file("sigmas.txt", std::ios_base::trunc);
    std::ofstream imu_pq_file("imu_pq.txt", std::ios_base::trunc);
    std::ofstream imu_pq_error_file("imu_pq_error.txt", std::ios_base::trunc);
    std::ofstream imu_candidate_file("imu_candidate.txt", std::ios_base::trunc);
    std::ofstream cam_pq_file("cam_pq.txt", std::ios_base::trunc);
    std::ofstream cam_candidate_file("cam_candidate.txt", std::ios_base::trunc);
    std::ofstream batch_file("batch.txt", std::ios_base::trunc);
    std::ofstream pose_file("timings.txt", std::ios_base::trunc);

  }
  srand(0);
  GetPot cl(argc, argv);
  if (cl.search("--help")) {
    LOG(INFO) << g_usage;
    exit(-1);
  }

  if (cl.search("-use_system_time")) {
    LOG(INFO) << "Using system time.";
    use_system_time = true;
  }

  if (cl.search("-startnow")) {
    LOG(INFO) << "Starting now.";
    is_running = true;
  }

  if(cl.search("-gt")){
    // Use the XML file as ground truth for calibraiton parameters
    LOG(INFO) << "Using XML file parameters as ground truth";
    has_gt = true;
  }


  VLOG(1) << "Initializing camera..." ;
  LoadCameras(cl);


  pyramid_levels = 3;
  patch_size = 7;
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
  tracker_options.dense_ncc_threshold = ncc_threshold;
  tracker_options.harris_score_threshold = 2e6;
  tracker_options.gn_scaling = 1.0;
  tracker.Initialize(keypoint_options, tracker_options, &rig);



  // Initialize the online calibration component.
  Eigen::VectorXd camera_weights(rig.cameras_[0]->NumParams());
  if (camera_weights.rows() > 4) {
    camera_weights << 1.0, 1.0, 1.7, 1.7, 320000;
  } else {
    camera_weights << 1.0, 1.0, 1.7, 1.7;
  }

  // TOOD: Run a large optimization to find out what the scaling weights for
  // the IMU transform are
  Eigen::VectorXd imu_weights(6);
  imu_weights << 500, 500, 5000, 500000, 500000, 500000;
  //imu_weights << 1,1,1,1,1,1;

  Eigen::VectorXd batch_weights(rig.cameras_[0]->NumParams() + Sophus::SE3d::DoF);
  batch_weights << camera_weights, imu_weights;


  InitGui();

  // Initialize camera calibration (camera intrisnsics)
  std::shared_ptr<Calibration> cam_calib(new Calibration);
  cam_calib->type = CalibrationType::Camera;
  cam_calib->num_self_cal_segments = 5;
  cam_calib->do_self_cal = do_cam_self_cal;
  cam_calib->self_cal_segment_length = min_poses_for_camera;
  cam_calib->unknown_calibration = unknown_cam_calibration;
  cam_calib->plot_graphs = true;
  cam_calib->online_calibrator.Init
      (&aac_mutex, &online_calibrator_mutex,
       &selfcal_rig, cam_calib->num_self_cal_segments,
       cam_calib->self_cal_segment_length, camera_weights,
       imu_time_offset, &imu_buffer);

  // start the async priority queue thread
  cam_calib->online_calibrator.pq_thread
      = std::shared_ptr<std::thread>(
        new std::thread(&sdtrack::OnlineCalibrator::DoPriorityQueueThread,
                        &(cam_calib->online_calibrator)));


  calibrations[Camera] = cam_calib;


  // Initialize camera-inertial calibration (camera to imu: Tvs)
  std::shared_ptr<Calibration> imu_calib(new Calibration);
  imu_calib->type = CalibrationType::IMU;
  imu_calib->num_self_cal_segments = 5;
  imu_calib->do_self_cal = do_imu_self_cal;
  imu_calib->self_cal_segment_length = min_poses_for_imu;
  imu_calib->unknown_calibration = unknown_imu_calibration;
  imu_calib->plot_graphs = true;
  imu_calib->online_calibrator.Init
      (&aac_mutex, &online_calibrator_mutex,
       &selfcal_rig, imu_calib->num_self_cal_segments,
       imu_calib->self_cal_segment_length, imu_weights,
       imu_time_offset, &imu_buffer);

  // Set priority queue parameters
  std::shared_ptr<sdtrack::PriorityQueueParams> params =
      imu_calib->online_calibrator.PriorityQueueParameters();
  params->do_tvs = true;
  params->use_imu = true;
  params->callback = &ApplyPriorityQueueResults;



  // start the async priority queue thread
  imu_calib->online_calibrator.pq_thread
      = std::shared_ptr<std::thread>(
        new std::thread(&sdtrack::OnlineCalibrator::DoPriorityQueueThread,
                        &(imu_calib->online_calibrator)));


  calibrations[IMU] = imu_calib;


  // Initialize batch calibration queue (camera intrinsics + Tvs)
  std::shared_ptr<Calibration> batch_calib(new Calibration);
  batch_calib->type = CalibrationType::Batch;
  // Only used for the initial joint batch solution. Should not run a
  // priority queue.
  batch_calib->do_self_cal = false;
  batch_calib->num_self_cal_segments = 10;
  batch_calib->self_cal_segment_length = min_poses_for_camera;
  batch_calib->online_calibrator.Init
      (&aac_mutex, &online_calibrator_mutex,
       &selfcal_rig, batch_calib->num_self_cal_segments,
       batch_calib->self_cal_segment_length, batch_weights,
       imu_time_offset, &imu_buffer);
  calibrations[Batch] = batch_calib;

  //imu_time_offset = -0.0697;

  aac_thread = std::shared_ptr<std::thread>(new std::thread(&DoAAC));

  Run();

  return 0;
}
