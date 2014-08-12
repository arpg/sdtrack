// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.
#undef NDEBUG
#include <assert.h>
#include <vector>
#include <HAL/Camera/CameraDevice.h>
#include <miniglog/logging.h>
#include <calibu/utils/Xml.h>
// #include <Utils/MathTypes.h>
// #include <Utils/PrintMatrix.h>
// #include <Utils/Utils.h>
#include "GetPot"
#include <sdtrack/TicToc.h>
// #include <HAL/Utils/TicToc.h>
#include <unistd.h>
#include <SceneGraph/SceneGraph.h>
#include <pangolin/pangolin.h>
#include <ba/BundleAdjuster.h>
#include "CVars/CVar.h"
#include <sdtrack/utils.h>
#include "math_types.h"
#include "gui_common.h"
#include "etc_common.h"
#include "common_cvars.h"
#ifdef CHECK_NANS
#include <xmmintrin.h>
#endif

#include <sdtrack/semi_dense_tracker.h>

#include "online_calibrator.h"


static uint32_t& num_ba_poses =
    CVarUtils::CreateCVar<>("sd.NumBAPoses",10u, "");
static bool& do_keyframing =
    CVarUtils::CreateCVar<>("sd.DoKeyframing", true, "");
static int& num_ba_iterations =
    CVarUtils::CreateCVar<>("sd.NumBAIterations", 200, "");
static bool& reset_outliers =
    CVarUtils::CreateCVar<>("sd.ResetOutliers", false, "");
static bool& use_dogleg =
    CVarUtils::CreateCVar<>("sd.UseDogleg", true, "");
static bool& use_robust_norm_for_proj =
    CVarUtils::CreateCVar<>("sd.UseRobustNormForProj", true, "");
static double& outlier_threshold =
    CVarUtils::CreateCVar<>("sd.OutlierThreshold", 1.0, "");
static bool& do_outlier_rejection =
    CVarUtils::CreateCVar<>("sd.DoOutlierRejection", true, "");
static int& pyramid_levels =
    CVarUtils::CreateCVar<>("sd.PyramidLevels", 3, "");
static int& patch_size =
    CVarUtils::CreateCVar<>("sd.PatchSize", 7, "");
static int& num_features =
    CVarUtils::CreateCVar<>("sd.NumFeatures",128, "");
static int& feature_cells =
    CVarUtils::CreateCVar<>("sd.FeatureCells",8, "");
static double& ncc_threshold =
    CVarUtils::CreateCVar<>("sd.NCCThreshold", 0.875, "");


uint32_t keyframe_tracks = UINT_MAX;
double start_time = 0;
uint32_t frame_count = 0;
Sophus::SE3d last_t_ba, prev_delta_t_ba, prev_t_ba;
// Self calibration params
bool unknown_calibration = true;
bool compare_self_cal_with_batch = false;
bool do_self_cal = true;
uint32_t num_self_cal_segments = 5;
uint32_t self_cal_segment_length = 10;

const int window_width = 640;
const int window_height = 480;
std::string g_usage = "SD SELFCAL. Example usage:\n"
    "-cam file:[loop=1]///Path/To/Dataset/[left,right]*pgm -cmod cameras.xml";
bool is_keyframe = true, is_prev_keyframe = true;
bool optimize_landmarks = true;
bool is_running = false;
bool is_stepping = false;
bool is_manual_mode = false;
bool do_bundle_adjustment = true;
bool do_start_new_landmarks = true;
int image_width;
int image_height;
calibu::CameraRigT<Scalar> old_rig;
calibu::Rig<Scalar> rig;
hal::Camera camera_device;
sdtrack::SemiDenseTracker tracker;

sdtrack::OnlineCalibrator online_calib;
std::vector<pangolin::DataLog> plot_logs;
std::vector<pangolin::Plotter*> plot_views;

TrackerGuiVars gui_vars;
std::shared_ptr<GetPot> cl;

// TrackCenterMap current_track_centers;
std::list<std::shared_ptr<sdtrack::DenseTrack>>* current_tracks = nullptr;
int last_optimization_level = 0;
// std::shared_ptr<sdtrack::DenseTrack> selected_track = nullptr;
std::shared_ptr<pb::Image> camera_img;
std::vector<std::vector<std::shared_ptr<SceneGraph::ImageView>>> patches;
std::vector<std::shared_ptr<sdtrack::TrackerPose>> poses;
std::vector<std::unique_ptr<SceneGraph::GLAxis> > axes;
ba::BundleAdjuster<double, 1, 6, 0> bundle_adjuster;

// State variables
std::vector<cv::KeyPoint> keypoints;

void DoBundleAdjustment(uint32_t num_active_poses, uint32_t id)
{
  if (reset_outliers) {
    for (std::shared_ptr<sdtrack::TrackerPose> pose : poses) {
      for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
        track->is_outlier = false;
      }
    }
    reset_outliers = false;
  }

  ba::Options<double> options;
  options.use_dogleg = use_dogleg;
  options.use_sparse_solver = true;
  options.param_change_threshold = 1e-10;
  options.error_change_threshold = 1e-3;
  options.use_robust_norm_for_proj_residuals = use_robust_norm_for_proj;
  options.projection_outlier_threshold = outlier_threshold;
  // options.error_change_threshold = 1e-3;
  // options.use_sparse_solver = false;
  uint32_t num_outliers = 0;
  Sophus::SE3d t_ba;
  uint32_t start_active_pose, start_pose;

  GetBaPoseRange(poses, num_active_poses, start_pose, start_active_pose);

  if (start_pose == poses.size()) {
    return;
  }

  bool all_poses_active = start_active_pose == start_pose;

  // Do a bundle adjustment on the current set
  if (current_tracks && poses.size() > 1) {
    std::shared_ptr<sdtrack::TrackerPose> last_pose = poses.back();
    bundle_adjuster.Init(options, poses.size(),
                         current_tracks->size() * poses.size());
    for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
      bundle_adjuster.AddCamera(rig.cameras_[cam_id], rig.t_wc_[cam_id]);
    }

    // First add all the poses and landmarks to ba.
    for (uint32_t ii = start_pose ; ii < poses.size() ; ++ii) {
      std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
      pose->opt_id[id] = bundle_adjuster.AddPose(
            pose->t_wp, ii >= start_active_pose );
      for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
        const bool constrains_active =
            track->keypoints.size() + ii > start_active_pose;
        if (track->num_good_tracked_frames == 1 || track->is_outlier ||
            !constrains_active) {
          track->external_id[id] = UINT_MAX;
          continue;
        }
        Eigen::Vector4d ray;
        ray.head<3>() = track->ref_keypoint.ray;
        ray[3] = track->ref_keypoint.rho;
        ray = sdtrack::MultHomogeneous(pose->t_wp  * rig.t_wc_[0], ray);
        bool active = track->id != tracker.longest_track_id() ||
            !all_poses_active;
        if (!active) {
          std::cerr << "Landmark " << track->id << " inactive. " << std::endl;
        }
        track->external_id[id] =
            bundle_adjuster.AddLandmark(ray, pose->opt_id[id], 0, active);
      }
    }

    // Now add all reprojections to ba)
    for (uint32_t ii = start_pose ; ii < poses.size() ; ++ii) {
      std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
      for (std::shared_ptr<sdtrack::DenseTrack> track : pose->tracks) {
        if (track->external_id[id] == UINT_MAX) {
          continue;
        }
        for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
          for (size_t jj = 0; jj < track->keypoints.size() ; ++jj) {
            if (track->keypoints[jj][cam_id].tracked) {
              const Eigen::Vector2d& z = track->keypoints[jj][cam_id].kp;
              bundle_adjuster.AddProjectionResidual(
                z, pose->opt_id[id] + jj, track->external_id[id], cam_id);
            }
          }
        }
      }
    }

    // Optimize the poses
    bundle_adjuster.Solve(num_ba_iterations);

    // Get the pose of the last pose. This is used to calculate the relative
    // transform from the pose to the current pose.
    last_pose->t_wp = bundle_adjuster.GetPose(last_pose->opt_id[id]).t_wp;
    // std::cerr << "last pose t_wp: " << std::endl << last_pose->t_wp.matrix() <<
    //              std::endl;

    // Read out the pose and landmark values.
    for (uint32_t ii = start_pose ; ii < poses.size() ; ++ii) {
      std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
      const ba::PoseT<double>& ba_pose =
          bundle_adjuster.GetPose(pose->opt_id[id]);

      pose->t_wp = ba_pose.t_wp;
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
            bundle_adjuster.GetLandmark(track->external_id[id]);
        double ratio =
            bundle_adjuster.LandmarkOutlierRatio(track->external_id[id]);
        auto landmark =
            bundle_adjuster.GetLandmarkObj(track->external_id[id]);

        if (do_outlier_rejection) {
          if (ratio > 0.3 &&
              ((track->keypoints.size() == num_ba_poses - 1) ||
               track->tracked == false)) {
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
              (pose->t_wp * rig.t_wc_[0]).inverse(), x_w);
        // Normalize the xyz component of the ray to compare to the original
        // ray.
        x_r /= x_r.head<3>().norm();
        track->ref_keypoint.rho = x_r[3];
      }
    }

  }
  std::cerr << "Rejected " << num_outliers << " outliers." << std::endl;
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

void BaAndStartNewLandmarks()
{
  if (!is_keyframe) {
    return;
  }

  uint32_t keyframe_id = poses.size();
  if (do_self_cal && unknown_calibration && poses.size() > 2) {
    sdtrack::CalibrationWindow batch_window;
    online_calib.AnalyzeCalibrationWindow(poses, current_tracks,
          0, poses.size(), batch_window, 50, true);
    for (uint32_t ii = 0 ; ii < poses.size() ; ++ii) {
      for (std::shared_ptr<sdtrack::DenseTrack> track: poses[ii]->tracks) {
        if (track->external_id[0] == UINT_MAX) {
          continue;
        }
        // We also have to backproject this track again.
        tracker.BackProjectTrack(track);
      }
    }

    const double score =
        online_calib.GetWindowScore(batch_window);
    // Write this to the batch file.
    std::ofstream("batch.txt", std::ios_base::app) << keyframe_id << ", " <<
      batch_window.covariance.diagonal().transpose().format(
      sdtrack::kLongCsvFmt) << ", " << score << ", " <<
      batch_window.mean.transpose().format(sdtrack::kLongCsvFmt) << std::endl;

    std::cerr << "Batch means are: " << batch_window.mean.transpose() <<
                 std::endl;
    std::cerr << "Batch sigmas are:\n" <<
                 batch_window.covariance << std::endl;
    std::cerr << "Batch score: " << score << std::endl;

    // If the determinant is smaller than a heuristic, switch to self_cal.
    if (score < 1e5 && score != 0) {
      std::cerr << "Determinant small enough, switching to self-cal" <<
                   std::endl;
      unknown_calibration = false;
    }
  }

  if (do_bundle_adjustment) {
    DoBundleAdjustment(10, 0);

    if (do_self_cal && poses.size() >= self_cal_segment_length) {
      uint32_t start_pose =
          std::max(0, (int)poses.size() - (int)self_cal_segment_length);
      uint32_t end_pose = poses.size();
      sdtrack::CalibrationWindow window;
      online_calib.AnalyzeCalibrationWindow(poses, current_tracks,
                                            start_pose, end_pose, window, 20);

      std::ofstream("sigmas.txt", std::ios_base::app) << keyframe_id << ", " <<
        window.covariance.diagonal(). transpose().format(
          sdtrack::kLongCsvFmt) << ", " << window.score << std::endl;

      // Now potentially add this to the priority queue.
      bool added = online_calib.AnalyzeCalibrationWindow(window);

      // If the priority queue was modified, calculate the new results for it.
      if (added) {
        sdtrack::CalibrationWindow pq_window;
        bool apply_results = !unknown_calibration;
        online_calib.AnalyzePriorityQueue(poses, current_tracks,
                                          pq_window, 50, apply_results);
        if (apply_results) {
          for (size_t ii = 0; ii < poses.size(); ++ii) {
            for (std::shared_ptr<sdtrack::DenseTrack> track : poses[ii]->tracks) {
              tracker.BackProjectTrack(track);
            }
          }
        }
        std::cerr << "Analyzed priority queue with mean " <<
                     pq_window.mean.transpose() << " and cov\n " <<
                     pq_window.covariance << std::endl;
        online_calib.SetPriorityQueueDistribution(pq_window.covariance,
                                                  pq_window.mean);


        const double score =
            online_calib.GetWindowScore(pq_window);

        // Write this to the pq file.
        std::ofstream("pq.txt", std::ios_base::app) << keyframe_id << ", " <<
          pq_window.covariance.diagonal().transpose().format(
            sdtrack::kLongCsvFmt) << ", " << score << ", " <<
          pq_window.mean.transpose().format(sdtrack::kLongCsvFmt) << std::endl;

        if (compare_self_cal_with_batch && !unknown_calibration) {
          // Also analyze the full batch solution.
          sdtrack::CalibrationWindow batch_window;
          online_calib.AnalyzeCalibrationWindow(poses, current_tracks,
                0, poses.size(), batch_window, 50);

          const double batch_score =
              online_calib.GetWindowScore(batch_window);

          // Write this to the batch file.
          std::ofstream("batch.txt", std::ios_base::app) << keyframe_id << ", " <<
            batch_window.covariance.diagonal().transpose().format(
            sdtrack::kLongCsvFmt) << ", " << batch_score << ", " <<
            batch_window.mean.transpose().format(sdtrack::kLongCsvFmt) <<
            std::endl;

          std::cerr << "Batch means are: " << batch_window.mean.transpose() <<
                       std::endl;
          std::cerr << "Batch sigmas are:\n" <<
                       batch_window.covariance << std::endl;
          std::cerr << "Batch score: " << batch_score << std::endl;
        }
      }
    }
  }

  if (do_start_new_landmarks) {
    tracker.StartNewLandmarks();
  }

  std::shared_ptr<sdtrack::TrackerPose> new_pose = poses.back();
  // Update the tracks on this new pose.
  new_pose->tracks = tracker.GetNewTracks();

  if (!do_bundle_adjustment) {
    tracker.TransformTrackTabs(tracker.t_ba());
  }
}

void ProcessImage(std::vector<cv::Mat>& images)
{
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
    }
    poses.push_back(new_pose);
    axes.push_back(std::unique_ptr<SceneGraph::GLAxis>(
                      new SceneGraph::GLAxis(0.05)));
    gui_vars.scene_graph.AddChild(axes.back().get());
  }

  guess = prev_delta_t_ba * prev_t_ba;
  if(guess.translation() == Eigen::Vector3d(0,0,0) &&
     poses.size() > 1) {
    guess.translation() = Eigen::Vector3d(0,0,-0.01);
  }

  tracker.AddImage(images, guess);
  tracker.EvaluateTrackResiduals(0, tracker.GetImagePyramid(),
                                 tracker.GetCurrentTracks());

  if (!is_manual_mode) {
    tracker.OptimizeTracks(-1, optimize_landmarks);
    tracker.PruneTracks();
  }
  // Update the pose t_ab based on the result from the tracker.
  UpdateCurrentPose();

  if (do_keyframing) {
    const double track_ratio = (double)tracker.num_successful_tracks() /
        (double)keyframe_tracks;
    const double total_trans = tracker.t_ba().translation().norm();
    const double total_rot = tracker.t_ba().so3().log().norm();

    bool keyframe_condition = track_ratio < 0.8 || total_trans > 0.2 ||
        total_rot > 0.1;

    std::cerr << "\tRatio: " << track_ratio << " trans: " << total_trans <<
                 " rot: " << total_rot << std::endl;

    if (keyframe_tracks != 0) {
      if (keyframe_condition) {
        is_keyframe = true;
      } else {
        is_keyframe = false;
      }
    }

    // If this is a keyframe, set it as one on the tracker.
    prev_delta_t_ba = tracker.t_ba() * prev_t_ba.inverse();

    if (is_keyframe) {
      tracker.AddKeyframe();
    }
    is_prev_keyframe = is_keyframe;
  } else {
    tracker.AddKeyframe();
  }

  std::cerr << "Num successful : " << tracker.num_successful_tracks() <<
               " keyframe tracks: " << keyframe_tracks << std::endl;

  if (!is_manual_mode) {
    BaAndStartNewLandmarks();
  }

  if (do_self_cal || unknown_calibration) {
    for (size_t ii = 0; ii < rig.cameras_[0]->NumParams() ; ++ii) {
      plot_logs[ii].Log(rig.cameras_[0]->GetParams()[ii],
          old_rig.cameras[0].camera.GenericParams()[ii]);
    }
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

  for (uint32_t ii = 0; ii < poses.size() ; ++ii) {
    axes[ii]->SetPose(poses[ii]->t_wp.matrix());
  }

  // Draw the tracks
  for (std::shared_ptr<sdtrack::DenseTrack>& track : *current_tracks) {
    Eigen::Vector2d center;
    if (track->keypoints.back()[cam_id].tracked) {
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

  for (int cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
    gui_vars.camera_view[cam_id]->RenderChildren();
  }
}

void Run()
{
  std::vector<pangolin::GlTexture> gl_tex;

  // pangolin::Timer timer;
  bool capture_success = false;
  std::shared_ptr<pb::ImageArray> images = pb::ImageArray::Create();
  camera_device.Capture(*images);
  while(!pangolin::ShouldQuit()) {
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
      gl_tex.resize(images->Size());

      for (uint32_t cam_id = 0 ; cam_id < images->Size() ; ++cam_id) {
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
      ProcessImage(cvmat_images);
    }

    if (camera_img && camera_img->data()) {
      for (uint32_t cam_id = 0 ; cam_id < rig.cameras_.size() &&
           cam_id < images->Size(); ++cam_id) {
        camera_img = images->at(cam_id);
        gui_vars.camera_view[cam_id]->ActivateAndScissor();
        gl_tex[cam_id].Upload(camera_img->data(), camera_img->Format(),
                      camera_img->Type());
        gl_tex[cam_id].RenderToViewportFlipY();
        DrawImageData(cam_id);
      }

      gui_vars.grid_view->ActivateAndScissor(gui_vars.gl_render3d);

      if (draw_landmarks) {
        DrawLandmarks(min_lm_measurements_for_drawing, poses, rig,
                      gui_vars.handler, selected_track_id);
      }
    }
    pangolin::FinishFrame();
  }
}

void InitGui() {
  InitTrackerGui(gui_vars, window_width, window_height, image_width,
                 image_height, rig.cameras_.size());
//  pangolin::CreateWindowAndBind("2dtracker", window_width * 2, window_height);

//  render_state.SetModelViewMatrix( pangolin::IdentityMatrix() );
//  render_state.SetProjectionMatrix(
//        pangolin::ProjectionMatrixOrthographic(0, window_width, 0,
//                                               window_height, 0, 1000));
//  handler = new TrackerHandler(render_state, image_width, image_height);

//  glPixelStorei(GL_PACK_ALIGNMENT,1);
//  glPixelStorei(GL_UNPACK_ALIGNMENT,1);

//  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//  glEnable( GL_BLEND );

//  grid.SetNumLines(20);
//  grid.SetLineSpacing(5.0);
//  gui_vars.scene_graph.AddChild(&grid);

//  // Add named OpenGL viewport to window and provide 3D Handler
//  camera_view = &pangolin::Display("image")
//      .SetAspect(-(float)window_width/(float)window_height);
//  grid_view = &pangolin::Display("grid")
//      .SetAspect(-(float)window_width/(float)window_height);

//  gl_render3d.SetProjectionMatrix(
//        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.01,5000));
//  gl_render3d.SetModelViewMatrix(
//        pangolin::ModelViewLookAt(-3,-3,-4, 0,0,0, pangolin::AxisNegZ));
//  sg_handler_.reset(new SceneGraph::HandlerSceneGraph(
//                      scene_graph, gl_render3d, pangolin::AxisNegZ, 50.0f));
//  grid_view->SetHandler(sg_handler_.get());
//  grid_view->SetDrawFunction(SceneGraph::ActivateDrawFunctor(
//                               scene_graph, gl_render3d));

//  //.SetBounds(0.0, 1.0, 0, 1.0, -(float)window_width/(float)window_height);

//  pangolin::Display("multi")
//      .SetBounds(1.0, 0.0, 0.0, 1.0)
//      .SetLayout(pangolin::LayoutEqual)
//      .AddDisplay(*camera_view)
//      .AddDisplay(*grid_view);

//  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();
//  glClearColor(0.0,0.0,0.0,1.0);

//  std::cerr << "Viewport: " << camera_view->v.l << " " <<
//               camera_view->v.r() << " " << camera_view->v.b << " " <<
//               camera_view->v.t() << std::endl;

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
  });

  pangolin::RegisterKeyPressCallback(' ', [&]() {
    is_running = !is_running;
  });

  pangolin::RegisterKeyPressCallback('c', [&]() {
    do_self_cal = !do_self_cal;
  });

  pangolin::RegisterKeyPressCallback('b', [&]() {
    last_optimization_level = 0;
    tracker.OptimizeTracks();
  });

  pangolin::RegisterKeyPressCallback('B', [&]() {
    do_bundle_adjustment = !do_bundle_adjustment;
    std::cerr << "Do BA:" << do_bundle_adjustment << std::endl;
  });

  pangolin::RegisterKeyPressCallback('k', [&]() {
    is_keyframe = !is_keyframe;
    std::cerr << "is_keyframe:" << is_keyframe << std::endl;
  });

  pangolin::RegisterKeyPressCallback('S', [&]() {
    do_start_new_landmarks = !do_start_new_landmarks;
    std::cerr << "Do SNL:" << do_start_new_landmarks << std::endl;
  });

  pangolin::RegisterKeyPressCallback('2', [&]() {
    last_optimization_level = 2;
    tracker.OptimizeTracks(last_optimization_level,
                           optimize_landmarks);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('3', [&]() {
    last_optimization_level = 3;
    tracker.OptimizeTracks(last_optimization_level,
                           optimize_landmarks);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('1', [&]() {
    last_optimization_level = 1;
    tracker.OptimizeTracks(last_optimization_level,
                           optimize_landmarks);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('0', [&]() {
    last_optimization_level = 0;
    tracker.OptimizeTracks(last_optimization_level,
                           optimize_landmarks);
    UpdateCurrentPose();
  });

  pangolin::RegisterKeyPressCallback('9', [&]() {
    last_optimization_level = 0;
    tracker.OptimizeTracks(-1, optimize_landmarks);
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

  pangolin::RegisterKeyPressCallback('m', [&]() {
    is_manual_mode = !is_manual_mode;
    std::cerr << "Manual mode:" << is_manual_mode << std::endl;
  });

  // set up the plotters.
  if (do_self_cal) {
    plot_views.resize(rig.cameras_[0]->NumParams());
    plot_logs.resize(rig.cameras_[0]->NumParams());
    double bottom = 0;
    for (size_t ii = 0; ii < rig.cameras_[0]->NumParams(); ++ii) {
      plot_views[ii] = new pangolin::Plotter(&plot_logs[ii]);
      plot_views[ii]->SetBounds(bottom, bottom + 0.1, 0.6, 1.0);
      bottom += 0.1;
      pangolin::DisplayBase().AddDisplay(*plot_views[ii]);
    }
  }
}

bool LoadCameras(GetPot& cl)
{
  LoadCameraAndRig(cl, camera_device, old_rig);
  // If we require self-calibration from an unknown initial calibration, then
  // perturb the values.
  rig.Clear();
  calibu::CreateFromOldRig(&old_rig, &rig);
  if (unknown_calibration) {
    Eigen::VectorXd params = old_rig.cameras[0].camera.GenericParams();
    // fov in rads.
    const double fov_rads = 90 * M_PI / 180.0;
    const double f_x =
        0.5 * old_rig.cameras[0].camera.Height() / tan(fov_rads / 2);
    std::cerr << "Changing fx from " << params[0] << " to " << f_x << std::endl;
    std::cerr << "Changing fy from " << params[1] << " to " << f_x << std::endl;
    params[0] = f_x;
    params[1] = f_x;
    params[2] = old_rig.cameras[0].camera.Width() / 2;
    params[3] = old_rig.cameras[0].camera.Height() / 2;
    if (params.rows() > 4) {
      params[4] = 1.0;
    }

    rig.cameras_[0]->SetParams(params);

    // Add a marker in the batch file for this initial, unknown calibration.
    Eigen::VectorXd initial_covariance(params.rows());
    initial_covariance.setOnes();
    std::ofstream("batch.txt", std::ios_base::app) << 0 << ", " <<
      initial_covariance.transpose().format(sdtrack::kLongCsvFmt) <<
      ", " << 0 << ", " << params.transpose().format(sdtrack::kLongCsvFmt) <<
      std::endl;
  }
  return true;
}

int main(int argc, char** argv) {
  srand(0);
  GetPot cl(argc, argv);
  if (cl.search("--help")) {
    LOG(INFO) << g_usage;
    exit(-1);
  }

  if (cl.search("-startnow")) {
    is_running = true;
  }

  LOG(INFO) << "Initializing camera...";
  LoadCameras(cl);

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
  tracker_options.dense_ncc_threshold = ncc_threshold;
  tracker_options.harris_score_threshold = 2e6;
  tracker_options.gn_scaling = 1.0;
  tracker.Initialize(keypoint_options, tracker_options, &rig);

  // Initialize the online calibration component.
  Eigen::VectorXd weights(rig.cameras_[0]->NumParams());
  if (weights.rows() > 4) {
    weights << 1.0, 1.0, 1.7, 1.7, 320000;
    //weights << 1.0, 1.0, 1.0, 1.0, 1.0;
  } else {
    weights << 1.0, 1.0, 1.7, 1.7;
  }

  online_calib.Init(&rig, num_self_cal_segments,
                    self_cal_segment_length, weights);

  InitGui();

  ba::debug_level_threshold = -1;

  Run();

  return 0;
}
