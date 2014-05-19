// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.
#undef NDEBUG
#include <assert.h>
#include <Eigen/Eigen>
#include <HAL/Camera/CameraDevice.h>
#include <miniglog/logging.h>
#include <calibu/utils/Xml.h>
#include "GetPot"
#include <sdtrack/TicToc.h>
#include <HAL/IMU/IMUDevice.h>
#include <PbMsgs/Matrix.h>
#include <unistd.h>
#include <SceneGraph/SceneGraph.h>
#include <pangolin/pangolin.h>
#include <ba/BundleAdjuster.h>
#include <ba/InterpolationBuffer.h>
#include "CVars/CVar.h"
#include <sdtrack/utils.h>
#include "math_types.h"
#include "gui_common.h"
#include "etc_common.h"
#ifdef CHECK_NANS
#include <xmmintrin.h>
#endif

#include <sdtrack/semi_dense_tracker.h>


static int& num_ba_poss =
    CVarUtils::CreateCVar<>("sd.NumBaPoses", 20, "");
static bool& regularize_biases_in_batch =
    CVarUtils::CreateCVar<>("sd.RegularizeBiasesInBatch", true, "");
static bool& do_keyframing =
    CVarUtils::CreateCVar<>("sd.DoKeyframing", true, "");
static int& num_ba_iterations =
    CVarUtils::CreateCVar<>("sd.NumBAIterations", 20, "");
static int& min_poses_for_imu =
    CVarUtils::CreateCVar<>("sd.MinPosesForImu", 5, "");


uint32_t keyframe_tracks = UINT_MAX;
uint32_t frame_count = 0;
Sophus::SE3d last_t_ba, prev_delta_t_ba, prev_t_ba;

const int window_width = 640;
const int window_height = 480;
const char* g_usage = "";
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
hal::IMU imu_device;
sdtrack::SemiDenseTracker tracker;

pangolin::View* camera_view, *grid_view;
pangolin::View patch_view;
pangolin::OpenGlRenderState  gl_render3d;
std::unique_ptr<SceneGraph::HandlerSceneGraph> sg_handler_;
SceneGraph::GLSceneGraph  scene_graph;
SceneGraph::GLGrid grid;

// TrackCenterMap current_track_centers;
std::list<std::shared_ptr<sdtrack::DenseTrack>>* current_tracks = nullptr;
int last_optimization_level = 0;
// std::shared_ptr<sdtrack::DenseTrack> selected_track = nullptr;
std::shared_ptr<pb::Image> camera_img;
std::vector<std::vector<std::shared_ptr<SceneGraph::ImageView>>> patches;
std::vector<std::shared_ptr<sdtrack::TrackerPose>> poses;
std::vector<std::unique_ptr<SceneGraph::GLAxis> > axes_;

// inertial stuff
ba::BundleAdjuster<double, 1, 6, 0> bundle_adjuster;
ba::BundleAdjuster<double, 1, 15, 0> vi_bundle_adjuster;
ba::InterpolationBufferT<ba::ImuMeasurementT<Scalar>, Scalar> imu_buffer;
std::vector<uint32_t> imu_residual_ids;

TrackerHandler *handler;
pangolin::OpenGlRenderState render_state;

// State variables
std::vector<cv::KeyPoint> keypoints;

void ImuCallback(const pb::ImuMsg& ref) {
  Eigen::VectorXd a, w;
  pb::ReadVector(ref.accel(), &a);
  pb::ReadVector(ref.gyro(), &w);
  imu_buffer.AddElement(ba::ImuMeasurementT<Scalar>(w, a, ref.device_time()));
  // std::cerr << "Added accel: " << a.transpose() << " and gyro " <<
  //              w.transpose() << " at time " << ref.device_time() << std::endl;
}

template <typename BaType>
void DoBundleAdjustment(BaType& ba, bool use_imu, uint32_t num_active_poses)
{
  imu_residual_ids.clear();
  ba::Options<double> options;
  options.projection_outlier_threshold = 1.0;
  options.trust_region_size = 10;
  options.regularize_biases_in_batch = regularize_biases_in_batch;
  uint32_t num_outliers = 0;
  Sophus::SE3d t_ba;
  // Find the earliest pose touched by the current tracks.
  uint32_t start_active_pose, start_pose;

  GetBaPoseRange(poses, num_active_poses, start_pose, start_active_pose);

  if (start_pose == poses.size()) {
    return;
  }

  bool all_poses_active = start_active_pose == start_pose;

  // Do a bundle adjustment on the current set
  if (current_tracks && poses.size() > 1) {
    std::shared_ptr<sdtrack::TrackerPose> last_pose = poses.back();
    ba.Init(options, poses.size(),
                         current_tracks->size() * poses.size());
    ba.AddCamera(rig.cameras_[0], rig.t_wc_[0]);
    // First add all the poses and landmarks to ba.
    for (uint32_t ii = start_pose ; ii < poses.size() ; ++ii) {
      std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
      const bool is_active = ii >= start_active_pose;
      if (use_imu) {
        pose->opt_id = ba.AddPose(pose->t_wp, Sophus::SE3t(), Eigen::VectorXt(),
                                  pose->v_w, pose->b, is_active, pose->time);
      } else {
        pose->opt_id = ba.AddPose(pose->t_wp, is_active, pose->time);
      }
      if (use_imu && ii > start_active_pose) {
        std::vector<ba::ImuMeasurementT<Scalar>> meas =
            imu_buffer.GetRange(poses[ii - 1]->time, pose->time);
        /*std::cerr << "Adding imu residual between poses " << ii - 1 << " with "
                     " time " << poses[ii - 1]->time <<  " and " << ii <<
                     " with time " << pose->time << " with " << meas.size() <<
                     " measurements" << std::endl;
                     */
        imu_residual_ids.push_back(
              ba.AddImuResidual(poses[ii - 1]->opt_id, pose->opt_id, meas));
      }
      for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
        const bool constrains_active =
            track->keypoints.size() + ii > start_active_pose;
        if (track->num_good_tracked_frames == 1 || track->is_outlier ||
            !constrains_active) {
          track->external_id = UINT_MAX;
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
        track->external_id =
            ba.AddLandmark(ray, pose->opt_id, 0, active);
      }
    }

    // Now add all reprojections to ba)
    for (uint32_t ii = start_pose ; ii < poses.size() ; ++ii) {
      std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
      for (std::shared_ptr<sdtrack::DenseTrack> track : pose->tracks) {
        if (track->external_id == UINT_MAX) {
          continue;
        }
        for (size_t jj = 0; jj < track->keypoints.size() ; ++jj) {
          if (track->keypoints_tracked[jj]) {
            const Eigen::Vector2d& z = track->keypoints[jj];
            const uint32_t res_id =
                ba.AddProjectionResidual(
                  z, pose->opt_id + jj, track->external_id, 0);
          }
        }
      }
    }

    // Optimize the poses
    ba.Solve(num_ba_iterations);

    // Get the pose of the last pose. This is used to calculate the relative
    // transform from the pose to the current pose.
    last_pose->t_wp = ba.GetPose(last_pose->opt_id).t_wp;
    // std::cerr << "last pose t_wp: " << std::endl << last_pose->t_wp.matrix() <<
    //              std::endl;

    // Read out the pose and landmark values.
    for (uint32_t ii = start_pose ; ii < poses.size() ; ++ii) {
      std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
      const ba::PoseT<double>& ba_pose = ba.GetPose(pose->opt_id);

      pose->t_wp = ba_pose.t_wp;
      if (use_imu) {
        pose->v_w = ba_pose.v_w;
        pose->b = ba_pose.b;
        // if (!regularize_biases_in_batch) {
          std::cerr << "b for pose " << ii << " is " << pose->b.transpose() <<
                       std::endl;
        // }
      }
      // Here the last pose is actually t_wb and the current pose t_wa.
      last_t_ba = t_ba;
      t_ba = last_pose->t_wp.inverse() * pose->t_wp;
      for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
        if (track->external_id == UINT_MAX) {
          continue;
        }
        track->t_ba = t_ba;

        // Get the landmark location in the world frame.
        const Eigen::Vector4d& x_w =
            ba.GetLandmark(track->external_id);
        double ratio = ba.LandmarkOutlierRatio(track->external_id);
        auto landmark =
            ba.GetLandmarkObj(track->external_id);
        if (ratio > 0.4) {
          num_outliers++;
          track->is_outlier = true;
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

  if (do_bundle_adjustment) {
    if (poses.size() > min_poses_for_imu) {
      DoBundleAdjustment(vi_bundle_adjuster, true, num_ba_poss);
    } else {
      DoBundleAdjustment(bundle_adjuster, false, num_ba_poss);
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

void ProcessImage(cv::Mat& image, double timestamp)
{
  // std::cerr << "Processing image with timestamp " << timestamp << std::endl;
#ifdef CHECK_NANS
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() &
                         ~(_MM_MASK_INVALID | _MM_MASK_OVERFLOW |
                           _MM_MASK_DIV_ZERO));
#endif

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
      // Set the initial velocity and bias. The initial pose is initialized to
      // 0 through the SE3 default constructor.
      new_pose->v_w.setZero();
      new_pose->b.setZero();
      new_pose->b << -0.0107279, 0.0125927, 0.00212173,
          0.0933186, 0.517752, 0.63016;
    }
    poses.push_back(new_pose);
    axes_.push_back(std::unique_ptr<SceneGraph::GLAxis>(
                      new SceneGraph::GLAxis(0.05)));
    scene_graph.AddChild(axes_.back().get());
  }

  // Set the timestamp of the latest pose to this image's timestamp.
  poses.back()->time = timestamp;

  guess = prev_delta_t_ba * prev_t_ba;
  if(guess.translation() == Eigen::Vector3d(0,0,0) &&
     poses.size() > 1) {
    guess.translation() = Eigen::Vector3d(0,0,0.01);
  }

  tracker.AddImage(image, guess);
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
               std::endl;
}

void DrawImageData()
{
  handler->track_centers.clear();

  for (uint32_t ii = 0; ii < poses.size() ; ++ii) {
    axes_[ii]->SetPose(poses[ii]->t_wp.matrix());
  }

  // Draw the tracks
  for (std::shared_ptr<sdtrack::DenseTrack>& track : *current_tracks) {  
    Eigen::Vector2d center;
    DrawTrackData(track, image_width, image_height, last_optimization_level,
                  center);
    handler->track_centers.push_back(
          std::pair<Eigen::Vector2d, std::shared_ptr<sdtrack::DenseTrack>>(
            center, track));
  }

  // Populate the first column with the reference from the selected track.
  if (handler->selected_track != nullptr) {
    DrawTrackPatches(handler->selected_track, patches);
  }
}

void Run()
{
  pangolin::GlTexture gl_tex;

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
      camera_img = images->at(0);
      image_width = camera_img->Width();
      image_height = camera_img->Height();
      handler->image_height = image_height;
      handler->image_width = image_width;
      if (!gl_tex.tid) {
        GLint internal_format = (camera_img->Format() == GL_LUMINANCE ?
                                   GL_LUMINANCE : GL_RGBA);
        // Only initialise now we know format.
        gl_tex.Reinitialise(camera_img->Width() , camera_img->Height(),
                            internal_format, false, 0,
                            camera_img->Format(), camera_img->Type(), 0);
      }

      ProcessImage(camera_img->Mat(), images->Timestamp());
    }
    if (camera_img && camera_img->data()) {
      camera_view->ActivateAndScissor();
      gl_tex.Upload(camera_img->data(), camera_img->Format(),
                    camera_img->Type());
      gl_tex.RenderToViewportFlipY();
      DrawImageData();
      // camera_view->RenderChildren();

      grid_view->ActivateAndScissor(gl_render3d);
      const ba::ImuCalibrationT<Scalar>& imu =
          vi_bundle_adjuster.GetImuCalibration();
      std::vector<ba::ImuPoseT<Scalar>> poses;

      glLineWidth(2.0f);
      glColor3f(1.0, 0.0, 1.0);
      // glPushMatrix();
      // glMultMatrixT(t_world_frame.matrix().data());
      // Draw the inertial residuals
      for (uint32_t id : imu_residual_ids) {
        const ba::ImuResidualT<Scalar>& res = vi_bundle_adjuster.GetImuResidual(id);
        const ba::PoseT<Scalar>& pose = vi_bundle_adjuster.GetPose(res.pose1_id);
        res.IntegrateResidual(pose, res.measurements,
                              pose.b.head<3>(),
                              pose.b.tail<3>(), imu.g_vec, poses);
        // std::cerr << "integrating residual with " << res.measurements.size() <<
        //              " measurements " << std::endl;

        for (size_t ii = 1 ; ii < poses.size() ; ++ii) {
          ba::ImuPoseT<Scalar>& prev_imu_pose = poses[ii - 1];
          ba::ImuPoseT<Scalar>& imu_pose = poses[ii];
          pangolin::glDrawLine(prev_imu_pose.t_wp.translation()[0],
                               prev_imu_pose.t_wp.translation()[1],
                               prev_imu_pose.t_wp.translation()[2],
                               imu_pose.t_wp.translation()[0],
                               imu_pose.t_wp.translation()[1],
                               imu_pose.t_wp.translation()[2]);
          /*std::cerr << "Drawing line from " <<
                       prev_imu_pose.t_wp.translation().transpose() << " to " <<
                       imu_pose.t_wp.translation() << std::endl;*/

        }
      }
      // grid_view->RenderChildren();
    }
    pangolin::FinishFrame();
  }
}

void InitGui()
{
  pangolin::CreateWindowAndBind("2dtracker", window_width * 2, window_height);

  render_state.SetModelViewMatrix( pangolin::IdentityMatrix() );
  render_state.SetProjectionMatrix(
        pangolin::ProjectionMatrixOrthographic(0, window_width, 0,
                                               window_height, 0, 1000));
  handler = new TrackerHandler(render_state, image_width, image_height);

  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1);

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable( GL_BLEND );

  grid.SetNumLines(20);
  grid.SetLineSpacing(5.0);
  scene_graph.AddChild(&grid);

  // Add named OpenGL viewport to window and provide 3D Handler
  camera_view = &pangolin::Display("image")
      .SetAspect(-(float)window_width/(float)window_height);
  grid_view = &pangolin::Display("grid")
      .SetAspect(-(float)window_width/(float)window_height);

  gl_render3d.SetProjectionMatrix(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.01,5000));
  gl_render3d.SetModelViewMatrix(
        pangolin::ModelViewLookAt(-3,-3,-4, 0,0,0, pangolin::AxisNegZ));
  sg_handler_.reset(new SceneGraph::HandlerSceneGraph(
                      scene_graph, gl_render3d, pangolin::AxisNegZ, 50.0f));
  grid_view->SetHandler(sg_handler_.get());
  grid_view->SetDrawFunction(SceneGraph::ActivateDrawFunctor(
                               scene_graph, gl_render3d));

  //.SetBounds(0.0, 1.0, 0, 1.0, -(float)window_width/(float)window_height);

  pangolin::Display("multi")
      .SetBounds(1.0, 0.0, 0.0, 1.0)
      .SetLayout(pangolin::LayoutEqual)
      .AddDisplay(*camera_view)
      .AddDisplay(*grid_view);

  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();
  glClearColor(0.0,0.0,0.0,1.0);

  std::cerr << "Viewport: " << camera_view->v.l << " " <<
               camera_view->v.r() << " " << camera_view->v.b << " " <<
               camera_view->v.t() << std::endl;

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

  // Create the patch grid.
  camera_view->AddDisplay(patch_view);
  camera_view->SetHandler(handler);
  patch_view.SetBounds(0.01, 0.31, 0.69, .99, 1.0f/1.0f);

  CreatePatchGrid(3, 3,  patches, patch_view);
}

bool LoadCameras(GetPot& cl)
{
  LoadCameraAndRig(cl, camera_device, old_rig);
  calibu::CreateFromOldRig(&old_rig, &rig);
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

   // Load the imu
  std::string imu_str = cl.follow("","-imu");
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
  std::shared_ptr<pb::ImageArray> images = pb::ImageArray::Create();
  camera_device.Capture(*images);

  // Set the initial gravity from the first bit of IMU data.
  if (imu_buffer.elements.size() == 0) {
    LOG(ERROR) << "No initial IMU measurements were found.";
  }
  const Eigen::Vector3d grav = imu_buffer.elements.front().a.normalized() *
      ba::Gravity;
  std::cerr << "Setting initial gravity to " << grav << std::endl;
  vi_bundle_adjuster.SetGravity(grav);


  sdtrack::DescriptorOptions descriptor_options;
  sdtrack::KeypointOptions keypoint_options;
  keypoint_options.gftt_feature_block_size = 9;
  keypoint_options.max_num_features = 1000;
  keypoint_options.gftt_absolute_strength_threshold = 0.005;
  sdtrack::TrackerOptions tracker_options;
  tracker_options.pyramid_levels = 3;
  tracker_options.detector_type = sdtrack::TrackerOptions::Detector_GFTT;
  tracker_options.num_active_tracks = 150;
  tracker_options.use_robust_norm_ = false;
  tracker_options.robust_norm_threshold_ = 30;
  tracker_options.patch_dim = 9;
  tracker_options.default_rho = 1.0/5.0;
  tracker_options.feature_cells = 6;
  tracker_options.iteration_exponent = 2;
  tracker_options.dense_ncc_threshold = 0.85;
  tracker_options.harris_score_threshold = 2e6;
  tracker.Initialize(keypoint_options, tracker_options, &rig);

  InitGui();

  ba::debug_level_threshold = 0;

  Run();

  return 0;
}
