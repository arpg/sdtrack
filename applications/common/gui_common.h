#pragma once
#include <vector>
#include <SceneGraph/GLDynamicGrid.h>
#include <SceneGraph/SceneGraph.h>
#include <HAL/Camera/CameraDevice.h>
#include <calibu/cam/camera_crtp.h>
#include <calibu/cam/CameraRig.h>
#include <miniglog/logging.h>
#include <sdtrack/utils.h>
#include "math_types.h"
#include <sdtrack/keypoint.h>
#include <sdtrack/track.h>
#include <sdtrack/semi_dense_tracker.h>
#include "etc_common.h"
#include "timer.h"
#include "timer_view.h"

typedef std::shared_ptr<sdtrack::DenseTrack> TrackPtr;
typedef std::vector<std::pair<Eigen::Vector2d, TrackPtr>> TrackCenterMap;

static Sophus::SE3d t_wp_old;

struct TrackerHandler : pangolin::Handler3D {
  TrackerHandler(pangolin::OpenGlRenderState& cam_state, uint32_t image_w,
                 uint32_t image_h,
                 pangolin::AxisDirection enforce_up = pangolin::AxisNone,
                 float trans_scale = 0.01f)
      : pangolin::Handler3D(cam_state, enforce_up, trans_scale),
        image_width(image_w),
        image_height(image_h) {}

  void Mouse(pangolin::View& view, pangolin::MouseButton button, int x, int y,
             bool pressed, int button_state) {
    bool handled = false;
    double x_val =
        image_width * (((double)x - (double)view.v.l) / (double)view.v.w);
    double y_val =
        image_height * (1. - ((double)y - (double)view.v.b) / (double)view.v.h);

    // Figure out which track we have clicked
    selected_track = nullptr;
    for (size_t ii = 0; ii < track_centers.size(); ++ii) {
      Eigen::Vector2d center = track_centers[ii].first;
      const double dist = sqrt(sdtrack::powi(x_val - center[0], 2) +
                               sdtrack::powi(y_val - center[1], 2));
      if (dist < 10) {
        // Then this is the selected track.
        selected_track = track_centers[ii].second;
        sdtrack::DenseKeypoint& kp = selected_track->ref_keypoint;
        std::cerr << "selected kp " << selected_track->id
                  << "with response: " << kp.response << " response2 "
                  << kp.response2 << " rmse: ";
        for (size_t cam_id = 0; cam_id < selected_track->transfer.size();
             ++cam_id) {
          std::cerr << selected_track->transfer[cam_id].rmse << ", ";
        }

        std::cerr << "ncc: ";
        for (size_t cam_id = 0; cam_id < selected_track->transfer.size();
             ++cam_id) {
          std::cerr << selected_track->transfer[cam_id].ncc << ", ";
        }

        std::cerr << " rho " << selected_track->ref_keypoint.rho
                  << " tracked: " << selected_track->tracked << " outlier: "
                  << selected_track->is_outlier << std::endl;
        break;
      }
    }

    if (!handled) {
      Handler3D::Mouse(view, button, x_val, y_val, pressed, button_state);
    }
  }

  uint32_t image_width;
  uint32_t image_height;
  TrackCenterMap track_centers;
  std::shared_ptr<sdtrack::DenseTrack> selected_track;
};

struct TrackerGuiVars {
  pangolin::OpenGlRenderState render_state;
  TrackerHandler* handler;
  int image_width;
  int image_height;
  SceneGraph::GLSceneGraph scene_graph;
  SceneGraph::GLDynamicGrid grid;
  pangolin::View* grid_view;
  std::vector<pangolin::View*> camera_view;
  pangolin::View patch_view;
  pangolin::OpenGlRenderState gl_render3d;
  std::unique_ptr<SceneGraph::HandlerSceneGraph> sg_handler_;
  std::vector<std::vector<std::shared_ptr<SceneGraph::ImageView>>> patches;
  Timer     timer;
  TimerView timer_view;
};

inline Eigen::Vector2d ImageToWindowCoords(int image_width, int image_height,
                                           double x, double y) {
  Eigen::Vector2d p_win((((x / image_width) - 0.5)) * 2,
                        ((-(y / image_height) + 0.5)) * 2);

  return p_win;
}

void DrawTrackPatches(
    std::shared_ptr<sdtrack::DenseTrack>& track,
    std::vector<std::vector<std::shared_ptr<SceneGraph::ImageView>>>& patches) {
  sdtrack::DenseKeypoint& kp = track->ref_keypoint;
  //  for (uint32_t ii = 0; ii < kp.patch_pyramid.size() &&
  //       ii <= 0 ; ++ii) {
  // int ii = 0;
  const sdtrack::Patch& ref_patch = kp.patch_pyramid[track->transfer[0].level];
  std::vector<unsigned char> disp_values;
  std::vector<unsigned char> disp_proj_values;
  std::vector<unsigned char> res_values;
  disp_values.reserve(ref_patch.values.size());
  disp_proj_values.reserve(disp_values.size());
  res_values.reserve(disp_values.size());

  for (size_t ii = 0 ; ii < track->transfer.size() ; ++ii) {
    for (size_t jj = 0; jj < ref_patch.values.size(); ++jj) {
      disp_values.push_back(ref_patch.values[jj]);
      disp_proj_values.push_back(track->transfer[ii].projected_values[jj]);
      res_values.push_back(fabs(track->transfer[ii].residuals[jj]));
    }
    patches[ii][0]->SetImage(&disp_values[0], ref_patch.dim, ref_patch.dim,
                             GL_LUMINANCE8, GL_LUMINANCE);

    patches[ii][1]->SetImage(&disp_proj_values[0], ref_patch.dim, ref_patch.dim,
                             GL_LUMINANCE8, GL_LUMINANCE);

    patches[ii][2]->SetImage(&res_values[0], ref_patch.dim, ref_patch.dim,
                             GL_LUMINANCE8, GL_LUMINANCE);
  }
  // }
}

void DrawLandmarks(const uint32_t min_lm_measurements_for_drawing,
                   std::vector<std::shared_ptr<sdtrack::TrackerPose>>& poses,
                   calibu::Rig<Scalar>& rig, TrackerHandler* handler,
                   int& selected_track_id) {
  glPointSize(0.5);
  glBegin(GL_POINTS);
  for (std::shared_ptr<sdtrack::TrackerPose> pose : poses) {
    for (std::shared_ptr<sdtrack::DenseTrack> track : pose->tracks) {
      if (selected_track_id == track->id) {
        handler->selected_track = track;
        selected_track_id = -1;
      }

      if (track->num_good_tracked_frames < min_lm_measurements_for_drawing) {
        continue;
      }
      Eigen::Vector4d ray;
      ray.head<3>() = track->ref_keypoint.ray;
      ray[3] = track->ref_keypoint.rho;
      ray = sdtrack::MultHomogeneous(pose->t_wp * rig.t_wc_[0], ray);
      ray /= ray[3];
      if (handler->selected_track == track) {
        glColor3f(1.0, 1.0, 0.2);
      } else if (track->is_outlier) {
        glColor3f(1.0, 0.2, 0.1);
      } else {
        glColor3f(1.0, 1.0, 1.0);
      }
      glVertex3f(ray[0], ray[1], ray[2]);
    }
  }
  glEnd();
}

void DrawTrackData(std::shared_ptr<sdtrack::DenseTrack>& track,
                   uint32_t image_width, uint32_t image_height,
                   Eigen::Vector2d& center, bool is_selected, uint32_t cam_id) {
  Eigen::Vector3d rgb;
  // const double error = std::min(1.0, track->rmse / 15.0) * 0.7 + 0.3;
  // hsv2rgb(Eigen::Vector3d(1.0 - error, 1.0, 1.0), rgb);
  if (is_selected) {
    rgb = Eigen::Vector3d(1.0, 1.0, 0.2);
  } else if (track->is_outlier == false) {
    const double error =
        (1.0 - (std::max(track->transfer[cam_id].ncc - 0.8, 0.0) / 0.2)) * 0.7 +
        0.3;
    sdtrack::hsv2rgb(Eigen::Vector3d(1.0 - error, 1.0, 1.0), rgb);
  } else {
    rgb = Eigen::Vector3d(1.0, 0.2, 0.0);
  }

  glColor4d(rgb[0], rgb[1], rgb[2], 1.0);
  glBegin(GL_LINE_STRIP);
  double alpha = track->keypoints.size() == 1 ? 1.0 : 0.15;
  const double alpha_increment = track->keypoints.size() == 1
                                     ? 0
                                     : (1.0 - alpha) / track->keypoints.size();
  for (std::vector<sdtrack::Keypoint> keypoints : track->keypoints) {
    const Eigen::Vector2d& point = keypoints[cam_id].kp;
    if (keypoints[cam_id].tracked) {
      glColor4d(rgb[0], rgb[1], rgb[2], alpha);
      Eigen::Vector2d px_win =
          ImageToWindowCoords(image_width, image_height, point[0], point[1]);
      glVertex3f(px_win[0], px_win[1], 0);
    }
    alpha += alpha_increment;
  }
  glEnd();

  glColor4d(rgb[0], rgb[1], rgb[2], cam_id == track->ref_cam_id ? 1.0 : 0.2);
  std::vector<Eigen::Vector2d> perimiter;
  track->transfer[cam_id].GetProjectedPerimiter(perimiter, center);

  glBegin(GL_LINE_STRIP);
  for (Eigen::Vector2d point : perimiter) {
    Eigen::Vector2d px_win =
        ImageToWindowCoords(image_width, image_height, point[0], point[1]);
    glVertex3f(px_win[0], px_win[1], 0);
  }
  glEnd();

  glColor3f(0.0, 1.0, 0.0);
  glPointSize(2);
  Eigen::Vector2d px_win = ImageToWindowCoords(
      image_width, image_height, track->keypoints.back()[cam_id].kp[0],
      track->keypoints.back()[cam_id].kp[1]);
  glBegin(GL_POINTS);
  glVertex3f(px_win[0], px_win[1], 0);
  glEnd();
}

// create image views and layout
void CreatePatchGrid(
    const uint32_t num_cols, const uint32_t num_rows,
    std::vector<std::vector<std::shared_ptr<SceneGraph::ImageView>>>& patches,
    pangolin::View& patch_view) {
  float xstep = 1.0f / num_cols;
  float ystep = 1.0f / num_rows;
  float ystep_delta = ystep * 0.9;
  float xstep_delta = xstep * 0.9;

  patches.resize(num_rows);
  for (unsigned int ii = 0; ii < num_rows; ++ii) {
    float top = 1.0 - ii * ystep;
    float bottom = top - ystep_delta;
    patches[ii].resize(num_cols);
    for (unsigned int jj = 0; jj < num_cols; ++jj) {
      float left = jj * xstep;
      float right = left + xstep_delta;
      patches[ii][jj] =
          std::shared_ptr<SceneGraph::ImageView>(new SceneGraph::ImageView);
      patch_view.AddDisplay(*patches[ii][jj].get());
      patches[ii][jj]->SetSamplingLinear(false);
      patches[ii][jj]->SetBounds(bottom, top, left, right);
    }
  }
}

void InitTrackerGui(TrackerGuiVars& vars, uint32_t window_width,
                    uint32_t window_height, uint32_t handler_image_width,
                    uint32_t handler_image_height, uint32_t num_cameras) {
  vars.image_height = handler_image_height;
  vars.image_width = handler_image_width;
  pangolin::CreateWindowAndBind("2dtracker", window_width, window_height);

  vars.render_state.SetModelViewMatrix(pangolin::IdentityMatrix());
  vars.render_state.SetProjectionMatrix(pangolin::ProjectionMatrixOrthographic(
      0, window_width, 0, window_height, 0, 1000));
  vars.handler = new TrackerHandler(vars.render_state, vars.image_width,
                                    vars.image_height);

  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);

  // vars.grid.SetNumLines(20);
  // vars.grid.SetLineSpacing(5.0);
  vars.scene_graph.AddChild(&vars.grid);

  // Add named OpenGL viewport to window and provide 3D Handler
  vars.camera_view.resize(num_cameras);
  for (size_t cam_id = 0; cam_id < num_cameras; ++cam_id) {
    vars.camera_view[cam_id] =
        &pangolin::CreateDisplay().SetAspect(-(float)window_width /
                                             (float)window_height);
  }
  vars.grid_view = &pangolin::Display("grid")
                        .SetAspect(-(float)window_width / (float)window_height);

  vars.gl_render3d.SetProjectionMatrix(
      pangolin::ProjectionMatrix(window_width, window_height,
                                 420, 420, 320, 240, 0.01, 5000));
  vars.gl_render3d.SetModelViewMatrix(
      pangolin::ModelViewLookAt(-3, -3, -4, 0, 0, 0, pangolin::AxisNegZ));
  vars.sg_handler_.reset(new SceneGraph::HandlerSceneGraph(
      vars.scene_graph, vars.gl_render3d, pangolin::AxisNegZ, 50.0f));
  vars.grid_view->SetHandler(vars.sg_handler_.get());
  vars.grid_view->SetDrawFunction(
      SceneGraph::ActivateDrawFunctor(vars.scene_graph, vars.gl_render3d));

  //.SetBounds(0.0, 1.0, 0, 1.0, -(float)window_width/(float)window_height);

  pangolin::Display("multi").SetBounds(1.0, 0.0, 0.0, 1.0).SetLayout(
      pangolin::LayoutEqual);

  for (size_t cam_id = 0; cam_id < num_cameras; ++cam_id) {
    pangolin::Display("multi").AddDisplay(*vars.camera_view[cam_id]);
  }

  pangolin::Display("multi").AddDisplay(*vars.grid_view);

  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();
  glClearColor(0.0, 0.0, 0.0, 1.0);

  // Create the patch grid.
  vars.camera_view[0]->AddDisplay(vars.patch_view);
  vars.camera_view[0]->SetHandler(vars.handler);
  vars.patch_view.SetBounds(0.01, 0.31, 0.69, .99, 1.0f / 1.0f);

  CreatePatchGrid(3, 3, vars.patches, vars.patch_view);

  vars.timer_view.SetBounds(0.7, 1, 0.7, 1.0);
  pangolin::DisplayBase().AddDisplay(vars.timer_view);
  vars.timer_view.InitReset();
}

bool LoadCameraAndRig(GetPot& cl, hal::Camera& camera_device,
                      calibu::CameraRigT<Scalar>& rig,
                      bool transform_to_robotics_coords = true) {
  std::string cam_string = cl.follow("", "-cam");
  try {
    camera_device = hal::Camera(hal::Uri(cam_string));
  }
  catch (hal::DeviceException& e) {
    LOG(ERROR) << "Error loading camera device: " << e.what();
    return false;
  }

  std::string def_dir("");
  def_dir = camera_device.GetDeviceProperty(hal::DeviceDirectory);
  std::string src_dir = cl.follow(def_dir.c_str(), "-sdir");

  LOG(INFO) << "Loading camera models...";
  std::string cmod_file = cl.follow("cameras.xml", "-cmod");
  std::string filename = src_dir + "/" + cmod_file;
  LOG(INFO) << "Loading camera models from " << filename;

  calibu::CameraRigT<Scalar> xmlrig = calibu::ReadXmlRig(filename);
  if (xmlrig.cameras.empty()) {
    LOG(FATAL) << "XML Camera rig is empty!";
  }

  calibu::CameraRigT<Scalar> crig = xmlrig;
  if (transform_to_robotics_coords) {
    crig = calibu::ToCoordinateConvention<Scalar>(
        xmlrig, calibu::RdfRobotics.cast<Scalar>());

    Sophus::SE3t M_rv;
    M_rv.so3() = calibu::RdfRobotics;
    for (calibu::CameraModelAndTransformT<Scalar>& model : crig.cameras) {
      model.T_wc = model.T_wc * M_rv;
    }
  }

  LOG(INFO) << "Starting Tvs: " << crig.cameras[0].T_wc.matrix();

  rig.cameras.clear();
  for (uint32_t cam_id = 0; cam_id < crig.cameras.size(); ++cam_id) {
    rig.Add(crig.cameras[cam_id]);
  }

  for (size_t ii = 0; ii < rig.cameras.size(); ++ii) {
    LOG(INFO) << ">>>>>>>> Camera " << ii << ":" << std::endl
              << "Model: " << std::endl << rig.cameras[ii].camera.K()
              << std::endl << "Pose: " << std::endl
              << rig.cameras[ii].T_wc.matrix();
  }
  return true;
}

inline pangolin::OpenGlMatrix LookAtModelViewMatrix(
    const Eigen::Vector4t& target,
    const Eigen::Vector4t& near_source,
    const Eigen::Matrix4t& t_wp) {
  const Eigen::Vector3t up = -t_wp.col(2).head<3>();
  auto w_target = t_wp * target;
  auto w_near_source = t_wp * near_source;

  const Eigen::Vector3t dir = (w_target - w_near_source).head<3>().normalized();
  Scalar dot = dir.dot(up.normalized());
  CHECK(!std::isnan(dot));

  return pangolin::ModelViewLookAt(
      w_near_source[0], w_near_source[1], w_near_source[2],
      w_target[0], w_target[1], w_target[2],
      up[0], up[1], up[2]);
}

//pangolin::OpenGlMatrix FollowingCameraModelView(Sophus::SE3t t_wp) {
//  static const Eigen::Vector4t target(0, 0, 0, 1);
//  static const Eigen::Vector4t near_source(-30, 0, -9, 1);
//  return LookAtModelViewMatrix(target, near_source, t_wp.matrix());
//}

void FollowCamera(TrackerGuiVars& vars, const Sophus::SE3t& t_wp_new) {
  pangolin::OpenGlMatrix mat = vars.gl_render3d.GetModelViewMatrix();
  #define M(row,col)  mat.m[col*4+row]
  double e_x = M(0,3);
  double e_y = M(1,3);
  double e_z = M(2,3);
  double ex = -(M(0,0)*e_x + M(1,0)*e_y + M(2,0)*e_z);
  double ey = -(M(0,1)*e_x + M(1,1)*e_y + M(2,1)*e_z);
  double ez = -(M(0,2)*e_x + M(1,2)*e_y + M(2,2)*e_z);
  Eigen::Vector3d vec(ex, ey, ez);
  Eigen::Vector3d offset = vec - t_wp_old.translation();
  // vec = sdtrack::MultHomogeneous(poses.back()->t_wp * old_twp.inverse(), vec);
  vec = t_wp_new.translation() + offset;
  ex = vec[0];
  ey = vec[1];
  ez = vec[2];

  M(0,3) = -(M(0,0)*ex + M(0,1)*ey + M(0,2)*ez);
  M(1,3) = -(M(1,0)*ex + M(1,1)*ey + M(1,2)*ez);
  M(2,3) = -(M(2,0)*ex + M(2,1)*ey + M(2,2)*ez);
  #undef M
  vars.gl_render3d.SetModelViewMatrix(mat);
  t_wp_old = t_wp_new;
}
