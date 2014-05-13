#pragma once
#include <vector>
#include <SceneGraph/SceneGraph.h>
#include <HAL/Camera/CameraDevice.h>
#include <calibu/cam/camera_crtp.h>
#include <calibu/cam/CameraRig.h>
#include <miniglog/logging.h>
#include <sdtrack/utils.h>
#include "math_types.h"
#include <sdtrack/keypoint.h>
#include <sdtrack/track.h>

typedef std::shared_ptr<sdtrack::DenseTrack> TrackPtr ;
typedef std::vector<std::pair<Eigen::Vector2d, TrackPtr>> TrackCenterMap;

struct TrackerHandler : pangolin::Handler3D
{
  TrackerHandler(pangolin::OpenGlRenderState& cam_state,
          uint32_t image_w, uint32_t image_h,
          pangolin::AxisDirection enforce_up=pangolin::AxisNone,
          float trans_scale=0.01f)
    : pangolin::Handler3D(cam_state, enforce_up, trans_scale),
  image_width(image_w), image_height(image_h){}

  void Mouse(pangolin::View& view, pangolin::MouseButton button, int x, int y,
             bool pressed, int button_state)
  {
    bool handled = false;
    double x_val =
        image_width * (((double)x - (double)view.v.l) / (double)view.v.w);
    double y_val =
        image_height * (1. - ((double)y - (double)view.v.b) / (double)view.v.h);

    // Figure out which track we have clicked
    selected_track = nullptr;
    for (size_t ii = 0; ii < track_centers.size() ; ++ii) {
      Eigen::Vector2d center = track_centers[ii].first;
      const double dist = sqrt(sdtrack::powi(x_val - center[0], 2) +
          sdtrack::powi(y_val - center[1], 2));
      std::cerr << "Dist is " << dist << " w " << image_width << " h " <<
                   image_height << std::endl;
      if (dist < 4) {
        // Then this is the selected track.
        selected_track = track_centers[ii].second;
        sdtrack::DenseKeypoint& kp = selected_track->ref_keypoint;
        sdtrack::Patch& p = selected_track->ref_keypoint.patch_pyramid[0];
        const double ncc = sdtrack::ScorePatchesNCC(
              p.values, p.projected_values, 9, 9);
        std::cerr << "selected kp " << selected_track->id <<
                     "with response: " << kp.response << " rmse " <<
                     selected_track->rmse << " ncc: " << ncc << " track ncc " <<
                     selected_track->ncc << std::endl;
        break;
      }
    }

    if(!handled) {
      Handler3D::Mouse(view,button,x_val,y_val,pressed,button_state);
    }
  }

  uint32_t image_width;
  uint32_t image_height;
  TrackCenterMap track_centers;
  std::shared_ptr<sdtrack::DenseTrack> selected_track;
};

inline Eigen::Vector2d ImageToWindowCoords(int image_width, int image_height,
                                           double x, double y)
{
  Eigen::Vector2d p_win((((x / image_width) - 0.5)) * 2,
                        ((-(y / image_height) + 0.5)) * 2);

  return p_win;
}

void DrawTrackData(std::shared_ptr<sdtrack::DenseTrack>& track,
                   uint32_t image_width, uint32_t image_height,
                   uint32_t opt_level, Eigen::Vector2d& center)
{
  Eigen::Vector3d rgb;
  // const double error = std::min(1.0, track->rmse / 15.0) * 0.7 + 0.3;
  // hsv2rgb(Eigen::Vector3d(1.0 - error, 1.0, 1.0), rgb);
  if (track->is_outlier == false) {
    const double error =  (1.0 - (std::max(track->ncc - 0.8, 0.0) / 0.2))
        * 0.7 + 0.3;
    sdtrack::hsv2rgb(Eigen::Vector3d(1.0 - error, 1.0, 1.0), rgb);
  } else {
    rgb = Eigen::Vector3d(1.0, 0.2, 0.0);
  }


  glBegin(GL_LINE_STRIP);
  double alpha = track->keypoints.size() == 1 ? 1.0 : 0.15;
  const double alpha_increment = track->keypoints.size() == 1  ?
        0 : (1.0 - alpha) / track->keypoints.size();
  for (Eigen::Vector2d point : track->keypoints) {
    glColor4d(rgb[0], rgb[1], rgb[2], alpha);
    alpha += alpha_increment;
    Eigen::Vector2d px_win = ImageToWindowCoords(image_width, image_height,
                                                 point[0], point[1]);
    glVertex3f(px_win[0], px_win[1], 0);
  }
  glEnd();

  const sdtrack::Patch& ref_patch =
      track->ref_keypoint.patch_pyramid[opt_level];
  std::vector<Eigen::Vector2d> perimiter;
  ref_patch.GetProjectedPerimiter(perimiter, center);

  glBegin(GL_LINE_STRIP);
  for (Eigen::Vector2d point : perimiter) {
    Eigen::Vector2d px_win = ImageToWindowCoords(image_width, image_height,
                                                 point[0], point[1]);
    glVertex3f(px_win[0], px_win[1], 0);
  }
  glEnd();

  glColor3f(0.0, 1.0, 0.0);
  glPointSize(2);
  Eigen::Vector2d px_win = ImageToWindowCoords(image_width, image_height,
        track->keypoints.back()[0],
      track->keypoints.back()[1]);
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
  for (unsigned int ii=0; ii < num_rows; ++ii) {
    float top =  1.0-ii*ystep;
    float bottom = top - ystep_delta;
    patches[ii].resize(num_cols);
    for (unsigned int jj=0; jj < num_cols; ++jj) {
      float left  = jj*xstep;
      float right = left + xstep_delta;
      patches[ii][jj] =
          std::shared_ptr<SceneGraph::ImageView>(new SceneGraph::ImageView);
      patch_view.AddDisplay(*patches[ii][jj].get());
      patches[ii][jj]->SetSamplingLinear(false);
      patches[ii][jj]->SetBounds(bottom,top,left,right);
    }
  }
}

bool LoadCameraAndRig(GetPot& cl,
             hal::Camera& camera_device,
             calibu::CameraRigT<Scalar>& rig)
{
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

  calibu::CameraRigT<Scalar> crig;
  crig = calibu::ToCoordinateConvention<Scalar>(
        xmlrig, calibu::RdfRobotics.cast<Scalar>());

  Sophus::SE3t M_rv;
  M_rv.so3() = calibu::RdfRobotics;
  for (calibu::CameraModelAndTransformT<Scalar>& model : crig.cameras) {
    model.T_wc = model.T_wc*M_rv;
  }
  LOG(INFO) << "Starting Tvs: " << crig.cameras[0].T_wc.matrix();

  rig.cameras.clear();
  rig.Add( crig.cameras[0]);

  for (size_t ii=0; ii < rig.cameras.size(); ++ii) {
    LOG(INFO) << ">>>>>>>> Camera " << ii << ":"  << std::endl
              << "Model: " << std::endl
              << rig.cameras[ii].camera.K() << std::endl
              << "Pose: " << std::endl
              << rig.cameras[ii].T_wc.matrix();
  }
  return true;
}
