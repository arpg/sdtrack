#pragma once
#include <ceres/ceres.h>
#include <calibu/cam/camera_crtp.h>

///
/// \brief Cost function for inverse depth projection residuals. Optimizes the
/// inverse depth of the landmark, as well as the pose of the reference and
/// measurement cameras.
///
template<typename CameraType>
struct InvepthCost {
  ///
  /// \brief InvepthCost
  /// \param t_vc_r Camera to vehicle frame transformation at the ref. pose.
  /// \param t_vc_m Camera to vehicle frame transformation at the meas. pose.
  /// \param l_c_r Back-projected ray in the ref. pose camera frame.
  /// \param z 2d measurement location in the meas. camera.
  /// \param params Camera parameter vector.
  ///
  InvepthCost(const Sophus::SE3d& t_vc_r,
              const Sophus::SE3d& t_vc_m,
              const Eigen::Vector3d& l_c_r,
              const Eigen::Vector2d& z,
              const Eigen::VectorXd& params)
      : t_vc_r_(t_vc_r), t_cv_m_(t_vc_m.inverse()), l_c_r_(l_c_r), z_(z),
        params_(params)
  {}

  ///
  /// \brief operator ()
  /// \param _t_wv_r Reference pose update parameter (6d).
  /// \param _t_wv_m Measurement pose update parameter (6d).
  /// \param _rho Inverse depth parameter for the landmark (1d).
  /// \param _r 2d projection residual.
  /// \return
  ///
  template<typename T>
  bool operator()(const T* const _t_wv_r,
                  const T* const _t_wv_m,
                  const T* const _rho,
                  T* _r) const {
    Eigen::Map<Eigen::Matrix<T, 2, 1>> r(_r); // residual vector
    const Eigen::Matrix<T, Eigen::Dynamic, 1> params_t = params_.cast<T>();

    const Eigen::Map<const Sophus::SE3Group<T>> t_wv_r(_t_wv_r); // ref pose
    const Eigen::Map<const Sophus::SE3Group<T>> t_wv_m(_t_wv_m); // meas pose

    // Transformation from reference to measurement camera.
    const Sophus::SE3Group<T> t_mr =
        t_cv_m_.cast<T>() * t_wv_m.inverse() * t_wv_r * t_vc_r_.cast<T>();
    const Eigen::Matrix<T, 3, 1> l_c_m =
        t_mr.rotationMatrix() * l_c_r_.cast<T>() + *_rho * t_mr.translation();
    Eigen::Matrix<T, 2, 1> pix;
    CameraType::template Project<T>(l_c_m.data(), params_t.data(), pix.data());
    // Compute the residual.
    r =  z_.cast<T>() - pix;
    return true;
  }

  Sophus::SE3d t_vc_r_;
  Sophus::SE3d t_cv_m_;
  Eigen::Vector3d l_c_r_;
  Eigen::Vector2d z_;
  Eigen::VectorXd params_;
};

///
/// \brief Cost function for inverse depth projection residuals. Optimizes the
/// inverse depth of the landmark, the pose of the reference and measurement
/// cameras, the transform between the vehicle and camera coordinates and also
/// the calibration parameters for the SINGLE camera for both reference and
/// measurement frames.
/// If different cameras are involved in the reference and measurement frames,
/// use the DifCamCalibratingInvDepthCost function.
///
template<typename CameraType>
struct CalibratingInvDepthCost {
  CalibratingInvDepthCost(const Eigen::Vector2d& z, const Eigen::Vector2d& z_ref)
      : z_(z), z_ref_(z_ref)
  {}

  template<typename T>
  bool operator()(const T* const _t_wv_r,
                  const T* const _t_wv_m,
                  const T* const _t_vc,
                  const T* const _params,
                  const T* const _rho,
                  T* _r) const {
    Eigen::Map<Eigen::Matrix<T, 2, 1>> r(_r); // residual vector

    const Eigen::Map<const Sophus::SE3Group<T>> t_wv_r(_t_wv_r); // ref pose
    const Eigen::Map<const Sophus::SE3Group<T>> t_wv_m(_t_wv_m); // meas pose

    const Eigen::Map<const Sophus::SE3Group<T>> t_vc(_t_vc);

    // Reference pixel.
    const Eigen::Matrix<T, 2, 1> pix_r = z_ref_.cast<T>();
    Eigen::Matrix<T, 3, 1> l_c_r;
    CameraType::template Unproject<T>(pix_r.data(), _params, l_c_r.data());

    // Transformation from reference to measurement camera.
    const Sophus::SE3Group<T> t_mr =
        t_vc.inverse() * t_wv_m.inverse() * t_wv_r * t_vc;
    const Eigen::Matrix<T, 3, 1> l_c =
        t_mr.rotationMatrix() * l_c_r + *_rho * t_mr.translation();
    Eigen::Matrix<T, 2, 1> pix;
    CameraType::template Project<T>(l_c.data(), _params, pix.data());
    // Compute the residual.
    r =  z_.cast<T>() - pix;
    return true;
  }

  Eigen::Vector2d z_;    // measurement location in the image.
  Eigen::Vector2d z_ref_;    // reference location of landmark in the first img.
};

///
/// \brief Cost function for inverse depth projection residuals. Optimizes the
/// inverse depth of the landmark, the pose of the reference and measurement
/// cameras,
///
template<typename CameraType>
struct DifCamCalibratingInvDepthCost {
  DifCamCalibratingInvDepthCost(const Eigen::Vector2d& z,
                                const Eigen::Vector2d& z_ref)
      : z_(z), z_ref_(z_ref)
  {}

  template<typename T>
  bool operator()(const T* const _t_wv_r,
                  const T* const _t_wv_m,
                  const T* const _t_vc_r,
                  const T* const _t_vc_m,
                  const T* const _params_r,
                  const T* const _params_m,
                  const T* const _rho,
                  T* _r) const {
    Eigen::Map<Eigen::Matrix<T, 2, 1>> r(_r); // residual vector

    const Eigen::Map<const Sophus::SE3Group<T>> t_wv_r(_t_wv_r); // ref pose
    const Eigen::Map<const Sophus::SE3Group<T>> t_wv_m(_t_wv_m); // meas pose

    const Eigen::Map<const Sophus::SE3Group<T>> t_vc_r(_t_vc_r);
    const Eigen::Map<const Sophus::SE3Group<T>> t_vc_m(_t_vc_m);

    // Reference pixel.
    const Eigen::Matrix<T, 2, 1> pix_r = z_ref_.cast<T>();
    Eigen::Matrix<T, 3, 1> l_c_r;
    CameraType::template Unproject<T>(pix_r.data(), _params_r, l_c_r.data());

    // Transformation from reference to measurement camera.
    const Sophus::SE3Group<T> t_mr =
        t_vc_m.inverse() * t_wv_m.inverse() * t_wv_r * t_vc_r;
    const Eigen::Matrix<T, 3, 1> l_c =
        t_mr.rotationMatrix() * l_c_r + *_rho * t_mr.translation();
    Eigen::Matrix<T, 2, 1> pix;
    CameraType::template Project<T>(l_c.data(), _params_m, pix.data());
    // Compute the residual.
    r =  z_.cast<T>() - pix;
    return true;
  }

  Eigen::Vector2d z_;    // measurement location in the image.
  Eigen::Vector2d z_ref_;    // reference location of landmark in the first img.
};

template <typename CamType>
ceres::ResidualBlockId AddProjectionResidualBlockToCeres(
    ceres::Problem& problem,
    std::shared_ptr<sdtrack::TrackerPose>& meas_pose,
    std::shared_ptr<sdtrack::TrackerPose>& ref_pose,
    std::shared_ptr<sdtrack::DenseTrack>& track,
    uint32_t meas_cam_id,
    uint32_t ref_cam_id,
    calibu::Rig<Scalar>& cam_rig,
    bool calibrating,
    const Eigen::Vector2d& z)
{
  ceres::ResidualBlockId residual_id;
  if (calibrating) {
    const Eigen::Vector2d& z_ref = track->keypoints[0][ref_cam_id].kp;
    if (meas_cam_id == ref_cam_id) {
      residual_id =
          problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<
            CalibratingInvDepthCost<CamType>, 2, 7, 7, 7,
            CamType::kParamSize, 1>(
              new CalibratingInvDepthCost<CamType>(z, z_ref)),
            NULL,
            ref_pose->t_wp.data(),
            meas_pose->t_wp.data(),
            cam_rig.t_wc_[ref_cam_id].data(),
            cam_rig.cameras_[ref_cam_id]->GetParams().data(),
            &track->ref_keypoint.rho);
    } else {
      residual_id =
          problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<
            DifCamCalibratingInvDepthCost<CamType>, 2, 7, 7, 7, 7,
            CamType::kParamSize, CamType::kParamSize, 1>(
              new DifCamCalibratingInvDepthCost<CamType>(z, z_ref)),
            NULL,
            ref_pose->t_wp.data(),
            meas_pose->t_wp.data(),
            cam_rig.t_wc_[ref_cam_id].data(),
            cam_rig.t_wc_[meas_cam_id].data(),
            cam_rig.cameras_[ref_cam_id]->GetParams().data(),
            cam_rig.cameras_[meas_cam_id]->GetParams().data(),
            &track->ref_keypoint.rho);
    }
  } else {
    residual_id =
        problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<
          InvepthCost<CamType>, 2, 7, 7, 1>(
            new InvepthCost<CamType>(
              cam_rig.t_wc_[ref_cam_id],       // Ref. cam extrinsics.
              cam_rig.t_wc_[meas_cam_id],  // Meas. cam extrinsics.
              track->ref_keypoint.ray,
              z,
              cam_rig.cameras_[meas_cam_id]->GetParams())),
          NULL,
          ref_pose->t_wp.data(),
          meas_pose->t_wp.data(),
          &track->ref_keypoint.rho);
  }

  return residual_id;
}

ceres::ResidualBlockId AddProjectionResidualToCeres(
    ceres::Problem& problem, std::shared_ptr<sdtrack::DenseTrack>& track,
    uint32_t ref_pose_id, uint32_t kp_id, uint32_t cam_id,
    std::vector<std::shared_ptr<sdtrack::TrackerPose>>& pose_vec,
    calibu::Rig<Scalar>& cam_rig, bool calibrating)
{
  const uint32_t ref_cam_id = 0;
  const Eigen::Vector2d& z = track->keypoints[kp_id][cam_id].kp;
  std::shared_ptr<sdtrack::TrackerPose>& meas_pose =
      pose_vec[ref_pose_id + kp_id];
  std::shared_ptr<sdtrack::TrackerPose>& ref_pose = pose_vec[ref_pose_id];
  ceres::ResidualBlockId residual_id;
  if (dynamic_cast<calibu::FovCamera<double>*>(cam_rig.cameras_[cam_id])) {
    residual_id =
    AddProjectionResidualBlockToCeres<calibu::FovCamera<double>>(
        problem, meas_pose, ref_pose, track, cam_id, ref_cam_id, cam_rig,
        calibrating, z);
  } else if (dynamic_cast<calibu::LinearCamera<double>*>(
               cam_rig.cameras_[cam_id])) {
    residual_id =
    AddProjectionResidualBlockToCeres<calibu::LinearCamera<double>>(
        problem, meas_pose, ref_pose, track, cam_id, ref_cam_id, cam_rig,
        calibrating, z);
  } else {
    LOG(FATAL) << "Unsupported camera type.";
  }

  return residual_id;
}



