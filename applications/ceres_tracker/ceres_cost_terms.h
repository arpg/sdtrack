#pragma once
#include <calibu/cam/camera_crtp.h>

////////////////////////////////////////////////////////////////////////////////
/// \brief The InverseDepthCostFunctor struct
template<typename CameraType>
struct InverseDepthCostFunctor {
  InverseDepthCostFunctor(Sophus::SE3d t_vc_r, Sophus::SE3d t_vc_m,
                          Eigen::Vector3d l_v, Eigen::Vector2d z,
                          Eigen::VectorXd params)
      : t_vc_r_(t_vc_r), t_cv_m_(t_vc_m.inverse()), l_v_(l_v), z_(z),
        params_(params)
  {}

  template<typename T>
  bool operator()(const T* const _t_wv_r, const T* const _t_wv_m,
                  const T* const _rho, T* _r) const {
    Eigen::Map<Eigen::Matrix<T, 2, 1>> r(_r); // residual vector
    const Eigen::Matrix<T, Eigen::Dynamic, 1> params_t = params_.cast<T>();

    const Eigen::Map<const Sophus::SE3Group<T>> t_wv_r(_t_wv_r); // ref pose
    const Eigen::Map<const Sophus::SE3Group<T>> t_wv_m(_t_wv_m); // meas pose

    // Transformation from reference to measurement camera.
    const Sophus::SE3Group<T> t_mr =
        t_cv_m_.cast<T>() * t_wv_m.inverse() * t_wv_r * t_vc_r_.cast<T>();
    const Eigen::Matrix<T, 3, 1> l_c =
        t_mr.rotationMatrix() * l_v_.cast<T>() + *_rho * t_mr.translation();
    Eigen::Matrix<T, 2, 1> pix;
    CameraType::template Project<T>(l_c.data(), params_t.data(), pix.data());
    // Compute the residual.
    r =  z_.cast<T>() - pix;
    return true;
  }

  Sophus::SE3d t_vc_r_;    // camera to vehicle transformation
  Sophus::SE3d t_cv_m_;
  Eigen::Vector3d l_v_;  // position of landmark in the vehicle frame
  Eigen::Vector2d z_;    // measurement location in the image
  Eigen::VectorXd params_;
};

