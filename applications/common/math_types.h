#pragma once

#include <Eigen/Eigen>
#include <sophus/se3.hpp>

namespace Sophus {
  typedef SE3Group<Scalar> SE3t;
}

namespace Eigen {
  typedef Matrix<Scalar,2,1> Vector2t;
  typedef Matrix<Scalar,3,1> Vector3t;
}

