#pragma once

#include <sophus/se3.hpp>

namespace Sophus {
namespace sdtrack {

struct AutoDiffLocalParamSO3 {
  
  template <typename T>
  bool operator()(const T* x_raw, const T* delta_raw, T* x_plus_delta_raw) const {
    const Eigen::Map<const Sophus::SO3Group<T>> x(x_raw);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> delta(delta_raw);
    Eigen::Map<Sophus::SO3Group<T>> x_plus_delta(x_plus_delta_raw);
    x_plus_delta = x * Sophus::SO3Group<T>::exp(delta);
    return true;
  }
};

}
}
