#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <unsupported/Eigen/MatrixFunctions>
#include <ceres/ceres.h>
#include "sophus/so3.hpp"

using ceres::SizedCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::HuberLoss;

namespace sdtrack {

////////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
struct NoiselessRotationCostFunctor {
  NoiselessRotationCostFunctor(const Sophus::SO3d& R_a,
                      const Sophus::SO3d& R_b)
    : R_a(R_a),
      R_b(R_b)
  {
  }

  template<typename T>
  bool operator()(const T* const R_ab_, T* residuals) const{
    // Pose to optimize over
    const Eigen::Map< const Sophus::SO3Group<T> > R_ab(R_ab_);

    // Residual vector in tangent space
    Eigen::Map< Eigen::Matrix<T, 3, 1> > rotation_residuals(residuals);

    rotation_residuals = Sophus::SO3Group<T>::log((R_a.cast<T>()*R_ab).inverse()*
                         R_ab*R_b.cast<T>());
    return true;
  }

  Sophus::SO3d R_a;
  Sophus::SO3d R_b;

};



} // namespace sdtrack
