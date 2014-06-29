#pragma once
#include <vector>
#include <Eigen/Eigen>
#include "keypoint.h"
#include <sophus/se3.hpp>

namespace sdtrack
{
  struct Track
  {
    static const uint32_t kUntrackedKeypoint = UINT_MAX;
    uint32_t id;
    std::vector<Eigen::Vector2d> keypoints;
    std::vector<unsigned char> descriptor;
    bool tracked;
  };

  struct PatchTransfer
  {
    Eigen::Vector2d center_projection;
    Eigen::Matrix<double, 2, 3> center_dprojection;
    std::vector<Eigen::Vector2d> valid_projections;
    std::vector<unsigned int> valid_rays;
    std::vector<Eigen::Matrix<double, 2, 4>> dprojections;
    uint32_t pixels_attempted;
    double mean_value;
    uint32_t level;
    double dimension;
  };

  struct DenseTrack
  {
    DenseTrack(uint32_t num_pyrmaid_levels,
               const std::vector<uint32_t>& pyramid_dims):
      ref_keypoint(num_pyrmaid_levels, pyramid_dims)
    {
      ref_keypoint.track = this;
      external_id.resize(2);
    }

    PatchTransfer transfer;
    uint32_t tracked_pixels;
    uint32_t pixels_attempted;
    double jtj = 0;
    double rmse = 0;
    double ncc = 1.0;
    double center_error = 0;
    uint32_t opt_id;
    uint32_t residual_offset;
    std::vector<uint32_t> external_id;
    uint32_t id;
    uint32_t num_good_tracked_frames = 0;
    DenseKeypoint ref_keypoint;
    std::vector<Eigen::Vector2d> keypoints;
    std::vector<uint32_t> keypoint_external_data;
    std::vector<bool> keypoints_tracked;
    bool residual_used = false;
    bool tracked = false;
    bool is_new = true;
    bool is_outlier = false;
    bool inverse_depth_ray = false;
    Sophus::SE3d t_ba;

    uint32_t external_data = UINT_MAX;
    uint32_t external_data2 = UINT_MAX;
  };
}
