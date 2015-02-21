#include "semi_dense_tracker.h"
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace sdtrack {
  class OptimizeTrack {
  public:
    SemiDenseTracker& tracker;
    const PyramidLevelOptimizationOptions& options;
    OptimizationStats& stats;
    std::vector<std::shared_ptr<DenseTrack>>& tracks;
    uint32_t level;
    const std::vector<std::vector<cv::Mat>>& image_pyrmaid;
    int g_sdtrack_debug;

    // Reduced quantities.
    Eigen::Matrix<double, 6, 6> u;
    Eigen::Matrix<double, 6, 1> r_p;
    double residual;

    OptimizeTrack(SemiDenseTracker& tracker_ref,
                  const PyramidLevelOptimizationOptions& opt,
                  std::vector<std::shared_ptr<DenseTrack>>& track_vec,
                  OptimizationStats& opt_stats,
                  uint32_t lvl,
                  const std::vector<std::vector<cv::Mat>>& pyr,
                  int debug_level);

    OptimizeTrack(
        const OptimizeTrack& other,
        tbb::split);


    void join(OptimizeTrack& other);

    void operator() (const tbb::blocked_range<int>& r);
  };

  class ParallelExtractKeypoints {
  public:
    SemiDenseTracker& tracker;
    const cv::Mat& image;
    const std::vector<cv::Rect>& bounds;

    // Reduced quantities.
    std::vector<cv::KeyPoint> keypoints;

    ParallelExtractKeypoints(SemiDenseTracker& tracker_ref,
                             const cv::Mat& img_ref,
                             const std::vector<cv::Rect>& bounds_ref);

    ParallelExtractKeypoints(const ParallelExtractKeypoints& other,
                             tbb::split);

    void join(ParallelExtractKeypoints& other);

    void operator() (const tbb::blocked_range<int>& r);
  };

  class Parallel2dAlignment {
  public:
    SemiDenseTracker& tracker;
    const AlignmentOptions &options;
    const std::vector<std::vector<cv::Mat>>& image_pyramid;
    std::vector<std::shared_ptr<DenseTrack>>& tracks;
    Sophus::SE3d t_cv;
    uint32_t level;
    uint32_t cam_id;

    Parallel2dAlignment(SemiDenseTracker& tracker_ref,
                        const AlignmentOptions &options_ref,
                        const std::vector<std::vector<cv::Mat>>& pyr,
                        std::vector<std::shared_ptr<DenseTrack>>& tracks_v,
                        Sophus::SE3d tcv,
                        uint32_t lvl,
                        uint32_t cam);

    void operator() (const tbb::blocked_range<int>& r) const;
  };
}
