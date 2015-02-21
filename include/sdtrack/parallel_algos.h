#include "semi_dense_tracker.h"
#include <tbb/parallel_reduce.h>
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
        tbb::split
        );


    void join(OptimizeTrack& other);

    void operator() (const tbb::blocked_range<int>& r);
  };
}
