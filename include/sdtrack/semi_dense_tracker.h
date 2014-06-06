#pragma once
#include <miniglog/logging.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "options.h"
#include "track.h"
#include "keypoint.h"
#include "utils.h"
//#include <Utils/PatchUtils.h>
#include "TicToc.h"
#include <calibu/cam/CameraRig.h>
#include "FeatureMask.h"
//#include <Utils/Utils.h>
#include <fstream>
#include <calibu/cam/camera_crtp_interop.h>
#include <Eigen/Eigenvalues>
#include <random>

#define LM_DIM 1

namespace sdtrack
{
  struct TrackerPose
  {
    std::list<std::shared_ptr<DenseTrack>> tracks;
    Sophus::SE3t t_wp;
    Eigen::Vector3t v_w;
    Eigen::Vector6t b;
    uint32_t opt_id;
    double time;
    uint32_t longest_track;
  };

  class SemiDenseTracker
  {
  public:
    SemiDenseTracker() :
    generator_(0) {}
    void Initialize(const KeypointOptions& keypoint_options,
                    const TrackerOptions& tracker_options,
                    calibu::Rig<Scalar> *rig);

    void StartNewLandmarks();

    double EvaluateTrackResiduals(uint32_t level,
        const std::vector<cv::Mat> &image_pyrmaid,
        std::list<std::shared_ptr<DenseTrack>> &tracks,
        bool transfer_jacobians = false,
        bool optimized_tracks_only = false);

    void TransformTrackTabs(const Sophus::SE3t& t_cb);

    void OptimizeTracks(int level = -1, bool optimize_landmarks = true,
                        bool optimize_pose = true);

    void PruneTracks();

    void OptimizePyramidLevel(uint32_t level,
        const std::vector<cv::Mat> &image_pyrmaid,
        std::list<std::shared_ptr<DenseTrack>> &tracks,
        const OptimizationOptions &options,
        OptimizationStats &stats);

    void AddImage(const cv::Mat &image,
                  const Sophus::SE3t& t_ab_guess);
    void AddKeyframe() { last_image_was_keyframe_ = true; }

    std::vector<cv::Mat>& GetImagePyramid() { return image_pyrmaid_; }
    void PruneOutliers();
    std::list<std::shared_ptr<DenseTrack>>& GetCurrentTracks()
      { return current_tracks_; }
    std::list<std::shared_ptr<DenseTrack>>& GetNewTracks()
      { return new_tracks_; }
    const Sophus::SE3t& t_ba() { return t_ba_; }
    void set_t_ba(const Sophus::SE3t& t_ba) { t_ba_ = t_ba; }
    uint32_t num_successful_tracks() { return num_successful_tracks_; }
    uint32_t longest_track_id() { return longest_track_id_; }

    void BackProjectTrack(std::shared_ptr<DenseTrack> track,
                          bool initialize_pixel_vals = false);

  private:
    uint32_t StartNewTracks(std::vector<cv::Mat>& image_pyrmaid,
                           std::vector<cv::KeyPoint>& cv_keypoints,
                           uint32_t num_to_start);


    void TransferPatch(std::shared_ptr<DenseTrack> track,
                       uint32_t level,
                       const Sophus::SE3t& t_ba,
                       calibu::CameraInterface<Scalar>* cam,
                       PatchTransfer &result, bool transfer_jacobians);


    inline double GetSubPix(const cv::Mat& image, double x, double y);

    void ReprojectTrackCenters();

    void ExtractKeypoints(const cv::Mat& image,
                          std::vector<cv::KeyPoint> &keypoints);

    bool IsKeypointValid(const cv::KeyPoint& kp,
                         uint32_t image_width, uint32_t image_height);

    bool IsReprojectionValid(const Eigen::Vector2t& pix, const cv::Mat &image);

    Sophus::SE3t t_ba_;
    bool last_image_was_keyframe_ = true;
    double lm_per_cell_;
    double average_track_length;
    TrackerOptions  tracker_options_;
    KeypointOptions keypoint_options_;
    std::vector<DenseKeypoint> previous_keypoints_;
    cv::FeatureDetector* detector_;
    std::list<std::shared_ptr<DenseTrack>> current_tracks_;
    std::list<std::shared_ptr<DenseTrack>> new_tracks_;
    uint32_t num_successful_tracks_;
    uint32_t next_track_id_;
    uint32_t longest_track_id_;
    FeatureMask mask_;
    std::vector<uint32_t> pyramid_patch_dims_;
    std::vector<std::vector<uint32_t>> pyramid_patch_corner_dims_;
    std::vector<std::vector<std::vector<double>>> pyramid_patch_interp_factors_;
    std::vector<Eigen::Vector2t> pyramid_coord_ratio_;
    std::vector<cv::Mat> image_pyrmaid_;
    std::vector<double> pyramid_error_thresholds_;
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> feature_cells_;
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> prev_feature_cells_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> feature_cell_rho_;
    calibu::Rig<Scalar>* camera_rig_;
    Eigen::Matrix4d generators_[6];
    std::default_random_engine generator_;
  };
}
