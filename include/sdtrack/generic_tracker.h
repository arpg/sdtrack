#pragma once
#include <HAL/Camera/CameraDevice.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <CommonFrontEnd/FundamentalMatrix.h>
#include "options.h"
#include "track.h"
#include "keypoint.h"
#include "utils.h"
#include <calibu/cam/CameraRig.h>

namespace rslam
{
  class GenericTracker
  {
  public:
    void Initialize(const KeypointOptions& keypoint_options,
                    const DescriptorOptions& descriptor_options,
                    const TrackerOptions& tracker_options,
                    calibu::CameraRigT<Scalar>* rig);
    void ExtractKeypoints(const cv::Mat& image,
                          std::vector<cv::KeyPoint> &keypoints);

    void ComputeDescriptors(const cv::Mat& image,
                            std::vector<cv::KeyPoint> &keypoints,
                            cv::Mat& descriptors);
    void AddImage(const cv::Mat &image);
    std::list<Track>& GetCurrentTracks() { return current_tracks_; }

    double GetKeypointMatchScore(const Keypoint &kp1, const Keypoint &kp2);
  private:
    double match_score_threshold_;
    TrackerOptions  tracker_options_;
    std::vector<Keypoint> current_keypoints_;
    std::vector<Keypoint> previous_keypoints_;
    cv::FeatureDetector* detector_;
    cv::DescriptorExtractor* descriptor_extractor_;
    std::list<Track> current_tracks_;
    uint32_t next_track_id_;
    cv::HammingSse hsse_;
    cv::L2<float> sl2_;
    calibu::CameraRigT<Scalar> *rig_;
  };
}
