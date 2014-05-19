#pragma once

#include <sdtrack/track.h>
#include <sdtrack/utils.h>
#include <sdtrack/semi_dense_tracker.h>

void GetBaPoseRange(
    const std::vector<std::shared_ptr<sdtrack::TrackerPose>>& poses,
    const uint32_t num_active_poses, uint32_t& start_pose,
    uint32_t& start_active_pose)
{
  start_active_pose = poses.size() > num_active_poses ?
      poses.size() - num_active_poses : 0;
  start_pose = poses.size();
  for (uint32_t ii = start_active_pose ; ii < poses.size() ; ++ii) {
    std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
    // std::cerr << "Start id: " << start_pose << " pose longest track " <<
    //              pose->longest_track << " for pose id " << ii << std::endl;
    start_pose = std::min(ii - (pose->longest_track - 1), start_pose);
  }

  std::cerr << "Num poses: " << poses.size() << " start pose " <<
               start_pose << " start active pose " << start_active_pose <<
               std::endl;
}
