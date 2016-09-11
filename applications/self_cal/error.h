#ifndef ERROR_H
#define ERROR_H

class Error {
  public:
  Error(){

  }

  Eigen::Vector3d& Translation(){
    return translation;
  }

  Eigen::Vector3d& Rotation(){
    return rotation;
  }

  double& MaxTransError(){
    return max_trans_error;
  }

  double& MaxRotError(){
    return max_rot_error;
  }

  unsigned& NumPoses(){
    return num_poses;
  }

  double& DistanceTraveled(){
    return distance_traveled;
  }

  double& PercentAvgTranslationError(){
    return percent_avg_trans_error;
  }

  double GetAverageTransError(){
    if(num_poses > 0){
      return translation.norm()/num_poses;
    }else{
      return -1;
    }
  }

  double GetPercentAverageTansError(){
    if(num_poses > 0){
      return percent_avg_trans_error/num_poses;
    }else{
      return -1;
    }
  }

  double GetAverageRotError(){
    if(num_poses > 0){
      return rotation.norm()/num_poses;
    }else{
      return -1;
    }
  }

  private:
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d rotation = Eigen::Vector3d::Zero();
    double max_trans_error = 0;
    double max_rot_error = 0;
    unsigned num_poses = 0;
    double distance_traveled = 0;
    double percent_avg_trans_error = 0;
};

#endif // ERROR_H

