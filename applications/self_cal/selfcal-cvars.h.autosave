#pragma once

#include "../common/common_cvars.h"

static int& ba_debug_level =
    CVarUtils::CreateCVar<>("debug.BaDebugLevel", -1, "");
static int& vi_ba_debug_level =
    CVarUtils::CreateCVar<>("debug.ViBaDebugLevel", -1, "");
static int& aac_ba_debug_level =
    CVarUtils::CreateCVar<>("debug.AacBaDebugLevel", -1, "");
static int& selfcal_ba_debug_level =
    CVarUtils::CreateCVar<>("debug.SelfcalBaDebugLevel", -1, "");
static int& selfcal_debug_level =
    CVarUtils::CreateCVar<>("debug.SelfcalDebugLevel", 2, "");


static double& gyro_sigma =
    CVarUtils::CreateCVar<>("sd.GyroUncertainty", 1.3088444e-1 /*IMU_GYRO_SIGMA*/, "");
static double& gyro_bias_sigma =
    CVarUtils::CreateCVar<>("sd.GyroBiasUncertainty", IMU_GYRO_BIAS_SIGMA, "");
static double& accel_sigma =
    CVarUtils::CreateCVar<>("sd.AccelUncertainty", IMU_ACCEL_SIGMA, "");
static double& accel_bias_sigma =
    CVarUtils::CreateCVar<>("sd.AccelBiasUncertainty", IMU_ACCEL_BIAS_SIGMA, "");

static Eigen::Vector3d& gravity_vector =
    CVarUtils::CreateCVar<>("sd.Gravity",
                            (Eigen::Vector3d)(Eigen::Vector3d(0, 0, -1) *
                                              ba::Gravity), "");
static uint32_t& num_ba_poses =
    CVarUtils::CreateCVar<>("sd.NumBAPoses",10u, "");
static uint32_t& num_selfcal_ba_iterations =
    CVarUtils::CreateCVar<>("sd.NumSelfCalBAIterations",10u, "");
static uint32_t& min_poses_for_imu =
    CVarUtils::CreateCVar<>("sd.MinPosesForImu", 20u, "");
static uint32_t& min_poses_for_imu_rotation_init =
    CVarUtils::CreateCVar<>("sd.MinPosesForImuRotationInit", 20u, "");
static uint32_t& min_poses_for_camera =
    CVarUtils::CreateCVar<>("sd.MinPosesForCamera", 10u, "");
static double& imu_extra_integration_time =
    CVarUtils::CreateCVar<>("sd.ImuExtraIntegrationTime", 0.3, "");
static double& imu_time_offset =
    CVarUtils::CreateCVar<>("sd.ImuTimeOffset", 0.0, "");
static double& adaptive_threshold =
    CVarUtils::CreateCVar<>("sd.AdaptiveThreshold", 0.1, "");
static bool& use_imu_for_guess =
    CVarUtils::CreateCVar<>("sd.UseImuForGuess", true, "");
static bool& do_async_ba =
    CVarUtils::CreateCVar<>("sd.DoAsyncBA", true, "");
static bool& do_serial_ac =
    CVarUtils::CreateCVar<>("sd.DoSerialAC", false, "");
static bool& use_imu_measurements =
    CVarUtils::CreateCVar<>("sd.UseImu", true, "");
static bool& do_cam_self_cal =
    CVarUtils::CreateCVar<>("sd.DoCamSelfCal", false, "");
static bool& show_panel =
    CVarUtils::CreateCVar<>("sd.ShowPanel", true, "");
static bool& do_imu_self_cal =
    CVarUtils::CreateCVar<>("sd.DoIMUSelfCal", true, "");
static bool& use_batch_estimates =
    CVarUtils::CreateCVar<>("sd.UseBatchEstimates", false, "");
static bool& do_async_pq =
    CVarUtils::CreateCVar<>("sd.DoAsyncPQ", true, "");
static uint32_t& num_aac_poses =
    CVarUtils::CreateCVar<>("sd.NumAACPoses",20u, "");
static bool& do_keyframing =
    CVarUtils::CreateCVar<>("sd.DoKeyframing", true, "");
static bool& do_adaptive =
    CVarUtils::CreateCVar<>("sd.DoAdaptiveConditioning", true, "");
static int& num_ba_iterations =
    CVarUtils::CreateCVar<>("sd.NumBAIterations", 200, "");
static bool& reset_outliers =
    CVarUtils::CreateCVar<>("sd.ResetOutliers", false, "");
static bool& use_dogleg =
    CVarUtils::CreateCVar<>("sd.UseDogleg", true, "");
static bool& use_robust_norm_for_proj =
    CVarUtils::CreateCVar<>("sd.UseRobustNormForProj", true, "");
static double& outlier_threshold =
    CVarUtils::CreateCVar<>("sd.OutlierThreshold", 1.0, "");
static bool& do_outlier_rejection =
    CVarUtils::CreateCVar<>("sd.DoOutlierRejection", true, "");
static int& pyramid_levels =
    CVarUtils::CreateCVar<>("sd.PyramidLevels", 4, "");
static int& patch_size =
    CVarUtils::CreateCVar<>("sd.PatchSize", 9, "");
static int& num_features =
    CVarUtils::CreateCVar<>("sd.NumFeatures",128, "");
static int& feature_cells =
    CVarUtils::CreateCVar<>("sd.FeatureCells",8, "");
static double& ncc_threshold =
    CVarUtils::CreateCVar<>("sd.NCCThreshold", 0.875, "");
static bool& regularize_biases_in_batch =
    CVarUtils::CreateCVar<>("sd.RegularizeBiasesInBatch", false, "");

