#include <gtest/gtest.h>

#include <limits>

#include <opencv2/core.hpp>

#include "Atlas.h"
#include "InertialMaturity.h"
#include "ImuTypes.h"

namespace {

TEST(ImuPreintegration, KeepsRotationMatricesInFloatPrecision)
{
    const cv::Mat transform = cv::Mat::eye(4, 4, CV_32F);
    const ORB_SLAM3::IMU::Calib calibration(
        transform, transform, 1.0e-3F, 1.0e-2F, 1.0e-5F, 1.0e-4F);
    ORB_SLAM3::IMU::Preintegrated preintegrated(
        ORB_SLAM3::IMU::Bias(), calibration);

    EXPECT_NO_THROW(preintegrated.IntegrateNewMeasurement(
        cv::Point3f(0.0F, 0.0F, 9.81F),
        cv::Point3f(0.01F, -0.02F, 0.03F),
        0.01F));
    EXPECT_EQ(preintegrated.dR.type(), CV_32F);
}

TEST(ImuPreintegration, RejectsNonPositiveAndNonFiniteTimeSteps)
{
    const cv::Mat transform = cv::Mat::eye(4, 4, CV_32F);
    const ORB_SLAM3::IMU::Calib calibration(
        transform, transform, 1.0e-3F, 1.0e-2F, 1.0e-5F, 1.0e-4F);
    ORB_SLAM3::IMU::Preintegrated preintegrated(
        ORB_SLAM3::IMU::Bias(), calibration);
    const cv::Point3f acceleration(0.0F, 0.0F, 9.81F);
    const cv::Point3f angular_velocity(0.01F, -0.02F, 0.03F);

    preintegrated.IntegrateNewMeasurement(acceleration, angular_velocity, 0.0F);
    preintegrated.IntegrateNewMeasurement(
        acceleration, angular_velocity,
        std::numeric_limits<float>::quiet_NaN());
    preintegrated.IntegrateNewMeasurement(acceleration, angular_velocity, -0.01F);

    EXPECT_FLOAT_EQ(preintegrated.dT, 0.0F);
    EXPECT_TRUE(cv::checkRange(preintegrated.dR));
    EXPECT_TRUE(cv::checkRange(preintegrated.dV));
    EXPECT_TRUE(cv::checkRange(preintegrated.dP));
    EXPECT_TRUE(cv::checkRange(preintegrated.C));
}

TEST(ImuPreintegration, RejectsNonFiniteMeasurements)
{
    const cv::Mat transform = cv::Mat::eye(4, 4, CV_32F);
    const ORB_SLAM3::IMU::Calib calibration(
        transform, transform, 1.0e-3F, 1.0e-2F, 1.0e-5F, 1.0e-4F);
    ORB_SLAM3::IMU::Preintegrated preintegrated(
        ORB_SLAM3::IMU::Bias(), calibration);
    const cv::Point3f angular_velocity(0.01F, -0.02F, 0.03F);

    preintegrated.IntegrateNewMeasurement(
        cv::Point3f(std::numeric_limits<float>::infinity(), 0.0F, 9.81F),
        angular_velocity, 0.01F);
    preintegrated.IntegrateNewMeasurement(
        cv::Point3f(0.0F, 0.0F, 9.81F),
        cv::Point3f(0.01F, std::numeric_limits<float>::quiet_NaN(), 0.03F),
        0.01F);

    EXPECT_FLOAT_EQ(preintegrated.dT, 0.0F);
    EXPECT_TRUE(cv::checkRange(preintegrated.dR));
    EXPECT_TRUE(cv::checkRange(preintegrated.dV));
    EXPECT_TRUE(cv::checkRange(preintegrated.dP));
    EXPECT_TRUE(cv::checkRange(preintegrated.C));
}

TEST(InertialMaturity, AdvancesThroughBothRefinementStagesOnce)
{
    using ORB_SLAM3::InertialRefinementStage;
    using ORB_SLAM3::NextInertialRefinementStage;

    EXPECT_EQ(NextInertialRefinementStage(4.99, false, false),
              InertialRefinementStage::None);
    EXPECT_EQ(NextInertialRefinementStage(5.0, false, false),
              InertialRefinementStage::BA1);
    EXPECT_EQ(NextInertialRefinementStage(14.99, true, false),
              InertialRefinementStage::None);
    EXPECT_EQ(NextInertialRefinementStage(15.0, true, false),
              InertialRefinementStage::BA2);
    EXPECT_EQ(NextInertialRefinementStage(60.0, true, true),
              InertialRefinementStage::None);
}

TEST(InertialMaturity, DoesNotSkipFirstRefinementStage)
{
    EXPECT_EQ(ORB_SLAM3::NextInertialRefinementStage(60.0, false, false),
              ORB_SLAM3::InertialRefinementStage::BA1);
}

TEST(Atlas, ReportsImuInitializationForCurrentMap)
{
    ORB_SLAM3::Atlas atlas(0);
    atlas.SetImuInitialized();
    EXPECT_TRUE(atlas.isImuInitialized());

    atlas.CreateNewMap();
    EXPECT_FALSE(atlas.isImuInitialized());

    atlas.SetImuInitialized();
    EXPECT_TRUE(atlas.isImuInitialized());
}

}  // namespace
