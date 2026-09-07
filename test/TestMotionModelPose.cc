#include <gtest/gtest.h>

#include "MotionModelPose.h"

#include <opencv2/core.hpp>

#include <limits>

namespace {

double orthogonalityError(const cv::Mat &pose)
{
    const cv::Mat rotation = pose(cv::Rect(0, 0, 3, 3));
    return cv::norm(rotation.t() * rotation -
                    cv::Mat::eye(3, 3, rotation.type()));
}

TEST(MotionModelPose, ProjectsFinitePoseToProperRigidTransform)
{
    cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);
    pose.at<float>(0, 0) = 1.002F;
    pose.at<float>(0, 1) = -0.04F;
    pose.at<float>(1, 0) = 0.041F;
    pose.at<float>(1, 1) = 0.998F;
    pose.at<float>(0, 3) = 1.25F;
    pose.at<float>(1, 3) = -0.5F;
    pose.at<float>(2, 3) = 0.75F;

    cv::Mat projected;
    ASSERT_TRUE(ORB_SLAM3::ProjectMotionModelPoseToSE3(pose, projected));

    EXPECT_LT(orthogonalityError(projected), 1e-5);
    EXPECT_NEAR(cv::determinant(projected(cv::Rect(0, 0, 3, 3))), 1.0, 1e-5);
    EXPECT_FLOAT_EQ(projected.at<float>(0, 3), 1.25F);
    EXPECT_FLOAT_EQ(projected.at<float>(1, 3), -0.5F);
    EXPECT_FLOAT_EQ(projected.at<float>(2, 3), 0.75F);
    EXPECT_FLOAT_EQ(projected.at<float>(3, 0), 0.0F);
    EXPECT_FLOAT_EQ(projected.at<float>(3, 3), 1.0F);
}

TEST(MotionModelPose, RepeatedPredictionDoesNotAmplifyRotationError)
{
    cv::Mat velocity = cv::Mat::eye(4, 4, CV_32F);
    velocity.at<float>(0, 0) = 1.0005F;
    velocity.at<float>(0, 1) = -0.02F;
    velocity.at<float>(1, 0) = 0.0205F;
    velocity.at<float>(1, 1) = 0.9995F;
    cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);

    ASSERT_TRUE(ORB_SLAM3::ProjectMotionModelPoseToSE3(velocity, velocity));
    for (int frame = 0; frame < 1000; ++frame) {
        const cv::Mat prediction = velocity * pose;
        ASSERT_TRUE(ORB_SLAM3::ProjectMotionModelPoseToSE3(prediction, pose));
    }

    EXPECT_LT(orthogonalityError(pose), 1e-5);
    EXPECT_NEAR(cv::determinant(pose(cv::Rect(0, 0, 3, 3))), 1.0, 1e-5);
}

TEST(MotionModelPose, PublishesDynamicOutputsWithoutClaimingAquaImuState)
{
    EXPECT_TRUE(ORB_SLAM3::CanPublishEstimatorOutputs(true, false, true));
    EXPECT_TRUE(ORB_SLAM3::CanPublishEstimatorOutputs(true, true, true));
    EXPECT_TRUE(ORB_SLAM3::CanPublishEstimatorOutputs(false, true, true));
    EXPECT_FALSE(ORB_SLAM3::CanPublishEstimatorOutputs(false, false, true));
    EXPECT_FALSE(ORB_SLAM3::CanPublishEstimatorOutputs(true, false, false));
}

TEST(MotionModelPose, DynamicStereoInitializationRequiresPersistentDepthSupport)
{
    EXPECT_FALSE(ORB_SLAM3::ShouldCommitStereoInitialization(
        true, false, 80, 40, 0, 0));
    EXPECT_FALSE(ORB_SLAM3::ShouldCommitStereoInitialization(
        true, true, 80, 40, 0, 14));
    EXPECT_FALSE(ORB_SLAM3::ShouldCommitStereoInitialization(
        true, true, 80, 40, 20, 0));
    EXPECT_TRUE(ORB_SLAM3::ShouldCommitStereoInitialization(
        true, true, 1, 1, 1, 1));
}

TEST(MotionModelPose, StereoInitializationStillRequiresCurrentFrameGeometry)
{
    EXPECT_FALSE(ORB_SLAM3::ShouldCommitStereoInitialization(
        true, true, 0, 40, 20, 20));
    EXPECT_FALSE(ORB_SLAM3::ShouldCommitStereoInitialization(
        true, true, 80, 0, 20, 20));
    EXPECT_TRUE(ORB_SLAM3::ShouldCommitStereoInitialization(
        false, false, 80, 40, 0, 0));
}

}  // namespace
