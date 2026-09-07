#include "MotionModelPose.h"

#include <opencv2/core.hpp>

namespace ORB_SLAM3 {

bool ProjectMotionModelPoseToSE3(const cv::Mat &pose, cv::Mat &projected)
{
    if (pose.rows != 4 || pose.cols != 4 || pose.channels() != 1 ||
        (pose.depth() != CV_32F && pose.depth() != CV_64F) ||
        !cv::checkRange(pose)) {
        return false;
    }

    cv::Mat source64;
    pose.convertTo(source64, CV_64F);
    const cv::Mat rotation = source64(cv::Rect(0, 0, 3, 3)).clone();
    cv::SVD svd(rotation, cv::SVD::FULL_UV);
    cv::Mat properRotation = svd.u * svd.vt;
    if (cv::determinant(properRotation) < 0.0) {
        svd.u.col(2) *= -1.0;
        properRotation = svd.u * svd.vt;
    }

    cv::Mat rigid64 = cv::Mat::eye(4, 4, CV_64F);
    properRotation.copyTo(rigid64(cv::Rect(0, 0, 3, 3)));
    source64(cv::Rect(3, 0, 1, 3)).copyTo(
        rigid64(cv::Rect(3, 0, 1, 3)));
    rigid64.convertTo(projected, pose.type());
    return cv::checkRange(projected);
}

bool CanPublishEstimatorOutputs(bool dynamicBackend,
                                bool aquaImuInitialized,
                                bool hasKeyframes)
{
    return hasKeyframes && (dynamicBackend || aquaImuInitialized);
}

bool ShouldCommitStereoInitialization(bool dynamicBackend,
                                      bool hasCandidate,
                                      int currentFeatures,
                                      int currentDepthPoints,
                                      int persistentDepthMatches,
                                      int persistentDepthBowMatches)
{
    if (dynamicBackend)
        return hasCandidate && currentFeatures > 0 && currentDepthPoints > 0 &&
               persistentDepthMatches > 0 && persistentDepthBowMatches > 0;
    return currentFeatures > 40 && currentDepthPoints > 20;
}

}  // namespace ORB_SLAM3
