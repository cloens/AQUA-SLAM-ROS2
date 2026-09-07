#ifndef MOTION_MODEL_POSE_H
#define MOTION_MODEL_POSE_H

#include <opencv2/core/mat.hpp>

namespace ORB_SLAM3 {

bool ProjectMotionModelPoseToSE3(const cv::Mat &pose, cv::Mat &projected);
bool CanPublishEstimatorOutputs(bool dynamicBackend,
                                bool aquaImuInitialized,
                                bool hasKeyframes);
bool ShouldCommitStereoInitialization(bool dynamicBackend,
                                      bool hasCandidate,
                                      int currentFeatures,
                                      int currentDepthPoints,
                                      int persistentDepthMatches,
                                      int persistentDepthBowMatches);
}  // namespace ORB_SLAM3

#endif
