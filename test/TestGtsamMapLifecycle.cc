#include <gtest/gtest.h>

#include "Frame.h"
#include "GtsamMapAdapter.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"

#include <cmath>

namespace {

void configureFrame(ORB_SLAM3::Frame& frame, std::uint64_t id,
                    double timestampSec, int octave, float quality)
{
    frame.mnId = id;
    frame.mTimeStamp = timestampSec;
    frame.N = 1;
    frame.Nleft = -1;
    frame.mpCamera = nullptr;
    frame.mpCamera2 = nullptr;
    frame.mb = 0.1F;
    frame.mbf = 40.0F;
    frame.mnScaleLevels = 3;
    frame.mvScaleFactors = {1.0F, 1.5F, 2.0F};
    frame.mvLevelSigma2 = {1.0F, 2.0F, 4.0F};
    frame.mvKeysUn.emplace_back(cv::Point2f(30.0F, 20.0F), 1.0F);
    frame.mvKeysUn.back().octave = octave;
    frame.mvKeysUn.back().response = quality;
    frame.mvuRight.push_back(20.0F);
    frame.mvpMapPoints.push_back(nullptr);
    frame.mTcw = cv::Mat::eye(4, 4, CV_32F);
}

TEST(GtsamMapLifecycle, CarriesStereoQualityAndLandmarkDegree)
{
    ORB_SLAM3::Map map;
    ORB_SLAM3::Frame firstFrame;
    ORB_SLAM3::Frame secondFrame;
    configureFrame(firstFrame, 7, 1.0, 2, 0.8F);
    configureFrame(secondFrame, 8, 2.0, 1, 0.7F);
    ORB_SLAM3::KeyFrame first(firstFrame, &map, nullptr);
    ORB_SLAM3::KeyFrame second(secondFrame, &map, nullptr);
    ORB_SLAM3::MapPoint landmark(
        (cv::Mat_<float>(3, 1) << 1.0F, 2.0F, 3.0F), &first, &map);
    landmark.mnId = 10;
    first.AddMapPoint(&landmark, 0);
    second.AddMapPoint(&landmark, 0);
    landmark.AddObservation(&first, 0);
    landmark.AddObservation(&second, 0);
    landmark.UpdateNormalAndDepth();
    map.AddKeyFrame(&first);
    map.AddKeyFrame(&second);
    map.AddMapPoint(&landmark);

    const auto snapshot = ORB_SLAM3::GtsamMapAdapter::snapshot(map, 0);

    ASSERT_EQ(snapshot.landmarks.size(), 1U);
    ASSERT_EQ(snapshot.observations.size(), 2U);
    EXPECT_EQ(snapshot.landmarks.front().observationDegree, 2U);
    EXPECT_EQ(snapshot.observations.front().octave, 2);
    EXPECT_DOUBLE_EQ(snapshot.observations.front().octaveVariancePx2, 4.0);
    EXPECT_FLOAT_EQ(static_cast<float>(snapshot.observations.front().quality),
                    0.8F);
    EXPECT_DOUBLE_EQ(snapshot.observations.front().stereoSkewPx, 0.0);
    EXPECT_TRUE(std::isfinite(snapshot.landmarks.front().minimumDepthMeters));
    EXPECT_TRUE(std::isfinite(snapshot.landmarks.front().maximumDepthMeters));
    EXPECT_GT(snapshot.landmarks.front().minimumDepthMeters, 0.0);
    EXPECT_GE(snapshot.landmarks.front().maximumDepthMeters,
              snapshot.landmarks.front().minimumDepthMeters);
}

TEST(GtsamMapLifecycle, RejectsObsoleteResultBeforeMapWrite)
{
    ORB_SLAM3::Map map;
    ORB_SLAM3::BackendMapResult result;
    result.accepted = true;
    result.sourceVersion = 7;
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::commit(result, map, 8));
}

TEST(GtsamMapLifecycle, RejectsUnacceptedResult)
{
    ORB_SLAM3::Map map;
    ORB_SLAM3::BackendMapResult result;
    result.sourceVersion = 0;
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::commit(result, map, 0));
}

TEST(GtsamMapLifecycle, AppendsAcceptedLoopConstraintFromSnapshotPoses)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    ORB_SLAM3::BackendKeyframeState first;
    first.id = 10;
    first.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                  0.0F, 1.0F, 0.0F, 0.0F,
                  0.0F, 0.0F, 1.0F, 0.0F,
                  0.0F, 0.0F, 0.0F, 1.0F};
    ORB_SLAM3::BackendKeyframeState second = first;
    second.id = 11;
    second.pose[3] = 0.2F;
    snapshot.keyframes = {first, second};

    ASSERT_TRUE(ORB_SLAM3::GtsamMapAdapter::appendAcceptedLoopEdge(
        snapshot, 10, 11, 0.03));
    ASSERT_EQ(snapshot.graphEdges.size(), 1U);
    EXPECT_EQ(snapshot.graphEdges.front().kind,
              ORB_SLAM3::BackendGraphEdgeKind::Loop);
    EXPECT_NEAR(snapshot.graphEdges.front().relativePose[3], 0.2F, 1e-6F);
}

TEST(GtsamMapLifecycle, ProjectsOnlyConstraintsThroughCommittedWatermark)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    snapshot.version = 4;
    ORB_SLAM3::BackendKeyframeState committed;
    committed.id = 10;
    committed.timestampSec = 1.0;
    committed.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                      0.0F, 1.0F, 0.0F, 0.0F,
                      0.0F, 0.0F, 1.0F, 0.0F,
                      0.0F, 0.0F, 0.0F, 1.0F};
    ORB_SLAM3::BackendKeyframeState future = committed;
    future.id = 11;
    future.timestampSec = 2.0;
    snapshot.keyframes = {committed, future};
    snapshot.landmarks = {{7, {1.0F, 2.0F, 3.0F}},
                          {8, {4.0F, 5.0F, 6.0F}}};
    snapshot.observations = {{10, 7, 30.0, 20.0, 10.0, 1.0},
                             {11, 7, 31.0, 21.0, 11.0, 1.0},
                             {11, 8, 32.0, 22.0, 12.0, 1.0}};
    snapshot.graphEdges = {{10, 11, ORB_SLAM3::BackendGraphEdgeKind::Covisibility,
                            committed.pose, 0.1}};

    ORB_SLAM3::BackendMapSnapshot projected;
    std::string reason;
    ASSERT_TRUE(ORB_SLAM3::GtsamMapAdapter::projectCommitted(
        snapshot, {10, 1.0, 4}, &projected, &reason));
    ASSERT_EQ(projected.keyframes.size(), 1U);
    EXPECT_EQ(projected.keyframes.front().id, 10U);
    ASSERT_EQ(projected.landmarks.size(), 1U);
    EXPECT_EQ(projected.landmarks.front().id, 7U);
    ASSERT_EQ(projected.observations.size(), 1U);
    EXPECT_EQ(projected.observations.front().keyframeId, 10U);
    EXPECT_EQ(projected.observations.front().landmarkId, 7U);
    EXPECT_TRUE(projected.graphEdges.empty());
}

TEST(GtsamMapLifecycle, RejectsCommittedWatermarkFromAnotherMapVersion)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    snapshot.version = 4;
    ORB_SLAM3::BackendKeyframeState keyframe;
    keyframe.id = 10;
    keyframe.timestampSec = 1.0;
    snapshot.keyframes.push_back(keyframe);

    ORB_SLAM3::BackendMapSnapshot projected;
    std::string reason;
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::projectCommitted(
        snapshot, {10, 1.0, 5}, &projected, &reason));
    EXPECT_EQ(reason, "snapshot and committed-keyframe watermark versions differ");
}

TEST(GtsamMapLifecycle, RejectsResultWhenMapTopologyChangedWithoutVersionBump)
{
    ORB_SLAM3::Map map;
    const auto snapshot = ORB_SLAM3::GtsamMapAdapter::snapshot(map, 0);
    ORB_SLAM3::BackendMapResult result;
    result.accepted = true;
    result.sourceVersion = snapshot.version;
    result.sourceMapId = snapshot.mapId;
    result.sourceTopologySignature = snapshot.topologySignature;
    ORB_SLAM3::MapPoint landmark;
    landmark.mnId = 99;
    landmark.SetWorldPos((cv::Mat_<float>(3, 1) << 1.0F, 2.0F, 3.0F));
    map.AddMapPoint(&landmark);

    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::commit(result, map, 0));
}

TEST(GtsamMapLifecycle, RejectsLocalResultWithExcessivePoseCorrection)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    ORB_SLAM3::BackendKeyframeState source;
    source.id = 10;
    source.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                   0.0F, 1.0F, 0.0F, 0.0F,
                   0.0F, 0.0F, 1.0F, 0.0F,
                   0.0F, 0.0F, 0.0F, 1.0F};
    snapshot.keyframes.push_back(source);
    ORB_SLAM3::BackendMapResult result;
    result.accepted = true;
    ORB_SLAM3::BackendKeyframeState optimized = source;
    optimized.pose[3] = 0.7F;
    result.keyframes.push_back(optimized);

    const auto validation = ORB_SLAM3::GtsamMapAdapter::validateLocalResult(
        snapshot, result, 0.25, 2.0);

    EXPECT_FALSE(validation.accepted);
    EXPECT_NEAR(validation.maximumPoseTranslation, 0.7, 1e-6);
}

TEST(GtsamMapLifecycle, RejectsLocalResultWithExcessiveLandmarkCorrection)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    ORB_SLAM3::BackendLandmarkState source;
    source.id = 20;
    source.position = {1.0F, 2.0F, 3.0F};
    snapshot.landmarks.push_back(source);
    ORB_SLAM3::BackendMapResult result;
    result.accepted = true;
    ORB_SLAM3::BackendLandmarkState optimized = source;
    optimized.position[2] = 12.0F;
    result.landmarks.push_back(optimized);

    const auto validation = ORB_SLAM3::GtsamMapAdapter::validateLocalResult(
        snapshot, result, 0.25, 2.0);

    EXPECT_FALSE(validation.accepted);
    EXPECT_NEAR(validation.maximumLandmarkTranslation, 9.0, 1e-6);
}

}  // namespace
