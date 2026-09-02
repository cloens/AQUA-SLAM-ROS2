#include <gtest/gtest.h>

#include "GtsamMapAdapter.h"
#include "Map.h"
#include "MapPoint.h"

namespace {

TEST(GtsamMapLifecycle, CarriesStereoQualityAndLandmarkDegree)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    ORB_SLAM3::BackendLandmarkState landmark;
    landmark.id = 10;
    landmark.observationDegree = 2U;
    snapshot.landmarks.push_back(landmark);

    ORB_SLAM3::BackendStereoObservation first;
    first.keyframeId = 7;
    first.landmarkId = landmark.id;
    first.octave = 2;
    first.quality = 0.8;
    snapshot.observations.push_back(first);
    ORB_SLAM3::BackendStereoObservation second = first;
    second.keyframeId = 8;
    second.octave = 1;
    second.quality = 0.7;
    snapshot.observations.push_back(second);

    ASSERT_EQ(snapshot.landmarks.size(), 1U);
    ASSERT_EQ(snapshot.observations.size(), 2U);
    EXPECT_EQ(snapshot.landmarks.front().observationDegree, 2U);
    EXPECT_EQ(snapshot.observations.front().octave, 2);
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
