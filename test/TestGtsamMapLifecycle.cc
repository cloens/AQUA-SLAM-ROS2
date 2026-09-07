#include <gtest/gtest.h>

#include "Frame.h"
#include "GtsamBackendAdapter.h"
#include "GtsamMapAdapter.h"
#include "KeyFrame.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapPoint.h"
#include "Tracking.h"

#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <future>
#include <chrono>
#include <thread>

namespace ORB_SLAM3
{
class TrackingTestAccess
{
public:
    static std::unique_ptr<Tracking> Create()
    {
        return std::unique_ptr<Tracking>(new Tracking());
    }

    static void RecordCommittedWatermark(
        Tracking& tracking, std::uint64_t keyframeId, double timestampSec,
        Map& map)
    {
        tracking.RecordCommittedKeyframeWatermark(
            keyframeId, timestampSec, map);
    }

    static bool AdvanceCommittedWatermarkMapVersion(
        Tracking& tracking, std::uint64_t expectedVersion,
        std::uint64_t newVersion)
    {
        return tracking.AdvanceCommittedWatermarkMapVersion(
            expectedVersion, newVersion);
    }

    static std::shared_ptr<GtsamBackendAdapter> InstallAndAcquireBackend(
        Tracking& tracking, std::shared_ptr<GtsamBackendAdapter> backend)
    {
        tracking.mpGtsamBackend = std::move(backend);
        return tracking.AcquireGtsamBackend();
    }

    static void ResetDynamicBackendState(Tracking& tracking)
    {
        tracking.ResetDynamicBackendState();
    }

    static void SetRecoveryFrame(Tracking& tracking, double timestamp)
    {
        tracking.mCurrentFrame.mTimeStamp = timestamp;
        tracking.time_recently_lost = 1.0;
    }
};

class LocalMappingTestAccess
{
public:
    static std::unique_ptr<LocalMapping> Create(Tracking& tracking)
    {
        auto mapper = std::unique_ptr<LocalMapping>(new LocalMapping());
        mapper->mpTracker = &tracking;
        mapper->mbResetRequested = false;
        mapper->mbResetRequestedActiveMap = false;
        mapper->mbStopRequested = false;
        mapper->mbFinishRequested = false;
        mapper->mbAbortBA.store(false);
        return mapper;
    }

    static bool WaitForCommit(LocalMapping& mapper)
    {
        return mapper.AcquireDynamicCommitTransaction().owns_lock();
    }

    static void RequestReset(LocalMapping& mapper, bool activeMap)
    {
        std::lock_guard<std::mutex> lock(mapper.mMutexReset);
        if (activeMap)
            mapper.mbResetRequestedActiveMap = true;
        else
            mapper.mbResetRequested = true;
    }
};

class LoopClosingTestAccess
{
public:
    static std::unique_ptr<LoopClosing> Create(Tracking& tracking)
    {
        auto loop = std::unique_ptr<LoopClosing>(new LoopClosing());
        loop->mpTracker = &tracking;
        loop->mpLocalMapper = nullptr;
        loop->mbResetRequested = false;
        loop->mbResetActiveMapRequested = false;
        loop->mpLoopLastCurrentKF = nullptr;
        loop->mpLoopMatchedKF = nullptr;
        loop->mpMergeLastCurrentKF = nullptr;
        loop->mpMergeMatchedKF = nullptr;
        return loop;
    }

    static bool ResetRequested(LoopClosing& loop)
    {
        std::lock_guard<std::mutex> lock(loop.mMutexReset);
        return loop.mbResetRequested;
    }

    static void CorrectAndAcknowledgeReset(LoopClosing& loop, bool merge)
    {
        if (merge)
            loop.CorrectMerge();
        else
            loop.CorrectLoop();
        loop.ResetIfRequested();
    }
};
}  // namespace ORB_SLAM3

namespace {

TEST(GtsamMapLifecycle, RecoveryRequiresPredictionAndExpiresWithoutVisualSupport)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    tracking->mState = ORB_SLAM3::Tracking::OK;
    ORB_SLAM3::TrackingTestAccess::SetRecoveryFrame(*tracking, 2.0);
    tracking->UpdateDynamicRecoveryState(false, true);
    EXPECT_EQ(tracking->mState, ORB_SLAM3::Tracking::RECENTLY_LOST);
    ORB_SLAM3::TrackingTestAccess::SetRecoveryFrame(*tracking, 2.5);
    tracking->UpdateDynamicRecoveryState(false, true);
    EXPECT_EQ(tracking->mState, ORB_SLAM3::Tracking::RECENTLY_LOST);
    ORB_SLAM3::TrackingTestAccess::SetRecoveryFrame(*tracking, 3.1);
    tracking->UpdateDynamicRecoveryState(false, true);
    EXPECT_EQ(tracking->mState, ORB_SLAM3::Tracking::LOST);
    tracking->UpdateDynamicRecoveryState(true, false);
    EXPECT_EQ(tracking->mState, ORB_SLAM3::Tracking::OK);
    tracking->UpdateDynamicRecoveryState(false, false);
    EXPECT_EQ(tracking->mState, ORB_SLAM3::Tracking::LOST);
}

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

ORB_SLAM3::BackendLoopPointCorrespondence mergeCorrespondence(
    std::uint64_t currentId, std::uint64_t candidateId, double x)
{
    ORB_SLAM3::BackendLoopPointCorrespondence correspondence;
    correspondence.currentLandmarkId = currentId;
    correspondence.candidateLandmarkId = candidateId;
    correspondence.currentPoint = {x, 0.0, 0.0};
    correspondence.candidatePoint = {x + 1.0, 0.0, 0.0};
    return correspondence;
}

ORB_SLAM3::BackendLoopCandidateResult acceptedMergeTransform(double x)
{
    ORB_SLAM3::BackendLoopCandidateResult result;
    result.accepted = true;
    result.currentFromCandidate = {
        1.0F, 0.0F, 0.0F, static_cast<float>(x),
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F};
    result.inliers = 8U;
    result.rmsErrorMeters = 0.02;
    return result;
}

TEST(GtsamMergeHypothesis,
     AccumulatesAcrossCandidateKeyframesInTheSameMapAndDeduplicates)
{
    ORB_SLAM3::BackendMergeHypothesis hypothesis(8U, 2U);

    const auto first = hypothesis.observe(
        41U, 100U, 10U,
        {mergeCorrespondence(1U, 11U, 0.0),
         mergeCorrespondence(2U, 12U, 1.0)});
    const auto second = hypothesis.observe(
        41U, 101U, 11U,
        {mergeCorrespondence(2U, 12U, 1.1),
         mergeCorrespondence(3U, 13U, 2.0)});

    EXPECT_FALSE(first.reset);
    EXPECT_FALSE(second.reset);
    EXPECT_EQ(hypothesis.candidateMapId(), 41U);
    EXPECT_EQ(hypothesis.currentKeyframeId(), 101U);
    EXPECT_EQ(hypothesis.candidateKeyframeId(), 11U);
    EXPECT_EQ(hypothesis.correspondences().size(), 3U);
    EXPECT_EQ(second.added, 1U);
}

TEST(GtsamMergeHypothesis, BoundsThePersistentCorrespondenceCache)
{
    ORB_SLAM3::BackendMergeHypothesis hypothesis(3U, 2U);

    hypothesis.observe(
        41U, 100U, 10U,
        {mergeCorrespondence(1U, 11U, 0.0),
         mergeCorrespondence(2U, 12U, 1.0),
         mergeCorrespondence(3U, 13U, 2.0),
         mergeCorrespondence(4U, 14U, 3.0)});

    ASSERT_EQ(hypothesis.correspondences().size(), 3U);
    EXPECT_EQ(hypothesis.correspondences().front().currentLandmarkId, 2U);
    EXPECT_EQ(hypothesis.correspondences().back().currentLandmarkId, 4U);
}

TEST(GtsamMergeHypothesis, CandidateMapChangeResetsAccumulatedEvidence)
{
    ORB_SLAM3::BackendMergeHypothesis hypothesis(8U, 2U);
    hypothesis.observe(
        41U, 100U, 10U,
        {mergeCorrespondence(1U, 11U, 0.0),
         mergeCorrespondence(2U, 12U, 1.0)});

    const auto update = hypothesis.observe(
        42U, 101U, 20U, {mergeCorrespondence(3U, 13U, 2.0)});

    EXPECT_TRUE(update.reset);
    EXPECT_EQ(update.resetReason, "candidate_map_changed");
    EXPECT_EQ(hypothesis.candidateMapId(), 42U);
    ASSERT_EQ(hypothesis.correspondences().size(), 1U);
    EXPECT_EQ(hypothesis.correspondences().front().currentLandmarkId, 3U);
}

TEST(GtsamMergeHypothesis, TwoConsecutiveMissesResetTheHypothesis)
{
    ORB_SLAM3::BackendMergeHypothesis hypothesis(8U, 2U);
    hypothesis.observe(
        41U, 100U, 10U, {mergeCorrespondence(1U, 11U, 0.0)});
    std::string reason;

    EXPECT_FALSE(hypothesis.recordMiss(&reason));
    EXPECT_TRUE(reason.empty());
    EXPECT_TRUE(hypothesis.recordMiss(&reason));
    EXPECT_EQ(reason, "consecutive_miss_limit");
    EXPECT_FALSE(hypothesis.active());
    EXPECT_TRUE(hypothesis.correspondences().empty());
}

TEST(GtsamMergeHypothesis,
     AdvancesConfirmationOnlyForConsistentAggregateTransforms)
{
    ORB_SLAM3::BackendMergeHypothesis hypothesis(8U, 2U);
    hypothesis.observe(
        41U, 100U, 10U, {mergeCorrespondence(1U, 11U, 0.0)});
    double translationDelta = -1.0;
    double rotationDelta = -1.0;
    std::string reason;

    EXPECT_TRUE(hypothesis.recordAccepted(
        acceptedMergeTransform(0.0), 0.5, 0.35,
        &translationDelta, &rotationDelta, &reason));
    EXPECT_EQ(hypothesis.confirmations(), 1U);
    EXPECT_TRUE(hypothesis.recordAccepted(
        acceptedMergeTransform(0.2), 0.5, 0.35,
        &translationDelta, &rotationDelta, &reason));
    EXPECT_EQ(hypothesis.confirmations(), 2U);
    EXPECT_NEAR(translationDelta, 0.2, 1e-6);
    EXPECT_FALSE(hypothesis.recordAccepted(
        acceptedMergeTransform(2.0), 0.5, 0.35,
        &translationDelta, &rotationDelta, &reason));
    EXPECT_EQ(reason, "transform_inconsistent");
    EXPECT_FALSE(hypothesis.active());
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

TEST(GtsamMapLifecycle, CommitsVerifiedLandmarkReplacementAfterAllValidation)
{
    ORB_SLAM3::Map map;
    ORB_SLAM3::Frame firstFrame;
    ORB_SLAM3::Frame secondFrame;
    configureFrame(firstFrame, 7U, 1.0, 0, 1.0F);
    configureFrame(secondFrame, 8U, 2.0, 0, 1.0F);
    firstFrame.mDescriptors = cv::Mat::zeros(1, 256, CV_32F);
    firstFrame.mDescriptors.at<float>(0, 0) = 1.0F;
    secondFrame.mDescriptors = firstFrame.mDescriptors.clone();
    ORB_SLAM3::KeyFrame first(firstFrame, &map, nullptr);
    ORB_SLAM3::KeyFrame second(secondFrame, &map, nullptr);
    const cv::Mat position = (cv::Mat_<float>(3, 1) << 1.0F, 2.0F, 4.0F);
    ORB_SLAM3::MapPoint target(position, &first, &map);
    ORB_SLAM3::MapPoint source(position, &second, &map);
    target.mnId = 100U;
    source.mnId = 200U;
    first.AddMapPoint(&target, 0);
    second.AddMapPoint(&source, 0);
    target.AddObservation(&first, 0);
    source.AddObservation(&second, 0);
    map.AddKeyFrame(&first);
    map.AddKeyFrame(&second);
    map.AddMapPoint(&target);
    map.AddMapPoint(&source);
    const auto snapshot = ORB_SLAM3::GtsamMapAdapter::snapshot(map, 0U);
    ORB_SLAM3::BackendMapResult result;
    result.accepted = true;
    result.sourceMapId = snapshot.mapId;
    result.sourceVersion = snapshot.version;
    result.sourceTopologySignature = snapshot.topologySignature;
    result.keyframes = snapshot.keyframes;
    result.landmarkReplacements = {{source.mnId, 9999U}};
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::commit(result, map, 0U));
    EXPECT_FALSE(source.isBad());
    EXPECT_EQ(second.GetMapPoint(0), &source);
    result.landmarkReplacements = {{source.mnId, target.mnId}};
    ASSERT_TRUE(ORB_SLAM3::GtsamMapAdapter::commit(result, map, 0U));
    EXPECT_TRUE(source.isBad());
    EXPECT_EQ(source.GetReplaced(), &target);
    EXPECT_EQ(second.GetMapPoint(0), &target);
    EXPECT_EQ(map.GetAllMapPoints().size(), 1U);
    EXPECT_EQ(target.GetObservations().size(), 2U);
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

TEST(GtsamMapLifecycle, AppendsVerifiedLoopCorrectionToSnapshotPoses)
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
    std::array<float, 16> currentFromCandidate = first.pose;
    currentFromCandidate[3] = 0.3F;

    ASSERT_TRUE(ORB_SLAM3::GtsamMapAdapter::appendAcceptedLoopEdge(
        snapshot, 10, 11, currentFromCandidate, 0.03));
    ASSERT_EQ(snapshot.graphEdges.size(), 1U);
    EXPECT_EQ(snapshot.graphEdges.front().kind,
              ORB_SLAM3::BackendGraphEdgeKind::Loop);
    EXPECT_NEAR(snapshot.graphEdges.front().relativePose[3], 0.5F, 1e-6F);
}

TEST(GtsamMapLifecycle, TransformsSnapshotIntoCandidateMapCoordinates)
{
    ORB_SLAM3::BackendMapSnapshot current;
    current.mapId = 7;
    ORB_SLAM3::BackendKeyframeState keyframe;
    keyframe.id = 20;
    keyframe.pose = {1.0F, 0.0F, 0.0F, 2.0F,
                     0.0F, 1.0F, 0.0F, 0.0F,
                     0.0F, 0.0F, 1.0F, 0.0F,
                     0.0F, 0.0F, 0.0F, 1.0F};
    keyframe.velocity = {2.0F, 0.0F, 0.0F};
    current.keyframes.push_back(keyframe);
    current.landmarks.push_back({30, {1.0F, 2.0F, 3.0F}});
    const std::array<float, 16> candidateFromCurrent = {
        0.0F, -1.0F, 0.0F, 10.0F,
        1.0F,  0.0F, 0.0F,  0.0F,
        0.0F,  0.0F, 1.0F,  0.0F,
        0.0F,  0.0F, 0.0F,  1.0F};

    ORB_SLAM3::BackendMapSnapshot transformed;
    ASSERT_TRUE(ORB_SLAM3::GtsamMapAdapter::transformSnapshot(
        current, candidateFromCurrent, &transformed));

    ASSERT_EQ(transformed.keyframes.size(), 1U);
    EXPECT_NEAR(transformed.keyframes.front().pose[3], 10.0F, 1e-6F);
    EXPECT_NEAR(transformed.keyframes.front().pose[7], 2.0F, 1e-6F);
    EXPECT_NEAR(transformed.keyframes.front().velocity[0], 0.0F, 1e-6F);
    EXPECT_NEAR(transformed.keyframes.front().velocity[1], 2.0F, 1e-6F);
    ASSERT_EQ(transformed.landmarks.size(), 1U);
    EXPECT_NEAR(transformed.landmarks.front().position[0], 8.0F, 1e-6F);
    EXPECT_NEAR(transformed.landmarks.front().position[1], 1.0F, 1e-6F);
    EXPECT_NEAR(transformed.landmarks.front().position[2], 3.0F, 1e-6F);
}

TEST(GtsamMapLifecycle, AppendsMergeEdgeWithMeasuredCovariance)
{
    ORB_SLAM3::BackendMapSnapshot combined;
    ORB_SLAM3::BackendKeyframeState current;
    current.id = 20;
    current.pose = {1.0F, 0.0F, 0.0F, 1.0F,
                    0.0F, 1.0F, 0.0F, 0.0F,
                    0.0F, 0.0F, 1.0F, 0.0F,
                    0.0F, 0.0F, 0.0F, 1.0F};
    ORB_SLAM3::BackendKeyframeState candidate = current;
    candidate.id = 10;
    candidate.pose[3] = 0.0F;
    combined.keyframes = {candidate, current};
    const std::array<float, 16> currentFromCandidate = {
        1.0F, 0.0F, 0.0F, 1.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F};
    std::array<double, 36> covariance{};
    covariance[0] = 0.04;
    covariance[7] = 0.05;
    covariance[14] = 0.06;
    covariance[21] = 0.01;
    covariance[28] = 0.02;
    covariance[35] = 0.03;

    ASSERT_TRUE(ORB_SLAM3::GtsamMapAdapter::appendAcceptedMergeEdge(
        combined, 20, 10, currentFromCandidate, covariance));
    ASSERT_EQ(combined.graphEdges.size(), 1U);
    const auto& edge = combined.graphEdges.front();
    EXPECT_EQ(edge.kind, ORB_SLAM3::BackendGraphEdgeKind::Merge);
    EXPECT_TRUE(edge.hasCovariance);
    EXPECT_DOUBLE_EQ(edge.covariance[0], 0.04);
    EXPECT_DOUBLE_EQ(edge.covariance[35], 0.03);
}

TEST(GtsamMapLifecycle, RejectsMergeEdgeWithNonPositiveCovariance)
{
    ORB_SLAM3::BackendMapSnapshot combined;
    ORB_SLAM3::BackendKeyframeState first;
    first.id = 10;
    first.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                  0.0F, 1.0F, 0.0F, 0.0F,
                  0.0F, 0.0F, 1.0F, 0.0F,
                  0.0F, 0.0F, 0.0F, 1.0F};
    ORB_SLAM3::BackendKeyframeState second = first;
    second.id = 20;
    combined.keyframes = {first, second};
    std::array<double, 36> covariance{};
    covariance[0] = -0.01;
    covariance[7] = 0.01;
    covariance[14] = 0.01;
    covariance[21] = 0.01;
    covariance[28] = 0.01;
    covariance[35] = 0.01;

    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::appendAcceptedMergeEdge(
        combined, 20, 10, first.pose, covariance));
    EXPECT_TRUE(combined.graphEdges.empty());
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

TEST(GtsamTrackingDiagnostics,
     KeepsPublishedDiagnosticsBoundToActualWatermarkAdvance)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    ORB_SLAM3::Map map;
    for (int index = 0; index < 4; ++index)
        map.IncreaseChangeIndex();

    ORB_SLAM3::TrackingTestAccess::RecordCommittedWatermark(
        *tracking, 10U, 1.0, map);
    ORB_SLAM3::CommittedKeyframeWatermark capturedWatermark;
    ASSERT_TRUE(tracking->CaptureCommittedKeyframeWatermark(&capturedWatermark));
    EXPECT_EQ(capturedWatermark.mapVersion, 4U);

    ORB_SLAM3::BackendMapSnapshot matchingSnapshot;
    matchingSnapshot.version = capturedWatermark.mapVersion;
    matchingSnapshot.keyframes.push_back({10U, 1.0});
    ORB_SLAM3::BackendMapSnapshot projected;
    std::string rejectionReason;
    EXPECT_TRUE(ORB_SLAM3::GtsamMapAdapter::projectCommitted(
        matchingSnapshot, capturedWatermark, &projected, &rejectionReason));

    ORB_SLAM3::BackendMapDiagnostics diagnostics;
    diagnostics.capturedWatermark = capturedWatermark;
    tracking->RecordDynamicDiagnostics(diagnostics);

    map.IncreaseChangeIndex();
    ORB_SLAM3::TrackingTestAccess::RecordCommittedWatermark(
        *tracking, 11U, 2.0, map);
    ORB_SLAM3::CommittedKeyframeWatermark liveWatermark;
    ASSERT_TRUE(tracking->CaptureCommittedKeyframeWatermark(&liveWatermark));
    EXPECT_EQ(liveWatermark.keyframeId, 11U);
    EXPECT_EQ(liveWatermark.mapVersion, 5U);

    const auto publishedDiagnostics = tracking->LatestDynamicDiagnostics();
    ASSERT_TRUE(publishedDiagnostics.has_value());
    EXPECT_EQ(publishedDiagnostics->capturedWatermark.keyframeId, 10U);
    EXPECT_DOUBLE_EQ(publishedDiagnostics->capturedWatermark.timestampSec, 1.0);
    EXPECT_EQ(publishedDiagnostics->capturedWatermark.mapVersion, 4U);
}

TEST(GtsamTrackingLifecycle,
     AdvancesCommittedWatermarkAcrossAcceptedMapCorrection)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    ORB_SLAM3::Map map;
    ORB_SLAM3::TrackingTestAccess::RecordCommittedWatermark(
        *tracking, 10U, 1.0, map);

    map.IncreaseChangeIndex();
    ASSERT_TRUE(
        ORB_SLAM3::TrackingTestAccess::AdvanceCommittedWatermarkMapVersion(
            *tracking, 0U, 1U));

    ORB_SLAM3::CommittedKeyframeWatermark watermark;
    ASSERT_TRUE(tracking->CaptureCommittedKeyframeWatermark(&watermark));
    EXPECT_EQ(watermark.keyframeId, 10U);
    EXPECT_DOUBLE_EQ(watermark.timestampSec, 1.0);
    EXPECT_EQ(watermark.mapVersion, 1U);

    ORB_SLAM3::BackendMapSnapshot correctedSnapshot;
    correctedSnapshot.version = 1U;
    correctedSnapshot.keyframes.push_back({10U, 1.0});
    ORB_SLAM3::BackendMapSnapshot projected;
    std::string reason;
    EXPECT_TRUE(ORB_SLAM3::GtsamMapAdapter::projectCommitted(
        correctedSnapshot, watermark, &projected, &reason));
}

TEST(GtsamTrackingLifecycle,
     DoesNotRollBackCommittedWatermarkMapVersion)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    ORB_SLAM3::Map map;
    map.IncreaseChangeIndex();
    ORB_SLAM3::TrackingTestAccess::RecordCommittedWatermark(
        *tracking, 11U, 2.0, map);

    EXPECT_FALSE(
        ORB_SLAM3::TrackingTestAccess::AdvanceCommittedWatermarkMapVersion(
            *tracking, 0U, 1U));
    ORB_SLAM3::CommittedKeyframeWatermark watermark;
    ASSERT_TRUE(tracking->CaptureCommittedKeyframeWatermark(&watermark));
    EXPECT_EQ(watermark.keyframeId, 11U);
    EXPECT_DOUBLE_EQ(watermark.timestampSec, 2.0);
    EXPECT_EQ(watermark.mapVersion, 1U);
}

TEST(GtsamTrackingLifecycle, KeepsRebaseLeaseAliveAcrossBackendReset)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    const cv::Mat camera = (cv::Mat_<float>(3, 3) <<
        456.0F, 0.0F, 320.0F, 0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    const auto backend = std::make_shared<ORB_SLAM3::GtsamBackendAdapter>(
        camera, 63.0F, cv::Mat::eye(4, 4, CV_32F));
    const auto rebaseLease =
        ORB_SLAM3::TrackingTestAccess::InstallAndAcquireBackend(
            *tracking, backend);
    ASSERT_TRUE(rebaseLease);

    ORB_SLAM3::TrackingTestAccess::ResetDynamicBackendState(*tracking);
    EXPECT_FALSE(tracking->RebaseGtsamBackend(ORB_SLAM3::BackendMapSnapshot()));
    EXPECT_NO_THROW(rebaseLease->rebase(ORB_SLAM3::BackendMapSnapshot()));
}

TEST(GtsamTrackingLifecycle, AppliesLocalMapCorrectionInTheCommittedDirection)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    std::mutex poseMutex;
    tracking->mLastFrame.mpExtrinsic_mutex = &poseMutex;
    tracking->mCurrentFrame.mpExtrinsic_mutex = &poseMutex;
    tracking->mLastFrame.mTcw = cv::Mat::eye(4, 4, CV_32F);
    tracking->mCurrentFrame.mTcw = cv::Mat::eye(4, 4, CV_32F);

    const cv::Mat oldCameraFromMap = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat newCameraFromMap = cv::Mat::eye(4, 4, CV_32F);
    newCameraFromMap.at<float>(0, 3) = 1.0F;

    tracking->SynchronizeAfterMapCorrection(
        oldCameraFromMap, newCameraFromMap);

    EXPECT_LT(cv::norm(tracking->mCurrentFrame.mTcw - newCameraFromMap), 1e-6);
}

TEST(GtsamTrackingLifecycle, MapCorrectionWaitsForTrackingTransaction)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    std::mutex poseMutex;
    tracking->mLastFrame.mpExtrinsic_mutex = &poseMutex;
    tracking->mCurrentFrame.mpExtrinsic_mutex = &poseMutex;
    tracking->mLastFrame.mTcw = cv::Mat::eye(4, 4, CV_32F);
    tracking->mCurrentFrame.mTcw = cv::Mat::eye(4, 4, CV_32F);
    const cv::Mat before = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat after = before.clone();
    after.at<float>(0, 3) = 1.0F;
    auto frameTransaction = tracking->AcquireDynamicMapTransaction();
    std::promise<void> started;
    auto worker = std::async(std::launch::async, [&] {
        started.set_value();
        tracking->SynchronizeAfterMapCorrection(before, after);
    });
    started.get_future().wait();
    EXPECT_EQ(worker.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    EXPECT_LT(cv::norm(tracking->mCurrentFrame.mTcw - before), 1e-6);
    frameTransaction.unlock();
    worker.get();
    EXPECT_LT(cv::norm(tracking->mCurrentFrame.mTcw - after), 1e-6);
}

TEST(GtsamTrackingLifecycle, BackgroundTransactionYieldsWhileTrackingWaits)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    auto frameTransaction = tracking->AcquireDynamicMapTransaction();
    auto worker = std::async(std::launch::async, [&] {
        return tracking->TryAcquireDynamicMapTransaction().owns_lock();
    });
    const auto status = worker.wait_for(std::chrono::seconds(2));
    EXPECT_EQ(status, std::future_status::ready);
    frameTransaction.unlock();
    EXPECT_FALSE(worker.get());
    EXPECT_TRUE(tracking->TryAcquireDynamicMapTransaction().owns_lock());
}

TEST(GtsamTrackingLifecycle, LocalCommitWaitsThroughContentionAndThenAcquires)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    auto mapper = ORB_SLAM3::LocalMappingTestAccess::Create(*tracking);
    auto frameTransaction = tracking->AcquireDynamicMapTransaction();
    auto worker = std::async(std::launch::async, [&] {
        return ORB_SLAM3::LocalMappingTestAccess::WaitForCommit(*mapper);
    });
    EXPECT_EQ(worker.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    frameTransaction.unlock();
    EXPECT_EQ(worker.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(worker.get());
}

TEST(GtsamTrackingLifecycle, LocalCommitCancelsPendingWaitForResetStopOrFinish)
{
    for (int request = 0; request < 4; ++request) {
        SCOPED_TRACE(request);
        auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
        auto mapper = ORB_SLAM3::LocalMappingTestAccess::Create(*tracking);
        auto frameTransaction = tracking->AcquireDynamicMapTransaction();
        auto worker = std::async(std::launch::async, [&] {
            return ORB_SLAM3::LocalMappingTestAccess::WaitForCommit(*mapper);
        });
        EXPECT_EQ(worker.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
        if (request < 2)
            ORB_SLAM3::LocalMappingTestAccess::RequestReset(*mapper, request == 1);
        else if (request == 2)
            mapper->RequestStop();
        else
            mapper->RequestFinish();
        const auto status = worker.wait_for(std::chrono::seconds(2));
        EXPECT_EQ(status, std::future_status::ready);
        frameTransaction.unlock();
        EXPECT_FALSE(worker.get());
    }
}

TEST(GtsamTrackingLifecycle, CorrectionsYieldBeforeStopAndAcknowledgeReset)
{
    for (const bool merge : {false, true}) {
        SCOPED_TRACE(merge ? "merge" : "loop");
        auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
        auto loop = ORB_SLAM3::LoopClosingTestAccess::Create(*tracking);
        auto frameTransaction = tracking->AcquireDynamicMapTransaction();
        auto reset = std::async(std::launch::async, [&] { loop->RequestReset(); });
        auto worker = std::async(std::launch::async, [&] {
            while (!ORB_SLAM3::LoopClosingTestAccess::ResetRequested(*loop))
                std::this_thread::yield();
            // No LocalMapper is installed: contention must return before RequestStop.
            ORB_SLAM3::LoopClosingTestAccess::CorrectAndAcknowledgeReset(*loop, merge);
        });
        const auto status = reset.wait_for(std::chrono::seconds(2));
        EXPECT_EQ(status, std::future_status::ready);
        frameTransaction.unlock();
        worker.get();
        reset.get();
    }
}

TEST(GtsamTrackingLifecycle, DynamicMapCommitSynchronizesTrackingFrames)
{
    auto tracking = ORB_SLAM3::TrackingTestAccess::Create();
    std::mutex poseMutex;
    tracking->mLastFrame.mpExtrinsic_mutex = &poseMutex;
    tracking->mCurrentFrame.mpExtrinsic_mutex = &poseMutex;
    tracking->mLastFrame.mTcw = cv::Mat::eye(4, 4, CV_32F);
    tracking->mCurrentFrame.mTcw = cv::Mat::eye(4, 4, CV_32F);

    ORB_SLAM3::Map map;
    ORB_SLAM3::KeyFrame keyframe;
    keyframe.mnId = 10U;
    keyframe.UpdateMap(&map);
    keyframe.SetPose(cv::Mat::eye(4, 4, CV_32F));
    map.AddKeyFrame(&keyframe);
    const auto snapshot = ORB_SLAM3::GtsamMapAdapter::snapshot(map, 0U);
    ASSERT_EQ(snapshot.keyframes.size(), 1U);

    ORB_SLAM3::BackendMapResult result;
    result.accepted = true;
    result.sourceVersion = snapshot.version;
    result.sourceMapId = snapshot.mapId;
    result.sourceTopologySignature = snapshot.topologySignature;
    auto optimized = snapshot.keyframes.front();
    optimized.pose[3] = 1.0F;
    result.keyframes.push_back(optimized);

    ASSERT_TRUE(tracking->CommitDynamicMapResult(
        result, map, snapshot.version, &keyframe));
    EXPECT_LT(cv::norm(tracking->mCurrentFrame.mTcw - keyframe.GetPose()), 1e-6);
    EXPECT_NEAR(keyframe.GetPose().at<float>(0, 3), -1.0F, 1e-6F);
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

TEST(GtsamMapLifecycle, AcceptsFiniteLargePoseCorrection)
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
        snapshot, result);

    EXPECT_TRUE(validation.accepted);
    EXPECT_NEAR(validation.maximumPoseTranslation, 0.7, 1e-6);
    result.keyframes.front().pose[3] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::validateLocalResult(
        snapshot, result).accepted);
}

TEST(GtsamMapLifecycle, AcceptsFiniteLargeStereoResidual)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    snapshot.calibration.fx = 100.0;
    snapshot.calibration.fy = 100.0;
    snapshot.calibration.cx = 50.0;
    snapshot.calibration.cy = 50.0;
    snapshot.calibration.baseline = 0.1;
    ORB_SLAM3::BackendKeyframeState source;
    source.id = 10;
    source.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                   0.0F, 1.0F, 0.0F, 0.0F,
                   0.0F, 0.0F, 1.0F, 0.0F,
                   0.0F, 0.0F, 0.0F, 1.0F};
    snapshot.keyframes.push_back(source);
    ORB_SLAM3::BackendLandmarkState landmark;
    landmark.id = 20;
    landmark.position = {0.0F, 0.0F, 1.0F};
    snapshot.landmarks.push_back(landmark);
    snapshot.observations.push_back({10U, 20U, 50.0, 40.0, 50.0,
                                     1.0, 0, 1.0, 1.0, 0.0});

    ORB_SLAM3::BackendMapResult result;
    result.accepted = true;
    auto optimized = source;
    optimized.pose[3] = 0.1F;
    result.keyframes.push_back(optimized);
    result.landmarks.push_back(landmark);
    result.acceptedStereoObservations.push_back(snapshot.observations.front());

    const auto validation = ORB_SLAM3::GtsamMapAdapter::validateLocalResult(
        snapshot, result);

    EXPECT_TRUE(validation.accepted);
    result.landmarks.front().position[2] = -1.0F;
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::validateLocalResult(
        snapshot, result).accepted);
}

TEST(GtsamMapLifecycle, ProjectionValidationIgnoresNisRejectedObservation)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    snapshot.calibration.fx = 100.0;
    snapshot.calibration.fy = 100.0;
    snapshot.calibration.cx = 50.0;
    snapshot.calibration.cy = 50.0;
    snapshot.calibration.baseline = 0.1;
    ORB_SLAM3::BackendKeyframeState pose;
    pose.id = 10;
    pose.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                 0.0F, 1.0F, 0.0F, 0.0F,
                 0.0F, 0.0F, 1.0F, 0.0F,
                 0.0F, 0.0F, 0.0F, 1.0F};
    snapshot.keyframes.push_back(pose);
    ORB_SLAM3::BackendLandmarkState landmark;
    landmark.id = 20;
    landmark.position = {0.0F, 0.0F, 1.0F};
    snapshot.landmarks.push_back(landmark);
    const ORB_SLAM3::BackendStereoObservation accepted = {
        10U, 20U, 50.0, 40.0, 50.0, 1.0, 0, 1.0, 1.0, 0.0};
    const ORB_SLAM3::BackendStereoObservation rejected = {
        10U, 20U, 150.0, 140.0, 150.0, 1.0, 0, 1.0, 1.0, 0.0};
    snapshot.observations = {accepted, rejected};

    ORB_SLAM3::BackendMapResult result;
    result.accepted = true;
    result.keyframes.push_back(pose);
    result.landmarks.push_back(landmark);
    result.acceptedStereoObservations.push_back(accepted);

    const auto validation = ORB_SLAM3::GtsamMapAdapter::validateLocalResult(
        snapshot, result);

    EXPECT_TRUE(validation.accepted);
}

TEST(GtsamMapLifecycle, AcceptsFiniteLargeLandmarkCorrection)
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
        snapshot, result);

    EXPECT_TRUE(validation.accepted);
    EXPECT_NEAR(validation.maximumLandmarkTranslation, 9.0, 1e-6);
    result.landmarks.front().position[2] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::validateLocalResult(
        snapshot, result).accepted);
}

TEST(GtsamMapLifecycle, AcceptsOneBoundedLandmarkOutlierWhenP95IsStable)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    ORB_SLAM3::BackendMapResult result;
    result.accepted = true;
    for (std::uint64_t id = 0; id < 100U; ++id) {
        ORB_SLAM3::BackendLandmarkState source;
        source.id = id;
        source.position = {static_cast<float>(id), 0.0F, 3.0F};
        snapshot.landmarks.push_back(source);
        auto optimized = source;
        optimized.position[1] += id == 99U ? 3.0F : 0.1F;
        result.landmarks.push_back(optimized);
    }

    const auto validation = ORB_SLAM3::GtsamMapAdapter::validateLocalResult(
        snapshot, result);

    EXPECT_TRUE(validation.accepted);
    EXPECT_NEAR(validation.landmarkTranslationP95, 0.1, 1e-5);
    EXPECT_NEAR(validation.maximumLandmarkTranslation, 3.0, 1e-6);
}

TEST(GtsamMapLifecycle, RequiresConsistentMergeTransformAcrossConfirmations)
{
    const std::array<float, 16> identity = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F};
    auto nearby = identity;
    nearby[3] = 0.2F;
    auto distant = identity;
    distant[3] = 2.0F;

    EXPECT_TRUE(ORB_SLAM3::GtsamMapAdapter::consistentMergeTransform(
        identity, nearby, 0.5, 0.35));
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::consistentMergeTransform(
        identity, distant, 0.5, 0.35));
}

TEST(GtsamMapLifecycle, AcceptsFiniteLargePoseRotation)
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
    optimized.pose = {0.0F, -1.0F, 0.0F, 0.0F,
                      1.0F, 0.0F, 0.0F, 0.0F,
                      0.0F, 0.0F, 1.0F, 0.0F,
                      0.0F, 0.0F, 0.0F, 1.0F};
    result.keyframes.push_back(optimized);

    const auto validation = ORB_SLAM3::GtsamMapAdapter::validateMapResult(
        snapshot, result);

    EXPECT_TRUE(validation.accepted);
    EXPECT_NEAR(validation.maximumPoseRotation, 1.57079632679, 1e-6);
}

TEST(DynamicLocalOptimizationTransaction, CancelsWhenAbortPrecedesSolve)
{
    std::atomic_bool abortRequested{true};
    ORB_SLAM3::DynamicLocalOptimizationTransaction transaction;

    EXPECT_FALSE(transaction.MayStartSolve(abortRequested));
    EXPECT_FALSE(transaction.MayCommitValidatedResult(false));
}

TEST(DynamicLocalOptimizationTransaction,
     CommitsValidatedResultWhenAbortArrivesAfterSolve)
{
    std::atomic_bool abortRequested{false};
    ORB_SLAM3::DynamicLocalOptimizationTransaction transaction;

    ASSERT_TRUE(transaction.MayStartSolve(abortRequested));
    transaction.MarkResultValidation(true);
    abortRequested.store(true, std::memory_order_release);

    EXPECT_TRUE(transaction.MayCommitValidatedResult(false));
    EXPECT_FALSE(transaction.MayCommitValidatedResult(true));
}

}  // namespace
