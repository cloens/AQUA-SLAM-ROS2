#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>

#include "GtsamBackendAdapter.h"
#include "GtsamMapOptimizationAdapter.h"
#include "ImuTypes.h"
#include "fixtures/joint_initializer_fixture.h"

namespace {

struct JointInitializationFixture
{
    joint_initializer_fixture::Motion motion = joint_initializer_fixture::motion();
    ORB_SLAM3::BackendMapSnapshot snapshot;
    ORB_SLAM3::BackendFrameInput sensors;
    cv::Mat bodyFromCamera = cv::Mat(4, 4, CV_32F);

    JointInitializationFixture()
    {
        snapshot.version = 17U;
        snapshot.mapId = 3U;
        snapshot.topologySignature = 29U;
        const auto& calibration = *motion.input.calibration;
        snapshot.calibration = {calibration.fx(), calibration.fy(),
            calibration.px(), calibration.py(), calibration.baseline()};
        const auto bodyPose = motion.input.bodyFromCamera.matrix();
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                bodyFromCamera.at<float>(row, col) = bodyPose(row, col);
        for (const auto& frame : motion.input.frames) {
            ORB_SLAM3::BackendKeyframeState keyframe;
            keyframe.id = frame.keyframeId;
            keyframe.timestampSec = frame.timestampSec;
            const auto pose = frame.mapFromBodySeed.compose(motion.input.bodyFromCamera).matrix();
            for (int row = 0; row < 4; ++row)
                for (int col = 0; col < 4; ++col)
                    keyframe.pose[4 * row + col] = pose(row, col);
            snapshot.keyframes.push_back(keyframe);
            for (const auto& stereo : frame.observations) {
                ORB_SLAM3::BackendStereoObservation observation;
                observation.keyframeId = frame.keyframeId;
                observation.landmarkId = stereo.trackId;
                observation.uLeft = stereo.pixel.uL();
                observation.uRight = stereo.pixel.uR();
                observation.v = stereo.pixel.v();
                observation.octave = 0;
                observation.octaveVariancePx2 = frame.stereoCovariance(0, 0);
                snapshot.observations.push_back(observation);
            }
        }
        for (const auto& stereo : motion.input.frames.front().observations) {
            ORB_SLAM3::BackendLandmarkState landmark;
            landmark.id = stereo.trackId;
            for (std::size_t axis = 0; axis < 3U; ++axis)
                landmark.position[axis] = stereo.landmarkMap(axis);
            snapshot.landmarks.push_back(landmark);
        }
        for (const auto& sample : motion.input.imu) {
            ORB_SLAM3::ImuSample converted;
            converted.timestampSec = sample.timestampSec;
            for (std::size_t axis = 0; axis < 3U; ++axis) {
                converted.acceleration[axis] = sample.acceleration(axis);
                converted.angularVelocity[axis] = sample.angularVelocity(axis);
            }
            sensors.imu.push_back(converted);
        }
        sensors.hasImuCovariance = true;
        const auto config = joint_initializer_fixture::config();
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) {
                const auto index = 3 * row + col;
                sensors.imuAccelerometerCovariance[index] = config.accelerometerCovariance(row, col);
                sensors.imuGyroscopeCovariance[index] = config.gyroscopeCovariance(row, col);
                sensors.imuAccelerometerBiasCovariance[index] = config.accelerometerBiasCovariance(row, col);
                sensors.imuGyroscopeBiasCovariance[index] = config.gyroscopeBiasCovariance(row, col);
            }
    }
};

TEST(GtsamBackendAdapter, AdoptsFullInitializationAndPreservesMapIdentities)
{
    JointInitializationFixture fixture;
    std::reverse(fixture.snapshot.keyframes.begin(), fixture.snapshot.keyframes.end());
    auto initialized = ORB_SLAM3::GtsamBackendAdapter::prepareInitialization(
        fixture.snapshot, fixture.sensors, fixture.bodyFromCamera, {});
    ASSERT_NE(initialized.backend, nullptr) << initialized.diagnostic;
    ASSERT_TRUE(initialized.mapResult.accepted);
    EXPECT_EQ(initialized.mapResult.sourceVersion, fixture.snapshot.version);
    EXPECT_EQ(initialized.mapResult.sourceMapId, fixture.snapshot.mapId);
    EXPECT_EQ(initialized.mapResult.sourceTopologySignature, fixture.snapshot.topologySignature);
    ASSERT_EQ(initialized.mapResult.keyframes.size(), fixture.snapshot.keyframes.size());
    EXPECT_EQ(initialized.mapResult.landmarks.size(), fixture.snapshot.landmarks.size());
    EXPECT_LT(initialized.mapResult.keyframes.front().timestampSec,
              initialized.mapResult.keyframes.back().timestampSec);
    for (std::size_t axis = 0; axis < 3U; ++axis)
        EXPECT_NEAR(initialized.gravityMap[axis], fixture.motion.gravity(axis), 0.03);
    EXPECT_NEAR(std::sqrt(initialized.gravityMap[0] * initialized.gravityMap[0] +
        initialized.gravityMap[1] * initialized.gravityMap[1] +
        initialized.gravityMap[2] * initialized.gravityMap[2]), 9.81, 1e-12);
    ORB_SLAM3::BackendFrameInput repeated;
    repeated.timestampSec = initialized.mapResult.keyframes.back().timestampSec;
    const auto replayed = initialized.backend->optimize(repeated);
    EXPECT_TRUE(replayed.accepted);
    EXPECT_FALSE(replayed.incrementalCommitted);
    EXPECT_EQ(replayed.mapVersion, initialized.mapResult.keyframes.size());
    const auto prepared = initialized.backend->prepareMapOptimization(fixture.snapshot);
    ASSERT_NE(prepared, nullptr);
    EXPECT_TRUE(initialized.backend->commitMapOptimization(*prepared,
        [](const auto& result) { return result.accepted; }));
}

TEST(GtsamBackendAdapter, InitializationRejectsMissingMeasurementCovariance)
{
    JointInitializationFixture fixture;
    fixture.sensors.hasImuCovariance = false;
    auto result = ORB_SLAM3::GtsamBackendAdapter::prepareInitialization(
        fixture.snapshot, fixture.sensors, fixture.bodyFromCamera, {});
    EXPECT_EQ(result.backend, nullptr);
    EXPECT_FALSE(result.mapResult.accepted);
    EXPECT_EQ(result.diagnostic, "initializer requires registered IMU covariance");
    fixture.sensors.hasImuCovariance = true;
    fixture.snapshot.observations.front().octaveVariancePx2 = 0.0;
    result = ORB_SLAM3::GtsamBackendAdapter::prepareInitialization(
        fixture.snapshot, fixture.sensors, fixture.bodyFromCamera, {});
    EXPECT_EQ(result.backend, nullptr);
    EXPECT_FALSE(result.mapResult.accepted);
    EXPECT_EQ(result.diagnostic, "initializer observation registration is invalid");
}

ORB_SLAM3::BackendMapSnapshot validStereoMapSnapshot(
    std::size_t observationCount = 6U)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    snapshot.version = 23U;
    snapshot.calibration = {400.0, 400.0, 320.0, 240.0, 0.12};

    ORB_SLAM3::BackendKeyframeState keyframe;
    keyframe.id = 42U;
    keyframe.timestampSec = 1.0;
    keyframe.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                     0.0F, 1.0F, 0.0F, 0.0F,
                     0.0F, 0.0F, 1.0F, 0.0F,
                     0.0F, 0.0F, 0.0F, 1.0F};
    snapshot.keyframes.push_back(keyframe);

    for (std::size_t index = 0; index < observationCount; ++index) {
        const double x = -0.6 + 0.2 * static_cast<double>(index);
        const double y = -0.2 + 0.05 * static_cast<double>(index);
        const double z = 4.0 + 0.1 * static_cast<double>(index);
        ORB_SLAM3::BackendLandmarkState landmark;
        landmark.id = 100U + index;
        landmark.position = {static_cast<float>(x), static_cast<float>(y),
                             static_cast<float>(z)};
        snapshot.landmarks.push_back(landmark);

        ORB_SLAM3::BackendStereoObservation observation;
        observation.keyframeId = keyframe.id;
        observation.landmarkId = landmark.id;
        observation.uLeft = 400.0 * x / z + 320.0;
        observation.uRight = observation.uLeft - 400.0 * 0.12 / z;
        observation.v = 400.0 * y / z + 240.0;
        observation.sigmaPx = 0.5;
        observation.octave = 0;
        observation.octaveVariancePx2 = 0.25;
        observation.quality = 1.0;
        observation.stereoSkewPx = 0.0;
        snapshot.observations.push_back(observation);
    }
    return snapshot;
}

ORB_SLAM3::BackendMapSnapshot onePoseMap(std::uint64_t mapId,
                                         std::uint64_t keyframeId,
                                         float translationX)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    snapshot.mapId = mapId;
    snapshot.version = mapId + 10U;
    snapshot.topologySignature = mapId + 100U;
    snapshot.calibration = {400.0, 400.0, 320.0, 240.0, 0.12};
    ORB_SLAM3::BackendKeyframeState keyframe;
    keyframe.id = keyframeId;
    keyframe.timestampSec = static_cast<double>(keyframeId);
    keyframe.pose = {1.0F, 0.0F, 0.0F, translationX,
                     0.0F, 1.0F, 0.0F, 0.0F,
                     0.0F, 0.0F, 1.0F, 0.0F,
                     0.0F, 0.0F, 0.0F, 1.0F};
    snapshot.keyframes.push_back(keyframe);
    return snapshot;
}

std::array<double, 36> mergeCovariance()
{
    std::array<double, 36> covariance{};
    covariance[0] = 0.04;
    covariance[7] = 0.04;
    covariance[14] = 0.09;
    covariance[21] = 0.01;
    covariance[28] = 0.01;
    covariance[35] = 0.02;
    return covariance;
}

// Duplicate IDs must remain ambiguous even when observations use an index.
TEST(GtsamMapOptimizationAdapter, RejectsDuplicateSnapshotIds)
{
    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    auto snapshot = validStereoMapSnapshot();
    snapshot.keyframes.push_back(snapshot.keyframes.front());
    EXPECT_EQ(adapter.evaluateStereoObservation(
        snapshot, snapshot.observations.front()).reason, "invalid_keyframe_id");
    const auto duplicateKeyframes = adapter.optimizeGlobal(snapshot);
    EXPECT_FALSE(duplicateKeyframes.accepted) << duplicateKeyframes.diagnostic;
    EXPECT_EQ(duplicateKeyframes.diagnostic, "duplicate snapshot keyframe identity");
    snapshot.keyframes.pop_back();
    snapshot.landmarks.push_back(snapshot.landmarks.front());
    EXPECT_EQ(adapter.evaluateStereoObservation(
        snapshot, snapshot.observations.front()).reason, "invalid_landmark_id");
    const auto duplicateResult = adapter.optimizeGlobal(snapshot);
    EXPECT_FALSE(duplicateResult.accepted) << duplicateResult.diagnostic;
    EXPECT_EQ(duplicateResult.diagnostic, "duplicate snapshot landmark identity");
}

// Stereo NIS uses all three pixel residuals and the registered variance.
TEST(GtsamMapOptimizationAdapter, PreservesStereoResidualAndNis)
{
    auto snapshot = validStereoMapSnapshot();
    auto& observation = snapshot.observations.front();
    const auto& point = snapshot.landmarks.front().position;
    observation.uLeft = 400.0 * point[0] / point[2] + 320.0 + 0.2;
    observation.uRight = 400.0 * (point[0] - 0.12) / point[2] + 320.0 - 0.3;
    observation.v = 400.0 * point[1] / point[2] + 240.0 + 0.4;
    const auto decision = ORB_SLAM3::GtsamMapOptimizationAdapter()
        .evaluateStereoObservation(snapshot, observation);
    EXPECT_TRUE(decision.accepted);
    EXPECT_NEAR(decision.residualPx, std::sqrt(0.29), 1e-9);
    EXPECT_NEAR(decision.nis, 0.29 / 0.25, 1e-9);
}

TEST(GtsamMapOptimizationAdapter, RejectsNonpositiveStereoDisparity)
{
    auto snapshot = validStereoMapSnapshot();
    snapshot.observations.front().uRight = snapshot.observations.front().uLeft;

    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    const auto decision = adapter.evaluateStereoObservation(
        snapshot, snapshot.observations.front());

    EXPECT_FALSE(decision.accepted);
    EXPECT_EQ(decision.reason, "nonpositive_disparity");
}

TEST(GtsamMapOptimizationAdapter, ProjectsSnapshotWorldFromCameraPoseUsingItsInverse)
{
    auto snapshot = validStereoMapSnapshot(1U);
    // Backend snapshots store Twc. The camera is at x=1 m, so Tcw has
    // translation -1 m and the point projects at u=220 px.
    snapshot.keyframes.front().pose[3] = 1.0F;
    auto& observation = snapshot.observations.front();
    const auto& landmark = snapshot.landmarks.front();
    const double cameraX = static_cast<double>(landmark.position[0]) - 1.0;
    const double depth = static_cast<double>(landmark.position[2]);
    observation.uLeft = snapshot.calibration.fx * cameraX / depth +
                        snapshot.calibration.cx;
    observation.uRight = observation.uLeft -
                         snapshot.calibration.fx * snapshot.calibration.baseline /
                             depth;
    observation.v = snapshot.calibration.fy * landmark.position[1] / depth +
                    snapshot.calibration.cy;

    const auto decision =
        ORB_SLAM3::GtsamMapOptimizationAdapter().evaluateStereoObservation(
            snapshot, observation);

    ASSERT_TRUE(decision.accepted) << decision.reason;
    EXPECT_NEAR(decision.residualPx, 0.0, 1e-6);
    EXPECT_NEAR(decision.nis, 0.0, 1e-6);
}

// Large finite residuals remain in the robust optimization problem.
TEST(GtsamMapOptimizationAdapter, RetainsLargeStereoOutlierWithNisDiagnostic)
{
    auto snapshot = validStereoMapSnapshot();
    auto outlier = snapshot.observations.front();
    outlier.uLeft += 100.0;
    snapshot.observations.push_back(outlier);

    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    const auto result = adapter.optimizeGlobal(snapshot);

    ASSERT_TRUE(result.accepted);
    EXPECT_EQ(result.diagnostics.stereo.attempted, 7U);
    EXPECT_EQ(result.diagnostics.stereo.accepted, 7U);
    EXPECT_EQ(result.diagnostics.stereo.rejected, 0U);
    EXPECT_EQ(result.diagnostics.stereo.reasonCounts.count("nis_gate"), 0U);
}

// A fixed camera and one valid stereo point form a fully constrained graph.
TEST(GtsamMapOptimizationAdapter, OptimizesSingleStereoLandmark)
{
    const auto snapshot = validStereoMapSnapshot(1U);
    const auto result = ORB_SLAM3::GtsamMapOptimizationAdapter()
        .optimizeGlobal(snapshot);
    ASSERT_TRUE(result.accepted);
    EXPECT_EQ(result.diagnostics.stereo.accepted, 1U);
    ASSERT_EQ(result.landmarks.size(), 1U);
    for (std::size_t axis = 0; axis < 3U; ++axis)
        EXPECT_NEAR(result.landmarks.front().position[axis],
                    snapshot.landmarks.front().position[axis], 1e-5);
}

// No stereo measurements leave the map without a measurement graph to solve.
TEST(GtsamMapOptimizationAdapter, RejectsEmptyStereoGraph)
{
    auto snapshot = validStereoMapSnapshot(1U);
    snapshot.observations.clear();
    EXPECT_FALSE(ORB_SLAM3::GtsamMapOptimizationAdapter()
        .optimizeGlobal(snapshot).accepted);
}

// Quality and finite stereo skew must not veto a registered observation.
TEST(GtsamMapOptimizationAdapter, RetainsLowQualityLargeSkewObservation)
{
    auto snapshot = validStereoMapSnapshot();
    auto& observation = snapshot.observations.front();
    observation.quality = 0.0;
    observation.stereoSkewPx = 20.0;
    observation.uLeft += 100.0;

    const auto decision = ORB_SLAM3::GtsamMapOptimizationAdapter()
        .evaluateStereoObservation(snapshot, observation);

    EXPECT_TRUE(decision.accepted);
    EXPECT_GT(decision.nis, 7.815);
    EXPECT_TRUE(std::isfinite(decision.residualPx));
}

// Landmark scale-invariance ranges are not projection-domain validity bounds.
TEST(GtsamMapOptimizationAdapter, RetainsLandmarksOutsideDepthInvarianceRange)
{
    auto snapshot = validStereoMapSnapshot();
    auto& landmark = snapshot.landmarks.front();
    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    landmark.minimumDepthMeters = 100.0;
    landmark.maximumDepthMeters = 200.0;
    EXPECT_TRUE(adapter.evaluateStereoObservation(
        snapshot, snapshot.observations.front()).accepted);
    landmark.minimumDepthMeters = 0.01;
    landmark.maximumDepthMeters = 0.02;
    EXPECT_TRUE(adapter.evaluateStereoObservation(
        snapshot, snapshot.observations.front()).accepted);
}

// Removing quality gates preserves actual covariance and projection failures.
TEST(GtsamMapOptimizationAdapter, RejectsInvalidStereoModelInputs)
{
    auto snapshot = validStereoMapSnapshot();
    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    auto observation = snapshot.observations.front();
    observation.stereoSkewPx = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(adapter.evaluateStereoObservation(snapshot, observation).reason,
              "invalid_stereo_skew");
    observation = snapshot.observations.front();
    observation.octaveVariancePx2 = 0.0;
    EXPECT_EQ(adapter.evaluateStereoObservation(snapshot, observation).reason,
              "invalid_covariance");
    snapshot.landmarks.front().position[2] = -1.0F;
    EXPECT_EQ(adapter.evaluateStereoObservation(
        snapshot, snapshot.observations.front()).reason, "behind_camera");
}

TEST(GtsamMapOptimizationAdapter, ReportsOptimizedLandmarkDegreeHistogram)
{
    auto snapshot = validStereoMapSnapshot();
    for (std::size_t index = 0; index < snapshot.landmarks.size(); ++index)
        snapshot.landmarks[index].observationDegree = index + 1U;

    const ORB_SLAM3::BackendMapResult result =
        ORB_SLAM3::GtsamMapOptimizationAdapter().optimizeGlobal(snapshot);

    ASSERT_TRUE(result.accepted);
    EXPECT_EQ(result.diagnostics.landmarkObservationDegree.count, 6U);
    std::uint64_t histogramCount = 0U;
    for (const std::uint64_t bin :
         result.diagnostics.landmarkObservationDegree.bins)
        histogramCount += bin;
    EXPECT_EQ(histogramCount, 6U);
}

TEST(GtsamMapOptimizationAdapter,
     MergeOptimizationUsesMeasuredCovarianceAndHistoricalAnchor)
{
    const auto current = onePoseMap(7U, 20U, 0.0F);
    const auto historical = onePoseMap(2U, 10U, 0.0F);
    std::array<float, 16> currentFromCandidate = {
        1.0F, 0.0F, 0.0F, 5.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F};
    ORB_SLAM3::BackendMergeSnapshot transaction;
    ASSERT_TRUE(ORB_SLAM3::GtsamMapAdapter::prepareMerge(
        current, historical, 20U, 10U, currentFromCandidate,
        mergeCovariance(), &transaction));
    ASSERT_EQ(transaction.graph.keyframes.size(), 2U);
    ASSERT_EQ(transaction.graph.graphEdges.size(), 1U);
    transaction.graph.graphEdges.back().sigma =
        std::numeric_limits<double>::quiet_NaN();
    for (auto& keyframe : transaction.graph.keyframes)
        if (keyframe.id == 20U)
            keyframe.pose[3] += 2.0F;

    const auto result =
        ORB_SLAM3::GtsamMapOptimizationAdapter().optimizeMerge(transaction);

    ASSERT_TRUE(result.accepted);
    EXPECT_EQ(result.currentSource.mapId, 7U);
    EXPECT_EQ(result.candidateSource.mapId, 2U);
    const auto historicalPose = std::find_if(
        result.keyframes.begin(), result.keyframes.end(),
        [](const auto& state) { return state.id == 10U; });
    const auto currentPose = std::find_if(
        result.keyframes.begin(), result.keyframes.end(),
        [](const auto& state) { return state.id == 20U; });
    ASSERT_NE(historicalPose, result.keyframes.end());
    ASSERT_NE(currentPose, result.keyframes.end());
    EXPECT_NEAR(historicalPose->pose[3], 0.0F, 1e-6F);
    EXPECT_NEAR(currentPose->pose[3], -5.0F, 1e-3F);
}

TEST(GtsamMapOptimizationAdapter, RejectsMergeWithDuplicateKeyframeIdentity)
{
    const auto current = onePoseMap(7U, 10U, 0.0F);
    const auto historical = onePoseMap(2U, 10U, 0.0F);
    const std::array<float, 16> identity = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F};
    ORB_SLAM3::BackendMergeSnapshot transaction;

    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::prepareMerge(
        current, historical, 10U, 10U, identity, mergeCovariance(),
        &transaction));
}

TEST(GtsamMapOptimizationAdapter, BuildsBoundedMergeCovarianceFromGeometry)
{
    const auto highQuality = ORB_SLAM3::GtsamMapAdapter::mergeCovariance(
        80U, 100U, 0.02);
    const auto lowQuality = ORB_SLAM3::GtsamMapAdapter::mergeCovariance(
        8U, 100U, 0.20);

    ASSERT_TRUE(highQuality.has_value());
    ASSERT_TRUE(lowQuality.has_value());
    for (std::size_t index = 0; index < 6U; ++index) {
        const std::size_t diagonal = index * 6U + index;
        EXPECT_GT((*highQuality)[diagonal], 0.0);
        EXPECT_GE((*lowQuality)[diagonal], (*highQuality)[diagonal]);
    }
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::mergeCovariance(
        0U, 100U, 0.02).has_value());
    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::mergeCovariance(
        10U, 5U, 0.02).has_value());
}

TEST(GtsamMapOptimizationAdapter, RejectsStaleMergeResultSource)
{
    const auto current = onePoseMap(7U, 20U, 0.0F);
    const auto historical = onePoseMap(2U, 10U, 0.0F);
    const std::array<float, 16> identity = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F};
    ORB_SLAM3::BackendMergeSnapshot transaction;
    ASSERT_TRUE(ORB_SLAM3::GtsamMapAdapter::prepareMerge(
        current, historical, 20U, 10U, identity, mergeCovariance(),
        &transaction));
    auto result = ORB_SLAM3::GtsamMapOptimizationAdapter().optimizeMerge(
        transaction);
    ASSERT_TRUE(result.accepted);
    ++result.currentSource.version;

    EXPECT_FALSE(ORB_SLAM3::GtsamMapAdapter::validateMergeResult(
        transaction, result).accepted);
}

// Conflicting but valid stereo observations keep their landmark in the graph.
TEST(GtsamMapOptimizationAdapter,
     RetainsLandmarkWithLargeStereoResiduals)
{
    auto snapshot = validStereoMapSnapshot();
    ORB_SLAM3::BackendLandmarkState rejectedLandmark;
    rejectedLandmark.id = 999U;
    rejectedLandmark.position = {0.0F, 0.0F, 4.0F};
    snapshot.landmarks.push_back(rejectedLandmark);

    ORB_SLAM3::BackendStereoObservation first = snapshot.observations.front();
    first.landmarkId = rejectedLandmark.id;
    first.uLeft += 100.0;
    ORB_SLAM3::BackendStereoObservation second = first;
    second.uLeft += 100.0;
    snapshot.observations.push_back(first);
    snapshot.observations.push_back(second);

    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    const auto result = adapter.optimizeGlobal(snapshot);

    ASSERT_TRUE(result.accepted);
    EXPECT_EQ(result.diagnostics.stereo.reasonCounts.count("nis_gate"), 0U);
    EXPECT_EQ(result.diagnostics.stereo.accepted, snapshot.observations.size());
    EXPECT_TRUE(std::any_of(result.landmarks.begin(), result.landmarks.end(),
        [&rejectedLandmark](const auto& landmark) {
            return landmark.id == rejectedLandmark.id;
        }));
}

// Residual magnitude alone must not delete a nonanchor keyframe.
TEST(GtsamMapOptimizationAdapter,
     RetainsNonanchorKeyframeWithLargeStereoResiduals)
{
    auto snapshot = validStereoMapSnapshot();
    const auto anchorObservations = snapshot.observations;
    auto rejectedKeyframe = snapshot.keyframes.front();
    rejectedKeyframe.id = 43U;
    snapshot.keyframes.push_back(rejectedKeyframe);
    for (const auto& anchorObservation : anchorObservations) {
        snapshot.observations.push_back(anchorObservation);
        snapshot.observations.push_back(anchorObservation);
        auto first = anchorObservation;
        first.keyframeId = rejectedKeyframe.id;
        first.uLeft += 1000.0;
        auto second = first;
        second.uLeft += 1000.0;
        snapshot.observations.push_back(first);
        snapshot.observations.push_back(second);
    }

    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    const auto result = adapter.optimizeGlobal(snapshot);

    ASSERT_TRUE(result.accepted)
        << testing::PrintToString(result.diagnostics.stereo.reasonCounts);
    ASSERT_EQ(result.keyframes.size(), 2U)
        << testing::PrintToString(result.diagnostics.stereo.reasonCounts);
    EXPECT_TRUE(std::any_of(result.keyframes.begin(), result.keyframes.end(),
        [&rejectedKeyframe](const auto& keyframe) {
            return keyframe.id == rejectedKeyframe.id;
        }));
}

TEST(GtsamBackendAdapter, AcceptsFiniteMetricStereoBootstrap)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 0.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        input.stereo.push_back(observation);
    }
    const auto result = backend.optimize(input);
    EXPECT_TRUE(result.accepted) << result.diagnostic;
    EXPECT_TRUE(result.incrementalCommitted);
    EXPECT_EQ(result.mapVersion, 1U);
    EXPECT_TRUE(cv::checkRange(result.bodyPose));
    EXPECT_EQ(result.sensorDiagnostics.stereo.input, 6U);
    EXPECT_EQ(result.sensorDiagnostics.stereo.accepted, 6U);
    EXPECT_EQ(result.sensorDiagnostics.stereo.rejected, 0U);
    EXPECT_DOUBLE_EQ(result.sensorDiagnostics.stereo.covariance[0], 1.0);
    EXPECT_DOUBLE_EQ(result.sensorDiagnostics.stereo.information[0], 1.0);
    EXPECT_DOUBLE_EQ(result.sensorDiagnostics.stereo.huberK, 1.345);
    EXPECT_FALSE(result.sensorDiagnostics.imu.present);
    EXPECT_FALSE(result.sensorDiagnostics.dvl.enabled);
    EXPECT_FALSE(result.sensorDiagnostics.dvl.present);
    EXPECT_FALSE(result.sensorDiagnostics.pressure.enabled);
    EXPECT_FALSE(result.sensorDiagnostics.pressure.present);
    EXPECT_GT(result.graphDiagnostics.totalFactors, 0U);
    EXPECT_GT(result.graphDiagnostics.totalValues, 0U);
}

TEST(GtsamMapOptimizationAdapter, KeepsOnlyVerifiedUnambiguousLandmarkIdentities)
{
    std::vector<ORB_SLAM3::BackendLoopPointCorrespondence> correspondences;
    for (std::uint64_t index = 0; index < 8U; ++index) {
        ORB_SLAM3::BackendLoopPointCorrespondence correspondence;
        correspondence.currentLandmarkId = index + 100U;
        correspondence.candidateLandmarkId = index + 200U;
        correspondence.candidatePoint = {
            static_cast<double>(index % 2U), static_cast<double>((index / 2U) % 2U),
            static_cast<double>(index / 4U)};
        correspondence.currentPoint = correspondence.candidatePoint;
        correspondence.currentPoint[0] += 0.1;
        correspondences.push_back(correspondence);
    }
    auto outlier = correspondences.front();
    outlier.currentLandmarkId = 999U;
    outlier.candidateLandmarkId = 1999U;
    outlier.currentPoint[0] += 10.0;
    correspondences.push_back(outlier);
    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    const auto verified = adapter.verifyLoopCandidateSE3(correspondences, 0.25, 6U);
    ASSERT_TRUE(verified.accepted);
    ASSERT_EQ(verified.landmarkReplacements.size(), 8U);
    for (std::size_t index = 0; index < 8U; ++index) {
        EXPECT_EQ(verified.landmarkReplacements[index].replacedId, index + 100U);
        EXPECT_EQ(verified.landmarkReplacements[index].retainedId, index + 200U);
    }
    auto ambiguous = correspondences.front();
    ambiguous.candidateLandmarkId = 2999U;
    correspondences.push_back(ambiguous);
    EXPECT_TRUE(adapter.verifyLoopCandidateSE3(correspondences, 0.25, 6U)
                    .landmarkReplacements.empty());
}

TEST(GtsamBackendAdapter, RefinesVisualBootstrapWithoutIncrementalGraph)
{
    const cv::Mat camera = (cv::Mat_<float>(3, 3)
        << 400.0F, 0.0F, 320.0F, 0.0F, 400.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    ORB_SLAM3::BackendFrameInput input;
    for (std::size_t index = 0; index < 12U; ++index) {
        const double x = -0.8 + 0.4 * static_cast<double>(index % 4U);
        const double y = -0.4 + 0.4 * static_cast<double>(index / 4U);
        const double z = 4.0 + 0.2 * static_cast<double>(index % 3U);
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = 100U + index;
        observation.uLeft = static_cast<float>(400.0 * x / z + 320.0);
        observation.uRight = static_cast<float>(400.0 * (x - 0.12) / z + 320.0);
        observation.vLeft = static_cast<float>(400.0 * y / z + 240.0);
        observation.hasLandmark = true;
        observation.landmarkMap = {x, y, z};
        input.stereo.push_back(observation);
    }
    cv::Mat initialCameraFromMap = cv::Mat::eye(4, 4, CV_32F);
    initialCameraFromMap.at<float>(0, 3) = -0.03F;
    const auto result = ORB_SLAM3::GtsamBackendAdapter::refineVisualPose(
        input, camera, 48.0F, cv::Mat::eye(4, 4, CV_32F), initialCameraFromMap);
    ASSERT_TRUE(result.accepted) << result.diagnostic;
    EXPECT_FALSE(result.incrementalCommitted);
    EXPECT_EQ(result.mapVersion, 0U);
    EXPECT_EQ(result.graphDiagnostics.totalFactors, 0U);
    EXPECT_NEAR(result.bodyPose.at<float>(0, 3), 0.0, 1e-3);
}

TEST(GtsamBackendAdapter, FormatsStructuredSensorAndGraphDiagnostics)
{
    ORB_SLAM3::BackendFrameResult result;
    result.accepted = true;
    result.sensorDiagnostics.stereo.input = 8U;
    result.sensorDiagnostics.stereo.accepted = 6U;
    result.sensorDiagnostics.stereo.rejected = 2U;
    result.sensorDiagnostics.stereo.huberK = 1.345;
    result.sensorDiagnostics.dvl.enabled = false;
    result.sensorDiagnostics.dvl.present = false;
    result.sensorDiagnostics.pressure.enabled = false;
    result.sensorDiagnostics.pressure.present = false;
    result.graphDiagnostics.submittedFactors = 12U;
    result.graphDiagnostics.totalFactors = 42U;

    const std::string message =
        ORB_SLAM3::AquaBackendAdapter::formatDynamicSensorDiagnostics(
            1.25, result);

    EXPECT_NE(message.find(" t=1.250000000 "), std::string::npos);
    EXPECT_NE(message.find("stereo={input=8,accepted=6,rejected=2"),
              std::string::npos);
    EXPECT_NE(message.find("huber_k=1.345"), std::string::npos);
    EXPECT_NE(message.find("dvl={enabled=0,present=0"), std::string::npos);
    EXPECT_NE(message.find("source_age_s=unavailable"), std::string::npos);
    EXPECT_NE(message.find("baseline_source=unavailable"), std::string::npos);
    EXPECT_NE(message.find("baseline_covariance=unavailable"), std::string::npos);
    EXPECT_NE(message.find("dynamic_covariance=unavailable"), std::string::npos);
    EXPECT_NE(message.find("dynamic_weight=unavailable"), std::string::npos);
    EXPECT_NE(message.find("downweighted=0"), std::string::npos);
    EXPECT_NE(message.find("pressure={enabled=0,present=0"),
              std::string::npos);
    EXPECT_NE(message.find("graph={submitted_factors=12,total_factors=42"),
              std::string::npos);
}

TEST(GtsamBackendAdapter, FormatsCompleteDvlWeightDiagnostics)
{
    ORB_SLAM3::BackendFrameResult result;
    result.accepted = true;
    result.incrementalCommitted = true;
    auto& dvl = result.sensorDiagnostics.dvl;
    dvl.enabled = true;
    dvl.present = true;
    dvl.healthAccepted = true;
    dvl.timeOffsetSec = -0.05;
    dvl.sourceAgeSec = 0.05;
    dvl.baselineSource = "configured_baseline";
    dvl.baselineCovariance = {0.04, 0.0, 0.0,
                              0.0, 0.04, 0.0,
                              0.0, 0.0, 0.04};
    dvl.dynamicCovariance = {0.08, 0.0, 0.0,
                             0.0, 0.08, 0.0,
                             0.0, 0.0, 0.08};
    dvl.covariance = {0.1, 0.0, 0.0,
                      0.0, 0.1, 0.0,
                      0.0, 0.0, 0.1};
    dvl.information = {10.0, 0.0, 0.0,
                       0.0, 10.0, 0.0,
                       0.0, 0.0, 10.0};
    dvl.weightAvailable = true;
    dvl.nisAvailable = true;
    dvl.nis = 8.0;
    dvl.dynamicWeight = 0.8;
    dvl.downweighted = true;
    result.factorDiagnostics.dvl.attempted = 1U;
    result.factorDiagnostics.dvl.accepted = 1U;
    result.factorDiagnostics.dvl.downweighted = 1U;
    result.factorDiagnostics.dvl.reason = "accepted";

    const std::string message =
        ORB_SLAM3::AquaBackendAdapter::formatDynamicSensorDiagnostics(
            1.25, result);

    EXPECT_NE(message.find("source_age_s=0.05"), std::string::npos);
    EXPECT_NE(message.find("baseline_source=configured_baseline"),
              std::string::npos);
    EXPECT_NE(message.find("baseline_covariance=[0.04"), std::string::npos);
    EXPECT_NE(message.find("dynamic_covariance=[0.08"), std::string::npos);
    EXPECT_NE(message.find("nis=8"), std::string::npos);
    EXPECT_NE(message.find("dynamic_weight=0.8"), std::string::npos);
    EXPECT_NE(message.find("downweighted=1"), std::string::npos);
    EXPECT_NE(message.find("downweighted_count=1"), std::string::npos);
}

TEST(GtsamBackendAdapter, PredictsFiniteCameraPoseForVisualRecovery)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    bodyFromCamera.at<float>(0, 3) = 0.1F;
    ORB_SLAM3::GtsamBackendAdapter backend(
        camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput first;
    first.timestampSec = 0.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        first.stereo.push_back(observation);
    }
    ASSERT_TRUE(backend.optimize(first).incrementalCommitted);

    ORB_SLAM3::BackendFrameInput second = first;
    second.timestampSec = 0.1;
    second.hasImuCovariance = true;
    for (int index = 0; index <= 10; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.01 * static_cast<double>(index);
        sample.acceleration = {0.0, 0.0, 9.81};
        second.imu.push_back(sample);
    }
    const auto secondResult = backend.optimize(second);
    ASSERT_TRUE(secondResult.incrementalCommitted);
    ASSERT_TRUE(secondResult.sensorDiagnostics.imu.present);
    EXPECT_EQ(secondResult.sensorDiagnostics.imu.sampleCount, 11U);
    EXPECT_NEAR(secondResult.sensorDiagnostics.imu.spanSec, 0.1, 1e-12);
    EXPECT_NEAR(secondResult.sensorDiagnostics.imu.maximumGapSec, 0.01, 1e-12);
    EXPECT_NEAR(secondResult.sensorDiagnostics.imu.accelerationRms, 9.81, 1e-12);
    EXPECT_DOUBLE_EQ(
        secondResult.sensorDiagnostics.imu.covariance.accelerometer[0], 1.0);
    EXPECT_DOUBLE_EQ(
        secondResult.sensorDiagnostics.imu.information.accelerometer[0], 1.0);

    std::vector<ORB_SLAM3::ImuSample> recoveryImu;
    for (int index = 0; index <= 100; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.1 + 0.01 * static_cast<double>(index);
        sample.acceleration = {0.0, 0.0, 9.81};
        recoveryImu.push_back(sample);
    }

    const auto prediction = backend.predictCameraPose(1.1, recoveryImu);

    ASSERT_TRUE(prediction.accepted) << prediction.diagnostic;
    ASSERT_EQ(prediction.cameraFromMap.rows, 4);
    ASSERT_EQ(prediction.cameraFromMap.cols, 4);
    EXPECT_TRUE(cv::checkRange(prediction.cameraFromMap));
    EXPECT_NEAR(prediction.cameraFromMap.at<float>(0, 3), -0.1F, 0.05F);
}

TEST(GtsamBackendAdapter, RebasePreservesBackendOwnedVelocityAndBias)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    const cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(
        camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput first;
    first.timestampSec = 0.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        first.stereo.push_back(observation);
    }
    const auto firstResult = backend.optimize(first);
    ASSERT_TRUE(firstResult.incrementalCommitted);

    ORB_SLAM3::BackendFrameInput second = first;
    second.timestampSec = 0.1;
    second.hasImuCovariance = true;
    for (int index = 0; index <= 10; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.01 * static_cast<double>(index);
        sample.acceleration = {1.0, 0.0, 9.81};
        second.imu.push_back(sample);
    }
    const auto secondResult = backend.optimize(second);
    ASSERT_TRUE(secondResult.incrementalCommitted);

    const auto keyframeState = [](std::uint64_t id, double timestamp,
                                  const cv::Mat& pose) {
        ORB_SLAM3::BackendKeyframeState state;
        state.id = id;
        state.timestampSec = timestamp;
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                state.pose[static_cast<std::size_t>(row * 4 + col)] =
                    pose.at<float>(row, col);
        state.velocity = {100.0F, 0.0F, 0.0F};
        state.accelerometerBias = {10.0F, 0.0F, 0.0F};
        return state;
    };
    ORB_SLAM3::BackendMapSnapshot map;
    map.keyframes.push_back(keyframeState(0U, 0.0, firstResult.bodyPose));
    map.keyframes.push_back(keyframeState(1U, 0.1, secondResult.bodyPose));
    ASSERT_TRUE(backend.rebase(map));

    std::vector<ORB_SLAM3::ImuSample> recoveryImu;
    for (int index = 0; index <= 10; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.1 + 0.01 * static_cast<double>(index);
        sample.acceleration = {1.0, 0.0, 9.81};
        recoveryImu.push_back(sample);
    }
    const auto prediction = backend.predictCameraPose(0.2, recoveryImu);

    ASSERT_TRUE(prediction.accepted) << prediction.diagnostic;
    EXPECT_LT(std::abs(prediction.cameraFromMap.at<float>(0, 3)), 1.0F);
}

TEST(GtsamBackendAdapter, DoesNotAttemptDvlFactorWithoutExplicitDvlMode)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 0.0;
    ORB_SLAM3::DvlMeasurement dvl;
    dvl.velocity = {0.2, -0.1, 0.0};
    input.dvl.push_back(dvl);
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        input.stereo.push_back(observation);
    }

    const auto result = backend.optimize(input);

    ASSERT_TRUE(result.accepted) << result.diagnostic;
    EXPECT_EQ(result.factorDiagnostics.dvl.attempted, 0U);
    EXPECT_EQ(result.factorDiagnostics.dvl.accepted, 0U);
    EXPECT_FALSE(result.sensorDiagnostics.dvl.enabled);
    EXPECT_TRUE(result.sensorDiagnostics.dvl.present);
    EXPECT_FALSE(result.sensorDiagnostics.dvl.weightAvailable);
}

TEST(GtsamBackendAdapter, AcceptsDvlFactorWithExplicitRegisteredMode)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamOptionalSensorConfig optionalSensors;
    optionalSensors.dvlEnabled = true;
    optionalSensors.bodyFromDvl = cv::Mat::eye(4, 4, CV_32F);
    optionalSensors.dvlTrackMode = ORB_SLAM3::DvlTrackMode::BottomTrack;
    optionalSensors.uncertaintyProfilePath =
        UW_DYNAMIC_BACKEND_ROOT_PATH "/test/fixtures/uncertainty_valid.yaml";
    ORB_SLAM3::GtsamBackendAdapter backend(
        camera, 63.0F, bodyFromCamera, cv::Mat(), cv::Mat(),
        optionalSensors);
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 0.0;
    ORB_SLAM3::DvlMeasurement dvl;
    dvl.timestampSec = 0.0;
    dvl.velocity = {0.2, -0.1, 0.0};
    dvl.covariance = {0.04, 0.0, 0.0,
                      0.0, 0.04, 0.0,
                      0.0, 0.0, 0.04};
    dvl.angularVelocityBody = {0.0, 0.0, 0.0};
    dvl.healthAccepted = true;
    dvl.trackMode = ORB_SLAM3::DvlTrackMode::BottomTrack;
    dvl.validBeamRatio = 1.0;
    dvl.altitudeValid = true;
    dvl.altitudeMeters = 2.0;
    dvl.errorVelocityMetersPerSec = 0.01;
    input.dvl.push_back(dvl);
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        input.stereo.push_back(observation);
    }

    const auto seed = backend.optimize(input);
    ASSERT_TRUE(seed.accepted) << seed.diagnostic;
    EXPECT_EQ(seed.factorDiagnostics.dvl.attempted, 0U);

    input.timestampSec = 0.1;
    input.dvl.front().timestampSec = 0.1;
    input.hasImuCovariance = true;
    for (int index = 0; index <= 10; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.01 * static_cast<double>(index);
        sample.acceleration = {0.0, 0.0, 9.81};
        input.imu.push_back(sample);
    }
    const auto result = backend.optimize(input);

    ASSERT_TRUE(result.accepted) << result.diagnostic;
    EXPECT_EQ(result.factorDiagnostics.dvl.attempted, 1U);
    EXPECT_EQ(result.factorDiagnostics.dvl.accepted, 1U);
    EXPECT_TRUE(result.sensorDiagnostics.dvl.enabled);
    EXPECT_TRUE(result.sensorDiagnostics.dvl.present);
    EXPECT_TRUE(result.sensorDiagnostics.dvl.healthAccepted);
    EXPECT_TRUE(result.sensorDiagnostics.dvl.weightAvailable);
    EXPECT_NEAR(result.sensorDiagnostics.dvl.covariance[0], 0.0004, 1e-12);
    EXPECT_NEAR(result.sensorDiagnostics.dvl.information[0], 2500.0, 1e-9);
    EXPECT_TRUE(result.sensorDiagnostics.dvl.nisAvailable);
    EXPECT_TRUE(std::isfinite(result.sensorDiagnostics.dvl.nis));
}

TEST(GtsamBackendAdapter, IntegratesEveryDvlUpdateInCameraInterval)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    ORB_SLAM3::GtsamOptionalSensorConfig optionalSensors;
    optionalSensors.dvlEnabled = true;
    optionalSensors.bodyFromDvl = cv::Mat::eye(4, 4, CV_32F);
    optionalSensors.dvlTrackMode = ORB_SLAM3::DvlTrackMode::BottomTrack;
    optionalSensors.uncertaintyProfilePath =
        UW_DYNAMIC_BACKEND_ROOT_PATH "/test/fixtures/uncertainty_valid.yaml";
    ORB_SLAM3::GtsamBackendAdapter backend(
        camera, 63.0F, cv::Mat::eye(4, 4, CV_32F),
        cv::Mat(), cv::Mat(), optionalSensors);

    ORB_SLAM3::BackendFrameInput first;
    first.timestampSec = 0.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        first.stereo.push_back(observation);
    }
    ORB_SLAM3::DvlMeasurement seed;
    seed.timestampSec = 0.0;
    seed.velocity = {1.0, 0.0, 0.0};
    seed.covariance = {0.04, 0.0, 0.0,
                       0.0, 0.04, 0.0,
                       0.0, 0.0, 0.04};
    seed.healthAccepted = true;
    seed.trackMode = ORB_SLAM3::DvlTrackMode::BottomTrack;
    seed.validBeamRatio = 1.0;
    seed.altitudeValid = true;
    seed.altitudeMeters = 2.0;
    seed.errorVelocityMetersPerSec = 0.01;
    first.dvl.push_back(seed);
    const auto seedResult = backend.optimize(first);
    ASSERT_TRUE(seedResult.incrementalCommitted);

    ORB_SLAM3::BackendFrameInput second = first;
    second.timestampSec = 0.1;
    second.dvl.clear();
    ORB_SLAM3::DvlMeasurement update = seed;
    update.timestampSec = 0.05;
    update.velocity = {2.0, 0.0, 0.0};
    second.dvl.push_back(update);
    second.hasImuCovariance = true;
    second.imu.clear();
    for (int index = 0; index <= 10; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.01 * static_cast<double>(index);
        sample.acceleration = {0.0, 0.0, 9.81};
        second.imu.push_back(sample);
    }

    const auto result = backend.optimize(second);
    ASSERT_TRUE(result.incrementalCommitted) << result.diagnostic;
    ASSERT_EQ(result.factorDiagnostics.dvl.accepted, 1U)
        << result.factorDiagnostics.dvl.reason;
    ASSERT_TRUE(result.sensorDiagnostics.dvl.weightAvailable);
    ASSERT_TRUE(std::isfinite(result.sensorDiagnostics.dvl.dynamicWeight));
    EXPECT_GT(result.sensorDiagnostics.dvl.dynamicWeight, 0.0);
    EXPECT_LE(result.sensorDiagnostics.dvl.dynamicWeight, 1.0);
    const double expectedIntegratedCovariance =
        seedResult.sensorDiagnostics.dvl.dynamicCovariance[0] * 0.05 * 0.05 +
        result.sensorDiagnostics.dvl.dynamicCovariance[0] * 0.05 * 0.05;
    EXPECT_NEAR(result.sensorDiagnostics.dvl.covariance[0] *
                    result.sensorDiagnostics.dvl.dynamicWeight,
                expectedIntegratedCovariance, 1e-12);
}

TEST(GtsamBackendAdapter, DvlModeRejectsUnknownRegisteredTrackMode)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    ORB_SLAM3::GtsamOptionalSensorConfig optionalSensors;
    optionalSensors.dvlEnabled = true;
    optionalSensors.bodyFromDvl = cv::Mat::eye(4, 4, CV_32F);
    optionalSensors.uncertaintyProfilePath =
        UW_DYNAMIC_BACKEND_ROOT_PATH "/test/fixtures/uncertainty_valid.yaml";

    EXPECT_THROW(ORB_SLAM3::GtsamBackendAdapter(
                     camera, 63.0F, cv::Mat::eye(4, 4, CV_32F),
                     cv::Mat(), cv::Mat(), optionalSensors),
                 std::invalid_argument);
}

TEST(GtsamOptionalSensorConfig, TrackingDvlModeUsesImuDvlCalibration)
{
    cv::Mat imuFromCamera = cv::Mat::eye(4, 4, CV_32F);
    imuFromCamera.at<float>(0, 3) = 0.2F;
    cv::Mat dvlFromCamera = cv::Mat::eye(4, 4, CV_32F);
    dvlFromCamera.at<float>(1, 3) = -0.3F;
    const ORB_SLAM3::IMU::Calib calibration(imuFromCamera, dvlFromCamera);

    const auto optionalSensors =
        ORB_SLAM3::GtsamOptionalSensorConfig::fromTrackingSensor(
            true, &calibration,
            ORB_SLAM3::DvlTrackMode::BottomTrack, "uncertainty.yaml");

    EXPECT_TRUE(optionalSensors.dvlEnabled);
    EXPECT_EQ(cv::norm(optionalSensors.bodyFromDvl,
                       calibration.mT_imu_dvl, cv::NORM_INF),
              0.0);
    EXPECT_EQ(optionalSensors.dvlTrackMode,
              ORB_SLAM3::DvlTrackMode::BottomTrack);
    EXPECT_EQ(optionalSensors.uncertaintyProfilePath, "uncertainty.yaml");
}

TEST(GtsamOptionalSensorConfig, TrackingDvlModeRejectsIncompleteRegistration)
{
    const cv::Mat transform = cv::Mat::eye(4, 4, CV_32F);
    const ORB_SLAM3::IMU::Calib calibration(transform, transform);

    EXPECT_THROW(
        ORB_SLAM3::GtsamOptionalSensorConfig::fromTrackingSensor(
            true, &calibration),
        std::invalid_argument);
    EXPECT_THROW(
        ORB_SLAM3::GtsamOptionalSensorConfig::fromTrackingSensor(
            true, &calibration,
            ORB_SLAM3::DvlTrackMode::BottomTrack),
        std::invalid_argument);
}

TEST(GtsamOptionalSensorConfig, TrackingImuStereoLeavesDvlDisabled)
{
    const cv::Mat transform = cv::Mat::eye(4, 4, CV_32F);
    const ORB_SLAM3::IMU::Calib calibration(transform, transform);

    const auto optionalSensors =
        ORB_SLAM3::GtsamOptionalSensorConfig::fromTrackingSensor(
            false, &calibration);

    EXPECT_FALSE(optionalSensors.dvlEnabled);
    EXPECT_TRUE(optionalSensors.bodyFromDvl.empty());
}

TEST(GtsamBackendAdapter, RepeatedOptimizationOfSameCameraFrameIsIdempotent)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 1.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        input.stereo.push_back(observation);
    }

    const auto first = backend.optimize(input);
    const auto repeated = backend.optimize(input);

    ASSERT_TRUE(first.accepted);
    EXPECT_TRUE(first.incrementalCommitted);
    EXPECT_TRUE(repeated.accepted);
    EXPECT_FALSE(repeated.incrementalCommitted);
    EXPECT_EQ(repeated.mapVersion, first.mapVersion);
    EXPECT_EQ(cv::norm(repeated.bodyPose, first.bodyPose, cv::NORM_INF), 0.0);
}

TEST(GtsamBackendAdapter, RetainedMapOptimizationUsesTimestampAndLandmarkIdentity)
{
    const cv::Mat camera = (cv::Mat_<float>(3, 3)
        << 400.0F, 0.0F, 320.0F, 0.0F, 400.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    ORB_SLAM3::GtsamInitialNavigationState initialNavigation;
    initialNavigation.velocityMap = {0.3, -0.2, 0.1};
    initialNavigation.accelerometerBias = {0.02, -0.01, 0.03};
    initialNavigation.gyroscopeBias = {0.001, 0.002, -0.003};
    ORB_SLAM3::GtsamBackendAdapter backend(
        camera, 48.0F, cv::Mat::eye(4, 4, CV_32F),
        cv::Mat(), cv::Mat(), ORB_SLAM3::GtsamOptionalSensorConfig(),
        initialNavigation);
    auto snapshot = validStereoMapSnapshot();
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = snapshot.keyframes.front().timestampSec;
    for (const auto& source : snapshot.observations) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = source.landmarkId;
        observation.uLeft = source.uLeft;
        observation.uRight = source.uRight;
        observation.vLeft = source.v;
        input.stereo.push_back(observation);
    }
    ASSERT_TRUE(backend.optimize(input).incrementalCommitted);
    const auto originalLandmarks = snapshot.landmarks;
    for (auto& landmark : snapshot.landmarks)
        landmark.position[0] += 0.1F;
    const auto candidate = backend.prepareMapOptimization(snapshot);
    ASSERT_TRUE(candidate);
    ASSERT_TRUE(candidate->result().accepted);
    ASSERT_EQ(candidate->result().keyframes.size(), 1U);
    EXPECT_EQ(candidate->result().keyframes.front().id, 42U);
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        EXPECT_NEAR(candidate->result().keyframes.front().velocity[axis],
                    initialNavigation.velocityMap[axis], 1e-7);
        EXPECT_NEAR(candidate->result().keyframes.front().accelerometerBias[axis],
                    initialNavigation.accelerometerBias[axis], 1e-7);
        EXPECT_NEAR(candidate->result().keyframes.front().gyroscopeBias[axis],
                    initialNavigation.gyroscopeBias[axis], 1e-7);
    }
    ASSERT_EQ(candidate->result().landmarks.size(), originalLandmarks.size());
    for (std::size_t index = 0; index < originalLandmarks.size(); ++index) {
        EXPECT_EQ(candidate->result().landmarks[index].id, originalLandmarks[index].id);
        for (std::size_t axis = 0; axis < 3U; ++axis)
            EXPECT_NEAR(candidate->result().landmarks[index].position[axis],
                        originalLandmarks[index].position[axis], 1e-4);
    }
    EXPECT_FALSE(backend.commitMapOptimization(*candidate,
        [](const ORB_SLAM3::BackendMapResult&) { return false; }));
    EXPECT_TRUE(backend.commitMapOptimization(*candidate,
        [](const ORB_SLAM3::BackendMapResult&) { return true; }));
    bool called = false;
    EXPECT_FALSE(backend.commitMapOptimization(*candidate,
        [&called](const ORB_SLAM3::BackendMapResult&) { called = true; return true; }));
    EXPECT_FALSE(called);
    snapshot.keyframes.front().timestampSec += 0.01;
    EXPECT_FALSE(backend.prepareMapOptimization(snapshot));
}

TEST(GtsamBackendAdapter, RejectsRebaseOlderThanCommittedGraph)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 2.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        input.stereo.push_back(observation);
    }
    ASSERT_TRUE(backend.optimize(input).incrementalCommitted);

    ORB_SLAM3::BackendMapSnapshot stale;
    ORB_SLAM3::BackendKeyframeState keyframe;
    keyframe.id = 4;
    keyframe.timestampSec = 1.0;
    keyframe.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                     0.0F, 1.0F, 0.0F, 0.0F,
                     0.0F, 0.0F, 1.0F, 0.0F,
                     0.0F, 0.0F, 0.0F, 1.0F};
    stale.keyframes.push_back(keyframe);

    EXPECT_FALSE(backend.rebase(stale));
}

TEST(GtsamBackendAdapter, ResetStartsNewGraphAtCommittedMapPose)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 1.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        input.stereo.push_back(observation);
    }
    ASSERT_TRUE(backend.optimize(input).accepted);
    cv::Mat translated = cv::Mat::eye(4, 4, CV_32F);
    translated.at<float>(0, 3) = 2.0F;
    backend.reset(translated);
    input.timestampSec = 0.0;
    const auto result = backend.optimize(input);
    ASSERT_TRUE(result.accepted) << result.diagnostic;
    EXPECT_NEAR(result.bodyPose.at<float>(0, 3), 2.0F, 1e-4F);
    EXPECT_EQ(result.mapVersion, 1U);
}

TEST(GtsamBackendAdapter, LargeFrontEndDisagreementPreservesCommittedGraph)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput first;
    first.timestampSec = 0.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        first.stereo.push_back(observation);
    }
    ASSERT_TRUE(backend.optimize(first).incrementalCommitted);

    ORB_SLAM3::BackendFrameInput second = first;
    second.timestampSec = 0.1;
    second.frontEndCameraFromMap = cv::Mat::eye(4, 4, CV_32F);
    second.frontEndCameraFromMap.at<float>(0, 3) = -10.0F;
    for (int index = 0; index <= 10; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.01 * static_cast<double>(index);
        sample.acceleration = {0.0, 0.0, 9.81};
        sample.angularVelocity = {0.0, 0.0, 0.0};
        second.imu.push_back(sample);
    }
    second.hasImuCovariance = true;

    const auto accepted = backend.optimize(second);

    ASSERT_TRUE(accepted.accepted) << accepted.diagnostic;
    EXPECT_TRUE(accepted.incrementalCommitted);
    EXPECT_EQ(accepted.mapVersion, 2U);
    EXPECT_EQ(accepted.graphDiagnostics.totalPoseStates, 2U);
    EXPECT_NEAR(accepted.bodyPose.at<float>(0, 3), 0.0F, 1e-4F);
}

TEST(GtsamBackendAdapter, MissingImuIntervalPreservesCommittedGraph)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 0.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        input.stereo.push_back(observation);
    }
    ASSERT_TRUE(backend.optimize(input).incrementalCommitted);
    auto committedMap = validStereoMapSnapshot();
    committedMap.keyframes.front().timestampSec = 0.0;
    ASSERT_TRUE(backend.rebase(committedMap));

    input.timestampSec = 1.5;
    input.frontEndCameraFromMap = cv::Mat::eye(4, 4, CV_32F);
    input.frontEndCameraFromMap.at<float>(1, 3) = -1.0F;

    const auto recovered = backend.optimize(input);

    EXPECT_FALSE(recovered.incrementalCommitted);
    EXPECT_EQ(recovered.mapVersion, 1U);
    EXPECT_EQ(recovered.diagnostic.find("resynchronized"), std::string::npos);

    input.hasImuCovariance = true;
    for (int index = 0; index <= 15; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.1 * static_cast<double>(index);
        sample.acceleration = {0.0, 0.0, 9.81};
        input.imu.push_back(sample);
    }
    const auto retry = backend.optimize(input);
    ASSERT_TRUE(retry.incrementalCommitted) << retry.diagnostic;
    EXPECT_EQ(retry.mapVersion, 2U);
    EXPECT_EQ(retry.graphDiagnostics.totalPoseStates, 2U);
}

TEST(GtsamBackendAdapter, ReusesPersistentLandmarksAcrossCameraFrames)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 63.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput first;
    first.timestampSec = 0.0;
    ORB_SLAM3::BackendFrameInput second;
    second.timestampSec = 0.5;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = 100U + index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        first.stereo.push_back(observation);
        second.stereo.push_back(observation);
    }
    for (int index = 0; index <= 10; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.05 * static_cast<double>(index);
        sample.acceleration = {0.0, 0.0, 9.81};
        sample.angularVelocity = {0.0, 0.0, 0.0};
        second.imu.push_back(sample);
    }
    second.hasImuCovariance = true;
    second.imuAccelerometerCovariance = {1e-2, 0.0, 0.0,
                                         0.0, 1e-2, 0.0,
                                         0.0, 0.0, 1e-2};
    second.imuGyroscopeCovariance = {1e-4, 0.0, 0.0,
                                     0.0, 1e-4, 0.0,
                                     0.0, 0.0, 1e-4};
    second.imuAccelerometerBiasCovariance = {1e-8, 0.0, 0.0,
                                             0.0, 1e-8, 0.0,
                                             0.0, 0.0, 1e-8};
    second.imuGyroscopeBiasCovariance = {1e-10, 0.0, 0.0,
                                         0.0, 1e-10, 0.0,
                                         0.0, 0.0, 1e-10};
    const auto firstResult = backend.optimize(first);
    const auto secondResult = backend.optimize(second);
    ASSERT_TRUE(firstResult.accepted);
    EXPECT_TRUE(secondResult.accepted) << secondResult.diagnostic;
    EXPECT_TRUE(secondResult.incrementalCommitted);
    EXPECT_EQ(secondResult.mapVersion, 2U);
}

TEST(GtsamBackendAdapter, AppliesExplicitGravityAlignedInitialBodyPose)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
                      0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat initialMapFromBody = cv::Mat::eye(4, 4, CV_32F);
    initialMapFromBody.at<float>(1, 1) = 0.0F;
    initialMapFromBody.at<float>(1, 2) = 1.0F;
    initialMapFromBody.at<float>(2, 1) = -1.0F;
    initialMapFromBody.at<float>(2, 2) = 0.0F;
    ORB_SLAM3::GtsamBackendAdapter backend(
        camera, 63.0F, bodyFromCamera, initialMapFromBody);
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 0.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        input.stereo.push_back(observation);
    }

    const auto result = backend.optimize(input);

    ASSERT_TRUE(result.accepted);
    EXPECT_LT(cv::norm(result.bodyPose, initialMapFromBody, cv::NORM_INF),
              1e-5);
}

TEST(GtsamBackendAdapter, ComputesGravityInExistingAquaMapFrame)
{
    std::vector<ORB_SLAM3::ImuSample> imu;
    for (int index = 0; index <= 10; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.1 * static_cast<double>(index);
        sample.acceleration = {0.0, -9.81, 0.0};
        imu.push_back(sample);
    }

    cv::Mat mapFromImu = cv::Mat::eye(4, 4, CV_32F);
    mapFromImu.at<float>(1, 1) = 0.0F;
    mapFromImu.at<float>(1, 2) = 1.0F;
    mapFromImu.at<float>(2, 1) = -1.0F;
    mapFromImu.at<float>(2, 2) = 0.0F;

    const cv::Mat gravityMap =
        ORB_SLAM3::GtsamBackendAdapter::gravityMapFromImu(
            imu, mapFromImu);

    EXPECT_NEAR(gravityMap.at<float>(0), 0.0F, 1e-4F);
    EXPECT_NEAR(gravityMap.at<float>(1), 0.0F, 1e-4F);
    EXPECT_NEAR(gravityMap.at<float>(2), -9.81F, 1e-4F);
}

TEST(GtsamBackendAdapter, RefinesDoublePrecisionPoseWithoutMutatingIncrementalGraph)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 400.0F, 0.0F, 320.0F,
                      0.0F, 400.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 48.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 1.0;
    for (std::uint64_t index = 0; index < 6; ++index) {
        const double x = -0.5 + 0.2 * static_cast<double>(index % 3U);
        const double y = -0.2 + 0.4 * static_cast<double>(index / 3U);
        const double z = 3.0 + 0.1 * static_cast<double>(index % 2U);
        const double cameraX = x - 0.2;
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = 100 + index;
        observation.uLeft = static_cast<float>(400.0 * cameraX / z + 320.0);
        observation.uRight = static_cast<float>(
            observation.uLeft - 400.0 * 0.12 / z);
        observation.vLeft = static_cast<float>(400.0 * y / z + 240.0);
        observation.hasLandmark = true;
        observation.landmarkMap = {x, y, z};
        input.stereo.push_back(observation);
    }
    cv::Mat initialCameraFromMap = cv::Mat::eye(4, 4, CV_64F);
    initialCameraFromMap.at<double>(0, 3) = -0.5;

    // AQUA may expose one persistent MapPoint at more than one feature index.
    // The frame optimizer must deduplicate that key within its transaction.
    input.stereo.push_back(input.stereo.front());

    const auto refined = backend.refinePose(input, initialCameraFromMap);
    const auto firstCommitted = backend.optimize(input);

    ASSERT_TRUE(refined.accepted) << refined.diagnostic;
    EXPECT_FALSE(refined.incrementalCommitted);
    EXPECT_LT(std::abs(refined.bodyPose.at<float>(0, 3) - 0.2F), 0.3F);
    ASSERT_TRUE(firstCommitted.accepted) << firstCommitted.diagnostic;
    EXPECT_TRUE(firstCommitted.incrementalCommitted);
    EXPECT_EQ(firstCommitted.mapVersion, 1U);
}

TEST(GtsamBackendAdapter, RejectsInvalidRuntimePoseWithoutThrowing)
{
    cv::Mat camera = (cv::Mat_<float>(3, 3) << 400.0F, 0.0F, 320.0F,
                      0.0F, 400.0F, 240.0F, 0.0F, 0.0F, 1.0F);
    cv::Mat bodyFromCamera = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::GtsamBackendAdapter backend(camera, 48.0F, bodyFromCamera);
    ORB_SLAM3::BackendFrameInput input;

    ORB_SLAM3::BackendFrameResult result;
    EXPECT_NO_THROW(result = backend.refinePose(input, cv::Mat()));
    EXPECT_FALSE(result.accepted);
    EXPECT_NE(result.diagnostic.find("invalid front-end pose"),
              std::string::npos);
}

TEST(GtsamMapOptimizationAdapter, ReturnsVersionedFiniteGlobalResult)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    snapshot.version = 11;
    snapshot.calibration = {400.0, 400.0, 320.0, 240.0, 0.12};
    ORB_SLAM3::BackendKeyframeState first;
    first.id = 42;
    first.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                  0.0F, 1.0F, 0.0F, 0.0F,
                  0.0F, 0.0F, 1.0F, 0.0F,
                  0.0F, 0.0F, 0.0F, 1.0F};
    ORB_SLAM3::BackendKeyframeState second = first;
    second.id = 43;
    second.pose[3] = 0.5F;
    snapshot.keyframes = {first, second};
    for (std::uint64_t index = 0; index < 6; ++index) {
        const double x = -0.5 + 0.2 * static_cast<double>(index % 3U);
        const double y = -0.2 + 0.4 * static_cast<double>(index / 3U);
        const double z = 3.0 + 0.1 * static_cast<double>(index % 2U);
        ORB_SLAM3::BackendLandmarkState landmark;
        landmark.id = 100 + index;
        landmark.position = {static_cast<float>(x), static_cast<float>(y),
                             static_cast<float>(z)};
        snapshot.landmarks.push_back(landmark);
        for (const auto keyframeAndX :
             {std::pair<std::uint64_t, double>{42, 0.0}, {43, 0.2}}) {
            const double cameraX = x - keyframeAndX.second;
            const double uLeft = 400.0 * cameraX / z + 320.0;
            ORB_SLAM3::BackendStereoObservation observation;
            observation.keyframeId = keyframeAndX.first;
            observation.landmarkId = landmark.id;
            observation.uLeft = uLeft;
            observation.uRight = uLeft - 400.0 * 0.12 / z;
            observation.v = 400.0 * y / z + 240.0;
            observation.sigmaPx = 0.5;
            observation.octave = 0;
            observation.octaveVariancePx2 = 0.25;
            observation.quality = 1.0;
            observation.stereoSkewPx = 0.0;
            snapshot.observations.push_back(observation);
        }
    }
    const ORB_SLAM3::GtsamMapOptimizationAdapter optimizer;
    const auto result = optimizer.optimizeGlobal(snapshot);
    const auto localResult = optimizer.optimizeLocal(snapshot);
    EXPECT_TRUE(result.accepted) << result.diagnostic;
    EXPECT_EQ(result.sourceVersion, 11U);
    ASSERT_EQ(result.keyframes.size(), 2U);
    ASSERT_EQ(result.landmarks.size(), 6U);
    for (const auto& landmark : result.landmarks) {
        EXPECT_TRUE(std::isfinite(landmark.position[0]));
        EXPECT_TRUE(std::isfinite(landmark.position[1]));
        EXPECT_TRUE(std::isfinite(landmark.position[2]));
    }
    EXPECT_LT(std::abs(result.keyframes[1].pose[3] - 0.2F),
              std::abs(second.pose[3] - 0.2F));
    ASSERT_TRUE(localResult.accepted) << localResult.diagnostic;
    ASSERT_EQ(localResult.keyframes.size(), 2U);
    EXPECT_LT(std::abs(localResult.keyframes[1].pose[3] - 0.2F),
              std::abs(second.pose[3] - 0.2F));
}

TEST(GtsamMapOptimizationAdapter, VerifiesLoopCandidateWithFixedScaleSe3)
{
    const cv::Mat rotation = (cv::Mat_<float>(3, 3) <<
        1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, -1.0F,
        0.0F, 1.0F, 0.0F);
    const cv::Vec3f translation(1.0F, -0.2F, 0.4F);
    ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    std::vector<ORB_SLAM3::BackendLoopPointCorrespondence> points;
    const std::array<std::array<double, 3>, 4> candidates = {{
        {{0.0, 0.0, 2.0}}, {{0.2, -0.1, 2.3}},
        {{-0.4, 0.3, 2.7}}, {{0.1, 0.5, 3.1}}}};
    for (const auto& candidate : candidates) {
        std::array<double, 3> current{};
        const cv::Mat value = rotation *
            (cv::Mat_<float>(3, 1) << static_cast<float>(candidate[0]),
             static_cast<float>(candidate[1]), static_cast<float>(candidate[2]));
        for (int axis = 0; axis < 3; ++axis)
            current[static_cast<std::size_t>(axis)] =
                value.at<float>(axis) + translation[axis];
        points.push_back({current, candidate});
    }
    const auto result = adapter.verifyLoopCandidateSE3(points, 1e-4, 4);
    ASSERT_TRUE(result.accepted);
    EXPECT_EQ(result.inliers, 4U);
    EXPECT_NEAR(result.currentFromCandidate[3], 1.0F, 1e-4F);
    EXPECT_NEAR(result.currentFromCandidate[7], -0.2F, 1e-4F);
    EXPECT_NEAR(result.currentFromCandidate[11], 0.4F, 1e-4F);
}

TEST(GtsamMapOptimizationAdapter, LocalBaUsesEightMostRecentKeyframes)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    snapshot.version = 17;
    snapshot.mapId = 3;
    snapshot.topologySignature = 91;
    snapshot.calibration = {400.0, 400.0, 320.0, 240.0, 0.12};
    for (int index = 11; index >= 0; --index) {
        ORB_SLAM3::BackendKeyframeState keyframe;
        keyframe.id = static_cast<std::uint64_t>(100 + index);
        keyframe.timestampSec = 10.0 + static_cast<double>(index);
        keyframe.pose = {1.0F, 0.0F, 0.0F, 0.05F * index + 0.02F,
                         0.0F, 1.0F, 0.0F, 0.0F,
                         0.0F, 0.0F, 1.0F, 0.0F,
                         0.0F, 0.0F, 0.0F, 1.0F};
        snapshot.keyframes.push_back(keyframe);
    }
    for (std::uint64_t landmarkIndex = 0; landmarkIndex < 6; ++landmarkIndex) {
        const double x = -0.5 + 0.2 * static_cast<double>(landmarkIndex % 3U);
        const double y = -0.2 + 0.4 * static_cast<double>(landmarkIndex / 3U);
        const double z = 4.0 + 0.1 * static_cast<double>(landmarkIndex % 2U);
        ORB_SLAM3::BackendLandmarkState landmark;
        landmark.id = 200 + landmarkIndex;
        landmark.position = {static_cast<float>(x), static_cast<float>(y),
                             static_cast<float>(z)};
        snapshot.landmarks.push_back(landmark);
        for (int keyframeIndex = 0; keyframeIndex < 12; ++keyframeIndex) {
            const double cameraX = 0.05 * static_cast<double>(keyframeIndex);
            const double uLeft = 400.0 * (x - cameraX) / z + 320.0;
            ORB_SLAM3::BackendStereoObservation observation;
            observation.keyframeId =
                static_cast<std::uint64_t>(100 + keyframeIndex);
            observation.landmarkId = landmark.id;
            observation.uLeft = uLeft;
            observation.uRight = uLeft - 400.0 * 0.12 / z;
            observation.v = 400.0 * y / z + 240.0;
            observation.sigmaPx = 0.5;
            observation.octave = 0;
            observation.octaveVariancePx2 = 0.25;
            observation.quality = 1.0;
            observation.stereoSkewPx = 0.0;
            snapshot.observations.push_back(observation);
        }
    }

    const ORB_SLAM3::GtsamMapOptimizationAdapter optimizer;
    const auto globalResult = optimizer.optimizeGlobal(snapshot);
    const auto localResult = optimizer.optimizeLocal(snapshot);

    ASSERT_TRUE(globalResult.accepted) << globalResult.diagnostic;
    EXPECT_EQ(globalResult.keyframes.size(), 12U);
    ASSERT_TRUE(localResult.accepted) << localResult.diagnostic;
    ASSERT_EQ(localResult.keyframes.size(), 8U);
    EXPECT_EQ(localResult.sourceVersion, snapshot.version);
    EXPECT_EQ(localResult.sourceMapId, snapshot.mapId);
    EXPECT_EQ(localResult.sourceTopologySignature,
              snapshot.topologySignature);
    for (std::size_t index = 0; index < localResult.keyframes.size(); ++index)
        EXPECT_EQ(localResult.keyframes[index].id,
                  static_cast<std::uint64_t>(104 + index));
}

TEST(GtsamMapOptimizationAdapter,
     SelectsMostRecentLocalKeyframesAndCovisibleFixedAnchor)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    for (const auto keyframeAndTimestamp :
         {std::pair<std::uint64_t, double>{31U, 31.0},
          {29U, 29.0}, {30U, 30.0}}) {
        ORB_SLAM3::BackendKeyframeState keyframe;
        keyframe.id = keyframeAndTimestamp.first;
        keyframe.timestampSec = keyframeAndTimestamp.second;
        snapshot.keyframes.push_back(keyframe);
    }
    for (const std::uint64_t landmarkId : {500U, 501U, 502U}) {
        ORB_SLAM3::BackendLandmarkState landmark;
        landmark.id = landmarkId;
        snapshot.landmarks.push_back(landmark);
    }
    for (const auto keyframeAndLandmark :
         {std::pair<std::uint64_t, std::uint64_t>{29U, 500U},
          {29U, 501U}, {30U, 500U}, {30U, 502U},
          {31U, 501U}, {31U, 502U}}) {
        ORB_SLAM3::BackendStereoObservation observation;
        observation.keyframeId = keyframeAndLandmark.first;
        observation.landmarkId = keyframeAndLandmark.second;
        snapshot.observations.push_back(observation);
    }

    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    const auto selection = adapter.selectLocalGraph(snapshot, 2U);

    ASSERT_TRUE(selection.accepted);
    EXPECT_EQ(selection.localKeyframeIds,
              (std::vector<std::uint64_t>{30U, 31U}));
    EXPECT_EQ(selection.fixedKeyframeIds,
              (std::vector<std::uint64_t>{29U}));
}

TEST(GtsamMapOptimizationAdapter, BoundsCovisibleFixedAnchorsToLocalWindowSize)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    for (std::uint64_t id = 1U; id <= 6U; ++id) {
        ORB_SLAM3::BackendKeyframeState keyframe;
        keyframe.id = id;
        keyframe.timestampSec = static_cast<double>(id);
        snapshot.keyframes.push_back(keyframe);
    }
    ORB_SLAM3::BackendLandmarkState landmark;
    landmark.id = 500U;
    snapshot.landmarks.push_back(landmark);
    for (std::uint64_t keyframeId = 1U; keyframeId <= 6U; ++keyframeId) {
        ORB_SLAM3::BackendStereoObservation observation;
        observation.keyframeId = keyframeId;
        observation.landmarkId = landmark.id;
        snapshot.observations.push_back(observation);
    }

    const auto selection =
        ORB_SLAM3::GtsamMapOptimizationAdapter().selectLocalGraph(snapshot, 2U);

    ASSERT_TRUE(selection.accepted);
    EXPECT_EQ(selection.localKeyframeIds,
              (std::vector<std::uint64_t>{5U, 6U}));
    EXPECT_EQ(selection.fixedKeyframeIds,
              (std::vector<std::uint64_t>{4U, 3U}));
}

TEST(GtsamMapOptimizationAdapter, RejectsTopologyOnlyLocalGraphSelection)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    for (const auto keyframeAndTimestamp :
         {std::pair<std::uint64_t, double>{29U, 29.0},
          {30U, 30.0}, {31U, 31.0}}) {
        ORB_SLAM3::BackendKeyframeState keyframe;
        keyframe.id = keyframeAndTimestamp.first;
        keyframe.timestampSec = keyframeAndTimestamp.second;
        snapshot.keyframes.push_back(keyframe);
    }
    ORB_SLAM3::BackendGraphEdge edge;
    edge.fromKeyframeId = 29U;
    edge.toKeyframeId = 31U;
    snapshot.graphEdges.push_back(edge);

    const ORB_SLAM3::GtsamMapOptimizationAdapter adapter;
    const auto selection = adapter.selectLocalGraph(snapshot, 2U);

    EXPECT_FALSE(selection.accepted);
    EXPECT_TRUE(selection.localKeyframeIds.empty());
    EXPECT_TRUE(selection.fixedKeyframeIds.empty());
    EXPECT_FALSE(adapter.optimizeLocal(snapshot).accepted);
}

TEST(GtsamMapOptimizationAdapter, RejectsMalformedEssentialGraphWithoutThrowing)
{
    ORB_SLAM3::BackendMapSnapshot snapshot;
    ORB_SLAM3::BackendKeyframeState first;
    first.id = 1U;
    first.pose = {1.0F, 0.0F, 0.0F, 0.0F,
                  0.0F, 1.0F, 0.0F, 0.0F,
                  0.0F, 0.0F, 1.0F, 0.0F,
                  0.0F, 0.0F, 0.0F, 1.0F};
    ORB_SLAM3::BackendKeyframeState second = first;
    second.id = 2U;
    snapshot.keyframes = {first, second};
    ORB_SLAM3::BackendGraphEdge edge;
    edge.fromKeyframeId = first.id;
    edge.toKeyframeId = second.id;
    edge.sigma = 0.1;
    edge.relativePose = first.pose;
    edge.relativePose[3] = std::numeric_limits<float>::quiet_NaN();
    snapshot.graphEdges.push_back(edge);

    ORB_SLAM3::BackendMapResult result;
    EXPECT_NO_THROW(result =
        ORB_SLAM3::GtsamMapOptimizationAdapter().optimizeEssentialGraph(snapshot));
    EXPECT_FALSE(result.accepted);
}

}  // namespace
