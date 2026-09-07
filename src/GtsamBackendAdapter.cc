#include "GtsamBackendAdapter.h"

#include "ImuTypes.h"

#include <uw_dynamic_backend/incremental_backend.h>
#include <uw_dynamic_backend/frame_optimizer.h>
#include <uw_dynamic_backend/uncertainty_config.h>
#include <uw_dynamic_backend/stereo_inertial_initializer.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Geometry>

#include <cmath>
#include <limits>
#include <mutex>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace ORB_SLAM3
{
namespace
{
gtsam::Pose3 toPose3(const cv::Mat& matrix)
{
    const bool supportedType =
        matrix.channels() == 1 &&
        (matrix.depth() == CV_32F || matrix.depth() == CV_64F);
    const bool finite = supportedType && cv::checkRange(matrix);
    if (matrix.rows != 4 || matrix.cols != 4 || !finite) {
        std::ostringstream message;
        message << "pose must be a finite single-channel 4x4 float matrix"
                << " (rows=" << matrix.rows
                << " cols=" << matrix.cols
                << " type=" << matrix.type()
                << " finite=" << (finite ? 1 : 0) << ")";
        throw std::invalid_argument(message.str());
    }
    gtsam::Matrix4 value;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            value(row, col) = matrix.depth() == CV_32F
                ? static_cast<double>(matrix.at<float>(row, col))
                : matrix.at<double>(row, col);
    return gtsam::Pose3(value);
}

gtsam::Matrix3 toMatrix3(const std::array<double, 9>& values)
{
    gtsam::Matrix3 matrix;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            matrix(row, col) = values[static_cast<std::size_t>(row * 3 + col)];
    return matrix;
}

std::array<double, 9> fromMatrix3(const gtsam::Matrix3& matrix)
{
    std::array<double, 9> values{};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            values[static_cast<std::size_t>(row * 3 + col)] = matrix(row, col);
    return values;
}

void copyImuCovariance(
    const uw_slam::dynamic_backend::ImuCovarianceDecision& input,
    IncrementalImuCovarianceDiagnostics* output)
{
    output->accelerometer = fromMatrix3(input.accelerometer);
    output->gyroscope = fromMatrix3(input.gyroscope);
    output->accelerometerBias = fromMatrix3(input.accelerometerBias);
    output->gyroscopeBias = fromMatrix3(input.gyroscopeBias);
}

gtsam::Vector3 toVector3(const std::array<double, 3>& values)
{
    return (gtsam::Vector3() << values[0], values[1], values[2]).finished();
}

gtsam::Vector3 toVector3(const std::array<float, 3>& values)
{
    return (gtsam::Vector3() << static_cast<double>(values[0]),
            static_cast<double>(values[1]), static_cast<double>(values[2]))
        .finished();
}

gtsam::Vector3 toVector3(const cv::Mat& values)
{
    if (values.rows != 3 || values.cols != 1 || values.type() != CV_32F ||
        !cv::checkRange(values))
        throw std::invalid_argument("gravityMap must be finite 3x1 CV_32F");
    return (gtsam::Vector3() << values.at<float>(0), values.at<float>(1),
            values.at<float>(2)).finished();
}

cv::Mat fromPose3(const gtsam::Pose3& pose)
{
    const auto matrix = pose.matrix();
    cv::Mat result(4, 4, CV_32F);
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            result.at<float>(row, col) = static_cast<float>(matrix(row, col));
    return result;
}

BackendMapResult mapResultFromGraph(
    const BackendMapSnapshot& snapshot,
    const uw_slam::dynamic_backend::IncrementalSnapshot& optimized,
    const std::unordered_map<std::uint64_t, std::uint64_t>& landmarkKeys,
    const gtsam::Pose3& bodyFromCamera,
    const std::unordered_set<std::uint64_t>& replacedIds = {})
{
    BackendMapResult result;
    result.sourceVersion = snapshot.version;
    result.sourceMapId = snapshot.mapId;
    result.sourceTopologySignature = snapshot.topologySignature;
    std::unordered_set<std::uint64_t> keyframeIds;
    std::unordered_set<double> keyframeTimes;
    for (const auto& source : snapshot.keyframes) {
        if (!keyframeIds.insert(source.id).second ||
            !std::isfinite(source.timestampSec) ||
            !keyframeTimes.insert(source.timestampSec).second)
            return {};
        const auto timestamp = std::lower_bound(optimized.timestamps.begin(),
            optimized.timestamps.end(), source.timestampSec);
        if (timestamp == optimized.timestamps.end() || *timestamp != source.timestampSec)
            continue;
        const std::size_t index = static_cast<std::size_t>(
            timestamp - optimized.timestamps.begin());
        BackendKeyframeState state = source;
        const auto pose = optimized.poses[index].compose(bodyFromCamera).matrix();
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                state.pose[static_cast<std::size_t>(4 * row + col)] =
                    static_cast<float>(pose(row, col));
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            state.velocity[axis] = static_cast<float>(optimized.velocities[index](axis));
            state.accelerometerBias[axis] = static_cast<float>(
                optimized.biases[index].accelerometer()(axis));
            state.gyroscopeBias[axis] = static_cast<float>(
                optimized.biases[index].gyroscope()(axis));
        }
        result.keyframes.push_back(state);
    }
    if (result.keyframes.empty())
        return {};
    std::sort(result.keyframes.begin(), result.keyframes.end(),
        [](const auto& left, const auto& right) {
            return left.timestampSec < right.timestampSec;
        });
    std::unordered_set<std::uint64_t> landmarkIds;
    for (const auto& source : snapshot.landmarks) {
        if (!landmarkIds.insert(source.id).second)
            return {};
        if (replacedIds.count(source.id) != 0U)
            continue;
        const auto mapped = landmarkKeys.find(source.id);
        if (mapped == landmarkKeys.end())
            continue;
        const gtsam::Key key = gtsam::Symbol('l', mapped->second);
        if (!optimized.values.exists(key))
            continue;
        const auto point = optimized.values.at<gtsam::Point3>(key);
        BackendLandmarkState landmark = source;
        for (std::size_t axis = 0; axis < 3U; ++axis)
            landmark.position[axis] = static_cast<float>(point(axis));
        result.landmarks.push_back(landmark);
    }
    result.accepted = true;
    return result;
}

uw_slam::dynamic_backend::DvlTrackMode toBackendTrackMode(DvlTrackMode mode)
{
    switch (mode) {
    case DvlTrackMode::BottomTrack:
        return uw_slam::dynamic_backend::DvlTrackMode::kBottomTrack;
    case DvlTrackMode::WaterTrack:
        return uw_slam::dynamic_backend::DvlTrackMode::kWaterTrack;
    case DvlTrackMode::Unknown:
        return uw_slam::dynamic_backend::DvlTrackMode::kUnknown;
    }
    return uw_slam::dynamic_backend::DvlTrackMode::kUnknown;
}
}  // namespace

struct GtsamBackendAdapter::Implementation
{
    std::mutex mutex;
    std::shared_ptr<gtsam::Cal3_S2Stereo> calibration;
    uw_slam::dynamic_backend::IncrementalBackendConfig backendConfig;
    gtsam::Pose3 bodyFromCamera;
    std::shared_ptr<uw_slam::dynamic_backend::IncrementalBackend> backend;
    std::unique_ptr<uw_slam::dynamic_backend::FrameOptimizer> frameOptimizer;
    double lastAcceptedTimestamp = -std::numeric_limits<double>::infinity();
    BackendFrameResult lastAcceptedResult;
    std::uint64_t lastPublishedVersion = 0U;
    // AQUA mnId values are external identities. Allocate compact, process-
    // local GTSAM landmark keys so resets/relocalization cannot collide with
    // stale IDs or the reserved transient namespace.
    std::unordered_map<std::uint64_t, std::uint64_t> landmarkKeys;
    std::uint64_t nextLandmarkKey = 1U;
};

struct GtsamPreparedMapOptimization::Implementation
{
    std::shared_ptr<uw_slam::dynamic_backend::IncrementalBackend> sourceBackend;
    std::shared_ptr<uw_slam::dynamic_backend::PreparedIncrementalOptimization> graph;
    BackendMapResult result;
    cv::Mat latestBodyPose;
    std::unordered_map<std::uint64_t, std::uint64_t> landmarkKeys;
};

GtsamPreparedMapOptimization::GtsamPreparedMapOptimization()
    : implementation_(std::make_unique<Implementation>())
{
}

GtsamPreparedMapOptimization::~GtsamPreparedMapOptimization() = default;

const BackendMapResult& GtsamPreparedMapOptimization::result() const
{
    return implementation_->result;
}

std::shared_ptr<GtsamPreparedMapOptimization>
GtsamBackendAdapter::prepareMapOptimization(
    const BackendMapSnapshot& snapshot,
    const std::vector<BackendLandmarkReplacement>& replacements) const
{
    auto candidate = std::shared_ptr<GtsamPreparedMapOptimization>(
        new GtsamPreparedMapOptimization());
    uw_slam::dynamic_backend::IncrementalSnapshot graph;
    std::unordered_map<std::uint64_t, std::uint64_t> landmarkKeys;
    gtsam::Pose3 bodyFromCamera;
    {
        std::lock_guard<std::mutex> lock(implementation_->mutex);
        candidate->implementation_->sourceBackend = implementation_->backend;
        graph = implementation_->backend->snapshot();
        landmarkKeys = implementation_->landmarkKeys;
        bodyFromCamera = implementation_->bodyFromCamera;
    }
    if (graph.timestamps.empty() ||
        std::none_of(snapshot.keyframes.begin(), snapshot.keyframes.end(),
            [&graph](const BackendKeyframeState& keyframe) {
                return keyframe.timestampSec == graph.timestamps.back();
            }))
        return {};
    std::unordered_set<std::uint64_t> replacedIds;
    std::unordered_set<std::uint64_t> retainedIds;
    std::unordered_set<std::uint64_t> liveLandmarkIds;
    for (const auto& landmark : snapshot.landmarks)
        liveLandmarkIds.insert(landmark.id);
    std::map<gtsam::Key, gtsam::Key> graphAliases;
    for (const auto& replacement : replacements) {
        const auto source = landmarkKeys.find(replacement.replacedId);
        const auto target = landmarkKeys.find(replacement.retainedId);
        if (replacement.replacedId == replacement.retainedId ||
            !replacedIds.insert(replacement.replacedId).second ||
            !retainedIds.insert(replacement.retainedId).second ||
            liveLandmarkIds.count(replacement.replacedId) == 0U ||
            liveLandmarkIds.count(replacement.retainedId) == 0U ||
            source == landmarkKeys.end() || target == landmarkKeys.end())
            return {};
        if (source->second != target->second)
            graphAliases.emplace(gtsam::Symbol('l', source->second),
                                 gtsam::Symbol('l', target->second));
    }
    for (const auto id : replacedIds)
        if (retainedIds.count(id) != 0U)
            return {};
    candidate->implementation_->graph =
        uw_slam::dynamic_backend::IncrementalBackend::prepareOptimization(graph, graphAliases);
    if (!candidate->implementation_->graph)
        return {};
    const auto& optimized = candidate->implementation_->graph->snapshot();
    auto& result = candidate->implementation_->result;
    for (auto& mapping : landmarkKeys) {
        const auto alias = graphAliases.find(gtsam::Symbol('l', mapping.second));
        if (alias != graphAliases.end())
            mapping.second = gtsam::Symbol(alias->second).index();
    }
    candidate->implementation_->landmarkKeys = landmarkKeys;
    result = mapResultFromGraph(snapshot, optimized, landmarkKeys,
                                bodyFromCamera, replacedIds);
    if (!result.accepted)
        return {};
    result.landmarkReplacements = replacements;
    candidate->implementation_->latestBodyPose = fromPose3(optimized.poses.back());
    result.accepted = true;
    return candidate;
}

bool GtsamBackendAdapter::commitMapOptimization(
    GtsamPreparedMapOptimization& candidate,
    const std::function<bool(const BackendMapResult&)>& commitMap)
{
    std::lock_guard<std::mutex> lock(implementation_->mutex);
    auto& prepared = *candidate.implementation_;
    if (!commitMap || !prepared.result.accepted || !prepared.graph ||
        prepared.sourceBackend != implementation_->backend ||
        !implementation_->backend->canCommitOptimization(*prepared.graph))
        return false;
    if (!commitMap(prepared.result))
        return false;
    implementation_->backend->commitOptimization(*prepared.graph);
    implementation_->landmarkKeys.swap(prepared.landmarkKeys);
    std::swap(implementation_->lastAcceptedResult.bodyPose, prepared.latestBodyPose);
    prepared.result.accepted = false;
    return true;
}

GtsamOptionalSensorConfig GtsamOptionalSensorConfig::fromTrackingSensor(
    bool dvlStereoEnabled, const IMU::Calib* imuCalibration,
    DvlTrackMode dvlTrackMode,
    const std::string& uncertaintyProfilePath)
{
    GtsamOptionalSensorConfig config;
    if (!dvlStereoEnabled)
        return config;
    if (!imuCalibration)
        throw std::invalid_argument("DVL mode requires IMU calibration");
    if (dvlTrackMode == DvlTrackMode::Unknown)
        throw std::invalid_argument("DVL track mode registration is missing");
    if (uncertaintyProfilePath.empty())
        throw std::invalid_argument("DVL uncertainty profile is missing");
    config.dvlEnabled = true;
    config.bodyFromDvl = imuCalibration->mT_imu_dvl.clone();
    config.dvlTrackMode = dvlTrackMode;
    config.uncertaintyProfilePath = uncertaintyProfilePath;
    return config;
}

GtsamBackendAdapter::GtsamBackendAdapter(const cv::Mat& cameraMatrix,
                                         float baselineFx,
                                         const cv::Mat& bodyFromCamera,
                                         const cv::Mat& initialMapFromBody,
                                         const cv::Mat& gravityMap,
                                         const GtsamOptionalSensorConfig& optionalSensors,
                                         const GtsamInitialNavigationState& initialNavigation)
    : implementation_(std::make_unique<Implementation>())
{
    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3 ||
        cameraMatrix.type() != CV_32F || !cv::checkRange(cameraMatrix) ||
        !(std::isfinite(baselineFx) && baselineFx > 0.0F))
        throw std::invalid_argument("invalid stereo calibration");
    implementation_->calibration = std::make_shared<gtsam::Cal3_S2Stereo>(
        cameraMatrix.at<float>(0, 0), cameraMatrix.at<float>(1, 1), 0.0,
        cameraMatrix.at<float>(0, 2), cameraMatrix.at<float>(1, 2),
        baselineFx / cameraMatrix.at<float>(0, 0));
    uw_slam::dynamic_backend::IncrementalBackendConfig config;
    config.initialVelocityMap = toVector3(initialNavigation.velocityMap);
    config.initialBias = gtsam::imuBias::ConstantBias(
        toVector3(initialNavigation.accelerometerBias),
        toVector3(initialNavigation.gyroscopeBias));
    if (!gravityMap.empty())
        config.gravityMap = toVector3(gravityMap);
    if (optionalSensors.dvlEnabled) {
        if (optionalSensors.dvlTrackMode == DvlTrackMode::Unknown)
            throw std::invalid_argument("DVL track mode registration is missing");
        if (optionalSensors.uncertaintyProfilePath.empty())
            throw std::invalid_argument("DVL uncertainty profile is missing");
        config.dvlEnabled = true;
        config.uncertaintyConfig =
            uw_slam::dynamic_backend::loadUncertaintyConfig(
                optionalSensors.uncertaintyProfilePath);
        config.dvlRegistration =
            uw_slam::dynamic_backend::DvlFactorRegistration{
                toPose3(optionalSensors.bodyFromDvl),
                toBackendTrackMode(optionalSensors.dvlTrackMode)};
    }
    implementation_->bodyFromCamera = toPose3(bodyFromCamera);
    implementation_->backendConfig = config;
    const gtsam::Pose3 initialPose = initialMapFromBody.empty()
        ? gtsam::Pose3()
        : toPose3(initialMapFromBody);
    implementation_->backend =
        std::make_unique<uw_slam::dynamic_backend::IncrementalBackend>(
            config, implementation_->calibration,
            implementation_->bodyFromCamera, initialPose);
    // Startup navigation belongs to this initial timestamp only. Later graph
    // reset/recovery boundaries retain their explicit legacy zero seed.
    implementation_->backendConfig.initialVelocityMap.setZero();
    implementation_->backendConfig.initialBias = gtsam::imuBias::ConstantBias();
    uw_slam::dynamic_backend::FrameOptimizerConfig frameConfig;
    implementation_->frameOptimizer =
        std::make_unique<uw_slam::dynamic_backend::FrameOptimizer>(
            frameConfig, implementation_->calibration);
}

GtsamBackendAdapter::~GtsamBackendAdapter() = default;
GtsamBackendAdapter::GtsamBackendAdapter(GtsamBackendAdapter&&) noexcept = default;
GtsamBackendAdapter& GtsamBackendAdapter::operator=(GtsamBackendAdapter&&) noexcept = default;

GtsamInitializationResult GtsamBackendAdapter::prepareInitialization(
    const BackendMapSnapshot& snapshot,
    const BackendFrameInput& sensorInput,
    const cv::Mat& bodyFromCamera,
    const GtsamOptionalSensorConfig& optionalSensors)
{
    GtsamInitializationResult result;
    try {
        if (!sensorInput.hasImuCovariance)
            throw std::invalid_argument("initializer requires registered IMU covariance");
        const auto& calibration = snapshot.calibration;
        if (!(std::isfinite(calibration.fx) && calibration.fx > 0.0 &&
              std::isfinite(calibration.fy) && calibration.fy > 0.0 &&
              std::isfinite(calibration.cx) && std::isfinite(calibration.cy) &&
              std::isfinite(calibration.baseline) && calibration.baseline > 0.0))
            throw std::invalid_argument("initializer stereo calibration is invalid");
        cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) <<
            calibration.fx, 0.0, calibration.cx,
            0.0, calibration.fy, calibration.cy, 0.0, 0.0, 1.0);
        auto adapter = std::make_shared<GtsamBackendAdapter>(cameraMatrix,
            static_cast<float>(calibration.fx * calibration.baseline),
            bodyFromCamera, cv::Mat(), cv::Mat(), optionalSensors);
        auto& implementation = *adapter->implementation_;
        uw_slam::dynamic_backend::InitializerConfig config;
        config.accelerometerCovariance = toMatrix3(sensorInput.imuAccelerometerCovariance);
        config.gyroscopeCovariance = toMatrix3(sensorInput.imuGyroscopeCovariance);
        config.accelerometerBiasCovariance = toMatrix3(sensorInput.imuAccelerometerBiasCovariance);
        config.gyroscopeBiasCovariance = toMatrix3(sensorInput.imuGyroscopeBiasCovariance);
        // Existing preintegration numerical error model, not sensor calibration.
        config.integrationCovariance = gtsam::Matrix3::Identity() * 1e-8;
        uw_slam::dynamic_backend::InitializerInput input;
        input.calibration = implementation.calibration;
        input.bodyFromCamera = implementation.bodyFromCamera;
        std::vector<BackendKeyframeState> keyframes = snapshot.keyframes;
        for (const auto& keyframe : keyframes)
            if (!std::isfinite(keyframe.timestampSec))
                throw std::invalid_argument("initializer keyframe identity is invalid");
        std::sort(keyframes.begin(), keyframes.end(),
            [](const auto& left, const auto& right) {
                return left.timestampSec < right.timestampSec;
            });
        std::unordered_map<std::uint64_t, std::size_t> frameIndices;
        for (const auto& keyframe : keyframes) {
            if (!frameIndices.emplace(keyframe.id, input.frames.size()).second ||
                !std::isfinite(keyframe.timestampSec))
                throw std::invalid_argument("initializer keyframe identity is invalid");
            cv::Mat mapFromCamera(4, 4, CV_32F);
            std::copy(keyframe.pose.begin(), keyframe.pose.end(), mapFromCamera.ptr<float>());
            uw_slam::dynamic_backend::InitializerFrame frame;
            frame.keyframeId = keyframe.id;
            frame.timestampSec = keyframe.timestampSec;
            frame.mapFromBodySeed = toPose3(mapFromCamera).compose(input.bodyFromCamera.inverse());
            frame.stereoCovariance = gtsam::Matrix3::Identity();
            input.frames.push_back(std::move(frame));
        }
        std::unordered_map<std::uint64_t, gtsam::Point3> landmarks;
        for (const auto& landmark : snapshot.landmarks)
            if (!landmarks.emplace(landmark.id, toVector3(landmark.position)).second)
                throw std::invalid_argument("initializer landmark identity is duplicated");
        for (const auto& observation : snapshot.observations) {
            const auto frame = frameIndices.find(observation.keyframeId);
            const auto landmark = landmarks.find(observation.landmarkId);
            if (frame == frameIndices.end() || landmark == landmarks.end() ||
                !std::isfinite(observation.octaveVariancePx2) ||
                observation.octaveVariancePx2 <= 0.0)
                throw std::invalid_argument("initializer observation registration is invalid");
            uw_slam::dynamic_backend::StereoTrackObservation stereo;
            stereo.trackId = observation.landmarkId;
            stereo.pixel = gtsam::StereoPoint2(observation.uLeft, observation.uRight, observation.v);
            stereo.pyramidSigmaPx = std::sqrt(observation.octaveVariancePx2);
            stereo.landmarkMap = landmark->second;
            stereo.hasLandmark = true;
            input.frames[frame->second].observations.push_back(stereo);
        }
        for (const auto& sample : sensorInput.imu)
            input.imu.push_back({0, sample.timestampSec,
                toVector3(sample.acceleration), toVector3(sample.angularVelocity)});
        const auto initialized =
            uw_slam::dynamic_backend::StereoInertialInitializer(config).update(input);
        result.diagnostic = initialized.state.diagnostic;
        if (!initialized.graph ||
            initialized.state.lifecycle != uw_slam::dynamic_backend::Lifecycle::kRunning)
            return result;
        if (!implementation.backend->adoptInitializedGraph(*initialized.graph))
            throw std::runtime_error("joint initializer graph adoption failed");
        for (const auto& mapping : initialized.graph->landmarkKeys) {
            const auto index = gtsam::Symbol(mapping.second).index();
            implementation.landmarkKeys.emplace(mapping.first, index);
            implementation.nextLandmarkKey = std::max(implementation.nextLandmarkKey, index + 1U);
        }
        const auto adopted = implementation.backend->snapshot();
        result.mapResult = mapResultFromGraph(snapshot, adopted,
            implementation.landmarkKeys, implementation.bodyFromCamera);
        if (!result.mapResult.accepted || result.mapResult.keyframes.size() != keyframes.size())
            throw std::runtime_error("joint initializer map result is incomplete");
        implementation.lastAcceptedTimestamp = adopted.timestamps.back();
        implementation.lastPublishedVersion = adopted.version;
        implementation.lastAcceptedResult.accepted = true;
        implementation.lastAcceptedResult.incrementalCommitted = true;
        implementation.lastAcceptedResult.mapVersion = adopted.version;
        implementation.lastAcceptedResult.bodyPose = fromPose3(adopted.poses.back());
        implementation.lastAcceptedResult.diagnostic = initialized.state.diagnostic;
        const gtsam::Vector3 gravity = adopted.gravityMagnitudeMps2 *
            adopted.values.at<gtsam::Unit3>(*adopted.gravityKey).unitVector();
        for (std::size_t axis = 0; axis < 3U; ++axis)
            result.gravityMap[axis] = gravity(axis);
        result.backend = std::move(adapter);
    } catch (const std::exception& exception) {
        result.backend.reset();
        result.mapResult = BackendMapResult();
        result.diagnostic = exception.what();
    }
    return result;
}

void GtsamBackendAdapter::reset(const cv::Mat& initialMapFromBody)
{
    std::lock_guard<std::mutex> lock(implementation_->mutex);
    const gtsam::Pose3 initialPose = initialMapFromBody.empty()
        ? gtsam::Pose3()
        : toPose3(initialMapFromBody);
    implementation_->backend =
        std::make_unique<uw_slam::dynamic_backend::IncrementalBackend>(
            implementation_->backendConfig, implementation_->calibration,
            implementation_->bodyFromCamera, initialPose);
    implementation_->lastAcceptedTimestamp =
        -std::numeric_limits<double>::infinity();
    implementation_->lastAcceptedResult = BackendFrameResult();
    implementation_->lastPublishedVersion = 0U;
    implementation_->landmarkKeys.clear();
    implementation_->nextLandmarkKey = 1U;
}

bool GtsamBackendAdapter::rebase(const BackendMapSnapshot& snapshot)
{
    std::lock_guard<std::mutex> lock(implementation_->mutex);
    if (snapshot.keyframes.empty())
        return false;
    const auto navigation = implementation_->backend->snapshot();
    if (navigation.timestamps.empty() ||
        navigation.timestamps.size() != navigation.velocities.size() ||
        navigation.timestamps.size() != navigation.biases.size())
        return false;
    uw_slam::dynamic_backend::IncrementalRebaseState state;
    std::vector<BackendKeyframeState> keyframes = snapshot.keyframes;
    std::sort(keyframes.begin(), keyframes.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.timestampSec < rhs.timestampSec;
              });
    if (!std::isfinite(keyframes.back().timestampSec) ||
        keyframes.back().timestampSec < implementation_->lastAcceptedTimestamp)
        return false;
    state.timestamps.reserve(navigation.timestamps.size());
    state.poses.reserve(navigation.timestamps.size());
    state.velocities = navigation.velocities;
    state.biases = navigation.biases;
    for (std::size_t index = 0; index < navigation.timestamps.size(); ++index) {
        const double timestamp = navigation.timestamps[index];
        const auto keyframe = std::lower_bound(
            keyframes.begin(), keyframes.end(), timestamp,
            [](const BackendKeyframeState& candidate, double value) {
                return candidate.timestampSec < value;
            });
        if (keyframe == keyframes.end() || keyframe->timestampSec != timestamp)
            return false;
        cv::Mat pose(4, 4, CV_32F);
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                pose.at<float>(row, col) =
                    keyframe->pose[static_cast<std::size_t>(row * 4 + col)];
        state.timestamps.push_back(timestamp);
        state.poses.push_back(toPose3(pose).compose(
            implementation_->bodyFromCamera.inverse()));
    }
    state.landmarks.reserve(snapshot.landmarks.size());
    implementation_->landmarkKeys.clear();
    std::uint64_t nextKey = 1U;
    for (const auto& landmark : snapshot.landmarks) {
        if (landmark.id >= (std::uint64_t{1} << 55U))
            return false;
        const gtsam::Point3 point(landmark.position[0], landmark.position[1],
                                  landmark.position[2]);
        state.landmarks.emplace_back(landmark.id, point);
        implementation_->landmarkKeys[landmark.id] = landmark.id;
        nextKey = std::max(nextKey, landmark.id + 1U);
    }
    if (!implementation_->backend->rebase(state)) {
        implementation_->landmarkKeys.clear();
        return false;
    }
    implementation_->nextLandmarkKey = nextKey;
    implementation_->lastAcceptedTimestamp = state.timestamps.back();
    implementation_->lastAcceptedResult = BackendFrameResult();
    return true;
}

BackendFrameResult GtsamBackendAdapter::optimize(const BackendFrameInput& input)
{
    std::lock_guard<std::mutex> lock(implementation_->mutex);
    if (input.timestampSec == implementation_->lastAcceptedTimestamp) {
        BackendFrameResult repeated = implementation_->lastAcceptedResult;
        repeated.incrementalCommitted = false;
        repeated.diagnostic = "idempotent keyframe already committed";
        return repeated;
    }

    BackendFrameResult output;
    uw_slam::dynamic_backend::IncrementalFrameBatch batch;
    batch.timestampSec = input.timestampSec;
    if (!input.frontEndCameraFromMap.empty()) {
        try {
            const gtsam::Pose3 cameraFromMap =
                toPose3(input.frontEndCameraFromMap);
            batch.referenceMapFromBody = cameraFromMap.inverse().compose(
                implementation_->bodyFromCamera.inverse());
        } catch (const std::exception& exception) {
            output.diagnostic = "invalid integration map pose: " +
                                std::string(exception.what());
            return output;
        }
    }
    batch.stereoCovariance = toMatrix3(input.stereoCovariance);
    batch.observations.reserve(input.stereo.size());
    for (const auto& observation : input.stereo) {
        if (!std::isfinite(observation.uLeft) ||
            !std::isfinite(observation.vLeft) ||
            !std::isfinite(observation.uRight) ||
            observation.uLeft <= observation.uRight)
            continue;
        uw_slam::dynamic_backend::StereoTrackObservation converted;
        const auto [entry, inserted] = implementation_->landmarkKeys.emplace(
            observation.trackId, implementation_->nextLandmarkKey);
        if (inserted)
            ++implementation_->nextLandmarkKey;
        if (entry->second >= (std::uint64_t{1} << 55U))
            throw std::overflow_error("GTSAM landmark key namespace exhausted");
        converted.trackId = entry->second;
        converted.pixel = gtsam::StereoPoint2(
            observation.uLeft, observation.uRight, observation.vLeft);
        converted.pyramidSigmaPx = observation.pyramidSigmaPx;
        converted.hasLandmark = observation.hasLandmark;
        if (observation.hasLandmark) {
            converted.landmarkMap = gtsam::Point3(
                observation.landmarkMap[0], observation.landmarkMap[1],
                observation.landmarkMap[2]);
        }
        batch.observations.push_back(converted);
    }
    for (const auto& sample : input.imu) {
        uw_slam::dynamic_backend::ReplayImuSample converted;
        converted.timestampSec = sample.timestampSec;
        converted.acceleration = toVector3(sample.acceleration);
        converted.angularVelocity = toVector3(sample.angularVelocity);
        batch.imu.push_back(converted);
    }
    batch.dvl.reserve(input.dvl.size());
    for (const auto& measurement : input.dvl) {
        uw_slam::dynamic_backend::OptionalDvlMeasurement dvl;
        dvl.timestampSec = measurement.timestampSec;
        dvl.velocity = toVector3(measurement.velocity);
        dvl.covariance = toMatrix3(measurement.covariance);
        dvl.angularVelocityBody = toVector3(measurement.angularVelocityBody);
        dvl.healthAccepted = measurement.healthAccepted;
        dvl.trackMode = toBackendTrackMode(measurement.trackMode);
        dvl.validBeamRatio = measurement.validBeamRatio;
        dvl.altitudeValid = measurement.altitudeValid;
        dvl.altitudeMeters = measurement.altitudeMeters;
        dvl.errorVelocityMetersPerSec = measurement.errorVelocityMetersPerSec;
        dvl.status = measurement.status;
        batch.dvl.push_back(dvl);
    }
    if (input.hasPressure) {
        uw_slam::dynamic_backend::OptionalPressureMeasurement pressure;
        pressure.timestampSec = input.pressure.timestampSec;
        pressure.depthMeters = input.pressure.depthMeters;
        pressure.varianceMetersSquared = input.pressure.varianceMetersSquared;
        batch.pressure = pressure;
    }
    if (input.hasImuCovariance) {
        uw_slam::dynamic_backend::ImuCovarianceDecision covariance;
        covariance.accelerometer = toMatrix3(input.imuAccelerometerCovariance);
        covariance.gyroscope = toMatrix3(input.imuGyroscopeCovariance);
        covariance.accelerometerBias = toMatrix3(input.imuAccelerometerBiasCovariance);
        covariance.gyroscopeBias = toMatrix3(input.imuGyroscopeBiasCovariance);
        batch.imuCovariance = covariance;
    }
    const auto result = implementation_->backend->update(batch);
    output.accepted = result.accepted;
    output.incrementalCommitted = result.accepted;
    output.mapVersion = result.accepted
        ? std::max(result.version,
                   implementation_->lastPublishedVersion + 1U)
        : implementation_->lastPublishedVersion;
    output.diagnostic = result.diagnostic;
    output.factorDiagnostics.dvl.attempted = result.factorDiagnostics.dvl.attempted;
    output.factorDiagnostics.dvl.accepted = result.factorDiagnostics.dvl.accepted;
    output.factorDiagnostics.dvl.rejected = result.factorDiagnostics.dvl.rejected;
    output.factorDiagnostics.dvl.downweighted =
        result.factorDiagnostics.dvl.downweighted;
    output.factorDiagnostics.dvl.reason = result.factorDiagnostics.dvl.reason;
    output.factorDiagnostics.pressure.attempted = result.factorDiagnostics.pressure.attempted;
    output.factorDiagnostics.pressure.accepted = result.factorDiagnostics.pressure.accepted;
    output.factorDiagnostics.pressure.rejected = result.factorDiagnostics.pressure.rejected;
    output.factorDiagnostics.pressure.reason = result.factorDiagnostics.pressure.reason;
    const auto& sensors = result.sensorDiagnostics;
    output.sensorDiagnostics.stereo.input = sensors.stereo.input;
    output.sensorDiagnostics.stereo.accepted = sensors.stereo.accepted;
    output.sensorDiagnostics.stereo.rejected = sensors.stereo.rejected;
    output.sensorDiagnostics.stereo.weightAvailable =
        sensors.stereo.covariance.array().isFinite().all() &&
        sensors.stereo.information.array().isFinite().all();
    output.sensorDiagnostics.stereo.covariance =
        fromMatrix3(sensors.stereo.covariance);
    output.sensorDiagnostics.stereo.information =
        fromMatrix3(sensors.stereo.information);
    output.sensorDiagnostics.stereo.huberK = sensors.stereo.huberK;
    output.sensorDiagnostics.imu.present = sensors.imu.present;
    output.sensorDiagnostics.imu.sampleCount = sensors.imu.sampleCount;
    output.sensorDiagnostics.imu.spanSec = sensors.imu.spanSec;
    output.sensorDiagnostics.imu.maximumGapSec = sensors.imu.maximumGapSec;
    output.sensorDiagnostics.imu.accelerationRms = sensors.imu.accelerationRms;
    output.sensorDiagnostics.imu.angularVelocityRms =
        sensors.imu.angularVelocityRms;
    output.sensorDiagnostics.imu.weightAvailable =
        sensors.imu.covariance.accelerometer.array().isFinite().all() &&
        sensors.imu.information.accelerometer.array().isFinite().all();
    copyImuCovariance(sensors.imu.covariance,
                      &output.sensorDiagnostics.imu.covariance);
    copyImuCovariance(sensors.imu.information,
                      &output.sensorDiagnostics.imu.information);
    output.sensorDiagnostics.dvl.enabled = sensors.dvl.enabled;
    output.sensorDiagnostics.dvl.present = sensors.dvl.present;
    output.sensorDiagnostics.dvl.healthAccepted = sensors.dvl.healthAccepted;
    output.sensorDiagnostics.dvl.timeOffsetSec = sensors.dvl.timeOffsetSec;
    output.sensorDiagnostics.dvl.sourceAgeSec = sensors.dvl.sourceAgeSec;
    output.sensorDiagnostics.dvl.baselineSource = sensors.dvl.baselineSource;
    output.sensorDiagnostics.dvl.baselineCovariance =
        fromMatrix3(sensors.dvl.baselineCovariance);
    output.sensorDiagnostics.dvl.weightAvailable =
        sensors.dvl.covariance.array().isFinite().all() &&
        sensors.dvl.information.array().isFinite().all();
    output.sensorDiagnostics.dvl.covariance = fromMatrix3(sensors.dvl.covariance);
    output.sensorDiagnostics.dvl.dynamicCovariance =
        fromMatrix3(sensors.dvl.dynamicCovariance);
    output.sensorDiagnostics.dvl.information =
        fromMatrix3(sensors.dvl.information);
    output.sensorDiagnostics.dvl.nisAvailable = sensors.dvl.nis.has_value();
    if (sensors.dvl.nis)
        output.sensorDiagnostics.dvl.nis = *sensors.dvl.nis;
    output.sensorDiagnostics.dvl.dynamicWeight = sensors.dvl.dynamicWeight;
    output.sensorDiagnostics.dvl.downweighted = sensors.dvl.downweighted;
    output.sensorDiagnostics.pressure.enabled = sensors.pressure.enabled;
    output.sensorDiagnostics.pressure.present = sensors.pressure.present;
    output.sensorDiagnostics.pressure.timeOffsetSec = sensors.pressure.timeOffsetSec;
    output.sensorDiagnostics.pressure.weightAvailable =
        std::isfinite(sensors.pressure.variance) &&
        std::isfinite(sensors.pressure.information);
    output.sensorDiagnostics.pressure.variance = sensors.pressure.variance;
    output.sensorDiagnostics.pressure.information = sensors.pressure.information;
    output.sensorDiagnostics.pressure.nisAvailable =
        sensors.pressure.nis.has_value();
    if (sensors.pressure.nis)
        output.sensorDiagnostics.pressure.nis = *sensors.pressure.nis;
    const auto& graph = result.observability;
    output.graphDiagnostics.submittedFactors = graph.submittedFactors;
    output.graphDiagnostics.submittedValues = graph.submittedValues;
    output.graphDiagnostics.totalFactors = graph.totalFactors;
    output.graphDiagnostics.totalValues = graph.totalValues;
    output.graphDiagnostics.totalLandmarks = graph.totalLandmarks;
    output.graphDiagnostics.totalPoseStates = graph.totalPoseStates;
    output.graphDiagnostics.archivedPoseCount = graph.archivedPoseCount;
    output.graphDiagnostics.variablesRelinearized = graph.variablesRelinearized;
    output.graphDiagnostics.variablesReeliminated = graph.variablesReeliminated;
    output.graphDiagnostics.factorsRecalculated = graph.factorsRecalculated;
    output.graphDiagnostics.cliques = graph.cliques;
    output.graphDiagnostics.treeNnz = graph.treeNnz;
    output.graphDiagnostics.batchReorderTriggered = graph.batchReorderTriggered;
    output.graphDiagnostics.errorBefore = graph.errorBefore;
    output.graphDiagnostics.errorAfter = graph.errorAfter;
    output.graphDiagnostics.candidateCopyMs = graph.candidateCopyMs;
    output.graphDiagnostics.isamUpdateMs = graph.isamUpdateMs;
    output.graphDiagnostics.estimateCalculationMs = graph.estimateCalculationMs;
    output.graphDiagnostics.transactionMs = graph.transactionMs;
    output.bodyPose = fromPose3(result.pose);
    if (!output.accepted &&
        output.diagnostic.find("key already exists") != std::string::npos &&
        implementation_->lastAcceptedResult.accepted) {
        // An iSAM transaction can reject a duplicate landmark key after
        // retaining the previous finite estimate. Keep tracking on that
        // estimate and expose the degraded transaction instead of forcing an
        // AQUA map reset/reinitialization loop.
        BackendFrameResult degraded = implementation_->lastAcceptedResult;
        degraded.incrementalCommitted = false;
        degraded.diagnostic = "degraded duplicate-landmark transaction: " +
                              output.diagnostic;
        return degraded;
    }
    if (output.accepted) {
        implementation_->lastPublishedVersion = output.mapVersion;
        implementation_->lastAcceptedTimestamp = input.timestampSec;
        implementation_->lastAcceptedResult = output;
    }
    return output;
}

namespace
{
BackendFrameResult refineVisualFrame(
    const BackendFrameInput& input,
    const cv::Mat& initialCameraFromMap,
    const gtsam::Pose3& bodyFromCamera,
    const uw_slam::dynamic_backend::FrameOptimizer& frameOptimizer)
{
    BackendFrameResult output;
    gtsam::Pose3 cameraFromMap;
    try {
        cameraFromMap = toPose3(initialCameraFromMap);
    } catch (const std::exception& exception) {
        output.diagnostic = "invalid front-end pose: " +
                            std::string(exception.what());
        return output;
    }
    const gtsam::Pose3 mapFromCamera = cameraFromMap.inverse();
    const gtsam::Pose3 mapFromBody = mapFromCamera.compose(
        bodyFromCamera.inverse());
    uw_slam::dynamic_backend::FrameOptimizationProblem problem;
    problem.previousPose = mapFromBody;
    problem.predictedPose = mapFromBody;
    problem.bodyFromCamera = bodyFromCamera;
    problem.stereoCovariance = toMatrix3(input.stereoCovariance);
    for (const auto& observation : input.stereo) {
        if (!observation.hasLandmark)
            continue;
        uw_slam::dynamic_backend::LandmarkObservation converted;
        converted.trackId = observation.trackId;
        converted.pixel = gtsam::StereoPoint2(
            observation.uLeft, observation.uRight, observation.vLeft);
        converted.landmarkMap = gtsam::Point3(
            observation.landmarkMap[0], observation.landmarkMap[1],
            observation.landmarkMap[2]);
        problem.observations.push_back(converted);
    }
    for (const auto& observation : input.monocularLandmarks) {
        uw_slam::dynamic_backend::MonocularLandmarkObservation converted;
        converted.trackId = observation.trackId;
        converted.pixel = gtsam::Point2(observation.u, observation.v);
        converted.landmarkMap = gtsam::Point3(
            observation.landmarkMap[0], observation.landmarkMap[1],
            observation.landmarkMap[2]);
        problem.monocularObservations.push_back(converted);
    }
    const auto result = frameOptimizer.optimize(problem);
    output.accepted = result.accepted;
    output.diagnostic = result.diagnostic;
    output.bodyPose = fromPose3(result.pose);
    output.optimizedPoseInlierTrackIds = result.optimizedPoseInlierTrackIds;
    output.frontEndPoseInlierTrackIds = result.previousPoseInlierTrackIds;
    return output;
}
}  // namespace

BackendFrameResult GtsamBackendAdapter::refinePose(
    const BackendFrameInput& input, const cv::Mat& initialCameraFromMap) const
{
    std::lock_guard<std::mutex> lock(implementation_->mutex);
    return refineVisualFrame(input, initialCameraFromMap,
        implementation_->bodyFromCamera, *implementation_->frameOptimizer);
}

BackendFrameResult GtsamBackendAdapter::refineVisualPose(
    const BackendFrameInput& input, const cv::Mat& cameraMatrix,
    float baselineFx, const cv::Mat& bodyFromCamera,
    const cv::Mat& initialCameraFromMap)
{
    BackendFrameResult output;
    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3 ||
        cameraMatrix.type() != CV_32F || !cv::checkRange(cameraMatrix) ||
        !(std::isfinite(baselineFx) && baselineFx > 0.0F &&
          cameraMatrix.at<float>(0, 0) > 0.0F &&
          cameraMatrix.at<float>(1, 1) > 0.0F)) {
        output.diagnostic = "invalid stereo calibration for visual bootstrap";
        return output;
    }
    try {
        const auto calibration = std::make_shared<gtsam::Cal3_S2Stereo>(
            cameraMatrix.at<float>(0, 0), cameraMatrix.at<float>(1, 1), 0.0,
            cameraMatrix.at<float>(0, 2), cameraMatrix.at<float>(1, 2),
            baselineFx / cameraMatrix.at<float>(0, 0));
        uw_slam::dynamic_backend::FrameOptimizerConfig config;
        const uw_slam::dynamic_backend::FrameOptimizer optimizer(config, calibration);
        return refineVisualFrame(input, initialCameraFromMap,
                                 toPose3(bodyFromCamera), optimizer);
    } catch (const std::exception& exception) {
        output.diagnostic = "visual bootstrap refinement: " +
                            std::string(exception.what());
        return output;
    }
}

GtsamCameraPrediction GtsamBackendAdapter::predictCameraPose(
    double timestampSec, const std::vector<ImuSample>& imu) const
{
    std::lock_guard<std::mutex> lock(implementation_->mutex);
    std::vector<uw_slam::dynamic_backend::ReplayImuSample> converted;
    converted.reserve(imu.size());
    for (const auto& sample : imu) {
        uw_slam::dynamic_backend::ReplayImuSample value;
        value.timestampSec = sample.timestampSec;
        value.acceleration = toVector3(sample.acceleration);
        value.angularVelocity = toVector3(sample.angularVelocity);
        converted.push_back(value);
    }

    const auto prediction = implementation_->backend->predict(
        timestampSec, converted);
    GtsamCameraPrediction output;
    output.accepted = prediction.accepted;
    output.diagnostic = prediction.diagnostic;
    if (prediction.accepted) {
        output.cameraFromMap = fromPose3(
            prediction.pose.compose(implementation_->bodyFromCamera).inverse());
    }
    return output;
}

cv::Mat GtsamBackendAdapter::cameraPose(const BackendFrameResult& result) const
{
    std::lock_guard<std::mutex> lock(implementation_->mutex);
    if (!result.accepted || result.bodyPose.empty())
        return cv::Mat();
    const gtsam::Pose3 bodyPose = toPose3(result.bodyPose);
    return fromPose3(bodyPose.compose(implementation_->bodyFromCamera).inverse());
}

cv::Mat GtsamBackendAdapter::gravityMapFromImu(
    const std::vector<ImuSample>& imu, const cv::Mat& mapFromImu)
{
    if (imu.size() < 2U)
        throw std::invalid_argument("insufficient IMU samples for gravity alignment");
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    double previousTimestamp = -std::numeric_limits<double>::infinity();
    for (const auto& sample : imu) {
        const Eigen::Vector3d acceleration(sample.acceleration[0],
                                           sample.acceleration[1],
                                           sample.acceleration[2]);
        if (!std::isfinite(sample.timestampSec) ||
            sample.timestampSec <= previousTimestamp ||
            !acceleration.array().isFinite().all())
            throw std::invalid_argument(
                "gravity-alignment IMU must be finite and strictly increasing");
        mean += acceleration;
        previousTimestamp = sample.timestampSec;
    }
    mean /= static_cast<double>(imu.size());
    const double magnitude = mean.norm();
    if (!(std::isfinite(magnitude) && magnitude > 0.0))
        throw std::invalid_argument("gravity alignment requires a nonzero mean acceleration");
    if (mapFromImu.rows != 4 || mapFromImu.cols != 4 ||
        mapFromImu.type() != CV_32F || !cv::checkRange(mapFromImu))
        throw std::invalid_argument("mapFromImu must be finite 4x4 CV_32F");
    Eigen::Matrix3d rotation;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            rotation(row, col) = mapFromImu.at<float>(row, col);
    const Eigen::Vector3d gravityMap =
        rotation * (-9.81 * mean.normalized());
    cv::Mat result(3, 1, CV_32F);
    for (int row = 0; row < 3; ++row)
        result.at<float>(row) = static_cast<float>(gravityMap(row));
    return result;
}

}  // namespace ORB_SLAM3
