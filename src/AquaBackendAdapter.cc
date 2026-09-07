#include "AquaBackendAdapter.h"

#include "Frame.h"
#include "ImuTypes.h"
#include "MapPoint.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ORB_SLAM3
{
namespace
{
// GTSAM Symbol reserves the upper 8 bits for the character prefix and keeps
// only a 56-bit index. Keep transient IDs inside that index range.
constexpr std::uint64_t kTransientMask = std::uint64_t{1} << 55U;
constexpr std::uint64_t kMaximumFrameId = (std::uint64_t{1} << 23U) - 1U;

bool finite(float value)
{
    return std::isfinite(static_cast<double>(value));
}

bool validPose(const cv::Mat& pose)
{
    if (pose.rows != 4 || pose.cols != 4 || pose.type() != CV_32F)
        return false;
    for (int row = 0; row < pose.rows; ++row)
        for (int col = 0; col < pose.cols; ++col)
            if (!finite(pose.at<float>(row, col)))
                return false;
    return std::abs(pose.at<float>(3, 3) - 1.0F) < 1e-4F;
}

std::uint64_t transientTrackId(const Frame& frame, int featureIndex)
{
    if (frame.mnId > kMaximumFrameId || featureIndex < 0)
        throw std::overflow_error("frame identity exceeds transient track namespace");
    return kTransientMask | (static_cast<std::uint64_t>(frame.mnId) << 32U) |
           static_cast<std::uint32_t>(featureIndex);
}
}  // namespace

BackendFrameInput AquaBackendAdapter::fromFrame(const Frame& frame,
                                                const RuntimeProfile& profile)
{
    if (!std::isfinite(frame.mTimeStamp))
        throw std::invalid_argument("frame timestamp must be finite");
    const int observationCount = frame.Nleft > 0 ? frame.Nleft : frame.N;
    if (frame.N < 0 || observationCount < 0 ||
        frame.mvKeysUn.size() < static_cast<std::size_t>(observationCount))
        throw std::invalid_argument("frame keypoint count is inconsistent");
    if (!frame.mvuRight.empty() &&
        frame.mvuRight.size() < static_cast<std::size_t>(observationCount))
        throw std::invalid_argument("frame right-coordinate count is inconsistent");
    if (!frame.mvDepth.empty() &&
        frame.mvDepth.size() < static_cast<std::size_t>(observationCount))
        throw std::invalid_argument("frame depth count is inconsistent");

    BackendFrameInput input;
    input.timestampSec = frame.mTimeStamp;
    input.frontEndCameraFromMap = frame.mTcw.clone();
    input.calibration.sha256 = profile.calibrationSha256;
    input.stereo.reserve(static_cast<std::size_t>(observationCount));
    std::vector<double> pyramidSigmas;
    pyramidSigmas.reserve(static_cast<std::size_t>(observationCount));
    for (int index = 0; index < observationCount; ++index)
    {
        const auto& key = frame.mvKeysUn[static_cast<std::size_t>(index)].pt;
        if (!finite(key.x) || !finite(key.y))
            throw std::invalid_argument("frame keypoint is non-finite");
        StereoTrackObservation observation;
        observation.trackId = transientTrackId(frame, index);
        observation.uLeft = key.x;
        observation.vLeft = key.y;
        const int octave =
            frame.mvKeysUn[static_cast<std::size_t>(index)].octave;
        if (octave >= 0 &&
            static_cast<std::size_t>(octave) < frame.mvLevelSigma2.size() &&
            frame.mvLevelSigma2[static_cast<std::size_t>(octave)] > 0.0F)
            observation.pyramidSigmaPx = std::sqrt(
                frame.mvLevelSigma2[static_cast<std::size_t>(octave)]);
        if (!frame.mvuRight.empty())
            observation.uRight = frame.mvuRight[static_cast<std::size_t>(index)];
        if (!frame.mvDepth.empty())
            observation.depth = frame.mvDepth[static_cast<std::size_t>(index)];
        if (!finite(observation.uRight) || !finite(observation.depth))
            throw std::invalid_argument("stereo observation is non-finite");
        const bool validStereo = observation.uRight >= 0.0F &&
            observation.depth > 0.0F && observation.uLeft > observation.uRight;
        bool hasLandmark = false;
        if (index < static_cast<int>(frame.mvpMapPoints.size()) &&
            frame.mvpMapPoints[static_cast<std::size_t>(index)] != nullptr) {
            const std::uint64_t mapPointId = static_cast<std::uint64_t>(
                frame.mvpMapPoints[static_cast<std::size_t>(index)]->mnId);
            if (mapPointId >= kTransientMask)
                throw std::overflow_error("map point identity exceeds persistent namespace");
            observation.trackId = mapPointId;
            const cv::Mat world =
                frame.mvpMapPoints[static_cast<std::size_t>(index)]->GetWorldPos();
            if (world.rows != 3 || world.cols != 1 || world.type() != CV_32F ||
                !cv::checkRange(world))
                throw std::invalid_argument("map landmark is invalid");
            for (int axis = 0; axis < 3; ++axis)
                observation.landmarkMap[static_cast<std::size_t>(axis)] =
                    world.at<float>(axis);
            observation.hasLandmark = true;
            hasLandmark = true;
        }
        if (validStereo) {
            pyramidSigmas.push_back(observation.pyramidSigmaPx);
            input.stereo.push_back(observation);
        } else if (hasLandmark) {
            MonocularTrackObservation monocular;
            monocular.trackId = observation.trackId;
            monocular.u = observation.uLeft;
            monocular.v = observation.vLeft;
            monocular.pyramidSigmaPx = observation.pyramidSigmaPx;
            monocular.landmarkMap = observation.landmarkMap;
            pyramidSigmas.push_back(monocular.pyramidSigmaPx);
            input.monocularLandmarks.push_back(monocular);
        }
    }
    if (!pyramidSigmas.empty()) {
        const auto middle = pyramidSigmas.begin() +
            static_cast<std::ptrdiff_t>(pyramidSigmas.size() / 2U);
        std::nth_element(pyramidSigmas.begin(), middle, pyramidSigmas.end());
        const double variance = (*middle) * (*middle);
        input.stereoCovariance = {variance, 0.0, 0.0,
                                  0.0, variance, 0.0,
                                  0.0, 0.0, variance};
    }
    return input;
}

bool AquaBackendAdapter::applyDvlMeasurement(
    BackendFrameInput& input, const IMU::GyroDvlPoint& measurement)
{
    if (!measurement.isDvlMeasurement || !std::isfinite(measurement.t))
        return false;
    const std::array<double, 3> velocity{
        measurement.v.x, measurement.v.y, measurement.v.z};
    const std::array<double, 3> angularVelocity{
        measurement.dvlAngularVelocityBody.x,
        measurement.dvlAngularVelocityBody.y,
        measurement.dvlAngularVelocityBody.z};
    if (!std::all_of(velocity.begin(), velocity.end(),
                     [](double value) { return std::isfinite(value); }) ||
        !std::all_of(angularVelocity.begin(), angularVelocity.end(),
                     [](double value) { return std::isfinite(value); }) ||
        !std::all_of(measurement.dvlCovariance.begin(),
                     measurement.dvlCovariance.end(),
                     [](double value) { return std::isfinite(value); }) ||
        measurement.dvlTrackMode == DvlTrackMode::Unknown ||
        !std::isfinite(measurement.dvlValidBeamRatio) ||
        measurement.dvlValidBeamRatio < 0.0 ||
        measurement.dvlValidBeamRatio > 1.0 ||
        !std::isfinite(measurement.dvlErrorVelocityMetersPerSec) ||
        measurement.dvlErrorVelocityMetersPerSec < 0.0 ||
        (measurement.dvlAltitudeValid &&
         (!std::isfinite(measurement.dvlAltitudeMeters) ||
          measurement.dvlAltitudeMeters < 0.0)))
        return false;
    DvlMeasurement converted;
    converted.timestampSec = measurement.t;
    converted.velocity = velocity;
    converted.covariance = measurement.dvlCovariance;
    converted.angularVelocityBody = angularVelocity;
    converted.healthAccepted = measurement.dvlHealthAccepted;
    converted.trackMode = measurement.dvlTrackMode;
    converted.validBeamRatio = measurement.dvlValidBeamRatio;
    converted.altitudeValid = measurement.dvlAltitudeValid;
    converted.altitudeMeters = measurement.dvlAltitudeMeters;
    converted.errorVelocityMetersPerSec =
        measurement.dvlErrorVelocityMetersPerSec;
    converted.status = measurement.dvlStatus;
    input.dvl.push_back(converted);
    return true;
}

bool AquaBackendAdapter::applyDynamicImuCovariance(
    BackendFrameInput& input, const IMU::Calib& calibration,
    double nominalFrequencyHz)
{
    input.hasImuCovariance = false;
    if (!(std::isfinite(nominalFrequencyHz) && nominalFrequencyHz > 0.0))
        return false;
    // IMU::Calib::Set stores independent channels. Refuse unsupported
    // correlations instead of silently discarding them at the graph boundary.
    const auto validCalibration = [](const cv::Mat& covariance) {
        if (covariance.rows != 6 || covariance.cols != 6 ||
            covariance.type() != CV_32F)
            return false;
        for (int row = 0; row < 6; ++row)
            for (int col = 0; col < 6; ++col) {
                const double value = covariance.at<float>(row, col);
                if (!std::isfinite(value) ||
                    (row == col ? value <= 0.0 : value != 0.0))
                    return false;
            }
        return true;
    };
    if (!validCalibration(calibration.Cov) ||
        !validCalibration(calibration.CovWalk))
        return false;

    // Tracking passes Ng*sqrt(f), Na*sqrt(f), Ngw/sqrt(f), Naw/sqrt(f)
    // to Calib::Set, which squares them. GTSAM expects continuous-time
    // covariance densities: measurement Cov/f and bias random walk CovWalk*f.
    const auto convert = [](const cv::Mat& covariance, int offset,
                            double scale, std::array<double, 9>& output) {
        output.fill(0.0);
        for (int axis = 0; axis < 3; ++axis) {
            const double variance =
                static_cast<double>(covariance.at<float>(offset + axis,
                                                        offset + axis)) * scale;
            if (!(std::isfinite(variance) && variance > 0.0))
                return false;
            output[4U * axis] = variance;
        }
        return true;
    };
    if (!convert(calibration.Cov, 0, 1.0 / nominalFrequencyHz,
                 input.imuGyroscopeCovariance) ||
        !convert(calibration.Cov, 3, 1.0 / nominalFrequencyHz,
                 input.imuAccelerometerCovariance) ||
        !convert(calibration.CovWalk, 0, nominalFrequencyHz,
                 input.imuGyroscopeBiasCovariance) ||
        !convert(calibration.CovWalk, 3, nominalFrequencyHz,
                 input.imuAccelerometerBiasCovariance))
        return false;
    input.hasImuCovariance = true;
    return true;
}

std::string AquaBackendAdapter::formatDynamicSensorDiagnostics(
    double timestampSec, const BackendFrameResult& result)
{
    const auto appendMatrix = [](std::ostringstream& stream,
                                 bool available,
                                 const std::array<double, 9>& matrix) {
        if (!available) {
            stream << "unavailable";
            return;
        }
        stream << '[';
        for (std::size_t index = 0; index < matrix.size(); ++index) {
            if (index > 0U)
                stream << ',';
            stream << matrix[index];
        }
        stream << ']';
    };
    const auto appendOptional = [](std::ostringstream& stream,
                                   bool available, double value) {
        if (available)
            stream << value;
        else
            stream << "unavailable";
    };

    const auto& sensors = result.sensorDiagnostics;
    const auto& graph = result.graphDiagnostics;
    std::ostringstream stream;
    stream << "dynamic_sensor_diagnostics"
           << " t=" << std::fixed << std::setprecision(9) << timestampSec
           << std::defaultfloat << std::setprecision(9)
           << " accepted=" << result.accepted
           << " committed=" << result.incrementalCommitted
           << " stereo={input=" << sensors.stereo.input
           << ",accepted=" << sensors.stereo.accepted
           << ",rejected=" << sensors.stereo.rejected
           << ",huber_k=" << sensors.stereo.huberK << ",covariance=";
    appendMatrix(stream, sensors.stereo.weightAvailable,
                 sensors.stereo.covariance);
    stream << ",information=";
    appendMatrix(stream, sensors.stereo.weightAvailable,
                 sensors.stereo.information);
    stream << "} imu={present=" << sensors.imu.present
           << ",samples=" << sensors.imu.sampleCount << ",span_s=";
    appendOptional(stream, sensors.imu.present && std::isfinite(sensors.imu.spanSec),
                   sensors.imu.spanSec);
    stream << ",max_gap_s=";
    appendOptional(stream, sensors.imu.present && sensors.imu.sampleCount >= 2U &&
                       std::isfinite(sensors.imu.maximumGapSec),
                   sensors.imu.maximumGapSec);
    stream << ",accel_rms=";
    appendOptional(stream, sensors.imu.present &&
                       std::isfinite(sensors.imu.accelerationRms),
                   sensors.imu.accelerationRms);
    stream << ",gyro_rms=";
    appendOptional(stream, sensors.imu.present &&
                       std::isfinite(sensors.imu.angularVelocityRms),
                   sensors.imu.angularVelocityRms);
    stream << ",accel_covariance=";
    appendMatrix(stream, sensors.imu.weightAvailable,
                 sensors.imu.covariance.accelerometer);
    stream << ",accel_information=";
    appendMatrix(stream, sensors.imu.weightAvailable,
                 sensors.imu.information.accelerometer);
    stream << ",gyro_covariance=";
    appendMatrix(stream, sensors.imu.weightAvailable,
                 sensors.imu.covariance.gyroscope);
    stream << ",gyro_information=";
    appendMatrix(stream, sensors.imu.weightAvailable,
                 sensors.imu.information.gyroscope);
    stream << ",accel_bias_covariance=";
    appendMatrix(stream, sensors.imu.weightAvailable,
                 sensors.imu.covariance.accelerometerBias);
    stream << ",accel_bias_information=";
    appendMatrix(stream, sensors.imu.weightAvailable,
                 sensors.imu.information.accelerometerBias);
    stream << ",gyro_bias_covariance=";
    appendMatrix(stream, sensors.imu.weightAvailable,
                 sensors.imu.covariance.gyroscopeBias);
    stream << ",gyro_bias_information=";
    appendMatrix(stream, sensors.imu.weightAvailable,
                 sensors.imu.information.gyroscopeBias);
    stream << "} dvl={enabled=" << sensors.dvl.enabled
           << ",present=" << sensors.dvl.present
           << ",health_accepted=" << sensors.dvl.healthAccepted
           << ",time_offset_s=";
    appendOptional(stream, sensors.dvl.present &&
                       std::isfinite(sensors.dvl.timeOffsetSec),
                   sensors.dvl.timeOffsetSec);
    stream << ",source_age_s=";
    appendOptional(stream, sensors.dvl.present &&
                       std::isfinite(sensors.dvl.sourceAgeSec),
                   sensors.dvl.sourceAgeSec);
    stream << ",baseline_source="
           << (sensors.dvl.baselineSource.empty()
                   ? "unavailable"
                   : sensors.dvl.baselineSource)
           << ",baseline_covariance=";
    appendMatrix(stream, !sensors.dvl.baselineSource.empty(),
                 sensors.dvl.baselineCovariance);
    stream << ",dynamic_covariance=";
    appendMatrix(stream, !sensors.dvl.baselineSource.empty(),
                 sensors.dvl.dynamicCovariance);
    stream << ",covariance=";
    appendMatrix(stream, sensors.dvl.weightAvailable, sensors.dvl.covariance);
    stream << ",information=";
    appendMatrix(stream, sensors.dvl.weightAvailable, sensors.dvl.information);
    stream << ",nis=";
    appendOptional(stream, sensors.dvl.nisAvailable, sensors.dvl.nis);
    stream << ",dynamic_weight=";
    appendOptional(stream, std::isfinite(sensors.dvl.dynamicWeight),
                   sensors.dvl.dynamicWeight);
    stream << ",downweighted=" << sensors.dvl.downweighted;
    stream << ",attempted=" << result.factorDiagnostics.dvl.attempted
           << ",accepted=" << result.factorDiagnostics.dvl.accepted
           << ",rejected=" << result.factorDiagnostics.dvl.rejected
           << ",downweighted_count="
           << result.factorDiagnostics.dvl.downweighted
           << ",reason=" << (result.factorDiagnostics.dvl.reason.empty()
                                  ? "none"
                                  : result.factorDiagnostics.dvl.reason)
           << "} pressure={enabled=" << sensors.pressure.enabled
           << ",present=" << sensors.pressure.present << ",time_offset_s=";
    appendOptional(stream, sensors.pressure.present &&
                       std::isfinite(sensors.pressure.timeOffsetSec),
                   sensors.pressure.timeOffsetSec);
    stream << ",variance=";
    appendOptional(stream, sensors.pressure.weightAvailable,
                   sensors.pressure.variance);
    stream << ",information=";
    appendOptional(stream, sensors.pressure.weightAvailable,
                   sensors.pressure.information);
    stream << ",nis=";
    appendOptional(stream, sensors.pressure.nisAvailable, sensors.pressure.nis);
    stream << ",attempted=" << result.factorDiagnostics.pressure.attempted
           << ",accepted=" << result.factorDiagnostics.pressure.accepted
           << ",rejected=" << result.factorDiagnostics.pressure.rejected
           << ",reason=" << (result.factorDiagnostics.pressure.reason.empty()
                                  ? "none"
                                  : result.factorDiagnostics.pressure.reason)
           << "} graph={submitted_factors=" << graph.submittedFactors
           << ",total_factors=" << graph.totalFactors
           << ",submitted_values=" << graph.submittedValues
           << ",total_values=" << graph.totalValues
           << ",landmarks=" << graph.totalLandmarks
           << ",poses=" << graph.totalPoseStates
           << ",archived_poses=" << graph.archivedPoseCount
           << ",relinearized=" << graph.variablesRelinearized
           << ",reeliminated=" << graph.variablesReeliminated
           << ",recalculated=" << graph.factorsRecalculated
           << ",cliques=" << graph.cliques
           << ",tree_nnz=" << graph.treeNnz
           << ",batch_reorder=" << graph.batchReorderTriggered
           << ",error_before=";
    appendOptional(stream, std::isfinite(graph.errorBefore), graph.errorBefore);
    stream << ",error_after=";
    appendOptional(stream, std::isfinite(graph.errorAfter), graph.errorAfter);
    stream << ",copy_ms=" << graph.candidateCopyMs
           << ",update_ms=" << graph.isamUpdateMs
           << ",estimate_ms=" << graph.estimateCalculationMs
           << ",transaction_ms=" << graph.transactionMs << '}';
    return stream.str();
}

bool AquaBackendAdapter::applyFrameResult(const BackendFrameResult& result,
                                          Frame& frame)
{
    if (!result.accepted || !validPose(result.bodyPose))
        return false;
    frame.SetPose(result.bodyPose);
    return true;
}

bool AquaBackendAdapter::canPreserveFrontEndPose(
    const BackendFrameResult& result, const Frame& frame)
{
    if (result.accepted || !validPose(frame.mTcw))
        return false;
    return result.diagnostic.find("underconstrained stereo frame") !=
               std::string::npos ||
           result.diagnostic.find("duplicate-landmark transaction") !=
               std::string::npos ||
           result.diagnostic.find("key already exists") != std::string::npos;
}

int AquaBackendAdapter::persistentStereoSupport(const BackendFrameInput& input)
{
    return static_cast<int>(std::count_if(
        input.stereo.begin(), input.stereo.end(),
        [](const StereoTrackObservation& observation) {
            return observation.hasLandmark;
        }));
}

int AquaBackendAdapter::trackingSupportAfterPoseCorrection(
    bool committedKeyframe, bool correctionAccepted,
    const BackendFrameResult& result, int persistentStereoSupport)
{
    if (committedKeyframe || correctionAccepted)
        return persistentStereoSupport;
    const auto& inliers = correctionAccepted
        ? result.optimizedPoseInlierTrackIds
        : result.frontEndPoseInlierTrackIds;
    return static_cast<int>(inliers.size());
}

}  // namespace ORB_SLAM3
