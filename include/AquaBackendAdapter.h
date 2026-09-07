#ifndef AQUA_BACKEND_ADAPTER_H
#define AQUA_BACKEND_ADAPTER_H

#include "BackendFacade.h"
#include "ImuTypes.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ORB_SLAM3
{

class Frame;

struct CalibrationIdentity
{
    std::string sha256;
};

struct StereoTrackObservation
{
    std::uint64_t trackId = 0;
    float uLeft = 0.0F;
    float vLeft = 0.0F;
    float uRight = -1.0F;
    float depth = -1.0F;
    float pyramidSigmaPx = 1.0F;
    std::array<double, 3> landmarkMap{0.0, 0.0, 0.0};
    bool hasLandmark = false;
};

struct MonocularTrackObservation
{
    std::uint64_t trackId = 0;
    float u = 0.0F;
    float v = 0.0F;
    float pyramidSigmaPx = 1.0F;
    std::array<double, 3> landmarkMap{0.0, 0.0, 0.0};
};

struct ImuSample
{
    double timestampSec = 0.0;
    std::array<double, 3> acceleration{0.0, 0.0, 0.0};
    std::array<double, 3> angularVelocity{0.0, 0.0, 0.0};
};

struct DvlMeasurement
{
    double timestampSec = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 3> velocity{0.0, 0.0, 0.0};
    std::array<double, 9> covariance{};
    std::array<double, 3> angularVelocityBody{};
    bool healthAccepted = false;
    DvlTrackMode trackMode = DvlTrackMode::Unknown;
    double validBeamRatio = std::numeric_limits<double>::quiet_NaN();
    bool altitudeValid = false;
    double altitudeMeters = std::numeric_limits<double>::quiet_NaN();
    double errorVelocityMetersPerSec =
        std::numeric_limits<double>::quiet_NaN();
    std::int64_t status = 0;
};

struct PressureMeasurement
{
    double timestampSec = std::numeric_limits<double>::quiet_NaN();
    double depthMeters = std::numeric_limits<double>::quiet_NaN();
    double varianceMetersSquared = std::numeric_limits<double>::quiet_NaN();
};

struct IncrementalFactorFamilyDiagnostics
{
    std::uint64_t attempted = 0;
    std::uint64_t accepted = 0;
    std::uint64_t rejected = 0;
    std::uint64_t downweighted = 0;
    std::string reason;
};

struct IncrementalFactorDiagnostics
{
    IncrementalFactorFamilyDiagnostics dvl;
    IncrementalFactorFamilyDiagnostics pressure;
};

struct IncrementalStereoSensorDiagnostics
{
    std::size_t input = 0;
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    bool weightAvailable = false;
    std::array<double, 9> covariance{};
    std::array<double, 9> information{};
    double huberK = std::numeric_limits<double>::quiet_NaN();
};

struct IncrementalImuCovarianceDiagnostics
{
    std::array<double, 9> accelerometer{};
    std::array<double, 9> gyroscope{};
    std::array<double, 9> accelerometerBias{};
    std::array<double, 9> gyroscopeBias{};
};

struct IncrementalImuSensorDiagnostics
{
    bool present = false;
    std::size_t sampleCount = 0;
    double spanSec = std::numeric_limits<double>::quiet_NaN();
    double maximumGapSec = std::numeric_limits<double>::quiet_NaN();
    double accelerationRms = std::numeric_limits<double>::quiet_NaN();
    double angularVelocityRms = std::numeric_limits<double>::quiet_NaN();
    bool weightAvailable = false;
    IncrementalImuCovarianceDiagnostics covariance;
    IncrementalImuCovarianceDiagnostics information;
};

struct IncrementalDvlSensorDiagnostics
{
    bool enabled = false;
    bool present = false;
    bool healthAccepted = false;
    double timeOffsetSec = std::numeric_limits<double>::quiet_NaN();
    double sourceAgeSec = std::numeric_limits<double>::quiet_NaN();
    std::string baselineSource;
    std::array<double, 9> baselineCovariance{};
    bool weightAvailable = false;
    std::array<double, 9> covariance{};
    std::array<double, 9> dynamicCovariance{};
    std::array<double, 9> information{};
    bool nisAvailable = false;
    double nis = std::numeric_limits<double>::quiet_NaN();
    double dynamicWeight = std::numeric_limits<double>::quiet_NaN();
    bool downweighted = false;
};

struct IncrementalPressureSensorDiagnostics
{
    bool enabled = false;
    bool present = false;
    double timeOffsetSec = std::numeric_limits<double>::quiet_NaN();
    bool weightAvailable = false;
    double variance = std::numeric_limits<double>::quiet_NaN();
    double information = std::numeric_limits<double>::quiet_NaN();
    bool nisAvailable = false;
    double nis = std::numeric_limits<double>::quiet_NaN();
};

struct IncrementalSensorDiagnostics
{
    IncrementalStereoSensorDiagnostics stereo;
    IncrementalImuSensorDiagnostics imu;
    IncrementalDvlSensorDiagnostics dvl;
    IncrementalPressureSensorDiagnostics pressure;
};

struct IncrementalGraphDiagnostics
{
    std::size_t submittedFactors = 0;
    std::size_t submittedValues = 0;
    std::size_t totalFactors = 0;
    std::size_t totalValues = 0;
    std::size_t totalLandmarks = 0;
    std::size_t totalPoseStates = 0;
    std::size_t archivedPoseCount = 0;
    std::size_t variablesRelinearized = 0;
    std::size_t variablesReeliminated = 0;
    std::size_t factorsRecalculated = 0;
    std::size_t cliques = 0;
    std::size_t treeNnz = 0;
    bool batchReorderTriggered = false;
    double errorBefore = std::numeric_limits<double>::quiet_NaN();
    double errorAfter = std::numeric_limits<double>::quiet_NaN();
    double candidateCopyMs = 0.0;
    double isamUpdateMs = 0.0;
    double estimateCalculationMs = 0.0;
    double transactionMs = 0.0;
};

// Buffers raw acquisition-time IMU samples until AQUA accepts a new keyframe.
// The left boundary is retained after commit so adjacent intervals share one
// physical sample without duplicating it in the next preintegration.
class DynamicKeyframeImuBuffer
{
public:
    void append(const std::vector<ImuSample>& samples)
    {
        for (const auto& sample : samples) {
            const auto finiteVector = [](const std::array<double, 3>& values) {
                return std::all_of(values.begin(), values.end(),
                                   [](double value) { return std::isfinite(value); });
            };
            if (std::isfinite(sample.timestampSec) &&
                finiteVector(sample.acceleration) &&
                finiteVector(sample.angularVelocity) &&
                (buffer_.empty() ||
                 sample.timestampSec > buffer_.back().timestampSec))
                buffer_.push_back(sample);
        }
    }

    std::vector<ImuSample> interval(double previousKeyframeTime,
                                    double currentKeyframeTime) const
    {
        std::vector<ImuSample> result;
        if (!(std::isfinite(previousKeyframeTime) &&
              std::isfinite(currentKeyframeTime) &&
              currentKeyframeTime > previousKeyframeTime)) return result;
        auto first = std::lower_bound(
            buffer_.begin(), buffer_.end(), previousKeyframeTime,
            [](const ImuSample& sample, double timestamp) {
                return sample.timestampSec < timestamp;
            });
        if (first != buffer_.begin() &&
            (first == buffer_.end() ||
             first->timestampSec > previousKeyframeTime))
            --first;
        for (auto sample = first; sample != buffer_.end(); ++sample) {
            result.push_back(*sample);
            if (sample->timestampSec > currentKeyframeTime)
                break;
        }
        return result;
    }

    void commit(double currentKeyframeTime)
    {
        if (!std::isfinite(currentKeyframeTime) || buffer_.empty()) return;
        const auto firstAfter = std::upper_bound(
            buffer_.begin(), buffer_.end(), currentKeyframeTime,
            [](double timestamp, const ImuSample& sample) {
                return timestamp < sample.timestampSec;
            });
        const auto retained = firstAfter == buffer_.begin()
            ? firstAfter
            : std::prev(firstAfter);
        buffer_.erase(buffer_.begin(), retained);
    }

    std::vector<ImuSample> pendingInterval(double currentKeyframeTime) const
    {
        if (!hasAcceptedKeyframe_) return {};
        auto pending = interval(lastAcceptedKeyframeTime_, currentKeyframeTime);
        if (pending.size() < 2U ||
            pending.front().timestampSec > lastAcceptedKeyframeTime_ ||
            pending.back().timestampSec < currentKeyframeTime)
            return {};
        return pending;
    }

    bool hasPhysicalBrackets(double currentKeyframeTime) const
    {
        if (!hasAcceptedKeyframe_)
            return false;
        const auto pending = interval(lastAcceptedKeyframeTime_,
                                      currentKeyframeTime);
        return pending.size() >= 2U &&
               pending.front().timestampSec <= lastAcceptedKeyframeTime_ &&
               pending.back().timestampSec >= currentKeyframeTime;
    }

    bool acceptKeyframe(double currentKeyframeTime)
    {
        if (!std::isfinite(currentKeyframeTime) ||
            (hasAcceptedKeyframe_ &&
             currentKeyframeTime <= lastAcceptedKeyframeTime_))
            return false;
        commit(currentKeyframeTime);
        lastAcceptedKeyframeTime_ = currentKeyframeTime;
        hasAcceptedKeyframe_ = true;
        return true;
    }

    void clear()
    {
        buffer_.clear();
        lastAcceptedKeyframeTime_ = 0.0;
        hasAcceptedKeyframe_ = false;
    }
    bool hasAcceptedKeyframe() const { return hasAcceptedKeyframe_; }
    std::size_t size() const { return buffer_.size(); }
    double firstTimestamp() const
    {
        return buffer_.empty() ? std::numeric_limits<double>::quiet_NaN()
                               : buffer_.front().timestampSec;
    }
    double lastTimestamp() const
    {
        return buffer_.empty() ? std::numeric_limits<double>::quiet_NaN()
                               : buffer_.back().timestampSec;
    }
    bool canAcceptKeyframe(double currentKeyframeTime) const
    {
        return std::isfinite(currentKeyframeTime) &&
               (!hasAcceptedKeyframe_ ||
                currentKeyframeTime > lastAcceptedKeyframeTime_);
    }
    double lastAcceptedKeyframeTime() const
    {
        return lastAcceptedKeyframeTime_;
    }
    const std::vector<ImuSample>& samples() const { return buffer_; }

private:
    std::vector<ImuSample> buffer_;
    double lastAcceptedKeyframeTime_ = 0.0;
    bool hasAcceptedKeyframe_ = false;
};

struct BackendFrameInput
{
    double timestampSec = 0.0;
    cv::Mat frontEndCameraFromMap;
    CalibrationIdentity calibration;
    std::vector<StereoTrackObservation> stereo;
    std::vector<MonocularTrackObservation> monocularLandmarks;
    std::vector<ImuSample> imu;
    std::vector<DvlMeasurement> dvl;
    bool hasPressure = false;
    PressureMeasurement pressure;
    std::array<double, 9> stereoCovariance{1.0, 0.0, 0.0,
                                            0.0, 1.0, 0.0,
                                            0.0, 0.0, 1.0};
    bool hasImuCovariance = false;
    std::array<double, 9> imuAccelerometerCovariance{1.0, 0.0, 0.0,
                                                      0.0, 1.0, 0.0,
                                                      0.0, 0.0, 1.0};
    std::array<double, 9> imuGyroscopeCovariance{1.0, 0.0, 0.0,
                                                  0.0, 1.0, 0.0,
                                                  0.0, 0.0, 1.0};
    std::array<double, 9> imuAccelerometerBiasCovariance{1.0, 0.0, 0.0,
                                                          0.0, 1.0, 0.0,
                                                          0.0, 0.0, 1.0};
    std::array<double, 9> imuGyroscopeBiasCovariance{1.0, 0.0, 0.0,
                                                      0.0, 1.0, 0.0,
                                                      0.0, 0.0, 1.0};
};

struct BackendFrameResult
{
    bool accepted = false;
    bool incrementalCommitted = false;
    std::uint64_t mapVersion = 0;
    cv::Mat bodyPose;
    std::vector<std::uint64_t> optimizedPoseInlierTrackIds;
    std::vector<std::uint64_t> frontEndPoseInlierTrackIds;
    std::string diagnostic;
    IncrementalFactorDiagnostics factorDiagnostics;
    IncrementalSensorDiagnostics sensorDiagnostics;
    IncrementalGraphDiagnostics graphDiagnostics;
};

class AquaBackendAdapter
{
public:
    static BackendFrameInput fromFrame(const Frame& frame,
                                       const RuntimeProfile& profile);
    static bool applyDvlMeasurement(BackendFrameInput& input,
                                    const IMU::GyroDvlPoint& measurement);
    static bool applyDynamicImuCovariance(BackendFrameInput& input,
                                          const IMU::Calib& calibration,
                                          double nominalFrequencyHz);
    static std::string formatDynamicSensorDiagnostics(
        double timestampSec, const BackendFrameResult& result);
    static bool applyFrameResult(const BackendFrameResult& result,
                                 Frame& frame);
    static bool canPreserveFrontEndPose(const BackendFrameResult& result,
                                        const Frame& frame);
    static int persistentStereoSupport(const BackendFrameInput& input);
    static int trackingSupportAfterPoseCorrection(
        bool committedKeyframe, bool correctionAccepted,
        const BackendFrameResult& result, int persistentStereoSupport);
};

}  // namespace ORB_SLAM3

#endif  // AQUA_BACKEND_ADAPTER_H
