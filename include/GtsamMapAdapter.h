#ifndef AQUA_GTSAM_MAP_ADAPTER_H
#define AQUA_GTSAM_MAP_ADAPTER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ORB_SLAM3
{
class Map;

struct BackendKeyframeState
{
    std::uint64_t id = 0;
    double timestampSec = 0.0;
    // Fixed-scale world/map-from-camera pose.
    std::array<float, 16> pose{};
    std::array<float, 3> velocity{};
    std::array<float, 3> accelerometerBias{};
    std::array<float, 3> gyroscopeBias{};
};

struct BackendLandmarkState
{
    std::uint64_t id = 0;
    std::array<float, 3> position{};
    std::size_t observationDegree = 0;
    double minimumDepthMeters = std::numeric_limits<double>::quiet_NaN();
    double maximumDepthMeters = std::numeric_limits<double>::quiet_NaN();
};

struct BackendStereoCalibration
{
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double baseline = 0.0;
};

struct BackendStereoObservation
{
    std::uint64_t keyframeId = 0;
    std::uint64_t landmarkId = 0;
    double uLeft = 0.0;
    double uRight = 0.0;
    double v = 0.0;
    double sigmaPx = 1.0;
    int octave = -1;
    double octaveVariancePx2 = std::numeric_limits<double>::quiet_NaN();
    double quality = std::numeric_limits<double>::quiet_NaN();
    double stereoSkewPx = 0.0;
};

struct ScalarDistribution
{
    static constexpr std::size_t kBinCount = 32U;

    std::uint64_t count = 0;
    double sumSquares = 0.0;
    double maximum = 0.0;
    double rms = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    std::array<std::uint64_t, kBinCount> bins{};
};

struct BackendMapDiagnostics
{
    ScalarDistribution stereoResidual;
    ScalarDistribution stereoNis;
    ScalarDistribution stereoDisparity;
    ScalarDistribution stereoSkew;
    ScalarDistribution stereoQuality;
    ScalarDistribution landmarkObservationDegree;
};

enum class BackendGraphEdgeKind
{
    SpanningTree,
    Covisibility,
    Loop
};

struct BackendGraphEdge
{
    std::uint64_t fromKeyframeId = 0;
    std::uint64_t toKeyframeId = 0;
    BackendGraphEdgeKind kind = BackendGraphEdgeKind::Covisibility;
    std::array<float, 16> relativePose{};
    double sigma = 0.1;
};

struct BackendMapSnapshot
{
    std::uint64_t version = 0;
    std::uint64_t mapId = 0;
    std::uint64_t topologySignature = 0;
    BackendStereoCalibration calibration;
    std::vector<BackendKeyframeState> keyframes;
    std::vector<BackendLandmarkState> landmarks;
    std::vector<BackendStereoObservation> observations;
    std::vector<BackendGraphEdge> graphEdges;
};

struct CommittedKeyframeWatermark
{
    std::uint64_t keyframeId = 0;
    double timestampSec = 0.0;
    std::uint64_t mapVersion = 0;
};

struct BackendMapResult
{
    bool accepted = false;
    std::uint64_t sourceVersion = 0;
    std::uint64_t sourceMapId = 0;
    std::uint64_t sourceTopologySignature = 0;
    std::vector<BackendKeyframeState> keyframes;
    std::vector<BackendLandmarkState> landmarks;
    BackendMapDiagnostics diagnostics;
};

struct BackendMapResultValidation
{
    bool accepted = false;
    double maximumPoseTranslation = 0.0;
    double maximumLandmarkTranslation = 0.0;
};

struct BackendLoopPointCorrespondence
{
    std::array<double, 3> currentPoint{};
    std::array<double, 3> candidatePoint{};
};

struct BackendLoopCandidateResult
{
    bool accepted = false;
    std::array<float, 16> currentFromCandidate{};
    std::size_t inliers = 0;
    double rmsErrorMeters = 0.0;
    std::string diagnostic;
};

class GtsamMapAdapter
{
public:
    static BackendMapSnapshot snapshot(Map& map, std::uint64_t version);
    static bool projectCommitted(
        const BackendMapSnapshot& source,
        const CommittedKeyframeWatermark& watermark,
        BackendMapSnapshot* projected,
        std::string* rejectionReason);
    static bool appendAcceptedLoopEdge(BackendMapSnapshot& snapshot,
                                       std::uint64_t fromKeyframeId,
                                       std::uint64_t toKeyframeId,
                                       double sigma);
    static BackendMapResultValidation validateLocalResult(
        const BackendMapSnapshot& snapshot,
        const BackendMapResult& result,
        double maximumPoseTranslation,
        double maximumLandmarkTranslation);
    static bool commit(const BackendMapResult& result, Map& map,
                       std::uint64_t expectedVersion);
};
}  // namespace ORB_SLAM3

#endif  // AQUA_GTSAM_MAP_ADAPTER_H
