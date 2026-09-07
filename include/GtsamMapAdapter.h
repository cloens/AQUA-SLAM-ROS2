#ifndef AQUA_GTSAM_MAP_ADAPTER_H
#define AQUA_GTSAM_MAP_ADAPTER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
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

struct CommittedKeyframeWatermark
{
    std::uint64_t keyframeId = 0;
    double timestampSec = 0.0;
    std::uint64_t mapVersion = 0;
};

struct FactorFamilyDiagnostics
{
    std::uint64_t attempted = 0;
    std::uint64_t accepted = 0;
    std::uint64_t rejected = 0;
    std::uint64_t deferred = 0;
    std::map<std::string, std::uint64_t> reasonCounts;
};

struct BackendMapDiagnostics
{
    CommittedKeyframeWatermark capturedWatermark;
    std::uint64_t sourceMapId = 0;
    std::uint64_t sourceTopologySignature = 0;
    bool dvlEnabled = false;
    bool pressureEnabled = false;
    FactorFamilyDiagnostics stereo;
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
    Loop,
    Merge
};

struct BackendGraphEdge
{
    std::uint64_t fromKeyframeId = 0;
    std::uint64_t toKeyframeId = 0;
    BackendGraphEdgeKind kind = BackendGraphEdgeKind::Covisibility;
    std::array<float, 16> relativePose{};
    double sigma = 0.1;
    std::array<double, 36> covariance{};
    bool hasCovariance = false;
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

struct BackendLandmarkReplacement
{
    std::uint64_t replacedId = 0;
    std::uint64_t retainedId = 0;
};

struct BackendMapResult
{
    bool accepted = false;
    std::string diagnostic;
    std::uint64_t sourceVersion = 0;
    std::uint64_t sourceMapId = 0;
    std::uint64_t sourceTopologySignature = 0;
    std::vector<BackendKeyframeState> keyframes;
    std::vector<BackendLandmarkState> landmarks;
    std::vector<BackendLandmarkReplacement> landmarkReplacements;
    std::vector<BackendStereoObservation> acceptedStereoObservations;
    BackendMapDiagnostics diagnostics;
};

struct BackendMapResultValidation
{
    bool accepted = false;
    double maximumPoseTranslation = 0.0;
    double maximumPoseRotation = 0.0;
    double maximumLandmarkTranslation = 0.0;
    double landmarkTranslationP95 = 0.0;
};

struct BackendMapSourceIdentity
{
    std::uint64_t mapId = 0;
    std::uint64_t version = 0;
    std::uint64_t topologySignature = 0;
};

struct BackendMergeSnapshot
{
    BackendMapSourceIdentity currentSource;
    BackendMapSourceIdentity candidateSource;
    std::uint64_t currentKeyframeId = 0;
    std::uint64_t candidateKeyframeId = 0;
    std::array<float, 16> candidateFromCurrent{};
    std::vector<std::uint64_t> currentKeyframeIds;
    std::vector<std::uint64_t> currentLandmarkIds;
    std::vector<std::uint64_t> candidateKeyframeIds;
    std::vector<std::uint64_t> candidateLandmarkIds;
    BackendMapSnapshot graph;
};

struct BackendMergeResult
{
    bool accepted = false;
    BackendMapSourceIdentity currentSource;
    BackendMapSourceIdentity candidateSource;
    std::uint64_t currentKeyframeId = 0;
    std::uint64_t candidateKeyframeId = 0;
    std::array<float, 16> candidateFromCurrent{};
    std::vector<std::uint64_t> currentKeyframeIds;
    std::vector<std::uint64_t> currentLandmarkIds;
    std::vector<std::uint64_t> candidateKeyframeIds;
    std::vector<std::uint64_t> candidateLandmarkIds;
    std::vector<BackendKeyframeState> keyframes;
    std::vector<BackendLandmarkState> landmarks;
    BackendMapDiagnostics diagnostics;
};

struct BackendLoopPointCorrespondence
{
    std::array<double, 3> currentPoint{};
    std::array<double, 3> candidatePoint{};
    std::uint64_t currentLandmarkId = 0;
    std::uint64_t candidateLandmarkId = 0;
};

struct BackendLoopCandidateResult
{
    bool accepted = false;
    std::array<float, 16> currentFromCandidate{};
    std::size_t inliers = 0;
    double rmsErrorMeters = 0.0;
    std::string diagnostic;
    std::vector<BackendLandmarkReplacement> landmarkReplacements;
};

struct BackendMergeHypothesisUpdate
{
    bool reset = false;
    std::string resetReason;
    std::size_t added = 0;
    std::size_t total = 0;
};

class BackendMergeHypothesis
{
public:
    explicit BackendMergeHypothesis(std::size_t maximumCorrespondences = 256U,
                                    std::size_t maximumConsecutiveMisses = 2U);

    BackendMergeHypothesisUpdate observe(
        std::uint64_t candidateMapId,
        std::uint64_t currentKeyframeId,
        std::uint64_t candidateKeyframeId,
        const std::vector<BackendLoopPointCorrespondence>& correspondences);
    bool recordAccepted(const BackendLoopCandidateResult& result,
                        double maximumTranslationMeters,
                        double maximumRotationRadians,
                        double* translationDeltaMeters,
                        double* rotationDeltaRadians,
                        std::string* resetReason);
    bool recordMiss(std::string* resetReason);
    void reset();

    bool active() const;
    std::uint64_t candidateMapId() const;
    std::uint64_t currentKeyframeId() const;
    std::uint64_t candidateKeyframeId() const;
    std::size_t confirmations() const;
    std::size_t consecutiveMisses() const;
    const std::vector<BackendLoopPointCorrespondence>& correspondences() const;
    const BackendLoopCandidateResult& aggregateResult() const;

private:
    std::size_t maximumCorrespondences_;
    std::size_t maximumConsecutiveMisses_;
    std::optional<std::uint64_t> candidateMapId_;
    std::uint64_t currentKeyframeId_ = 0;
    std::uint64_t candidateKeyframeId_ = 0;
    std::size_t confirmations_ = 0;
    std::size_t consecutiveMisses_ = 0;
    std::vector<BackendLoopPointCorrespondence> correspondences_;
    BackendLoopCandidateResult aggregateResult_;
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
                                       const std::array<float, 16>& currentFromCandidate,
                                       double sigma);
    static bool appendAcceptedMergeEdge(
        BackendMapSnapshot& snapshot,
        std::uint64_t currentKeyframeId,
        std::uint64_t candidateKeyframeId,
        const std::array<float, 16>& currentFromCandidate,
        const std::array<double, 36>& covariance);
    static bool transformSnapshot(
        const BackendMapSnapshot& source,
        const std::array<float, 16>& targetFromSource,
        BackendMapSnapshot* transformed);
    static bool prepareMerge(
        const BackendMapSnapshot& current,
        const BackendMapSnapshot& candidate,
        std::uint64_t currentKeyframeId,
        std::uint64_t candidateKeyframeId,
        const std::array<float, 16>& currentFromCandidate,
        const std::array<double, 36>& covariance,
        BackendMergeSnapshot* transaction);
    static std::optional<std::array<double, 36>> mergeCovariance(
        std::size_t inliers, std::size_t correspondenceCount,
        double rmsErrorMeters);
    static bool consistentMergeTransform(
        const std::array<float, 16>& firstCurrentFromCandidate,
        const std::array<float, 16>& secondCurrentFromCandidate,
        double maximumTranslationMeters,
        double maximumRotationRadians);
    static BackendMapResultValidation validateMergeResult(
        const BackendMergeSnapshot& snapshot,
        const BackendMergeResult& result);
    static bool commitMerge(const BackendMergeResult& result,
                            Map& currentMap, Map& candidateMap,
                            std::string* rejectionReason = nullptr);
    static BackendMapResultValidation validateMapResult(
        const BackendMapSnapshot& snapshot,
        const BackendMapResult& result);
    static BackendMapResultValidation validateLocalResult(
        const BackendMapSnapshot& snapshot,
        const BackendMapResult& result);
    static bool commit(const BackendMapResult& result, Map& map,
                       std::uint64_t expectedVersion);
};
}  // namespace ORB_SLAM3

#endif  // AQUA_GTSAM_MAP_ADAPTER_H
