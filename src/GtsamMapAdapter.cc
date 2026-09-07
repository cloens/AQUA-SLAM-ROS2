#include "GtsamMapAdapter.h"

#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace ORB_SLAM3
{
namespace
{
bool finitePose(const std::array<float, 16>& pose)
{
    for (float value : pose)
        if (!std::isfinite(static_cast<double>(value)))
            return false;
    return std::abs(pose[15] - 1.0F) < 1e-4F;
}

bool copyPose(const cv::Mat& source, std::array<float, 16>& destination)
{
    if (source.rows != 4 || source.cols != 4 || source.type() != CV_32F ||
        !cv::checkRange(source))
        return false;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            destination[static_cast<std::size_t>(row * 4 + col)] =
                source.at<float>(row, col);
    return finitePose(destination);
}

bool projectionConsistent(const BackendMapSnapshot& snapshot,
                          const BackendMapResult& result)
{
    if (result.acceptedStereoObservations.empty() ||
        !(std::isfinite(snapshot.calibration.fx) && snapshot.calibration.fx > 0.0) ||
        !(std::isfinite(snapshot.calibration.fy) && snapshot.calibration.fy > 0.0) ||
        !(std::isfinite(snapshot.calibration.baseline) && snapshot.calibration.baseline > 0.0))
        return true;
    std::unordered_map<std::uint64_t, const BackendKeyframeState*> keyframes;
    std::unordered_map<std::uint64_t, const BackendLandmarkState*> landmarks;
    for (const auto& state : result.keyframes)
        keyframes.emplace(state.id, &state);
    for (const auto& state : result.landmarks)
        landmarks.emplace(state.id, &state);

    for (const auto& observation : result.acceptedStereoObservations) {
        const auto keyframe = keyframes.find(observation.keyframeId);
        const auto landmark = landmarks.find(observation.landmarkId);
        if (keyframe == keyframes.end() || landmark == landmarks.end())
            continue;
        const auto& pose = keyframe->second->pose;
        const auto& point = landmark->second->position;
        // Snapshots store AQUA's map-from-camera (Twc) pose. Projection uses
        // its inverse, i.e. Rcw * (Pw - twc).
        const double dx = point[0] - pose[3];
        const double dy = point[1] - pose[7];
        const double dz = point[2] - pose[11];
        const double x = pose[0] * dx + pose[4] * dy + pose[8] * dz;
        const double y = pose[1] * dx + pose[5] * dy + pose[9] * dz;
        const double z = pose[2] * dx + pose[6] * dy + pose[10] * dz;
        if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && z > 0.0))
            return false;
        const double uLeft = snapshot.calibration.fx * x / z + snapshot.calibration.cx;
        const double uRight = uLeft - snapshot.calibration.fx * snapshot.calibration.baseline / z;
        const double v = snapshot.calibration.fy * y / z + snapshot.calibration.cy;
        const double duLeft = uLeft - observation.uLeft;
        const double duRight = uRight - observation.uRight;
        const double dv = v - observation.v;
        const double error = std::sqrt(duLeft * duLeft + duRight * duRight + dv * dv);
        if (!std::isfinite(error))
            return false;
    }
    return true;
}

cv::Mat toPoseMatrix(const std::array<float, 16>& source);

bool mergeTransformDelta(
    const std::array<float, 16>& firstCurrentFromCandidate,
    const std::array<float, 16>& secondCurrentFromCandidate,
    double* translationMeters, double* rotationRadians)
{
    if (!translationMeters || !rotationRadians ||
        !finitePose(firstCurrentFromCandidate) ||
        !finitePose(secondCurrentFromCandidate))
        return false;
    const cv::Mat delta = toPoseMatrix(firstCurrentFromCandidate).inv() *
                          toPoseMatrix(secondCurrentFromCandidate);
    if (!cv::checkRange(delta))
        return false;
    *translationMeters = cv::norm(delta.rowRange(0, 3).col(3));
    const double trace = delta.at<float>(0, 0) + delta.at<float>(1, 1) +
                         delta.at<float>(2, 2);
    *rotationRadians = std::acos(std::clamp(
        (trace - 1.0) * 0.5, -1.0, 1.0));
    return std::isfinite(*translationMeters) &&
           std::isfinite(*rotationRadians);
}

bool validCovariance(const std::array<double, 36>& values)
{
    cv::Mat covariance(6, 6, CV_64F);
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            const double value = values[static_cast<std::size_t>(row * 6 + col)];
            if (!std::isfinite(value))
                return false;
            covariance.at<double>(row, col) = value;
            const double reverse = values[static_cast<std::size_t>(col * 6 + row)];
            if (!std::isfinite(reverse) || std::abs(value - reverse) > 1e-10)
                return false;
        }
    }
    cv::Mat eigenvalues;
    if (!cv::eigen(covariance, eigenvalues) || eigenvalues.rows != 6)
        return false;
    for (int index = 0; index < eigenvalues.rows; ++index)
        if (!(eigenvalues.at<double>(index) > 0.0))
            return false;
    return true;
}

std::array<float, 3> copyVector3(const cv::Mat& source)
{
    if (source.rows != 3 || source.cols != 1 || !cv::checkRange(source))
        return {};
    cv::Mat value;
    source.convertTo(value, CV_32F);
    return {value.at<float>(0), value.at<float>(1), value.at<float>(2)};
}

cv::Mat toPoseMatrix(const std::array<float, 16>& source)
{
    cv::Mat pose(4, 4, CV_32F);
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            pose.at<float>(row, col) =
                source[static_cast<std::size_t>(row * 4 + col)];
    return pose;
}

std::uint64_t topologySignature(
    const std::vector<KeyFrame*>& keyframes,
    const std::vector<MapPoint*>& landmarks)
{
    std::vector<std::uint64_t> keyframeIds;
    std::vector<std::uint64_t> landmarkIds;
    for (KeyFrame* keyframe : keyframes)
        if (keyframe && !keyframe->isBad())
            keyframeIds.push_back(keyframe->mnId);
    for (MapPoint* landmark : landmarks)
        if (landmark && !landmark->isBad())
            landmarkIds.push_back(landmark->mnId);
    std::sort(keyframeIds.begin(), keyframeIds.end());
    std::sort(landmarkIds.begin(), landmarkIds.end());
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](std::uint64_t value) {
        for (unsigned int byte = 0; byte < 8U; ++byte) {
            hash ^= (value >> (byte * 8U)) & 0xffU;
            hash *= 1099511628211ULL;
        }
    };
    mix(0x4b45594652414d45ULL);
    for (std::uint64_t id : keyframeIds)
        mix(id);
    mix(0x4c414e444d41524bULL);
    for (std::uint64_t id : landmarkIds)
        mix(id);
    return hash;
}
}  // namespace

BackendMergeHypothesis::BackendMergeHypothesis(
    std::size_t maximumCorrespondences,
    std::size_t maximumConsecutiveMisses)
    : maximumCorrespondences_(std::max<std::size_t>(1U, maximumCorrespondences)),
      maximumConsecutiveMisses_(
          std::max<std::size_t>(1U, maximumConsecutiveMisses))
{
}

BackendMergeHypothesisUpdate BackendMergeHypothesis::observe(
    std::uint64_t candidateMapId,
    std::uint64_t currentKeyframeId,
    std::uint64_t candidateKeyframeId,
    const std::vector<BackendLoopPointCorrespondence>& correspondences)
{
    BackendMergeHypothesisUpdate update;
    if (candidateMapId_ && *candidateMapId_ != candidateMapId) {
        reset();
        update.reset = true;
        update.resetReason = "candidate_map_changed";
    }
    candidateMapId_ = candidateMapId;
    currentKeyframeId_ = currentKeyframeId;
    candidateKeyframeId_ = candidateKeyframeId;

    for (const auto& correspondence : correspondences) {
        const auto duplicate = std::find_if(
            correspondences_.begin(), correspondences_.end(),
            [&correspondence](const BackendLoopPointCorrespondence& existing) {
                return existing.currentLandmarkId ==
                           correspondence.currentLandmarkId &&
                       existing.candidateLandmarkId ==
                           correspondence.candidateLandmarkId;
            });
        if (duplicate != correspondences_.end()) {
            *duplicate = correspondence;
            continue;
        }
        correspondences_.push_back(correspondence);
        ++update.added;
        if (correspondences_.size() > maximumCorrespondences_)
            correspondences_.erase(correspondences_.begin());
    }
    update.total = correspondences_.size();
    return update;
}

bool BackendMergeHypothesis::recordAccepted(
    const BackendLoopCandidateResult& result,
    double maximumTranslationMeters,
    double maximumRotationRadians,
    double* translationDeltaMeters,
    double* rotationDeltaRadians,
    std::string* resetReason)
{
    if (resetReason)
        resetReason->clear();
    if (!active() || !result.accepted)
        return false;

    double translationDelta = 0.0;
    double rotationDelta = 0.0;
    if (confirmations_ > 0U &&
        (!mergeTransformDelta(aggregateResult_.currentFromCandidate,
                              result.currentFromCandidate,
                              &translationDelta, &rotationDelta) ||
         translationDelta > maximumTranslationMeters ||
         rotationDelta > maximumRotationRadians)) {
        reset();
        if (resetReason)
            *resetReason = "transform_inconsistent";
        if (translationDeltaMeters)
            *translationDeltaMeters = translationDelta;
        if (rotationDeltaRadians)
            *rotationDeltaRadians = rotationDelta;
        return false;
    }
    aggregateResult_ = result;
    ++confirmations_;
    consecutiveMisses_ = 0;
    if (translationDeltaMeters)
        *translationDeltaMeters = translationDelta;
    if (rotationDeltaRadians)
        *rotationDeltaRadians = rotationDelta;
    return true;
}

bool BackendMergeHypothesis::recordMiss(std::string* resetReason)
{
    if (resetReason)
        resetReason->clear();
    if (!active())
        return false;
    ++consecutiveMisses_;
    if (consecutiveMisses_ < maximumConsecutiveMisses_)
        return false;
    reset();
    if (resetReason)
        *resetReason = "consecutive_miss_limit";
    return true;
}

void BackendMergeHypothesis::reset()
{
    candidateMapId_.reset();
    currentKeyframeId_ = 0;
    candidateKeyframeId_ = 0;
    confirmations_ = 0;
    consecutiveMisses_ = 0;
    correspondences_.clear();
    aggregateResult_ = {};
}

bool BackendMergeHypothesis::active() const
{
    return candidateMapId_.has_value();
}

std::uint64_t BackendMergeHypothesis::candidateMapId() const
{
    return candidateMapId_.value_or(0U);
}

std::uint64_t BackendMergeHypothesis::currentKeyframeId() const
{
    return currentKeyframeId_;
}

std::uint64_t BackendMergeHypothesis::candidateKeyframeId() const
{
    return candidateKeyframeId_;
}

std::size_t BackendMergeHypothesis::confirmations() const
{
    return confirmations_;
}

std::size_t BackendMergeHypothesis::consecutiveMisses() const
{
    return consecutiveMisses_;
}

const std::vector<BackendLoopPointCorrespondence>&
BackendMergeHypothesis::correspondences() const
{
    return correspondences_;
}

const BackendLoopCandidateResult& BackendMergeHypothesis::aggregateResult() const
{
    return aggregateResult_;
}

bool GtsamMapAdapter::projectCommitted(
    const BackendMapSnapshot& source,
    const CommittedKeyframeWatermark& watermark,
    BackendMapSnapshot* projected,
    std::string* rejectionReason)
{
    const auto reject = [rejectionReason](const char* reason) {
        if (rejectionReason)
            *rejectionReason = reason;
        return false;
    };
    if (!projected || !std::isfinite(watermark.timestampSec))
        return reject("invalid committed-keyframe watermark");
    if (source.version != watermark.mapVersion)
        return reject("snapshot and committed-keyframe watermark versions differ");

    BackendMapSnapshot result = source;
    std::sort(result.keyframes.begin(), result.keyframes.end(),
              [](const BackendKeyframeState& left,
                 const BackendKeyframeState& right) {
                  if (left.timestampSec != right.timestampSec)
                      return left.timestampSec < right.timestampSec;
                  return left.id < right.id;
              });
    if (result.keyframes.empty())
        return reject("map snapshot has no keyframes");
    for (std::size_t index = 0; index < result.keyframes.size(); ++index) {
        const auto& keyframe = result.keyframes[index];
        if (!std::isfinite(keyframe.timestampSec) ||
            (index > 0U && keyframe.timestampSec <=
                              result.keyframes[index - 1U].timestampSec))
            return reject("map snapshot keyframe timestamps are not strictly increasing");
    }

    const auto firstFuture = std::upper_bound(
        result.keyframes.begin(), result.keyframes.end(), watermark.timestampSec,
        [](double timestamp, const BackendKeyframeState& keyframe) {
            return timestamp < keyframe.timestampSec;
        });
    result.keyframes.erase(firstFuture, result.keyframes.end());
    if (result.keyframes.empty() ||
        result.keyframes.back().id != watermark.keyframeId ||
        result.keyframes.back().timestampSec != watermark.timestampSec)
        return reject("snapshot does not end at committed-keyframe watermark");

    std::unordered_set<std::uint64_t> keyframeIds;
    for (const auto& keyframe : result.keyframes)
        keyframeIds.insert(keyframe.id);
    std::unordered_set<std::uint64_t> sourceLandmarkIds;
    for (const auto& landmark : source.landmarks)
        sourceLandmarkIds.insert(landmark.id);

    result.observations.clear();
    std::unordered_set<std::uint64_t> retainedLandmarkIds;
    for (const auto& observation : source.observations) {
        if (keyframeIds.count(observation.keyframeId) == 0U ||
            sourceLandmarkIds.count(observation.landmarkId) == 0U)
            continue;
        result.observations.push_back(observation);
        retainedLandmarkIds.insert(observation.landmarkId);
    }
    result.landmarks.clear();
    for (const auto& landmark : source.landmarks)
        if (retainedLandmarkIds.count(landmark.id) != 0U)
            result.landmarks.push_back(landmark);

    result.graphEdges.clear();
    for (const auto& edge : source.graphEdges)
        if (keyframeIds.count(edge.fromKeyframeId) != 0U &&
            keyframeIds.count(edge.toKeyframeId) != 0U)
            result.graphEdges.push_back(edge);

    *projected = std::move(result);
    if (rejectionReason)
        rejectionReason->clear();
    return true;
}

BackendMapSnapshot GtsamMapAdapter::snapshot(Map& map,
                                             std::uint64_t version)
{
    std::shared_lock<std::shared_timed_mutex> mapLock(map.mMutexMapUpdate);
    BackendMapSnapshot result;
    result.version = version;
    result.mapId = map.GetId();
    const std::vector<KeyFrame*> keyframes = map.GetAllKeyFrames();
    const std::vector<MapPoint*> landmarks = map.GetAllMapPoints();
    result.topologySignature = topologySignature(keyframes, landmarks);
    std::unordered_map<std::uint64_t, KeyFrame*> validKeyframes;
    std::unordered_map<std::uint64_t, MapPoint*> validLandmarks;
    for (KeyFrame* keyframe : keyframes) {
        if (!keyframe || keyframe->isBad())
            continue;
        BackendKeyframeState state;
        state.id = keyframe->mnId;
        state.timestampSec = keyframe->mTimeStamp;
        state.velocity = copyVector3(keyframe->GetVelocityOld());
        const IMU::Bias bias = keyframe->GetImuBias();
        state.accelerometerBias = {static_cast<float>(bias.bax),
                                   static_cast<float>(bias.bay),
                                   static_cast<float>(bias.baz)};
        state.gyroscopeBias = {static_cast<float>(bias.bwx),
                               static_cast<float>(bias.bwy),
                               static_cast<float>(bias.bwz)};
        if (copyPose(keyframe->GetPoseInverse(), state.pose)) {
            result.keyframes.push_back(state);
            validKeyframes.emplace(state.id, keyframe);
            if (result.keyframes.size() == 1U) {
                result.calibration.fx = keyframe->fx;
                result.calibration.fy = keyframe->fy;
                result.calibration.cx = keyframe->cx;
                result.calibration.cy = keyframe->cy;
                result.calibration.baseline = keyframe->mb;
            }
        }
    }
    for (MapPoint* landmark : landmarks) {
        if (!landmark || landmark->isBad())
            continue;
        const cv::Mat position = landmark->GetWorldPos();
        if (position.rows != 3 || position.cols != 1 ||
            position.type() != CV_32F || !cv::checkRange(position))
            continue;
        BackendLandmarkState state;
        state.id = landmark->mnId;
        for (int axis = 0; axis < 3; ++axis)
            state.position[static_cast<std::size_t>(axis)] =
                position.at<float>(axis);
        state.observationDegree = landmark->GetObservations().size();
        const double minimumDepth = landmark->GetMinDistanceInvariance();
        const double maximumDepth = landmark->GetMaxDistanceInvariance();
        if (std::isfinite(minimumDepth) && std::isfinite(maximumDepth) &&
            minimumDepth > 0.0 && maximumDepth >= minimumDepth) {
            state.minimumDepthMeters = minimumDepth;
            state.maximumDepthMeters = maximumDepth;
        }
        result.landmarks.push_back(state);
        validLandmarks.emplace(state.id, landmark);
    }
    std::vector<KeyFrame*> orderedKeyframes;
    orderedKeyframes.reserve(validKeyframes.size());
    for (const auto& entry : validKeyframes)
        orderedKeyframes.push_back(entry.second);
    std::sort(orderedKeyframes.begin(), orderedKeyframes.end(),
              [](const KeyFrame* left, const KeyFrame* right) {
                  if (left->mTimeStamp != right->mTimeStamp)
                      return left->mTimeStamp < right->mTimeStamp;
                  return left->mnId < right->mnId;
              });
    for (KeyFrame* keyframe : orderedKeyframes) {
        const std::vector<MapPoint*> matches = keyframe->GetMapPointMatches();
        const std::size_t count = std::min(
            {matches.size(), keyframe->mvKeysUn.size(), keyframe->mvuRight.size()});
        for (std::size_t index = 0; index < count; ++index) {
            MapPoint* landmark = matches[index];
            if (!landmark || validLandmarks.count(landmark->mnId) == 0U ||
                !(keyframe->mvuRight[index] >= 0.0F))
                continue;
            const cv::KeyPoint& keypoint = keyframe->mvKeysUn[index];
            if (!std::isfinite(static_cast<double>(keypoint.pt.x)) ||
                !std::isfinite(static_cast<double>(keypoint.pt.y)) ||
                !std::isfinite(static_cast<double>(keypoint.response)))
                continue;
            BackendStereoObservation observation;
            observation.keyframeId = keyframe->mnId;
            observation.landmarkId = landmark->mnId;
            observation.uLeft = keypoint.pt.x;
            observation.uRight = keyframe->mvuRight[index];
            observation.v = keypoint.pt.y;
            observation.octave = keypoint.octave;
            observation.quality = keypoint.response;
            // AQUA retains only the rectified right x-coordinate, so vertical skew is zero.
            observation.stereoSkewPx = 0.0;
            if (keypoint.octave >= 0 &&
                static_cast<std::size_t>(keypoint.octave) <
                    keyframe->mvLevelSigma2.size()) {
                observation.octaveVariancePx2 = keyframe->mvLevelSigma2[
                    static_cast<std::size_t>(keypoint.octave)];
                if (std::isfinite(observation.octaveVariancePx2) &&
                    observation.octaveVariancePx2 > 0.0)
                    observation.sigmaPx = std::sqrt(
                        observation.octaveVariancePx2);
            }
            result.observations.push_back(observation);
        }
    }

    std::set<std::tuple<std::uint64_t, std::uint64_t, BackendGraphEdgeKind>>
        insertedEdges;
    auto appendEdge = [&](KeyFrame* from, KeyFrame* to,
                          BackendGraphEdgeKind kind, double sigma) {
        if (!from || !to || from == to ||
            validKeyframes.count(from->mnId) == 0U ||
            validKeyframes.count(to->mnId) == 0U)
            return;
        const auto ordered = std::minmax(from->mnId, to->mnId);
        if (!insertedEdges.emplace(ordered.first, ordered.second, kind).second)
            return;
        BackendGraphEdge edge;
        edge.fromKeyframeId = from->mnId;
        edge.toKeyframeId = to->mnId;
        edge.kind = kind;
        edge.sigma = sigma;
        const cv::Mat relative = from->GetPose() * to->GetPoseInverse();
        if (copyPose(relative, edge.relativePose))
            result.graphEdges.push_back(edge);
    };
    for (const auto& entry : validKeyframes) {
        KeyFrame* keyframe = entry.second;
        appendEdge(keyframe, keyframe->GetParent(),
                   BackendGraphEdgeKind::SpanningTree, 0.05);
        for (KeyFrame* connected : keyframe->GetCovisiblesByWeight(100)) {
            const int weight = keyframe->GetWeight(connected);
            appendEdge(keyframe, connected, BackendGraphEdgeKind::Covisibility,
                       std::clamp(5.0 / static_cast<double>(weight), 0.02, 0.2));
        }
        for (KeyFrame* loop : keyframe->GetLoopEdges())
            appendEdge(keyframe, loop, BackendGraphEdgeKind::Loop, 0.03);
    }
    return result;
}

bool GtsamMapAdapter::appendAcceptedLoopEdge(
    BackendMapSnapshot& snapshot,
    std::uint64_t fromKeyframeId,
    std::uint64_t toKeyframeId,
    const std::array<float, 16>& currentFromCandidate,
    double sigma)
{
    if (fromKeyframeId == toKeyframeId ||
        !finitePose(currentFromCandidate) ||
        !(std::isfinite(sigma) && sigma > 0.0))
        return false;
    const BackendKeyframeState* from = nullptr;
    const BackendKeyframeState* to = nullptr;
    for (const auto& keyframe : snapshot.keyframes) {
        if (keyframe.id == fromKeyframeId)
            from = &keyframe;
        if (keyframe.id == toKeyframeId)
            to = &keyframe;
    }
    if (!from || !to || !finitePose(from->pose) || !finitePose(to->pose))
        return false;
    for (const auto& edge : snapshot.graphEdges) {
        if (edge.kind != BackendGraphEdgeKind::Loop)
            continue;
        const bool sameDirection =
            edge.fromKeyframeId == fromKeyframeId &&
            edge.toKeyframeId == toKeyframeId;
        const bool reverseDirection =
            edge.fromKeyframeId == toKeyframeId &&
            edge.toKeyframeId == fromKeyframeId;
        if (sameDirection || reverseDirection)
            return false;
    }
    const cv::Mat mapFromFrom = toPoseMatrix(from->pose);
    const cv::Mat mapFromTo = toPoseMatrix(to->pose);
    const cv::Mat verifiedCurrentFromCandidate =
        toPoseMatrix(currentFromCandidate);
    BackendGraphEdge edge;
    edge.fromKeyframeId = fromKeyframeId;
    edge.toKeyframeId = toKeyframeId;
    edge.kind = BackendGraphEdgeKind::Loop;
    edge.sigma = sigma;
    if (!copyPose(mapFromFrom.inv() * verifiedCurrentFromCandidate * mapFromTo,
                  edge.relativePose))
        return false;
    snapshot.graphEdges.push_back(edge);
    return true;
}

bool GtsamMapAdapter::appendAcceptedMergeEdge(
    BackendMapSnapshot& snapshot,
    std::uint64_t currentKeyframeId,
    std::uint64_t candidateKeyframeId,
    const std::array<float, 16>& currentFromCandidate,
    const std::array<double, 36>& covariance)
{
    if (currentKeyframeId == candidateKeyframeId ||
        !finitePose(currentFromCandidate) || !validCovariance(covariance))
        return false;
    const BackendKeyframeState* current = nullptr;
    const BackendKeyframeState* candidate = nullptr;
    for (const auto& keyframe : snapshot.keyframes) {
        if (keyframe.id == currentKeyframeId)
            current = &keyframe;
        if (keyframe.id == candidateKeyframeId)
            candidate = &keyframe;
    }
    if (!current || !candidate || !finitePose(current->pose) ||
        !finitePose(candidate->pose))
        return false;
    for (const auto& edge : snapshot.graphEdges) {
        if (edge.kind != BackendGraphEdgeKind::Merge)
            continue;
        const bool sameDirection =
            edge.fromKeyframeId == currentKeyframeId &&
            edge.toKeyframeId == candidateKeyframeId;
        const bool reverseDirection =
            edge.fromKeyframeId == candidateKeyframeId &&
            edge.toKeyframeId == currentKeyframeId;
        if (sameDirection || reverseDirection)
            return false;
    }

    BackendGraphEdge edge;
    edge.fromKeyframeId = currentKeyframeId;
    edge.toKeyframeId = candidateKeyframeId;
    edge.kind = BackendGraphEdgeKind::Merge;
    edge.covariance = covariance;
    edge.hasCovariance = true;
    const cv::Mat mapFromCurrent = toPoseMatrix(current->pose);
    const cv::Mat mapFromCandidate = toPoseMatrix(candidate->pose);
    const cv::Mat verifiedCurrentFromCandidate =
        toPoseMatrix(currentFromCandidate);
    if (!copyPose(mapFromCurrent.inv() * verifiedCurrentFromCandidate *
                      mapFromCandidate,
                  edge.relativePose))
        return false;
    snapshot.graphEdges.push_back(edge);
    return true;
}

bool GtsamMapAdapter::transformSnapshot(
    const BackendMapSnapshot& source,
    const std::array<float, 16>& targetFromSource,
    BackendMapSnapshot* transformed)
{
    if (!transformed || !finitePose(targetFromSource))
        return false;
    const cv::Mat transform = toPoseMatrix(targetFromSource);
    const cv::Mat rotation = transform.rowRange(0, 3).colRange(0, 3);
    BackendMapSnapshot result = source;
    for (auto& keyframe : result.keyframes) {
        if (!finitePose(keyframe.pose))
            return false;
        if (!copyPose(transform * toPoseMatrix(keyframe.pose), keyframe.pose))
            return false;
        const cv::Mat velocity = (cv::Mat_<float>(3, 1)
            << keyframe.velocity[0], keyframe.velocity[1], keyframe.velocity[2]);
        if (!cv::checkRange(velocity))
            return false;
        const cv::Mat rotated = rotation * velocity;
        keyframe.velocity = {rotated.at<float>(0), rotated.at<float>(1),
                             rotated.at<float>(2)};
    }
    for (auto& landmark : result.landmarks) {
        const cv::Mat homogeneous = (cv::Mat_<float>(4, 1)
            << landmark.position[0], landmark.position[1], landmark.position[2],
               1.0F);
        if (!cv::checkRange(homogeneous))
            return false;
        const cv::Mat point = transform * homogeneous;
        if (!cv::checkRange(point))
            return false;
        landmark.position = {point.at<float>(0), point.at<float>(1),
                             point.at<float>(2)};
    }
    *transformed = std::move(result);
    return true;
}

bool GtsamMapAdapter::prepareMerge(
    const BackendMapSnapshot& current,
    const BackendMapSnapshot& candidate,
    std::uint64_t currentKeyframeId,
    std::uint64_t candidateKeyframeId,
    const std::array<float, 16>& currentFromCandidate,
    const std::array<double, 36>& covariance,
    BackendMergeSnapshot* transaction)
{
    if (!transaction || current.mapId == candidate.mapId ||
        currentKeyframeId == candidateKeyframeId ||
        !finitePose(currentFromCandidate) || !validCovariance(covariance) ||
        current.keyframes.empty() || candidate.keyframes.empty())
        return false;
    const auto calibrationMatches = [](const BackendStereoCalibration& left,
                                       const BackendStereoCalibration& right) {
        const double valuesLeft[] = {left.fx, left.fy, left.cx, left.cy,
                                     left.baseline};
        const double valuesRight[] = {right.fx, right.fy, right.cx, right.cy,
                                      right.baseline};
        for (std::size_t index = 0; index < 5U; ++index)
            if (!std::isfinite(valuesLeft[index]) ||
                !std::isfinite(valuesRight[index]) ||
                std::abs(valuesLeft[index] - valuesRight[index]) > 1e-6)
                return false;
        return true;
    };
    if (!calibrationMatches(current.calibration, candidate.calibration))
        return false;

    BackendMapSnapshot endpoints;
    const auto appendEndpoint = [&endpoints](
        const BackendMapSnapshot& source, std::uint64_t id) {
        const auto found = std::find_if(
            source.keyframes.begin(), source.keyframes.end(),
            [id](const BackendKeyframeState& state) { return state.id == id; });
        if (found == source.keyframes.end())
            return false;
        endpoints.keyframes.push_back(*found);
        return true;
    };
    if (!appendEndpoint(current, currentKeyframeId) ||
        !appendEndpoint(candidate, candidateKeyframeId) ||
        !appendAcceptedMergeEdge(endpoints, currentKeyframeId,
                                 candidateKeyframeId, currentFromCandidate,
                                 covariance))
        return false;

    const cv::Mat currentFromCandidateMatrix =
        toPoseMatrix(currentFromCandidate);
    std::array<float, 16> candidateFromCurrent{};
    if (!copyPose(currentFromCandidateMatrix.inv(), candidateFromCurrent))
        return false;
    BackendMapSnapshot transformedCurrent;
    if (!transformSnapshot(current, candidateFromCurrent, &transformedCurrent))
        return false;

    BackendMergeSnapshot result;
    result.currentSource = {current.mapId, current.version,
                            current.topologySignature};
    result.candidateSource = {candidate.mapId, candidate.version,
                              candidate.topologySignature};
    result.currentKeyframeId = currentKeyframeId;
    result.candidateKeyframeId = candidateKeyframeId;
    result.candidateFromCurrent = candidateFromCurrent;
    for (const auto& keyframe : current.keyframes)
        result.currentKeyframeIds.push_back(keyframe.id);
    for (const auto& landmark : current.landmarks)
        result.currentLandmarkIds.push_back(landmark.id);
    for (const auto& keyframe : candidate.keyframes)
        result.candidateKeyframeIds.push_back(keyframe.id);
    for (const auto& landmark : candidate.landmarks)
        result.candidateLandmarkIds.push_back(landmark.id);
    result.graph = candidate;
    result.graph.mapId = current.mapId;
    result.graph.version = current.version;
    result.graph.topologySignature = current.topologySignature;

    std::unordered_set<std::uint64_t> keyframeIds;
    std::unordered_set<std::uint64_t> landmarkIds;
    for (const auto& keyframe : result.graph.keyframes)
        if (!keyframeIds.insert(keyframe.id).second)
            return false;
    for (const auto& landmark : result.graph.landmarks)
        if (!landmarkIds.insert(landmark.id).second)
            return false;
    for (const auto& keyframe : transformedCurrent.keyframes)
        if (!keyframeIds.insert(keyframe.id).second)
            return false;
    for (const auto& landmark : transformedCurrent.landmarks)
        if (!landmarkIds.insert(landmark.id).second)
            return false;

    result.graph.keyframes.insert(result.graph.keyframes.end(),
                                  transformedCurrent.keyframes.begin(),
                                  transformedCurrent.keyframes.end());
    result.graph.landmarks.insert(result.graph.landmarks.end(),
                                  transformedCurrent.landmarks.begin(),
                                  transformedCurrent.landmarks.end());
    result.graph.observations.insert(result.graph.observations.end(),
                                     transformedCurrent.observations.begin(),
                                     transformedCurrent.observations.end());
    result.graph.graphEdges.insert(result.graph.graphEdges.end(),
                                   transformedCurrent.graphEdges.begin(),
                                   transformedCurrent.graphEdges.end());
    result.graph.graphEdges.push_back(endpoints.graphEdges.front());
    *transaction = std::move(result);
    return true;
}

std::optional<std::array<double, 36>> GtsamMapAdapter::mergeCovariance(
    std::size_t inliers, std::size_t correspondenceCount,
    double rmsErrorMeters)
{
    if (inliers < 6U || correspondenceCount < inliers ||
        !(std::isfinite(rmsErrorMeters) && rmsErrorMeters >= 0.0))
        return std::nullopt;
    const double inlierRatio = static_cast<double>(inliers) /
                               static_cast<double>(correspondenceCount);
    const double residualScale = std::max(0.25,
        (rmsErrorMeters / 0.05) * (rmsErrorMeters / 0.05));
    const double supportScale = 20.0 / static_cast<double>(inliers);
    const double scale = std::clamp(
        residualScale * supportScale / std::max(0.05, inlierRatio),
        0.25, 25.0);
    const double rotationVariance = std::clamp(0.0004 * scale,
                                               0.0001, 0.04);
    const double translationVariance = std::clamp(0.0025 * scale,
                                                  0.0004, 0.25);
    std::array<double, 36> covariance{};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        covariance[axis * 6U + axis] = rotationVariance;
        const std::size_t translationAxis = axis + 3U;
        covariance[translationAxis * 6U + translationAxis] =
            translationVariance;
    }
    return covariance;
}

bool GtsamMapAdapter::consistentMergeTransform(
    const std::array<float, 16>& firstCurrentFromCandidate,
    const std::array<float, 16>& secondCurrentFromCandidate,
    double maximumTranslationMeters,
    double maximumRotationRadians)
{
    if (!finitePose(firstCurrentFromCandidate) ||
        !finitePose(secondCurrentFromCandidate) ||
        !(std::isfinite(maximumTranslationMeters) &&
          maximumTranslationMeters > 0.0) ||
        !(std::isfinite(maximumRotationRadians) &&
          maximumRotationRadians > 0.0))
        return false;
    double translation = 0.0;
    double rotation = 0.0;
    return mergeTransformDelta(firstCurrentFromCandidate,
                               secondCurrentFromCandidate,
                               &translation, &rotation) &&
           translation <= maximumTranslationMeters &&
           rotation <= maximumRotationRadians;
}

BackendMapResultValidation GtsamMapAdapter::validateMergeResult(
    const BackendMergeSnapshot& snapshot,
    const BackendMergeResult& result)
{
    if (!result.accepted ||
        result.currentSource.mapId == result.candidateSource.mapId ||
        result.currentSource.mapId != snapshot.currentSource.mapId ||
        result.currentSource.version != snapshot.currentSource.version ||
        result.currentSource.topologySignature !=
            snapshot.currentSource.topologySignature ||
        result.candidateSource.mapId != snapshot.candidateSource.mapId ||
        result.candidateSource.version != snapshot.candidateSource.version ||
        result.candidateSource.topologySignature !=
            snapshot.candidateSource.topologySignature)
        return {};
    BackendMapResult mapResult;
    mapResult.accepted = true;
    mapResult.sourceVersion = snapshot.graph.version;
    mapResult.sourceMapId = snapshot.graph.mapId;
    mapResult.sourceTopologySignature = snapshot.graph.topologySignature;
    mapResult.keyframes = result.keyframes;
    mapResult.landmarks = result.landmarks;
    return validateMapResult(snapshot.graph, mapResult);
}

bool GtsamMapAdapter::commitMerge(const BackendMergeResult& result,
                                  Map& currentMap, Map& candidateMap,
                                  std::string* rejectionReason)
{
    const auto reject = [rejectionReason](const char* reason) {
        if (rejectionReason)
            *rejectionReason = reason;
        return false;
    };
    if (!result.accepted || &currentMap == &candidateMap ||
        result.currentSource.mapId != currentMap.GetId() ||
        result.candidateSource.mapId != candidateMap.GetId())
        return reject("invalid result or source map identity");
    Map* first = &currentMap < &candidateMap ? &currentMap : &candidateMap;
    Map* second = first == &currentMap ? &candidateMap : &currentMap;
    std::unique_lock<std::shared_timed_mutex> firstLock(
        first->mMutexMapUpdate, std::defer_lock);
    std::unique_lock<std::shared_timed_mutex> secondLock(
        second->mMutexMapUpdate, std::defer_lock);
    std::lock(firstLock, secondLock);

    const std::vector<KeyFrame*> currentKeyframes =
        currentMap.GetAllKeyFrames();
    const std::vector<KeyFrame*> candidateKeyframes =
        candidateMap.GetAllKeyFrames();
    const std::vector<MapPoint*> currentLandmarks =
        currentMap.GetAllMapPoints();
    const std::vector<MapPoint*> candidateLandmarks =
        candidateMap.GetAllMapPoints();
    if (static_cast<std::uint64_t>(currentMap.GetMapChangeIndex()) !=
        result.currentSource.version)
        return reject("current map version changed");
    if (static_cast<std::uint64_t>(candidateMap.GetMapChangeIndex()) !=
        result.candidateSource.version)
        return reject("candidate map version changed");
    if (topologySignature(candidateKeyframes, candidateLandmarks) !=
        result.candidateSource.topologySignature)
        return reject("candidate map topology changed");

    std::unordered_map<std::uint64_t, KeyFrame*> keyframes;
    std::unordered_map<std::uint64_t, MapPoint*> landmarks;
    for (KeyFrame* keyframe : currentKeyframes)
        if (keyframe && !keyframe->isBad() &&
            !keyframes.emplace(keyframe->mnId, keyframe).second)
            return reject("duplicate current keyframe id");
    for (KeyFrame* keyframe : candidateKeyframes)
        if (keyframe && !keyframe->isBad() &&
            !keyframes.emplace(keyframe->mnId, keyframe).second)
            return reject("duplicate candidate keyframe id");
    for (MapPoint* landmark : currentLandmarks)
        if (landmark && !landmark->isBad() &&
            !landmarks.emplace(landmark->mnId, landmark).second)
            return reject("duplicate current landmark id");
    for (MapPoint* landmark : candidateLandmarks)
        if (landmark && !landmark->isBad() &&
            !landmarks.emplace(landmark->mnId, landmark).second)
            return reject("duplicate candidate landmark id");
    for (const std::uint64_t id : result.currentKeyframeIds)
        if (keyframes.count(id) != 1U ||
            keyframes.at(id)->GetMap() != &currentMap)
            return reject("snapshotted current keyframe missing or moved");
    for (const std::uint64_t id : result.currentLandmarkIds)
        if (landmarks.count(id) != 1U ||
            landmarks.at(id)->GetMap() != &currentMap)
            return reject("snapshotted current landmark missing or moved");
    for (const std::uint64_t id : result.candidateKeyframeIds)
        if (keyframes.count(id) != 1U ||
            keyframes.at(id)->GetMap() != &candidateMap)
            return reject("snapshotted candidate keyframe missing or moved");
    for (const std::uint64_t id : result.candidateLandmarkIds)
        if (landmarks.count(id) != 1U ||
            landmarks.at(id)->GetMap() != &candidateMap)
            return reject("snapshotted candidate landmark missing or moved");
    std::unordered_set<std::uint64_t> optimizedKeyframeIds;
    std::unordered_set<std::uint64_t> optimizedLandmarkIds;
    for (const auto& state : result.keyframes) {
        if (!finitePose(state.pose) || keyframes.count(state.id) != 1U)
            return reject("optimized keyframe is invalid or missing");
        if (!optimizedKeyframeIds.insert(state.id).second)
            return reject("duplicate optimized keyframe id");
        for (float value : state.velocity)
            if (!std::isfinite(static_cast<double>(value)))
                return reject("optimized keyframe velocity is non-finite");
        for (float value : state.accelerometerBias)
            if (!std::isfinite(static_cast<double>(value)))
                return reject("optimized accelerometer bias is non-finite");
        for (float value : state.gyroscopeBias)
            if (!std::isfinite(static_cast<double>(value)))
                return reject("optimized gyroscope bias is non-finite");
    }
    for (const auto& state : result.landmarks) {
        if (landmarks.count(state.id) != 1U)
            return reject("optimized landmark is missing");
        if (!optimizedLandmarkIds.insert(state.id).second)
            return reject("duplicate optimized landmark id");
        for (float value : state.position)
            if (!std::isfinite(static_cast<double>(value)))
                return reject("optimized landmark is non-finite");
    }
    for (KeyFrame* keyframe : candidateKeyframes)
        if (keyframe && !keyframe->isBad() &&
            optimizedKeyframeIds.count(keyframe->mnId) == 0U)
            return reject("candidate keyframe absent from optimized result");
    for (MapPoint* landmark : candidateLandmarks)
        if (landmark && !landmark->isBad() &&
            optimizedLandmarkIds.count(landmark->mnId) == 0U)
            return reject("candidate landmark absent from optimized result");
    const auto currentEndpoint = keyframes.find(result.currentKeyframeId);
    const auto candidateEndpoint = keyframes.find(result.candidateKeyframeId);
    if (currentEndpoint == keyframes.end() ||
        candidateEndpoint == keyframes.end())
        return reject("merge endpoint missing");
    KeyFrame* currentKeyframe = currentEndpoint->second;
    KeyFrame* candidateKeyframe = candidateEndpoint->second;

    if (!finitePose(result.candidateFromCurrent))
        return reject("candidate-from-current transform is invalid");
    const cv::Mat candidateFromCurrent =
        toPoseMatrix(result.candidateFromCurrent);
    const cv::Mat rotation = candidateFromCurrent.rowRange(0, 3).colRange(0, 3);
    std::unordered_map<std::uint64_t, cv::Mat> preservedKeyframePoses;
    std::unordered_map<std::uint64_t, cv::Mat> preservedKeyframeVelocities;
    std::unordered_map<std::uint64_t, cv::Mat> preservedLandmarkPositions;
    for (KeyFrame* keyframe : currentKeyframes) {
        if (!keyframe || keyframe->isBad() ||
            optimizedKeyframeIds.count(keyframe->mnId) != 0U)
            continue;
        const cv::Mat transformed =
            candidateFromCurrent * keyframe->GetPoseInverse();
        const cv::Mat velocity = rotation * keyframe->GetVelocityOld();
        if (!cv::checkRange(transformed) || !cv::checkRange(velocity))
            return reject("appended current keyframe transform is invalid");
        preservedKeyframePoses.emplace(keyframe->mnId, transformed.inv());
        preservedKeyframeVelocities.emplace(keyframe->mnId, velocity);
    }
    for (MapPoint* landmark : currentLandmarks) {
        if (!landmark || landmark->isBad() ||
            optimizedLandmarkIds.count(landmark->mnId) != 0U)
            continue;
        const cv::Mat position = landmark->GetWorldPos();
        if (position.rows != 3 || position.cols != 1 ||
            position.type() != CV_32F || !cv::checkRange(position))
            return reject("appended current landmark position is invalid");
        const cv::Mat homogeneous = (cv::Mat_<float>(4, 1)
            << position.at<float>(0), position.at<float>(1),
               position.at<float>(2), 1.0F);
        const cv::Mat transformed = candidateFromCurrent * homogeneous;
        if (!cv::checkRange(transformed))
            return reject("appended current landmark transform is invalid");
        preservedLandmarkPositions.emplace(
            landmark->mnId, transformed.rowRange(0, 3).clone());
    }

    for (const auto& state : result.keyframes) {
        KeyFrame* keyframe = keyframes.at(state.id);
        keyframe->SetPose(toPoseMatrix(state.pose).inv());
        keyframe->SetVelocity((cv::Mat_<float>(3, 1)
            << state.velocity[0], state.velocity[1], state.velocity[2]));
        keyframe->SetNewBias(IMU::Bias(
            state.accelerometerBias[0], state.accelerometerBias[1],
            state.accelerometerBias[2], state.gyroscopeBias[0],
            state.gyroscopeBias[1], state.gyroscopeBias[2]));
    }
    for (const auto& state : result.landmarks) {
        MapPoint* landmark = landmarks.at(state.id);
        landmark->SetWorldPos((cv::Mat_<float>(3, 1)
            << state.position[0], state.position[1], state.position[2]));
    }
    for (const auto& [id, pose] : preservedKeyframePoses) {
        keyframes.at(id)->SetPose(pose);
        keyframes.at(id)->SetVelocity(preservedKeyframeVelocities.at(id));
    }
    for (const auto& [id, position] : preservedLandmarkPositions)
        landmarks.at(id)->SetWorldPos(position);
    for (KeyFrame* keyframe : candidateKeyframes) {
        if (!keyframe || keyframe->isBad())
            continue;
        keyframe->UpdateMap(&currentMap);
        currentMap.AddKeyFrame(keyframe);
        candidateMap.EraseKeyFrame(keyframe);
    }
    for (MapPoint* landmark : candidateLandmarks) {
        if (!landmark || landmark->isBad())
            continue;
        landmark->UpdateMap(&currentMap);
        currentMap.AddMapPoint(landmark);
        candidateMap.EraseMapPoint(landmark);
        landmark->UpdateNormalAndDepth();
    }
    currentKeyframe->AddMergeEdge(candidateKeyframe);
    candidateKeyframe->AddMergeEdge(currentKeyframe);
    KeyFrame* currentOrigin = currentMap.GetOriginKF();
    if (currentOrigin && currentOrigin != candidateKeyframe &&
        currentOrigin->GetParent() == nullptr)
        currentOrigin->ChangeParent(candidateKeyframe);
    currentMap.IncreaseChangeIndex();
    currentMap.InformNewBigChange();
    candidateMap.SetBad();
    if (rejectionReason)
        rejectionReason->clear();
    return true;
}

BackendMapResultValidation GtsamMapAdapter::validateLocalResult(
    const BackendMapSnapshot& snapshot,
    const BackendMapResult& result)
{
    return validateMapResult(snapshot, result);
}

BackendMapResultValidation GtsamMapAdapter::validateMapResult(
    const BackendMapSnapshot& snapshot,
    const BackendMapResult& result)
{
    BackendMapResultValidation validation;
    if (!result.accepted)
        return validation;

    std::unordered_map<std::uint64_t, const BackendKeyframeState*> keyframes;
    std::unordered_map<std::uint64_t, const BackendLandmarkState*> landmarks;
    std::vector<double> landmarkTranslations;
    for (const auto& state : snapshot.keyframes)
        keyframes.emplace(state.id, &state);
    for (const auto& state : snapshot.landmarks)
        landmarks.emplace(state.id, &state);

    for (const auto& state : result.keyframes) {
        const auto found = keyframes.find(state.id);
        if (found == keyframes.end() || !finitePose(state.pose))
            return validation;
        const cv::Mat sourceFromOptimized =
            toPoseMatrix(found->second->pose).inv() * toPoseMatrix(state.pose);
        if (!cv::checkRange(sourceFromOptimized))
            return validation;
        const double dx = sourceFromOptimized.at<float>(0, 3);
        const double dy = sourceFromOptimized.at<float>(1, 3);
        const double dz = sourceFromOptimized.at<float>(2, 3);
        const double shift = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double trace = sourceFromOptimized.at<float>(0, 0) +
                             sourceFromOptimized.at<float>(1, 1) +
                             sourceFromOptimized.at<float>(2, 2);
        const double cosine = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
        const double rotation = std::acos(cosine);
        if (!std::isfinite(shift) || !std::isfinite(rotation))
            return validation;
        validation.maximumPoseTranslation = std::max(
            validation.maximumPoseTranslation, shift);
        validation.maximumPoseRotation = std::max(
            validation.maximumPoseRotation, rotation);
    }
    for (const auto& state : result.landmarks) {
        const auto found = landmarks.find(state.id);
        if (found == landmarks.end())
            return validation;
        const auto& source = *found->second;
        const double dx = state.position[0] - source.position[0];
        const double dy = state.position[1] - source.position[1];
        const double dz = state.position[2] - source.position[2];
        const double shift = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!std::isfinite(shift))
            return validation;
        validation.maximumLandmarkTranslation = std::max(
            validation.maximumLandmarkTranslation, shift);
        landmarkTranslations.push_back(shift);
    }
    if (!landmarkTranslations.empty()) {
        const std::size_t p95Index = static_cast<std::size_t>(std::ceil(
            0.95 * static_cast<double>(landmarkTranslations.size()))) - 1U;
        std::nth_element(landmarkTranslations.begin(),
                         landmarkTranslations.begin() + p95Index,
                         landmarkTranslations.end());
        validation.landmarkTranslationP95 = landmarkTranslations[p95Index];
    }
    if (!projectionConsistent(snapshot, result))
        return validation;
    validation.accepted = true;
    return validation;
}

bool GtsamMapAdapter::commit(const BackendMapResult& result, Map& map,
                             std::uint64_t expectedVersion)
{
    std::unique_lock<std::shared_timed_mutex> mapLock(map.mMutexMapUpdate);
    if (!result.accepted || result.sourceVersion != expectedVersion ||
        static_cast<std::uint64_t>(map.GetMapChangeIndex()) != expectedVersion ||
        result.sourceMapId != static_cast<std::uint64_t>(map.GetId()) ||
        result.sourceTopologySignature != topologySignature(
            map.GetAllKeyFrames(), map.GetAllMapPoints()))
        return false;
    std::unordered_map<std::uint64_t, KeyFrame*> keyframes;
    for (KeyFrame* keyframe : map.GetAllKeyFrames())
        if (keyframe)
            keyframes.emplace(keyframe->mnId, keyframe);
    std::unordered_map<std::uint64_t, MapPoint*> landmarks;
    for (MapPoint* landmark : map.GetAllMapPoints())
        if (landmark)
            landmarks.emplace(landmark->mnId, landmark);
    std::vector<cv::Mat> cameraFromMapPoses;
    cameraFromMapPoses.reserve(result.keyframes.size());
    for (const auto& state : result.keyframes) {
        if (!finitePose(state.pose))
            return false;
        cv::Mat cameraFromMap = toPoseMatrix(state.pose).inv();
        if (!cv::checkRange(cameraFromMap))
            return false;
        cameraFromMapPoses.push_back(cameraFromMap);
        const auto found = keyframes.find(state.id);
        if (found == keyframes.end())
            return false;
        for (float value : state.velocity)
            if (!std::isfinite(static_cast<double>(value)))
                return false;
        for (float value : state.accelerometerBias)
            if (!std::isfinite(static_cast<double>(value)))
                return false;
        for (float value : state.gyroscopeBias)
            if (!std::isfinite(static_cast<double>(value)))
                return false;
    }
    for (const auto& state : result.landmarks) {
        if (landmarks.count(state.id) == 0U)
            return false;
        for (float value : state.position)
            if (!std::isfinite(static_cast<double>(value)))
                return false;
    }
    std::unordered_set<std::uint64_t> replacedIds;
    std::unordered_set<std::uint64_t> retainedIds;
    for (const auto& replacement : result.landmarkReplacements) {
        if (replacement.replacedId == replacement.retainedId ||
            !replacedIds.insert(replacement.replacedId).second ||
            !retainedIds.insert(replacement.retainedId).second ||
            landmarks.count(replacement.replacedId) == 0U ||
            landmarks.count(replacement.retainedId) == 0U)
            return false;
        MapPoint* source = landmarks.at(replacement.replacedId);
        MapPoint* target = landmarks.at(replacement.retainedId);
        if (source->isBad() || target->isBad() ||
            source->GetMap() != &map || target->GetMap() != &map)
            return false;
    }
    for (const auto id : replacedIds)
        if (retainedIds.count(id) != 0U)
            return false;
    for (std::size_t index = 0; index < result.keyframes.size(); ++index) {
        const auto& state = result.keyframes[index];
        KeyFrame* keyframe = keyframes.at(state.id);
        keyframe->SetPose(cameraFromMapPoses[index]);
        keyframe->SetVelocity((cv::Mat_<float>(3, 1)
            << state.velocity[0], state.velocity[1], state.velocity[2]));
        keyframe->SetNewBias(IMU::Bias(
            state.accelerometerBias[0], state.accelerometerBias[1],
            state.accelerometerBias[2], state.gyroscopeBias[0],
            state.gyroscopeBias[1], state.gyroscopeBias[2]));
    }
    for (const auto& state : result.landmarks) {
        MapPoint* landmark = landmarks.at(state.id);
        landmark->SetWorldPos((cv::Mat_<float>(3, 1)
            << state.position[0], state.position[1], state.position[2]));
        landmark->UpdateNormalAndDepth();
    }
    for (const auto& replacement : result.landmarkReplacements) {
        MapPoint* target = landmarks.at(replacement.retainedId);
        landmarks.at(replacement.replacedId)->Replace(target);
        target->UpdateNormalAndDepth();
    }
    return true;
}

}  // namespace ORB_SLAM3
