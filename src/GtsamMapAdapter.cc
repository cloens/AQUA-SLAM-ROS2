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
    double sigma)
{
    if (fromKeyframeId == toKeyframeId ||
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
    BackendGraphEdge edge;
    edge.fromKeyframeId = fromKeyframeId;
    edge.toKeyframeId = toKeyframeId;
    edge.kind = BackendGraphEdgeKind::Loop;
    edge.sigma = sigma;
    if (!copyPose(mapFromFrom.inv() * mapFromTo, edge.relativePose))
        return false;
    snapshot.graphEdges.push_back(edge);
    return true;
}

BackendMapResultValidation GtsamMapAdapter::validateLocalResult(
    const BackendMapSnapshot& snapshot,
    const BackendMapResult& result,
    double maximumPoseTranslation,
    double maximumLandmarkTranslation)
{
    BackendMapResultValidation validation;
    if (!result.accepted ||
        !(std::isfinite(maximumPoseTranslation) &&
          maximumPoseTranslation > 0.0) ||
        !(std::isfinite(maximumLandmarkTranslation) &&
          maximumLandmarkTranslation > 0.0))
        return validation;

    std::unordered_map<std::uint64_t, const BackendKeyframeState*> keyframes;
    std::unordered_map<std::uint64_t, const BackendLandmarkState*> landmarks;
    for (const auto& state : snapshot.keyframes)
        keyframes.emplace(state.id, &state);
    for (const auto& state : snapshot.landmarks)
        landmarks.emplace(state.id, &state);

    for (const auto& state : result.keyframes) {
        const auto found = keyframes.find(state.id);
        if (found == keyframes.end() || !finitePose(state.pose))
            return validation;
        const auto& source = *found->second;
        const double dx = state.pose[3] - source.pose[3];
        const double dy = state.pose[7] - source.pose[7];
        const double dz = state.pose[11] - source.pose[11];
        const double shift = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!std::isfinite(shift))
            return validation;
        validation.maximumPoseTranslation = std::max(
            validation.maximumPoseTranslation, shift);
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
    }
    validation.accepted =
        validation.maximumPoseTranslation <= maximumPoseTranslation &&
        validation.maximumLandmarkTranslation <= maximumLandmarkTranslation;
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
    for (const auto& state : result.keyframes) {
        if (!finitePose(state.pose))
            return false;
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
    for (const auto& state : result.keyframes) {
        cv::Mat mapFromCamera = toPoseMatrix(state.pose);
        cv::Mat cameraFromMap = mapFromCamera.inv();
        if (!cv::checkRange(cameraFromMap))
            return false;
        KeyFrame* keyframe = keyframes.at(state.id);
        keyframe->SetPose(cameraFromMap);
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
    return true;
}

}  // namespace ORB_SLAM3
