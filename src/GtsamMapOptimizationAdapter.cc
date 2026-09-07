#include "GtsamMapOptimizationAdapter.h"

#include <uw_dynamic_backend/global_optimizer.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/inference/Symbol.h>

#include <cmath>
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ORB_SLAM3
{
namespace
{
gtsam::Pose3 toPose(const std::array<float, 16>& values);

StereoFactorDecision rejectStereo(std::string reason)
{
    StereoFactorDecision decision;
    decision.reason = std::move(reason);
    return decision;
}

bool validCalibration(const BackendStereoCalibration& calibration)
{
    return std::isfinite(calibration.fx) && calibration.fx > 0.0 &&
           std::isfinite(calibration.fy) && calibration.fy > 0.0 &&
           std::isfinite(calibration.cx) && std::isfinite(calibration.cy) &&
           std::isfinite(calibration.baseline) && calibration.baseline > 0.0;
}

struct SnapshotIndex
{
    explicit SnapshotIndex(const BackendMapSnapshot& snapshot)
    {
        if (validCalibration(snapshot.calibration)) {
            const auto& values = snapshot.calibration;
            calibration = std::make_shared<gtsam::Cal3_S2Stereo>(
                values.fx, values.fy, 0.0, values.cx, values.cy, values.baseline);
        }
        keyframes.reserve(snapshot.keyframes.size());
        landmarks.reserve(snapshot.landmarks.size());
        for (const auto& keyframe : snapshot.keyframes) {
            auto inserted = keyframes.emplace(keyframe.id, &keyframe);
            if (!inserted.second)
                inserted.first->second = nullptr;
        }
        for (const auto& landmark : snapshot.landmarks) {
            auto inserted = landmarks.emplace(landmark.id, &landmark);
            if (!inserted.second)
                inserted.first->second = nullptr;
        }
    }

    std::unordered_map<std::uint64_t, const BackendKeyframeState*> keyframes;
    std::unordered_map<std::uint64_t, const BackendLandmarkState*> landmarks;
    std::shared_ptr<gtsam::Cal3_S2Stereo> calibration;
};

gtsam::SharedNoiseModel stereoNoise(double variancePx2)
{
    const auto gaussian = gtsam::noiseModel::Isotropic::Sigma(
        3, std::sqrt(variancePx2));
    return gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(1.345), gaussian);
}

StereoFactorDecision evaluateStereo(
    const BackendMapSnapshot& snapshot,
    const SnapshotIndex& index,
    const BackendStereoObservation& observation,
    const gtsam::Pose3* poseOverride = nullptr,
    const gtsam::Point3* landmarkOverride = nullptr)
{
    if (!validCalibration(snapshot.calibration))
        return rejectStereo("invalid_calibration");
    const auto keyframeEntry = index.keyframes.find(observation.keyframeId);
    const BackendKeyframeState* keyframe = keyframeEntry == index.keyframes.end()
        ? nullptr : keyframeEntry->second;
    if (keyframe == nullptr)
        return rejectStereo("invalid_keyframe_id");
    const auto landmarkEntry = index.landmarks.find(observation.landmarkId);
    const BackendLandmarkState* landmark = landmarkEntry == index.landmarks.end()
        ? nullptr : landmarkEntry->second;
    if (landmark == nullptr)
        return rejectStereo("invalid_landmark_id");
    if (!(std::isfinite(observation.uLeft) &&
          std::isfinite(observation.uRight) && std::isfinite(observation.v)))
        return rejectStereo("nonfinite_measurement");
    if (!(std::isfinite(observation.sigmaPx) && observation.sigmaPx > 0.0 &&
          std::isfinite(observation.octaveVariancePx2) &&
          observation.octaveVariancePx2 > 0.0))
        return rejectStereo("invalid_covariance");
    if (observation.octave < 0)
        return rejectStereo("invalid_octave");
    if (!(std::isfinite(observation.quality) && observation.quality >= 0.0))
        return rejectStereo("invalid_quality");
    if (!std::isfinite(observation.stereoSkewPx))
        return rejectStereo("invalid_stereo_skew");
    if (!(observation.uLeft > observation.uRight))
        return rejectStereo("nonpositive_disparity");
    const gtsam::Point3 initialLandmark(landmark->position[0],
                                        landmark->position[1],
                                        landmark->position[2]);
    if (!initialLandmark.array().isFinite().all())
        return rejectStereo("invalid_landmark");

    // Backend snapshots store Twc (map from camera), which is also the pose
    // convention expected by GTSAM StereoCamera/GenericStereoFactor.
    gtsam::Pose3 initialPose;
    try {
        initialPose = toPose(keyframe->pose);
    } catch (const std::exception&) {
        return rejectStereo("invalid_keyframe_pose");
    }
    const gtsam::Pose3& pose = poseOverride == nullptr ? initialPose : *poseOverride;
    const gtsam::Point3& point =
        landmarkOverride == nullptr ? initialLandmark : *landmarkOverride;
    const gtsam::Point3 cameraPoint = pose.transformTo(point);
    if (!(cameraPoint.array().isFinite().all() && cameraPoint.z() > 0.0))
        return rejectStereo("behind_camera");
    StereoFactorDecision decision;
    try {
        const gtsam::StereoCamera camera(pose, index.calibration);
        const gtsam::Vector3 residual = (camera.project(point) -
            gtsam::StereoPoint2(observation.uLeft, observation.uRight,
                                observation.v)).vector();
        decision.residualPx = residual.norm();
        decision.nis = residual.squaredNorm() / observation.octaveVariancePx2;
    } catch (const std::exception&) {
        return rejectStereo("stereo_evaluation_failure");
    }
    if (!(std::isfinite(decision.residualPx) && std::isfinite(decision.nis)))
        return rejectStereo("nonfinite_residual");
    decision.accepted = true;
    decision.reason = "accepted";
    return decision;
}

void recordScalar(ScalarDistribution* distribution, double value)
{
    if (!(distribution != nullptr && std::isfinite(value) && value >= 0.0))
        return;
    ++distribution->count;
    distribution->sumSquares += value * value;
    distribution->maximum = std::max(distribution->maximum, value);
    distribution->rms = std::sqrt(
        distribution->sumSquares / static_cast<double>(distribution->count));
    const double logarithm = std::log2(std::max(value, 1e-12));
    const int bin = std::max(0, std::min(
        static_cast<int>(ScalarDistribution::kBinCount) - 1,
        static_cast<int>(std::floor(logarithm)) + 16));
    ++distribution->bins[static_cast<std::size_t>(bin)];
    const auto percentile = [distribution](double fraction) {
        const std::uint64_t threshold = static_cast<std::uint64_t>(
            std::ceil(fraction * static_cast<double>(distribution->count)));
        std::uint64_t cumulative = 0;
        for (std::size_t index = 0; index < distribution->bins.size(); ++index) {
            cumulative += distribution->bins[index];
            if (cumulative >= threshold)
                return std::pow(2.0, static_cast<int>(index) - 16);
        }
        return distribution->maximum;
    };
    distribution->p50 = percentile(0.50);
    distribution->p95 = percentile(0.95);
}

void recordStereoDecision(BackendMapDiagnostics* diagnostics,
                          const BackendStereoObservation& observation,
                          const StereoFactorDecision& decision)
{
    if (diagnostics == nullptr)
        return;
    ++diagnostics->stereo.attempted;
    if (decision.deferred)
        ++diagnostics->stereo.deferred;
    else if (decision.accepted)
        ++diagnostics->stereo.accepted;
    else
        ++diagnostics->stereo.rejected;
    ++diagnostics->stereo.reasonCounts[decision.reason];
    recordScalar(&diagnostics->stereoResidual, decision.residualPx);
    recordScalar(&diagnostics->stereoNis, decision.nis);
    recordScalar(&diagnostics->stereoDisparity,
                 observation.uLeft - observation.uRight);
    recordScalar(&diagnostics->stereoSkew, std::abs(observation.stereoSkewPx));
    recordScalar(&diagnostics->stereoQuality, observation.quality);
}

gtsam::Pose3 toPose(const std::array<float, 16>& values)
{
    if (!std::isfinite(values[0]) || std::abs(values[15] - 1.0F) > 1e-4F)
        throw std::invalid_argument("invalid map pose");
    gtsam::Matrix4 matrix;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) {
            const float value = values[static_cast<std::size_t>(row * 4 + col)];
            if (!std::isfinite(value))
                throw std::invalid_argument("non-finite map pose");
            matrix(row, col) = value;
        }
    return gtsam::Pose3(matrix);
}

std::array<float, 16> fromPose(const gtsam::Pose3& pose)
{
    std::array<float, 16> result{};
    const auto matrix = pose.matrix();
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            result[static_cast<std::size_t>(row * 4 + col)] =
                static_cast<float>(matrix(row, col));
    return result;
}

BackendMapSnapshot localWindow(const BackendMapSnapshot& snapshot,
                               const LocalGraphSelection& selection)
{
    BackendMapSnapshot window = snapshot;
    window.keyframes.clear();
    window.landmarks.clear();
    window.observations.clear();
    window.graphEdges.clear();
    if (!selection.accepted)
        return window;

    std::unordered_map<std::uint64_t, const BackendKeyframeState*> keyframes;
    for (const auto& keyframe : snapshot.keyframes)
        keyframes.emplace(keyframe.id, &keyframe);
    for (const std::uint64_t keyframeId : selection.localKeyframeIds)
        window.keyframes.push_back(*keyframes.at(keyframeId));
    for (const std::uint64_t keyframeId : selection.fixedKeyframeIds)
        window.keyframes.push_back(*keyframes.at(keyframeId));

    const std::unordered_set<std::uint64_t> localKeyframeIds(
        selection.localKeyframeIds.begin(), selection.localKeyframeIds.end());
    const std::unordered_set<std::uint64_t> fixedKeyframeIds(
        selection.fixedKeyframeIds.begin(), selection.fixedKeyframeIds.end());
    std::unordered_set<std::uint64_t> landmarkIds;
    for (const auto& observation : snapshot.observations) {
        if (localKeyframeIds.count(observation.keyframeId) == 0U)
            continue;
        landmarkIds.insert(observation.landmarkId);
    }
    for (const auto& landmark : snapshot.landmarks)
        if (landmarkIds.count(landmark.id) != 0U)
            window.landmarks.push_back(landmark);
    for (const auto& observation : snapshot.observations) {
        const bool selectedKeyframe =
            localKeyframeIds.count(observation.keyframeId) != 0U ||
            fixedKeyframeIds.count(observation.keyframeId) != 0U;
        if (selectedKeyframe && landmarkIds.count(observation.landmarkId) != 0U)
            window.observations.push_back(observation);
    }
    return window;
}

BackendMapResult optimize(const BackendMapSnapshot& snapshot, bool essential,
                          const std::vector<std::uint64_t>& fixedPoseIds = {},
                          const std::vector<std::uint64_t>& resultKeyframeIds = {})
{
    BackendMapResult result;
    result.sourceVersion = snapshot.version;
    result.sourceMapId = snapshot.mapId;
    result.sourceTopologySignature = snapshot.topologySignature;
    if (snapshot.keyframes.empty())
        return result;
    const SnapshotIndex snapshotIndex(snapshot);
    for (const auto& entry : snapshotIndex.keyframes) {
        if (entry.second == nullptr) {
            result.diagnostic = "duplicate snapshot keyframe identity";
            return result;
        }
    }
    for (const auto& entry : snapshotIndex.landmarks) {
        if (entry.second == nullptr) {
            result.diagnostic = "duplicate snapshot landmark identity";
            return result;
        }
    }

    std::vector<gtsam::Pose3> poses;
    std::vector<BackendLandmarkState> optimizedLandmarks;
    poses.reserve(snapshot.keyframes.size());
    for (const auto& keyframe : snapshot.keyframes)
        poses.push_back(toPose(keyframe.pose));

    uw_slam::dynamic_backend::GlobalOptimizer optimizer;
    std::unordered_map<std::uint64_t, gtsam::Pose3> optimizedPoses;
    if (essential) {
        std::vector<uw_slam::dynamic_backend::EssentialGraphEdge> edges;
        std::unordered_map<std::uint64_t, std::size_t> poseIndices;
        for (std::size_t index = 0; index < snapshot.keyframes.size(); ++index)
            poseIndices.emplace(snapshot.keyframes[index].id, index);
        for (const auto& graphEdge : snapshot.graphEdges) {
            const auto from = poseIndices.find(graphEdge.fromKeyframeId);
            const auto to = poseIndices.find(graphEdge.toKeyframeId);
            if (from == poseIndices.end() || to == poseIndices.end())
                return result;
            uw_slam::dynamic_backend::EssentialGraphEdge edge;
            edge.from = from->second;
            edge.to = to->second;
            edge.relativePose = toPose(graphEdge.relativePose);
            if (graphEdge.hasCovariance) {
                gtsam::Matrix covariance(6, 6);
                for (int row = 0; row < 6; ++row)
                    for (int col = 0; col < 6; ++col)
                        covariance(row, col) = graphEdge.covariance[
                            static_cast<std::size_t>(row * 6 + col)];
                try {
                    edge.noise =
                        gtsam::noiseModel::Gaussian::Covariance(covariance);
                } catch (const std::exception&) {
                    return result;
                }
            } else {
                if (!(std::isfinite(graphEdge.sigma) && graphEdge.sigma > 0.0))
                    return result;
                edge.noise = gtsam::noiseModel::Isotropic::Sigma(
                    6, graphEdge.sigma);
            }
            edges.push_back(edge);
        }
        if (poses.size() > 1U && edges.empty())
            return result;
        const std::vector<gtsam::Pose3> optimized =
            optimizer.optimizeEssentialGraph(poses, edges);
        if (optimized.size() != snapshot.keyframes.size())
            return result;
        for (std::size_t index = 0; index < optimized.size(); ++index)
            optimizedPoses.emplace(snapshot.keyframes[index].id,
                                   optimized[index]);
    } else {
        const auto& calibration = snapshot.calibration;
        if (!validCalibration(calibration))
            return result;
        const auto stereoCalibration = snapshotIndex.calibration;
        uw_slam::dynamic_backend::GraphSnapshot graph;
        graph.version = snapshot.version;
        for (std::size_t index = 0; index < poses.size(); ++index) {
            const gtsam::Key key =
                gtsam::Symbol('x', snapshot.keyframes[index].id);
            graph.values.insert(key, poses[index]);
            graph.poseKeys.push_back(key);
        }
        const auto addPriors = [&fixedPoseIds, &snapshot](
            uw_slam::dynamic_backend::GraphSnapshot* target) {
            if (fixedPoseIds.empty()) {
                target->graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                    target->poseKeys.front(),
                    target->values.at<gtsam::Pose3>(target->poseKeys.front()),
                    gtsam::noiseModel::Constrained::All(6));
                return true;
            }
            for (const std::uint64_t keyframeId : fixedPoseIds) {
                const gtsam::Key key = gtsam::Symbol('x', keyframeId);
                if (!target->values.exists(key))
                    return false;
                target->graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                    key, target->values.at<gtsam::Pose3>(key),
                    gtsam::noiseModel::Constrained::All(6));
            }
            return true;
        };
        if (!addPriors(&graph))
            return result;
        const auto addStereoFactor = [&stereoCalibration](
            uw_slam::dynamic_backend::GraphSnapshot* target,
            const BackendStereoObservation& observation) {
            target->graph.emplace_shared<
                gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>>(
                gtsam::StereoPoint2(observation.uLeft,
                                    observation.uRight,
                                    observation.v),
                stereoNoise(observation.octaveVariancePx2),
                gtsam::Symbol('x', observation.keyframeId),
                gtsam::Symbol('l', observation.landmarkId), stereoCalibration);
        };
        std::vector<const BackendStereoObservation*> candidates;
        std::vector<StereoFactorDecision> decisions;
        candidates.reserve(snapshot.observations.size());
        decisions.reserve(snapshot.observations.size());
        for (const auto& observation : snapshot.observations) {
            StereoFactorDecision decision = evaluateStereo(
                snapshot, snapshotIndex, observation);
            if (!decision.accepted) {
                recordStereoDecision(&result.diagnostics, observation, decision);
                continue;
            }
            candidates.push_back(&observation);
            decisions.push_back(std::move(decision));
        }
        if (candidates.empty())
            return result;
        std::unordered_set<std::uint64_t> insertedLandmarkIds;
        for (const BackendStereoObservation* observation : candidates) {
            if (!insertedLandmarkIds.insert(observation->landmarkId).second)
                continue;
            const BackendLandmarkState* landmark =
                snapshotIndex.landmarks.at(observation->landmarkId);
            if (landmark == nullptr)
                return result;
            graph.values.insert(gtsam::Symbol('l', landmark->id),
                gtsam::Point3(landmark->position[0], landmark->position[1],
                              landmark->position[2]));
        }
        for (const BackendStereoObservation* observation : candidates)
            addStereoFactor(&graph, *observation);
        const auto firstBatch = optimizer.optimizeBatch(graph);
        if (!firstBatch.accepted) {
            result.diagnostic = "first stereo solve: " + firstBatch.diagnostic;
            for (std::size_t index = 0; index < candidates.size(); ++index) {
                decisions[index].accepted = false;
                decisions[index].deferred = true;
                decisions[index].reason = "first_stereo_solve_failure";
                recordStereoDecision(&result.diagnostics, *candidates[index],
                                     decisions[index]);
            }
            return result;
        }
        std::vector<std::size_t> acceptedFactorIndices;
        acceptedFactorIndices.reserve(candidates.size());
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto& observation = *candidates[index];
            const gtsam::Key poseKey = gtsam::Symbol('x', observation.keyframeId);
            const gtsam::Key landmarkKey = gtsam::Symbol('l', observation.landmarkId);
            const gtsam::Pose3 solvedPose =
                firstBatch.values.at<gtsam::Pose3>(poseKey);
            const gtsam::Point3 solvedLandmark =
                firstBatch.values.at<gtsam::Point3>(landmarkKey);
            decisions[index] = evaluateStereo(
                snapshot, snapshotIndex, observation, &solvedPose, &solvedLandmark);
            if (decisions[index].accepted)
                acceptedFactorIndices.push_back(index);
            else
                recordStereoDecision(&result.diagnostics, observation,
                                     decisions[index]);
        }
        if (acceptedFactorIndices.empty())
            return result;
        uw_slam::dynamic_backend::GraphSnapshot finalGraph;
        finalGraph.version = snapshot.version;
        std::unordered_set<std::uint64_t> finalPoseIds(
            fixedPoseIds.begin(), fixedPoseIds.end());
        if (fixedPoseIds.empty())
            finalPoseIds.insert(snapshot.keyframes.front().id);
        for (const std::size_t index : acceptedFactorIndices)
            finalPoseIds.insert(candidates[index]->keyframeId);
        for (const auto& keyframe : snapshot.keyframes) {
            if (finalPoseIds.count(keyframe.id) == 0U)
                continue;
            const gtsam::Key poseKey = gtsam::Symbol('x', keyframe.id);
            finalGraph.poseKeys.push_back(poseKey);
            finalGraph.values.insert(
                poseKey, firstBatch.values.at<gtsam::Pose3>(poseKey));
        }
        std::unordered_set<std::uint64_t> finalLandmarkIds;
        for (const std::size_t index : acceptedFactorIndices) {
            const std::uint64_t landmarkId = candidates[index]->landmarkId;
            if (!finalLandmarkIds.insert(landmarkId).second)
                continue;
            const gtsam::Key landmarkKey = gtsam::Symbol('l', landmarkId);
            finalGraph.values.insert(
                landmarkKey, firstBatch.values.at<gtsam::Point3>(landmarkKey));
        }
        if (!addPriors(&finalGraph))
            return result;
        for (const std::size_t index : acceptedFactorIndices)
            addStereoFactor(&finalGraph, *candidates[index]);
        const auto finalBatch = optimizer.optimizeBatch(finalGraph);
        if (!finalBatch.accepted) {
            result.diagnostic = "second stereo solve: " + finalBatch.diagnostic;
            for (const std::size_t index : acceptedFactorIndices) {
                decisions[index].accepted = false;
                decisions[index].deferred = true;
                decisions[index].reason = "second_stereo_solve_failure";
                recordStereoDecision(&result.diagnostics, *candidates[index],
                                     decisions[index]);
            }
            return result;
        }
        result.acceptedStereoObservations.reserve(acceptedFactorIndices.size());
        for (const std::size_t index : acceptedFactorIndices) {
            recordStereoDecision(&result.diagnostics, *candidates[index],
                                 decisions[index]);
            result.acceptedStereoObservations.push_back(*candidates[index]);
        }
        for (const auto& keyframe : snapshot.keyframes) {
            const gtsam::Key poseKey = gtsam::Symbol('x', keyframe.id);
            if (finalBatch.values.exists(poseKey))
                optimizedPoses.emplace(
                    keyframe.id, finalBatch.values.at<gtsam::Pose3>(poseKey));
        }
        optimizedLandmarks.reserve(snapshot.landmarks.size());
        for (const auto& landmark : snapshot.landmarks) {
            const gtsam::Key key = gtsam::Symbol('l', landmark.id);
            if (!finalBatch.values.exists(key))
                continue;
            const gtsam::Point3 point = finalBatch.values.at<gtsam::Point3>(key);
            if (!point.array().isFinite().all())
                return result;
            BackendLandmarkState optimizedLandmark = landmark;
            optimizedLandmark.position = {
                static_cast<float>(point.x()), static_cast<float>(point.y()),
                static_cast<float>(point.z())};
            optimizedLandmarks.push_back(optimizedLandmark);
        }
    }
    const std::unordered_set<std::uint64_t> resultKeyframeIdSet(
        resultKeyframeIds.begin(), resultKeyframeIds.end());
    result.keyframes.reserve(resultKeyframeIds.empty()
                                 ? optimizedPoses.size()
                                 : resultKeyframeIds.size());
    for (const auto& keyframe : snapshot.keyframes) {
        if (!resultKeyframeIdSet.empty() &&
            resultKeyframeIdSet.count(keyframe.id) == 0U)
            continue;
        const auto optimized = optimizedPoses.find(keyframe.id);
        if (optimized == optimizedPoses.end())
            continue;
        BackendKeyframeState state = keyframe;
        state.pose = fromPose(optimized->second);
        result.keyframes.push_back(state);
    }
    result.landmarks = std::move(optimizedLandmarks);
    for (const auto& landmark : result.landmarks)
        recordScalar(&result.diagnostics.landmarkObservationDegree,
                     static_cast<double>(landmark.observationDegree));
    result.accepted = true;
    return result;
}
}  // namespace

StereoFactorDecision GtsamMapOptimizationAdapter::evaluateStereoObservation(
    const BackendMapSnapshot& snapshot,
    const BackendStereoObservation& observation) const
{
    return evaluateStereo(snapshot, SnapshotIndex(snapshot), observation);
}

LocalGraphSelection GtsamMapOptimizationAdapter::selectLocalGraph(
    const BackendMapSnapshot& snapshot, std::size_t maximumLocalKeyframes) const
{
    LocalGraphSelection selection;
    if (maximumLocalKeyframes == 0U || snapshot.keyframes.empty())
        return selection;

    std::unordered_map<std::uint64_t, const BackendKeyframeState*> keyframes;
    for (const auto& keyframe : snapshot.keyframes) {
        if (!std::isfinite(keyframe.timestampSec) ||
            !keyframes.emplace(keyframe.id, &keyframe).second)
            return selection;
    }
    std::unordered_set<std::uint64_t> landmarkIds;
    for (const auto& landmark : snapshot.landmarks)
        if (!landmarkIds.insert(landmark.id).second)
            return selection;

    std::vector<const BackendKeyframeState*> ordered;
    ordered.reserve(snapshot.keyframes.size());
    for (const auto& keyframe : snapshot.keyframes)
        ordered.push_back(&keyframe);
    std::sort(ordered.begin(), ordered.end(),
              [](const BackendKeyframeState* left,
                 const BackendKeyframeState* right) {
                  if (left->timestampSec != right->timestampSec)
                      return left->timestampSec < right->timestampSec;
                  return left->id < right->id;
              });
    const std::size_t firstLocal = ordered.size() > maximumLocalKeyframes
        ? ordered.size() - maximumLocalKeyframes
        : 0U;
    std::unordered_set<std::uint64_t> precedingKeyframeIds;
    for (std::size_t index = 0; index < firstLocal; ++index)
        precedingKeyframeIds.insert(ordered[index]->id);
    std::unordered_set<std::uint64_t> localKeyframeIds;
    for (std::size_t index = firstLocal; index < ordered.size(); ++index) {
        selection.localKeyframeIds.push_back(ordered[index]->id);
        localKeyframeIds.insert(ordered[index]->id);
    }

    std::unordered_set<std::uint64_t> retainedLandmarkIds;
    for (const auto& observation : snapshot.observations) {
        if (localKeyframeIds.count(observation.keyframeId) != 0U &&
            landmarkIds.count(observation.landmarkId) != 0U)
            retainedLandmarkIds.insert(observation.landmarkId);
    }
    if (retainedLandmarkIds.empty()) {
        selection.localKeyframeIds.clear();
        return selection;
    }

    std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>>
        sharedLandmarkIds;
    for (const auto& observation : snapshot.observations) {
        if (localKeyframeIds.count(observation.keyframeId) != 0U ||
            retainedLandmarkIds.count(observation.landmarkId) == 0U)
            continue;
        const auto keyframe = keyframes.find(observation.keyframeId);
        if (keyframe == keyframes.end() ||
            precedingKeyframeIds.count(observation.keyframeId) == 0U)
            continue;
        sharedLandmarkIds[observation.keyframeId].insert(observation.landmarkId);
    }

    std::vector<std::uint64_t> candidates;
    candidates.reserve(sharedLandmarkIds.size());
    for (const auto& candidate : sharedLandmarkIds)
        if (!candidate.second.empty())
            candidates.push_back(candidate.first);
    std::sort(candidates.begin(), candidates.end(),
              [&keyframes, &sharedLandmarkIds](std::uint64_t left,
                                                std::uint64_t right) {
                  const std::size_t leftCount = sharedLandmarkIds.at(left).size();
                  const std::size_t rightCount = sharedLandmarkIds.at(right).size();
                  if (leftCount != rightCount)
                      return leftCount > rightCount;
                  const double leftTimestamp = keyframes.at(left)->timestampSec;
                  const double rightTimestamp = keyframes.at(right)->timestampSec;
                  if (leftTimestamp != rightTimestamp)
                      return leftTimestamp > rightTimestamp;
                  return left < right;
              });
    if (candidates.size() > maximumLocalKeyframes)
        candidates.resize(maximumLocalKeyframes);
    selection.fixedKeyframeIds = std::move(candidates);
    selection.accepted = true;
    return selection;
}

BackendMapResult GtsamMapOptimizationAdapter::optimizeGlobal(
    const BackendMapSnapshot& snapshot) const
{
    return optimize(snapshot, false);
}

BackendMapResult GtsamMapOptimizationAdapter::optimizeLocal(
    const BackendMapSnapshot& snapshot) const
{
    const LocalGraphSelection selection = selectLocalGraph(snapshot);
    return optimize(localWindow(snapshot, selection), false,
                    selection.fixedKeyframeIds, selection.localKeyframeIds);
}

BackendMapResult GtsamMapOptimizationAdapter::optimizeEssentialGraph(
    const BackendMapSnapshot& snapshot) const
{
    try {
        return optimize(snapshot, true);
    } catch (const std::exception&) {
        BackendMapResult rejected;
        rejected.sourceVersion = snapshot.version;
        rejected.sourceMapId = snapshot.mapId;
        rejected.sourceTopologySignature = snapshot.topologySignature;
        return rejected;
    }
}

BackendMergeResult GtsamMapOptimizationAdapter::optimizeMerge(
    const BackendMergeSnapshot& snapshot) const
{
    BackendMergeResult result;
    result.currentSource = snapshot.currentSource;
    result.candidateSource = snapshot.candidateSource;
    result.currentKeyframeId = snapshot.currentKeyframeId;
    result.candidateKeyframeId = snapshot.candidateKeyframeId;
    result.candidateFromCurrent = snapshot.candidateFromCurrent;
    result.currentKeyframeIds = snapshot.currentKeyframeIds;
    result.currentLandmarkIds = snapshot.currentLandmarkIds;
    result.candidateKeyframeIds = snapshot.candidateKeyframeIds;
    result.candidateLandmarkIds = snapshot.candidateLandmarkIds;
    const BackendMapResult optimized = optimizeEssentialGraph(snapshot.graph);
    result.accepted = optimized.accepted;
    result.keyframes = optimized.keyframes;
    result.landmarks = optimized.landmarks;
    result.diagnostics = optimized.diagnostics;
    return result;
}

BackendLoopCandidateResult GtsamMapOptimizationAdapter::verifyLoopCandidateSE3(
    const std::vector<BackendLoopPointCorrespondence>& correspondences,
    double maximumResidualMeters, std::size_t minimumInliers) const
{
    std::vector<uw_slam::dynamic_backend::LoopPointCorrespondence> points;
    points.reserve(correspondences.size());
    for (const auto& correspondence : correspondences) {
        points.push_back({
            gtsam::Point3(correspondence.currentPoint[0],
                          correspondence.currentPoint[1],
                          correspondence.currentPoint[2]),
            gtsam::Point3(correspondence.candidatePoint[0],
                          correspondence.candidatePoint[1],
                          correspondence.candidatePoint[2])});
    }
    const auto verified = uw_slam::dynamic_backend::GlobalOptimizer()
        .verifyLoopCandidateSE3(points, maximumResidualMeters, minimumInliers);
    BackendLoopCandidateResult result;
    result.accepted = verified.accepted;
    result.inliers = verified.inliers;
    result.rmsErrorMeters = verified.rmsErrorMeters;
    result.diagnostic = verified.diagnostic;
    if (verified.accepted) {
        result.currentFromCandidate = fromPose(verified.currentFromCandidate);
        std::unordered_map<std::uint64_t, std::uint64_t> sources;
        std::unordered_map<std::uint64_t, std::uint64_t> targets;
        for (std::size_t index = 0; index < correspondences.size(); ++index) {
            const auto& correspondence = correspondences[index];
            if (correspondence.currentLandmarkId == correspondence.candidateLandmarkId ||
                (verified.currentFromCandidate.transformFrom(points[index].candidatePoint) -
                 points[index].currentPoint).norm() > maximumResidualMeters)
                continue;
            const auto source = sources.emplace(correspondence.currentLandmarkId,
                                                correspondence.candidateLandmarkId);
            const auto target = targets.emplace(correspondence.candidateLandmarkId,
                                                correspondence.currentLandmarkId);
            if ((!source.second && source.first->second != correspondence.candidateLandmarkId) ||
                (!target.second && target.first->second != correspondence.currentLandmarkId)) {
                result.landmarkReplacements.clear();
                return result;
            }
            if (source.second)
                result.landmarkReplacements.push_back({correspondence.currentLandmarkId,
                                                       correspondence.candidateLandmarkId});
        }
        for (const auto& source : sources) {
            if (targets.count(source.first) != 0U) {
                result.landmarkReplacements.clear();
                break;
            }
        }
    }
    return result;
}

}  // namespace ORB_SLAM3
