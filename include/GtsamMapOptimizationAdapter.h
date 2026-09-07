#ifndef AQUA_GTSAM_MAP_OPTIMIZATION_ADAPTER_H
#define AQUA_GTSAM_MAP_OPTIMIZATION_ADAPTER_H

#include "GtsamMapAdapter.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ORB_SLAM3
{

struct LocalGraphSelection
{
    bool accepted = false;
    std::vector<std::uint64_t> localKeyframeIds;
    std::vector<std::uint64_t> fixedKeyframeIds;
};

struct StereoFactorDecision
{
    bool accepted = false;
    bool deferred = false;
    std::string reason;
    double residualPx = std::numeric_limits<double>::quiet_NaN();
    double nis = std::numeric_limits<double>::quiet_NaN();
};

class GtsamMapOptimizationAdapter
{
public:
    StereoFactorDecision evaluateStereoObservation(
        const BackendMapSnapshot& snapshot,
        const BackendStereoObservation& observation) const;
    LocalGraphSelection selectLocalGraph(
        const BackendMapSnapshot& snapshot,
        std::size_t maximumLocalKeyframes = 8U) const;
    BackendMapResult optimizeLocal(const BackendMapSnapshot& snapshot) const;
    BackendMapResult optimizeGlobal(const BackendMapSnapshot& snapshot) const;
    BackendMapResult optimizeEssentialGraph(const BackendMapSnapshot& snapshot) const;
    BackendMergeResult optimizeMerge(const BackendMergeSnapshot& snapshot) const;

    BackendLoopCandidateResult verifyLoopCandidateSE3(
        const std::vector<BackendLoopPointCorrespondence>& correspondences,
        double maximumResidualMeters, std::size_t minimumInliers) const;
};

}  // namespace ORB_SLAM3

#endif  // AQUA_GTSAM_MAP_OPTIMIZATION_ADAPTER_H
