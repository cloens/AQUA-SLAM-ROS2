#include "BackendFacade.h"

#include <utility>

namespace ORB_SLAM3
{

BackendMode loadBackendMode(const std::string& value)
{
    if (value == "GTSAM_DYNAMIC")
        return BackendMode::GtsamDynamic;
    throw std::invalid_argument(
        "only GTSAM_DYNAMIC backend is supported (got: " + value + ")");
}

EstimatorMode selectEstimatorMode(const RuntimeProfile& profile)
{
    if (profile.dvlEnabled)
    {
        if (profile.sensorMode != "stereo_inertial_dvl")
            throw std::invalid_argument(
                "DVL-enabled profile must select stereo_inertial_dvl");
        return EstimatorMode::DvlStereo;
    }

    if (profile.sensorMode == "stereo_inertial")
        return EstimatorMode::ImuStereo;
    throw std::invalid_argument(
        "DVL-disabled profile must select stereo_inertial");
}

bool usesDynamicBackend(BackendMode mode, OptimizationStage stage) noexcept
{
    switch (stage)
    {
    case OptimizationStage::Frame:
    case OptimizationStage::Initialization:
    case OptimizationStage::LocalBundleAdjustment:
    case OptimizationStage::GlobalBundleAdjustment:
    case OptimizationStage::LoopCandidateVerification:
    case OptimizationStage::EssentialGraph:
        return mode == BackendMode::GtsamDynamic;
    }
    return false;
}

BackendFacade::BackendFacade(BackendMode mode)
    : mMode(mode)
{
}

BackendMode BackendFacade::mode() const noexcept
{
    return mMode;
}

void BackendFacade::setMode(BackendMode)
{
    throw std::logic_error("backend mode is immutable for process lifetime");
}

}  // namespace ORB_SLAM3
