#ifndef AQUA_BACKEND_FACADE_H
#define AQUA_BACKEND_FACADE_H

#include <stdexcept>
#include <string>

namespace ORB_SLAM3
{

enum class BackendMode
{
    GtsamDynamic
};

enum class EstimatorMode
{
    ImuStereo,
    DvlStereo
};

enum class OptimizationStage
{
    Frame,
    Initialization,
    LocalBundleAdjustment,
    GlobalBundleAdjustment,
    LoopCandidateVerification,
    EssentialGraph
};

struct RuntimeProfile
{
    std::string sensorMode;
    std::string calibrationSha256;
    bool dvlEnabled = false;
};

BackendMode loadBackendMode(const std::string& value);
EstimatorMode selectEstimatorMode(const RuntimeProfile& profile);
bool usesDynamicBackend(BackendMode mode, OptimizationStage stage) noexcept;

class BackendFacade
{
public:
    explicit BackendFacade(BackendMode mode);

    BackendMode mode() const noexcept;

    // Backend selection is process immutable; this method exists only to
    // produce an explicit diagnostic for accidental runtime switching.
    void setMode(BackendMode mode);

private:
    const BackendMode mMode;
};

}  // namespace ORB_SLAM3

#endif  // AQUA_BACKEND_FACADE_H
