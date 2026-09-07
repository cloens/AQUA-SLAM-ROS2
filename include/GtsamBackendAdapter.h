#ifndef AQUA_GTSAM_BACKEND_ADAPTER_H
#define AQUA_GTSAM_BACKEND_ADAPTER_H

#include "AquaBackendAdapter.h"
#include "GtsamMapAdapter.h"

#include <memory>
#include <functional>
#include <unordered_map>

namespace ORB_SLAM3
{

namespace IMU
{
class Calib;
}

struct GtsamOptionalSensorConfig
{
    bool dvlEnabled = false;
    cv::Mat bodyFromDvl;
    DvlTrackMode dvlTrackMode = DvlTrackMode::Unknown;
    std::string uncertaintyProfilePath;

    // Translates Tracking's sensor selection and registered IMU calibration
    // without exposing dynamic-backend types to the tracking boundary.
    static GtsamOptionalSensorConfig fromTrackingSensor(
        bool dvlStereoEnabled, const IMU::Calib* imuCalibration,
        DvlTrackMode dvlTrackMode = DvlTrackMode::Unknown,
        const std::string& uncertaintyProfilePath = std::string());
};

struct GtsamCameraPrediction
{
    bool accepted = false;
    cv::Mat cameraFromMap;
    std::string diagnostic;
};

struct GtsamInitialNavigationState
{
    std::array<double, 3> velocityMap{};
    std::array<double, 3> accelerometerBias{};
    std::array<double, 3> gyroscopeBias{};
};

class GtsamBackendAdapter;

struct GtsamInitializationResult
{
    std::shared_ptr<GtsamBackendAdapter> backend;
    BackendMapResult mapResult;
    std::array<double, 3> gravityMap{};
    std::string diagnostic;
};

// Pimpl keeps GTSAM headers and ownership entirely behind the AQUA boundary.
// The implementation is compiled only when UW_DYNAMIC_BACKEND_ROOT is set.
class GtsamPreparedMapOptimization
{
public:
    ~GtsamPreparedMapOptimization();
    const BackendMapResult& result() const;

private:
    friend class GtsamBackendAdapter;
    GtsamPreparedMapOptimization();
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

class GtsamBackendAdapter
{
public:
    GtsamBackendAdapter(const cv::Mat& cameraMatrix,
                        float baselineFx,
                        const cv::Mat& bodyFromCamera,
                        const cv::Mat& initialMapFromBody = cv::Mat(),
                        const cv::Mat& gravityMap = cv::Mat(),
                        const GtsamOptionalSensorConfig& optionalSensors =
                            GtsamOptionalSensorConfig(),
                        const GtsamInitialNavigationState& initialNavigation =
                            GtsamInitialNavigationState());
    ~GtsamBackendAdapter();

    GtsamBackendAdapter(GtsamBackendAdapter&&) noexcept;
    GtsamBackendAdapter& operator=(GtsamBackendAdapter&&) noexcept;
    GtsamBackendAdapter(const GtsamBackendAdapter&) = delete;
    GtsamBackendAdapter& operator=(const GtsamBackendAdapter&) = delete;

    BackendFrameResult optimize(const BackendFrameInput& input);
    static GtsamInitializationResult prepareInitialization(
        const BackendMapSnapshot& snapshot,
        const BackendFrameInput& sensorInput,
        const cv::Mat& bodyFromCamera,
        const GtsamOptionalSensorConfig& optionalSensors);
    // Atomically discard the incremental graph after an external map commit.
    // The next keyframe becomes a fresh graph anchor in the supplied map frame.
    void reset(const cv::Mat& initialMapFromBody = cv::Mat());
    bool rebase(const BackendMapSnapshot& snapshot);
    std::shared_ptr<GtsamPreparedMapOptimization> prepareMapOptimization(
        const BackendMapSnapshot& snapshot,
        const std::vector<BackendLandmarkReplacement>& replacements = {}) const;
    // Caller holds Tracking's map transaction. The callback runs only after
    // generation/revision checks, immediately before the prepared solver swap.
    bool commitMapOptimization(
        GtsamPreparedMapOptimization& candidate,
        const std::function<bool(const BackendMapResult&)>& commitMap);
    BackendFrameResult refinePose(const BackendFrameInput& input,
                                  const cv::Mat& initialCameraFromMap) const;
    static BackendFrameResult refineVisualPose(
        const BackendFrameInput& input, const cv::Mat& cameraMatrix,
        float baselineFx, const cv::Mat& bodyFromCamera,
        const cv::Mat& initialCameraFromMap);
    GtsamCameraPrediction predictCameraPose(
        double timestampSec, const std::vector<ImuSample>& imu) const;
    cv::Mat cameraPose(const BackendFrameResult& result) const;
    static cv::Mat gravityMapFromImu(
        const std::vector<ImuSample>& imu,
        const cv::Mat& mapFromImu);

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace ORB_SLAM3

#endif  // AQUA_GTSAM_BACKEND_ADAPTER_H
