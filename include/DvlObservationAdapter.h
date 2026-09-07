#ifndef DVL_OBSERVATION_ADAPTER_H
#define DVL_OBSERVATION_ADAPTER_H

#include <optional>

#include <uw_slam_bridge/msg/dvl_observation.hpp>

#include "ImuTypes.h"

namespace ORB_SLAM3
{

std::optional<IMU::GyroDvlPoint> fromDvlObservation(
    const uw_slam_bridge::msg::DvlObservation& message);

} // namespace ORB_SLAM3

#endif
