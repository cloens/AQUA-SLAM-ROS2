#include "DvlObservationAdapter.h"

#include <algorithm>
#include <cmath>

namespace ORB_SLAM3
{

std::optional<IMU::GyroDvlPoint> fromDvlObservation(
    const uw_slam_bridge::msg::DvlObservation& message)
{
    DvlTrackMode trackMode = DvlTrackMode::Unknown;
    if (message.track_mode == message.TRACK_MODE_BOTTOM)
        trackMode = DvlTrackMode::BottomTrack;
    else if (message.track_mode == message.TRACK_MODE_WATER)
        trackMode = DvlTrackMode::WaterTrack;

    const double timestamp = static_cast<double>(message.header.stamp.sec) +
        static_cast<double>(message.header.stamp.nanosec) * 1e-9;
    if (trackMode == DvlTrackMode::Unknown || !std::isfinite(timestamp) ||
        !std::isfinite(message.velocity.x) ||
        !std::isfinite(message.velocity.y) ||
        !std::isfinite(message.velocity.z) ||
        !std::all_of(message.covariance.begin(), message.covariance.end(),
                     [](double value) { return std::isfinite(value); }) ||
        !std::isfinite(message.valid_beam_ratio) ||
        message.valid_beam_ratio < 0.0 || message.valid_beam_ratio > 1.0 ||
        !std::isfinite(message.altitude) ||
        (message.altitude_valid && message.altitude < 0.0) ||
        !std::isfinite(message.error_velocity) || message.error_velocity < 0.0)
        return std::nullopt;

    IMU::GyroDvlPoint measurement(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        message.velocity.x, message.velocity.y, message.velocity.z,
        0.0, 0.0, 0.0, 0.0, timestamp);
    measurement.isDvlMeasurement = true;
    measurement.dvlHealthAccepted = message.velocity_valid;
    measurement.dvlCovariance = message.covariance;
    measurement.dvlTrackMode = trackMode;
    measurement.dvlValidBeamRatio = message.valid_beam_ratio;
    measurement.dvlAltitudeValid = message.altitude_valid;
    measurement.dvlAltitudeMeters = message.altitude;
    measurement.dvlErrorVelocityMetersPerSec = message.error_velocity;
    measurement.dvlStatus = message.status;
    return measurement;
}

} // namespace ORB_SLAM3
