#ifndef AQUA_DYNAMIC_KEYFRAME_DVL_BUFFER_H
#define AQUA_DYNAMIC_KEYFRAME_DVL_BUFFER_H

#include "AquaBackendAdapter.h"

namespace ORB_SLAM3
{

class DynamicKeyframeDvlBuffer
{
public:
    bool append(const DvlMeasurement& measurement)
    {
        if (!std::isfinite(measurement.timestampSec) ||
            measurement.timestampSec <= lastTimestamp_)
            return false;
        samples_.push_back(measurement);
        lastTimestamp_ = measurement.timestampSec;
        return true;
    }

    std::vector<DvlMeasurement> pendingInterval(double timestamp) const
    {
        const auto end = std::upper_bound(
            samples_.begin(), samples_.end(), timestamp,
            [](double time, const DvlMeasurement& sample) {
                return time < sample.timestampSec;
            });
        return {samples_.begin(), end};
    }

    void commit(double timestamp)
    {
        const auto end = std::upper_bound(
            samples_.begin(), samples_.end(), timestamp,
            [](double time, const DvlMeasurement& sample) {
                return time < sample.timestampSec;
            });
        samples_.erase(samples_.begin(), end);
    }

    void clear()
    {
        samples_.clear();
        lastTimestamp_ = -std::numeric_limits<double>::infinity();
    }

private:
    std::vector<DvlMeasurement> samples_;
    double lastTimestamp_ = -std::numeric_limits<double>::infinity();
};

} // namespace ORB_SLAM3

#endif
