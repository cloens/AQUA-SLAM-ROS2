#ifndef INERTIAL_MATURITY_H
#define INERTIAL_MATURITY_H

namespace ORB_SLAM3
{

enum class InertialRefinementStage
{
    None,
    BA1,
    BA2,
};

inline InertialRefinementStage NextInertialRefinementStage(
    double elapsed_seconds, bool has_ba1, bool has_ba2)
{
    if (!has_ba1 && elapsed_seconds >= 5.0)
        return InertialRefinementStage::BA1;
    if (has_ba1 && !has_ba2 && elapsed_seconds >= 15.0)
        return InertialRefinementStage::BA2;
    return InertialRefinementStage::None;
}

}  // namespace ORB_SLAM3

#endif  // INERTIAL_MATURITY_H
