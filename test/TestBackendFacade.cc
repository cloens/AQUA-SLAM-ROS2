#include <gtest/gtest.h>

#include <stdexcept>

#include "BackendFacade.h"

namespace {

TEST(BackendFacade, RejectsUnknownModeBeforeRuntimeStarts)
{
    EXPECT_THROW(ORB_SLAM3::loadBackendMode("not-a-backend"), std::invalid_argument);
}

TEST(BackendFacade, SelectedModeCannotChange)
{
    ORB_SLAM3::BackendFacade facade(ORB_SLAM3::BackendMode::GtsamDynamic);
    EXPECT_EQ(facade.mode(), ORB_SLAM3::BackendMode::GtsamDynamic);
    EXPECT_THROW(facade.setMode(ORB_SLAM3::BackendMode::GtsamDynamic), std::logic_error);
    EXPECT_EQ(facade.mode(), ORB_SLAM3::BackendMode::GtsamDynamic);
}

TEST(BackendFacade, MissingDvlKeepsImuStereoMode)
{
    ORB_SLAM3::RuntimeProfile profile;
    profile.sensorMode = "stereo_inertial";
    profile.dvlEnabled = false;
    EXPECT_EQ(ORB_SLAM3::selectEstimatorMode(profile), ORB_SLAM3::EstimatorMode::ImuStereo);
}

TEST(BackendFacade, ExplicitDvlSelectsDvlStereoMode)
{
    ORB_SLAM3::RuntimeProfile profile;
    profile.sensorMode = "stereo_inertial_dvl";
    profile.dvlEnabled = true;
    EXPECT_EQ(ORB_SLAM3::selectEstimatorMode(profile), ORB_SLAM3::EstimatorMode::DvlStereo);
}

TEST(BackendFacade, OneBackendOwnsEveryOptimizationStage)
{
    using ORB_SLAM3::BackendMode;
    using ORB_SLAM3::OptimizationStage;
    for (const auto stage : {OptimizationStage::Frame,
                             OptimizationStage::Initialization,
                             OptimizationStage::LocalBundleAdjustment,
                             OptimizationStage::GlobalBundleAdjustment,
                             OptimizationStage::LoopCandidateVerification,
                             OptimizationStage::EssentialGraph}) {
        EXPECT_TRUE(ORB_SLAM3::usesDynamicBackend(
            BackendMode::GtsamDynamic, stage));
        EXPECT_TRUE(ORB_SLAM3::usesDynamicBackend(
            BackendMode::GtsamDynamic, stage));
    }
}

}  // namespace
