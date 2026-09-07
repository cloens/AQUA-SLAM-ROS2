#include <gtest/gtest.h>

#include "AquaBackendAdapter.h"
#include "DvlObservationAdapter.h"
#include "Frame.h"
#include "MapPoint.h"

#include <limits>

namespace {

TEST(AquaBackendAdapter, PreservesFiniteFrontEndPoseForUnderconstrainedFrame)
{
    ORB_SLAM3::Frame frame;
    frame.mTcw = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::BackendFrameResult result;
    result.diagnostic = "underconstrained stereo frame";

    EXPECT_TRUE(ORB_SLAM3::AquaBackendAdapter::canPreserveFrontEndPose(
        result, frame));
}

TEST(AquaBackendAdapter, RejectsUnderconstrainedFrameWithoutFiniteFrontEndPose)
{
    ORB_SLAM3::Frame frame;
    ORB_SLAM3::BackendFrameResult result;
    result.diagnostic = "underconstrained stereo frame";

    EXPECT_FALSE(ORB_SLAM3::AquaBackendAdapter::canPreserveFrontEndPose(
        result, frame));
}

TEST(AquaBackendAdapter, CountsOnlyPersistentLandmarksAsTrackingSupport)
{
    ORB_SLAM3::BackendFrameInput input;
    input.stereo.resize(12U);
    for (std::size_t index = 0U; index < 4U; ++index)
        input.stereo[index].hasLandmark = true;

    EXPECT_EQ(ORB_SLAM3::AquaBackendAdapter::persistentStereoSupport(input),
              4);
}

TEST(AquaBackendAdapter, AcceptsFiniteLargePoseCorrectionWithoutInlierGate)
{
    std::mutex poseMutex;
    ORB_SLAM3::Frame frame;
    frame.mpExtrinsic_mutex = &poseMutex;
    frame.mTcw = cv::Mat::eye(4, 4, CV_32F);
    ORB_SLAM3::BackendFrameResult result;
    result.accepted = true;
    result.bodyPose = frame.mTcw.clone();
    result.bodyPose.at<float>(0, 3) = 10.0F;
    const float angle = 0.8F;
    result.bodyPose.at<float>(0, 0) = std::cos(angle);
    result.bodyPose.at<float>(0, 1) = -std::sin(angle);
    result.bodyPose.at<float>(1, 0) = std::sin(angle);
    result.bodyPose.at<float>(1, 1) = std::cos(angle);

    ASSERT_TRUE(ORB_SLAM3::AquaBackendAdapter::applyFrameResult(result, frame));
    EXPECT_EQ(cv::norm(frame.mTcw, result.bodyPose, cv::NORM_INF), 0.0);
    result.bodyPose.at<float>(0, 3) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(ORB_SLAM3::AquaBackendAdapter::applyFrameResult(result, frame));
}

TEST(AquaBackendAdapter, SelectsSupportForThePoseRetainedByTracking)
{
    ORB_SLAM3::BackendFrameResult result;
    result.optimizedPoseInlierTrackIds = {1U, 2U, 3U};
    result.frontEndPoseInlierTrackIds = {4U, 5U};

    EXPECT_EQ(ORB_SLAM3::AquaBackendAdapter::trackingSupportAfterPoseCorrection(
                  false, false, result, 42),
              2);
    EXPECT_EQ(ORB_SLAM3::AquaBackendAdapter::trackingSupportAfterPoseCorrection(
                  false, true, result, 42),
              42);
    EXPECT_EQ(ORB_SLAM3::AquaBackendAdapter::trackingSupportAfterPoseCorrection(
                  true, false, result, 42),
              42);
}

TEST(AquaBackendAdapter, DoesNotUseRawAssociationsWithoutGeometricInliers)
{
    ORB_SLAM3::BackendFrameResult result;

    EXPECT_EQ(ORB_SLAM3::AquaBackendAdapter::trackingSupportAfterPoseCorrection(
                  false, false, result, 247),
              0);
}

TEST(AquaBackendAdapter, VisualSuccessRequiresActualPersistentObservations)
{
    ORB_SLAM3::BackendFrameResult result;
    result.accepted = true;
    EXPECT_EQ(ORB_SLAM3::AquaBackendAdapter::trackingSupportAfterPoseCorrection(
                  false, true, result, 1), 1);
    EXPECT_EQ(ORB_SLAM3::AquaBackendAdapter::trackingSupportAfterPoseCorrection(
                  false, true, result, 0), 0);
}

TEST(AquaBackendAdapter, PreservesTimestampCalibrationAndStereoTracks)
{
    ORB_SLAM3::Frame frame;
    frame.mTimeStamp = 12.5;
    frame.N = 1;
    frame.mvKeysUn.emplace_back(cv::Point2f(10.0F, 20.0F), 1.0F);
    frame.mvKeysUn.back().octave = 0;
    frame.mvLevelSigma2.push_back(4.0F);
    frame.mvuRight.push_back(8.0F);
    frame.mvDepth.push_back(2.0F);
    ORB_SLAM3::RuntimeProfile profile;
    profile.calibrationSha256 = "cave-calibration";

    const auto input = ORB_SLAM3::AquaBackendAdapter::fromFrame(frame, profile);
    EXPECT_DOUBLE_EQ(input.timestampSec, frame.mTimeStamp);
    ASSERT_EQ(input.stereo.size(), 1U);
    EXPECT_EQ(input.calibration.sha256, "cave-calibration");
    EXPECT_FLOAT_EQ(input.stereo.front().uRight, 8.0F);
    EXPECT_FLOAT_EQ(input.stereo.front().pyramidSigmaPx, 2.0F);
    EXPECT_DOUBLE_EQ(input.stereoCovariance[0], 4.0);
    EXPECT_DOUBLE_EQ(input.stereoCovariance[4], 4.0);
    EXPECT_DOUBLE_EQ(input.stereoCovariance[8], 4.0);
}

TEST(AquaBackendAdapter, CopiesRawDvlMeasurementIntoBackendInput)
{
    ORB_SLAM3::BackendFrameInput input;
    ORB_SLAM3::IMU::GyroDvlPoint measurement(
        0.0, 0.0, 0.0, 0.01, -0.02, 0.03,
        0.4, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 12.45);
    measurement.isDvlMeasurement = true;
    measurement.dvlHealthAccepted = true;
    measurement.dvlCovariance = {0.0004, 0.0, 0.0,
                                 0.0, 0.0004, 0.0,
                                 0.0, 0.0, 0.0004};
    measurement.dvlAngularVelocityBody = cv::Point3d(0.01, -0.02, 0.03);
    measurement.dvlTrackMode = ORB_SLAM3::DvlTrackMode::BottomTrack;
    measurement.dvlValidBeamRatio = 1.0;
    measurement.dvlErrorVelocityMetersPerSec = 0.02;

    ASSERT_TRUE(ORB_SLAM3::AquaBackendAdapter::applyDvlMeasurement(
        input, measurement));
    ASSERT_EQ(input.dvl.size(), 1U);
    EXPECT_DOUBLE_EQ(input.dvl.front().timestampSec, 12.45);
    EXPECT_DOUBLE_EQ(input.dvl.front().velocity[0], 0.4);
    EXPECT_DOUBLE_EQ(input.dvl.front().velocity[1], -0.2);
    EXPECT_DOUBLE_EQ(input.dvl.front().velocity[2], 0.1);
    EXPECT_DOUBLE_EQ(input.dvl.front().angularVelocityBody[0], 0.01);
    EXPECT_DOUBLE_EQ(input.dvl.front().covariance[0], 0.0004);
    EXPECT_TRUE(input.dvl.front().healthAccepted);
}

TEST(AquaBackendAdapter, AppendsEveryDvlMeasurementInAcquisitionOrder)
{
    ORB_SLAM3::BackendFrameInput input;
    ORB_SLAM3::IMU::GyroDvlPoint first(
        0.0, 0.0, 0.0, 0.01, -0.02, 0.03,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.40);
    ORB_SLAM3::IMU::GyroDvlPoint second = first;
    second.t = 12.45;
    second.v = cv::Point3d(2.0, 0.0, 0.0);
    for (auto* measurement : {&first, &second}) {
        measurement->isDvlMeasurement = true;
        measurement->dvlHealthAccepted = true;
        measurement->dvlCovariance = {0.04, 0.0, 0.0,
                                      0.0, 0.04, 0.0,
                                      0.0, 0.0, 0.04};
        measurement->dvlTrackMode = ORB_SLAM3::DvlTrackMode::BottomTrack;
        measurement->dvlValidBeamRatio = 1.0;
        measurement->dvlErrorVelocityMetersPerSec = 0.02;
    }

    ASSERT_TRUE(ORB_SLAM3::AquaBackendAdapter::applyDvlMeasurement(input, first));
    ASSERT_TRUE(ORB_SLAM3::AquaBackendAdapter::applyDvlMeasurement(input, second));

    ASSERT_EQ(input.dvl.size(), 2U);
    EXPECT_DOUBLE_EQ(input.dvl[0].timestampSec, 12.40);
    EXPECT_DOUBLE_EQ(input.dvl[1].timestampSec, 12.45);
    EXPECT_DOUBLE_EQ(input.dvl[0].velocity[0], 1.0);
    EXPECT_DOUBLE_EQ(input.dvl[1].velocity[0], 2.0);
}

TEST(AquaBackendAdapter, PreservesDvlQualityEvidenceAtBackendBoundary)
{
    ORB_SLAM3::BackendFrameInput input;
    ORB_SLAM3::IMU::GyroDvlPoint measurement(
        0.0, 0.0, 0.0, 0.01, -0.02, 0.03,
        0.4, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 12.45);
    measurement.isDvlMeasurement = true;
    measurement.dvlHealthAccepted = true;
    measurement.dvlCovariance = {0.0004, 0.0, 0.0,
                                 0.0, 0.0004, 0.0,
                                 0.0, 0.0, 0.0004};
    measurement.dvlTrackMode = ORB_SLAM3::DvlTrackMode::BottomTrack;
    measurement.dvlValidBeamRatio = 0.75;
    measurement.dvlAltitudeValid = true;
    measurement.dvlAltitudeMeters = 1.25;
    measurement.dvlErrorVelocityMetersPerSec = 0.02;
    measurement.dvlStatus = 0;

    ASSERT_TRUE(ORB_SLAM3::AquaBackendAdapter::applyDvlMeasurement(
        input, measurement));
    ASSERT_EQ(input.dvl.size(), 1U);
    EXPECT_DOUBLE_EQ(input.dvl.front().timestampSec, 12.45);
    EXPECT_EQ(input.dvl.front().trackMode, ORB_SLAM3::DvlTrackMode::BottomTrack);
    EXPECT_DOUBLE_EQ(input.dvl.front().validBeamRatio, 0.75);
    EXPECT_TRUE(input.dvl.front().altitudeValid);
    EXPECT_DOUBLE_EQ(input.dvl.front().altitudeMeters, 1.25);
    EXPECT_DOUBLE_EQ(input.dvl.front().errorVelocityMetersPerSec, 0.02);
    EXPECT_EQ(input.dvl.front().status, 0);
}

TEST(AquaBackendAdapter, RejectsIncompleteDvlQualityEvidence)
{
    ORB_SLAM3::BackendFrameInput input;
    ORB_SLAM3::IMU::GyroDvlPoint measurement(
        0.0, 0.0, 0.0, 0.01, -0.02, 0.03,
        0.4, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 12.45);
    measurement.isDvlMeasurement = true;
    measurement.dvlHealthAccepted = true;
    measurement.dvlCovariance = {0.0004, 0.0, 0.0,
                                 0.0, 0.0004, 0.0,
                                 0.0, 0.0, 0.0004};
    measurement.dvlTrackMode = ORB_SLAM3::DvlTrackMode::BottomTrack;
    measurement.dvlAltitudeValid = true;
    measurement.dvlAltitudeMeters = 1.25;
    measurement.dvlErrorVelocityMetersPerSec = 0.02;

    EXPECT_FALSE(ORB_SLAM3::AquaBackendAdapter::applyDvlMeasurement(
        input, measurement));
    EXPECT_TRUE(input.dvl.empty());
}

TEST(DvlObservationAdapter, PreservesCompleteNormalizedObservation)
{
    uw_slam_bridge::msg::DvlObservation message;
    message.header.stamp.sec = 12;
    message.header.stamp.nanosec = 450000000U;
    message.velocity.x = 0.4;
    message.velocity.y = -0.2;
    message.velocity.z = 0.1;
    message.velocity_valid = true;
    message.covariance_valid = true;
    message.covariance = {0.0004, 0.0, 0.0,
                          0.0, 0.0004, 0.0,
                          0.0, 0.0, 0.0004};
    message.track_mode = message.TRACK_MODE_BOTTOM;
    message.valid_beam_ratio = 0.75;
    message.altitude_valid = true;
    message.altitude = 1.25;
    message.error_velocity = 0.02;
    message.status = 7;

    const auto measurement = ORB_SLAM3::fromDvlObservation(message);

    ASSERT_TRUE(measurement.has_value());
    EXPECT_DOUBLE_EQ(measurement->t, 12.45);
    EXPECT_DOUBLE_EQ(measurement->v.x, 0.4);
    EXPECT_DOUBLE_EQ(measurement->v.y, -0.2);
    EXPECT_DOUBLE_EQ(measurement->v.z, 0.1);
    EXPECT_TRUE(measurement->isDvlMeasurement);
    EXPECT_TRUE(measurement->dvlHealthAccepted);
    EXPECT_DOUBLE_EQ(measurement->dvlCovariance[0], 0.0004);
    EXPECT_EQ(measurement->dvlTrackMode,
              ORB_SLAM3::DvlTrackMode::BottomTrack);
    EXPECT_DOUBLE_EQ(measurement->dvlValidBeamRatio, 0.75);
    EXPECT_TRUE(measurement->dvlAltitudeValid);
    EXPECT_DOUBLE_EQ(measurement->dvlAltitudeMeters, 1.25);
    EXPECT_DOUBLE_EQ(measurement->dvlErrorVelocityMetersPerSec, 0.02);
    EXPECT_EQ(measurement->dvlStatus, 7);
}

TEST(DvlObservationAdapter, PreservesRejectedSourceObservation)
{
    uw_slam_bridge::msg::DvlObservation message;
    message.header.stamp.sec = 12;
    message.velocity_valid = false;
    message.covariance_valid = true;
    message.covariance = {0.0004, 0.0, 0.0,
                          0.0, 0.0004, 0.0,
                          0.0, 0.0, 0.0004};
    message.track_mode = message.TRACK_MODE_BOTTOM;
    message.valid_beam_ratio = 0.5;
    message.altitude_valid = false;
    message.altitude = -1.0;
    message.error_velocity = 0.02;

    const auto measurement = ORB_SLAM3::fromDvlObservation(message);

    ASSERT_TRUE(measurement.has_value());
    EXPECT_FALSE(measurement->dvlHealthAccepted);
}

TEST(DvlObservationAdapter, PreservesValidJsonV1WithoutInventingCovariance)
{
    uw_slam_bridge::msg::DvlObservation message;
    message.header.stamp.sec = 12;
    message.velocity.x = 0.4;
    message.velocity_valid = true;
    message.covariance_valid = false;
    message.covariance.fill(0.0);
    message.track_mode = message.TRACK_MODE_BOTTOM;
    message.valid_beam_ratio = 1.0;
    message.altitude_valid = true;
    message.altitude = 1.25;
    message.error_velocity = 0.02;

    const auto measurement = ORB_SLAM3::fromDvlObservation(message);

    ASSERT_TRUE(measurement.has_value());
    EXPECT_TRUE(measurement->dvlHealthAccepted);
    EXPECT_TRUE(std::all_of(measurement->dvlCovariance.begin(),
                            measurement->dvlCovariance.end(),
                            [](double value) { return value == 0.0; }));
}

TEST(AquaBackendAdapter, ExcludesUnmatchedFeaturesFromStereoFactors)
{
    ORB_SLAM3::Frame frame;
    frame.mnId = 4;
    frame.mTimeStamp = 12.5;
    frame.N = 2;
    frame.mvKeysUn.emplace_back(cv::Point2f(10.0F, 20.0F), 1.0F);
    frame.mvKeysUn.emplace_back(cv::Point2f(11.0F, 21.0F), 1.0F);
    frame.mvuRight = {8.0F, -1.0F};
    frame.mvDepth = {2.0F, -1.0F};
    ORB_SLAM3::RuntimeProfile profile;

    const auto input = ORB_SLAM3::AquaBackendAdapter::fromFrame(frame, profile);

    ASSERT_EQ(input.stereo.size(), 1U);
    EXPECT_FLOAT_EQ(input.stereo.front().uRight, 8.0F);
    EXPECT_FLOAT_EQ(input.stereo.front().depth, 2.0F);
}

TEST(AquaBackendAdapter, PreservesPersistentLandmarkWithoutStereoDepthForPoseRefinement)
{
    ORB_SLAM3::MapPoint mapPoint;
    mapPoint.mnId = 37;
    mapPoint.SetWorldPos((cv::Mat_<float>(3, 1) << 1.0F, 2.0F, 3.0F));
    ORB_SLAM3::Frame frame;
    frame.mnId = 4;
    frame.mTimeStamp = 12.5;
    frame.N = 1;
    frame.mvKeysUn.emplace_back(cv::Point2f(10.0F, 20.0F), 1.0F);
    frame.mvuRight = {-1.0F};
    frame.mvDepth = {-1.0F};
    frame.mvpMapPoints = {&mapPoint};
    ORB_SLAM3::RuntimeProfile profile;

    const auto input = ORB_SLAM3::AquaBackendAdapter::fromFrame(frame, profile);

    EXPECT_TRUE(input.stereo.empty());
    ASSERT_EQ(input.monocularLandmarks.size(), 1U);
    EXPECT_EQ(input.monocularLandmarks.front().trackId, 37U);
    EXPECT_FLOAT_EQ(input.monocularLandmarks.front().u, 10.0F);
    EXPECT_FLOAT_EQ(input.monocularLandmarks.front().v, 20.0F);
}

TEST(AquaBackendAdapter, RejectsNonFinitePoseResult)
{
    ORB_SLAM3::Frame frame;
    ORB_SLAM3::BackendFrameResult result;
    result.accepted = true;
    result.bodyPose = cv::Mat::eye(4, 4, CV_32F);
    result.bodyPose.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(ORB_SLAM3::AquaBackendAdapter::applyFrameResult(result, frame));
}

TEST(AquaBackendAdapter, UsesPersistentMapPointIdentityAcrossFrames)
{
    ORB_SLAM3::MapPoint mapPoint;
    mapPoint.mnId = 37;
    mapPoint.SetWorldPos((cv::Mat_<float>(3, 1) << 1.0F, 2.0F, 3.0F));
    ORB_SLAM3::RuntimeProfile profile;

    ORB_SLAM3::Frame first;
    first.mnId = 10;
    first.mTimeStamp = 1.0;
    first.N = 1;
    first.mvKeysUn.emplace_back(cv::Point2f(10.0F, 20.0F), 1.0F);
    first.mvuRight.push_back(8.0F);
    first.mvDepth.push_back(2.0F);
    first.mvpMapPoints.push_back(&mapPoint);
    ORB_SLAM3::Frame second = first;
    second.mnId = 11;
    second.mTimeStamp = 2.0;

    const auto firstInput =
        ORB_SLAM3::AquaBackendAdapter::fromFrame(first, profile);
    const auto secondInput =
        ORB_SLAM3::AquaBackendAdapter::fromFrame(second, profile);

    EXPECT_EQ(firstInput.stereo.front().trackId, 37U);
    EXPECT_EQ(secondInput.stereo.front().trackId, 37U);
}

TEST(AquaBackendAdapter, DoesNotReuseUnassociatedFeatureIdentityAcrossFrames)
{
    ORB_SLAM3::RuntimeProfile profile;
    ORB_SLAM3::Frame first;
    first.mnId = 10;
    first.mTimeStamp = 1.0;
    first.N = 1;
    first.mvKeysUn.emplace_back(cv::Point2f(10.0F, 20.0F), 1.0F);
    first.mvuRight.push_back(8.0F);
    first.mvDepth.push_back(2.0F);
    ORB_SLAM3::Frame second = first;
    second.mnId = 11;
    second.mTimeStamp = 2.0;

    const auto firstInput =
        ORB_SLAM3::AquaBackendAdapter::fromFrame(first, profile);
    const auto secondInput =
        ORB_SLAM3::AquaBackendAdapter::fromFrame(second, profile);

    EXPECT_NE(firstInput.stereo.front().trackId,
              secondInput.stereo.front().trackId);
}

TEST(AquaBackendAdapter, DoesNotTreatPhysicalAccelerationAsMeasurementNoise)
{
    ORB_SLAM3::IMU::Calib calibration;
    calibration.Set(cv::Mat::eye(4, 4, CV_32F),
                    0.1F, 0.2F, 0.003F, 0.004F);
    ORB_SLAM3::BackendFrameInput stable;
    for (int index = 0; index < 6; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.003 * static_cast<double>(index);
        sample.acceleration = {0.0, 0.0, 9.81};
        sample.angularVelocity = {0.01, 0.01, 0.01};
        stable.imu.push_back(sample);
    }
    ORB_SLAM3::BackendFrameInput vibrating = stable;
    for (std::size_t index = 0; index < vibrating.imu.size(); ++index)
        vibrating.imu[index].acceleration[0] =
            index % 2U == 0U ? -2.0 : 2.0;

    ASSERT_TRUE(ORB_SLAM3::AquaBackendAdapter::applyDynamicImuCovariance(
        stable, calibration, 333.0));
    ASSERT_TRUE(ORB_SLAM3::AquaBackendAdapter::applyDynamicImuCovariance(
        vibrating, calibration, 333.0));

    EXPECT_EQ(vibrating.imuAccelerometerCovariance,
              stable.imuAccelerometerCovariance);
    EXPECT_DOUBLE_EQ(vibrating.imuGyroscopeCovariance[0],
                     stable.imuGyroscopeCovariance[0]);
    EXPECT_GT(stable.imuAccelerometerCovariance[0], 0.0);
    EXPECT_GT(stable.imuGyroscopeCovariance[0], 0.0);
    EXPECT_GT(stable.imuAccelerometerBiasCovariance[0], 0.0);
    EXPECT_GT(stable.imuGyroscopeBiasCovariance[0], 0.0);
}

TEST(AquaBackendAdapter, RecoversContinuousImuNoiseAtDifferentSamplingRates)
{
    for (const float frequency : {100.0F, 400.0F}) {
        const float scale = std::sqrt(frequency);
        ORB_SLAM3::IMU::Calib calibration;
        calibration.Set(cv::Mat::eye(4, 4, CV_32F),
                        0.01F * scale, 0.2F * scale,
                        0.003F / scale, 0.004F / scale);
        ORB_SLAM3::BackendFrameInput input;

        ASSERT_TRUE(ORB_SLAM3::AquaBackendAdapter::applyDynamicImuCovariance(
            input, calibration, frequency));

        EXPECT_TRUE(input.hasImuCovariance);
        for (std::size_t index = 0U; index < 9U; ++index) {
            const bool diagonal = index % 4U == 0U;
            EXPECT_NEAR(input.imuGyroscopeCovariance[index],
                        diagonal ? 0.0001 : 0.0, 1e-10);
            EXPECT_NEAR(input.imuAccelerometerCovariance[index],
                        diagonal ? 0.04 : 0.0, 1e-8);
            EXPECT_NEAR(input.imuGyroscopeBiasCovariance[index],
                        diagonal ? 0.000009 : 0.0, 1e-11);
            EXPECT_NEAR(input.imuAccelerometerBiasCovariance[index],
                        diagonal ? 0.000016 : 0.0, 1e-11);
        }
    }
}

TEST(AquaBackendAdapter, RejectsInvalidImuCalibrationWithoutKeepingOldNoise)
{
    ORB_SLAM3::IMU::Calib calibration;
    calibration.Set(cv::Mat::eye(4, 4, CV_32F),
                    0.1F, 0.2F, 0.003F, 0.004F);
    ORB_SLAM3::BackendFrameInput input;
    for (const double frequency : {0.0, -1.0,
                                  std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::infinity()}) {
        input.hasImuCovariance = true;
        EXPECT_FALSE(ORB_SLAM3::AquaBackendAdapter::applyDynamicImuCovariance(
            input, calibration, frequency));
        EXPECT_FALSE(input.hasImuCovariance);
    }
    for (const float invalid : {0.0F, -1.0F,
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity()}) {
        calibration.CovWalk.at<float>(0, 0) = invalid;
        input.hasImuCovariance = true;
        EXPECT_FALSE(ORB_SLAM3::AquaBackendAdapter::applyDynamicImuCovariance(
            input, calibration, 100.0));
        EXPECT_FALSE(input.hasImuCovariance);
    }
    calibration.Set(cv::Mat::eye(4, 4, CV_32F),
                    0.1F, 0.2F, 0.003F, 0.004F);
    calibration.Cov.at<float>(0, 1) = 0.001F;
    EXPECT_FALSE(ORB_SLAM3::AquaBackendAdapter::applyDynamicImuCovariance(
        input, calibration, 100.0));
    calibration.Cov.release();
    EXPECT_FALSE(ORB_SLAM3::AquaBackendAdapter::applyDynamicImuCovariance(
        input, calibration, 100.0));
}

}  // namespace
