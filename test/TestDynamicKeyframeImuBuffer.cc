#include <gtest/gtest.h>

#include "AquaBackendAdapter.h"
#include "DynamicKeyframeDvlBuffer.h"

#include <limits>

namespace {

TEST(DynamicKeyframeDvlBuffer, PreservesAllSamplesUntilSuccessfulCommit)
{
    ORB_SLAM3::DynamicKeyframeDvlBuffer buffer;
    ORB_SLAM3::DvlMeasurement observation;
    for (double timestamp : {0.0, 0.1, 0.2, 0.3}) {
        observation.timestampSec = timestamp;
        observation.velocity[0] = timestamp * 10.0;
        ASSERT_TRUE(buffer.append(observation));
    }
    const auto firstAttempt = buffer.pendingInterval(0.2);
    ASSERT_EQ(firstAttempt.size(), 3U);
    EXPECT_DOUBLE_EQ(firstAttempt.back().velocity[0], 2.0);
    EXPECT_EQ(buffer.pendingInterval(0.3).size(), 4U);
    buffer.commit(0.2);
    const auto next = buffer.pendingInterval(0.3);
    ASSERT_EQ(next.size(), 1U);
    EXPECT_DOUBLE_EQ(next.front().timestampSec, 0.3);
    EXPECT_FALSE(buffer.append(observation));
    observation.timestampSec = 0.1;
    EXPECT_FALSE(buffer.append(observation));
    buffer.clear();
    EXPECT_TRUE(buffer.pendingInterval(1.0).empty());
    EXPECT_TRUE(buffer.append(observation));
}

ORB_SLAM3::ImuSample sample(double timestamp)
{
    ORB_SLAM3::ImuSample result;
    result.timestampSec = timestamp;
    return result;
}

TEST(DynamicKeyframeImuBuffer, MergesFramesAndKeepsOneRightBracket)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer buffer;
    buffer.append({sample(0.0), sample(0.01)});
    buffer.append({sample(0.01), sample(0.02), sample(0.025), sample(0.03)});

    const auto interval = buffer.interval(0.0, 0.02);

    ASSERT_EQ(interval.size(), 4U);
    EXPECT_DOUBLE_EQ(interval[0].timestampSec, 0.0);
    EXPECT_DOUBLE_EQ(interval[1].timestampSec, 0.01);
    EXPECT_DOUBLE_EQ(interval[2].timestampSec, 0.02);
    EXPECT_DOUBLE_EQ(interval[3].timestampSec, 0.025);
}

TEST(DynamicKeyframeImuBuffer, SuccessfulCommitRetainsBoundaryAndFuture)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer buffer;
    buffer.append({sample(0.0), sample(0.01), sample(0.02)});

    buffer.commit(0.01);

    ASSERT_EQ(buffer.samples().size(), 2U);
    EXPECT_DOUBLE_EQ(buffer.samples().front().timestampSec, 0.01);
}

TEST(DynamicKeyframeImuBuffer, RetainsPhysicalBracketsAroundCameraBoundary)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer buffer;
    buffer.append({sample(0.009), sample(0.011), sample(0.019), sample(0.021)});

    buffer.commit(0.015);
    const auto interval = buffer.interval(0.015, 0.020);

    ASSERT_EQ(interval.size(), 3U);
    EXPECT_DOUBLE_EQ(interval[0].timestampSec, 0.011);
    EXPECT_DOUBLE_EQ(interval[1].timestampSec, 0.019);
    EXPECT_DOUBLE_EQ(interval[2].timestampSec, 0.021);
}

TEST(DynamicKeyframeImuBuffer, FailedCommitDoesNotConsumeSamples)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer buffer;
    buffer.append({sample(0.0), sample(0.01), sample(0.02)});

    const auto before = buffer.samples();

    EXPECT_EQ(buffer.samples().size(), before.size());
    EXPECT_DOUBLE_EQ(buffer.samples().front().timestampSec,
                     before.front().timestampSec);
}

TEST(DynamicKeyframeImuBuffer, ResetClearsAllState)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer buffer;
    buffer.append({sample(0.0), sample(0.01)});

    buffer.clear();

    EXPECT_TRUE(buffer.samples().empty());
    EXPECT_FALSE(buffer.hasAcceptedKeyframe());
}

TEST(DynamicKeyframeImuBuffer, AdvancesOnlyWhenKeyframeIsAccepted)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer buffer;
    buffer.append({sample(0.0), sample(0.01), sample(0.02), sample(0.03)});

    EXPECT_TRUE(buffer.pendingInterval(0.01).empty());
    buffer.acceptKeyframe(0.01);
    const auto rejectedCandidate = buffer.pendingInterval(0.02);
    const auto laterCandidate = buffer.pendingInterval(0.03);

    ASSERT_TRUE(buffer.hasAcceptedKeyframe());
    EXPECT_DOUBLE_EQ(buffer.lastAcceptedKeyframeTime(), 0.01);
    ASSERT_FALSE(rejectedCandidate.empty());
    ASSERT_FALSE(laterCandidate.empty());
    EXPECT_DOUBLE_EQ(rejectedCandidate.front().timestampSec,
                     laterCandidate.front().timestampSec);
}

TEST(DynamicKeyframeImuBuffer, RejectsIntervalsMissingEitherPhysicalBracket)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer missingLeft;
    missingLeft.append({sample(0.011), sample(0.019), sample(0.021)});
    missingLeft.acceptKeyframe(0.01);
    EXPECT_TRUE(missingLeft.pendingInterval(0.02).empty());

    ORB_SLAM3::DynamicKeyframeImuBuffer missingRight;
    missingRight.append({sample(0.009), sample(0.011), sample(0.019)});
    missingRight.acceptKeyframe(0.01);
    EXPECT_TRUE(missingRight.pendingInterval(0.02).empty());
}

TEST(DynamicKeyframeImuBuffer, RetainsIntervalsWithInternalSamplingGap)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer buffer;
    buffer.append({sample(0.0), sample(0.01), sample(0.25), sample(0.26)});
    buffer.acceptKeyframe(0.0);

    EXPECT_EQ(buffer.pendingInterval(0.26).size(), 4U);
}

TEST(DynamicKeyframeImuBuffer, PreservesPhysicalBracketsAcrossInternalGap)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer buffer;
    buffer.append({sample(0.0), sample(0.01), sample(0.25), sample(0.27)});
    buffer.acceptKeyframe(0.0);

    EXPECT_EQ(buffer.pendingInterval(0.26).size(), 4U);
    EXPECT_TRUE(buffer.hasPhysicalBrackets(0.26));
}

TEST(DynamicKeyframeImuBuffer, DiscardsNonFinitePhysicalSamples)
{
    ORB_SLAM3::DynamicKeyframeImuBuffer buffer;
    auto invalidAcceleration = sample(0.01);
    invalidAcceleration.acceleration[1] =
        std::numeric_limits<double>::quiet_NaN();
    auto invalidGyroscope = sample(0.02);
    invalidGyroscope.angularVelocity[2] =
        std::numeric_limits<double>::infinity();

    buffer.append({sample(0.0), invalidAcceleration, invalidGyroscope,
                   sample(0.03)});

    ASSERT_EQ(buffer.samples().size(), 2U);
    EXPECT_DOUBLE_EQ(buffer.samples().front().timestampSec, 0.0);
    EXPECT_DOUBLE_EQ(buffer.samples().back().timestampSec, 0.03);
}

}  // namespace
