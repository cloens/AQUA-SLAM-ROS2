#include "GtsamBackendAdapter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <thread>
#include <vector>

namespace
{
cv::Mat cameraMatrix()
{
    return (cv::Mat_<float>(3, 3) << 456.0F, 0.0F, 320.0F,
            0.0F, 456.0F, 240.0F, 0.0F, 0.0F, 1.0F);
}

ORB_SLAM3::BackendFrameInput bootstrapInput()
{
    ORB_SLAM3::BackendFrameInput input;
    input.timestampSec = 0.0;
    for (std::uint64_t index = 0; index < 6U; ++index) {
        ORB_SLAM3::StereoTrackObservation observation;
        observation.trackId = index;
        observation.uLeft = 100.0F + static_cast<float>(index * 12U);
        observation.uRight = observation.uLeft - 20.0F;
        observation.vLeft = 150.0F + static_cast<float>(index * 8U);
        input.stereo.push_back(observation);
    }
    return input;
}
}

TEST(GtsamBackendConcurrency, ConcurrentIdenticalKeyframesCommitExactlyOnce)
{
    ORB_SLAM3::GtsamBackendAdapter backend(
        cameraMatrix(), 63.0F, cv::Mat::eye(4, 4, CV_32F));
    const auto input = bootstrapInput();
    std::promise<void> release;
    const auto start = release.get_future().share();
    std::atomic<unsigned> ready{0U};
    std::vector<std::future<ORB_SLAM3::BackendFrameResult>> workers;
    for (unsigned index = 0; index < 8U; ++index) {
        workers.push_back(std::async(std::launch::async, [&] {
            ++ready;
            start.wait();
            return backend.optimize(input);
        }));
    }
    while (ready.load() != workers.size())
        std::this_thread::yield();
    release.set_value();
    unsigned committed = 0U;
    for (auto& worker : workers) {
        const auto result = worker.get();
        EXPECT_TRUE(result.accepted) << result.diagnostic;
        EXPECT_EQ(result.mapVersion, 1U);
        committed += result.incrementalCommitted ? 1U : 0U;
    }
    EXPECT_EQ(committed, 1U);
}

TEST(GtsamBackendConcurrency, RebaseAndPredictionPreserveCommittedPose)
{
    ORB_SLAM3::GtsamBackendAdapter backend(
        cameraMatrix(), 63.0F, cv::Mat::eye(4, 4, CV_32F));
    const auto result = backend.optimize(bootstrapInput());
    ASSERT_TRUE(result.incrementalCommitted);
    ORB_SLAM3::BackendMapSnapshot snapshot;
    ORB_SLAM3::BackendKeyframeState keyframe;
    keyframe.id = 0U;
    keyframe.timestampSec = 0.0;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            keyframe.pose[static_cast<std::size_t>(row * 4 + col)] =
                result.bodyPose.at<float>(row, col);
    snapshot.keyframes.push_back(keyframe);
    std::vector<ORB_SLAM3::ImuSample> imu;
    for (int index = 0; index <= 10; ++index) {
        ORB_SLAM3::ImuSample sample;
        sample.timestampSec = 0.01 * static_cast<double>(index);
        sample.acceleration = {0.0, 0.0, 9.81};
        imu.push_back(sample);
    }
    const auto expected = backend.predictCameraPose(0.1, imu);
    ASSERT_TRUE(expected.accepted) << expected.diagnostic;
    std::promise<void> release;
    const auto start = release.get_future().share();
    auto rebasing = std::async(std::launch::async, [&] {
        start.wait();
        for (unsigned index = 0; index < 100U; ++index)
            EXPECT_TRUE(backend.rebase(snapshot));
    });
    auto predicting = std::async(std::launch::async, [&] {
        start.wait();
        for (unsigned index = 0; index < 100U; ++index) {
            const auto prediction = backend.predictCameraPose(0.1, imu);
            EXPECT_TRUE(prediction.accepted) << prediction.diagnostic;
            if (!prediction.accepted)
                continue;
            EXPECT_TRUE(cv::checkRange(prediction.cameraFromMap));
            EXPECT_LT(cv::norm(prediction.cameraFromMap -
                               expected.cameraFromMap), 1e-5);
        }
    });
    release.set_value();
    rebasing.get();
    predicting.get();
}
