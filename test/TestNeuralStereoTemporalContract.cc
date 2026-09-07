#include <gtest/gtest.h>

#include <algorithm>

#include "Frame.h"
#include "MapPoint.h"
#include "NeuralFeatureFrontend.h"

namespace ORB_SLAM3
{
namespace
{

FeatureSet TranslatedFeatures(bool add_outlier)
{
    const std::vector<cv::Point2f> points = {
        {10.0F, 10.0F}, {20.0F, 10.0F}, {30.0F, 10.0F},
        {10.0F, 20.0F}, {20.0F, 20.0F}, {30.0F, 20.0F},
        {15.0F, 30.0F}, {25.0F, 30.0F}, {35.0F, 30.0F},
        {40.0F, 40.0F}, {50.0F, 15.0F}, {60.0F, 25.0F}};
    FeatureSet features;
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        cv::Point2f point = points[index] + cv::Point2f(4.0F, 0.0F);
        if (add_outlier && index == points.size() - 1)
            point.y += 25.0F;
        features.keypoints.emplace_back(point, 1.0F);
    }
    features.descriptors = cv::Mat::zeros(
        static_cast<int>(points.size()), 256, CV_32F);
    return features;
}

TEST(NeuralStereoTemporalContract, FundamentalRansacRejectsTemporalOutlier)
{
    FeatureSet previous = TranslatedFeatures(false);
    for (auto& keypoint : previous.keypoints)
        keypoint.pt.x -= 4.0F;
    FeatureSet current = TranslatedFeatures(true);
    std::vector<NeuralMatch> matches;
    for (int index = 0; index < static_cast<int>(previous.keypoints.size()); ++index)
        matches.push_back({index, index, 0.9F});

    const auto inliers = NeuralFeatureFrontend::FilterFundamentalInliers(
        previous, current, matches, 1.0, 0.999);

    EXPECT_GE(inliers.size(), 8U);
    EXPECT_EQ(std::count_if(inliers.begin(), inliers.end(), [](const auto& match) {
                  return match.query_index == 11;
              }), 0);
}

TEST(NeuralStereoTemporalContract, TransfersOnlyUniqueInlierMapPointIdentity)
{
    Frame previous;
    Frame current;
    MapPoint first;
    MapPoint second;
    MapPoint occupied;
    previous.mvpMapPoints = {&first, &second, nullptr};
    current.mvpMapPoints = {nullptr, &occupied, nullptr};
    const std::vector<NeuralMatch> inliers = {
        {0, 0, 0.9F}, {1, 1, 0.8F}, {2, 2, 0.7F}};

    const int transferred = current.TransferTemporalMapPoints(previous, inliers);

    EXPECT_EQ(transferred, 1);
    EXPECT_EQ(current.mvpMapPoints[0], &first);
    EXPECT_EQ(current.mvpMapPoints[1], &occupied);
    EXPECT_EQ(current.mvpMapPoints[2], nullptr);
}

TEST(NeuralStereoTemporalContract, DoesNotTransferOneMapPointTwice)
{
    Frame previous;
    Frame current;
    MapPoint shared;
    previous.mvpMapPoints = {&shared, &shared};
    current.mvpMapPoints = {nullptr, nullptr};
    const std::vector<NeuralMatch> inliers = {
        {0, 0, 0.9F}, {1, 1, 0.8F}};

    const int transferred = current.TransferTemporalMapPoints(previous, inliers);

    EXPECT_EQ(transferred, 1);
    EXPECT_EQ(current.mvpMapPoints[0], &shared);
    EXPECT_EQ(current.mvpMapPoints[1], nullptr);
}

TEST(NeuralStereoTemporalContract, DoesNotDuplicateAnExistingCurrentMapPoint)
{
    Frame previous;
    Frame current;
    MapPoint shared;
    previous.mvpMapPoints = {&shared};
    current.mvpMapPoints = {&shared, nullptr};
    const std::vector<NeuralMatch> inliers = {{0, 1, 0.9F}};

    const int transferred = current.TransferTemporalMapPoints(previous, inliers);

    EXPECT_EQ(transferred, 0);
    EXPECT_EQ(current.mvpMapPoints[0], &shared);
    EXPECT_EQ(current.mvpMapPoints[1], nullptr);
}

} // namespace
} // namespace ORB_SLAM3
