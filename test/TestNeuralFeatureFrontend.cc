#include <gtest/gtest.h>

#include <cmath>
#include <set>

#include "NeuralFeatureFrontend.h"

namespace ORB_SLAM3
{
namespace
{

NeuralFeatureFrontend::Config ValidConfig()
{
    NeuralFeatureFrontend::Config config;
    config.superpoint_model =
        "/home/leong/UW_SLAM/work/onnxruntime_cuda/models/superpoint.onnx";
    config.superpoint_sha256 =
        "234d12c9f523292efb34e0ca513b011050b0c052700da9c01787b9356a1138d2";
    config.lightglue_model =
        "/home/leong/UW_SLAM/work/onnxruntime_cuda/models/superpoint_lightglue_fused.onnx";
    config.lightglue_sha256 =
        "8463182c165254b8cf182def813160691a5f3a455d95b4346e1b3ebf8a7709cf";
    config.projection_descriptor_threshold = 0.195335388F;
    return config;
}

TEST(NeuralFeatureFrontendTest, RejectsUnexpectedModelDigest)
{
    NeuralFeatureFrontend::Config config = ValidConfig();
    config.superpoint_sha256 = std::string(64, '0');

    EXPECT_THROW(NeuralFeatureFrontend frontend(config), std::runtime_error);
}

TEST(NeuralFeatureFrontendTest, RejectsInvalidKeypointThreshold)
{
    NeuralFeatureFrontend::Config config = ValidConfig();
    config.keypoint_threshold = 1.1F;

    EXPECT_THROW(NeuralFeatureFrontend frontend(config), std::runtime_error);
}

TEST(NeuralFeatureFrontendTest, RejectsInvalidProjectionDescriptorThreshold)
{
    NeuralFeatureFrontend::Config config = ValidConfig();
    config.projection_descriptor_threshold = 2.1F;
    EXPECT_THROW(NeuralFeatureFrontend frontend(config), std::runtime_error);

    config.projection_descriptor_threshold = 0.0F;
    EXPECT_THROW(NeuralFeatureFrontend frontend(config), std::runtime_error);
}

TEST(NeuralFeatureFrontendTest, ConstructsCudaSessionsWithPinnedModels)
{
    NeuralFeatureFrontend frontend(ValidConfig());

    EXPECT_EQ(frontend.Provider(), "CUDAExecutionProvider");
    EXPECT_FLOAT_EQ(frontend.ProjectionDescriptorThreshold(), 0.195335388F);
}

TEST(NeuralFeatureFrontendTest, ExtractsFiniteNormalizedSuperPointDescriptors)
{
    auto config = ValidConfig();
    config.max_features = 500;
    NeuralFeatureFrontend frontend(config);
    cv::Mat image(512, 612, CV_8UC1);
    cv::RNG random(7);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);

    const FeatureSet features = frontend.Extract(image);

    ASSERT_GT(features.keypoints.size(), 0U);
    ASSERT_LE(features.keypoints.size(), 500U);
    ASSERT_EQ(features.descriptors.rows,
              static_cast<int>(features.keypoints.size()));
    ASSERT_EQ(features.descriptors.cols, 256);
    ASSERT_EQ(features.descriptors.type(), CV_32F);
    for (int row = 0; row < features.descriptors.rows; ++row)
    {
        EXPECT_TRUE(cv::checkRange(features.descriptors.row(row)));
        EXPECT_NEAR(cv::norm(features.descriptors.row(row)), 1.0, 1e-4);
        EXPECT_GE(features.keypoints[row].pt.x, 0.0F);
        EXPECT_LT(features.keypoints[row].pt.x, image.cols);
        EXPECT_GE(features.keypoints[row].pt.y, 0.0F);
        EXPECT_LT(features.keypoints[row].pt.y, image.rows);
    }
}

TEST(NeuralFeatureFrontendTest, LightGlueMatchesIdenticalFeaturesOneToOne)
{
    auto config = ValidConfig();
    config.max_features = 300;
    NeuralFeatureFrontend frontend(config);
    cv::Mat image(512, 612, CV_8UC1);
    cv::RNG random(11);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    const FeatureSet features = frontend.Extract(image);

    const auto matches = frontend.Match(features, image.size(), features, image.size());

    ASSERT_FALSE(matches.empty());
    std::set<int> query_indices;
    std::set<int> train_indices;
    for (const auto& match : matches)
    {
        EXPECT_EQ(match.query_index, match.train_index);
        EXPECT_TRUE(std::isfinite(match.confidence));
        EXPECT_TRUE(query_indices.insert(match.query_index).second);
        EXPECT_TRUE(train_indices.insert(match.train_index).second);
    }
}

TEST(NeuralFeatureFrontendTest, LightGlueAcceptsFourFeaturesWithoutPaddedIndices)
{
    NeuralFeatureFrontend frontend(ValidConfig());
    FeatureSet features;
    features.keypoints = {
        cv::KeyPoint(10.0F, 10.0F, 1.0F),
        cv::KeyPoint(30.0F, 10.0F, 1.0F),
        cv::KeyPoint(10.0F, 30.0F, 1.0F),
        cv::KeyPoint(30.0F, 30.0F, 1.0F)};
    features.descriptors = cv::Mat::zeros(4, 256, CV_32F);
    for (int row = 0; row < 4; ++row)
        features.descriptors.at<float>(row, row) = 1.0F;

    const auto matches = frontend.Match(features, cv::Size(40, 40),
                                        features, cv::Size(40, 40));

    for (const auto& match : matches)
    {
        EXPECT_GE(match.query_index, 0);
        EXPECT_LT(match.query_index, 4);
        EXPECT_GE(match.train_index, 0);
        EXPECT_LT(match.train_index, 4);
    }
}

TEST(NeuralFeatureFrontendTest, RejectsInvalidLightGlueConfidence)
{
    EXPECT_FALSE(ORB_SLAM3::NeuralFeatureFrontend::IsUsableMatchConfidence(
        std::numeric_limits<float>::quiet_NaN(), 0.1F));
    EXPECT_FALSE(ORB_SLAM3::NeuralFeatureFrontend::IsUsableMatchConfidence(
        std::numeric_limits<float>::infinity(), 0.1F));
    EXPECT_FALSE(ORB_SLAM3::NeuralFeatureFrontend::IsUsableMatchConfidence(
        0.09F, 0.1F));
    EXPECT_TRUE(ORB_SLAM3::NeuralFeatureFrontend::IsUsableMatchConfidence(
        0.1F, 0.1F));
}

TEST(NeuralFeatureFrontendTest, RectifiedStereoRejectsNegativeDisparityAndVerticalOutlier)
{
    FeatureSet left;
    left.keypoints = {
        cv::KeyPoint(20.0F, 10.0F, 1.0F),
        cv::KeyPoint(20.0F, 20.0F, 1.0F),
        cv::KeyPoint(20.0F, 30.0F, 1.0F)};
    FeatureSet right;
    right.keypoints = {
        cv::KeyPoint(15.0F, 10.0F, 1.0F),
        cv::KeyPoint(25.0F, 20.0F, 1.0F),
        cv::KeyPoint(15.0F, 35.0F, 1.0F)};
    const std::vector<NeuralMatch> matches = {
        {0, 0, 0.9F}, {1, 1, 0.9F}, {2, 2, 0.9F}};

    const StereoGeometry stereo = NeuralFeatureFrontend::FilterRectifiedStereo(
        left, right, matches, 50.0F, 2.0F, 0.5F, 100.0F);

    ASSERT_EQ(stereo.right_u.size(), 3U);
    EXPECT_FLOAT_EQ(stereo.right_u[0], 15.0F);
    EXPECT_FLOAT_EQ(stereo.depth[0], 10.0F);
    EXPECT_FLOAT_EQ(stereo.right_u[1], -1.0F);
    EXPECT_FLOAT_EQ(stereo.depth[1], -1.0F);
    EXPECT_FLOAT_EQ(stereo.right_u[2], -1.0F);
    EXPECT_FLOAT_EQ(stereo.depth[2], -1.0F);
    ASSERT_EQ(stereo.matches.size(), 1U);
    EXPECT_EQ(stereo.matches[0].query_index, 0);
}

} // namespace
} // namespace ORB_SLAM3
