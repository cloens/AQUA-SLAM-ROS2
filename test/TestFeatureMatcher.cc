#include <gtest/gtest.h>

#include <limits>

#include "CameraModels/Pinhole.h"
#include "FeatureMatcher.h"
#include "NeuralFeatureFrontend.h"

namespace ORB_SLAM3
{
namespace
{

cv::Mat UnitDescriptor(int component, float sign = 1.0F)
{
    cv::Mat descriptor = cv::Mat::zeros(1, 256, CV_32F);
    descriptor.at<float>(0, component) = sign;
    return descriptor;
}

NeuralFeatureFrontend::Config ValidFrontendConfig()
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
    config.max_features = 100;
    return config;
}

TEST(FeatureMatcherTest, UsesNormalizedSuperPointCosineDistance)
{
    const cv::Mat first = UnitDescriptor(0);
    const cv::Mat same = UnitDescriptor(0);
    const cv::Mat orthogonal = UnitDescriptor(1);
    const cv::Mat opposite = UnitDescriptor(0, -1.0F);

    EXPECT_FLOAT_EQ(FeatureMatcher::DescriptorDistance(first, same), 0.0F);
    EXPECT_FLOAT_EQ(FeatureMatcher::DescriptorDistance(first, orthogonal), 1.0F);
    EXPECT_FLOAT_EQ(FeatureMatcher::DescriptorDistance(first, opposite), 2.0F);
}

TEST(FeatureMatcherTest, RejectsDescriptorsOutsideFloatNormalizedContract)
{
    const cv::Mat valid = UnitDescriptor(0);
    cv::Mat wrong_type = cv::Mat::zeros(1, 256, CV_8U);
    cv::Mat wrong_width = cv::Mat::zeros(1, 32, CV_32F);
    cv::Mat zero = cv::Mat::zeros(1, 256, CV_32F);
    cv::Mat non_finite = valid.clone();
    non_finite.at<float>(0, 3) = std::numeric_limits<float>::infinity();

    EXPECT_THROW(FeatureMatcher::DescriptorDistance(valid, wrong_type),
                 std::runtime_error);
    EXPECT_THROW(FeatureMatcher::DescriptorDistance(valid, wrong_width),
                 std::runtime_error);
    EXPECT_THROW(FeatureMatcher::DescriptorDistance(valid, zero),
                 std::runtime_error);
    EXPECT_THROW(FeatureMatcher::DescriptorDistance(valid, non_finite),
                 std::runtime_error);
}

TEST(FeatureMatcherTest, MatchesAfterKeyFrameImagePixelsAreReleased)
{
    NeuralFeatureFrontend frontend(ValidFrontendConfig());
    cv::Mat image(512, 612, CV_8UC1);
    cv::RNG random(17);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::Mat K = (cv::Mat_<float>(3, 3) <<
        655.0F, 0.0F, 306.0F,
        0.0F, 655.0F, 256.0F,
        0.0F, 0.0F, 1.0F);
    cv::Mat distortion = cv::Mat::zeros(4, 1, CV_32F);
    Pinhole camera({655.0F, 655.0F, 306.0F, 256.0F});
    IMU::Calib imuCalibration;
    Frame frame(image, image, 0.0, frontend, K, distortion,
                50.0F, 100.0F, 4.0F, 2.0F,
                &camera, false, nullptr, imuCalibration);
    frame.SetPose(cv::Mat::eye(4, 4, CV_32F));
    Map map(0);
    KeyFrame keyframe(frame, &map, nullptr);
    keyframe.imgLeft.release();
    std::vector<MapPoint*> matches;
    FeatureMatcher matcher(frontend);

    EXPECT_NO_THROW(matcher.SearchByNeuralPair(&keyframe, frame, matches));
}

TEST(FeatureMatcherTest, EmptyStereoFrameInitializesPoseAndImuState)
{
    auto config = ValidFrontendConfig();
    config.keypoint_threshold = 1.0F;
    NeuralFeatureFrontend frontend(config);
    const cv::Mat image = cv::Mat::zeros(512, 612, CV_8UC1);
    cv::Mat K = (cv::Mat_<float>(3, 3) <<
        655.0F, 0.0F, 306.0F,
        0.0F, 655.0F, 256.0F,
        0.0F, 0.0F, 1.0F);
    cv::Mat distortion = cv::Mat::zeros(4, 1, CV_32F);
    Pinhole camera({655.0F, 655.0F, 306.0F, 256.0F});
    IMU::Calib imuCalibration;
    for (float radialDistortion : {0.0F, 0.1F}) {
        SCOPED_TRACE(radialDistortion);
        distortion.at<float>(0) = radialDistortion;
        Frame frame(image, image, 0.0, frontend, K, distortion,
                    50.0F, 100.0F, 4.0F, 2.0F,
                    &camera, false, nullptr, imuCalibration);

        ASSERT_EQ(frame.N, 0);
        ASSERT_EQ(frame.mVw.rows, 3);
        ASSERT_EQ(frame.mVw.cols, 1);
        EXPECT_EQ(frame.Nleft, -1);
        EXPECT_EQ(frame.Nright, -1);
        EXPECT_EQ(frame.monoLeft, -1);
        EXPECT_EQ(frame.monoRight, -1);
        EXPECT_TRUE(frame.mvKeysUn.empty());
        EXPECT_TRUE(frame.mvpMapPoints.empty());
        EXPECT_TRUE(frame.mvbOutlier.empty());
        ASSERT_NE(frame.mpExtrinsic_mutex, nullptr);
        EXPECT_FALSE(frame.imuIsPreintegrated());
        frame.setIntegrated();
        EXPECT_TRUE(frame.imuIsPreintegrated());
        frame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        EXPECT_TRUE(cv::checkRange(frame.mTcw));
    }
}

} // namespace
} // namespace ORB_SLAM3
