#ifndef NEURAL_FEATURE_FRONTEND_H
#define NEURAL_FEATURE_FRONTEND_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace ORB_SLAM3
{

struct FeatureSet
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct NeuralMatch
{
    int query_index;
    int train_index;
    float confidence;
};

struct StereoGeometry
{
    std::vector<float> right_u;
    std::vector<float> depth;
    std::vector<NeuralMatch> matches;
};

class NeuralFeatureFrontend
{
public:
    struct Config
    {
        std::string superpoint_model;
        std::string superpoint_sha256;
        std::string lightglue_model;
        std::string lightglue_sha256;
        int max_features = 1000;
        float keypoint_threshold = 0.004F;
        float match_threshold = 0.10F;
        float projection_descriptor_threshold = -1.0F;
    };

    explicit NeuralFeatureFrontend(const Config& config);
    ~NeuralFeatureFrontend();

    std::string Provider() const;
    float ProjectionDescriptorThreshold() const;
    FeatureSet Extract(const cv::Mat& grayscale);
    std::vector<NeuralMatch> Match(const FeatureSet& first,
                                   const cv::Size& first_size,
                                   const FeatureSet& second,
                                   const cv::Size& second_size);
    static bool IsUsableMatchConfidence(float confidence, float threshold);
    static StereoGeometry FilterRectifiedStereo(
        const FeatureSet& left, const FeatureSet& right,
        const std::vector<NeuralMatch>& matches, float baseline_times_fx,
        float max_vertical_error, float min_disparity, float max_depth);
    static std::vector<NeuralMatch> FilterFundamentalInliers(
        const FeatureSet& previous, const FeatureSet& current,
        const std::vector<NeuralMatch>& matches, double max_error,
        double confidence);

    NeuralFeatureFrontend(const NeuralFeatureFrontend&) = delete;
    NeuralFeatureFrontend& operator=(const NeuralFeatureFrontend&) = delete;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ORB_SLAM3

#endif
