#include "NeuralFeatureFrontend.h"

#include <array>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

#include <openssl/evp.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace ORB_SLAM3
{
namespace
{

std::string FileSha256(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open neural model: " + path);

    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context || EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1)
    {
        EVP_MD_CTX_free(context);
        throw std::runtime_error("cannot initialize SHA-256");
    }
    std::array<char, 64 * 1024> buffer{};
    while (stream)
    {
        stream.read(buffer.data(), buffer.size());
        if (EVP_DigestUpdate(context, buffer.data(), stream.gcount()) != 1)
        {
            EVP_MD_CTX_free(context);
            throw std::runtime_error("cannot update SHA-256 for neural model: " + path);
        }
    }
    if (!stream.eof())
    {
        EVP_MD_CTX_free(context);
        throw std::runtime_error("cannot read neural model: " + path);
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(context, digest.data(), &digest_size) != 1)
    {
        EVP_MD_CTX_free(context);
        throw std::runtime_error("cannot finalize SHA-256 for neural model: " + path);
    }
    EVP_MD_CTX_free(context);
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index)
        encoded << std::setw(2) << static_cast<unsigned int>(digest[index]);
    return encoded.str();
}

void VerifyModel(const std::string& path, const std::string& expected)
{
    if (expected.size() != 64)
        throw std::runtime_error("invalid expected SHA-256 for neural model: " + path);
    const std::string actual = FileSha256(path);
    if (actual != expected)
        throw std::runtime_error("unexpected SHA-256 for neural model: " + path);
}

void ValidateConfig(const NeuralFeatureFrontend::Config& config)
{
    if (config.max_features <= 0)
        throw std::runtime_error("FeatureFrontend.max_features must be positive");
    if (!std::isfinite(config.keypoint_threshold) ||
        config.keypoint_threshold < 0.0F || config.keypoint_threshold > 1.0F)
        throw std::runtime_error(
            "FeatureFrontend.keypoint_threshold must be within [0,1]");
    if (!std::isfinite(config.match_threshold) ||
        config.match_threshold < 0.0F || config.match_threshold > 1.0F)
        throw std::runtime_error(
            "FeatureFrontend.match_threshold must be within [0,1]");
    if (!std::isfinite(config.projection_descriptor_threshold) ||
        config.projection_descriptor_threshold <= 0.0F ||
        config.projection_descriptor_threshold > 2.0F)
        throw std::runtime_error(
            "FeatureFrontend.projection_descriptor_threshold must be within (0,2]");
}

std::vector<std::string> TensorNames(Ort::Session& session, bool inputs)
{
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        auto name = inputs ? session.GetInputNameAllocated(index, allocator)
                           : session.GetOutputNameAllocated(index, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

void VerifyTensorNames(Ort::Session& session,
                       const std::vector<std::string>& expected_inputs,
                       const std::vector<std::string>& expected_outputs,
                       const std::string& model_name)
{
    if (TensorNames(session, true) != expected_inputs ||
        TensorNames(session, false) != expected_outputs)
        throw std::runtime_error(model_name + " tensor names do not match the pinned contract");
}

} // namespace

class NeuralFeatureFrontend::Impl
{
public:
    explicit Impl(const Config& config)
	        : environment(ORT_LOGGING_LEVEL_WARNING, "aqua-neural-frontend"),
		          max_features(config.max_features),
		          keypoint_threshold(config.keypoint_threshold),
		          match_threshold(config.match_threshold),
		          projection_descriptor_threshold(
		              config.projection_descriptor_threshold)
    {
        const auto providers = Ort::GetAvailableProviders();
        if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end())
            throw std::runtime_error("CUDAExecutionProvider is unavailable");

        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_options);

        options.SetLogId("aqua-superpoint");
        superpoint = std::make_unique<Ort::Session>(
            environment, config.superpoint_model.c_str(), options);
        VerifyTensorNames(*superpoint, {"image"},
                          {"keypoints", "scores", "descriptors"}, "SuperPoint");
        options.SetLogId("aqua-lightglue");
        lightglue = std::make_unique<Ort::Session>(
            environment, config.lightglue_model.c_str(), options);
        VerifyTensorNames(*lightglue,
                          {"kpts0", "kpts1", "desc0", "desc1"},
                          {"matches0", "mscores0"}, "LightGlue");
    }

    FeatureSet Extract(const cv::Mat& grayscale)
    {
        if (grayscale.empty() || grayscale.type() != CV_8UC1)
            throw std::runtime_error("SuperPoint input must be a non-empty CV_8UC1 image");

        const float scale = 512.0F /
            static_cast<float>(std::max(grayscale.rows, grayscale.cols));
        const cv::Size resized_size(
            static_cast<int>(std::round(grayscale.cols * scale)),
            static_cast<int>(std::round(grayscale.rows * scale)));
        cv::Mat resized;
        cv::resize(grayscale, resized, resized_size, 0.0, 0.0, cv::INTER_AREA);
        cv::Mat input;
        resized.convertTo(input, CV_32F, 1.0 / 255.0);
        if (!input.isContinuous())
            input = input.clone();

        const std::array<int64_t, 4> input_shape{
            1, 1, input.rows, input.cols};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory, input.ptr<float>(), input.total(),
            input_shape.data(), input_shape.size());
        const char* input_names[] = {"image"};
        const char* output_names[] = {"keypoints", "scores", "descriptors"};
        auto outputs = superpoint->Run(Ort::RunOptions{nullptr},
                                       input_names, &tensor, 1,
                                       output_names, 3);

        const auto keypoint_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto score_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        const auto descriptor_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
        if (keypoint_shape.size() != 3 || keypoint_shape[0] != 1 || keypoint_shape[2] != 2 ||
            score_shape.size() != 2 || score_shape[0] != 1 ||
            descriptor_shape.size() != 3 || descriptor_shape[0] != 1 ||
            descriptor_shape[2] != 256 || score_shape[1] != keypoint_shape[1] ||
            descriptor_shape[1] != keypoint_shape[1])
            throw std::runtime_error("SuperPoint output shapes do not match [1,N,2]/[1,N]/[1,N,256]");

        const std::size_t count = static_cast<std::size_t>(keypoint_shape[1]);
        const auto* keypoints = outputs[0].GetTensorData<int64_t>();
        const auto* scores = outputs[1].GetTensorData<float>();
        const auto* descriptors = outputs[2].GetTensorData<float>();
	        std::vector<std::size_t> order(count);
	        std::iota(order.begin(), order.end(), 0);
	        order.erase(std::remove_if(order.begin(), order.end(),
	                                   [scores, this](std::size_t index) {
	                                       return scores[index] < keypoint_threshold;
	                                   }),
	                    order.end());
        std::stable_sort(order.begin(), order.end(),
                         [scores](std::size_t first, std::size_t second) {
                             return scores[first] > scores[second];
                         });
        order.resize(std::min(order.size(), static_cast<std::size_t>(max_features)));

        FeatureSet result;
        result.keypoints.reserve(order.size());
        result.descriptors.create(static_cast<int>(order.size()), 256, CV_32F);
        for (std::size_t output_index = 0; output_index < order.size(); ++output_index)
        {
            const std::size_t source_index = order[output_index];
            const float x = static_cast<float>(keypoints[source_index * 2]) / scale;
            const float y = static_cast<float>(keypoints[source_index * 2 + 1]) / scale;
            if (!std::isfinite(x) || !std::isfinite(y) ||
                x < 0.0F || x >= grayscale.cols || y < 0.0F || y >= grayscale.rows)
                throw std::runtime_error("SuperPoint returned an invalid keypoint");
            cv::KeyPoint keypoint(cv::Point2f(x, y), 1.0F, -1.0F, scores[source_index], 0);
            result.keypoints.push_back(keypoint);
            std::copy_n(descriptors + source_index * 256, 256,
                        result.descriptors.ptr<float>(static_cast<int>(output_index)));
            cv::Mat row = result.descriptors.row(static_cast<int>(output_index));
            if (!cv::checkRange(row))
                throw std::runtime_error("SuperPoint returned a non-finite descriptor");
            const double norm = cv::norm(row);
            if (!std::isfinite(norm) || norm <= 0.0)
                throw std::runtime_error("SuperPoint returned a zero descriptor");
            row /= norm;
        }
        return result;
    }

    std::vector<NeuralMatch> Match(const FeatureSet& first,
                                   const cv::Size& first_size,
                                   const FeatureSet& second,
                                   const cv::Size& second_size)
    {
        ValidateFeatures(first, "first");
        ValidateFeatures(second, "second");
        if (first.keypoints.empty() || second.keypoints.empty())
            return {};
        if (first_size.width <= 0 || first_size.height <= 0 ||
            second_size.width <= 0 || second_size.height <= 0)
            throw std::runtime_error("LightGlue image sizes must be positive");

        std::vector<float> first_keypoints = NormalizeKeypoints(first.keypoints, first_size);
        std::vector<float> second_keypoints = NormalizeKeypoints(second.keypoints, second_size);
        cv::Mat first_descriptors = first.descriptors.isContinuous()
            ? first.descriptors : first.descriptors.clone();
        cv::Mat second_descriptors = second.descriptors.isContinuous()
            ? second.descriptors : second.descriptors.clone();
        const std::array<int64_t, 3> first_keypoint_shape{
            1, static_cast<int64_t>(first.keypoints.size()), 2};
        const std::array<int64_t, 3> second_keypoint_shape{
            1, static_cast<int64_t>(second.keypoints.size()), 2};
        const std::array<int64_t, 3> first_descriptor_shape{
            1, static_cast<int64_t>(first.keypoints.size()), 256};
        const std::array<int64_t, 3> second_descriptor_shape{
            1, static_cast<int64_t>(second.keypoints.size()), 256};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtDeviceAllocator, OrtMemTypeCPU);
        std::array<Ort::Value, 4> inputs{
            Ort::Value::CreateTensor<float>(memory, first_keypoints.data(),
                                            first_keypoints.size(), first_keypoint_shape.data(), 3),
            Ort::Value::CreateTensor<float>(memory, second_keypoints.data(),
                                            second_keypoints.size(), second_keypoint_shape.data(), 3),
            Ort::Value::CreateTensor<float>(memory, first_descriptors.ptr<float>(),
                                            first_descriptors.total(), first_descriptor_shape.data(), 3),
            Ort::Value::CreateTensor<float>(memory, second_descriptors.ptr<float>(),
                                            second_descriptors.total(), second_descriptor_shape.data(), 3)};
        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0", "mscores0"};
        auto outputs = lightglue->Run(Ort::RunOptions{nullptr},
                                      input_names, inputs.data(), inputs.size(),
                                      output_names, 2);
        const auto match_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto score_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (match_shape.size() != 2 || match_shape[1] != 2 ||
            score_shape.size() != 1 || score_shape[0] != match_shape[0])
            throw std::runtime_error("LightGlue output shapes do not match [M,2]/[M]");

        const auto* indices = outputs[0].GetTensorData<int64_t>();
        const auto* scores = outputs[1].GetTensorData<float>();
        std::set<int> used_first;
        std::set<int> used_second;
        std::vector<NeuralMatch> result;
        for (int64_t index = 0; index < match_shape[0]; ++index)
        {
            if (!NeuralFeatureFrontend::IsUsableMatchConfidence(
                    scores[index], match_threshold))
                continue;
            const int query = static_cast<int>(indices[index * 2]);
            const int train = static_cast<int>(indices[index * 2 + 1]);
            if (query < 0 || query >= static_cast<int>(first.keypoints.size()) ||
                train < 0 || train >= static_cast<int>(second.keypoints.size()))
                throw std::runtime_error("LightGlue returned an out-of-range index");
            if (!used_first.insert(query).second || !used_second.insert(train).second)
                throw std::runtime_error("LightGlue returned duplicate match indices");
            result.push_back({query, train, scores[index]});
        }
        return result;
    }

    float ProjectionDescriptorThreshold() const
    {
        return projection_descriptor_threshold;
    }

private:
    static void ValidateFeatures(const FeatureSet& features, const std::string& name)
    {
        if (features.descriptors.type() != CV_32F || features.descriptors.cols != 256 ||
            features.descriptors.rows != static_cast<int>(features.keypoints.size()) ||
            !cv::checkRange(features.descriptors))
            throw std::runtime_error(name + " LightGlue features violate the N x 256 float contract");
    }

    static std::vector<float> NormalizeKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                                                 const cv::Size& size)
    {
        const float shift_x = static_cast<float>(size.width) / 2.0F;
        const float shift_y = static_cast<float>(size.height) / 2.0F;
        const float scale = static_cast<float>(std::max(size.width, size.height)) / 2.0F;
        std::vector<float> normalized;
        normalized.reserve(keypoints.size() * 2);
        for (const auto& keypoint : keypoints)
        {
            normalized.push_back((keypoint.pt.x - shift_x) / scale);
            normalized.push_back((keypoint.pt.y - shift_y) / scale);
        }
        return normalized;
    }

    Ort::Env environment;
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> superpoint;
    std::unique_ptr<Ort::Session> lightglue;
	    int max_features;
	    float keypoint_threshold;
	    float match_threshold;
	    float projection_descriptor_threshold;
};

NeuralFeatureFrontend::NeuralFeatureFrontend(const Config& config)
{
    ValidateConfig(config);
    VerifyModel(config.superpoint_model, config.superpoint_sha256);
    VerifyModel(config.lightglue_model, config.lightglue_sha256);
    impl_ = std::make_unique<Impl>(config);
}

NeuralFeatureFrontend::~NeuralFeatureFrontend() = default;

std::string NeuralFeatureFrontend::Provider() const
{
    return "CUDAExecutionProvider";
}

float NeuralFeatureFrontend::ProjectionDescriptorThreshold() const
{
    return impl_->ProjectionDescriptorThreshold();
}

FeatureSet NeuralFeatureFrontend::Extract(const cv::Mat& grayscale)
{
    return impl_->Extract(grayscale);
}

std::vector<NeuralMatch> NeuralFeatureFrontend::Match(
    const FeatureSet& first, const cv::Size& first_size,
    const FeatureSet& second, const cv::Size& second_size)
{
    return impl_->Match(first, first_size, second, second_size);
}

bool NeuralFeatureFrontend::IsUsableMatchConfidence(float confidence,
                                                     float threshold)
{
    return std::isfinite(confidence) && std::isfinite(threshold) &&
           confidence >= threshold;
}

StereoGeometry NeuralFeatureFrontend::FilterRectifiedStereo(
    const FeatureSet& left, const FeatureSet& right,
    const std::vector<NeuralMatch>& matches, float baseline_times_fx,
    float max_vertical_error, float min_disparity, float max_depth)
{
    if (!std::isfinite(baseline_times_fx) || baseline_times_fx <= 0.0F ||
        !std::isfinite(max_vertical_error) || max_vertical_error < 0.0F ||
        !std::isfinite(min_disparity) || min_disparity <= 0.0F ||
        !std::isfinite(max_depth) || max_depth <= 0.0F)
        throw std::runtime_error("invalid rectified stereo geometry threshold");

    StereoGeometry result;
    result.right_u.assign(left.keypoints.size(), -1.0F);
    result.depth.assign(left.keypoints.size(), -1.0F);
    result.matches.reserve(matches.size());
    for (const auto& match : matches)
    {
        if (match.query_index < 0 ||
            match.query_index >= static_cast<int>(left.keypoints.size()) ||
            match.train_index < 0 ||
            match.train_index >= static_cast<int>(right.keypoints.size()))
            throw std::runtime_error("rectified stereo match index is out of range");
        const cv::Point2f& left_point = left.keypoints[match.query_index].pt;
        const cv::Point2f& right_point = right.keypoints[match.train_index].pt;
        const float disparity = left_point.x - right_point.x;
        if (!std::isfinite(disparity) || disparity < min_disparity ||
            std::fabs(left_point.y - right_point.y) > max_vertical_error)
            continue;
        const float depth = baseline_times_fx / disparity;
        if (!std::isfinite(depth) || depth <= 0.0F || depth > max_depth)
            continue;
        result.right_u[match.query_index] = right_point.x;
        result.depth[match.query_index] = depth;
        result.matches.push_back(match);
    }
    return result;
}

std::vector<NeuralMatch> NeuralFeatureFrontend::FilterFundamentalInliers(
    const FeatureSet& previous, const FeatureSet& current,
    const std::vector<NeuralMatch>& matches, double max_error,
    double confidence)
{
    if (!std::isfinite(max_error) || max_error <= 0.0 ||
        !std::isfinite(confidence) || confidence <= 0.0 || confidence >= 1.0)
        throw std::runtime_error("invalid fundamental-matrix RANSAC parameters");
    if (matches.size() < 8)
        return {};

    std::set<int> previous_indices;
    std::set<int> current_indices;
    std::vector<cv::Point2f> previous_points;
    std::vector<cv::Point2f> current_points;
    previous_points.reserve(matches.size());
    current_points.reserve(matches.size());
    for (const auto& match : matches)
    {
        if (match.query_index < 0 ||
            match.query_index >= static_cast<int>(previous.keypoints.size()) ||
            match.train_index < 0 ||
            match.train_index >= static_cast<int>(current.keypoints.size()))
            throw std::runtime_error("temporal match index is out of range");
        if (!previous_indices.insert(match.query_index).second ||
            !current_indices.insert(match.train_index).second)
            throw std::runtime_error("temporal matches are not one-to-one");
        previous_points.push_back(previous.keypoints[match.query_index].pt);
        current_points.push_back(current.keypoints[match.train_index].pt);
    }

    cv::Mat inlier_mask;
    const cv::Mat fundamental = cv::findFundamentalMat(
        previous_points, current_points, cv::FM_RANSAC,
        max_error, confidence, inlier_mask);
    if (fundamental.empty() || inlier_mask.total() != matches.size())
        return {};

    std::vector<NeuralMatch> inliers;
    inliers.reserve(matches.size());
    for (std::size_t index = 0; index < matches.size(); ++index)
    {
        if (inlier_mask.at<unsigned char>(static_cast<int>(index)) != 0)
            inliers.push_back(matches[index]);
    }
    return inliers;
}

} // namespace ORB_SLAM3
