#include "NeuralFeatureFrontend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <onnxruntime_cxx_api.h>

namespace
{

struct Arguments
{
    std::string settings;
    std::string manifest;
    std::string output;
    bool calibration = false;
};

struct ImagePair
{
    std::string sequence_id;
    std::string left;
    std::string right;
};

std::string JsonEscape(const std::string& value)
{
    std::ostringstream escaped;
    for (const char character : value)
    {
        switch (character)
        {
        case '"': escaped << "\\\""; break;
        case '\\': escaped << "\\\\"; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (static_cast<unsigned char>(character) < 0x20)
                throw std::runtime_error("manifest contains a control character");
            escaped << character;
        }
    }
    return escaped.str();
}

Arguments ParseArguments(int argc, char** argv)
{
    Arguments arguments;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--calibration")
        {
            arguments.calibration = true;
            continue;
        }
        if ((argument == "--settings" || argument == "--manifest" || argument == "--output") &&
            index + 1 < argc)
        {
            std::string* destination = argument == "--settings" ? &arguments.settings :
                                       argument == "--manifest" ? &arguments.manifest :
                                                                  &arguments.output;
            *destination = argv[++index];
            continue;
        }
        throw std::runtime_error(
            "usage: neural_frontend_benchmark --settings SETTINGS --manifest MANIFEST "
            "--output OUTPUT [--calibration]");
    }
    if (arguments.settings.empty() || arguments.manifest.empty() || arguments.output.empty())
        throw std::runtime_error("settings, manifest, and output are required");
    return arguments;
}

std::vector<ImagePair> ReadManifest(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open manifest: " + path);
    std::string line;
    if (!std::getline(stream, line) || line != "aqua-neural-frontend-manifest-v1")
        throw std::runtime_error("unsupported neural frontend manifest");

    std::vector<ImagePair> pairs;
    while (std::getline(stream, line))
    {
        const std::size_t first_tab = line.find('\t');
        const std::size_t second_tab = line.find('\t', first_tab + 1);
        if (first_tab == std::string::npos || second_tab == std::string::npos ||
            line.find('\t', second_tab + 1) != std::string::npos)
            throw std::runtime_error("manifest record must contain sequence id, left path, right path");
        ImagePair pair{line.substr(0, first_tab),
                       line.substr(first_tab + 1, second_tab - first_tab - 1),
                       line.substr(second_tab + 1)};
        if (pair.sequence_id.empty() || pair.left.empty() || pair.right.empty())
            throw std::runtime_error("manifest record contains an empty field");
        pairs.push_back(std::move(pair));
    }
    if (pairs.empty())
        throw std::runtime_error("manifest has no image pairs");
    return pairs;
}

std::string ReadString(cv::FileStorage& settings, const char* key)
{
    const cv::FileNode node = settings[key];
    if (node.empty() || !node.isString())
        throw std::runtime_error(std::string("settings requires string ") + key);
    return static_cast<std::string>(node);
}

float ReadFloat(cv::FileStorage& settings, const char* key)
{
    const cv::FileNode node = settings[key];
    if (node.empty() || (!node.isInt() && !node.isReal()))
        throw std::runtime_error(std::string("settings requires numeric ") + key);
    const float value = static_cast<float>(node.real());
    if (!std::isfinite(value))
        throw std::runtime_error(std::string("settings contains non-finite ") + key);
    return value;
}

ORB_SLAM3::NeuralFeatureFrontend::Config ReadFrontendConfig(const std::string& path)
{
    cv::FileStorage settings(path, cv::FileStorage::READ);
    if (!settings.isOpened())
        throw std::runtime_error("cannot open settings: " + path);
    if (ReadString(settings, "FeatureFrontend.type") != "SUPERPOINT_LIGHTGLUE" ||
        ReadString(settings, "FeatureFrontend.provider") != "CUDA")
        throw std::runtime_error("settings must select SUPERPOINT_LIGHTGLUE with CUDA");

    ORB_SLAM3::NeuralFeatureFrontend::Config config;
    config.superpoint_model = ReadString(settings, "FeatureFrontend.superpoint_onnx");
    config.superpoint_sha256 = ReadString(settings, "FeatureFrontend.superpoint_sha256");
    config.lightglue_model = ReadString(settings, "FeatureFrontend.lightglue_onnx");
    config.lightglue_sha256 = ReadString(settings, "FeatureFrontend.lightglue_sha256");
    const cv::FileNode max_features = settings["FeatureFrontend.max_features"];
    if (max_features.empty() || !max_features.isInt())
        throw std::runtime_error("settings requires integer FeatureFrontend.max_features");
    config.max_features = static_cast<int>(max_features);
    config.keypoint_threshold = ReadFloat(settings, "FeatureFrontend.keypoint_threshold");
    config.match_threshold = ReadFloat(settings, "FeatureFrontend.match_threshold");
    config.projection_descriptor_threshold = ReadFloat(
        settings, "FeatureFrontend.projection_descriptor_threshold");
    return config;
}

float DescriptorDistance(const cv::Mat& first, int first_index,
                         const cv::Mat& second, int second_index)
{
    const float distance = 1.0F - first.row(first_index).dot(second.row(second_index));
    if (!std::isfinite(distance))
        throw std::runtime_error("non-finite SuperPoint descriptor distance");
    return distance;
}

std::vector<float> GlobalNearestImpostorDistances(
    const ORB_SLAM3::FeatureSet& previous, const ORB_SLAM3::FeatureSet& current,
    const std::vector<ORB_SLAM3::NeuralMatch>& temporal_matches,
    const std::vector<ORB_SLAM3::NeuralMatch>& inliers)
{
    std::set<int> matched_current_indices;
    for (const ORB_SLAM3::NeuralMatch& match : temporal_matches)
        matched_current_indices.insert(match.train_index);

    std::vector<float> distances;
    for (const ORB_SLAM3::NeuralMatch& inlier : inliers)
    {
        int candidate = -1;
        float nearest_descriptor_distance = std::numeric_limits<float>::infinity();
        for (int index = 0; index < static_cast<int>(current.keypoints.size()); ++index)
        {
            if (matched_current_indices.count(index) != 0)
                continue;
            const float distance = DescriptorDistance(previous.descriptors, inlier.query_index,
                                                      current.descriptors, index);
            if (distance < nearest_descriptor_distance ||
                (distance == nearest_descriptor_distance && index < candidate))
            {
                candidate = index;
                nearest_descriptor_distance = distance;
            }
        }
        if (candidate >= 0)
            distances.push_back(nearest_descriptor_distance);
    }
    return distances;
}

void WriteDistances(std::ostream& stream, const std::vector<float>& distances)
{
    stream << '[';
    for (std::size_t index = 0; index < distances.size(); ++index)
    {
        if (index != 0)
            stream << ',';
        stream << distances[index];
    }
    stream << ']';
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Arguments arguments = ParseArguments(argc, argv);
        if (std::filesystem::exists(arguments.output))
            throw std::runtime_error("refusing to overwrite benchmark output: " + arguments.output);
        const std::vector<ImagePair> pairs = ReadManifest(arguments.manifest);
        const ORB_SLAM3::NeuralFeatureFrontend::Config config =
            ReadFrontendConfig(arguments.settings);
        ORB_SLAM3::NeuralFeatureFrontend frontend(config);
        if (frontend.Provider() != "CUDAExecutionProvider")
            throw std::runtime_error("neural frontend did not select CUDAExecutionProvider");

        std::ofstream output(arguments.output);
        if (!output)
            throw std::runtime_error("cannot create benchmark output: " + arguments.output);
        output << std::setprecision(9) << std::fixed;
        output << "{\"type\":\"metadata\",\"provider\":\"CUDAExecutionProvider\""
               << ",\"onnxruntime_version\":\"" << JsonEscape(Ort::GetVersionString())
               << "\",\"superpoint_model_path\":\"" << JsonEscape(config.superpoint_model)
               << "\",\"superpoint_sha256\":\"" << JsonEscape(config.superpoint_sha256)
               << "\",\"lightglue_model_path\":\"" << JsonEscape(config.lightglue_model)
               << "\",\"lightglue_sha256\":\"" << JsonEscape(config.lightglue_sha256)
               << "\"}\n";

        ORB_SLAM3::FeatureSet previous;
        cv::Size previous_size;
        for (const ImagePair& pair : pairs)
        {
            const cv::Mat left = cv::imread(pair.left, cv::IMREAD_GRAYSCALE);
            const cv::Mat right = cv::imread(pair.right, cv::IMREAD_GRAYSCALE);
            if (left.empty() || right.empty())
                throw std::runtime_error("cannot load stereo pair " + pair.sequence_id);

            const auto started = std::chrono::steady_clock::now();
            ORB_SLAM3::FeatureSet current = frontend.Extract(left);
            ORB_SLAM3::FeatureSet right_features = frontend.Extract(right);
            const std::vector<ORB_SLAM3::NeuralMatch> stereo_matches =
                frontend.Match(current, left.size(), right_features, right.size());
            std::vector<ORB_SLAM3::NeuralMatch> temporal_matches;
            std::vector<ORB_SLAM3::NeuralMatch> temporal_inliers;
            if (!previous.keypoints.empty())
            {
                temporal_matches = frontend.Match(previous, previous_size, current, left.size());
                temporal_inliers = ORB_SLAM3::NeuralFeatureFrontend::FilterFundamentalInliers(previous, current, temporal_matches, 3.0, 0.99);
            }
            const auto finished = std::chrono::steady_clock::now();

            std::vector<float> positive_distances;
            std::vector<float> negative_distances;
            if (arguments.calibration)
            {
                positive_distances.reserve(temporal_inliers.size());
                for (const ORB_SLAM3::NeuralMatch& inlier : temporal_inliers)
                    positive_distances.push_back(DescriptorDistance(
                        previous.descriptors, inlier.query_index,
                        current.descriptors, inlier.train_index));
                negative_distances = GlobalNearestImpostorDistances(
                    previous, current, temporal_matches, temporal_inliers);
            }

            const double latency_ms = std::chrono::duration<double, std::milli>(finished - started).count();
            output << "{\"type\":\"pair\",\"sequence_id\":\"" << JsonEscape(pair.sequence_id)
                   << "\",\"latency_ms\":" << latency_ms
                   << ",\"left_features\":" << current.keypoints.size()
                   << ",\"right_features\":" << right_features.keypoints.size()
                   << ",\"stereo_matches\":" << stereo_matches.size()
                   << ",\"temporal_matches\":" << temporal_inliers.size()
                   << ",\"positive_distances\":";
            WriteDistances(output, positive_distances);
            output << ",\"negative_distances\":";
            WriteDistances(output, negative_distances);
            output << "}\n";
            previous = std::move(current);
            previous_size = left.size();
        }
        if (!output)
            throw std::runtime_error("cannot write benchmark output: " + arguments.output);
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "neural_frontend_benchmark: " << exception.what() << std::endl;
        return 1;
    }
}
