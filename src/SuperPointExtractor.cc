#include "SuperPointExtractor.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace ORB_SLAM3
{

static int popcnt32(unsigned int v)
{
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return (int)((((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
}

SuperPointExtractor::SuperPointExtractor(const std::string& onnxPath, int nfeatures,
                                         float confThresh, int nmsDist,
                                         cv::Size modelInputSize)
    : mnFeatures(nfeatures), mfConfThresh(confThresh), mnNmsDist(nmsDist),
      mModelInputSize(modelInputSize)
{
    if (onnxPath.empty())
        return;
    try {
        mNet = cv::dnn::readNetFromONNX(onnxPath);
        mNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        mNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        mbReady = true;
        std::cout << "SuperPointExtractor loaded: " << onnxPath << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "SuperPointExtractor failed to load " << onnxPath << ": " << e.what() << std::endl;
    }
}

cv::Mat SuperPointExtractor::FloatDescToORB(const cv::Mat& descFloat)
{
    cv::Mat packed;
    if (descFloat.empty() || descFloat.cols != 256 || descFloat.type() != CV_32F)
        return packed;
    packed = cv::Mat::zeros(descFloat.rows, 32, CV_8U);
    for (int r = 0; r < descFloat.rows; ++r) {
        const float* src = descFloat.ptr<float>(r);
        uchar* dst = packed.ptr<uchar>(r);
        for (int b = 0; b < 32; ++b) {
            uchar v = 0;
            for (int i = 0; i < 8; ++i) {
                if (src[b * 8 + i] > 0.f)
                    v |= static_cast<uchar>(1u << i);
            }
            dst[b] = v;
        }
    }
    return packed;
}

int SuperPointExtractor::HammingPacked(const cv::Mat& a, const cv::Mat& b)
{
    if (a.empty() || b.empty() || a.cols != b.cols)
        return 256;
    const int* pa = a.ptr<int32_t>();
    const int* pb = b.ptr<int32_t>();
    int dist = 0;
    for (int i = 0; i < a.cols / 4; ++i)
        dist += popcnt32((unsigned int)(pa[i] ^ pb[i]));
    return dist;
}

bool SuperPointExtractor::RunNet(const cv::Mat& gray01, cv::Mat& heatmap, cv::Mat& coarseDesc)
{
    if (!mbReady || gray01.empty())
        return false;
    cv::Mat blob = cv::dnn::blobFromImage(gray01, 1.0, gray01.size(), cv::Scalar(), false, false, CV_32F);
    std::lock_guard<std::mutex> lock(mMutex);
    mNet.setInput(blob);
    std::vector<cv::Mat> outs;
    mNet.forward(outs, std::vector<std::string>{"heatmap", "desc"});
    if (outs.size() < 2)
        return false;
    heatmap = outs[0].clone();
    coarseDesc = outs[1].clone();
    return !heatmap.empty() && !coarseDesc.empty();
}

void SuperPointExtractor::Decode(const cv::Mat& heatmap4, const cv::Mat& coarseDesc4,
                                 std::vector<cv::KeyPoint>& keys, cv::Mat& descFloat)
{
    keys.clear();
    descFloat.release();
    cv::Mat heat;
    if (heatmap4.dims == 4)
        heat = cv::Mat(heatmap4.size[2], heatmap4.size[3], CV_32F,
                       (void*)heatmap4.ptr<float>()).clone();
    else if (heatmap4.dims == 3)
        heat = cv::Mat(heatmap4.size[1], heatmap4.size[2], CV_32F,
                       (void*)heatmap4.ptr<float>()).clone();
    else
        heat = heatmap4.clone();

    const int h = heat.rows;
    const int w = heat.cols;
    std::vector<cv::KeyPoint> candidates;
    candidates.reserve(2048);
    for (int y = 4; y < h - 4; ++y) {
        const float* row = heat.ptr<float>(y);
        for (int x = 4; x < w - 4; ++x) {
            if (row[x] >= mfConfThresh)
                candidates.emplace_back(cv::Point2f((float)x, (float)y), 8.f, -1.f, row[x]);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.response > b.response;
    });
    cv::Mat nms = cv::Mat::zeros(h, w, CV_8U);
    const int radius = std::max(1, mnNmsDist);
    for (const auto& kp : candidates) {
        const int x = (int)std::round(kp.pt.x);
        const int y = (int)std::round(kp.pt.y);
        if (nms.at<uchar>(y, x))
            continue;
        keys.push_back(kp);
        for (int yy = std::max(0, y - radius); yy <= std::min(h - 1, y + radius); ++yy)
            for (int xx = std::max(0, x - radius); xx <= std::min(w - 1, x + radius); ++xx)
                nms.at<uchar>(yy, xx) = 1;
        if ((int)keys.size() >= mnFeatures)
            break;
    }
    if (keys.empty() || coarseDesc4.dims != 4)
        return;

    const int dimensions = coarseDesc4.size[1];
    const int hc = coarseDesc4.size[2];
    const int wc = coarseDesc4.size[3];
    const float* coarse = coarseDesc4.ptr<float>();
    descFloat.create((int)keys.size(), dimensions, CV_32F);
    for (int i = 0; i < (int)keys.size(); ++i) {
        const float x = keys[i].pt.x / 8.f;
        const float y = keys[i].pt.y / 8.f;
        const int x0 = std::clamp((int)std::floor(x), 0, wc - 1);
        const int y0 = std::clamp((int)std::floor(y), 0, hc - 1);
        const int x1 = std::min(wc - 1, x0 + 1);
        const int y1 = std::min(hc - 1, y0 + 1);
        const float ax = x - x0;
        const float ay = y - y0;
        float* output = descFloat.ptr<float>(i);
        double norm = 0.0;
        for (int c = 0; c < dimensions; ++c) {
            auto at = [&](int yy, int xx) { return coarse[(c * hc + yy) * wc + xx]; };
            const float value = (1-ax)*(1-ay)*at(y0,x0) + ax*(1-ay)*at(y0,x1)
                              + (1-ax)*ay*at(y1,x0) + ax*ay*at(y1,x1);
            output[c] = value;
            norm += (double)value * value;
        }
        norm = std::sqrt(std::max(norm, 1e-12));
        for (int c = 0; c < dimensions; ++c)
            output[c] = (float)(output[c] / norm);
    }
}

int SuperPointExtractor::Extract(const cv::Mat& gray, std::vector<cv::KeyPoint>& keys,
                                 cv::Mat& descBinary)
{
    keys.clear();
    descBinary.release();
    if (gray.empty())
        return 0;
    cv::Mat mono;
    if (gray.channels() == 3)
        cv::cvtColor(gray, mono, cv::COLOR_BGR2GRAY);
    else if (gray.channels() == 4)
        cv::cvtColor(gray, mono, cv::COLOR_BGRA2GRAY);
    else
        mono = gray;
    cv::Mat input;
    mono.convertTo(input, CV_32F, 1.0 / 255.0);
    const cv::Size inputSize = input.size();
    const cv::Size modelInput = mModelInputSize.width > 0 && mModelInputSize.height > 0
        ? mModelInputSize : inputSize;
    if (inputSize != modelInput)
        cv::resize(input, input, modelInput, 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat heatmap, coarseDesc, floatDesc;
    if (!RunNet(input, heatmap, coarseDesc))
        return 0;
    Decode(heatmap, coarseDesc, keys, floatDesc);
    const float inputScaleX = static_cast<float>(inputSize.width) / modelInput.width;
    const float inputScaleY = static_cast<float>(inputSize.height) / modelInput.height;
    for (auto& key : keys) {
        key.pt.x *= inputScaleX;
        key.pt.y *= inputScaleY;
        key.size *= std::sqrt(inputScaleX * inputScaleY);
    }
    descBinary = FloatDescToORB(floatDesc);
    return (int)keys.size();
}

int SuperPointExtractor::ExtractPair(const cv::Mat& grayL, const cv::Mat& grayR,
                                     std::vector<cv::KeyPoint>& keysL, cv::Mat& descL,
                                     std::vector<cv::KeyPoint>& keysR, cv::Mat& descR,
                                     std::vector<cv::Point2f>& matchL,
                                     std::vector<cv::Point2f>& matchR,
                                     std::vector<float>& matchScore,
                                     std::vector<int>* leftToRight)
{
    Extract(grayL, keysL, descL);
    Extract(grayR, keysR, descR);
    matchL.clear();
    matchR.clear();
    matchScore.clear();
    if (leftToRight)
        leftToRight->assign(keysL.size(), -1);
    if (keysL.empty() || keysR.empty() || descL.empty() || descR.empty())
        return 0;

    std::vector<int> bestRight(keysL.size(), -1);
    std::vector<int> bestDistance(keysL.size(), 256);
    for (size_t i = 0; i < keysL.size(); ++i) {
        int best = 256, second = 256, bestIndex = -1;
        for (size_t j = 0; j < keysR.size(); ++j) {
            if (std::fabs(keysL[i].pt.y - keysR[j].pt.y) > 3.0f ||
                keysR[j].pt.x >= keysL[i].pt.x)
                continue;
            const int distance = HammingPacked(descL.row((int)i), descR.row((int)j));
            if (distance < best) {
                second = best;
                best = distance;
                bestIndex = (int)j;
            } else if (distance < second) {
                second = distance;
            }
        }
        if (bestIndex >= 0 && best < 110 && (second >= 256 || best < (int)(0.9f * second + 2))) {
            bestRight[i] = bestIndex;
            bestDistance[i] = best;
        }
    }
    std::vector<int> used(keysR.size(), -1);
    for (size_t i = 0; i < keysL.size(); ++i) {
        const int j = bestRight[i];
        if (j < 0)
            continue;
        if (used[j] >= 0 && bestDistance[used[j]] <= bestDistance[i])
            continue;
        used[j] = (int)i;
    }
    for (size_t j = 0; j < used.size(); ++j) {
        const int i = used[j];
        if (i < 0)
            continue;
        matchL.push_back(keysL[i].pt);
        matchR.push_back(keysR[j].pt);
        matchScore.push_back(1.f - bestDistance[i] / 256.f);
        if (leftToRight)
            (*leftToRight)[i] = (int)j;
    }
    return (int)matchL.size();
}

} // namespace ORB_SLAM3
