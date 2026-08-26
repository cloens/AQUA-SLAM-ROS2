#ifndef SUPERPOINTEXTRACTOR_H
#define SUPERPOINTEXTRACTOR_H

#include <mutex>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{

class SuperPointExtractor
{
public:
    SuperPointExtractor(const std::string& onnxPath, int nfeatures=1000,
                        float confThresh=0.005f, int nmsDist=3,
                        cv::Size modelInputSize=cv::Size());
    bool Ready() const { return mbReady; }
    int Extract(const cv::Mat& gray, std::vector<cv::KeyPoint>& keys, cv::Mat& descBinary);
    int ExtractPair(const cv::Mat& grayL, const cv::Mat& grayR,
                    std::vector<cv::KeyPoint>& keysL, cv::Mat& descL,
                    std::vector<cv::KeyPoint>& keysR, cv::Mat& descR,
                    std::vector<cv::Point2f>& matchL,
                    std::vector<cv::Point2f>& matchR,
                    std::vector<float>& matchScore,
                    std::vector<int>* leftToRight = nullptr);
    static cv::Mat FloatDescToORB(const cv::Mat& descFloat);
    static int HammingPacked(const cv::Mat& a, const cv::Mat& b);
private:
    bool RunNet(const cv::Mat& gray01, cv::Mat& heatmap, cv::Mat& coarseDesc);
    void Decode(const cv::Mat& heatmap, const cv::Mat& coarseDesc,
                std::vector<cv::KeyPoint>& keys, cv::Mat& descFloat);
    cv::dnn::Net mNet;
    std::mutex mMutex;
    bool mbReady = false;
    int mnFeatures;
    float mfConfThresh;
    int mnNmsDist;
    cv::Size mModelInputSize;
};

} // namespace ORB_SLAM3
#endif
