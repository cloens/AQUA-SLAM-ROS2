#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <optional>

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <opencv2/core/core.hpp>

#include "System.h"
#include "ImuTypes.h"

// Optional: StereoMatches from uw_slam_bridge when built in same workspace.
#if __has_include("uw_slam_bridge/msg/stereo_matches.hpp")
#include "uw_slam_bridge/msg/stereo_matches.hpp"
#define UW_SLAM_HAS_STEREO_MATCHES 1
#else
#define UW_SLAM_HAS_STEREO_MATCHES 0
#endif

using namespace std;

class ImuGrabber
{
public:
    void GrabImu(const sensor_msgs::msg::Imu::ConstSharedPtr &imu_msg)
    {
        unique_lock<mutex> lock(mBufMutex);
        imuBuf.push(imu_msg);
    }

    queue<sensor_msgs::msg::Imu::ConstSharedPtr> imuBuf;
    mutex mBufMutex;
};

class DVLGrabber
{
public:
    void GrabDVL(const nav_msgs::msg::Odometry::SharedPtr &msg)
    {
        unique_lock<mutex> lock(mBufMutex);
        dvlBuf.push(msg);
    }

    queue<nav_msgs::msg::Odometry::SharedPtr> dvlBuf;
    mutex mBufMutex;
};

struct ExternalFrontEnd
{
    vector<cv::Point2f> left;
    vector<cv::Point2f> right;
    vector<float> score;
    cv::Mat depth;
    double stamp = 0.0;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System *pSLAM, ImuGrabber *pImuGb, DVLGrabber *pDvlGb)
        : mpSLAM(pSLAM), mpImuGb(pImuGb), mpDvlGb(pDvlGb) {}

    void GrabImageLeft(const sensor_msgs::msg::Image::SharedPtr &msg)
    {
        unique_lock<mutex> lock(mBufMutexLeft);
        imgLeftBuf.push(msg);
    }

    void GrabImageRight(const sensor_msgs::msg::Image::SharedPtr &msg)
    {
        unique_lock<mutex> lock(mBufMutexRight);
        imgRightBuf.push(msg);
    }

    void GrabDepth(const sensor_msgs::msg::Image::SharedPtr &msg)
    {
        unique_lock<mutex> lock(mBufMutexDepth);
        depthBuf.push(msg);
    }

#if UW_SLAM_HAS_STEREO_MATCHES
    void GrabMatches(const uw_slam_bridge::msg::StereoMatches::SharedPtr &msg)
    {
        unique_lock<mutex> lock(mBufMutexMatches);
        matchesBuf.push(msg);
    }
#endif

    cv::Mat GetImage(const sensor_msgs::msg::Image::SharedPtr &img_msg)
    {
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            // Accept rgb8 from bridge or bgr8 from legacy sources.
            if (img_msg->encoding == sensor_msgs::image_encodings::RGB8) {
                cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::RGB8);
                cv::Mat bgr;
                cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_RGB2BGR);
                return bgr;
            }
            cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"), "cv_bridge exception: %s", e.what());
            return cv::Mat();
        }
        return cv_ptr->image.clone();
    }

    cv::Mat GetDepth(const sensor_msgs::msg::Image::SharedPtr &img_msg)
    {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(img_msg);
            cv::Mat depth;
            if (cv_ptr->image.type() == CV_32FC1)
                depth = cv_ptr->image.clone();
            else
                cv_ptr->image.convertTo(depth, CV_32FC1);
            return depth;
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"), "depth cv_bridge exception: %s", e.what());
            return cv::Mat();
        }
    }

    bool PopNearestDepth(double tImg, cv::Mat &outDepth)
    {
        unique_lock<mutex> lock(mBufMutexDepth);
        if (depthBuf.empty())
            return false;
        const double maxDt = 0.15;
        auto best = depthBuf.front();
        double bestDt = abs(rclcpp::Time(best->header.stamp).seconds() - tImg);
        while (depthBuf.size() > 1) {
            auto cand = depthBuf.front();
            double dt = abs(rclcpp::Time(cand->header.stamp).seconds() - tImg);
            if (dt <= bestDt) {
                best = cand;
                bestDt = dt;
                depthBuf.pop();
            } else {
                break;
            }
        }
        if (bestDt > maxDt)
            return false;
        if (!depthBuf.empty() && depthBuf.front() == best)
            depthBuf.pop();
        outDepth = GetDepth(best);
        return !outDepth.empty();
    }

#if UW_SLAM_HAS_STEREO_MATCHES
    bool PopNearestMatches(double tImg, vector<cv::Point2f> &left, vector<cv::Point2f> &right, vector<float> &score)
    {
        unique_lock<mutex> lock(mBufMutexMatches);
        if (matchesBuf.empty())
            return false;
        const double maxDt = 0.15;
        auto best = matchesBuf.front();
        double bestDt = abs(rclcpp::Time(best->header.stamp).seconds() - tImg);
        while (matchesBuf.size() > 1) {
            auto cand = matchesBuf.front();
            double dt = abs(rclcpp::Time(cand->header.stamp).seconds() - tImg);
            if (dt <= bestDt) {
                best = cand;
                bestDt = dt;
                matchesBuf.pop();
            } else {
                break;
            }
        }
        if (bestDt > maxDt)
            return false;
        if (!matchesBuf.empty() && matchesBuf.front() == best)
            matchesBuf.pop();
        const size_t n = best->u_left.size();
        if (n == 0 || best->u_right.size() != n || best->v_left.size() != n || best->v_right.size() != n)
            return false;
        left.clear(); right.clear(); score.clear();
        left.reserve(n); right.reserve(n); score.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            left.emplace_back(best->u_left[i], best->v_left[i]);
            right.emplace_back(best->u_right[i], best->v_right[i]);
            score.push_back(i < best->score.size() ? best->score[i] : 1.0f);
        }
        return true;
    }
#endif

    void SyncWithImu()
    {
        const double maxTimeDiff = 0.1;
        while (true) {
            cv::Mat imLeft, imRight;
            double tImLeft = 0, tImRight = 0;

            if (!imgLeftBuf.empty() && !imgRightBuf.empty() && !mpImuGb->imuBuf.empty()) {
                tImLeft  = rclcpp::Time(imgLeftBuf.front()->header.stamp).seconds();
                tImRight = rclcpp::Time(imgRightBuf.front()->header.stamp).seconds();

                {
                    unique_lock<mutex> lock(mBufMutexRight);
                    while ((tImLeft - tImRight) > maxTimeDiff && imgRightBuf.size() > 1) {
                        imgRightBuf.pop();
                        tImRight = rclcpp::Time(imgRightBuf.front()->header.stamp).seconds();
                    }
                }
                {
                    unique_lock<mutex> lock(mBufMutexLeft);
                    while ((tImRight - tImLeft) > maxTimeDiff && imgLeftBuf.size() > 1) {
                        imgLeftBuf.pop();
                        tImLeft = rclcpp::Time(imgLeftBuf.front()->header.stamp).seconds();
                    }
                }

                if (abs(tImLeft - tImRight) > maxTimeDiff)
                    continue;

                {
                    unique_lock<mutex> lock(mpImuGb->mBufMutex);
                    if (tImLeft > rclcpp::Time(mpImuGb->imuBuf.back()->header.stamp).seconds())
                        continue;
                }

                {
                    unique_lock<mutex> lock(mBufMutexLeft);
                    imLeft = GetImage(imgLeftBuf.front());
                    imgLeftBuf.pop();
                }
                {
                    unique_lock<mutex> lock(mBufMutexRight);
                    imRight = GetImage(imgRightBuf.front());
                    imgRightBuf.pop();
                }
                if (imLeft.empty() || imRight.empty())
                    continue;

                vector<ORB_SLAM3::IMU::GyroDvlPoint> vGyroDVLMeas;
                vector<ORB_SLAM3::IMU::DvlPoint>     vDVLMeas;

                {
                    unique_lock<mutex> lock(mpImuGb->mBufMutex);
                    while (!mpImuGb->imuBuf.empty() &&
                           rclcpp::Time(mpImuGb->imuBuf.front()->header.stamp).seconds() <= tImLeft) {
                        double t = rclcpp::Time(mpImuGb->imuBuf.front()->header.stamp).seconds();
                        const auto &imu = mpImuGb->imuBuf.front();
                        float ax = imu->linear_acceleration.x;
                        float ay = imu->linear_acceleration.y;
                        float az = imu->linear_acceleration.z;
                        float wx = imu->angular_velocity.x;
                        float wy = imu->angular_velocity.y;
                        float wz = imu->angular_velocity.z;
                        vGyroDVLMeas.push_back(ORB_SLAM3::IMU::GyroDvlPoint(
                            ax, ay, az, wx, wy, wz, 0, 0, 0, 0, 0, 0, 0, t));
                        mpImuGb->imuBuf.pop();
                    }
                }

                {
                    unique_lock<mutex> lock(mpDvlGb->mBufMutex);
                    while (!mpDvlGb->dvlBuf.empty() &&
                           rclcpp::Time(mpDvlGb->dvlBuf.front()->header.stamp).seconds() <= tImLeft) {
                        double t = rclcpp::Time(mpDvlGb->dvlBuf.front()->header.stamp).seconds();
                        const auto &dvl = mpDvlGb->dvlBuf.front();
                        float vx = dvl->twist.twist.linear.x;
                        float vy = dvl->twist.twist.linear.y;
                        float vz = dvl->twist.twist.linear.z;
                        vDVLMeas.push_back(ORB_SLAM3::IMU::DvlPoint(vx, vy, vz, 0, 0, 0, 0, t));
                        vGyroDVLMeas.push_back(ORB_SLAM3::IMU::GyroDvlPoint(
                            0, 0, 0, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, t));
                        mpDvlGb->dvlBuf.pop();
                    }
                }

                if (vDVLMeas.size() >= 2)
                    vDVLMeas.resize(1);

                // DVL-optional: IMU-only is enough to proceed with bDVL=false.
                if (vGyroDVLMeas.empty())
                    continue;

                sort(vGyroDVLMeas.begin(), vGyroDVLMeas.end(),
                     [](const ORB_SLAM3::IMU::GyroDvlPoint &a,
                        const ORB_SLAM3::IMU::GyroDvlPoint &b) { return a.t < b.t; });

                vector<cv::Point2f> vExtLeft, vExtRight;
                vector<float> vExtScore;
                cv::Mat extDepth;
#if UW_SLAM_HAS_STEREO_MATCHES
                PopNearestMatches(tImLeft, vExtLeft, vExtRight, vExtScore);
#endif
                PopNearestDepth(tImLeft, extDepth);

                mpSLAM->TrackStereoGroDVL(imLeft, imRight, tImLeft, vGyroDVLMeas, !vDVLMeas.empty(), "",
                                          vExtLeft, vExtRight, vExtScore, extDepth);
            }

            this_thread::sleep_for(chrono::milliseconds(1));
        }
    }

    queue<sensor_msgs::msg::Image::SharedPtr> imgLeftBuf, imgRightBuf, depthBuf;
    mutex mBufMutexLeft, mBufMutexRight, mBufMutexDepth;
#if UW_SLAM_HAS_STEREO_MATCHES
    queue<uw_slam_bridge::msg::StereoMatches::SharedPtr> matchesBuf;
    mutex mBufMutexMatches;
#endif

    ORB_SLAM3::System *mpSLAM;
    ImuGrabber        *mpImuGb;
    DVLGrabber        *mpDvlGb;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("aqua_slam_node");

    if (argc < 3) {
        RCLCPP_ERROR(node->get_logger(),
                     "Usage: aqua_slam_node <path_to_vocabulary> <path_to_settings>");
        rclcpp::shutdown();
        return 1;
    }

    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    string imu_topic   = fsSettings["ImuTopic"];
    string dvl_topic   = fsSettings["DvlTopic"];
    string img_l_topic = fsSettings["LeftImgTopic"];
    string img_r_topic = fsSettings["RightImgTopic"];
    string depth_topic;
    string matches_topic;
    if (!fsSettings["DepthTopic"].empty())
        depth_topic = (string)fsSettings["DepthTopic"];
    if (!fsSettings["StereoMatchesTopic"].empty())
        matches_topic = (string)fsSettings["StereoMatchesTopic"];
    if (depth_topic.empty())
        depth_topic = "/uw_slam/depth/scaled";
    if (matches_topic.empty())
        matches_topic = "/uw_slam/stereo_matches";

    RCLCPP_INFO(node->get_logger(), "IMU topic:   %s", imu_topic.c_str());
    RCLCPP_INFO(node->get_logger(), "DVL topic:   %s (optional)", dvl_topic.c_str());
    RCLCPP_INFO(node->get_logger(), "Left image:  %s", img_l_topic.c_str());
    RCLCPP_INFO(node->get_logger(), "Right image: %s", img_r_topic.c_str());
    RCLCPP_INFO(node->get_logger(), "Depth topic: %s", depth_topic.c_str());
    RCLCPP_INFO(node->get_logger(), "Matches:     %s", matches_topic.c_str());

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::DVL_STEREO, node, false);

    ImuGrabber   imugb;
    DVLGrabber   dvlgb;
    ImageGrabber igb(&SLAM, &imugb, &dvlgb);

    auto imu_sub = node->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, 100,
        [&imugb](const sensor_msgs::msg::Imu::ConstSharedPtr &msg) {
            imugb.GrabImu(msg);
        });

    // DVL remains subscribed, but empty DVL is allowed.
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr dvl_sub;
    if (!dvl_topic.empty()) {
        dvl_sub = node->create_subscription<nav_msgs::msg::Odometry>(
            dvl_topic, 100,
            [&dvlgb](const nav_msgs::msg::Odometry::SharedPtr msg) {
                dvlgb.GrabDVL(msg);
            });
    }

    auto it = image_transport::create_subscription(
        node.get(), img_l_topic,
        [&igb](const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
            igb.GrabImageLeft(std::make_shared<sensor_msgs::msg::Image>(*msg));
        },
        "raw");

    auto it_r = image_transport::create_subscription(
        node.get(), img_r_topic,
        [&igb](const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
            igb.GrabImageRight(std::make_shared<sensor_msgs::msg::Image>(*msg));
        },
        "raw");

    auto depth_sub = node->create_subscription<sensor_msgs::msg::Image>(
        depth_topic, 10,
        [&igb](const sensor_msgs::msg::Image::SharedPtr msg) {
            igb.GrabDepth(msg);
        });

#if UW_SLAM_HAS_STEREO_MATCHES
    auto matches_sub = node->create_subscription<uw_slam_bridge::msg::StereoMatches>(
        matches_topic, 10,
        [&igb](const uw_slam_bridge::msg::StereoMatches::SharedPtr msg) {
            igb.GrabMatches(msg);
        });
#else
    RCLCPP_WARN(node->get_logger(),
                "uw_slam_bridge/StereoMatches not found at build time; ORB stereo fallback remains active");
#endif

    thread sync_thread(&ImageGrabber::SyncWithImu, &igb);
    rclcpp::spin(node);
    rclcpp::shutdown();
    sync_thread.join();
    return 0;
}
