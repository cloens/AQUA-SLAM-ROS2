/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef SYSTEM_H
#define SYSTEM_H

//#define SAVE_TIMES

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <opencv2/core/core.hpp>

// include ros
// #include <ros/ros.h>  // original
#include <rclcpp/rclcpp.hpp>
// #include <image_transport/image_transport.h>  // original
#include <image_transport/image_transport.hpp>
// #include <cv_bridge/cv_bridge.h>  // original
#include <cv_bridge/cv_bridge.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
// #include <sensor_msgs/image_encodings.h>  // original
#include <sensor_msgs/image_encodings.hpp>
// #include <sensor_msgs/Image.h>  // original
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
// #include <sensor_msgs/Imu.h>  // original
#include <sensor_msgs/msg/imu.hpp>
// #include <nav_msgs/Odometry.h>  // original
#include <nav_msgs/msg/odometry.hpp>
// #include <geometry_msgs/PoseStamped.h>  // original
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <std_srvs/srv/empty.hpp>

#include "RosHandling.h"
// #include "MapDrawer.h"
#include "DenseMapper.h"

#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "ImuTypes.h"


namespace ORB_SLAM3
{

class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET=0,
        VERBOSITY_NORMAL=1,
        VERBOSITY_VERBOSE=2,
        VERBOSITY_VERY_VERBOSE=3,
        VERBOSITY_DEBUG=4
    };

    static eLevel th;

public:
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            cout << str << endl;
    }

    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

class Viewer;
class FrameDrawer;
class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;

class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_MONOCULAR=3,
        IMU_STEREO=4,
        DVL_STEREO=5
    };

    // File type
    enum eFileType{
        TEXT_FILE=0,
        BINARY_FILE=1,
    };

public: bool mbResetActiveMap;
public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    // System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true, const int initFr = 0, const string &strSequence = std::string(), const string &strLoadingFile = std::string());  // original
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, rclcpp::Node::SharedPtr node, const bool bUseViewer = true, const int initFr = 0, const string &strSequence = std::string(), const string &strLoadingFile = std::string());

	cv::Mat TrackStereoGroDVL(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::ImuPoint>& vImuMeas = vector<IMU::ImuPoint>(), bool bDVL= false, string filename="",
	                         const std::vector<cv::Point2f>& vExtLeft = std::vector<cv::Point2f>(),
	                         const std::vector<cv::Point2f>& vExtRight = std::vector<cv::Point2f>(),
	                         const std::vector<float>& vExtScore = std::vector<float>(),
	                         const cv::Mat& extDepthScaled = cv::Mat());
	cv::Mat TrackStereoGroDVL(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::GyroDvlPoint>& vDVLGyroMeas = vector<IMU::GyroDvlPoint>(), bool bDVL= false, string filename="",
	                         const std::vector<cv::Point2f>& vExtLeft = std::vector<cv::Point2f>(),
	                         const std::vector<cv::Point2f>& vExtRight = std::vector<cv::Point2f>(),
	                         const std::vector<float>& vExtScore = std::vector<float>(),
	                         const cv::Mat& extDepthScaled = cv::Mat());
	cv::Mat TrackStereoImu(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp,
	                      const vector<IMU::ImuPoint>& vImuMeas = vector<IMU::ImuPoint>(), string filename="",
	                      const std::vector<cv::Point2f>& vExtLeft = std::vector<cv::Point2f>(),
	                      const std::vector<cv::Point2f>& vExtRight = std::vector<cv::Point2f>(),
	                      const std::vector<float>& vExtScore = std::vector<float>(),
	                      const cv::Mat& extDepthScaled = cv::Mat());
	cv::Mat TrackStereoGroDVLKLT(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::ImuPoint>& vImuMeas = vector<IMU::ImuPoint>(), bool bDVL= false, string filename="");

    void dvlCallBack(const nav_msgs::msg::Odometry::SharedPtr &dvl);

    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();

    // Reset the system (clear Atlas or the active map)
    void Reset();
    void ResetActiveMap();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);
	void SaveKeyFrameDataToMat(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename);

    // save all KeyFrame Pose
    void SaveKeyFrameTrajectory(const string &filename);

    // Save data used for initialization debug
    void SaveDebugData(const int &iniIdx);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

    // For debugging
    double GetTimeFromIMUInit();
    bool isLost();
    bool isFinished();

    void ChangeDataset();

    void SaveAtlas(const string &out_path, int type);
	bool LoadAtlas(string filename, int type);
	DenseMapper* mpDenseMapper;
	string mStrVocabularyFilePath;

private:


    string CalculateCheckSum(string filename, int type);

    // Input sensor
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;
    Atlas* mpAtlas;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    // Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;
    // MapDrawer* mpMapDrawer;


    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;

	// Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Tracking state
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;

    // DVL EKF
    bool mDepth_initialized=false;
    bool mOdom_initialized=false;
    bool mGT_initialized=false;

    double mInitialized_depth=0;
    // from origin to the first EKF frame
    Eigen::Isometry3d mT_w_e0;
    // from origin to the first Ground Truth frame
    Eigen::Isometry3d mT_w_g0;
	// transformation from camera to EKF
//	Eigen::Isometry3d mT_e_c;
	// velocity from EKF
	Eigen::Matrix<double,6,1> mV_e;
	// velocity
//	Eigen::Vector3d mV_angular_e;

    // whether use EKF readings in the current Frame
    // because DVL has low frequency. we only use the first EKF reading after DVL reading
    mutex mDVL_state_lock;
    bool mDVL_updated=false;

    image_transport::Publisher* mImg_l_pub=NULL;
    image_transport::Publisher* mImg_r_pub=NULL;
// //     ros::Publisher* mGt_pub=NULL;  // original  // original
    RosHandling* mRosHandler;

    rclcpp::Node::SharedPtr mp_node;


};

}// namespace ORB_SLAM

#endif // SYSTEM_H
