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


#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "GtsamMapAdapter.h"

#include "Atlas.h"
#include <boost/algorithm/string.hpp>
#include <array>
#include <thread>
#include <mutex>
// #include <ros/ros.h>  // original
#include <rclcpp/rclcpp.hpp>
// #include <image_transport/image_transport.h>  // original
#include <image_transport/image_transport.hpp>
// #include <sensor_msgs/Image.h>  // original
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <cv_bridge/cv_bridge.h>  // original
#include <cv_bridge/cv_bridge.hpp>

namespace ORB_SLAM3
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;
class Map;
class RosHandling;
class NeuralFeatureFrontend;


class LoopClosing
{
    friend class LoopClosingTestAccess;
public:

    typedef std::pair<std::set<KeyFrame*>,int> ConsistentGroup;

public:

    LoopClosing(Atlas* pAtlas, KeyFrameDatabase* pDB,
                RosHandling* pRosHandler, const bool bFixScale,
                int mergingThreshold, rclcpp::Node::SharedPtr node,
                NeuralFeatureFrontend* featureFrontend);

    void SetTracker(Tracking* pTracker);

	void SetTargetMap(int ID);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();
    void RequestResetActiveMap(Map* pMap);

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF);

    bool isRunningGBA(){
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

	void ClearQueue();

    // Viewer* mpViewer;
// //     ros::NodeHandlePtr mpNH;  // original  // original
    rclcpp::Node::SharedPtr mpNode;
    std::shared_ptr<image_transport::ImageTransport> mpIt;
    // publisher for current keyframe
    image_transport::Publisher mImgPub_cur_keyframe;
	// publisher for previous keyframe in the map which may have the loop
	image_transport::Publisher mImgPub_map_keyframe;


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    LoopClosing() = default;

    bool CheckNewKeyFrames();


    //Methods to implement the new place recognition algorithm
    bool NewDetectCommonRegions();
    void CorrectLoop();
    void CorrectMerge();


    void ResetIfRequested();
    bool mbResetRequested;
    bool mbResetActiveMapRequested;
    Map* mpMapToReset;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Atlas* mpAtlas;
    Tracking* mpTracker;
	RosHandling* mpRosHandler;

    KeyFrameDatabase* mpKeyFrameDB;
    NeuralFeatureFrontend* mpFeatureFrontend;

    LocalMapping *mpLocalMapper;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame* mpCurrentKF;
    KeyFrame* mpLastCurrentKF;
    KeyFrame* mpMatchedKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
    // all covisiable KF of current KF
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;
    std::vector<MapPoint*> mvpCurrentMatchedPoints;
    std::vector<MapPoint*> mvpLoopMapPoints;
    cv::Mat mScw;

    //-------
    Map* mpLastMap;

    bool mbLoopDetected;
    int mnLoopNumCoincidences;
    int mnLoopNumNotFound;
	// the last keyframe when the last loop detected
    KeyFrame* mpLoopLastCurrentKF;
	/**
	 * the best matched keyframe
	 * it will be set when the loop detected successfully
	 * it will be erase when the loop detection failed
	 */
    KeyFrame* mpLoopMatchedKF;
    std::vector<BackendLandmarkReplacement> mLoopLandmarkReplacements;
    std::vector<MapPoint*> mvpLoopMPs;
    std::vector<MapPoint*> mvpLoopMatchedMPs;
    bool mbMergeDetected;
    int mnMergeNumCoincidences;
    int mnMergeNumNotFound;
    KeyFrame* mpMergeLastCurrentKF;
    // the best keyframe selected from the candidate ketframes
    KeyFrame* mpMergeMatchedKF;
    std::array<float, 16> mMergeCurrentFromCandidate{};
    std::size_t mMergeInliers = 0;
    std::size_t mMergeCorrespondenceCount = 0;
    double mMergeRmsErrorMeters = 0.0;
    BackendMergeHypothesis mMergeHypothesis{256U, 2U};
    std::vector<MapPoint*> mvpMergeMPs;
    std::vector<MapPoint*> mvpMergeMatchedMPs;
    std::vector<KeyFrame*> mvpMergeConnectedKFs;

    //-------

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;


    int mnFullBAIdx;

    // service for planner
// // 	ros::ServiceClient mMergingSrv;  // original  // original
	int mMergingThreshold;
	int mTargetMapID;



    vector<double> vdPR_CurrentTime;
    vector<double> vdPR_MatchedTime;
    vector<int> vnPR_TypeRecogn;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
