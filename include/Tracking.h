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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

// #include <ros/ros.h>  // original
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
// #include "MapDrawer.h"
#include <DVLGroPreIntegration.h>

#include "ImuTypes.h"

#include "GeometricCamera.h"

#include <mutex>
#include <shared_mutex>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ORB_SLAM3
{
class LKTracker;
class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class RosHandling;
class DenseMapper;
class Integrator;

class Tracking
{

public:
	Tracking(System *pSys,
	         ORBVocabulary *pVoc,
	         FrameDrawer *pFrameDrawer,
	         Atlas *pAtlas,
	         KeyFrameDatabase *pKFDB,
	         RosHandling *pRosHandler,
	         DenseMapper *pDenseMapper,
	         const string &strSettingPath,
	         const int sensor,
	         const string &_nameSeq = std::string());

	~Tracking();

	// Parse the config file
	bool ParseCamParamFile(cv::FileStorage &fSettings);
	bool ParseORBParamFile(cv::FileStorage &fSettings);
	bool ParseIMUParamFile(cv::FileStorage &fSettings);

	// Preprocess the input and call Track(). Extract features and performs stereo matching.
	cv::Mat GrabImageStereoDvl(const cv::Mat &imRectLeft,
	                           const cv::Mat &imRectRight,
	                           const double &timestamp,
	                           bool bDvl,
	                           string filename,
	                           const std::vector<cv::Point2f>& vExtLeft = std::vector<cv::Point2f>(),
	                           const std::vector<cv::Point2f>& vExtRight = std::vector<cv::Point2f>(),
	                           const std::vector<float>& vExtScore = std::vector<float>(),
	                           const cv::Mat& extDepthScaled = cv::Mat());
	cv::Mat GrabImageStereoDvlgyro(const cv::Mat &imRectLeft,
	                               const cv::Mat &imRectRight,
	                               const double &timestamp,
	                               bool bDvl,
	                               string filename,
	                               const std::vector<cv::Point2f>& vExtLeft = std::vector<cv::Point2f>(),
	                               const std::vector<cv::Point2f>& vExtRight = std::vector<cv::Point2f>(),
	                               const std::vector<float>& vExtScore = std::vector<float>(),
	                               const cv::Mat& extDepthScaled = cv::Mat());

	cv::Mat GrabImageStereoDvlKLT(const cv::Mat &imRectLeft,
	                              const cv::Mat &imRectRight,
	                              const double &timestamp,
	                              bool bDvl,
	                              string filename);

	void GrabImuData(const IMU::ImuPoint &imuMeasurement);
	void GrabDVLGyroData(const IMU::GyroDvlPoint &DVLGyrosMeasurement);

	void SetLocalMapper(LocalMapping *pLocalMapper);
	void SetLoopClosing(LoopClosing *pLoopClosing);
	// void SetViewer(Viewer *pViewer);
	void SetStepByStep(bool bSet);
	void SetExtrinsicPara(IMU::Calib &calib);

	// Load new settings
	// The focal lenght should be similar or scale prediction will fail when projecting points
	void ChangeCalibration(const string &strSettingPath);

	// Use this function if you have deactivated local mapping and you only want to localize the camera.
	void InformOnlyTracking(const bool &flag);

	void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame);
	void UpdateFrameDVLGyro(const IMU::Bias &b, KeyFrame *pCurrentKeyFrame);
	KeyFrame *GetLastKeyFrame()
	{
		return mpLastKeyFrame;
	}

	void CreateMapInAtlas();
	std::mutex mMutexTracks;

	//--
	void NewDataset();
	int GetNumberDataset();
	int GetMatchesInliers();

	// EKF DVL function
	void LoadEKFReading(const Eigen::Isometry3d &T_e0_ej, const double &timeEKF, bool good_EKF,
	                    const Eigen::Isometry3d &T_g0_gj, const Eigen::Matrix<double, 6, 1> &V_e);

	void topicPublishDVLOnly();

	inline std::string getImageType(int number)
	{
		// find type
		int imgTypeInt = number % 8;
		std::string imgTypeString;

		switch (imgTypeInt) {
		case 0:imgTypeString = "8U";
			break;
		case 1:imgTypeString = "8S";
			break;
		case 2:imgTypeString = "16U";
			break;
		case 3:imgTypeString = "16S";
			break;
		case 4:imgTypeString = "32S";
			break;
		case 5:imgTypeString = "32F";
			break;
		case 6:imgTypeString = "64F";
			break;
		default:break;
		}

		// find channel
		int channel = (number / 8) + 1;

		std::stringstream type;
		type << "CV_" << imgTypeString << "C" << channel;

		return type.str();
	}
public:

	// Tracking states
	enum eTrackingState
	{
		SYSTEM_NOT_READY = -1,
		NO_IMAGES_YET = 0,
		NOT_INITIALIZED = 1,
		OK = 2,
		RECENTLY_LOST = 3,
		LOST = 4,
		OK_KLT = 5,
		VISUAL_LOST = 6
	};

	eTrackingState mState;
	eTrackingState mLastProcessedState;

	// Input sensor
	int mSensor;

	// Current Frame
	Frame mCurrentFrame;
	Frame mLastFrame;
	// Frame mLastFrameBeforeLoss;
//    Frame mLastFrameBeforeLoss;
	// for debug
	Frame mCurrentFrame_orb;
	Frame mCurrentFrame_orb_ekf;
	Frame mCurrentFrame_ekf;
	Frame mLastFrame_orb;
	Frame mLastFrame_orb_ekf;
	Frame mLastFrame_ekf;

	cv::Mat mImLeft;

	// Initialization Variables (Monocular)
	std::vector<int> mvIniLastMatches;
	std::vector<int> mvIniMatches;
	std::vector<cv::Point2f> mvbPrevMatched;
	std::vector<cv::Point3f> mvIniP3D;
	Frame mInitialFrame;

	// Lists used to recover the full camera trajectory at the end of the execution.
	// Basically we store the reference keyframe for each frame and its relative transformation
	list<cv::Mat> mlRelativeFramePoses;
	list<KeyFrame *> mlpReferences;
	list<double> mlFrameTimes;
	list<bool> mlbLost;

	// frames with estimated pose
	int mTrackedFr;
	bool mbStep;

	// True if local mapping is deactivated and we are performing only localization
	bool mbOnlyTracking;

	void Reset(bool bLocMap = false);
	void ResetActiveMap(bool bLocMap = false);
	IMU::Calib GetExtrinsicPara();

	float mMeanTrack;
	bool mbInitWith3KFs;
	double t0; // time-stamp of first read frame
	double t0vis; // time-stamp of first inserted keyframe
	double t0IMU; // time-stamp of IMU initialization


	vector<MapPoint *> GetLocalMapMPS();

	//TEST--
	bool mbNeedRectify;
	//cv::Mat M1l, M2l;
	//cv::Mat M1r, M2r;

	bool mbWriteStats;

	// EKF DVL

	// the ekf reading for the current frame
	Eigen::Isometry3d mCurT_e0_ej;
	// velocity and angular_velocity
	Eigen::Matrix<double, 6, 1> mV_e;
//	Eigen::Vector3d mV_angular_e;
	double mCurTimeEKF;
	// calibration information between EKF and camera
	Eigen::Isometry3d mT_e_c;
	// from camera frame(x forward) to camera model frame(z forward)
	Eigen::Isometry3d mT_c_cm;
	// from EKF to GT
	Eigen::Isometry3d mT_g_e;
	Eigen::Isometry3d mCurT_g0_gj;

	bool mGood_EKF = false;

	//Tightly coupled DVL-Gro
	bool mbDVL = false;

	// weights for optimization
	double mlamda_visual;
	double mlamda_DVL;
	double mlamda_DVL_debug;
	double mDVL_func_debug;

protected:

	// Main tracking function. It is independent of the input sensor.
	void Track();
	void TrackDVLGyro();
	void TrackDVLImu();
	void TrackKLT();

	// Map initialization for stereo and RGB-D
	void StereoInitialization();
	void StereoInitializationKLT();

	// Map initialization for monocular
	void MonocularInitialization();
	void CreateNewMapPoints();
	cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);
	void CreateInitialMapMonocular();

	void CheckReplacedInLastFrame();
	bool TrackReferenceKeyFrame();
	void UpdateLastFrame();
	bool TrackWithMotionModel();
	bool TrackWithMotionModelAndEKF();
	void saveMatchingResults(const cv::Mat &velocity_orb, const cv::Mat &velocity_dvl);
	bool PredictStateIMU();
	bool PredictStateDvlGro();
    bool PredictStateGyro();

	bool Relocalization();

	void UpdateLocalMap();
	void UpdateLocalPoints();
	void UpdateLocalKeyFrames();

	bool TrackLocalMap();
	bool TrackLocalMapWithDvlGyro();
	void SaveOptimizationResult();
	void drawOptimizationResult();
	bool TrackLocalMap_old();
	void SearchLocalPoints();

	bool NeedNewKeyFrame();
	void CreateNewKeyFrame();
	void CreateNewKeyFrameKLT();

	// Perform preintegration from last frame
	void PreintegrateIMU();

	void PreintegrateDvlGro();
	/***
	 * 4 beam DVL model and Gyro pre_integration
	 */
	void PreintegrateDvlGro2();
	/***
	 * 4 beam DVL model and Gyro + Acc pre_integration
	 */
	void PreintegrateDvlGro3();

	// Reset IMU biases and compute frame velocity
	void ResetFrameIMU();
	void ComputeGyroBias(const vector<Frame *> &vpFs, float &bwx, float &bwy, float &bwz);
	void ComputeVelocitiesAccBias(const vector<Frame *> &vpFs, float &bax, float &bay, float &baz);

	// DVL EKF Tracking
	bool TrackWithVisualAndEKF();
	// update EKF pose according to velocity rading
	void IntegrateDVLVelocity();

	bool mbMapUpdated;

	// Imu preintegration from last frame
	IMU::Preintegrated *mpImuPreintegratedFromLastKF;
	DVLGroPreIntegration *mpDvlPreintegratedFromLastKF;

	// Queue of IMU measurements between frames
	std::list<IMU::ImuPoint> mlQueueImuData;

	std::list<IMU::GyroDvlPoint> mlQueueDVLGyroData;

	// Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
	std::vector<IMU::ImuPoint> mvImuFromLastFrame;
	std::vector<IMU::GyroDvlPoint> mvGyroDVLFromLastFrame;
	std::mutex mMutexImuQueue;

	// Imu calibration parameters
	IMU::Calib *mpImuCalib;
	std::mutex mMutexExtrinsic;

	// Last Bias Estimation (at keyframe creation)
	std::shared_mutex mBiasMutex;
	IMU::Bias mLastBias;

	// In case of performing only localization, this flag is true when there are no matches to
	// points in the map. Still tracking will continue if there are enough matches with temporal points.
	// In that case we are doing visual odometry. The system will try to do relocalization to recover
	// "zero-drift" localization to the map.
	bool mbVO;

	//Other Thread Pointers
	LocalMapping *mpLocalMapper;
	LoopClosing *mpLoopClosing;

	//ORB
	ORBextractor *mpORBextractorLeft, *mpORBextractorRight;
	ORBextractor *mpIniORBextractor;

	//BoW
	ORBVocabulary *mpORBVocabulary;
	KeyFrameDatabase *mpKeyFrameDB;

	// Initalization (only for monocular)
	Initializer *mpInitializer;
	bool mbSetInit;

	//Local Map
	KeyFrame *mpReferenceKF;
	std::vector<KeyFrame *> mvpLocalKeyFrames;
	std::vector<MapPoint *> mvpLocalMapPoints;

	// System
	System *mpSystem;

	//Drawers
	// Viewer *mpViewer;
	FrameDrawer *mpFrameDrawer;
	// MapDrawer *mpMapDrawer;
	bool bStepByStep;

	//Atlas
	Atlas *mpAtlas;
	Map *mpMapToReset;
	DenseMapper *mpDenseMapper;

	//Calibration matrix
	cv::Mat mK;
	cv::Mat mDistCoef;
	float mbf;

	//New KeyFrame rules (according to fps)
	int mMinFrames;
	int mMaxFrames;

	int mnFirstImuFrameId;
	int mnFramesToResetIMU;

	// Threshold close/far points
	// Points seen as close by the stereo/RGBD sensor are considered reliable
	// and inserted from just one frame. Far points requiere a match in two keyframes.
	float mThDepth;
	float mThFarDepth;
	float mThCloseDepth;

	// For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
	float mDepthMapFactor;

	//Current matches in frame
	int mnMatchesInliers;

	//Last Frame, KeyFrame and Relocalisation Info
	KeyFrame *mpLastKeyFrame;
	unsigned int mnLastKeyFrameId;
	unsigned int mnLastRelocFrameId;
	double mTimeStampLost;
	double time_recently_lost;

	unsigned int mnFirstFrameId;
	unsigned int mnInitialFrameId;
	unsigned int mnLastInitFrameId;

	bool mbCreatedMap;

	//Motion Model
	cv::Mat mVelocity;

	//Color order (true RGB, false BGR, ignored if grayscale)
	bool mbRGB;

	list<MapPoint *> mlpTemporalPoints;

	//int nMapChangeIndex;

	int mnNumDataset;

	ofstream f_track_stats;

	ofstream f_track_times;
	double mTime_PreIntIMU;
	double mTime_PosePred;
	double mTime_LocalMapTrack;
	double mTime_NewKF_Dec;

	GeometricCamera *mpCamera, *mpCamera2;

	int initID, lastID;

	cv::Mat mTlr;

	// ros pose publisher
// // 	ros::Publisher mPose_pub;  // original  // original
// // 	ros::Publisher mEKFPose_pub;  // original  // original

	float mKF_init_step;
	float mImageScale = 1.0f;

	Eigen::Vector4d mAlpha, mBeta;
	std::mutex mBeamOriMutex;

public:
	cv::Mat mImRight;
	RosHandling *mpRosHandler;
	bool mCalibrated;
    bool mInitialized;
	bool mVisualIntegration;
	int mKF_num_for_init;
	int mKFThresholdForMap;
	bool mDetectLoop = true;

	std::mutex mLossIntegrationRefMutex;
	DVLGroPreIntegration *mpDvlPreintegratedFromLastKFBeforeLost;
	bool mDoLossIntegration;
	DVLGroPreIntegration *getLossIntegrationRef();

	void SetBeamOrientation(const Eigen::Vector4d &alpha, const Eigen::Vector4d &beta);
	void GetBeamOrientation(Eigen::Vector4d &alpha, Eigen::Vector4d &beta);
	Integrator* mpIntegrator;

	LKTracker *mpLKTracker;

	IMU::Bias mLastBiasBeforeLost;

    //store all KF during camera loss
    std::shared_mutex mLossKFMutex;
    protected: set<KeyFrame*,KFComparator> mvpLossKF;

    int mLossFNum = 0;
    int mLossLastOptID = 0;
    // optimized v
    std::shared_mutex mVMutex;
    Eigen::Vector3d mVd_opt;
public:
    set<KeyFrame*, KFComparator> getMvpLossKf();
    void clearPartLossKF();

    void setOptV(const Eigen::Vector3d& v);
    void getOptV(Eigen::Vector3d& v);
};

} //namespace ORB_SLAM

#endif // TRACKING_H
