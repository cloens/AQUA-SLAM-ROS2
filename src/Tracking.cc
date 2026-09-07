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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/core/eigen.hpp>

#include"LKTracker.h"
#include"FeatureMatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Initializer.h"
#ifndef AQUA_HAS_GTSAM_DYNAMIC
#include"G2oTypes.h"
#include"Optimizer.h"
#endif
#include"PnPsolver.h"
#include "Pinhole.h"
// #include"Viewer.h"
#include"FrameDrawer.h"
#include"Atlas.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include "System.h"
#include "src/Integrator.h"
#include <DVLGroPreIntegration.h>
#include "MotionModelPose.h"

#include<iostream>
#include <filesystem>
#include <stdexcept>
#include <fstream>

#include<mutex>
#include<chrono>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <include/MLPnPsolver.h>
#include <RosHandling.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


using namespace std;

namespace ORB_SLAM3
{


Tracking::Tracking(System *pSys,
	               FrameDrawer *pFrameDrawer,
                   Atlas *pAtlas,
                   KeyFrameDatabase *pKFDB,
                   RosHandling *pRosHandler,
                   DenseMapper *pDenseMapper,
                   const string &strSettingPath,
                   const int sensor,
                   const string &_nameSeq,
                   std::shared_ptr<const BackendFacade> backendFacade)
	:
	mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
	mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpKeyFrameDB(pKFDB),
	mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys),
	mpFrameDrawer(pFrameDrawer),  mpAtlas(pAtlas), mnLastRelocFrameId(0),
	time_recently_lost(5.0), mpMapToReset(nullptr), mpDenseMapper(pDenseMapper),
	mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpRosHandler(pRosHandler),
    mpDvlPreintegratedFromLastKF(nullptr), mpBackendFacade(std::move(backendFacade))
{
	// initialize the pose pulisher
// // 	ros::NodeHandle n;  // original  // original
	// mPose_pub = n.advertise<geometry_msgs::PoseStamped>("/AQUA_SLAM/orb_pose", 10);
	// mEKFPose_pub = n.advertise<geometry_msgs::PoseStamped>("/AQUA_SLAM/preintegrated_pose", 10);
	// Load camera parameters from settings file
	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

	// set the transformation from calibrated camera to camera model
	mT_c_cm = Eigen::Isometry3d::Identity();
	Eigen::AngleAxisd r1(M_PI / 2, Eigen::Vector3d::UnitY());
	Eigen::AngleAxisd r2(-M_PI / 2, Eigen::Vector3d::UnitZ());
	mT_c_cm.rotate(r1);
	mT_c_cm.rotate(r2);

	bool b_parse_cam = ParseCamParamFile(fSettings);
	if (!b_parse_cam) {
		std::cout << "*Error with the camera parameters in the config file*" << std::endl;
	}

	if (!ParseFeatureFrontendParamFile(fSettings))
		throw std::runtime_error("invalid FeatureFrontend configuration");

	initID = 0;
	lastID = 0;

	// Load IMU parameters
	bool b_parse_imu = true;
	if (sensor == System::IMU_MONOCULAR || sensor == System::IMU_STEREO || sensor == System::DVL_STEREO) {
	b_parse_imu = ParseIMUParamFile(fSettings);
		if (!b_parse_imu) {
			std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
		}

		mnFramesToResetIMU = mMaxFrames;
	}
	if (!mpBackendFacade)
		throw std::logic_error("Tracking requires the process backend facade");
	mBackendMode = mpBackendFacade->mode();
	if (sensor == System::DVL_STEREO &&
	    mBackendMode == BackendMode::GtsamDynamic) {
		const std::string trackMode = (std::string)fSettings["DvlTrackMode"];
		if (trackMode == "bottom_track")
			mDvlTrackMode = DvlTrackMode::BottomTrack;
		else if (trackMode == "water_track")
			mDvlTrackMode = DvlTrackMode::WaterTrack;
		else
			throw std::runtime_error("DvlTrackMode must be bottom_track or water_track");
		const std::string messageContract =
			(std::string)fSettings["DvlMessageContract"];
		if (messageContract != "uw_slam_bridge/msg/DvlObservation")
			throw std::runtime_error("DVL requires uw_slam_bridge/msg/DvlObservation");
		std::filesystem::path uncertaintyProfile(
			(std::string)fSettings["DvlUncertaintyProfile"]);
		if (uncertaintyProfile.empty())
			throw std::runtime_error("DvlUncertaintyProfile is missing");
		if (uncertaintyProfile.is_relative())
			uncertaintyProfile = std::filesystem::path(strSettingPath).parent_path() /
				uncertaintyProfile;
		uncertaintyProfile = uncertaintyProfile.lexically_normal();
		if (!std::filesystem::is_regular_file(uncertaintyProfile))
			throw std::runtime_error("DvlUncertaintyProfile does not exist: " +
			                         uncertaintyProfile.string());
		mDvlUncertaintyProfilePath = uncertaintyProfile.string();
	}
	// DVL is an optional measurement channel.  Its absence must not gate
	// stereo-inertial startup; when present it is forwarded to the GTSAM
	// transaction by the sensor-specific adapter path.
#ifndef AQUA_HAS_GTSAM_DYNAMIC
	if (mBackendMode == BackendMode::GtsamDynamic)
		throw std::runtime_error("GTSAM_DYNAMIC requested but AQUA was built without UW_DYNAMIC_BACKEND_ROOT");
#else
	mCollectDynamicInitializationImu =
		mBackendMode == BackendMode::GtsamDynamic;
#endif
	if (sensor == System::IMU_STEREO || sensor == System::DVL_STEREO) {
		mKF_init_step = (double)fSettings["Optimizer.KF_init_step"];
		mKF_num_for_init = fSettings["Optimizer.KF_num_for_init"];
		mCalibrated = ((double)fSettings["Optimizer.calibrated"]) == 1;
		mInitialized = sensor != System::DVL_STEREO;
		mKFThresholdForMap = (int)fSettings["Tracker.KFThresholdForMap"];
		time_recently_lost = (double)fSettings["Tracker.ReLocalTime"];
		mDetectLoop = ((int)fSettings["EnableLoopDetection"]) == 1;
		mVisualIntegration = ((int)fSettings["VisualIntegration"]) == 1;
	}
	// Load Optimizer parameters
	if (sensor == System::DVL_STEREO) {
		mlamda_visual = fSettings["Optimizer.visual_lamda"];
		mlamda_DVL = fSettings["Optimizer.DVL_lamda"];
		mlamda_DVL_debug = fSettings["Optimizer.DVL_lamda_debug"];
		mDVL_func_debug = fSettings["Optimizer.DVL_func"];
		cout << "Optimizer.visual_lamda: " << mlamda_visual << endl;
		cout << "Optimizer.DVL_lamda: " << mlamda_DVL << endl;
		cout << "Optimizer.DVL_lamda_debug: " << mlamda_DVL_debug << endl;
		cout << "Optimizer.DVL_func: " << mDVL_func_debug << endl;
	}

	mbInitWith3KFs = false;


	mnNumDataset = 0;

	if (!b_parse_cam || !b_parse_imu) {
		std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
		try {
			throw -1;
		}
		catch (exception &e) {

		}
	}


	mCurT_e0_ej = Eigen::Isometry3d::Identity();
	mCurTimeEKF = 0;
	// set the transformation from camera to EKF T_e_c
	mT_e_c = Eigen::Isometry3d::Identity();
	Eigen::Matrix3d r;
	Eigen::Vector3d t;
	r << 0.9947841924, -0.09307539064, 0.0417298708,
		0.09538717271, 0.9937878685, -0.05733201293,
		-0.03613443986, 0.06101347458, 0.9974826606;
	t << 0.2784991001, -0.1103631965, 0.07719669752;
	mT_e_c.rotate(r);
	mT_e_c.pretranslate(t);


	mT_e_c = mT_e_c * mT_c_cm;

	mT_g_e = Eigen::Isometry3d::Identity();
//    t=Eigen::Vector3d(-0.144514, 0.0205092, 0.0137455);
//    r<<0.998376, 	-0.00950652, 	-0.0561734,
//	0.0124389,		0.998565,		0.0520846,
//	0.0555977,		-0.0526987,		0.997062;

	t = Eigen::Vector3d(-0.132592, 0.00587571, 0.00178882);
	r << 0.997896, -0.014342, -0.0632293,
		0.0177779, 0.998376, 0.0541173,
		0.0623505, -0.0551275, 0.996531;
	r1 = Eigen::AngleAxisd(-M_PI, Eigen::Vector3d::UnitX());
	// Eigen::AngleAxisd r2(-M_PI / 2, Eigen::Vector3d::UnitZ());
	mT_g_e.rotate(r1);
	mT_g_e.rotate(r);
	mT_g_e.pretranslate(t);
	mCurT_g0_gj = Eigen::Isometry3d::Identity();

	// init LK tracker
	// mpLKTracker = new LKTracker(mSensor == System::DVL_STEREO);

    mpIntegrator = new Integrator();
    mpIntegrator->CreateNewIntFromKF_C2C(IMU::Bias(),GetExtrinsicPara(),mAlpha,mBeta);
}

Tracking::~Tracking()
{
	//f_track_stats.close();
#ifdef SAVE_TIMES
	f_track_times.close();
#endif
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
	mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
	cout << endl << "Camera Parameters: " << endl;
	bool b_miss_params = false;

	string sCameraName = fSettings["Camera.type"];
	if (sCameraName == "PinHole") {
		float fx, fy, cx, cy, image_scale;

		image_scale = (float)fSettings["ImageScale"];
		mImageScale = image_scale;
		cout << "set image scale: " << mImageScale << endl;

		// Camera calibration parameters
		cv::FileNode node = fSettings["Camera.fx"];
		if (!node.empty() && node.isReal()) {
			fx = node.real();
		}
		else {
			std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.fy"];
		if (!node.empty() && node.isReal()) {
			fy = node.real();
		}
		else {
			std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.cx"];
		if (!node.empty() && node.isReal()) {
			cx = node.real();
		}
		else {
			std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.cy"];
		if (!node.empty() && node.isReal()) {
			cy = node.real();
		}
		else {
			std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		// Distortion parameters
		node = fSettings["Camera.k1"];
		if (!node.empty() && node.isReal()) {
			mDistCoef.at<float>(0) = node.real();
		}
		else {
			std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.k2"];
		if (!node.empty() && node.isReal()) {
			mDistCoef.at<float>(1) = node.real();
		}
		else {
			std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.p1"];
		if (!node.empty() && node.isReal()) {
			mDistCoef.at<float>(2) = node.real();
		}
		else {
			std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.p2"];
		if (!node.empty() && node.isReal()) {
			mDistCoef.at<float>(3) = node.real();
		}
		else {
			std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.k3"];
		if (!node.empty() && node.isReal()) {
			mDistCoef.resize(5);
			mDistCoef.at<float>(4) = node.real();
		}

		if (b_miss_params) {
			return false;
		}

		fx = fx * mImageScale;
		fy = fy * mImageScale;
		cx = cx * mImageScale;
		cy = cy * mImageScale;
		vector<float> vCamCalib{fx, fy, cx, cy};

		mpCamera = new Pinhole(vCamCalib);

		mpAtlas->AddCamera(mpCamera);


		std::cout << "- Camera: Pinhole" << std::endl;
		std::cout << "- fx: " << fx << std::endl;
		std::cout << "- fy: " << fy << std::endl;
		std::cout << "- cx: " << cx << std::endl;
		std::cout << "- cy: " << cy << std::endl;
		std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
		std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


		std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
		std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

		if (mDistCoef.rows == 5) {
			std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;
		}

		mK = cv::Mat::eye(3, 3, CV_32F);
		mK.at<float>(0, 0) = fx;
		mK.at<float>(1, 1) = fy;
		mK.at<float>(0, 2) = cx;
		mK.at<float>(1, 2) = cy;

	}
	else if (sCameraName == "KannalaBrandt8") {
		float fx, fy, cx, cy;
		float k1, k2, k3, k4;

		// Camera calibration parameters
		cv::FileNode node = fSettings["Camera.fx"];
		if (!node.empty() && node.isReal()) {
			fx = node.real();
		}
		else {
			std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}
		node = fSettings["Camera.fy"];
		if (!node.empty() && node.isReal()) {
			fy = node.real();
		}
		else {
			std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.cx"];
		if (!node.empty() && node.isReal()) {
			cx = node.real();
		}
		else {
			std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.cy"];
		if (!node.empty() && node.isReal()) {
			cy = node.real();
		}
		else {
			std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		// Distortion parameters
		node = fSettings["Camera.k1"];
		if (!node.empty() && node.isReal()) {
			k1 = node.real();
		}
		else {
			std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}
		node = fSettings["Camera.k2"];
		if (!node.empty() && node.isReal()) {
			k2 = node.real();
		}
		else {
			std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.k3"];
		if (!node.empty() && node.isReal()) {
			k3 = node.real();
		}
		else {
			std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		node = fSettings["Camera.k4"];
		if (!node.empty() && node.isReal()) {
			k4 = node.real();
		}
		else {
			std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

		if (!b_miss_params) {
			vector<float> vCamCalib{fx, fy, cx, cy, k1, k2, k3, k4};
			mpCamera = new KannalaBrandt8(vCamCalib);

			std::cout << "- Camera: Fisheye" << std::endl;
			std::cout << "- fx: " << fx << std::endl;
			std::cout << "- fy: " << fy << std::endl;
			std::cout << "- cx: " << cx << std::endl;
			std::cout << "- cy: " << cy << std::endl;
			std::cout << "- k1: " << k1 << std::endl;
			std::cout << "- k2: " << k2 << std::endl;
			std::cout << "- k3: " << k3 << std::endl;
			std::cout << "- k4: " << k4 << std::endl;

			mK = cv::Mat::eye(3, 3, CV_32F);
			mK.at<float>(0, 0) = fx;
			mK.at<float>(1, 1) = fy;
			mK.at<float>(0, 2) = cx;
			mK.at<float>(1, 2) = cy;
		}

		if (mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::DVL_STEREO) {
			// Right camera
			// Camera calibration parameters
			cv::FileNode node = fSettings["Camera2.fx"];
			if (!node.empty() && node.isReal()) {
				fx = node.real();
			}
			else {
				std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
				b_miss_params = true;
			}
			node = fSettings["Camera2.fy"];
			if (!node.empty() && node.isReal()) {
				fy = node.real();
			}
			else {
				std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
				b_miss_params = true;
			}

			node = fSettings["Camera2.cx"];
			if (!node.empty() && node.isReal()) {
				cx = node.real();
			}
			else {
				std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
				b_miss_params = true;
			}

			node = fSettings["Camera2.cy"];
			if (!node.empty() && node.isReal()) {
				cy = node.real();
			}
			else {
				std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
				b_miss_params = true;
			}

			// Distortion parameters
			node = fSettings["Camera2.k1"];
			if (!node.empty() && node.isReal()) {
				k1 = node.real();
			}
			else {
				std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
				b_miss_params = true;
			}
			node = fSettings["Camera2.k2"];
			if (!node.empty() && node.isReal()) {
				k2 = node.real();
			}
			else {
				std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
				b_miss_params = true;
			}

			node = fSettings["Camera2.k3"];
			if (!node.empty() && node.isReal()) {
				k3 = node.real();
			}
			else {
				std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
				b_miss_params = true;
			}

			node = fSettings["Camera2.k4"];
			if (!node.empty() && node.isReal()) {
				k4 = node.real();
			}
			else {
				std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
				b_miss_params = true;
			}


			int leftLappingBegin = -1;
			int leftLappingEnd = -1;

			int rightLappingBegin = -1;
			int rightLappingEnd = -1;

			node = fSettings["Camera.lappingBegin"];
			if (!node.empty() && node.isInt()) {
				leftLappingBegin = node.operator int();
			}
			else {
				std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
			}
			node = fSettings["Camera.lappingEnd"];
			if (!node.empty() && node.isInt()) {
				leftLappingEnd = node.operator int();
			}
			else {
				std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
			}
			node = fSettings["Camera2.lappingBegin"];
			if (!node.empty() && node.isInt()) {
				rightLappingBegin = node.operator int();
			}
			else {
				std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
			}
			node = fSettings["Camera2.lappingEnd"];
			if (!node.empty() && node.isInt()) {
				rightLappingEnd = node.operator int();
			}
			else {
				std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
			}

			node = fSettings["Tlr"];
			if (!node.empty()) {
				mTlr = node.mat();
				if (mTlr.rows != 3 || mTlr.cols != 4) {
					std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
					b_miss_params = true;
				}
			}
			else {
				std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
				b_miss_params = true;
			}

			if (!b_miss_params) {
				static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
				static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

				mpFrameDrawer->both = true;

				vector<float> vCamCalib2{fx, fy, cx, cy, k1, k2, k3, k4};
				mpCamera2 = new KannalaBrandt8(vCamCalib2);

				static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
				static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

				std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

				std::cout << std::endl << "Camera2 Parameters:" << std::endl;
				std::cout << "- Camera: Fisheye" << std::endl;
				std::cout << "- fx: " << fx << std::endl;
				std::cout << "- fy: " << fy << std::endl;
				std::cout << "- cx: " << cx << std::endl;
				std::cout << "- cy: " << cy << std::endl;
				std::cout << "- k1: " << k1 << std::endl;
				std::cout << "- k2: " << k2 << std::endl;
				std::cout << "- k3: " << k3 << std::endl;
				std::cout << "- k4: " << k4 << std::endl;

				std::cout << "- mTlr: \n" << mTlr << std::endl;

				std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
			}
		}

		if (b_miss_params) {
			return false;
		}

		mpAtlas->AddCamera(mpCamera);
		mpAtlas->AddCamera(mpCamera2);
	}
	else {
		std::cerr << "*Not Supported Camera Sensor*" << std::endl;
		std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
	}

	if (mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::DVL_STEREO) {
		cv::FileNode node = fSettings["Camera.bf"];
		if (!node.empty() && node.isReal()) {
			mbf = node.real();
			mbf = mbf * mImageScale;
		}
		else {
			std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

	}

	float fps = fSettings["Camera.fps"];
	if (fps == 0) {
		fps = 30;
	}

	// Max/Min Frames to insert keyframes and to check relocalisation
	mMinFrames = 0;
	mMaxFrames = fps;

	cout << "- fps: " << fps << endl;


	int nRGB = fSettings["Camera.RGB"];
	mbRGB = nRGB;

	if (mbRGB) {
		cout << "- color order: RGB (ignored if grayscale)" << endl;
	}
	else {
		cout << "- color order: BGR (ignored if grayscale)" << endl;
	}

	if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO
		|| mSensor == System::DVL_STEREO) {
		float fx = mpCamera->getParameter(0);
		mThFarDepth = (float)fSettings["ThFarDepth"];
		mThCloseDepth = (float)fSettings["ThCloaseDepth"];
		cv::FileNode node = fSettings["ThDepth"];
		if (!node.empty() && node.isReal()) {
			mThDepth = node.real();
			mThDepth = mbf * mThDepth / fx;
			cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
		}
		else {
			std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

	}

	if (mSensor == System::RGBD) {
		cv::FileNode node = fSettings["DepthMapFactor"];
		if (!node.empty() && node.isReal()) {
			mDepthMapFactor = node.real();
			if (fabs(mDepthMapFactor) < 1e-5) {
				mDepthMapFactor = 1;
			}
			else {
				mDepthMapFactor = 1.0f / mDepthMapFactor;
			}
		}
		else {
			std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
			b_miss_params = true;
		}

	}

	if (b_miss_params) {
		return false;
	}

	return true;
}

bool Tracking::ParseFeatureFrontendParamFile(cv::FileStorage &fSettings)
{
	bool b_miss_params = false;
	NeuralFeatureFrontend::Config config;
	auto readString = [&](const char* key, std::string& value) {
		const cv::FileNode node = fSettings[key];
		if (node.empty() || !node.isString()) {
			std::cerr << "*" << key << " must be a string*" << std::endl;
			b_miss_params = true;
			return;
		}
		value = static_cast<std::string>(node);
	};
	auto readReal = [&](const char* key, float& value) {
		const cv::FileNode node = fSettings[key];
		if (node.empty() || (!node.isReal() && !node.isInt())) {
			std::cerr << "*" << key << " must be numeric*" << std::endl;
			b_miss_params = true;
			return;
		}
		value = static_cast<float>(node.real());
	};

	std::string type;
	std::string provider;
	readString("FeatureFrontend.type", type);
	readString("FeatureFrontend.provider", provider);
	readString("FeatureFrontend.superpoint_onnx", config.superpoint_model);
	readString("FeatureFrontend.superpoint_sha256", config.superpoint_sha256);
	readString("FeatureFrontend.lightglue_onnx", config.lightglue_model);
	readString("FeatureFrontend.lightglue_sha256", config.lightglue_sha256);
	const cv::FileNode maxFeatures = fSettings["FeatureFrontend.max_features"];
	if (maxFeatures.empty() || !maxFeatures.isInt()) {
		std::cerr << "*FeatureFrontend.max_features must be an integer*" << std::endl;
		b_miss_params = true;
	} else {
		config.max_features = static_cast<int>(maxFeatures);
	}
	readReal("FeatureFrontend.match_threshold", config.match_threshold);
	readReal("FeatureFrontend.keypoint_threshold", config.keypoint_threshold);
	readReal("FeatureFrontend.projection_descriptor_threshold",
	         config.projection_descriptor_threshold);
	readReal("FeatureFrontend.stereo_max_vertical_error_px",
	         mStereoMaxVerticalError);
	if (type != "SUPERPOINT_LIGHTGLUE") {
		std::cerr << "*FeatureFrontend.type must be SUPERPOINT_LIGHTGLUE*" << std::endl;
		b_miss_params = true;
	}
	if (provider != "CUDA") {
		std::cerr << "*FeatureFrontend.provider must be CUDA*" << std::endl;
		b_miss_params = true;
	}

	if (b_miss_params) {
		return false;
	}
	mnFeatureTarget = config.max_features;
	mpFeatureFrontend = std::make_unique<NeuralFeatureFrontend>(config);
	cout << endl << "Feature Frontend: SUPERPOINT_LIGHTGLUE" << endl;
	cout << "- Provider: " << mpFeatureFrontend->Provider() << endl;
	cout << "- Maximum Features: " << mnFeatureTarget << endl;
	cout << "- Match Threshold: " << config.match_threshold << endl;
	cout << "- Stereo Vertical Error: " << mStereoMaxVerticalError << " px" << endl;

	return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
	bool b_miss_params = false;

	cv::Mat Tbc;
	cv::FileNode node = fSettings["Tbc"];
//	if (!node.empty()) {
//		Tbc = node.mat();
//		if (Tbc.rows != 4 || Tbc.cols != 4) {
//			std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
//			b_miss_params = true;
//		}
//	}
//	else {
//		std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
//		b_miss_params = true;
//	}

	cv::Mat T_imu_c;
	node = fSettings["T_imu_c"];
	if (!node.empty()) {
		T_imu_c = node.mat();
		if (T_imu_c.rows != 4 || T_imu_c.cols != 4) {
			std::cerr << "*T_imu_c matrix have to be a 4x4 transformation matrix*" << std::endl;
			b_miss_params = true;
		}
	}
	else {
		std::cerr << "*T_imu_c matrix doesn't exist*" << std::endl;
		b_miss_params = true;
	}

	cv::Mat T_dvl_c;
	node = fSettings["T_dvl_c"];
	if (!node.empty()) {
		T_dvl_c = node.mat();
		if (T_dvl_c.rows != 4 || T_dvl_c.cols != 4) {
			std::cerr << "*T_dvl_c matrix have to be a 4x4 transformation matrix*" << std::endl;
			b_miss_params = true;
		}
	}
	else {
		std::cerr << "*T_dvl_c matrix doesn't exist*" << std::endl;
		b_miss_params = true;
	}

	cout << endl;

	cv::Mat alpha;
	node = fSettings["alpha"];
	if (!node.empty()) {
		alpha = node.mat();
		if (T_dvl_c.rows != 4 || T_dvl_c.cols != 4) {
			std::cerr << "*T_dvl_c matrix have to be a 4x4 transformation matrix*" << std::endl;
			b_miss_params = true;
		}
	}
	else {
		std::cerr << "*T_dvl_c matrix doesn't exist*" << std::endl;
		b_miss_params = true;
	}
	cv::Mat beta;
	node = fSettings["beta"];
	if (!node.empty()) {
		beta = node.mat();
		if (T_dvl_c.rows != 4 || T_dvl_c.cols != 4) {
			std::cerr << "*T_dvl_c matrix have to be a 4x4 transformation matrix*" << std::endl;
			b_miss_params = true;
		}
	}
	else {
		std::cerr << "*T_dvl_c matrix doesn't exist*" << std::endl;
		b_miss_params = true;
	}

	cout << "DVL beam orientation Alpha: " << alpha << endl;
	cout << "DVL beam orientation Beta: " << beta << endl;
	Eigen::Vector4d alpha_e, beta_e;
	cv::cv2eigen(alpha, alpha_e);
	cv::cv2eigen(beta, beta_e);
	SetBeamOrientation(alpha_e, beta_e);

	cout << "Camera to IMU Transform (T_imu_c): " << endl << T_imu_c << endl;

	float freq, Ng, Na, Ngw, Naw;

	node = fSettings["IMU.Frequency"];
	if (!node.empty() && node.isInt()) {
		freq = node.operator int();
	}
	else {
		std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
		b_miss_params = true;
	}

	node = fSettings["IMU.NoiseGyro"];
	if (!node.empty() && node.isReal()) {
		Ng = node.real();
	}
	else {
		std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
		b_miss_params = true;
	}

	node = fSettings["IMU.NoiseAcc"];
	if (!node.empty() && node.isReal()) {
		Na = node.real();
	}
	else {
		std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
		b_miss_params = true;
	}

	node = fSettings["IMU.GyroWalk"];
	if (!node.empty() && node.isReal()) {
		Ngw = node.real();
	}
	else {
		std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
		b_miss_params = true;
	}

	node = fSettings["IMU.AccWalk"];
	if (!node.empty() && node.isReal()) {
		Naw = node.real();
	}
	else {
		std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
		b_miss_params = true;
	}

	if (b_miss_params) {
		return false;
	}

	const float sf = sqrt(freq);
	mImuFrequencyHz = freq;
	cout << endl;
	cout << "IMU frequency: " << freq << " Hz" << endl;
	cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
	cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
	cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
	cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

	cv::Mat T_c_cm;
	cv::eigen2cv(mT_c_cm.matrix(), T_c_cm);
	T_c_cm.convertTo(T_c_cm, CV_32FC1);
	cv::Mat T_bi_be;
	Eigen::AngleAxisd r_i_e(M_PI, Eigen::Vector3d::UnitY());
	Eigen::Isometry3d T_bi_be_eigen = Eigen::Isometry3d::Identity();
	T_bi_be_eigen.rotate(r_i_e);
	cv::eigen2cv(T_bi_be_eigen.matrix(), T_bi_be);
	T_bi_be.convertTo(T_bi_be, CV_32FC1);
	Eigen::Isometry3d T_imu_c_eigen;
	cv::cv2eigen(T_imu_c, T_imu_c_eigen.matrix());
	Eigen::Isometry3d T_dvl_c_eigen;
	cv::cv2eigen(T_dvl_c, T_dvl_c_eigen.matrix());
//    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);
	mpImuCalib = new IMU::Calib(T_imu_c, T_dvl_c, Ng*sf,Na*sf,Ngw/sf,Naw/sf);

	cv::FileNode T_body_imu_node = fSettings["T_body_imu"];
	if (!T_body_imu_node.empty()) {
		cv::Mat T_body_imu = T_body_imu_node.mat();
		if (T_body_imu.rows == 4 && T_body_imu.cols == 4) {
			mpImuCalib->mT_body_imu = T_body_imu.clone();
		}
	}
//	IMU::Calib c_test=GetExtrinsicPara();
//	IMU::Calib c_test2;
//	SetExtrinsicPara(c_test2);
//	cout<<"Tbc in file: "<<Tbc<<endl;
//    cout<<"Tbc: "<<mpImuCalib->Tbc<<endl;
//	cout<<"Tcb: "<<mpImuCalib->Tcb<<endl;

	mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), GetExtrinsicPara());
//	mpDvlPreintegratedFromLastKF = new DVLGroPreIntegration(IMU::Bias(), GetExtrinsicPara());


	return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
	mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
	mpLoopClosing = pLoopClosing;
}

// void Tracking::SetViewer(Viewer *pViewer)
// {
// 	mpViewer = pViewer;
// }

void Tracking::SetStepByStep(bool bSet)
{
	bStepByStep = bSet;
}

cv::Mat Tracking::GrabImageStereoDvl(const cv::Mat &imRectLeft,
                                     const cv::Mat &imRectRight,
                                     const double &timestamp,
                                     bool bDvl, string filename,
                                     const std::vector<cv::Point2f>& vExtLeft,
                                     const std::vector<cv::Point2f>& vExtRight,
                                     const std::vector<float>& vExtScore,
                                     const cv::Mat& extDepthScaled)
{
#ifdef AQUA_HAS_GTSAM_DYNAMIC
    auto transaction = AcquireDynamicMapTransaction();
#endif
	mImLeft = imRectLeft.clone();
	cv::Mat imGrayRight = imRectRight.clone();
	mImRight = imRectRight.clone();

	cv::resize(mImLeft, mImLeft, cv::Size(mImLeft.cols * mImageScale, mImLeft.rows * mImageScale));
	cv::resize(imGrayRight, imGrayRight, cv::Size(imGrayRight.cols * mImageScale, imGrayRight.rows * mImageScale));
	cv::resize(mImRight, mImRight, cv::Size(mImRight.cols * mImageScale, mImRight.rows * mImageScale));

//	if (mImGray.channels() == 3) {
//		if (mbRGB) {
//			cvtColor(mImGray, mImGray, CV_RGB2GRAY);
//			cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
//		}
//		else {
//			cvtColor(mImGray, mImGray, CV_BGR2GRAY);
//			cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
//		}
//	}
//	else if (mImGray.channels() == 4) {
//		if (mbRGB) {
//			cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
//			cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
//		}
//		else {
//			cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
//			cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
//		}
//	}


	if ((mSensor == System::DVL_STEREO || mSensor == System::IMU_STEREO) && !mpCamera2) {
		// EKF DVL
		// tighly coupled DVL
		mCurrentFrame = Frame(mImLeft,
		                      imGrayRight,
			                      timestamp,
			                      *mpFeatureFrontend,
			                      mK,
		                      mDistCoef,
		                      mbf,
		                      mThFarDepth,
		                      mThCloseDepth,
		                      mStereoMaxVerticalError,
		                      mpCamera,
		                      bDvl,
		                      &mLastFrame,
		                      GetExtrinsicPara());
	}
	else {
// 		ROS_ERROR_STREAM("Wrong Mode!");  // original
        assert(0);
	}


	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	
	if (!vExtLeft.empty() && !vExtRight.empty()) {
		mCurrentFrame.ApplyExternalStereoMatches(vExtLeft, vExtRight, vExtScore);
	}
	if (!extDepthScaled.empty()) {
		mCurrentFrame.SetExternalDepth(extDepthScaled);
	}
mCurrentFrame.mNameFile = filename;
	mCurrentFrame.mnDataset = mnNumDataset;

	Track();

#ifdef AQUA_HAS_GTSAM_DYNAMIC
    if (mBackendMode == BackendMode::GtsamDynamic && mState == OK)
        TryInitializeGtsamBackend();
#endif

	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	double t_track = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t1 - t0).count();
	/*cout << "trracking time: " << t_track << endl;
	f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ",";
	f_track_stats << mvpLocalKeyFrames.size() << ",";
	f_track_stats << mvpLocalMapPoints.size() << ",";
	f_track_stats << setprecision(6) << t_track << endl;*/

#ifdef SAVE_TIMES
																										f_track_times << mCurrentFrame.mTimeFeatureExtraction << ",";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
    f_track_times << mTime_PreIntIMU << ",";
    f_track_times << mTime_PosePred << ",";
    f_track_times << mTime_LocalMapTrack << ",";
    f_track_times << mTime_NewKF_Dec << ",";
    f_track_times << t_track << endl;
#endif

	return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageStereoDvlgyro(const Mat &imRectLeft,
                                         const Mat &imRectRight,
                                         const double &timestamp,
                                         bool bDvl,
                                         string filename,
                                         const std::vector<cv::Point2f>& vExtLeft,
                                         const std::vector<cv::Point2f>& vExtRight,
                                         const std::vector<float>& vExtScore,
                                         const cv::Mat& extDepthScaled)
{
#ifdef AQUA_HAS_GTSAM_DYNAMIC
    auto transaction = AcquireDynamicMapTransaction();
#endif
	mImLeft = imRectLeft.clone();
	cv::Mat imGrayRight = imRectRight.clone();
	mImRight = imRectRight.clone();

	cv::resize(mImLeft, mImLeft, cv::Size(mImLeft.cols * mImageScale, mImLeft.rows * mImageScale));
	cv::resize(imGrayRight, imGrayRight, cv::Size(imGrayRight.cols * mImageScale, imGrayRight.rows * mImageScale));
	cv::resize(mImRight, mImRight, cv::Size(mImRight.cols * mImageScale, mImRight.rows * mImageScale));

	if ((mSensor == System::DVL_STEREO || mSensor == System::IMU_STEREO) && !mpCamera2) {
		// EKF DVL
		// tighly coupled DVL
		mCurrentFrame = Frame(mImLeft,
		                      imGrayRight,
			                      timestamp,
			                      *mpFeatureFrontend,
			                      mK,
		                      mDistCoef,
		                      mbf,
		                      mThFarDepth,
		                      mThCloseDepth,
		                      mStereoMaxVerticalError,
		                      mpCamera,
		                      bDvl,
		                      &mLastFrame,
		                      GetExtrinsicPara());
	}
	else {
// 		ROS_ERROR_STREAM("Wrong Mode!");  // original
        assert(0);
	}


	
	if (!vExtLeft.empty() && !vExtRight.empty()) {
		mCurrentFrame.ApplyExternalStereoMatches(vExtLeft, vExtRight, vExtScore);
	}
	if (!extDepthScaled.empty()) {
		mCurrentFrame.SetExternalDepth(extDepthScaled);
	}
mCurrentFrame.mNameFile = filename;
	mCurrentFrame.mnDataset = mnNumDataset;

	if (mSensor == System::IMU_STEREO)
		Track();
	else
		TrackDVLGyro();

#ifdef AQUA_HAS_GTSAM_DYNAMIC
    if (mBackendMode == BackendMode::GtsamDynamic && mState == OK)
        TryInitializeGtsamBackend();
#endif

	return mCurrentFrame.mTcw.clone();
}

void Tracking::GrabImuData(const IMU::ImuPoint &imuMeasurement)
{
	unique_lock<mutex> lock(mMutexImuQueue);
	mlQueueImuData.push_back(imuMeasurement);
#ifdef AQUA_HAS_GTSAM_DYNAMIC
	ImuSample dynamicSample;
	dynamicSample.timestampSec = imuMeasurement.t;
	dynamicSample.acceleration = {imuMeasurement.a.x, imuMeasurement.a.y,
	                              imuMeasurement.a.z};
	dynamicSample.angularVelocity = {imuMeasurement.w.x, imuMeasurement.w.y,
	                                 imuMeasurement.w.z};
	// Keyframe intervals must be formed from raw IMU arrival, not from the
	// delayed enhanced-stereo frame window consumed by PreintegrateIMU().
	mDynamicKeyframeImu.append({dynamicSample});
	if (mCollectDynamicInitializationImu &&
		(std::isfinite(dynamicSample.timestampSec)) &&
		(mDynamicInitializationImu.empty() ||
		 dynamicSample.timestampSec >
		     mDynamicInitializationImu.back().timestampSec)) {
		mDynamicInitializationImu.push_back(dynamicSample);
	}
#endif
}

#ifdef AQUA_HAS_GTSAM_DYNAMIC
std::shared_ptr<GtsamBackendAdapter> Tracking::AcquireGtsamBackend() const
{
    std::lock_guard<std::mutex> lock(mMutexDynamicState);
    return mpGtsamBackend;
}

bool Tracking::DynamicLastCommitSucceeded() const
{
    std::lock_guard<std::mutex> lock(mMutexDynamicState);
    return mDynamicLastCommitSucceeded;
}

std::uint64_t Tracking::RecordCommittedKeyframeWatermark(
    std::uint64_t keyframeId, double timestampSec, Map& map)
{
    if (!std::isfinite(timestampSec))
        return 0U;
    const std::uint64_t mapVersion =
        static_cast<std::uint64_t>(map.GetMapChangeIndex());
    std::lock_guard<std::mutex> lock(mMutexDynamicState);
    mDynamicCommittedWatermark = {keyframeId, timestampSec, mapVersion};
    mHasDynamicCommittedWatermark = true;
    return ++mDynamicCommitSequence;
}

void Tracking::ResetDynamicBackendState()
{
    std::lock_guard<std::mutex> lock(mMutexDynamicState);
    mpGtsamBackend.reset();
    mDynamicLastCommitSucceeded = false;
    mHasDynamicCommittedWatermark = false;
    mLastDynamicInitializationTopology.reset();
    mLatestDynamicDiagnostics.reset();
    mDynamicRebaseRejectionCount = 0U;
    mDynamicRebaseWarningLatched = false;
}

bool Tracking::PredictStateWithDynamicBackend()
{
    const auto backend = AcquireGtsamBackend();
    if (!backend)
        return false;

    std::vector<ImuSample> imu;
    {
        std::lock_guard<std::mutex> lock(mMutexImuQueue);
        imu = mDynamicKeyframeImu.pendingInterval(mCurrentFrame.mTimeStamp);
    }
    if (imu.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                    "dynamic recovery lacks a bracketed IMU interval at %.9f",
                    mCurrentFrame.mTimeStamp);
        return false;
    }

    const auto prediction = backend->predictCameraPose(
        mCurrentFrame.mTimeStamp, imu);
    if (!prediction.accepted || prediction.cameraFromMap.empty() ||
        !cv::checkRange(prediction.cameraFromMap)) {
        RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                    "dynamic recovery prediction rejected at %.9f: %s",
                    mCurrentFrame.mTimeStamp,
                    prediction.diagnostic.empty()
                        ? "invalid camera pose"
                        : prediction.diagnostic.c_str());
        return false;
    }
    mCurrentFrame.SetPose(prediction.cameraFromMap);
    return true;
}

bool Tracking::TryInitializeGtsamBackend()
{
    if (AcquireGtsamBackend())
        return true;
    if (mCurrentFrame.mTimeStamp - mLastDynamicInitializationAttempt < 0.5)
        return false;
    mLastDynamicInitializationAttempt = mCurrentFrame.mTimeStamp;
    std::vector<ImuSample> initializationImu;
    {
        std::unique_lock<std::mutex> lock(mMutexImuQueue);
        if (mDynamicInitializationImu.size() < 2U ||
            mDynamicInitializationImu.back().timestampSec <=
                mDynamicInitializationImu.front().timestampSec)
            return false;
        initializationImu = mDynamicInitializationImu;
    }
    try {
        if (!mpImuCalib)
            throw std::runtime_error("IMU calibration is unavailable");
        // The dynamic backend preintegrates raw IMU-frame measurements, so its
        // body state is the IMU frame.  T_body_imu remains an output-only
        // transform and must not enter the estimator state convention.
        const cv::Mat bodyFromCamera = mpImuCalib->mT_imu_c;
        auto transaction = AcquireDynamicMapTransaction();
        Map* map = mpAtlas ? mpAtlas->GetCurrentMap() : nullptr;
        if (!map)
            return false;
        const std::uint64_t version = map->GetMapChangeIndex();
        // Tracking has released the map lock. All inserted bootstrap keyframes
        // enter this graph once; the next keyframe starts a new IMU interval.
        auto snapshot = GtsamMapAdapter::snapshot(*map, version);
        if (snapshot.keyframes.empty())
            return false;
        const auto latest = std::max_element(snapshot.keyframes.begin(), snapshot.keyframes.end(),
            [](const auto& left, const auto& right) {
                return left.timestampSec < right.timestampSec;
            });
        if (initializationImu.back().timestampSec < latest->timestampSec ||
            mLastDynamicInitializationTopology == snapshot.topologySignature)
            return false;
        if (!mpLocalMapper || mpLocalMapper->stopRequested())
            return false;
        mpLocalMapper->RequestStop();
        struct ResumeMapping {
            LocalMapping* mapper;
            ~ResumeMapping() { mapper->Release(); }
        } resumeMapping{mpLocalMapper};
        // The mapper drains queued keyframes before reporting stopped. Its
        // culling/fusion must finish before the initialization snapshot is taken.
        while (!mpLocalMapper->isStopped()) {
            if (mpLocalMapper->isFinished())
                return false;
            usleep(1000);
        }
        snapshot = GtsamMapAdapter::snapshot(*map, version);
        mLastDynamicInitializationTopology = snapshot.topologySignature;
        RCLCPP_INFO(
            rclcpp::get_logger("aqua_slam"),
            "dynamic GTSAM initialization inputs: keyframes=%zu stereo_observations=%zu imu_span_s=%.9f",
            snapshot.keyframes.size(), snapshot.observations.size(),
            initializationImu.back().timestampSec -
                initializationImu.front().timestampSec);
        BackendFrameInput noise;
        if (!AquaBackendAdapter::applyDynamicImuCovariance(
                noise, *mpImuCalib, mImuFrequencyHz))
            throw std::runtime_error("invalid configured IMU noise");
        const auto optionalSensors = GtsamOptionalSensorConfig::fromTrackingSensor(
            mSensor == System::DVL_STEREO, mpImuCalib,
            mDvlTrackMode, mDvlUncertaintyProfilePath);
        noise.imu = initializationImu;
        const auto solveStart = std::chrono::steady_clock::now();
        auto initialization = GtsamBackendAdapter::prepareInitialization(
            snapshot, noise, bodyFromCamera, optionalSensors);
        RCLCPP_INFO(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM initialization solve_ms=%.3f accepted=%d",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - solveStart).count(),
                    initialization.backend ? 1 : 0);
        if (!initialization.backend || initialization.mapResult.keyframes.empty())
            throw std::runtime_error(initialization.diagnostic);
        const auto& last = initialization.mapResult.keyframes.back();
        KeyFrame* reference = nullptr;
        for (KeyFrame* keyframe : map->GetAllKeyFrames())
            if (keyframe && keyframe->mnId == last.id)
                reference = keyframe;
        {
            std::lock_guard<std::mutex> lock(mMutexImuQueue);
            if (!mDynamicKeyframeImu.canAcceptKeyframe(last.timestampSec))
                return false;
        }
        if (!reference || !CommitDynamicMapResult(
                initialization.mapResult, *map, version, reference)) {
            const auto current = GtsamMapAdapter::snapshot(*map, map->GetMapChangeIndex());
            RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                        "dynamic GTSAM initialization map commit rejected reference=%d "
                        "source_version=%lu current_version=%lu source_topology=%lu current_topology=%lu",
                        reference ? 1 : 0, static_cast<unsigned long>(version),
                        static_cast<unsigned long>(current.version),
                        static_cast<unsigned long>(snapshot.topologySignature),
                        static_cast<unsigned long>(current.topologySignature));
            return false;
        }
        map->IncreaseChangeIndex();
        {
            std::lock_guard<std::mutex> lock(mMutexDynamicState);
            mpGtsamBackend = initialization.backend;
        }
        RecordCommittedKeyframeWatermark(last.id, last.timestampSec, *map);
        {
            std::lock_guard<std::mutex> lock(mMutexImuQueue);
            mDynamicKeyframeImu.acceptKeyframe(last.timestampSec);
            mDynamicKeyframeDvl.commit(last.timestampSec);
            mCollectDynamicInitializationImu = false;
            mDynamicInitializationImu.clear();
        }
        RCLCPP_INFO(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM initialized from %.3f seconds of IMU; retained_keyframes=%zu initialization_dvl=unused",
                    initializationImu.back().timestampSec -
                        initializationImu.front().timestampSec,
                    initialization.mapResult.keyframes.size());
        return true;
    } catch (const std::exception& exception) {
        RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM waiting for physical initialization: %s",
                    exception.what());
        return false;
    }
}

bool Tracking::ResetGtsamBackendAtKeyFrame(const cv::Mat& cameraFromMap,
                                           double keyframeTimestamp)
{
    const auto backend = AcquireGtsamBackend();
    if (!backend || !mpImuCalib ||
        cameraFromMap.rows != 4 || cameraFromMap.cols != 4 ||
        !cv::checkRange(cameraFromMap) || !std::isfinite(keyframeTimestamp))
        return false;
    try {
        const cv::Mat bodyFromCamera = mpImuCalib->mT_imu_c;
        const cv::Mat initialMapFromBody =
            cameraFromMap.inv() * bodyFromCamera.inv();
        if (!cv::checkRange(initialMapFromBody))
            return false;
        backend->reset(initialMapFromBody);

        // Preserve already-acquired samples while moving the logical IMU
        // boundary to the committed keyframe. No source timestamp changes.
        {
            std::unique_lock<std::mutex> lock(mMutexImuQueue);
            const std::vector<ImuSample> buffered =
                mDynamicKeyframeImu.samples();
            mDynamicKeyframeImu.clear();
            mDynamicKeyframeImu.append(buffered);
            if (!mDynamicKeyframeImu.acceptKeyframe(keyframeTimestamp))
                return false;
        }
        std::lock_guard<std::mutex> lock(mMutexDynamicState);
        mDynamicLastCommitSucceeded = false;
        return true;
    } catch (const std::exception& exception) {
        RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM map resynchronization failed: %s",
                    exception.what());
        return false;
    }
}

bool Tracking::RebaseGtsamBackend(const BackendMapSnapshot& snapshot)
{
    const auto backend = AcquireGtsamBackend();
    if (!backend)
        return false;
    const bool rebased = backend->rebase(snapshot);
    RCLCPP_INFO(rclcpp::get_logger("aqua_slam"),
                "dynamic GTSAM full-map rebase %s keyframes=%zu landmarks=%zu",
                rebased ? "succeeded" : "failed", snapshot.keyframes.size(),
                snapshot.landmarks.size());
	return rebased;
}

std::unique_lock<std::recursive_timed_mutex> Tracking::AcquireDynamicMapTransaction()
{
    return std::unique_lock<std::recursive_timed_mutex>(mMutexDynamicMapTransaction);
}

std::unique_lock<std::recursive_timed_mutex> Tracking::TryAcquireDynamicMapTransaction()
{
    return std::unique_lock<std::recursive_timed_mutex>(
        mMutexDynamicMapTransaction, std::try_to_lock);
}

bool Tracking::CaptureCommittedKeyframeWatermark(
    CommittedKeyframeWatermark* watermark) const
{
    if (!watermark)
        return false;
    std::lock_guard<std::mutex> lock(mMutexDynamicState);
    if (!mHasDynamicCommittedWatermark)
        return false;
    *watermark = mDynamicCommittedWatermark;
    return true;
}

bool Tracking::AdvanceCommittedWatermarkMapVersion(
    std::uint64_t expectedVersion, std::uint64_t newVersion)
{
    if (newVersion <= expectedVersion)
        return false;
    std::lock_guard<std::mutex> lock(mMutexDynamicState);
    if (!mHasDynamicCommittedWatermark ||
        mDynamicCommittedWatermark.mapVersion != expectedVersion)
        return false;
    mDynamicCommittedWatermark.mapVersion = newVersion;
    return true;
}

void Tracking::RecordDynamicDiagnostics(
    const BackendMapDiagnostics& diagnostics)
{
    std::lock_guard<std::mutex> lock(mMutexDynamicState);
    mLatestDynamicDiagnostics = diagnostics;
}

std::optional<BackendMapDiagnostics> Tracking::LatestDynamicDiagnostics() const
{
    std::lock_guard<std::mutex> lock(mMutexDynamicState);
    return mLatestDynamicDiagnostics;
}

void Tracking::RecordGtsamRebaseProjectionRejected(const std::string& reason)
{
    const auto now = std::chrono::steady_clock::now();
    bool warn = false;
    {
        std::lock_guard<std::mutex> lock(mMutexDynamicState);
        if (mDynamicRebaseRejectionCount++ == 0U)
            mDynamicRebaseStaleSince = now;
        if (!mDynamicRebaseWarningLatched &&
            now - mDynamicRebaseStaleSince >= std::chrono::seconds(1)) {
            mDynamicRebaseWarningLatched = true;
            warn = true;
        }
    }
    if (warn)
        RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"),
                     "dynamic GTSAM rebase stale for at least 1 s: %s",
                     reason.c_str());
    else
        RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM rebase projection rejected: %s",
                    reason.c_str());
}

void Tracking::RecordGtsamRebaseSucceeded()
{
    std::lock_guard<std::mutex> lock(mMutexDynamicState);
    mDynamicRebaseRejectionCount = 0U;
    mDynamicRebaseWarningLatched = false;
}

void Tracking::SynchronizeAfterMapCorrection(
    const cv::Mat& oldCameraFromMap, const cv::Mat& newCameraFromMap)
{
    auto transaction = AcquireDynamicMapTransaction();
    if (oldCameraFromMap.rows != 4 || oldCameraFromMap.cols != 4 ||
        newCameraFromMap.rows != 4 || newCameraFromMap.cols != 4 ||
        !cv::checkRange(oldCameraFromMap) || !cv::checkRange(newCameraFromMap))
        return;
    // MapPoint write-back defines p_new = S * p_old.  Camera-from-map poses
    // therefore transform as Tcw_new = Tcw_old * S^-1.
    const cv::Mat mapCorrection =
        oldCameraFromMap.inv() * newCameraFromMap;
    if (!cv::checkRange(mapCorrection))
        return;
    if (!mLastFrame.mTcw.empty())
        mLastFrame.SetPose(mLastFrame.mTcw * mapCorrection);
    if (!mCurrentFrame.mTcw.empty())
        mCurrentFrame.SetPose(mCurrentFrame.mTcw * mapCorrection);
    if (!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty())
        mVelocity = mCurrentFrame.mTcw * mLastFrame.mTcw.inv();
}

bool Tracking::CommitDynamicMapResult(const BackendMapResult& result, Map& map,
                                      std::uint64_t expectedVersion,
                                      KeyFrame* correctionReference)
{
    auto transaction = AcquireDynamicMapTransaction();
    if (mpAtlas && mpAtlas->GetCurrentMap() != &map)
        return false;
    if (!correctionReference)
        return false;
    const cv::Mat oldCameraFromMap = correctionReference->GetPose();
    if (!GtsamMapAdapter::commit(result, map, expectedVersion))
        return false;
    SynchronizeAfterMapCorrection(
        oldCameraFromMap, correctionReference->GetPose());
    return true;
}

std::shared_ptr<GtsamPreparedMapOptimization> Tracking::PrepareDynamicMapOptimization(
    const BackendMapSnapshot& snapshot,
    const std::vector<BackendLandmarkReplacement>& replacements)
{
    const auto backend = AcquireGtsamBackend();
    return backend ? backend->prepareMapOptimization(snapshot, replacements) : nullptr;
}

bool Tracking::CommitDynamicMapOptimization(
    GtsamPreparedMapOptimization& candidate, Map& map,
    std::uint64_t expectedVersion, KeyFrame* correctionReference)
{
    auto transaction = AcquireDynamicMapTransaction();
    const auto backend = AcquireGtsamBackend();
    CommittedKeyframeWatermark watermark;
    if (!backend || !CaptureCommittedKeyframeWatermark(&watermark) ||
        watermark.mapVersion != expectedVersion ||
        !mpAtlas || mpAtlas->GetCurrentMap() != &map || !correctionReference ||
        correctionReference->GetMap() != &map ||
        std::none_of(candidate.result().keyframes.begin(),
                     candidate.result().keyframes.end(),
                     [&](const BackendKeyframeState& state) {
                         return state.id == correctionReference->mnId;
                     }))
        return false;
    if (!backend->commitMapOptimization(candidate, [&](const BackendMapResult& result) {
            return CommitDynamicMapResult(result, map, expectedVersion,
                                          correctionReference);
        }))
        return false;
    map.IncreaseChangeIndex();
    AdvanceCommittedWatermarkMapVersion(
        expectedVersion, static_cast<std::uint64_t>(map.GetMapChangeIndex()));
    return true;
}
#endif

int Tracking::OptimizeCurrentFrameWithBackend(bool commitIncremental,
                                              std::uint64_t committingKeyframeId)
{
#ifdef AQUA_HAS_GTSAM_DYNAMIC
	if (commitIncremental) {
		std::lock_guard<std::mutex> lock(mMutexDynamicState);
		mDynamicLastCommitSucceeded = false;
	}
    RuntimeProfile profile;
    if (mSensor == System::DVL_STEREO) {
        profile.sensorMode = "stereo_inertial_dvl";
        profile.dvlEnabled = true;
    } else {
        profile.sensorMode = "stereo_inertial";
        profile.dvlEnabled = false;
    }
	    BackendFrameInput input = AquaBackendAdapter::fromFrame(mCurrentFrame, profile);
		    if (profile.dvlEnabled && commitIncremental) {
                std::lock_guard<std::mutex> lock(mMutexImuQueue);
                input.dvl = mDynamicKeyframeDvl.pendingInterval(input.timestampSec);
		    }
	    const int persistentStereoSupport =
        AquaBackendAdapter::persistentStereoSupport(input) +
        static_cast<int>(input.monocularLandmarks.size());
    if (!AcquireGtsamBackend()) {
        if (!mpImuCalib)
            return 0;
        const auto bootstrap = GtsamBackendAdapter::refineVisualPose(
            input, mK, mbf, mpImuCalib->mT_imu_c, mCurrentFrame.mTcw);
        if (!bootstrap.accepted || persistentStereoSupport == 0)
            return 0;
        mCurrentFrame.SetPose((bootstrap.bodyPose * mpImuCalib->mT_imu_c).inv());
        input.frontEndCameraFromMap = mCurrentFrame.mTcw.clone();
        return persistentStereoSupport;
    }
    const auto backend = AcquireGtsamBackend();
    if (!backend)
        return 0;
    input.imu.reserve(mvImuFromLastFrame.size());
    for (const auto& sample : mvImuFromLastFrame) {
        ImuSample converted;
        converted.timestampSec = sample.t;
        converted.acceleration = {sample.a.x, sample.a.y, sample.a.z};
        converted.angularVelocity = {sample.w.x, sample.w.y, sample.w.z};
        input.imu.push_back(converted);
    }
#ifdef AQUA_HAS_GTSAM_DYNAMIC
    if (commitIncremental) {
        bool missingPhysicalBrackets = false;
        {
            std::unique_lock<std::mutex> lock(mMutexImuQueue);
            input.imu = mDynamicKeyframeImu.pendingInterval(input.timestampSec);
            const bool rejectedInterval =
                mDynamicKeyframeImu.hasAcceptedKeyframe() && input.imu.empty();
            if (rejectedInterval) {
                missingPhysicalBrackets = true;
                const auto buffered = mDynamicKeyframeImu.samples();
                double maximumGapSec = 0.0;
                double gapLeftSec = std::numeric_limits<double>::quiet_NaN();
                double gapRightSec = std::numeric_limits<double>::quiet_NaN();
                for (std::size_t index = 1U; index < buffered.size(); ++index) {
                    const double gap = buffered[index].timestampSec -
                                       buffered[index - 1U].timestampSec;
                    if (gap > maximumGapSec) {
                        maximumGapSec = gap;
                        gapLeftSec = buffered[index - 1U].timestampSec;
                        gapRightSec = buffered[index].timestampSec;
                    }
                }
                RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                            "dynamic IMU boundary evidence current=%.9f previous=%.9f "
                            "buffer_size=%zu first=%.9f last=%.9f max_gap=%.9f "
                            "gap_left=%.9f gap_right=%.9f",
                            input.timestampSec,
                            mDynamicKeyframeImu.lastAcceptedKeyframeTime(),
                            mDynamicKeyframeImu.size(),
                            mDynamicKeyframeImu.firstTimestamp(),
                            mDynamicKeyframeImu.lastTimestamp(), maximumGapSec,
                            gapLeftSec, gapRightSec);
            }
        }
        if (missingPhysicalBrackets) {
            RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                        "dynamic GTSAM deferring keyframe %.9f until its IMU interval has physical brackets",
                        input.timestampSec);
            return 0;
        }
    }
#endif
    // PreintegrateIMU intentionally retains the first sample after a camera
    // boundary so the next frame can use it as its left endpoint.  Under a
    // delayed enhanced-stereo worker that retained endpoint can also be
    // copied into the current batch.  Remove only repeated/non-monotonic
    // boundary samples before deriving dynamic covariance; source timestamps
    // remain untouched and no synthetic IMU is introduced.
    if (input.imu.size() > 1U) {
        std::vector<ImuSample> monotonic;
        monotonic.reserve(input.imu.size());
        monotonic.push_back(input.imu.front());
        for (std::size_t index = 1U; index < input.imu.size(); ++index) {
            const auto& sample = input.imu[index];
            if (std::isfinite(sample.timestampSec) &&
                sample.timestampSec > monotonic.back().timestampSec)
                monotonic.push_back(sample);
        }
        input.imu.swap(monotonic);
    }
    // Noise density belongs to the calibrated sensor, not the measured motion.
    if (!mpImuCalib ||
        !AquaBackendAdapter::applyDynamicImuCovariance(
            input, *mpImuCalib, mImuFrequencyHz)) {
        RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM invalid calibrated IMU noise at %.9f",
                    input.timestampSec);
        return 0;
    }
    // Validate the logical keyframe watermark before entering GTSAM. This is
    // a read-only check: the IMU interval remains available if optimization
    // rejects the transaction.
    if (commitIncremental) {
        std::lock_guard<std::mutex> lock(mMutexImuQueue);
        if (!mDynamicKeyframeImu.canAcceptKeyframe(input.timestampSec)) {
            RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"),
                         "dynamic GTSAM refusing non-monotonic keyframe timestamp %.9f",
                         input.timestampSec);
            return 0;
        }
    }
    const auto result = commitIncremental
        ? backend->optimize(input)
        : backend->refinePose(input, mCurrentFrame.mTcw);
    if (commitIncremental) {
        const std::string sensorDiagnostics =
            AquaBackendAdapter::formatDynamicSensorDiagnostics(
                input.timestampSec, result);
        RCLCPP_INFO(rclcpp::get_logger("aqua_slam"), "%s",
                    sensorDiagnostics.c_str());
    }
    if (commitIncremental && result.accepted && !result.incrementalCommitted) {
        RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM did not commit keyframe %.9f; preserving IMU interval: %s",
                    input.timestampSec,
                    result.diagnostic.empty() ? "transaction degraded" : result.diagnostic.c_str());
        return persistentStereoSupport;
    }
    bool acceptedKeyframeTimestamp = true;
    std::uint64_t commitSequence = 0U;
    if (commitIncremental && result.incrementalCommitted) {
        std::lock_guard<std::mutex> lock(mMutexImuQueue);
        acceptedKeyframeTimestamp =
            mDynamicKeyframeImu.acceptKeyframe(input.timestampSec);
        mDynamicKeyframeDvl.commit(input.timestampSec);
        if (acceptedKeyframeTimestamp) {
            Map* map = mpAtlas == nullptr ? nullptr : mpAtlas->GetCurrentMap();
            if (map == nullptr)
                acceptedKeyframeTimestamp = false;
            else
                commitSequence = RecordCommittedKeyframeWatermark(
                    committingKeyframeId, input.timestampSec, *map);
        }
    }
    if (commitIncremental && result.incrementalCommitted &&
        !acceptedKeyframeTimestamp) {
        RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"),
                     "dynamic GTSAM accepted a non-monotonic keyframe timestamp %.9f",
                     input.timestampSec);
        return 0;
    }
    const bool committedKeyframe =
        commitIncremental && result.incrementalCommitted;
    if (committedKeyframe) {
        RCLCPP_INFO(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM committed keyframe t=%.9f backend_version=%lu commit_sequence=%lu imu_samples=%zu",
                    input.timestampSec,
                    static_cast<unsigned long>(result.mapVersion),
                    static_cast<unsigned long>(commitSequence),
                    input.imu.size());
    }
    // A duplicate-landmark transaction can be rejected before the first
    // committed dynamic estimate exists (for example when initialization was
    // refined but not committed). Keep the finite front-end pose for this
    // frame so AQUA does not interpret a transactional backend diagnostic as
    // camera loss and reset the active map.
    if (AquaBackendAdapter::canPreserveFrontEndPose(result, mCurrentFrame)) {
        RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM degraded frame %.9f; preserving finite front-end pose: %s",
                    input.timestampSec, result.diagnostic.c_str());
        return AquaBackendAdapter::trackingSupportAfterPoseCorrection(
            committedKeyframe, false, result, persistentStereoSupport);
    }
    const cv::Mat cameraPose = backend->cameraPose(result);
    if (!result.accepted || cameraPose.empty() || !cv::checkRange(cameraPose)) {
        RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                    "dynamic GTSAM rejected frame %.9f: %s",
                    input.timestampSec,
                    result.diagnostic.empty() ? "no diagnostic" : result.diagnostic.c_str());
        return 0;
    }
    if (committedKeyframe) {
        std::lock_guard<std::mutex> lock(mMutexDynamicState);
        mDynamicLastCommitSucceeded = true;
    }
    mCurrentFrame.SetPose(cameraPose);
    return AquaBackendAdapter::trackingSupportAfterPoseCorrection(
        committedKeyframe, true, result, persistentStereoSupport);
#else
    return 0;
#endif
}
void Tracking::GrabDVLGyroData(const GyroDvlPoint &DVLGyrosMeasurement)
{
	unique_lock<mutex> lock(mMutexImuQueue);
	mlQueueDVLGyroData.push_back(DVLGyrosMeasurement);
#ifdef AQUA_HAS_GTSAM_DYNAMIC
    BackendFrameInput input;
    if (AquaBackendAdapter::applyDvlMeasurement(input, DVLGyrosMeasurement))
        mDynamicKeyframeDvl.append(input.dvl.front());
#endif
}

void Tracking::PreintegrateIMU()
{
	//cout << "start preintegration" << endl;

	if (!mCurrentFrame.mpPrevFrame) {
		Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
		mCurrentFrame.setIntegrated();
		return;
	}

	// cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;

	mvImuFromLastFrame.clear();
	mvImuFromLastFrame.reserve(mlQueueImuData.size());
	if (mlQueueImuData.size() == 0) {
		Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
		mCurrentFrame.setIntegrated();
		return;
	}

	while (true) {
		bool bSleep = false;
		{
			unique_lock<mutex> lock(mMutexImuQueue);
			if (!mlQueueImuData.empty()) {
				IMU::ImuPoint *m = &mlQueueImuData.front();
				cout.precision(17);
				// IMU measurement is before Frame_i
				if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - 0.001l) {
					mlQueueImuData.pop_front();
				}
					// IMU measurement is between Frame_i and Frame_j
				else if (m->t < mCurrentFrame.mTimeStamp - 0.001l) {
					mvImuFromLastFrame.push_back(*m);
					mlQueueImuData.pop_front();
				}
					// IMU measurement is after Frame_j
				else {
					mvImuFromLastFrame.push_back(*m);
					break;
				}
			}
			else {
				break;
				bSleep = true;
			}
		}
		if (bSleep) {
			usleep(500);
		}
	}


	const int n = mvImuFromLastFrame.size() - 1;
	IMU::Preintegrated
		*pImuPreintegratedFromLastFrame =
		new IMU::Preintegrated(mLastFrame.mImuBias, mCurrentFrame.GetExtrinsicParamters());

	for (int i = 0; i < n; i++) {
		float tstep;
		cv::Point3f acc, angVel;
		// first IMU meas after Frame_i
		// and this IMU meas is not the last two
		if ((i == 0) && (i < (n - 1))) {
			// delta_t between IMU meas
			float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
			// time gap between first IMU meas and Frame_i
			float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
			// estimate the acc from Frame_i to second IMU meas
			acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
				(mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) * 0.5f;
			// estimate the angVel from Frame_i to second IMU meas
			angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
				(mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) * 0.5f;
			tstep = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
		}
			// not first and not last, the middle IMU meas
		else if (i < (n - 1)) {
			acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;
			angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f;
			tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
		}
			// last two IMU meas before Frame_j
			// and this IMU meas is not the first IMU meas after Frame_i
		else if ((i > 0) && (i == (n - 1))) {
			float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
			float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
			// estimate the acc from the last second IMU meas to Frame_j
			acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
				(mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) * 0.5f;
			// estimate the angVel from the last second IMU meas to Frame_j
			angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
				(mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) * 0.5f;
			tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
		}
			// last two IMU meas before Frame_j
			// and this IMU meas is also the first IMU meas after Frame_i
		else if ((i == 0) && (i == (n - 1))) {
			acc = mvImuFromLastFrame[i].a;
			angVel = mvImuFromLastFrame[i].w;
			tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
		}

		if (!mpImuPreintegratedFromLastKF) {
			cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
		}
		// keyframe integration
		mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
		// frame integration
		pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
	}

	mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
	mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
	mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

	mCurrentFrame.setIntegrated();

	Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}

void Tracking::PreintegrateDvlGro()
{
	//cout << "start preintegration" << endl;

	if (mCurrentFrame.mpPrevFrame->mTimeStamp == 0) {
// 		ROS_INFO_STREAM("non prev frame ");  // original
//		Verbose::PrintMess(, Verbose::VERBOSITY_NORMAL);
		mCurrentFrame.setIntegrated();
		return;
	}

	// cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;

	mvImuFromLastFrame.clear();
	mvImuFromLastFrame.reserve(mlQueueImuData.size());
	if (mlQueueImuData.size() == 0) {
		Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
		// ROS_WARN_STREAM("Not IMU data in mlQueueImuData!!");  // original
		RCLCPP_WARN_STREAM(rclcpp::get_logger("aqua_slam"), "Not IMU data in mlQueueImuData!!");
		mCurrentFrame.setIntegrated();
		return;
	}

	// put IMU Meas from mlQueueImuData to mvImuFromLastFrame
	while (true) {
		bool bSleep = false;
		{
			unique_lock<mutex> lock(mMutexImuQueue);
			if (!mlQueueImuData.empty()) {
				IMU::ImuPoint *m = &mlQueueImuData.front();
//				cout<<"IMU Meas acc: "<<m->a<<" timestamp: "<< m->t <<endl;
				cout.precision(17);
				// IMU measurement is before Frame_i
				if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - 0.001l) {
					mlQueueImuData.pop_front();
				}
					// IMU measurement is between Frame_i and Frame_j
				else if (m->t <= mCurrentFrame.mTimeStamp + 0.001l) {
//					cout<<"push_back IMU Meas acc: "<<m->a<<" timestamp: "<< m->t <<endl;
					mvImuFromLastFrame.push_back(*m);
					mlQueueImuData.pop_front();
				}
					// IMU measurement is after Frame_j
				else {
//					cout << "Error! IMU measurement is after Frame_j!!!" << endl;
					mvImuFromLastFrame.push_back(*m);
					break;
				}
			}
			else {
				break;
				bSleep = true;
			}
		}
		if (bSleep) {
			usleep(500);
		}
	}


//	const int n = mvImuFromLastFrame.size()-1;
	const int n = mvImuFromLastFrame.size();
	// first mImuBias is set to 0
	// first mImuCalib comes from settings file
//	IMU::Preintegrated
//		*pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias, mCurrentFrame.mImuCalib);
	//todo_tightly
	//	pass velocity from last frame to DVLGroPreIntegration of current frame
	// mVelocity: T_cj_ci
	DVLGroPreIntegration *pDvlGroPreIntegratedFromLastFrame = NULL;
	cv::Mat v_di_cv = mLastFrame.GetDvlVelocity();
	if (!v_di_cv.empty()) {
		v_di_cv.convertTo(v_di_cv, CV_32F);

//		cv::Point3f
//		v_di(v_dkfi_cv.at<float>(0), v_dkfi_cv.at<float>(1), v_dkfi_cv.at<float>(2));
		cv::Point3f v_di(v_di_cv.at<float>(0), v_di_cv.at<float>(1), v_di_cv.at<float>(2));

		pDvlGroPreIntegratedFromLastFrame =
			new DVLGroPreIntegration(mLastFrame.mImuBias,
			                         mCurrentFrame.GetExtrinsicParamters(),
			                         v_di,
			                         mCurrentFrame.mbDVL);

	}
	else {
		pDvlGroPreIntegratedFromLastFrame =
			new DVLGroPreIntegration(mLastFrame.mImuBias, mCurrentFrame.GetExtrinsicParamters(), mCurrentFrame.mbDVL);
	}


	for (int i = 0; i < n; i++) {
		float tstep;
		cv::Point3f acc, angVel;
		// first IMU meas after Frame_i
		// and this IMU meas is not the last two
		if ((i == 0) && (i < (n - 1))) {
			// delta_t between IMU meas
			float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
			if (tab == 0) {
				tab = 0.001;
			}
			// time gap between first IMU meas and Frame_i
			float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
			// when acc != 0
			// velocity saved in acc
			acc = mvImuFromLastFrame[i].a;
			// estimate the angVel from Frame_i to second IMU meas
			angVel = mvImuFromLastFrame[i].w;
			tstep = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
		}
			// not first and not last, the middle IMU meas
		else if (i < (n - 1)) {
			acc = mvImuFromLastFrame[i].a;
			angVel = mvImuFromLastFrame[i].w;
			tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
		}
			// last two IMU meas before Frame_j
			// and this IMU meas is not the first IMU meas after Frame_i
		else if ((i > 0) && (i == (n - 1))) {
			float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
			if (tab == 0) {
				tab = 0.001;
			}
			float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
			// estimate the acc from the last second IMU meas to Frame_j
			acc = mvImuFromLastFrame[i].a;
			// estimate the angVel from the last second IMU meas to Frame_j
			angVel = mvImuFromLastFrame[i].w;
			tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
		}
			// last two IMU meas before Frame_j
			// and this IMU meas is also the first IMU meas after Frame_i
		else if ((i == 0) && (i == (n - 1))) {
			acc = mvImuFromLastFrame[i].a;
			angVel = mvImuFromLastFrame[i].w;
			tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
		}

		if (!mpImuPreintegratedFromLastKF) {
			cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
		}
		// keyframe integration

		mpDvlPreintegratedFromLastKF->IntegrateGroMeasurement(angVel, tstep);
		{
			DVLGroPreIntegration *pDvlPreintegratedFromLastKFBeforeLost = getLossIntegrationRef();
			std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);
			if (mDoLossIntegration) {
				pDvlPreintegratedFromLastKFBeforeLost->IntegrateGroMeasurement(angVel, tstep);
			}
//			else{
// //				ROS_INFO_STREAM("pDvlPreintegratedFromLastKFBeforeLost do not exits");  // original
//			}
		}

		// frame integration

//		cout<<"angVel: "<<angVel<<endl;
		pDvlGroPreIntegratedFromLastFrame->IntegrateGroMeasurement(angVel, tstep);
		if (acc.x != 0 && acc.y != 0 && acc.z != 0) {
//			cout<<"velocity measurement: "<<acc<<endl;
			mpDvlPreintegratedFromLastKF->IntegrateDVLMeasurement(acc, tstep);
// //			ROS_INFO_STREAM("add kf velocity measurement:"<<acc);  // original
			pDvlGroPreIntegratedFromLastFrame->IntegrateDVLMeasurement(acc, tstep);
			{
				DVLGroPreIntegration *pDvlPreintegratedFromLastKFBeforeLost = getLossIntegrationRef();
				std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);
				if (mDoLossIntegration) {
					pDvlPreintegratedFromLastKFBeforeLost->IntegrateDVLMeasurement(acc, tstep);
// //					ROS_INFO_STREAM("add loss ref velocity measurement:"<<acc);  // original
				}
//				else{
// //					ROS_INFO_STREAM("pDvlPreintegratedFromLastKFBeforeLost do not exits");  // original
//				}
			}
		}
	}

//	mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
	mCurrentFrame.mpDvlPreintegrationFrame = pDvlGroPreIntegratedFromLastFrame;
//	mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
	mCurrentFrame.mpDvlPreintegrationKeyFrame = mpDvlPreintegratedFromLastKF;
	mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

	mCurrentFrame.setIntegrated();

}

void Tracking::PreintegrateDvlGro2()
{
	//cout << "start preintegration" << endl;

	if (mCurrentFrame.mpPrevFrame->mTimeStamp == 0) {
// 		ROS_INFO_STREAM("non prev frame ");  // original
//		Verbose::PrintMess(, Verbose::VERBOSITY_NORMAL);
		mCurrentFrame.setIntegrated();
		return;
	}

	// cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;

	mvGyroDVLFromLastFrame.clear();
	mvGyroDVLFromLastFrame.reserve(mlQueueDVLGyroData.size());
	if (mlQueueDVLGyroData.size() == 0) {
		Verbose::PrintMess("Not IMU data in mlQueueDVLGyroData!!", Verbose::VERBOSITY_NORMAL);
		// ROS_WARN_STREAM("Not IMU data in mlQueueDVLGyroData!!");  // original
		RCLCPP_WARN_STREAM(rclcpp::get_logger("aqua_slam"), "Not IMU data in mlQueueDVLGyroData!!");
		mCurrentFrame.setIntegrated();
		return;
	}

	// put Gyros DVL Meas from mlQueueDVLGyroData to mvGyroDVLFromLastFrame
	while (true) {
		bool bSleep = false;
		{
			unique_lock<mutex> lock(mMutexImuQueue);
			if (!mlQueueDVLGyroData.empty()) {
				IMU::GyroDvlPoint *m = &mlQueueDVLGyroData.front();
//				cout<<"IMU Meas acc: "<<m->a<<" timestamp: "<< m->t <<endl;
				cout.precision(17);
				// IMU measurement is before Frame_i
				if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - 0.00001l) {
					mlQueueDVLGyroData.pop_front();
				}
					// IMU measurement is between Frame_i and Frame_j
				else if (m->t <= mCurrentFrame.mTimeStamp + 0.00001l) {
//					cout<<"push_back IMU Meas acc: "<<m->a<<" timestamp: "<< m->t <<endl;
					mvGyroDVLFromLastFrame.push_back(*m);
					mlQueueDVLGyroData.pop_front();
				}
					// IMU measurement is after Frame_j
				else {
//					cout << "Error! IMU measurement is after Frame_j!!!" << endl;
					mvGyroDVLFromLastFrame.push_back(*m);
					break;
				}
			}
			else {
				break;
				bSleep = true;
			}
		}
		if (bSleep) {
			usleep(500);
		}
	}


//	const int n = mvGyroDVLFromLastFrame.size()-1;
	const int n = mvGyroDVLFromLastFrame.size();
	// first mImuBias is set to 0
	// first mImuCalib comes from settings file
//	IMU::Preintegrated
//		*pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias, mCurrentFrame.mImuCalib);
	//todo_tightly
	//	pass velocity from last frame to DVLGroPreIntegration of current frame
	// mVelocity: T_cj_ci
	DVLGroPreIntegration *pDvlGroPreIntegratedFromLastFrame = NULL;
	cv::Mat v_di_cv = mLastFrame.GetDvlVelocity();
	if (!v_di_cv.empty()) {
		v_di_cv.convertTo(v_di_cv, CV_32F);

//		cv::Point3f
//		v_di(v_dkfi_cv.at<float>(0), v_dkfi_cv.at<float>(1), v_dkfi_cv.at<float>(2));
		cv::Point3f v_di(v_di_cv.at<float>(0), v_di_cv.at<float>(1), v_di_cv.at<float>(2));

		pDvlGroPreIntegratedFromLastFrame =
			new DVLGroPreIntegration(mLastFrame.mImuBias,
			                         mCurrentFrame.GetExtrinsicParamters(),
			                         v_di,
			                         mCurrentFrame.mbDVL);

	}
	else {
		pDvlGroPreIntegratedFromLastFrame =
			new DVLGroPreIntegration(mLastFrame.mImuBias, mCurrentFrame.GetExtrinsicParamters(), mCurrentFrame.mbDVL);
	}

	Eigen::Vector4d alpha, beta;
	GetBeamOrientation(alpha, beta);
	pDvlGroPreIntegratedFromLastFrame->SetBeamOrientation(alpha, beta);
	mpDvlPreintegratedFromLastKF->SetBeamOrientation(alpha, beta);

	for (int i = 0; i < n; i++) {
		float tstep;
		cv::Point3d v_d, angVel, acc;
		Eigen::Vector4d v_beam;
		// first IMU meas after Frame_i
		// and this IMU meas is not the last two
		if ((i == 0) && (i < (n - 1))) {
			// delta_t between IMU meas
			float tab = mvGyroDVLFromLastFrame[i + 1].t - mvGyroDVLFromLastFrame[i].t;
			if (tab == 0) {
				tab = 0.001;
			}
			// time gap between first IMU meas and Frame_i
			float tini = mvGyroDVLFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
			// when acc != 0
			// velocity saved in acc
			v_d = mvGyroDVLFromLastFrame[i].v;
			// estimate the angVel from Frame_i to second IMU meas
			angVel = mvGyroDVLFromLastFrame[i].angular_v;
			acc = mvGyroDVLFromLastFrame[i].acc;
			v_beam = mvGyroDVLFromLastFrame[i].vb;
			tstep = mvGyroDVLFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
		}
			// not first and not last, the middle IMU meas
		else if (i < (n - 1)) {
			v_d = mvGyroDVLFromLastFrame[i].v;
			// estimate the angVel from Frame_i to second IMU meas
			acc = mvGyroDVLFromLastFrame[i].acc;
			angVel = mvGyroDVLFromLastFrame[i].angular_v;
			v_beam = mvGyroDVLFromLastFrame[i].vb;
			tstep = mvGyroDVLFromLastFrame[i + 1].t - mvGyroDVLFromLastFrame[i].t;
		}
			// last two IMU meas before Frame_j
			// and this IMU meas is not the first IMU meas after Frame_i
		else if ((i > 0) && (i == (n - 1))) {
			float tab = mvGyroDVLFromLastFrame[i + 1].t - mvGyroDVLFromLastFrame[i].t;
			if (tab == 0) {
				tab = 0.001;
			}
			float tend = mvGyroDVLFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
			// estimate the acc from the last second IMU meas to Frame_j
			v_d = mvGyroDVLFromLastFrame[i].v;
			// estimate the angVel from Frame_i to second IMU meas
			acc = mvGyroDVLFromLastFrame[i].acc;
			angVel = mvGyroDVLFromLastFrame[i].angular_v;
			v_beam = mvGyroDVLFromLastFrame[i].vb;
			tstep = mCurrentFrame.mTimeStamp - mvGyroDVLFromLastFrame[i].t;
		}
			// last two IMU meas before Frame_j
			// and this IMU meas is also the first IMU meas after Frame_i
		else if ((i == 0) && (i == (n - 1))) {
			v_d = mvGyroDVLFromLastFrame[i].v;
			// estimate the angVel from Frame_i to second IMU meas
			angVel = mvGyroDVLFromLastFrame[i].angular_v;
			acc = mvGyroDVLFromLastFrame[i].acc;
			v_beam = mvGyroDVLFromLastFrame[i].vb;
			tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
		}

		if (!mpImuPreintegratedFromLastKF) {
			cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
		}
		// keyframe integration

		if ((angVel.x != 0) && (angVel.y != 0) && (angVel.z != 0)) {
			mpDvlPreintegratedFromLastKF->IntegrateGroMeasurement(angVel, tstep);
			{
				DVLGroPreIntegration *pDvlPreintegratedFromLastKFBeforeLost = getLossIntegrationRef();
				std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);
				if (mDoLossIntegration) {
					pDvlPreintegratedFromLastKFBeforeLost->SetBeamOrientation(alpha, beta);
					pDvlPreintegratedFromLastKFBeforeLost->IntegrateGroMeasurement(angVel, tstep);
				}
//			else{
// //				ROS_INFO_STREAM("pDvlPreintegratedFromLastKFBeforeLost do not exits");  // original
//			}
			}
			pDvlGroPreIntegratedFromLastFrame->IntegrateGroMeasurement(angVel, tstep);
		}
		if (v_d.x != 0 && v_d.y != 0 && v_d.z != 0) {
//			cout<<"velocity measurement: "<<acc<<endl;
			mpDvlPreintegratedFromLastKF->IntegrateDVLMeasurement2(v_beam, tstep);
			mpDvlPreintegratedFromLastKF->v_dk_dvl = v_d;
			mpDvlPreintegratedFromLastKF->SetDVLDebugVelocity(v_d);
// //			ROS_INFO_STREAM("add kf velocity measurement:"<<acc);  // original
			pDvlGroPreIntegratedFromLastFrame->IntegrateDVLMeasurement2(v_beam, tstep);
			pDvlGroPreIntegratedFromLastFrame->v_dk_dvl = v_d;
			pDvlGroPreIntegratedFromLastFrame->SetDVLDebugVelocity(v_d);
			{
				DVLGroPreIntegration *pDvlPreintegratedFromLastKFBeforeLost = getLossIntegrationRef();
				std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);

				if (mDoLossIntegration) {
					pDvlPreintegratedFromLastKFBeforeLost->SetBeamOrientation(alpha, beta);
					pDvlPreintegratedFromLastKFBeforeLost->IntegrateDVLMeasurement2(v_beam, tstep);
					pDvlPreintegratedFromLastKFBeforeLost->v_dk_dvl = v_d;
					pDvlPreintegratedFromLastKFBeforeLost->SetDVLDebugVelocity(v_d);
// //					ROS_INFO_STREAM("add loss ref velocity measurement:"<<acc);  // original
				}
//				else{
// //					ROS_INFO_STREAM("pDvlPreintegratedFromLastKFBeforeLost do not exits");  // original
//				}
			}
		}
	}

//	mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
	mCurrentFrame.mpDvlPreintegrationFrame = pDvlGroPreIntegratedFromLastFrame;
//	mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
	mCurrentFrame.mpDvlPreintegrationKeyFrame = mpDvlPreintegratedFromLastKF;
	mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

	mCurrentFrame.setIntegrated();

}

void Tracking::PreintegrateDvlGro3()
{
	mpIntegrator->IntegrateMeasurements(mCurrentFrame,mlQueueDVLGyroData,mMutexImuQueue,mvGyroDVLFromLastFrame);

//	mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
	mCurrentFrame.mpDvlPreintegrationFrame = mpIntegrator->mpIntFromF_C2C;
//	mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
	mCurrentFrame.mpDvlPreintegrationKeyFrame = mpIntegrator->mpIntFromKF_C2C;
	mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

	mCurrentFrame.setIntegrated();

}

bool Tracking::PredictStateIMU()
{
	if (!mCurrentFrame.mpPrevFrame) {
		Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
		return false;
	}

	if (mbMapUpdated && mpLastKeyFrame) {
		const cv::Mat twb1 = mpLastKeyFrame->GetImuPosition();
		const cv::Mat Rwb1 = mpLastKeyFrame->GetImuRotation();
		const cv::Mat Vwb1 = mpLastKeyFrame->GetVelocityOld();

		const cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
		const float t12 = mpImuPreintegratedFromLastKF->dT;

		cv::Mat Rwb2 =
			IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
		cv::Mat twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz
			+ Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
		cv::Mat Vwb2 =
			Vwb1 + t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
		mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);
		mCurrentFrame.mPredRwb = Rwb2.clone();
		mCurrentFrame.mPredtwb = twb2.clone();
		mCurrentFrame.mPredVwb = Vwb2.clone();
		mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
		mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
		return true;
	}
	else if (!mbMapUpdated) {
		const cv::Mat twb1 = mLastFrame.GetImuPosition();
		const cv::Mat Rwb1 = mLastFrame.GetImuRotation();
		const cv::Mat Vwb1 = mLastFrame.mVw;
		const cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
		const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

		cv::Mat Rwb2 =
			IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
		cv::Mat twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz
			+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
		cv::Mat Vwb2 =
			Vwb1 + t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

		mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);
		mCurrentFrame.mPredRwb = Rwb2.clone();
		mCurrentFrame.mPredtwb = twb2.clone();
		mCurrentFrame.mPredVwb = Vwb2.clone();
		mCurrentFrame.mImuBias = mLastFrame.mImuBias;
		mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
		return true;
	}
	else {
		cout << "not IMU prediction!!" << endl;
	}

	return false;
}

bool Tracking::PredictStateDvlGro()
{
#ifdef AQUA_HAS_GTSAM_DYNAMIC
    if (mBackendMode == BackendMode::GtsamDynamic)
        return PredictStateWithDynamicBackend();
#endif
	//	if (mState == NOT_INITIALIZED && mpAtlas->GetAllMaps().size() > 1) {
	{
		// std::lock_guard<std::mutex> lock_LossRfe(mLossIntegrationRefMutex);
		if (mpIntegrator->GetDoLossIntegration()) {

            // DVLGroPreIntegration *pDvlPreintegratedFromLastKFBeforeLost = mpIntegrator->mpIntFromKFBeforeLost_C2C;
            // cv::Mat T_c0_cf_cv = mpIntegrator->mpLossRefKF->GetPoseInverse();
            // Eigen::Isometry3d T_g_d,T_d_c,T_c0_cf;

            // KeyFrame* pKF =mpIntegrator->mpLossRefKF;
            // DVLGroPreIntegration *pDvlPreintegratedFromKF = mpIntegrator->mpIntFromKFBeforeLost_C2C;
//             // ROS_INFO_STREAM(fixed<<setprecision(9)<<"loss KF timestamp: "<<pKF->mTimeStamp);  // original
//             // ROS_INFO_STREAM(fixed<<setprecision(9)<<"intgeration expected time: "<<pKF->mTimeStamp+pDvlPreintegratedFromKF->dT);  // original
//             // ROS_INFO_STREAM(fixed<<setprecision(9)<<"current frame time: "<<mCurrentFrame.mTimeStamp);  // original
            KeyFrame* pKF =*getMvpLossKf().rbegin();
//            KeyFrame* pKF_loss_end =*getMvpLossKf().rbegin();
//            KeyFrame* pKF_loss_start =*getMvpLossKf().begin();
// //            ROS_INFO_STREAM("loss kf start:"<<pKF_loss_start->mnId<<" end:"<<pKF_loss_end->mnId);  // original
            Eigen::Matrix3d R_b0_w = mpAtlas->getRGravity();
            DVLGroPreIntegration *pDvlPreintegratedFromKF = mpIntegrator->mpIntFromKF_C2C;
// //            ROS_INFO_STREAM("loss bias: x:"<<  // original
//            pDvlPreintegratedFromKF->mb.bwx<<" y:"<<pDvlPreintegratedFromKF->mb.bwy<<" z:"<<
//            pDvlPreintegratedFromKF->mb.bwz);
            cv::Mat T_c0_cf_cv = pKF->GetPoseInverse();
            Eigen::Vector3d v_df;
            pKF->GetDvlVelocity(v_df);
            Eigen::Isometry3d T_b_d,T_d_c,T_c0_cf,T_b_c;
            cv::Mat T_g_d_cv = GetExtrinsicPara().mT_imu_dvl;
            cv::Mat T_d_c_cv = GetExtrinsicPara().mT_dvl_c;
            cv::cv2eigen(T_g_d_cv, T_b_d.matrix());
            cv::cv2eigen(T_d_c_cv,T_d_c.matrix());
            cv::cv2eigen(T_c0_cf_cv,T_c0_cf.matrix());
            Eigen::Matrix3d R_c0_cf = T_c0_cf.rotation();
            Eigen::Vector3d t_c0_cf = T_c0_cf.translation();
            T_b_c = T_b_d * T_d_c;
            Eigen::Matrix3d R_b_d = T_b_d.rotation();
            Eigen::Vector3d t_b_d = T_b_d.translation();
            Eigen::Isometry3d T_b0_bf = T_b_c * T_c0_cf * T_b_c.inverse();
            Eigen::Matrix3d R_b0_bf = T_b0_bf.rotation();
            Eigen::Vector3d t_b0_bf =  T_b0_bf.translation();
            Eigen::Matrix3d R_bf_b1 = Eigen::Matrix3d::Identity();
            Eigen::Vector3d t_b0_b1 = Eigen::Vector3d::Zero();
            Eigen::Vector3d Dt_bf_b1 = Eigen::Vector3d::Zero();
            Eigen::Vector3d t_df_d1 = Eigen::Vector3d::Zero();
            cv::cv2eigen(pDvlPreintegratedFromKF->GetDeltaRotation(pDvlPreintegratedFromKF->mb), R_bf_b1);
            cv::cv2eigen(pDvlPreintegratedFromKF->GetDeltaPosition(pDvlPreintegratedFromKF->mb), Dt_bf_b1);
            cv::cv2eigen(pDvlPreintegratedFromKF->GetDVLPosition(pDvlPreintegratedFromKF->mb), t_df_d1);
            t_b0_b1 = R_b0_bf * Dt_bf_b1 + t_b0_bf + R_b0_bf * T_b_d.rotation()  * v_df * pDvlPreintegratedFromKF->dT
                      + 0.5*R_b0_w*Eigen::Vector3d(0,0,-9.8)*pDvlPreintegratedFromKF->dT*pDvlPreintegratedFromKF->dT;

            Eigen::Isometry3d T_d0_df = T_d_c * T_c0_cf * T_d_c.inverse();
            Eigen::Matrix3d R_df_d1 = R_b_d.transpose() * R_bf_b1 * R_b_d;
            Eigen::Isometry3d T_df_d1 = Eigen::Isometry3d::Identity();
            T_df_d1.rotate(R_df_d1);
            T_df_d1.pretranslate(t_df_d1);

            Eigen::Isometry3d T_d0_d1 = T_d0_df * T_df_d1;
            Eigen::Isometry3d T_c0_c1 = T_d_c.inverse() * T_d0_d1 *T_d_c;



            // Eigen::Vector3d t_b0_b1_dvl = t_b0_bf + R_b0_bf*(-R_bf_b1*t_b_d + R_b_d*t_df_d1 + t_b_d);
            // // Eigen::Vector3d t_b0_b1_dvl =  R_b_d*t_df_d1;
            // Eigen::Vector3d t_b0_b1_acc_v = t_b0_bf + R_b0_bf * T_b_d.rotation()  * v_df * pDvlPreintegratedFromKF->dT;
            //
            // Eigen::Matrix3d R_b0_b1 = R_b0_bf * R_bf_b1;
            //
            // Eigen::Isometry3d T_b0_b1 = Eigen::Isometry3d::Identity();
            // T_b0_b1.rotate(R_b0_b1);
            // T_b0_b1.pretranslate(t_b0_b1);
            //
            // Eigen::Isometry3d T_b0_b1_dvl = Eigen::Isometry3d::Identity();
            // T_b0_b1_dvl.rotate(R_b0_b1);
            // T_b0_b1_dvl.pretranslate(t_b0_b1_dvl);
            //
            // Eigen::Isometry3d T_b0_b1_acc_v = Eigen::Isometry3d::Identity();
            // T_b0_b1_acc_v.rotate(R_b0_b1);
            // T_b0_b1_acc_v.pretranslate(t_b0_b1_acc_v);
            //
            // Eigen::Isometry3d T_c0_c1 = T_b_c.inverse() * T_b0_b1 * T_b_c;
            // Eigen::Isometry3d T_c0_c1_dvl = T_b_c.inverse() * T_b0_b1_dvl * T_b_c;
            // Eigen::Isometry3d T_c0_c1_acc_v = T_b_c.inverse() * T_b0_b1_acc_v * T_b_c;
// 			// // ROS_INFO_STREAM("Gravity: \n" << R_b0_w);  // original
// 			// ROS_INFO_STREAM(fixed<<setprecision(6)<<"Loss KF[" << pKF->mnId << "]"<< " time: " << pKF->mTimeStamp);  // original
// 			// ROS_INFO_STREAM(fixed<<setprecision(6)<<"Loss integration time: "  // original
            // << pDvlPreintegratedFromKF->dT<<" v_d: "<<pDvlPreintegratedFromKF->mVelocity.t());
// 			// ROS_INFO_STREAM(fixed<<setprecision(6)<<"current frame time: " << mCurrentFrame.mTimeStamp);  // original
//             // ROS_INFO_STREAM("t_c0_c1:" << T_c0_c1.translation().transpose());  // original
//             // ROS_INFO_STREAM("t_c0_c1_dvl:" << T_c0_c1_dvl.translation().transpose());  // original
//             // ROS_INFO_STREAM("t_c0_c1_acc_v:" << T_c0_c1_acc_v.translation().transpose());  // original
//             // // ROS_INFO_STREAM("Dt_bf_b1: " << Dt_bf_b1.transpose());  // original
            cv::Mat T_c1_c0_cv(4,4,CV_32F);
            cv::eigen2cv(T_c0_c1.inverse().matrix(),T_c1_c0_cv);
            T_c1_c0_cv.convertTo(T_c1_c0_cv,CV_32F);
            mCurrentFrame.SetPose(T_c1_c0_cv);




            return true;
		}
	}


	if (mCurrentFrame.mpPrevFrame->mTimeStamp == 0) {
		Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
		return false;
	}
    DVLGroPreIntegration *pDvlPreintegratedFromKF = mCurrentFrame.mpDvlPreintegrationKeyFrame;
    auto pKF = mCurrentFrame.mpLastKeyFrame;
	if(!pDvlPreintegratedFromKF) {
		cout << "not IMU prediction!!" << endl;
		return false;
	}

    Eigen::Matrix3d R_b0_w = mpAtlas->getRGravity();
    cv::Mat T_c0_cf_cv = pKF->GetPoseInverse();
    Eigen::Vector3d v_df,v_df_mea;
    pKF->GetDvlVelocity(v_df);
    pKF->GetDvlVelocityMeasurement(v_df_mea);
    // ROS_DEBUG_STREAM("predict pose KF[" << pKF->mnId << "] dvl velocity: "<<v_df_mea.transpose()<<" opt velocity: "<<v_df.transpose());  // original
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "predict pose KF[" << pKF->mnId << "] dvl velocity: "<<v_df_mea.transpose()<<" opt velocity: "<<v_df.transpose());

    Eigen::Isometry3d T_b_d,T_d_c,T_c0_cf,T_b_c;
    cv::Mat T_g_d_cv = GetExtrinsicPara().mT_imu_dvl;
    cv::Mat T_d_c_cv = GetExtrinsicPara().mT_dvl_c;
    cv::cv2eigen(T_g_d_cv, T_b_d.matrix());
    cv::cv2eigen(T_d_c_cv,T_d_c.matrix());
    cv::cv2eigen(T_c0_cf_cv,T_c0_cf.matrix());
    Eigen::Matrix3d R_c0_cf = T_c0_cf.rotation();
    Eigen::Vector3d t_c0_cf = T_c0_cf.translation();
    T_b_c = T_b_d * T_d_c;
    Eigen::Matrix3d R_b_d = T_b_d.rotation();
    Eigen::Vector3d t_b_d = T_b_d.translation();
    Eigen::Isometry3d T_b0_bf = T_b_c * T_c0_cf * T_b_c.inverse();
    Eigen::Matrix3d R_b0_bf = T_b0_bf.rotation();
    Eigen::Vector3d t_b0_bf =  T_b0_bf.translation();
    Eigen::Matrix3d R_bf_b1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_b0_b1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d Dt_bf_b1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d t_df_d1 = Eigen::Vector3d::Zero();
    cv::cv2eigen(pDvlPreintegratedFromKF->GetDeltaRotation(pDvlPreintegratedFromKF->mb), R_bf_b1);
    cv::cv2eigen(pDvlPreintegratedFromKF->GetDeltaPosition(pDvlPreintegratedFromKF->mb), Dt_bf_b1);
    cv::cv2eigen(pDvlPreintegratedFromKF->GetDVLPosition(pDvlPreintegratedFromKF->mb,v_df), t_df_d1);
    t_b0_b1 = R_b0_bf * Dt_bf_b1 + t_b0_bf + R_b0_bf * T_b_d.rotation()  * v_df * pDvlPreintegratedFromKF->dT
              + 0.5*R_b0_w*Eigen::Vector3d(0,0,-9.8)*pDvlPreintegratedFromKF->dT*pDvlPreintegratedFromKF->dT;


    Eigen::Isometry3d T_c0_c1 = T_c0_cf;
    // Eigen::Isometry3d T_d0_df = T_d_c * T_c0_cf * T_d_c.inverse();
    // Eigen::Matrix3d R_df_d1 = R_b_d.transpose() * R_bf_b1 * R_b_d;
    // Eigen::Isometry3d T_df_d1 = Eigen::Isometry3d::Identity();
    // T_df_d1.rotate(R_df_d1);
    // T_df_d1.pretranslate(t_df_d1);
    // Eigen::Isometry3d T_d0_d1 = T_d0_df * T_df_d1;
    // T_c0_c1 = T_d_c.inverse() * T_d0_d1 *T_d_c;
    if(0){
        //use IMU as prediction
        // ROS_DEBUG_STREAM("use IMU as prediction");  // original
        RCLCPP_DEBUG(rclcpp::get_logger("aqua_slam"), "use IMU as prediction");
        Eigen::Matrix3d R_b0_b1 = R_b0_bf * R_bf_b1;
        Eigen::Isometry3d T_b0_b1 = Eigen::Isometry3d::Identity();
        T_b0_b1.rotate(R_b0_b1);
        T_b0_b1.pretranslate(t_b0_b1);
        T_c0_c1 = T_b_c.inverse() * T_b0_b1 * T_b_c;
    }
    else{
        //use DVL as prediction
        // ROS_DEBUG_STREAM("use DVL as prediction");  // original
        RCLCPP_DEBUG(rclcpp::get_logger("aqua_slam"), "use DVL as prediction");
        Eigen::Isometry3d T_d0_df = T_d_c * T_c0_cf * T_d_c.inverse();
        Eigen::Matrix3d R_df_d1 = R_b_d.transpose() * R_bf_b1 * R_b_d;
        Eigen::Isometry3d T_df_d1 = Eigen::Isometry3d::Identity();
        T_df_d1.rotate(R_df_d1);
        T_df_d1.pretranslate(t_df_d1);
        Eigen::Isometry3d T_d0_d1 = T_d0_df * T_df_d1;
        T_c0_c1 = T_d_c.inverse() * T_d0_d1 *T_d_c;
    }


    cv::Mat T_c1_c0_cv(4,4,CV_32F);
    cv::eigen2cv(T_c0_c1.inverse().matrix(),T_c1_c0_cv);
    T_c1_c0_cv.convertTo(T_c1_c0_cv,CV_32F);
    mCurrentFrame.SetPose(T_c1_c0_cv);
    return true;

    // return false;
}

void Tracking::ComputeGyroBias(const vector<Frame *> &vpFs, float &bwx, float &bwy, float &bwz)
{
	const int N = vpFs.size();
	vector<float> vbx;
	vbx.reserve(N);
	vector<float> vby;
	vby.reserve(N);
	vector<float> vbz;
	vbz.reserve(N);

	cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
	cv::Mat grad = cv::Mat::zeros(3, 1, CV_32F);
	for (int i = 1; i < N; i++) {
		Frame *pF2 = vpFs[i];
		Frame *pF1 = vpFs[i - 1];
		cv::Mat VisionR = pF1->GetImuRotation().t() * pF2->GetImuRotation();
		cv::Mat JRg = pF2->mpImuPreintegratedFrame->JRg;
		cv::Mat E = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaRotation().t() * VisionR;
		cv::Mat e = IMU::LogSO3(E);
		assert(fabs(pF2->mTimeStamp - pF1->mTimeStamp - pF2->mpImuPreintegratedFrame->dT) < 0.01);

		cv::Mat J = -IMU::InverseRightJacobianSO3(e) * E.t() * JRg;
		grad += J.t() * e;
		H += J.t() * J;
	}

	cv::Mat bg = -H.inv(cv::DECOMP_SVD) * grad;
	bwx = bg.at<float>(0);
	bwy = bg.at<float>(1);
	bwz = bg.at<float>(2);

	for (int i = 1; i < N; i++) {
		Frame *pF = vpFs[i];
		pF->mImuBias.bwx = bwx;
		pF->mImuBias.bwy = bwy;
		pF->mImuBias.bwz = bwz;
		pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
		pF->mpImuPreintegratedFrame->Reintegrate();
	}
}

void Tracking::ComputeVelocitiesAccBias(const vector<Frame *> &vpFs, float &bax, float &bay, float &baz)
{
	const int N = vpFs.size();
	const int nVar = 3 * N + 3; // 3 velocities/frame + acc bias
	const int nEqs = 6 * (N - 1);

	cv::Mat J(nEqs, nVar, CV_32F, cv::Scalar(0));
	cv::Mat e(nEqs, 1, CV_32F, cv::Scalar(0));
	cv::Mat g = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);

	for (int i = 0; i < N - 1; i++) {
		Frame *pF2 = vpFs[i + 1];
		Frame *pF1 = vpFs[i];
		cv::Mat twb1 = pF1->GetImuPosition();
		cv::Mat twb2 = pF2->GetImuPosition();
		cv::Mat Rwb1 = pF1->GetImuRotation();
		cv::Mat dP12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaPosition();
		cv::Mat dV12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaVelocity();
		cv::Mat JP12 = pF2->mpImuPreintegratedFrame->JPa;
		cv::Mat JV12 = pF2->mpImuPreintegratedFrame->JVa;
		float t12 = pF2->mpImuPreintegratedFrame->dT;
		// Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
		J.rowRange(6 * i, 6 * i + 3).colRange(3 * i, 3 * i + 3) += cv::Mat::eye(3, 3, CV_32F) * t12;
		J.rowRange(6 * i, 6 * i + 3).colRange(3 * N, 3 * N + 3) += Rwb1 * JP12;
		e.rowRange(6 * i, 6 * i + 3) = twb2 - twb1 - 0.5f * g * t12 * t12 - Rwb1 * dP12;
		// Velocity v2=v1+g*t+R1*dV12
		J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * i, 3 * i + 3) += -cv::Mat::eye(3, 3, CV_32F);
		J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * (i + 1), 3 * (i + 1) + 3) += cv::Mat::eye(3, 3, CV_32F);
		J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * N, 3 * N + 3) -= Rwb1 * JV12;
		e.rowRange(6 * i + 3, 6 * i + 6) = g * t12 + Rwb1 * dV12;
	}

	cv::Mat H = J.t() * J;
	cv::Mat B = J.t() * e;
	cv::Mat x(nVar, 1, CV_32F);
	cv::solve(H, B, x);

	bax = x.at<float>(3 * N);
	bay = x.at<float>(3 * N + 1);
	baz = x.at<float>(3 * N + 2);

	for (int i = 0; i < N; i++) {
		Frame *pF = vpFs[i];
		x.rowRange(3 * i, 3 * i + 3).copyTo(pF->mVw);
		if (i > 0) {
			pF->mImuBias.bax = bax;
			pF->mImuBias.bay = bay;
			pF->mImuBias.baz = baz;
			pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
		}
	}
}

void Tracking::ResetFrameIMU()
{
	// TODO To implement...
}

void Tracking::topicPublishDVLOnly()
{
//    Eigen::Isometry3d T_e0_el=mLastFrameBeforeLoss.mT_e0_ej;
//    Eigen::Isometry3d T_ej_el=mCurrentFrame.mT_e0_ej.inverse()*T_e0_el;
//    Eigen::Isometry3d T_cj_cl=mT_e_c.inverse()*T_ej_el*mT_e_c;
//    Eigen::Isometry3d T_cl_c0=Eigen::Isometry3d::Identity();
//    cv::cv2eigen(mLastFrameBeforeLoss.mTcw,T_cl_c0.matrix());
//    Eigen::Isometry3d T_cj_c0=T_cj_cl*T_cl_c0;
//    cv::Mat T_c_w(4,4,CV_32FC1);
//    cv::eigen2cv(T_cj_c0.matrix(),T_c_w);
//    T_c_w.convertTo(T_c_w,CV_32FC1);
//
//    mCurrentFrame.SetPose(T_c_w);
	// ORB pose
	if (mCurrentFrame.mTcw.empty() || !mInitialized) {
		return;
	}

	cv::Mat T_c_w = mCurrentFrame.mTcw.clone();

	Eigen::Isometry3d T_dvl_camera;
	cv::cv2eigen(mpImuCalib->mT_dvl_c, T_dvl_camera.matrix());

//    Eigen::Isometry3d T_cj_c0=Eigen::Isometry3d::Identity();
//    cv::cv2eigen(T_c_w, T_cj_c0.matrix());
	Eigen::Isometry3d T_c0_cj;
	cv::Mat T_d_c_cv = mpImuCalib->mT_dvl_c;
	Eigen::Isometry3d T_d_c;
	cv::cv2eigen(T_d_c_cv, T_d_c.matrix());

	cv::cv2eigen(T_c_w, T_c0_cj.matrix());
	T_c0_cj = T_c0_cj.inverse();
	Eigen::Isometry3d T_c0_cj_camera;
	T_c0_cj_camera.matrix() = T_c0_cj.matrix().replicate(1, 1);
	Eigen::Isometry3d T_c0_cj_orb = mT_c_cm * T_c0_cj * mT_c_cm.inverse();
	Eigen::Isometry3d T_d0_cj = T_d_c * T_c0_cj;

// // 	mpRosHandler->PublishOrb(T_c0_cj, T_d_c, ros::Time(mCurrentFrame.mTimeStamp));  // original  // original
	mpRosHandler->PublishOrb(T_c0_cj, T_d_c, mCurrentFrame.mTimeStamp, mCurrentFrame.mVw);
// // 	mpRosHandler->PublishCamera(T_c0_cj_camera, ros::Time(mCurrentFrame.mTimeStamp));  // original  // original
	// EKF pose
	Eigen::Isometry3d T_e0_ej_ekf = mCurrentFrame.mT_e0_ej;



    mpFrameDrawer->Update(this);
	cv::Mat img_with_info = mpFrameDrawer->DrawFrame(true);
	std_msgs::msg::Header header; // empty header
// // 	header.stamp = ros::Time::now(); // time  // original  // original
	cv_bridge::CvImage img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, img_with_info);
	mpRosHandler->PublishImgWithInfo(img_bridge.toImageMsg());



	// mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
}

void Tracking::Track()
{
#ifdef SAVE_TIMES
																															mTime_PreIntIMU = 0;
    mTime_PosePred = 0;
    mTime_LocalMapTrack = 0;
    mTime_NewKF_Dec = 0;
#endif

	if (bStepByStep) {
		while (!mbStep) {
			usleep(500);
		}
		mbStep = false;
	}
	while (mpSystem->mbResetActiveMap) {
		usleep(500);
	}

	if (mpLocalMapper->mbBadImu) {
		cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
		mpSystem->ResetActiveMap();
		return;
	}

	Map *pCurrentMap = mpAtlas->GetCurrentMap();

	if (mState != NO_IMAGES_YET) {
		if (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp) {
			cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
			unique_lock<mutex> lock(mMutexImuQueue);
			mlQueueImuData.clear();
			CreateMapInAtlas();
			return;
		}
		else if (mBackendMode != BackendMode::GtsamDynamic &&
                 mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 10.0) {
			cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
			if (mpAtlas->isInertial()) {

				if (mpAtlas->isImuInitialized()) {
					cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
					if (!pCurrentMap->GetIniertialBA2()) {
						mpSystem->ResetActiveMap();
					}
					else {
						CreateMapInAtlas();
					}
				}
				else {
					cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
					mpMapToReset = pCurrentMap;
					mpSystem->ResetActiveMap();
				}
			}

			return;
		}
	}


	if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mpLastKeyFrame) {
		mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());
	}

	if (mState == NO_IMAGES_YET) {
		mState = NOT_INITIALIZED;
	}

	mLastProcessedState = mState;

	if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && !mbCreatedMap) {
#ifdef SAVE_TIMES
		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
#endif
		PreintegrateIMU();
#ifdef SAVE_TIMES
																																std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        mTime_PreIntIMU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();
#endif

	}


	if (mSensor == System::DVL_STEREO && !mbCreatedMap) {
		PreintegrateDvlGro();
		{
//			std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);
//			if (mpDvlPreintegratedFromLastKFBeforeLost) {
////				cout<<"ref integration DV:"<<mpDvlPreintegratedFromLastKFBeforeLost->dV.t()<<" dt: "<<mpDvlPreintegratedFromLastKFBeforeLost->dT<<endl;
//				mpDvlPreintegratedFromLastKFBeforeLost->output();
//			}
//			if (mpDvlPreintegratedFromLastKF) {
////				cout<<"KF integration DV:"<<mpDvlPreintegratedFromLastKF->dV.t()<<" dt: "<<mpDvlPreintegratedFromLastKF->dT<<endl;
//			}
		}

	}
	mbCreatedMap = false;

	// Get Map Mutex -> Map cannot be changed
		unique_lock<shared_timed_mutex> lock(pCurrentMap->mMutexMapUpdate);

	mbMapUpdated = false;

	// set mbMapUpdated to update map later
	int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
	int nMapChangeIndex = pCurrentMap->GetLastMapChange();
	if (nCurMapChangeIndex > nMapChangeIndex) {
		// cout << "Map update detected" << endl;
		pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
		mbMapUpdated = true;
	}

	// Initialization
	if (mState == NOT_INITIALIZED) {
		cout << "try to initialize" << endl;
		if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO
			|| mSensor == System::DVL_STEREO) {
			StereoInitialization();
		}
		else {
			MonocularInitialization();
		}

		mpFrameDrawer->Update(this);

		if (mState != OK) // If rightly initialized, mState=OK
		{
			if (mSensor == System::DVL_STEREO) {
				topicPublishDVLOnly();
			}
			mLastFrame = Frame(mCurrentFrame);
			return;
		}
//        cout<<"initialization success"<<endl;
		if (mpAtlas->GetAllMaps().size() == 1) {
			mnFirstFrameId = mCurrentFrame.mnId;
		}
	}
		// after Initialization
	else {
		// System is initialized. Track Frame.
		bool bOK;

		// Initial camera pose estimation using motion model or relocalization (if tracking is lost)
		if (!mbOnlyTracking) {
#ifdef SAVE_TIMES
			std::chrono::steady_clock::time_point timeStartPosePredict = std::chrono::steady_clock::now();
#endif

			// State OK
			// Local Mapping is activated. This is the normal behaviour, unless
			// you explicitly activate the "only tracking" mode.
			if (mState == OK) {

				// Local Mapping might have changed some MapPoints tracked in last frame
				CheckReplacedInLastFrame();

				// no IMU, visual only, no velocity information
				if (mSensor == System::DVL_STEREO) {
					// Track from ReferenceKeyFrame after relocalization
					if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
						bOK = TrackReferenceKeyFrame();
					}
					else {


						bOK = TrackWithMotionModel();
//						bOK = mpLKTracker->trackFrame(mCurrentFrame,mLastFrame);

					}

					if (!bOK) {
						cout << "Fail to track with motion model!" << endl;
					}

				}
				else if ((mVelocity.empty() && !pCurrentMap->isImuInitialized())
					|| mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
					//Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
					bOK = TrackReferenceKeyFrame();
				}
				else {
					//Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
					bOK = TrackWithMotionModel();

					// just did relocalization
					if (!bOK) {
						bOK = TrackReferenceKeyFrame();
					}
				}


				if (!bOK) {
					if (mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
						(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) {
						mState = LOST;
					}
//					else if (mSensor == System::DVL_STEREO) {
//						mState = VISUAL_LOST;
//					}
					else if (pCurrentMap->KeyFramesInMap() > 10) {
						cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
						mState = RECENTLY_LOST;
						mTimeStampLost = mCurrentFrame.mTimeStamp;
						//mCurrentFrame.SetPose(mLastFrame.mTcw);
					}
					else {
						mState = LOST;
					}
				}
			}
			else {

				if (mState == RECENTLY_LOST) {
					Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

					bOK = true;
					// when lost recently, using IMU to predict pose
					// then try to tracklocalmap()
					// if slam can tracklocalmap() successfully in time_recently_lost, then continue
					// if failed to tracklocalmap()  in time_recently_lost, then lost
						if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) {
#ifdef AQUA_HAS_GTSAM_DYNAMIC
							if (mBackendMode == BackendMode::GtsamDynamic)
								bOK = PredictStateWithDynamicBackend();
							else
#endif
								bOK = pCurrentMap->isImuInitialized() && PredictStateIMU();

							if (mCurrentFrame.mTimeStamp - mTimeStampLost > time_recently_lost) {
								mState = LOST;
								Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
								bOK = false;
							}
					}
					else if ((mSensor == System::DVL_STEREO && mCalibrated)) {
						PredictStateDvlGro();

						if (mCurrentFrame.mTimeStamp - mTimeStampLost > time_recently_lost) {
							mState = LOST;
							Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
							cout << "Track Lost..." << endl;
							bOK = false;

						}
					}
					else {
						// TODO fix relocalization
						bOK = Relocalization();
						if (!bOK) {
							mState = LOST;
							Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
							bOK = false;
						}
					}
				}
				else if (mState == LOST) {

					Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

					if (pCurrentMap->KeyFramesInMap() < mKFThresholdForMap) {
						mpSystem->ResetActiveMap();
						mpMapToReset = mpAtlas->GetCurrentMap();
						cout << "Reseting current map..." << endl;
					}
					else {
						CreateMapInAtlas();
					}

					if (mpLastKeyFrame) {
						mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
					}

					Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

					return;
				}

			}


#ifdef SAVE_TIMES
																																	std::chrono::steady_clock::time_point timeEndPosePredict = std::chrono::steady_clock::now();

        mTime_PosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(timeEndPosePredict - timeStartPosePredict).count();
#endif

		}
		else {
			// Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
			if (mState == LOST) {
				if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
					Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
				}
				bOK = Relocalization();
			}
			else {
				if (!mbVO) {
					// In last frame we tracked enough MapPoints in the map
					if (!mVelocity.empty()) {
						bOK = TrackWithMotionModel();
					}
					else {
						bOK = TrackReferenceKeyFrame();
					}
				}
				else {
					// In last frame we tracked mainly "visual odometry" points.

					// We compute two camera poses, one from motion model and one doing relocalization.
					// If relocalization is sucessfull we choose that solution, otherwise we retain
					// the "visual odometry" solution.

					bool bOKMM = false;
					bool bOKReloc = false;
					vector<MapPoint *> vpMPsMM;
					vector<bool> vbOutMM;
					cv::Mat TcwMM;
					if (!mVelocity.empty()) {
						bOKMM = TrackWithMotionModel();
						vpMPsMM = mCurrentFrame.mvpMapPoints;
						vbOutMM = mCurrentFrame.mvbOutlier;
						TcwMM = mCurrentFrame.mTcw.clone();
					}
					bOKReloc = Relocalization();

					if (bOKMM && !bOKReloc) {
						mCurrentFrame.SetPose(TcwMM);
						mCurrentFrame.mvpMapPoints = vpMPsMM;
						mCurrentFrame.mvbOutlier = vbOutMM;

						if (mbVO) {
							for (int i = 0; i < mCurrentFrame.N; i++) {
								if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
									mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
								}
							}
						}
					}
					else if (bOKReloc) {
						mbVO = false;
					}

					bOK = bOKReloc || bOKMM;
				}
			}
		}

		if (!mCurrentFrame.mpReferenceKF) {
			mCurrentFrame.mpReferenceKF = mpReferenceKF;
		}

		// track local map
		// If we have an initial estimation of the camera pose and matching. Track the local map.
		if (!mbOnlyTracking) {
			if (bOK) {
//                if((mSensor==System::DVL_STEREO) && (mState==VISUAL_LOST))
				if (mSensor == System::DVL_STEREO) {
					//todo_tightly
					//	decide whether use DVL+ gyros to optimize in tracking thread or not
					//	use them make visual tracking easy to lose
//					bOK = TrackLocalMapWithDvlGyro();
					bOK = TrackLocalMap();
//					drawOptimizationResult();
				}

				else {
					bOK = TrackLocalMap();
				}
			}
			if (!bOK) {
				cout << "Fail to track local map!" << endl;
			}
		}
		else {
			// mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
			// a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
			// the camera we will use the local map again.
			if (bOK && !mbVO) {
				bOK = TrackLocalMap();
			}
		}

		if (bOK) {
			mState = OK;
		}
			// lost first
		else if (mState == OK) {
			if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
				Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
					if (mBackendMode != BackendMode::GtsamDynamic &&
						(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())) {
					cout << "IMU is not or recently initialized. Reseting active map..." << endl;
					mpSystem->ResetActiveMap();
					mpMapToReset = mpAtlas->GetCurrentMap();
				}

				mState = RECENTLY_LOST;
			}
			else if (mSensor == System::DVL_STEREO) {
				if (!mCalibrated) {
					cout << "Gyros is not initialized. Reseting active map..." << endl;
					mpSystem->ResetActiveMap();
				}

				mState = RECENTLY_LOST;
			}
			else {
				mState = LOST;
			} // visual to lost

			if (mCurrentFrame.mnId > mnLastRelocFrameId + mMaxFrames) {
				mTimeStampLost = mCurrentFrame.mTimeStamp;
			}
		}

		// Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
		// ?????????????????
		if ((mCurrentFrame.mnId < (mnLastRelocFrameId + mnFramesToResetIMU))
			&& (mCurrentFrame.mnId > mnFramesToResetIMU)
			&& ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO))
			&& pCurrentMap->isImuInitialized()) {
			// TODO check this situation
			Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
			Frame *pF = new Frame(mCurrentFrame);
			pF->mpPrevFrame = new Frame(mLastFrame);

			// Load preintegration
			pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
		}

		// ??????????????????????
		// ResetFrameIMU() no implementation
		if (pCurrentMap->isImuInitialized()) {
			if (bOK) {
				if (mCurrentFrame.mnId == (mnLastRelocFrameId + mnFramesToResetIMU)) {
					cout << "RESETING FRAME!!!" << endl;
					ResetFrameIMU();
				}
				else if (mCurrentFrame.mnId > (mnLastRelocFrameId + 30)) {
					mLastBias = mCurrentFrame.mImuBias;
				}
			}
		}

		// Update drawer
		mpFrameDrawer->Update(this);
		// publish current pose estimation
		if (!mCurrentFrame.mTcw.empty()) {
			// mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
			// Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
			// twc = -Rwc*mCameraPose.rowRange(0,3).col(3);


			Frame cur_frame_backup = mCurrentFrame;
//            PredictStateDvlGro();
			// ORB pose
			cv::Mat current_pose = mCurrentFrame.mTcw.clone();
			Eigen::Isometry3d T_cj_c0 = Eigen::Isometry3d::Identity();
			cv::cv2eigen(current_pose, T_cj_c0.matrix());
			Eigen::Isometry3d T_c0_cj = T_cj_c0.inverse();
			Eigen::Isometry3d T_dvl_camera;
			cv::cv2eigen(mpImuCalib->mT_dvl_c, T_dvl_camera.matrix());
			Eigen::Isometry3d T_c0_cj_camera;
			T_c0_cj_camera.matrix() = T_c0_cj.matrix().replicate(1, 1);
			Eigen::Isometry3d T_c0_cj_orb = mT_c_cm * T_c0_cj * mT_c_cm.inverse();
			// EKF pose
			Eigen::Isometry3d T_e0_ej_ekf = mCurrentFrame.mT_e0_ej;
			mCurrentFrame = cur_frame_backup;

			cv::Mat T_d_c_cv = mpImuCalib->mT_dvl_c;
			Eigen::Isometry3d T_d_c;
			cv::cv2eigen(T_d_c_cv, T_d_c.matrix());
			Eigen::Isometry3d T_d0_cj = T_d_c * T_c0_cj;

// // 			mpRosHandler->PublishOrb(T_c0_cj, T_d_c, ros::Time(mCurrentFrame.mTimeStamp));  // original  // original
				mpRosHandler->PublishOrb(T_c0_cj, T_d_c, mCurrentFrame.mTimeStamp, mCurrentFrame.mVw);
// // 			mpRosHandler->PublishCamera(T_c0_cj_camera, ros::Time(mCurrentFrame.mTimeStamp));  // original  // original
//			mpRosHandler->UpdateMap(mpAtlas);
			cv::Mat img_with_info = mpFrameDrawer->DrawFrame(true);
			std_msgs::msg::Header header; // empty header
// // 			header.stamp = ros::Time::now(); // time  // original  // original
			cv_bridge::CvImage
				img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, img_with_info);
			mpRosHandler->PublishImgWithInfo(img_bridge.toImageMsg());

		}


		// handle the result of tracklocalmap()
		if (bOK || mState == RECENTLY_LOST) {
			// Update motion model

			if (!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty()) {
				cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
				mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
				mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
				// T_cj_ci
				mVelocity = mCurrentFrame.mTcw * LastTwc;
				bool motionModelValid = true;
				if (mBackendMode == BackendMode::GtsamDynamic) {
					motionModelValid =
						ProjectMotionModelPoseToSE3(mVelocity, mVelocity);
				}
				if (!motionModelValid) {
					RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"),
					             "invalid dynamic motion-model update: frame=%lu "
					             "timestamp=%.9f current_finite=%d last_finite=%d "
					             "last_inverse_finite=%d current_type=%d last_type=%d",
					             mCurrentFrame.mnId, mCurrentFrame.mTimeStamp,
					             cv::checkRange(mCurrentFrame.mTcw),
					             cv::checkRange(mLastFrame.mTcw),
					             cv::checkRange(LastTwc), mCurrentFrame.mTcw.type(),
					             mLastFrame.mTcw.type());
					mVelocity.release();
				}
				if (motionModelValid) {
				Eigen::Isometry3d T_ci_cj = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mVelocity, T_ci_cj.matrix());
				T_ci_cj = T_ci_cj.inverse();
				Eigen::Isometry3d T_dvl_c = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mpImuCalib->mT_dvl_c, T_dvl_c.matrix());
				Eigen::Isometry3d T_di_dj = T_dvl_c * T_ci_cj * T_dvl_c.inverse();
				float dt = mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp;
				Eigen::Vector3d v_di
					(T_di_dj.translation().x() / dt, T_di_dj.translation().y() / dt, T_di_dj.translation().z() / dt);
				Eigen::Isometry3d T_c0_ci = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mLastFrame.mTcw, T_c0_ci.matrix());
				T_c0_ci = T_c0_ci.inverse();

				Eigen::Isometry3d T_c_dvl = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mCurrentFrame.mImuCalib.mT_c_dvl, T_c_dvl.matrix());
				Eigen::Vector3d v_c0 = T_c0_ci.rotation() * T_c_dvl.rotation() * v_di;
				cv::Mat v_c0_cv;
				cv::eigen2cv(v_c0, v_c0_cv);
				v_c0_cv.convertTo(v_c0_cv, CV_32F);
				// cause issue???
				mCurrentFrame.SetVelocity(v_c0_cv);
//				cout << "set velocity v_di: " << v_di.x() << " " << v_di.y() << " " << v_di.z() << endl;
//				cout << "set velocity v_c0: " << v_c0.x() << " " << v_c0.y() << " " << v_c0.z() << endl;
				}

			}
			else {
				mVelocity = cv::Mat();
			}

			if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
				// mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
			}

			// Clean VO matches
			for (int i = 0; i < mCurrentFrame.N; i++) {
				MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
				if (pMP) {
					if (pMP->Observations() < 1) {
						mCurrentFrame.mvbOutlier[i] = false;
						mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
					}
				}
			}

			// Delete temporal MapPoints
			for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
			     lit != lend; lit++) {
				MapPoint *pMP = *lit;
				delete pMP;
			}
			mlpTemporalPoints.clear();

#ifdef SAVE_TIMES
			std::chrono::steady_clock::time_point timeStartNewKF = std::chrono::steady_clock::now();
#endif
			bool bNeedKF = NeedNewKeyFrame();
#ifdef SAVE_TIMES
																																	std::chrono::steady_clock::time_point timeEndNewKF = std::chrono::steady_clock::now();

            mTime_NewKF_Dec = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(timeEndNewKF - timeStartNewKF).count();
#endif



			// Check if we need to insert a new keyframe
//			if (bNeedKF && (bOK
//				|| (mState == RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)))) {
//				CreateNewKeyFrame();
//			}
			if (bNeedKF && bOK) {
				CreateNewKeyFrame();
				if (pCurrentMap->KeyFramesInMap() >= mKFThresholdForMap) {
					mDoLossIntegration = false;
				}
			}

			// We allow points with high innovation (considererd outliers by the Huber Function)
			// pass to the new keyframe, so that bundle adjustment will finally decide
			// if they are outliers or not. We don't want next frame to estimate its position
			// with those points so we discard them in the frame. Only has effect if lastframe is tracked
			for (int i = 0; i < mCurrentFrame.N; i++) {
				if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]) {
					mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				}
			}
		}

		// Reset if the camera get lost soon after initialization
		if (mState == LOST || mState == VISUAL_LOST) {
			if (pCurrentMap->KeyFramesInMap() <= mKFThresholdForMap) {
				mpSystem->ResetActiveMap();
				mpMapToReset = mpAtlas->GetCurrentMap();
				return;
			}
			if ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)
				|| (mSensor == System::DVL_STEREO)) {
				if (!mCalibrated) {
					Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
					mpSystem->ResetActiveMap();
					mpMapToReset = mpAtlas->GetCurrentMap();
					return;
				}
			}

			CreateMapInAtlas();
		}

		if (!mCurrentFrame.mpReferenceKF) {
			mCurrentFrame.mpReferenceKF = mpReferenceKF;
		}

		mLastFrame = Frame(mCurrentFrame);
	}


	if (mState == OK || mState == RECENTLY_LOST) {
		// Store frame pose information to retrieve the complete camera trajectory afterwards.
		if (!mCurrentFrame.mTcw.empty()) {
			cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
			mlRelativeFramePoses.push_back(Tcr);
			mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
			mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
			mlbLost.push_back(mState == LOST);
		}
		else {
			// This can happen if tracking is lost
			mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
			mlpReferences.push_back(mlpReferences.back());
			mlFrameTimes.push_back(mlFrameTimes.back());
			mlbLost.push_back(mState == LOST);
		}

	}
}

void Tracking::UpdateDynamicRecoveryState(bool visualTracked, bool predictionAccepted)
{
    if (visualTracked) {
        mState = OK;
        return;
    }
    if (mState == OK)
        mTimeStampLost = mCurrentFrame.mTimeStamp;
    mState = predictionAccepted &&
        mCurrentFrame.mTimeStamp - mTimeStampLost <= time_recently_lost
        ? RECENTLY_LOST : LOST;
}

void Tracking::TrackDVLGyro()
{

    // ROS_DEBUG_STREAM(fixed << setprecision(6)<<"Frame["<<mCurrentFrame.mnId<<"] "<<mCurrentFrame.mTimeStamp);  // original
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), std::fixed << std::setprecision(6) << "Frame[" << mCurrentFrame.mnId << "] " << mCurrentFrame.mTimeStamp);
	if (bStepByStep) {
		while (!mbStep) {
			usleep(500);
		}
		mbStep = false;
	}
	while (mpSystem->mbResetActiveMap) {
		usleep(500);
	}


	Map *pCurrentMap = mpAtlas->GetCurrentMap();

	if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mpLastKeyFrame) {
		mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());
	}

	if (mState == NO_IMAGES_YET) {
		mState = NOT_INITIALIZED;
	}

	mLastProcessedState = mState;



	mbCreatedMap = false;

	// Get Map Mutex -> Map cannot be changed
	unique_lock<shared_timed_mutex> lock(pCurrentMap->mMutexMapUpdate);

	mbMapUpdated = false;

	// set mbMapUpdated to update map later
	int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
	int nMapChangeIndex = pCurrentMap->GetLastMapChange();
//     // ROS_INFO_STREAM("Map change check in tracking: "<<nCurMapChangeIndex);  // original
	if (nCurMapChangeIndex > nMapChangeIndex) {
		// cout << "Map update detected" << endl;
		pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
		mbMapUpdated = true;
	}
    PreintegrateDvlGro3();

	// Initialization
	if (mState == NOT_INITIALIZED) {
        StereoInitialization();

		mpFrameDrawer->Update(this);
        topicPublishDVLOnly();

		if (mState != OK) // If rightly initialized, mState=OK
		{
            // topicPublishDVLOnly();
			mLastFrame = Frame(mCurrentFrame);
//             ROS_INFO_STREAM(fixed<<setprecision(6)<<"fail to initialize, Frame["<<mCurrentFrame.mnId<<"], "<<mCurrentFrame.mTimeStamp);  // original
			return;
		}
//        cout<<"initialization success"<<endl;
		if (mpAtlas->GetAllMaps().size() == 1) {
			mnFirstFrameId = mCurrentFrame.mnId;
		}
//         ROS_INFO_STREAM(fixed<<setprecision(6)<<"initialization is done, Frame["<<mCurrentFrame.mnId<<"], "<<mCurrentFrame.mTimeStamp);  // original
        mLastFrame = Frame(mCurrentFrame);
	}
		// after Initialization
	else {
		// System is initialized. Track Frame.
		bool bOK = false;
        bool predictionAccepted = false;

		// do DVL-IMU printegration

		// if (mSensor == System::DVL_STEREO ) {
        //     PreintegrateDvlGro3();
		// }

		// Initial camera pose estimation using motion model or relocalization (if tracking is lost)
		// State OK
		if (mState == OK || mState == RECENTLY_LOST) {

			// Local Mapping might have changed some MapPoints tracked in last frame
			CheckReplacedInLastFrame();

			// no IMU, visual only, no velocity information
			if (mSensor == System::DVL_STEREO) {
				// Track from ReferenceKeyFrame after relocalization
				if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
					bOK = TrackReferenceKeyFrame();
				}
				else {


					bOK = TrackWithMotionModel();
//						bOK = mpLKTracker->trackFrame(mCurrentFrame,mLastFrame);

				}

				if (!bOK) {
                    bOK = TrackReferenceKeyFrame();

//                     ROS_INFO_STREAM("Fail to track with motion model!");  // original
					// cout << "Fail to track with motion model!" << endl;
                    if (!bOK) {
                        predictionAccepted = PredictStateDvlGro();
                        bOK = predictionAccepted;
                    }


				}


			}


		}
		else {
            Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

            if (pCurrentMap->KeyFramesInMap() < mKFThresholdForMap) {
                mpSystem->ResetActiveMap();
                mpMapToReset = mpAtlas->GetCurrentMap();
                mLastFrame = Frame(mCurrentFrame);
                cout << "Reseting current map..." << endl;
            }
            else {
                CreateMapInAtlas();
            }

            // if (mpLastKeyFrame) {
            //     mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
            // }

            return;

		}

		if (!mCurrentFrame.mpReferenceKF) {
			mCurrentFrame.mpReferenceKF = mpReferenceKF;
		}

		// track local map
		// If we have an initial estimation of the camera pose and matching. Track the local map.
		if (bOK) {
			//todo_tightly
			//	decide whether use DVL+ gyros to optimize in tracking thread or not
			//	use them make visual tracking easy to lose
			if(mCalibrated)
				// bOK = TrackLocalMapWithDvlGyro();
				bOK = TrackLocalMap();
			else//
				bOK = TrackLocalMap();
            // if(mCurrentFrame.mPoorVision){
                // PredictStateDvlGro();
            // }
            if (!bOK) {
//                 ROS_INFO_STREAM("Fail to track local map!");  // original
                predictionAccepted = PredictStateDvlGro();
            }
		}

        UpdateDynamicRecoveryState(bOK, predictionAccepted);


		// Update drawer
		mpFrameDrawer->Update(this);
		// publish current pose estimation
		if (!mCurrentFrame.mTcw.empty()&&mInitialized) {
			// mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
			// Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
			// twc = -Rwc*mCameraPose.rowRange(0,3).col(3);


			Frame cur_frame_backup = mCurrentFrame;
//            PredictStateDvlGro();
			// ORB pose
			cv::Mat current_pose = mCurrentFrame.mTcw.clone();
			Eigen::Isometry3d T_cj_c0 = Eigen::Isometry3d::Identity();
			cv::cv2eigen(current_pose, T_cj_c0.matrix());
			Eigen::Isometry3d T_c0_cj = T_cj_c0.inverse();
			Eigen::Isometry3d T_dvl_camera;
			cv::cv2eigen(mpImuCalib->mT_dvl_c, T_dvl_camera.matrix());
			Eigen::Isometry3d T_c0_cj_camera;
			T_c0_cj_camera.matrix() = T_c0_cj.matrix().replicate(1, 1);
			Eigen::Isometry3d T_c0_cj_orb = mT_c_cm * T_c0_cj * mT_c_cm.inverse();
			// EKF pose
			Eigen::Isometry3d T_e0_ej_ekf = mCurrentFrame.mT_e0_ej;
			mCurrentFrame = cur_frame_backup;

			cv::Mat T_d_c_cv = mpImuCalib->mT_dvl_c;
			Eigen::Isometry3d T_d_c;
			cv::cv2eigen(T_d_c_cv, T_d_c.matrix());
			Eigen::Isometry3d T_d0_cj = T_d_c * T_c0_cj;

// // 			mpRosHandler->PublishOrb(T_c0_cj, T_d_c, ros::Time(mCurrentFrame.mTimeStamp));  // original  // original
				mpRosHandler->PublishOrb(T_c0_cj, T_d_c, mCurrentFrame.mTimeStamp, mCurrentFrame.mVw);
// // 			mpRosHandler->PublishCamera(T_c0_cj_camera, ros::Time(mCurrentFrame.mTimeStamp));  // original  // original
//			mpRosHandler->UpdateMap(mpAtlas);
			cv::Mat img_with_info = mpFrameDrawer->DrawFrame(true);
			std_msgs::msg::Header header; // empty header
// // 			header.stamp = ros::Time::now(); // time  // original  // original
			cv_bridge::CvImage
				img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, img_with_info);
			mpRosHandler->PublishImgWithInfo(img_bridge.toImageMsg());

		}


		// handle the result of tracklocalmap()
		if (bOK) {
			// Update motion model

			if (!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty()) {
				cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
				mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
				mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
				// T_cj_ci
				mVelocity = mCurrentFrame.mTcw * LastTwc;
				Eigen::Isometry3d T_ci_cj = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mVelocity, T_ci_cj.matrix());
				T_ci_cj = T_ci_cj.inverse();
				Eigen::Isometry3d T_dvl_c = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mpImuCalib->mT_dvl_c, T_dvl_c.matrix());
				Eigen::Isometry3d T_di_dj = T_dvl_c * T_ci_cj * T_dvl_c.inverse();
				float dt = mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp;
				Eigen::Vector3d v_di
					(T_di_dj.translation().x() / dt, T_di_dj.translation().y() / dt, T_di_dj.translation().z() / dt);
				Eigen::Isometry3d T_c0_ci = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mLastFrame.mTcw, T_c0_ci.matrix());
				T_c0_ci = T_c0_ci.inverse();

				Eigen::Isometry3d T_c_dvl = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mCurrentFrame.mImuCalib.mT_c_dvl, T_c_dvl.matrix());
				Eigen::Vector3d v_c0 = T_c0_ci.rotation() * T_c_dvl.rotation() * v_di;
				cv::Mat v_c0_cv;
				cv::eigen2cv(v_c0, v_c0_cv);
				v_c0_cv.convertTo(v_c0_cv, CV_32F);
				// cause issue???
				mCurrentFrame.SetVelocity(v_c0_cv);
//				cout << "set velocity v_di: " << v_di.x() << " " << v_di.y() << " " << v_di.z() << endl;
//				cout << "set velocity v_c0: " << v_c0.x() << " " << v_c0.y() << " " << v_c0.z() << endl;

			}
			else {
				mVelocity = cv::Mat();
			}


			// Clean VO matches
			for (int i = 0; i < mCurrentFrame.N; i++) {
				MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
				if (pMP) {
					if (pMP->Observations() < 1) {
						mCurrentFrame.mvbOutlier[i] = false;
						mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
					}
				}
			}

			// Delete temporal MapPoints
			for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
			     lit != lend; lit++) {
				MapPoint *pMP = *lit;
				delete pMP;
			}
			mlpTemporalPoints.clear();

			bool bNeedKF = NeedNewKeyFrame();



			// Check if we need to insert a new keyframe
//			if (bNeedKF && (bOK
//				|| (mState == RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)))) {
//				CreateNewKeyFrame();
//			}
			if (bNeedKF && bOK) {
                if (pCurrentMap->KeyFramesInMap() >= mKFThresholdForMap) {
                    // mDoLossIntegration = false;
                    if(mpIntegrator->GetDoLossIntegration()){
                        mpIntegrator->SetDoLossIntegration(false);
                        mCurrentFrame.mpDvlPreintegrationLossRefKF = mpIntegrator->mpIntFromKFBeforeLost_C2C;
                        mCurrentFrame.mpLossRefKF = mpIntegrator->mpLossRefKF;
                        CreateNewKeyFrame();
//                         ROS_INFO_STREAM(fixed<<setprecision(6)<<"KF["<<mpLastKeyFrame->mnId  // original
//                                              <<"] "<<mpLastKeyFrame->mTimeStamp<<" Loss Reference has beed set to KF["  // original
//                                              <<mpLastKeyFrame->mpLossRefKF->mnId<<"] "<<mpLastKeyFrame->mpLossRefKF->mTimeStamp);  // original
                        KeyFrame* pkf = mpLastKeyFrame;
                        std::unique_lock<std::shared_mutex> lock(mLossKFMutex);
                        while(mvpLossKF.count(pkf)==0 && (pkf->mPrevKF!=nullptr)){
                            if (!pkf->isBad()){
                                mvpLossKF.insert(pkf);
                                pkf = pkf->mPrevKF;
                            }
                        }
                    }
                    else{
                        CreateNewKeyFrame();
                    }

                }
                else{
                    CreateNewKeyFrame();
                    if(mpIntegrator->GetDoLossIntegration()){
                        KeyFrame* pkf = mpLastKeyFrame;
                        std::unique_lock<std::shared_mutex> lock(mLossKFMutex);
                        mvpLossKF.insert(pkf);
                        // ROS_DEBUG_STREAM("add KF["<<pkf->mnId<<"] to loss KF set");  // original
                        RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "add KF[" << pkf->mnId << "] to loss KF set");
                        auto loss_kf = mvpLossKF;
                        // Optimizer::PoseOnlyOptimizationDVLIMU(loss_kf, mpAtlas);
                    }
                }



			}

			// We allow points with high innovation (considererd outliers by the Huber Function)
			// pass to the new keyframe, so that bundle adjustment will finally decide
			// if they are outliers or not. We don't want next frame to estimate its position
			// with those points so we discard them in the frame. Only has effect if lastframe is tracked
			for (int i = 0; i < mCurrentFrame.N; i++) {
				if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]) {
					mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				}
			}
		}

		// Reset if the camera get lost soon after initialization
		if (mState == LOST) {
			if (pCurrentMap->KeyFramesInMap() <= mKFThresholdForMap) {
                mpSystem->ResetActiveMap();
                mpMapToReset = mpAtlas->GetCurrentMap();
                mLastFrame = Frame(mCurrentFrame);
                return;
            }
            else{
                CreateMapInAtlas();
                return;
            }


		}

		if (!mCurrentFrame.mpReferenceKF) {
			mCurrentFrame.mpReferenceKF = mpReferenceKF;
		}

		mLastFrame = Frame(mCurrentFrame);
	}


	if (mState == OK) {
		// Store frame pose information to retrieve the complete camera trajectory afterwards.
		if (!mCurrentFrame.mTcw.empty()) {
			cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
			mlRelativeFramePoses.push_back(Tcr);
			mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
			mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
			mlbLost.push_back(mState == LOST);
		}
		else {
			// This can happen if tracking is lost
			mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
			mlpReferences.push_back(mlpReferences.back());
			mlFrameTimes.push_back(mlFrameTimes.back());
			mlbLost.push_back(mState == LOST);
		}

	}
}

void Tracking::TrackDVLImu()
{

}

void Tracking::TrackKLT()
{

	Map *pCurrentMap = mpAtlas->GetCurrentMap();

	if (mState == NO_IMAGES_YET) {
		mState = NOT_INITIALIZED;
	}

	mLastProcessedState = mState;


	if (mSensor == System::DVL_STEREO && !mbCreatedMap) {
		PreintegrateDvlGro();
		{
//			std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);
//			if (mpDvlPreintegratedFromLastKFBeforeLost) {
////				cout<<"ref integration DV:"<<mpDvlPreintegratedFromLastKFBeforeLost->dV.t()<<" dt: "<<mpDvlPreintegratedFromLastKFBeforeLost->dT<<endl;
//				mpDvlPreintegratedFromLastKFBeforeLost->output();
//			}
//			if (mpDvlPreintegratedFromLastKF) {
////				cout<<"KF integration DV:"<<mpDvlPreintegratedFromLastKF->dV.t()<<" dt: "<<mpDvlPreintegratedFromLastKF->dT<<endl;
//			}
		}

	}
	mbCreatedMap = false;

	// Get Map Mutex -> Map cannot be changed
	unique_lock<shared_timed_mutex> lock(pCurrentMap->mMutexMapUpdate);

	mbMapUpdated = false;

	// set mbMapUpdated to update map later
	int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
	int nMapChangeIndex = pCurrentMap->GetLastMapChange();
	if (nCurMapChangeIndex > nMapChangeIndex) {
		// cout << "Map update detected" << endl;
		pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
		mbMapUpdated = true;
	}

	// Initialization
	if (mState == NOT_INITIALIZED) {

		StereoInitialization();

		mpFrameDrawer->Update(this);

		if (mState != OK) // If rightly initialized, mState=OK
		{
			if (mSensor == System::DVL_STEREO) {
				topicPublishDVLOnly();
			}
			mLastFrame = Frame(mCurrentFrame);
			return;
		}
		cout << "initialization success" << endl;
		if (mpAtlas->GetAllMaps().size() == 1) {
			mnFirstFrameId = mCurrentFrame.mnId;
		}
	}
		// after Initialization
	else {
		// System is initialized. Track Frame.
		bool bOK;

		// Initial camera pose estimation using motion model or relocalization (if tracking is lost)

		// State OK
		// Local Mapping is activated. This is the normal behaviour, unless
		// you explicitly activate the "only tracking" mode.
		if (mState == OK) {

			// Local Mapping might have changed some MapPoints tracked in last frame
			CheckReplacedInLastFrame();

			// no IMU, visual only, no velocity information
			// Track from ReferenceKeyFrame after relocalization
			if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
				bOK = TrackReferenceKeyFrame();
			}
			else {
				bOK = TrackWithMotionModel();
			}
			if (!bOK) {
				cout << "Fail to track with motion model!" << endl;
			}


			if (!bOK) {
				if (mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
					(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) {
					mState = LOST;
				}
				else if (pCurrentMap->KeyFramesInMap() > 10) {
					cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
					mState = RECENTLY_LOST;
					mTimeStampLost = mCurrentFrame.mTimeStamp;
					//mCurrentFrame.SetPose(mLastFrame.mTcw);
				}
				else {
					mState = LOST;
				}
			}
		}
		else {
			if (mState == RECENTLY_LOST) {
				Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

				bOK = true;
				// when lost recently, using IMU to predict pose
				// then try to tracklocalmap()
				// if slam can tracklocalmap() successfully in time_recently_lost, then continue
				// if failed to tracklocalmap()  in time_recently_lost, then lost
				PredictStateDvlGro();

				if (mCurrentFrame.mTimeStamp - mTimeStampLost > time_recently_lost) {
					mState = LOST;
					Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
					cout << "Track Lost..." << endl;
					bOK = false;

				}

			}
			else if (mState == LOST) {

				Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

				if (pCurrentMap->KeyFramesInMap() < mKFThresholdForMap) {
					mpSystem->ResetActiveMap();
					cout << "Reseting current map..." << endl;
				}
				else {
					CreateMapInAtlas();
				}

				if (mpLastKeyFrame) {
					mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
				}

				Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

				return;
			}

		}


		if (!mCurrentFrame.mpReferenceKF) {
			mCurrentFrame.mpReferenceKF = mpReferenceKF;
		}

		// track local map
		// If we have an initial estimation of the camera pose and matching. Track the local map.
		if (bOK) {
			//todo_tightly
			//	decide whether use DVL+ gyros to optimize in tracking thread or not
			//	use them make visual tracking easy to lose
			//bOK = TrackLocalMapWithDvlGyro();
			bOK = TrackLocalMap();
		}
		if (!bOK) {
			cout << "Fail to track local map!" << endl;
		}


		if (bOK) {
			mState = OK;
		}
			// lost first
		else if (mState == OK) {
			if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
				Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
				if (!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2()) {
					cout << "IMU is not or recently initialized. Reseting active map..." << endl;
					mpSystem->ResetActiveMap();
				}

				mState = RECENTLY_LOST;
			}
			else if (mSensor == System::DVL_STEREO) {
				if (!mCalibrated) {
					cout << "Gyros is not initialized. Reseting active map..." << endl;
					mpSystem->ResetActiveMap();
				}

				mState = RECENTLY_LOST;
			}
			else {
				mState = LOST;
			} // visual to lost

			if (mCurrentFrame.mnId > mnLastRelocFrameId + mMaxFrames) {
				mTimeStampLost = mCurrentFrame.mTimeStamp;
			}
		}

		// Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
		// ?????????????????
		if ((mCurrentFrame.mnId < (mnLastRelocFrameId + mnFramesToResetIMU))
			&& (mCurrentFrame.mnId > mnFramesToResetIMU)
			&& ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO))
			&& pCurrentMap->isImuInitialized()) {
			// TODO check this situation
			Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
			Frame *pF = new Frame(mCurrentFrame);
			pF->mpPrevFrame = new Frame(mLastFrame);

			// Load preintegration
			pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
		}

		// ??????????????????????
		// ResetFrameIMU() no implementation
		if (pCurrentMap->isImuInitialized()) {
			if (bOK) {
				if (mCurrentFrame.mnId == (mnLastRelocFrameId + mnFramesToResetIMU)) {
					cout << "RESETING FRAME!!!" << endl;
					ResetFrameIMU();
				}
				else if (mCurrentFrame.mnId > (mnLastRelocFrameId + 30)) {
					mLastBias = mCurrentFrame.mImuBias;
				}
			}
		}

		// Update drawer
		mpFrameDrawer->Update(this);
		// publish current pose estimation
		if (!mCurrentFrame.mTcw.empty()) {

			// Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
			// twc = -Rwc*mCameraPose.rowRange(0,3).col(3);


			Frame cur_frame_backup = mCurrentFrame;
//            PredictStateDvlGro();
			// ORB pose
			cv::Mat current_pose = mCurrentFrame.mTcw.clone();
			Eigen::Isometry3d T_cj_c0 = Eigen::Isometry3d::Identity();
			cv::cv2eigen(current_pose, T_cj_c0.matrix());
			Eigen::Isometry3d T_c0_cj = T_cj_c0.inverse();
			Eigen::Isometry3d T_dvl_camera;
			cv::cv2eigen(mpImuCalib->mT_dvl_c, T_dvl_camera.matrix());
			Eigen::Isometry3d T_c0_cj_camera;
			T_c0_cj_camera.matrix() = T_c0_cj.matrix().replicate(1, 1);
			Eigen::Isometry3d T_c0_cj_orb = mT_c_cm * T_c0_cj * mT_c_cm.inverse();
			// EKF pose
			Eigen::Isometry3d T_e0_ej_ekf = mCurrentFrame.mT_e0_ej;
			mCurrentFrame = cur_frame_backup;

			cv::Mat T_d_c_cv = mpImuCalib->mT_dvl_c;
			Eigen::Isometry3d T_d_c;
			cv::cv2eigen(T_d_c_cv, T_d_c.matrix());
			Eigen::Isometry3d T_d0_cj = T_d_c * T_c0_cj;

// // 			mpRosHandler->PublishOrb(T_c0_cj, T_d_c, ros::Time(mCurrentFrame.mTimeStamp));  // original  // original
				mpRosHandler->PublishOrb(T_c0_cj, T_d_c, mCurrentFrame.mTimeStamp, mCurrentFrame.mVw);
// // 			mpRosHandler->PublishCamera(T_c0_cj_camera, ros::Time(mCurrentFrame.mTimeStamp));  // original  // original
//			mpRosHandler->UpdateMap(mpAtlas);
			cv::Mat img_with_info = mpFrameDrawer->DrawFrame(true);
			std_msgs::msg::Header header; // empty header
// // 			header.stamp = ros::Time::now(); // time  // original  // original
			cv_bridge::CvImage
				img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, img_with_info);
			mpRosHandler->PublishImgWithInfo(img_bridge.toImageMsg());

		}


		// handle the result of tracklocalmap()
		if (bOK || mState == RECENTLY_LOST) {
			// Update motion model

			if (!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty()) {
				cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
				mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
				mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
				// T_cj_ci
				mVelocity = mCurrentFrame.mTcw * LastTwc;
				Eigen::Isometry3d T_ci_cj = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mVelocity, T_ci_cj.matrix());
				T_ci_cj = T_ci_cj.inverse();
				Eigen::Isometry3d T_dvl_c = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mpImuCalib->mT_dvl_c, T_dvl_c.matrix());
				Eigen::Isometry3d T_di_dj = T_dvl_c * T_ci_cj * T_dvl_c.inverse();
				float dt = mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp;
				Eigen::Vector3d v_di
					(T_di_dj.translation().x() / dt, T_di_dj.translation().y() / dt, T_di_dj.translation().z() / dt);
				Eigen::Isometry3d T_c0_ci = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mLastFrame.mTcw, T_c0_ci.matrix());
				T_c0_ci = T_c0_ci.inverse();

				Eigen::Isometry3d T_c_dvl = Eigen::Isometry3d::Identity();
				cv::cv2eigen(mCurrentFrame.mImuCalib.mT_c_dvl, T_c_dvl.matrix());
				Eigen::Vector3d v_c0 = T_c0_ci.rotation() * T_c_dvl.rotation() * v_di;
				cv::Mat v_c0_cv;
				cv::eigen2cv(v_c0, v_c0_cv);
				v_c0_cv.convertTo(v_c0_cv, CV_32F);
				// cause issue???
				mCurrentFrame.SetVelocity(v_c0_cv);
//				cout << "set velocity v_di: " << v_di.x() << " " << v_di.y() << " " << v_di.z() << endl;
//				cout << "set velocity v_c0: " << v_c0.x() << " " << v_c0.y() << " " << v_c0.z() << endl;

			}
			else {
				mVelocity = cv::Mat();
			}



			// Clean VO matches
			for (int i = 0; i < mCurrentFrame.N; i++) {
				MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
				if (pMP) {
					if (pMP->Observations() < 1) {
						mCurrentFrame.mvbOutlier[i] = false;
						mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
					}
				}
			}

			// Delete temporal MapPoints
			for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
			     lit != lend; lit++) {
				MapPoint *pMP = *lit;
				delete pMP;
			}
			mlpTemporalPoints.clear();


			bool bNeedKF = NeedNewKeyFrame();



			// Check if we need to insert a new keyframe
//			if (bNeedKF && (bOK
//				|| (mState == RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)))) {
//				CreateNewKeyFrame();
//			}
			if (bNeedKF && bOK) {
				CreateNewKeyFrame();
				if (pCurrentMap->KeyFramesInMap() >= mKFThresholdForMap) {
					mDoLossIntegration = false;
				}
			}

			// We allow points with high innovation (considererd outliers by the Huber Function)
			// pass to the new keyframe, so that bundle adjustment will finally decide
			// if they are outliers or not. We don't want next frame to estimate its position
			// with those points so we discard them in the frame. Only has effect if lastframe is tracked
			for (int i = 0; i < mCurrentFrame.N; i++) {
				if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]) {
					mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				}
			}
		}

		// Reset if the camera get lost soon after initialization
		if (mState == LOST || mState == VISUAL_LOST) {
			if (pCurrentMap->KeyFramesInMap() <= mKFThresholdForMap) {
				mpSystem->ResetActiveMap();
				return;
			}
			if ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)
				|| (mSensor == System::DVL_STEREO)) {
				if (!mCalibrated) {
					Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
					mpSystem->ResetActiveMap();
					return;
				}
			}

			CreateMapInAtlas();
		}

		if (!mCurrentFrame.mpReferenceKF) {
			mCurrentFrame.mpReferenceKF = mpReferenceKF;
		}

		mLastFrame = Frame(mCurrentFrame);
	}


	if (mState == OK || mState == RECENTLY_LOST) {
		// Store frame pose information to retrieve the complete camera trajectory afterwards.
		if (!mCurrentFrame.mTcw.empty()) {
			cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
			mlRelativeFramePoses.push_back(Tcr);
			mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
			mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
			mlbLost.push_back(mState == LOST);
		}
		else {
			// This can happen if tracking is lost
			mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
			mlpReferences.push_back(mlpReferences.back());
			mlFrameTimes.push_back(mlFrameTimes.back());
			mlbLost.push_back(mState == LOST);
		}

	}
}

void Tracking::StereoInitialization()
{
//     ROS_INFO_STREAM("try to initialize");  // original
	int nDepth = 0;
	for (int i = 0; i < mCurrentFrame.N; ++i) {
		if (mCurrentFrame.mvDepth[i] > 0)
			++nDepth;
	}
	RCLCPP_DEBUG(rclcpp::get_logger("aqua_slam"),
	             "StereoInit attempt N=%d depthValid=%d", mCurrentFrame.N, nDepth);
	const bool dynamicStereoInitialization =
		mBackendMode == BackendMode::GtsamDynamic &&
		mSensor == System::IMU_STEREO;
	int persistentDepthMatches = 0;
	int persistentDepthAppearanceMatches = 0;
	if (dynamicStereoInitialization && mHasDynamicStereoInitCandidate &&
		mCurrentFrame.N > 0 && nDepth > 0) {
		FeatureSet initialFeatures{mInitialFrame.mvKeysUn,
		                           mInitialFrame.mDescriptors};
		FeatureSet currentFeatures{mCurrentFrame.mvKeysUn,
		                           mCurrentFrame.mDescriptors};
		const auto matches = mpFeatureFrontend->Match(
			initialFeatures, mInitialFrame.imgLeft.size(),
			currentFeatures, mCurrentFrame.imgLeft.size());
		mvIniMatches.assign(mInitialFrame.N, -1);
		for (const auto& match : matches)
			mvIniMatches[match.query_index] = match.train_index;
		for (size_t i = 0; i < mvIniMatches.size() && i < mInitialFrame.mvDepth.size(); ++i) {
			if (mvIniMatches[i] >= 0 && mInitialFrame.mvDepth[i] > 0) {
				++persistentDepthMatches;
				++persistentDepthAppearanceMatches;
			}
		}
	}

	if (!ShouldCommitStereoInitialization(dynamicStereoInitialization,
	                                      mHasDynamicStereoInitCandidate,
	                                      mCurrentFrame.N,
	                                      nDepth,
	                                      persistentDepthMatches,
	                                      persistentDepthAppearanceMatches)) {
		if (dynamicStereoInitialization && mCurrentFrame.N > 0 && nDepth > 0) {
			mInitialFrame = Frame(mCurrentFrame);
			mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
			for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); ++i)
				mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
			mvIniMatches.assign(mCurrentFrame.mvKeysUn.size(), -1);
			mHasDynamicStereoInitCandidate = true;
			RCLCPP_INFO(rclcpp::get_logger("aqua_slam"),
			            "StereoInit pending: frame=%lu features=%d depthValid=%d "
				            "persistentDepthMatches=%d persistentDepthAppearanceMatches=%d",
			            mCurrentFrame.mnId, mCurrentFrame.N, nDepth,
				            persistentDepthMatches,
				            persistentDepthAppearanceMatches);
		}
		else if (dynamicStereoInitialization) {
			mHasDynamicStereoInitCandidate = false;
			mvbPrevMatched.clear();
			mvIniMatches.clear();
		}
		else if (mSensor == System::DVL_STEREO &&
		         mpIntegrator->GetDoLossIntegration()) {
			PredictStateDvlGro();
		}
		return;
	}

	mHasDynamicStereoInitCandidate = false;
	mvbPrevMatched.clear();
	mvIniMatches.clear();
	{
		if (mSensor == System::DVL_STEREO) {
            if (mpIntegrator->GetDoLossIntegration()) {
                PredictStateDvlGro();
            }
            else {
                mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            }
            if(mInitialized)
            {
                std::shared_lock<std::shared_mutex> lock(mBiasMutex);
                mpIntegrator->CreateNewIntFromKF_C2C(mLastBias, GetExtrinsicPara(), mAlpha, mBeta);
//                 ROS_INFO_STREAM("create new preintegration from KF with bias: " << mLastBias.bax << " "  // original
//                                                                                 << mLastBias.bay << " " << mLastBias.baz << " " << mLastBias.bwx << " " << mLastBias.bwy  // original
//                                                                                 << " " << mLastBias.bwz);  // original
            }
            else{
// 				ROS_INFO_STREAM("create new preintegration from KF with 0 bias");  // original
                mpIntegrator->CreateNewIntFromKF_C2C(IMU::Bias(), GetExtrinsicPara(), mAlpha, mBeta);
            }
            if(!mlQueueDVLGyroData.empty()){
                mpIntegrator->IntegrateMeasurements(mCurrentFrame,mlQueueDVLGyroData,mMutexImuQueue,mvGyroDVLFromLastFrame);
                mCurrentFrame.mpDvlPreintegrationFrame = mpIntegrator->mpIntFromF_C2C;
                mCurrentFrame.mpDvlPreintegrationKeyFrame = mpIntegrator->mpIntFromKF_C2C;
                mCurrentFrame.setIntegrated();
            }

		}
		else {
			mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
			cout << "use identity as new initialization pose" << endl;
		}


		// Create KeyFrame
		KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
        if (mpLastKeyFrame) {
            pKFini->mPrevKF = mpLastKeyFrame;
            mpLastKeyFrame->mNextKF = pKFini;
            if (mSensor == System::DVL_STEREO && pKFini->mpDvlPreintegrationKeyFrame) {
                pKFini->SetNewBias(pKFini->mpDvlPreintegrationKeyFrame->mb);
            }
        }
        // else: first init or re-init after map reset — no prev KF to link

		mpAtlas->AddKeyFrame(pKFini);
		mpLastKeyFrame = pKFini;

		// Create MapPoints and asscoiate to KeyFrame
		if (!mpCamera2) {
			for (int i = 0; i < mCurrentFrame.N; i++) {
				float z = mCurrentFrame.mvDepth[i];
				if (z > 0) {
					// x3D.at<fload>(col,row)
					cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
					// calculate the map point in world coordinate
					MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
					pNewMP->AddObservation(pKFini, i);
					pKFini->AddMapPoint(pNewMP, i);
					pNewMP->ComputeDistinctiveDescriptors();
					pNewMP->UpdateNormalAndDepth();
					mpAtlas->AddMapPoint(pNewMP);

					mCurrentFrame.mvpMapPoints[i] = pNewMP;
				}
			}
		}
		else {
			for (int i = 0; i < mCurrentFrame.Nleft; i++) {
				int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
				if (rightIndex != -1) {
					cv::Mat x3D = mCurrentFrame.mvStereo3Dpoints[i];

					MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

					pNewMP->AddObservation(pKFini, i);
					pNewMP->AddObservation(pKFini, rightIndex + mCurrentFrame.Nleft);

					pKFini->AddMapPoint(pNewMP, i);
					pKFini->AddMapPoint(pNewMP, rightIndex + mCurrentFrame.Nleft);

					pNewMP->ComputeDistinctiveDescriptors();
					pNewMP->UpdateNormalAndDepth();
					mpAtlas->AddMapPoint(pNewMP);

					mCurrentFrame.mvpMapPoints[i] = pNewMP;
					mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft] = pNewMP;
				}
			}
		}

		//Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

		mpLocalMapper->InsertKeyFrame(pKFini);

		mLastFrame = Frame(mCurrentFrame);
		mnLastKeyFrameId = mCurrentFrame.mnId;
		mpLastKeyFrame = pKFini;
		mnLastRelocFrameId = mCurrentFrame.mnId;

		mvpLocalKeyFrames.push_back(pKFini);
		mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
		mpReferenceKF = pKFini;
		mCurrentFrame.mpReferenceKF = pKFini;

		mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

		mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

		// mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

		mState = OK;
		RCLCPP_INFO(rclcpp::get_logger("aqua_slam"),
		            "StereoInit SUCCESS: KF=%lu mapPoints=%zu depthValid=%d",
		            pKFini->mnId, mpAtlas->MapPointsInMap(), nDepth);

        if(mpIntegrator->GetDoLossIntegration()){
            std::unique_lock<std::shared_mutex> lock(mLossKFMutex);
            mvpLossKF.insert(pKFini);
            // ROS_DEBUG_STREAM("add KF["<<pKFini->mnId<<"] to loss KF set");  // original
            RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "add KF[" << pKFini->mnId << "] to loss KF set");
        }
        auto all_loss_kf =getMvpLossKf();
        if(all_loss_kf.size()>0 && (mpIntegrator->GetDoLossIntegration())){
            #ifndef AQUA_HAS_GTSAM_DYNAMIC
            if (mBackendMode != BackendMode::GtsamDynamic)
                Optimizer::PoseOnlyOptimizationDVLIMU(all_loss_kf, mpAtlas, mLossLastOptID);
            #endif
            // ROS_DEBUG_STREAM("DVL IMU Optimization done");  // original
            RCLCPP_DEBUG(rclcpp::get_logger("aqua_slam"), "DVL IMU Optimization done");
            for (auto pKF : all_loss_kf) {
                // ROS_DEBUG_STREAM(fixed<<setprecision(6)<< "KF[" << pKF->mnId << "] bias: "...);  // original
                RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), std::fixed << std::setprecision(6)
                    << "KF[" << pKF->mnId << "] bias: " << pKF->GetImuBias()
                    << " integration duration: " << pKF->mpDvlPreintegrationKeyFrame->dT);
            }
            // UpdateFrameDVLGyro(pKFini->GetImuBias(),pKFini);
			PredictStateDvlGro();
        }
        // mpRosHandler->PublishLossKF(all_loss_kf);
	}
	}

void Tracking::StereoInitializationKLT()
{
	if (mCurrentFrame.N > 500) {


		if (mpDvlPreintegratedFromLastKF) {
			delete mpDvlPreintegratedFromLastKF;
			mpDvlPreintegratedFromLastKF = new DVLGroPreIntegration(IMU::Bias(), GetExtrinsicPara());
		}

		DVLGroPreIntegration *pDvlPreintegratedFromLastKFBeforeLost = NULL;
		{
			std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);
			if (!mpDvlPreintegratedFromLastKFBeforeLost) {
				mpDvlPreintegratedFromLastKFBeforeLost = new DVLGroPreIntegration(IMU::Bias(), GetExtrinsicPara());
				// mLastFrameBeforeLoss = Frame(mCurrentFrame);
				// mLastFrameBeforeLoss.SetPose(cv::Mat::eye(4, 4, CV_32F));
			}
			pDvlPreintegratedFromLastKFBeforeLost = mpDvlPreintegratedFromLastKFBeforeLost;
		}

		if (mDoLossIntegration) {
			PredictStateDvlGro();
			topicPublishDVLOnly();
		}
		else {
			cv::Mat T_cj_c0 = cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_cj_d0 = T_cj_c0 * mCurrentFrame.mImuCalib.mT_c_dvl;
			mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
//				mCurrentFrame.SetPose(T_cj_d0);
//				cv::Mat Rwgyro0 = mCurrentFrame.mImuCalib.mT_c_imu.rowRange(0, 3).colRange(0, 3).clone();
//				cv::Mat twdvl0 = mCurrentFrame.mImuCalib.mT_c_dvl.rowRange(0, 3).col(3).clone();
//				mCurrentFrame.SetDvlPoseVelocity(Rwgyro0, twdvl0, cv::Mat::zeros(3, 1, CV_32F));
		}
		mCurrentFrame.mpDvlPreintegrationKeyFrame = mpDvlPreintegratedFromLastKF;



		// Create KeyFrame
		KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

		if (mpDvlPreintegratedFromLastKF) {
//			delete mpDvlPreintegratedFromLastKF;
			mpDvlPreintegratedFromLastKF = new DVLGroPreIntegration(IMU::Bias(), GetExtrinsicPara());
		}

		// Insert KeyFrame in the map
		mpAtlas->AddKeyFrame(pKFini);

		// Create MapPoints and asscoiate to KeyFrame
		for (int i = 0; i < mCurrentFrame.N; i++) {
			float z = mCurrentFrame.mvDepth[i];
			if (z > 0) {
				// x3D.at<fload>(col,row)
				cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
				// calculate the map point in world coordinate
				MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
				pNewMP->AddObservation(pKFini, i);
				pKFini->AddMapPoint(pNewMP, i);
				pNewMP->ComputeDistinctiveDescriptors();
				pNewMP->UpdateNormalAndDepth();
				mpAtlas->AddMapPoint(pNewMP);

				mCurrentFrame.mvpMapPoints[i] = pNewMP;
			}
		}

		//Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

		mpLocalMapper->InsertKeyFrame(pKFini);

		mLastFrame = Frame(mCurrentFrame);
		mnLastKeyFrameId = mCurrentFrame.mnId;
		mpLastKeyFrame = pKFini;
		mnLastRelocFrameId = mCurrentFrame.mnId;

		mvpLocalKeyFrames.push_back(pKFini);
		mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
		mpReferenceKF = pKFini;
		mCurrentFrame.mpReferenceKF = pKFini;

		mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

		mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

		// mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

		mState = OK;
	}
}

void Tracking::MonocularInitialization()
{

	if (!mpInitializer) {
		// Set Reference Frame
		if (mCurrentFrame.mvKeys.size() > 100) {

			mInitialFrame = Frame(mCurrentFrame);
			mLastFrame = Frame(mCurrentFrame);
			mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
			for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
				mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

			if (mpInitializer) {
				delete mpInitializer;
			}

			mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

			fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

			if (mSensor == System::IMU_MONOCULAR) {
				if (mpImuPreintegratedFromLastKF) {
					delete mpImuPreintegratedFromLastKF;
				}
				mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), GetExtrinsicPara());
				mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

			}
			return;
		}
	}
	else {
		if (((int)mCurrentFrame.mvKeys.size() <= 100)
			|| ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp > 1.0))) {
			delete mpInitializer;
			mpInitializer = static_cast<Initializer *>(NULL);
			fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

			return;
		}

		// Find correspondences
		FeatureMatcher matcher(*mpFeatureFrontend, 0.9F);
		int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

		// Check if there are enough correspondences
		if (nmatches < 100) {
			delete mpInitializer;
			mpInitializer = static_cast<Initializer *>(NULL);
			fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
			return;
		}

		cv::Mat Rcw; // Current Camera Rotation
		cv::Mat tcw; // Current Camera Translation
		vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

		if (mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,
		                                      mCurrentFrame.mvKeysUn,
		                                      mvIniMatches,
		                                      Rcw,
		                                      tcw,
		                                      mvIniP3D,
		                                      vbTriangulated)) {
			for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
				if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
					mvIniMatches[i] = -1;
					nmatches--;
				}
			}

			// Set Frame Poses
			mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
			cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
			Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
			tcw.copyTo(Tcw.rowRange(0, 3).col(3));
			mCurrentFrame.SetPose(Tcw);

			CreateInitialMapMonocular();

			// Just for video
			// bStepByStep = true;
		}
	}
}

void Tracking::CreateInitialMapMonocular()
{
	// Create KeyFrames
	KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
	KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

	if (mSensor == System::IMU_MONOCULAR) {
		pKFini->mpImuPreintegrated = (IMU::Preintegrated *)(NULL);
	}


	// Insert KFs in the map
	mpAtlas->AddKeyFrame(pKFini);
	mpAtlas->AddKeyFrame(pKFcur);

	for (size_t i = 0; i < mvIniMatches.size(); i++) {
		if (mvIniMatches[i] < 0) {
			continue;
		}

		//Create MapPoint.
		cv::Mat worldPos(mvIniP3D[i]);
		MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());

		pKFini->AddMapPoint(pMP, i);
		pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

		pMP->AddObservation(pKFini, i);
		pMP->AddObservation(pKFcur, mvIniMatches[i]);

		pMP->ComputeDistinctiveDescriptors();
		pMP->UpdateNormalAndDepth();

		//Fill Current Frame structure
		mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
		mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

		//Add to Map
		mpAtlas->AddMapPoint(pMP);
	}


	// Update Connections
	pKFini->UpdateConnections();
	pKFcur->UpdateConnections();

	std::set<MapPoint *> sMPs;
	sMPs = pKFini->GetMapPoints();

	pKFcur->PrintPointDistribution();

	float medianDepth = pKFini->ComputeSceneMedianDepth(2);
	float invMedianDepth;
	if (mSensor == System::IMU_MONOCULAR) {
		invMedianDepth = 4.0f / medianDepth; // 4.0f
	}
	else {
		invMedianDepth = 1.0f / medianDepth;
	}

	if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) // TODO Check, originally 100 tracks
	{
		Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_NORMAL);
		mpSystem->ResetActiveMap();
		return;
	}

	// Scale initial baseline
	cv::Mat Tc2w = pKFcur->GetPose();
	Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
	pKFcur->SetPose(Tc2w);

	// Scale points
	vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
	for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
		if (vpAllMapPoints[iMP]) {
			MapPoint *pMP = vpAllMapPoints[iMP];
			pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
			pMP->UpdateNormalAndDepth();
		}
	}

	if (mSensor == System::IMU_MONOCULAR) {
		pKFcur->mPrevKF = pKFini;
		pKFini->mNextKF = pKFcur;
		pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

		mpImuPreintegratedFromLastKF =
			new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), pKFcur->mImuCalib);
	}


	mpLocalMapper->InsertKeyFrame(pKFini);
	mpLocalMapper->InsertKeyFrame(pKFcur);
	mpLocalMapper->mFirstTs = pKFcur->mTimeStamp;

	mCurrentFrame.SetPose(pKFcur->GetPose());
	mnLastKeyFrameId = mCurrentFrame.mnId;
	mpLastKeyFrame = pKFcur;
	mnLastRelocFrameId = mInitialFrame.mnId;

	mvpLocalKeyFrames.push_back(pKFcur);
	mvpLocalKeyFrames.push_back(pKFini);
	mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
	mpReferenceKF = pKFcur;
	mCurrentFrame.mpReferenceKF = pKFcur;

	// Compute here initial velocity
	vector<KeyFrame *> vKFs = mpAtlas->GetAllKeyFrames();

	mVelocity = cv::Mat();

	mLastFrame = Frame(mCurrentFrame);

	mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

	// mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

	mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

	mState = OK;

	initID = pKFcur->mnId;
}

void Tracking::CreateMapInAtlas()
{
	cout << "create map" << endl;
	mpLoopClosing->RequestResetActiveMap(mpAtlas->GetCurrentMap());
#ifdef AQUA_HAS_GTSAM_DYNAMIC
	if (mBackendMode == BackendMode::GtsamDynamic) {
		ResetDynamicBackendState();
		std::lock_guard<std::mutex> lock(mMutexImuQueue);
		mDynamicKeyframeImu.clear();
		mDynamicKeyframeDvl.clear();
		mDynamicInitializationImu.clear();
        mLastDynamicInitializationAttempt = -std::numeric_limits<double>::infinity();
		mCollectDynamicInitializationImu = true;
	}
#endif
	mnLastInitFrameId = mCurrentFrame.mnId;
	mpAtlas->CreateNewMap();
	if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR) {
		mpAtlas->SetInertialSensor();
	}
	mbSetInit = false;

	mnInitialFrameId = mCurrentFrame.mnId + 1;
	mState = NO_IMAGES_YET;
	mHasDynamicStereoInitCandidate = false;

	// Restart the variable with information about the last KF
	mVelocity = cv::Mat();
	mnLastRelocFrameId =
		mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
	Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId + 1), Verbose::VERBOSITY_NORMAL);
	mbVO = false; // Init value for know if there are enough MapPoints in the last KF
	if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR) {
		if (mpInitializer) {
			delete mpInitializer;
		}
		mpInitializer = static_cast<Initializer *>(NULL);
	}

	if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mpImuPreintegratedFromLastKF) {
		delete mpImuPreintegratedFromLastKF;
		mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), GetExtrinsicPara());
	}
	else if (mSensor == System::DVL_STEREO) {
//        GetBeamOrientation()
//         mpIntegrator->CreateNewIntFromKF_C2C(IMU::Bias(),GetExtrinsicPara(), mAlpha, mBeta);
        // mpIntegrator->CreateNewIntFromKF_D2D(IMU::Bias(),GetExtrinsicPara(), mAlpha, mBeta);
	}
	if (mSensor == System::DVL_STEREO) {
		// std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);
		// if (mpDvlPreintegratedFromLastKFBeforeLost) {
		// 	delete mpDvlPreintegratedFromLastKFBeforeLost;
		// 	mpDvlPreintegratedFromLastKFBeforeLost = NULL;
		// }
		// mpDvlPreintegratedFromLastKFBeforeLost = new DVLGroPreIntegration(IMU::Bias(), GetExtrinsicPara());
		// mLastFrameBeforeLoss = Frame(mCurrentFrame);
		// mDoLossIntegration = true;

        mpIntegrator->CreateNewIntFromKFBeforeLoss_D2D(mpIntegrator->mpIntFromKF_C2C);
        mpIntegrator->SetLossRefKF(mpLastKeyFrame);
        mpIntegrator->SetDoLossIntegration(true);
//         ROS_INFO_STREAM("Create new DVL preintegration before loss, bias: "  // original
//         <<mpIntegrator->mpIntFromKFBeforeLost_C2C->mb.bax<<", "  // original
//         <<mpIntegrator->mpIntFromKFBeforeLost_C2C->mb.bay<<", "  // original
//         <<mpIntegrator->mpIntFromKFBeforeLost_C2C->mb.baz);  // original
        std::unique_lock<std::shared_mutex> lock(mLossKFMutex);
        mvpLossKF.clear();
		// insert 5 KF before mpLastKeyFrame to mvpLossKF
        auto pkf =mpLastKeyFrame;
        mLossLastOptID = mpLastKeyFrame->mnId;
        // remember to change in optimization, if change the number if KF inserted
		while(mvpLossKF.size() < 5 && pkf){
// 			ROS_INFO_STREAM("pKF["<<pkf->mnId<<"] inserted");  // original
			mvpLossKF.insert(pkf);
            pkf = pkf->mPrevKF;
		}


	}

	mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

	if (mpReferenceKF) {
		mpReferenceKF = static_cast<KeyFrame *>(NULL);
	}

	mLastFrame = Frame(mCurrentFrame);
//	mLastFrame.mTimeStamp = mCurrentFrame.mTimeStamp;
	mCurrentFrame = Frame();
	mvIniMatches.clear();
	mvbPrevMatched.clear();

	mbCreatedMap = true;

}

void Tracking::CheckReplacedInLastFrame()
{
	for (int i = 0; i < mLastFrame.N; i++) {
		MapPoint *pMP = mLastFrame.mvpMapPoints[i];

		if (pMP) {
			MapPoint *pRep = pMP->GetReplaced();
			if (pRep) {
				mLastFrame.mvpMapPoints[i] = pRep;
			}
		}
	}
}

bool Tracking::TrackReferenceKeyFrame()
{
	// Match the current frame to its reference keyframe before pose refinement.
	FeatureMatcher matcher(*mpFeatureFrontend, 0.7F);
	vector<MapPoint *> vpMapPointMatches;

	int nmatches = matcher.SearchByNeuralPair(mpReferenceKF, mCurrentFrame,
	                                           vpMapPointMatches);

	if (mBackendMode == BackendMode::GtsamDynamic ? nmatches == 0 : nmatches < 15) {
		RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
		            "TrackReferenceKeyFrame rejected: frame=%lu dt=%.6f features=%d "
		            "descriptors=%dx%d type=%d bow_matches=%d reference_kf=%lu",
		            mCurrentFrame.mnId,
		            mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp,
		            mCurrentFrame.N,
		            mCurrentFrame.mDescriptors.rows,
		            mCurrentFrame.mDescriptors.cols,
		            mCurrentFrame.mDescriptors.type(),
		            nmatches,
		            mpReferenceKF ? mpReferenceKF->mnId : 0);
		return false;
	}

	mCurrentFrame.mvpMapPoints = vpMapPointMatches;
	// just use the pose of last frame as the initial data for optimizetion
	mCurrentFrame.SetPose(mLastFrame.mTcw);

	//mCurrentFrame.PrintPointDistribution();


	// cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
	const int poseInliers = OptimizeCurrentFrameWithBackend();

	// Discard outliers
	int nmatchesMap = 0;
	for (int i = 0; i < mCurrentFrame.N; i++) {
		//if(i >= mCurrentFrame.Nleft) break;
		if (mCurrentFrame.mvpMapPoints[i]) {
			if (mCurrentFrame.mvbOutlier[i]) {
				MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

				mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				mCurrentFrame.mvbOutlier[i] = false;
				if (i < mCurrentFrame.Nleft) {
					pMP->mbTrackInView = false;
				}
				else {
					pMP->mbTrackInViewR = false;
				}
				pMP->mbTrackInView = false;
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
				nmatches--;
			}
			else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
				nmatchesMap++;
			}
		}
	}

	if (nmatchesMap < 10) {
		RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
		            "TrackReferenceKeyFrame weak result: frame=%lu dt=%.6f "
		            "bow_matches=%d pose_inliers=%d map_matches=%d reference_kf=%lu",
		            mCurrentFrame.mnId,
		            mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp,
		            nmatches,
		            poseInliers,
		            nmatchesMap,
		            mpReferenceKF ? mpReferenceKF->mnId : 0);
	}

	if (mBackendMode == BackendMode::GtsamDynamic)
		return poseInliers > 0;

	// TODO check these conditions
	if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
		return true;
	}
	else {
		return nmatchesMap >= 10;
	}
}

bool Tracking::TrackWithVisualAndEKF()
{
	// Match the current frame to its reference keyframe before pose refinement.
	FeatureMatcher matcher(*mpFeatureFrontend, 0.7F);
	vector<MapPoint *> vpMapPointMatches;

	int nmatches = matcher.SearchByNeuralPair(mpReferenceKF, mCurrentFrame,
	                                           vpMapPointMatches);

	if (nmatches < 15) {
//        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
		return false;
	}

	mCurrentFrame.mvpMapPoints = vpMapPointMatches;

	// first use PNP to get a initial data for optimization
	// vector<bool> inliers;
	// int inliners_num;
	// bool bad_result;
	// MLPnPsolver* pnpSolver=new MLPnPsolver(mCurrentFrame,vpMapPointMatches);
	// pnpSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
	// cv::Mat initialT_cw=pnpSolver->iterate(5,bad_result,inliers,inliners_num);

	// if(bad_result)
	//     return false;

	// just use the pose of last frame as the initial data for optimizetion
	mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

	//mCurrentFrame.PrintPointDistribution();


	// cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
	if (mBackendMode == BackendMode::GtsamDynamic)
		OptimizeCurrentFrameWithBackend();
	#ifndef AQUA_HAS_GTSAM_DYNAMIC
	else
		Optimizer::PoseOptimizationWithBA_and_EKF(&mCurrentFrame, &mLastFrame, mlamda_visual, mlamda_DVL);
	#endif

	// Discard outliers
	int nmatchesMap = 0;
	for (int i = 0; i < mCurrentFrame.N; i++) {
		//if(i >= mCurrentFrame.Nleft) break;
		if (mCurrentFrame.mvpMapPoints[i]) {
			if (mCurrentFrame.mvbOutlier[i]) {
				MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

				mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				mCurrentFrame.mvbOutlier[i] = false;
				if (i < mCurrentFrame.Nleft) {
					pMP->mbTrackInView = false;
				}
				else {
					pMP->mbTrackInViewR = false;
				}
				pMP->mbTrackInView = false;
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
				nmatches--;
			}
			else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
				nmatchesMap++;
			}
		}
	}

	// TODO check these conditions
	if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
		return true;
	}
	else {
		return nmatchesMap >= 10;
	}
}

void Tracking::UpdateLastFrame()
{
	// Update pose according to reference keyframe
    if(mLastFrame.mpReferenceKF){
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();
        mLastFrame.SetPose(Tlr * pRef->GetPose());
    }


	if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR
		|| !mbOnlyTracking) {
		return;
	}

	// Create "visual odometry" MapPoints
	// We sort points according to their measured depth by the stereo/RGB-D sensor
	vector<pair<float, int> > vDepthIdx;
	vDepthIdx.reserve(mLastFrame.N);
	for (int i = 0; i < mLastFrame.N; i++) {
		float z = mLastFrame.mvDepth[i];
		if (z > 0) {
			vDepthIdx.push_back(make_pair(z, i));
		}
	}

	if (vDepthIdx.empty()) {
		return;
	}

	sort(vDepthIdx.begin(), vDepthIdx.end());

	// We insert all close points (depth<mThDepth)
	// If less than 100 close points, we insert the 100 closest ones.
	int nPoints = 0;
	for (size_t j = 0; j < vDepthIdx.size(); j++) {
		int i = vDepthIdx[j].second;

		bool bCreateNew = false;

		MapPoint *pMP = mLastFrame.mvpMapPoints[i];
		if (!pMP) {
			bCreateNew = true;
		}
		else if (pMP->Observations() < 1) {
			bCreateNew = true;
		}

		if (bCreateNew) {
			cv::Mat x3D = mLastFrame.UnprojectStereo(i);
			MapPoint *pNewMP = new MapPoint(x3D, mpAtlas->GetCurrentMap(), &mLastFrame, i);

			mLastFrame.mvpMapPoints[i] = pNewMP;

			mlpTemporalPoints.push_back(pNewMP);
			nPoints++;
		}
		else {
			nPoints++;
		}

		if (vDepthIdx[j].first > mThDepth && nPoints > 100) {
			break;
		}
	}
}

void Tracking::saveMatchingResults(const cv::Mat &velocity_orb, const cv::Mat &velocity_dvl)
{
	FeatureMatcher matcher(*mpFeatureFrontend, 0.9F);

	// Update last frame pose according to its reference keyframe
	// Create "visual odometry" points if in Localization Mode
	UpdateLastFrame();

	Frame curFrame_orb = mCurrentFrame;
	Frame curFrame_ekf = mCurrentFrame;
	Frame curFrame_gt = mCurrentFrame;
	Frame lastFrame = mLastFrame;

	Eigen::Isometry3d T_g0_gi = mLastFrame.mT_g0_gj;
	Eigen::Isometry3d T_g0_gj = mCurrentFrame.mT_g0_gj;
	Eigen::Isometry3d T_ei_ej_gt = mT_g_e.inverse() * T_g0_gi.inverse() * T_g0_gj * mT_g_e;
	Eigen::Isometry3d velocity_gt_eigen = mT_e_c.inverse() * T_ei_ej_gt.inverse() * mT_e_c;

	cv::Mat v_gt(4, 4, CV_32FC1);
	cv::eigen2cv(velocity_gt_eigen.matrix(), v_gt);
	v_gt.convertTo(v_gt, CV_32FC1);

	cv::Mat v_orb = velocity_orb.clone();
	cv::Mat v_dvl = velocity_dvl.clone();
	v_orb.convertTo(v_orb, CV_32FC1);
	v_dvl.convertTo(v_dvl, CV_32FC1);

	Eigen::Isometry3d velocity_dvl_eigen = Eigen::Isometry3d::Identity();
	Eigen::Isometry3d velocity_orb_eigen = Eigen::Isometry3d::Identity();
	cv::cv2eigen(v_orb, velocity_orb_eigen.matrix());
	cv::cv2eigen(v_dvl, velocity_dvl_eigen.matrix());

	fstream file_v;
	file_v.open("/home/da/project/ORB_SLAM3_edit/matching_result/velocity_dvl.txt", ios::out | ios::app);
	file_v << fixed;
	if (!file_v) {
		cout << "fail to create file" << endl;
		return;
	}
	Eigen::Vector3d euler_dvl = velocity_dvl_eigen.rotation().eulerAngles(0, 1, 2);
	file_v << setprecision(6) <<
	       velocity_dvl_eigen.translation().x() << " " << velocity_dvl_eigen.translation().y() << " "
	       << velocity_dvl_eigen.translation().z() << " "
	       << euler_dvl.x() << " " << euler_dvl.y() << " " << euler_dvl.z() << endl;
	file_v.close();

	file_v.open("/home/da/project/ORB_SLAM3_edit/matching_result/velocity_orb.txt", ios::out | ios::app);
	file_v << fixed;
	if (!file_v) {
		cout << "fail to create file" << endl;
		return;
	}
	Eigen::Vector3d euler_orb = velocity_orb_eigen.rotation().eulerAngles(0, 1, 2);
	file_v << setprecision(6) <<
	       velocity_orb_eigen.translation().x() << " " << velocity_orb_eigen.translation().y() << " "
	       << velocity_orb_eigen.translation().z() << " "
	       << euler_orb.x() << " " << euler_orb.y() << " " << euler_orb.z() << endl;
	file_v.close();

	file_v.open("/home/da/project/ORB_SLAM3_edit/matching_result/velocity_gt.txt", ios::out | ios::app);
	file_v << fixed;
	if (!file_v) {
		cout << "fail to create file" << endl;
		return;
	}
	Eigen::Vector3d euler_gt = velocity_gt_eigen.rotation().eulerAngles(0, 1, 2);
	file_v << setprecision(6) <<
	       velocity_gt_eigen.translation().x() << " " << velocity_gt_eigen.translation().y() << " "
	       << velocity_gt_eigen.translation().z() << " "
	       << euler_gt.x() << " " << euler_gt.y() << " " << euler_gt.z() << endl;
	file_v.close();


	curFrame_orb.SetPose(v_orb * lastFrame.mTcw);
	curFrame_ekf.SetPose(v_dvl * lastFrame.mTcw);
	curFrame_gt.SetPose(v_gt * lastFrame.mTcw);


	fill(curFrame_orb.mvpMapPoints.begin(), curFrame_orb.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
	fill(curFrame_ekf.mvpMapPoints.begin(), curFrame_ekf.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

	// Project points seen in previous frame
	int th;

	if (mSensor == System::STEREO || mSensor == System::DVL_STEREO) {
		th = 7;
	}
	else {
		th = 15;
	}

	// write matching results to a file, for comparasion with original
	fstream file_orb;
	file_orb << fixed;
	file_orb
		.open("/home/da/project/ORB_SLAM3_edit/matching_result/matching_results_withoutDVL.txt", ios::out | ios::app);
	if (!file_orb) {
		cout << "fail to create file" << endl;
		return;
	}
	int nmatches = matcher.SearchByProjection(curFrame_orb,
	                                          lastFrame,
	                                          th,
	                                          mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
	file_orb << setprecision(6) << mCurrentFrame.mTimeStamp << " " << mLastFrame.mTimeStamp << " " << mCurrentFrame.N
	         << " " << nmatches << " ";

	// If few matches, uses a wider window search
	if (nmatches < 20) {
		fill(curFrame_orb.mvpMapPoints.begin(), curFrame_orb.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

		nmatches = matcher.SearchByProjection(curFrame_orb,
		                                      lastFrame,
		                                      2 * th,
		                                      mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
		file_orb << nmatches << " ";
	}
	else {
		file_orb << "0" << " ";
	}

	// Optimize frame pose with all matches
	#ifndef AQUA_HAS_GTSAM_DYNAMIC
	Optimizer::PoseOptimization(&curFrame_orb);
	#endif

	// Discard outliers
	int nmatchesMap = 0;
	for (int i = 0; i < curFrame_orb.N; i++) {
		if (curFrame_orb.mvpMapPoints[i]) {
			if (curFrame_orb.mvbOutlier[i]) {
				MapPoint *pMP = curFrame_orb.mvpMapPoints[i];

				curFrame_orb.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				curFrame_orb.mvbOutlier[i] = false;
				if (i < curFrame_orb.Nleft) {
					pMP->mbTrackInView = false;
				}
				else {
					pMP->mbTrackInViewR = false;
				}
				pMP->mnLastFrameSeen = curFrame_orb.mnId;
				nmatches--;
			}
			else if (curFrame_orb.mvpMapPoints[i]->Observations() > 0) {
				nmatchesMap++;
			}
		}
	}
	file_orb << nmatches << " " << nmatchesMap << " " << mLastFrame.mGood_EKF;

	file_orb << '\n';
	file_orb.close();

	fstream file_ekf;
	file_ekf << fixed;
	file_ekf.open("/home/da/project/ORB_SLAM3_edit/matching_result/matching_results_withDVL.txt", ios::out | ios::app);
	if (!file_ekf) {
		cout << "fail to create file" << endl;
		return;
	}
	nmatches = matcher.SearchByProjection(curFrame_ekf,
	                                      lastFrame,
	                                      th,
	                                      mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
	file_ekf << setprecision(6) << mCurrentFrame.mTimeStamp << " " << mLastFrame.mTimeStamp << " " << mCurrentFrame.N
	         << " " << nmatches << " ";

	// If few matches, uses a wider window search
	if (nmatches < 20) {
		fill(curFrame_orb.mvpMapPoints.begin(), curFrame_orb.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

		nmatches = matcher.SearchByProjection(curFrame_ekf,
		                                      lastFrame,
		                                      2 * th,
		                                      mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
		file_ekf << nmatches << " ";
	}
	else {
		file_ekf << "0" << " ";
	}

	// Optimize frame pose with all matches
	#ifndef AQUA_HAS_GTSAM_DYNAMIC
	Optimizer::PoseOptimizationWithBA_and_EKF(&curFrame_ekf, &lastFrame, mlamda_visual, mlamda_DVL);
	#endif

	// Discard outliers
	nmatchesMap = 0;
	for (int i = 0; i < curFrame_ekf.N; i++) {
		if (curFrame_ekf.mvpMapPoints[i]) {
			if (curFrame_ekf.mvbOutlier[i]) {
				MapPoint *pMP = curFrame_ekf.mvpMapPoints[i];

				curFrame_ekf.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				curFrame_ekf.mvbOutlier[i] = false;
				if (i < curFrame_ekf.Nleft) {
					pMP->mbTrackInView = false;
				}
				else {
					pMP->mbTrackInViewR = false;
				}
				pMP->mnLastFrameSeen = curFrame_ekf.mnId;
				nmatches--;
			}
			else if (curFrame_ekf.mvpMapPoints[i]->Observations() > 0) {
				nmatchesMap++;
			}
		}
	}
	file_ekf << nmatches << " " << nmatchesMap << " " << mLastFrame.mGood_EKF;
	file_ekf << '\n';
	file_ekf.close();

	fstream file_gt;
	file_gt << fixed;
	file_gt.open("/home/da/project/ORB_SLAM3_edit/matching_result/matching_results_gt.txt", ios::out | ios::app);
	if (!file_gt) {
		cout << "fail to create file" << endl;
		return;
	}
	nmatches = matcher.SearchByProjection(curFrame_gt,
	                                      lastFrame,
	                                      th,
	                                      mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
	file_gt << setprecision(6) << mCurrentFrame.mTimeStamp << " " << mLastFrame.mTimeStamp << " " << mCurrentFrame.N
	        << " " << nmatches << " ";

	// If few matches, uses a wider window search
	if (nmatches < 20) {
		fill(curFrame_orb.mvpMapPoints.begin(), curFrame_orb.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

		nmatches = matcher.SearchByProjection(curFrame_gt,
		                                      lastFrame,
		                                      2 * th,
		                                      mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
		file_gt << nmatches << " ";
	}
	else {
		file_gt << "0" << " ";
	}

	// Optimize frame pose with all matches
//	Optimizer::PoseOptimizationWithBA_and_EKF(&curFrame_gt,&lastFrame,mlamda_visual,mlamda_DVL);
	#ifndef AQUA_HAS_GTSAM_DYNAMIC
	Optimizer::PoseOptimization(&curFrame_gt);
	#endif

	// Discard outliers
	nmatchesMap = 0;
	for (int i = 0; i < curFrame_gt.N; i++) {
		if (curFrame_gt.mvpMapPoints[i]) {
			if (curFrame_gt.mvbOutlier[i]) {
				MapPoint *pMP = curFrame_gt.mvpMapPoints[i];

				curFrame_gt.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				curFrame_gt.mvbOutlier[i] = false;
				if (i < curFrame_gt.Nleft) {
					pMP->mbTrackInView = false;
				}
				else {
					pMP->mbTrackInViewR = false;
				}
				pMP->mnLastFrameSeen = curFrame_gt.mnId;
				nmatches--;
			}
			else if (curFrame_gt.mvpMapPoints[i]->Observations() > 0) {
				nmatchesMap++;
			}
		}
	}
	file_gt << nmatches << " " << nmatchesMap << " " << mLastFrame.mGood_EKF;
	file_gt << '\n';
	file_gt.close();
}

int Tracking::MatchTemporalMapPoints()
{
	FeatureSet previous{mLastFrame.mvKeysUn, mLastFrame.mDescriptors};
	FeatureSet current{mCurrentFrame.mvKeysUn, mCurrentFrame.mDescriptors};
	const auto matches = mpFeatureFrontend->Match(
		previous, mLastFrame.imgLeft.size(), current, mCurrentFrame.imgLeft.size());
	const auto inliers = NeuralFeatureFrontend::FilterFundamentalInliers(
		previous, current, matches, 3.0, 0.99);
	return mCurrentFrame.TransferTemporalMapPoints(mLastFrame, inliers);
}

bool Tracking::TrackWithMotionModel()
{
	FeatureMatcher matcher(*mpFeatureFrontend, 0.9F);

	// Update last frame pose according to its reference keyframe
	// Create "visual odometry" points if in Localization Mode
	UpdateLastFrame();



	// if IMU is ready, use IMU pridict Pose of current frame
	if (mSensor == System::DVL_STEREO && mCalibrated && mpAtlas->GetCurrentMap()->isImuInitialized()) {
		// PredictStateDvlGro();
//		return true;
		mVelocity.convertTo(mVelocity, CV_32FC1);
		// mVelocity: T_cj_ci
		// mLastFrame.mTcw: T_ci_c0
			cv::Mat predictedPose = mVelocity * mLastFrame.mTcw;
			if (mBackendMode == BackendMode::GtsamDynamic &&
				!ProjectMotionModelPoseToSE3(predictedPose, predictedPose)) {
				RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"),
				             "invalid dynamic motion-model prediction: frame=%lu "
				             "timestamp=%.9f velocity_finite=%d last_finite=%d "
			             "velocity_type=%d last_type=%d",
			             mCurrentFrame.mnId, mCurrentFrame.mTimeStamp,
				             cv::checkRange(mVelocity), cv::checkRange(mLastFrame.mTcw),
				             mVelocity.type(), mLastFrame.mTcw.type());
				mVelocity.release();
				return false;
			}
		mCurrentFrame.SetPose(predictedPose);
	}
	else {
		// string v_type = getImageType(mVelocity.type());
		// string t_type = getImageType(mLastFrame.mTcw.type());
		mVelocity.convertTo(mVelocity, CV_32FC1);
		// mVelocity: T_cj_ci
		// mLastFrame.mTcw: T_ci_c0
		cv::Mat predictedPose = mVelocity * mLastFrame.mTcw;
		if (mBackendMode == BackendMode::GtsamDynamic &&
			!ProjectMotionModelPoseToSE3(predictedPose, predictedPose)) {
			RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"),
			             "invalid dynamic motion-model prediction: frame=%lu "
			             "timestamp=%.9f velocity_finite=%d last_finite=%d "
			             "velocity_type=%d last_type=%d",
			             mCurrentFrame.mnId, mCurrentFrame.mTimeStamp,
			             cv::checkRange(mVelocity), cv::checkRange(mLastFrame.mTcw),
			             mVelocity.type(), mLastFrame.mTcw.type());
			mVelocity.release();
			return false;
		}
		if (mBackendMode != BackendMode::GtsamDynamic &&
			!cv::checkRange(predictedPose)) {
			RCLCPP_ERROR(rclcpp::get_logger("aqua_slam"),
			             "motion-model prediction became non-finite: frame=%lu "
			             "timestamp=%.9f velocity_finite=%d last_finite=%d "
			             "velocity_type=%d last_type=%d",
			             mCurrentFrame.mnId, mCurrentFrame.mTimeStamp,
			             cv::checkRange(mVelocity), cv::checkRange(mLastFrame.mTcw),
			             mVelocity.type(), mLastFrame.mTcw.type());
		}
		mCurrentFrame.SetPose(predictedPose);
//		PredictStateDvlGro();
	}


	fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
	int nmatches = MatchTemporalMapPoints();

	// Project points seen in previous frame
	int th;

	if (mSensor == System::STEREO || mSensor == System::DVL_STEREO) {
		th = 7;
	}
	else {
		th = 15;
	}

	nmatches += matcher.SearchByProjection(mCurrentFrame,
	                                       mLastFrame,
	                                       th,
	                                       mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);

	// If few matches, uses a wider window search
	if (nmatches < 20) {
		Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
		nmatches += matcher.SearchByProjection(mCurrentFrame,
		                                       mLastFrame,
		                                       2 * th,
		                                       mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
		Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

	}

	if (mBackendMode != BackendMode::GtsamDynamic && nmatches < 20) {
		Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
		if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
			return true;
		}
		else {
			return false;
		}
	}

	// Optimize frame pose with all matches
	const int poseSupport = OptimizeCurrentFrameWithBackend();

	// Discard outliers
	int nmatchesMap = 0;
	for (int i = 0; i < mCurrentFrame.N; i++) {
		if (mCurrentFrame.mvpMapPoints[i]) {
			if (mCurrentFrame.mvbOutlier[i]) {
				MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

				mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				mCurrentFrame.mvbOutlier[i] = false;
				if (i < mCurrentFrame.Nleft) {
					pMP->mbTrackInView = false;
				}
				else {
					pMP->mbTrackInViewR = false;
				}
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
				nmatches--;
			}
			else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
				nmatchesMap++;
			}
		}
	}

	if (mBackendMode == BackendMode::GtsamDynamic)
		return poseSupport > 0;

	if (mbOnlyTracking) {
		mbVO = nmatchesMap < 10;
		return nmatches > 20;
	}

	if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
		return true;
	}
	else {
		return nmatchesMap >= 10;
	}
}

bool Tracking::TrackWithMotionModelAndEKF()
{
	FeatureMatcher matcher(*mpFeatureFrontend, 0.9F);

	// Update last frame pose according to its reference keyframe
	// Create "visual odometry" points if in Localization Mode
	UpdateLastFrame();



	// if IMU is ready, use IMU pridict Pose of current frame
	if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU)) {
		// Predict ste with IMU if it is initialized and it doesnt need reset
		// calibrate pose of current frame by IMU, velocity could be calcuted further by pose
		if (!PredictStateIMU())
			return false;
	}
	else {
		string v_type = getImageType(mVelocity.type());
		string t_type = getImageType(mLastFrame.mTcw.type());
		mVelocity.convertTo(mVelocity, CV_32FC1);
		// mVelocity: T_cj_ci
		// mLastFrame.mTcw: T_ci_c0
		mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
	}


	fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
	int nmatches = MatchTemporalMapPoints();

	// Project points seen in previous frame
	int th;

	if (mSensor == System::STEREO || mSensor == System::DVL_STEREO) {
		th = 7;
	}
	else {
		th = 15;
	}

	nmatches += matcher.SearchByProjection(mCurrentFrame,
	                                       mLastFrame,
	                                       th,
	                                       mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);

	// If few matches, uses a wider window search
	if (nmatches < 10) {
		Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
		nmatches += matcher.SearchByProjection(mCurrentFrame,
		                                       mLastFrame,
		                                       2 * th,
		                                       mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
		Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

	}

	if (nmatches < 10) {
		Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
		if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
			return true;
		}
		else {
			return false;
		}
	}

	// Optimize frame pose with all matches
	if (mBackendMode == BackendMode::GtsamDynamic)
		OptimizeCurrentFrameWithBackend();
	#ifndef AQUA_HAS_GTSAM_DYNAMIC
	else
		Optimizer::PoseOptimizationWithBA_and_EKF(&mCurrentFrame, &mLastFrame, mlamda_visual, mlamda_DVL);
	#endif

	// Discard outliers
	int nmatchesMap = 0;
	for (int i = 0; i < mCurrentFrame.N; i++) {
		if (mCurrentFrame.mvpMapPoints[i]) {
			if (mCurrentFrame.mvbOutlier[i]) {
				MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

				mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				mCurrentFrame.mvbOutlier[i] = false;
				if (i < mCurrentFrame.Nleft) {
					pMP->mbTrackInView = false;
				}
				else {
					pMP->mbTrackInViewR = false;
				}
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
				nmatches--;
			}
			else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
				nmatchesMap++;
			}
		}
	}

	if (mbOnlyTracking) {
		mbVO = nmatchesMap < 10;
		return nmatches > 20;
	}

	if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
		return true;
	}
	else if (mSensor == System::DVL_STEREO) {
		return nmatchesMap >= 3;
	}
}

void Tracking::SaveOptimizationResult()
{
	FeatureMatcher matcher(*mpFeatureFrontend, 0.9F);

	// Update last frame pose according to its reference keyframe
	// Create "visual odometry" points if in Localization Mode
	UpdateLastFrame();



	// if IMU is ready, use IMU pridict Pose of current frame
	if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU)) {
		// Predict ste with IMU if it is initialized and it doesnt need reset
		// calibrate pose of current frame by IMU, velocity could be calcuted further by pose
		PredictStateIMU();
		return;
	}
	else {
		string v_type = getImageType(mVelocity.type());
		string t_type = getImageType(mLastFrame.mTcw.type());
		mVelocity.convertTo(mVelocity, CV_32FC1);
		// mVelocity: T_cj_ci
		// mLastFrame.mTcw: T_ci_c0
		mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
	}


	fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

	// Project points seen in previous frame
	int th;

	if (mSensor == System::STEREO || mSensor == System::DVL_STEREO) {
		th = 7;
	}
	else {
		th = 15;
	}

	int nmatches = matcher.SearchByProjection(mCurrentFrame,
	                                          mLastFrame,
	                                          th,
	                                          mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);


	cout << "frame id:" << mCurrentFrame.mnId << endl;
	Frame f_orb(mCurrentFrame);
	Frame f_ekf(mCurrentFrame);
	Frame f_orb_ekf(mCurrentFrame);
	// Optimize frame pose with all matches
	#ifndef AQUA_HAS_GTSAM_DYNAMIC
	Optimizer::PoseOptimization(&f_orb);
	if (mDVL_func_debug == 1) {
		Optimizer::PoseOptimizationWithBA_and_EKF(&f_orb_ekf, &mLastFrame, mlamda_visual, mlamda_DVL_debug);
	}
	else {
		Optimizer::PoseOptimizationWithBA_and_EKF2(&f_orb_ekf, &mLastFrame, mlamda_visual, mlamda_DVL_debug);
	}
	Optimizer::PoseOptimizationWithEKF(&f_ekf, &mLastFrame);
	#endif
//	T_e0_ej_gt = T_e_c * T_c0_ci * T_c_e * T_ei_ej_gt
	Eigen::Isometry3d T_gt = Eigen::Isometry3d::Identity();
	Eigen::Isometry3d T_ei_ej_gt = Eigen::Isometry3d::Identity();
	T_ei_ej_gt = mT_g_e.inverse() * mLastFrame.mT_g0_gj.inverse() * mCurrentFrame.mT_g0_gj * mT_g_e;
//	T_ei_ej
	Eigen::Isometry3d T_orb = Eigen::Isometry3d::Identity();
//	T_ei_ej
	Eigen::Isometry3d T_orb_ekf = Eigen::Isometry3d::Identity();
//	T_ei_ej = T_e_c * T_ci_c0 * T_c0_cj * T_c_e
	Eigen::Isometry3d T_ekf = Eigen::Isometry3d::Identity();
//	T_ei_ej=T_e_c * T_c0_ci * T_c_e * T_ei_ej
	Eigen::Isometry3d T_ekf_ori = Eigen::Isometry3d::Identity();
	Eigen::Isometry3d T_ci_c0 = Eigen::Isometry3d::Identity();
	cv::cv2eigen(mLastFrame.mTcw, T_ci_c0.matrix());
//	T_ci_c0=T_ci_c0.inverse();
	Eigen::Isometry3d T_ei_ej = mLastFrame.mT_e0_ej.inverse() * mCurrentFrame.mT_e0_ej;
	T_ekf_ori = T_ei_ej;
	T_gt = T_ei_ej_gt;

	// T_cj_c0
	Eigen::Isometry3d T_c0_cj_orb = Eigen::Isometry3d::Identity();
	Eigen::Isometry3d T_c0_cj_orb_ekf = Eigen::Isometry3d::Identity();
	Eigen::Isometry3d T_c0_cj_ekf = Eigen::Isometry3d::Identity();
	cv::cv2eigen(f_orb.mTcw, T_c0_cj_orb.matrix());
	cv::cv2eigen(f_orb_ekf.mTcw, T_c0_cj_orb_ekf.matrix());
	cv::cv2eigen(f_ekf.mTcw, T_c0_cj_ekf.matrix());
	T_c0_cj_orb = T_c0_cj_orb.inverse();
	T_c0_cj_orb_ekf = T_c0_cj_orb_ekf.inverse();
	T_c0_cj_ekf = T_c0_cj_ekf.inverse();

	T_orb = mT_e_c * T_ci_c0 * T_c0_cj_orb * mT_e_c.inverse();
	T_orb_ekf = mT_e_c * T_ci_c0 * T_c0_cj_orb_ekf * mT_e_c.inverse();
	T_ekf = mT_e_c * T_ci_c0 * T_c0_cj_ekf * mT_e_c.inverse();
	fstream file;
	file.open("data/Optmization_results.txt", ios::out | ios::app);
	if (!file) {
		cout << "fail to open data/Optmization_results.txt" << endl;
	}
	file << mCurrentFrame.mnId << " " << T_gt.translation().transpose() << " " << T_orb.translation().transpose() << " "
	     << T_orb_ekf.translation().transpose() << " " << T_ekf.translation().transpose()
	     << " " << T_ekf_ori.translation().transpose() << " " << T_gt.rotation().eulerAngles(0, 1, 2).transpose() << " "
	     << T_orb.rotation().eulerAngles(0, 1, 2).transpose() << " "
	     << T_orb_ekf.rotation().eulerAngles(0, 1, 2).transpose()
	     << " " << T_ekf.rotation().eulerAngles(0, 1, 2).transpose() << " "
	     << T_ekf_ori.rotation().eulerAngles(0, 1, 2).transpose() << endl;
	file.close();

}

void Tracking::drawOptimizationResult()
{
	// add all keyframes share the map points in current frame and their neighbors to mvpLocalKeyFrames
	// add all map points in mvpLocalKeyFrames to mvpLocalMapPoints
	UpdateLocalMap();
	// add more map points in local map to mCurrentFrame.mvpMapPoints
	SearchLocalPoints();

	// TOO check outliers before PO
	int aux1 = 0, aux2 = 0;
	for (int i = 0; i < mCurrentFrame.N; i++) {
		if (mCurrentFrame.mvpMapPoints[i]) {
			aux1++;
			if (mCurrentFrame.mvbOutlier[i]) {
				aux2++;
			}
		}
	}

	Frame f_dvl_gyro(mCurrentFrame);
	Frame f_orb(mCurrentFrame);

	#ifndef AQUA_HAS_GTSAM_DYNAMIC
	Optimizer::PoseOptimization(&f_orb);

	Optimizer::PoseDvlGyrosOPtimizationLastFrame(&f_dvl_gyro, mlamda_DVL);
	#endif

	cv::Mat img_dvl_gyro(mCurrentFrame.imgLeft.rows, mCurrentFrame.imgLeft.cols, CV_8UC1);
	cv::Mat img_orb(mCurrentFrame.imgLeft.rows, mCurrentFrame.imgLeft.cols, CV_8UC1);


	mCurrentFrame.imgLeft.copyTo(img_dvl_gyro);
	mCurrentFrame.imgLeft.copyTo(img_orb);

	if (mCurrentFrame.imgLeft.channels() < 3) {
		// cv::cvtColor(img_dvl_gyro, img_dvl_gyro, CV_GRAY2BGR);  // original
		// cv::cvtColor(img_orb, img_orb, CV_GRAY2BGR);  // original
		cv::cvtColor(img_dvl_gyro, img_dvl_gyro, cv::COLOR_GRAY2BGR);
		cv::cvtColor(img_orb, img_orb, cv::COLOR_GRAY2BGR);
	}

	int inlier_dvl_gyro = 0;
	int inlier_orb = 0;
	int outlier_dvl_gyro = 0;
	int outlier_orb = 0;
	for (int i = 0; i < mCurrentFrame.N; i++) {
		if (f_dvl_gyro.mvpMapPoints[i]) {
			if (!f_dvl_gyro.mvbOutlier[i]) {
				cv::circle(img_dvl_gyro, f_dvl_gyro.mvKeys[i].pt, 2, cv::Scalar(0, 255, 0));
				inlier_dvl_gyro++;
			}
			else {
				cv::circle(img_dvl_gyro, f_dvl_gyro.mvKeys[i].pt, 2, cv::Scalar(255, 0, 0));
				outlier_dvl_gyro++;
			}

			if (!f_orb.mvbOutlier[i]) {
				cv::circle(img_orb, f_dvl_gyro.mvKeys[i].pt, 2, cv::Scalar(0, 255, 0));
				inlier_orb++;
			}
			else {
				cv::circle(img_orb, f_dvl_gyro.mvKeys[i].pt, 2, cv::Scalar(255, 0, 0));
				outlier_orb++;
			}
		}
	}

	stringstream ss_dvl_gyro;
	stringstream ss_orb;
	ss_dvl_gyro << "dvl + guro optimization | " << "inlier: " << inlier_dvl_gyro << " | " << "outlier: "
	            << outlier_dvl_gyro;
	ss_orb << "orb optimization | " << "inlier: " << inlier_orb << " | " << "outlier: "
	       << outlier_orb;

	int baseline = 0;
	cv::Size textSize = cv::getTextSize(ss_dvl_gyro.str(), cv::FONT_HERSHEY_PLAIN, 2, 1, &baseline);
	cv::Mat img_dvl_gyro_withinfo(img_dvl_gyro.rows + textSize.height + 10, img_dvl_gyro.cols, img_dvl_gyro.type());
	img_dvl_gyro.copyTo(img_dvl_gyro_withinfo.rowRange(0, img_dvl_gyro.rows).colRange(0, img_dvl_gyro.cols));
	img_dvl_gyro_withinfo.rowRange(img_dvl_gyro.rows, img_dvl_gyro_withinfo.rows) =
		cv::Mat::zeros(textSize.height + 10, img_dvl_gyro_withinfo.cols, img_dvl_gyro.type());
	cv::putText(img_dvl_gyro_withinfo,
	            ss_dvl_gyro.str(),
	            cv::Point(5, img_dvl_gyro_withinfo.rows - 5),
	            cv::FONT_HERSHEY_PLAIN,
	            1,
	            cv::Scalar(255, 255, 255),
	            1,
	            8);

	baseline = 0;
	textSize = cv::getTextSize(ss_orb.str(), cv::FONT_HERSHEY_PLAIN, 2, 1, &baseline);
	cv::Mat img_orb_withinfo(img_orb.rows + textSize.height + 10, img_orb.cols, img_orb.type());
	img_orb.copyTo(img_orb_withinfo.rowRange(0, img_orb.rows).colRange(0, img_orb.cols));
	img_orb_withinfo.rowRange(img_orb.rows, img_orb_withinfo.rows) =
		cv::Mat::zeros(textSize.height + 10, img_orb_withinfo.cols, img_orb.type());
	cv::putText(img_orb_withinfo,
	            ss_orb.str(),
	            cv::Point(5, img_orb_withinfo.rows - 5),
	            cv::FONT_HERSHEY_PLAIN,
	            1,
	            cv::Scalar(255, 255, 255),
	            1,
	            8);
	cv::Mat img_with_debuginfo
		(img_dvl_gyro_withinfo.rows, img_dvl_gyro_withinfo.cols + img_orb_withinfo.cols, img_dvl_gyro_withinfo.type());

	img_dvl_gyro_withinfo
		.copyTo(img_with_debuginfo.rowRange(0, img_with_debuginfo.rows).colRange(0, img_dvl_gyro_withinfo.cols));
	img_orb_withinfo.copyTo(img_with_debuginfo.rowRange(0, img_with_debuginfo.rows).colRange(img_dvl_gyro_withinfo.cols,
	                                                                                         img_dvl_gyro_withinfo.cols
		                                                                                         + img_orb_withinfo
			                                                                                         .cols));
	std_msgs::msg::Header header; // empty header
// // 	header.stamp = ros::Time::now(); // time  // original  // original
//	cv::Mat img_with_info=mpFrameDrawer->DrawFrame(true);

	if (img_with_debuginfo.channels() < 3) {
		// cv::cvtColor(img_with_debuginfo, img_with_debuginfo, CV_GRAY2BGR);  // original
		cv::cvtColor(img_with_debuginfo, img_with_debuginfo, cv::COLOR_GRAY2BGR);
	}
	cv_bridge::CvImage img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, img_with_debuginfo);
	mpRosHandler->PublishImgWithInfo(img_bridge.toImageMsg());

}

void Tracking::IntegrateDVLVelocity()
{
	double delta_time = mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp;
	Eigen::Vector3d v_t(mLastFrame.mV_e(0, 0), mLastFrame.mV_e(1, 0), mLastFrame.mV_e(2, 0));
	Eigen::Vector3d v_angular(mLastFrame.mV_e(3, 0), mLastFrame.mV_e(4, 0), mLastFrame.mV_e(5, 0));
	Eigen::Vector3d t = delta_time * v_t;
	double z_ekf = mCurrentFrame.mT_e0_ej.translation().z() - mLastFrame.mT_e0_ej.translation().z();
	t(2, 0) = z_ekf;
	Eigen::Vector3d eular = delta_time * v_angular;
//	cout<<"velocity: "<<t.matrix()<<endl;
//	cout<<"angular velocity: "<<eular.matrix()<<endl;
	Eigen::Matrix3d r = (Eigen::AngleAxisd(eular(0, 0), Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(eular(1, 0),
	                                                                                                  Eigen::Vector3d::UnitY())
		* Eigen::AngleAxisd(eular(2, 0), Eigen::Vector3d::UnitZ())).toRotationMatrix();

//	cv::Mat v_ori=mVelocity.clone();
//	Eigen::Isometry3d T_ci_cj_ekf=Eigen::Isometry3d::Identity();
//	cv::cv2eigen(v_ori,T_ci_cj_ekf.matrix());
//	T_ci_cj_ekf=T_ci_cj_ekf.inverse();
//	Eigen::Vector3d t_ekf=T_ci_cj_ekf.translation();
//	Eigen::Vector3d eular_ekf=T_ci_cj_ekf.rotation().eulerAngles(0, 1, 2);
//	cout << "velocity from EKF frame: " << t_ekf.matrix() << endl;
//	cout<<"angular velocity from EKF frame: "<<eular_ekf.matrix()<<endl;

	Eigen::Isometry3d T_ei_ej = Eigen::Isometry3d::Identity();
	T_ei_ej.rotate(r);
	T_ei_ej.pretranslate(t);
	Eigen::Isometry3d T_cj_ci = mT_e_c.inverse() * T_ei_ej.inverse() * mT_e_c;
	cv::eigen2cv(T_cj_ci.matrix(), mVelocity);

//	Eigen::Isometry3d T_ci_c0=Eigen::Isometry3d::Identity();
//	Eigen::Isometry3d T_cj_c0=Eigen::Isometry3d::Identity();
//	cv::cv2eigen(mLastFrame.mTcw,T_ci_c0.matrix());
//	T_cj_c0=T_cj_ci*T_ci_c0;
	mCurrentFrame.mT_e0_ej = mLastFrame.mT_e0_ej * T_ei_ej;

}

bool Tracking::TrackLocalMap()
{

	// We have an estimation of the camera pose and some map points tracked in the frame.
	// We retrieve the local map and try to find matches to points in the local map.
	mTrackedFr++;

	// add all keyframes share the map points in current frame and their neighbors to mvpLocalKeyFrames
	// add all map points in mvpLocalKeyFrames to mvpLocalMapPoints
	int associationsBeforeLocalMap = 0;
	for (MapPoint *pMP : mCurrentFrame.mvpMapPoints) {
		if (pMP)
			++associationsBeforeLocalMap;
	}
	UpdateLocalMap();
	// add more map points in local map to mCurrentFrame.mvpMapPoints
	SearchLocalPoints();

	// TOO check outliers before PO
	int aux1 = 0, aux2 = 0;
	for (int i = 0; i < mCurrentFrame.N; i++)
		if (mCurrentFrame.mvpMapPoints[i]) {
			aux1++;
			if (mCurrentFrame.mvbOutlier[i]) {
				aux2++;
			}
		}

	// Refine every camera frame; dynamic graph state is committed only for a
	// newly created AQUA keyframe.
	const int poseInliers = OptimizeCurrentFrameWithBackend(false);
//	if (!mpAtlas->isImuInitialized()) {
//		Optimizer::PoseOptimization(&mCurrentFrame);
//	}
//	else {
//		if (mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU) {
//			Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
//			Optimizer::PoseOptimization(&mCurrentFrame);
//		}
//		else {
//			// if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
//			if (!mbMapUpdated) //  && (mnMatchesInliers>30))
//			{
//				Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
//				inliers =
//					Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
//			}
//			else {
//				Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
//				inliers =
//					Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
//			}
//		}
//	}

	aux1 = 0, aux2 = 0;
	for (int i = 0; i < mCurrentFrame.N; i++)
		if (mCurrentFrame.mvpMapPoints[i]) {
			aux1++;
			if (mCurrentFrame.mvbOutlier[i]) {
				aux2++;
			}
		}

	mnMatchesInliers = 0;

	// Update MapPoints Statistics
	for (int i = 0; i < mCurrentFrame.N; i++) {
		if (mCurrentFrame.mvpMapPoints[i]) {
			if (!mCurrentFrame.mvbOutlier[i]) {
				mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
				if (!mbOnlyTracking) {
					if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
						mnMatchesInliers++;
					}
				}
				else {
					mnMatchesInliers++;
				}
			}
			else if (mSensor == System::STEREO || mSensor == System::DVL_STEREO) {
				mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
			}
		}
	}

	// Decide if the tracking was succesful
	// More restrictive if there was a relocalization recently
	mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
	if(mnMatchesInliers < mnFeatureTarget * 0.10){
        mCurrentFrame.mPoorVision = true;
    }
//     // ROS_INFO_STREAM("Matching Inliers: "<<mnMatchesInliers);  // original
	// if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50) {
	// 	return false;
	// }

//	if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST)) {
//		return true;
//	}

    // if(mnMatchesInliers<20){
    //     PredictStateDvlGro();
    // }
	    // Only persistent MapPoints constrain motion across frames. Per-frame
	    // stereo depth without a MapPoint is useful for initialization, but must
	    // not keep tracking alive after persistent map support is lost.
    const int trackingSupport =
        mBackendMode == BackendMode::GtsamDynamic ? poseInliers : mnMatchesInliers;
    if (mBackendMode == BackendMode::GtsamDynamic ? trackingSupport == 0
                                               : trackingSupport < 10) {
		RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
		            "TrackLocalMap rejected: frame=%lu dt=%.6f features=%d "
		            "associations_before=%d associations_after=%d pose_inliers=%d "
			    "map_inliers=%d local_keyframes=%zu local_points=%zu",
		            mCurrentFrame.mnId,
		            mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp,
		            mCurrentFrame.N,
		            associationsBeforeLocalMap,
		            aux1,
		            poseInliers,
		            mnMatchesInliers,
			    mvpLocalKeyFrames.size(),
			    mvpLocalMapPoints.size());
		return false;
    }
    else {
        return true;
    }
}
bool Tracking::TrackLocalMapWithDvlGyro()
{

	// We have an estimation of the camera pose and some map points tracked in the frame.
	// We retrieve the local map and try to find matches to points in the local map.
	mTrackedFr++;

	// add all keyframes share the map points in current frame and their neighbors to mvpLocalKeyFrames
	// add all map points in mvpLocalKeyFrames to mvpLocalMapPoints
	UpdateLocalMap();
	// add more map points in local map to mCurrentFrame.mvpMapPoints
	SearchLocalPoints();

	// TOO check outliers before PO
	int aux1 = 0, aux2 = 0;
	for (int i = 0; i < mCurrentFrame.N; i++)
		if (mCurrentFrame.mvpMapPoints[i]) {
			aux1++;
			if (mCurrentFrame.mvbOutlier[i]) {
				aux2++;
			}
		}

	int inliers;
//	Optimizer::PoseOptimization(&mCurrentFrame);
//

	if (mBackendMode == BackendMode::GtsamDynamic) {
		inliers = OptimizeCurrentFrameWithBackend();
	}
	#ifndef AQUA_HAS_GTSAM_DYNAMIC
	else if (!mbMapUpdated) //  && (mnMatchesInliers>30))
	{
//		cout << "track local map from last Frame" << endl;
		inliers =
			Optimizer::PoseDvlGyrosOPtimizationLastFrame(&mCurrentFrame, mlamda_DVL);
	}
	else {
//		cout << "track local map from last Keyframe frame" << endl;
		inliers =
			Optimizer::PoseDvlGyrosOPtimizationLastKeyFrame(&mCurrentFrame, mlamda_DVL);
	}
	#endif

	aux1 = 0, aux2 = 0;
	for (int i = 0; i < mCurrentFrame.N; i++)
		if (mCurrentFrame.mvpMapPoints[i]) {
			aux1++;
			if (mCurrentFrame.mvbOutlier[i]) {
				aux2++;
			}
		}

	mnMatchesInliers = 0;

	// Update MapPoints Statistics
	for (int i = 0; i < mCurrentFrame.N; i++) {
		if (mCurrentFrame.mvpMapPoints[i]) {
			if (!mCurrentFrame.mvbOutlier[i]) {
				mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
				if (!mbOnlyTracking) {
					if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
						mnMatchesInliers++;
					}
				}
				else {
					mnMatchesInliers++;
				}
			}
			else if (mSensor == System::STEREO || mSensor == System::DVL_STEREO) {
				mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
			}
		}
	}

	// Decide if the tracking was succesful
	// More restrictive if there was a relocalization recently
	mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
	if(mnMatchesInliers < mnFeatureTarget * 0.15){
        mCurrentFrame.mPoorVision = true;
    }
//	if ( (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames) && (mnMatchesInliers < 50)) {
//		return false;
//	}
//
//	if ((mnMatchesInliers > 30) && (mState == RECENTLY_LOST)) {
//		return true;
//	}


//	if (mSensor == System::IMU_MONOCULAR) {
//		if (mnMatchesInliers < 15) {
//			return false;
//		}
//		else {
//			return true;
//		}
//	}
//	else if (mSensor == System::IMU_STEREO) {
//		if (mnMatchesInliers < 15) {
//			return false;
//		}
//		else {
//			return true;
//		}
//	}

	if (mBackendMode == BackendMode::GtsamDynamic)
		return inliers > 0;

	if (mSensor == System::DVL_STEREO) {
//		cout << "inliers: " << mnMatchesInliers << endl;
		if (mnMatchesInliers < 30) {
			return false;
		}
		else {
			return true;
		}
	}
	else {
		if (mnMatchesInliers < 30) {
			return false;
		}
		else {
			return true;
		}
	}
}

bool Tracking::NeedNewKeyFrame()
{
	// DVL is an optional asynchronous observation; visual-inertial tracking
	// remains eligible when the current image has no fresh DVL sample.
	const bool sensorReady = true;

	if (!mCalibrated) {
		if (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp >= mKF_init_step
			&& sensorReady) {
			return true;
		}
		else{
            return false;
        }
	}
    else if (!mInitialized) {
        if (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp >= mKF_init_step
            && sensorReady) {
            return true;
        }
        else{
            return false;
        }
    }



	// If Local Mapping is freezed by a Loop Closure do not insert keyframes
	if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
		return false;
	}

	// Return false if IMU is initialazing
	if (mpLocalMapper->IsInitializing()) {
		return false;
	}
	const int nKFs = mpAtlas->KeyFramesInMap();

	// Do not insert keyframes if not enough frames have passed from last relocalisation
	if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames) {
		return false;
	}

	// Tracked MapPoints in the reference keyframe
	int nMinObs = 3;
	if (nKFs <= 2) {
		nMinObs = 2;
	}
	int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

	// Local Mapping accept keyframes?
	bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

	// Check how many "close" points are being tracked and how many could be potentially created.
	int nNonTrackedClose = 0;
	int nTrackedClose = 0;


	bool bNeedToInsertClose;
    int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
    for(int i =0; i<N; i++)
    {
        if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
        {
            if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                nTrackedClose++;
            else
                nNonTrackedClose++;

        }
    }
	bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70) && (mpAtlas->GetAllKeyFrames().size()>mKFThresholdForMap);
    // if(!mCurrentFrame.mPoorVision){
    //     mCurrentFrame.mPoorVision = bNeedToInsertClose;
    // }
	// Thresholds
	float thRefRatio = 0.75f;

	// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
	const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
	// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
	const bool c1b = ((mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames) && bLocalMappingIdle);
	//Condition 1c: tracking is weak
	const bool c1c = mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose;
	// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
	const bool c2 = (((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose)) && mnMatchesInliers > 15);
    bool c3 = (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp >= mKF_init_step) || (mCurrentFrame.mbDVL);


	if (((c1a || c1b ) && c2 || c3)) {
		// If the mapping accepts keyframes, insert keyframe.
		// Otherwise send a signal to interrupt BA
		if (bLocalMappingIdle) {
// //            ROS_INFO_STREAM("Idle BA");  // original
			return true;
		}
		else {
			mpLocalMapper->InterruptBA();
            return false;
		}
	}
	else {
		return false;
	}
}

void Tracking::CreateNewKeyFrame()
{
	if (mpLocalMapper->IsInitializing()) {
		return;
	}

	if (!mpLocalMapper->SetNotStop(true)) {
		return;
	}
//	int assigned_feature=0;
//	for (int i = 0; i < FRAME_GRID_COLS; i++)
//	{
//		for (int j = 0; j < FRAME_GRID_ROWS; j++)
//		{
//			assigned_feature+=mCurrentFrame.mGrid[i][j].size();
//			if (mCurrentFrame.Nleft != 0)
//			{
//				assigned_feature+=mCurrentFrame.mGridRight[i][j].size();
//			}
//		}
//	}
//	cout<<"before insert new keyframe, assigned feature in frame: "<<assigned_feature<<endl;
KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
#ifdef AQUA_HAS_GTSAM_DYNAMIC
	if (mBackendMode == BackendMode::GtsamDynamic) {
		// Before inertial initialization, visual keyframes maintain map support.
        // Once fusion starts, only a committed transaction may publish a keyframe.
		const int trackingSupport = OptimizeCurrentFrameWithBackend(true, pKF->mnId);
        const bool visualBootstrap = !AcquireGtsamBackend() && trackingSupport > 0;
		if (!DynamicLastCommitSucceeded() && !visualBootstrap) {
			RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
			            "dynamic GTSAM keyframe transaction was not committed; "
			            "discarding uncommitted front-end keyframe and preserving IMU interval");
			delete pKF;
			mpLocalMapper->SetNotStop(false);
			return;
		} else {
			pKF->SetPose(mCurrentFrame.mTcw);
		}
	}
#endif
	mpDenseMapper->InsertNewKF(pKF);
//	assigned_feature=0;
//	for (int i = 0; i < FRAME_GRID_COLS; i++)
//	{
//		for (int j = 0; j < FRAME_GRID_ROWS; j++)
//		{
//			assigned_feature+=mCurrentFrame.mGrid[i][j].size();
//			if (mCurrentFrame.Nleft != 0)
//			{
//				assigned_feature+=mCurrentFrame.mGridRight[i][j].size();
//			}
//		}
//	}
//	cout<<"after insert new keyframe, assigned feature in frame: "<<assigned_feature<<endl;

	if (mpAtlas->isImuInitialized()) {
		pKF->bImu = true;
	}
    if (mpLastKeyFrame) {
        pKF->mPrevKF = mpLastKeyFrame;
        // Eigen::Vector3d Vd;
        // mpLastKeyFrame->GetDvlVelocity(Vd);
        // pKF->SetDvlVelocity(Vd);
        mpLastKeyFrame->mNextKF = pKF;
    }
    // else: first KF after map reset — no prev KF to link
    // //Optimize bias of new KF
    // if(mInitialized){
    //     set<KeyFrame*, KFComparator> spKFs;
    //     spKFs.insert(pKF);
    //     spKFs.insert(mpLastKeyFrame);
    //     if (spKFs.size() > 1) {
    //         Optimizer::PoseOptimizationDVLIMUBiasOnly(spKFs,mpAtlas);
    //         UpdateFrameDVLGyro(pKF->GetImuBias(),pKF);
    //     }
    //     else{
//     //         ROS_ERROR_STREAM("Only one KF in spKFs, cannot optimize bias!");  // original
    //         assert(-1);
    //     }
    // }
    // else{
    //     pKF->SetNewBias(mpLastKeyFrame->GetImuBias());
    // }
//     // ROS_INFO_STREAM(fixed<<setprecision(6)<<"New KF["<<pKF->mnId<<"] time"<<pKF->mTimeStamp<<"dvl: "<<pKF->mbDVL);  // original


	// // pKF->SetNewBias(mpLastKeyFrame->GetImuBias());
    // KeyFrame* pLastKF = mpLastKeyFrame;
    // while(pLastKF && pLastKF->mnId>0){
    //     if(pLastKF->GetImuBias().bax!=0 || pLastKF->GetImuBias().bay!=0 || pLastKF->GetImuBias().baz!=0){
    //         pKF->SetNewBias(pLastKF->GetImuBias());
//     //         ROS_INFO_STREAM("New KF["<<pKF->mnId<<"] bias: "<<pKF->GetImuBias().bax<<","<<pKF->GetImuBias().bay  // original
    //                                  <<","<<pKF->GetImuBias().baz);
    //         break;
    //     }
    //     pLastKF = pLastKF->mPrevKF;
    // }
	mpReferenceKF = pKF;
	mCurrentFrame.mpReferenceKF = pKF;


	// Reset preintegration from last KF (Create new object)
	if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
		mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
	}
	if (mSensor == System::DVL_STEREO) {
        std::unique_lock<std::shared_mutex> lock(mBiasMutex);
		// mpDvlPreintegratedFromLastKF = new DVLGroPreIntegration(pKF->GetImuBias(), pKF->mImuCalib);
        mpIntegrator->CreateNewIntFromKF_C2C(mLastBias, GetExtrinsicPara(),mAlpha,mBeta);
        pKF->SetNewBias(mLastBias);
//         ROS_INFO_STREAM("create new KF["<<pKF->mnId<<"]"<<" bias(acc gyros):"<<mLastBias.bax<<","<<mLastBias.bay<<","<<mLastBias.baz<<","<<mLastBias.bwx<<","<<mLastBias.bwy<<","<<mLastBias.bwz);  // original
        // mpIntegrator->CreateNewIntFromKF_D2D(pKF->GetImuBias(), pKF->mImuCalib,mAlpha,mBeta);
	}

	if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
	{
		mCurrentFrame.UpdatePoseMatrices();
		// cout << "create new MPs" << endl;
		// We sort points by the measured depth by the stereo/RGBD sensor.
		// We create all those MapPoints whose depth < mThDepth.
		// If there are less than 100 close points we create the 100 closest.
		int maxPoint = 100;
		if (mSensor == System::IMU_STEREO) {
			maxPoint = 100;
		}

		vector<pair<float, int> > vDepthIdx;
		int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
		vDepthIdx.reserve(mCurrentFrame.N);
		for (int i = 0; i < N; i++) {
			float z = mCurrentFrame.mvDepth[i];
			if (z > 0) {
				vDepthIdx.push_back(make_pair(z, i));
			}
		}

		if (!vDepthIdx.empty()) {
			sort(vDepthIdx.begin(), vDepthIdx.end());

			int nPoints = 0;
			for (size_t j = 0; j < vDepthIdx.size(); j++) {
				int i = vDepthIdx[j].second;

				bool bCreateNew = false;

				MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
				if (!pMP) {
					bCreateNew = true;
				}
				else if (pMP->Observations() < 1) {
					bCreateNew = true;
					mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				}

				if (bCreateNew) {
					cv::Mat x3D;

					if (mCurrentFrame.Nleft == -1) {
						x3D = mCurrentFrame.UnprojectStereo(i);
					}
					else {
						x3D = mCurrentFrame.UnprojectStereoFishEye(i);
					}

					MapPoint *pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
					pNewMP->AddObservation(pKF, i);

					//Check if it is a stereo observation in order to not
					//duplicate mappoints
					if (mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0) {
						mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
						pNewMP->AddObservation(pKF, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
						pKF->AddMapPoint(pNewMP, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
					}

					pKF->AddMapPoint(pNewMP, i);
					pNewMP->ComputeDistinctiveDescriptors();
					pNewMP->UpdateNormalAndDepth();
					mpAtlas->AddMapPoint(pNewMP);

					mCurrentFrame.mvpMapPoints[i] = pNewMP;
					nPoints++;
				}
				else {
					nPoints++; // TODO check ???
				}

				if (vDepthIdx[j].first > mThDepth && nPoints > maxPoint) {
					break;
				}
			}

			Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);

		}
	}


	mpLocalMapper->InsertKeyFrame(pKF);

	mpLocalMapper->SetNotStop(false);

	mnLastKeyFrameId = mCurrentFrame.mnId;
	mpLastKeyFrame = pKF;
	//cout  << "end creating new KF" << endl;
    // if(mpLastKeyFrame->mnId>2){
    //     std::ofstream o_file1("/home/da/project/ros/orb_dvl2_ws/src/dvl2/data/g2o/1.txt", std::ios::out);
    //     std::ifstream i_file1("/home/da/project/ros/orb_dvl2_ws/src/dvl2/data/g2o/1.txt");
    //     std::ofstream o_file2("/home/da/project/ros/orb_dvl2_ws/src/dvl2/data/g2o/2.txt", std::ios::out);
    //     std::ifstream i_file2("/home/da/project/ros/orb_dvl2_ws/src/dvl2/data/g2o/2.txt");
    //
    //     // DvlImuCamPose pose_test(pKFi);
    //     DVLGroPreIntegration *pDVLGroPreIntegration = mpLastKeyFrame->mpDvlPreintegrationKeyFrame;
    //     boost::archive::text_oarchive oa1(o_file1);
    //     oa1 << pDVLGroPreIntegration;
    //     o_file1.close();
    //     DVLGroPreIntegration* pDVLGroPreIntegration2 = new DVLGroPreIntegration();
    //     boost::archive::text_iarchive ia1(i_file1);
    //     ia1 >> pDVLGroPreIntegration2;
//     //     ROS_INFO_STREAM("pInt1 dR: \n"<<pDVLGroPreIntegration->GetDeltaRotation(pDVLGroPreIntegration->mb));  // original
//     //     ROS_INFO_STREAM("pInt2 dR: \n"<<pDVLGroPreIntegration2->GetDeltaRotation(pDVLGroPreIntegration2->mb));  // original
//     //     ROS_INFO_STREAM("pInt1 dV: \n"<<pDVLGroPreIntegration->GetDeltaVelocity(pDVLGroPreIntegration->mb));  // original
//     //     ROS_INFO_STREAM("pInt2 dV: \n"<<pDVLGroPreIntegration2->GetDeltaVelocity(pDVLGroPreIntegration2->mb));  // original
//     //     ROS_INFO_STREAM("pInt1 dP_dvl: \n"<<pDVLGroPreIntegration->GetDVLPosition(pDVLGroPreIntegration->mb));  // original
//     //     ROS_INFO_STREAM("pInt2 dP_dvl: \n"<<pDVLGroPreIntegration2->GetDVLPosition(pDVLGroPreIntegration2->mb));  // original
//     //     ROS_INFO_STREAM("pInt1 dP_imu: \n"<<pDVLGroPreIntegration->GetDeltaPosition(pDVLGroPreIntegration->mb));  // original
//     //     ROS_INFO_STREAM("pInt2 dP_imu: \n"<<pDVLGroPreIntegration2->GetDeltaPosition(pDVLGroPreIntegration2->mb));  // original
    //
    //     DvlImuCamPose pose_test(mpLastKeyFrame);
    //     boost::archive::text_oarchive oa3(o_file2);
    //     oa3 << pose_test;
    //     o_file2.close();
    //     DvlImuCamPose pose_test2;
    //     boost::archive::text_iarchive ia2(i_file2);
    //     ia2 >> pose_test2;
//     //     ROS_INFO_STREAM("Pose1 Rwc: \n"<<pose_test.Rwc);  // original
//     //     ROS_INFO_STREAM("Pose2 Rwc: \n"<<pose_test2.Rwc);  // original
//     //     ROS_INFO_STREAM("Pose1 twc: \n"<<pose_test.twc);  // original
//     //     ROS_INFO_STREAM("Pose2 twc: \n"<<pose_test2.twc);  // original
//     //     ROS_INFO_STREAM("Pose1 Rcw: \n"<<pose_test.Rcw[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 Rcw: \n"<<pose_test2.Rcw[0]);  // original
//     //     ROS_INFO_STREAM("Pose1 tcw: \n"<<pose_test.tcw[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 tcw: \n"<<pose_test2.tcw[0]);  // original
//     //     ROS_INFO_STREAM("Pose1 Rgc: \n"<<pose_test.R_imu_c[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 Rgc: \n"<<pose_test2.R_imu_c[0]);  // original
//     //     ROS_INFO_STREAM("Pose1 tgc: \n"<<pose_test.t_imu_c[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 tgc: \n"<<pose_test2.t_imu_c[0]);  // original
//     //     ROS_INFO_STREAM("Pose1 Rdc: \n"<<pose_test.R_dvl_c[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 Rdc: \n"<<pose_test2.R_dvl_c[0]);  // original
//     //     ROS_INFO_STREAM("Pose1 tdc: \n"<<pose_test.t_dvl_c[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 tdc: \n"<<pose_test2.t_dvl_c[0]);  // original
//     //     ROS_INFO_STREAM("Pose1 Rcd: \n"<<pose_test.R_c_dvl[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 Rcd: \n"<<pose_test2.R_c_dvl[0]);  // original
//     //     ROS_INFO_STREAM("Pose1 tcd: \n"<<pose_test.t_c_dvl[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 tcd: \n"<<pose_test2.t_c_dvl[0]);  // original
//     //     ROS_INFO_STREAM("Pose1 Rcg: \n"<<pose_test.R_c_imu[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 Rcg: \n"<<pose_test2.R_c_imu[0]);  // original
//     //     ROS_INFO_STREAM("Pose1 tcg: \n"<<pose_test.t_c_imu[0]);  // original
//     //     ROS_INFO_STREAM("Pose2 tcg: \n"<<pose_test2.t_c_imu[0]);  // original
    //     auto pc1 = static_cast<Pinhole*>(pose_test.pCamera[0]);
    //     auto pc2 = static_cast<Pinhole*>(pose_test2.pCamera[0]);
//     //     ROS_INFO_STREAM("Pose1 camera: param"<<pc1->mvParameters[0]<<","<<pc1->mvParameters[1]<<","<<pc1->mvParameters[2]<<","<<pc1->mvParameters[3]<<", type:"<<pc1->mnType<<", ID:"<<pc1->mnId);  // original
//     //     ROS_INFO_STREAM("Pose2 camera: param"<<pc2->mvParameters[0]<<","<<pc2->mvParameters[1]<<","<<pc2->mvParameters[2]<<","<<pc2->mvParameters[3]<<", type:"<<pc2->mnType<<", ID:"<<pc2->mnId);  // original
    //
    // }
}

void Tracking::CreateNewKeyFrameKLT()
{
	if (mpLocalMapper->IsInitializing()) {
		return;
	}

	if (!mpLocalMapper->SetNotStop(true)) {
		return;
	}
	KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

	mpDenseMapper->InsertNewKF(pKF, true);


	if (mpAtlas->isImuInitialized()) {
		pKF->bImu = true;
	}

	pKF->SetNewBias(mCurrentFrame.mImuBias);
	mpReferenceKF = pKF;
	mCurrentFrame.mpReferenceKF = pKF;

	if (mpLastKeyFrame) {
		pKF->mPrevKF = mpLastKeyFrame;
		mpLastKeyFrame->mNextKF = pKF;
	}
	else {
		Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);
	}

	// Reset preintegration from last KF (Create new object)
	if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
		mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
	}
	if (mSensor == System::DVL_STEREO) {
		mpDvlPreintegratedFromLastKF = new DVLGroPreIntegration(pKF->GetImuBias(), pKF->mImuCalib);
	}

	if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
	{
		mCurrentFrame.UpdatePoseMatrices();
		// cout << "create new MPs" << endl;
		// We sort points by the measured depth by the stereo/RGBD sensor.
		// We create all those MapPoints whose depth < mThDepth.
		// If there are less than 100 close points we create the 100 closest.
		int maxPoint = 100;
		if (mSensor == System::IMU_STEREO) {
			maxPoint = 100;
		}

		vector<pair<float, int> > vDepthIdx;
		int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
		vDepthIdx.reserve(mCurrentFrame.N);
		for (int i = 0; i < N; i++) {
			float z = mCurrentFrame.mvDepth[i];
			if (z > 0) {
				vDepthIdx.push_back(make_pair(z, i));
			}
		}

		if (!vDepthIdx.empty()) {
			sort(vDepthIdx.begin(), vDepthIdx.end());

			int nPoints = 0;
			for (size_t j = 0; j < vDepthIdx.size(); j++) {
				int i = vDepthIdx[j].second;

				bool bCreateNew = false;

				cv::Mat mask = mpLKTracker->mask.clone();
				MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
				if (!pMP) {
					bCreateNew = true;
				}
				else if (pMP->Observations() < 1) {
					bCreateNew = true;
					mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
				}
				if (mask.at<uchar>(mCurrentFrame.mvKeys[i].pt) == 0) {
					bCreateNew = false;
				}

				if (bCreateNew) {

					cv::Mat x3D;

					if (mCurrentFrame.Nleft == -1) {
						x3D = mCurrentFrame.UnprojectStereo(i);
					}
					else {
						x3D = mCurrentFrame.UnprojectStereoFishEye(i);
					}

					MapPoint *pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
					pNewMP->AddObservation(pKF, i);

					//Check if it is a stereo observation in order to not
					//duplicate mappoints
					if (mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0) {
						mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
						pNewMP->AddObservation(pKF, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
						pKF->AddMapPoint(pNewMP, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
					}

					pKF->AddMapPoint(pNewMP, i);
					pNewMP->ComputeDistinctiveDescriptors();
					pNewMP->UpdateNormalAndDepth();
					mpAtlas->AddMapPoint(pNewMP);

					mCurrentFrame.mvpMapPoints[i] = pNewMP;
					nPoints++;

					mpLKTracker->ids.push_back(pNewMP->mnId);
				}
				else {
					nPoints++; // TODO check ???
				}

				if (vDepthIdx[j].first > mThDepth && nPoints > maxPoint) {
					break;
				}
			}

			Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);

		}
	}


	mpLocalMapper->InsertKeyFrame(pKF);

	mpLocalMapper->SetNotStop(false);

	mnLastKeyFrameId = mCurrentFrame.mnId;
	mpLastKeyFrame = pKF;
	//cout  << "end creating new KF" << endl;
}

void Tracking::SearchLocalPoints()
{
	// Do not search map points already matched
	for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
	     vit != vend; vit++) {
		MapPoint *pMP = *vit;
		if (pMP) {
			if (pMP->isBad()) {
				*vit = static_cast<MapPoint *>(NULL);
			}
			else {
				pMP->IncreaseVisible();
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
				pMP->mbTrackInView = false;
				pMP->mbTrackInViewR = false;
			}
		}
	}

	int nToMatch = 0;

	// Project points in frame and check its visibility
	for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend;
	     vit++) {
		MapPoint *pMP = *vit;

		if (pMP->mnLastFrameSeen == mCurrentFrame.mnId) {
			continue;
		}
		if (pMP->isBad()) {
			continue;
		}
		// Project (this fills MapPoint variables for matching)
		if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
			pMP->IncreaseVisible();
			nToMatch++;
		}
		if (pMP->mbTrackInView) {
			mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
		}
	}

	if (nToMatch > 0) {
		FeatureMatcher matcher(*mpFeatureFrontend, 0.8F);
		int th = 1;
		if (mSensor == System::RGBD) {
			th = 3;
		}
		if (mpAtlas->isImuInitialized()) {
			if (mpAtlas->GetCurrentMap()->GetIniertialBA2()) {
				th = 2;
			}
			else {
				th = 3;
			}
		}
		else if (!mpAtlas->isImuInitialized() && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) {
			th = 10;
		}

		// If the camera has been relocalised recently, perform a coarser search
		if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
			th = 5;
		}

		if (mState == LOST || mState == RECENTLY_LOST) { // Lost for less than 1 second
			th = 15;
		} // 15
		// add pointer of matched map point too mCurrentFrame.mvpMapPoints
		int matches = matcher.SearchByProjection(mCurrentFrame,
		                                         mvpLocalMapPoints,
		                                         th,
		                                         mpLocalMapper->mbFarPoints,
		                                         mpLocalMapper->mThFarPoints);
	}
}

void Tracking::UpdateLocalMap()
{
	// This is for visualization
	mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

	// Update
	UpdateLocalKeyFrames();
	UpdateLocalPoints();
}
// add all map points in mvpLocalKeyFrames to mvpLocalMapPoints
void Tracking::UpdateLocalPoints()
{
	mvpLocalMapPoints.clear();

	int count_pts = 0;

	for (vector<KeyFrame *>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(),
		     itEndKF = mvpLocalKeyFrames.rend(); itKF != itEndKF; ++itKF) {
		KeyFrame *pKF = *itKF;
		const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

		for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++) {

			MapPoint *pMP = *itMP;
			if (!pMP) {
				continue;
			}
			if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId) {
				continue;
			}
			if (!pMP->isBad()) {
				count_pts++;
				mvpLocalMapPoints.push_back(pMP);
				pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
			}
		}
	}
}

// add all keyframes share the map points in current frame
// and their neighbors to mvpLocalKeyFrames
void Tracking::UpdateLocalKeyFrames()
{
	// Each map point vote for the keyframes in which it has been observed
	// keyframeCounter include all keyframes share the map points in current frame
	map<KeyFrame *, int> keyframeCounter;
	if (!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2)) {
		for (int i = 0; i < mCurrentFrame.N; i++) {
			MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
			if (pMP) {
				if (!pMP->isBad()) {
					const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
					for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(),
						     itend = observations.end(); it != itend; it++)
						keyframeCounter[it->first]++;
				}
				else {
					mCurrentFrame.mvpMapPoints[i] = NULL;
				}
			}
		}
	}
	else {
		for (int i = 0; i < mLastFrame.N; i++) {
			// Using lastframe since current frame has not matches yet
			if (mLastFrame.mvpMapPoints[i]) {
				MapPoint *pMP = mLastFrame.mvpMapPoints[i];
				if (!pMP) {
					continue;
				}
				if (!pMP->isBad()) {
					const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
					for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(),
						     itend = observations.end(); it != itend; it++)
						keyframeCounter[it->first]++;
				}
				else {
					// MODIFICATION
					mLastFrame.mvpMapPoints[i] = NULL;
				}
			}
		}
	}


	int max = 0;
	KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

	mvpLocalKeyFrames.clear();
	mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

	// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
	for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd;
	     it++) {
		KeyFrame *pKF = it->first;

		if (pKF->isBad()) {
			continue;
		}

		if (it->second > max) {
			max = it->second;
			pKFmax = pKF;
		}

		mvpLocalKeyFrames.push_back(pKF);
		pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
	}

	// Include also some not-already-included keyframes that are neighbors to already-included keyframes
	for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
	     itKF != itEndKF; itKF++) {
		// Limit the number of keyframes
		if (mvpLocalKeyFrames.size() > 80) { // 80
			break;
		}

		KeyFrame *pKF = *itKF;

		const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


		for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
		     itNeighKF != itEndNeighKF; itNeighKF++) {
			KeyFrame *pNeighKF = *itNeighKF;
			if (!pNeighKF->isBad()) {
				if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
					mvpLocalKeyFrames.push_back(pNeighKF);
					pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
					break;
				}
			}
		}

		const set<KeyFrame *> spChilds = pKF->GetChilds();
		for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
			KeyFrame *pChildKF = *sit;
			if (!pChildKF->isBad()) {
				if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
					mvpLocalKeyFrames.push_back(pChildKF);
					pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
					break;
				}
			}
		}

		KeyFrame *pParent = pKF->GetParent();
		if (pParent) {
			if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
				mvpLocalKeyFrames.push_back(pParent);
				pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
				break;
			}
		}
	}

	// Add 10 last temporal KFs (mainly for IMU)
	if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mvpLocalKeyFrames.size() < 80) {
		//cout << "CurrentKF: " << mCurrentFrame.mnId << endl;
		KeyFrame *tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

		const int Nd = 20;
		for (int i = 0; i < Nd; i++) {
			if (!tempKeyFrame) {
				break;
			}
			//cout << "tempKF: " << tempKeyFrame << endl;
			if (tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
				mvpLocalKeyFrames.push_back(tempKeyFrame);
				tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.mnId;
				tempKeyFrame = tempKeyFrame->mPrevKF;
			}
		}
	}

	if (pKFmax) {
		mpReferenceKF = pKFmax;
		mCurrentFrame.mpReferenceKF = mpReferenceKF;
	}
}

bool Tracking::Relocalization()
{
	Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
	// Relocalization is performed when tracking is lost
	// Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
	vector<KeyFrame *>
		vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

	if (vpCandidateKFs.empty()) {
		Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
		return false;
	}

	const int nKFs = vpCandidateKFs.size();

	// We perform first an ORB matching with each candidate
	// If enough matches are found we setup a PnP solver
	FeatureMatcher matcher(*mpFeatureFrontend, 0.75F);

	vector<MLPnPsolver *> vpMLPnPsolvers;
	vpMLPnPsolvers.resize(nKFs);

	vector<vector<MapPoint *> > vvpMapPointMatches;
	vvpMapPointMatches.resize(nKFs);

	vector<bool> vbDiscarded;
	vbDiscarded.resize(nKFs);

	int nCandidates = 0;

	for (int i = 0; i < nKFs; i++) {
		KeyFrame *pKF = vpCandidateKFs[i];
		if (pKF->isBad()) {
			vbDiscarded[i] = true;
		}
		else {
			int nmatches = matcher.SearchByNeuralPair(
				pKF, mCurrentFrame, vvpMapPointMatches[i]);
			if (nmatches < 15) {
				vbDiscarded[i] = true;
				continue;
			}
			else {
				MLPnPsolver *pSolver = new MLPnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
				pSolver->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991);  //This solver needs at least 6 points
				vpMLPnPsolvers[i] = pSolver;
			}
		}
	}

	// Alternatively perform some iterations of P4P RANSAC
	// Until we found a camera pose supported by enough inliers
	bool bMatch = false;
	FeatureMatcher matcher2(*mpFeatureFrontend, 0.9F);

	while (nCandidates > 0 && !bMatch) {
		for (int i = 0; i < nKFs; i++) {
			if (vbDiscarded[i]) {
				continue;
			}

			// Perform 5 Ransac Iterations
			vector<bool> vbInliers;
			int nInliers;
			bool bNoMore;

			MLPnPsolver *pSolver = vpMLPnPsolvers[i];
			cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

			// If Ransac reachs max. iterations discard keyframe
			if (bNoMore) {
				vbDiscarded[i] = true;
				nCandidates--;
			}

			// If a Camera Pose is computed, optimize
			if (!Tcw.empty()) {
				Tcw.copyTo(mCurrentFrame.mTcw);

				set<MapPoint *> sFound;

				const int np = vbInliers.size();

				for (int j = 0; j < np; j++) {
					if (vbInliers[j]) {
						mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
						sFound.insert(vvpMapPointMatches[i][j]);
					}
					else {
						mCurrentFrame.mvpMapPoints[j] = NULL;
					}
				}

				int nGood = OptimizeCurrentFrameWithBackend();

				if (nGood < 10) {
					continue;
				}

				for (int io = 0; io < mCurrentFrame.N; io++)
					if (mCurrentFrame.mvbOutlier[io]) {
						mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);
					}

				// If few inliers, search by projection in a coarse window and optimize again
				if (nGood < 50) {
					int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10);

					if (nadditional + nGood >= 50) {
						nGood = OptimizeCurrentFrameWithBackend();

						// If many inliers but still not enough, search by projection again in a narrower window
						// the camera has been already optimized with many points
						if (nGood > 30 && nGood < 50) {
							sFound.clear();
							for (int ip = 0; ip < mCurrentFrame.N; ip++)
								if (mCurrentFrame.mvpMapPoints[ip]) {
									sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
								}
							nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3);

							// Final optimization
							if (nGood + nadditional >= 50) {
								nGood = OptimizeCurrentFrameWithBackend();

								for (int io = 0; io < mCurrentFrame.N; io++)
									if (mCurrentFrame.mvbOutlier[io]) {
										mCurrentFrame.mvpMapPoints[io] = NULL;
									}
							}
						}
					}
				}


				// If the pose is supported by enough inliers stop ransacs and continue
				if (nGood >= 50) {
					bMatch = true;
					break;
				}
			}
		}
	}

	if (!bMatch) {
		return false;
	}
	else {
		mnLastRelocFrameId = mCurrentFrame.mnId;
		cout << "Relocalized!!" << endl;
		return true;
	}

}

void Tracking::Reset(bool bLocMap)
{
	Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);
#ifdef AQUA_HAS_GTSAM_DYNAMIC
		if (mBackendMode == BackendMode::GtsamDynamic) {
			ResetDynamicBackendState();
			std::lock_guard<std::mutex> lock(mMutexImuQueue);
			mDynamicKeyframeImu.clear();
			mDynamicKeyframeDvl.clear();
			mDynamicInitializationImu.clear();
            mLastDynamicInitializationAttempt = -std::numeric_limits<double>::infinity();
			mCollectDynamicInitializationImu = true;
	}
#endif

	// Reset Local Mapping
	if (!bLocMap) {
		Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
		mpLocalMapper->RequestReset();
		Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
	}


	// Reset Loop Closing
	Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
	mpLoopClosing->RequestReset();
	Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

	// Clear BoW Database
	Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
	mpKeyFrameDB->clear();
	Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

	// Clear Map (this erase MapPoints and KeyFrames)
	mpAtlas->clearAtlas();
	mpAtlas->CreateNewMap();
	if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR) {
		mpAtlas->SetInertialSensor();
	}
	mnInitialFrameId = 0;

	KeyFrame::nNextId = 0;
	Frame::nNextId = 0;
	mState = NO_IMAGES_YET;
	mHasDynamicStereoInitCandidate = false;

	if (mpInitializer) {
		delete mpInitializer;
		mpInitializer = static_cast<Initializer *>(NULL);
	}
	mbSetInit = false;

	mlRelativeFramePoses.clear();
	mlpReferences.clear();
	mlFrameTimes.clear();
	mlbLost.clear();
	mCurrentFrame = Frame();
	mnLastRelocFrameId = 0;
	mLastFrame = Frame();
	mpReferenceKF = static_cast<KeyFrame *>(NULL);
	mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
	mvIniMatches.clear();
	mvbPrevMatched.clear();


	Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
//     ROS_INFO_STREAM("reset done");  // original
}

void Tracking::ResetActiveMap(bool bLocMap)
{
	Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
#ifdef AQUA_HAS_GTSAM_DYNAMIC
		if (mBackendMode == BackendMode::GtsamDynamic) {
			ResetDynamicBackendState();
			std::lock_guard<std::mutex> lock(mMutexImuQueue);
			mDynamicKeyframeImu.clear();
			mDynamicKeyframeDvl.clear();
			mDynamicInitializationImu.clear();
            mLastDynamicInitializationAttempt = -std::numeric_limits<double>::infinity();
			mCollectDynamicInitializationImu = true;
	}
#endif

	Map *pCurMap = mpAtlas->GetCurrentMap();
	Map *pMapToReset = mpMapToReset ? mpMapToReset : pCurMap;
	std::unique_lock<std::shared_timed_mutex> lock(pMapToReset->mMutexMapUpdate, std::defer_lock);
	std::unique_lock<std::shared_timed_mutex> lock2(pCurMap->mMutexMapUpdate, std::defer_lock);
    lock.lock();
	if (pCurMap != pMapToReset) {
		lock2.lock();
	}
	// Clear BoW Database
//	Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
	std::thread clear_thread(&KeyFrameDatabase::clearMap, mpKeyFrameDB, pMapToReset);
//	mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
	Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

	if (!bLocMap) {
		Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        lock.unlock();
		if (pCurMap != pMapToReset) {
			lock2.unlock();
		}
		mpLocalMapper->RequestResetActiveMap(pMapToReset);
        lock.lock();
        if (pCurMap != pMapToReset) {
            lock2.lock();
        }
		Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
	}

	// Reset Loop Closing
	Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    lock.unlock();
	if (pCurMap != pMapToReset) {
		lock2.unlock();
	}
	mpLoopClosing->RequestResetActiveMap(pMapToReset);
    lock.lock();
    if (pCurMap != pMapToReset) {
        lock2.lock();
    }
	Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);


	// Clear Map (this erase MapPoints and KeyFrames)
//	cout << "clear keyframes and mappoints" << endl;
	clear_thread.join();
	mpAtlas->clearMap(pMapToReset);
//	mpAtlas->clearSmallMaps();
	// The cleared map owns the previous keyframe. A re-initialized stereo map
	// must not link its first keyframe to that deleted object.
	mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

	//KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
	//Frame::nNextId = mnLastInitFrameId;
	mnLastInitFrameId = Frame::nNextId;
	mnLastRelocFrameId = mnLastInitFrameId;
	mState = NO_IMAGES_YET; //NOT_INITIALIZED;
	mHasDynamicStereoInitCandidate = false;

	if (mpInitializer) {
		delete mpInitializer;
		mpInitializer = static_cast<Initializer *>(NULL);
	}

	list<bool> lbLost;
	// lbLost.reserve(mlbLost.size());
	unsigned int index = mnFirstFrameId;
//    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
	for (Map *pMap: mpAtlas->GetAllMaps()) {
		if (pMap->GetAllKeyFrames().size() > 0) {
			if (index > pMap->GetLowerKFID()) {
				index = pMap->GetLowerKFID();
			}
		}
	}

	//cout << "First Frame id: " << index << endl;
	int num_lost = 0;
//    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

	for (list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++) {
		if (index < mnInitialFrameId) {
			lbLost.push_back(*ilbL);
		}
		else {
			lbLost.push_back(true);
			num_lost += 1;
		}

		index++;
	}
//    cout << num_lost << " Frames set to lost" << endl;

	mlbLost = lbLost;

	mnInitialFrameId = mCurrentFrame.mnId;
	mnLastRelocFrameId = mCurrentFrame.mnId;


	mLastFrame = Frame(mCurrentFrame);
//	mLastFrame.mTimeStamp = mCurrentFrame.mTimeStamp;
	mCurrentFrame = Frame();
	mpReferenceKF = static_cast<KeyFrame *>(NULL);
	// mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
	mvIniMatches.clear();
	mvbPrevMatched.clear();



	// if no reference for integration, create
	// {
		// std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);
		// if (!mpDvlPreintegratedFromLastKFBeforeLost) {
		// 	mpDvlPreintegratedFromLastKFBeforeLost = new DVLGroPreIntegration(IMU::Bias(), GetExtrinsicPara());
		// 	mLastFrameBeforeLoss = Frame(mCurrentFrame);
		// }
	// }

    // mpIntegrator->CreateNewIntFromKF_C2C(IMU::Bias(),GetExtrinsicPara(), mAlpha, mBeta);
    // mpIntegrator->CreateNewIntFromKF_D2D(IMU::Bias(),GetExtrinsicPara(), mAlpha, mBeta);

	Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint *> Tracking::GetLocalMapMPS()
{
	return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	float fx = fSettings["Camera.fx"];
	float fy = fSettings["Camera.fy"];
	float cx = fSettings["Camera.cx"];
	float cy = fSettings["Camera.cy"];

	cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
	K.at<float>(0, 0) = fx;
	K.at<float>(1, 1) = fy;
	K.at<float>(0, 2) = cx;
	K.at<float>(1, 2) = cy;
	K.copyTo(mK);

	cv::Mat DistCoef(4, 1, CV_32F);
	DistCoef.at<float>(0) = fSettings["Camera.k1"];
	DistCoef.at<float>(1) = fSettings["Camera.k2"];
	DistCoef.at<float>(2) = fSettings["Camera.p1"];
	DistCoef.at<float>(3) = fSettings["Camera.p2"];
	const float k3 = fSettings["Camera.k3"];
	if (k3 != 0) {
		DistCoef.resize(5);
		DistCoef.at<float>(4) = k3;
	}
	DistCoef.copyTo(mDistCoef);

	mbf = fSettings["Camera.bf"];

	Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
	mbOnlyTracking = flag;
}
// used for initialize IMU and refine IMU
void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame)
{
	Map *pMap = pCurrentKeyFrame->GetMap();
	unsigned int index = mnFirstFrameId;
	list<ORB_SLAM3::KeyFrame *>::iterator lRit = mlpReferences.begin();
	list<bool>::iterator lbL = mlbLost.begin();
	for (list<cv::Mat>::iterator lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end(); lit != lend;
	     lit++, lRit++, lbL++) {
		if (*lbL) {
			continue;
		}

		KeyFrame *pKF = *lRit;

		while (pKF->isBad()) {
			pKF = pKF->GetParent();
		}

		if (pKF->GetMap() == pMap) {
			(*lit).rowRange(0, 3).col(3) = (*lit).rowRange(0, 3).col(3) * s;
		}
	}

	// mLastBias = b;

	mpLastKeyFrame = pCurrentKeyFrame;

	// mLastFrame.SetNewBias(mLastBias);
	// mCurrentFrame.SetNewBias(mLastBias);

	cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);

	cv::Mat twb1;
	cv::Mat Rwb1;
	cv::Mat Vwb1;
	float t12;

	// Local Mapping cannot complete the tracker-owned preintegration. Waiting
	// here can deadlock the whole IMU_STEREO pipeline during initialization.
	if (!mCurrentFrame.imuIsPreintegrated()) {
		return;
	}


	if (mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId) {
		mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
		                              mLastFrame.mpLastKeyFrame->GetImuPosition(),
									  mLastFrame.mpLastKeyFrame->GetVelocityOld());
	}
	else {
		twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
		Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
		Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocityOld();
		t12 = mLastFrame.mpImuPreintegrated->dT;

		mLastFrame.SetImuPoseVelocity(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
		                              twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz
			                              + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
		                              Vwb1 + Gz * t12
			                              + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
	}

	if (mCurrentFrame.mpImuPreintegrated) {
		twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
		Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
		Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocityOld();
		t12 = mCurrentFrame.mpImuPreintegrated->dT;

		mCurrentFrame.SetImuPoseVelocity(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
		                                 twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz
			                                 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
		                                 Vwb1 + Gz * t12
			                                 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
	}

	mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::UpdateFrameDVLGyro(const IMU::Bias &b, KeyFrame *pCurrentKeyFrame)
{
	{
		std::unique_lock<std::shared_mutex> lock(mBiasMutex);
		mLastBias = b;
	}
// 	mpLastKeyFrame = pCurrentKeyFrame;
//
// 	mLastFrame.SetNewBias(b);
// 	mCurrentFrame.SetNewBias(b);
// 	mLastFrame.SetExtrinsicParamters(pCurrentKeyFrame->mImuCalib);
// 	mCurrentFrame.SetExtrinsicParamters(pCurrentKeyFrame->mImuCalib);
// 	cv::Mat T_ci_c0_cv = mLastFrame.mTcw.clone();
// 	Eigen::Isometry3d T_ci_c0 = Eigen::Isometry3d::Identity();
// 	cv::cv2eigen(T_ci_c0_cv, T_ci_c0.matrix());
// 	Eigen::Isometry3d T_c0_ci = T_ci_c0.inverse();
// //	cout << "pose before update bias"<<T_c0_ci.matrix() << endl;
//
//
// 	cv::Mat twdvl1;
// 	cv::Mat Rwgyro1;
// 	cv::Mat Rwdvl1;
// 	cv::Mat Vwdvl1;
// 	float t12;
//

//
//
    //update lastframe pose
    if(mLastFrame.mpDvlPreintegrationKeyFrame && pCurrentKeyFrame == mLastFrame.mpLastKeyFrame){
        DVLGroPreIntegration *pDvlPreintegratedFromKF = mLastFrame.mpDvlPreintegrationKeyFrame;
        cv::Mat T_c0_cf_cv = mLastFrame.mpLastKeyFrame->GetPoseInverse();
        Eigen::Isometry3d T_g_d,T_d_c,T_c0_cf;
        cv::Mat T_g_d_cv = GetExtrinsicPara().mT_imu_dvl;
        cv::Mat T_d_c_cv = GetExtrinsicPara().mT_dvl_c;
        cv::cv2eigen(T_g_d_cv,T_g_d.matrix());
        cv::cv2eigen(T_d_c_cv,T_d_c.matrix());
        cv::cv2eigen(T_c0_cf_cv,T_c0_cf.matrix());
        Eigen::Matrix3d R_gf_g1 = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_df_df_d1 = Eigen::Vector3d::Identity();
        cv::cv2eigen(pDvlPreintegratedFromKF->GetDeltaRotation(pCurrentKeyFrame->GetImuBias()), R_gf_g1);
        cv::cv2eigen(pDvlPreintegratedFromKF->GetDVLPosition(pCurrentKeyFrame->GetImuBias()), t_df_df_d1);
        Eigen::Matrix3d R_df_d1 =  T_g_d.rotation().inverse() * R_gf_g1 * T_g_d.rotation();
        Eigen::Isometry3d T_df_d1 = Eigen::Isometry3d::Identity();
        T_df_d1.pretranslate(t_df_df_d1);
        T_df_d1.rotate(R_df_d1);
        Eigen::Isometry3d T_c0_c1 = T_c0_cf * T_d_c.inverse() * T_df_d1 * T_d_c;
        cv::Mat T_c1_c0_cv(4,4,CV_32F);
        cv::eigen2cv(T_c0_c1.inverse().matrix(),T_c1_c0_cv);
        T_c1_c0_cv.convertTo(T_c1_c0_cv,CV_32F);
        mLastFrame.SetPose(T_c1_c0_cv);
//         // ROS_INFO_STREAM("update last frame pose");  // original
    }


    // while (!) {
    //     usleep(100);
    // }
    // update currentFrame pose
    if(mCurrentFrame.mpLastKeyFrame == pCurrentKeyFrame&&mCurrentFrame.mpDvlPreintegrationKeyFrame){
        DVLGroPreIntegration *pDvlPreintegratedFromKF = mCurrentFrame.mpDvlPreintegrationKeyFrame;
        cv::Mat T_c0_cf_cv = mCurrentFrame.mpLastKeyFrame->GetPoseInverse();
        Eigen::Isometry3d T_g_d,T_d_c,T_c0_cf;
        cv::Mat T_g_d_cv = GetExtrinsicPara().mT_imu_dvl;
        cv::Mat T_d_c_cv = GetExtrinsicPara().mT_dvl_c;
        cv::cv2eigen(T_g_d_cv,T_g_d.matrix());
        cv::cv2eigen(T_d_c_cv,T_d_c.matrix());
        cv::cv2eigen(T_c0_cf_cv,T_c0_cf.matrix());
        Eigen::Matrix3d R_gf_g1 = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_df_df_d1 = Eigen::Vector3d::Identity();
        cv::cv2eigen(pDvlPreintegratedFromKF->GetDeltaRotation(pCurrentKeyFrame->GetImuBias()), R_gf_g1);
        cv::cv2eigen(pDvlPreintegratedFromKF->GetDVLPosition(pCurrentKeyFrame->GetImuBias()), t_df_df_d1);
        Eigen::Matrix3d R_df_d1 =  T_g_d.rotation().inverse() * R_gf_g1 * T_g_d.rotation();
        Eigen::Isometry3d T_df_d1 = Eigen::Isometry3d::Identity();
        T_df_d1.pretranslate(t_df_df_d1);
        T_df_d1.rotate(R_df_d1);
        Eigen::Isometry3d T_c0_c1 = T_c0_cf * T_d_c.inverse() * T_df_d1 * T_d_c;
        cv::Mat T_c1_c0_cv(4,4,CV_32F);
        cv::eigen2cv(T_c0_c1.inverse().matrix(),T_c1_c0_cv);
        T_c1_c0_cv.convertTo(T_c1_c0_cv,CV_32F);
        mCurrentFrame.SetPose(T_c1_c0_cv);
//         // ROS_INFO_STREAM("update current frame pose");  // original
    }
    // PredictStateDvlGro();


}

cv::Mat Tracking::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
	cv::Mat R1w = pKF1->GetRotation();
	cv::Mat t1w = pKF1->GetTranslation();
	cv::Mat R2w = pKF2->GetRotation();
	cv::Mat t2w = pKF2->GetTranslation();

	cv::Mat R12 = R1w * R2w.t();
	cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

	cv::Mat t12x = Converter::tocvSkewMatrix(t12);

	const cv::Mat &K1 = pKF1->mK;
	const cv::Mat &K2 = pKF2->mK;


	return K1.t().inv() * t12x * R12 * K2.inv();
}

void Tracking::CreateNewMapPoints()
{
	// Retrieve neighbor keyframes in covisibility graph
	const vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();

	FeatureMatcher matcher(*mpFeatureFrontend, 0.6F);

	cv::Mat Rcw1 = mpLastKeyFrame->GetRotation();
	cv::Mat Rwc1 = Rcw1.t();
	cv::Mat tcw1 = mpLastKeyFrame->GetTranslation();
	cv::Mat Tcw1(3, 4, CV_32F);
	Rcw1.copyTo(Tcw1.colRange(0, 3));
	tcw1.copyTo(Tcw1.col(3));
	cv::Mat Ow1 = mpLastKeyFrame->GetCameraCenter();

	const float &fx1 = mpLastKeyFrame->fx;
	const float &fy1 = mpLastKeyFrame->fy;
	const float &cx1 = mpLastKeyFrame->cx;
	const float &cy1 = mpLastKeyFrame->cy;
	const float &invfx1 = mpLastKeyFrame->invfx;
	const float &invfy1 = mpLastKeyFrame->invfy;

	const float ratioFactor = 1.5f * mpLastKeyFrame->mfScaleFactor;

	int nnew = 0;

	// Search matches with epipolar restriction and triangulate
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKF2 = vpKFs[i];
		if (pKF2 == mpLastKeyFrame) {
			continue;
		}

		// Check first that baseline is not too short
		cv::Mat Ow2 = pKF2->GetCameraCenter();
		cv::Mat vBaseline = Ow2 - Ow1;
		const float baseline = cv::norm(vBaseline);

		if ((mSensor != System::MONOCULAR) || (mSensor != System::IMU_MONOCULAR)) {
			if (baseline < pKF2->mb) {
				continue;
			}
		}
		else {
			const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
			const float ratioBaselineDepth = baseline / medianDepthKF2;

			if (ratioBaselineDepth < 0.01) {
				continue;
			}
		}

		// Compute Fundamental Matrix
		cv::Mat F12 = ComputeF12(mpLastKeyFrame, pKF2);

		// Search matches that fullfil epipolar constraint
		vector<pair<size_t, size_t> > vMatchedIndices;
		matcher.SearchForNeuralTriangulation(
			mpLastKeyFrame, pKF2, F12, vMatchedIndices, false);

		cv::Mat Rcw2 = pKF2->GetRotation();
		cv::Mat Rwc2 = Rcw2.t();
		cv::Mat tcw2 = pKF2->GetTranslation();
		cv::Mat Tcw2(3, 4, CV_32F);
		Rcw2.copyTo(Tcw2.colRange(0, 3));
		tcw2.copyTo(Tcw2.col(3));

		const float &fx2 = pKF2->fx;
		const float &fy2 = pKF2->fy;
		const float &cx2 = pKF2->cx;
		const float &cy2 = pKF2->cy;
		const float &invfx2 = pKF2->invfx;
		const float &invfy2 = pKF2->invfy;

		// Triangulate each match
		const int nmatches = vMatchedIndices.size();
		for (int ikp = 0; ikp < nmatches; ikp++) {
			const int &idx1 = vMatchedIndices[ikp].first;
			const int &idx2 = vMatchedIndices[ikp].second;

			const cv::KeyPoint &kp1 = mpLastKeyFrame->mvKeysUn[idx1];
			const float kp1_ur = mpLastKeyFrame->mvuRight[idx1];
			bool bStereo1 = kp1_ur >= 0;

			const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
			const float kp2_ur = pKF2->mvuRight[idx2];
			bool bStereo2 = kp2_ur >= 0;

			// Check parallax between rays
			cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
			cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

			cv::Mat ray1 = Rwc1 * xn1;
			cv::Mat ray2 = Rwc2 * xn2;
			const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

			float cosParallaxStereo = cosParallaxRays + 1;
			float cosParallaxStereo1 = cosParallaxStereo;
			float cosParallaxStereo2 = cosParallaxStereo;

			if (bStereo1) {
				cosParallaxStereo1 = cos(2 * atan2(mpLastKeyFrame->mb / 2, mpLastKeyFrame->mvDepth[idx1]));
			}
			else if (bStereo2) {
				cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));
			}

			cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

			cv::Mat x3D;
			if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0
				&& (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
				// Linear Triangulation Method
				cv::Mat A(4, 4, CV_32F);
				A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
				A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
				A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
				A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

				cv::Mat w, u, vt;
				cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

				x3D = vt.row(3).t();

				if (x3D.at<float>(3) == 0) {
					continue;
				}

				// Euclidean coordinates
				x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

			}
			else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
				x3D = mpLastKeyFrame->UnprojectStereo(idx1);
			}
			else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
				x3D = pKF2->UnprojectStereo(idx2);
			}
			else {
				continue;
			} //No stereo and very low parallax

			cv::Mat x3Dt = x3D.t();

			//Check triangulation in front of cameras
			float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
			if (z1 <= 0) {
				continue;
			}

			float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
			if (z2 <= 0) {
				continue;
			}

			//Check reprojection error in first keyframe
			const float &sigmaSquare1 = mpLastKeyFrame->mvLevelSigma2[kp1.octave];
			const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
			const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
			const float invz1 = 1.0 / z1;

			if (!bStereo1) {
				float u1 = fx1 * x1 * invz1 + cx1;
				float v1 = fy1 * y1 * invz1 + cy1;
				float errX1 = u1 - kp1.pt.x;
				float errY1 = v1 - kp1.pt.y;
				if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1) {
					continue;
				}
			}
			else {
				float u1 = fx1 * x1 * invz1 + cx1;
				float u1_r = u1 - mpLastKeyFrame->mbf * invz1;
				float v1 = fy1 * y1 * invz1 + cy1;
				float errX1 = u1 - kp1.pt.x;
				float errY1 = v1 - kp1.pt.y;
				float errX1_r = u1_r - kp1_ur;
				if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1) {
					continue;
				}
			}

			//Check reprojection error in second keyframe
			const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
			const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
			const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
			const float invz2 = 1.0 / z2;
			if (!bStereo2) {
				float u2 = fx2 * x2 * invz2 + cx2;
				float v2 = fy2 * y2 * invz2 + cy2;
				float errX2 = u2 - kp2.pt.x;
				float errY2 = v2 - kp2.pt.y;
				if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2) {
					continue;
				}
			}
			else {
				float u2 = fx2 * x2 * invz2 + cx2;
				float u2_r = u2 - mpLastKeyFrame->mbf * invz2;
				float v2 = fy2 * y2 * invz2 + cy2;
				float errX2 = u2 - kp2.pt.x;
				float errY2 = v2 - kp2.pt.y;
				float errX2_r = u2_r - kp2_ur;
				if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2) {
					continue;
				}
			}

			//Check scale consistency
			cv::Mat normal1 = x3D - Ow1;
			float dist1 = cv::norm(normal1);

			cv::Mat normal2 = x3D - Ow2;
			float dist2 = cv::norm(normal2);

			if (dist1 == 0 || dist2 == 0) {
				continue;
			}

			const float ratioDist = dist2 / dist1;
			const float ratioOctave = mpLastKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

			if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor) {
				continue;
			}

			// Triangulation is succesfull
			MapPoint *pMP = new MapPoint(x3D, mpLastKeyFrame, mpAtlas->GetCurrentMap());

			pMP->AddObservation(mpLastKeyFrame, idx1);
			pMP->AddObservation(pKF2, idx2);

			mpLastKeyFrame->AddMapPoint(pMP, idx1);
			pKF2->AddMapPoint(pMP, idx2);

			pMP->ComputeDistinctiveDescriptors();

			pMP->UpdateNormalAndDepth();

			mpAtlas->AddMapPoint(pMP);
			nnew++;
		}
	}
	TrackReferenceKeyFrame();
}

void Tracking::NewDataset()
{
	mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
	return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
	return mnMatchesInliers;
}

void Tracking::LoadEKFReading(const Eigen::Isometry3d &T_e0_ej,
                              const double &timeEKF,
                              bool good_EKF,
                              const Eigen::Isometry3d &T_g0_gj,
                              const Eigen::Matrix<double, 6, 1> &V_e)
{
	mCurT_e0_ej = T_e0_ej;
	mCurTimeEKF = timeEKF;
	mGood_EKF = good_EKF;
	mCurT_g0_gj = T_g0_gj;
	mV_e = V_e;
}
IMU::Calib Tracking::GetExtrinsicPara()
{
	std::lock_guard<std::mutex> lock(mMutexExtrinsic);
	if (mpImuCalib) {
		return *mpImuCalib;
	}
	else { return IMU::Calib(); }
}
void Tracking::SetExtrinsicPara(Calib &calib)
{
	std::lock_guard<std::mutex> lock(mMutexExtrinsic);
	delete mpImuCalib;
	mpImuCalib = new IMU::Calib(calib);
}

DVLGroPreIntegration *Tracking::getLossIntegrationRef()
{
	std::lock_guard<std::mutex> lock(mLossIntegrationRefMutex);
	return mpDvlPreintegratedFromLastKFBeforeLost;
}
cv::Mat Tracking::GrabImageStereoDvlKLT(const Mat &imRectLeft,
                                        const Mat &imRectRight,
                                        const double &timestamp,
                                        bool bDvl,
                                        string filename)
{
	mImLeft = imRectLeft.clone();
	cv::Mat imGrayRight = imRectRight.clone();
	mImRight = imRectRight.clone();

	cv::resize(mImLeft, mImLeft, cv::Size(mImLeft.cols * mImageScale, mImLeft.rows * mImageScale));
	cv::resize(imGrayRight, imGrayRight, cv::Size(imGrayRight.cols * mImageScale, imGrayRight.rows * mImageScale));
	cv::resize(mImRight, mImRight, cv::Size(mImRight.cols * mImageScale, mImRight.rows * mImageScale));

	mCurrentFrame = Frame(mImLeft,
	                      imGrayRight,
		                      timestamp,
		                      *mpFeatureFrontend,
		                      mK,
	                      mDistCoef,
	                      mbf,
	                      mThFarDepth,
	                      mThCloseDepth,
	                      mStereoMaxVerticalError,
	                      mpCamera,
	                      bDvl,
	                      &mLastFrame,
	                      GetExtrinsicPara());


	TrackKLT();

	return mCurrentFrame.mTcw.clone();
}
void Tracking::SetBeamOrientation(const Eigen::Vector4d &alpha, const Eigen::Vector4d &beta)
{
	std::lock_guard<std::mutex> lock(mBeamOriMutex);
	mAlpha = alpha;
	mBeta = beta;
}
void Tracking::GetBeamOrientation(Eigen::Vector4d &alpha, Eigen::Vector4d &beta)
{
	std::lock_guard<std::mutex> lock(mBeamOriMutex);
	alpha = mAlpha;
	beta = mBeta;
}

set<KeyFrame*, KFComparator> Tracking::getMvpLossKf()
{
    std::shared_lock<std::shared_mutex> lock(mLossKFMutex);
    return mvpLossKF;
}

void Tracking::clearPartLossKF()
{
    std::shared_lock<std::shared_mutex> lock(mLossKFMutex);
    if (mvpLossKF.size() > 10) {
        auto it = mvpLossKF.begin();
        std::advance(it, mvpLossKF.size() - 10); // Move iterator to 5th element from the end
        mvpLossKF.erase(mvpLossKF.begin(), it); // Erase elements from the beginning to the iterator
    }
}

void Tracking::setOptV(const Eigen::Vector3d &v)
{
    std::unique_lock<std::shared_mutex> lock(mVMutex);
    mVd_opt = v;
}

void Tracking::getOptV(Eigen::Vector3d &v)
{
    std::shared_lock<std::shared_mutex> lock(mVMutex);
    v = mVd_opt;
}

} //namespace ORB_SLAM
