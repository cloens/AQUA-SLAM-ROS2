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

#include "Frame.h"

#ifndef AQUA_HAS_GTSAM_DYNAMIC
#include "G2oTypes.h"
#endif
#include "MapPoint.h"
#include "KeyFrame.h"
#include "NeuralFeatureFrontend.h"
#include "Converter.h"
#include "FeatureMatcher.h"
#include "GeometricCamera.h"

#include <opencv2/core/eigen.hpp>

#include <thread>
#include <cmath>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId = 0;

bool Frame::mbInitialComputations = true;

float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;

float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;

float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;


Frame::Frame()
	: mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL),
	  mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbImuPreintegrated(false), mTimeStamp(0)
{
}

//Copy Constructor
Frame::Frame(const Frame &frame)
	: mpcpi(frame.mpcpi),
	  mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
	  mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
	  mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
	  mvDepth(frame.mvDepth),
	  mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
	  mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier),
	  mnCloseMPs(frame.mnCloseMPs),
	  mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame),
	  mImuBias(frame.mImuBias),
	  mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
	  mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
	  mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile),
	  mnDataset(frame.mnDataset),
	  mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame),
	  mpLastKeyFrame(frame.mpLastKeyFrame), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
	  mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
	  monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
	  mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
	  mTlr(frame.mTlr.clone()), mRlr(frame.mRlr.clone()), mtlr(frame.mtlr.clone()), mTrl(frame.mTrl.clone()),
	  mTimeStereoMatch(frame.mTimeStereoMatch), mTimeFeatureExtraction(frame.mTimeFeatureExtraction), mT_e0_ej(frame.mT_e0_ej),
	  mTime_ekf(frame.mTime_ekf), mGood_EKF(frame.mGood_EKF), mT_e_c(frame.mT_e_c),
	  mT_e_g(frame.mT_e_g), mT_g0_gj(frame.mT_g0_gj), mV_e(frame.mV_e), mbDVL(frame.mbDVL),
	  mpDvlPreintegrationFrame(frame.mpDvlPreintegrationFrame),
	  mpDvlPreintegrationKeyFrame(frame.mpDvlPreintegrationKeyFrame), mpDvlPreintegrationLossRefKF(frame.mpDvlPreintegrationLossRefKF),
      mpLossRefKF(frame.mpLossRefKF), mT_c0_cj_dvl(frame.mT_c0_cj_dvl), mImuCalib(frame.mImuCalib), mpExtrinsic_mutex(frame.mpExtrinsic_mutex),
      mPoorVision(frame.mPoorVision)
{
	for (int i = 0; i < FRAME_GRID_COLS; i++)
		for (int j = 0; j < FRAME_GRID_ROWS; j++) {
			mGrid[i][j] = frame.mGrid[i][j];
			if (frame.Nleft > 0) {
				mGridRight[i][j] = frame.mGridRight[i][j];
			}
		}

	if (!frame.mTcw.empty()) {
		SetPose(frame.mTcw);
	}

	if (!frame.mVw.empty()) {
		mVw = frame.mVw.clone();
	}

	mmProjectPoints = frame.mmProjectPoints;
	mmMatchedInImage = frame.mmMatchedInImage;

	frame.imgLeft.copyTo(imgLeft);
	frame.imgRight.copyTo(imgRight);
	if (!frame.imgDepthScaled.empty())
		frame.imgDepthScaled.copyTo(imgDepthScaled);
}

Frame::Frame(const cv::Mat &imLeft,
			 const cv::Mat &imRight,
				 const double &timeStamp,
				 NeuralFeatureFrontend &featureFrontend,
			 cv::Mat &K,
			 cv::Mat &distCoef,
			 const float &bf,
			 const float &thFarDepth,
			 const float &thCloseDepth,
			 const float &stereoMaxVerticalError,
			 GeometricCamera *pCamera,
			 bool bDVL,
			 Frame *pPrevF,
			 const IMU::Calib &ImuCalib)
		: mpcpi(NULL),
	  mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thFarDepth), mCloseThDepth(thCloseDepth),
	  mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL),
	  mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbImuPreintegrated(false),
	  mpCamera(pCamera), mpCamera2(nullptr), mTimeStereoMatch(0), mTimeFeatureExtraction(0), mbDVL(bDVL)
{
//	imLeft.copyTo(imgLeft);
	// Frame ID
	mnId = nNextId++;
	// save image into Frame
	imgLeft = imLeft.clone();
	imgRight = imRight.clone();

	cv::Mat imgLeftGray = imLeft.clone();
	cv::Mat imgRightGray = imRight.clone();
	if (imgLeftGray.channels() == 3) {
		// cv::cvtColor(imgLeftGray, imgLeftGray, CV_BGR2GRAY);  // original
		// cv::cvtColor(imgRightGray, imgRightGray, CV_BGR2GRAY);  // original
		cv::cvtColor(imgLeftGray, imgLeftGray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(imgRightGray, imgRightGray, cv::COLOR_BGR2GRAY);
	}
	else if (imgLeftGray.channels() == 4) {
		// cv::cvtColor(imgLeftGray, imgLeftGray, CV_BGRA2GRAY);  // original
		// cv::cvtColor(imgRightGray, imgRightGray, CV_BGRA2GRAY);  // original
		cv::cvtColor(imgLeftGray, imgLeftGray, cv::COLOR_BGRA2GRAY);
		cv::cvtColor(imgRightGray, imgRightGray, cv::COLOR_BGRA2GRAY);
	}

	// Scale Level Info
	mnScaleLevels = 1;
	mfScaleFactor = 1.0F;
	mfLogScaleFactor = 0.0F;
	mvScaleFactors = {1.0F};
	mvInvScaleFactors = {1.0F};
	mvLevelSigma2 = {1.0F};
	mvInvLevelSigma2 = {1.0F};

	// Neural feature extraction
	#ifdef SAVE_TIMES
	std::chrono::steady_clock::time_point featureExtractionStarted = std::chrono::steady_clock::now();
#endif
	ExtractNeuralStereo(imgLeftGray, imgRightGray, featureFrontend,
	                    stereoMaxVerticalError);
#ifdef SAVE_TIMES
	std::chrono::steady_clock::time_point featureExtractionFinished = std::chrono::steady_clock::now();

		mTimeFeatureExtraction = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(featureExtractionFinished - featureExtractionStarted).count();
#endif

	N = mvKeys.size();

#ifdef SAVE_TIMES
	std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
#ifdef SAVE_TIMES
	std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

		mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

	mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
	mvbOutlier = vector<bool>(N, false);
	mmProjectPoints.clear(); // = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
	mmMatchedInImage.clear();

	// This is done only for the first Frame (or after a change in the calibration)
	if (mbInitialComputations) {
		ComputeImageBounds(imgLeftGray);

		mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
		mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

		fx = K.at<float>(0, 0);
		fy = K.at<float>(1, 1);
		cx = K.at<float>(0, 2);
		cy = K.at<float>(1, 2);
		invfx = 1.0f / fx;
		invfy = 1.0f / fy;

		mbInitialComputations = false;
	}

	mb = mbf / fx;

	if (pPrevF) {
		if (!pPrevF->mVw.empty()) {
			mVw = pPrevF->mVw.clone();
		}
	}
	else {
		mVw = cv::Mat::zeros(3, 1, CV_32F);
	}


	mpMutexImu = new std::mutex();
	mpExtrinsic_mutex = new std::mutex();

	//Set no stereo fisheye information
	Nleft = -1;
	Nright = -1;
	mvLeftToRightMatch = vector<int>(0);
	mvRightToLeftMatch = vector<int>(0);
	mTlr = cv::Mat(3, 4, CV_32F);
	mTrl = cv::Mat(3, 4, CV_32F);
	mvStereo3Dpoints = vector<cv::Mat>(0);
	monoLeft = -1;
	monoRight = -1;

	AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
	// Fill matrix with points
	const int nCells = FRAME_GRID_COLS * FRAME_GRID_ROWS;

	int nReserve = 0.5f * N / (nCells);

	for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
		for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
			mGrid[i][j].reserve(nReserve);
			if (Nleft != -1) {
				mGridRight[i][j].reserve(nReserve);
			}
		}

	for (int i = 0; i < N; i++) {
//            if(mvKeysRight.empty()||mvKeys.empty())
//                break;
		const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
											   : (i < Nleft) ? mvKeys[i]
															 : mvKeysRight[i - Nleft];

		int nGridPosX, nGridPosY;
		if (PosInGrid(kp, nGridPosX, nGridPosY)) {
			if (Nleft == -1 || i < Nleft) {
				mGrid[nGridPosX][nGridPosY].push_back(i);
			}
			else {
				mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
			}
		}
	}
}

void Frame::ExtractNeuralStereo(const cv::Mat &imLeftGray,
                               const cv::Mat &imRightGray,
                               NeuralFeatureFrontend &featureFrontend,
                               float stereoMaxVerticalError)
{
	FeatureSet left = featureFrontend.Extract(imLeftGray);
	FeatureSet right = featureFrontend.Extract(imRightGray);
	const auto matches = featureFrontend.Match(
		left, imLeftGray.size(), right, imRightGray.size());
    const float maxDepth = mThDepth > 0.0F
        ? mThDepth : std::numeric_limits<float>::max();
	const StereoGeometry stereo = NeuralFeatureFrontend::FilterRectifiedStereo(
		left, right, matches, mbf, stereoMaxVerticalError, 0.5F, maxDepth);

    mvKeys = std::move(left.keypoints);
    mDescriptors = std::move(left.descriptors);
    mvKeysRight = std::move(right.keypoints);
    mDescriptorsRight = std::move(right.descriptors);
    N = static_cast<int>(mvKeys.size());
    UndistortKeyPoints();
    mvuRight = stereo.right_u;
    mvDepth = stereo.depth;
}

int Frame::TransferTemporalMapPoints(
    const Frame& previous, const std::vector<NeuralMatch>& inliers)
{
	std::set<int> previousIndices;
	std::set<int> currentIndices;
	std::set<MapPoint*> transferredMapPoints;
	for (MapPoint* mapPoint : mvpMapPoints) {
		if (mapPoint)
			transferredMapPoints.insert(mapPoint);
	}
	int transferred = 0;
	for (const auto& match : inliers) {
		if (match.query_index < 0 ||
		    match.query_index >= static_cast<int>(previous.mvpMapPoints.size()) ||
		    match.train_index < 0 ||
		    match.train_index >= static_cast<int>(mvpMapPoints.size()))
			throw std::runtime_error("temporal MapPoint match index is out of range");
		if (!previousIndices.insert(match.query_index).second ||
		    !currentIndices.insert(match.train_index).second)
			throw std::runtime_error("temporal MapPoint matches are not one-to-one");
		MapPoint* mapPoint = previous.mvpMapPoints[match.query_index];
		if (mapPoint && !mapPoint->isBad() && !mvpMapPoints[match.train_index] &&
		    transferredMapPoints.insert(mapPoint).second) {
			mvpMapPoints[match.train_index] = mapPoint;
			++transferred;
		}
	}
	return transferred;
}

void Frame::SetPose(cv::Mat Tcw)
{
    {
        std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
        mTcw = Tcw.clone();
    }
	UpdatePoseMatrices();
}

const void Frame::GetPose(cv::Mat &Tcw)
{
    std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	Tcw = mTcw.clone();
}

void Frame::SetNewBias(const IMU::Bias &b)
{
	mImuBias = b;
	if (mpImuPreintegrated) {
		mpImuPreintegrated->SetNewBias(b);
	}
	if (mpDvlPreintegrationKeyFrame) {
		mpDvlPreintegrationKeyFrame->SetNewBias(b);
	}
}

void Frame::SetVelocity(const cv::Mat &Vwb)
{
	mVw = Vwb.clone();
}

void Frame::SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb)
{
	{
		std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
		mVw = Vwb.clone();
		cv::Mat Rbw = Rwb.t();
		cv::Mat tbw = -Rbw * twb;
		cv::Mat Tbw = cv::Mat::eye(4, 4, CV_32F);
		Rbw.copyTo(Tbw.rowRange(0, 3).colRange(0, 3));
		tbw.copyTo(Tbw.rowRange(0, 3).col(3));
		mTcw = mImuCalib.Tcb * Tbw;
		mT_c0_cj_dvl = Eigen::Isometry3d::Identity();
		cv::cv2eigen(mTcw, mT_c0_cj_dvl.matrix());
	}
	UpdatePoseMatrices();
}

void Frame::SetDvlPoseVelocity(const cv::Mat &R_c0_gyroj, const cv::Mat &c0_t_c0_dj, const cv::Mat &c0_V_di_dj)
{
	{
		std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
		cv::Mat R_c0_dj;
		cv::Mat R_imu_dvl;
		R_imu_dvl = mImuCalib.mT_imu_dvl.rowRange(0, 3).colRange(0, 3).clone();
		R_c0_dj = R_c0_gyroj * R_imu_dvl;
		mVw = c0_V_di_dj.clone();
		cv::Mat R_dj_c0 = R_c0_dj.t();
		cv::Mat dj_t_dj_c0 = -R_dj_c0 * c0_t_c0_dj;
		cv::Mat T_dj_c0 = cv::Mat::eye(4, 4, CV_32F);
		R_dj_c0.copyTo(T_dj_c0.rowRange(0, 3).colRange(0, 3));
		dj_t_dj_c0.copyTo(T_dj_c0.rowRange(0, 3).col(3));
		mTcw = mImuCalib.mT_c_dvl * T_dj_c0;
		mT_c0_cj_dvl = Eigen::Isometry3d::Identity();
		cv::cv2eigen(mTcw, mT_c0_cj_dvl.matrix());
	}
	UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{
    std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
	mRwc = mRcw.t();
	mtcw = mTcw.rowRange(0, 3).col(3);
	// t_w_c
	mOw = -mRcw.t() * mtcw;
}

cv::Mat Frame::GetImuPosition()
{
	std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	// R_w_c * t_c_b + t_w_c
	return mRwc * mImuCalib.Tcb.rowRange(0, 3).col(3) + mOw;
}

cv::Mat Frame::GetImuRotation()
{
	std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	// R_w_c * R_c_b
	return mRwc * mImuCalib.Tcb.rowRange(0, 3).colRange(0, 3);
}

cv::Mat Frame::GetDvlPosition()
{
	std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	// R_w_c * t_c_dvl + t_w_c = c0_t_c0_dj
	return mRwc * mImuCalib.mT_c_dvl.rowRange(0, 3).col(3) + mOw;
}

cv::Mat Frame::GetDvlRotation()
{
	std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	// R_w_c * R_c_dvl
	return mRwc * mImuCalib.mT_c_dvl.rowRange(0, 3).colRange(0, 3);
}

cv::Mat Frame::GetDvlVelocity() const
{
	std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	if (mVw.empty()||mRwc.empty()) {
		return cv::Mat();
	}
	// R_c0_cj * R_c_dvl = R_c0_dj
	cv::Mat R_w_d = mRwc * mImuCalib.mT_c_dvl.rowRange(0, 3).colRange(0, 3);
	// R_dj_c0
	cv::Mat R_d_w = R_w_d.t();
	return R_d_w * mVw;
}

cv::Mat Frame::GetGyroRotation()
{
	std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	// R_w_c * R_c_imu
	return mRwc * mImuCalib.mT_c_imu.rowRange(0, 3).colRange(0, 3);
}

cv::Mat Frame::GetImuPose()
{
	std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	cv::Mat Twb = cv::Mat::eye(4, 4, CV_32F);
	Twb.rowRange(0, 3).colRange(0, 3) = mRwc * mImuCalib.Tcb.rowRange(0, 3).colRange(0, 3);
	Twb.rowRange(0, 3).col(3) = mRwc * mImuCalib.Tcb.rowRange(0, 3).col(3) + mOw;
	return Twb.clone();
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(pMP->isBad()){
        return false;
    }
	if (Nleft == -1) {
		// cout << "\na";
		pMP->mbTrackInView = false;
		pMP->mTrackProjX = -1;
		pMP->mTrackProjY = -1;

		// 3D in absolute coordinates
		cv::Mat P = pMP->GetWorldPos();

		// cout << "b";

		// 3D in camera coordinates
		const cv::Mat Pc = mRcw * P + mtcw;
		const float Pc_dist = cv::norm(Pc);

		// Check positive depth
		const float &PcZ = Pc.at<float>(2);
		const float invz = 1.0f / PcZ;
		if (PcZ < 0.0f) {
			return false;
		}

		const cv::Point2f uv = mpCamera->project(Pc);

		// cout << "c";

		if (uv.x < mnMinX || uv.x > mnMaxX) {
			return false;
		}
		if (uv.y < mnMinY || uv.y > mnMaxY) {
			return false;
		}

		// cout << "d";
		pMP->mTrackProjX = uv.x;
		pMP->mTrackProjY = uv.y;

		// Check distance is in the scale invariance region of the MapPoint
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		const cv::Mat PO = P - mOw;
		const float dist = cv::norm(PO);

		if (dist < minDistance || dist > maxDistance) {
			return false;
		}

		// cout << "e";

		// Check viewing angle
		cv::Mat Pn = pMP->GetNormal();

		// cout << "f";

		const float viewCos = PO.dot(Pn) / dist;

		if (viewCos < viewingCosLimit) {
			return false;
		}

		// Predict scale in the image
		const int nPredictedLevel = pMP->PredictScale(dist, this);

		// cout << "g";

		// Data used by the tracking
		pMP->mbTrackInView = true;
		pMP->mTrackProjX = uv.x;
		pMP->mTrackProjXR = uv.x - mbf * invz;

		pMP->mTrackDepth = Pc_dist;
		// cout << "h";

		pMP->mTrackProjY = uv.y;
		pMP->mnTrackScaleLevel = nPredictedLevel;
		pMP->mTrackViewCos = viewCos;

		// cout << "i";

		return true;
	}
	else {
		pMP->mbTrackInView = false;
		pMP->mbTrackInViewR = false;
		pMP->mnTrackScaleLevel = -1;
		pMP->mnTrackScaleLevelR = -1;

		pMP->mbTrackInView = isInFrustumChecks(pMP, viewingCosLimit);
		pMP->mbTrackInViewR = isInFrustumChecks(pMP, viewingCosLimit, true);

		return pMP->mbTrackInView || pMP->mbTrackInViewR;
	}
}

bool Frame::ProjectPointDistort(MapPoint *pMP, cv::Point2f &kp, float &u, float &v)
{

	// 3D in absolute coordinates
	cv::Mat P = pMP->GetWorldPos();

	// 3D in camera coordinates
	const cv::Mat Pc = mRcw * P + mtcw;
	const float &PcX = Pc.at<float>(0);
	const float &PcY = Pc.at<float>(1);
	const float &PcZ = Pc.at<float>(2);

	// Check positive depth
	if (PcZ < 0.0f) {
		cout << "Negative depth: " << PcZ << endl;
		return false;
	}

	// Project in image and check it is not outside
	const float invz = 1.0f / PcZ;
	u = fx * PcX * invz + cx;
	v = fy * PcY * invz + cy;

	// cout << "c";

	if (u < mnMinX || u > mnMaxX) {
		return false;
	}
	if (v < mnMinY || v > mnMaxY) {
		return false;
	}

	float u_distort, v_distort;

	float x = (u - cx) * invfx;
	float y = (v - cy) * invfy;
	float r2 = x * x + y * y;
	float k1 = mDistCoef.at<float>(0);
	float k2 = mDistCoef.at<float>(1);
	float p1 = mDistCoef.at<float>(2);
	float p2 = mDistCoef.at<float>(3);
	float k3 = 0;
	if (mDistCoef.total() == 5) {
		k3 = mDistCoef.at<float>(4);
	}

	// Radial distorsion
	float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
	float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

	// Tangential distorsion
	x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
	y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

	u_distort = x_distort * fx + cx;
	v_distort = y_distort * fy + cy;

	u = u_distort;
	v = v_distort;

	kp = cv::Point2f(u, v);

	return true;
}

cv::Mat Frame::inRefCoordinates(cv::Mat pCw)
{
	return mRcw * pCw + mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x,
										const float &y,
										const float &r,
										const int minLevel,
										const int maxLevel,
										const bool bRight) const
{
	vector<size_t> vIndices;
	vIndices.reserve(N);

	float factorX = r;
	float factorY = r;

	/*cout << "fX " << factorX << endl;
cout << "fY " << factorY << endl;*/

	const int nMinCellX = max(0, (int)floor((x - mnMinX - factorX) * mfGridElementWidthInv));
	if (nMinCellX >= FRAME_GRID_COLS) {
		return vIndices;
	}

	const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + factorX) * mfGridElementWidthInv));
	if (nMaxCellX < 0) {
		return vIndices;
	}

	const int nMinCellY = max(0, (int)floor((y - mnMinY - factorY) * mfGridElementHeightInv));
	if (nMinCellY >= FRAME_GRID_ROWS) {
		return vIndices;
	}

	const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + factorY) * mfGridElementHeightInv));
	if (nMaxCellY < 0) {
		return vIndices;
	}

	const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

	for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
		for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
			const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
			if (vCell.empty()) {
				continue;
			}

			for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
				const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
														 : (!bRight) ? mvKeys[vCell[j]]
																	 : mvKeysRight[vCell[j]];
				if (bCheckLevels) {
					if (kpUn.octave < minLevel) {
						continue;
					}
					if (maxLevel >= 0) {
						if (kpUn.octave > maxLevel) {
							continue;
						}
					}
				}

				const float distx = kpUn.pt.x - x;
				const float disty = kpUn.pt.y - y;

				if (fabs(distx) < factorX && fabs(disty) < factorY) {
					vIndices.push_back(vCell[j]);
				}
			}
		}
	}

	return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
	posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
	posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

	//Keypoint's coordinates are undistorted, which could cause to go out of the image
	if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS) {
		return false;
	}

	return true;
}

void Frame::UndistortKeyPoints()
{
	if (mvKeys.empty() || mDistCoef.at<float>(0) == 0.0) {
		mvKeysUn = mvKeys;
		return;
	}

	// Fill matrix with points
	cv::Mat mat(N, 2, CV_32F);

	for (int i = 0; i < N; i++) {
		mat.at<float>(i, 0) = mvKeys[i].pt.x;
		mat.at<float>(i, 1) = mvKeys[i].pt.y;
	}

	// Undistort points
	mat = mat.reshape(2);
	cv::undistortPoints(mat, mat, static_cast<Pinhole *>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
	mat = mat.reshape(1);

	// Fill undistorted keypoint vector
	mvKeysUn.resize(N);
	for (int i = 0; i < N; i++) {
		cv::KeyPoint kp = mvKeys[i];
		kp.pt.x = mat.at<float>(i, 0);
		kp.pt.y = mat.at<float>(i, 1);
		mvKeysUn[i] = kp;
	}
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
	if (mDistCoef.at<float>(0) != 0.0) {
		cv::Mat mat(4, 2, CV_32F);
		mat.at<float>(0, 0) = 0.0;
		mat.at<float>(0, 1) = 0.0;
		mat.at<float>(1, 0) = imLeft.cols;
		mat.at<float>(1, 1) = 0.0;
		mat.at<float>(2, 0) = 0.0;
		mat.at<float>(2, 1) = imLeft.rows;
		mat.at<float>(3, 0) = imLeft.cols;
		mat.at<float>(3, 1) = imLeft.rows;

		mat = mat.reshape(2);
		cv::undistortPoints(mat, mat, static_cast<Pinhole *>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
		mat = mat.reshape(1);

		// Undistort corners
		mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
		mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
		mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
		mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
	}
	else {
		mnMinX = 0.0f;
		mnMaxX = imLeft.cols;
		mnMinY = 0.0f;
		mnMaxY = imLeft.rows;
	}
}

void Frame::ApplyExternalStereoMatches(const std::vector<cv::Point2f>& vLeft,
                                      const std::vector<cv::Point2f>& vRight,
                                      const std::vector<float>& vScore)
{
    (void)vScore;
    mvuRight = vector<float>(N, -1.0f);
    mvDepth = vector<float>(N, -1.0f);
    if (N == 0 || vLeft.empty() || vLeft.size() != vRight.size())
        return;

    const float maxAssocDist2 = 9.0f; // 3px radius
    for (size_t m = 0; m < vLeft.size(); ++m) {
        const float ul = vLeft[m].x;
        const float vl = vLeft[m].y;
        const float ur = vRight[m].x;
        float disparity = ul - ur;
        if (disparity <= 0.5f)
            continue;

        float bestDist2 = maxAssocDist2;
        int bestIdx = -1;
        for (int i = 0; i < N; ++i) {
            const float dx = mvKeysUn[i].pt.x - ul;
            const float dy = mvKeysUn[i].pt.y - vl;
            const float d2 = dx * dx + dy * dy;
            if (d2 < bestDist2) {
                bestDist2 = d2;
                bestIdx = i;
            }
        }
        if (bestIdx < 0)
            continue;

        // Keep measured disparity, attach it to the nearest ORB keypoint.
        const float uL = mvKeysUn[bestIdx].pt.x;
        mvuRight[bestIdx] = uL - disparity;
        mvDepth[bestIdx] = mbf / disparity;
    }
}

void Frame::SetExternalDepth(const cv::Mat& depthScaled)
{
    if (depthScaled.empty()) {
        imgDepthScaled.release();
        return;
    }
    if (depthScaled.type() == CV_32FC1)
        imgDepthScaled = depthScaled.clone();
    else
        depthScaled.convertTo(imgDepthScaled, CV_32FC1);
}

void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
	mvuRight = vector<float>(N, -1);
	mvDepth = vector<float>(N, -1);

	for (int i = 0; i < N; i++) {
		const cv::KeyPoint &kp = mvKeys[i];
		const cv::KeyPoint &kpU = mvKeysUn[i];

		const float &v = kp.pt.y;
		const float &u = kp.pt.x;

		const float d = imDepth.at<float>(v, u);

		if (d > 0) {
			mvDepth[i] = d;
			mvuRight[i] = kpU.pt.x - mbf / d;
		}
	}
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
	const float z = mvDepth[i];
	if (z > 0) {
		const float u = mvKeysUn[i].pt.x;
		const float v = mvKeysUn[i].pt.y;
		const float x = (u - cx) * z * invfx;
		const float y = (v - cy) * z * invfy;
		cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
		return mRwc * x3Dc + mOw;
	}
	else {
		return cv::Mat();
	}
}

bool Frame::imuIsPreintegrated()
{
	unique_lock<std::mutex> lock(*mpMutexImu);
	return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
	unique_lock<std::mutex> lock(*mpMutexImu);
	mbImuPreintegrated = true;
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight)
{
	// 3D in absolute coordinates
	cv::Mat P = pMP->GetWorldPos();

	cv::Mat mR, mt, twc;
	if (bRight) {
		cv::Mat Rrl = mTrl.colRange(0, 3).rowRange(0, 3);
		cv::Mat trl = mTrl.col(3);
		mR = Rrl * mRcw;
		mt = Rrl * mtcw + trl;
		twc = mRwc * mTlr.rowRange(0, 3).col(3) + mOw;
	}
	else {
		mR = mRcw;
		mt = mtcw;
		twc = mOw;
	}

	// 3D in camera coordinates
	cv::Mat Pc = mR * P + mt;
	const float Pc_dist = cv::norm(Pc);
	const float &PcZ = Pc.at<float>(2);

	// Check positive depth
	if (PcZ < 0.0f) {
		return false;
	}

	// Project in image and check it is not outside
	cv::Point2f uv;
	if (bRight) {
		uv = mpCamera2->project(Pc);
	}
	else {
		uv = mpCamera->project(Pc);
	}

	if (uv.x < mnMinX || uv.x > mnMaxX) {
		return false;
	}
	if (uv.y < mnMinY || uv.y > mnMaxY) {
		return false;
	}

	// Check distance is in the scale invariance region of the MapPoint
	const float maxDistance = pMP->GetMaxDistanceInvariance();
	const float minDistance = pMP->GetMinDistanceInvariance();
	const cv::Mat PO = P - twc;
	const float dist = cv::norm(PO);

	if (dist < minDistance || dist > maxDistance) {
		return false;
	}

	// Check viewing angle
	cv::Mat Pn = pMP->GetNormal();

	const float viewCos = PO.dot(Pn) / dist;

	if (viewCos < viewingCosLimit) {
		return false;
	}

	// Predict scale in the image
	const int nPredictedLevel = pMP->PredictScale(dist, this);

	if (bRight) {
		pMP->mTrackProjXR = uv.x;
		pMP->mTrackProjYR = uv.y;
		pMP->mnTrackScaleLevelR = nPredictedLevel;
		pMP->mTrackViewCosR = viewCos;
		pMP->mTrackDepthR = Pc_dist;
	}
	else {
		pMP->mTrackProjX = uv.x;
		pMP->mTrackProjY = uv.y;
		pMP->mnTrackScaleLevel = nPredictedLevel;
		pMP->mTrackViewCos = viewCos;
		pMP->mTrackDepth = Pc_dist;
	}

	return true;
}

cv::Mat Frame::UnprojectStereoFishEye(const int &i)
{
	return mRwc * mvStereo3Dpoints[i] + mOw;
}
void Frame::SetExtrinsicParamters(const IMU::Calib &calib)
{
	std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	mImuCalib = calib;
}
IMU::Calib &Frame::GetExtrinsicParamters()
{
	std::lock_guard<std::mutex> lock(*mpExtrinsic_mutex);
	return mImuCalib;
}

} // namespace ORB_SLAM3
