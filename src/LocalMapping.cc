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

#include "KeyFrame.h"
#include "Atlas.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include "LocalMapping.h"
#include "InertialMaturity.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "DvlGyroOptimizer.h"
#include "Converter.h"
#include "System.h"
#include <sophus/geometry.hpp>

#include<mutex>
#include<chrono>

namespace ORB_SLAM3
{

LocalMapping::LocalMapping(System *pSys,
						   Atlas *pAtlas,
						   DenseMapper* pDenseMapper,
						   const float bMonocular,
						   bool bInertial,
						   bool bDvlGyro,
						   const string &strSettingPath)
	:
	mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false),
	mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false),
	mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
	mbNewInit(false), mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0),
	infoInertial(Eigen::MatrixXd::Zero(9, 9)), mstrSettingPath(strSettingPath), mpDenseMapper(pDenseMapper),
	mbDvlGyro(bDvlGyro)
{
	mnMatchesInliers = 0;

	mbBadImu = false;

	mTinit = 0.f;

	mNumLM = 0;
	mNumKFCulling = 0;

	//DEBUG: times and data from LocalMapping in each frame

//	strSequence = "";//_strSeqName;

	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	double dCalibrated;
	dCalibrated = fSettings["Optimizer.calibrated"];
	mbCalibrated = dCalibrated == 1;
    mInitTranslationThred = fSettings["IMUInitTranslation"];
    mInitRotationThred = fSettings["IMUInitRotation"];
    stringstream ss;
    ss<<" Local Mapper Parameter Setting: "<<endl;
    ss<<" IMUInitTranslation: "<<mInitTranslationThred<<endl;
    ss<<" IMUInitRotation: "<<mInitRotationThred<<endl;
//     ROS_INFO_STREAM(ss.str());  // original


	//f_lm.open("localMapping_times" + strSequence + ".txt");
	/*f_lm.open("localMapping_times.txt");

	f_lm << "# Timestamp KF, Num CovKFs, Num KFs, Num RecentMPs, Num MPs, processKF, MPCulling, CreateMP, SearchNeigh, BA, KFCulling, [numFixKF_LBA]" << endl;
	f_lm << fixed;*/
}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser)
{
	mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
	mpTracker = pTracker;
}

void LocalMapping::Run()
{

	mbFinished = false;
    int new_KF = 0;
    int current_KF_num = 0;
    int pose_graph_kf_num =0;
	while (1) {
		// Tracking will see that Local Mapping is busy
		SetAcceptKeyFrames(false);

		// Check if there are keyframes in the queue
		if (CheckNewKeyFrames() && !mbBadImu) {
            new_KF++;
			// std::cout << "LM" << std::endl;
			std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

			// BoW conversion and insertion in Map
			ProcessNewKeyFrame();
			std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

			// Check recent MapPoints
			MapPointCulling();
			std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

			// Triangulate new MapPoints
			CreateNewMapPoints();
			std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

			// Save here:
			// # Cov KFs
			// # tot Kfs
			// # recent added MPs
			// # tot MPs
			// # localMPs in LBA
			// # fixedKFs in LBA

			mbAbortBA = false;
            // current_KF_num = mpAtlas->GetAllKeyFramesinAllMap().size();

			if (!CheckNewKeyFrames()) {
				// Find more matches in neighbor keyframes and fuse point duplications
				SearchInNeighbors();
			}
			//--
			int num_FixedKF_BA = 0;

			if (!CheckNewKeyFrames() && !stopRequested()) {
				if (mpAtlas->KeyFramesInMap() >= 2) {
					if (mbInertial) {
						if (mpCurrentKeyFrame->GetMap()->isImuInitialized()) {
							Optimizer::LocalInertialBA(mpCurrentKeyFrame,
							                           &mbAbortBA,
							                           mpCurrentKeyFrame->GetMap());
						}
						else if (mpAtlas->KeyFramesInMap() > 2) {
							Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,
							                                 &mbAbortBA,
							                                 mpCurrentKeyFrame->GetMap(),
							                                 num_FixedKF_BA);
						}
					}
					else if (mbDvlGyro) {
						if (mpAtlas->IsIMUCalibrated()) {
							DvlGyroOptimizer::LocalDVLIMUBundleAdjustment(mpAtlas,
							                                                  mpCurrentKeyFrame,
							                                                  &mbAbortBA,
							                                                  mpCurrentKeyFrame->GetMap(),
							                                                  num_FixedKF_BA,
							                                                  mpTracker->mlamda_DVL,
							                                                  mpTracker->mlamda_visual);
							mpTracker->UpdateFrameDVLGyro(mpCurrentKeyFrame->GetImuBias(),
							                                  mpCurrentKeyFrame);
						}
						else if (mpAtlas->KeyFramesInMap() > 2) {
							Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,
							                                 &mbAbortBA,
							                                 mpCurrentKeyFrame->GetMap(),
							                                 num_FixedKF_BA);
						}
					}
					else if (mpAtlas->KeyFramesInMap() > 2) {
						Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,
						                                 &mbAbortBA,
						                                 mpCurrentKeyFrame->GetMap(),
						                                 num_FixedKF_BA);
					}
				}

				if (mpTracker->mState == Tracking::OK) {
					if (mbInertial) {
						Map* pCurrentMap = mpCurrentKeyFrame->GetMap();
						if (!pCurrentMap->isImuInitialized()) {
							InitializeIMU();
						}
						else {
							const double elapsed = mpCurrentKeyFrame->mTimeStamp - mFirstTs;
							switch (NextInertialRefinementStage(
								elapsed,
								pCurrentMap->GetIniertialBA1(),
								pCurrentMap->GetIniertialBA2())) {
							case InertialRefinementStage::BA1:
								InitializeIMU(1.0F, 1.0e5F, true);
								pCurrentMap->SetIniertialBA1();
								break;
							case InertialRefinementStage::BA2:
								InitializeIMU(0.0F, 0.0F, true);
								pCurrentMap->SetIniertialBA2();
								break;
							case InertialRefinementStage::None:
								break;
							}
						}
					}
					else if (mbDvlGyro) {
						if (mpAtlas->GetAllMaps().size() == 1) {
							auto dis = GetTravelDistance();
							if (!mpAtlas->isImuInitialized()) {
								if (mpAtlas->KeyFramesInMap() > 10)
									InitializeDvlIMU();
							}
							else if (!mpAtlas->IsIMUCalibrated()
							         && dis.first > mInitTranslationThred
							         && dis.second > mInitRotationThred) {
								InitializeDvlIMU();
							}
						}
						else if (mpAtlas->GetAllKeyFramesinAllMap().size() > 200
						         && current_KF_num < 200) {
							RefineGravityDvlIMU();
							current_KF_num = mpAtlas->GetAllKeyFramesinAllMap().size();
						}
						else if (mpAtlas->GetAllKeyFramesinAllMap().size() > 400
						         && current_KF_num < 400) {
							RefineGravityDvlIMU();
							current_KF_num = mpAtlas->GetAllKeyFramesinAllMap().size();
						}
					}
				}




                // Check redundant local Keyframes
				// if(mpTracker->mCalibrated){
				// 	KeyFrameCulling();
				// }

			}
            // mpTracker->mpRosHandler->PublishIntegration(mpAtlas);
            // if(mpAtlas->GetAllMaps().front()->isImuInitialized()&&new_KF>=50){
            //     mpTracker->mpRosHandler->UpdateMap(mpAtlas);
            //     new_KF = 0;
            // }

			mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);


		}
		else if (Stop() && !mbBadImu) {
			// Safe area to stop
			while (isStopped() && !CheckFinish()) {
				// cout << "LM: usleep if is stopped" << endl;
				usleep(3000);
			}
			if (CheckFinish()) {
				break;
			}
		}

// 		// ROS_INFO_STREAM("Local Mapping KF in queue: "<<KeyframesInQueue());  // original
		ResetIfRequested();
// 		// ROS_INFO_STREAM("Local Mapping after reset KF in queue: "<<KeyframesInQueue());  // original

		// Tracking will see that Local Mapping is busy
		SetAcceptKeyFrames(true);

		if (CheckFinish()) {
			break;
		}

		// cout << "LM: normal usleep" << endl;
		usleep(3000);
	}

	//f_lm.close();

	SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
	unique_lock<mutex> lock(mMutexNewKFs);
	mlNewKeyFrames.push_back(pKF);
	mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames()
{
	unique_lock<mutex> lock(mMutexNewKFs);
	return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
	//cout << "ProcessNewKeyFrame: " << mlNewKeyFrames.size() << endl;
	{
		unique_lock<mutex> lock(mMutexNewKFs);
		mpCurrentKeyFrame = mlNewKeyFrames.front();
		mlNewKeyFrames.pop_front();
	}

	// Compute Bags of Words structures
	mpCurrentKeyFrame->ComputeBoW();

	// Associate MapPoints to the new keyframe and update normal and descriptor
	const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

	for (size_t i = 0; i < vpMapPointMatches.size(); i++) {
		MapPoint *pMP = vpMapPointMatches[i];
		if (pMP) {
			if (!pMP->isBad()) {
				// for old existing points, we need to AddObservation() to current Key frame
				// but we have not did that in CreateNewKeyFrame(), so we AddObservation() here
				if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) {
					pMP->AddObservation(mpCurrentKeyFrame, i);
					pMP->UpdateNormalAndDepth();
					pMP->ComputeDistinctiveDescriptors();
				}
					// for new points, we need to AddObservation() to current Key frame
					// and we have did that in CreateNewKeyFrame()
				else // this can only happen for new stereo points inserted by the Tracking
				{
					mlpRecentAddedMapPoints.push_back(pMP);
				}
			}
		}
	}

	// Update links in the Covisibility Graph
	mpCurrentKeyFrame->UpdateConnections();

	// Insert Keyframe in Map
	mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::EmptyQueue()
{
	while (CheckNewKeyFrames()) {
		ProcessNewKeyFrame();
	}
}

void LocalMapping::MapPointCulling()
{
	// Check Recent Added MapPoints
	list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
	const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

	int nThObs;
	if (mbMonocular) {
		nThObs = 2;
	}
	else {
		nThObs = 3;
	} // MODIFICATION_STEREO_IMU here 3
	const int cnThObs = nThObs;

	int borrar = mlpRecentAddedMapPoints.size();

	while (lit != mlpRecentAddedMapPoints.end()) {
		MapPoint *pMP = *lit;

		if (pMP->isBad()) {
			lit = mlpRecentAddedMapPoints.erase(lit);
		}
		else if (pMP->GetFoundRatio() < 0.25f) {
			pMP->SetBadFlag();
			lit = mlpRecentAddedMapPoints.erase(lit);
		}
		else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs) {
			pMP->SetBadFlag();
			lit = mlpRecentAddedMapPoints.erase(lit);
		}
		else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3) {
			lit = mlpRecentAddedMapPoints.erase(lit);
		}
		else {
			lit++;
			borrar--;
		}
	}
	//cout << "erase MP: " << borrar << endl;
}

void LocalMapping::CreateNewMapPoints()
{
	// Retrieve neighbor keyframes in covisibility graph
	int nn = 10;
	// For stereo inertial case
	if (mbMonocular) {
		nn = 20;
	}
	vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

	if (mbInertial) {
		KeyFrame *pKF = mpCurrentKeyFrame;
		int count = 0;
		while ((vpNeighKFs.size() <= nn) && (pKF->mPrevKF) && (count++ < nn)) {
			vector<KeyFrame *>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
			if (it == vpNeighKFs.end()) {
				vpNeighKFs.push_back(pKF->mPrevKF);
			}
			pKF = pKF->mPrevKF;
		}
	}

	float th = 0.6f;

	ORBmatcher matcher(th, false);

	cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
	cv::Mat Rwc1 = Rcw1.t();
	cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
	cv::Mat Tcw1(3, 4, CV_32F);
	Rcw1.copyTo(Tcw1.colRange(0, 3));
	tcw1.copyTo(Tcw1.col(3));
	cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

	const float &fx1 = mpCurrentKeyFrame->fx;
	const float &fy1 = mpCurrentKeyFrame->fy;
	const float &cx1 = mpCurrentKeyFrame->cx;
	const float &cy1 = mpCurrentKeyFrame->cy;
	const float &invfx1 = mpCurrentKeyFrame->invfx;
	const float &invfy1 = mpCurrentKeyFrame->invfy;

	const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

	// Search matches with epipolar restriction and triangulate
	for (size_t i = 0; i < vpNeighKFs.size(); i++) {
		if (i > 0 && CheckNewKeyFrames()) {// && (mnMatchesInliers>50))
			return;
		}

		KeyFrame *pKF2 = vpNeighKFs[i];

		GeometricCamera *pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

		// Check first that baseline is not too short
		cv::Mat Ow2 = pKF2->GetCameraCenter();
		cv::Mat vBaseline = Ow2 - Ow1;
		const float baseline = cv::norm(vBaseline);

		if (!mbMonocular) {
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
		cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

		// Search matches that fullfil epipolar constraint
		vector<pair<size_t, size_t> > vMatchedIndices;
		bool bCoarse = mbInertial &&
			((!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && mpCurrentKeyFrame->GetMap()->GetIniertialBA1()) ||
				mpTracker->mState == Tracking::RECENTLY_LOST);
		matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false, bCoarse);

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

			const cv::KeyPoint &kp1 = (mpCurrentKeyFrame->NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
																	   : (idx1 < mpCurrentKeyFrame->NLeft)
																		 ? mpCurrentKeyFrame->mvKeys[idx1]
																		 : mpCurrentKeyFrame->mvKeysRight[idx1
						- mpCurrentKeyFrame->NLeft];
			const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
			bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur >= 0);
			const bool bRight1 = (mpCurrentKeyFrame->NLeft == -1 || idx1 < mpCurrentKeyFrame->NLeft) ? false
																									 : true;

			const cv::KeyPoint &kp2 = (pKF2->NLeft == -1) ? pKF2->mvKeysUn[idx2]
														  : (idx2 < pKF2->NLeft) ? pKF2->mvKeys[idx2]
																				 : pKF2->mvKeysRight[idx2
						- pKF2->NLeft];

			const float kp2_ur = pKF2->mvuRight[idx2];
			bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur >= 0);
			const bool bRight2 = (pKF2->NLeft == -1 || idx2 < pKF2->NLeft) ? false
																		   : true;

			if (mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2) {
				if (bRight1 && bRight2) {
					Rcw1 = mpCurrentKeyFrame->GetRightRotation();
					Rwc1 = Rcw1.t();
					tcw1 = mpCurrentKeyFrame->GetRightTranslation();
					Tcw1 = mpCurrentKeyFrame->GetRightPose();
					Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

					Rcw2 = pKF2->GetRightRotation();
					Rwc2 = Rcw2.t();
					tcw2 = pKF2->GetRightTranslation();
					Tcw2 = pKF2->GetRightPose();
					Ow2 = pKF2->GetRightCameraCenter();

					pCamera1 = mpCurrentKeyFrame->mpCamera2;
					pCamera2 = pKF2->mpCamera2;
				}
				else if (bRight1 && !bRight2) {
					Rcw1 = mpCurrentKeyFrame->GetRightRotation();
					Rwc1 = Rcw1.t();
					tcw1 = mpCurrentKeyFrame->GetRightTranslation();
					Tcw1 = mpCurrentKeyFrame->GetRightPose();
					Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

					Rcw2 = pKF2->GetRotation();
					Rwc2 = Rcw2.t();
					tcw2 = pKF2->GetTranslation();
					Tcw2 = pKF2->GetPose();
					Ow2 = pKF2->GetCameraCenter();

					pCamera1 = mpCurrentKeyFrame->mpCamera2;
					pCamera2 = pKF2->mpCamera;
				}
				else if (!bRight1 && bRight2) {
					Rcw1 = mpCurrentKeyFrame->GetRotation();
					Rwc1 = Rcw1.t();
					tcw1 = mpCurrentKeyFrame->GetTranslation();
					Tcw1 = mpCurrentKeyFrame->GetPose();
					Ow1 = mpCurrentKeyFrame->GetCameraCenter();

					Rcw2 = pKF2->GetRightRotation();
					Rwc2 = Rcw2.t();
					tcw2 = pKF2->GetRightTranslation();
					Tcw2 = pKF2->GetRightPose();
					Ow2 = pKF2->GetRightCameraCenter();

					pCamera1 = mpCurrentKeyFrame->mpCamera;
					pCamera2 = pKF2->mpCamera2;
				}
				else {
					Rcw1 = mpCurrentKeyFrame->GetRotation();
					Rwc1 = Rcw1.t();
					tcw1 = mpCurrentKeyFrame->GetTranslation();
					Tcw1 = mpCurrentKeyFrame->GetPose();
					Ow1 = mpCurrentKeyFrame->GetCameraCenter();

					Rcw2 = pKF2->GetRotation();
					Rwc2 = Rcw2.t();
					tcw2 = pKF2->GetTranslation();
					Tcw2 = pKF2->GetPose();
					Ow2 = pKF2->GetCameraCenter();

					pCamera1 = mpCurrentKeyFrame->mpCamera;
					pCamera2 = pKF2->mpCamera;
				}
			}

			// Check parallax between rays
			cv::Mat xn1 = pCamera1->unprojectMat(kp1.pt);
			cv::Mat xn2 = pCamera2->unprojectMat(kp2.pt);

			cv::Mat ray1 = Rwc1 * xn1;
			cv::Mat ray2 = Rwc2 * xn2;
			const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

			float cosParallaxStereo = cosParallaxRays + 1;
			float cosParallaxStereo1 = cosParallaxStereo;
			float cosParallaxStereo2 = cosParallaxStereo;

			if (bStereo1) {
				cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2, mpCurrentKeyFrame->mvDepth[idx1]));
			}
			else if (bStereo2) {
				cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));
			}

			cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

			cv::Mat x3D;
			if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && (bStereo1 || bStereo2 ||
				(cosParallaxRays < 0.9998 && mbInertial) || (cosParallaxRays < 0.9998 && !mbInertial))) {
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
				x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
			}
			else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
				x3D = pKF2->UnprojectStereo(idx2);
			}
			else {
				continue; //No stereo and very low parallax
			}

			cv::Mat x3Dt = x3D.t();

			if (x3Dt.empty()) { continue; }
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
			const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
			const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
			const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
			const float invz1 = 1.0 / z1;

			if (!bStereo1) {
				cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1, y1, z1));
				float errX1 = uv1.x - kp1.pt.x;
				float errY1 = uv1.y - kp1.pt.y;

				if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1) {
					continue;
				}

			}
			else {
				float u1 = fx1 * x1 * invz1 + cx1;
				float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
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
				cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2, y2, z2));
				float errX2 = uv2.x - kp2.pt.x;
				float errY2 = uv2.y - kp2.pt.y;
				if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2) {
					continue;
				}
			}
			else {
				float u2 = fx2 * x2 * invz2 + cx2;
				float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
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

			if (mbFarPoints && (dist1 >= mThFarPoints || dist2 >= mThFarPoints)) { // MODIFICATION
				continue;
			}

			const float ratioDist = dist2 / dist1;
			const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

			if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor) {
				continue;
			}

			// Triangulation is succesfull
			MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());

			pMP->AddObservation(mpCurrentKeyFrame, idx1);
			pMP->AddObservation(pKF2, idx2);

			mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
			pKF2->AddMapPoint(pMP, idx2);

			pMP->ComputeDistinctiveDescriptors();

			pMP->UpdateNormalAndDepth();

			mpAtlas->AddMapPoint(pMP);
			mlpRecentAddedMapPoints.push_back(pMP);
		}
	}
}

void LocalMapping::SearchInNeighbors()
{
	// Retrieve neighbor keyframes
	int nn = 10;
	if (mbMonocular) {
		nn = 20;
	}
	const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
	vector<KeyFrame *> vpTargetKFs;
	for (vector<KeyFrame *>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++) {
		KeyFrame *pKFi = *vit;
		if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId) {
			continue;
		}
		vpTargetKFs.push_back(pKFi);
		pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
	}

	// Add some covisible of covisible
	// Extend to some second neighbors if abort is not requested
	for (int i = 0, imax = vpTargetKFs.size(); i < imax; i++) {
		const vector<KeyFrame *> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
		for (vector<KeyFrame *>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end();
			 vit2 != vend2; vit2++) {
			KeyFrame *pKFi2 = *vit2;
			if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId
				|| pKFi2->mnId == mpCurrentKeyFrame->mnId) {
				continue;
			}
			vpTargetKFs.push_back(pKFi2);
			pKFi2->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
		}
		if (mbAbortBA) {
			break;
		}
	}

	// Extend to temporal neighbors
	if (mbInertial) {
		KeyFrame *pKFi = mpCurrentKeyFrame->mPrevKF;
		while (vpTargetKFs.size() < 20 && pKFi) {
			if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId) {
				pKFi = pKFi->mPrevKF;
				continue;
			}
			vpTargetKFs.push_back(pKFi);
			pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
			pKFi = pKFi->mPrevKF;
		}
	}

	// Search matches by projection from current KF in target KFs
	ORBmatcher matcher;
	vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
	for (vector<KeyFrame *>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++) {
		KeyFrame *pKFi = *vit;

		matcher.Fuse(pKFi, vpMapPointMatches);
		if (pKFi->NLeft != -1) { matcher.Fuse(pKFi, vpMapPointMatches, true); }
	}

	if (mbAbortBA) {
		return;
	}

	// Search matches by projection from target KFs in current KF
	vector<MapPoint *> vpFuseCandidates;
	vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

	for (vector<KeyFrame *>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF;
		 vitKF++) {
		KeyFrame *pKFi = *vitKF;

		vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

		for (vector<MapPoint *>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end();
			 vitMP != vendMP; vitMP++) {
			MapPoint *pMP = *vitMP;
			if (!pMP) {
				continue;
			}
			if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId) {
				continue;
			}
			pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
			vpFuseCandidates.push_back(pMP);
		}
	}

	matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);
	if (mpCurrentKeyFrame->NLeft != -1) { matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates, true); }


	// Update points
	vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
	for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
		MapPoint *pMP = vpMapPointMatches[i];
		if (pMP) {
			if (!pMP->isBad()) {
				pMP->ComputeDistinctiveDescriptors();
				pMP->UpdateNormalAndDepth();
			}
		}
	}

	// Update connections in covisibility graph
	mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
	cv::Mat R1w = pKF1->GetRotation();
	cv::Mat t1w = pKF1->GetTranslation();
	cv::Mat R2w = pKF2->GetRotation();
	cv::Mat t2w = pKF2->GetTranslation();

	cv::Mat R12 = R1w * R2w.t();
	cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

	cv::Mat t12x = SkewSymmetricMatrix(t12);

	const cv::Mat &K1 = pKF1->mpCamera->toK();
	const cv::Mat &K2 = pKF2->mpCamera->toK();


	return K1.t().inv() * t12x * R12 * K2.inv();
}

void LocalMapping::RequestStop()
{
	unique_lock<mutex> lock(mMutexStop);
	mbStopRequested = true;
	unique_lock<mutex> lock2(mMutexNewKFs);
	mbAbortBA = true;
}

bool LocalMapping::Stop()
{
	unique_lock<mutex> lock(mMutexStop);
	if (mbStopRequested && !mbNotStop) {
		mbStopped = true;
		cout << "Local Mapping STOP" << endl;
		return true;
	}

	return false;
}

bool LocalMapping::isStopped()
{
	unique_lock<mutex> lock(mMutexStop);
	return mbStopped;
}

bool LocalMapping::stopRequested()
{
	unique_lock<mutex> lock(mMutexStop);
	return mbStopRequested;
}

void LocalMapping::Release()
{
	unique_lock<mutex> lock(mMutexStop);
	unique_lock<mutex> lock2(mMutexFinish);
	if (mbFinished) {
		return;
	}
	mbStopped = false;
	mbStopRequested = false;
	for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
		delete *lit;
	mlNewKeyFrames.clear();

	cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
	unique_lock<mutex> lock(mMutexAccept);
	return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
	unique_lock<mutex> lock(mMutexAccept);
	mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
	unique_lock<mutex> lock(mMutexStop);

	if (flag && mbStopped) {
		return false;
	}

	mbNotStop = flag;

	return true;
}

void LocalMapping::InterruptBA()
{
	mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
	// Check redundant keyframes (only local keyframes)
	// A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
	// in at least other 3 keyframes (in the same or finer scale)
	// We only consider close stereo points
	const int Nd = 21; // MODIFICATION_STEREO_IMU 20 This should be the same than that one from LIBA
	mpCurrentKeyFrame->UpdateBestCovisibles();
	vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

	float redundant_th;
	if (!mbInertial) {
		redundant_th = 0.9;
	}
	else if (mbMonocular) {
		redundant_th = 0.9;
	}
	else {
		redundant_th = 0.5;
	}

	const bool bInitImu = mpAtlas->isImuInitialized();
	int count = 0;

	// Compoute last KF from optimizable window:
	unsigned int last_ID;
	if (mbInertial) {
		int count = 0;
		KeyFrame *aux_KF = mpCurrentKeyFrame;
		while (count < Nd && aux_KF->mPrevKF) {
			aux_KF = aux_KF->mPrevKF;
			count++;
		}
		last_ID = aux_KF->mnId;
	}


	for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend;
		 vit++) {
		count++;
		KeyFrame *pKF = *vit;

		if ((pKF->mnId == pKF->GetMap()->GetInitKFid()) || pKF->isBad()) {
			continue;
		}
		const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

		int nObs = 3;
		const int thObs = nObs;
		int nRedundantObservations = 0;
		int nMPs = 0;
		for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
			MapPoint *pMP = vpMapPoints[i];
			if (pMP) {
				if (!pMP->isBad()) {
					if (!mbMonocular) {
						if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0) {
							continue;
						}
					}

					nMPs++;
					if (pMP->Observations() > thObs) {
						const int &scaleLevel = (pKF->NLeft == -1) ? pKF->mvKeysUn[i].octave
																   : (i < pKF->NLeft) ? pKF->mvKeys[i].octave
																					  : pKF->mvKeysRight[i].octave;
						const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
						int nObs = 0;
						for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(),
								 mend = observations.end(); mit != mend; mit++) {
							KeyFrame *pKFi = mit->first;
							if (pKFi == pKF) {
								continue;
							}
							tuple<int, int> indexes = mit->second;
							int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
							int scaleLeveli = -1;
							if (pKFi->NLeft == -1) {
								scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
							}
							else {
								if (leftIndex != -1) {
									scaleLeveli = pKFi->mvKeys[leftIndex].octave;
								}
								if (rightIndex != -1) {
									int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
									scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
																								  : scaleLeveli;
								}
							}

							if (scaleLeveli <= scaleLevel + 1) {
								nObs++;
								if (nObs > thObs) {
									break;
								}
							}
						}
						if (nObs > thObs) {
							nRedundantObservations++;
						}
					}
				}
			}
		}

		if (nRedundantObservations > redundant_th * nMPs) {
			if (mbInertial) {
				if (mpAtlas->KeyFramesInMap() <= Nd) {
					continue;
				}

				if (pKF->mnId > (mpCurrentKeyFrame->mnId - 2)) {
					continue;
				}

				if (pKF->mPrevKF && pKF->mNextKF) {
					const float t = pKF->mNextKF->mTimeStamp - pKF->mPrevKF->mTimeStamp;

					if ((bInitImu && (pKF->mnId < last_ID) && t < 3.) || (t < 0.5)) {
						pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
						pKF->mNextKF->mPrevKF = pKF->mPrevKF;
						pKF->mPrevKF->mNextKF = pKF->mNextKF;
						pKF->mNextKF = NULL;
						pKF->mPrevKF = NULL;
						pKF->SetBadFlag();
					}
					else if (!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()
						&& (cv::norm(pKF->GetImuPosition() - pKF->mPrevKF->GetImuPosition()) < 0.02) && (t < 3)) {
						pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
						pKF->mNextKF->mPrevKF = pKF->mPrevKF;
						pKF->mPrevKF->mNextKF = pKF->mNextKF;
						pKF->mNextKF = NULL;
						pKF->mPrevKF = NULL;
						pKF->SetBadFlag();
					}
				}
			}
			else if (mbDvlGyro) {
				if (mpAtlas->KeyFramesInMap() <= Nd) {
					continue;
				}

				if (pKF->mnId > (mpCurrentKeyFrame->mnId - 2)) {
					continue;
				}

				if (!mbCalibrated) {
					continue;
				}

				if (pKF->mPrevKF && pKF->mNextKF) {
					pKF->mNextKF->mpDvlPreintegrationKeyFrame->MergePrevious2(pKF->mpDvlPreintegrationKeyFrame);
					pKF->mNextKF->mPrevKF = pKF->mPrevKF;
					pKF->mPrevKF->mNextKF = pKF->mNextKF;
					pKF->mNextKF = NULL;
					pKF->mPrevKF = NULL;
					pKF->SetBadFlag();
					cout << "key frame culling! ID: " << pKF->mnId << endl;
				}
			}
			else {
				pKF->SetBadFlag();
			}
		}
		if ((count > 20 && mbAbortBA)
			|| count > 100) // MODIFICATION originally 20 for mbabortBA check just 10 keyframes
		{
			break;
		}
	}
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
	return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
		v.at<float>(2), 0, -v.at<float>(0),
		-v.at<float>(1), v.at<float>(0), 0);
}

void LocalMapping::RequestReset()
{
	{
		unique_lock<mutex> lock(mMutexReset);
		cout << "LM: Map reset recieved" << endl;
		mbResetRequested = true;
	}
	cout << "LM: Map reset, waiting..." << endl;

	while (1) {
		{
			unique_lock<mutex> lock2(mMutexReset);
			if (!mbResetRequested) {
				break;
			}
		}
		usleep(3000);
	}
	cout << "LM: Map reset, Done!!!" << endl;
}

void LocalMapping::RequestResetActiveMap(Map *pMap)
{
	{
		unique_lock<mutex> lock(mMutexReset);
//		cout << "LM: Active map reset recieved" << endl;
		mbResetRequestedActiveMap = true;
		mpMapToReset = pMap;
	}
//	cout << "LM: Active map reset, waiting..." << endl;

	while (1) {
		{
			unique_lock<mutex> lock2(mMutexReset);
			if (!mbResetRequestedActiveMap) {
				break;
			}
		}
		usleep(3000);
	}
	//cout << "LM: Active map reset, Done!!!" << endl;
}

void LocalMapping::ResetIfRequested()
{
	bool executed_reset = false;
	{
		unique_lock<mutex> lock(mMutexReset);
		if (mbResetRequested) {
			executed_reset = true;

			cout << "LM: Reseting Atlas in Local Mapping..." << endl;
			mlNewKeyFrames.clear();
			mlpRecentAddedMapPoints.clear();
			mbResetRequested = false;
			mbResetRequestedActiveMap = false;

			// Inertial parameters
			mTinit = 0.f;
			mbNotBA2 = true;
			mbNotBA1 = true;
			mbBadImu = false;

			mIdxInit = 0;

			//cout << "LM: End reseting Local Mapping..." << endl;
		}

		if (mbResetRequestedActiveMap) {
			executed_reset = true;
//			cout << "LM: Reseting current map in Local Mapping..." << endl;
			mlNewKeyFrames.clear();
			mlpRecentAddedMapPoints.clear();

			// Inertial parameters
			mTinit = 0.f;
			mbNotBA2 = true;
			mbNotBA1 = true;
			mbBadImu = false;

			mbResetRequestedActiveMap = false;
			//cout << "LM: End reseting Local Mapping..." << endl;
		}
	}
	if (executed_reset) {
		//cout << "LM: Reset free the mutex" << endl;
	}

}

void LocalMapping::RequestFinish()
{
	unique_lock<mutex> lock(mMutexFinish);
	mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
	unique_lock<mutex> lock(mMutexFinish);
	return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
	unique_lock<mutex> lock(mMutexFinish);
	mbFinished = true;
	unique_lock<mutex> lock2(mMutexStop);
	mbStopped = true;
}

bool LocalMapping::isFinished()
{
	unique_lock<mutex> lock(mMutexFinish);
	return mbFinished;
}

void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
	if (mbResetRequested) {
		return;
	}

	float minTime;
	int nMinKF;
	if (mbMonocular) {
		minTime = 2.0;
		nMinKF = 10;
	}
	else {
		minTime = 1.0;
		nMinKF = 10;
	}


	if (mpAtlas->KeyFramesInMap() < nMinKF) {
		return;
	}

	// Retrieve all keyframe in temporal order
	list<KeyFrame *> lpKF;
	KeyFrame *pKF = mpCurrentKeyFrame;
	while (pKF->mPrevKF) {
		lpKF.push_front(pKF);
		pKF = pKF->mPrevKF;
	}
	lpKF.push_front(pKF);
	vector<KeyFrame *> vpKF(lpKF.begin(), lpKF.end());

	if (vpKF.size() < nMinKF) {
		return;
	}

	mFirstTs = vpKF.front()->mTimeStamp;
	if (mpCurrentKeyFrame->mTimeStamp - mFirstTs < minTime) {
		return;
	}

	bInitializing = true;

	while (CheckNewKeyFrames()) {
		ProcessNewKeyFrame();
		vpKF.push_back(mpCurrentKeyFrame);
		lpKF.push_back(mpCurrentKeyFrame);
	}

	const int N = vpKF.size();
	IMU::Bias b(0, 0, 0, 0, 0, 0);

	// Compute and KF velocities mRwg estimation
	if (!mpCurrentKeyFrame->GetMap()->isImuInitialized()) {
		cv::Mat cvRwg;
		cv::Mat dirG = cv::Mat::zeros(3, 1, CV_32F);
		for (vector<KeyFrame *>::iterator itKF = vpKF.begin(); itKF != vpKF.end(); itKF++) {
			if (!(*itKF)->mpImuPreintegrated) {
				continue;
			}
			if (!(*itKF)->mPrevKF) {
				continue;
			}

			dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
			cv::Mat _vel =
				((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition()) / (*itKF)->mpImuPreintegrated->dT;
			(*itKF)->SetVelocity(_vel);
			(*itKF)->mPrevKF->SetVelocity(_vel);
		}

		dirG = dirG / cv::norm(dirG);
		cv::Mat gI = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, -1.0f);
		cv::Mat v = gI.cross(dirG);
		const float nv = cv::norm(v);
		const float cosg = gI.dot(dirG);
		const float ang = acos(cosg);
		cv::Mat vzg = v * ang / nv;
		cvRwg = IMU::ExpSO3(vzg);
		mRwg = Converter::toMatrix3d(cvRwg);
		mTinit = mpCurrentKeyFrame->mTimeStamp - mFirstTs;
	}
	else {
		mRwg = Eigen::Matrix3d::Identity();
		mbg = Converter::toVector3d(mpCurrentKeyFrame->GetGyroBias());
		mba = Converter::toVector3d(mpCurrentKeyFrame->GetAccBias());
	}

	mScale = 1.0;

	mInitTime = mpTracker->mLastFrame.mTimeStamp - vpKF.front()->mTimeStamp;

	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	// set new bias for keyframes after optimization
		Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(),
									mRwg,
									mScale,
									mbg,
									mba,
									mbMonocular,
									infoInertial,
									false,
									false,
										priorG,
										priorA);
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	/*cout << "scale after inertial-only optimization: " << mScale << endl;
	cout << "bg after inertial-only optimization: " << mbg << endl;
	cout << "ba after inertial-only optimization: " << mba << endl;*/


	if (mScale < 1e-1) {
		cout << "scale too small" << endl;
		bInitializing = false;
		return;
	}



	// Before this line we are not changing the map

	// unique_lock<timed_mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
		if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
			mpAtlas->GetCurrentMap()->ApplyScaledRotation(Converter::toCvMat(mRwg).t(), mScale, true);
			// set new bias for last frame and current frame
			mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
		}
	std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

	// Check if initialization OK
	if (!mpAtlas->isImuInitialized()) {
		for (int i = 0; i < N; i++) {
			KeyFrame *pKF2 = vpKF[i];
			pKF2->bImu = true;
		}
	}

	/*cout << "Before GIBA: " << endl;
	cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
	cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;*/

	std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
	if (bFIBA) {
		if (priorA != 0.f) {
			Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, 0, NULL, true, priorG, priorA);
		}
		else {
			Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, 0, NULL, false);
		}
	}

	std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

		// If initialization is OK
		mpTracker->UpdateFrameIMU(1.0, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
	if (!mpAtlas->isImuInitialized()) {
		cout << "IMU in Map " << mpAtlas->GetCurrentMap()->GetId() << " is initialized" << endl;
		mpAtlas->SetImuInitialized();
		mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
		mpCurrentKeyFrame->bImu = true;
	}


	mbNewInit = true;
	mnKFs = vpKF.size();
	mIdxInit++;

	for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++) {
		(*lit)->SetBadFlag();
		delete *lit;
	}
	mlNewKeyFrames.clear();

	mpTracker->mState = Tracking::OK;
	bInitializing = false;


	/*cout << "After GIBA: " << endl;
	cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
	cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;
	double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
	double t_update = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
	double t_viba = std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count();
	cout << t_inertial_only << ", " << t_update << ", " << t_viba << endl;*/

	mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

	return;
}

void LocalMapping::ScaleRefinement()
{
	// Minimum number of keyframes to compute a solution
	// Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
	// unique_lock<mutex> lock0(mMutexImuInit);
	if (mbResetRequested) {
		return;
	}

	// Retrieve all keyframes in temporal order
	list<KeyFrame *> lpKF;
	KeyFrame *pKF = mpCurrentKeyFrame;
	while (pKF->mPrevKF) {
		lpKF.push_front(pKF);
		pKF = pKF->mPrevKF;
	}
	lpKF.push_front(pKF);
	vector<KeyFrame *> vpKF(lpKF.begin(), lpKF.end());

	while (CheckNewKeyFrames()) {
		ProcessNewKeyFrame();
		vpKF.push_back(mpCurrentKeyFrame);
		lpKF.push_back(mpCurrentKeyFrame);
	}

	const int N = vpKF.size();

	mRwg = Eigen::Matrix3d::Identity();
	mScale = 1.0;

	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	if (mScale < 1e-1) // 1e-1
	{
		cout << "scale too small" << endl;
		bInitializing = false;
		return;
	}

	// Before this line we are not changing the map
	// unique_lock<timed_mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
		mpAtlas->GetCurrentMap()->ApplyScaledRotation(Converter::toCvMat(mRwg).t(), mScale, true);
		mpTracker->UpdateFrameIMU(mScale, mpCurrentKeyFrame->GetImuBias(), mpCurrentKeyFrame);
	}
	std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

	for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++) {
		(*lit)->SetBadFlag();
		delete *lit;
	}
	mlNewKeyFrames.clear();

	double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

	// To perform pose-inertial opt w.r.t. last keyframe
	mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

	return;
}

bool LocalMapping::IsInitializing()
{
	return bInitializing;
}

double LocalMapping::GetCurrKFTime()
{

	if (mpCurrentKeyFrame) {
		return mpCurrentKeyFrame->mTimeStamp;
	}
	else {
		return 0.0;
	}
}

KeyFrame *LocalMapping::GetCurrKF()
{
	return mpCurrentKeyFrame;
}

void LocalMapping::InitializeDvlGyro(float priorG, bool bFirst)
{
	if (mbResetRequested) {
		return;
	}

	float minTime;
	int nMinKF;
	if (mbMonocular) {
		minTime = 2.0;
		nMinKF = 10;
	}
	else {
		minTime = 1.0;
		nMinKF = 10;
	}
	nMinKF = mpTracker->mKF_num_for_init;

	if (mpAtlas->KeyFramesInMap() < nMinKF) {
		return;
	}

	// Retrieve all keyframe in temporal order
	list<KeyFrame *> lpKF;
	KeyFrame *pKF = mpCurrentKeyFrame;
	while (pKF->mPrevKF) {
		lpKF.push_front(pKF);
		pKF = pKF->mPrevKF;
	}
	lpKF.push_front(pKF);
	vector<KeyFrame *> vpKF(lpKF.begin(), lpKF.end());

	if (vpKF.size() < nMinKF) {
		return;
	}

//	mFirstTs = vpKF.front()->mTimeStamp;
//	if (mpCurrentKeyFrame->mTimeStamp - mFirstTs < minTime) {
//		return;
//	}

	bInitializing = true;

	while (CheckNewKeyFrames()) {
		ProcessNewKeyFrame();
		vpKF.push_back(mpCurrentKeyFrame);
		lpKF.push_back(mpCurrentKeyFrame);
	}

	const int N = vpKF.size();
	IMU::Bias b(0, 0, 0, 0, 0, 0);

	// Compute and KF velocities mRwg estimation
	mRwg = Eigen::Matrix3d::Identity();
	mbg = Converter::toVector3d(mpCurrentKeyFrame->GetGyroBias());
	mba = Converter::toVector3d(mpCurrentKeyFrame->GetAccBias());

	mScale = 1.0;

	mInitTime = mpTracker->mLastFrame.mTimeStamp - vpKF.front()->mTimeStamp;

	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	// new bias has been set to all keyframes after optimization
//	Optimizer::DvlGyroInitOptimization(mpAtlas->GetCurrentMap(), mbg, mbMonocular, priorG);
	Optimizer::DvlGyroInitOptimization3(mpAtlas->GetCurrentMap(), mbg, mbMonocular, priorG);
//	Optimizer::DvlGyroInitOptimization6(mpAtlas->GetCurrentMap(), mbg, mbMonocular, priorG);
//	Optimizer::DvlGyroInitOptimization5(mpAtlas->GetCurrentMap(), mbg, mbMonocular, priorG);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	// set new bias for tracking thread, update last frame and current frame pose
//	mpTracker->mLastKfBeforeLoss=NULL;
//	mpTracker->mpDvlPreintegratedFromLastKFBeforeLost=NULL;

	if (!mpAtlas->isImuInitialized()) {
		for (auto pKF:vpKF) {
			pKF->bImu = true;
		}
		mpAtlas->SetImuInitialized();
	}
	mbCalibrated = true;

	/**todo_tightly
	 * 	execute a full DVL_Gyro BA after initilization
	 */

	DvlGyroOptimizer::FullDVLGyroBundleAdjustment(nullptr, mpAtlas->GetCurrentMap(), mpTracker->mlamda_DVL);
	cout<<"full BA excuted after calibration!"<<endl;

	//todo_tightly
	//	not sure whether this is safe in different thread
	mpTracker->UpdateFrameDVLGyro(vpKF.front()->GetImuBias(), mpCurrentKeyFrame);
	mpTracker->SetExtrinsicPara(vpKF.front()->mImuCalib);
	mpTracker->mCalibrated = true;
	bInitializing = false;
	// mpTracker->mpRosHandler->UpdateMap(mpAtlas);
	// mpTracker->mpRosHandler->PublishIntegration(mpAtlas);
	return;
}

void LocalMapping::InitializeDvlIMU()
{
	if (mbResetRequested) {
		return;
	}

	int nMinKF = 10;

	if (mpAtlas->KeyFramesInMap() < nMinKF) {
		return;
	}

	// Retrieve all keyframe in temporal order
	list<KeyFrame *> lpKF;
	KeyFrame *pKF = mpCurrentKeyFrame;
    Map* pCurMap = mpCurrentKeyFrame->GetMap();
	while (pKF->mPrevKF) {
		lpKF.push_front(pKF);
		pKF = pKF->mPrevKF;
	}
	lpKF.push_front(pKF);
	vector<KeyFrame *> vpKF(lpKF.begin(), lpKF.end());

	if (vpKF.size() < nMinKF) {
		return;
	}

//	mFirstTs = vpKF.front()->mTimeStamp;
//	if (mpCurrentKeyFrame->mTimeStamp - mFirstTs < minTime) {
//		return;
//	}
    auto dis = GetTravelDistance();

	bInitializing = true;

	while (CheckNewKeyFrames()) {
		ProcessNewKeyFrame();
		vpKF.push_back(mpCurrentKeyFrame);
		lpKF.push_back(mpCurrentKeyFrame);
	}

	const int N = vpKF.size();
	IMU::Bias b(0, 0, 0, 0, 0, 0);

	// Compute and KF velocities mRwg estimation
	mRwg = Eigen::Matrix3d::Identity();
	mbg = Converter::toVector3d(mpCurrentKeyFrame->GetGyroBias());
	mba = Converter::toVector3d(mpCurrentKeyFrame->GetAccBias());

	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	// new bias has been set to all keyframes after optimization
//	Optimizer::DvlGyroInitOptimization(mpAtlas->GetCurrentMap(), mbg, mbMonocular, priorG);
    double clib_avg_error = 0;
    if(dis.first<mInitTranslationThred||dis.second<mInitRotationThred){
        clib_avg_error = Optimizer::DvlIMUInitOptimization(mpAtlas->GetCurrentMap(),1e2,1e8);
    }
    else{
        clib_avg_error = Optimizer::DvlIMUInitOptimization(mpAtlas->GetCurrentMap(),1,1e6);
    }

//	Optimizer::DvlGyroInitOptimization6(mpAtlas->GetCurrentMap(), mbg, mbMonocular, priorG);
//	Optimizer::DvlGyroInitOptimization5(mpAtlas->GetCurrentMap(), mbg, mbMonocular, priorG);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    auto all_kf = mpAtlas->GetAllKeyFrames();
    if(clib_avg_error<0.01&&(all_kf.size()>20)){
        unique_lock<shared_timed_mutex> lock(pCurMap->mMutexMapUpdate);
//         ROS_INFO_STREAM("init avg error<100, avg err < 0.01");  // original
        pCurMap->SetImuInitialized();
        mpAtlas->SetDvlImuInitialized();
        mpAtlas->setRGravity(pCurMap->getRGravity());
        mpTracker->mpRosHandler->UpdateMap(mpAtlas);
        // mpTracker->UpdateFrameDVLGyro(vpKF.front()->GetImuBias(), mpCurrentKeyFrame);
        mpTracker->SetExtrinsicPara(vpKF.front()->mImuCalib);
        mpTracker->mCalibrated = true;
        mpTracker->mInitialized = true;
        bInitializing = false;
        mpAtlas->SetIMUCalibrated();
        FullBA();
        mpTracker->UpdateFrameDVLGyro(vpKF.front()->GetImuBias(), mpCurrentKeyFrame);
        return;

    }
    else if (dis.first<mInitTranslationThred||dis.second<mInitRotationThred){
        unique_lock<shared_timed_mutex> lock(pCurMap->mMutexMapUpdate);
//         ROS_INFO_STREAM("init motion is not enough");  // original
        ResetKFBias();
        pCurMap->SetImuInitialized();
        // mpAtlas->SetDvlImuInitialized();
        mpAtlas->setRGravity(pCurMap->getRGravity());
        mpTracker->mpRosHandler->UpdateMap(mpAtlas);
        mpTracker->UpdateFrameDVLGyro(vpKF.front()->GetImuBias(), mpCurrentKeyFrame);
        mpTracker->SetExtrinsicPara(vpKF.front()->mImuCalib);
        mpTracker->mCalibrated = true;
        mpTracker->mInitialized = true;
        bInitializing = false;
        // FullBA();
        return;
    }
    else if(clib_avg_error<0.1){
        unique_lock<shared_timed_mutex> lock(pCurMap->mMutexMapUpdate);
//         ROS_INFO_STREAM("init motion is enough");  // original
        mpAtlas->SetIMUCalibrated();
        pCurMap->SetImuInitialized();
        mpAtlas->SetDvlImuInitialized();
        mpAtlas->setRGravity(pCurMap->getRGravity());
        mpAtlas->SetDvlImuInitialized();
        mpTracker->mpRosHandler->UpdateMap(mpAtlas);
        // mpTracker->UpdateFrameDVLGyro(vpKF.front()->GetImuBias(), mpCurrentKeyFrame);
        mpTracker->SetExtrinsicPara(vpKF.front()->mImuCalib);
        mpTracker->mCalibrated = true;
        mpTracker->mInitialized = true;
        bInitializing = false;
        FullBA();
        mpTracker->UpdateFrameDVLGyro(vpKF.front()->GetImuBias(), mpCurrentKeyFrame);
        return;
        // FullBA();
    }
//     ROS_INFO_STREAM("init motion is enough, but error too large");  // original
    unique_lock<shared_timed_mutex> lock(pCurMap->mMutexMapUpdate);
    pCurMap->SetImuInitialized();
    mpAtlas->SetDvlImuInitialized();
    mpAtlas->setRGravity(pCurMap->getRGravity());
    mpAtlas->SetDvlImuInitialized();
	mpAtlas->SetIMUCalibrated();
    mpTracker->mpRosHandler->UpdateMap(mpAtlas);
    mpTracker->UpdateFrameDVLGyro(vpKF.front()->GetImuBias(), mpCurrentKeyFrame);
    mpTracker->SetExtrinsicPara(vpKF.front()->mImuCalib);
    mpTracker->mCalibrated = true;
    mpTracker->mInitialized = true;
    bInitializing = false;
    // FullBA();


	// mpTracker->mpRosHandler->PublishIntegration(mpAtlas);
	return;
}


void LocalMapping::RefineGravityDvlIMU()
{
    if(mpAtlas->GetAllKeyFrames().size()<10)
        return;
    if (mbResetRequested) {
        return;
    }

    Optimizer::DvlIMURefineOptimization(mpAtlas);

    // mpTracker->mpRosHandler->UpdateMap(mpAtlas);
    // mpTracker->mpRosHandler->PublishIntegration(mpAtlas);
}

void LocalMapping::FullBA()
{
    DvlGyroOptimizer::FullDVLIMUBundleAdjustment(mpAtlas,
                                                 mpCurrentKeyFrame,
                                                 &mbAbortBA,
                                                 mpCurrentKeyFrame->GetMap(),
                                                 10,
                                                 mpTracker->mlamda_DVL,
                                                 mpTracker->mlamda_visual);
}

std::pair<double,double> LocalMapping::GetTravelDistance()
{
    auto all_kfs = mpAtlas->GetAllKeyFrames();
    //sort
    std::sort(all_kfs.begin(),all_kfs.end(),[](KeyFrame* a,KeyFrame* b){return a->mnId<b->mnId;});
    Eigen::Isometry3d T_c0_ci = Eigen::Isometry3d::Identity();
    double t_dis = 0;
    double R_dis = 0;
    for(auto pkfi:all_kfs){
        cv::Mat Tc0cj_cv = pkfi->GetPoseInverse();
        Eigen::Isometry3d T_c0_cj = Eigen::Isometry3d::Identity();
        cv::cv2eigen(Tc0cj_cv, T_c0_cj.matrix());
        Eigen::Isometry3d T_ci_cj = T_c0_ci.inverse() * T_c0_cj;
        Eigen::Vector3d t_ci_cj = T_ci_cj.translation();
        // Re-orthogonalize via SVD: float→double conversion from cv::Mat
        // can leave the 3×3 rotation slightly non-orthonormal, which
        // causes Sophus::SO3 to abort.
        Eigen::Matrix3d R_ci_cj = T_ci_cj.rotation();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_ci_cj,
            Eigen::ComputeFullU | Eigen::ComputeFullV);
        R_ci_cj = svd.matrixU() * svd.matrixV().transpose();
        Sophus::SO3<double> R_ci_cj_SO3(R_ci_cj);
        Eigen::Vector3d R_ci_cj_so3 = R_ci_cj_SO3.log();
        t_dis += t_ci_cj.norm();
        R_dis += abs(R_ci_cj_so3.y());
        T_c0_ci = T_c0_cj;
    }
//     ROS_INFO_STREAM("travel distance(t R): "<<t_dis<<" "<<R_dis);  // original
    return std::make_pair(t_dis,R_dis);
}

void LocalMapping::ResetKFBias()
{
//     ROS_INFO_STREAM("reset bias");  // original
    auto all_kf = mpAtlas->GetAllKeyFrames();
    for(auto pkf:all_kf){
        IMU::Bias b;
        pkf->SetNewBias(b);
    }

}
} //namespace ORB_SLAM
