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


#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H


#include "Initializer.h"
#include "RosHandling.h"

#include <atomic>
#include <mutex>


namespace ORB_SLAM3
{

class System;
class KeyFrame;
class Tracking;
class LoopClosing;
class Atlas;
class KeyFrameDatabase;
class DenseMapper;
class NeuralFeatureFrontend;

// A Local Mapping scheduling abort can cancel work only before solve begins.
// Once snapshot-bound result validation succeeds, commit guards own rejection.
class DynamicLocalOptimizationTransaction
{
public:
    bool MayStartSolve(const std::atomic_bool& abortRequested) const noexcept
    {
        return !abortRequested.load();
    }

    void MarkResultValidation(bool accepted) noexcept
    {
        mbResultValidated = accepted;
    }

    bool MayCommitValidatedResult(bool stopRequested) const noexcept
    {
        return mbResultValidated && !stopRequested;
    }

private:
    bool mbResultValidated = false;
};

class LocalMapping
{
    friend class LocalMappingTestAccess;
public:
    LocalMapping(System* pSys, Atlas* pAtlas, DenseMapper* pDenseMapper,
                 const float bMonocular, bool bInertial, bool bDvlGyro,
                 NeuralFeatureFrontend* featureFrontend,
                 const string &strSettingPath=std::string());

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);
    void EmptyQueue();

    // Thread Synch
    void RequestStop();
    void RequestReset();
    void RequestResetActiveMap(Map* pMap);
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    bool IsInitializing();
    double GetCurrKFTime();
    KeyFrame* GetCurrKF();

    std::mutex mMutexImuInit;

    Eigen::MatrixXd mcovInertial;
    Eigen::Matrix3d mRwg;
    Eigen::Vector3d mbg;
    Eigen::Vector3d mba;
    double mScale;
    double mInitTime;
    double mCostTime;
    bool mbNewInit;
    unsigned int mInitSect;
    unsigned int mIdxInit;
    unsigned int mnKFs;
    double mFirstTs;
    int mnMatchesInliers;

    // For debugging (erase in normal mode)
    int mInitFr;
    int mIdxIteration;
    string mstrSettingPath;


    bool mbNotBA1;
    bool mbNotBA2;
    bool mbBadImu;

    bool mbWriteStats;

    // not consider far points (clouds)
    bool mbFarPoints;
    float mThFarPoints;
protected:

    LocalMapping() = default;
    std::unique_lock<std::recursive_timed_mutex> AcquireDynamicCommitTransaction();

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();
    bool OptimizeLocalMapWithDynamicBackend(Map* map);

    void MapPointCulling();
    void SearchInNeighbors();
    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    System *mpSystem;

	DenseMapper* mpDenseMapper;

    bool mbMonocular;
    bool mbInertial;

    void ResetIfRequested();
    bool mbResetRequested;
    bool mbResetRequestedActiveMap;
    Map* mpMapToReset;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Atlas* mpAtlas;

    LoopClosing* mpLoopCloser;
    NeuralFeatureFrontend* mpFeatureFrontend;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    std::atomic_bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;

    std::pair<double,double> GetTravelDistance();
    void InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);
	void InitializeDvlGyro(float priorG = 1e2, bool bFirst = false);
	void InitializeDvlIMU();
    void RefineGravityDvlIMU();
    void ScaleRefinement();
    void FullBA();
    void ResetKFBias();

    bool bInitializing;

    Eigen::MatrixXd infoInertial;
    int mNumLM;
    int mNumKFCulling;

    float mTinit;

    int mBiasRefineCount = 0;

    bool mbDvlGyro;

    bool mbCalibrated;

    double mInitTranslationThred, mInitRotationThred;

    //DEBUG
    ofstream f_lm;

	friend class RosHandling;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
