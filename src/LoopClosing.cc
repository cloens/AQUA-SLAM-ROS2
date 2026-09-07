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


#include "LoopClosing.h"
#include "Tracking.h"
#include "LocalMapping.h"
#include "KeyFrameRegistry.h"
#include "GtsamMapAdapter.h"
#include "GtsamMapOptimizationAdapter.h"
#include "GtsamBackendAdapter.h"
#include "FeatureMatcher.h"
#include "System.h"

#include<mutex>
#include<thread>
#include<chrono>

#if !defined(AQUA_HAS_GTSAM_DYNAMIC)
#error "AQUA loop closing requires the GTSAM dynamic backend"
#endif


namespace ORB_SLAM3
{

LoopClosing::LoopClosing(Atlas *pAtlas, KeyFrameDatabase *pDB, RosHandling* pRosHandler, const bool bFixScale, int mergingThreshold, rclcpp::Node::SharedPtr node, NeuralFeatureFrontend* featureFrontend):
    mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
    mpKeyFrameDB(pDB), mpFeatureFrontend(featureFrontend), mpMatchedKF(NULL),
    mpLoopLastCurrentKF(NULL), mpLoopMatchedKF(NULL),
    mpMergeLastCurrentKF(nullptr), mpMergeMatchedKF(nullptr),
    mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0), mnLoopNumCoincidences(0), mnMergeNumCoincidences(0),
    mbLoopDetected(false), mbMergeDetected(false), mnLoopNumNotFound(0), mnMergeNumNotFound(0), mpRosHandler(pRosHandler), mMergingThreshold(mergingThreshold),
    mpNode(node)
{
    mnCovisibilityConsistencyTh = mMergingThreshold;
    mpLastCurrentKF = static_cast<KeyFrame*>(NULL);
// //     mpNH=boost::make_shared<ros::NodeHandle>();  // original  // original
    // mpIt=boost::make_shared<image_transport::ImageTransport>(*mpNH);  // original
    // mpIt=boost::make_shared<image_transport::ImageTransport>(mpNode);  // original
    mpIt=std::make_shared<image_transport::ImageTransport>(mpNode);
    mImgPub_cur_keyframe=mpIt->advertise("/aqua_slam/loop/cur_img",10);
    mImgPub_map_keyframe=mpIt->advertise("AQUA_SLAM/loop/map_img",10);
	mTargetMapID = -1;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        //NEW LOOP AND MERGE DETECTION ALGORITHM
        //----------------------------
        if(CheckNewKeyFrames())
        {
            if(mpLastCurrentKF)
            {
                mpLastCurrentKF->mvpLoopCandKFs.clear();
                mpLastCurrentKF->mvpMergeCandKFs.clear();
            }
            if(NewDetectCommonRegions())
            {
                // after NewDetectCommonRegions() we have data association

#ifdef AQUA_HAS_GTSAM_DYNAMIC
                if (mpTracker->backendMode() == BackendMode::GtsamDynamic) {
                    if (mbMergeDetected) {
                        CorrectMerge();
                        if (mpMergeLastCurrentKF)
                            mpMergeLastCurrentKF->SetErase();
                        if (mpMergeMatchedKF)
                            mpMergeMatchedKF->SetErase();
                        mpMergeLastCurrentKF = nullptr;
                        mpMergeMatchedKF = nullptr;
                        mnMergeNumCoincidences = 0;
                        mnMergeNumNotFound = 0;
                        mvpMergeMatchedMPs.clear();
                        mvpMergeMPs.clear();
                        mMergeInliers = 0;
                        mMergeCorrespondenceCount = 0;
                        mMergeRmsErrorMeters = 0.0;
                        mMergeHypothesis.reset();
                        mbMergeDetected = false;
                    }
                    if (mbLoopDetected) {
                        mvpLoopMapPoints = mvpLoopMPs;
                        CorrectLoop();
                        if (mpLoopLastCurrentKF)
                            mpLoopLastCurrentKF->SetErase();
                        if (mpLoopMatchedKF)
                            mpLoopMatchedKF->SetErase();
                        mpLoopLastCurrentKF = nullptr;
                        mpLoopMatchedKF = nullptr;
                        mnLoopNumCoincidences = 0;
                        mvpLoopMatchedMPs.clear();
                        mLoopLandmarkReplacements.clear();
                        mvpLoopMPs.clear();
                        mnLoopNumNotFound = 0;
                        mbLoopDetected = false;
                    }
                    mpLastCurrentKF = mpCurrentKF;
                    continue;
                }
#endif


            }
            mpLastCurrentKF = mpCurrentKF;
        }

        ResetIfRequested();

        if(CheckFinish()){
            // cout << "LC: Finish requested" << endl;
            break;
        }

        usleep(5000);
    }

    //ofstream f_stats;
    //f_stats.open("PlaceRecognition_stats" + mpLocalMapper->strSequence + ".txt");
    //f_stats << "# current_timestamp, matched_timestamp, [0:Loop, 1:Merge]" << endl;
    //f_stats << fixed;
    //for(int i=0; i< vdPR_CurrentTime.size(); ++i)
    //{
    //    f_stats  << 1e9*vdPR_CurrentTime[i] << "," << 1e9*vdPR_MatchedTime[i] << "," << vnPR_TypeRecogn[i] << endl;
    //}

    //f_stats.close();

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::NewDetectCommonRegions()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
		mlpLoopKeyFrameQueue.pop_front();
		if(mpCurrentKF->isBad()){
			return false;
		}
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
        mpCurrentKF->mbCurrentPlaceRecognition = true;

        mpLastMap = mpCurrentKF->GetMap();
    }
	if (!mpTracker->mDetectLoop)
		return false;
	//if using stereo and less than 5 keyframes in map, return false
    if(mpLastMap->GetAllKeyFrames().size() < mpTracker->mKFThresholdForMap+10) //12
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }
//	if less than 3 keyframes in map, return false
    if(mpLastMap->GetAllKeyFrames().size() < 3)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

#ifdef AQUA_HAS_GTSAM_DYNAMIC
    // Dynamic mode performs candidate matching and fixed-scale SE(3)
    // verification before handing accepted edges to GTSAM.
    if (mpTracker->backendMode() == BackendMode::GtsamDynamic) {
        const auto resetLoopHypothesis = [this]() {
            if (mpLoopLastCurrentKF)
                mpLoopLastCurrentKF->SetErase();
            if (mpLoopMatchedKF)
                mpLoopMatchedKF->SetErase();
            mpLoopLastCurrentKF = nullptr;
            mpLoopMatchedKF = nullptr;
            mnLoopNumCoincidences = 0;
            mnLoopNumNotFound = 0;
            mvpLoopMatchedMPs.clear();
            mvpLoopMPs.clear();
            mbLoopDetected = false;
        };
        const auto resetMergeHypothesis = [this]() {
            if (mpMergeLastCurrentKF)
                mpMergeLastCurrentKF->SetErase();
            if (mpMergeMatchedKF)
                mpMergeMatchedKF->SetErase();
            mpMergeLastCurrentKF = nullptr;
            mpMergeMatchedKF = nullptr;
            mnMergeNumCoincidences = 0;
            mnMergeNumNotFound = 0;
            mvpMergeMatchedMPs.clear();
            mvpMergeMPs.clear();
            mMergeInliers = 0;
            mMergeCorrespondenceCount = 0;
            mMergeRmsErrorMeters = 0.0;
            mMergeHypothesis.reset();
            mbMergeDetected = false;
        };
        std::vector<KeyFrame*> loopCandidates;
        std::vector<KeyFrame*> mergeCandidates;
        mpKeyFrameDB->DetectNBestCandidates(
            mpCurrentKF, loopCandidates, mergeCandidates, 3);
        if (!mpFeatureFrontend)
            throw std::runtime_error("LoopClosing requires the neural feature frontend");
        FeatureMatcher matcher(*mpFeatureFrontend, 0.9F);
        const auto currentMapPoints = mpCurrentKF->GetMapPointMatches();
        const auto collectCandidateCorrespondences =
            [&matcher, &currentMapPoints, this](
                KeyFrame* candidate, std::vector<MapPoint*>* matched,
                std::vector<BackendLoopPointCorrespondence>* correspondences) {
            if (!candidate || !matched || !correspondences || candidate->isBad())
                return false;
            if (matcher.SearchByNeuralPair(mpCurrentKF, candidate, *matched) < 20)
                return false;
            const std::size_t count = std::min(currentMapPoints.size(),
                                               matched->size());
            correspondences->clear();
            correspondences->reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                MapPoint* current = currentMapPoints[index];
                MapPoint* candidatePoint = (*matched)[index];
                if (!current || !candidatePoint || current->isBad() ||
                    candidatePoint->isBad())
                    continue;
                const cv::Mat currentPosition = current->GetWorldPos();
                const cv::Mat candidatePosition = candidatePoint->GetWorldPos();
                if (currentPosition.rows != 3 || currentPosition.cols != 1 ||
                    candidatePosition.rows != 3 || candidatePosition.cols != 1 ||
                    currentPosition.type() != CV_32F ||
                    candidatePosition.type() != CV_32F ||
                    !cv::checkRange(currentPosition) ||
                    !cv::checkRange(candidatePosition))
                    continue;
                BackendLoopPointCorrespondence correspondence;
                correspondence.currentLandmarkId = current->mnId;
                correspondence.candidateLandmarkId = candidatePoint->mnId;
                for (int axis = 0; axis < 3; ++axis) {
                    correspondence.currentPoint[static_cast<std::size_t>(axis)] =
                        currentPosition.at<float>(axis);
                    correspondence.candidatePoint[static_cast<std::size_t>(axis)] =
                        candidatePosition.at<float>(axis);
                }
                correspondences->push_back(correspondence);
            }
            return !correspondences->empty();
        };
        for (KeyFrame* candidate : loopCandidates) {
            if (!candidate || candidate->isBad() ||
                candidate->GetMap() != mpCurrentKF->GetMap())
                continue;
            if (mnLoopNumCoincidences > 0 && candidate != mpLoopMatchedKF) {
                const std::set<KeyFrame*> connected =
                    candidate->GetConnectedKeyFrames();
                const std::set<KeyFrame*> previousConnected =
                    mpLoopMatchedKF->GetConnectedKeyFrames();
                if (connected.count(mpLoopMatchedKF) == 0U &&
                    previousConnected.count(candidate) == 0U)
                    continue;
            }
            std::vector<MapPoint*> matched;
            std::vector<BackendLoopPointCorrespondence> correspondences;
            BackendLoopCandidateResult verification;
            if (!collectCandidateCorrespondences(
                    candidate, &matched, &correspondences))
                continue;
            verification = GtsamMapOptimizationAdapter().verifyLoopCandidateSE3(
                correspondences, 0.25, 6U);
            if (!verification.accepted || verification.landmarkReplacements.size() < 6U)
                continue;
            if (mnLoopNumCoincidences > 0) {
                if (mpLoopLastCurrentKF)
                    mpLoopLastCurrentKF->SetErase();
                if (candidate != mpLoopMatchedKF) {
                    if (mpLoopMatchedKF)
                        mpLoopMatchedKF->SetErase();
                    candidate->SetNotErase();
                    mpLoopMatchedKF = candidate;
                }
            } else {
                candidate->SetNotErase();
                mpLoopMatchedKF = candidate;
            }
            mpLoopLastCurrentKF = mpCurrentKF;
            mvpLoopMPs = candidate->GetMapPointMatches();
            mvpLoopMatchedMPs = matched;
            mLoopLandmarkReplacements = verification.landmarkReplacements;
            ++mnLoopNumCoincidences;
            mnLoopNumNotFound = 0;
            mbLoopDetected = mnLoopNumCoincidences >= 3;
            mpKeyFrameDB->add(mpCurrentKF);
            return mbLoopDetected;
        }
        bool mergeEvidenceObserved = false;
        for (KeyFrame* candidate : mergeCandidates) {
            if (!candidate || candidate->isBad() || !candidate->GetMap() ||
                candidate->GetMap() == mpCurrentKF->GetMap() ||
                candidate->GetMap()->IsBad())
                continue;
            std::vector<MapPoint*> matched;
            std::vector<BackendLoopPointCorrespondence> correspondences;
            if (!collectCandidateCorrespondences(
                    candidate, &matched, &correspondences))
                continue;
            mergeEvidenceObserved = true;
            const BackendMergeHypothesisUpdate hypothesisUpdate =
                mMergeHypothesis.observe(
                    candidate->GetMap()->GetId(), mpCurrentKF->mnId,
                    candidate->mnId, correspondences);
            if (hypothesisUpdate.reset) {
                if (mpMergeLastCurrentKF)
                    mpMergeLastCurrentKF->SetErase();
                if (mpMergeMatchedKF)
                    mpMergeMatchedKF->SetErase();
                mpMergeLastCurrentKF = nullptr;
                mpMergeMatchedKF = nullptr;
                RCLCPP_INFO(
                    mpNode->get_logger(),
                    "dynamic merge hypothesis reset reason=%s candidate_map=%lu",
                    hypothesisUpdate.resetReason.c_str(),
                    static_cast<unsigned long>(candidate->GetMap()->GetId()));
            }
            BackendLoopCandidateResult verification =
                GtsamMapOptimizationAdapter().verifyLoopCandidateSE3(
                    mMergeHypothesis.correspondences(), 0.25, 6U);
            if (!verification.accepted) {
                RCLCPP_INFO(
                    mpNode->get_logger(),
                    "dynamic merge aggregate rejected current_map=%lu candidate_map=%lu "
                    "current_kf=%lu candidate_kf=%lu correspondences=%zu added=%zu reason=%s",
                    static_cast<unsigned long>(mpCurrentKF->GetMap()->GetId()),
                    static_cast<unsigned long>(candidate->GetMap()->GetId()),
                    mpCurrentKF->mnId, candidate->mnId,
                    mMergeHypothesis.correspondences().size(),
                    hypothesisUpdate.added, verification.diagnostic.c_str());
                continue;
            }
            constexpr double kMergeConfirmationTranslationMeters = 0.5;
            constexpr double kMergeConfirmationRotationRadians = 0.35;
            double translationDeltaMeters = 0.0;
            double rotationDeltaRadians = 0.0;
            std::string hypothesisResetReason;
            if (!mMergeHypothesis.recordAccepted(
                    verification, kMergeConfirmationTranslationMeters,
                    kMergeConfirmationRotationRadians,
                    &translationDeltaMeters, &rotationDeltaRadians,
                    &hypothesisResetReason)) {
                if (hypothesisResetReason == "transform_inconsistent") {
                    if (mpMergeLastCurrentKF)
                        mpMergeLastCurrentKF->SetErase();
                    if (mpMergeMatchedKF)
                        mpMergeMatchedKF->SetErase();
                    mpMergeLastCurrentKF = nullptr;
                    mpMergeMatchedKF = nullptr;
                    RCLCPP_INFO(
                        mpNode->get_logger(),
                        "dynamic merge hypothesis reset reason=%s translation_delta_m=%.6f rotation_delta_rad=%.6f",
                        hypothesisResetReason.c_str(), translationDeltaMeters,
                        rotationDeltaRadians);
                    mMergeHypothesis.observe(
                        candidate->GetMap()->GetId(), mpCurrentKF->mnId,
                        candidate->mnId, correspondences);
                    if (!mMergeHypothesis.recordAccepted(
                            verification, kMergeConfirmationTranslationMeters,
                            kMergeConfirmationRotationRadians,
                            &translationDeltaMeters, &rotationDeltaRadians,
                            &hypothesisResetReason))
                        continue;
                } else {
                    continue;
                }
            }
            if (mMergeHypothesis.confirmations() > 1U) {
                if (mpMergeLastCurrentKF)
                    mpMergeLastCurrentKF->SetErase();
                if (candidate != mpMergeMatchedKF) {
                    if (mpMergeMatchedKF)
                        mpMergeMatchedKF->SetErase();
                    candidate->SetNotErase();
                    mpMergeMatchedKF = candidate;
                }
            } else {
                candidate->SetNotErase();
                mpMergeMatchedKF = candidate;
            }
            mpMergeLastCurrentKF = mpCurrentKF;
            mvpMergeMPs = candidate->GetMapPointMatches();
            mvpMergeMatchedMPs = matched;
            const BackendLoopCandidateResult& aggregateResult =
                mMergeHypothesis.aggregateResult();
            mMergeCurrentFromCandidate = aggregateResult.currentFromCandidate;
            mMergeInliers = aggregateResult.inliers;
            mMergeCorrespondenceCount =
                mMergeHypothesis.correspondences().size();
            mMergeRmsErrorMeters = aggregateResult.rmsErrorMeters;
            mnMergeNumCoincidences = static_cast<int>(
                mMergeHypothesis.confirmations());
            mnMergeNumNotFound = 0;
            mbMergeDetected = mnMergeNumCoincidences >= 3;
            RCLCPP_INFO(mpNode->get_logger(),
                        "dynamic merge candidate current_map=%lu candidate_map=%lu "
                        "current_kf=%lu candidate_kf=%lu confirmations=%d inliers=%zu "
                        "correspondences=%zu added=%zu rms_m=%.6f "
                        "translation_delta_m=%.6f rotation_delta_rad=%.6f accepted=%d",
                        static_cast<unsigned long>(mpCurrentKF->GetMap()->GetId()),
                        static_cast<unsigned long>(candidate->GetMap()->GetId()),
                        mpCurrentKF->mnId, candidate->mnId,
                        mnMergeNumCoincidences, aggregateResult.inliers,
                        mMergeCorrespondenceCount, hypothesisUpdate.added,
                        aggregateResult.rmsErrorMeters, translationDeltaMeters,
                        rotationDeltaRadians, mbMergeDetected ? 1 : 0);
            mpKeyFrameDB->add(mpCurrentKF);
            return mbMergeDetected;
        }
        mpKeyFrameDB->add(mpCurrentKF);
        if (mnLoopNumCoincidences > 0) {
            ++mnLoopNumNotFound;
            if (mnLoopNumNotFound >= 2)
                resetLoopHypothesis();
        }
        if (!mergeEvidenceObserved && mMergeHypothesis.active()) {
            std::string resetReason;
            if (mMergeHypothesis.recordMiss(&resetReason)) {
                RCLCPP_INFO(mpNode->get_logger(),
                            "dynamic merge hypothesis reset reason=%s",
                            resetReason.c_str());
                resetMergeHypothesis();
            } else {
                mnMergeNumNotFound = static_cast<int>(
                    mMergeHypothesis.consecutiveMisses());
            }
        }
        mpCurrentKF->SetErase();
        mpCurrentKF->mbCurrentPlaceRecognition = false;
        return false;
    }
#endif

    return false;
}

void LoopClosing::CorrectMerge()
{
    // Never wait on Tracking: it can be waiting for our reset acknowledgement.
    auto mapTransaction = mpTracker->TryAcquireDynamicMapTransaction();
    if (!mapTransaction.owns_lock())
        return;
    Map* currentMap = mpCurrentKF ? mpCurrentKF->GetMap() : nullptr;
    Map* candidateMap = mpMergeMatchedKF ? mpMergeMatchedKF->GetMap() : nullptr;
    if (!currentMap || !candidateMap || currentMap == candidateMap ||
        mpAtlas->GetCurrentMap() != currentMap) {
        RCLCPP_WARN(mpNode->get_logger(),
                    "dynamic merge rejected before snapshot: invalid map ownership");
        return;
    }

    mpLocalMapper->RequestStop();
    struct ReleaseLocalMapper
    {
        LocalMapping* mapper;
        ~ReleaseLocalMapper() { mapper->Release(); }
    } release{mpLocalMapper};
    const auto stopDeadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(5);
    while (!mpLocalMapper->isStopped()) {
        if (std::chrono::steady_clock::now() >= stopDeadline) {
            RCLCPP_WARN(mpNode->get_logger(),
                        "dynamic merge rejected: Local Mapping stop timeout");
            return;
        }
        usleep(1000);
    }

    CommittedKeyframeWatermark watermark;
    if (!mpTracker->CaptureCommittedKeyframeWatermark(&watermark)) {
        RCLCPP_WARN(mpNode->get_logger(),
                    "dynamic merge rejected: missing committed watermark");
        return;
    }
    const std::uint64_t currentVersion = static_cast<std::uint64_t>(
        currentMap->GetMapChangeIndex());
    const std::uint64_t candidateVersion = static_cast<std::uint64_t>(
        candidateMap->GetMapChangeIndex());
    const BackendMapSnapshot currentFull =
        GtsamMapAdapter::snapshot(*currentMap, currentVersion);
    BackendMapSnapshot currentSnapshot;
    std::string rejectionReason;
    if (!GtsamMapAdapter::projectCommitted(
            currentFull, watermark, &currentSnapshot, &rejectionReason)) {
        RCLCPP_WARN(mpNode->get_logger(),
                    "dynamic merge rejected: incomplete committed projection (%s)",
                    rejectionReason.c_str());
        return;
    }
    const BackendMapSnapshot candidateSnapshot =
        GtsamMapAdapter::snapshot(*candidateMap, candidateVersion);
    const auto covariance = GtsamMapAdapter::mergeCovariance(
        mMergeInliers, mMergeCorrespondenceCount, mMergeRmsErrorMeters);
    if (!covariance) {
        RCLCPP_WARN(mpNode->get_logger(),
                    "dynamic merge rejected: invalid covariance inputs inliers=%zu correspondences=%zu rms_m=%.6f",
                    mMergeInliers, mMergeCorrespondenceCount,
                    mMergeRmsErrorMeters);
        return;
    }

    BackendMergeSnapshot transaction;
    if (!GtsamMapAdapter::prepareMerge(
            currentSnapshot, candidateSnapshot, mpCurrentKF->mnId,
            mpMergeMatchedKF->mnId, mMergeCurrentFromCandidate,
            *covariance, &transaction)) {
        RCLCPP_WARN(mpNode->get_logger(),
                    "dynamic merge rejected: transaction preparation failed");
        return;
    }
    transaction.currentSource = {
        currentFull.mapId, currentFull.version, currentFull.topologySignature};
    transaction.currentKeyframeIds.clear();
    transaction.currentLandmarkIds.clear();
    for (const auto& keyframe : currentFull.keyframes)
        transaction.currentKeyframeIds.push_back(keyframe.id);
    for (const auto& landmark : currentFull.landmarks)
        transaction.currentLandmarkIds.push_back(landmark.id);
    const BackendMergeResult result =
        GtsamMapOptimizationAdapter().optimizeMerge(transaction);
    const BackendMapResultValidation validation =
        GtsamMapAdapter::validateMergeResult(
            transaction, result);
    RCLCPP_INFO(
        mpNode->get_logger(),
        "dynamic merge transaction current_map=%lu candidate_map=%lu current_kf=%lu candidate_kf=%lu "
        "optimized=%d validated=%d inliers=%zu correspondences=%zu rms_m=%.6f "
        "covariance_diag=[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]",
        static_cast<unsigned long>(currentMap->GetId()),
        static_cast<unsigned long>(candidateMap->GetId()),
        mpCurrentKF->mnId, mpMergeMatchedKF->mnId,
        result.accepted ? 1 : 0, validation.accepted ? 1 : 0,
        mMergeInliers, mMergeCorrespondenceCount, mMergeRmsErrorMeters,
        (*covariance)[0], (*covariance)[7], (*covariance)[14],
        (*covariance)[21], (*covariance)[28], (*covariance)[35]);
    if (!validation.accepted)
        return;

    const cv::Mat oldCurrentPose = mpCurrentKF->GetPose();
    std::string commitRejectionReason;
    if (!GtsamMapAdapter::commitMerge(
            result, *currentMap, *candidateMap, &commitRejectionReason)) {
        RCLCPP_WARN(mpNode->get_logger(),
                    "dynamic merge rejected at commit gate: %s",
                    commitRejectionReason.c_str());
        return;
    }
    const std::uint64_t mergedVersion = static_cast<std::uint64_t>(
        currentMap->GetMapChangeIndex());
    if (!mpTracker->AdvanceCommittedWatermarkMapVersion(
            currentVersion, mergedVersion)) {
        RCLCPP_WARN(mpNode->get_logger(),
                    "dynamic merge committed but watermark version advance failed");
        return;
    }
    mpTracker->SynchronizeAfterMapCorrection(
        oldCurrentPose, mpCurrentKF->GetPose());

    CommittedKeyframeWatermark mergedWatermark;
    const BackendMapSnapshot mergedFull =
        GtsamMapAdapter::snapshot(*currentMap, mergedVersion);
    BackendMapSnapshot mergedSnapshot;
    if (!mpTracker->CaptureCommittedKeyframeWatermark(&mergedWatermark) ||
        !GtsamMapAdapter::projectCommitted(
            mergedFull, mergedWatermark, &mergedSnapshot, &rejectionReason) ||
        !mpTracker->RebaseGtsamBackend(mergedSnapshot)) {
        mpTracker->RecordGtsamRebaseProjectionRejected(
            rejectionReason.empty() ? "merge commit rebase rejected"
                                    : rejectionReason);
        return;
    }
    mpTracker->RecordGtsamRebaseSucceeded();
    RCLCPP_INFO(mpNode->get_logger(),
                "dynamic merge committed current_map=%lu absorbed_map=%lu keyframes=%zu landmarks=%zu",
                static_cast<unsigned long>(currentMap->GetId()),
                static_cast<unsigned long>(candidateMap->GetId()),
                mergedSnapshot.keyframes.size(), mergedSnapshot.landmarks.size());
}

void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

#ifdef AQUA_HAS_GTSAM_DYNAMIC
    // Dynamic loop correction is a transactional GTSAM Essential Graph update.
    if (mpTracker->backendMode() == BackendMode::GtsamDynamic) {
        // Acquire before stopping Local Mapping; its commit gate yields to us.
        auto mapTransaction = mpTracker->TryAcquireDynamicMapTransaction();
        if (!mapTransaction.owns_lock())
            return;
        mpLocalMapper->RequestStop();
        while (!mpLocalMapper->isStopped())
            usleep(1000);
        Map* map = mpCurrentKF ? mpCurrentKF->GetMap() : nullptr;
        bool committed = false;
        if (map && mpLoopMatchedKF) {
            CommittedKeyframeWatermark watermark;
            if (!mpTracker->CaptureCommittedKeyframeWatermark(&watermark)) {
                RCLCPP_WARN(mpNode->get_logger(),
                            "discarded dynamic loop correction without committed watermark");
            } else {
            const std::uint64_t version =
                static_cast<std::uint64_t>(map->GetMapChangeIndex());
            const BackendMapSnapshot fullSnapshot =
                GtsamMapAdapter::snapshot(*map, version);
            BackendMapSnapshot snapshot;
            std::string projectionReason;
            if (!GtsamMapAdapter::projectCommitted(
                    fullSnapshot, watermark, &snapshot, &projectionReason)) {
                mpTracker->RecordGtsamRebaseProjectionRejected(projectionReason);
            } else if (!mLoopLandmarkReplacements.empty() &&
                       mpLoopMatchedKF->GetMap() == map) {
                snapshot.landmarks = fullSnapshot.landmarks;
                mapTransaction.unlock();
                const auto prepared = mpTracker->PrepareDynamicMapOptimization(
                    snapshot, mLoopLandmarkReplacements);
                const BackendMapResult result = prepared ? prepared->result()
                                                         : BackendMapResult();
                const BackendMapResultValidation validation =
                    GtsamMapAdapter::validateMapResult(
                        snapshot, result);
                RCLCPP_INFO(
                    mpNode->get_logger(),
                    "dynamic loop optimized=%d validated=%d keyframes=%zu landmarks=%zu "
                    "max_pose_shift_m=%.6f max_pose_rotation_rad=%.6f max_landmark_shift_m=%.6f",
                    result.accepted ? 1 : 0, validation.accepted ? 1 : 0,
                    result.keyframes.size(), result.landmarks.size(),
                    validation.maximumPoseTranslation,
                    validation.maximumPoseRotation,
                    validation.maximumLandmarkTranslation);
                mapTransaction.try_lock();
				committed = prepared && validation.accepted &&
                            mapTransaction.owns_lock() &&
				            mpTracker->CommitDynamicMapOptimization(
				                *prepared, *map, version, mpCurrentKF);
                if (committed) {
                    map->InformNewBigChange();
                    mpTracker->RecordGtsamRebaseSucceeded();
                    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF);
                    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF);
                }
            }
            }
        }
        if (!committed)
            RCLCPP_WARN(mpNode->get_logger(),
                        "discarded dynamic GTSAM loop correction");
        mpLocalMapper->Release();
        return;
    }
#endif
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::RequestResetActiveMap(Map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset = pMap;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetActiveMapRequested)
                break;
        }
        usleep(100);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        cout << "Loop closer reset requested..." << endl;
        mlpLoopKeyFrameQueue.clear();
        if (mpLoopLastCurrentKF)
            mpLoopLastCurrentKF->SetErase();
        if (mpLoopMatchedKF)
            mpLoopMatchedKF->SetErase();
        mpLoopLastCurrentKF = nullptr;
        mpLoopMatchedKF = nullptr;
        mnLoopNumCoincidences = 0;
        mnLoopNumNotFound = 0;
        mbLoopDetected = false;
        if (mpMergeLastCurrentKF)
            mpMergeLastCurrentKF->SetErase();
        if (mpMergeMatchedKF)
            mpMergeMatchedKF->SetErase();
        mpMergeLastCurrentKF = nullptr;
        mpMergeMatchedKF = nullptr;
        mnMergeNumCoincidences = 0;
        mnMergeNumNotFound = 0;
        mMergeHypothesis.reset();
        mbMergeDetected = false;
        mLastLoopKFid=0;  //TODO old variable, it is not use in the new algorithm
        mbResetRequested=false;
        mbResetActiveMapRequested = false;
    }
    else if(mbResetActiveMapRequested)
    {

		mlpLoopKeyFrameQueue.clear();
		if (mpLoopLastCurrentKF)
			mpLoopLastCurrentKF->SetErase();
		if (mpLoopMatchedKF)
			mpLoopMatchedKF->SetErase();
		mpLoopLastCurrentKF = nullptr;
		mpLoopMatchedKF = nullptr;
			mnLoopNumCoincidences = 0;
			mnLoopNumNotFound = 0;
			mbLoopDetected = false;
			if (mpMergeLastCurrentKF)
				mpMergeLastCurrentKF->SetErase();
			if (mpMergeMatchedKF)
				mpMergeMatchedKF->SetErase();
			mpMergeLastCurrentKF = nullptr;
			mpMergeMatchedKF = nullptr;
			mnMergeNumCoincidences = 0;
			mnMergeNumNotFound = 0;
			mMergeHypothesis.reset();
			mbMergeDetected = false;

        mLastLoopKFid=mpAtlas->GetLastInitKFid(); //TODO old variable, it is not use in the new algorithm
        mbResetActiveMapRequested=false;

    }
}

void LoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF)
{
    Verbose::PrintMess("Starting Global Bundle Adjustment", Verbose::VERBOSITY_NORMAL);

    const bool bImuInit = pActiveMap->isImuInitialized();

    {
        const std::uint64_t version =
            static_cast<std::uint64_t>(pActiveMap->GetMapChangeIndex());
        const BackendMapSnapshot fullSnapshot =
            GtsamMapAdapter::snapshot(*pActiveMap, version);
        CommittedKeyframeWatermark watermark;
        BackendMapSnapshot snapshot;
        const bool projected = mpTracker &&
            mpTracker->CaptureCommittedKeyframeWatermark(&watermark) &&
            GtsamMapAdapter::projectCommitted(fullSnapshot, watermark, &snapshot,
                                             nullptr);
        if (projected)
            snapshot.landmarks = fullSnapshot.landmarks;
        const auto prepared = projected
            ? mpTracker->PrepareDynamicMapOptimization(snapshot) : nullptr;
        if (!prepared || !mpTracker->CommitDynamicMapOptimization(
                *prepared, *pActiveMap, version, mpCurrentKF)) {
            RCLCPP_WARN(rclcpp::get_logger("aqua_slam"),
                        "dynamic global BA result was rejected (version=%lu)",
                        static_cast<unsigned long>(version));
            std::unique_lock<std::mutex> lock(mMutexGBA);
            mbFinishedGBA = true;
            mbRunningGBA = false;
            return;
        }
        pActiveMap->InformNewBigChange();
        std::unique_lock<std::mutex> lock(mMutexGBA);
        mbFinishedGBA = true;
        mbRunningGBA = false;
        return;
    }


    int idx =  mnFullBAIdx;
    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!bImuInit && pActiveMap->isImuInitialized())
            return;

        if(!mbStopGBA)
        {
            Verbose::PrintMess("Global Bundle Adjustment finished", Verbose::VERBOSITY_NORMAL);
            Verbose::PrintMess("Updating map ...", Verbose::VERBOSITY_NORMAL);

            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<shared_timed_mutex> lock(pActiveMap->mMutexMapUpdate);
            // cout << "LC: Update Map Mutex adquired" << endl;

            //pActiveMap->PrintEssentialGraph();
            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(pActiveMap->mvpKeyFrameOrigins.begin(),pActiveMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                //cout << "---Updating KF " << pKF->mnId << " with " << sChilds.size() << " childs" << endl;
                //cout << " KF mnBAGlobalForKF: " << pKF->mnBAGlobalForKF << endl;
                cv::Mat Twc = pKF->GetPoseInverse();
                //cout << "Twc: " << Twc << endl;
                //cout << "GBA: Correct KeyFrames" << endl;
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(!pChild || pChild->isBad())
                        continue;

                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        //cout << "++++New child with flag " << pChild->mnBAGlobalForKF << "; LoopKF: " << nLoopKF << endl;
                        //cout << " child id: " << pChild->mnId << endl;
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        //cout << "Child pose: " << Tchildc << endl;
                        //cout << "pKF->mTcwGBA: " << pKF->mTcwGBA << endl;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;

                        cv::Mat Rcor = pChild->mTcwGBA.rowRange(0,3).colRange(0,3).t()*pChild->GetRotation();
                        if(!pChild->GetVelocityOld().empty()){
                            //cout << "Child velocity: " << pChild->GetVelocity() << endl;
                            pChild->mVwbGBA = Rcor* pChild->GetVelocityOld();
                        }
                        else
                            Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);


                        //cout << "Child bias: " << pChild->GetImuBias() << endl;
                        pChild->mBiasGBA = pChild->GetImuBias();


                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                //cout << "-------Update pose" << endl;
                pKF->mTcwBefGBA = pKF->GetPose();
                //cout << "pKF->mTcwBefGBA: " << pKF->mTcwBefGBA << endl;
                pKF->SetPose(pKF->mTcwGBA);
                /*cv::Mat Tco_cn = pKF->mTcwBefGBA * pKF->mTcwGBA.inv();
                cv::Vec3d trasl = Tco_cn.rowRange(0,3).col(3);
                double dist = cv::norm(trasl);
                cout << "GBA: KF " << pKF->mnId << " had been moved " << dist << " meters" << endl;
                double desvX = 0;
                double desvY = 0;
                double desvZ = 0;
                if(pKF->mbHasHessian)
                {
                    cv::Mat hessianInv = pKF->mHessianPose.inv();

                    double covX = hessianInv.at<double>(3,3);
                    desvX = std::sqrt(covX);
                    double covY = hessianInv.at<double>(4,4);
                    desvY = std::sqrt(covY);
                    double covZ = hessianInv.at<double>(5,5);
                    desvZ = std::sqrt(covZ);
                    pKF->mbHasHessian = false;
                }
                if(dist > 1)
                {
                    cout << "--To much distance correction: It has " << pKF->GetConnectedKeyFrames().size() << " connected KFs" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(80).size() << " connected KF with 80 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(50).size() << " connected KF with 50 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(20).size() << " connected KF with 20 common matches or more" << endl;

                    cout << "--STD in meters(x, y, z): " << desvX << ", " << desvY << ", " << desvZ << endl;


                    string strNameFile = pKF->mNameFile;
                    cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

                    cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

                    vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
                    int num_MPs = 0;
                    for(int i=0; i<vpMapPointsKF.size(); ++i)
                    {
                        if(!vpMapPointsKF[i] || vpMapPointsKF[i]->isBad())
                        {
                            continue;
                        }
                        num_MPs += 1;
                        string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                        cv::circle(imLeft, pKF->mvKeys[i].pt, 2, cv::Scalar(0, 255, 0));
                        cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    }
                    cout << "--It has " << num_MPs << " MPs matched in the map" << endl;

                    string namefile = "./test_GBA/GBA_" + to_string(nLoopKF) + "_KF" + to_string(pKF->mnId) +"_D" + to_string(dist) +".png";
                    cv::imwrite(namefile, imLeft);
                }*/


                if(pKF->bImu)
                {
                    //cout << "-------Update inertial values" << endl;
                    pKF->mVwbBefGBA = pKF->GetVelocityOld();
                    if (pKF->mVwbGBA.empty())
                        Verbose::PrintMess("pKF->mVwbGBA is empty", Verbose::VERBOSITY_NORMAL);

                    assert(!pKF->mVwbGBA.empty());
                    pKF->SetVelocity(pKF->mVwbGBA);
                    pKF->SetNewBias(pKF->mBiasGBA);
                }

                lpKFtoCheck.pop_front();
            }

            //cout << "GBA: Correct MapPoints" << endl;
            // Correct MapPoints
            const vector<MapPoint*> vpMPs = pActiveMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    if(pRefKF->mTcwBefGBA.empty())
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }

            pActiveMap->InformNewBigChange();
            pActiveMap->IncreaseChangeIndex();

            // TODO Check this update
            // mpTracker->UpdateFrameIMU(1.0f, mpTracker->GetLastKeyFrame()->GetImuBias(), mpTracker->GetLastKeyFrame());

            mpLocalMapper->Release();

            Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    // cout << "LC: Finish requested" << endl;
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

//void LoopClosing::CallMapMerge()
//{
//	Map* pmap = mpMergeMatchedKF->GetMap();
//	if(pmap){
//		int merged_map_id = mpMergeMatchedKF->GetMap()->GetId();
//		int current_map_id = mpCurrentKF->GetMap()->GetId();
//		vehicle_interface::MapMergingInfo map_info;
//		map_info.request.current_map_id=current_map_id;
//		map_info.request.merged_map_id=merged_map_id;
//
//		mMergingSrv.call(map_info);
//		{
//			std::lock_guard<std::mutex> guard(mpTracker->mMerged_flag_mutex);
//			mpTracker->mMapMerged = true;
//		}
//		cout<<"map merging info send to planner!"<<endl;
//	}
//	else{
//		ROS_WARN_STREAM("map merge error, skip to send");
//	}
//
//
//
//}

void LoopClosing::SetTargetMap(int ID)
{
	mTargetMapID = ID;
}
void LoopClosing::ClearQueue()
{
	std::lock_guard<std::mutex> lock(mMutexLoopQueue);
	mlpLoopKeyFrameQueue.clear();
}

} //namespace ORB_SLAM
