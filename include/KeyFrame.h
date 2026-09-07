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


#ifndef KEYFRAME_H
#define KEYFRAME_H


#include "ImuTypes.h"
#include <DVLGroPreIntegration.h>

#include "GeometricCamera.h"

#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>


namespace ORB_SLAM3
{
using std::cout;
using std::endl;
using std::list;
using std::make_pair;
using std::map;
using std::max;
using std::min;
using std::mutex;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::tuple;
using std::unique_lock;
using std::vector;
using std::get;


class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class GeometricCamera;

class KeyFrame
{


    template<class Archive>
    void serializeMatrix(Archive& ar, cv::Mat& mat, const unsigned int version)
    {
        int cols, rows, type;
        bool continuous;

        if (Archive::is_saving::value) {
            cols = mat.cols; rows = mat.rows; type = mat.type();
            continuous = mat.isContinuous();
        }

        ar & cols & rows & type & continuous;

        if (Archive::is_loading::value)
            mat.create(rows, cols, type);

        if (continuous) {
            const unsigned int data_size = rows * cols * mat.elemSize();
            ar & boost::serialization::make_array(mat.ptr(), data_size);
        } else {
            const unsigned int row_size = cols*mat.elemSize();
            for (int i = 0; i < rows; i++) {
                ar & boost::serialization::make_array(mat.ptr(i), row_size);
            }
        }
    }


    template<class Archive>
    void serializeMatrix(Archive& ar, const cv::Mat& mat, const unsigned int version)
    {
        cv::Mat matAux = mat;

        serializeMatrix(ar, matAux,version);

        if (Archive::is_loading::value)
        {
            cv::Mat* ptr;
            ptr = (cv::Mat*)( &mat );
            *ptr = matAux;
        }
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serializeVectorKeyPoints(Archive& ar, const vector<cv::KeyPoint>& vKP, const unsigned int version)
    {
        int NumEl;

        if (Archive::is_saving::value) {
            NumEl = vKP.size();
        }

        ar & NumEl;

        vector<cv::KeyPoint> vKPaux = vKP;
        if (Archive::is_loading::value)
            vKPaux.reserve(NumEl);

        for(int i=0; i < NumEl; ++i)
        {
            cv::KeyPoint KPi;

            if (Archive::is_loading::value)
                KPi = cv::KeyPoint();

            if (Archive::is_saving::value)
                KPi = vKPaux[i];

            ar & KPi.angle;
            ar & KPi.response;
            ar & KPi.size;
            ar & KPi.pt.x;
            ar & KPi.pt.y;
            ar & KPi.class_id;
            ar & KPi.octave;

            if (Archive::is_loading::value)
                vKPaux.push_back(KPi);
        }


        if (Archive::is_loading::value)
        {
            vector<cv::KeyPoint> *ptr;
            ptr = (vector<cv::KeyPoint>*)( &vKP );
            *ptr = vKPaux;
        }
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
	    if (Archive::is_saving::value) {
//		    if(std::isnan(mfGridElementWidthInv)){
//			    mfGridElementWidthInv = 0;
//			}
//		    if(std::isnan(mfGridElementHeightInv)){
//			    mfGridElementHeightInv = 0;
//		    }
		    if(std::isnan(mLoopScore)){
			    mLoopScore = 0;
		    }
		    if(std::isnan(mRelocScore)){
			    mRelocScore = 0;
		    }
		    if(std::isnan(mMergeScore)){
			    mMergeScore = 0;
		    }
		    if(std::isnan(mPlaceRecognitionScore)){
			    mPlaceRecognitionScore = 0;
		    }
		    if(std::isnan(mfScaleMerge)){
			    mfScaleMerge = 0;
		    }
		    if(std::isnan(mfScale)){
			    mfScale = 0;
		    }
//		    if(std::isnan(mfScaleFactor)){
//			    mfScaleFactor = 0;
//		    }
//		    if(std::isnan(mfLogScaleFactor)){
//			    mfLogScaleFactor = 0;
//		    }
		    if(std::isnan(mHalfBaseline)){
			    mHalfBaseline = 0;
		    }
	    }
        ar & mnId;
//		cout<<"load KF id: "<<mnId<<endl;
        ar & const_cast<long unsigned int&>(mnFrameId);
        ar & const_cast<double&>(mTimeStamp);
        // Grid
        ar & const_cast<int&>(mnGridCols);
        ar & const_cast<int&>(mnGridRows);
        ar & const_cast<float&>(mfGridElementWidthInv);
        ar & const_cast<float&>(mfGridElementHeightInv);
        // Variables of tracking
        ar & mnTrackReferenceForFrame;
        ar & mnFuseTargetForKF;
        // Variables of local mapping
        ar & mnBALocalForKF;
        ar & mnBAFixedForKF;
        ar & mnNumberOfOpt;
        // Variables used by KeyFrameDatabase
        ar & mnLoopQuery;
        ar & mnLoopWords;
        ar & mLoopScore;
        ar & mnRelocQuery;
        ar & mnRelocWords;
        ar & mRelocScore;
        ar & mnMergeQuery;
        ar & mnMergeWords;
        ar & mMergeScore;
        ar & mnPlaceRecognitionQuery;
        ar & mnPlaceRecognitionWords;
        ar & mPlaceRecognitionScore;
        ar & mbCurrentPlaceRecognition;
        // Variables of loop closing
        serializeMatrix(ar,mTcwGBA,version);
        serializeMatrix(ar,mTcwBefGBA,version);
        serializeMatrix(ar,mVwbGBA,version);
        serializeMatrix(ar,mVwbBefGBA,version);
        ar & mBiasGBA;
        ar & mnBAGlobalForKF;
        // Variables of Merging
        serializeMatrix(ar,mTcwMerge,version);
        serializeMatrix(ar,mTcwBefMerge,version);
        serializeMatrix(ar,mTwcBefMerge,version);
        serializeMatrix(ar,mVwbMerge,version);
        serializeMatrix(ar,mVwbBefMerge,version);
        ar & mBiasMerge;
        ar & mnMergeCorrectedForKF;
        ar & mnMergeForKF;
        ar & mfScaleMerge;
        ar & mnBALocalForMerge;
        // Scale
        ar & mfScale;
        // Calibration parameters
        ar & const_cast<float&>(fx);
        ar & const_cast<float&>(fy);
        ar & const_cast<float&>(invfx);
        ar & const_cast<float&>(invfy);
        ar & const_cast<float&>(cx);
        ar & const_cast<float&>(cy);
        ar & const_cast<float&>(mbf);
        ar & const_cast<float&>(mb);
        ar & const_cast<float&>(mThDepth);
        serializeMatrix(ar,mDistCoef,version);
        // Number of Keypoints
        ar & const_cast<int&>(N);
        // KeyPoints
        serializeVectorKeyPoints(ar,mvKeys,version);
        serializeVectorKeyPoints(ar,mvKeysUn,version);
        ar & const_cast<vector<float>& >(mvuRight);
        ar & const_cast<vector<float>& >(mvDepth);
        serializeMatrix(ar,mDescriptors,version);
        // Pose relative to parent
        serializeMatrix(ar,mTcp,version);
        // Scale
        ar & const_cast<int&>(mnScaleLevels);
        ar & const_cast<float&>(mfScaleFactor);
        ar & const_cast<float&>(mfLogScaleFactor);
        ar & const_cast<vector<float>& >(mvScaleFactors);
        ar & const_cast<vector<float>& >(mvLevelSigma2);
        ar & const_cast<vector<float>& >(mvInvLevelSigma2);
        // Image bounds and calibration
        ar & const_cast<int&>(mnMinX);
        ar & const_cast<int&>(mnMinY);
        ar & const_cast<int&>(mnMaxX);
        ar & const_cast<int&>(mnMaxY);
        ar & const_cast<int&>(mnImageWidth);
        ar & const_cast<int&>(mnImageHeight);
        serializeMatrix(ar,mK,version);
//		ar & mPrevKF;
//		ar & mNextKF;
		ar & mpImuPreintegrated;
		ar & mImuCalib;
		ar & mnOriginMapId;
		ar & mNameFile;
		ar & mnDataset;
//		ar & mvpLoopCandKFs;
//		ar & mvpMergeCandKFs;
		ar & mbHasHessian;
	    serializeMatrix(ar,mHessianPose,version);
		ar & mGrid;
	    ar & mGridRight;
        // Pose
        serializeMatrix(ar,Tcw,version);
	    serializeMatrix(ar,Twc,version);
	    serializeMatrix(ar,Ow,version);
	    serializeMatrix(ar,Cw,version);
	    serializeMatrix(ar,Owb,version);
	    serializeMatrix(ar,Vw,version);

        // MapPointsId associated to keypoints
        ar & mvBackupMapPointsId;
        // Grid
//        ar & mGrid;
        // Connected KeyFrameWeight
        ar & mBackupConnectedKeyFrameIdWeights;
        // Spanning Tree and Loop Edges
        ar & mbFirstConnection;
        ar & mBackupParentId;
        ar & mvBackupChildrensId;
        ar & mvBackupLoopEdgesId;
        ar & mvBackupMergeEdgesId;
        // Bad flags
        ar & mbNotErase;
        ar & mbToBeErased;
        ar & mbBad;

        ar & mHalfBaseline;

        // Camera variables
        ar & mnBackupIdCamera;
        ar & mnBackupIdCamera2;

        // Fisheye variables
        ar & mvLeftToRightMatch;
        ar & mvRightToLeftMatch;
        ar & const_cast<int&>(NLeft);
        ar & const_cast<int&>(NRight);
        serializeMatrix(ar, mTlr, version);
	    serializeMatrix(ar, mTrl, version);
        serializeVectorKeyPoints(ar, mvKeysRight, version);

        // Inertial variables
        ar & mImuBias;
        ar & mBackupImuPreintegrated;
        ar & mImuCalib;
        ar & mBackupPrevKFId;
        ar & mBackupNextKFId;
        ar & bImu;
        serializeMatrix(ar, Vw, version);

	    ar & mpDvlPreintegrationKeyFrame;
		ar & mImuCalib;

    }

public:
    KeyFrame();
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    void SetPose(const cv::Mat &Tcw);
    void SetVelocity(const cv::Mat &Vw_);

    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();
    cv::Mat GetImuPosition();
    cv::Mat GetImuRotation();
    cv::Mat GetImuPose();
    cv::Mat GetStereoCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();
    cv::Mat GetVelocityOld();
    cv::Mat GetDvlPosition();
    cv::Mat GetGyroRotation();
	cv::Mat GetDvlRotation();
    void GetDvlVelocity(Eigen::Vector3d& v_d);
    void GetDvlVelocityMeasurement(Eigen::Vector3d& v_d);
    void SetDvlVelocity(const Eigen::Vector3d& v_d);

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, const int &weight);
    void EraseConnection(KeyFrame* pKF);

    void UpdateConnections(bool upParent=true);
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);
    void SetFirstConnection(bool bFirst);

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopEdges();

    // Merge Edges
    void AddMergeEdge(KeyFrame* pKF);
    set<KeyFrame*> GetMergeEdges();

    // MapPoint observation functions
    int GetNumberMPs();
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const int &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const int &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const bool bRight = false) const;
    cv::Mat UnprojectStereo(int i);

    // Image
    bool IsInImage(const float &x, const float &y) const;
    cv::Size ImageSize() const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp( int a, int b){
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }

    Map* GetMap();
    void UpdateMap(Map* pMap);

    void SetNewBias(const IMU::Bias &b);
    cv::Mat GetGyroBias();
    cv::Mat GetAccBias();
    IMU::Bias GetImuBias();

    bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);
    bool ProjectPointUnDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

    void PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP, set<GeometricCamera*>& spCam);
    void PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid, map<unsigned int, GeometricCamera*>& mpCamId);


    void SetKeyFrameDatabase(KeyFrameDatabase* pKFDB);

    bool bImu;

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    //Number of optimizations by BA(amount of iterations in BA)
    long unsigned int mnNumberOfOpt;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;
    long unsigned int mnMergeQuery;
    int mnMergeWords;
    float mMergeScore;
    long unsigned int mnPlaceRecognitionQuery;
    int mnPlaceRecognitionWords;
    float mPlaceRecognitionScore;

    bool mbCurrentPlaceRecognition;


    // Variables used by loop closing
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    cv::Mat mVwbGBA;
    cv::Mat mVwbBefGBA;
    IMU::Bias mBiasGBA;
    long unsigned int mnBAGlobalForKF;

    // Variables used by merging
    cv::Mat mTcwMerge;
    cv::Mat mTcwBefMerge;
    cv::Mat mTwcBefMerge;
    cv::Mat mVwbMerge;
    cv::Mat mVwbBefMerge;
    IMU::Bias mBiasMerge;
    long unsigned int mnMergeCorrectedForKF;
    long unsigned int mnMergeForKF;
    float mfScaleMerge;
    long unsigned int mnBALocalForMerge;

    float mfScale;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;
    cv::Mat mDistCoef;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const int mnImageWidth;
    const int mnImageHeight;
    const cv::Mat mK;

    // Preintegrated IMU measurements from previous keyframe
    KeyFrame* mPrevKF;
    KeyFrame* mNextKF;

    IMU::Preintegrated* mpImuPreintegrated;
    IMU::Calib mImuCalib;


    unsigned int mnOriginMapId;

    string mNameFile;

    int mnDataset;

    std::vector <KeyFrame*> mvpLoopCandKFs;
    std::vector <KeyFrame*> mvpMergeCandKFs;

    bool mbHasHessian;
    cv::Mat mHessianPose;

	// Grid over the image to speed up feature matching
	std::vector< std::vector <std::vector<size_t> > > mGrid;
	std::vector< std::vector <std::vector<size_t> > > mGridRight;
//	std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
//	std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw;
    cv::Mat Twc;
    cv::Mat Ow;
    cv::Mat Cw; // Stereo middel point. Only for visualization

    // IMU position
    cv::Mat Owb;

    // Velocity (Only used for inertial SLAM)
    cv::Mat Vw;
	Eigen::Vector3d mVd; //velocity under DVL frame(FOR DVL model)

    // Imu bias
    IMU::Bias mImuBias;

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;
    // For save relation without pointer, this is necessary for save/load function
    std::vector<long long int> mvBackupMapPointsId;

    KeyFrameDatabase* mpKeyFrameDB;



    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;
    // For save relation without pointer, this is necessary for save/load function
    std::map<long unsigned int, int> mBackupConnectedKeyFrameIdWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;
    std::set<KeyFrame*> mspLoopEdges;
    std::set<KeyFrame*> mspMergeEdges;
    // For save relation without pointer, this is necessary for save/load function
    long long int mBackupParentId;
    std::vector<long unsigned int> mvBackupChildrensId;
    std::vector<long unsigned int> mvBackupLoopEdgesId;
    std::vector<long unsigned int> mvBackupMergeEdgesId;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;    

    float mHalfBaseline; // Only for visualization

    Map* mpMap;

    std::mutex mMutexPose; // for pose, velocity and biases
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
    std::mutex mMutexMap;

    // Backup variables for inertial
    long long int mBackupPrevKFId;
    long long int mBackupNextKFId;
    IMU::Preintegrated mBackupImuPreintegrated;

    // Backup for Cameras
    unsigned int mnBackupIdCamera, mnBackupIdCamera2;

public:
    GeometricCamera* mpCamera, *mpCamera2;

    //Indexes of stereo observations correspondences
    std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

    //Transformation matrix between cameras in stereo fisheye
    cv::Mat mTlr;
    cv::Mat mTrl;

    //KeyPoints in the right image (for stereo fisheye, coordinates are needed)
    const std::vector<cv::KeyPoint> mvKeysRight;

    // when NLeft==-1, using left index.
    // NLeft!=-1, using right index
    const int NLeft, NRight;



    cv::Mat GetRightPose();
    cv::Mat GetRightPoseInverse();
    cv::Mat GetRightPoseInverseH();
    cv::Mat GetRightCameraCenter();
    cv::Mat GetRightRotation();
    cv::Mat GetRightTranslation();

    cv::Mat imgLeft, imgRight; //TODO Backup??
    cv::Mat imgDepthScaled; // UW_SLAM: metric scaled DA2 depth

	// DVL EKF
	Eigen::Isometry3d mT_e0_ej;
	Eigen::Isometry3d mT_ei_ej;
	// velocity
	Eigen::Matrix<double,6,1> mV_e;
	// Eigen::Vector<double,6> mVelocity_EKF;
	double mTime_ekf;
	bool mGood_EKF=false;
	// transform from camera to ekf
	Eigen::Isometry3d mT_e_c;
	// from EKF to GT
	Eigen::Isometry3d mT_e_g;
	// current ground truth pose
	Eigen::Isometry3d mT_g0_gj;

	// tightly coupled DVL
	bool mbDVL= false;

	// Dvl and Gyro preintegration from last Keyframe
	DVLGroPreIntegration *mpDvlPreintegrationKeyFrame = nullptr;
    DVLGroPreIntegration *mpDvlPreintegrationLossRefKF=nullptr;
    KeyFrame* mpLossRefKF = nullptr;
    bool mPoorVision;

	void IntegrateDVL(KeyFrame* kp_i);

    void PrintPointDistribution(){
        int left = 0, right = 0;
        int Nlim = (NLeft != -1) ? NLeft : N;
        for(int i = 0; i < N; i++){
            if(mvpMapPoints[i]){
                if(i < Nlim) left++;
                else right++;
            }
        }
        cout << "Point distribution in KeyFrame: left-> " << left << " --- right-> " << right << endl;
    }


};

struct KFComparator{
    bool operator()(const KeyFrame* a, const KeyFrame* b) const {
        return a->mnId < b->mnId;
    }
};


} //namespace ORB_SLAM

#endif // KEYFRAME_H
