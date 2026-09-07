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


#ifndef FRAME_H
#define FRAME_H

//#define SAVE_TIMES

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ImuTypes.h"
#include "NeuralFeatureFrontend.h"
#include <DVLGroPreIntegration.h>

#include <mutex>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{
using std::cout;
using std::endl;
using std::map;
using std::string;
using std::vector;

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;
class ConstraintPoseImu;
class GeometricCamera;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

	// Constructor for tightly coupled stereo-gro-dvl.
	Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
	      NeuralFeatureFrontend &featureFrontend,
	      cv::Mat &K, cv::Mat &distCoef, const float &bf,
	      const float &thFarDepth, const float &thCloseDepth,
	      const float &stereoMaxVerticalError, GeometricCamera* pCamera,
	      bool bDVL, Frame* pPrevF = static_cast<Frame*>(NULL),
	      const IMU::Calib &ImuCalib = IMU::Calib());

    // Destructor
    // ~Frame();

    void ExtractNeuralStereo(const cv::Mat &imLeftGray,
                             const cv::Mat &imRightGray,
                             NeuralFeatureFrontend &featureFrontend,
                             float stereoMaxVerticalError);
    int TransferTemporalMapPoints(const Frame& previous,
                                  const std::vector<NeuralMatch>& inliers);

    // Set the camera pose. (Imu pose is not modified!)
    void SetPose(cv::Mat Tcw);
    const void GetPose(cv::Mat &Tcw);

    // Set IMU velocity
    void SetVelocity(const cv::Mat &Vwb);

    // Set IMU pose and velocity (implicitly changes camera pose)
    void SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb);

	void SetDvlPoseVelocity(const cv::Mat &R_c0_gyroj, const cv::Mat &c0_t_c0_dj, const cv::Mat &c0_V_di_dj);

	void SetExtrinsicParamters(const IMU::Calib& calib);
	IMU::Calib& GetExtrinsicParamters();
	std::mutex* mpExtrinsic_mutex;

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    cv::Mat GetImuPosition();
    cv::Mat GetImuRotation();
    cv::Mat GetImuPose();

	cv::Mat GetDvlPosition();
	cv::Mat GetDvlRotation();
	cv::Mat GetDvlVelocity() const;
	cv::Mat GetGyroRotation();

    void SetNewBias(const IMU::Bias &b);

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

    cv::Mat inRefCoordinates(cv::Mat pCw);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1, const bool bRight = false) const;

    // UW_SLAM seam: overwrite stereo geometry from external LEFT↔RIGHT matches.
    void ApplyExternalStereoMatches(const std::vector<cv::Point2f>& vLeft,
                                   const std::vector<cv::Point2f>& vRight,
                                   const std::vector<float>& vScore = std::vector<float>());

    // UW_SLAM seam: metric Stereo-Scaled DA2 depth for DenseMapper.
    void SetExternalDepth(const cv::Mat& depthScaled);

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

    ConstraintPoseImu* mpcpi;

    bool imuIsPreintegrated();
    void setIntegrated();

    // R_c0_cj
    cv::Mat mRwc;
    // c0_t_c0_cj
    cv::Mat mOw;
public:
    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;
	float mCloseThDepth;

    // Number of KeyPoints.
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    std::vector<MapPoint*> mvpMapPoints;
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Normalized SuperPoint descriptor, one CV_32F row per keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;
    int mnCloseMPs;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];


    // Camera pose.
    // for dvl-gyros, T_cj_c0
    cv::Mat mTcw;

    // IMU linear velocity
    // for dvl-gyros, c0_V_di_dj
    cv::Mat mVw;

    cv::Mat mPredRwb, mPredtwb, mPredVwb;
    IMU::Bias mPredBias;

    // IMU bias
    IMU::Bias mImuBias;

    // Imu calibration
    IMU::Calib mImuCalib;

    // Imu preintegration from last keyframe
    IMU::Preintegrated* mpImuPreintegrated;
	// DVL-Imu preintegration from last keyframe
	DVLGroPreIntegration *mpDvlPreintegrationKeyFrame=nullptr;
    // DVL-Imu preintegration from reference loss keyframe
    DVLGroPreIntegration *mpDvlPreintegrationLossRefKF=nullptr;
    KeyFrame* mpLossRefKF = nullptr;
    KeyFrame* mpLastKeyFrame;

    // Pointer to previous frame
    Frame* mpPrevFrame;
	// Imu preintegration from last frame
    IMU::Preintegrated* mpImuPreintegratedFrame;
	// dvl preintegration from last frame
    DVLGroPreIntegration *mpDvlPreintegrationFrame=nullptr;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

    map<long unsigned int, cv::Point2f> mmProjectPoints;
    map<long unsigned int, cv::Point2f> mmMatchedInImage;

    string mNameFile;

    int mnDataset;

    double mTimeStereoMatch;
    double mTimeFeatureExtraction;

    // DVL EKF
    Eigen::Isometry3d mT_e0_ej;
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

    // tighly coupled DVL
    bool mbDVL=false;
	Eigen::Isometry3d mT_c0_cj_dvl;
    bool mPoorVision = false;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    //R_cj_c0
    cv::Mat mRcw;
    //cj_t_cj_c0
    cv::Mat mtcw;
    //==mtwc

    bool mbImuPreintegrated;

    std::mutex *mpMutexImu;

public:
    GeometricCamera* mpCamera, *mpCamera2;

    //Number of KeyPoints extracted in the left and right images
    int Nleft, Nright;
    //Number of Non Lapping Keypoints
    int monoLeft, monoRight;

    //For stereo matching
    std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

    // Triangulated stereo observations using the left camera as reference.
    std::vector<cv::Mat> mvStereo3Dpoints;

    //Grid for the right image
    std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    cv::Mat mTlr, mRlr, mtlr, mTrl;

    bool isInFrustumChecks(MapPoint* pMP, float viewingCosLimit, bool bRight = false);

    cv::Mat UnprojectStereoFishEye(const int &i);

    cv::Mat imgLeft, imgRight;
    cv::Mat imgDepthScaled; // UW_SLAM: metric CV_32FC1 depth, empty if unused

    void PrintPointDistribution(){
        int left = 0, right = 0;
        int Nlim = (Nleft != -1) ? Nleft : N;
        for(int i = 0; i < N; i++){
            if(mvpMapPoints[i] && !mvbOutlier[i]){
                if(i < Nlim) left++;
                else right++;
            }
        }
        cout << "Point distribution in Frame: left-> " << left << " --- right-> " << right << endl;
    }
};

}// namespace ORB_SLAM

#endif // FRAME_H
