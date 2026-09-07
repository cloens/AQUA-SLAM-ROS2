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


#ifndef FEATUREMATCHER_H
#define FEATUREMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"


namespace ORB_SLAM3
{

class NeuralFeatureFrontend;

class FeatureMatcher
{    
public:

    FeatureMatcher(NeuralFeatureFrontend& frontend, float nnratio=0.6);

    static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

	int searchByLKTracking(Frame &CurrentFrame, const Frame &LastFrame);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3, const bool bFarPoints = false, const float thFarPoints = 50.0f);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th);


    /***
     * Project MapPoints using a Similarity Transformation and search matches.
     * Used in Place Recognition (Loop Closing and Merging)
     * @param pKF(in) current keyframe
     * @param Scw(in) pose of current keyframe
     * @param vpPoints(in) map points for matching
     * @param vpPointsKFs(in) pointers to keyframes have good covisibility with the candidate keyframe(previous keyframe in the map)
     * 						and the candidate keyframe. vpPointsKFs has the same index with vpPointsKFs
     * 						and their index is the index of keypoint in the keyframe
     * @param vpMatched(out)
     * @param vpMatchedKF(out)
     * @return the number of matches
     */
    int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs, std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th);

    int SearchByNeuralPair(KeyFrame* keyframe, Frame& frame,
                           std::vector<MapPoint*>& mapPointMatches);
    int SearchByNeuralPair(KeyFrame* first, KeyFrame* second,
                           std::vector<MapPoint*>& matches);

    // Matching for the Map Initialization (only used in the monocular case)
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10);

    int SearchForNeuralTriangulation(
        KeyFrame* first, KeyFrame* second, const cv::Mat& fundamental,
        std::vector<std::pair<size_t, size_t>>& matchedPairs,
        bool onlyStereo);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0, const bool bRight = false);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);

public:

    static const int HISTO_LENGTH;


protected:

    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF, const bool b1=false);
    bool CheckDistEpipolarLine2(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF, const float unc);

    float RadiusByViewingCos(const float &viewCos);

    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    float mfDescriptorThreshold;
    bool mbCheckOrientation;
    NeuralFeatureFrontend* mpNeuralFrontend = nullptr;
};

}// namespace ORB_SLAM

#endif // FEATUREMATCHER_H
