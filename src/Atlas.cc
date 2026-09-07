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

#include "Atlas.h"
// #include <ros/ros.h>  // original
// #include "Viewer.h"

#include "GeometricCamera.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "KeyFrameRegistry.h"

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ORB_SLAM3
{

Atlas::Atlas():mDvlImuInitailized(false), mIMUBiasCalibrated(false){
    mpCurrentMap = static_cast<Map*>(NULL);
    mR_b0_w.setIdentity();
}

Atlas::Atlas(int initKFid): mnLastInitKFidMap(initKFid), mHasViewer(false), mDvlImuInitailized(false),
mIMUBiasCalibrated(false)
{
    mpCurrentMap = static_cast<Map*>(NULL);
    CreateNewMap();
    mR_b0_w.setIdentity();
}

Atlas::~Atlas()
{
    for(std::set<Map*>::iterator it = mspMaps.begin(), end = mspMaps.end(); it != end;)
    {
        Map* pMi = *it;

        if(pMi)
        {
            delete pMi;
            pMi = static_cast<Map*>(NULL);

            it = mspMaps.erase(it);
        }
        else
            ++it;

    }
}

void Atlas::CreateNewMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    cout << "Creation of new map with id: " << Map::nNextId << endl;
    if(mpCurrentMap){
        cout << "Exits current map " << endl;
        if(!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
            mnLastInitKFidMap = mpCurrentMap->GetMaxKFid()+1; //The init KF is the next of current maximum

        mpCurrentMap->SetStoredMap();
        cout << "Saved map with ID: " << mpCurrentMap->GetId() << endl;

        //if(mHasViewer)
        //    mpViewer->AddMapToCreateThumbnail(mpCurrentMap);
    }
//    cout << "Creation of new map with last KF id: " << mnLastInitKFidMap << endl;
	cv::RNG rng(mnLastInitKFidMap);
	float r	= rng.uniform(0.0,1.0);
	float g	= rng.uniform(0.0,1.0);
	float b	= rng.uniform(0.0,1.0);
	Eigen::Vector3d color(r,g,b);
    mpCurrentMap = new Map(mnLastInitKFidMap,color);
    mpCurrentMap->SetCurrentMap();
    mspMaps.insert(mpCurrentMap);
}

void Atlas::ChangeMap(Map *pMapFrom, Map *pMapTo)
{
    unique_lock<mutex> lock(mMutexAtlas);

    if(pMapFrom){
	    cout << "Chage map from "<<pMapFrom->GetId()<<" to : " << pMapTo->GetId() << endl;
		pMapFrom->SetStoredMap();
    }

	if(mpCurrentMap==pMapFrom)
	{
		mpCurrentMap = pMapTo;
		mpCurrentMap->SetCurrentMap();
	}
	else{
// 		ROS_ERROR_STREAM("try to merge inactive maps!!");  // original
	}

}

unsigned long int Atlas::GetLastInitKFid()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mnLastInitKFidMap;
}

// void Atlas::SetViewer(Viewer* pViewer)
// {
//     mpViewer = pViewer;
//     mHasViewer = true;
// }

void Atlas::AddKeyFrame(KeyFrame* pKF)
{
    Map* pMapKF = pKF->GetMap();
    pMapKF->AddKeyFrame(pKF);
}

void Atlas::AddMapPoint(MapPoint* pMP)
{
    Map* pMapMP = pMP->GetMap();
    pMapMP->AddMapPoint(pMP);
}

void Atlas::AddCamera(GeometricCamera* pCam)
{
    mvpCameras.push_back(pCam);
}

void Atlas::SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs)
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetReferenceMapPoints(vpMPs);
}

void Atlas::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->InformNewBigChange();
}

int Atlas::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetLastBigChangeIdx();
}

long unsigned int Atlas::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->MapPointsInMap();
}

long unsigned Atlas::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->KeyFramesInMap();
}

std::vector<KeyFrame*> Atlas::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllKeyFrames();
}

std::vector<KeyFrame*> Atlas::GetAllKeyFramesinAllMap()
{
    std::vector<KeyFrame*> all_kf;
    unique_lock<mutex> lock(mMutexAtlas);
    for(auto m:mspMaps){
        auto all_kf_cur_map = m->GetAllKeyFrames();
        all_kf.insert(all_kf.end(),all_kf_cur_map.begin(),all_kf_cur_map.end());
    }
    return all_kf;
}

std::vector<MapPoint*> Atlas::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllMapPoints();
}

std::vector<MapPoint*> Atlas::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetReferenceMapPoints();
}

vector<Map*> Atlas::GetAllMaps()
{
    unique_lock<mutex> lock(mMutexAtlas);
    struct compFunctor
    {
        inline bool operator()(Map* elem1 ,Map* elem2)
        {
            return elem1->GetId() < elem2->GetId();
        }
    };
    vector<Map*> vMaps(mspMaps.begin(),mspMaps.end());
    sort(vMaps.begin(), vMaps.end(), compFunctor());
    return vMaps;
}

int Atlas::CountMaps()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mspMaps.size();
}

void Atlas::clearMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->clear();
}
void Atlas::clearSmallMaps()
{
//	unique_lock<mutex> lock(mMutexAtlas);
	auto all_Maps = GetAllMaps();
	for(auto pMap:all_Maps)
	{
		auto all_KFs = pMap->GetAllKeyFrames();
		if (all_KFs.size()<5){
			pMap->clear();
		}
	}
}

void Atlas::clearAtlas()
{
    unique_lock<mutex> lock(mMutexAtlas);
    /*for(std::set<Map*>::iterator it=mspMaps.begin(), send=mspMaps.end(); it!=send; it++)
    {
        (*it)->clear();
        delete *it;
    }*/
    mspMaps.clear();
    mpCurrentMap = static_cast<Map*>(NULL);
    mnLastInitKFidMap = 0;
}

Map* Atlas::GetCurrentMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    if(!mpCurrentMap||mpCurrentMap->IsBad())
	{
		lock.unlock();
		CreateNewMap();
// 		ROS_ERROR_STREAM("current map bad");  // original
	}

//    while(mpCurrentMap->IsBad())
//        usleep(3000);

    return mpCurrentMap;
}

void Atlas::SetMapBad(Map* pMap)
{
//	usleep(500);
    mspMaps.erase(pMap);

    mspBadMaps.insert(pMap);
}

void Atlas::RemoveBadMaps()
{
    for(Map* pMap : mspBadMaps)
    {
//		clearMap(pMap);
		pMap->SetBad();
    }
	unique_lock<mutex> lock(mMutexAtlas);
    mspBadMaps.clear();
}

bool Atlas::isInertial()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->IsInertial();
}

void Atlas::SetInertialSensor()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetInertialSensor();
}

void Atlas::SetImuInitialized()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetImuInitialized();
}

bool Atlas::isImuInitialized()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap && mpCurrentMap->isImuInitialized();
}

void Atlas::PreSave()
{
    if(mpCurrentMap){
        if(!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
            mnLastInitKFidMap = mpCurrentMap->GetMaxKFid()+1; //The init KF is the next of current maximum
    }

    struct compFunctor
    {
        inline bool operator()(Map* elem1 ,Map* elem2)
        {
            return elem1->GetId() < elem2->GetId();
        }
    };
    std::copy(mspMaps.begin(), mspMaps.end(), std::back_inserter(mvpBackupMaps));
    sort(mvpBackupMaps.begin(), mvpBackupMaps.end(), compFunctor());

    std::set<GeometricCamera*> spCams(mvpCameras.begin(), mvpCameras.end());
    cout << "There are " << spCams.size() << " cameras in the atlas" << endl;
    for(Map* pMi : mvpBackupMaps)
    {
        cout << "Pre-save of map " << pMi->GetId() << endl;
        pMi->PreSave(spCams);
    }
    cout << "Maps stored" << endl;
    for(GeometricCamera* pCam : mvpCameras)
    {
        cout << "Pre-save of camera " << pCam->GetId() << endl;
        if(pCam->GetType() == pCam->CAM_PINHOLE)
        {
            mvpBackupCamPin.push_back((Pinhole*) pCam);
        }
        else if(pCam->GetType() == pCam->CAM_FISHEYE)
        {
            mvpBackupCamKan.push_back((KannalaBrandt8*) pCam);
        }
    }

}

void Atlas::PostLoad()
{
//    mvpCameras.clear();
    map<unsigned int,GeometricCamera*> mpCams;
	for(auto pCamera:mvpCameras){
		mpCams[pCamera->GetId()] = pCamera;
	}
//    for(Pinhole* pCam : mvpBackupCamPin)
//    {
//        //mvpCameras.push_back((GeometricCamera*)pCam);
//        mvpCameras.push_back(pCam);
//        mpCams[pCam->GetId()] = pCam;
//    }
//    for(KannalaBrandt8* pCam : mvpBackupCamKan)
//    {
//        //mvpCameras.push_back((GeometricCamera*)pCam);
//        mvpCameras.push_back(pCam);
//        mpCams[pCam->GetId()] = pCam;
//    }

    mspMaps.clear();
    unsigned long int numKF = 0, numMP = 0;
    map<long unsigned int, KeyFrame*> mpAllKeyFrameId;
	//recover KeyFrameDB
	for(Map* pMi : mvpBackupMaps){
		pMi->PostLoadKFID(mpAllKeyFrameId);
	}
	mpKeyFrameDB->PostLoad(mpAllKeyFrameId);
    for(Map* pMi : mvpBackupMaps)
    {
        cout << "Map id:" << pMi->GetId() << endl;
        mspMaps.insert(pMi);
        map<long unsigned int, KeyFrame*> mpKeyFrameId;
        pMi->PostLoad(mpKeyFrameDB, mpKeyFrameId, mpCams);
        mpAllKeyFrameId.insert(mpKeyFrameId.begin(), mpKeyFrameId.end());
        numKF += pMi->GetAllKeyFrames().size();
	    for(auto pKF:(pMi->GetAllKeyFrames())){
			cout<<"KF camera1 pointer: "<<pKF->mpCamera<<"KF camera2 pointer: "<<pKF->mpCamera2<<endl;
		}
        numMP += pMi->GetAllMapPoints().size();
    }

    cout << "Number Map: " << mspMaps.size() << "Number KF:" << numKF << "; number MP:" << numMP << endl;

	for(auto pMap:mspMaps){
		cout << "Map ID:" << pMap->GetId() << endl;
	}
    mvpBackupMaps.clear();
}

void Atlas::SetKeyFrameDababase(KeyFrameDatabase* pKFDB)
{
    mpKeyFrameDB = pKFDB;
}

KeyFrameDatabase* Atlas::GetKeyFrameDatabase()
{
    return mpKeyFrameDB;
}

long unsigned int Atlas::GetNumLivedKF()
{
    unique_lock<mutex> lock(mMutexAtlas);
    long unsigned int num = 0;
    for(Map* mMAPi : mspMaps)
    {
        num += mMAPi->GetAllKeyFrames().size();
    }

    return num;
}

long unsigned int Atlas::GetNumLivedMP() {
    unique_lock<mutex> lock(mMutexAtlas);
    long unsigned int num = 0;
    for (Map *mMAPi : mspMaps) {
        num += mMAPi->GetAllMapPoints().size();
    }

    return num;
}
void Atlas::clearMap(Map *pMap)
{
	unique_lock<mutex> lock(mMutexAtlas);
	if(pMap&&!pMap->IsBad()){
		pMap->clear();
//		pMap->SetBad();
//		if(mspMaps.find(pMap)!=mspMaps.end()){
//			mspMaps.erase(mspMaps.find(pMap));
//		}
		cout<<"clear map: "<<pMap->GetId()<<endl;
	}
	else
    {
// 		ROS_ERROR_STREAM("fail to clear map!!!");  // original
    }
}
bool Atlas::isDvlImuInitialized()
{
	unique_lock<mutex> lock(mMutexAtlas);
	return mDvlImuInitailized;
}
void Atlas::SetDvlImuInitialized()
{
	unique_lock<mutex> lock(mMutexAtlas);
	mDvlImuInitailized = true;
}

Eigen::Matrix3d Atlas::getRGravity()
{
    std::shared_lock<std::shared_mutex> lock(mMutexGravity);
    return mR_b0_w;
}

void Atlas::setRGravity(const Eigen::Matrix3d &mRB0W)
{
    std::unique_lock<std::shared_mutex> lock(mMutexGravity);
    mR_b0_w = mRB0W;
}

bool Atlas::IsIMUCalibrated()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mIMUBiasCalibrated;
}

void Atlas::SetIMUCalibrated()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mIMUBiasCalibrated = true;
}

} //namespace ORB_SLAM3
