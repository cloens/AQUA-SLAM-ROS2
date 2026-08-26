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

#include "Optimizer.h"

#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/eigen.hpp>

#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "G2oTypes.h"
#include "Converter.h"
#include "System.h"
#include <sophus/geometry.hpp>

#include <mutex>

#include "OptimizableTypes.h"

namespace ORB_SLAM3
{

bool sortByVal(const pair<MapPoint *, int> &a, const pair<MapPoint *, int> &b)
{
	return (a.second < b.second);
}

void Optimizer::GlobalBundleAdjustemnt(Map *pMap,
                                       int nIterations,
                                       bool *pbStopFlag,
                                       const unsigned long nLoopKF,
                                       const bool bRobust)
{
	vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
	vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
	BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

// vpKFs: the keyframe(transformation) to optimiza
// vpMP: observed map points
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
	// whether a map point has connection to a keyframe
	vector<bool> vbNotIncludedMP;
	vbNotIncludedMP.resize(vpMP.size());

	// Map : a set of map points and key frames
	Map *pMap = vpKFs[0]->GetMap();

	// set up g2o solver
	g2o::SparseOptimizer optimizer;

	// ??? what does 3 means
	// 6: 6 variables to optimize(6-degree-pose), 3: dimension of error(camera tranlation)
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);
	// set up g2o solver finished

	// ????
	if (pbStopFlag) {
		optimizer.setForceStopFlag(pbStopFlag);
	}

	// the max id of Keyframe
	long unsigned int maxKFid = 0;

	// the number of all edges
	const int nExpectedSize = (vpKFs.size()) * vpMP.size();

	// edges between map points(3) and camera pose(6)
	// for monocular
	vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	// edges between map points(3) and camera pose(6)
	// for stereo, and only for those
	vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody *> vpEdgesBody;
	vpEdgesBody.reserve(nExpectedSize);

	// points to keyframes
	// ?? why allocate nExpectedSize(num_map_points*num_keyframe)
	vector<KeyFrame *> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);
	// ?? why allocate nExpectedSize(num_map_points*num_keyframe)
	vector<KeyFrame *> vpEdgeKFBody;
	vpEdgeKFBody.reserve(nExpectedSize);

	// points to mappoints
	// ?? why allocate nExpectedSize(num_map_points*num_keyframe)
	vector<MapPoint *> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);
	// ?? why allocate nExpectedSize(num_map_points*num_keyframe)
	vector<MapPoint *> vpMapPointEdgeBody;
	vpMapPointEdgeBody.reserve(nExpectedSize);

	// stereo edge
	vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
	vpEdgesStereo.reserve(nExpectedSize);
	// stereo keyframe
	vector<KeyFrame *> vpEdgeKFStereo;
	vpEdgeKFStereo.reserve(nExpectedSize);
	// stereo map points
	vector<MapPoint *> vpMapPointEdgeStereo;
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	// Set KeyFrame vertices

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKF = vpKFs[i];
		if (pKF->isBad()) {
			continue;
		}
		// add camera pose as vertex to the graph
		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
		vSE3->setId(pKF->mnId);
		// set the first frame as fixed, because we do not optimize the first frame.
		vSE3->setFixed(pKF->mnId == pMap->GetInitKFid());
		optimizer.addVertex(vSE3);
		if (pKF->mnId > maxKFid) {
			maxKFid = pKF->mnId;
		}
		//cout << "KF id: " << pKF->mnId << endl;
	}

	// RobustKernelHuber parameter
	const float thHuber2D = sqrt(5.99);
	const float thHuber3D = sqrt(7.815);

	// Set MapPoint vertices
	//cout << "start inserting MPs" << endl;

	for (size_t i = 0; i < vpMP.size(); i++) {
		MapPoint *pMP = vpMP[i];
		// ignore bad points
		if (pMP->isBad()) {
			continue;
		}

		// add map points to graph,
		g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		const int id = pMP->mnId + maxKFid + 1;
		vPoint->setId(id);
		// do not optimize map points
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		// when using monocular, first int: map point index
		// when using stereo, first int: left index, second int: right index
		const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

		int nEdges = 0;
		//SET EDGES
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end();
		     mit++) {
			KeyFrame *pKF = mit->first;
			// ignore bad frame, ignore out bound Key frame
			if (pKF->isBad() || pKF->mnId > maxKFid) {
				continue;
			}
			// ignore map points which are not in the graph
			if (optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL) {
				continue;
			}
			nEdges++;

			// number of features in left(stereo)
			const int leftIndex = get<0>(mit->second);

			// using monocular
			if (leftIndex != -1 && pKF->mvuRight[get<0>(mit->second)] < 0) {
				//undistorted feature points
				const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];
				//set observation
				Eigen::Matrix<double, 2, 1> obs;
				obs << kpUn.pt.x, kpUn.pt.y;

				ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
				e->setMeasurement(obs);

				// set information according to octave(pyramid layer)/distance
				const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
				e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

				if (bRobust) {
					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuber2D);
				}

				e->pCamera = pKF->mpCamera;

				optimizer.addEdge(e);

				vpEdgesMono.push_back(e);
				vpEdgeKFMono.push_back(pKF);
				vpMapPointEdgeMono.push_back(pMP);
			}
				// using stereo,
				// for the features are both in the left image and the right image
			else if (leftIndex != -1 && pKF->mvuRight[leftIndex] >= 0) //Stereo observation
			{
				//undistorted feature points
				const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

				Eigen::Matrix<double, 3, 1> obs;
				// get<0>(mit->second): leftindex
				// kp_ur: the position of x in thr right image
				const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
				obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

				g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
				e->setMeasurement(obs);
				// set information according to octave(pyramid layer)/distance
				const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
				Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
				e->setInformation(Info);

				if (bRobust) {
					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuber3D);
				}

				e->fx = pKF->fx;
				e->fy = pKF->fy;
				e->cx = pKF->cx;
				e->cy = pKF->cy;
				e->bf = pKF->mbf;

				optimizer.addEdge(e);

				vpEdgesStereo.push_back(e);
				vpEdgeKFStereo.push_back(pKF);
				vpMapPointEdgeStereo.push_back(pMP);
			}

			// using stereo,
			// for the features are only in the right image
			if (pKF->mpCamera2) {
				int rightIndex = get<1>(mit->second);

				if (rightIndex != -1 && rightIndex < pKF->mvKeysRight.size()) {
					rightIndex -= pKF->NLeft;

					Eigen::Matrix<double, 2, 1> obs;
					cv::KeyPoint kp = pKF->mvKeysRight[rightIndex];
					obs << kp.pt.x, kp.pt.y;

					ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuber2D);

					e->mTrl = Converter::toSE3Quat(pKF->mTrl);

					e->pCamera = pKF->mpCamera2;

					optimizer.addEdge(e);
					vpEdgesBody.push_back(e);
					vpEdgeKFBody.push_back(pKF);
					vpMapPointEdgeBody.push_back(pMP);
				}
			}
		}

		if (nEdges == 0) {
			optimizer.removeVertex(vPoint);
			vbNotIncludedMP[i] = true;
		}
		else {
			vbNotIncludedMP[i] = false;
		}
	}

	//cout << "end inserting MPs" << endl;
	// Optimize!
	optimizer.setVerbose(false);
	optimizer.initializeOptimization();
	optimizer.optimize(nIterations);
	Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

	// Recover optimized data

	//Keyframes
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKF = vpKFs[i];
		if (pKF->isBad()) {
			continue;
		}
		g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));

		g2o::SE3Quat SE3quat = vSE3->estimate();

		// ???? what is nLoopKF
		// if current frame is the first keyframe of this map
		if (nLoopKF == pMap->GetOriginKF()->mnId) {
			pKF->SetPose(Converter::toCvMat(SE3quat));
		}
		else {
			/*if(!vSE3->fixed())
            {
                //cout << "KF " << pKF->mnId << ": " << endl;
                pKF->mHessianPose = cv::Mat(6, 6, CV_64F);
                pKF->mbHasHessian = true;
                for(int r=0; r<6; ++r)
                {
                    for(int c=0; c<6; ++c)
                    {
                        //cout  << vSE3->hessian(r, c) << ", ";
                        pKF->mHessianPose.at<double>(r, c) = vSE3->hessian(r, c);
                    }
                    //cout << endl;
                }
            }*/

			//????? in else, where to uodate the pose of keyframe

			// optimizaed transformation matrix from world to camera
			pKF->mTcwGBA.create(4, 4, CV_32F);
			Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
			pKF->mnBAGlobalForKF = nLoopKF;

			// the inverse of previous transformation matrix from world to camera
			cv::Mat mTwc = pKF->GetPoseInverse();
			// transformation matrix from previous camera to optimized camera
			cv::Mat mTcGBA_c = pKF->mTcwGBA * mTwc;
			cv::Vec3d vector_dist = mTcGBA_c.rowRange(0, 3).col(3);
			double dist = cv::norm(vector_dist);
			if (dist > 1) {
				int numMonoBadPoints = 0, numMonoOptPoints = 0;
				int numStereoBadPoints = 0, numStereoOptPoints = 0;
				vector<MapPoint *> vpMonoMPsOpt, vpStereoMPsOpt;

				//using mono
				// iterate all edges, find edges include current keyframe
				// and find all map points connected to current keyframe, and add to vpMonoMPsOpt
				for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
					ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
					MapPoint *pMP = vpMapPointEdgeMono[i];
					KeyFrame *pKFedge = vpEdgeKFMono[i];

					if (pKF != pKFedge) {
						continue;
					}

					if (pMP->isBad()) {
						continue;
					}

					if (e->chi2() > 5.991 || !e->isDepthPositive()) {
						numMonoBadPoints++;
					}
					else {
						numMonoOptPoints++;
						vpMonoMPsOpt.push_back(pMP);
					}
				}

				// using stereo
				// iterate all edges, find edges include current keyframe
				// and find all map points connected to current keyframe, and add to vpMonoMPsOpt
				for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
					g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
					MapPoint *pMP = vpMapPointEdgeStereo[i];
					KeyFrame *pKFedge = vpEdgeKFMono[i];

					if (pKF != pKFedge) {
						continue;
					}

					if (pMP->isBad()) {
						continue;
					}

					if (e->chi2() > 7.815 || !e->isDepthPositive()) {
						numStereoBadPoints++;
					}
					else {
						numStereoOptPoints++;
						vpStereoMPsOpt.push_back(pMP);
					}
				}
				Verbose::PrintMess("GBA: KF " + to_string(pKF->mnId) + " had been moved " + to_string(dist) + " meters",
				                   Verbose::VERBOSITY_DEBUG);
				Verbose::PrintMess("--Number of observations: " + to_string(numMonoOptPoints) + " in mono and "
					                   + to_string(numStereoOptPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
				Verbose::PrintMess(
					"--Number of discarded observations: " + to_string(numMonoBadPoints) + " in mono and "
						+ to_string(numStereoBadPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
			}
		}
	}
	Verbose::PrintMess("BA: KFs updated", Verbose::VERBOSITY_DEBUG);

	//Points
	for (size_t i = 0; i < vpMP.size(); i++) {
		// ???
		if (vbNotIncludedMP[i]) {
			continue;
		}

		MapPoint *pMP = vpMP[i];

		if (pMP->isBad()) {
			continue;
		}
		g2o::VertexSBAPointXYZ
			*vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

		// if current frame is the first keyframe of this map
		if (nLoopKF == pMap->GetOriginKF()->mnId) {
			pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
			pMP->UpdateNormalAndDepth();
		}
		else {
			pMP->mPosGBA.create(3, 1, CV_32F);
			// ????
			Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
			pMP->mnBAGlobalForKF = nLoopKF;
		}
	}
}

void Optimizer::FullInertialBA(Map *pMap,
                               int its,
                               const bool bFixLocal,
                               const long unsigned int nLoopId,
                               bool *pbStopFlag,
                               bool bInit,
                               float priorG,
                               float priorA,
                               Eigen::VectorXd *vSingVal,
                               bool *bHess)
{
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
	const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	solver->setUserLambdaInit(1e-5);
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);

	if (pbStopFlag) {
		optimizer.setForceStopFlag(pbStopFlag);
	}

	int nNonFixed = 0;

	// Set KeyFrame vertices
	KeyFrame *pIncKF;
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		pIncKF = pKFi;
		bool bFixed = false;
		if (bFixLocal) {
			bFixed = (pKFi->mnBALocalForKF >= (maxKFid - 1)) || (pKFi->mnBAFixedForKF >= (maxKFid - 1));
			if (!bFixed) {
				nNonFixed++;
			}
			VP->setFixed(bFixed);
		}
		optimizer.addVertex(VP);

		if (pKFi->bImu) {
			VertexVelocity *VV = new VertexVelocity(pKFi);
			VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
			VV->setFixed(bFixed);
			optimizer.addVertex(VV);
			if (!bInit) {
				VertexGyroBias *VG = new VertexGyroBias(pKFi);
				VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
				VG->setFixed(bFixed);
				optimizer.addVertex(VG);
				VertexAccBias *VA = new VertexAccBias(pKFi);
				VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
				VA->setFixed(bFixed);
				optimizer.addVertex(VA);
			}
		}
	}

	if (bInit) {
		VertexGyroBias *VG = new VertexGyroBias(pIncKF);
		VG->setId(4 * maxKFid + 2);
		VG->setFixed(false);
		optimizer.addVertex(VG);
		VertexAccBias *VA = new VertexAccBias(pIncKF);
		VA->setId(4 * maxKFid + 3);
		VA->setFixed(false);
		optimizer.addVertex(VA);
	}

	if (bFixLocal) {
		if (nNonFixed < 3) {
			return;
		}
	}

	// IMU links
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (!pKFi->mPrevKF) {
			Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
			continue;
		}

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (pKFi->bImu && pKFi->mPrevKF->bImu) {
				if (!pKFi->mpImuPreintegrated) {
					continue;
				}
				pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
				g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1);

				g2o::HyperGraph::Vertex *VG1;
				g2o::HyperGraph::Vertex *VA1;
				g2o::HyperGraph::Vertex *VG2;
				g2o::HyperGraph::Vertex *VA2;
				if (!bInit) {
					VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2);
					VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3);
					VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2);
					VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3);
				}
				else {
					VG1 = optimizer.vertex(4 * maxKFid + 2);
					VA1 = optimizer.vertex(4 * maxKFid + 3);
				}

				g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1);

				if (!bInit) {
					if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2) {
						cout << "Error" << VP1 << ", " << VV1 << ", " << VG1 << ", " << VA1 << ", " << VP2 << ", "
						     << VV2 << ", " << VG2 << ", " << VA2 << endl;

						continue;
					}
				}
				else {
					if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2) {
						cout << "Error" << VP1 << ", " << VV1 << ", " << VG1 << ", " << VA1 << ", " << VP2 << ", "
						     << VV2 << endl;

						continue;
					}
				}

				EdgeInertial *ei = new EdgeInertial(pKFi->mpImuPreintegrated);
				ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
				ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
				ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG1));
				ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA1));
				ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
				ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));

				g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
				ei->setRobustKernel(rki);
				rki->setDelta(sqrt(16.92));

				optimizer.addEdge(ei);

				if (!bInit) {
					EdgeGyroRW *egr = new EdgeGyroRW();
					egr->setVertex(0, VG1);
					egr->setVertex(1, VG2);
					cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
					Eigen::Matrix3d InfoG;
					for (int r = 0; r < 3; r++)
						for (int c = 0; c < 3; c++)
							InfoG(r, c) = cvInfoG.at<float>(r, c);
					egr->setInformation(InfoG);
					egr->computeError();
					optimizer.addEdge(egr);

					EdgeAccRW *ear = new EdgeAccRW();
					ear->setVertex(0, VA1);
					ear->setVertex(1, VA2);
					cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
					Eigen::Matrix3d InfoA;
					for (int r = 0; r < 3; r++)
						for (int c = 0; c < 3; c++)
							InfoA(r, c) = cvInfoA.at<float>(r, c);
					ear->setInformation(InfoA);
					ear->computeError();
					optimizer.addEdge(ear);
				}
			}
			else {
				cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
			}
		}
	}

	if (bInit) {
		g2o::HyperGraph::Vertex *VG = optimizer.vertex(4 * maxKFid + 2);
		g2o::HyperGraph::Vertex *VA = optimizer.vertex(4 * maxKFid + 3);

		// Add prior to comon biases
		EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F));
		epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
		double infoPriorA = priorA; //
		epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
		optimizer.addEdge(epa);

		EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
		epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
		double infoPriorG = priorG; //
		epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
		optimizer.addEdge(epg);
	}

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	const unsigned long iniMPid = maxKFid * 5;

	vector<bool> vbNotIncludedMP(vpMPs.size(), false);

	for (size_t i = 0; i < vpMPs.size(); i++) {
		MapPoint *pMP = vpMPs[i];
		g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		unsigned long id = pMP->mnId + iniMPid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

		bool bAllFixed = true;

		//Set edges
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (pKFi->mnId > maxKFid) {
				continue;
			}

			if (!pKFi->isBad()) {
				const int leftIndex = get<0>(mit->second);
				cv::KeyPoint kpUn;

				if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] < 0) // Monocular observation
				{
					kpUn = pKFi->mvKeysUn[leftIndex];
					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					EdgeMono *e = new EdgeMono(0);

					g2o::OptimizableGraph::Vertex
						*VP = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId));
					if (bAllFixed) {
						if (!VP->fixed()) {
							bAllFixed = false;
						}
					}

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, VP);
					e->setMeasurement(obs);
					const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					optimizer.addEdge(e);
				}
				else if (leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0) // stereo observation
				{
					kpUn = pKFi->mvKeysUn[leftIndex];
					const float kp_ur = pKFi->mvuRight[leftIndex];
					Eigen::Matrix<double, 3, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

					EdgeStereo *e = new EdgeStereo(0);

					g2o::OptimizableGraph::Vertex
						*VP = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId));
					if (bAllFixed) {
						if (!VP->fixed()) {
							bAllFixed = false;
						}
					}

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, VP);
					e->setMeasurement(obs);
					const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

					e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberStereo);

					optimizer.addEdge(e);
				}

				if (pKFi->mpCamera2) { // Monocular right observation
					int rightIndex = get<1>(mit->second);

					if (rightIndex != -1 && rightIndex < pKFi->mvKeysRight.size()) {
						rightIndex -= pKFi->NLeft;

						Eigen::Matrix<double, 2, 1> obs;
						kpUn = pKFi->mvKeysRight[rightIndex];
						obs << kpUn.pt.x, kpUn.pt.y;

						EdgeMono *e = new EdgeMono(1);

						g2o::OptimizableGraph::Vertex
							*VP = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId));
						if (bAllFixed) {
							if (!VP->fixed()) {
								bAllFixed = false;
							}
						}

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
						e->setVertex(1, VP);
						e->setMeasurement(obs);
						const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(thHuberMono);

						optimizer.addEdge(e);
					}
				}
			}
		}

		if (bAllFixed) {
			optimizer.removeVertex(vPoint);
			vbNotIncludedMP[i] = true;
		}
	}

	if (pbStopFlag) {
		if (*pbStopFlag) {
			return;
		}
	}

	optimizer.initializeOptimization();
	optimizer.optimize(its);

	// Recover optimized data
	//Keyframes
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
		if (nLoopId == 0) {
			cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
			pKFi->SetPose(Tcw);
		}
		else {
			pKFi->mTcwGBA = cv::Mat::eye(4, 4, CV_32F);
			Converter::toCvMat(VP->estimate().Rcw[0]).copyTo(pKFi->mTcwGBA.rowRange(0, 3).colRange(0, 3));
			Converter::toCvMat(VP->estimate().tcw[0]).copyTo(pKFi->mTcwGBA.rowRange(0, 3).col(3));
			pKFi->mnBAGlobalForKF = nLoopId;
		}
		if (pKFi->bImu) {
			VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
			if (nLoopId == 0) {
				pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
			}
			else {
				pKFi->mVwbGBA = Converter::toCvMat(VV->estimate());
			}

			VertexGyroBias *VG;
			VertexAccBias *VA;
			if (!bInit) {
				VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
				VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
			}
			else {
				VG = static_cast<VertexGyroBias *>(optimizer.vertex(4 * maxKFid + 2));
				VA = static_cast<VertexAccBias *>(optimizer.vertex(4 * maxKFid + 3));
			}

			Vector6d vb;
			vb << VG->estimate(), VA->estimate();
			IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);
			if (nLoopId == 0) {
				pKFi->SetNewBias(b);
			}
			else {
				pKFi->mBiasGBA = b;
			}
		}
	}

	//Points
	for (size_t i = 0; i < vpMPs.size(); i++) {
		if (vbNotIncludedMP[i]) {
			continue;
		}

		MapPoint *pMP = vpMPs[i];
		g2o::VertexSBAPointXYZ
			*vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + iniMPid + 1));

		if (nLoopId == 0) {
			pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
			pMP->UpdateNormalAndDepth();
		}
		else {
			pMP->mPosGBA.create(3, 1, CV_32F);
			Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
			pMP->mnBAGlobalForKF = nLoopId;
		}
	}

	pMap->IncreaseChangeIndex();
}

int Optimizer::PoseOptimization(Frame *pFrame)
{
	// setup solver
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// number of edges
	int nInitialCorrespondences = 0;

	// Set Frame vertex
	g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
	vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);

	// Set MapPoint vertices

	// Number of KeyPoints.
	const int N = pFrame->N;

	// setup edges pointer
	vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
	vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
	vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
	vpEdgesMono.reserve(N);
	vpEdgesMono_FHR.reserve(N);
	vnIndexEdgeMono.reserve(N);
	vnIndexEdgeRight.reserve(N);

	vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesStereo.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	// robust kernel
	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

	// add edges
	{
		unique_lock<mutex> lock(MapPoint::mGlobalMutex);

		// iterate all key points
		// N: Number of KeyPoints in the frame
		for (int i = 0; i < N; i++) {
			MapPoint *pMP = pFrame->mvpMapPoints[i];
			// if the key points has its correspending map points created
			if (pMP) {
				//Conventional SLAM
				if (!pFrame->mpCamera2) {
					// Monocular observation
					// add edge
					if (pFrame->mvuRight[i] < 0) {
						nInitialCorrespondences++;
						pFrame->mvbOutlier[i] = false;

						Eigen::Matrix<double, 2, 1> obs;
						//undistorted feature points
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						obs << kpUn.pt.x, kpUn.pt.y;

						ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(deltaMono);

						e->pCamera = pFrame->mpCamera;
						// xw: position of the map point under world coordinate
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesMono.push_back(e);
						vnIndexEdgeMono.push_back(i);
					}
						// Stereo observation
						// add edge, for those key ponits in both left and right
					else {
						nInitialCorrespondences++;
						pFrame->mvbOutlier[i] = false;

						//SET EDGE
						Eigen::Matrix<double, 3, 1> obs;

						//undistorted feature points
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						// kp_ur: the position of x in thr right image
						const float &kp_ur = pFrame->mvuRight[i];
						obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

						g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
						e->setInformation(Info);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(deltaStereo);

						e->fx = pFrame->fx;
						e->fy = pFrame->fy;
						e->cx = pFrame->cx;
						e->cy = pFrame->cy;
						e->bf = pFrame->mbf;
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesStereo.push_back(e);
						vnIndexEdgeStereo.push_back(i);
					}
				}
					// SLAM with respect a rigid body
					// using stereo, only in one image
					// add edge, for those key ponits only in left or only in right
				else {
					nInitialCorrespondences++;

					cv::KeyPoint kpUn;

					if (i < pFrame->Nleft) { //Left camera observation
						kpUn = pFrame->mvKeys[i];

						pFrame->mvbOutlier[i] = false;

						Eigen::Matrix<double, 2, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y;

						ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(deltaMono);

						e->pCamera = pFrame->mpCamera;
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesMono.push_back(e);
						vnIndexEdgeMono.push_back(i);
					}
					else { //Right camera observation
						//continue;
						kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

						Eigen::Matrix<double, 2, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y;

						pFrame->mvbOutlier[i] = false;

						ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody
							*e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(deltaMono);

						e->pCamera = pFrame->mpCamera2;
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						e->mTrl = Converter::toSE3Quat(pFrame->mTrl);

						optimizer.addEdge(e);

						vpEdgesMono_FHR.push_back(e);
						vnIndexEdgeRight.push_back(i);
					}
				}
			}
		}
	}

	//cout << "PO: vnIndexEdgeMono.size() = " << vnIndexEdgeMono.size() << "   vnIndexEdgeRight.size() = " << vnIndexEdgeRight.size() << endl;
	if (nInitialCorrespondences < 3) {
		return 0;
	}

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
	const int its[4] = {10, 10, 10, 10};

	int nBad = 0;
	for (size_t it = 0; it < 4; it++) {
		// set estimate of camera vertex as the pose of current frame
		vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;

		// 3 for loop for check outlier

		// using mono, or using stereo but feature only in left image
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			// ??? what is the meaning of chi2
			const float chi2 = e->chi2();

			if (chi2 > chi2Mono[it]) {
				pFrame->mvbOutlier[idx] = true;
				// ???? what does level work?
				e->setLevel(1);
				nBad++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}
		// using stereo, but feature only in right
		for (size_t i = 0, iend = vpEdgesMono_FHR.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = vpEdgesMono_FHR[i];

			const size_t idx = vnIndexEdgeRight[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Mono[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBad++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}
		// using stereo, feature in both left and right image
		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

			const size_t idx = vnIndexEdgeStereo[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Stereo[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBad++;
			}
			else {
				e->setLevel(0);
				pFrame->mvbOutlier[idx] = false;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		if (optimizer.edges().size() < 10) {
			break;
		}
	}

	// Recover optimized pose and return number of inliers
	g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
	g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	pFrame->SetPose(pose);

	//cout << "[PoseOptimization]: initial correspondences-> " << nInitialCorrespondences << " --- outliers-> " << nBad << endl;

	return nInitialCorrespondences - nBad;
}

void Optimizer::PoseOnlyOptimizationDVLIMU(set<KeyFrame*, KFComparator> &loss_kfs, Atlas* pAtlas, int& optimized_kf_id)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);


    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    long maxKFid = (*loss_kfs.rbegin())->mnId;


	stringstream ss;
	ss << "DVL_IMU_PoseOnly optimization: " << endl;
    // iterate ls_kf from last to first, and add vertex to optimizer, and set the last 10 of them as fixed
    for (auto rit = loss_kfs.rbegin(); rit != loss_kfs.rend(); ++rit) {
        KeyFrame* pKFi = *rit;
        if (pKFi->isBad()){
//             ROS_ERROR_STREAM("bad KF");  // original
            assert(-1);
        }
        if(pKFi->GetPose().empty()){
//             ROS_ERROR_STREAM("empty pose");  // original
            continue;
        }
        VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
        VP->setId(pKFi->mnId);
        if (pKFi->mnId > (maxKFid-4)){
            VP->setFixed(false);
            ss << "optimize: " << pKFi->mnId << endl;
        }
        else {
            VP->setFixed(true);
            ss << "fixed: " << pKFi->mnId << endl;
        }
        // if (pKFi->mnId == (*loss_kfs.rbegin())->mnId) {
        //     VP->setFixed(false);
		// 	ss << "optimizable: " << pKFi->mnId << endl;
        // }
        // else if (pKFi->mnId == (*loss_kfs.begin())->mnId) {
        //     ss << "fixed: " << pKFi->mnId << endl;
        //     VP->setFixed(true);
        // }
		// else if (pKFi->mnId > (optimized_kf_id-4)) {
		// 	VP->setFixed(false);
		// 	ss << "optimizable: " << pKFi->mnId << endl;
		// }
        // else{
        //     VP->setFixed(true);
        //     ss << "fixed: " << pKFi->mnId << endl;
        // }
        optimizer.addVertex(VP);
    }
//     // ROS_INFO_STREAM(ss.str());  // original
    optimized_kf_id = maxKFid;


    // set<KeyFrame*, KFComparator>::reverse_iterator rit;
    // for (rit = loss_kfs.rbegin(); rit != loss_kfs.rend(); ++rit) {
    //     KeyFrame* pKFi = *rit;
    //     if (pKFi->isBad()){
//     //         ROS_ERROR_STREAM("bad KF");  // original
    //         assert(-1);
    //     }
    //     VertexPoseDvlGro *VP = new VertexPoseDvlGro(pKFi);
    //     VP->setId(pKFi->mnId);
    //     VP->setFixed(true);
    //     if (pKFi->mnId == (*loss_kfs.rbegin())->mnId) {
    //         VP->setFixed(false);
    //     }
    //     optimizer.addVertex(VP);
    // }
    // for(auto pKFi:loss_kfs) {
    //     if (pKFi->isBad()){
//     //         ROS_ERROR_STREAM("bad KF");  // original
    //         assert(-1);
    //     }
    //     VertexPoseDvlGro *VP = new VertexPoseDvlGro(pKFi);
    //     VP->setId(pKFi->mnId);
    //     VP->setFixed(true);
    //     if (pKFi->mnId == (*loss_kfs.rbegin())->mnId) {
    //         VP->setFixed(false);
    //     }
    //     optimizer.addVertex(VP);
//     //     // ROS_INFO_STREAM("opt KF: "<<pKFi->mnId);  // original
    // }

    // Biases
    vector<VertexGyroBias*> vpgb;
    vector<VertexAccBias*> vpab;
    //todo_tightly
    //	set fixed for debuging
    for(auto pKFi:loss_kfs) {
        VertexGyroBias *VG = new VertexGyroBias(pKFi);
        VG->setId(maxKFid + 1 + pKFi->mnId);
        VG->setFixed(true);
        optimizer.addVertex(VG);
        vpgb.push_back(VG);

        VertexAccBias *VA = new VertexAccBias(pKFi);
        VA->setId((maxKFid + 1)*2 + pKFi->mnId);
        VA->setFixed(true);
        if (pKFi->GetMap() != (*loss_kfs.begin())->GetMap() && pKFi->mnId > (maxKFid-4)) {
            vpab.push_back(VA);
            VA->setFixed(true);
            // ROS_DEBUG_STREAM("optimizable bias: " << pKFi->mnId);  // original
            RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "optimizable bias: " << pKFi->mnId);
        }
        optimizer.addVertex(VA);

        VertexVelocity *VV = new VertexVelocity(pKFi);
        VV->setId((maxKFid + 1)*3 + pKFi->mnId);
        VV->setFixed(true);
        optimizer.addVertex(VV);
    }

    // extrinsic parameter
    g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
    vT_d_c->setEstimate(Converter::toSE3Quat((*loss_kfs.begin())->mImuCalib.mT_dvl_c));
    vT_d_c->setId((maxKFid + 1)*4);
    vT_d_c->setFixed(true);
    optimizer.addVertex(vT_d_c);

    g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
    vT_g_d->setEstimate(Converter::toSE3Quat((*loss_kfs.begin())->mImuCalib.mT_imu_dvl));
    vT_g_d->setId((maxKFid + 1)*4+1);
    vT_g_d->setFixed(true);
    optimizer.addVertex(vT_g_d);


    Eigen::Matrix3d R_b0_w = pAtlas->getRGravity();
    VertexGDir *VGDir = new VertexGDir(R_b0_w);
    VGDir->setId((maxKFid + 1)*4+2);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);

    // add map points
    set<MapPoint *, MapPointComp> LocalMapPoints;
    for(auto pKFi:loss_kfs) {
        vector<MapPoint *> vpMPs = pKFi->GetMapPointMatches();
        for (vector<MapPoint *>::iterator it = vpMPs.begin(); it != vpMPs.end(); it++) {
            MapPoint *pMP = *it;
            if (pMP) {
                //				cout<<"find local map point"<<endl;
                if (!pMP->isBad()) {
                    if(LocalMapPoints.count(pMP)==0){
                        LocalMapPoints.insert(pMP);
                    }
                }
            }
        }
    }
    int N_map_points = LocalMapPoints.size();

    vector<EdgeMonoBA_DvlGyros *> mono_edges;
    vector<EdgeStereoBA_DvlGyros *> stereo_edges;


    // Graph edges
    vector<EdgeDvlGyroBA *> dvl_edges;
    // dvl_edges.reserve(loss_kfs.size());
    vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
    //	vppUsedKF.reserve(OptKFs.size() + FixedKFs.size());
    //	std::cout << "build optimization graph" << std::endl;
    EdgeDvlIMUInit *bias_edge = NULL;
    EdgeDvlIMU *fixed_bias_edge = NULL;
    std::set<EdgeDvlIMU*> dvlimu_edge;
    std::set<EdgeDvlIMUGravityRefine*> dvlimuG_edge;
    for(auto pKFi:loss_kfs) {
        if (pKFi->mPrevKF&&loss_kfs.count(pKFi->mPrevKF)) {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
                continue;
            }
            // ROS_DEBUG_STREAM("add dvl-imu edge: " << pKFi->mPrevKF->mnId << " -> " << pKFi->mnId);  // original
            RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "add dvl-imu edge: " << pKFi->mPrevKF->mnId << " -> " << pKFi->mnId);
            VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
            //				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
            VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
            //				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex((maxKFid + 1)*3 + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex((maxKFid + 1)*3 + pKFi->mnId);
            g2o::HyperGraph::Vertex *VG1 = optimizer.vertex(maxKFid + 1 + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VA1 = optimizer.vertex((maxKFid + 1) * 2 + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VG2 = optimizer.vertex(maxKFid + 1 + pKFi->mnId);
            g2o::HyperGraph::Vertex *VA2 = optimizer.vertex((maxKFid + 1) * 2 + pKFi->mnId);
            g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex((maxKFid + 1)*4);
            g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex((maxKFid + 1)*4+1);
            g2o::HyperGraph::Vertex *VR_b0_w = optimizer.vertex((maxKFid + 1) * 4 + 2);

            if (!VP1 || !VG2 || !VP2) {
                cout << "Error" << VP1 << ", " << VG2 << ", " << VP2 << endl;

                continue;
            }

            EdgeAccRW* e_bias = new EdgeAccRW();
            e_bias->setLevel(1);
            e_bias->setVertex(0,VA1);
            e_bias->setVertex(1,VA2);
        	Eigen::Matrix3d info_acc_bias = Eigen::Matrix3d::Identity();
        	cv::Mat cvInfoA = pKFi->mpDvlPreintegrationKeyFrame->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
        	cv::cv2eigen(cvInfoA,info_acc_bias);
            e_bias->setInformation(info_acc_bias);
            optimizer.addEdge(e_bias);


        	EdgeGyroRW* eg_bias = new EdgeGyroRW();
        	eg_bias->setLevel(1);
        	eg_bias->setVertex(0,VG1);
        	eg_bias->setVertex(1,VG2);
        	Eigen::Matrix3d info_gyro_bias = Eigen::Matrix3d::Identity()*1e4;
        	cv::Mat cvInfoG = pKFi->mpDvlPreintegrationKeyFrame->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
        	cv::cv2eigen(cvInfoG,info_gyro_bias);
        	eg_bias->setInformation(info_gyro_bias);
        	// if(info_gyro_bias(0,0)==0)
        	optimizer.addEdge(eg_bias);

            EdgeDvlIMU *e_di = new EdgeDvlIMU(pKFi->mpDvlPreintegrationKeyFrame);
            e_di->setLevel(1);
            e_di->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            e_di->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            e_di->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            e_di->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            e_di->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG2));
            e_di->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA2));
            e_di->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
            e_di->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
            e_di->setVertex(8, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VR_b0_w));
            Eigen::Matrix<double,9,9> info = Eigen::Matrix<double,9,9>::Identity();
        	Eigen::Matrix<double,9,9> info_DI=Eigen::Matrix<double,9,9>::Identity();
        	cv::Mat cvInfo = pKFi->mpDvlPreintegrationKeyFrame->C.rowRange(0,9).colRange(0,9).inv(cv::DECOMP_SVD);
        	cv::cv2eigen(cvInfo,info_DI);

            // info.block(0,0,3,3) = Eigen::Matrix3d::Identity() * 100;
            // info(0,0) = info(0,0) * 5e3; // 10_24
            // info(1,1) = info(1,1) * 5e3; // before 10_24
            // info(0,0) = info(0,0) * 1e2; // 10_24
            // info.block(3,3,3,3) = Eigen::Matrix3d::Identity()*100;
            // info.block(6,6,3,3) = Eigen::Matrix3d::Identity()*100;
            e_di->setInformation(info_DI);
            e_di->setId((maxKFid + 1) * 2 + pKFi->mnId);
            optimizer.addEdge(e_di);
            // fixed_bias_edge = e_di;
            // dvlimu_edge.insert(e_di);

        	EdgeDvlGyroBA *ei = new EdgeDvlGyroBA(pKFi->mpDvlPreintegrationKeyFrame);
        	//				ei->setVertex(0, VP1);

        	//			g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        	//			ei->setRobustKernel(rk);
        	//			rk->setDelta(sqrt(7.815));
        	ei->setLevel(0);
        	ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
        	ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
        	ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG2));
        	ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
        	ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
        	Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
        	// set first 3*3 of cov to be 100 * identity matrix
        	cov.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * 100;
        	// set last 3*3 of cov to be 1000 * identity matrix
        	cov.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity() * 100;
        	ei->setInformation(cov);
        	ei->setId(pKFi->mnId);
        	dvl_edges.push_back(ei);
        	optimizer.addEdge(ei);

            EdgeDvlIMUGravityRefine* eg =new EdgeDvlIMUGravityRefine(pKFi->mpDvlPreintegrationKeyFrame);
            eg->setLevel(0);
            eg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            eg->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            eg->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            eg->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            eg->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG2));
            eg->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA2));
            eg->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
            eg->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
            eg->setVertex(8, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VR_b0_w));
            eg->setInformation(Eigen::Matrix<double, 3, 3>::Identity() *100);
            eg->setId((maxKFid+1)*2 + pKFi->mnId);
            optimizer.addEdge(eg);
            // dvlimuG_edge.insert(eg);
        }
    }

    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};

    // Compute error for different scales
    //	std::cout << "start optimization" << std::endl;
    // optimizer.setVerbose(true);

//     // ROS_INFO_STREAM("bias before: ");  // original
    // for(auto p:vpab){
    //     int kf_id = p->id() - (maxKFid + 1)*2;
    //     p->setFixed(false);
//     //     ROS_INFO_STREAM("KF["<<kf_id<<"] bias: "<<p->estimate().transpose());  // original
    // }
    // for(auto eg:dvlimuG_edge){
    //     eg->setLevel(0);
    // }
    // for(auto edi:dvlimu_edge){
    //     edi->setLevel(1);
    // }
    // optimizer.setVerbose(true);
    // optimizer.initializeOptimization(0);
    // optimizer.optimize(5);
//     // ROS_INFO_STREAM("bias after: ");  // original
    // for(auto p:vpab){
    //     int kf_id = p->id() - (maxKFid + 1)*2;
    //     p->setFixed(false);
//     //     ROS_INFO_STREAM("KF["<<kf_id<<"] bias: "<<p->estimate().transpose());  // original
    // }
    //
    // for(auto p:vpab){
    //     p->setFixed(true);
    // }
    // for(auto eg:dvlimuG_edge){
    //     eg->setLevel(1);
    // }
    // for(auto edi:dvlimu_edge){
    //     edi->setLevel(0);
    // }
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    // for(auto p:vpab){
    //     p->setFixed(false);
    // }
    // optimizer.initializeOptimization(0);
    // optimizer.optimize(2);

    // fixed_bias_edge->setLevel(0);
    // optimizer.initializeOptimization(0);
    // optimizer.optimize(2);


    int visual_edge_num = mono_edges.size() + stereo_edges.size();
    int dvl_edge_num = dvl_edges.size();

    //	std::cout << "end optimization" << std::endl;


    // Recover optimized data
    for(auto pKFi:loss_kfs) {
        int kf_id = pKFi->mnId;
        if (pKFi->mnId > maxKFid) {
            continue;
        }
        //Bias
        // int gyros_bias_vertex_id = kf_id + maxKFid + 1;
        // int acc_bias_vertex_id = kf_id + (maxKFid + 1)*2;
        // VertexGyroBias *v_gb = dynamic_cast<VertexGyroBias *>(optimizer.vertex(gyros_bias_vertex_id));
        // VertexAccBias *v_ab = dynamic_cast<VertexAccBias *>(optimizer.vertex(acc_bias_vertex_id));
        // // bg << v_gb->estimate();
        // IMU::Bias b(v_ab->estimate().x(), v_ab->estimate().y(), v_ab->estimate().z(),
        //             v_gb->estimate().x(), v_gb->estimate().y(), v_gb->estimate().z());
        // pKFi->SetNewBias(b);
//         // ROS_INFO_STREAM("recover KF[" << pKFi->mnId << "] bias(gyros acc): " << b.bwx<<","<<b.bwy<<","<<b.bwz<<","<<b.bax<<","<<b.bay<<","<<b.baz);  // original

        // Pose
        VertexPoseDvlIMU *VP = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
        if(VP->fixed()){
            continue;
        }
        Eigen::Quaterniond Rwc(VP->estimate().Rwc);
        Eigen::Vector3d twc = VP->estimate().twc;
        // ROS_DEBUG_STREAM("recover KF[" << pKFi->mnId << "] pose: from"<<pKFi->GetPoseInverse().col(3).rowRange(0,3).t()<<" to: " << twc.transpose());  // original
        RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "recover KF[" << pKFi->mnId << "] pose: from"<<pKFi->GetPoseInverse().col(3).rowRange(0,3).t()<<" to: " << twc.transpose());
        Eigen::Isometry3d Twc = Eigen::Isometry3d::Identity();
        Twc.pretranslate(twc);
        Twc.rotate(Rwc);
        Eigen::Isometry3d Tcw = Twc.inverse();
        cv::Mat Tcw_cv;
        cv::eigen2cv(Tcw.matrix(), Tcw_cv);
        Tcw_cv.convertTo(Tcw_cv, CV_32F);
        pKFi->SetPose(Tcw_cv);
    }
//     // ROS_INFO_STREAM(ss.str());  // original


}

void Optimizer::OptimizationDVLIMU(set<KeyFrame*, KFComparator> &loss_kfs, Atlas* pAtlas, double lamda_DVL)
{
    unique_lock<shared_timed_mutex> lock(pAtlas->GetCurrentMap()->mMutexMapUpdate);
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);


    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    long maxKFid = (*loss_kfs.rbegin())->mnId;

    stringstream ss;
    ss << "DVL_IMU optimization: " << endl;
    // iterate ls_kf from last to first, and add vertex to optimizer
    for (auto pKFi :loss_kfs) {
        if (pKFi->isBad()){
//             ROS_ERROR_STREAM("bad KF");  // original
            assert(-1);
        }
        if(pKFi->GetPose().empty()){
//             ROS_ERROR_STREAM("empty pose");  // original
            continue;
        }
        VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
        VP->setId(pKFi->mnId);
        if (pKFi->mnId == (*loss_kfs.begin())->mnId) {
            VP->setFixed(true);
            ss << "fixed: " << pKFi->mnId << endl;
        }
        else if(pKFi->GetMap()!=(*loss_kfs.begin())->GetMap()){
            VP->setFixed(false);
            ss << "optimizable: " << pKFi->mnId << endl;
        }
        else{
            VP->setFixed(true);
            ss << "fixed: " << pKFi->mnId << endl;
        }
        optimizer.addVertex(VP);
    }
//     ROS_INFO_STREAM(ss.str());  // original




    // Biases
    vector<VertexGyroBias*> vpgb;
    vector<VertexAccBias*> vpab;
    //todo_tightly
    //	set fixed for debuging
    for(auto pKFi:loss_kfs) {
        VertexGyroBias *VG = new VertexGyroBias(pKFi);
        VG->setId(maxKFid + 1 + pKFi->mnId);
        VG->setFixed(true);
        optimizer.addVertex(VG);
        vpgb.push_back(VG);

        VertexAccBias *VA = new VertexAccBias(pKFi);
        VA->setId((maxKFid + 1)*2 + pKFi->mnId);
        VA->setFixed(true);
        optimizer.addVertex(VA);
        if(pKFi->GetMap()!=(*loss_kfs.begin())->GetMap()){
            vpab.push_back(VA);
        }


        VertexVelocity *VV = new VertexVelocity(pKFi);
        VV->setId((maxKFid + 1)*3 + pKFi->mnId);
        VV->setFixed(true);
        optimizer.addVertex(VV);
    }

    // extrinsic parameter
    g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
    vT_d_c->setEstimate(Converter::toSE3Quat((*loss_kfs.begin())->mImuCalib.mT_dvl_c));
    vT_d_c->setId((maxKFid + 1)*4);
    vT_d_c->setFixed(true);
    optimizer.addVertex(vT_d_c);

    g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
    vT_g_d->setEstimate(Converter::toSE3Quat((*loss_kfs.begin())->mImuCalib.mT_imu_dvl));
    vT_g_d->setId((maxKFid + 1)*4+1);
    vT_g_d->setFixed(true);
    optimizer.addVertex(vT_g_d);


    Eigen::Matrix3d R_b0_w = pAtlas->getRGravity();
    VertexGDir *VGDir = new VertexGDir(R_b0_w);
    VGDir->setId((maxKFid + 1)*4+2);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);

    // add map points
    set<MapPoint *, MapPointComp> LocalMapPoints;
    for(auto pKFi:loss_kfs) {
        vector<MapPoint *> vpMPs = pKFi->GetMapPointMatches();
        for (vector<MapPoint *>::iterator it = vpMPs.begin(); it != vpMPs.end(); it++) {
            MapPoint *pMP = *it;
            if (pMP) {
                //				cout<<"find local map point"<<endl;
                if (!pMP->isBad()) {
                    if(LocalMapPoints.count(pMP)==0){
                        // LocalMapPoints.insert(pMP);
                    }
                }
            }
        }
    }
    int N_map_points = LocalMapPoints.size();


    vector<EdgeMonoBA_DvlGyros *> mono_edges;
    vector<EdgeStereoBA_DvlGyros *> stereo_edges;

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (auto pMP: LocalMapPoints) {
            if (pMP) {
                g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
                vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
                int id = pMP->mnId + (maxKFid + 1)*5;
                vPoint->setId(id);
                if (pMP->GetMap()==(*loss_kfs.begin())->GetMap()){
                    vPoint->setFixed(true);
                }
                else{
                    vPoint->setFixed(false);
                }
                vPoint->setMarginalized(true);
                optimizer.addVertex(vPoint);

                const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

                for (auto ob: observations) {
                    KeyFrame *pKFi = ob.first;
                    if(!loss_kfs.count(pKFi)){
                        continue;
                    }
                    if (!pKFi->isBad()) {
                        const int leftIndex = get<0>(ob.second);

                        // Monocular observation
                        if (leftIndex != -1 && pKFi->mvuRight[get<0>(ob.second)] < 0) {
                            const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                            Eigen::Matrix<double, 2, 1> obs;
                            obs << kpUn.pt.x, kpUn.pt.y;

                            EdgeMonoBA_DvlGyros *e = new EdgeMonoBA_DvlGyros();
                            e->setLevel(0);
                            e->setVertex(0,
                                         dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                            e->setVertex(1, vPoint);
                            e->setMeasurement(obs);
                            const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                            e->setInformation(Eigen::Matrix2d::Identity() * invSigma2 * 0.0001);

                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(sqrt(5.991));

                            mono_edges.push_back(e);
                            optimizer.addEdge(e);
                        }
                        else if (leftIndex != -1 && pKFi->mvuRight[get<0>(ob.second)] >= 0) // Stereo observation
                        {
                            const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                            Eigen::Matrix<double, 3, 1> obs;
                            const float kp_ur = pKFi->mvuRight[get<0>(ob.second)];
                            obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                            EdgeStereoBA_DvlGyros *e = new EdgeStereoBA_DvlGyros();
                            e->setLevel(0);
                            e->setVertex(0,
                                         dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                            e->setVertex(1, vPoint);
                            e->setMeasurement(obs);
                            const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                            Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2 * 0.0001;
                            e->setInformation(Info);

                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(sqrt(7.815));

                            stereo_edges.push_back(e);
                            optimizer.addEdge(e);
                        }

                    }

                }
            }
        }
    }

    // Graph edges
    vector<EdgeDvlGyroBA *> dvl_edges;
    // dvl_edges.reserve(loss_kfs.size());
    vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
    //	vppUsedKF.reserve(OptKFs.size() + FixedKFs.size());
    //	std::cout << "build optimization graph" << std::endl;
    EdgeDvlIMUInit *bias_edge = NULL;
    EdgeDvlIMU *fixed_bias_edge = NULL;
    for(auto pKFi:loss_kfs) {
        if (loss_kfs.count(pKFi->mPrevKF)) {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
                continue;
            }
            // ROS_DEBUG_STREAM("add dvl-imu edge: " << pKFi->mPrevKF->mnId << " -> " << pKFi->mnId);  // original
            RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "add dvl-imu edge: " << pKFi->mPrevKF->mnId << " -> " << pKFi->mnId);
            VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
            //				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
            VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
            //				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex((maxKFid + 1)*3 + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex((maxKFid + 1)*3 + pKFi->mnId);
            g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid + 1 + pKFi->mnId);
            g2o::HyperGraph::Vertex *VA = optimizer.vertex((maxKFid + 1)*2 + pKFi->mnId);
            g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex((maxKFid + 1)*4);
            g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex((maxKFid + 1)*4+1);
            g2o::HyperGraph::Vertex *VR_b0_w = optimizer.vertex((maxKFid + 1) * 4 + 2);

            if (!VP1 || !VG || !VP2) {
                cout << "Error" << VP1 << ", " << VG << ", " << VP2 << endl;

                continue;
            }

            std::vector<cv::Point3d> vbias_a, vbias_g;
            vbias_a.push_back(cv::Point3d(pKFi->mPrevKF->GetImuBias().bax, pKFi->mPrevKF->GetImuBias().bay,
                                          pKFi->mPrevKF->GetImuBias().baz));
            vbias_g.push_back(cv::Point3d(pKFi->mPrevKF->GetImuBias().bwx, pKFi->mPrevKF->GetImuBias().bwy,
                                          pKFi->mPrevKF->GetImuBias().bwz));
            cv::Mat acc_prior = cv::Mat(vbias_a);
            cv::Mat gyr_prior = cv::Mat(vbias_g);
            acc_prior.convertTo(acc_prior, CV_32F);
            gyr_prior.convertTo(gyr_prior, CV_32F);
            acc_prior = acc_prior.reshape(1);
            gyr_prior = gyr_prior.reshape(1);
            EdgePriorAcc *epa = new EdgePriorAcc(acc_prior);
            epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            epa->setLevel(0);
            epa->setInformation(Eigen::Matrix3d::Identity() * 100);
            optimizer.addEdge(epa);

            EdgePriorGyro *epg = new EdgePriorGyro(gyr_prior);
            epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            epg->setLevel(0);
            epg->setInformation(Eigen::Matrix3d::Identity() * 100);
            optimizer.addEdge(epg);

            //velocity edge
            Eigen::Vector3d dvl_v1;
            pKFi->mPrevKF->GetDvlVelocityMeasurement(dvl_v1);
            Eigen::Vector3d dvl_v2;
            pKFi->GetDvlVelocityMeasurement(dvl_v2);
            EdgeDvlVelocity *ev = new EdgeDvlVelocity(dvl_v1);
            ev->setLevel(0);
            ev->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            if(pKFi->mbDVL){
                ev->setInformation(Eigen::Matrix3d::Identity()*100);
            }
            else{
                ev->setInformation(Eigen::Matrix3d::Identity()*0.1);
            }
            optimizer.addEdge(ev);
            // add edge for v2
            EdgeDvlVelocity *ev2 = new EdgeDvlVelocity(dvl_v2);
            ev2->setLevel(0);
            ev2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            if (pKFi->mbDVL) {
                ev2->setInformation(Eigen::Matrix3d::Identity() * 100);
            }
            else {
                ev2->setInformation(Eigen::Matrix3d::Identity()*0.1);
            }
            optimizer.addEdge(ev2);

            //				EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            EdgeDvlGyroBA *ei = new EdgeDvlGyroBA(pKFi->mpDvlPreintegrationKeyFrame);
            //				ei->setVertex(0, VP1);

            //			g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            //			ei->setRobustKernel(rk);
            //			rk->setDelta(sqrt(7.815));
            ei->setLevel(1);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
            Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
            // set first 3*3 of cov to be 100 * identity matrix
            cov.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * 100;
            // set last 3*3 of cov to be 1000 * identity matrix
            cov.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity() * 100;
            ei->setInformation(cov);
            ei->setId(pKFi->mnId);
            dvl_edges.push_back(ei);
            optimizer.addEdge(ei);

            EdgeDvlIMU *eG = new EdgeDvlIMU(pKFi->mpDvlPreintegrationKeyFrame);
            eG->setLevel(0);
            eG->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            eG->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            eG->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            eG->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            eG->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            eG->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            eG->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
            eG->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
            eG->setVertex(8, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VR_b0_w));
            eG->setInformation(Eigen::Matrix<double, 9, 9>::Identity() *1000);
            eG->setId((maxKFid+1)*2 + pKFi->mnId);
            optimizer.addEdge(eG);
            fixed_bias_edge = eG;
        }
    }

    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};

    // Compute error for different scales
    //	std::cout << "start optimization" << std::endl;
    // optimizer.setVerbose(true);

    optimizer.initializeOptimization(0);
    optimizer.optimize(20);

    // for(auto p:vpab){
    //     p->setFixed(false);
    // }
    // optimizer.initializeOptimization(0);
    // optimizer.optimize(2);
    //
    // fixed_bias_edge->setLevel(0);
    // optimizer.initializeOptimization(0);
    // optimizer.optimize(2);


    int visual_edge_num = mono_edges.size() + stereo_edges.size();
    int dvl_edge_num = dvl_edges.size();

    //	std::cout << "end optimization" << std::endl;


    // Recover optimized data
    for (auto pMap:pAtlas->GetAllMaps()) {
        for(auto pKFi:loss_kfs) {
            int kf_id = pKFi->mnId;
            if (pKFi->mnId > maxKFid) {
                continue;
            }
            if (pKFi->GetMap()!=pMap){
                continue;
            }
            //Bias
            int gyros_bias_vertex_id = kf_id + maxKFid + 1;
            int acc_bias_vertex_id = kf_id + (maxKFid + 1)*2;
            VertexGyroBias *v_gb = dynamic_cast<VertexGyroBias *>(optimizer.vertex(gyros_bias_vertex_id));
            VertexAccBias *v_ab = dynamic_cast<VertexAccBias *>(optimizer.vertex(acc_bias_vertex_id));
            // bg << v_gb->estimate();
            IMU::Bias b(v_ab->estimate().x(), v_ab->estimate().y(), v_ab->estimate().z(),
                        v_gb->estimate().x(), v_gb->estimate().y(), v_gb->estimate().z());
            pKFi->SetNewBias(b);
            // ROS_DEBUG_STREAM("recover KF[" << pKFi->mnId << "] bias: " << b);  // original
            RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "recover KF[" << pKFi->mnId << "] bias: " << b);

            // Pose
            VertexPoseDvlIMU *VP = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
            if(VP->fixed()){
                continue;
            }
            Eigen::Quaterniond Rwc(VP->estimate().Rwc);
            Eigen::Vector3d twc = VP->estimate().twc;
            // ROS_DEBUG_STREAM("recover KF[" << pKFi->mnId << "] pose: from"<<pKFi->GetPoseInverse().col(3).rowRange(0,3).t()<<" to: " << twc.transpose());  // original
            RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "recover KF[" << pKFi->mnId << "] pose: from"<<pKFi->GetPoseInverse().col(3).rowRange(0,3).t()<<" to: " << twc.transpose());
            Eigen::Isometry3d Twc = Eigen::Isometry3d::Identity();
            Twc.pretranslate(twc);
            Twc.rotate(Rwc);
            Eigen::Isometry3d Tcw = Twc.inverse();
            cv::Mat Tcw_cv;
            cv::eigen2cv(Tcw.matrix(), Tcw_cv);
            Tcw_cv.convertTo(Tcw_cv, CV_32F);
            pKFi->SetPose(Tcw_cv);
        }
        for (auto pMP:LocalMapPoints) {
            if(pMP->isBad()){
                continue;
            }
            if (pMP->GetMap()!=pMap){
                continue;
            }
            int id = pMP->mnId + (maxKFid + 1)*5;
            g2o::VertexSBAPointXYZ
                    *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex((maxKFid + 1) * 5 + pMP->mnId));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
    }
}


int Optimizer::PoseOptimizationWithBA_and_EKF(Frame *pFrame, Frame *pLastFrame, double lamda_visual, double lamda_DVL)
{
	// setup solver
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// number of edges
	int nInitialCorrespondences = 0;

	// Set Frame vertex
	g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
	vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);

	// Set MapPoint vertices

	// Number of KeyPoints.
	const int N = pFrame->N;

	// setup edges pointer
	vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
	vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
	vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
	vpEdgesMono.reserve(N);
	vpEdgesMono_FHR.reserve(N);
	vnIndexEdgeMono.reserve(N);
	vnIndexEdgeRight.reserve(N);

	vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesStereo.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	// robust kernel
	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

	// add edges
	{
		unique_lock<mutex> lock(MapPoint::mGlobalMutex);

		// iterate all key points
		// N: Number of KeyPoints in the frame
		for (int i = 0; i < N; i++) {
			MapPoint *pMP = pFrame->mvpMapPoints[i];
			// if the key points has its correspending map points created
			if (pMP) {
				//Conventional SLAM
				double visual_lamda = lamda_visual;
				double ekf_lamda = lamda_DVL;
				if (!pFrame->mpCamera2) {
					// Monocular observation
					// add edge
					if (pFrame->mvuRight[i] < 0) {
						nInitialCorrespondences++;
						pFrame->mvbOutlier[i] = false;

						Eigen::Matrix<double, 2, 1> obs;
						//undistorted feature points
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						obs << kpUn.pt.x, kpUn.pt.y;

						ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						// TODO remove setRobustKernel
						e->setRobustKernel(rk);
						rk->setDelta(deltaMono);

						e->pCamera = pFrame->mpCamera;
						// xw: position of the map point under world coordinate
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesMono.push_back(e);
						vnIndexEdgeMono.push_back(i);
					}
						// Stereo observation
						// add edge, for those key ponits in both left and right
					else {
						nInitialCorrespondences++;
						pFrame->mvbOutlier[i] = false;

						//SET EDGE
						Eigen::Matrix<double, 3, 1> obs;

						//undistorted feature points
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						// kp_ur: the position of x in thr right image
						const float &kp_ur = pFrame->mvuRight[i];
						obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

						g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
						e->setInformation(Info);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(deltaStereo);

						e->fx = pFrame->fx;
						e->fy = pFrame->fy;
						e->cx = pFrame->cx;
						e->cy = pFrame->cy;
						e->bf = pFrame->mbf;
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesStereo.push_back(e);
						vnIndexEdgeStereo.push_back(i);
					}
				}
					// SLAM with respect a rigid body
					// using stereo, only in one image
					// add edge, for those key ponits only in left or only in right
				else {
					nInitialCorrespondences++;

					cv::KeyPoint kpUn;

					if (i < pFrame->Nleft) { //Left camera observation
						kpUn = pFrame->mvKeys[i];

						pFrame->mvbOutlier[i] = false;

						Eigen::Matrix<double, 2, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y;

						ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2 * visual_lamda);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(deltaMono);

						e->pCamera = pFrame->mpCamera;
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesMono.push_back(e);
						vnIndexEdgeMono.push_back(i);
					}
					else { //Right camera observation
						//continue;
						kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

						Eigen::Matrix<double, 2, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y;

						pFrame->mvbOutlier[i] = false;

						ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody
							*e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2 * visual_lamda);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(deltaMono);

						e->pCamera = pFrame->mpCamera2;
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						e->mTrl = Converter::toSE3Quat(pFrame->mTrl);

						optimizer.addEdge(e);

						vpEdgesMono_FHR.push_back(e);
						vnIndexEdgeRight.push_back(i);
					}
				}

				// add EKF constrains
				// the position map point under Last camera frame
				Eigen::Vector3d p_ci_test;
				cv::Mat P_c0 = pMP->GetWorldPos();
				Eigen::Vector3d p_c0;
				p_c0 << P_c0.at<float>(0), P_c0.at<float>(1), P_c0.at<float>(2);
				cv::Mat T_ciw = pLastFrame->mTcw;
				Eigen::Isometry3d T_ci_c0 = Eigen::Isometry3d::Identity();
				cv::cv2eigen(T_ciw, T_ci_c0.matrix());
				EdgeSE3DVLPoseOnly *edge = new EdgeSE3DVLPoseOnly(pFrame->mT_e0_ej.inverse(),
				                                                  pLastFrame->mT_e0_ej.inverse(),
				                                                  T_ci_c0,
				                                                  pFrame->mT_e_c,
				                                                  p_c0);
				edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//					edge->setRobustKernel(rk);
//					rk->setDelta(deltaMono);
				// edge->setId(index);
				// set information according to covariance of EKF
				edge->setInformation(ekf_lamda * Eigen::Matrix3d::Identity());
				optimizer.addEdge(edge);
			}
		}
	}

	//cout << "PO: vnIndexEdgeMono.size() = " << vnIndexEdgeMono.size() << "   vnIndexEdgeRight.size() = " << vnIndexEdgeRight.size() << endl;
	if (nInitialCorrespondences < 3) {
		return 0;
	}

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
	const int its[4] = {10, 10, 10, 10};

	int nBad = 0;
	for (size_t it = 0; it < 4; it++) {
		// set estimate of camera vertex as the pose of current frame
		vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;

		// 3 for loop for check outlier

		// using mono, or using stereo but feature only in left image
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			// ??? what is the meaning of chi2
			const float chi2 = e->chi2();

			if (chi2 > chi2Mono[it]) {
				pFrame->mvbOutlier[idx] = true;
				// ???? what does level work?
				e->setLevel(1);
				nBad++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}
		// using stereo, but feature only in right
		for (size_t i = 0, iend = vpEdgesMono_FHR.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = vpEdgesMono_FHR[i];

			const size_t idx = vnIndexEdgeRight[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Mono[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBad++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}
		// using stereo, feature in both left and right image
		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

			const size_t idx = vnIndexEdgeStereo[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Stereo[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBad++;
			}
			else {
				e->setLevel(0);
				pFrame->mvbOutlier[idx] = false;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		if (optimizer.edges().size() < 10) {
			break;
		}
	}

	// Recover optimized pose and return number of inliers
	g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
	g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	pFrame->SetPose(pose);

	//cout << "[PoseOptimization]: initial correspondences-> " << nInitialCorrespondences << " --- outliers-> " << nBad << endl;

	return nInitialCorrespondences - nBad;
}

int Optimizer::PoseOptimizationWithBA_and_EKF2(Frame *pFrame, Frame *pLastFrame, double lamda_visual, double lamda_DVL)
{
	// setup solver
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// number of edges
	int nInitialCorrespondences = 0;

	// Set Frame vertex
	g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
	vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);

	// Set MapPoint vertices

	// Number of KeyPoints.
	const int N = pFrame->N;

	// setup edges pointer
	vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
	vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
	vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
	vpEdgesMono.reserve(N);
	vpEdgesMono_FHR.reserve(N);
	vnIndexEdgeMono.reserve(N);
	vnIndexEdgeRight.reserve(N);

	vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesStereo.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	// robust kernel
	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

	// add edges
	{
		unique_lock<mutex> lock(MapPoint::mGlobalMutex);

		// iterate all key points
		// N: Number of KeyPoints in the frame
		for (int i = 0; i < N; i++) {
			MapPoint *pMP = pFrame->mvpMapPoints[i];
			// if the key points has its correspending map points created
			if (pMP) {
				//Conventional SLAM
				double visual_lamda = lamda_visual;
				double ekf_lamda = lamda_DVL;
				if (!pFrame->mpCamera2) {
					// Monocular observation
					// add edge
					if (pFrame->mvuRight[i] < 0) {
						nInitialCorrespondences++;
						pFrame->mvbOutlier[i] = false;

						Eigen::Matrix<double, 2, 1> obs;
						//undistorted feature points
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						obs << kpUn.pt.x, kpUn.pt.y;

						ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

//                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						// TODO remove setRobustKernel
//                            e->setRobustKernel(rk);
//                            rk->setDelta(deltaMono);

						e->pCamera = pFrame->mpCamera;
						// xw: position of the map point under world coordinate
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesMono.push_back(e);
						vnIndexEdgeMono.push_back(i);
					}
						// Stereo observation
						// add edge, for those key ponits in both left and right
					else {
						nInitialCorrespondences++;
						pFrame->mvbOutlier[i] = false;

						//SET EDGE
						Eigen::Matrix<double, 3, 1> obs;

						//undistorted feature points
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						// kp_ur: the position of x in thr right image
						const float &kp_ur = pFrame->mvuRight[i];
						obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

						g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
						e->setInformation(Info);

//                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                            e->setRobustKernel(rk);
//                            rk->setDelta(deltaStereo);

						e->fx = pFrame->fx;
						e->fy = pFrame->fy;
						e->cx = pFrame->cx;
						e->cy = pFrame->cy;
						e->bf = pFrame->mbf;
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesStereo.push_back(e);
						vnIndexEdgeStereo.push_back(i);
					}
				}
					// SLAM with respect a rigid body
					// using stereo, only in one image
					// add edge, for those key ponits only in left or only in right
				else {
					nInitialCorrespondences++;

					cv::KeyPoint kpUn;

					if (i < pFrame->Nleft) { //Left camera observation
						kpUn = pFrame->mvKeys[i];

						pFrame->mvbOutlier[i] = false;

						Eigen::Matrix<double, 2, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y;

						ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2 * visual_lamda);

//                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                            e->setRobustKernel(rk);
//                            rk->setDelta(deltaMono);

						e->pCamera = pFrame->mpCamera;
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesMono.push_back(e);
						vnIndexEdgeMono.push_back(i);
					}
					else { //Right camera observation
						//continue;
						kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

						Eigen::Matrix<double, 2, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y;

						pFrame->mvbOutlier[i] = false;

						ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody
							*e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2 * visual_lamda);

//                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                            e->setRobustKernel(rk);
//                            rk->setDelta(deltaMono);

						e->pCamera = pFrame->mpCamera2;
						cv::Mat Xw = pMP->GetWorldPos();
						e->Xw[0] = Xw.at<float>(0);
						e->Xw[1] = Xw.at<float>(1);
						e->Xw[2] = Xw.at<float>(2);

						e->mTrl = Converter::toSE3Quat(pFrame->mTrl);

						optimizer.addEdge(e);

						vpEdgesMono_FHR.push_back(e);
						vnIndexEdgeRight.push_back(i);
					}
				}

				// add EKF constrains
				// the position map point under Last camera frame
			}
		}

		Eigen::Vector3d p_ci_test;
		cv::Mat T_ciw = pLastFrame->mTcw;
		Eigen::Isometry3d T_ci_c0 = Eigen::Isometry3d::Identity();
		cv::cv2eigen(T_ciw, T_ci_c0.matrix());
		EdgeSE3DVLPoseOnly2 *edge = new EdgeSE3DVLPoseOnly2(pFrame->mT_e0_ej.inverse(),
		                                                    pLastFrame->mT_e0_ej.inverse(),
		                                                    T_ci_c0,
		                                                    pFrame->mT_e_c);
		edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//					edge->setRobustKernel(rk);
//					rk->setDelta(deltaMono);
		// edge->setId(index);
		// set information according to covariance of EKF
		edge->setInformation(lamda_DVL * Eigen::Matrix<double, 6, 6>::Identity());
		optimizer.addEdge(edge);
	}

	//cout << "PO: vnIndexEdgeMono.size() = " << vnIndexEdgeMono.size() << "   vnIndexEdgeRight.size() = " << vnIndexEdgeRight.size() << endl;
	if (nInitialCorrespondences < 3) {
		return 0;
	}

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
	const int its[4] = {10, 10, 10, 10};

	int nBad = 0;
	for (size_t it = 0; it < 4; it++) {
		// set estimate of camera vertex as the pose of current frame
		vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;

		// 3 for loop for check outlier

		// using mono, or using stereo but feature only in left image
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			// ??? what is the meaning of chi2
			const float chi2 = e->chi2();

			if (chi2 > chi2Mono[it]) {
				pFrame->mvbOutlier[idx] = true;
				// ???? what does level work?
				e->setLevel(1);
				nBad++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}
		// using stereo, but feature only in right
		for (size_t i = 0, iend = vpEdgesMono_FHR.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = vpEdgesMono_FHR[i];

			const size_t idx = vnIndexEdgeRight[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Mono[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBad++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}
		// using stereo, feature in both left and right image
		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

			const size_t idx = vnIndexEdgeStereo[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Stereo[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBad++;
			}
			else {
				e->setLevel(0);
				pFrame->mvbOutlier[idx] = false;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		if (optimizer.edges().size() < 10) {
			break;
		}
	}

	// Recover optimized pose and return number of inliers
	g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
	g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	pFrame->SetPose(pose);

	//cout << "[PoseOptimization]: initial correspondences-> " << nInitialCorrespondences << " --- outliers-> " << nBad << endl;

	return nInitialCorrespondences - nBad;
}

void Optimizer::PoseOptimizationWithEKF(Frame *pFrame, Frame *pLastFrame)
{
	// setup solver
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// number of edges
	int nInitialCorrespondences = 0;

	// Set Frame vertex
	g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
	vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);

	// Set MapPoint vertices

	// Number of KeyPoints.
	const int N = pFrame->N;

	// setup edges pointer
	vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
	vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
	vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
	vpEdgesMono.reserve(N);
	vpEdgesMono_FHR.reserve(N);
	vnIndexEdgeMono.reserve(N);
	vnIndexEdgeRight.reserve(N);

	vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesStereo.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	// robust kernel
	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

	// add edges
	{
		unique_lock<mutex> lock(MapPoint::mGlobalMutex);

		// iterate all key points
		// N: Number of KeyPoints in the frame
		for (int i = 0; i < N; i++) {
			MapPoint *pMP = pFrame->mvpMapPoints[i];
			// if the key points has its correspending map points created
			if (pMP) {

				// add EKF constrains
				// the position map point under Last camera frame
				Eigen::Vector3d p_ci_test;
				cv::Mat P_c0 = pMP->GetWorldPos();
				Eigen::Vector3d p_c0;
				p_c0 << P_c0.at<float>(0), P_c0.at<float>(1), P_c0.at<float>(2);
				cv::Mat T_ciw = pLastFrame->mTcw;
				Eigen::Isometry3d T_ci_c0 = Eigen::Isometry3d::Identity();
				cv::cv2eigen(T_ciw, T_ci_c0.matrix());
				EdgeSE3DVLPoseOnly *edge = new EdgeSE3DVLPoseOnly(pFrame->mT_e0_ej.inverse(),
				                                                  pLastFrame->mT_e0_ej.inverse(),
				                                                  T_ci_c0,
				                                                  pFrame->mT_e_c,
				                                                  p_c0);
				edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//					edge->setRobustKernel(rk);
//					rk->setDelta(deltaMono);
				// edge->setId(index);
				// set information according to covariance of EKF
				edge->setInformation(Eigen::Matrix3d::Identity());
				optimizer.addEdge(edge);
			}
		}
	}
	optimizer.initializeOptimization(0);
	optimizer.optimize(20);



	// Recover optimized pose and return number of inliers
	g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
	g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	pFrame->SetPose(pose);

	//cout << "[PoseOptimization]: initial correspondences-> " << nInitialCorrespondences << " --- outliers-> " << nBad << endl;

}

// ??? what is vpNonEnoughOptKFs
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, vector<KeyFrame *> &vpNonEnoughOptKFs)
{
	// Local KeyFrames: First Breath Search from Current Keyframe
	list<KeyFrame *> lLocalKeyFrames;

	lLocalKeyFrames.push_back(pKF);

	// ??? what is mnBALocalForKF
	pKF->mnBALocalForKF = pKF->mnId;
	Map *pCurrentMap = pKF->GetMap();

	// get all key frames share covisibility
	const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
	// add all key frames share covisibility to lLocalKeyFrames
	for (int i = 0, iend = vNeighKFs.size(); i < iend; i++) {
		KeyFrame *pKFi = vNeighKFs[i];
		// ??? what is mnBALocalForKF
		pKFi->mnBALocalForKF = pKF->mnId;
		if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
			lLocalKeyFrames.push_back(pKFi);
		}
	}
	// add some of key frames in vpNonEnoughOptKFs to lLocalKeyFrames
	for (KeyFrame *pKFi: vpNonEnoughOptKFs) {
		// pKFi is not bad, pKFi is in the same map with pKF, and pKFi is not in lLocalKeyFrames
		if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap && pKFi->mnBALocalForKF != pKF->mnId) {
			pKFi->mnBALocalForKF = pKF->mnId;
			lLocalKeyFrames.push_back(pKFi);
		}
	}

	// Local MapPoints seen in Local KeyFrames
	list<MapPoint *> lLocalMapPoints;
	set<MapPoint *> sNumObsMP;
	// ??? what is num_fixedKF
	int num_fixedKF;
	// add all map points associated to keypoints in the key frames in lLocalKeyFrames to lLocalMapPoints
	for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		if (pKFi->mnId == pCurrentMap->GetInitKFid()) {
			num_fixedKF = 1;
		}
		// get all MapPoints associated to keypoints in the key frame pKFi
		vector<MapPoint *> vpMPs = pKFi->GetMapPointMatches();
		for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
			MapPoint *pMP = *vit;
			if (pMP) {
				if (!pMP->isBad() && pMP->GetMap() == pCurrentMap) {

					if (pMP->mnBALocalForKF != pKF->mnId) {
						lLocalMapPoints.push_back(pMP);
						pMP->mnBALocalForKF = pKF->mnId;
					}
				}
			}
		}
	}

	// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
	list<KeyFrame *> lFixedCameras;
	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		map<KeyFrame *, tuple<int, int>> observations = (*lit)->GetObservations();
		for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
				pKFi->mnBAFixedForKF = pKF->mnId;
				if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
					lFixedCameras.push_back(pKFi);
				}
			}
		}
	}
	num_fixedKF = lFixedCameras.size() + num_fixedKF;

	// if fixedKF is too less, add fixed key franme
	if (num_fixedKF < 2) {
		//Verbose::PrintMess("LM-LBA: New Fixed KFs had been set", Verbose::VERBOSITY_NORMAL);
		//TODO We set 2 KFs to fixed to avoid a degree of freedom in scale
		list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin();
		int lowerId = pKF->mnId;
		KeyFrame *pLowerKf;
		int secondLowerId = pKF->mnId;
		KeyFrame *pSecondLowerKF;

		for (; lit != lLocalKeyFrames.end(); lit++) {
			KeyFrame *pKFi = *lit;
			if (pKFi == pKF || pKFi->mnId == pCurrentMap->GetInitKFid()) {
				continue;
			}

			if (pKFi->mnId < lowerId) {
				lowerId = pKFi->mnId;
				pLowerKf = pKFi;
			}
			else if (pKFi->mnId < secondLowerId) {
				secondLowerId = pKFi->mnId;
				pSecondLowerKF = pKFi;
			}
		}
		lFixedCameras.push_back(pLowerKf);
		lLocalKeyFrames.remove(pLowerKf);
		num_fixedKF++;
		if (num_fixedKF < 2) {
			lFixedCameras.push_back(pSecondLowerKF);
			lLocalKeyFrames.remove(pSecondLowerKF);
			num_fixedKF++;
		}
	}

	if (num_fixedKF == 0) {
		Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
		//return;
	}
	//Verbose::PrintMess("LM-LBA: There are " + to_string(lLocalKeyFrames.size()) + " KFs and " + to_string(lLocalMapPoints.size()) + " MPs to optimize. " + to_string(num_fixedKF) + " KFs are fixed", Verbose::VERBOSITY_DEBUG);

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	if (pCurrentMap->IsInertial()) {
		solver->setUserLambdaInit(100.0);
	} // TODO uncomment
	//cout << "LM-LBA: lambda init: " << solver->userLambdaInit() << endl;

	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);

	if (pbStopFlag) {
		optimizer.setForceStopFlag(pbStopFlag);
	}

	unsigned long maxKFid = 0;

	// Set Local KeyFrame vertices
	for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
		vSE3->setId(pKFi->mnId);
		// fixed the first key frame of map
		vSE3->setFixed(pKFi->mnId == pCurrentMap->GetInitKFid());
		optimizer.addVertex(vSE3);
		if (pKFi->mnId > maxKFid) {
			maxKFid = pKFi->mnId;
		}
	}

	// Set Fixed KeyFrame vertices
	for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
		vSE3->setId(pKFi->mnId);
		vSE3->setFixed(true);
		optimizer.addVertex(vSE3);
		if (pKFi->mnId > maxKFid) {
			maxKFid = pKFi->mnId;
		}
	}

	Verbose::PrintMess(
		"LM-LBA: opt/fixed KFs: " + to_string(lLocalKeyFrames.size()) + "/" + to_string(lFixedCameras.size()),
		Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess("LM-LBA: local MPs: " + to_string(lLocalMapPoints.size()), Verbose::VERBOSITY_DEBUG);

	// Set MapPoint vertices
	const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

	vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody *> vpEdgesBody;
	vpEdgesBody.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFBody;
	vpEdgeKFBody.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeBody;
	vpMapPointEdgeBody.reserve(nExpectedSize);

	vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
	vpEdgesStereo.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFStereo;
	vpEdgeKFStereo.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeStereo;
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	int nPoints = 0;

	int nKFs = lLocalKeyFrames.size() + lFixedCameras.size(), nEdges = 0;

	// add map points and edges
	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnId + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);
		nPoints++;

		const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

		//Set edges
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
				const int cam0Index = get<0>(mit->second);

				// Monocular observation of Camera 0
				if (cam0Index != -1 && pKFi->mvuRight[cam0Index] < 0) {
					const cv::KeyPoint &kpUn = pKFi->mvKeysUn[cam0Index];
					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					e->pCamera = pKFi->mpCamera;

					optimizer.addEdge(e);
					vpEdgesMono.push_back(e);
					vpEdgeKFMono.push_back(pKFi);
					vpMapPointEdgeMono.push_back(pMP);

					nEdges++;
				}
				else if (cam0Index != -1
					&& pKFi->mvuRight[cam0Index] >= 0) // Stereo observation (with rectified images)
				{
					const cv::KeyPoint &kpUn = pKFi->mvKeysUn[cam0Index];
					Eigen::Matrix<double, 3, 1> obs;
					const float kp_ur = pKFi->mvuRight[cam0Index];
					obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

					g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
					e->setInformation(Info);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberStereo);

					e->fx = pKFi->fx;
					e->fy = pKFi->fy;
					e->cx = pKFi->cx;
					e->cy = pKFi->cy;
					e->bf = pKFi->mbf;

					optimizer.addEdge(e);
					vpEdgesStereo.push_back(e);
					vpEdgeKFStereo.push_back(pKFi);
					vpMapPointEdgeStereo.push_back(pMP);

					nEdges++;
				}

				// Monocular observation of Camera 0
				if (pKFi->mpCamera2) {
					int rightIndex = get<1>(mit->second);

					if (rightIndex != -1) {
						rightIndex -= pKFi->NLeft;

						Eigen::Matrix<double, 2, 1> obs;
						cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
						obs << kp.pt.x, kp.pt.y;

						ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
						e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
						e->setMeasurement(obs);
						const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(thHuberMono);

						e->mTrl = Converter::toSE3Quat(pKFi->mTrl);

						e->pCamera = pKFi->mpCamera2;

						optimizer.addEdge(e);
						vpEdgesBody.push_back(e);
						vpEdgeKFBody.push_back(pKFi);
						vpMapPointEdgeBody.push_back(pMP);

						nEdges++;
					}
				}
			}
		}
	}

	if (pbStopFlag) {
		if (*pbStopFlag) {
			return;
		}
	}

	optimizer.initializeOptimization();

	//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	int numPerform_it = optimizer.optimize(5);
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	//std::cout << "LBA time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	//std::cout << "Keyframes: " << nKFs << " --- MapPoints: " << nPoints << " --- Edges: " << nEdges << endl;

	bool bDoMore = true;

	if (pbStopFlag) {
		if (*pbStopFlag) {
			bDoMore = false;
		}
	}

	if (bDoMore) {

		// Check inlier observations
		int nMonoBadObs = 0;
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
			MapPoint *pMP = vpMapPointEdgeMono[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 5.991 || !e->isDepthPositive()) {
				//e->setLevel(1);
				nMonoBadObs++;
			}

			//e->setRobustKernel(0);
		}

		int nBodyBadObs = 0;
		for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
			MapPoint *pMP = vpMapPointEdgeBody[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 5.991 || !e->isDepthPositive()) {
				//e->setLevel(1);
				nBodyBadObs++;
			}

			//e->setRobustKernel(0);
		}

		int nStereoBadObs = 0;
		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
			MapPoint *pMP = vpMapPointEdgeStereo[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 7.815 || !e->isDepthPositive()) {
				//e->setLevel(1);
				nStereoBadObs++;
			}

			//e->setRobustKernel(0);
		}
		Verbose::PrintMess(
			"LM-LBA: First optimization has " + to_string(nMonoBadObs) + " monocular and " + to_string(nStereoBadObs)
				+ " stereo bad observations", Verbose::VERBOSITY_DEBUG);

		// Optimize again without the outliers
		//Verbose::PrintMess("LM-LBA: second optimization", Verbose::VERBOSITY_DEBUG);
		//optimizer.initializeOptimization(0);
		//numPerform_it = optimizer.optimize(10);
		numPerform_it += optimizer.optimize(5);
	}

	vector<pair<KeyFrame *, MapPoint *>> vToErase;
	vToErase.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());

	// Check inlier observations
	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
		ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
		MapPoint *pMP = vpMapPointEdgeMono[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 5.991 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFMono[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++) {
		ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
		MapPoint *pMP = vpMapPointEdgeBody[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 5.991 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFBody[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
		g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
		MapPoint *pMP = vpMapPointEdgeStereo[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 7.815 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFStereo[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	Verbose::PrintMess("LM-LBA: outlier observations: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess(
		"LM-LBA: total of observations: " + to_string(vpMapPointEdgeMono.size() + vpMapPointEdgeStereo.size()),
		Verbose::VERBOSITY_DEBUG);
	bool bRedrawError = false;
	bool bWriteStats = false;

	// Get Map Mutex
	unique_lock<shared_timed_mutex> lock(pCurrentMap->mMutexMapUpdate);

	if (!vToErase.empty()) {

		//cout << "LM-LBA: There are " << vToErase.size() << " observations whose will be deleted from the map" << endl;
		for (size_t i = 0; i < vToErase.size(); i++) {
			KeyFrame *pKFi = vToErase[i].first;
			MapPoint *pMPi = vToErase[i].second;
			pKFi->EraseMapPointMatch(pMPi);
			pMPi->EraseObservation(pKFi);
		}
	}

	// Recover optimized data
	//Keyframes
	vpNonEnoughOptKFs.clear();
	for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		cv::Mat Tiw = Converter::toCvMat(SE3quat);
		cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
		cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
		double dist = cv::norm(trasl);
		pKFi->SetPose(Converter::toCvMat(SE3quat));

		pKFi->mnNumberOfOpt += numPerform_it;
		//cout << "LM-LBA: KF " << pKFi->mnId << " had performed " <<  pKFi->mnNumberOfOpt << " iterations" << endl;
		if (pKFi->mnNumberOfOpt < 10) {
			vpNonEnoughOptKFs.push_back(pKFi);
		}
	}

	//Points
	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		g2o::VertexSBAPointXYZ
			*vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
	}

	pCurrentMap->IncreaseChangeIndex();
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int &num_fixedKF)
{
	//cout << "LBA" << endl;
	// Local KeyFrames: First Breath Search from Current Keyframe
	list<KeyFrame *> lLocalKeyFrames;

	lLocalKeyFrames.push_back(pKF);
	pKF->mnBALocalForKF = pKF->mnId;
	Map *pCurrentMap = pKF->GetMap();

	const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
	for (int i = 0, iend = vNeighKFs.size(); i < iend; i++) {
		KeyFrame *pKFi = vNeighKFs[i];
		pKFi->mnBALocalForKF = pKF->mnId;
		if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
			lLocalKeyFrames.push_back(pKFi);
		}
	}

	// Local MapPoints seen in Local KeyFrames
	num_fixedKF = 0;
	list<MapPoint *> lLocalMapPoints;
	set<MapPoint *> sNumObsMP;
	for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		if (pKFi->mnId == pMap->GetInitKFid()) {
			num_fixedKF = 1;
		}
		vector<MapPoint *> vpMPs = pKFi->GetMapPointMatches();
		for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
			MapPoint *pMP = *vit;
			if (pMP) {
				if (!pMP->isBad() && pMP->GetMap() == pCurrentMap) {

					if (pMP->mnBALocalForKF != pKF->mnId) {
						lLocalMapPoints.push_back(pMP);
						pMP->mnBALocalForKF = pKF->mnId;
					}
				}
			}
		}
	}

	// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
	list<KeyFrame *> lFixedCameras;
	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		map<KeyFrame *, tuple<int, int>> observations = (*lit)->GetObservations();
		for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (pKFi && pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
				pKFi->mnBAFixedForKF = pKF->mnId;
				if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
					lFixedCameras.push_back(pKFi);
				}
			}
		}
	}
	num_fixedKF = lFixedCameras.size() + num_fixedKF;
	if (num_fixedKF < 2) {
		//Verbose::PrintMess("LM-LBA: New Fixed KFs had been set", Verbose::VERBOSITY_NORMAL);
		//TODO We set 2 KFs to fixed to avoid a degree of freedom in scale
		list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin();
		int lowerId = pKF->mnId;
		KeyFrame *pLowerKf;
		int secondLowerId = pKF->mnId;
		KeyFrame *pSecondLowerKF;

		for (; lit != lLocalKeyFrames.end(); lit++) {
			KeyFrame *pKFi = *lit;
			if(!pKFi){
				continue;
			}
			if (pKFi == pKF || pKFi->mnId == pMap->GetInitKFid()) {
				continue;
			}

			if (pKFi->mnId < lowerId) {
				lowerId = pKFi->mnId;
				pLowerKf = pKFi;
			}
			else if (pKFi->mnId < secondLowerId) {
				secondLowerId = pKFi->mnId;
				pSecondLowerKF = pKFi;
			}
		}
		lFixedCameras.push_back(pLowerKf);
		lLocalKeyFrames.remove(pLowerKf);
		num_fixedKF++;
		if (num_fixedKF < 2) {
			lFixedCameras.push_back(pSecondLowerKF);
			lLocalKeyFrames.remove(pSecondLowerKF);
			num_fixedKF++;
		}
	}

	if (num_fixedKF == 0) {
		Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
		//return;
	}
	//Verbose::PrintMess("LM-LBA: There are " + to_string(lLocalKeyFrames.size()) + " KFs and " + to_string(lLocalMapPoints.size()) + " MPs to optimize. " + to_string(num_fixedKF) + " KFs are fixed", Verbose::VERBOSITY_DEBUG);

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	if (pMap->IsInertial()) {
		solver->setUserLambdaInit(100.0);
	} // TODO uncomment
	//cout << "LM-LBA: lambda init: " << solver->userLambdaInit() << endl;

	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);

	if (pbStopFlag) {
		optimizer.setForceStopFlag(pbStopFlag);
	}

	unsigned long maxKFid = 0;

	// Set Local KeyFrame vertices
	for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
		vSE3->setId(pKFi->mnId);
		vSE3->setFixed(pKFi->mnId == pMap->GetInitKFid());
		optimizer.addVertex(vSE3);
		if (pKFi->mnId > maxKFid) {
			maxKFid = pKFi->mnId;
		}
	}
	//Verbose::PrintMess("LM-LBA: KFs to optimize added", Verbose::VERBOSITY_DEBUG);

	// Set Fixed KeyFrame vertices
	for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		if(!pKFi){
			continue;
		}
		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
		vSE3->setId(pKFi->mnId);
		vSE3->setFixed(true);
		optimizer.addVertex(vSE3);
		if (pKFi->mnId > maxKFid) {
			maxKFid = pKFi->mnId;
		}
	}

	// Set MapPoint vertices
	const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

	vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody *> vpEdgesBody;
	vpEdgesBody.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFBody;
	vpEdgeKFBody.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeBody;
	vpMapPointEdgeBody.reserve(nExpectedSize);

	vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
	vpEdgesStereo.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFStereo;
	vpEdgeKFStereo.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeStereo;
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	int nPoints = 0;

	int nKFs = lLocalKeyFrames.size() + lFixedCameras.size(), nEdges = 0;

	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnId + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);
		nPoints++;

		const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

		//Set edges
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
				const int leftIndex = get<0>(mit->second);

				// Monocular observation
				if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] < 0) {
					const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					e->pCamera = pKFi->mpCamera;

					optimizer.addEdge(e);
					vpEdgesMono.push_back(e);
					vpEdgeKFMono.push_back(pKFi);
					vpMapPointEdgeMono.push_back(pMP);

					nEdges++;
				}
				else if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] >= 0) // Stereo observation
				{
					const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
					Eigen::Matrix<double, 3, 1> obs;
					const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
					obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

					g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
					e->setInformation(Info);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberStereo);

					e->fx = pKFi->fx;
					e->fy = pKFi->fy;
					e->cx = pKFi->cx;
					e->cy = pKFi->cy;
					e->bf = pKFi->mbf;

					optimizer.addEdge(e);
					vpEdgesStereo.push_back(e);
					vpEdgeKFStereo.push_back(pKFi);
					vpMapPointEdgeStereo.push_back(pMP);

					nEdges++;
				}

				if (pKFi->mpCamera2) {
					int rightIndex = get<1>(mit->second);

					if (rightIndex != -1) {
						rightIndex -= pKFi->NLeft;

						Eigen::Matrix<double, 2, 1> obs;
						cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
						obs << kp.pt.x, kp.pt.y;

						ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
						e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
						e->setMeasurement(obs);
						const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(thHuberMono);

						e->mTrl = Converter::toSE3Quat(pKFi->mTrl);

						e->pCamera = pKFi->mpCamera2;

						optimizer.addEdge(e);
						vpEdgesBody.push_back(e);
						vpEdgeKFBody.push_back(pKFi);
						vpMapPointEdgeBody.push_back(pMP);

						nEdges++;
					}
				}
			}
		}
	}

	//Verbose::PrintMess("LM-LBA: total observations: " + to_string(vpMapPointEdgeMono.size()+vpMapPointEdgeStereo.size()), Verbose::VERBOSITY_DEBUG);

	if (pbStopFlag) {
		if (*pbStopFlag) {
			return;
		}
	}

	optimizer.initializeOptimization();

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	optimizer.optimize(5);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	//std::cout << "LBA time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	//std::cout << "Keyframes: " << nKFs << " --- MapPoints: " << nPoints << " --- Edges: " << nEdges << endl;

	bool bDoMore = true;

	if (pbStopFlag) {
		if (*pbStopFlag) {
			bDoMore = false;
		}
	}

	if (bDoMore) {

		// Check inlier observations
		int nMonoBadObs = 0;
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
			MapPoint *pMP = vpMapPointEdgeMono[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 5.991 || !e->isDepthPositive()) {
				// e->setLevel(1); // MODIFICATION
				nMonoBadObs++;
			}

			//e->setRobustKernel(0);
		}

		int nBodyBadObs = 0;
		for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
			MapPoint *pMP = vpMapPointEdgeBody[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 5.991 || !e->isDepthPositive()) {
				//e->setLevel(1);
				nBodyBadObs++;
			}

			//e->setRobustKernel(0);
		}

		int nStereoBadObs = 0;
		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
			MapPoint *pMP = vpMapPointEdgeStereo[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 7.815 || !e->isDepthPositive()) {
				//TODO e->setLevel(1);
				nStereoBadObs++;
			}

			//TODO e->setRobustKernel(0);
		}
		//Verbose::PrintMess("LM-LBA: First optimization has " + to_string(nMonoBadObs) + " monocular and " + to_string(nStereoBadObs) + " stereo bad observations", Verbose::VERBOSITY_DEBUG);

		// Optimize again without the outliers
		//Verbose::PrintMess("LM-LBA: second optimization", Verbose::VERBOSITY_DEBUG);
		optimizer.initializeOptimization(0);
		optimizer.optimize(10);
	}

	vector<pair<KeyFrame *, MapPoint *>> vToErase;
	vToErase.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());

	// Check inlier observations
	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
		ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
		MapPoint *pMP = vpMapPointEdgeMono[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 5.991 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFMono[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++) {
		ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
		MapPoint *pMP = vpMapPointEdgeBody[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 5.991 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFBody[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
		g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
		MapPoint *pMP = vpMapPointEdgeStereo[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 7.815 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFStereo[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	//Verbose::PrintMess("LM-LBA: outlier observations: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);
	bool bRedrawError = false;
	if (vToErase.size() >= (vpMapPointEdgeMono.size() + vpMapPointEdgeStereo.size()) * 0.5) {
		Verbose::PrintMess("LM-LBA: ERROR IN THE OPTIMIZATION, MOST OF THE POINTS HAS BECOME OUTLIERS",
		                   Verbose::VERBOSITY_NORMAL);

		return;
		bRedrawError = true;
		string folder_name = "test_LBA";
		string name = "_PreLM_LBA";
		name = "_PreLM_LBA_Fixed";
	}

	// Get Map Mutex
	unique_lock<shared_timed_mutex> lock(pMap->mMutexMapUpdate);

	if (!vToErase.empty()) {
		map<KeyFrame *, int> mspInitialConnectedKFs;
		map<KeyFrame *, int> mspInitialObservationKFs;
		if (bRedrawError) {
			for (KeyFrame *pKFi: lLocalKeyFrames) {

				mspInitialConnectedKFs[pKFi] = pKFi->GetConnectedKeyFrames().size();
				mspInitialObservationKFs[pKFi] = pKFi->GetNumberMPs();
			}
		}

		//cout << "LM-LBA: There are " << vToErase.size() << " observations whose will be deleted from the map" << endl;
		for (size_t i = 0; i < vToErase.size(); i++) {
			KeyFrame *pKFi = vToErase[i].first;
			MapPoint *pMPi = vToErase[i].second;
			pKFi->EraseMapPointMatch(pMPi);
			pMPi->EraseObservation(pKFi);
		}

		map<KeyFrame *, int> mspFinalConnectedKFs;
		map<KeyFrame *, int> mspFinalObservationKFs;
		if (bRedrawError) {
			ofstream f_lba;
			f_lba.open("test_LBA/LBA_failure_KF" + to_string(pKF->mnId) + ".txt");
			f_lba << "# KF id, Initial Num CovKFs, Final Num CovKFs, Initial Num MPs, Fimal Num MPs" << endl;
			f_lba << fixed;

			for (KeyFrame *pKFi: lLocalKeyFrames) {
				pKFi->UpdateConnections();
				int finalNumberCovKFs = pKFi->GetConnectedKeyFrames().size();
				int finalNumberMPs = pKFi->GetNumberMPs();
				f_lba << pKFi->mnId << ", " << mspInitialConnectedKFs[pKFi] << ", " << finalNumberCovKFs << ", "
				      << mspInitialObservationKFs[pKFi] << ", " << finalNumberMPs << endl;

				mspFinalConnectedKFs[pKFi] = finalNumberCovKFs;
				mspFinalObservationKFs[pKFi] = finalNumberMPs;
			}

			f_lba.close();
		}
	}

	// Recover optimized data
	//Keyframes
	bool bShowStats = false;
	for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		cv::Mat Tiw = Converter::toCvMat(SE3quat);
		cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
		cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
		double dist = cv::norm(trasl);
		pKFi->SetPose(Converter::toCvMat(SE3quat));

		if (dist > 1.0) {
			bShowStats = true;
			Verbose::PrintMess("LM-LBA: Too much distance in KF " + to_string(pKFi->mnId) + ", " + to_string(dist)
				                   + " meters. Current KF " + to_string(pKF->mnId), Verbose::VERBOSITY_DEBUG);
			Verbose::PrintMess("LM-LBA: Number of connections between the KFs " + to_string(pKF->GetWeight((pKFi))),
			                   Verbose::VERBOSITY_DEBUG);

			int numMonoMP = 0, numBadMonoMP = 0;
			int numStereoMP = 0, numBadStereoMP = 0;
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
				if (vpEdgeKFMono[i] != pKFi) {
					continue;
				}
				ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
				MapPoint *pMP = vpMapPointEdgeMono[i];

				if (pMP->isBad()) {
					continue;
				}

				if (e->chi2() > 5.991 || !e->isDepthPositive()) {
					numBadMonoMP++;
				}
				else {
					numMonoMP++;
				}
			}

			for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
				if (vpEdgeKFStereo[i] != pKFi) {
					continue;
				}
				g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
				MapPoint *pMP = vpMapPointEdgeStereo[i];

				if (pMP->isBad()) {
					continue;
				}

				if (e->chi2() > 7.815 || !e->isDepthPositive()) {
					numBadStereoMP++;
				}
				else {
					numStereoMP++;
				}
			}
			Verbose::PrintMess(
				"LM-LBA: Good observations in mono " + to_string(numMonoMP) + " and stereo " + to_string(numStereoMP),
				Verbose::VERBOSITY_DEBUG);
			Verbose::PrintMess("LM-LBA: Bad observations in mono " + to_string(numBadMonoMP) + " and stereo "
				                   + to_string(numBadStereoMP), Verbose::VERBOSITY_DEBUG);
		}
	}

	//Points
	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		g2o::VertexSBAPointXYZ
			*vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
	}

	if (bRedrawError) {
		string folder_name = "test_LBA";
		string name = "_PostLM_LBA";
		//pMap->printReprojectionError(lLocalKeyFrames, pKF, name, folder_name);
		name = "_PostLM_LBA_Fixed";
		//pMap->printReprojectionError(lFixedCameras, pKF, name, folder_name);
	}

	// TODO Check this changeindex
	pMap->IncreaseChangeIndex();
}

void Optimizer::LocalDVLBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int &num_fixedKF)
{
	Map *pCurrentMap = pKF->GetMap();
	int Nd = std::min(10, (int)pCurrentMap->KeyFramesInMap() - 2);// number of keyframes in current map
	const unsigned long maxKFid = pKF->mnId;

	vector<KeyFrame *> OptDVLKFs;
	const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
	list<KeyFrame *> OptVisualKFS;

	OptDVLKFs.reserve(Nd);
	OptDVLKFs.push_back(pKF);
	pKF->mnBALocalForKF = pKF->mnId;

	for (int i = 1; i < Nd; i++) {
		if (OptDVLKFs.back()->mPrevKF) {
			OptDVLKFs.push_back(OptDVLKFs.back()->mPrevKF);
			OptDVLKFs.back()->mnBALocalForKF = pKF->mnId;
		}
		else {
			break;
		}
	}
	int N = OptDVLKFs.size();

	//cout << "LBA" << endl;
	// Local KeyFrames: First Breath Search from Current Keyframe
//		list<KeyFrame *> lLocalKeyFrames;
//
//		lLocalKeyFrames.push_back(pKF);
//		pKF->mnBALocalForKF = pKF->mnId;
//
//
//
//		for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
//		{
//			KeyFrame *pKFi = vNeighKFs[i];
//			pKFi->mnBALocalForKF = pKF->mnId;
//			if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
//				lLocalKeyFrames.push_back(pKFi);
//		}

	// Local MapPoints seen in Local KeyFrames
	list<MapPoint *> lLocalMapPoints;
	for (int i = 0; i < N; i++) {
		vector<MapPoint *> vpMPs = OptDVLKFs[i]->GetMapPointMatches();
		for (vector<MapPoint *>::iterator it = vpMPs.begin(); it != vpMPs.end(); it++) {
			MapPoint *pMP = *it;
			if (pMP) {
				if (!pMP->isBad()) {
					if (pMP->mnBALocalForKF != pKF->mnId) {
						lLocalMapPoints.push_back(pMP);
						pMP->mnBALocalForKF = pKF->mnId;
					}
				}
			}
		}
	}

	// Fixed Keyframe
	list<KeyFrame *> lFixedKFs;
	if (OptDVLKFs.back()->mPrevKF) {
		lFixedKFs.push_back(OptDVLKFs.back()->mPrevKF);
		OptDVLKFs.back()->mPrevKF->mnBAFixedForKF = pKF->mnId;
	}
	else {
		OptDVLKFs.back()->mnBALocalForKF = 0;
		OptDVLKFs.back()->mnBAFixedForKF = pKF->mnId;
		lFixedKFs.push_back(OptDVLKFs.back());
		OptDVLKFs.pop_back();
	}

	const int maxFixedKF = 200;
	for (list<MapPoint *>::iterator it = lLocalMapPoints.begin(); it != lLocalMapPoints.end(); it++) {
		map<KeyFrame *, tuple<int, int>> observations = (*it)->GetObservations();
		for (map<KeyFrame *, tuple<int, int>>::iterator it_ob = observations.begin(); it_ob != observations.end();
		     it_ob++) {
			KeyFrame *pKFi = it_ob->first;
			if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
				pKFi->mnBAFixedForKF = pKF->mnId;
				if (!pKFi->isBad()) {
					lFixedKFs.push_back(pKFi);
					break;
				}
			}
		}
		if (lFixedKFs.size() >= maxFixedKF) {
			break;
		}
	}



	//Verbose::PrintMess("LM-LBA: There are " + to_string(lLocalKeyFrames.size()) + " KFs and " + to_string(lLocalMapPoints.size()) + " MPs to optimize. " + to_string(num_fixedKF) + " KFs are fixed", Verbose::VERBOSITY_DEBUG);

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	if (pMap->IsInertial()) {
		solver->setUserLambdaInit(100.0);
	} // TODO uncomment
	//cout << "LM-LBA: lambda init: " << solver->userLambdaInit() << endl;

	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);

	if (pbStopFlag) {
		optimizer.setForceStopFlag(pbStopFlag);
	}

//		unsigned long maxKFid = 0;

	// Set Local KeyFrame vertices
	for (int i = 0; i < N; i++) {
		KeyFrame *pKFi = OptDVLKFs[i];
		pKFi->IntegrateDVL(pKFi->mPrevKF);

		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
//			cout<<"id:"<<pKFi->mnId<<"initial pose: \n"<<pKFi->GetPose()<<endl;
		vSE3->setId(pKFi->mnId);
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);
	}
	//Verbose::PrintMess("LM-LBA: KFs to optimize added", Verbose::VERBOSITY_DEBUG);

	// Set Fixed KeyFrame vertices
	for (list<KeyFrame *>::iterator it = lFixedKFs.begin(), lend = lFixedKFs.end(); it != lend; it++) {
		KeyFrame *pKFi = *it;
		if (pKFi->mPrevKF) {
			pKFi->IntegrateDVL(pKFi->mPrevKF);
		}
		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
		vSE3->setId(pKFi->mnId);
		vSE3->setFixed(true);
		optimizer.addVertex(vSE3);
	}

	// Set MapPoint vertices
	const int nExpectedSize = (OptDVLKFs.size() + lFixedKFs.size()) * lLocalMapPoints.size();

	vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody *> vpEdgesBody;
	vpEdgesBody.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFBody;
	vpEdgeKFBody.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeBody;
	vpMapPointEdgeBody.reserve(nExpectedSize);

	vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
	vpEdgesStereo.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFStereo;
	vpEdgeKFStereo.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeStereo;
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	int nPoints = 0;

	int nKFs = OptDVLKFs.size() + lFixedKFs.size(), nEdges = 0;

	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnId + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);
		nPoints++;

		const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

		//Set edges
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
				continue;
			}

			if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
				const int leftIndex = get<0>(mit->second);

				// Monocular observation
				if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] < 0) {
					const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					e->pCamera = pKFi->mpCamera;

					optimizer.addEdge(e);
					vpEdgesMono.push_back(e);
					vpEdgeKFMono.push_back(pKFi);
					vpMapPointEdgeMono.push_back(pMP);

					nEdges++;
				}
				else if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] >= 0) // Stereo observation
				{
					const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
					Eigen::Matrix<double, 3, 1> obs;
					const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
					obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

					g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
					e->setInformation(Info);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberStereo);

					e->fx = pKFi->fx;
					e->fy = pKFi->fy;
					e->cx = pKFi->cx;
					e->cy = pKFi->cy;
					e->bf = pKFi->mbf;

					optimizer.addEdge(e);
					vpEdgesStereo.push_back(e);
					vpEdgeKFStereo.push_back(pKFi);
					vpMapPointEdgeStereo.push_back(pMP);

					nEdges++;
				}

				if (pKFi->mpCamera2) {
					int rightIndex = get<1>(mit->second);

					if (rightIndex != -1) {
						rightIndex -= pKFi->NLeft;

						Eigen::Matrix<double, 2, 1> obs;
						cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
						obs << kp.pt.x, kp.pt.y;

						ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
						e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
						e->setMeasurement(obs);
						const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(thHuberMono);

						e->mTrl = Converter::toSE3Quat(pKFi->mTrl);

						e->pCamera = pKFi->mpCamera2;

						optimizer.addEdge(e);
						vpEdgesBody.push_back(e);
						vpEdgeKFBody.push_back(pKFi);
						vpMapPointEdgeBody.push_back(pMP);

						nEdges++;
					}
				}
			}
		}
	}

	for (vector<KeyFrame *>::iterator it = OptDVLKFs.begin(); it != OptDVLKFs.end(); it++) {
		if ((*it)->mPrevKF) {
			KeyFrame *p_cur = *it;
			KeyFrame *p_pre = (*it)->mPrevKF;
//				p_cur->IntegrateDVL(p_pre);
			EdgeSE3DVLBA *e = new EdgeSE3DVLBA(p_cur->mT_ei_ej, p_cur->mT_e_c);
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(p_pre->mnId)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(p_cur->mnId)));
			e->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 50000000);
			optimizer.addEdge(e);
//				cout<<"add edge"<<endl;
			nEdges++;
		}
		else {
			cout << "no previous keyframe for DVL" << endl;
		}

	}
	//Verbose::PrintMess("LM-LBA: total observations: " + to_string(vpMapPointEdgeMono.size()+vpMapPointEdgeStereo.size()), Verbose::VERBOSITY_DEBUG);

	if (pbStopFlag) {
		if (*pbStopFlag) {
			return;
		}
	}

	optimizer.initializeOptimization();

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	optimizer.optimize(5);
//		cout<<"first optimization"<<endl;
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	//std::cout << "LBA time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	//std::cout << "Keyframes: " << nKFs << " --- MapPoints: " << nPoints << " --- Edges: " << nEdges << endl;

	bool bDoMore = true;

	if (pbStopFlag) {
		if (*pbStopFlag) {
			bDoMore = false;
		}
	}

	if (bDoMore) {

		// Check inlier observations
		int nMonoBadObs = 0;
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
			MapPoint *pMP = vpMapPointEdgeMono[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 5.991 || !e->isDepthPositive()) {
				// e->setLevel(1); // MODIFICATION
				nMonoBadObs++;
			}

			//e->setRobustKernel(0);
		}

		int nBodyBadObs = 0;
		for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
			MapPoint *pMP = vpMapPointEdgeBody[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 5.991 || !e->isDepthPositive()) {
				//e->setLevel(1);
				nBodyBadObs++;
			}

			//e->setRobustKernel(0);
		}

		int nStereoBadObs = 0;
		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
			MapPoint *pMP = vpMapPointEdgeStereo[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 7.815 || !e->isDepthPositive()) {
				//TODO e->setLevel(1);
				nStereoBadObs++;
			}

			//TODO e->setRobustKernel(0);
		}
		//Verbose::PrintMess("LM-LBA: First optimization has " + to_string(nMonoBadObs) + " monocular and " + to_string(nStereoBadObs) + " stereo bad observations", Verbose::VERBOSITY_DEBUG);

		// Optimize again without the outliers
		//Verbose::PrintMess("LM-LBA: second optimization", Verbose::VERBOSITY_DEBUG);
		optimizer.initializeOptimization(0);
		optimizer.optimize(10);
//			cout<<"second optimization"<<endl;
	}

	vector<pair<KeyFrame *, MapPoint *>> vToErase;
	vToErase.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());

	// Check inlier observations
	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
		ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
		MapPoint *pMP = vpMapPointEdgeMono[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 5.991 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFMono[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++) {
		ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
		MapPoint *pMP = vpMapPointEdgeBody[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 5.991 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFBody[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
		g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
		MapPoint *pMP = vpMapPointEdgeStereo[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 7.815 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFStereo[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	//Verbose::PrintMess("LM-LBA: outlier observations: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);
	bool bRedrawError = false;
	if (vToErase.size() >= (vpMapPointEdgeMono.size() + vpMapPointEdgeStereo.size()) * 0.5) {
		Verbose::PrintMess("LM-LBA: ERROR IN THE OPTIMIZATION, MOST OF THE POINTS HAS BECOME OUTLIERS",
		                   Verbose::VERBOSITY_NORMAL);

		return;
		bRedrawError = true;
		string folder_name = "test_LBA";
		string name = "_PreLM_LBA";
		name = "_PreLM_LBA_Fixed";
	}

	// Get Map Mutex
	unique_lock<shared_timed_mutex> lock(pMap->mMutexMapUpdate);

	if (!vToErase.empty()) {
		map<KeyFrame *, int> mspInitialConnectedKFs;
		map<KeyFrame *, int> mspInitialObservationKFs;
		if (bRedrawError) {
			for (KeyFrame *pKFi: OptDVLKFs) {

				mspInitialConnectedKFs[pKFi] = pKFi->GetConnectedKeyFrames().size();
				mspInitialObservationKFs[pKFi] = pKFi->GetNumberMPs();
			}
		}

		//cout << "LM-LBA: There are " << vToErase.size() << " observations whose will be deleted from the map" << endl;
		for (size_t i = 0; i < vToErase.size(); i++) {
			KeyFrame *pKFi = vToErase[i].first;
			MapPoint *pMPi = vToErase[i].second;
			pKFi->EraseMapPointMatch(pMPi);
			pMPi->EraseObservation(pKFi);
		}

		map<KeyFrame *, int> mspFinalConnectedKFs;
		map<KeyFrame *, int> mspFinalObservationKFs;
		if (bRedrawError) {
			ofstream f_lba;
			f_lba.open("test_LBA/LBA_failure_KF" + to_string(pKF->mnId) + ".txt");
			f_lba << "# KF id, Initial Num CovKFs, Final Num CovKFs, Initial Num MPs, Fimal Num MPs" << endl;
			f_lba << fixed;

			for (KeyFrame *pKFi: OptDVLKFs) {
				pKFi->UpdateConnections();
				int finalNumberCovKFs = pKFi->GetConnectedKeyFrames().size();
				int finalNumberMPs = pKFi->GetNumberMPs();
				f_lba << pKFi->mnId << ", " << mspInitialConnectedKFs[pKFi] << ", " << finalNumberCovKFs << ", "
				      << mspInitialObservationKFs[pKFi] << ", " << finalNumberMPs << endl;

				mspFinalConnectedKFs[pKFi] = finalNumberCovKFs;
				mspFinalObservationKFs[pKFi] = finalNumberMPs;
			}

			f_lba.close();
		}
	}

	// Recover optimized data
	//Keyframes
	bool bShowStats = false;
	for (vector<KeyFrame *>::iterator lit = OptDVLKFs.begin(), lend = OptDVLKFs.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		cv::Mat Tiw = Converter::toCvMat(SE3quat);
		cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
		cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
		double dist = cv::norm(trasl);
		pKFi->SetPose(Converter::toCvMat(SE3quat));

		if (dist > 1.0) {
			bShowStats = true;
			Verbose::PrintMess("LM-LBA: Too much distance in KF " + to_string(pKFi->mnId) + ", " + to_string(dist)
				                   + " meters. Current KF " + to_string(pKF->mnId), Verbose::VERBOSITY_DEBUG);
			Verbose::PrintMess("LM-LBA: Number of connections between the KFs " + to_string(pKF->GetWeight((pKFi))),
			                   Verbose::VERBOSITY_DEBUG);

			int numMonoMP = 0, numBadMonoMP = 0;
			int numStereoMP = 0, numBadStereoMP = 0;
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
				if (vpEdgeKFMono[i] != pKFi) {
					continue;
				}
				ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
				MapPoint *pMP = vpMapPointEdgeMono[i];

				if (pMP->isBad()) {
					continue;
				}

				if (e->chi2() > 5.991 || !e->isDepthPositive()) {
					numBadMonoMP++;
				}
				else {
					numMonoMP++;
				}
			}

			for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
				if (vpEdgeKFStereo[i] != pKFi) {
					continue;
				}
				g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
				MapPoint *pMP = vpMapPointEdgeStereo[i];

				if (pMP->isBad()) {
					continue;
				}

				if (e->chi2() > 7.815 || !e->isDepthPositive()) {
					numBadStereoMP++;
				}
				else {
					numStereoMP++;
				}
			}
			Verbose::PrintMess(
				"LM-LBA: Good observations in mono " + to_string(numMonoMP) + " and stereo " + to_string(numStereoMP),
				Verbose::VERBOSITY_DEBUG);
			Verbose::PrintMess("LM-LBA: Bad observations in mono " + to_string(numBadMonoMP) + " and stereo "
				                   + to_string(numBadStereoMP), Verbose::VERBOSITY_DEBUG);
		}
	}

	//Points
	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		g2o::VertexSBAPointXYZ
			*vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
	}

	if (bRedrawError) {
		string folder_name = "test_LBA";
		string name = "_PostLM_LBA";
		//pMap->printReprojectionError(lLocalKeyFrames, pKF, name, folder_name);
		name = "_PostLM_LBA_Fixed";
		//pMap->printReprojectionError(lFixedCameras, pKF, name, folder_name);
	}

	// TODO Check this changeindex
	pMap->IncreaseChangeIndex();
}

void Optimizer::LocalDVLRefinement(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int &num_fixedKF)
{
	Map *pCurrentMap = pKF->GetMap();
//	int Nd=std::min(20,(int)pCurrentMap->KeyFramesInMap()-2) ;// number of keyframes in current map
	int Nd = (int)pCurrentMap->KeyFramesInMap() - 2;// number of keyframes in current map
	const unsigned long maxKFid = pKF->mnId;

	vector<KeyFrame *> OptDVLKFs;
	const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
	list<KeyFrame *> OptVisualKFS;

	OptDVLKFs.reserve(Nd);
	OptDVLKFs.push_back(pKF);
	pKF->mnBALocalForKF = pKF->mnId;

	for (int i = 1; i < Nd; i++) {
		if (OptDVLKFs.back()->mPrevKF) {
			OptDVLKFs.push_back(OptDVLKFs.back()->mPrevKF);
			OptDVLKFs.back()->mnBALocalForKF = pKF->mnId;
		}
		else {
			break;
		}
	}
	int N = OptDVLKFs.size();

	//cout << "LBA" << endl;
	// Local KeyFrames: First Breath Search from Current Keyframe
//		list<KeyFrame *> lLocalKeyFrames;
//
//		lLocalKeyFrames.push_back(pKF);
//		pKF->mnBALocalForKF = pKF->mnId;
//
//
//
//		for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
//		{
//			KeyFrame *pKFi = vNeighKFs[i];
//			pKFi->mnBALocalForKF = pKF->mnId;
//			if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
//				lLocalKeyFrames.push_back(pKFi);
//		}

	// Local MapPoints seen in Local KeyFrames
	list<MapPoint *> lLocalMapPoints;
	for (int i = 0; i < N; i++) {
		vector<MapPoint *> vpMPs = OptDVLKFs[i]->GetMapPointMatches();
		for (vector<MapPoint *>::iterator it = vpMPs.begin(); it != vpMPs.end(); it++) {
			MapPoint *pMP = *it;
			if (pMP) {
				if (!pMP->isBad()) {
					if (pMP->mnBALocalForKF != pKF->mnId) {
						lLocalMapPoints.push_back(pMP);
						pMP->mnBALocalForKF = pKF->mnId;
					}
				}
			}
		}
	}

	// Fixed Keyframe
	list<KeyFrame *> lFixedKFs;
	if (OptDVLKFs.back()->mPrevKF) {
		lFixedKFs.push_back(OptDVLKFs.back()->mPrevKF);
		OptDVLKFs.back()->mPrevKF->mnBAFixedForKF = pKF->mnId;
	}
	else {
		OptDVLKFs.back()->mnBALocalForKF = 0;
		OptDVLKFs.back()->mnBAFixedForKF = pKF->mnId;
		lFixedKFs.push_back(OptDVLKFs.back());
		OptDVLKFs.pop_back();
	}

	const int maxFixedKF = 2000;
	for (list<MapPoint *>::iterator it = lLocalMapPoints.begin(); it != lLocalMapPoints.end(); it++) {
		map<KeyFrame *, tuple<int, int>> observations = (*it)->GetObservations();
		for (map<KeyFrame *, tuple<int, int>>::iterator it_ob = observations.begin(); it_ob != observations.end();
		     it_ob++) {
			KeyFrame *pKFi = it_ob->first;
			if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
				pKFi->mnBAFixedForKF = pKF->mnId;
				if (!pKFi->isBad()) {
					lFixedKFs.push_back(pKFi);
					break;
				}
			}
		}
		if (lFixedKFs.size() >= maxFixedKF) {
			break;
		}
	}



	//Verbose::PrintMess("LM-LBA: There are " + to_string(lLocalKeyFrames.size()) + " KFs and " + to_string(lLocalMapPoints.size()) + " MPs to optimize. " + to_string(num_fixedKF) + " KFs are fixed", Verbose::VERBOSITY_DEBUG);

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	if (pMap->IsInertial()) {
		solver->setUserLambdaInit(100.0);
	} // TODO uncomment
	//cout << "LM-LBA: lambda init: " << solver->userLambdaInit() << endl;

	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);

	if (pbStopFlag) {
		optimizer.setForceStopFlag(pbStopFlag);
	}


	Eigen::Isometry3d initial_T_e_c = pKF->mT_e_c;
	fstream file;
	file.open("data/Edge_T.txt", ios::out | ios::app);
	if (!file) {
		cout << "fail to open data/Edge_T.txt" << endl;
	}
//	file<<"initial T_e_c: "<<endl;
//
//	file<<"t: "<<initial_T_e_c.translation()<<endl;
//	file<<"r: "<<initial_T_e_c.rotation().eulerAngles(0,1,2)<<endl;

//	cv::Mat initial_T_e_c_mat;
//	cv::eigen2cv(initial_T_e_c.matrix(),initial_T_e_c_mat);
	g2o::VertexSE3Expmap *vSE3_e_c = new g2o::VertexSE3Expmap();
	vSE3_e_c->setEstimate(g2o::SE3Quat(initial_T_e_c.rotation(), initial_T_e_c.translation()));
	vSE3_e_c->setId(0);
	vSE3_e_c->setFixed(false);
	optimizer.addVertex(vSE3_e_c);
//	cout<<initial_T_e_c_mat<<endl;

	for (vector<KeyFrame *>::iterator it = OptDVLKFs.begin(); it != OptDVLKFs.end(); it++) {
		if ((*it)->mPrevKF) {
			KeyFrame *p_cur = *it;
			KeyFrame *p_pre = (*it)->mPrevKF;
			p_cur->IntegrateDVL(p_pre);
			Eigen::Isometry3d T_ci_c0 = Eigen::Isometry3d::Identity();
			Eigen::Isometry3d T_cj_c0 = Eigen::Isometry3d::Identity();
			cv::cv2eigen(p_pre->GetPose(), T_ci_c0.matrix());
			cv::cv2eigen(p_cur->GetPose(), T_cj_c0.matrix());
			EdgeDVLRefine *e = new EdgeDVLRefine(p_cur->mT_ei_ej, T_ci_c0, T_cj_c0);
//			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(p_pre->mnId)));
//			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(p_cur->mnId)));
			e->setVertex(0, vSE3_e_c);
			e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
			optimizer.addEdge(e);
		}
		else {
			cout << "no previous keyframe for DVL" << endl;
		}

	}
	//Verbose::PrintMess("LM-LBA: total observations: " + to_string(vpMapPointEdgeMono.size()+vpMapPointEdgeStereo.size()), Verbose::VERBOSITY_DEBUG);

	if (pbStopFlag) {
		if (*pbStopFlag) {
			return;
		}
	}

	optimizer.initializeOptimization();

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	optimizer.optimize(50);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	Eigen::Isometry3d T_e_c(vSE3_e_c->estimate());
	cout << "DVL refinement finished!" << endl;
	file << pKF->mnId << " " << T_e_c.translation().transpose() << " "
	     << T_e_c.rotation().eulerAngles(0, 1, 2).transpose() << endl;
//	file<<"r matrix: "<<T_e_c.rotation()<<endl;

	file.close();

}

void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *>> &LoopConnections, const bool &bFixScale)
{
	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
		new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
	g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	solver->setUserLambdaInit(1e-16);
	optimizer.setAlgorithm(solver);

	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
	const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

	const unsigned int nMaxKFid = pMap->GetMaxKFid();

	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
	vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

	vector<Eigen::Vector3d> vZvectors(nMaxKFid + 1); // For debugging
	Eigen::Vector3d z_vec;
	z_vec << 0.0, 0.0, 1.0;

	const int minFeat = 100; // MODIFICATION originally was set to 100

	// Set KeyFrame vertices
	for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
		KeyFrame *pKF = vpKFs[i];
		if (pKF->isBad()) {
			continue;
		}
		g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

		const int nIDi = pKF->mnId;

		LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

		if (it != CorrectedSim3.end()) {
			vScw[nIDi] = it->second;
			VSim3->setEstimate(it->second);
		}
		else {
			Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
			Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
			g2o::Sim3 Siw(Rcw, tcw, 1.0);
			vScw[nIDi] = Siw;
			VSim3->setEstimate(Siw);
		}

		if (pKF->mnId == pMap->GetInitKFid()) {
			VSim3->setFixed(true);
		}

		VSim3->setId(nIDi);
		VSim3->setMarginalized(false);
		VSim3->_fix_scale = bFixScale;

		optimizer.addVertex(VSim3);
		vZvectors[nIDi] = vScw[nIDi].rotation().toRotationMatrix() * z_vec; // For debugging

		vpVertices[nIDi] = VSim3;
	}

	set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

	const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

	// Set Loop edges
	int count_loop = 0;
	for (map<KeyFrame *, set<KeyFrame *>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end();
	     mit != mend; mit++) {
		KeyFrame *pKF = mit->first;
		const long unsigned int nIDi = pKF->mnId;
		const set<KeyFrame *> &spConnections = mit->second;
		const g2o::Sim3 Siw = vScw[nIDi];
		const g2o::Sim3 Swi = Siw.inverse();

		for (set<KeyFrame *>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send;
		     sit++) {
			const long unsigned int nIDj = (*sit)->mnId;
			if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat) {
				continue;
			}

			const g2o::Sim3 Sjw = vScw[nIDj];
			const g2o::Sim3 Sji = Sjw * Swi;

			g2o::EdgeSim3 *e = new g2o::EdgeSim3();
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
			e->setMeasurement(Sji);

			e->information() = matLambda;

			optimizer.addEdge(e);
			count_loop++;
			sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
		}
	}

	int count_spa_tree = 0;
	int count_cov = 0;
	int count_imu = 0;
	int count_kf = 0;
	// Set normal edges
	for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
		count_kf = 0;
		KeyFrame *pKF = vpKFs[i];

		const int nIDi = pKF->mnId;

		g2o::Sim3 Swi;

		LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

		if (iti != NonCorrectedSim3.end()) {
			Swi = (iti->second).inverse();
		}
		else {
			Swi = vScw[nIDi].inverse();
		}

		KeyFrame *pParentKF = pKF->GetParent();

		// Spanning tree edge
		if (pParentKF) {
			int nIDj = pParentKF->mnId;

			g2o::Sim3 Sjw;

			LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

			if (itj != NonCorrectedSim3.end()) {
				Sjw = itj->second;
			}
			else {
				Sjw = vScw[nIDj];
			}

			g2o::Sim3 Sji = Sjw * Swi;

			g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            auto v1 = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj));
            auto v2 = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi));
            if(v1&&v2){
                e->setVertex(1, v1);
                e->setVertex(0, v2);
                e->setMeasurement(Sji);
                count_kf++;
                count_spa_tree++;
                e->information() = matLambda;
                optimizer.addEdge(e);
            }
            else
                continue;

		}

		// Loop edges
		const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
		for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++) {
			KeyFrame *pLKF = *sit;
			if (pLKF->mnId < pKF->mnId) {
				g2o::Sim3 Slw;

				LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

				if (itl != NonCorrectedSim3.end()) {
					Slw = itl->second;
				}
				else {
					Slw = vScw[pLKF->mnId];
				}

				g2o::Sim3 Sli = Slw * Swi;
				g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                auto v1 = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId));
                auto v2 = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi));
                if(v1&&v2){
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                    count_kf++;
                    count_loop++;
                }
			}
		}

		// Covisibility graph edges
		const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
		for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++) {
			KeyFrame *pKFn = *vit;
			if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn)) {
				if (!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
					if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId)))) {
						continue;
					}

					g2o::Sim3 Snw;

					LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

					if (itn != NonCorrectedSim3.end()) {
						Snw = itn->second;
					}
					else {
						Snw = vScw[pKFn->mnId];
					}

					g2o::Sim3 Sni = Snw * Swi;

					g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                    auto v1 = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId));
                    auto v2 = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi));
                    if(v1&&v2){
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                        count_kf++;
                        count_cov++;
                    }
				}
			}
		}

		// Inertial edges if inertial
		// if (pKF->bImu && pKF->mPrevKF) {
		// 	g2o::Sim3 Spw;
		// 	LoopClosing::KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
		// 	if (itp != NonCorrectedSim3.end()) {
		// 		Spw = itp->second;
		// 	}
		// 	else {
		// 		Spw = vScw[pKF->mPrevKF->mnId];
		// 	}
        //
		// 	g2o::Sim3 Spi = Spw * Swi;
		// 	g2o::EdgeSim3 *ep = new g2o::EdgeSim3();
		// 	ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mPrevKF->mnId)));
		// 	ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
		// 	ep->setMeasurement(Spi);
		// 	ep->information() = matLambda;
		// 	optimizer.addEdge(ep);
		// 	count_kf++;
		// 	count_imu++;
		// }
		/*if(count_kf<3)
            cout << "EG: kf with " << count_kf << " edges!!    ID: " << pKF->mnId << endl;*/
	}

	//cout << "EG: Number of KFs: " << vpKFs.size() << endl;
	//cout << "EG: spaning tree edges: " << count_spa_tree << endl;
	//cout << "EG: Loop edges: " << count_loop << endl;
	//cout << "EG: covisible edges: " << count_cov << endl;
	//cout << "EG: imu edges: " << count_imu << endl;
	// Optimize!
	optimizer.initializeOptimization();
	optimizer.computeActiveErrors();
	float err0 = optimizer.activeRobustChi2();
	optimizer.optimize(20);
	optimizer.computeActiveErrors();
	float errEnd = optimizer.activeRobustChi2();
	//cout << "err_0/err_end: " << err0 << "/" << errEnd << endl;
	unique_lock<shared_timed_mutex> lock(pMap->mMutexMapUpdate);

	// SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		const int nIDi = pKFi->mnId;

		g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
		g2o::Sim3 CorrectedSiw = VSim3->estimate();
		vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
		Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
		Eigen::Vector3d eigt = CorrectedSiw.translation();
		double s = CorrectedSiw.scale();

		eigt *= (1. / s); //[R t/s;0 1]

		cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

		pKFi->SetPose(Tiw);
		// cout << "angle KF " << nIDi << ": " << (180.0/3.1415)*acos(vZvectors[nIDi].dot(eigR*z_vec)) << endl;
	}

	// Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
	for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
		MapPoint *pMP = vpMPs[i];

		if (pMP->isBad()) {
			continue;
		}

		int nIDr;
		if (pMP->mnCorrectedByKF == pCurKF->mnId) {
			nIDr = pMP->mnCorrectedReference;
		}
		else {
			KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
			nIDr = pRefKF->mnId;
		}

		g2o::Sim3 Srw = vScw[nIDr];
		g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

		cv::Mat P3Dw = pMP->GetWorldPos();
		Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
		Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

		cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
		pMP->SetWorldPos(cvCorrectedP3Dw);

		pMP->UpdateNormalAndDepth();
	}

	// TODO Check this changeindex
	pMap->IncreaseChangeIndex();
}

void Optimizer::OptimizeEssentialGraph6DoF(KeyFrame *pCurKF,
                                           vector<KeyFrame *> &vpFixedKFs,
                                           vector<KeyFrame *> &vpFixedCorrectedKFs,
                                           vector<KeyFrame *> &vpNonFixedKFs,
                                           vector<MapPoint *> &vpNonCorrectedMPs,
                                           double scale)
{
	Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedKFs.size()) + " KFs fixed in the merged map",
	                   Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess(
		"Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map",
		Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess(
		"Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map",
		Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess(
		"Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map",
		Verbose::VERBOSITY_DEBUG);

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver =
		new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	solver->setUserLambdaInit(1e-16);
	optimizer.setAlgorithm(solver);

	Map *pMap = pCurKF->GetMap();
	const unsigned int nMaxKFid = pMap->GetMaxKFid();

	vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vScw(nMaxKFid + 1);
	vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vScw_bef(nMaxKFid + 1);
	vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vCorrectedSwc(nMaxKFid + 1);
	vector<g2o::VertexSE3Expmap *> vpVertices(nMaxKFid + 1);
	vector<bool> vbFromOtherMap(nMaxKFid + 1);

	const int minFeat = 100;

	for (KeyFrame *pKFi: vpFixedKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		g2o::VertexSE3Expmap *VSE3 = new g2o::VertexSE3Expmap();

		const int nIDi = pKFi->mnId;

		Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
		Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
		g2o::SE3Quat Siw(Rcw, tcw);
		vScw[nIDi] = Siw;
		vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
		VSE3->setEstimate(Siw);

		VSE3->setFixed(true);

		VSE3->setId(nIDi);
		VSE3->setMarginalized(false);
		//VSim3->_fix_scale = true; //TODO
		vbFromOtherMap[nIDi] = false;

		optimizer.addVertex(VSE3);

		vpVertices[nIDi] = VSE3;
	}
	cout << "Opt_Essential: vpFixedKFs loaded" << endl;

	set<unsigned long> sIdKF;
	for (KeyFrame *pKFi: vpFixedCorrectedKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		g2o::VertexSE3Expmap *VSE3 = new g2o::VertexSE3Expmap();

		const int nIDi = pKFi->mnId;

		Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
		Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
		g2o::SE3Quat Siw(Rcw, tcw);
		vScw[nIDi] = Siw;
		vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
		VSE3->setEstimate(Siw);

		cv::Mat Tcw_bef = pKFi->mTcwBefMerge;
		Eigen::Matrix<double, 3, 3> Rcw_bef = Converter::toMatrix3d(Tcw_bef.rowRange(0, 3).colRange(0, 3));
		Eigen::Matrix<double, 3, 1> tcw_bef = Converter::toVector3d(Tcw_bef.rowRange(0, 3).col(3)) / scale;
		vScw_bef[nIDi] = g2o::SE3Quat(Rcw_bef, tcw_bef);

		VSE3->setFixed(true);

		VSE3->setId(nIDi);
		VSE3->setMarginalized(false);
		//VSim3->_fix_scale = true;
		vbFromOtherMap[nIDi] = true;

		optimizer.addVertex(VSE3);

		vpVertices[nIDi] = VSE3;

		sIdKF.insert(nIDi);
	}
	Verbose::PrintMess("Opt_Essential: vpFixedCorrectedKFs loaded", Verbose::VERBOSITY_DEBUG);

	for (KeyFrame *pKFi: vpNonFixedKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		const int nIDi = pKFi->mnId;

		if (sIdKF.count(nIDi)) { // It has already added in the corrected merge KFs
			continue;
		}

		g2o::VertexSE3Expmap *VSE3 = new g2o::VertexSE3Expmap();

		//cv::Mat Tcw = pKFi->mTcwBefMerge;
		//Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(Tcw.rowRange(0,3).colRange(0,3));
		//Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(Tcw.rowRange(0,3).col(3));
		Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
		Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation()) / scale;
		g2o::SE3Quat Siw(Rcw, tcw);
		vScw_bef[nIDi] = Siw;
		VSE3->setEstimate(Siw);

		VSE3->setFixed(false);

		VSE3->setId(nIDi);
		VSE3->setMarginalized(false);
		//VSim3->_fix_scale = true;
		vbFromOtherMap[nIDi] = true;

		optimizer.addVertex(VSE3);

		vpVertices[nIDi] = VSE3;

		sIdKF.insert(nIDi);
	}
	Verbose::PrintMess("Opt_Essential: vpNonFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

	vector<KeyFrame *> vpKFs;
	vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
	vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
	vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(), vpFixedCorrectedKFs.end());
	vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
	set<KeyFrame *> spKFs(vpKFs.begin(), vpKFs.end());

	Verbose::PrintMess("Opt_Essential: List of KF loaded", Verbose::VERBOSITY_DEBUG);

	const Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();

	for (KeyFrame *pKFi: vpKFs) {
		int num_connections = 0;
		const int nIDi = pKFi->mnId;

		g2o::SE3Quat Swi = vScw[nIDi].inverse();
		g2o::SE3Quat Swi_bef;
		if (vbFromOtherMap[nIDi]) {
			Swi_bef = vScw_bef[nIDi].inverse();
		}
		/*if(pKFi->mnMergeCorrectedForKF == pCurKF->mnId)
        {
             Swi = vScw[nIDi].inverse();
        }
        else
        {
            cv::Mat Twi = pKFi->mTwcBefMerge;
            Swi = g2o::Sim3(Converter::toMatrix3d(Twi.rowRange(0, 3).colRange(0, 3)),
                            Converter::toVector3d(Twi.rowRange(0, 3).col(3)),1.0);
        }*/

		KeyFrame *pParentKFi = pKFi->GetParent();

		// Spanning tree edge
		if (pParentKFi && spKFs.find(pParentKFi) != spKFs.end()) {
			int nIDj = pParentKFi->mnId;

			g2o::SE3Quat Sjw = vScw[nIDj];
			g2o::SE3Quat Sjw_bef;
			if (vbFromOtherMap[nIDj]) {
				Sjw_bef = vScw_bef[nIDj];
			}

			/*if(pParentKFi->mnMergeCorrectedForKF == pCurKF->mnId)
            {
                 Sjw =  vScw[nIDj];
            }
            else
            {
                cv::Mat Tjw = pParentKFi->mTcwBefMerge;
                Sjw = g2o::Sim3(Converter::toMatrix3d(Tjw.rowRange(0, 3).colRange(0, 3)),
                                Converter::toVector3d(Tjw.rowRange(0, 3).col(3)),1.0);
            }*/

			g2o::SE3Quat Sji;

			if (vbFromOtherMap[nIDi] && vbFromOtherMap[nIDj]) {
				Sji = Sjw_bef * Swi_bef;
			}
			else {
				Sji = Sjw * Swi;
			}

			g2o::EdgeSE3 *e = new g2o::EdgeSE3();
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
			e->setMeasurement(Sji);

			e->information() = matLambda;
			optimizer.addEdge(e);
			num_connections++;
		}

		// Loop edges
		const set<KeyFrame *> sLoopEdges = pKFi->GetLoopEdges();
		for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++) {
			KeyFrame *pLKF = *sit;
			if (spKFs.find(pLKF) != spKFs.end() && pLKF->mnId < pKFi->mnId) {
				g2o::SE3Quat Slw = vScw[pLKF->mnId];
				g2o::SE3Quat Slw_bef;
				if (vbFromOtherMap[pLKF->mnId]) {
					Slw_bef = vScw_bef[pLKF->mnId];
				}

				/*if(pLKF->mnMergeCorrectedForKF == pCurKF->mnId)
                {
                     Slw = vScw[pLKF->mnId];
                }
                else
                {
                    cv::Mat Tlw = pLKF->mTcwBefMerge;
                    Slw = g2o::Sim3(Converter::toMatrix3d(Tlw.rowRange(0, 3).colRange(0, 3)),
                                    Converter::toVector3d(Tlw.rowRange(0, 3).col(3)),1.0);
                }*/

				g2o::SE3Quat Sli;

				if (vbFromOtherMap[nIDi] && vbFromOtherMap[pLKF->mnId]) {
					Sli = Slw_bef * Swi_bef;
				}
				else {
					Sli = Slw * Swi;
				}

				g2o::EdgeSE3 *el = new g2o::EdgeSE3();
				el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
				el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
				el->setMeasurement(Sli);
				el->information() = matLambda;
				optimizer.addEdge(el);
				num_connections++;
			}
		}

		// Covisibility graph edges
		const vector<KeyFrame *> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
		for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++) {
			KeyFrame *pKFn = *vit;
			if (pKFn && pKFn != pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn)
				&& spKFs.find(pKFn) != spKFs.end()) {
				if (!pKFn->isBad() && pKFn->mnId < pKFi->mnId) {
					g2o::SE3Quat Snw = vScw[pKFn->mnId];

					g2o::SE3Quat Snw_bef;
					if (vbFromOtherMap[pKFn->mnId]) {
						Snw_bef = vScw_bef[pKFn->mnId];
					}
					/*if(pKFn->mnMergeCorrectedForKF == pCurKF->mnId)
                    {
                        Snw = vScw[pKFn->mnId];
                    }
                    else
                    {
                        cv::Mat Tnw = pKFn->mTcwBefMerge;
                        Snw = g2o::Sim3(Converter::toMatrix3d(Tnw.rowRange(0, 3).colRange(0, 3)),
                                        Converter::toVector3d(Tnw.rowRange(0, 3).col(3)),1.0);
                    }*/

					g2o::SE3Quat Sni;

					if (vbFromOtherMap[nIDi] && vbFromOtherMap[pKFn->mnId]) {
						Sni = Snw_bef * Swi_bef;
					}
					else {
						Sni = Snw * Swi;
					}

					g2o::EdgeSE3 *en = new g2o::EdgeSE3();
					en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
					en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
					en->setMeasurement(Sni);
					en->information() = matLambda;
					optimizer.addEdge(en);
					num_connections++;
				}
			}
		}

		if (num_connections == 0) {
			Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections",
			                   Verbose::VERBOSITY_DEBUG);
		}
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(20);

	Verbose::PrintMess("Opt_Essential: Finish the optimization", Verbose::VERBOSITY_DEBUG);

	unique_lock<shared_timed_mutex> lock(pMap->mMutexMapUpdate);

	Verbose::PrintMess("Opt_Essential: Apply the new pose to the KFs", Verbose::VERBOSITY_DEBUG);
	// SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
	for (KeyFrame *pKFi: vpNonFixedKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		const int nIDi = pKFi->mnId;

		g2o::VertexSE3Expmap *VSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(nIDi));
		g2o::SE3Quat CorrectedSiw = VSE3->estimate();
		vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
		Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
		Eigen::Vector3d eigt = CorrectedSiw.translation();
		//double s = CorrectedSiw.scale();

		//eigt *=(1./s); //[R t/s;0 1]

		cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

		/*{
            cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
            cv::Vec3d trasl = Tco_cn.rowRange(0,3).col(3);
            double dist = cv::norm(trasl);
            if(dist > 1.0)
            {
                cout << "--Distance: " << dist << " meters" << endl;
                cout << "--To much distance correction in EssentGraph: It has connected " << pKFi->GetVectorCovisibleKeyFrames().size() << " KFs" << endl;
            }

            string strNameFile = pKFi->mNameFile;
            cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

            cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

            vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();
            for(int j=0; j<vpMapPointsKFi.size(); ++j)
            {
                if(!vpMapPointsKFi[j] || vpMapPointsKFi[j]->isBad())
                {
                    continue;
                }
                string strNumOBs = to_string(vpMapPointsKFi[j]->Observations());
                cv::circle(imLeft, pKFi->mvKeys[j].pt, 2, cv::Scalar(0, 255, 0));
                cv::putText(imLeft, strNumOBs, pKFi->mvKeys[j].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
            }

            string namefile = "./test_OptEssent/Essent_" + to_string(pCurKF->mnId) + "_KF" + to_string(pKFi->mnId) +"_D" + to_string(dist) +".png";
            cv::imwrite(namefile, imLeft);
        }*/

		pKFi->mTcwBefMerge = pKFi->GetPose();
		pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
		pKFi->SetPose(Tiw);
	}

	Verbose::PrintMess("Opt_Essential: Apply the new pose to the MPs", Verbose::VERBOSITY_DEBUG);
	cout << "Opt_Essential: number of points -> " << vpNonCorrectedMPs.size() << endl;
	// Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
	for (MapPoint *pMPi: vpNonCorrectedMPs) {
		if (pMPi->isBad()) {
			continue;
		}

		//Verbose::PrintMess("Opt_Essential: MP id " + to_string(pMPi->mnId), Verbose::VERBOSITY_DEBUG);
		/*int nIDr;
        if(pMPi->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMPi->mnCorrectedReference;
        }
        else
        {

        }*/
		KeyFrame *pRefKF = pMPi->GetReferenceKeyFrame();
		g2o::SE3Quat Srw;
		g2o::SE3Quat correctedSwr;
		while (pRefKF->isBad()) {
			if (!pRefKF) {
				Verbose::PrintMess("MP " + to_string(pMPi->mnId) + " without a valid reference KF",
				                   Verbose::VERBOSITY_DEBUG);
				break;
			}

			pMPi->EraseObservation(pRefKF);
			pRefKF = pMPi->GetReferenceKeyFrame();
		}
		/*if(pRefKF->mnMergeCorrectedForKF == pCurKF->mnId)
        {
            int nIDr = pRefKF->mnId;

            Srw = vScw[nIDr];
            correctedSwr = vCorrectedSwc[nIDr];
        }
        else
        {*/
		//cv::Mat TNonCorrectedwr = pRefKF->mTwcBefMerge;
		//Eigen::Matrix<double,3,3> RNonCorrectedwr = Converter::toMatrix3d(TNonCorrectedwr.rowRange(0,3).colRange(0,3));
		//Eigen::Matrix<double,3,1> tNonCorrectedwr = Converter::toVector3d(TNonCorrectedwr.rowRange(0,3).col(3));
		Srw = vScw_bef[pRefKF->mnId]; //g2o::SE3Quat(RNonCorrectedwr,tNonCorrectedwr).inverse();

		cv::Mat Twr = pRefKF->GetPoseInverse();
		Eigen::Matrix<double, 3, 3> Rwr = Converter::toMatrix3d(Twr.rowRange(0, 3).colRange(0, 3));
		Eigen::Matrix<double, 3, 1> twr = Converter::toVector3d(Twr.rowRange(0, 3).col(3));
		correctedSwr = g2o::SE3Quat(Rwr, twr);
		//}
		//cout << "Opt_Essential: Loaded the KF reference position" << endl;

		cv::Mat P3Dw = pMPi->GetWorldPos() / scale;
		Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
		Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

		//cout << "Opt_Essential: Calculated the new MP position" << endl;
		cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
		//cout << "Opt_Essential: Converted the position to the OpenCV format" << endl;
		pMPi->SetWorldPos(cvCorrectedP3Dw);
		//cout << "Opt_Essential: Loaded the corrected position in the MP object" << endl;

		pMPi->UpdateNormalAndDepth();
	}

	Verbose::PrintMess("Opt_Essential: End of the optimization", Verbose::VERBOSITY_DEBUG);
}

void Optimizer::OptimizeEssentialGraph(KeyFrame *pCurKF,
                                       vector<KeyFrame *> &vpFixedKFs,
                                       vector<KeyFrame *> &vpFixedCorrectedKFs,
                                       vector<KeyFrame *> &vpNonFixedKFs,
                                       vector<MapPoint *> &vpNonCorrectedMPs,
                                       bool &bFinished,
                                       bool *bReSet)
{
	Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedKFs.size()) + " KFs fixed in the merged map",
	                   Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess(
		"Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map",
		Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess(
		"Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map",
		Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess(
		"Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map",
		Verbose::VERBOSITY_DEBUG);

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
		new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
	g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	solver->setUserLambdaInit(1e-16);
	optimizer.setAlgorithm(solver);

	Map *pMap = pCurKF->GetMap();
	const unsigned int nMaxKFid = pMap->GetMaxKFid();

	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
	vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

	const int minFeat = 100;

	for (KeyFrame *pKFi: vpFixedKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

		const int nIDi = pKFi->mnId;

		Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
		Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
		g2o::Sim3 Siw(Rcw, tcw, 1.0);
		vScw[nIDi] = Siw;
		vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
		VSim3->setEstimate(Siw);

		VSim3->setFixed(true);

		VSim3->setId(nIDi);
		VSim3->setMarginalized(false);
		VSim3->_fix_scale = true; //TODO

		optimizer.addVertex(VSim3);

		vpVertices[nIDi] = VSim3;
	}
	Verbose::PrintMess("Opt_Essential: vpFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

	set<unsigned long> sIdKF;
	for (KeyFrame *pKFi: vpFixedCorrectedKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

		const int nIDi = pKFi->mnId;

		Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
		Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
		g2o::Sim3 Siw(Rcw, tcw, 1.0);
		//vScw[nIDi] = Siw;
		vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
		VSim3->setEstimate(Siw);

		cv::Mat Tcw_bef = pKFi->mTcwBefMerge;
		Eigen::Matrix<double, 3, 3> Rcw_bef = Converter::toMatrix3d(Tcw_bef.rowRange(0, 3).colRange(0, 3));
		Eigen::Matrix<double, 3, 1> tcw_bef = Converter::toVector3d(Tcw_bef.rowRange(0, 3).col(3));
		vScw[nIDi] = g2o::Sim3(Rcw_bef, tcw_bef, 1.0);

		VSim3->setFixed(true);

		VSim3->setId(nIDi);
		VSim3->setMarginalized(false);

		optimizer.addVertex(VSim3);

		vpVertices[nIDi] = VSim3;

		sIdKF.insert(nIDi);
	}
	Verbose::PrintMess("Opt_Essential: vpFixedCorrectedKFs loaded", Verbose::VERBOSITY_DEBUG);

	for (KeyFrame *pKFi: vpNonFixedKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		const int nIDi = pKFi->mnId;

		if (sIdKF.count(nIDi)) { // It has already added in the corrected merge KFs
			continue;
		}

		g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

		//cv::Mat Tcw = pKFi->mTcwBefMerge;
		//Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(Tcw.rowRange(0,3).colRange(0,3));
		//Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(Tcw.rowRange(0,3).col(3));
		Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
		Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
		g2o::Sim3 Siw(Rcw, tcw, 1.0);
		vScw[nIDi] = Siw;
		VSim3->setEstimate(Siw);

		VSim3->setFixed(false);

		VSim3->setId(nIDi);
		VSim3->setMarginalized(false);

		optimizer.addVertex(VSim3);

		vpVertices[nIDi] = VSim3;

		sIdKF.insert(nIDi);
	}
	Verbose::PrintMess("Opt_Essential: vpNonFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

	vector<KeyFrame *> vpKFs;
	vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
	vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
	vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(), vpFixedCorrectedKFs.end());
	vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
	set<KeyFrame *> spKFs(vpKFs.begin(), vpKFs.end());

	Verbose::PrintMess("Opt_Essential: List of KF loaded", Verbose::VERBOSITY_DEBUG);

	const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

	for (KeyFrame *pKFi: vpKFs) {
		int num_connections = 0;
		const int nIDi = pKFi->mnId;

		g2o::Sim3 Swi = vScw[nIDi].inverse();
		/*if(pKFi->mnMergeCorrectedForKF == pCurKF->mnId)
        {
             Swi = vScw[nIDi].inverse();
        }
        else
        {
            cv::Mat Twi = pKFi->mTwcBefMerge;
            Swi = g2o::Sim3(Converter::toMatrix3d(Twi.rowRange(0, 3).colRange(0, 3)),
                            Converter::toVector3d(Twi.rowRange(0, 3).col(3)),1.0);
        }*/

		KeyFrame *pParentKFi = pKFi->GetParent();

		// Spanning tree edge
		if (pParentKFi && spKFs.find(pParentKFi) != spKFs.end()) {
			int nIDj = pParentKFi->mnId;

			g2o::Sim3 Sjw = vScw[nIDj];

			/*if(pParentKFi->mnMergeCorrectedForKF == pCurKF->mnId)
            {
                 Sjw =  vScw[nIDj];
            }
            else
            {
                cv::Mat Tjw = pParentKFi->mTcwBefMerge;
                Sjw = g2o::Sim3(Converter::toMatrix3d(Tjw.rowRange(0, 3).colRange(0, 3)),
                                Converter::toVector3d(Tjw.rowRange(0, 3).col(3)),1.0);
            }*/

			g2o::Sim3 Sji = Sjw * Swi;

			g2o::EdgeSim3 *e = new g2o::EdgeSim3();
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
			e->setMeasurement(Sji);

			e->information() = matLambda;
			optimizer.addEdge(e);
			num_connections++;
		}

		// Loop edges
		const set<KeyFrame *> sLoopEdges = pKFi->GetLoopEdges();
		for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++) {
			KeyFrame *pLKF = *sit;
			if (spKFs.find(pLKF) != spKFs.end() && pLKF->mnId < pKFi->mnId) {
				g2o::Sim3 Slw = vScw[pLKF->mnId];

				/*if(pLKF->mnMergeCorrectedForKF == pCurKF->mnId)
                {
                     Slw = vScw[pLKF->mnId];
                }
                else
                {
                    cv::Mat Tlw = pLKF->mTcwBefMerge;
                    Slw = g2o::Sim3(Converter::toMatrix3d(Tlw.rowRange(0, 3).colRange(0, 3)),
                                    Converter::toVector3d(Tlw.rowRange(0, 3).col(3)),1.0);
                }*/

				g2o::Sim3 Sli = Slw * Swi;
				g2o::EdgeSim3 *el = new g2o::EdgeSim3();
				el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
				el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
				el->setMeasurement(Sli);
				el->information() = matLambda;
				optimizer.addEdge(el);
				num_connections++;
			}
		}

		// Covisibility graph edges
		const vector<KeyFrame *> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
		for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++) {
			KeyFrame *pKFn = *vit;
			if (pKFn && pKFn != pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn)
				&& spKFs.find(pKFn) != spKFs.end()) {
				if (!pKFn->isBad() && pKFn->mnId < pKFi->mnId) {

					g2o::Sim3 Snw = vScw[pKFn->mnId];
					/*if(pKFn->mnMergeCorrectedForKF == pCurKF->mnId)
                    {
                        Snw = vScw[pKFn->mnId];
                    }
                    else
                    {
                        cv::Mat Tnw = pKFn->mTcwBefMerge;
                        Snw = g2o::Sim3(Converter::toMatrix3d(Tnw.rowRange(0, 3).colRange(0, 3)),
                                        Converter::toVector3d(Tnw.rowRange(0, 3).col(3)),1.0);
                    }*/

					g2o::Sim3 Sni = Snw * Swi;

					g2o::EdgeSim3 *en = new g2o::EdgeSim3();
					en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
					en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
					en->setMeasurement(Sni);
					en->information() = matLambda;
					optimizer.addEdge(en);
					num_connections++;
				}
			}
		}

		if (num_connections == 0) {
			Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections",
			                   Verbose::VERBOSITY_DEBUG);
		}
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(20);

	Verbose::PrintMess("Opt_Essential: Finish the optimization", Verbose::VERBOSITY_DEBUG);

	unique_lock<shared_timed_mutex> lock(pMap->mMutexMapUpdate, std::defer_lock);
	while (!lock.try_lock_for(std::chrono::milliseconds(300))) {
		if (*bReSet) {
			bFinished = false;
			return;
		}
	}

	Verbose::PrintMess("Opt_Essential: Apply the new pose to the KFs", Verbose::VERBOSITY_DEBUG);
	// SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
	for (KeyFrame *pKFi: vpNonFixedKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		const int nIDi = pKFi->mnId;

		g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
		g2o::Sim3 CorrectedSiw = VSim3->estimate();
		vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
		Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
		Eigen::Vector3d eigt = CorrectedSiw.translation();
		double s = CorrectedSiw.scale();

		eigt *= (1. / s); //[R t/s;0 1]

		cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

		/*{
            cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
            cv::Vec3d trasl = Tco_cn.rowRange(0,3).col(3);
            double dist = cv::norm(trasl);
            if(dist > 1.0)
            {
                cout << "--Distance: " << dist << " meters" << endl;
                cout << "--To much distance correction in EssentGraph: It has connected " << pKFi->GetVectorCovisibleKeyFrames().size() << " KFs" << endl;
            }
            string strNameFile = pKFi->mNameFile;
            cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);
            cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);
            vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();
            for(int j=0; j<vpMapPointsKFi.size(); ++j)
            {
                if(!vpMapPointsKFi[j] || vpMapPointsKFi[j]->isBad())
                {
                    continue;
                }
                string strNumOBs = to_string(vpMapPointsKFi[j]->Observations());
                cv::circle(imLeft, pKFi->mvKeys[j].pt, 2, cv::Scalar(0, 255, 0));
                cv::putText(imLeft, strNumOBs, pKFi->mvKeys[j].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
            }
            string namefile = "./test_OptEssent/Essent_" + to_string(pCurKF->mnId) + "_KF" + to_string(pKFi->mnId) +"_D" + to_string(dist) +".png";
            cv::imwrite(namefile, imLeft);
        }*/

		pKFi->mTcwBefMerge = pKFi->GetPose();
		pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
		pKFi->SetPose(Tiw);
	}

	Verbose::PrintMess("Opt_Essential: Apply the new pose to the MPs", Verbose::VERBOSITY_DEBUG);
	// Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
	for (MapPoint *pMPi: vpNonCorrectedMPs) {
		if (pMPi->isBad()) {
			continue;
		}

		Verbose::PrintMess("Opt_Essential: MP id " + to_string(pMPi->mnId), Verbose::VERBOSITY_DEBUG);
		/*int nIDr;
        if(pMPi->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMPi->mnCorrectedReference;
        }
        else
        {
        }*/
		KeyFrame *pRefKF = pMPi->GetReferenceKeyFrame();
		g2o::Sim3 Srw;
		g2o::Sim3 correctedSwr;
		while (pRefKF->isBad()) {
			if (!pRefKF) {
				Verbose::PrintMess("MP " + to_string(pMPi->mnId) + " without a valid reference KF",
				                   Verbose::VERBOSITY_DEBUG);
				break;
			}

			pMPi->EraseObservation(pRefKF);
			pRefKF = pMPi->GetReferenceKeyFrame();
		}
		/*if(pRefKF->mnMergeCorrectedForKF == pCurKF->mnId)
        {
            int nIDr = pRefKF->mnId;
            Srw = vScw[nIDr];
            correctedSwr = vCorrectedSwc[nIDr];
        }
        else
        {*/
		cv::Mat TNonCorrectedwr = pRefKF->mTwcBefMerge;
		Eigen::Matrix<double, 3, 3>
			RNonCorrectedwr = Converter::toMatrix3d(TNonCorrectedwr.rowRange(0, 3).colRange(0, 3));
		Eigen::Matrix<double, 3, 1> tNonCorrectedwr = Converter::toVector3d(TNonCorrectedwr.rowRange(0, 3).col(3));
		Srw = g2o::Sim3(RNonCorrectedwr, tNonCorrectedwr, 1.0).inverse();

		cv::Mat Twr = pRefKF->GetPoseInverse();
		Eigen::Matrix<double, 3, 3> Rwr = Converter::toMatrix3d(Twr.rowRange(0, 3).colRange(0, 3));
		Eigen::Matrix<double, 3, 1> twr = Converter::toVector3d(Twr.rowRange(0, 3).col(3));
		correctedSwr = g2o::Sim3(Rwr, twr, 1.0);
		//}
		//cout << "Opt_Essential: Loaded the KF reference position" << endl;

		cv::Mat P3Dw = pMPi->GetWorldPos();
		Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
		Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

		//cout << "Opt_Essential: Calculated the new MP position" << endl;
		cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
		//cout << "Opt_Essential: Converted the position to the OpenCV format" << endl;
		pMPi->SetWorldPos(cvCorrectedP3Dw);
		//cout << "Opt_Essential: Loaded the corrected position in the MP object" << endl;

		pMPi->UpdateNormalAndDepth();
	}

	Verbose::PrintMess("Opt_Essential: End of the optimization", Verbose::VERBOSITY_DEBUG);
}

void Optimizer::GlobalVAPoseGraphOptimization(KeyFrame* pCurKF, vector<KeyFrame*> &vpFixedKFs,
                                                     vector<KeyFrame*> &vpFixedCorrectedKFs,
                                                      vector<KeyFrame *> &vpNonFixedKFs,
                                                      vector<MapPoint *> &vpNonCorrectedMPs,
                                                     Atlas* pAtlas)
{
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    Map *pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
    vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

    const int minFeat = 100;
    set<unsigned long> sIdKF;
    set<unsigned long> sIdMP;

    for (KeyFrame *pKFi: vpFixedKFs) {
        if (pKFi->isBad()) {
            continue;
        }

        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
        g2o::Sim3 Siw(Rcw, tcw, 1.0);
        vScw[nIDi] = Siw;
        vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
        VSim3->setEstimate(Siw);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = true; //TODO

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;
        sIdKF.insert(nIDi);
    }


    for (KeyFrame *pKFi: vpFixedCorrectedKFs) {
        if (pKFi->isBad()) {
            continue;
        }

        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
        g2o::Sim3 Siw(Rcw, tcw, 1.0);
        //vScw[nIDi] = Siw;
        vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
        VSim3->setEstimate(Siw);

        cv::Mat Tcw_bef = pKFi->mTcwBefMerge;
        Eigen::Matrix<double, 3, 3> Rcw_bef = Converter::toMatrix3d(Tcw_bef.rowRange(0, 3).colRange(0, 3));
        Eigen::Matrix<double, 3, 1> tcw_bef = Converter::toVector3d(Tcw_bef.rowRange(0, 3).col(3));
        vScw[nIDi] = g2o::Sim3(Rcw_bef, tcw_bef, 1.0);

        VSim3->setFixed(true);
        VSim3->_fix_scale = true;
        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;

        sIdKF.insert(nIDi);
    }

    for (KeyFrame *pKFi: vpNonFixedKFs) {
        if (pKFi->isBad()) {
            continue;
        }

        const int nIDi = pKFi->mnId;

        if (sIdKF.count(nIDi)) { // It has already added in the corrected merge KFs
            continue;
        }

        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        //cv::Mat Tcw = pKFi->mTcwBefMerge;
        //Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(Tcw.rowRange(0,3).colRange(0,3));
        //Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(Tcw.rowRange(0,3).col(3));
        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
        g2o::Sim3 Siw(Rcw, tcw, 1.0);
        vScw[nIDi] = Siw;
        VSim3->setEstimate(Siw);

        VSim3->setFixed(false);
        VSim3->_fix_scale = true;
        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;

        sIdKF.insert(nIDi);
    }
    vector<KeyFrame*> KFs_in_othermaps;
    for(auto m:pAtlas->GetAllMaps()){
        for (KeyFrame *pKFi: m->GetAllKeyFrames()) {
            if (pKFi->isBad()) {
                // ROS_DEBUG_STREAM("KF: "<<pKFi->mnId<<"is bad");  // original
                RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), "KF: "<<pKFi->mnId<<"is bad");
                continue;
            }

            const int nIDi = pKFi->mnId;

            if (sIdKF.count(nIDi)) { // It has already added in the corrected merge KFs
                continue;
            }

            g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

            //cv::Mat Tcw = pKFi->mTcwBefMerge;
            //Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(Tcw.rowRange(0,3).colRange(0,3));
            //Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(Tcw.rowRange(0,3).col(3));
            Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
            Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
            g2o::Sim3 Siw(Rcw, tcw, 1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);

            VSim3->setFixed(false);
            VSim3->_fix_scale = true;
            VSim3->setId(nIDi);
            VSim3->setMarginalized(false);

            optimizer.addVertex(VSim3);

            vpVertices[nIDi] = VSim3;

            sIdKF.insert(nIDi);
            KFs_in_othermaps.push_back(pKFi);
        }
    }


    vector<KeyFrame *> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size()+ KFs_in_othermaps.size());
    vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(), vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
    vpKFs.insert(vpKFs.end(), KFs_in_othermaps.begin(), KFs_in_othermaps.end());
    set<KeyFrame *> spKFs(vpKFs.begin(), vpKFs.end());


    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

    for (KeyFrame *pKFi: vpKFs) {
        int num_connections = 0;
        const int nIDi = pKFi->mnId;

        g2o::Sim3 Swi = vScw[nIDi].inverse();

        KeyFrame *pParentKFi = pKFi->GetParent();

        // Spanning tree edge
        if (pParentKFi && spKFs.find(pParentKFi) != spKFs.end()) {
            int nIDj = pParentKFi->mnId;

            g2o::Sim3 Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
            num_connections++;
        }

        // Loop edges
        const set<KeyFrame *> sLoopEdges = pKFi->GetLoopEdges();
        for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++) {
            KeyFrame *pLKF = *sit;
            if (spKFs.find(pLKF) != spKFs.end() && pLKF->mnId < pKFi->mnId) {
                g2o::Sim3 Slw = vScw[pLKF->mnId];


                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
                num_connections++;
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame *> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++) {
            KeyFrame *pKFn = *vit;
            if (pKFn && pKFn != pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn)
                && spKFs.find(pKFn) != spKFs.end()) {
                if (!pKFn->isBad() && pKFn->mnId < pKFi->mnId) {

                    g2o::Sim3 Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                    num_connections++;
                }
            }
        }

        if (pKFi->mpDvlPreintegrationLossRefKF){
            KeyFrame *pKFn = pKFi->mpLossRefKF;
            if (pKFn && pKFn != pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn)
                && spKFs.find(pKFn) != spKFs.end()) {
                if (!pKFn->isBad() && pKFn->mnId < pKFi->mnId) {

                    g2o::Sim3 Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                    num_connections++;
                }
            }
//             ROS_INFO_STREAM("add multi-map constrain for global pose graph, from ID: "<<pKFi->mnId<<"to ID: "<<pKFn->mnId);  // original
        }

        if (num_connections == 0) {
            Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections",
                               Verbose::VERBOSITY_DEBUG);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    Verbose::PrintMess("Opt_Essential: Finish the optimization", Verbose::VERBOSITY_DEBUG);

    // unique_lock<timed_mutex> lock(pMap->mMutexMapUpdate, std::defer_lock);

    Verbose::PrintMess("Opt_Essential: Apply the new pose to the KFs", Verbose::VERBOSITY_DEBUG);
    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (KeyFrame *pKFi: vpKFs) {
        if (pKFi->isBad()) {
            continue;
        }

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw = VSim3->estimate();
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *= (1. / s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);


        pKFi->mTcwBefMerge = pKFi->GetPose();
        pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
//         // ROS_INFO_STREAM("set KF: "<<pKFi->mnId<<" pose before merging");  // original
        pKFi->SetPose(Tiw);

        if(pKFi->GetMapPointMatches().size()==0){
            continue;
        }
        for(auto pMPi:pKFi->GetMapPointMatches()){
            if(!pMPi){
                continue;
            }
            if (!sIdMP.count(pMPi->mnId)){
                if (pMPi->isBad()) {
                    continue;
                }
                sIdMP.insert(pMPi->mnId);

                KeyFrame *pRefKF = pKFi;
                g2o::Sim3 Srw;
                g2o::Sim3 correctedSwr;
                while (pRefKF->isBad()) {
                    if (!pRefKF) {
                        break;
                    }

                    pMPi->EraseObservation(pRefKF);
                    pRefKF = pMPi->GetReferenceKeyFrame();
                }
                // if (!sIdKF.count(pRefKF->mnId)){
//                 //     ROS_INFO_STREAM("skip update map point["<<pMPi->mnId<<"] refer to KF["<<pRefKF->mnId<<"]");  // original
                //     continue;
                // }

                cv::Mat TNonCorrectedwr = pRefKF->mTwcBefMerge;
                Eigen::Matrix<double, 3, 3>
                        RNonCorrectedwr = Converter::toMatrix3d(TNonCorrectedwr.rowRange(0, 3).colRange(0, 3));
                Eigen::Matrix<double, 3, 1> tNonCorrectedwr = Converter::toVector3d(TNonCorrectedwr.rowRange(0, 3).col(3));
                Srw = g2o::Sim3(RNonCorrectedwr, tNonCorrectedwr, 1.0).inverse();

                cv::Mat Twr = pRefKF->GetPoseInverse();
                Eigen::Matrix<double, 3, 3> Rwr = Converter::toMatrix3d(Twr.rowRange(0, 3).colRange(0, 3));
                Eigen::Matrix<double, 3, 1> twr = Converter::toVector3d(Twr.rowRange(0, 3).col(3));
                correctedSwr = g2o::Sim3(Rwr, twr, 1.0);
                //}
                //cout << "Opt_Essential: Loaded the KF reference position" << endl;

                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

                //cout << "Opt_Essential: Calculated the new MP position" << endl;
                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                //cout << "Opt_Essential: Converted the position to the OpenCV format" << endl;
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                //cout << "Opt_Essential: Loaded the corrected position in the MP object" << endl;

                pMPi->UpdateNormalAndDepth();
            }

        }
    }
    // for(auto m:pAtlas->GetAllMaps()){
    //     for (MapPoint *pMPi: m->GetAllMapPoints()) {
    //         if (pMPi->isBad()) {
    //             continue;
    //         }
    //
    //         KeyFrame *pRefKF = pMPi->GetReferenceKeyFrame();
    //         g2o::Sim3 Srw;
    //         g2o::Sim3 correctedSwr;
    //         while (pRefKF->isBad()) {
    //             if (!pRefKF) {
    //                 break;
    //             }
    //
    //             pMPi->EraseObservation(pRefKF);
    //             pRefKF = pMPi->GetReferenceKeyFrame();
    //         }
    //         // if (!sIdKF.count(pRefKF->mnId)){
//     //         //     ROS_INFO_STREAM("skip update map point["<<pMPi->mnId<<"] refer to KF["<<pRefKF->mnId<<"]");  // original
    //         //     continue;
    //         // }
    //
    //         cv::Mat TNonCorrectedwr = pRefKF->mTwcBefMerge;
    //         Eigen::Matrix<double, 3, 3>
    //                 RNonCorrectedwr = Converter::toMatrix3d(TNonCorrectedwr.rowRange(0, 3).colRange(0, 3));
    //         Eigen::Matrix<double, 3, 1> tNonCorrectedwr = Converter::toVector3d(TNonCorrectedwr.rowRange(0, 3).col(3));
    //         Srw = g2o::Sim3(RNonCorrectedwr, tNonCorrectedwr, 1.0).inverse();
    //
    //         cv::Mat Twr = pRefKF->GetPoseInverse();
    //         Eigen::Matrix<double, 3, 3> Rwr = Converter::toMatrix3d(Twr.rowRange(0, 3).colRange(0, 3));
    //         Eigen::Matrix<double, 3, 1> twr = Converter::toVector3d(Twr.rowRange(0, 3).col(3));
    //         correctedSwr = g2o::Sim3(Rwr, twr, 1.0);
    //         //}
    //         //cout << "Opt_Essential: Loaded the KF reference position" << endl;
    //
    //         cv::Mat P3Dw = pMPi->GetWorldPos();
    //         Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
    //         Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
    //
    //         //cout << "Opt_Essential: Calculated the new MP position" << endl;
    //         cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
    //         //cout << "Opt_Essential: Converted the position to the OpenCV format" << endl;
    //         pMPi->SetWorldPos(cvCorrectedP3Dw);
    //         //cout << "Opt_Essential: Loaded the corrected position in the MP object" << endl;
    //
    //         pMPi->UpdateNormalAndDepth();
    //     }
    // }

}

void Optimizer::OptimizeEssentialGraph(KeyFrame *pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3)
{
	// Setup optimizer
	Map *pMap = pCurKF->GetMap();
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
		new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
	g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	solver->setUserLambdaInit(1e-16);
	optimizer.setAlgorithm(solver);

	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
	const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

	const unsigned int nMaxKFid = pMap->GetMaxKFid();

	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
	vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

	const int minFeat = 100; // TODO Check. originally 100

	// Set KeyFrame vertices
	for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
		KeyFrame *pKF = vpKFs[i];
		if (pKF->isBad()) {
			continue;
		}
		g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

		const int nIDi = pKF->mnId;

		Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
		Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
		g2o::Sim3 Siw(Rcw, tcw, 1.0);
		vScw[nIDi] = Siw;
		VSim3->setEstimate(Siw);

		if (pKF->mnBALocalForKF == pCurKF->mnId || pKF->mnBAFixedForKF == pCurKF->mnId) {
			cout << "fixed fk: " << pKF->mnId << endl;
			VSim3->setFixed(true);
		}
		else {
			VSim3->setFixed(false);
		}

		VSim3->setId(nIDi);
		VSim3->setMarginalized(false);
		// TODO Check
		// VSim3->_fix_scale = bFixScale;

		optimizer.addVertex(VSim3);

		vpVertices[nIDi] = VSim3;
	}

	set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

	const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

	int count_edges[3] = {0, 0, 0};
	// Set normal edges
	for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
		KeyFrame *pKF = vpKFs[i];

		const int nIDi = pKF->mnId;

		g2o::Sim3 Swi;

		LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

		if (iti != NonCorrectedSim3.end()) {
			Swi = (iti->second).inverse();
		}
		else {
			Swi = vScw[nIDi].inverse();
		}

		KeyFrame *pParentKF = pKF->GetParent();

		// Spanning tree edge
		if (pParentKF) {
			int nIDj = pParentKF->mnId;

			g2o::Sim3 Sjw;
			LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

			if (itj != NonCorrectedSim3.end()) {
				Sjw = itj->second;
			}
			else {
				Sjw = vScw[nIDj];
			}

			g2o::Sim3 Sji = Sjw * Swi;

			g2o::EdgeSim3 *e = new g2o::EdgeSim3();
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
			e->setMeasurement(Sji);

			e->information() = matLambda;
			optimizer.addEdge(e);
			count_edges[0]++;
		}

		// Loop edges
		const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
		for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++) {
			KeyFrame *pLKF = *sit;
			if (pLKF->mnId < pKF->mnId) {
				g2o::Sim3 Slw;
				LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

				if (itl != NonCorrectedSim3.end()) {
					Slw = itl->second;
				}
				else {
					Slw = vScw[pLKF->mnId];
				}

				g2o::Sim3 Sli = Slw * Swi;
				g2o::EdgeSim3 *el = new g2o::EdgeSim3();
				el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
				el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
				el->setMeasurement(Sli);
				el->information() = matLambda;
				optimizer.addEdge(el);
				count_edges[1]++;
			}
		}

		// Covisibility graph edges
		const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
		for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++) {
			KeyFrame *pKFn = *vit;
			if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn)) {
				if (!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
					// just one edge between frames
					if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId)))) {
						continue;
					}

					g2o::Sim3 Snw;

					LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

					if (itn != NonCorrectedSim3.end()) {
						Snw = itn->second;
					}
					else {
						Snw = vScw[pKFn->mnId];
					}

					g2o::Sim3 Sni = Snw * Swi;

					g2o::EdgeSim3 *en = new g2o::EdgeSim3();
					en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
					en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
					en->setMeasurement(Sni);
					en->information() = matLambda;
					optimizer.addEdge(en);
					count_edges[2]++;
				}
			}
		}
	}

	Verbose::PrintMess("edges pose graph: " + to_string(count_edges[0]) + ", " + to_string(count_edges[1]) + ", "
		                   + to_string(count_edges[2]), Verbose::VERBOSITY_DEBUG);
	// Optimize!
	optimizer.initializeOptimization();
	optimizer.setVerbose(false);
	optimizer.optimize(20);

	unique_lock<shared_timed_mutex> lock(pMap->mMutexMapUpdate);

	// SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		const int nIDi = pKFi->mnId;

		g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
		g2o::Sim3 CorrectedSiw = VSim3->estimate();
		vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
		Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
		Eigen::Vector3d eigt = CorrectedSiw.translation();
		double s = CorrectedSiw.scale();

		eigt *= (1. / s); //[R t/s;0 1]

		cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

		pKFi->SetPose(Tiw);
	}

	// Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
	for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
		MapPoint *pMP = vpMPs[i];

		if (pMP->isBad()) {
			continue;
		}

		int nIDr;
		if (pMP->mnCorrectedByKF == pCurKF->mnId) {
			nIDr = pMP->mnCorrectedReference;
		}
		else {
			KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
			nIDr = pRefKF->mnId;
		}

		g2o::Sim3 Srw = vScw[nIDr];
		g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

		cv::Mat P3Dw = pMP->GetWorldPos();
		Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
		Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

		cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
		pMP->SetWorldPos(cvCorrectedP3Dw);

		pMP->UpdateNormalAndDepth();
	}

	// TODO Check this changeindex
	pMap->IncreaseChangeIndex();
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1,
                            KeyFrame *pKF2,
                            vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12,
                            const float th2,
                            const bool bFixScale)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// Calibration
	const cv::Mat &K1 = pKF1->mK;
	const cv::Mat &K2 = pKF2->mK;

	// Camera poses
	const cv::Mat R1w = pKF1->GetRotation();
	const cv::Mat t1w = pKF1->GetTranslation();
	const cv::Mat R2w = pKF2->GetRotation();
	const cv::Mat t2w = pKF2->GetTranslation();

	// Set Sim3 vertex
	g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
	vSim3->_fix_scale = bFixScale;
	vSim3->setEstimate(g2oS12);
	vSim3->setId(0);
	vSim3->setFixed(false);
	vSim3->_principle_point1[0] = K1.at<float>(0, 2);
	vSim3->_principle_point1[1] = K1.at<float>(1, 2);
	vSim3->_focal_length1[0] = K1.at<float>(0, 0);
	vSim3->_focal_length1[1] = K1.at<float>(1, 1);
	vSim3->_principle_point2[0] = K2.at<float>(0, 2);
	vSim3->_principle_point2[1] = K2.at<float>(1, 2);
	vSim3->_focal_length2[0] = K2.at<float>(0, 0);
	vSim3->_focal_length2[1] = K2.at<float>(1, 1);
	optimizer.addVertex(vSim3);

	// Set MapPoint vertices
	const int N = vpMatches1.size();
	const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
	vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;
	vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;
	vector<size_t> vnIndexEdge;

	vnIndexEdge.reserve(2 * N);
	vpEdges12.reserve(2 * N);
	vpEdges21.reserve(2 * N);

	const float deltaHuber = sqrt(th2);

	int nCorrespondences = 0;

	for (int i = 0; i < N; i++) {
		if (!vpMatches1[i]) {
			continue;
		}

		MapPoint *pMP1 = vpMapPoints1[i];
		MapPoint *pMP2 = vpMatches1[i];

		const int id1 = 2 * i + 1;
		const int id2 = 2 * (i + 1);

		const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKF2));

		if (pMP1 && pMP2) {
			if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0) {
				g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
				cv::Mat P3D1w = pMP1->GetWorldPos();
				cv::Mat P3D1c = R1w * P3D1w + t1w;
				vPoint1->setEstimate(Converter::toVector3d(P3D1c));
				vPoint1->setId(id1);
				vPoint1->setFixed(true);
				optimizer.addVertex(vPoint1);

				g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
				cv::Mat P3D2w = pMP2->GetWorldPos();
				cv::Mat P3D2c = R2w * P3D2w + t2w;
				vPoint2->setEstimate(Converter::toVector3d(P3D2c));
				vPoint2->setId(id2);
				vPoint2->setFixed(true);
				optimizer.addVertex(vPoint2);
			}
			else {
				continue;
			}
		}
		else {
			continue;
		}

		nCorrespondences++;

		// Set edge x1 = S12*X2
		Eigen::Matrix<double, 2, 1> obs1;
		const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
		obs1 << kpUn1.pt.x, kpUn1.pt.y;

		g2o::EdgeSim3ProjectXYZ *e12 = new g2o::EdgeSim3ProjectXYZ();
		e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
		e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
		e12->setMeasurement(obs1);
		const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
		e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

		g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
		e12->setRobustKernel(rk1);
		rk1->setDelta(deltaHuber);
		optimizer.addEdge(e12);

		// Set edge x2 = S21*X1
		Eigen::Matrix<double, 2, 1> obs2;
		const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
		obs2 << kpUn2.pt.x, kpUn2.pt.y;

		g2o::EdgeInverseSim3ProjectXYZ *e21 = new g2o::EdgeInverseSim3ProjectXYZ();

		e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
		e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
		e21->setMeasurement(obs2);
		float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
		e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

		g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
		e21->setRobustKernel(rk2);
		rk2->setDelta(deltaHuber);
		optimizer.addEdge(e21);

		vpEdges12.push_back(e12);
		vpEdges21.push_back(e21);
		vnIndexEdge.push_back(i);
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(5);

	// Check inliers
	int nBad = 0;
	for (size_t i = 0; i < vpEdges12.size(); i++) {
		g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
		g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
		if (!e12 || !e21) {
			continue;
		}

		if (e12->chi2() > th2 || e21->chi2() > th2) {
			size_t idx = vnIndexEdge[i];
			vpMatches1[idx] = static_cast<MapPoint *>(NULL);
			optimizer.removeEdge(e12);
			optimizer.removeEdge(e21);
			vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(NULL);
			vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(NULL);
			nBad++;
		}
	}

	int nMoreIterations;
	if (nBad > 0) {
		nMoreIterations = 10;
	}
	else {
		nMoreIterations = 5;
	}

	if (nCorrespondences - nBad < 10) {
		return 0;
	}

	// Optimize again only with inliers

	optimizer.initializeOptimization();
	optimizer.optimize(nMoreIterations);

	int nIn = 0;
	for (size_t i = 0; i < vpEdges12.size(); i++) {
		g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
		g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
		if (!e12 || !e21) {
			continue;
		}

		if (e12->chi2() > th2 || e21->chi2() > th2) {
			size_t idx = vnIndexEdge[i];
			vpMatches1[idx] = static_cast<MapPoint *>(NULL);
		}
		else {
			nIn++;
		}
	}

	// Recover optimized Sim3
	g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
	g2oS12 = vSim3_recov->estimate();

	return nIn;
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1,
                            KeyFrame *pKF2,
                            vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12,
                            const float th2,
                            const bool bFixScale,
                            Eigen::Matrix<double, 7, 7> &mAcumHessian,
                            const bool bAllPoints)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// Camera poses
	const cv::Mat R1w = pKF1->GetRotation();
	const cv::Mat t1w = pKF1->GetTranslation();
	const cv::Mat R2w = pKF2->GetRotation();
	const cv::Mat t2w = pKF2->GetTranslation();

	// Set Sim3 vertex
	ORB_SLAM3::VertexSim3Expmap *vSim3 = new ORB_SLAM3::VertexSim3Expmap();
	vSim3->_fix_scale = bFixScale;
	vSim3->setEstimate(g2oS12);
	vSim3->setId(0);
	vSim3->setFixed(false);
	vSim3->pCamera1 = pKF1->mpCamera;
	vSim3->pCamera2 = pKF2->mpCamera;
	optimizer.addVertex(vSim3);

	// Set MapPoint vertices
	const int N = vpMatches1.size();
	const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
	vector<ORB_SLAM3::EdgeSim3ProjectXYZ *> vpEdges12;
	vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ *> vpEdges21;
	vector<size_t> vnIndexEdge;
	vector<bool> vbIsInKF2;

	vnIndexEdge.reserve(2 * N);
	vpEdges12.reserve(2 * N);
	vpEdges21.reserve(2 * N);
	vbIsInKF2.reserve(2 * N);

	const float deltaHuber = sqrt(th2);

	int nCorrespondences = 0;
	int nBadMPs = 0;
	int nInKF2 = 0;
	int nOutKF2 = 0;
	int nMatchWithoutMP = 0;

	vector<int> vIdsOnlyInKF2;

	for (int i = 0; i < N; i++) {
		if (!vpMatches1[i]) {
			continue;
		}

		MapPoint *pMP1 = vpMapPoints1[i];
		MapPoint *pMP2 = vpMatches1[i];

		const int id1 = 2 * i + 1;
		const int id2 = 2 * (i + 1);

		const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKF2));
		/*if(i2 < 0)
            cout << "Sim3-OPT: Error, there is a matched which is not find it" << endl;*/

		cv::Mat P3D1c;
		cv::Mat P3D2c;

		if (pMP1 && pMP2) {
			//if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
			if (!pMP1->isBad() && !pMP2->isBad()) {
				g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
				cv::Mat P3D1w = pMP1->GetWorldPos();
				P3D1c = R1w * P3D1w + t1w;
				vPoint1->setEstimate(Converter::toVector3d(P3D1c));
				vPoint1->setId(id1);
				vPoint1->setFixed(true);
				optimizer.addVertex(vPoint1);

				g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
				cv::Mat P3D2w = pMP2->GetWorldPos();
				P3D2c = R2w * P3D2w + t2w;
				vPoint2->setEstimate(Converter::toVector3d(P3D2c));
				vPoint2->setId(id2);
				vPoint2->setFixed(true);
				optimizer.addVertex(vPoint2);
			}
			else {
				nBadMPs++;
				continue;
			}
		}
		else {
			nMatchWithoutMP++;

			//TODO The 3D position in KF1 doesn't exist
			if (!pMP2->isBad()) {
				g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
				cv::Mat P3D2w = pMP2->GetWorldPos();
				P3D2c = R2w * P3D2w + t2w;
				vPoint2->setEstimate(Converter::toVector3d(P3D2c));
				vPoint2->setId(id2);
				vPoint2->setFixed(true);
				optimizer.addVertex(vPoint2);

				vIdsOnlyInKF2.push_back(id2);
			}
			continue;
		}

		if (i2 < 0 && !bAllPoints) {
			Verbose::PrintMess("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints),
			                   Verbose::VERBOSITY_DEBUG);
			continue;
		}

		if (P3D2c.at<float>(2) < 0) {
			Verbose::PrintMess("Sim3: Z coordinate is negative", Verbose::VERBOSITY_DEBUG);
			continue;
		}

		nCorrespondences++;

		// Set edge x1 = S12*X2
		Eigen::Matrix<double, 2, 1> obs1;
		const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
		obs1 << kpUn1.pt.x, kpUn1.pt.y;

		ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ();

		e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
		e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
		e12->setMeasurement(obs1);
		const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
		e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

		g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
		e12->setRobustKernel(rk1);
		rk1->setDelta(deltaHuber);
		optimizer.addEdge(e12);

		// Set edge x2 = S21*X1
		Eigen::Matrix<double, 2, 1> obs2;
		cv::KeyPoint kpUn2;
		bool inKF2;
		if (i2 >= 0) {
			kpUn2 = pKF2->mvKeysUn[i2];
			obs2 << kpUn2.pt.x, kpUn2.pt.y;
			inKF2 = true;

			nInKF2++;
		}
		else {
			float invz = 1 / P3D2c.at<float>(2);
			float x = P3D2c.at<float>(0) * invz;
			float y = P3D2c.at<float>(1) * invz;

			obs2 << x, y;
			kpUn2 = cv::KeyPoint(cv::Point2f(x, y), pMP2->mnTrackScaleLevel);

			inKF2 = false;
			nOutKF2++;
		}

		ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = new ORB_SLAM3::EdgeInverseSim3ProjectXYZ();

		e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
		e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
		e21->setMeasurement(obs2);
		float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
		e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

		g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
		e21->setRobustKernel(rk2);
		rk2->setDelta(deltaHuber);
		optimizer.addEdge(e21);

		vpEdges12.push_back(e12);
		vpEdges21.push_back(e21);
		vnIndexEdge.push_back(i);

		vbIsInKF2.push_back(inKF2);
	}

	Verbose::PrintMess(
		"Sim3: There are " + to_string(nCorrespondences) + " matches, " + to_string(nInKF2) + " are in the KF and "
			+ to_string(nOutKF2) + " are in the connected KFs. There are " + to_string(nMatchWithoutMP)
			+ " matches which have not an associate MP", Verbose::VERBOSITY_DEBUG);

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(5);

	// Check inliers
	int nBad = 0;
	int nBadOutKF2 = 0;
	for (size_t i = 0; i < vpEdges12.size(); i++) {
		ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
		ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
		if (!e12 || !e21) {
			continue;
		}

		if (e12->chi2() > th2 || e21->chi2() > th2) {
			size_t idx = vnIndexEdge[i];
			vpMatches1[idx] = static_cast<MapPoint *>(NULL);
			optimizer.removeEdge(e12);
			optimizer.removeEdge(e21);
			vpEdges12[i] = static_cast<ORB_SLAM3::EdgeSim3ProjectXYZ *>(NULL);
			vpEdges21[i] = static_cast<ORB_SLAM3::EdgeInverseSim3ProjectXYZ *>(NULL);
			nBad++;

			if (!vbIsInKF2[i]) {
				nBadOutKF2++;
			}
			continue;
		}

		//Check if remove the robust adjustment improve the result
		e12->setRobustKernel(0);
		e21->setRobustKernel(0);
	}

	Verbose::PrintMess(
		"Sim3: First Opt -> Correspondences: " + to_string(nCorrespondences) + "; nBad: " + to_string(nBad)
			+ "; nBadOutKF2: " + to_string(nBadOutKF2), Verbose::VERBOSITY_DEBUG);

	int nMoreIterations;
	if (nBad > 0) {
		nMoreIterations = 10;
	}
	else {
		nMoreIterations = 5;
	}

	if (nCorrespondences - nBad < 10) {
		return 0;
	}

	// Optimize again only with inliers

	optimizer.initializeOptimization();
	optimizer.optimize(nMoreIterations);

	int nIn = 0;
	mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
	for (size_t i = 0; i < vpEdges12.size(); i++) {
		ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
		ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
		if (!e12 || !e21) {
			continue;
		}

		e12->computeError();
		e21->computeError();

		if (e12->chi2() > th2 || e21->chi2() > th2) {
			size_t idx = vnIndexEdge[i];
			vpMatches1[idx] = static_cast<MapPoint *>(NULL);
		}
		else {
			nIn++;
			//mAcumHessian += e12->GetHessian();
		}
	}

	// Recover optimized Sim3
	//Verbose::PrintMess("Sim3: Initial seed " + g2oS12, Verbose::VERBOSITY_DEBUG);
	g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
	g2oS12 = vSim3_recov->estimate();
	//Verbose::PrintMess("Sim3: Optimized solution " + g2oS12, Verbose::VERBOSITY_DEBUG);

	return nIn;
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1,
                            KeyFrame *pKF2,
                            vector<MapPoint *> &vpMatches1,
                            vector<KeyFrame *> &vpMatches1KF,
                            g2o::Sim3 &g2oS12,
                            const float th2,
                            const bool bFixScale,
                            Eigen::Matrix<double, 7, 7> &mAcumHessian,
                            const bool bAllPoints)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// Calibration
	const cv::Mat &K1 = pKF1->mK;
	const cv::Mat &K2 = pKF2->mK;

	// Camera poses
	const cv::Mat R1w = pKF1->GetRotation();
	const cv::Mat t1w = pKF1->GetTranslation();
	Verbose::PrintMess("Extracted rotation and traslation from the first KF ", Verbose::VERBOSITY_DEBUG);
	const cv::Mat R2w = pKF2->GetRotation();
	const cv::Mat t2w = pKF2->GetTranslation();
	Verbose::PrintMess("Extracted rotation and traslation from the second KF ", Verbose::VERBOSITY_DEBUG);

	// Set Sim3 vertex
	g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
	vSim3->_fix_scale = bFixScale;
	vSim3->setEstimate(g2oS12);
	vSim3->setId(0);
	vSim3->setFixed(false);
	vSim3->_principle_point1[0] = K1.at<float>(0, 2);
	vSim3->_principle_point1[1] = K1.at<float>(1, 2);
	vSim3->_focal_length1[0] = K1.at<float>(0, 0);
	vSim3->_focal_length1[1] = K1.at<float>(1, 1);
	vSim3->_principle_point2[0] = K2.at<float>(0, 2);
	vSim3->_principle_point2[1] = K2.at<float>(1, 2);
	vSim3->_focal_length2[0] = K2.at<float>(0, 0);
	vSim3->_focal_length2[1] = K2.at<float>(1, 1);
	optimizer.addVertex(vSim3);

	// Set MapPoint vertices
	const int N = vpMatches1.size();
	const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
	vector<ORB_SLAM3::EdgeSim3ProjectXYZ *> vpEdges12;
	vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ *> vpEdges21;
	vector<size_t> vnIndexEdge;

	vnIndexEdge.reserve(2 * N);
	vpEdges12.reserve(2 * N);
	vpEdges21.reserve(2 * N);

	const float deltaHuber = sqrt(th2);

	int nCorrespondences = 0;

	KeyFrame *pKFm = pKF2;
	for (int i = 0; i < N; i++) {
		if (!vpMatches1[i]) {
			continue;
		}

		MapPoint *pMP1 = vpMapPoints1[i];
		MapPoint *pMP2 = vpMatches1[i];

		const int id1 = 2 * i + 1;
		const int id2 = 2 * (i + 1);

		pKFm = vpMatches1KF[i];
		const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKFm));
		if (i2 < 0) {
			Verbose::PrintMess("Sim3-OPT: Error, there is a matched which is not find it", Verbose::VERBOSITY_DEBUG);
		}

		cv::Mat P3D2c;

		if (pMP1 && pMP2) {
			//if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
			if (!pMP1->isBad() && !pMP2->isBad()) {
				g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
				cv::Mat P3D1w = pMP1->GetWorldPos();
				cv::Mat P3D1c = R1w * P3D1w + t1w;
				vPoint1->setEstimate(Converter::toVector3d(P3D1c));
				vPoint1->setId(id1);
				vPoint1->setFixed(true);
				optimizer.addVertex(vPoint1);

				g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
				cv::Mat P3D2w = pMP2->GetWorldPos();
				P3D2c = R2w * P3D2w + t2w;
				vPoint2->setEstimate(Converter::toVector3d(P3D2c));
				vPoint2->setId(id2);
				vPoint2->setFixed(true);
				optimizer.addVertex(vPoint2);
			}
			else {
				continue;
			}
		}
		else {
			continue;
		}

		if (i2 < 0 && !bAllPoints) {
			Verbose::PrintMess("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints),
			                   Verbose::VERBOSITY_DEBUG);
			continue;
		}

		nCorrespondences++;

		// Set edge x1 = S12*X2
		Eigen::Matrix<double, 2, 1> obs1;
		const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
		obs1 << kpUn1.pt.x, kpUn1.pt.y;

		ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ();
		e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
		e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
		e12->setMeasurement(obs1);
		const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
		e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

		g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
		e12->setRobustKernel(rk1);
		rk1->setDelta(deltaHuber);
		optimizer.addEdge(e12);

		// Set edge x2 = S21*X1
		Eigen::Matrix<double, 2, 1> obs2;
		cv::KeyPoint kpUn2;
		if (i2 >= 0 && pKFm == pKF2) {
			kpUn2 = pKFm->mvKeysUn[i2];
			obs2 << kpUn2.pt.x, kpUn2.pt.y;
		}
		else {
			float invz = 1 / P3D2c.at<float>(2);
			float x = P3D2c.at<float>(0) * invz;
			float y = P3D2c.at<float>(1) * invz;

			// Project in image and check it is not outside
			float u = pKF2->fx * x + pKFm->cx;
			float v = pKF2->fy * y + pKFm->cy;
			obs2 << u, v;
			kpUn2 = cv::KeyPoint(cv::Point2f(u, v), pMP2->mnTrackScaleLevel);
		}

		ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = new ORB_SLAM3::EdgeInverseSim3ProjectXYZ();

		e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
		e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
		e21->setMeasurement(obs2);
		float invSigmaSquare2 = pKFm->mvInvLevelSigma2[kpUn2.octave];
		e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

		g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
		e21->setRobustKernel(rk2);
		rk2->setDelta(deltaHuber);
		optimizer.addEdge(e21);

		vpEdges12.push_back(e12);
		vpEdges21.push_back(e21);
		vnIndexEdge.push_back(i);
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(5);

	// Check inliers
	int nBad = 0;
	for (size_t i = 0; i < vpEdges12.size(); i++) {
		ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
		ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
		if (!e12 || !e21) {
			continue;
		}

		if (e12->chi2() > th2 || e21->chi2() > th2) {
			size_t idx = vnIndexEdge[i];
			vpMatches1[idx] = static_cast<MapPoint *>(NULL);
			optimizer.removeEdge(e12);
			optimizer.removeEdge(e21);
			vpEdges12[i] = static_cast<ORB_SLAM3::EdgeSim3ProjectXYZ *>(NULL);
			vpEdges21[i] = static_cast<ORB_SLAM3::EdgeInverseSim3ProjectXYZ *>(NULL);
			nBad++;
			continue;
		}

		//Check if remove the robust adjustment improve the result
		e12->setRobustKernel(0);
		e21->setRobustKernel(0);
	}

	//cout << "Sim3 -> Correspondences: " << nCorrespondences << "; nBad: " << nBad << endl;

	int nMoreIterations;
	if (nBad > 0) {
		nMoreIterations = 10;
	}
	else {
		nMoreIterations = 5;
	}

	if (nCorrespondences - nBad < 10) {
		return 0;
	}

	// Optimize again only with inliers

	optimizer.initializeOptimization();
	optimizer.optimize(nMoreIterations);

	int nIn = 0;
	mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
	for (size_t i = 0; i < vpEdges12.size(); i++) {
		ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
		ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
		if (!e12 || !e21) {
			continue;
		}

		e12->computeError();
		e21->computeError();

		if (e12->chi2() > th2 || e21->chi2() > th2) {
			size_t idx = vnIndexEdge[i];
			vpMatches1[idx] = static_cast<MapPoint *>(NULL);
		}
		else {
			nIn++;
			//mAcumHessian += e12->GetHessian();
		}
	}

	// Recover optimized Sim3
	ORB_SLAM3::VertexSim3Expmap *vSim3_recov = static_cast<ORB_SLAM3::VertexSim3Expmap *>(optimizer.vertex(0));
	g2oS12 = vSim3_recov->estimate();

	return nIn;
}

void Optimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, bool bLarge, bool bRecInit)
{
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	Map *pCurrentMap = pKF->GetMap();

	int maxOpt = 10;
	int opt_it = 10;
	if (bLarge) {
		maxOpt = 25;
		opt_it = 4;
	}
	const int Nd = std::min((int)pCurrentMap->KeyFramesInMap() - 2, maxOpt);
	const unsigned long maxKFid = pKF->mnId;

	vector<KeyFrame *> vpOptimizableKFs;
	const vector<KeyFrame *> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
	list<KeyFrame *> lpOptVisKFs;

	vpOptimizableKFs.reserve(Nd);
	vpOptimizableKFs.push_back(pKF);
	pKF->mnBALocalForKF = pKF->mnId;
	for (int i = 1; i < Nd; i++) {
		if (vpOptimizableKFs.back()->mPrevKF) {
			vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
			vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
		}
		else {
			break;
		}
	}

	int N = vpOptimizableKFs.size();

	// Optimizable points seen by temporal optimizable keyframes
	list<MapPoint *> lLocalMapPoints;
	for (int i = 0; i < N; i++) {
		vector<MapPoint *> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
		for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
			MapPoint *pMP = *vit;
			if (pMP) {
				if (!pMP->isBad()) {
					if (pMP->mnBALocalForKF != pKF->mnId) {
						lLocalMapPoints.push_back(pMP);
						pMP->mnBALocalForKF = pKF->mnId;
					}
				}
			}
		}
	}

	// Fixed Keyframe: First frame previous KF to optimization window)
	list<KeyFrame *> lFixedKeyFrames;
	if (vpOptimizableKFs.back()->mPrevKF) {
		lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
		vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF = pKF->mnId;
	}
	else {
		vpOptimizableKFs.back()->mnBALocalForKF = 0;
		vpOptimizableKFs.back()->mnBAFixedForKF = pKF->mnId;
		lFixedKeyFrames.push_back(vpOptimizableKFs.back());
		vpOptimizableKFs.pop_back();
	}

	// Optimizable visual KFs
	const int maxCovKF = 0;
	for (int i = 0, iend = vpNeighsKFs.size(); i < iend; i++) {
		if (lpOptVisKFs.size() >= maxCovKF) {
			break;
		}

		KeyFrame *pKFi = vpNeighsKFs[i];
		if (pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId) {
			continue;
		}
		pKFi->mnBALocalForKF = pKF->mnId;
		if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
			lpOptVisKFs.push_back(pKFi);

			vector<MapPoint *> vpMPs = pKFi->GetMapPointMatches();
			for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
				MapPoint *pMP = *vit;
				if (pMP) {
					if (!pMP->isBad()) {
						if (pMP->mnBALocalForKF != pKF->mnId) {
							lLocalMapPoints.push_back(pMP);
							pMP->mnBALocalForKF = pKF->mnId;
						}
					}
				}
			}
		}
	}

	// Fixed KFs which are not covisible optimizable
	const int maxFixKF = 200;

	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		map<KeyFrame *, tuple<int, int>> observations = (*lit)->GetObservations();
		for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
				pKFi->mnBAFixedForKF = pKF->mnId;
				if (!pKFi->isBad()) {
					lFixedKeyFrames.push_back(pKFi);
					break;
				}
			}
		}
		if (lFixedKeyFrames.size() >= maxFixKF) {
			break;
		}
	}

	bool bNonFixed = (lFixedKeyFrames.size() == 0);

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;
	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	if (bLarge) {
		g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
		optimizer.setAlgorithm(solver);
	}
	else {
		g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		solver->setUserLambdaInit(1e0);
		optimizer.setAlgorithm(solver);
	}

	// Set Local temporal KeyFrame vertices
	N = vpOptimizableKFs.size();
	for (int i = 0; i < N; i++) {
		KeyFrame *pKFi = vpOptimizableKFs[i];

		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(false);
		optimizer.addVertex(VP);

		if (pKFi->bImu) {
			VertexVelocity *VV = new VertexVelocity(pKFi);
			VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
			VV->setFixed(false);
			optimizer.addVertex(VV);
			VertexGyroBias *VG = new VertexGyroBias(pKFi);
			VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
			VG->setFixed(false);
			optimizer.addVertex(VG);
			VertexAccBias *VA = new VertexAccBias(pKFi);
			VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
			VA->setFixed(false);
			optimizer.addVertex(VA);
		}
	}

	// Set Local visual KeyFrame vertices
	for (list<KeyFrame *>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++) {
		KeyFrame *pKFi = *it;
		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(false);
		optimizer.addVertex(VP);
	}

	// Set Fixed KeyFrame vertices
	for (list<KeyFrame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		if (pKFi->bImu) // This should be done only for keyframe just before temporal window
		{
			VertexVelocity *VV = new VertexVelocity(pKFi);
			VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
			VV->setFixed(true);
			optimizer.addVertex(VV);
			VertexGyroBias *VG = new VertexGyroBias(pKFi);
			VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
			VG->setFixed(true);
			optimizer.addVertex(VG);
			VertexAccBias *VA = new VertexAccBias(pKFi);
			VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
			VA->setFixed(true);
			optimizer.addVertex(VA);
		}
	}

	// Create intertial constraints
	vector<EdgeInertial *> vei(N, (EdgeInertial *)NULL);
	vector<EdgeGyroRW *> vegr(N, (EdgeGyroRW *)NULL);//bias
	vector<EdgeAccRW *> vear(N, (EdgeAccRW *)NULL);//bias

	for (int i = 0; i < N; i++) {
		KeyFrame *pKFi = vpOptimizableKFs[i];

		if (!pKFi->mPrevKF) {
			cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
			continue;
		}
		if (pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated) {
			pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
			g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1);
			g2o::HyperGraph::Vertex *VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2);
			g2o::HyperGraph::Vertex *VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3);
			g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
			g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2);
			g2o::HyperGraph::Vertex *VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3);

			if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2) {
				cerr << "Error " << VP1 << ", " << VV1 << ", " << VG1 << ", " << VA1 << ", " << VP2 << ", " << VV2
				     << ", " << VG2 << ", " << VA2 << endl;
				continue;
			}

			vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

			vei[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			vei[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
			vei[i]->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG1));
			vei[i]->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA1));
			vei[i]->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			vei[i]->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));

			if (i == N - 1 || bRecInit) {
				// All inertial residuals are included without robust cost function, but not that one linking the
				// last optimizable keyframe inside of the local window and the first fixed keyframe out. The
				// information matrix for this measurement is also downweighted. This is done to avoid accumulating
				// error due to fixing variables.
				g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
				vei[i]->setRobustKernel(rki);
				if (i == N - 1) {
					vei[i]->setInformation(vei[i]->information() * 1e-2);
				}
				rki->setDelta(sqrt(16.92));
			}
			optimizer.addEdge(vei[i]);

			vegr[i] = new EdgeGyroRW();
			vegr[i]->setVertex(0, VG1);
			vegr[i]->setVertex(1, VG2);
			cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
			Eigen::Matrix3d InfoG;

			for (int r = 0; r < 3; r++)
				for (int c = 0; c < 3; c++)
					InfoG(r, c) = cvInfoG.at<float>(r, c);
			vegr[i]->setInformation(InfoG);
			optimizer.addEdge(vegr[i]);

			// cout << "b";
			vear[i] = new EdgeAccRW();
			vear[i]->setVertex(0, VA1);
			vear[i]->setVertex(1, VA2);
			cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
			Eigen::Matrix3d InfoA;
			for (int r = 0; r < 3; r++)
				for (int c = 0; c < 3; c++)
					InfoA(r, c) = cvInfoA.at<float>(r, c);
			vear[i]->setInformation(InfoA);

			optimizer.addEdge(vear[i]);
		}
		else {
			cout << "ERROR building inertial edge" << endl;
		}
	}

	// Set MapPoint vertices
	const int nExpectedSize = (N + lFixedKeyFrames.size()) * lLocalMapPoints.size();

	// Mono
	vector<EdgeMono *> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	// Stereo
	vector<EdgeStereo *> vpEdgesStereo;
	vpEdgesStereo.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFStereo;
	vpEdgeKFStereo.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeStereo;
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float chi2Mono2 = 5.991;
	const float thHuberStereo = sqrt(7.815);
	const float chi2Stereo2 = 7.815;

	const unsigned long iniMPid = maxKFid * 5;

	map<int, int> mVisEdges;
	for (int i = 0; i < N; i++) {
		KeyFrame *pKFi = vpOptimizableKFs[i];
		mVisEdges[pKFi->mnId] = 0;
	}
	for (list<KeyFrame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++) {
		mVisEdges[(*lit)->mnId] = 0;
	}

	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

		unsigned long id = pMP->mnId + iniMPid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);
		const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

		// Create visual constraints
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
				continue;
			}

			if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
				const int leftIndex = get<0>(mit->second);

				cv::KeyPoint kpUn;

				// Monocular left observation
				if (leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0) {
					mVisEdges[pKFi->mnId]++;

					kpUn = pKFi->mvKeysUn[leftIndex];
					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					EdgeMono *e = new EdgeMono(0);

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);

					// Add here uncerteinty
					const float unc2 = pKFi->mpCamera->uncertainty2(obs);

					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					optimizer.addEdge(e);
					vpEdgesMono.push_back(e);
					vpEdgeKFMono.push_back(pKFi);
					vpMapPointEdgeMono.push_back(pMP);
				}
					// Stereo-observation
				else if (leftIndex != -1) // Stereo observation
				{
					kpUn = pKFi->mvKeysUn[leftIndex];
					mVisEdges[pKFi->mnId]++;

					const float kp_ur = pKFi->mvuRight[leftIndex];
					Eigen::Matrix<double, 3, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

					EdgeStereo *e = new EdgeStereo(0);

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);

					// Add here uncerteinty
					const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));

					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
					e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberStereo);

					optimizer.addEdge(e);
					vpEdgesStereo.push_back(e);
					vpEdgeKFStereo.push_back(pKFi);
					vpMapPointEdgeStereo.push_back(pMP);
				}

				// Monocular right observation
				if (pKFi->mpCamera2) {
					int rightIndex = get<1>(mit->second);

					if (rightIndex != -1) {
						rightIndex -= pKFi->NLeft;
						mVisEdges[pKFi->mnId]++;

						Eigen::Matrix<double, 2, 1> obs;
						cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
						obs << kp.pt.x, kp.pt.y;

						EdgeMono *e = new EdgeMono(1);

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
						e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
						e->setMeasurement(obs);

						// Add here uncerteinty
						const float unc2 = pKFi->mpCamera->uncertainty2(obs);

						const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(thHuberMono);

						optimizer.addEdge(e);
						vpEdgesMono.push_back(e);
						vpEdgeKFMono.push_back(pKFi);
						vpMapPointEdgeMono.push_back(pMP);
					}
				}
			}
		}
	}

	//cout << "Total map points: " << lLocalMapPoints.size() << endl;
	for (map<int, int>::iterator mit = mVisEdges.begin(), mend = mVisEdges.end(); mit != mend; mit++) {
		assert(mit->second >= 3);
	}

	optimizer.initializeOptimization();
	optimizer.computeActiveErrors();
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	float err = optimizer.activeRobustChi2();
	optimizer.optimize(opt_it); // Originally to 2
	float err_end = optimizer.activeRobustChi2();
	if (pbStopFlag) {
		optimizer.setForceStopFlag(pbStopFlag);
	}

	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

	vector<pair<KeyFrame *, MapPoint *>> vToErase;
	vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

	// Check inlier observations
	// Mono
	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
		EdgeMono *e = vpEdgesMono[i];
		MapPoint *pMP = vpMapPointEdgeMono[i];
		bool bClose = pMP->mTrackDepth < 10.f;

		if (pMP->isBad()) {
			continue;
		}

		if ((e->chi2() > chi2Mono2 && !bClose) || (e->chi2() > 1.5f * chi2Mono2 && bClose) || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFMono[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	// Stereo
	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
		EdgeStereo *e = vpEdgesStereo[i];
		MapPoint *pMP = vpMapPointEdgeStereo[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > chi2Stereo2) {
			KeyFrame *pKFi = vpEdgeKFStereo[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	// Get Map Mutex and erase outliers
	unique_lock<shared_timed_mutex> lock(pMap->mMutexMapUpdate);

	// TODO: Some convergence problems have been detected here
	//cout << "err0 = " << err << endl;
	//cout << "err_end = " << err_end << endl;
	if ((2 * err < err_end || isnan(err) || isnan(err_end)) && !bLarge) //bGN)
	{
		cout << "FAIL LOCAL-INERTIAL BA!!!!" << endl;
		return;
	}

	if (!vToErase.empty()) {
		for (size_t i = 0; i < vToErase.size(); i++) {
			KeyFrame *pKFi = vToErase[i].first;
			MapPoint *pMPi = vToErase[i].second;
			pKFi->EraseMapPointMatch(pMPi);
			pMPi->EraseObservation(pKFi);
		}
	}

	// Display main statistcis of optimization
	Verbose::PrintMess("LIBA KFs: " + to_string(N), Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess("LIBA bNonFixed?: " + to_string(bNonFixed), Verbose::VERBOSITY_DEBUG);
	Verbose::PrintMess("LIBA KFs visual outliers: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);

	for (list<KeyFrame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
		(*lit)->mnBAFixedForKF = 0;

	// Recover optimized data
	// Local temporal Keyframes
	N = vpOptimizableKFs.size();
	for (int i = 0; i < N; i++) {
		KeyFrame *pKFi = vpOptimizableKFs[i];

		VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
		cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
		pKFi->SetPose(Tcw);
		pKFi->mnBALocalForKF = 0;

		if (pKFi->bImu) {
			VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
			pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
			VertexGyroBias *VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
			VertexAccBias *VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
			Vector6d b;
			b << VG->estimate(), VA->estimate();
			pKFi->SetNewBias(IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]));
		}
	}

	// Local visual KeyFrame
	for (list<KeyFrame *>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++) {
		KeyFrame *pKFi = *it;
		VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
		cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
		pKFi->SetPose(Tcw);
		pKFi->mnBALocalForKF = 0;
	}

	//Points
	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		g2o::VertexSBAPointXYZ
			*vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + iniMPid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
	}

	pMap->IncreaseChangeIndex();

	std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

	/*double t_const = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
    double t_opt = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    double t_rec = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
    /*std::cout << " Construction time: " << t_const << std::endl;
    std::cout << " Optimization time: " << t_opt << std::endl;
    std::cout << " Recovery time: " << t_rec << std::endl;
    std::cout << " Total time: " << t_const+t_opt+t_rec << std::endl;
    std::cout << " Optimization iterations: " << opt_it << std::endl;*/
}

Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
	// Goal
	// a  | ab | ac       a*  | 0 | ac*
	// ba | b  | bc  -->  0   | 0 | 0
	// ca | cb | c        ca* | 0 | c*

	// Size of block before block to marginalize
	const int a = start;
	// Size of block to marginalize
	const int b = end - start + 1;
	// Size of block after block to marginalize
	const int c = H.cols() - (end + 1);

	// Reorder as follows:
	// a  | ab | ac       a  | ac | ab
	// ba | b  | bc  -->  ca | c  | cb
	// ca | cb | c        ba | bc | b

	Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(), H.cols());
	if (a > 0) {
		Hn.block(0, 0, a, a) = H.block(0, 0, a, a);
		Hn.block(0, a + c, a, b) = H.block(0, a, a, b);
		Hn.block(a + c, 0, b, a) = H.block(a, 0, b, a);
	}
	if (a > 0 && c > 0) {
		Hn.block(0, a, a, c) = H.block(0, a + b, a, c);
		Hn.block(a, 0, c, a) = H.block(a + b, 0, c, a);
	}
	if (c > 0) {
		Hn.block(a, a, c, c) = H.block(a + b, a + b, c, c);
		Hn.block(a, a + c, c, b) = H.block(a + b, a, c, b);
		Hn.block(a + c, a, b, c) = H.block(a, a + b, b, c);
	}
	Hn.block(a + c, a + c, b, b) = H.block(a, a, b, b);

	// Perform marginalization (Schur complement)
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a + c, a + c, b, b), Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv = svd.singularValues();
	for (int i = 0; i < b; ++i) {
		if (singularValues_inv(i) > 1e-6) {
			singularValues_inv(i) = 1.0 / singularValues_inv(i);
		}
		else {
			singularValues_inv(i) = 0;
		}
	}
	Eigen::MatrixXd invHb = svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose();
	Hn.block(0, 0, a + c, a + c) =
		Hn.block(0, 0, a + c, a + c) - Hn.block(0, a + c, a + c, b) * invHb * Hn.block(a + c, 0, b, a + c);
	Hn.block(a + c, a + c, b, b) = Eigen::MatrixXd::Zero(b, b);
	Hn.block(0, a + c, a + c, b) = Eigen::MatrixXd::Zero(a + c, b);
	Hn.block(a + c, 0, b, a + c) = Eigen::MatrixXd::Zero(b, a + c);

	// Inverse reorder
	// a*  | ac* | 0       a*  | 0 | ac*
	// ca* | c*  | 0  -->  0   | 0 | 0
	// 0   | 0   | 0       ca* | 0 | c*
	Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(), H.cols());
	if (a > 0) {
		res.block(0, 0, a, a) = Hn.block(0, 0, a, a);
		res.block(0, a, a, b) = Hn.block(0, a + c, a, b);
		res.block(a, 0, b, a) = Hn.block(a + c, 0, b, a);
	}
	if (a > 0 && c > 0) {
		res.block(0, a + b, a, c) = Hn.block(0, a, a, c);
		res.block(a + b, 0, c, a) = Hn.block(a, 0, c, a);
	}
	if (c > 0) {
		res.block(a + b, a + b, c, c) = Hn.block(a, a, c, c);
		res.block(a + b, a, c, b) = Hn.block(a, a + c, c, b);
		res.block(a, a + b, b, c) = Hn.block(a + c, a, b, c);
	}

	res.block(a, a, b, b) = Hn.block(a + c, a + c, b, b);

	return res;
}

Eigen::MatrixXd Optimizer::Condition(const Eigen::MatrixXd &H, const int &start, const int &end)
{
	// Size of block before block to condition
	const int a = start;
	// Size of block to condition
	const int b = end + 1 - start;

	// Set to zero elements related to block b(start:end,start:end)
	// a  | ab | ac       a  | 0 | ac
	// ba | b  | bc  -->  0  | 0 | 0
	// ca | cb | c        ca | 0 | c

	Eigen::MatrixXd Hn = H;

	Hn.block(a, 0, b, H.cols()) = Eigen::MatrixXd::Zero(b, H.cols());
	Hn.block(0, a, H.rows(), b) = Eigen::MatrixXd::Zero(H.rows(), b);

	return Hn;
}

Eigen::MatrixXd Optimizer::Sparsify(const Eigen::MatrixXd &H,
                                    const int &start1,
                                    const int &end1,
                                    const int &start2,
                                    const int &end2)
{
	// Goal: remove link between a and b
	// p(a,b,c) ~ p(a,b,c)*p(a|c)/p(a|b,c) => H' = H + H1 - H2
	// H1: marginalize b and condition c
	// H2: condition b and c
	Eigen::MatrixXd Hac = Marginalize(H, start2, end2);
	Eigen::MatrixXd Hbc = Marginalize(H, start1, end1);
	Eigen::MatrixXd Hc = Marginalize(Hac, start1, end1);

	return Hac + Hbc - Hc;
}

void Optimizer::InertialOptimization(Map *pMap,
                                     Eigen::Matrix3d &Rwg,
                                     double &scale,
                                     Eigen::Vector3d &bg,
                                     Eigen::Vector3d &ba,
                                     bool bMono,
                                     Eigen::MatrixXd &covInertial,
                                     bool bFixedVel,
                                     bool bGauss,
                                     float priorG,
                                     float priorA)
{
	Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	if (priorG != 0.f) {
		solver->setUserLambdaInit(1e3);
	}

	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		VertexVelocity *VV = new VertexVelocity(pKFi);
		VV->setId(maxKFid + (pKFi->mnId) + 1);
		if (bFixedVel) {
			VV->setFixed(true);
		}
		else {
			VV->setFixed(false);
		}

		optimizer.addVertex(VV);
	}

	// Biases
	VertexGyroBias *VG = new VertexGyroBias(vpKFs.front());
	VG->setId(maxKFid * 2 + 2);
	if (bFixedVel) {
		VG->setFixed(true);
	}
	else {
		VG->setFixed(false);
	}
	optimizer.addVertex(VG);
	VertexAccBias *VA = new VertexAccBias(vpKFs.front());
	VA->setId(maxKFid * 2 + 3);
	if (bFixedVel) {
		VA->setFixed(true);
	}
	else {
		VA->setFixed(false);
	}

	optimizer.addVertex(VA);
	// prior acc bias
	EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F));
	epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
	double infoPriorA = priorA;
	epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
	optimizer.addEdge(epa);
	EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
	epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	double infoPriorG = priorG;
	epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
	optimizer.addEdge(epg);

	// Gravity and scale
	VertexGDir *VGDir = new VertexGDir(Rwg);
	VGDir->setId(maxKFid * 2 + 4);
	VGDir->setFixed(false);
	optimizer.addVertex(VGDir);
	VertexScale *VS = new VertexScale(scale);
	VS->setId(maxKFid * 2 + 5);
	VS->setFixed(!bMono); // Fixed for stereo case
	optimizer.addVertex(VS);

	// Graph edges
	// IMU links with gravity and scale
	vector<EdgeInertialGS *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	std::cout << "build optimization graph" << std::endl;

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpImuPreintegrated) {
				std::cout << "Not preintegrated measurement" << std::endl;
				continue;
			}

			pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
			g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
			g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid * 2 + 2);
			g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
			g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
			g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
			if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS) {
				cout << "Error" << VP1 << ", " << VV1 << ", " << VG << ", " << VA << ", " << VP2 << ", " << VV2 << ", "
				     << VGDir << ", " << VS << endl;

				continue;
			}
			EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
			ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
			ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

			vpei.push_back(ei);

			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}

	// Compute error for different scales
	std::set<g2o::HyperGraph::Edge *> setEdges = optimizer.edges();

	std::cout << "start optimization" << std::endl;
	optimizer.setVerbose(false);
	optimizer.initializeOptimization();
	optimizer.optimize(its);

	std::cout << "end optimization" << std::endl;

	scale = VS->estimate();

	// Recover optimized data
	// Biases
	VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid * 2 + 2));
	VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid * 2 + 3));
	Vector6d vb;
	vb << VG->estimate(), VA->estimate();
	bg << VG->estimate();
	ba << VA->estimate();
	scale = VS->estimate();

	IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);
	Rwg = VGDir->estimate().Rwg;

	cv::Mat cvbg = Converter::toCvMat(bg);

	//Keyframes velocities and biases
	std::cout << "update Keyframes velocities and biases" << std::endl;

	const int N = vpKFs.size();
		for (size_t i = 0; i < N; i++) {
			KeyFrame *pKFi = vpKFs[i];
			if (pKFi->mnId > maxKFid) {
				continue;
			}

			VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
			Eigen::Vector3d Vw = VV->estimate(); // Velocity is scaled after
			pKFi->SetVelocity(Converter::toCvMat(Vw));

			if (cv::norm(pKFi->GetGyroBias() - cvbg) > 0.01) {
				pKFi->SetNewBias(b);
				if (pKFi->mpImuPreintegrated) {
					pKFi->mpImuPreintegrated->Reintegrate();
				}
			}
			else {
				pKFi->SetNewBias(b);
			}
		}
}

void Optimizer::InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
{
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	solver->setUserLambdaInit(1e3);

	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		VertexVelocity *VV = new VertexVelocity(pKFi);
		VV->setId(maxKFid + (pKFi->mnId) + 1);
		VV->setFixed(false);

		optimizer.addVertex(VV);
	}

	// Biases
	VertexGyroBias *VG = new VertexGyroBias(vpKFs.front());
	VG->setId(maxKFid * 2 + 2);
	VG->setFixed(false);
	optimizer.addVertex(VG);

	VertexAccBias *VA = new VertexAccBias(vpKFs.front());
	VA->setId(maxKFid * 2 + 3);
	VA->setFixed(false);

	optimizer.addVertex(VA);
	// prior acc bias
	EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F));
	epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
	double infoPriorA = priorA;
	epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
	optimizer.addEdge(epa);
	EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
	epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	double infoPriorG = priorG;
	epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
	optimizer.addEdge(epg);

	// Gravity and scale
	VertexGDir *VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
	VGDir->setId(maxKFid * 2 + 4);
	VGDir->setFixed(true);
	optimizer.addVertex(VGDir);
	VertexScale *VS = new VertexScale(1.0);
	VS->setId(maxKFid * 2 + 5);
	VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
	optimizer.addVertex(VS);

	// Graph edges
	// IMU links with gravity and scale
	vector<EdgeInertialGS *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpImuPreintegrated) {
				continue;
			}

			pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
			g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
			g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid * 2 + 2);
			g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
			g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
			g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
			if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS) {
				cout << "Error" << VP1 << ", " << VV1 << ", " << VG << ", " << VA << ", " << VP2 << ", " << VV2 << ", "
				     << VGDir << ", " << VS << endl;

				continue;
			}
			EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
			ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
			ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

			vpei.push_back(ei);

			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}

	// Compute error for different scales
	optimizer.setVerbose(false);
	optimizer.initializeOptimization();
	optimizer.optimize(its);

	// Recover optimized data
	// Biases
	VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid * 2 + 2));
	VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid * 2 + 3));
	Vector6d vb;
	vb << VG->estimate(), VA->estimate();
	bg << VG->estimate();
	ba << VA->estimate();

	IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);

	cv::Mat cvbg = Converter::toCvMat(bg);

	//Keyframes velocities and biases
	const int N = vpKFs.size();
	for (size_t i = 0; i < N; i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}

		VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
		Eigen::Vector3d Vw = VV->estimate();
		pKFi->SetVelocity(Converter::toCvMat(Vw));

		if (cv::norm(pKFi->GetGyroBias() - cvbg) > 0.01) {
			pKFi->SetNewBias(b);
			if (pKFi->mpImuPreintegrated) {
				pKFi->mpImuPreintegrated->Reintegrate();
			}
		}
		else {
			pKFi->SetNewBias(b);
		}
	}
}

void Optimizer::InertialOptimization(vector<KeyFrame *> vpKFs,
                                     Eigen::Vector3d &bg,
                                     Eigen::Vector3d &ba,
                                     float priorG,
                                     float priorA)
{
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = vpKFs[0]->GetMap()->GetMaxKFid();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	solver->setUserLambdaInit(1e3);

	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		//if(pKFi->mnId>maxKFid)
		//    continue;
		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		VertexVelocity *VV = new VertexVelocity(pKFi);
		VV->setId(maxKFid + (pKFi->mnId) + 1);
		VV->setFixed(false);

		optimizer.addVertex(VV);
	}

	// Biases
	VertexGyroBias *VG = new VertexGyroBias(vpKFs.front());
	VG->setId(maxKFid * 2 + 2);
	VG->setFixed(false);
	optimizer.addVertex(VG);

	VertexAccBias *VA = new VertexAccBias(vpKFs.front());
	VA->setId(maxKFid * 2 + 3);
	VA->setFixed(false);

	optimizer.addVertex(VA);
	// prior acc bias
	EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F));
	epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
	double infoPriorA = priorA;
	epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
	optimizer.addEdge(epa);
	EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
	epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	double infoPriorG = priorG;
	epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
	optimizer.addEdge(epg);

	// Gravity and scale
	VertexGDir *VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
	VGDir->setId(maxKFid * 2 + 4);
	VGDir->setFixed(true);
	optimizer.addVertex(VGDir);
	VertexScale *VS = new VertexScale(1.0);
	VS->setId(maxKFid * 2 + 5);
	VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
	optimizer.addVertex(VS);

	// Graph edges
	// IMU links with gravity and scale
	vector<EdgeInertialGS *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpImuPreintegrated) {
				continue;
			}

			pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
			g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
			g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid * 2 + 2);
			g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
			g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
			g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
			if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS) {
				cout << "Error" << VP1 << ", " << VV1 << ", " << VG << ", " << VA << ", " << VP2 << ", " << VV2 << ", "
				     << VGDir << ", " << VS << endl;

				continue;
			}
			EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
			ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
			ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

			vpei.push_back(ei);

			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}

	// Compute error for different scales
	optimizer.setVerbose(false);
	optimizer.initializeOptimization();
	optimizer.optimize(its);

	// Recover optimized data
	// Biases
	VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid * 2 + 2));
	VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid * 2 + 3));
	Vector6d vb;
	vb << VG->estimate(), VA->estimate();
	bg << VG->estimate();
	ba << VA->estimate();

	IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);

	cv::Mat cvbg = Converter::toCvMat(bg);

	//Keyframes velocities and biases
	const int N = vpKFs.size();
	for (size_t i = 0; i < N; i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}

		VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
		Eigen::Vector3d Vw = VV->estimate();
		pKFi->SetVelocity(Converter::toCvMat(Vw));

		if (cv::norm(pKFi->GetGyroBias() - cvbg) > 0.01) {
			pKFi->SetNewBias(b);
			if (pKFi->mpImuPreintegrated) {
				pKFi->mpImuPreintegrated->Reintegrate();
			}
		}
		else {
			pKFi->SetNewBias(b);
		}
	}
}

void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale)
{
	int its = 10;
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (all variables are fixed)
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		VertexVelocity *VV = new VertexVelocity(pKFi);
		VV->setId(maxKFid + 1 + (pKFi->mnId));
		VV->setFixed(true);
		optimizer.addVertex(VV);

		// Vertex of fixed biases
		VertexGyroBias *VG = new VertexGyroBias(vpKFs.front());
		VG->setId(2 * (maxKFid + 1) + (pKFi->mnId));
		VG->setFixed(true);
		optimizer.addVertex(VG);
		VertexAccBias *VA = new VertexAccBias(vpKFs.front());
		VA->setId(3 * (maxKFid + 1) + (pKFi->mnId));
		VA->setFixed(true);
		optimizer.addVertex(VA);
	}

	// Gravity and scale
	VertexGDir *VGDir = new VertexGDir(Rwg);
	VGDir->setId(4 * (maxKFid + 1));
	VGDir->setFixed(false);
	optimizer.addVertex(VGDir);
	VertexScale *VS = new VertexScale(scale);
	VS->setId(4 * (maxKFid + 1) + 1);
	VS->setFixed(false);
	optimizer.addVertex(VS);

	// Graph edges
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}

			g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VV1 = optimizer.vertex((maxKFid + 1) + pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
			g2o::HyperGraph::Vertex *VV2 = optimizer.vertex((maxKFid + 1) + pKFi->mnId);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(2 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VA = optimizer.vertex(3 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(4 * (maxKFid + 1));
			g2o::HyperGraph::Vertex *VS = optimizer.vertex(4 * (maxKFid + 1) + 1);
			if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS) {
				Verbose::PrintMess(
					"Error" + to_string(VP1->id()) + ", " + to_string(VV1->id()) + ", " + to_string(VG->id()) + ", "
						+ to_string(VA->id()) + ", " + to_string(VP2->id()) + ", " + to_string(VV2->id()) + ", "
						+ to_string(VGDir->id()) + ", " + to_string(VS->id()), Verbose::VERBOSITY_NORMAL);

				continue;
			}
			EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
			ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
			ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

			optimizer.addEdge(ei);
		}
	}

	// Compute error for different scales
	optimizer.setVerbose(false);
	optimizer.initializeOptimization();
	optimizer.optimize(its);

	// Recover optimized data
	scale = VS->estimate();
	Rwg = VGDir->estimate().Rwg;
}

void Optimizer::MergeBundleAdjustmentVisual(KeyFrame *pCurrentKF,
                                            vector<KeyFrame *> vpWeldingKFs,
                                            vector<KeyFrame *> vpFixedKFs,
                                            bool *pbStopFlag)
{
	vector<MapPoint *> vpMPs;

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	if (pbStopFlag) {
		optimizer.setForceStopFlag(pbStopFlag);
	}

	long unsigned int maxKFid = 0;
	set<KeyFrame *> spKeyFrameBA;

	// Set not fixed KeyFrame vertices
	for (KeyFrame *pKFi: vpWeldingKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		pKFi->mnBALocalForKF = pCurrentKF->mnId;

		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
		vSE3->setId(pKFi->mnId);
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);
		if (pKFi->mnId > maxKFid) {
			maxKFid = pKFi->mnId;
		}

		set<MapPoint *> spViewMPs = pKFi->GetMapPoints();
		for (MapPoint *pMPi: spViewMPs) {
			if (pMPi) {
				if (!pMPi->isBad()) {
					if (pMPi->mnBALocalForKF != pCurrentKF->mnId) {
						vpMPs.push_back(pMPi);
						pMPi->mnBALocalForKF = pCurrentKF->mnId;
					}
				}
			}
		}

		spKeyFrameBA.insert(pKFi);
	}

	// Set fixed KeyFrame vertices
	for (KeyFrame *pKFi: vpFixedKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		pKFi->mnBALocalForKF = pCurrentKF->mnId;

		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
		vSE3->setId(pKFi->mnId);
		vSE3->setFixed(true);
		optimizer.addVertex(vSE3);
		if (pKFi->mnId > maxKFid) {
			maxKFid = pKFi->mnId;
		}

		set<MapPoint *> spViewMPs = pKFi->GetMapPoints();
		for (MapPoint *pMPi: spViewMPs) {
			if (pMPi) {
				if (!pMPi->isBad()) {
					if (pMPi->mnBALocalForKF != pCurrentKF->mnId) {
						vpMPs.push_back(pMPi);
						pMPi->mnBALocalForKF = pCurrentKF->mnId;
					}
				}
			}
		}

		spKeyFrameBA.insert(pKFi);
	}

	const int nExpectedSize = (vpWeldingKFs.size() + vpFixedKFs.size()) * vpMPs.size();

	vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
	vpEdgesStereo.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFStereo;
	vpEdgeKFStereo.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeStereo;
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	const float thHuber2D = sqrt(5.99);
	const float thHuber3D = sqrt(7.815);

	// Set MapPoint vertices
	for (unsigned int i = 0; i < vpMPs.size(); ++i) {
		MapPoint *pMPi = vpMPs[i];
		if (pMPi->isBad()) {
			continue;
		}

		g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMPi->GetWorldPos()));
		const int id = pMPi->mnId + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const map<KeyFrame *, tuple<int, int>> observations = pMPi->GetObservations();
		int nEdges = 0;
		//SET EDGES
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end();
		     mit++) {
			//cout << "--KF view init" << endl;

			KeyFrame *pKF = mit->first;
			if (spKeyFrameBA.find(pKF) == spKeyFrameBA.end() || pKF->isBad() || pKF->mnId > maxKFid
				|| pKF->mnBALocalForKF != pCurrentKF->mnId || !pKF->GetMapPoint(get<0>(mit->second))) {
				continue;
			}

			//cout << "-- KF view exists" << endl;
			nEdges++;

			const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];
			//cout << "-- KeyPoint loads" << endl;

			if (pKF->mvuRight[get<0>(mit->second)] < 0) //Monocular
			{
				Eigen::Matrix<double, 2, 1> obs;
				obs << kpUn.pt.x, kpUn.pt.y;

				g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
				e->setMeasurement(obs);
				const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
				e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
				//cout << "-- Sigma loads" << endl;

				g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(thHuber2D);

				e->fx = pKF->fx;
				e->fy = pKF->fy;
				e->cx = pKF->cx;
				e->cy = pKF->cy;
				//cout << "-- Calibration loads" << endl;

				optimizer.addEdge(e);
				//cout << "-- Edge added" << endl;

				vpEdgesMono.push_back(e);
				vpEdgeKFMono.push_back(pKF);
				vpMapPointEdgeMono.push_back(pMPi);
				//cout << "-- Added to vector" << endl;
			}
			else // RGBD or Stereo
			{
				Eigen::Matrix<double, 3, 1> obs;
				const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
				obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

				g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
				e->setMeasurement(obs);
				const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
				Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
				e->setInformation(Info);

				g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(thHuber3D);

				e->fx = pKF->fx;
				e->fy = pKF->fy;
				e->cx = pKF->cx;
				e->cy = pKF->cy;
				e->bf = pKF->mbf;

				optimizer.addEdge(e);

				vpEdgesStereo.push_back(e);
				vpEdgeKFStereo.push_back(pKF);
				vpMapPointEdgeStereo.push_back(pMPi);
			}
			//cout << "-- End to load point" << endl;
		}
	}

	//cout << "End to load MPs" << endl;

	if (pbStopFlag) {
		if (*pbStopFlag) {
			return;
		}
	}

	optimizer.initializeOptimization();
	optimizer.optimize(5);

	//cout << "End the first optimization" << endl;

	bool bDoMore = true;

	if (pbStopFlag) {
		if (*pbStopFlag) {
			bDoMore = false;
		}
	}

	if (bDoMore) {

		// Check inlier observations
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
			MapPoint *pMP = vpMapPointEdgeMono[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 5.991 || !e->isDepthPositive()) {
				e->setLevel(1);
			}

			e->setRobustKernel(0);
		}

		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
			MapPoint *pMP = vpMapPointEdgeStereo[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 7.815 || !e->isDepthPositive()) {
				e->setLevel(1);
			}

			e->setRobustKernel(0);
		}

		// Optimize again without the outliers

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);

		//cout << "End the second optimization (without outliers)" << endl;
	}

	vector<pair<KeyFrame *, MapPoint *>> vToErase;
	vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

	// Check inlier observations
	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
		g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
		MapPoint *pMP = vpMapPointEdgeMono[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 5.991 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFMono[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
		g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
		MapPoint *pMP = vpMapPointEdgeStereo[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 7.815 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFStereo[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	// Get Map Mutex
	unique_lock<shared_timed_mutex> lock(pCurrentKF->GetMap()->mMutexMapUpdate);

	if (!vToErase.empty()) {
		for (size_t i = 0; i < vToErase.size(); i++) {
			KeyFrame *pKFi = vToErase[i].first;
			MapPoint *pMPi = vToErase[i].second;
			pKFi->EraseMapPointMatch(pMPi);
			pMPi->EraseObservation(pKFi);
		}
	}
	//cout << "End to erase observations" << endl;

	// Recover optimized data

	//Keyframes
	for (KeyFrame *pKFi: vpWeldingKFs) {
		if (pKFi->isBad()) {
			continue;
		}

		g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		pKFi->SetPose(Converter::toCvMat(SE3quat));
	}
	//cout << "End to update the KeyFrames" << endl;

	//Points
	for (MapPoint *pMPi: vpMPs) {
		if (pMPi->isBad()) {
			continue;
		}

		g2o::VertexSBAPointXYZ
			*vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMPi->mnId + maxKFid + 1));
		pMPi->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMPi->UpdateNormalAndDepth();
	}
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pMainKF,
                                      vector<KeyFrame *> vpAdjustKF,
                                      vector<KeyFrame *> vpFixedKF,
                                      bool *pbStopFlag)
{
	bool bShowImages = false;

	vector<MapPoint *> vpMPs;

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	optimizer.setVerbose(false);

	if (pbStopFlag) {
		optimizer.setForceStopFlag(pbStopFlag);
	}

	long unsigned int maxKFid = 0;
	set<KeyFrame *> spKeyFrameBA;

	Map *pCurrentMap = pMainKF->GetMap();

	//set<MapPoint*> sNumObsMP;

	// Set fixed KeyFrame vertices
	for (KeyFrame *pKFi: vpFixedKF) {
		if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap) {
			Verbose::PrintMess("ERROR LBA: KF is bad or is not in the current map", Verbose::VERBOSITY_NORMAL);
			continue;
		}

		pKFi->mnBALocalForMerge = pMainKF->mnId;

		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
		vSE3->setId(pKFi->mnId);
		vSE3->setFixed(true);
		optimizer.addVertex(vSE3);
		if (pKFi->mnId > maxKFid) {
			maxKFid = pKFi->mnId;
		}

		set<MapPoint *> spViewMPs = pKFi->GetMapPoints();
		for (MapPoint *pMPi: spViewMPs) {
			if (pMPi) {
				if (!pMPi->isBad() && pMPi->GetMap() == pCurrentMap) {

					if (pMPi->mnBALocalForMerge != pMainKF->mnId) {
						vpMPs.push_back(pMPi);
						pMPi->mnBALocalForMerge = pMainKF->mnId;
					}
				}
			}
			/*if(sNumObsMP.find(pMPi) == sNumObsMP.end())
                    {
                        sNumObsMP.insert(pMPi);
                    }
                    else
                    {
                        if(pMPi->mnBALocalForMerge!=pMainKF->mnId)
                        {
                            vpMPs.push_back(pMPi);
                            pMPi->mnBALocalForMerge=pMainKF->mnId;
                        }
                    }*/
		}

		spKeyFrameBA.insert(pKFi);
	}

	//cout << "End to load Fixed KFs" << endl;

	// Set non fixed Keyframe vertices
	set<KeyFrame *> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
	for (KeyFrame *pKFi: vpAdjustKF) {
		if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap) {
			continue;
		}

		pKFi->mnBALocalForKF = pMainKF->mnId;

		g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
		vSE3->setId(pKFi->mnId);
		optimizer.addVertex(vSE3);
		if (pKFi->mnId > maxKFid) {
			maxKFid = pKFi->mnId;
		}

		set<MapPoint *> spViewMPs = pKFi->GetMapPoints();
		for (MapPoint *pMPi: spViewMPs) {
			if (pMPi) {
				if (!pMPi->isBad() && pMPi->GetMap() == pCurrentMap) {
					/*if(sNumObsMP.find(pMPi) == sNumObsMP.end())
                    {
                        sNumObsMP.insert(pMPi);
                    }*/
					if (pMPi->mnBALocalForMerge != pMainKF->mnId) {
						vpMPs.push_back(pMPi);
						pMPi->mnBALocalForMerge = pMainKF->mnId;
					}
				}
			}
		}

		spKeyFrameBA.insert(pKFi);
	}

	//Verbose::PrintMess("LBA: There are " + to_string(vpMPs.size()) + " MPs to optimize", Verbose::VERBOSITY_NORMAL);

	//cout << "End to load KFs for position adjust" << endl;

	const int nExpectedSize = (vpAdjustKF.size() + vpFixedKF.size()) * vpMPs.size();

	vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
	vpEdgesStereo.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFStereo;
	vpEdgeKFStereo.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeStereo;
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	const float thHuber2D = sqrt(5.99);
	const float thHuber3D = sqrt(7.815);

	// Set MapPoint vertices
	map<KeyFrame *, int> mpObsKFs;
	map<KeyFrame *, int> mpObsFinalKFs;
	map<MapPoint *, int> mpObsMPs;
	for (unsigned int i = 0; i < vpMPs.size(); ++i) {
		MapPoint *pMPi = vpMPs[i];
		if (pMPi->isBad()) {
			continue;
		}

		g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMPi->GetWorldPos()));
		const int id = pMPi->mnId + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const map<KeyFrame *, tuple<int, int>> observations = pMPi->GetObservations();
		int nEdges = 0;
		//SET EDGES
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end();
		     mit++) {
			//cout << "--KF view init" << endl;

			KeyFrame *pKF = mit->first;
			if (pKF->isBad() || pKF->mnId > maxKFid || pKF->mnBALocalForMerge != pMainKF->mnId
				|| !pKF->GetMapPoint(get<0>(mit->second))) {
				continue;
			}

			//cout << "-- KF view exists" << endl;
			nEdges++;

			const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];
			//cout << "-- KeyPoint loads" << endl;

			if (pKF->mvuRight[get<0>(mit->second)] < 0) //Monocular
			{
				mpObsMPs[pMPi]++;
				Eigen::Matrix<double, 2, 1> obs;
				obs << kpUn.pt.x, kpUn.pt.y;

				ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
				e->setMeasurement(obs);
				const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
				e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
				//cout << "-- Sigma loads" << endl;

				g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(thHuber2D);

				e->pCamera = pKF->mpCamera;
				//cout << "-- Calibration loads" << endl;

				optimizer.addEdge(e);
				//cout << "-- Edge added" << endl;

				vpEdgesMono.push_back(e);
				vpEdgeKFMono.push_back(pKF);
				vpMapPointEdgeMono.push_back(pMPi);
				//cout << "-- Added to vector" << endl;

				mpObsKFs[pKF]++;
			}
			else // RGBD or Stereo
			{
				mpObsMPs[pMPi] += 2;
				Eigen::Matrix<double, 3, 1> obs;
				const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
				obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

				g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
				e->setMeasurement(obs);
				const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
				Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
				e->setInformation(Info);

				g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(thHuber3D);

				e->fx = pKF->fx;
				e->fy = pKF->fy;
				e->cx = pKF->cx;
				e->cy = pKF->cy;
				e->bf = pKF->mbf;

				optimizer.addEdge(e);

				vpEdgesStereo.push_back(e);
				vpEdgeKFStereo.push_back(pKF);
				vpMapPointEdgeStereo.push_back(pMPi);

				mpObsKFs[pKF]++;
			}
			//cout << "-- End to load point" << endl;
		}
	}
	//Verbose::PrintMess("LBA: number total of edged -> " + to_string(vpEdgeKFMono.size() + vpEdgeKFStereo.size()), Verbose::VERBOSITY_NORMAL);

	map<int, int> mStatsObs;
	for (map<MapPoint *, int>::iterator it = mpObsMPs.begin(); it != mpObsMPs.end(); ++it) {
		MapPoint *pMPi = it->first;
		int numObs = it->second;

		mStatsObs[numObs]++;
		/*if(numObs < 5)
        {
            cout << "LBA: MP " << pMPi->mnId << " has " << numObs << " observations" << endl;
        }*/
	}

	/*for(map<int, int>::iterator it = mStatsObs.begin(); it != mStatsObs.end(); ++it)
    {
        cout << "LBA: There are " << it->second << " MPs with " << it->first << " observations" << endl;
    }*/

	//cout << "End to load MPs" << endl;

	if (pbStopFlag) {
		if (*pbStopFlag) {
			return;
		}
	}

	optimizer.save("/home/da/project/ros/orb_dvl2_ws/src/dvl2/orb3_result/merge_BA.g2o");
	optimizer.initializeOptimization();
	optimizer.optimize(5);

	//cout << "End the first optimization" << endl;

	bool bDoMore = true;

	if (pbStopFlag) {
		if (*pbStopFlag) {
			bDoMore = false;
		}
	}

	map<unsigned long int, int> mWrongObsKF;
	if (bDoMore) {

		// Check inlier observations
		int badMonoMP = 0, badStereoMP = 0;
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
			MapPoint *pMP = vpMapPointEdgeMono[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 5.991 || !e->isDepthPositive()) {
				e->setLevel(1);
				badMonoMP++;
			}

			e->setRobustKernel(0);
		}

		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
			MapPoint *pMP = vpMapPointEdgeStereo[i];

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 7.815 || !e->isDepthPositive()) {
				e->setLevel(1);
				badStereoMP++;
			}

			e->setRobustKernel(0);
		}
		Verbose::PrintMess(
			"LBA: First optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP)
				+ " sterero bad edges", Verbose::VERBOSITY_DEBUG);

		// Optimize again without the outliers

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);

		//cout << "End the second optimization (without outliers)" << endl;
	}

	vector<pair<KeyFrame *, MapPoint *>> vToErase;
	vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());
	set<MapPoint *> spErasedMPs;
	set<KeyFrame *> spErasedKFs;

	// Check inlier observations
	int badMonoMP = 0, badStereoMP = 0;
	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
		ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
		MapPoint *pMP = vpMapPointEdgeMono[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 5.991 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFMono[i];
			vToErase.push_back(make_pair(pKFi, pMP));
			mWrongObsKF[pKFi->mnId]++;
			badMonoMP++;

			spErasedMPs.insert(pMP);
			spErasedKFs.insert(pKFi);
		}
	}

	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
		g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
		MapPoint *pMP = vpMapPointEdgeStereo[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > 7.815 || !e->isDepthPositive()) {
			KeyFrame *pKFi = vpEdgeKFStereo[i];
			vToErase.push_back(make_pair(pKFi, pMP));
			mWrongObsKF[pKFi->mnId]++;
			badStereoMP++;

			spErasedMPs.insert(pMP);
			spErasedKFs.insert(pKFi);
		}
	}
	Verbose::PrintMess(
		"LBA: Second optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP)
			+ " sterero bad edges", Verbose::VERBOSITY_DEBUG);

	// Get Map Mutex
	unique_lock<shared_timed_mutex> lock(pMainKF->GetMap()->mMutexMapUpdate, std::defer_lock);
	if (!lock.try_lock()) {
		return;
	}

	if (!vToErase.empty()) {
		map<KeyFrame *, int> mpMPs_in_KF;
		for (KeyFrame *pKFi: spErasedKFs) {
			int num_MPs = pKFi->GetMapPoints().size();
			mpMPs_in_KF[pKFi] = num_MPs;
		}

		Verbose::PrintMess(
			"LBA: There are " + to_string(vToErase.size()) + " observations whose will be deleted from the map",
			Verbose::VERBOSITY_DEBUG);
		for (size_t i = 0; i < vToErase.size(); i++) {
			KeyFrame *pKFi = vToErase[i].first;
			MapPoint *pMPi = vToErase[i].second;
			pKFi->EraseMapPointMatch(pMPi);
			pMPi->EraseObservation(pKFi);
		}

		Verbose::PrintMess("LBA: " + to_string(spErasedMPs.size()) + " MPs had deleted observations",
		                   Verbose::VERBOSITY_DEBUG);
		Verbose::PrintMess("LBA: Current map is " + to_string(pMainKF->GetMap()->GetId()), Verbose::VERBOSITY_DEBUG);
		int numErasedMP = 0;
		for (MapPoint *pMPi: spErasedMPs) {
			if (pMPi->isBad()) {
				Verbose::PrintMess(
					"LBA: MP " + to_string(pMPi->mnId) + " has lost almost all the observations, its origin map is "
						+ to_string(pMPi->mnOriginMapId), Verbose::VERBOSITY_DEBUG);
				numErasedMP++;
			}
		}
		Verbose::PrintMess("LBA: " + to_string(numErasedMP) + " MPs had deleted from the map",
		                   Verbose::VERBOSITY_DEBUG);

		for (KeyFrame *pKFi: spErasedKFs) {
			int num_MPs = pKFi->GetMapPoints().size();
			int num_init_MPs = mpMPs_in_KF[pKFi];
			Verbose::PrintMess(
				"LBA: Initially KF " + to_string(pKFi->mnId) + " had " + to_string(num_init_MPs) + ", at the end has "
					+ to_string(num_MPs), Verbose::VERBOSITY_DEBUG);
		}
	}
	for (unsigned int i = 0; i < vpMPs.size(); ++i) {
		MapPoint *pMPi = vpMPs[i];
		if (pMPi->isBad()) {
			continue;
		}

		const map<KeyFrame *, tuple<int, int>> observations = pMPi->GetObservations();
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end();
		     mit++) {
			//cout << "--KF view init" << endl;

			KeyFrame *pKF = mit->first;
			if (pKF->isBad() || pKF->mnId > maxKFid || pKF->mnBALocalForKF != pMainKF->mnId
				|| !pKF->GetMapPoint(get<0>(mit->second))) {
				continue;
			}

			const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];
			//cout << "-- KeyPoint loads" << endl;

			if (pKF->mvuRight[get<0>(mit->second)] < 0) //Monocular
			{
				mpObsFinalKFs[pKF]++;
			}
			else // RGBD or Stereo
			{

				mpObsFinalKFs[pKF]++;
			}
			//cout << "-- End to load point" << endl;
		}
	}

	//cout << "End to erase observations" << endl;

	// Recover optimized data

	//Keyframes
	for (KeyFrame *pKFi: vpAdjustKF) {
		if (pKFi->isBad()) {
			continue;
		}

		g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		cv::Mat Tiw = Converter::toCvMat(SE3quat);
		cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
		cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
		double dist = cv::norm(trasl);

		int numMonoBadPoints = 0, numMonoOptPoints = 0;
		int numStereoBadPoints = 0, numStereoOptPoints = 0;
		vector<MapPoint *> vpMonoMPsOpt, vpStereoMPsOpt;
		vector<MapPoint *> vpMonoMPsBad, vpStereoMPsBad;

		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
			MapPoint *pMP = vpMapPointEdgeMono[i];
			KeyFrame *pKFedge = vpEdgeKFMono[i];

			if (pKFi != pKFedge) {
				continue;
			}

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 5.991 || !e->isDepthPositive()) {
				numMonoBadPoints++;
				vpMonoMPsBad.push_back(pMP);
			}
			else {
				numMonoOptPoints++;
				vpMonoMPsOpt.push_back(pMP);
			}
		}

		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
			MapPoint *pMP = vpMapPointEdgeStereo[i];
			KeyFrame *pKFedge = vpEdgeKFMono[i];

			if (pKFi != pKFedge) {
				continue;
			}

			if (pMP->isBad()) {
				continue;
			}

			if (e->chi2() > 7.815 || !e->isDepthPositive()) {
				numStereoBadPoints++;
				vpStereoMPsBad.push_back(pMP);
			}
			else {
				numStereoOptPoints++;
				vpStereoMPsOpt.push_back(pMP);
			}
		}

		if (numMonoOptPoints + numStereoOptPoints < 50) {
			Verbose::PrintMess("LBA ERROR: KF " + to_string(pKFi->mnId) + " has only " + to_string(numMonoOptPoints)
				                   + " monocular and " + to_string(numStereoOptPoints) + " stereo points",
			                   Verbose::VERBOSITY_DEBUG);
		}
		if (dist > 1.0) {
			if (bShowImages) {
				string strNameFile = pKFi->mNameFile;
				// cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);  // original
				cv::Mat imLeft = cv::imread(strNameFile, cv::IMREAD_UNCHANGED);

				// cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);  // original
				cv::cvtColor(imLeft, imLeft, cv::COLOR_GRAY2BGR);

				int numPointsMono = 0, numPointsStereo = 0;
				int numPointsMonoBad = 0, numPointsStereoBad = 0;
				for (int i = 0; i < vpMonoMPsOpt.size(); ++i) {
					if (!vpMonoMPsOpt[i] || vpMonoMPsOpt[i]->isBad()) {
						continue;
					}
					int index = get<0>(vpMonoMPsOpt[i]->GetIndexInKeyFrame(pKFi));
					if (index < 0) {
						//cout << "LBA ERROR: KF has a monocular observation which is not recognized by the MP" << endl;
						//cout << "LBA: KF " << pKFi->mnId << " and MP " << vpMonoMPsOpt[i]->mnId << " with index " << endl;
						continue;
					}

					//string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
					cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(255, 0, 0));
					//cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
					numPointsMono++;
				}

				for (int i = 0; i < vpStereoMPsOpt.size(); ++i) {
					if (!vpStereoMPsOpt[i] || vpStereoMPsOpt[i]->isBad()) {
						continue;
					}
					int index = get<0>(vpStereoMPsOpt[i]->GetIndexInKeyFrame(pKFi));
					if (index < 0) {
						//cout << "LBA: KF has a stereo observation which is not recognized by the MP" << endl;
						//cout << "LBA: KF " << pKFi->mnId << " and MP " << vpStereoMPsOpt[i]->mnId << endl;
						continue;
					}

					//string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
					cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(0, 255, 0));
					//cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
					numPointsStereo++;
				}

				for (int i = 0; i < vpMonoMPsBad.size(); ++i) {
					if (!vpMonoMPsBad[i] || vpMonoMPsBad[i]->isBad()) {
						continue;
					}
					int index = get<0>(vpMonoMPsBad[i]->GetIndexInKeyFrame(pKFi));
					if (index < 0) {
						//cout << "LBA ERROR: KF has a monocular observation which is not recognized by the MP" << endl;
						//cout << "LBA: KF " << pKFi->mnId << " and MP " << vpMonoMPsOpt[i]->mnId << " with index " << endl;
						continue;
					}

					//string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
					cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(0, 0, 255));
					//cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
					numPointsMonoBad++;
				}
				for (int i = 0; i < vpStereoMPsBad.size(); ++i) {
					if (!vpStereoMPsBad[i] || vpStereoMPsBad[i]->isBad()) {
						continue;
					}
					int index = get<0>(vpStereoMPsBad[i]->GetIndexInKeyFrame(pKFi));
					if (index < 0) {
						//cout << "LBA: KF has a stereo observation which is not recognized by the MP" << endl;
						//cout << "LBA: KF " << pKFi->mnId << " and MP " << vpStereoMPsOpt[i]->mnId << endl;
						continue;
					}

					//string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
					cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(0, 0, 255));
					//cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
					numPointsStereoBad++;
				}

				string namefile =
					"./test_LBA/LBA_KF" + to_string(pKFi->mnId) + "_" + to_string(numPointsMono + numPointsStereo)
						+ "_D" + to_string(dist) + ".png";
				cv::imwrite(namefile, imLeft);

				Verbose::PrintMess("--LBA in KF " + to_string(pKFi->mnId), Verbose::VERBOSITY_DEBUG);
				Verbose::PrintMess("--Distance: " + to_string(dist) + " meters", Verbose::VERBOSITY_DEBUG);
				Verbose::PrintMess("--Number of observations: " + to_string(numMonoOptPoints) + " in mono and "
					                   + to_string(numStereoOptPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
				Verbose::PrintMess(
					"--Number of discarded observations: " + to_string(numMonoBadPoints) + " in mono and "
						+ to_string(numStereoBadPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
				Verbose::PrintMess(
					"--To much distance correction in LBA: It has " + to_string(mpObsKFs[pKFi]) + " observated MPs",
					Verbose::VERBOSITY_DEBUG);
				Verbose::PrintMess("--To much distance correction in LBA: It has " + to_string(mpObsFinalKFs[pKFi])
					                   + " deleted observations", Verbose::VERBOSITY_DEBUG);
				Verbose::PrintMess("--------", Verbose::VERBOSITY_DEBUG);
			}
		}
		pKFi->SetPose(Tiw);
	}
	//cout << "End to update the KeyFrames" << endl;

	//Points
	for (MapPoint *pMPi: vpMPs) {
		if (pMPi->isBad()) {
			continue;
		}

		g2o::VertexSBAPointXYZ
			*vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMPi->mnId + maxKFid + 1));
		pMPi->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMPi->UpdateNormalAndDepth();
	}
	//cout << "End to update MapPoint" << endl;
	lock.unlock();
}

void Optimizer::MergeInertialBA(KeyFrame *pCurrKF,
                                KeyFrame *pMergeKF,
                                bool *pbStopFlag,
                                Map *pMap,
                                LoopClosing::KeyFrameAndPose &corrPoses)
{
	const int Nd = 6;
	const unsigned long maxKFid = pCurrKF->mnId;

	vector<KeyFrame *> vpOptimizableKFs;
	vpOptimizableKFs.reserve(2 * Nd);

	// For cov KFS, inertial parameters are not optimized
	const int maxCovKF = 30;
	vector<KeyFrame *> vpOptimizableCovKFs;
	vpOptimizableCovKFs.reserve(maxCovKF);

	// Add sliding window for current KF
	vpOptimizableKFs.push_back(pCurrKF);
	pCurrKF->mnBALocalForKF = pCurrKF->mnId;
	for (int i = 1; i < Nd; i++) {
		if (vpOptimizableKFs.back()->mPrevKF) {
			vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
			vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
		}
		else {
			break;
		}
	}

	list<KeyFrame *> lFixedKeyFrames;
	if (vpOptimizableKFs.back()->mPrevKF) {
		vpOptimizableCovKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
		vpOptimizableKFs.back()->mPrevKF->mnBALocalForKF = pCurrKF->mnId;
	}
	else {
		vpOptimizableCovKFs.push_back(vpOptimizableKFs.back());
		vpOptimizableKFs.pop_back();
	}

	KeyFrame *pKF0 = vpOptimizableCovKFs.back();
	cv::Mat Twc0 = pKF0->GetPoseInverse();

	// Add temporal neighbours to merge KF (previous and next KFs)
	vpOptimizableKFs.push_back(pMergeKF);
	pMergeKF->mnBALocalForKF = pCurrKF->mnId;

	// Previous KFs
	for (int i = 1; i < (Nd / 2); i++) {
		if (vpOptimizableKFs.back()->mPrevKF) {
			vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
			vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
		}
		else {
			break;
		}
	}

	// We fix just once the old map
	if (vpOptimizableKFs.back()->mPrevKF) {
		lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
		vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF = pCurrKF->mnId;
	}
	else {
		vpOptimizableKFs.back()->mnBALocalForKF = 0;
		vpOptimizableKFs.back()->mnBAFixedForKF = pCurrKF->mnId;
		lFixedKeyFrames.push_back(vpOptimizableKFs.back());
		vpOptimizableKFs.pop_back();
	}

	// Next KFs
	if (pMergeKF->mNextKF) {
		vpOptimizableKFs.push_back(pMergeKF->mNextKF);
		vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
	}

	while (vpOptimizableKFs.size() < (2 * Nd)) {
		if (vpOptimizableKFs.back()->mNextKF) {
			vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mNextKF);
			vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
		}
		else {
			break;
		}
	}

	int N = vpOptimizableKFs.size();

	// Optimizable points seen by optimizable keyframes
	list<MapPoint *> lLocalMapPoints;
	map<MapPoint *, int> mLocalObs;
	for (int i = 0; i < N; i++) {
		vector<MapPoint *> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
		for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
			// Using mnBALocalForKF we avoid redundance here, one MP can not be added several times to lLocalMapPoints
			MapPoint *pMP = *vit;
			if (pMP) {
				if (!pMP->isBad()) {
					if (pMP->mnBALocalForKF != pCurrKF->mnId) {
						mLocalObs[pMP] = 1;
						lLocalMapPoints.push_back(pMP);
						pMP->mnBALocalForKF = pCurrKF->mnId;
					}
					else {
						mLocalObs[pMP]++;
					}
				}
			}
		}
	}

	std::vector<std::pair<MapPoint *, int>> pairs;
	pairs.reserve(mLocalObs.size());
	for (auto itr = mLocalObs.begin(); itr != mLocalObs.end(); ++itr)
		pairs.push_back(*itr);
	sort(pairs.begin(), pairs.end(), sortByVal);

	// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
	int i = 0;
	for (vector<pair<MapPoint *, int>>::iterator lit = pairs.begin(), lend = pairs.end(); lit != lend; lit++, i++) {
		map<KeyFrame *, tuple<int, int>> observations = lit->first->GetObservations();
		if (i >= maxCovKF) {
			break;
		}
		for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (pKFi->mnBALocalForKF != pCurrKF->mnId
				&& pKFi->mnBAFixedForKF != pCurrKF->mnId) // If optimizable or already included...
			{
				pKFi->mnBALocalForKF = pCurrKF->mnId;
				if (!pKFi->isBad()) {
					vpOptimizableCovKFs.push_back(pKFi);
					break;
				}
			}
		}
	}

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;
	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	solver->setUserLambdaInit(1e3); // TODO uncomment

	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);

	// Set Local KeyFrame vertices
	N = vpOptimizableKFs.size();
	for (int i = 0; i < N; i++) {
		KeyFrame *pKFi = vpOptimizableKFs[i];

		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(false);
		optimizer.addVertex(VP);

		if (pKFi->bImu) {
			VertexVelocity *VV = new VertexVelocity(pKFi);
			VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
			VV->setFixed(false);
			optimizer.addVertex(VV);
			VertexGyroBias *VG = new VertexGyroBias(pKFi);
			VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
			VG->setFixed(false);
			optimizer.addVertex(VG);
			VertexAccBias *VA = new VertexAccBias(pKFi);
			VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
			VA->setFixed(false);
			optimizer.addVertex(VA);
		}
	}

	// Set Local cov keyframes vertices
	int Ncov = vpOptimizableCovKFs.size();
	for (int i = 0; i < Ncov; i++) {
		KeyFrame *pKFi = vpOptimizableCovKFs[i];

		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(false);
		optimizer.addVertex(VP);

		if (pKFi->bImu) {
			VertexVelocity *VV = new VertexVelocity(pKFi);
			VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
			VV->setFixed(false);
			optimizer.addVertex(VV);
			VertexGyroBias *VG = new VertexGyroBias(pKFi);
			VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
			VG->setFixed(false);
			optimizer.addVertex(VG);
			VertexAccBias *VA = new VertexAccBias(pKFi);
			VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
			VA->setFixed(false);
			optimizer.addVertex(VA);
		}
	}

	// Set Fixed KeyFrame vertices
	for (list<KeyFrame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++) {
		KeyFrame *pKFi = *lit;
		VertexPose *VP = new VertexPose(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		if (pKFi->bImu) {
			VertexVelocity *VV = new VertexVelocity(pKFi);
			VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
			VV->setFixed(true);
			optimizer.addVertex(VV);
			VertexGyroBias *VG = new VertexGyroBias(pKFi);
			VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
			VG->setFixed(true);
			optimizer.addVertex(VG);
			VertexAccBias *VA = new VertexAccBias(pKFi);
			VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
			VA->setFixed(true);
			optimizer.addVertex(VA);
		}
	}

	// Create intertial constraints
	vector<EdgeInertial *> vei(N, (EdgeInertial *)NULL);
	vector<EdgeGyroRW *> vegr(N, (EdgeGyroRW *)NULL);
	vector<EdgeAccRW *> vear(N, (EdgeAccRW *)NULL);
	for (int i = 0; i < N; i++) {
		//cout << "inserting inertial edge " << i << endl;
		KeyFrame *pKFi = vpOptimizableKFs[i];

		if (!pKFi->mPrevKF) {
			Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!!!!", Verbose::VERBOSITY_NORMAL);
			continue;
		}
		if (pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated) {
			pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
			g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1);
			g2o::HyperGraph::Vertex *VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2);
			g2o::HyperGraph::Vertex *VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3);
			g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
			g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2);
			g2o::HyperGraph::Vertex *VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3);

			if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2) {
				cerr << "Error " << VP1 << ", " << VV1 << ", " << VG1 << ", " << VA1 << ", " << VP2 << ", " << VV2
				     << ", " << VG2 << ", " << VA2 << endl;
				continue;
			}

			vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

			vei[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			vei[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
			vei[i]->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG1));
			vei[i]->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA1));
			vei[i]->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			vei[i]->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));

			// TODO Uncomment
			g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
			vei[i]->setRobustKernel(rki);
			rki->setDelta(sqrt(16.92));
			optimizer.addEdge(vei[i]);

			vegr[i] = new EdgeGyroRW();
			vegr[i]->setVertex(0, VG1);
			vegr[i]->setVertex(1, VG2);
			cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
			Eigen::Matrix3d InfoG;

			for (int r = 0; r < 3; r++)
				for (int c = 0; c < 3; c++)
					InfoG(r, c) = cvInfoG.at<float>(r, c);
			vegr[i]->setInformation(InfoG);
			optimizer.addEdge(vegr[i]);

			vear[i] = new EdgeAccRW();
			vear[i]->setVertex(0, VA1);
			vear[i]->setVertex(1, VA2);
			cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
			Eigen::Matrix3d InfoA;
			for (int r = 0; r < 3; r++)
				for (int c = 0; c < 3; c++)
					InfoA(r, c) = cvInfoA.at<float>(r, c);
			vear[i]->setInformation(InfoA);
			optimizer.addEdge(vear[i]);
		}
		else {
			Verbose::PrintMess("ERROR building inertial edge", Verbose::VERBOSITY_NORMAL);
		}
	}

	Verbose::PrintMess("end inserting inertial edges", Verbose::VERBOSITY_NORMAL);

	// Set MapPoint vertices
	const int nExpectedSize = (N + Ncov + lFixedKeyFrames.size()) * lLocalMapPoints.size();

	// Mono
	vector<EdgeMono *> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	// Stereo
	vector<EdgeStereo *> vpEdgesStereo;
	vpEdgesStereo.reserve(nExpectedSize);

	vector<KeyFrame *> vpEdgeKFStereo;
	vpEdgeKFStereo.reserve(nExpectedSize);

	vector<MapPoint *> vpMapPointEdgeStereo;
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float chi2Mono2 = 5.991;
	const float thHuberStereo = sqrt(7.815);
	const float chi2Stereo2 = 7.815;

	const unsigned long iniMPid = maxKFid * 5; // TODO: should be  maxKFid*4;

	Verbose::PrintMess("start inserting MPs", Verbose::VERBOSITY_NORMAL);
	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		if (!pMP) {
			continue;
		}

		g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

		unsigned long id = pMP->mnId + iniMPid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

		// Create visual constraints
		for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end();
		     mit != mend; mit++) {
			KeyFrame *pKFi = mit->first;

			if (!pKFi) {
				continue;
			}

			if ((pKFi->mnBALocalForKF != pCurrKF->mnId) && (pKFi->mnBAFixedForKF != pCurrKF->mnId)) {
				continue;
			}

			if (pKFi->mnId > maxKFid) {
				Verbose::PrintMess("ID greater than current KF is", Verbose::VERBOSITY_NORMAL);
				continue;
			}

			if (optimizer.vertex(id) == NULL || optimizer.vertex(pKFi->mnId) == NULL) {
				continue;
			}

			if (!pKFi->isBad()) {
				const cv::KeyPoint &kpUn = pKFi->mvKeysUn[get<0>(mit->second)];

				if (pKFi->mvuRight[get<0>(mit->second)] < 0) // Monocular observation
				{
					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					EdgeMono *e = new EdgeMono();
					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);
					optimizer.addEdge(e);
					vpEdgesMono.push_back(e);
					vpEdgeKFMono.push_back(pKFi);
					vpMapPointEdgeMono.push_back(pMP);
				}
				else // stereo observation
				{
					const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
					Eigen::Matrix<double, 3, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

					EdgeStereo *e = new EdgeStereo();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberStereo);

					optimizer.addEdge(e);
					vpEdgesStereo.push_back(e);
					vpEdgeKFStereo.push_back(pKFi);
					vpMapPointEdgeStereo.push_back(pMP);
				}
			}
		}
	}

	if (pbStopFlag) {
		if (*pbStopFlag) {
			return;
		}
	}
	optimizer.initializeOptimization();
	optimizer.optimize(3);
	if (pbStopFlag) {
		if (!*pbStopFlag) {
			optimizer.optimize(5);
		}
	}

	optimizer.setForceStopFlag(pbStopFlag);

	vector<pair<KeyFrame *, MapPoint *>> vToErase;
	vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

	// Check inlier observations
	// Mono
	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
		EdgeMono *e = vpEdgesMono[i];
		MapPoint *pMP = vpMapPointEdgeMono[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > chi2Mono2) {
			KeyFrame *pKFi = vpEdgeKFMono[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	// Stereo
	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
		EdgeStereo *e = vpEdgesStereo[i];
		MapPoint *pMP = vpMapPointEdgeStereo[i];

		if (pMP->isBad()) {
			continue;
		}

		if (e->chi2() > chi2Stereo2) {
			KeyFrame *pKFi = vpEdgeKFStereo[i];
			vToErase.push_back(make_pair(pKFi, pMP));
		}
	}

	// Get Map Mutex and erase outliers
	// unique_lock<timed_mutex> lock(pMap->mMutexMapUpdate);
	if (!vToErase.empty()) {
		for (size_t i = 0; i < vToErase.size(); i++) {
			KeyFrame *pKFi = vToErase[i].first;
			MapPoint *pMPi = vToErase[i].second;
			pKFi->EraseMapPointMatch(pMPi);
			pMPi->EraseObservation(pKFi);
		}
	}

	// Recover optimized data
	//Keyframes
	for (int i = 0; i < N; i++) {
		KeyFrame *pKFi = vpOptimizableKFs[i];

		VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
		cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
		pKFi->SetPose(Tcw);

		cv::Mat Tiw = pKFi->GetPose();
		cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
		cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
		g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
		corrPoses[pKFi] = g2oSiw;

		if (pKFi->bImu) {
			VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
			pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
			VertexGyroBias *VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
			VertexAccBias *VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
			Vector6d b;
			b << VG->estimate(), VA->estimate();
			pKFi->SetNewBias(IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]));
		}
	}

	for (int i = 0; i < Ncov; i++) {
		KeyFrame *pKFi = vpOptimizableCovKFs[i];

		VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
		cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
		pKFi->SetPose(Tcw);

		cv::Mat Tiw = pKFi->GetPose();
		cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
		cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
		g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
		corrPoses[pKFi] = g2oSiw;

		if (pKFi->bImu) {
			VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
			pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
			VertexGyroBias *VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
			VertexAccBias *VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
			Vector6d b;
			b << VG->estimate(), VA->estimate();
			pKFi->SetNewBias(IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]));
		}
	}

	//Points
	for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
		MapPoint *pMP = *lit;
		g2o::VertexSBAPointXYZ
			*vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + iniMPid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
	}

	pMap->IncreaseChangeIndex();
}

int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setVerbose(false);
	optimizer.setAlgorithm(solver);

	int nInitialMonoCorrespondences = 0;
	int nInitialStereoCorrespondences = 0;
	int nInitialCorrespondences = 0;

	// Set Frame vertex
	VertexPose *VP = new VertexPose(pFrame);
	VP->setId(0);
	VP->setFixed(false);
	optimizer.addVertex(VP);
	VertexVelocity *VV = new VertexVelocity(pFrame);
	VV->setId(1);
	VV->setFixed(false);
	optimizer.addVertex(VV);
	VertexGyroBias *VG = new VertexGyroBias(pFrame);
	VG->setId(2);
	VG->setFixed(false);
	optimizer.addVertex(VG);
	VertexAccBias *VA = new VertexAccBias(pFrame);
	VA->setId(3);
	VA->setFixed(false);
	optimizer.addVertex(VA);

	// Set MapPoint vertices
	const int N = pFrame->N;
	const int Nleft = pFrame->Nleft;
	const bool bRight = (Nleft != -1);

	vector<EdgeMonoOnlyPose *> vpEdgesMono;
	vector<EdgeStereoOnlyPose *> vpEdgesStereo;
	vector<size_t> vnIndexEdgeMono;
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesMono.reserve(N);
	vpEdgesStereo.reserve(N);
	vnIndexEdgeMono.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	{
		unique_lock<mutex> lock(MapPoint::mGlobalMutex);

		for (int i = 0; i < N; i++) {
			MapPoint *pMP = pFrame->mvpMapPoints[i];
			if (pMP) {
				cv::KeyPoint kpUn;

				// Left monocular observation
				if ((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft) {
					if (i < Nleft) { // pair left-right
						kpUn = pFrame->mvKeys[i];
					}
					else {
						kpUn = pFrame->mvKeysUn[i];
					}

					nInitialMonoCorrespondences++;
					pFrame->mvbOutlier[i] = false;

					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					EdgeMonoOnlyPose *e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 0);

					e->setVertex(0, VP);
					e->setMeasurement(obs);

					// Add here uncerteinty
					const float unc2 = pFrame->mpCamera->uncertainty2(obs);

					const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					optimizer.addEdge(e);

					vpEdgesMono.push_back(e);
					vnIndexEdgeMono.push_back(i);
				}
					// Stereo observation
				else if (!bRight) {
					nInitialStereoCorrespondences++;
					pFrame->mvbOutlier[i] = false;

					kpUn = pFrame->mvKeysUn[i];
					const float kp_ur = pFrame->mvuRight[i];
					Eigen::Matrix<double, 3, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

					EdgeStereoOnlyPose *e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

					e->setVertex(0, VP);
					e->setMeasurement(obs);

					// Add here uncerteinty
					const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

					const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
					e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberStereo);

					optimizer.addEdge(e);

					vpEdgesStereo.push_back(e);
					vnIndexEdgeStereo.push_back(i);
				}

				// Right monocular observation
				if (bRight && i >= Nleft) {
					nInitialMonoCorrespondences++;
					pFrame->mvbOutlier[i] = false;

					kpUn = pFrame->mvKeysRight[i - Nleft];
					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					EdgeMonoOnlyPose *e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 1);

					e->setVertex(0, VP);
					e->setMeasurement(obs);

					// Add here uncerteinty
					const float unc2 = pFrame->mpCamera->uncertainty2(obs);

					const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					optimizer.addEdge(e);

					vpEdgesMono.push_back(e);
					vnIndexEdgeMono.push_back(i);
				}
			}
		}
	}
	nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

	KeyFrame *pKF = pFrame->mpLastKeyFrame;
	VertexPose *VPk = new VertexPose(pKF);
	VPk->setId(4);
	VPk->setFixed(true);
	optimizer.addVertex(VPk);
	VertexVelocity *VVk = new VertexVelocity(pKF);
	VVk->setId(5);
	VVk->setFixed(true);
	optimizer.addVertex(VVk);
	VertexGyroBias *VGk = new VertexGyroBias(pKF);
	VGk->setId(6);
	VGk->setFixed(true);
	optimizer.addVertex(VGk);
	VertexAccBias *VAk = new VertexAccBias(pKF);
	VAk->setId(7);
	VAk->setFixed(true);
	optimizer.addVertex(VAk);

	EdgeInertial *ei = new EdgeInertial(pFrame->mpImuPreintegrated);

	ei->setVertex(0, VPk);
	ei->setVertex(1, VVk);
	ei->setVertex(2, VGk);
	ei->setVertex(3, VAk);
	ei->setVertex(4, VP);
	ei->setVertex(5, VV);
	optimizer.addEdge(ei);

	EdgeGyroRW *egr = new EdgeGyroRW();
	egr->setVertex(0, VGk);
	egr->setVertex(1, VG);
	cv::Mat cvInfoG = pFrame->mpImuPreintegrated->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
	Eigen::Matrix3d InfoG;
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			InfoG(r, c) = cvInfoG.at<float>(r, c);
	egr->setInformation(InfoG);
	optimizer.addEdge(egr);

	EdgeAccRW *ear = new EdgeAccRW();
	ear->setVertex(0, VAk);
	ear->setVertex(1, VA);
	cv::Mat cvInfoA = pFrame->mpImuPreintegrated->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
	Eigen::Matrix3d InfoA;
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			InfoA(r, c) = cvInfoA.at<float>(r, c);
	ear->setInformation(InfoA);
	optimizer.addEdge(ear);

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	float chi2Mono[4] = {12, 7.5, 5.991, 5.991};
	float chi2Stereo[4] = {15.6, 9.8, 7.815, 7.815};

	int its[4] = {10, 10, 10, 10};

	int nBad = 0;
	int nBadMono = 0;
	int nBadStereo = 0;
	int nInliersMono = 0;
	int nInliersStereo = 0;
	int nInliers = 0;
	bool bOut = false;
	for (size_t it = 0; it < 4; it++) {
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;
		nBadMono = 0;
		nBadStereo = 0;
		nInliers = 0;
		nInliersMono = 0;
		nInliersStereo = 0;
		float chi2close = 1.5 * chi2Mono[it];

		// For monocular observations
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			EdgeMonoOnlyPose *e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();
			bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.f;

			if ((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close) || !e->isDepthPositive()) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBadMono++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
				nInliersMono++;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		// For stereo observations
		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			EdgeStereoOnlyPose *e = vpEdgesStereo[i];

			const size_t idx = vnIndexEdgeStereo[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Stereo[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1); // not included in next optimization
				nBadStereo++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
				nInliersStereo++;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		nInliers = nInliersMono + nInliersStereo;
		nBad = nBadMono + nBadStereo;

		if (optimizer.edges().size() < 10) {
			cout << "PIOLKF: NOT ENOUGH EDGES" << endl;
			break;
		}
	}

	// If not too much tracks, recover not too bad points
	if ((nInliers < 30) && !bRecInit) {
		nBad = 0;
		const float chi2MonoOut = 18.f;
		const float chi2StereoOut = 24.f;
		EdgeMonoOnlyPose *e1;
		EdgeStereoOnlyPose *e2;
		for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++) {
			const size_t idx = vnIndexEdgeMono[i];
			e1 = vpEdgesMono[i];
			e1->computeError();
			if (e1->chi2() < chi2MonoOut) {
				pFrame->mvbOutlier[idx] = false;
			}
			else {
				nBad++;
			}
		}
		for (size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++) {
			const size_t idx = vnIndexEdgeStereo[i];
			e2 = vpEdgesStereo[i];
			e2->computeError();
			if (e2->chi2() < chi2StereoOut) {
				pFrame->mvbOutlier[idx] = false;
			}
			else {
				nBad++;
			}
		}
	}

	// Recover optimized pose, velocity and biases
	pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb),
	                           Converter::toCvMat(VP->estimate().twb),
	                           Converter::toCvMat(VV->estimate()));
	Vector6d b;
	b << VG->estimate(), VA->estimate();
	pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

	// Recover Hessian, marginalize keyFframe states and generate new prior for frame
	Eigen::Matrix<double, 15, 15> H;
	H.setZero();

	H.block<9, 9>(0, 0) += ei->GetHessian2();
	H.block<3, 3>(9, 9) += egr->GetHessian2();
	H.block<3, 3>(12, 12) += ear->GetHessian2();

	int tot_in = 0, tot_out = 0;
	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
		EdgeMonoOnlyPose *e = vpEdgesMono[i];

		const size_t idx = vnIndexEdgeMono[i];

		if (!pFrame->mvbOutlier[idx]) {
			H.block<6, 6>(0, 0) += e->GetHessian();
			tot_in++;
		}
		else {
			tot_out++;
		}
	}

	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
		EdgeStereoOnlyPose *e = vpEdgesStereo[i];

		const size_t idx = vnIndexEdgeStereo[i];

		if (!pFrame->mvbOutlier[idx]) {
			H.block<6, 6>(0, 0) += e->GetHessian();
			tot_in++;
		}
		else {
			tot_out++;
		}
	}

	pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,
	                                      VP->estimate().twb,
	                                      VV->estimate(),
	                                      VG->estimate(),
	                                      VA->estimate(),
	                                      H);

	return nInitialCorrespondences - nBad;
}

int Optimizer::PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);

	int nInitialMonoCorrespondences = 0;
	int nInitialStereoCorrespondences = 0;
	int nInitialCorrespondences = 0;

	// Set Current Frame vertex
	VertexPose *VP = new VertexPose(pFrame);
	VP->setId(0);
	VP->setFixed(false);
	optimizer.addVertex(VP);
	VertexVelocity *VV = new VertexVelocity(pFrame);
	VV->setId(1);
	VV->setFixed(false);
	optimizer.addVertex(VV);
	VertexGyroBias *VG = new VertexGyroBias(pFrame);
	VG->setId(2);
	VG->setFixed(false);
	optimizer.addVertex(VG);
	VertexAccBias *VA = new VertexAccBias(pFrame);
	VA->setId(3);
	VA->setFixed(false);
	optimizer.addVertex(VA);

	// Set MapPoint vertices
	const int N = pFrame->N;
	const int Nleft = pFrame->Nleft;
	const bool bRight = (Nleft != -1);

	vector<EdgeMonoOnlyPose *> vpEdgesMono;
	vector<EdgeStereoOnlyPose *> vpEdgesStereo;
	vector<size_t> vnIndexEdgeMono;
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesMono.reserve(N);
	vpEdgesStereo.reserve(N);
	vnIndexEdgeMono.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	{
		unique_lock<mutex> lock(MapPoint::mGlobalMutex);

		for (int i = 0; i < N; i++) {
			MapPoint *pMP = pFrame->mvpMapPoints[i];
			if (pMP) {
				cv::KeyPoint kpUn;
				// Left monocular observation
				if ((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft) {
					if (i < Nleft) { // pair left-right
						kpUn = pFrame->mvKeys[i];
					}
					else {
						kpUn = pFrame->mvKeysUn[i];
					}

					nInitialMonoCorrespondences++;
					pFrame->mvbOutlier[i] = false;

					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					EdgeMonoOnlyPose *e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 0);

					e->setVertex(0, VP);
					e->setMeasurement(obs);

					// Add here uncerteinty
					const float unc2 = pFrame->mpCamera->uncertainty2(obs);

					const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					optimizer.addEdge(e);

					vpEdgesMono.push_back(e);
					vnIndexEdgeMono.push_back(i);
				}
					// Stereo observation
				else if (!bRight) {
					nInitialStereoCorrespondences++;
					pFrame->mvbOutlier[i] = false;

					kpUn = pFrame->mvKeysUn[i];
					const float kp_ur = pFrame->mvuRight[i];
					Eigen::Matrix<double, 3, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

					EdgeStereoOnlyPose *e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

					e->setVertex(0, VP);
					e->setMeasurement(obs);

					// Add here uncerteinty
					const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

					const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
					e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberStereo);

					optimizer.addEdge(e);

					vpEdgesStereo.push_back(e);
					vnIndexEdgeStereo.push_back(i);
				}

				// Right monocular observation
				if (bRight && i >= Nleft) {
					nInitialMonoCorrespondences++;
					pFrame->mvbOutlier[i] = false;

					kpUn = pFrame->mvKeysRight[i - Nleft];
					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					EdgeMonoOnlyPose *e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 1);

					e->setVertex(0, VP);
					e->setMeasurement(obs);

					// Add here uncerteinty
					const float unc2 = pFrame->mpCamera->uncertainty2(obs);

					const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					optimizer.addEdge(e);

					vpEdgesMono.push_back(e);
					vnIndexEdgeMono.push_back(i);
				}
			}
		}
	}

	nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

	// Set Previous Frame Vertex
	Frame *pFp = pFrame->mpPrevFrame;

	VertexPose *VPk = new VertexPose(pFp);
	VPk->setId(4);
	VPk->setFixed(false);
	optimizer.addVertex(VPk);
	VertexVelocity *VVk = new VertexVelocity(pFp);
	VVk->setId(5);
	VVk->setFixed(false);
	optimizer.addVertex(VVk);
	VertexGyroBias *VGk = new VertexGyroBias(pFp);
	VGk->setId(6);
	VGk->setFixed(false);
	optimizer.addVertex(VGk);
	VertexAccBias *VAk = new VertexAccBias(pFp);
	VAk->setId(7);
	VAk->setFixed(false);
	optimizer.addVertex(VAk);

	EdgeInertial *ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

	ei->setVertex(0, VPk);
	ei->setVertex(1, VVk);
	ei->setVertex(2, VGk);
	ei->setVertex(3, VAk);
	ei->setVertex(4, VP);
	ei->setVertex(5, VV);
	optimizer.addEdge(ei);

	EdgeGyroRW *egr = new EdgeGyroRW();
	egr->setVertex(0, VGk);
	egr->setVertex(1, VG);
	cv::Mat cvInfoG = pFrame->mpImuPreintegratedFrame->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
	Eigen::Matrix3d InfoG;
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			InfoG(r, c) = cvInfoG.at<float>(r, c);
	egr->setInformation(InfoG);
	optimizer.addEdge(egr);

	EdgeAccRW *ear = new EdgeAccRW();
	ear->setVertex(0, VAk);
	ear->setVertex(1, VA);
	cv::Mat cvInfoA = pFrame->mpImuPreintegratedFrame->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
	Eigen::Matrix3d InfoA;
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			InfoA(r, c) = cvInfoA.at<float>(r, c);
	ear->setInformation(InfoA);
	optimizer.addEdge(ear);

	if (!pFp->mpcpi) {
		Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId),
		                   Verbose::VERBOSITY_NORMAL);
	}

	EdgePriorPoseImu *ep = new EdgePriorPoseImu(pFp->mpcpi);

	ep->setVertex(0, VPk);
	ep->setVertex(1, VVk);
	ep->setVertex(2, VGk);
	ep->setVertex(3, VAk);
	g2o::RobustKernelHuber *rkp = new g2o::RobustKernelHuber;
	ep->setRobustKernel(rkp);
	rkp->setDelta(5);
	optimizer.addEdge(ep);

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.

	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
	const int its[4] = {10, 10, 10, 10};

	int nBad = 0;
	int nBadMono = 0;
	int nBadStereo = 0;
	int nInliersMono = 0;
	int nInliersStereo = 0;
	int nInliers = 0;
	for (size_t it = 0; it < 4; it++) {
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;
		nBadMono = 0;
		nBadStereo = 0;
		nInliers = 0;
		nInliersMono = 0;
		nInliersStereo = 0;
		float chi2close = 1.5 * chi2Mono[it];

		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			EdgeMonoOnlyPose *e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];
			bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.f;

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if ((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close) || !e->isDepthPositive()) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBadMono++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
				nInliersMono++;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			EdgeStereoOnlyPose *e = vpEdgesStereo[i];

			const size_t idx = vnIndexEdgeStereo[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Stereo[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBadStereo++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
				nInliersStereo++;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		nInliers = nInliersMono + nInliersStereo;
		nBad = nBadMono + nBadStereo;

		if (optimizer.edges().size() < 10) {
			cout << "PIOLF: NOT ENOUGH EDGES" << endl;
			break;
		}
	}

	if ((nInliers < 30) && !bRecInit) {
		nBad = 0;
		const float chi2MonoOut = 18.f;
		const float chi2StereoOut = 24.f;
		EdgeMonoOnlyPose *e1;
		EdgeStereoOnlyPose *e2;
		for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++) {
			const size_t idx = vnIndexEdgeMono[i];
			e1 = vpEdgesMono[i];
			e1->computeError();
			if (e1->chi2() < chi2MonoOut) {
				pFrame->mvbOutlier[idx] = false;
			}
			else {
				nBad++;
			}
		}
		for (size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++) {
			const size_t idx = vnIndexEdgeStereo[i];
			e2 = vpEdgesStereo[i];
			e2->computeError();
			if (e2->chi2() < chi2StereoOut) {
				pFrame->mvbOutlier[idx] = false;
			}
			else {
				nBad++;
			}
		}
	}

	nInliers = nInliersMono + nInliersStereo;

	// Recover optimized pose, velocity and biases
	pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb),
	                           Converter::toCvMat(VP->estimate().twb),
	                           Converter::toCvMat(VV->estimate()));
	Vector6d b;
	b << VG->estimate(), VA->estimate();
	pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

	// Recover Hessian, marginalize previous frame states and generate new prior for frame
	Eigen::Matrix<double, 30, 30> H;
	H.setZero();

	H.block<24, 24>(0, 0) += ei->GetHessian();

	Eigen::Matrix<double, 6, 6> Hgr = egr->GetHessian();
	H.block<3, 3>(9, 9) += Hgr.block<3, 3>(0, 0);
	H.block<3, 3>(9, 24) += Hgr.block<3, 3>(0, 3);
	H.block<3, 3>(24, 9) += Hgr.block<3, 3>(3, 0);
	H.block<3, 3>(24, 24) += Hgr.block<3, 3>(3, 3);

	Eigen::Matrix<double, 6, 6> Har = ear->GetHessian();
	H.block<3, 3>(12, 12) += Har.block<3, 3>(0, 0);
	H.block<3, 3>(12, 27) += Har.block<3, 3>(0, 3);
	H.block<3, 3>(27, 12) += Har.block<3, 3>(3, 0);
	H.block<3, 3>(27, 27) += Har.block<3, 3>(3, 3);

	H.block<15, 15>(0, 0) += ep->GetHessian();

	int tot_in = 0, tot_out = 0;
	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
		EdgeMonoOnlyPose *e = vpEdgesMono[i];

		const size_t idx = vnIndexEdgeMono[i];

		if (!pFrame->mvbOutlier[idx]) {
			H.block<6, 6>(15, 15) += e->GetHessian();
			tot_in++;
		}
		else {
			tot_out++;
		}
	}

	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
		EdgeStereoOnlyPose *e = vpEdgesStereo[i];

		const size_t idx = vnIndexEdgeStereo[i];

		if (!pFrame->mvbOutlier[idx]) {
			H.block<6, 6>(15, 15) += e->GetHessian();
			tot_in++;
		}
		else {
			tot_out++;
		}
	}

	H = Marginalize(H, 0, 14);

	pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,
	                                      VP->estimate().twb,
	                                      VV->estimate(),
	                                      VG->estimate(),
	                                      VA->estimate(),
	                                      H.block<15, 15>(15, 15));
	delete pFp->mpcpi;
	pFp->mpcpi = NULL;

	return nInitialCorrespondences - nBad;
}

int Optimizer::PoseDvlGyrosOPtimizationLastFrame(Frame *pFrame, double lamda_DVL, bool bRecInit)
{
	g2o::SparseOptimizer optimizer;
//	g2o::BlockSolverX::LinearSolverType *linearSolver;

//	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
//
//	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

//	g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	solver->setUserLambdaInit(100);
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);

	int nInitialMonoCorrespondences = 0;
	int nInitialStereoCorrespondences = 0;
	int nInitialCorrespondences = 0;

	// Set Current Frame vertex
	VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pFrame);
	VP->setId(0);
	VP->setFixed(false);
	optimizer.addVertex(VP);

	VertexGyroBias *VG = new VertexGyroBias(pFrame);
	VG->setId(2);
	VG->setFixed(true);
	optimizer.addVertex(VG);

	g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
	vT_d_c->setEstimate(Converter::toSE3Quat(pFrame->GetExtrinsicParamters().mT_dvl_c));
	vT_d_c->setId(3);
	vT_d_c->setFixed(true);
	optimizer.addVertex(vT_d_c);

	g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
	vT_g_d->setEstimate(Converter::toSE3Quat(pFrame->GetExtrinsicParamters().mT_imu_dvl));
	vT_g_d->setId(4);
	vT_g_d->setFixed(true);
	optimizer.addVertex(vT_g_d);

	// Set MapPoint vertices
	const int N = pFrame->N;
	const int Nleft = pFrame->Nleft;
	const bool bRight = (Nleft != -1);

	vector<EdgeMonoOnlyPose_DvlGyros *> vpEdgesMono;
	vector<EdgeStereoOnlyPose_DvlGyros *> vpEdgesStereo;
	vector<size_t> vnIndexEdgeMono;
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesMono.reserve(N);
	vpEdgesStereo.reserve(N);
	vnIndexEdgeMono.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	// set visual constains
	{
		unique_lock<mutex> lock(MapPoint::mGlobalMutex);

		for (int i = 0; i < N; i++) {
			MapPoint *pMP = pFrame->mvpMapPoints[i];
			if (pMP) {

				// Left monocular observation
//				if ((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft) {
				if (!pFrame->mpCamera2) {

					if (pFrame->mvuRight[i] < 0) {

						nInitialMonoCorrespondences++;
						pFrame->mvbOutlier[i] = false;

						Eigen::Matrix<double, 2, 1> obs;
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						obs << kpUn.pt.x, kpUn.pt.y;

						EdgeMonoOnlyPose_DvlGyros *e = new EdgeMonoOnlyPose_DvlGyros(pMP->GetWorldPos(), 0);

						e->setVertex(0, VP);
						e->setMeasurement(obs);

						// Add here uncerteinty
						const float unc2 = pFrame->mpCamera->uncertainty2(obs);

						// octave (pyramid layer) from which the keypoint has been extracted
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(thHuberMono);

						optimizer.addEdge(e);

						vpEdgesMono.push_back(e);
						vnIndexEdgeMono.push_back(i);
					}
					else // Stereo observation
					{
						nInitialStereoCorrespondences++;
						pFrame->mvbOutlier[i] = false;
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						const float kp_ur = pFrame->mvuRight[i];
						Eigen::Matrix<double, 3, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

						EdgeStereoOnlyPose_DvlGyros *e = new EdgeStereoOnlyPose_DvlGyros(pMP->GetWorldPos());

						e->setVertex(0, VP);
						e->setMeasurement(obs);

						// Add here uncerteinty
						const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

						const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
						e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(thHuberStereo);

						optimizer.addEdge(e);

						vpEdgesStereo.push_back(e);
						vnIndexEdgeStereo.push_back(i);
					}

				}
			}
		}
	}

	nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;


	//todo_tightly
	//	use prior information to contrain pose of previous frame
	//	but do not understand the propogation of uncertainty(Hessian Matrix)
	//  just set it as fixed for now
	Frame *pFp = pFrame->mpPrevFrame;
	VertexPoseDvlIMU *VP2 = new VertexPoseDvlIMU(pFp);
	VP2->setId(1);
	VP2->setFixed(true);
	optimizer.addVertex(VP2);

	// set DVL_Gyros constrain
	//todo_tightly
	//	maybe add velocity to optimization
	EdgeDvlGyroTrack *ei = new EdgeDvlGyroTrack(pFrame->mpDvlPreintegrationFrame);

	ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP));
	ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
	ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vT_d_c));
	ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vT_g_d));
	ei->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * lamda_DVL*(vpEdgesStereo.size()+vpEdgesStereo.size()));
	ei->setId(pFrame->mnId);
	optimizer.addEdge(ei);


	if (!pFp->mpcpi) {
		Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId),
		                   Verbose::VERBOSITY_NORMAL);
	}
//	EdgePriorPoseImu *ep = new EdgePriorPoseImu(pFp->mpcpi);
//	ep->setVertex(0, VPk);
//	ep->setVertex(1, VVk);
//	ep->setVertex(2, VGk);
//	ep->setVertex(3, VAk);
//	g2o::RobustKernelHuber *rkp = new g2o::RobustKernelHuber;
//	ep->setRobustKernel(rkp);
//	rkp->setDelta(5);
//	optimizer.addEdge(ep);

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.

	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
	const int its[4] = {10, 10, 10, 10};

	int nBad = 0;
	int nBadMono = 0;
	int nBadStereo = 0;
	int nInliersMono = 0;
	int nInliersStereo = 0;
	int nInliers = 0;
	for (size_t it = 0; it < 4; it++) {
//		cout<<"optimization iteration: "<<it<<endl;
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;
		nBadMono = 0;
		nBadStereo = 0;
		nInliers = 0;
		nInliersMono = 0;
		nInliersStereo = 0;
		float chi2close = 1.5 * chi2Mono[it];

		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			EdgeMonoOnlyPose_DvlGyros *e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];
			bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.f;

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if ((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close)) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBadMono++;
//				cout<<"outlier edge id:"<<e->id()<<" chi2: "<<chi2<<endl;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
				nInliersMono++;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			EdgeStereoOnlyPose_DvlGyros *e = vpEdgesStereo[i];

			const size_t idx = vnIndexEdgeStereo[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Stereo[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBadStereo++;
//				cout<<"outlier edge id:"<<e->id()<<" chi2: "<<chi2<<endl;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
				nInliersStereo++;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		nInliers = nInliersMono + nInliersStereo;
		nBad = nBadMono + nBadStereo;

// //		ROS_INFO_STREAM("track local map inlier: "<<nInliers<<" outlier: "<<nBad);  // original
//		cout<<"inlier map points: "<<nInliers<<endl;
//		cout<<"outlier map points: "<<nBad<<endl;

		if (optimizer.edges().size() < 10) {
			cout << "PIOLF: NOT ENOUGH EDGES" << endl;
			break;
		}
	}

	if ((nInliers < 30) && !bRecInit) {
		nBad = 0;
		const float chi2MonoOut = 18.f;
		const float chi2StereoOut = 24.f;
		EdgeMonoOnlyPose_DvlGyros *e1;
		EdgeStereoOnlyPose_DvlGyros *e2;
		for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++) {
			const size_t idx = vnIndexEdgeMono[i];
			e1 = vpEdgesMono[i];
			e1->computeError();
			if (e1->chi2() < chi2MonoOut) {
				pFrame->mvbOutlier[idx] = false;
			}
			else {
				nBad++;
			}
		}
		for (size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++) {
			const size_t idx = vnIndexEdgeStereo[i];
			e2 = vpEdgesStereo[i];
			e2->computeError();
			if (e2->chi2() < chi2StereoOut) {
				pFrame->mvbOutlier[idx] = false;
			}
			else {
				nBad++;
			}
		}
	}

	nInliers = nInliersMono + nInliersStereo;

	if (nInliers > 30) {
		// Recover optimized pose, velocity and biases
//	pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb),
//							   Converter::toCvMat(VP->estimate().twb),
//							   Converter::toCvMat(VV->estimate()));
		Eigen::Matrix3d R_w_g = VP->estimate().Rwc * VP->estimate().R_c_imu[0];
		Eigen::Vector3d t_w_d = VP->estimate().twc + VP->estimate().Rwc * VP->estimate().t_c_dvl[0];
		Eigen::Isometry3d T_c_w = Eigen::Isometry3d::Identity();
		T_c_w.pretranslate(VP->estimate().tcw[0]);
		T_c_w.rotate(VP->estimate().Rcw[0]);
		cv::Mat R_w_g_cv;
		cv::eigen2cv(R_w_g, R_w_g_cv);
		R_w_g_cv.convertTo(R_w_g_cv, CV_32F);
		cv::Mat T_c_w_cv;
		cv::eigen2cv(T_c_w.matrix(), T_c_w_cv);
		T_c_w_cv.convertTo(T_c_w_cv, CV_32F);
		pFrame->SetPose(T_c_w_cv);
//		pFrame->SetDvlPoseVelocity(Converter::toCvMat(R_w_g),
//								   Converter::toCvMat(t_w_d),
//								   pFrame->GetDvlVelocity());

		Vector6d b;
		b << VG->estimate(), 0, 0, 0;
		pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);
	}




	return nInitialCorrespondences - nBad;
}

int Optimizer::PoseDvlGyrosOPtimizationLastKeyFrame(Frame *pFrame, double lamda_DVL, bool bRecInit)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

//	g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(false);

	int nInitialMonoCorrespondences = 0;
	int nInitialStereoCorrespondences = 0;
	int nInitialCorrespondences = 0;

	// Set Current Frame vertex
	VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pFrame);
	VP->setId(0);
	VP->setFixed(false);
	optimizer.addVertex(VP);

	VertexGyroBias *VG = new VertexGyroBias(pFrame);
	VG->setId(2);
	VG->setFixed(true);
	optimizer.addVertex(VG);

	g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
	vT_d_c->setEstimate(Converter::toSE3Quat(pFrame->GetExtrinsicParamters().mT_dvl_c));
	vT_d_c->setId(3);
	vT_d_c->setFixed(true);
	optimizer.addVertex(vT_d_c);

	g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
	vT_g_d->setEstimate(Converter::toSE3Quat(pFrame->GetExtrinsicParamters().mT_imu_dvl));
	vT_g_d->setId(4);
	vT_g_d->setFixed(true);
	optimizer.addVertex(vT_g_d);

	// Set MapPoint vertices
	const int N = pFrame->N;
	const int Nleft = pFrame->Nleft;
	const bool bRight = (Nleft != -1);

	vector<EdgeMonoOnlyPose_DvlGyros *> vpEdgesMono;
	vector<EdgeStereoOnlyPose_DvlGyros *> vpEdgesStereo;
	vector<size_t> vnIndexEdgeMono;
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesMono.reserve(N);
	vpEdgesStereo.reserve(N);
	vnIndexEdgeMono.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	// set visual constains
	{
		unique_lock<mutex> lock(MapPoint::mGlobalMutex);

		for (int i = 0; i < N; i++) {
			MapPoint *pMP = pFrame->mvpMapPoints[i];
			if (pMP) {

				// Left monocular observation
//				if ((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft) {
				if (!pFrame->mpCamera2) {

					if (pFrame->mvuRight[i] < 0) {

						nInitialMonoCorrespondences++;
						pFrame->mvbOutlier[i] = false;

						Eigen::Matrix<double, 2, 1> obs;
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						obs << kpUn.pt.x, kpUn.pt.y;

						EdgeMonoOnlyPose_DvlGyros *e = new EdgeMonoOnlyPose_DvlGyros(pMP->GetWorldPos(), 0);

						e->setVertex(0, VP);
						e->setMeasurement(obs);

						// Add here uncerteinty
						const float unc2 = pFrame->mpCamera->uncertainty2(obs);

						// octave (pyramid layer) from which the keypoint has been extracted
						const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(thHuberMono);

						optimizer.addEdge(e);

						vpEdgesMono.push_back(e);
						vnIndexEdgeMono.push_back(i);
					}
					else // Stereo observation
					{
						nInitialStereoCorrespondences++;
						pFrame->mvbOutlier[i] = false;
						const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
						const float kp_ur = pFrame->mvuRight[i];
						Eigen::Matrix<double, 3, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

						EdgeStereoOnlyPose_DvlGyros *e = new EdgeStereoOnlyPose_DvlGyros(pMP->GetWorldPos());

						e->setVertex(0, VP);
						e->setMeasurement(obs);

						// Add here uncerteinty
						const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

						const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
						e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

						g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
						rk->setDelta(thHuberStereo);

						optimizer.addEdge(e);

						vpEdgesStereo.push_back(e);
						vnIndexEdgeStereo.push_back(i);
					}

				}
			}
		}
	}

	nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;


	KeyFrame *pFK = pFrame->mpLastKeyFrame;
	VertexPoseDvlIMU *VP2 = new VertexPoseDvlIMU(pFK);
	VP2->setId(1);
	VP2->setFixed(true);
	optimizer.addVertex(VP2);

	// set DVL_Gyros constrain
	//todo_tightly
	//	maybe add velocity to optimization
	EdgeDvlGyroTrack *ei = new EdgeDvlGyroTrack(pFrame->mpDvlPreintegrationKeyFrame);

	ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP));
	ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
	ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vT_d_c));
	ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vT_g_d));
	ei->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * lamda_DVL*(vpEdgesStereo.size()+vpEdgesMono.size()));
	ei->setId(pFrame->mnId);
	optimizer.addEdge(ei);


//	if (!pFK->mpcpi) {
//		Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFK->mnId),
//						   Verbose::VERBOSITY_NORMAL);
//	}
//	EdgePriorPoseImu *ep = new EdgePriorPoseImu(pFp->mpcpi);
//	ep->setVertex(0, VPk);
//	ep->setVertex(1, VVk);
//	ep->setVertex(2, VGk);
//	ep->setVertex(3, VAk);
//	g2o::RobustKernelHuber *rkp = new g2o::RobustKernelHuber;
//	ep->setRobustKernel(rkp);
//	rkp->setDelta(5);
//	optimizer.addEdge(ep);

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.

	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
	const int its[4] = {10, 10, 10, 10};

	int nBad = 0;
	int nBadMono = 0;
	int nBadStereo = 0;
	int nInliersMono = 0;
	int nInliersStereo = 0;
	int nInliers = 0;
	for (size_t it = 0; it < 4; it++) {
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;
		nBadMono = 0;
		nBadStereo = 0;
		nInliers = 0;
		nInliersMono = 0;
		nInliersStereo = 0;
		float chi2close = 1.5 * chi2Mono[it];

		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
			EdgeMonoOnlyPose_DvlGyros *e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];
			bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.f;

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if ((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close)) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBadMono++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
				nInliersMono++;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
			EdgeStereoOnlyPose_DvlGyros *e = vpEdgesStereo[i];

			const size_t idx = vnIndexEdgeStereo[i];

			if (pFrame->mvbOutlier[idx]) {
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2 > chi2Stereo[it]) {
				pFrame->mvbOutlier[idx] = true;
				e->setLevel(1);
				nBadStereo++;
			}
			else {
				pFrame->mvbOutlier[idx] = false;
				e->setLevel(0);
				nInliersStereo++;
			}

			if (it == 2) {
				e->setRobustKernel(0);
			}
		}

		nInliers = nInliersMono + nInliersStereo;
		nBad = nBadMono + nBadStereo;

		if (optimizer.edges().size() < 10) {
			cout << "PIOLF: NOT ENOUGH EDGES" << endl;
			break;
		}
	}

	if ((nInliers < 30) && !bRecInit) {
		nBad = 0;
		const float chi2MonoOut = 18.f;
		const float chi2StereoOut = 24.f;
		EdgeMonoOnlyPose_DvlGyros *e1;
		EdgeStereoOnlyPose_DvlGyros *e2;
		for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++) {
			const size_t idx = vnIndexEdgeMono[i];
			e1 = vpEdgesMono[i];
			e1->computeError();
			if (e1->chi2() < chi2MonoOut) {
				pFrame->mvbOutlier[idx] = false;
			}
			else {
				nBad++;
			}
		}
		for (size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++) {
			const size_t idx = vnIndexEdgeStereo[i];
			e2 = vpEdgesStereo[i];
			e2->computeError();
			if (e2->chi2() < chi2StereoOut) {
				pFrame->mvbOutlier[idx] = false;
			}
			else {
				nBad++;
			}
		}
	}

	nInliers = nInliersMono + nInliersStereo;

	// Recover optimized pose, velocity and biases
//	pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb),
//							   Converter::toCvMat(VP->estimate().twb),
//							   Converter::toCvMat(VV->estimate()));
	Eigen::Matrix3d R_w_g = VP->estimate().Rwc * VP->estimate().R_c_imu[0];
	Eigen::Vector3d t_w_d = VP->estimate().twc + VP->estimate().Rwc * VP->estimate().t_c_dvl[0];
	Eigen::Isometry3d T_c_w = Eigen::Isometry3d::Identity();
	T_c_w.pretranslate(VP->estimate().tcw[0]);
	T_c_w.rotate(VP->estimate().Rcw[0]);
	cv::Mat R_w_g_cv;
	cv::eigen2cv(R_w_g, R_w_g_cv);
	R_w_g_cv.convertTo(R_w_g_cv, CV_32F);
	cv::Mat T_c_w_cv;
	cv::eigen2cv(T_c_w.matrix(), T_c_w_cv);
	T_c_w_cv.convertTo(T_c_w_cv, CV_32F);
	pFrame->SetPose(T_c_w_cv);
//	pFrame->SetDvlPoseVelocity(Converter::toCvMat(R_w_g),
//							   Converter::toCvMat(t_w_d),
//							   pFrame->GetDvlVelocity());
	Vector6d b;
	b << VG->estimate(), 0, 0, 0;
	pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

	//todo_tightly
	// do not understand how to get Hessian matrix
	// Recover Hessian, marginalize previous frame states and generate new prior for frame
//	Eigen::Matrix<double, 30, 30> H;
//	H.setZero();
//
//	H.block<24, 24>(0, 0) += ei->GetHessian();
//
//	Eigen::Matrix<double, 6, 6> Hgr = egr->GetHessian();
//	H.block<3, 3>(9, 9) += Hgr.block<3, 3>(0, 0);
//	H.block<3, 3>(9, 24) += Hgr.block<3, 3>(0, 3);
//	H.block<3, 3>(24, 9) += Hgr.block<3, 3>(3, 0);
//	H.block<3, 3>(24, 24) += Hgr.block<3, 3>(3, 3);
//
//	Eigen::Matrix<double, 6, 6> Har = ear->GetHessian();
//	H.block<3, 3>(12, 12) += Har.block<3, 3>(0, 0);
//	H.block<3, 3>(12, 27) += Har.block<3, 3>(0, 3);
//	H.block<3, 3>(27, 12) += Har.block<3, 3>(3, 0);
//	H.block<3, 3>(27, 27) += Har.block<3, 3>(3, 3);
//
//	H.block<15, 15>(0, 0) += ep->GetHessian();
//
//	int tot_in = 0, tot_out = 0;
//	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
//		EdgeMonoOnlyPose *e = vpEdgesMono[i];
//
//		const size_t idx = vnIndexEdgeMono[i];
//
//		if (!pFrame->mvbOutlier[idx]) {
//			H.block<6, 6>(15, 15) += e->GetHessian();
//			tot_in++;
//		}
//		else {
//			tot_out++;
//		}
//	}
//
//	for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
//		EdgeStereoOnlyPose *e = vpEdgesStereo[i];
//
//		const size_t idx = vnIndexEdgeStereo[i];
//
//		if (!pFrame->mvbOutlier[idx]) {
//			H.block<6, 6>(15, 15) += e->GetHessian();
//			tot_in++;
//		}
//		else {
//			tot_out++;
//		}
//	}
//
//	H = Marginalize(H, 0, 14);
//
//	pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,
//										  VP->estimate().twb,
//										  VV->estimate(),
//										  VG->estimate(),
//										  VA->estimate(),
//										  H.block<15, 15>(15, 15));
//	delete pFp->mpcpi;
//	pFp->mpcpi = NULL;

	return nInitialCorrespondences - nBad;
}

void Optimizer::OptimizeEssentialGraph4DoF(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                           const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                           const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                           const map<KeyFrame *, set<KeyFrame *>> &LoopConnections)
{
	typedef g2o::BlockSolver<g2o::BlockSolverTraits<4, 4>> BlockSolver_4_4;

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	g2o::BlockSolverX::LinearSolverType *linearSolver =
		new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	optimizer.setAlgorithm(solver);

	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
	const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

	const unsigned int nMaxKFid = pMap->GetMaxKFid();

	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);

	vector<VertexPose4DoF *> vpVertices(nMaxKFid + 1);

	const int minFeat = 100;
	// Set KeyFrame vertices
	for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
		KeyFrame *pKF = vpKFs[i];
		if (pKF->isBad()) {
			continue;
		}

		VertexPose4DoF *V4DoF;

		const int nIDi = pKF->mnId;

		LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

		if (it != CorrectedSim3.end()) {
			vScw[nIDi] = it->second;
			const g2o::Sim3 Swc = it->second.inverse();
			Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
			Eigen::Vector3d twc = Swc.translation();
			V4DoF = new VertexPose4DoF(Rwc, twc, pKF);
		}
		else {
			Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
			Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
			g2o::Sim3 Siw(Rcw, tcw, 1.0);
			vScw[nIDi] = Siw;
			V4DoF = new VertexPose4DoF(pKF);
		}

		if (pKF == pLoopKF) {
			V4DoF->setFixed(true);
		}

		V4DoF->setId(nIDi);
		V4DoF->setMarginalized(false);

		optimizer.addVertex(V4DoF);
		vpVertices[nIDi] = V4DoF;
	}
	cout << "PoseGraph4DoF: KFs loaded" << endl;

	set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

	// Edge used in posegraph has still 6Dof, even if updates of camera poses are just in 4DoF
	Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();
	matLambda(0, 0) = 1e3;
	matLambda(1, 1) = 1e3;
	matLambda(0, 0) = 1e3;

	// Set Loop edges
	Edge4DoF *e_loop;
	for (map<KeyFrame *, set<KeyFrame *>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end();
	     mit != mend; mit++) {
		KeyFrame *pKF = mit->first;
		const long unsigned int nIDi = pKF->mnId;
		const set<KeyFrame *> &spConnections = mit->second;
		const g2o::Sim3 Siw = vScw[nIDi];
		const g2o::Sim3 Swi = Siw.inverse();

		for (set<KeyFrame *>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send;
		     sit++) {
			const long unsigned int nIDj = (*sit)->mnId;
			if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat) {
				continue;
			}

			const g2o::Sim3 Sjw = vScw[nIDj];
			const g2o::Sim3 Sij = Siw * Sjw.inverse();
			Eigen::Matrix4d Tij;
			Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
			Tij.block<3, 1>(0, 3) = Sij.translation();
			Tij(3, 3) = 1.;

			Edge4DoF *e = new Edge4DoF(Tij);
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));

			e->information() = matLambda;
			e_loop = e;
			optimizer.addEdge(e);

			sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
		}
	}
	cout << "PoseGraph4DoF: Loop edges loaded" << endl;

	// 1. Set normal edges
	for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
		KeyFrame *pKF = vpKFs[i];

		const int nIDi = pKF->mnId;

		g2o::Sim3 Siw;

		// Use noncorrected poses for posegraph edges
		LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

		if (iti != NonCorrectedSim3.end()) {
			Siw = iti->second;
		}
		else {
			Siw = vScw[nIDi];
		}

		// 1.1.0 Spanning tree edge
		KeyFrame *pParentKF = static_cast<KeyFrame *>(NULL);
		if (pParentKF) {
			int nIDj = pParentKF->mnId;

			g2o::Sim3 Swj;

			LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

			if (itj != NonCorrectedSim3.end()) {
				Swj = (itj->second).inverse();
			}
			else {
				Swj = vScw[nIDj].inverse();
			}

			g2o::Sim3 Sij = Siw * Swj;
			Eigen::Matrix4d Tij;
			Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
			Tij.block<3, 1>(0, 3) = Sij.translation();
			Tij(3, 3) = 1.;

			Edge4DoF *e = new Edge4DoF(Tij);
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
			e->information() = matLambda;
			optimizer.addEdge(e);
		}

		// 1.1.1 Inertial edges
		KeyFrame *prevKF = pKF->mPrevKF;
		if (prevKF) {
			int nIDj = prevKF->mnId;

			g2o::Sim3 Swj;

			LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(prevKF);

			if (itj != NonCorrectedSim3.end()) {
				Swj = (itj->second).inverse();
			}
			else {
				Swj = vScw[nIDj].inverse();
			}

			g2o::Sim3 Sij = Siw * Swj;
			Eigen::Matrix4d Tij;
			Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
			Tij.block<3, 1>(0, 3) = Sij.translation();
			Tij(3, 3) = 1.;

			Edge4DoF *e = new Edge4DoF(Tij);
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
			e->information() = matLambda;
			optimizer.addEdge(e);
		}

		// 1.2 Loop edges
		const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
		for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++) {
			KeyFrame *pLKF = *sit;
			if (pLKF->mnId < pKF->mnId) {
				g2o::Sim3 Swl;

				LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

				if (itl != NonCorrectedSim3.end()) {
					Swl = itl->second.inverse();
				}
				else {
					Swl = vScw[pLKF->mnId].inverse();
				}

				g2o::Sim3 Sil = Siw * Swl;
				Eigen::Matrix4d Til;
				Til.block<3, 3>(0, 0) = Sil.rotation().toRotationMatrix();
				Til.block<3, 1>(0, 3) = Sil.translation();
				Til(3, 3) = 1.;

				Edge4DoF *e = new Edge4DoF(Til);
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
				e->information() = matLambda;
				optimizer.addEdge(e);
			}
		}

		// 1.3 Covisibility graph edges
		const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
		for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++) {
			KeyFrame *pKFn = *vit;
			if (pKFn && pKFn != pParentKF && pKFn != prevKF && pKFn != pKF->mNextKF && !pKF->hasChild(pKFn)
				&& !sLoopEdges.count(pKFn)) {
				if (!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
					if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId)))) {
						continue;
					}

					g2o::Sim3 Swn;

					LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

					if (itn != NonCorrectedSim3.end()) {
						Swn = itn->second.inverse();
					}
					else {
						Swn = vScw[pKFn->mnId].inverse();
					}

					g2o::Sim3 Sin = Siw * Swn;
					Eigen::Matrix4d Tin;
					Tin.block<3, 3>(0, 0) = Sin.rotation().toRotationMatrix();
					Tin.block<3, 1>(0, 3) = Sin.translation();
					Tin(3, 3) = 1.;
					Edge4DoF *e = new Edge4DoF(Tin);
					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
					e->information() = matLambda;
					optimizer.addEdge(e);
				}
			}
		}
	}
	cout << "PoseGraph4DoF: Covisibility edges loaded" << endl;

	optimizer.initializeOptimization();
	optimizer.computeActiveErrors();
	optimizer.optimize(20);

	unique_lock<shared_timed_mutex> lock(pMap->mMutexMapUpdate);

	// SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		const int nIDi = pKFi->mnId;

		VertexPose4DoF *Vi = static_cast<VertexPose4DoF *>(optimizer.vertex(nIDi));
		Eigen::Matrix3d Ri = Vi->estimate().Rcw[0];
		Eigen::Vector3d ti = Vi->estimate().tcw[0];

		g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri, ti, 1.);
		vCorrectedSwc[nIDi] = CorrectedSiw.inverse();

		cv::Mat Tiw = Converter::toCvSE3(Ri, ti);
		pKFi->SetPose(Tiw);
	}

	// Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
	for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
		MapPoint *pMP = vpMPs[i];

		if (pMP->isBad()) {
			continue;
		}

		int nIDr;

		KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
		nIDr = pRefKF->mnId;

		g2o::Sim3 Srw = vScw[nIDr];
		g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

		cv::Mat P3Dw = pMP->GetWorldPos();
		Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
		Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

		cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
		pMP->SetWorldPos(cvCorrectedP3Dw);

		pMP->UpdateNormalAndDepth();
	}
	pMap->IncreaseChangeIndex();
}

void Optimizer::DvlGyroInitOptimization(Map *pMap,
                                        Eigen::Vector3d &bg,
                                        bool bMono,
                                        float priorG)
{
	Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	if (priorG != 0.f) {
		solver->setUserLambdaInit(1e3);
	}

	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);
	}

	// Biases
	//todo_tightly
	//	set fixed for debuging
	VertexGyroBias *VG = new VertexGyroBias(vpKFs.front());
	VG->setId(maxKFid + 1);
	VG->setFixed(true);
	optimizer.addVertex(VG);

	// prior acc bias
//		EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
//		epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
//		double infoPriorG = priorG;
//		epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
//		optimizer.addEdge(epg);

	// extrinsic parameter
	//todo_tightly
	//	set fixed for debuging
	g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
	vT_d_c->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_dvl_c));
	vT_d_c->setId(maxKFid + 2);
	vT_d_c->setFixed(false);
	optimizer.addVertex(vT_d_c);

	g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
	vT_g_d->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_imu_dvl));
	vT_g_d->setId(maxKFid + 3);
	vT_g_d->setFixed(true);
	optimizer.addVertex(vT_g_d);


	// Graph edges
	vector<EdgeDvlGyroInit *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	std::cout << "build optimization graph" << std::endl;

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpDvlPreintegrationKeyFrame) {
				std::cout << "Not preintegrated measurement" << std::endl;
			}

			pKFi->mpDvlPreintegrationKeyFrame->SetNewBias(pKFi->mPrevKF->GetImuBias());
			VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
//				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
//				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid + 1);
			g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex(maxKFid + 2);
			g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex(maxKFid + 3);
//				g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
//				g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
//				g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
//				cout<<"VP1: Rcw[0]"<<VP1->estimate().Rcw[0]<<endl;
//				cout<<"VP1: Rwc"<<VP1->estimate().Rwc<<endl;
//				cout<<"VP1: tcw[0]"<<VP1->estimate().tcw[0]<<endl;
//				cout<<"VP1: twc"<<VP1->estimate().twc<<endl;
//
//				cout<<"VP2: Rcw[0]"<<VP2->estimate().Rcw[0]<<endl;
//				cout<<"VP2: Rwc"<<VP2->estimate().Rwc<<endl;
//				cout<<"VP2: tcw[0]"<<VP2->estimate().tcw[0]<<endl;
//				cout<<"VP2: twc"<<VP2->estimate().twc<<endl;

			if (!VP1 || !VG || !VP2) {
				cout << "Error" << VP1 << ", " << VG << ", " << VP2 << endl;

				continue;
			}
//				EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			EdgeDvlGyroInit *ei = new EdgeDvlGyroInit(pKFi->mpDvlPreintegrationKeyFrame);
//				ei->setVertex(0, VP1);

//			g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//			ei->setRobustKernel(rk);
//			rk->setDelta(sqrt(7.815));
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
			ei->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 100000);
			ei->setId(pKFi->mnId);


			vpei.push_back(ei);

			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}

	// Compute error for different scales
	std::set<g2o::HyperGraph::Edge *> setEdges = optimizer.edges();
	double total_error = 0;

	for (vector<EdgeDvlGyroInit *>::iterator it = vpei.begin(); it != vpei.end(); it++) {
		VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>((*it)->vertex(0));
		VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>((*it)->vertex(1));
		VertexGyroBias *VG = dynamic_cast<VertexGyroBias *>((*it)->vertex(2));
		g2o::VertexSE3Expmap *VT = dynamic_cast<g2o::VertexSE3Expmap *>((*it)->vertex(3));
		const IMU::Bias b(0, 0, 0, VG->estimate()[0], VG->estimate()[1], VG->estimate()[2]);

		Eigen::Isometry3d T_dvl_c = VT->estimate();
		const Eigen::Matrix3d R_dvl_c = T_dvl_c.rotation();
		const Eigen::Matrix3d R_c_dvl = T_dvl_c.inverse().rotation();
		const Eigen::Vector3d t_dvl_c = T_dvl_c.translation();
		const Eigen::Vector3d t_c_dvl = T_dvl_c.inverse().translation();

		const Eigen::Matrix3d dR = Converter::toMatrix3d((*it)->mpInt->GetDeltaRotation(b));
		const Eigen::Vector3d dP = Converter::toVector3d((*it)->mpInt->GetDVLPosition(b));

		Eigen::Isometry3d T_di_dj_mea = Eigen::Isometry3d::Identity();
		Eigen::Isometry3d T_di_dj_est = Eigen::Isometry3d::Identity();
		T_di_dj_mea.rotate(dR);
		T_di_dj_mea.pretranslate(dP);

		Eigen::Matrix3d R_est = R_dvl_c * VP1->estimate().Rcw[0] * VP2->estimate().Rwc * R_c_dvl;
		Eigen::Vector3d t_est = (t_dvl_c - R_dvl_c * VP1->estimate().Rcw[0] * VP2->estimate().Rwc * R_c_dvl * t_dvl_c) +
			(R_dvl_c * (VP1->estimate().Rcw[0] * VP2->estimate().twc - VP1->estimate().Rcw[0] * VP1->estimate().twc));
		T_di_dj_est.rotate(R_est);
		T_di_dj_est.pretranslate(t_est);

		(*it)->computeError();
		Eigen::Matrix<double, 6, 1> error = (*it)->error();
		cout << "edge id: " << (*it)->id() << "\n"
		     //			<<"VP1:\n"
		     //			<<"tcw: "<<VP1->estimate().tcw[0].transpose()<<"twc: "<<VP1->estimate().twc.transpose()<<"\n"
		     //			<<"VP2:\n"
		     //			<<"tcw: "<<VP2->estimate().tcw[0].transpose()<<"twc: "<<VP2->estimate().twc.transpose()<<"\n"
		     //			<<"T_dvl_c:\n"
		     //			<<T_dvl_c.matrix()<<"\n"
		     //			<<"t_est1:\n"
		     //			<<(t_dvl_c - R_dvl_c * VP1->estimate().Rcw[0] * VP2->estimate().Rwc * R_c_dvl * t_dvl_c).transpose()<<"\n"
		     //			<<"t_est2:\n"
		     //			<<(R_dvl_c* (VP1->estimate().Rcw[0]*VP2->estimate().twc - VP1->estimate().Rcw[0]*VP1->estimate().twc)).transpose()<<"\n"
		     //			<<"dP: \n"
		     //			<<(*it)->mpInt->dP.t()<<"\n"
		     //			<<"dR: \n"
		     //			<<(*it)->mpInt->dR<<"\n"
		     //			<<"T_di_dj_est: \n"
		     //			<<T_di_dj_est.matrix()<<"\n"
		     //			<<"T_di_dj_mea: \n"
		     //			<<T_di_dj_mea.matrix()<<"\n"

		     //			<<"have dvl: "<<(*it)->mpInt->bDVL<<"\n"
		     << " error: " << error.transpose() << endl;
		total_error += error.transpose() * error;

	}
	cout << "total error: " << total_error << endl;

	std::cout << "start optimization" << std::endl;
	optimizer.setVerbose(true);
	optimizer.initializeOptimization();
	optimizer.optimize(its);

	std::cout << "end optimization" << std::endl;


	// Recover optimized data
	// Biases
	VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid + 1));
	Vector6d vb;
	vb << VG->estimate(), 0, 0, 0;
	bg << VG->estimate();

	vT_d_c = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(maxKFid + 2));

	Eigen::Isometry3d T_dvl_c = vT_d_c->estimate();
	Eigen::Isometry3d T_gyros_dvl = vT_g_d->estimate();

	IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);

	Eigen::Matrix3d R_gt;
	R_gt << 0, 0, 1,
		-1, 0, 0,
		0, -1, 0;
	cout << "init optimization result: \n"
	     << "bias_gyro:\n" << bg << "\n"
	     << "R_dvl_c:\n" << T_dvl_c.rotation() << "\n"
	     << "R_dvl_c(eular yaw-pitch-roll):" << T_dvl_c.rotation().eulerAngles(2, 1, 0).transpose() << "\n"
	     << "t_dvl_c:" << T_dvl_c.translation().transpose() << "\n"
	     << "R_dvl_c distance with R_gt(LogSO3()): " << LogSO3(T_dvl_c.rotation().inverse() * R_gt).transpose() << "\n"
	     << "R_gyros_dvl:\n" << T_gyros_dvl.rotation() << "\n"
	     << "R_gyros_dvl(eular yaw-pitch-roll):" << T_gyros_dvl.rotation().eulerAngles(2, 1, 0).transpose() << "\n"
	     << endl;


	cv::Mat cvbg = Converter::toCvMat(bg);

	//Keyframes velocities and biases
	std::cout << "update Keyframes biases and extrinsic paramters" << std::endl;

	cv::Mat T_dvl_c_cv, T_gyros_dvl_cv, T_gyros_c_cv;
	cv::eigen2cv(T_dvl_c.matrix(), T_dvl_c_cv);
	T_dvl_c_cv.convertTo(T_dvl_c_cv, CV_32FC1);
	cv::eigen2cv(T_gyros_dvl.matrix(), T_gyros_dvl_cv);
	T_gyros_dvl_cv.convertTo(T_gyros_dvl_cv, CV_32FC1);
	Eigen::Isometry3d T_gyros_c = T_gyros_dvl * T_dvl_c;
	cv::eigen2cv(T_gyros_c.matrix(), T_gyros_c_cv);
	T_gyros_c_cv.convertTo(T_gyros_c_cv, CV_32FC1);

	// IMU::Calib extrinsic_para(T_gyros_c_cv, T_dvl_c_cv);

	const int N = vpKFs.size();
	for (size_t i = 0; i < N; i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		pKFi->mImuCalib.SetExtrinsic(T_gyros_c_cv, T_dvl_c_cv);

		if (cv::norm(pKFi->GetGyroBias() - cvbg) > 0.01) {
			pKFi->SetNewBias(b);
			if (pKFi->mpDvlPreintegrationKeyFrame) {
				pKFi->mpDvlPreintegrationKeyFrame->ReintegrateWithVelocity();
			}
		}
		else {
			pKFi->SetNewBias(b);
		}
	}

}
void Optimizer::DvlGyroInitOptimization2(Map *pMap,
                                         Eigen::Vector3d &bg,
                                         bool bMono,
                                         float priorG)
{
	Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	if (priorG != 0.f) {
		solver->setUserLambdaInit(1e3);
	}

	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	vector<VertexGyroBias *> vpgb;
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		// Biases
		VertexGyroBias *VG = new VertexGyroBias(pKFi);
		VG->setId(maxKFid + 1 + pKFi->mnId);
		VG->setFixed(true);
		optimizer.addVertex(VG);
		vpgb.push_back(VG);
	}



	// prior acc bias
	//		EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
	//		epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	//		double infoPriorG = priorG;
	//		epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
	//		optimizer.addEdge(epg);

	// extrinsic parameter
	g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
	vT_d_c->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_dvl_c));
	vT_d_c->setId(maxKFid + 1 + maxKFid + 1);
	vT_d_c->setFixed(false);
	optimizer.addVertex(vT_d_c);

	g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
	vT_g_d->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_imu_dvl));
	vT_g_d->setId(maxKFid + 1 + maxKFid + 2);
	vT_g_d->setFixed(true);
	optimizer.addVertex(vT_g_d);


	// Graph edges
	vector<EdgeDvlGyroInit *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	std::cout << "build optimization graph" << std::endl;

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpDvlPreintegrationKeyFrame) {
				std::cout << "Not preintegrated measurement" << std::endl;
			}

			pKFi->mpDvlPreintegrationKeyFrame->SetNewBias(pKFi->mPrevKF->GetImuBias());
			VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
			//				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
			//				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid + 1 + pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex(maxKFid + 1 + maxKFid + 1);
			g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex(maxKFid + 1 + maxKFid + 2);
			//				g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
			//				g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
			//				g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
			//				cout<<"VP1: Rcw[0]"<<VP1->estimate().Rcw[0]<<endl;
			//				cout<<"VP1: Rwc"<<VP1->estimate().Rwc<<endl;
			//				cout<<"VP1: tcw[0]"<<VP1->estimate().tcw[0]<<endl;
			//				cout<<"VP1: twc"<<VP1->estimate().twc<<endl;
			//
			//				cout<<"VP2: Rcw[0]"<<VP2->estimate().Rcw[0]<<endl;
			//				cout<<"VP2: Rwc"<<VP2->estimate().Rwc<<endl;
			//				cout<<"VP2: tcw[0]"<<VP2->estimate().tcw[0]<<endl;
			//				cout<<"VP2: twc"<<VP2->estimate().twc<<endl;

			if (!VP1 || !VG || !VP2) {
				cout << "Error" << VP1 << ", " << VG << ", " << VP2 << endl;

				continue;
			}
			//				EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			EdgeDvlGyroInit *ei = new EdgeDvlGyroInit(pKFi->mpDvlPreintegrationKeyFrame);
			//				ei->setVertex(0, VP1);

			//			g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
			//			ei->setRobustKernel(rk);
			//			rk->setDelta(sqrt(7.815));
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
			ei->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 100000);
			ei->setId(pKFi->mnId);


			vpei.push_back(ei);

			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}

	// Compute error for different scales
	std::set<g2o::HyperGraph::Edge *> setEdges = optimizer.edges();
	double total_error = 0;

	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
	optimizer.setVerbose(true);

	int mono_outlier;
	int stereo_outlier;
	float total_chi2;
	for (int i = 0; i < 4; i++) {
		if (i > 0) {
			vT_g_d->setFixed(true);
		}
//		if (i>1){
//			for(auto v_gb:vpgb){
//				v_gb->setFixed(false);
//			}
//		}

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);
//		for (auto ei:vpei) {
//			ei->computeError();
//			cout << "calibration optimization iteration : " << i << "edge of " << ei->id() - 1 << " and " << ei->id()
//				 << "\nchi2:" << ei->chi2() << endl;
//		}

	}

//	std::cout << "start optimization" << std::endl;
//	optimizer.setVerbose(true);
//	optimizer.initializeOptimization();
//	optimizer.optimize(its);
//
//	std::cout << "end optimization" << std::endl;


	// Recover optimized data
	// Biases
	for (auto pkf: vpKFs) {
		int kf_id = pkf->mnId;
		int bias_vertex_id = kf_id + maxKFid + 1;
		VertexGyroBias *v_gb = dynamic_cast<VertexGyroBias *>(optimizer.vertex(bias_vertex_id));
		bg << v_gb->estimate();
		IMU::Bias b(0, 0, 0, bg[0], bg[1], bg[2]);

//		cout << "kf id: " << pkf->mnId << " gyros bias: " << bg.transpose() << endl;


		cv::Mat cvbg;
		cv::eigen2cv(bg, cvbg);
		cvbg.convertTo(cvbg, CV_32F);
		if (cv::norm(pkf->GetGyroBias() - cvbg) > 0.01) {
			pkf->SetNewBias(b);
			if (pkf->mpDvlPreintegrationKeyFrame) {
				pkf->mpDvlPreintegrationKeyFrame->ReintegrateWithVelocity();
			}
		}
		else {
			pkf->SetNewBias(b);
		}
		pkf->SetNewBias(b);
	}


	Eigen::Isometry3d T_dvl_c = vT_d_c->estimate();
	Eigen::Isometry3d T_gyros_dvl = vT_g_d->estimate();

	Eigen::Matrix3d R_gt;
	R_gt << 0, 0, 1,
		-1, 0, 0,
		0, -1, 0;
	cout << "init optimization result: \n"
	     << "R_dvl_c:\n" << T_dvl_c.rotation() << "\n"
	     << "R_dvl_c(eular yaw-pitch-roll):" << T_dvl_c.rotation().eulerAngles(2, 1, 0).transpose() << "\n"
	     << "t_dvl_c:" << T_dvl_c.translation().transpose() << "\n"
	     << "R_dvl_c distance with R_gt(LogSO3()): " << LogSO3(T_dvl_c.rotation().inverse() * R_gt).transpose() << "\n"
	     << "R_gyros_dvl:\n" << T_gyros_dvl.rotation() << "\n"
	     << "R_gyros_dvl(eular yaw-pitch-roll):" << T_gyros_dvl.rotation().eulerAngles(2, 1, 0).transpose() << "\n"
	     << endl;


	cv::Mat cvbg = Converter::toCvMat(bg);

	//Keyframes velocities and biases
	std::cout << "update Keyframes biases and extrinsic paramters" << std::endl;

	cv::Mat T_dvl_c_cv, T_gyros_dvl_cv, T_gyros_c_cv;
	cv::eigen2cv(T_dvl_c.matrix(), T_dvl_c_cv);
	T_dvl_c_cv.convertTo(T_dvl_c_cv, CV_32FC1);
	cv::eigen2cv(T_gyros_dvl.matrix(), T_gyros_dvl_cv);
	T_gyros_dvl_cv.convertTo(T_gyros_dvl_cv, CV_32FC1);
	Eigen::Isometry3d T_gyros_c = T_gyros_dvl * T_dvl_c;
	cv::eigen2cv(T_gyros_c.matrix(), T_gyros_c_cv);
	T_gyros_c_cv.convertTo(T_gyros_c_cv, CV_32FC1);

	// IMU::Calib extrinsic_para(T_gyros_c_cv, T_dvl_c_cv);

	const int N = vpKFs.size();
	for (size_t i = 0; i < N; i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
        pKFi->mImuCalib.SetExtrinsic(T_gyros_c_cv, T_dvl_c_cv);
	}

}
void Optimizer::DvlGyroInitOptimization3(Map *pMap,
                                         Eigen::Vector3d &bg,
                                         bool bMono,
                                         float priorG)
{
	Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	if (priorG != 0.f) {
		solver->setUserLambdaInit(100);
	}

	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	vector<VertexGyroBias *> vpgb;
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		// Biases
		VertexGyroBias *VG = new VertexGyroBias(pKFi);
		VG->setId(maxKFid + 1 + pKFi->mnId);
		VG->setFixed(true);
		optimizer.addVertex(VG);
		vpgb.push_back(VG);
	}



	// prior acc bias
	//		EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
	//		epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	//		double infoPriorG = priorG;
	//		epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
	//		optimizer.addEdge(epg);

	// extrinsic parameter
	g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
	vT_d_c->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_dvl_c));
	vT_d_c->setId(maxKFid + 1 + maxKFid + 1);
	vT_d_c->setFixed(false);
	optimizer.addVertex(vT_d_c);

	g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
	vT_g_d->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_imu_dvl));
	vT_g_d->setId(maxKFid + 1 + maxKFid + 2);
	vT_g_d->setFixed(true);
	optimizer.addVertex(vT_g_d);


	// Graph edges
	vector<EdgeDvlGyroInit *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	std::cout << "build optimization graph" << std::endl;

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpDvlPreintegrationKeyFrame) {
				std::cout << "Not preintegrated measurement" << std::endl;
			}

			pKFi->mpDvlPreintegrationKeyFrame->SetNewBias(pKFi->mPrevKF->GetImuBias());
			VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
			//				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
			//				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid + 1 + pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex(maxKFid + 1 + maxKFid + 1);
			g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex(maxKFid + 1 + maxKFid + 2);
			//				g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
			//				g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
			//				g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
			//				cout<<"VP1: Rcw[0]"<<VP1->estimate().Rcw[0]<<endl;
			//				cout<<"VP1: Rwc"<<VP1->estimate().Rwc<<endl;
			//				cout<<"VP1: tcw[0]"<<VP1->estimate().tcw[0]<<endl;
			//				cout<<"VP1: twc"<<VP1->estimate().twc<<endl;
			//
			//				cout<<"VP2: Rcw[0]"<<VP2->estimate().Rcw[0]<<endl;
			//				cout<<"VP2: Rwc"<<VP2->estimate().Rwc<<endl;
			//				cout<<"VP2: tcw[0]"<<VP2->estimate().tcw[0]<<endl;
			//				cout<<"VP2: twc"<<VP2->estimate().twc<<endl;

			if (!VP1 || !VG || !VP2) {
				cout << "Error" << VP1 << ", " << VG << ", " << VP2 << endl;

				continue;
			}
			//				EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			EdgeDvlGyroInit *ei = new EdgeDvlGyroInit(pKFi->mpDvlPreintegrationKeyFrame);
			//				ei->setVertex(0, VP1);

			//			g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
			//			ei->setRobustKernel(rk);
			//			rk->setDelta(sqrt(7.815));
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
			ei->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 100000);
			ei->setId(pKFi->mnId);


			vpei.push_back(ei);

			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}


	optimizer.setVerbose(true);
	std::set<g2o::HyperGraph::Edge *> setEdges = optimizer.edges();
	double total_error = 0;

	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};

	int mono_outlier;
	int stereo_outlier;
	float total_chi2;
	for (int i = 0; i < 4; i++) {
		if (i > 1) {
			vT_g_d->setFixed(false);
		}

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);
//		for (auto ei:vpei) {
//			ei->computeError();
//			cout << "calibration optimization iteration : " << i << "edge of " << ei->id() - 1 << " and " << ei->id()
//				 << "\nchi2:" << ei->chi2() << endl;
//		}

	}

	//	std::cout << "start optimization" << std::endl;
	//	optimizer.setVerbose(true);
	//	optimizer.initializeOptimization();
	//	optimizer.optimize(its);
	//
	//	std::cout << "end optimization" << std::endl;


	// Recover optimized data
	// Biases
	for (auto pkf: vpKFs) {
		int kf_id = pkf->mnId;
		int bias_vertex_id = kf_id + maxKFid + 1;
		VertexGyroBias *v_gb = dynamic_cast<VertexGyroBias *>(optimizer.vertex(bias_vertex_id));
		bg << v_gb->estimate();
		IMU::Bias b(0, 0, 0, bg[0], bg[1], bg[2]);

//		cout << "kf id: " << pkf->mnId << " gyros bias: " << bg.transpose() << endl;


		cv::Mat cvbg;
		cv::eigen2cv(bg, cvbg);
		cvbg.convertTo(cvbg, CV_32F);
//		if (cv::norm(pkf->GetGyroBias() - cvbg) > 0.01) {
//			pkf->SetNewBias(b);
//			if (pkf->mpDvlPreintegrationKeyFrame) {
//				pkf->mpDvlPreintegrationKeyFrame->ReintegrateWithVelocity();
//			}
//		}
//		else {
//			pkf->SetNewBias(b);
//		}
		pkf->SetNewBias(b);
	}


	Eigen::Isometry3d T_dvl_c = vT_d_c->estimate();
	Eigen::Quaterniond q_dvl_c(T_dvl_c.rotation());
	Eigen::Isometry3d T_gyros_dvl = vT_g_d->estimate();
	Eigen::Isometry3d T_gyros_c = T_gyros_dvl * T_dvl_c;
	Eigen::Quaterniond q_gyros_c(T_gyros_c.rotation());

	Eigen::Isometry3d T_dvl_c_gt = Eigen::Isometry3d::Identity();
	Eigen::Matrix3d R_dvl_c_gt;
	R_dvl_c_gt << 0, 1, 0, -1, 0, 0, 0, 0, 1;
	Eigen::Vector3d t_dvl_c_gt(-0.88, -0.348, 1.056);
	T_dvl_c_gt.rotate(R_dvl_c_gt);
	T_dvl_c_gt.pretranslate(t_dvl_c_gt);

	Eigen::Isometry3d T_gyros_c_gt = Eigen::Isometry3d::Identity();
	Eigen::Matrix3d R_gyros_c_gt;
	R_gyros_c_gt << 0, 0, 1, -1, 0, 0, 0, -1, 0;
	Eigen::Vector3d t_gyros_c_gt(0, 0, 0);
	T_gyros_c_gt.rotate(R_gyros_c_gt);
	T_gyros_c_gt.pretranslate(t_gyros_c_gt);

	Eigen::Vector3d err_R_dvl_c = LogSO3(T_dvl_c.rotation().inverse() * T_dvl_c_gt.rotation());
	Eigen::Vector3d err_R_gyros_c = LogSO3(T_gyros_c.rotation().inverse() * T_gyros_c_gt.rotation());
	Eigen::Vector3d err_t_dvl_c = T_dvl_c.translation() - T_dvl_c_gt.translation();
	Eigen::Vector3d err_t_gyros_c = T_gyros_c.translation() - T_gyros_c_gt.translation();

	stringstream ss;
// // 	ss << "result_" << ros::Time::now().toNSec() << ".txt";  // original  // original
	ofstream f("/home/da/project/ros/orb_dvl2_ws/src/dvl2/calibration_results/" + ss.str());

	if (f.is_open()) {
		f << fixed;
		f
			<< "#T_dvl_camera: tranlation(x y z) quaternion(x y z w) tranlation_error(x y z) rotation_error(LogSO3 x y z)\n";
		f << setprecision(9) << T_dvl_c.translation().x() << " " << T_dvl_c.translation().y() << " "
		  << T_dvl_c.translation().z() << " " << q_dvl_c.x() << " " << q_dvl_c.y() << " " << q_dvl_c.z() << " "
		  << q_dvl_c.w() << "\n" << err_t_dvl_c.x() << " " << err_t_dvl_c.y() << " " << err_t_dvl_c.z() << "\n"
		  << err_R_dvl_c.x() << " " << err_R_dvl_c.y() << " " << err_R_dvl_c.z() << " " << endl;
		f << "#T_gyroscope_camera: quaternion(x y z w) rotation_error(LogSO3 x y z)\n";
		f << setprecision(9) << q_gyros_c.x() << " " << q_gyros_c.y() << " " << q_gyros_c.z() << " " << q_gyros_c.w()
		  << "\n" << err_R_gyros_c.x() << " " << err_R_gyros_c.y() << " " << err_R_gyros_c.z() << " " << endl;
	}


	cout << "init optimization result: \n"
	     << "bias_gyro:\n" << bg << "\n"
	     << "R_dvl_c:\n" << T_dvl_c.rotation() << "\n"
	     << "t_dvl_c:" << T_dvl_c.translation().transpose() << "\n"
	     << "R_gyros_c:\n" << T_gyros_c.rotation() << "\n"
	     << endl;


	cv::Mat cvbg = Converter::toCvMat(bg);

	//Keyframes velocities and biases
	std::cout << "update Keyframes biases and extrinsic paramters" << std::endl;

	cv::Mat T_dvl_c_cv, T_gyros_dvl_cv, T_gyros_c_cv;
	cv::eigen2cv(T_dvl_c.matrix(), T_dvl_c_cv);
	T_dvl_c_cv.convertTo(T_dvl_c_cv, CV_32FC1);
	cv::eigen2cv(T_gyros_dvl.matrix(), T_gyros_dvl_cv);
	T_gyros_dvl_cv.convertTo(T_gyros_dvl_cv, CV_32FC1);
	T_gyros_c = T_gyros_dvl * T_dvl_c;
	cv::eigen2cv(T_gyros_c.matrix(), T_gyros_c_cv);
	T_gyros_c_cv.convertTo(T_gyros_c_cv, CV_32FC1);

	// IMU::Calib extrinsic_para(T_gyros_c_cv, T_dvl_c_cv);

	const int N = vpKFs.size();
	for (size_t i = 0; i < N; i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
        pKFi->mImuCalib.SetExtrinsic(T_gyros_c_cv, T_dvl_c_cv);
	}

}

void Optimizer::DvlGyroInitOptimization5(Map *pMap, Eigen::Vector3d &bg, bool bMono, float priorG)
{
	Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	if (priorG != 0.f) {
		solver->setUserLambdaInit(100);
	}

	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	vector<VertexGyroBias *> vpgb;
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		// Biases
		VertexGyroBias *VG = new VertexGyroBias(pKFi);
		VG->setId(maxKFid + 1 + pKFi->mnId);
		VG->setFixed(true);
		optimizer.addVertex(VG);
		vpgb.push_back(VG);
	}



	// prior acc bias
	//		EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
	//		epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	//		double infoPriorG = priorG;
	//		epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
	//		optimizer.addEdge(epg);

	// extrinsic parameter
	g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
	vT_d_c->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_dvl_c));
	vT_d_c->setId(maxKFid + 1 + maxKFid + 1);
	vT_d_c->setFixed(true);
	optimizer.addVertex(vT_d_c);

	g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
	vT_g_d->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_imu_dvl));
	vT_g_d->setId(maxKFid + 1 + maxKFid + 2);
	vT_g_d->setFixed(true);
	optimizer.addVertex(vT_g_d);

	VertexDVLBeamOritenstion *v_beam_ori = new VertexDVLBeamOritenstion();
	Eigen::Matrix<double, 8, 1> alpha_beta;
//	alpha_beta << 67.5 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		45 / 180.0 * M_PI;
	alpha_beta << 1 / 180.0 * M_PI,
		1 / 180.0 * M_PI,
		1 / 180.0 * M_PI,
		1 / 180.0 * M_PI,
		45 / 180.0 * M_PI,
		45 / 180.0 * M_PI,
		45 / 180.0 * M_PI,
		45 / 180.0 * M_PI;
	v_beam_ori->setFixed(false);
	v_beam_ori->setId(maxKFid + 1 + maxKFid + 3);
	v_beam_ori->setEstimate(alpha_beta);
	optimizer.addVertex(v_beam_ori);

	// Graph edges
	vector<EdgeDvlGyroInit2 *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	std::cout << "build optimization graph" << std::endl;

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpDvlPreintegrationKeyFrame) {
				std::cout << "Not preintegrated measurement" << std::endl;
			}

			pKFi->mpDvlPreintegrationKeyFrame->SetNewBias(pKFi->mPrevKF->GetImuBias());
			VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
			//				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
			//				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid + 1 + pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex(maxKFid + 1 + maxKFid + 1);
			g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex(maxKFid + 1 + maxKFid + 2);
			g2o::HyperGraph::Vertex *Valpha_beta = optimizer.vertex(maxKFid + 1 + maxKFid + 3);

			if (!VP1 || !VG || !VP2) {
				cout << "Error" << VP1 << ", " << VG << ", " << VP2 << endl;

				continue;
			}
			//				EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			EdgeDvlGyroInit2 *ei = new EdgeDvlGyroInit2(pKFi->mpDvlPreintegrationKeyFrame);
			//				ei->setVertex(0, VP1);

			//			g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
			//			ei->setRobustKernel(rk);
			//			rk->setDelta(sqrt(7.815));
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
			ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(Valpha_beta));
			ei->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 100000);
			ei->setId(pKFi->mnId);


			vpei.push_back(ei);

			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}


	optimizer.setVerbose(true);
	std::set<g2o::HyperGraph::Edge *> setEdges = optimizer.edges();
	double total_error = 0;

	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};

	int mono_outlier;
	int stereo_outlier;
	float total_chi2;
	for (int i = 0; i < 4; i++) {
//		if (i > 1) {
//			vT_g_d->setFixed(false);
//		}

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);
//		for (auto ei:vpei) {
//			ei->computeError();
//			cout << "calibration optimization iteration : " << i << "edge of " << ei->id() - 1 << " and " << ei->id()
//				 << "\nchi2:" << ei->chi2() << endl;
//		}

	}

	//	std::cout << "start optimization" << std::endl;
	//	optimizer.setVerbose(true);
	//	optimizer.initializeOptimization();
	//	optimizer.optimize(its);
	//
	//	std::cout << "end optimization" << std::endl;


	// Recover optimized data
	// Biases
	for (auto pkf: vpKFs) {
		int kf_id = pkf->mnId;
		int bias_vertex_id = kf_id + maxKFid + 1;
		VertexGyroBias *v_gb = dynamic_cast<VertexGyroBias *>(optimizer.vertex(bias_vertex_id));
		bg << v_gb->estimate();
		IMU::Bias b(0, 0, 0, bg[0], bg[1], bg[2]);

//		cout << "kf id: " << pkf->mnId << " gyros bias: " << bg.transpose() << endl;


		cv::Mat cvbg;
		cv::eigen2cv(bg, cvbg);
		cvbg.convertTo(cvbg, CV_32F);
//		if (cv::norm(pkf->GetGyroBias() - cvbg) > 0.01) {
//			pkf->SetNewBias(b);
//			if (pkf->mpDvlPreintegrationKeyFrame) {
//				pkf->mpDvlPreintegrationKeyFrame->ReintegrateWithVelocity();
//			}
//		}
//		else {
//			pkf->SetNewBias(b);
//		}
		pkf->SetNewBias(b);
	}


	Eigen::Isometry3d T_dvl_c = vT_d_c->estimate();
	Eigen::Quaterniond q_dvl_c(T_dvl_c.rotation());
	Eigen::Isometry3d T_gyros_dvl = vT_g_d->estimate();
	Eigen::Isometry3d T_gyros_c = T_gyros_dvl * T_dvl_c;
	Eigen::Quaterniond q_gyros_c(T_gyros_c.rotation());

	Eigen::Isometry3d T_dvl_c_gt = Eigen::Isometry3d::Identity();
	Eigen::Matrix3d R_dvl_c_gt;
	R_dvl_c_gt << 0, 1, 0, -1, 0, 0, 0, 0, 1;
	Eigen::Vector3d t_dvl_c_gt(-0.88, -0.348, 1.056);
	T_dvl_c_gt.rotate(R_dvl_c_gt);
	T_dvl_c_gt.pretranslate(t_dvl_c_gt);

	Eigen::Isometry3d T_gyros_c_gt = Eigen::Isometry3d::Identity();
	Eigen::Matrix3d R_gyros_c_gt;
	R_gyros_c_gt << 0, 0, 1, -1, 0, 0, 0, -1, 0;
	Eigen::Vector3d t_gyros_c_gt(0, 0, 0);
	T_gyros_c_gt.rotate(R_gyros_c_gt);
	T_gyros_c_gt.pretranslate(t_gyros_c_gt);

	Eigen::Vector3d err_R_dvl_c = LogSO3(T_dvl_c.rotation().inverse() * T_dvl_c_gt.rotation());
	Eigen::Vector3d err_R_gyros_c = LogSO3(T_gyros_c.rotation().inverse() * T_gyros_c_gt.rotation());
	Eigen::Vector3d err_t_dvl_c = T_dvl_c.translation() - T_dvl_c_gt.translation();
	Eigen::Vector3d err_t_gyros_c = T_gyros_c.translation() - T_gyros_c_gt.translation();

	stringstream ss;
// // 	ss << "result_" << ros::Time::now().toNSec() << ".txt";  // original  // original
	ofstream f("/home/da/project/ros/orb_dvl2_ws/src/dvl2/calibration_results/" + ss.str());

	if (f.is_open()) {
		f << fixed;
		f
			<< "#T_dvl_camera: tranlation(x y z) quaternion(x y z w) tranlation_error(x y z) rotation_error(LogSO3 x y z)\n";
		f << setprecision(9) << T_dvl_c.translation().x() << " " << T_dvl_c.translation().y() << " "
		  << T_dvl_c.translation().z() << " " << q_dvl_c.x() << " " << q_dvl_c.y() << " " << q_dvl_c.z() << " "
		  << q_dvl_c.w() << "\n" << err_t_dvl_c.x() << " " << err_t_dvl_c.y() << " " << err_t_dvl_c.z() << "\n"
		  << err_R_dvl_c.x() << " " << err_R_dvl_c.y() << " " << err_R_dvl_c.z() << " " << endl;
		f << "#T_gyroscope_camera: quaternion(x y z w) rotation_error(LogSO3 x y z)\n";
		f << setprecision(9) << q_gyros_c.x() << " " << q_gyros_c.y() << " " << q_gyros_c.z() << " " << q_gyros_c.w()
		  << "\n" << err_R_gyros_c.x() << " " << err_R_gyros_c.y() << " " << err_R_gyros_c.z() << " " << endl;
	}

	alpha_beta = v_beam_ori->estimate();

	cout << "init optimization result: \n"
	     << "bias_gyro:\n" << bg << "\n"
	     << "R_dvl_c:\n" << T_dvl_c.rotation() << "\n"
	     << "t_dvl_c:" << T_dvl_c.translation().transpose() << "\n"
	     << "R_gyros_c:\n" << T_gyros_c.rotation() << "\n"
	     << "alpha_beta:\n" << alpha_beta.transpose() << "\n"
	     << endl;


	cv::Mat cvbg = Converter::toCvMat(bg);

	//Keyframes velocities and biases
	std::cout << "update Keyframes biases and extrinsic paramters" << std::endl;

	cv::Mat T_dvl_c_cv, T_gyros_dvl_cv, T_gyros_c_cv;
	cv::eigen2cv(T_dvl_c.matrix(), T_dvl_c_cv);
	T_dvl_c_cv.convertTo(T_dvl_c_cv, CV_32FC1);
	cv::eigen2cv(T_gyros_dvl.matrix(), T_gyros_dvl_cv);
	T_gyros_dvl_cv.convertTo(T_gyros_dvl_cv, CV_32FC1);
	T_gyros_c = T_gyros_dvl * T_dvl_c;
	cv::eigen2cv(T_gyros_c.matrix(), T_gyros_c_cv);
	T_gyros_c_cv.convertTo(T_gyros_c_cv, CV_32FC1);

	// IMU::Calib extrinsic_para(T_gyros_c_cv, T_dvl_c_cv);

	const int N = vpKFs.size();
	for (size_t i = 0; i < N; i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
        pKFi->mImuCalib.SetExtrinsic(T_gyros_c_cv, T_dvl_c_cv);
	}

}

void Optimizer::DvlGyroInitOptimization6(Map *pMap, Eigen::Vector3d &bg, bool bMono, float priorG)
{
	Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	if (priorG != 0.f) {
		solver->setUserLambdaInit(100);
	}

	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	vector<VertexGyroBias *> vpgb;
	vector<VertexVelocity *> vpv;
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		// Biases
		VertexGyroBias *VG = new VertexGyroBias(pKFi);
		VG->setId(maxKFid + 1 + pKFi->mnId);
		VG->setFixed(true);
		optimizer.addVertex(VG);
		vpgb.push_back(VG);

		VertexVelocity *VV = new VertexVelocity();
		Eigen::Vector3d v(0.1, 0.1, 0.1);
		VV->setEstimate(v);
		VV->setId(maxKFid + 1 + maxKFid + 1 + pKFi->mnId);
		VV->setFixed(false);
		optimizer.addVertex(VV);
		vpv.push_back(VV);

	}



	// prior acc bias
	//		EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
	//		epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	//		double infoPriorG = priorG;
	//		epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
	//		optimizer.addEdge(epg);

	// extrinsic parameter
	g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
	vT_d_c->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_dvl_c));
	vT_d_c->setId(maxKFid + 1 + maxKFid + 1 + maxKFid + 1);
	vT_d_c->setFixed(true);
	optimizer.addVertex(vT_d_c);

	g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
	vT_g_d->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_imu_dvl));
	vT_g_d->setId(maxKFid + 1 + maxKFid + 1 + maxKFid + 2);
	vT_g_d->setFixed(true);
	optimizer.addVertex(vT_g_d);

	VertexDVLBeamOritenstion *v_beam_ori = new VertexDVLBeamOritenstion();
//	Eigen::Matrix<double,8,1> alpha_beta;
//	alpha_beta << 67.5 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		45 / 180.0 * M_PI;
//	alpha_beta << 1 / 180.0 * M_PI,
//		1 / 180.0 * M_PI,
//		1 / 180.0 * M_PI,
//		1 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		45 / 180.0 * M_PI;
//	v_beam_ori->setFixed(false);
//	v_beam_ori->setId(maxKFid + 1 + maxKFid + 1 + maxKFid + 3);
//	v_beam_ori->setEstimate(alpha_beta);
//	optimizer.addVertex(v_beam_ori);

	// Graph edges
	vector<EdgeDvlGyroInit3 *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	std::cout << "build optimization graph" << std::endl;

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpDvlPreintegrationKeyFrame) {
				std::cout << "Not preintegrated measurement" << std::endl;
			}

			pKFi->mpDvlPreintegrationKeyFrame->SetNewBias(pKFi->mPrevKF->GetImuBias());
			VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
			//				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
			//				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid + 1 + pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VV = optimizer.vertex(maxKFid + 1 + maxKFid + 1 + pKFi->mnId);
			g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex(maxKFid + 1 + maxKFid + 1 + maxKFid + 1);
			g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex(maxKFid + 1 + maxKFid + 1 + maxKFid + 2);
//			g2o::HyperGraph::Vertex *Valpha_beta = optimizer.vertex(maxKFid + 1 + maxKFid + 1 + maxKFid + 3);

			if (!VP1 || !VG || !VP2) {
				cout << "Error" << VP1 << ", " << VG << ", " << VP2 << endl;

				continue;
			}
			//				EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			EdgeDvlGyroInit3 *ei = new EdgeDvlGyroInit3(pKFi->mpDvlPreintegrationKeyFrame);
			//				ei->setVertex(0, VP1);

			//			g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
			//			ei->setRobustKernel(rk);
			//			rk->setDelta(sqrt(7.815));
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
			ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
			ei->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 100000);
			ei->setId(pKFi->mnId);


			vpei.push_back(ei);

			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}


	optimizer.setVerbose(true);
	std::set<g2o::HyperGraph::Edge *> setEdges = optimizer.edges();
	double total_error = 0;

	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};

	int mono_outlier;
	int stereo_outlier;
	float total_chi2;
	for (int i = 0; i < 4; i++) {
//		if (i > 1) {
//			vT_g_d->setFixed(false);
//		}

		optimizer.initializeOptimization();
		optimizer.optimize(10);
//		for (auto ei:vpei) {
//			ei->computeError();
//			cout << "calibration optimization iteration : " << i << "edge of " << ei->id() - 1 << " and " << ei->id()
//				 << "\nchi2:" << ei->chi2() << endl;
//		}

	}

	//	std::cout << "start optimization" << std::endl;
	//	optimizer.setVerbose(true);
	//	optimizer.initializeOptimization();
	//	optimizer.optimize(its);
	//
	//	std::cout << "end optimization" << std::endl;


	// Recover optimized data
	// Biases
	for (auto pkf: vpKFs) {
		int kf_id = pkf->mnId;
		int bias_vertex_id = kf_id + maxKFid + 1;
		int velocity_vertex_id = kf_id + maxKFid + 1 + maxKFid + 1;
		VertexGyroBias *v_gb = dynamic_cast<VertexGyroBias *>(optimizer.vertex(bias_vertex_id));
		bg << v_gb->estimate();
		IMU::Bias b(0, 0, 0, bg[0], bg[1], bg[2]);

		VertexVelocity *v_v = dynamic_cast<VertexVelocity *>(optimizer.vertex(velocity_vertex_id));
		Eigen::Vector3d v_dk = v_v->estimate();
		pkf->mpDvlPreintegrationKeyFrame->v_dk_visual = v_dk;

//		cout << "kf id: " << pkf->mnId << " gyros bias: " << bg.transpose() << endl;
// 		ROS_INFO_STREAM("beam calibration: KeyFrame id:<<" << pkf->mnId << " dvl velocity: "  // original
// 		                                                   << pkf->mpDvlPreintegrationKeyFrame->v_dk_dvl  // original
// 		                                                   << " visual velocity: "  // original
// 		                                                   << pkf->mpDvlPreintegrationKeyFrame->v_dk_visual.transpose());  // original
	}

	DvlBeamOptimization(pMap);
	DvlBeamOptimization_dvl(pMap);




}

void Optimizer::DvlGyroInitOptimization4(Map *pMap,
                                         Eigen::Vector3d &bg,
                                         bool bMono,
                                         float priorG)
{
	Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	if (priorG != 0.f) {
		solver->setUserLambdaInit(1e3);
	}

	optimizer.setAlgorithm(solver);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	vector<VertexGyroBias *> vpgb;
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);

		// Biases
		VertexGyroBias *VG = new VertexGyroBias(pKFi);
		VG->setId(maxKFid + 1 + pKFi->mnId);
		VG->setFixed(true);
		optimizer.addVertex(VG);
		vpgb.push_back(VG);
	}



	// prior acc bias
	//		EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
	//		epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
	//		double infoPriorG = priorG;
	//		epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
	//		optimizer.addEdge(epg);

	// extrinsic parameter
	g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
	vT_d_c->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_dvl_c));
	vT_d_c->setId(maxKFid + 1 + maxKFid + 1);
	vT_d_c->setFixed(false);
	optimizer.addVertex(vT_d_c);

	g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
	vT_g_d->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_imu_dvl));
	vT_g_d->setId(maxKFid + 1 + maxKFid + 2);
	vT_g_d->setFixed(true);
	optimizer.addVertex(vT_g_d);


	// Graph edges
	vector<EdgeDvlGyroInit *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	std::cout << "build optimization graph" << std::endl;

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpDvlPreintegrationKeyFrame) {
				std::cout << "Not preintegrated measurement" << std::endl;
			}

			pKFi->mpDvlPreintegrationKeyFrame->SetNewBias(pKFi->mPrevKF->GetImuBias());
			VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
			//				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
			//				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
			g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid + 1 + pKFi->mPrevKF->mnId);
			g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex(maxKFid + 1 + maxKFid + 1);
			g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex(maxKFid + 1 + maxKFid + 2);
			//				g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
			//				g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
			//				g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
			//				cout<<"VP1: Rcw[0]"<<VP1->estimate().Rcw[0]<<endl;
			//				cout<<"VP1: Rwc"<<VP1->estimate().Rwc<<endl;
			//				cout<<"VP1: tcw[0]"<<VP1->estimate().tcw[0]<<endl;
			//				cout<<"VP1: twc"<<VP1->estimate().twc<<endl;
			//
			//				cout<<"VP2: Rcw[0]"<<VP2->estimate().Rcw[0]<<endl;
			//				cout<<"VP2: Rwc"<<VP2->estimate().Rwc<<endl;
			//				cout<<"VP2: tcw[0]"<<VP2->estimate().tcw[0]<<endl;
			//				cout<<"VP2: twc"<<VP2->estimate().twc<<endl;

			if (!VP1 || !VG || !VP2) {
				cout << "Error" << VP1 << ", " << VG << ", " << VP2 << endl;

				continue;
			}
			//				EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
			EdgeDvlGyroInit *ei = new EdgeDvlGyroInit(pKFi->mpDvlPreintegrationKeyFrame);
			//				ei->setVertex(0, VP1);

			//			g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
			//			ei->setRobustKernel(rk);
			//			rk->setDelta(sqrt(7.815));
			ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
			ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
			ei->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 100000);
			ei->setId(pKFi->mnId);


			vpei.push_back(ei);

			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}


	optimizer.setVerbose(true);
	std::set<g2o::HyperGraph::Edge *> setEdges = optimizer.edges();
	double total_error = 0;

	const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
	const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};

	int mono_outlier;
	int stereo_outlier;
	float total_chi2;
	for (int i = 0; i < 4; i++) {
		if (i > 0) {
			vT_g_d->setFixed(false);
		}
		if (i > 2) {
			for (auto v_gb: vpgb) {
				v_gb->setFixed(false);
			}
		}

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);
		//		for (auto ei:vpei) {
		//			ei->computeError();
		//			cout << "calibration optimization iteration : " << i << "edge of " << ei->id() - 1 << " and " << ei->id()
		//				 << "\nchi2:" << ei->chi2() << endl;
		//		}

	}

	//	std::cout << "start optimization" << std::endl;
	//	optimizer.setVerbose(true);
	//	optimizer.initializeOptimization();
	//	optimizer.optimize(its);
	//
	//	std::cout << "end optimization" << std::endl;


	// Recover optimized data
	// Biases
	for (auto pkf: vpKFs) {
		int kf_id = pkf->mnId;
		int bias_vertex_id = kf_id + maxKFid + 1;
		VertexGyroBias *v_gb = dynamic_cast<VertexGyroBias *>(optimizer.vertex(bias_vertex_id));
		bg << v_gb->estimate();
		IMU::Bias b(0, 0, 0, bg[0], bg[1], bg[2]);

		cout << "kf id: " << pkf->mnId << " gyros bias: " << bg.transpose() << endl;


		cv::Mat cvbg;
		cv::eigen2cv(bg, cvbg);
		cvbg.convertTo(cvbg, CV_32F);
		//		if (cv::norm(pkf->GetGyroBias() - cvbg) > 0.01) {
		//			pkf->SetNewBias(b);
		//			if (pkf->mpDvlPreintegrationKeyFrame) {
		//				pkf->mpDvlPreintegrationKeyFrame->ReintegrateWithVelocity();
		//			}
		//		}
		//		else {
		//			pkf->SetNewBias(b);
		//		}
		pkf->SetNewBias(b);
	}


	Eigen::Isometry3d T_dvl_c = vT_d_c->estimate();
	Eigen::Isometry3d T_gyros_dvl = vT_g_d->estimate();

	Eigen::Matrix3d R_gt;
	R_gt << 0, 0, 1,
		-1, 0, 0,
		0, -1, 0;
	cout << "init optimization result: \n"
	     << "R_dvl_c:\n" << T_dvl_c.rotation() << "\n"
	     << "R_dvl_c(eular yaw-pitch-roll):" << T_dvl_c.rotation().eulerAngles(2, 1, 0).transpose() << "\n"
	     << "t_dvl_c:" << T_dvl_c.translation().transpose() << "\n"
	     << "R_dvl_c distance with R_gt(LogSO3()): " << LogSO3(T_dvl_c.rotation().inverse() * R_gt).transpose() << "\n"
	     << "R_gyros_dvl:\n" << T_gyros_dvl.rotation() << "\n"
	     << "R_gyros_dvl(eular yaw-pitch-roll):" << T_gyros_dvl.rotation().eulerAngles(2, 1, 0).transpose() << "\n"
	     << endl;


	cv::Mat cvbg = Converter::toCvMat(bg);

	//Keyframes velocities and biases
	std::cout << "update Keyframes biases and extrinsic paramters" << std::endl;

	cv::Mat T_dvl_c_cv, T_gyros_dvl_cv, T_gyros_c_cv;
	cv::eigen2cv(T_dvl_c.matrix(), T_dvl_c_cv);
	T_dvl_c_cv.convertTo(T_dvl_c_cv, CV_32FC1);
	cv::eigen2cv(T_gyros_dvl.matrix(), T_gyros_dvl_cv);
	T_gyros_dvl_cv.convertTo(T_gyros_dvl_cv, CV_32FC1);
	Eigen::Isometry3d T_gyros_c = T_gyros_dvl * T_dvl_c;
	cv::eigen2cv(T_gyros_c.matrix(), T_gyros_c_cv);
	T_gyros_c_cv.convertTo(T_gyros_c_cv, CV_32FC1);

	// IMU::Calib extrinsic_para(T_gyros_c_cv, T_dvl_c_cv);

	const int N = vpKFs.size();
	for (size_t i = 0; i < N; i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
        pKFi->mImuCalib.SetExtrinsic(T_gyros_c_cv, T_dvl_c_cv);
	}

}

double Optimizer::DvlIMUInitOptimization(Map *pMap, double priori_g, double priori_a)
{
	// Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);


	optimizer.setAlgorithm(solver);
    // solver->setUserLambdaInit(1e3);

	// Set KeyFrame vertices (fixed poses and optimizable velocities)
	vector<VertexGyroBias *> vpgb;
    vector<VertexAccBias *> vpab;
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId > maxKFid) {
			continue;
		}
		VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
		VP->setId(pKFi->mnId);
		VP->setFixed(true);
		optimizer.addVertex(VP);



		//velocity
		VertexVelocity *VV = new VertexVelocity(pKFi);
        VV->setId((maxKFid + 1)*3 + pKFi->mnId);
        VV->setFixed(false);
        optimizer.addVertex(VV);
	}

    // Biases
    VertexGyroBias *VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid + 1 + 1);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias *VA = new VertexAccBias(vpKFs.front());
    VA->setId((maxKFid + 1)*2 + 1);
    VA->setFixed(true);
    optimizer.addVertex(VA);





	// extrinsic parameter
	g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
	vT_d_c->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_dvl_c));
	vT_d_c->setId((maxKFid + 1)*4);
	vT_d_c->setFixed(true);
	optimizer.addVertex(vT_d_c);

	g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
	vT_g_d->setEstimate(Converter::toSE3Quat(vpKFs[0]->mImuCalib.mT_imu_dvl));
	vT_g_d->setId((maxKFid + 1)*4+1);
	vT_g_d->setFixed(true);
	optimizer.addVertex(vT_g_d);

    VertexGDir *VGDir = new VertexGDir(pMap->getRGravity());
    VGDir->setId((maxKFid + 1)*4+2);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);


	// Graph edges
    EdgePriorGyro* eg_pri_bias = new EdgePriorGyro();
    eg_pri_bias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
    eg_pri_bias->setInformation(Eigen::Matrix3d::Identity()*priori_g);
    optimizer.addEdge(eg_pri_bias);
    EdgePriorAcc* e_pri_bias = new EdgePriorAcc();
    e_pri_bias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
    e_pri_bias->setInformation(Eigen::Matrix3d::Identity()*priori_a);
    optimizer.addEdge(e_pri_bias);

	vector<EdgeDvlIMUInitWithoutBias *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	// std::cout << "build optimization graph" << std::endl;
    vector<EdgeDvlIMU*> dvlimu_edges;
	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];
		if (pKFi->mnId<=1)
			continue;

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpDvlPreintegrationKeyFrame) {
				std::cout << "Not preintegrated measurement" << std::endl;
			}

			// pKFi->mpDvlPreintegrationKeyFrame->SetNewBias(pKFi->mPrevKF->GetImuBias());
			VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
			//				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
			VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
			//				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex((maxKFid + 1)*3 + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex((maxKFid + 1)*3 + pKFi->mnId);
            g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid + 1 + 1);
            g2o::HyperGraph::Vertex *VA = optimizer.vertex((maxKFid + 1) * 2 + 1);


			g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex((maxKFid + 1)*4);
			g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex((maxKFid + 1)*4+1);
            g2o::HyperGraph::Vertex *VR_w_b0 = optimizer.vertex((maxKFid + 1)*4+2);

			if (!VP1 || !VP2 || !VV1 || !VV2 || !VG || !VA  || !VT_d_c || !VT_g_d || !VR_w_b0) {
//                 ROS_ERROR_STREAM("DVL IMU initialzation Error, KF1 ID:"<< pKFi->mPrevKF->mnId << "KF2 ID:" << pKFi->mnId << "VP1: " << VP1 <<", VP2: " << VP2 << ", VV1: " << VV1  // original
// 								 << ", VV2: " << VV2 << ", VG: " << VG << ", VA: " << VA  // original
// 								 << ", VT_d_c: " << VT_d_c << ", VT_g_d: " << VT_g_d  // original
// 								 << ", VR_w_b0: " << VR_w_b0);  // original
				continue;
                // assert(-1);
			}
            // // prior acc bias
            // EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F));
            // epa->setLevel(0);
            // epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            // epa->setInformation(bias_info * Eigen::Matrix3d::Identity());
            // optimizer.addEdge(epa);
            // // prior gyro bias
            // EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
            // epg->setLevel(0);
            // epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            // epg->setInformation(bias_info * Eigen::Matrix3d::Identity());
            // optimizer.addEdge(epg);




			EdgeDvlIMU *ei2 = new EdgeDvlIMU(pKFi->mpDvlPreintegrationKeyFrame);
			ei2->setLevel(0);
			ei2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
			ei2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
			ei2->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
			ei2->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
			ei2->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
			ei2->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
			ei2->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
			ei2->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
			ei2->setVertex(8, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VR_w_b0));
            Eigen::Matrix<double, 9, 9> info = Eigen::Matrix<double, 9, 9>::Identity()* 1e6;
            cv::Mat cvInfo = pKFi->mpDvlPreintegrationKeyFrame->C.rowRange(0,9).colRange(0,9).inv(cv::DECOMP_SVD);
            cv::cv2eigen(cvInfo, info);
            // info.block(0,0,3,3) = Eigen::Matrix3d::Identity() * 1e6;
            // info.block(3,3,3,3) = Eigen::Matrix3d::Identity() * 1e4;
            // info.block(6,6,3,3) = Eigen::Matrix3d::Identity() * 1e4;
            // info.block(0,0,3,3) = Eigen::Matrix3d::Identity() * 1e6;
            // info(0,0) = info(0,0)*lamda_DVL * 5e3; // 10_24
            // info_DI(1,1) = 1e9; // before 10_24
            // info.block(3,3,3,3) = Eigen::Matrix3d::Identity() * 1e6;
            // info.block(6,6,3,3) = Eigen::Matrix3d::Identity() * 1e6;
			ei2->setInformation(info);
			// ei2->setId(pKFi->mnId);
			optimizer.addEdge(ei2);
            dvlimu_edges.push_back(ei2);

		}
	}


	optimizer.setVerbose(false);
    optimizer.initializeOptimization(0);
    optimizer.optimize(20);
    VG->setFixed(false);
    VA->setFixed(false);
    optimizer.initializeOptimization(0);
    optimizer.optimize(20);

    auto bias_g = VG->estimate();
    auto bias_a = VA->estimate();
//     ROS_INFO_STREAM("bias_g: "<< bias_g.transpose());  // original
//     ROS_INFO_STREAM("bias_a: "<< bias_a.transpose());  // original

    double total_dvl = 0;
    double avg_dvl = 0;
    for(auto e:dvlimu_edges){
        total_dvl += e->error().norm();
    }
    avg_dvl = total_dvl/dvlimu_edges.size();
//     ROS_INFO_STREAM("avg_dvl: "<< avg_dvl);  // original
//     ROS_INFO_STREAM("total_dvl:"<< total_dvl);  // original
    // VGDir->setFixed(true);
    // e_bias->setLevel(0);
    // e_bias_without->setLevel(1);
    // optimizer.initializeOptimization(0);
    // optimizer.optimize(2);

    // update gravity direction
    // Eigen::Matrix3d R_b0_w = NormalizeRotation(VGDir->estimate().Rwg);
    // Eigen::Matrix3d R_b0_w = VGDir->estimate().Rwg;
//     // ROS_INFO_STREAM("gravity calibration result: "<< R_b0_w);  // original
    // Sophus::SO3<double> R_b0_w_SO3(R_b0_w);
    // Eigen::Vector3d R_b0_w_so3 = R_b0_w_SO3.log();
//     // ROS_INFO_STREAM("gravity calibration result: \n"<< VGDir->estimate().Rwg);  // original
    pMap->setRGravity(VGDir->estimate().Rwg);
    // if(vpKFs.size()<200){
    // pMap->SetImuInitialized();
    if(avg_dvl>2000){
        return avg_dvl;
    }

	// Recover optimized data
	// Biases
	for (auto pkf: vpKFs) {
		int kf_id = pkf->mnId;
		int gyros_bias_vertex_id = 1 + maxKFid + 1;
        int acc_bias_vertex_id = 1 + (maxKFid + 1)*2;
		VertexGyroBias *v_gb = dynamic_cast<VertexGyroBias *>(optimizer.vertex(gyros_bias_vertex_id));
        VertexAccBias *v_ab = dynamic_cast<VertexAccBias *>(optimizer.vertex(acc_bias_vertex_id));
        // bg << v_gb->estimate();
		IMU::Bias b(v_ab->estimate().x(), v_ab->estimate().y(), v_ab->estimate().z(),
                    v_gb->estimate().x(), v_gb->estimate().y(), v_gb->estimate().z());
		cv::Mat cvbg;
		cv::eigen2cv(v_gb->estimate(), cvbg);
		cvbg.convertTo(cvbg, CV_32F);
		pkf->SetNewBias(b);
	}
    return avg_dvl;
	// pkf->SetNewBias(b)

}


void Optimizer::DvlIMURefineOptimization(Atlas* pAtlas)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);


    long unsigned int maxKFid = 0;
    for(auto m:pAtlas->GetAllMaps()){
        if(maxKFid<m->GetMaxKFid())
            maxKFid = m->GetMaxKFid();
    }

    // extrinsic parameter
    g2o::VertexSE3Expmap *vT_d_c = new g2o::VertexSE3Expmap();
    vT_d_c->setEstimate(Converter::toSE3Quat(pAtlas->GetAllMaps().front()->GetOriginKF()->mImuCalib.mT_dvl_c));
    vT_d_c->setId((maxKFid + 1)*4);
    vT_d_c->setFixed(true);
    optimizer.addVertex(vT_d_c);

    g2o::VertexSE3Expmap *vT_g_d = new g2o::VertexSE3Expmap();
    vT_g_d->setEstimate(Converter::toSE3Quat(pAtlas->GetAllMaps().front()->GetOriginKF()->mImuCalib.mT_imu_dvl));
    vT_g_d->setId((maxKFid + 1)*4+1);
    vT_g_d->setFixed(true);
    optimizer.addVertex(vT_g_d);

    VertexGDir *VGDir = new VertexGDir(pAtlas->getRGravity());
    VGDir->setId((maxKFid + 1)*4+2);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);

    stringstream ss;
    ss<<"Gravity refine optimization: \n";
    for(auto pMap:pAtlas->GetAllMaps()){
        ss << "map id: " << pMap->GetId() << endl;
        const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

        // Set KeyFrame vertices (fixed poses and optimizable velocities)
        vector<VertexGyroBias *> vpgb;
        vector<VertexAccBias *> vpab;
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKFi = vpKFs[i];
            if (pKFi->mnId > maxKFid) {
                continue;
            }
            VertexPoseDvlIMU *VP = new VertexPoseDvlIMU(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            // Biases
            VertexGyroBias *VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 1 + pKFi->mnId);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            vpgb.push_back(VG);

            VertexAccBias *VA = new VertexAccBias(pKFi);
            VA->setId((maxKFid + 1)*2 + pKFi->mnId);
            VA->setFixed(true);
            optimizer.addVertex(VA);
            vpab.push_back(VA);

            //velocity
            VertexVelocity *VV = new VertexVelocity(pKFi);
            VV->setId((maxKFid + 1)*3 + pKFi->mnId);
            VV->setFixed(true);
            optimizer.addVertex(VV);
        }







        // Graph edges
        vector<EdgeDvlIMUInitWithoutBias *> vpei;
        vpei.reserve(vpKFs.size());
        vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
        vppUsedKF.reserve(vpKFs.size());

        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKFi = vpKFs[i];

            if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
                if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
                    continue;
                }
                if (!pKFi->mpDvlPreintegrationKeyFrame) {
                    std::cout << "Not preintegrated measurement" << std::endl;
                }
                ss << "Add gravity edge from KF[" << pKFi->mnId << "] to KF{" << pKFi->mPrevKF->mnId<<"]"<< endl;
                // pKFi->mpDvlPreintegrationKeyFrame->SetNewBias(pKFi->mPrevKF->GetImuBias());
                VertexPoseDvlIMU *VP1 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mPrevKF->mnId));
                //				g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
                VertexPoseDvlIMU *VP2 = dynamic_cast<VertexPoseDvlIMU *>(optimizer.vertex(pKFi->mnId));
                //				g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
                g2o::HyperGraph::Vertex *VV1 = optimizer.vertex((maxKFid + 1)*3 + pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex *VV2 = optimizer.vertex((maxKFid + 1)*3 + pKFi->mnId);
                g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid + 1 + pKFi->mnId);
                g2o::HyperGraph::Vertex *VA = optimizer.vertex((maxKFid + 1)*2 + pKFi->mnId);
                g2o::HyperGraph::Vertex *VT_d_c = optimizer.vertex((maxKFid + 1)*4);
                g2o::HyperGraph::Vertex *VT_g_d = optimizer.vertex((maxKFid + 1)*4+1);
                g2o::HyperGraph::Vertex *VR_w_b0 = optimizer.vertex((maxKFid + 1)*4+2);
                //				g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
                //				g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
                //				g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
                //				cout<<"VP1: Rcw[0]"<<VP1->estimate().Rcw[0]<<endl;
                //				cout<<"VP1: Rwc"<<VP1->estimate().Rwc<<endl;
                //				cout<<"VP1: tcw[0]"<<VP1->estimate().tcw[0]<<endl;
                //				cout<<"VP1: twc"<<VP1->estimate().twc<<endl;
                //
                //				cout<<"VP2: Rcw[0]"<<VP2->estimate().Rcw[0]<<endl;
                //				cout<<"VP2: Rwc"<<VP2->estimate().Rwc<<endl;
                //				cout<<"VP2: tcw[0]"<<VP2->estimate().tcw[0]<<endl;
                //				cout<<"VP2: twc"<<VP2->estimate().twc<<endl;

                if (!VP1 || !VG || !VP2) {
                    cout << "Error" << VP1 << ", " << VG << ", " << VP2 << endl;

                    continue;
                }
                // prior acc bias
                EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F));
                epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
                epa->setInformation(100 * Eigen::Matrix3d::Identity());
                optimizer.addEdge(epa);
                // prior gyro bias
                EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
                epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
                epg->setInformation(100 * Eigen::Matrix3d::Identity());
                optimizer.addEdge(epg);
                //				EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
                EdgeDvlIMUInitWithoutBias *ei = new EdgeDvlIMUInitWithoutBias(pKFi->mpDvlPreintegrationKeyFrame);
                ei->setLevel(0);
                //				ei->setVertex(0, VP1);

                //			g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                //			ei->setRobustKernel(rk);
                //			rk->setDelta(sqrt(7.815));
                ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
                ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
                ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
                ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
                ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
                ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
                ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_d_c));
                ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VT_g_d));
                ei->setVertex(8, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VR_w_b0));
                ei->setInformation(Eigen::Matrix<double, 3, 3>::Identity()*100);
                ei->setId(pKFi->mnId);
                optimizer.addEdge(ei);
            }
        }


    }


    // optimizer.setVerbose(true);
    optimizer.initializeOptimization(0);
    optimizer.optimize(5);
    // ROS_DEBUG_STREAM(ss.str());  // original
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("aqua_slam"), ss.str());
    // for(VertexGyroBias* v:vpgb){
    //     v->setFixed(false);
    // }
	//
    // vT_d_c->setFixed(false);
    // optimizer.initializeOptimization();
    // optimizer.optimize(5);

    // update gravity direction
    // Eigen::Matrix3d R_b0_w = NormalizeRotation(VGDir->estimate().Rwg);
    Eigen::Matrix3d R_b0_w = VGDir->estimate().Rwg;
//     ROS_INFO_STREAM("gravity refine");  // original
    // Sophus::SO3<double> R_b0_w_SO3(R_b0_w);
    // Eigen::Vector3d R_b0_w_so3 = R_b0_w_SO3.log();
//     // ROS_INFO_STREAM("gravity calibration result: "<< R_b0_w_so3.transpose());  // original
    pAtlas->setRGravity(R_b0_w);
    for(auto m:pAtlas->GetAllMaps()){
        m->setRGravity(R_b0_w);
        m->SetImuInitialized();
    }

}


void Optimizer::DvlBeamOptimization(Map *pMap)
{
	Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);


	optimizer.setAlgorithm(solver);



	VertexDVLBeamOritenstion *v_beam_ori = new VertexDVLBeamOritenstion();
	Eigen::Matrix<double,8,1> alpha_beta;
	alpha_beta << 1.5 / 180.0 * M_PI,
		1.0 / 180.0 * M_PI,
		1.5 / 180.0 * M_PI,
		1.0 / 180.0 * M_PI,
		1.5 / 180.0 * M_PI,
		1.0 / 180.0 * M_PI,
		1.5 / 180.0 * M_PI,
		1.0 / 180.0 * M_PI;
//	alpha_beta << 67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI;
	v_beam_ori->setFixed(false);
	v_beam_ori->setId(0);
	v_beam_ori->setEstimate(alpha_beta);
	optimizer.addVertex(v_beam_ori);

	// Graph edges
	vector<EdgeDvlGyroInit3 *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	std::cout << "build optimization graph" << std::endl;

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpDvlPreintegrationKeyFrame) {
				std::cout << "Not preintegrated measurement" << std::endl;
			}


			EdgeDVLBeamCalibration1* ei = new EdgeDVLBeamCalibration1();
			ei->setVertex(0,v_beam_ori);
			ei->setInformation(Eigen::Matrix<double, 4, 4>::Identity() * 180.0 / M_PI);
			ei->setMeasurement(pKFi->mpDvlPreintegrationKeyFrame);


			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}


	optimizer.setVerbose(true);
	optimizer.initializeOptimization();
	optimizer.optimize(10);

	Eigen::Matrix<double, 8, 1> r_opt = v_beam_ori->estimate();

// 	ROS_INFO_STREAM(  // original
// 		"DVL Calibration(visual data):\nbeam1_theta=" << r_opt(0) / M_PI * 180.0 << " beam1_phi=" << r_opt(1) / M_PI * 180.0  // original
// 		                                 << "\nbeam2_theta=" << r_opt(2) / M_PI * 180.0 << " beam2_phi="  // original
// 		                                 << r_opt(3) / M_PI * 180.0  // original
// 		                                 << "\nbeam3_theta=" << r_opt(4) / M_PI * 180.0 << " beam3_phi="  // original
// 		                                 << r_opt(5) / M_PI * 180.0  // original
// 		                                 << "\nbeam4_theta=" << r_opt(6) / M_PI * 180.0 << " beam4_phi="  // original
// 		                                 << r_opt(7) / M_PI * 180.0);  // original



}

void Optimizer::DvlBeamOptimization_dvl(Map *pMap)
{
	Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
	int its = 200; // Check number of iterations
	long unsigned int maxKFid = pMap->GetMaxKFid();
	const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType *linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);


	optimizer.setAlgorithm(solver);



	VertexDVLBeamOritenstion *v_beam_ori = new VertexDVLBeamOritenstion();
	Eigen::Matrix<double,8,1> alpha_beta;
	alpha_beta << 1.5 / 180.0 * M_PI,
		1.0 / 180.0 * M_PI,
		1.5 / 180.0 * M_PI,
		1.0 / 180.0 * M_PI,
		1.5 / 180.0 * M_PI,
		1.0 / 180.0 * M_PI,
		1.5 / 180.0 * M_PI,
		1.0 / 180.0 * M_PI;
//	alpha_beta << 67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI,
//		67.5 / 180.0 * M_PI,
//		45 / 180.0 * M_PI;
	v_beam_ori->setFixed(false);
	v_beam_ori->setId(0);
	v_beam_ori->setEstimate(alpha_beta);
	optimizer.addVertex(v_beam_ori);

	// Graph edges
	vector<EdgeDvlGyroInit3 *> vpei;
	vpei.reserve(vpKFs.size());
	vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
	vppUsedKF.reserve(vpKFs.size());
	std::cout << "build optimization graph" << std::endl;

	for (size_t i = 0; i < vpKFs.size(); i++) {
		KeyFrame *pKFi = vpKFs[i];

		if (pKFi->mPrevKF && pKFi->mnId <= maxKFid) {
			if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid) {
				continue;
			}
			if (!pKFi->mpDvlPreintegrationKeyFrame) {
				std::cout << "Not preintegrated measurement" << std::endl;
			}


			EdgeDVLBeamCalibration2* ei = new EdgeDVLBeamCalibration2();
			ei->setVertex(0,v_beam_ori);
			ei->setInformation(Eigen::Matrix<double, 4, 4>::Identity() * 180.0 / M_PI);
			ei->setMeasurement(pKFi->mpDvlPreintegrationKeyFrame);


			vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
			optimizer.addEdge(ei);
		}
	}


	optimizer.setVerbose(true);
	optimizer.initializeOptimization();
	optimizer.optimize(10);

	Eigen::Matrix<double, 8, 1> r_opt = v_beam_ori->estimate();

// 	ROS_INFO_STREAM(  // original
// 		"DVL Calibration(DVL data):\nbeam1_theta=" << r_opt(0) / M_PI * 180.0 << " beam1_phi=" << r_opt(1) / M_PI * 180.0  // original
// 		                                 << "\nbeam2_theta=" << r_opt(2) / M_PI * 180.0 << " beam2_phi="  // original
// 		                                 << r_opt(3) / M_PI * 180.0  // original
// 		                                 << "\nbeam3_theta=" << r_opt(4) / M_PI * 180.0 << " beam3_phi="  // original
// 		                                 << r_opt(5) / M_PI * 180.0  // original
// 		                                 << "\nbeam4_theta=" << r_opt(6) / M_PI * 180.0 << " beam4_phi="  // original
// 		                                 << r_opt(7) / M_PI * 180.0);  // original



}

} // namespace ORB_SLAM3
