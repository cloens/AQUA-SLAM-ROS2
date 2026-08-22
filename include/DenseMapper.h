//
// Created by da on 03/12/2021.
//
#ifndef DENSEMAPPER_H
#define DENSEMAPPER_H

// #include <ros/ros.h>  // original
// #include <rosbag/bag.h>
// #include <rosbag/view.h>
#include <rclcpp/rclcpp.hpp>
// #include <sensor_msgs/Image.h>  // original
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/image_encodings.hpp>
// #include <image_transport/image_transport.h>  // original
#include <image_transport/image_transport.hpp>
// #include <image_transport/subscriber_filter.h>  // original
#include <image_transport/subscriber_filter.hpp>
#include <message_filters/sync_policies/approximate_time.h>
// #include <cv_bridge/cv_bridge.h>  // original
#include <cv_bridge/cv_bridge.hpp>
// #include <std_srvs/Empty.h>  // original
#include <std_srvs/srv/empty.hpp>


#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <mutex>

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{
class KeyFrame;

struct DepthEstParamters
{
	float _P1;
	float _P2;
	int _correlation_window_size;
	int _disp12MaxDiff;
	int _disparity_range;
	int _min_disparity;
	int _prefilter_cap;
	int _prefilter_size;
	int _speckle_range;
	int _speckle_size;
	int _texture_threshold;
	float _uniqueness_ratio;
	DepthEstParamters()
	{};
	DepthEstParamters(float P1,
					  float P2,
					  int correlation_window_size,
					  int disp12MaxDiff,
					  int disparity_range,
					  int min_disparity,
					  int prefilter_cap,
					  int prefilter_size,
					  int speckle_range,
					  int speckle_size,
					  int texture_threshold,
					  float uniqueness_ratio)
		: _P1(P1), _P2(P2), _correlation_window_size(correlation_window_size),
		  _disp12MaxDiff(disp12MaxDiff),
		  _disparity_range(disparity_range), _min_disparity(min_disparity),
		  _prefilter_cap(prefilter_cap), _prefilter_size(prefilter_size), _speckle_range(speckle_range),
		  _speckle_size(speckle_size), _texture_threshold(texture_threshold), _uniqueness_ratio(uniqueness_ratio)
	{
        //output all menber value
        cout << "P1: " << _P1 << endl;
        cout << "P2: " << _P2 << endl;
        cout << "correlation_window_size: " << _correlation_window_size << endl;
        cout << "disp12MaxDiff: " << _disp12MaxDiff << endl;
        cout << "disparity_range: " << _disparity_range << endl;
        cout << "min_disparity: " << _min_disparity << endl;
        cout << "prefilter_cap: " << _prefilter_cap << endl;
        cout << "prefilter_size: " << _prefilter_size << endl;
        cout << "speckle_range: " << _speckle_range << endl;
        cout << "speckle_size: " << _speckle_size << endl;
        cout << "texture_threshold: " << _texture_threshold << endl;
        cout << "uniqueness_ratio: " << _uniqueness_ratio << endl;
    };

};

class DenseMapper
{
public:
	// DenseMapper(string settingFile);  // original
	DenseMapper(string settingFile, rclcpp::Node::SharedPtr node);
	std::mutex mDenseMapMutex;
	/***
	 * blobal point cloud in first DVL frame(d0 frame)
	 */
	pcl::PointCloud<pcl::PointXYZRGB> mGlobalMap;

	// point cloud is in local camera frame
	std::mutex mKFMutex;
	std::map<KeyFrame *, pcl::PointCloud<pcl::PointXYZRGB>> mKFWithPointCloud;

	std::mutex mKFQueueMutex;
	std::queue<KeyFrame *> mKFQueue;

	// used for store KF before more than 5 KF in map
	std::queue<KeyFrame *> mKFQueueTemp;

	//ros publisher
	std::shared_ptr<image_transport::Publisher> mDepthPub;
	std::shared_ptr<image_transport::Publisher> mDepthConfPub;
// // 	std::shared_ptr<ros::Publisher> mMapPub;  // original  // original
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mMapPub;
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mMapAliasPub;

	bool mStop = false;
	bool mEnable = true;
	float mLeafSize;
	int mMeanK;
	float mStdThred;

public:
	void InsertNewKF(KeyFrame *pKF);
	void Update();

	void PublishMap();
    void ReconstructFromBagPose(string bag_path,string pose_path);

	void Run();
	void Stop();
	void Resume();
	void Save(string path);

protected:
	rclcpp::Node::SharedPtr mNode;
	double fx, fy, cx, cy, bf;
	DepthEstParamters mParam;

protected:
	inline void Project2Dto3D(int x, int y, float d, Eigen::Vector3d &point)
	{
		point.z() = bf / d;
		point.x() = point.z() * (x - cx) / fx;
		point.y() = point.z() * (y - cy) / fy;
	}

	void ComputeDisp(const Mat &left, const Mat &right, Mat &out, Mat &out_conf, DepthEstParamters &param);

	void GetSubMap(const Mat &img_l, const Mat &img_r, pcl::PointCloud<pcl::PointXYZRGB> &sub_map);
	void GetSubMapFromExternalDepth(const Mat &img_l, const Mat &depth_m, pcl::PointCloud<pcl::PointXYZRGB> &sub_map);
	inline void ProjectDepthTo3D(int x, int y, float z, Eigen::Vector3d &point)
	{
		point.z() = z;
		point.x() = point.z() * (x - cx) / fx;
		point.y() = point.z() * (y - cy) / fy;
	}

	void MergeSubMap(pcl::PointCloud<pcl::PointXYZRGB> &sub_map, KeyFrame *pKF);

};

}


#endif //DENSEMAPPER_H
