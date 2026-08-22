//
// Created by da on 24/03/2021.
//

#ifndef ROSHANDLING_H
#define ROSHANDLING_H

// #include <ros/ros.h>  // original
#include <rclcpp/rclcpp.hpp>
// #include <image_transport/image_transport.h>  // original
#include <image_transport/image_transport.hpp>
// #include <cv_bridge/cv_bridge.h>  // original
#include <cv_bridge/cv_bridge.hpp>

#include <octomap/OcTree.h>
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
// #include <sensor_msgs/image_encodings.h>  // original
#include <sensor_msgs/image_encodings.hpp>
// #include <sensor_msgs/Image.h>  // original
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
// #include <sensor_msgs/Imu.h>  // original
#include <sensor_msgs/msg/imu.hpp>
// #include <sensor_msgs/PointCloud2.h>  // original
#include <sensor_msgs/msg/point_cloud2.hpp>
// #include <nav_msgs/Path.h>  // original
#include <nav_msgs/msg/path.hpp>
// #include <nav_msgs/Odometry.h>  // original
#include <nav_msgs/msg/odometry.hpp>
// #include <geometry_msgs/PoseStamped.h>  // original
#include <geometry_msgs/msg/pose_stamped.hpp>
// #include <geometry_msgs/Transform.h>  // original
#include <geometry_msgs/msg/transform.hpp>
// #include <geometry_msgs/TransformStamped.h>  // original
#include <geometry_msgs/msg/transform_stamped.hpp>
// #include <visualization_msgs/Marker.h>  // original
#include <visualization_msgs/msg/marker.hpp>
// #include <visualization_msgs/MarkerArray.h>  // original
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// #include <octomap_msgs/Octomap.h>  // original
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>

// #include <tf/transform_broadcaster.h>  // original
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// #include <std_srvs/Empty.h>  // original
#include <std_srvs/srv/empty.hpp>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
// #include <pcl_ros/transforms.h>  // original
// #include <tf2_sensor_msgs/tf2_sensor_msgs.h>  // original
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>

#include <boost/shared_ptr.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <mutex>
#include <thread>

#include "KeyFrame.h"

using namespace std;

namespace ORB_SLAM3
{
class Atlas;
class System;
class LocalMapping;

class RosHandling
{
public:
	// RosHandling(System *pSys, LocalMapping *pLocal);  // original
	RosHandling(System *pSys, LocalMapping *pLocal, rclcpp::Node::SharedPtr node);
	void PublishLeftImg(const sensor_msgs::msg::Image::SharedPtr &img);
	void PublishRightImg(const sensor_msgs::msg::Image::SharedPtr &img);
	void PublishImgWithInfo(const sensor_msgs::msg::Image::SharedPtr &img);
	void PublishImgMergeCandidate(const cv::Mat &img);
	void PublishIntegration(Atlas *pAtlas);
    void PublishLossKF(set<KeyFrame*,KFComparator> &loss_kfs);
// // 	// void PublishGT(const Eigen::Isometry3d &T_g0_gj_gt, const ros::Time &stamp);  // original  // original
	void PublishOrb(const Eigen::Isometry3d &T_c0_cj_orb, const Eigen::Isometry3d &T_d_c,
	                double timestamp, const cv::Mat &Vwb = cv::Mat());
// // 	void PublishCamera(const Eigen::Isometry3d &T_c0_cj_orb, const ros::Time &stamp);  // original  // original
// // 	// void PublishEkf(const Eigen::Isometry3d &T_e0_ej_ekf, const ros::Time &stamp);  // original  // original
// // 	// void PublishDensePointCloudPose(const Eigen::Isometry3d &T_c0_cmj, const ros::Time &stamp);  // original  // original
	//publish pointcloud and octomap
	void UpdateMap(ORB_SLAM3::Atlas *pAtlas);
	void PublishMap(ORB_SLAM3::Atlas *pAtlas, int state);
    void Run(Atlas* pAtlas);
	void BroadcastTF(const Eigen::Isometry3d &T_c0_cj_orb,
// // 	                 const ros::Time &stamp,  // original  // original
	                 const string &id,
	                 const string &child_id);
    // publish reference integration path
	void GenerateFreePointcloud(const pcl::PointXYZRGB &start,
	                            const pcl::PointXYZRGB &end,
	                            float resolution,
	                            pcl::PointCloud<pcl::PointXYZRGB> &cloud_free);
	double LinearInterpolation(double start_x, double end_x, double start_y, double end_y, double x);
	void PublishLossInteration(const Eigen::Isometry3d &T_e0_er, const Eigen::Isometry3d &T_e0_ec);
	// bool SavePose(std::shared_ptr<std_srvs::srv::Empty::Request> &req, std::shared_ptr<std_srvs::srv::Empty::Response> &res);  // original (ROS2: void, no &)
	void SavePose(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res);
	// bool LoadMap(std::shared_ptr<std_srvs::srv::Empty::Request> &req, std::shared_ptr<std_srvs::srv::Empty::Response> &res);  // original
	void LoadMap(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res);
	// bool CalibrateDVLGyro(std::shared_ptr<std_srvs::srv::Empty::Request> &req, std::shared_ptr<std_srvs::srv::Empty::Response> &res);  // original
	void CalibrateDVLGyro(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res);
	// bool FullBA(std::shared_ptr<std_srvs::srv::Empty::Request> &req, std::shared_ptr<std_srvs::srv::Empty::Response> &res);  // original
	void FullBA(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res);

protected:
	System *mp_system;
	LocalMapping *mp_LocalMapping;
	rclcpp::Node::SharedPtr mp_node;

	std::shared_ptr<image_transport::ImageTransport> mp_it;
	std::shared_ptr<image_transport::Publisher> mp_img_l_pub;
	std::shared_ptr<image_transport::Publisher> mp_img_r_pub;
	std::shared_ptr<image_transport::Publisher> mp_img_info_pub;
	std::shared_ptr<image_transport::Publisher> mp_img_merge_cond_pub;

	//publish qualisys path
	nav_msgs::msg::Path m_integration_path;
// // 	std::shared_ptr<ros::Publisher> mp_integration_path_pub;  // original  // original
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mp_integration_path_pub;
    // publish reference integration path
    nav_msgs::msg::Path m_ref_integration_path;
// //     std::shared_ptr<ros::Publisher> mp_ref_integration_path_pub;  // original  // original
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mp_ref_integration_path_pub;
// //     std::shared_ptr<ros::Publisher> mp_markers_pub;  // original  // original
	rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mp_markers_pub;
	//publish qulisys pose(if exist),
// // 	std::shared_ptr<ros::Publisher> mp_gt_pub;  // original  // original
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mp_gt_pub;
	//publish qualisys path
	nav_msgs::msg::Path m_gt_path;
// // 	std::shared_ptr<ros::Publisher> mp_gt_path_pub;  // original  // original
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mp_gt_path_pub;

	//publish orb pose, odometry and path, in camera frame
// // 	std::shared_ptr<ros::Publisher> mp_pose_orb_pub;  // original  // original
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mp_pose_orb_pub;
// // 	std::shared_ptr<ros::Publisher> mp_odom_orb_pub;  // original  // original
	rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr mp_odom_orb_pub;
	nav_msgs::msg::Path m_path_orb;
// // 	std::shared_ptr<ros::Publisher> mp_path_orb_pub;  // original  // original
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mp_path_orb_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mp_pose_alias_pub;
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mp_path_alias_pub;
	rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr mp_odom_alias_pub;
	// body-frame (FLU) outputs — experimental
	rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr mp_odom_orb_body_pub;
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mp_path_orb_body_pub;
	nav_msgs::msg::Path m_path_orb_body;
// // 	std::shared_ptr<ros::Publisher> mp_pose_orb_camera_pub;  // original  // original
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mp_pose_orb_camera_pub;

	//publish ekf pose and path, in EKF frame
// // 	std::shared_ptr<ros::Publisher> mp_pose_ekf_pub;  // original  // original
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mp_pose_ekf_pub;
	nav_msgs::msg::Path m_path_ekf;
// // 	std::shared_ptr<ros::Publisher> mp_path_ekf_pub;  // original  // original
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mp_path_ekf_pub;

	//publish point cloud
// // 	std::shared_ptr<ros::Publisher> mp_pointcloud_pub;  // original  // original
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mp_pointcloud_pub;
// // 	std::shared_ptr<ros::Publisher> mp_octomap_pub;  // original  // original
	rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr mp_octomap_pub;
// // 	std::shared_ptr<ros::Publisher> mp_map_info_pub;  // original  // original
	rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mp_map_info_pub;

	std::mutex m_mutex_map;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr mp_cloud_occupied;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr mp_cloud_free;
	float m_octomap_resolution;
	std::shared_ptr<octomap::OcTree> mp_octree;

// // 	std::shared_ptr<ros::Publisher> mp_pose_integration_ref_pub;  // original  // original
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mp_pose_integration_ref_pub;
// // 	std::shared_ptr<ros::Publisher> mp_pose_integration_cur_pub;  // original  // original
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mp_pose_integration_cur_pub;

// // 	std::shared_ptr<ros::ServiceServer> mp_save_srv, m_load_srv, m_calib_srv, m_fullBA_srv;  // original  // original
	rclcpp::Service<std_srvs::srv::Empty>::SharedPtr mp_save_srv, mp_load_srv, mp_calib_srv, mp_fullBA_srv;

    // gravity dir of current map
    Eigen::Isometry3d mT_w_c0;
    // IMU←camera extrinsic (T_imu_c), set once from calibration
    Eigen::Isometry3d mT_imu_c;
    // body←IMU extrinsic (T_body_imu), identity if body == IMU
    Eigen::Isometry3d mT_body_imu;
    bool mb_calib_initialized{false};

public:
	void setLocalMapping(LocalMapping *mpLocalMapping)
	{
		mp_LocalMapping = mpLocalMapping;
	}
protected:


	//	todo unfinished

	//publish orb pose, in orb frame
// // 	std::shared_ptr<ros::Publisher> mp_pose_pointcloud_pub;  // original  // original
	// call service to send last N good keyframes once lost feature tracking
// // 	std::shared_ptr<ros::ServiceClient> mp_lost_srv;  // original  // original
	// tf2_ros::TransformBroadcaster m_tb;  // original (no default constructor in ROS2)
	std::shared_ptr<tf2_ros::TransformBroadcaster> m_tb;

};
}

#endif //ROSHANDLING_H
