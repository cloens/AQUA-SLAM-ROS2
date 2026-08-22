//
// Created by da on 24/03/2021.
//

#include "RosHandling.h"
#include "Atlas.h"
#include "KeyFrame.h"
#include "Map.h"
#include "System.h"
#include "LocalMapping.h"
// #include "visualization_msgs/Marker.h"  // original
#include <visualization_msgs/msg/marker.hpp>
#include <fstream>

using namespace ORB_SLAM3;
using namespace std;

// RosHandling::RosHandling(System *pSys, LocalMapping *pLocal)  // original
// 	: mp_system(pSys),mp_LocalMapping(pLocal)  // original
RosHandling::RosHandling(System *pSys, LocalMapping *pLocal, rclcpp::Node::SharedPtr node)
	: mp_system(pSys), mp_LocalMapping(pLocal), mp_node(node)
{
// // 	ros::NodeHandle nh_;  // original  // original
	mp_it = std::make_shared<image_transport::ImageTransport>(mp_node);

// 	image_transport::Publisher img_l_pub = mp_it->advertise("/aqua_slam/left/image_raw", 10);  // original
// 	mp_img_l_pub =  // original
// 		std::shared_ptr<image_transport::Publisher>(boost::make_shared<image_transport::Publisher>(img_l_pub));  // original
	mp_img_l_pub = std::make_shared<image_transport::Publisher>(mp_it->advertise("/aqua_slam/left/image_raw", 10));

// 	image_transport::Publisher img_r_pub = mp_it->advertise("/aqua_slam/right/image_raw", 10);  // original
// 	mp_img_r_pub =  // original
// 		std::shared_ptr<image_transport::Publisher>(boost::make_shared<image_transport::Publisher>(img_r_pub));  // original
	mp_img_r_pub = std::make_shared<image_transport::Publisher>(mp_it->advertise("/aqua_slam/right/image_raw", 10));

// 	image_transport::Publisher img_info_pub = mp_it->advertise("/aqua_slam/image/features", 10);  // original
// 	mp_img_info_pub =  // original
// 		std::shared_ptr<image_transport::Publisher>(boost::make_shared<image_transport::Publisher>(img_info_pub));  // original
	mp_img_info_pub = std::make_shared<image_transport::Publisher>(mp_it->advertise("/aqua_slam/image/features", 10));

// 	image_transport::Publisher img_merge_pub = mp_it->advertise("/aqua_slam/image/map_merge", 10);  // original
// 	mp_img_merge_cond_pub =  // original
// 		std::shared_ptr<image_transport::Publisher>(boost::make_shared<image_transport::Publisher>(img_merge_pub));  // original
	mp_img_merge_cond_pub = std::make_shared<image_transport::Publisher>(mp_it->advertise("/aqua_slam/image/map_merge", 10));


// // 	// ros::Publisher gt_pub = nh_.advertise<geometry_msgs::PoseStamped>("orb_dvl/gt", 10);  // original  // original
// // 	// mp_gt_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(gt_pub));  // original  // original
// // 	// ros::Publisher gt_path_pub = nh_.advertise<nav_msgs::msg::Path>("orb_path_gt", 10);  // original  // original
// // 	// mp_gt_path_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(gt_path_pub));  // original  // original

// // 	ros::Publisher integration_path_pub = nh_.advertise<nav_msgs::msg::Path>("/aqua_slam/dvl_imu_path", 10);  // original  // original
// // 		std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(integration_path_pub));  // original  // original
	mp_integration_path_pub = mp_node->create_publisher<nav_msgs::msg::Path>("/aqua_slam/dvl_imu_path", 10);
// //     ros::Publisher ref_integration_path_pub = nh_.advertise<nav_msgs::msg::Path>("/aqua_slam/dvl_imu_path_ref", 10);  // original  // original
// //             std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(ref_integration_path_pub));  // original  // original
    mp_ref_integration_path_pub = mp_node->create_publisher<nav_msgs::msg::Path>("/aqua_slam/dvl_imu_path_ref", 10);
    // initialize mp_markers_pub
// //     ros::Publisher markers_pub = nh_.advertise<visualization_msgs::MarkerArray>("/aqua_slam/markers", 10);  // original  // original
// //     mp_markers_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(markers_pub));  // original  // original
    mp_markers_pub = mp_node->create_publisher<visualization_msgs::msg::MarkerArray>("/aqua_slam/markers", 10);
// // 	ros::Publisher pose_orb_pub = nh_.advertise<geometry_msgs::PoseStamped>("/aqua_slam/orb_pose", 10);  // original  // original
// // 	mp_pose_orb_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(pose_orb_pub));  // original  // original
	mp_pose_orb_pub = mp_node->create_publisher<geometry_msgs::msg::PoseStamped>("/aqua_slam/orb_pose", 10);
// // 	ros::Publisher odom_orb_pub = nh_.advertise<nav_msgs::Odometry>("/aqua_slam/orb_odom", 10);  // original  // original
// // 	mp_odom_orb_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(odom_orb_pub));  // original  // original
	mp_odom_orb_pub = mp_node->create_publisher<nav_msgs::msg::Odometry>("/aqua_slam/orb_odom", 10);
// // 	ros::Publisher path_orb_pub = nh_.advertise<nav_msgs::msg::Path>("/aqua_slam/orb_path", 10);  // original  // original
// // 	mp_path_orb_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(path_orb_pub));  // original  // original
	mp_path_orb_pub = mp_node->create_publisher<nav_msgs::msg::Path>("/aqua_slam/orb_path", 10);
	mp_pose_alias_pub = mp_node->create_publisher<geometry_msgs::msg::PoseStamped>("/aqua_slam/pose", 10);
	mp_path_alias_pub = mp_node->create_publisher<nav_msgs::msg::Path>("/aqua_slam/path", 10);
	mp_odom_alias_pub = mp_node->create_publisher<nav_msgs::msg::Odometry>("/aqua_slam/odom", 10);
	mp_odom_orb_body_pub = mp_node->create_publisher<nav_msgs::msg::Odometry>("/aqua_slam/orb_odom_body", 10);
	mp_path_orb_body_pub = mp_node->create_publisher<nav_msgs::msg::Path>("/aqua_slam/orb_path_body", 10);
// // 	ros::Publisher pose_orb_camera_pub = nh_.advertise<nav_msgs::Odometry>("/aqua_slam/camera_pose", 10);  // original  // original
// // 	mp_pose_orb_camera_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(pose_orb_camera_pub));  // original  // original
	mp_pose_orb_camera_pub = mp_node->create_publisher<geometry_msgs::msg::PoseStamped>("/aqua_slam/camera_pose", 10);

// // 	// ros::Publisher pose_ekf_pub = nh_.advertise<geometry_msgs::PoseStamped>("orb_ekf_pose", 10);  // original  // original
// // 	// mp_pose_ekf_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(pose_ekf_pub));  // original  // original
// // 	ros::Publisher path_ekf_pub = nh_.advertise<nav_msgs::msg::Path>("/aqua_slam/ekf_path", 10);  // original  // original
// // 	mp_path_ekf_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(path_ekf_pub));  // original  // original
	mp_path_ekf_pub = mp_node->create_publisher<nav_msgs::msg::Path>("/aqua_slam/ekf_path", 10);

// // 	// ros::Publisher pose_pointcloud_pub = nh_.advertise<geometry_msgs::PoseStamped>("orb_point_pose", 10);  // original  // original
// // 	// mp_pose_pointcloud_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(pose_pointcloud_pub));  // original  // original

// // //	ros::ServiceClient lost_srv = nh_.serviceClient<vehicle_interface::AlarmStoppedTracking>("/ORBSLAM3/lost");  // original  // original
// // //	mp_lost_srv = std::shared_ptr<ros::ServiceClient>(boost::make_shared<ros::ServiceClient>(lost_srv));  // original  // original

// // 	ros::Publisher pointcloud_pub = nh_.advertise<sensor_msgs::msg::PointCloud2>("/aqua_slam/sparse_map", 100);  // original  // original
// // 	mp_pointcloud_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(pointcloud_pub));  // original  // original
	mp_pointcloud_pub = mp_node->create_publisher<sensor_msgs::msg::PointCloud2>("/aqua_slam/sparse_map", 100);

// // 	ros::Publisher octomap_pub = nh_.advertise<octomap_msgs::Octomap>("/aqua_slam/octomap", 10);  // original  // original
// // 	mp_octomap_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(octomap_pub));  // original  // original
	mp_octomap_pub = mp_node->create_publisher<octomap_msgs::msg::Octomap>("/aqua_slam/octomap", 10);
	mp_cloud_occupied = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	mp_cloud_occupied->reserve(5000);
	mp_cloud_free = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	mp_cloud_free->reserve(50000);
	m_octomap_resolution = 0.1;
// 	mp_octree = boost::make_shared<octomap::OcTree>(octomap::OcTree(m_octomap_resolution));  // original
	mp_octree = std::make_shared<octomap::OcTree>(m_octomap_resolution);
// // 	// ros::Publisher map_info_pub = nh_.advertise<vehicle_interface::MapInfo>("/aqua_slam/map_info", 10);  // original  // original
// // 	// mp_map_info_pub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(map_info_pub));  // original  // original

// // 	ros::Publisher pose_integration_ref = nh_.advertise<geometry_msgs::PoseStamped>("/aqua_slam/dvl_imu_pose_ref", 10);  // original  // original
// // 		std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(pose_integration_ref));  // original  // original
	mp_pose_integration_ref_pub = mp_node->create_publisher<geometry_msgs::msg::PoseStamped>("/aqua_slam/dvl_imu_pose_ref", 10);

// // 	ros::Publisher pose_integration_cur = nh_.advertise<geometry_msgs::PoseStamped>("/aqua_slam/dvl_imu_pose", 10);  // original  // original
// // 		std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(pose_integration_cur));  // original  // original
	mp_pose_integration_cur_pub = mp_node->create_publisher<geometry_msgs::msg::PoseStamped>("/aqua_slam/dvl_imu_pose", 10);

// // 	ros::ServiceServer save_srv = nh_.advertiseService("/aqua_slam/save", &RosHandling::SavePose, this);  // original  // original
// // 	mp_save_srv = std::shared_ptr<ros::ServiceServer>(boost::make_shared<ros::ServiceServer>(save_srv));  // original  // original
	mp_save_srv = mp_node->create_service<std_srvs::srv::Empty>("/aqua_slam/save",
		std::bind(&RosHandling::SavePose, this, std::placeholders::_1, std::placeholders::_2));

// // 	ros::ServiceServer load_srv = nh_.advertiseService("/aqua_slam/load_map", &RosHandling::LoadMap, this);  // original  // original
// // 	m_load_srv = std::shared_ptr<ros::ServiceServer>(boost::make_shared<ros::ServiceServer>(load_srv));  // original  // original
	mp_load_srv = mp_node->create_service<std_srvs::srv::Empty>("/aqua_slam/load_map",
		std::bind(&RosHandling::LoadMap, this, std::placeholders::_1, std::placeholders::_2));

// // 	ros::ServiceServer calib_srv = nh_.advertiseService("/aqua_slam/calibrate", &RosHandling::CalibrateDVLGyro, this);  // original  // original
// // 	m_calib_srv = std::shared_ptr<ros::ServiceServer>(boost::make_shared<ros::ServiceServer>(calib_srv));  // original  // original
	mp_calib_srv = mp_node->create_service<std_srvs::srv::Empty>("/aqua_slam/calibrate",
		std::bind(&RosHandling::CalibrateDVLGyro, this, std::placeholders::_1, std::placeholders::_2));

// //     ros::ServiceServer fullBA_srv = nh_.advertiseService("/aqua_slam/full_ba", &RosHandling::FullBA, this);  // original  // original
// //     m_fullBA_srv = std::shared_ptr<ros::ServiceServer>(boost::make_shared<ros::ServiceServer>(fullBA_srv));  // original  // original
	mp_fullBA_srv = mp_node->create_service<std_srvs::srv::Empty>("/aqua_slam/full_ba",
		std::bind(&RosHandling::FullBA, this, std::placeholders::_1, std::placeholders::_2));

	m_tb = std::make_shared<tf2_ros::TransformBroadcaster>(mp_node);

    mT_w_c0.setIdentity();

    // publish empty messages so RViz2 clears stale path/pointcloud on node restart
    {
        auto stamp = mp_node->now();

        nav_msgs::msg::Path empty_path;
        empty_path.header.stamp = stamp;
        empty_path.header.frame_id = "aqua_slam";
        mp_path_orb_pub->publish(empty_path);
        mp_path_orb_body_pub->publish(empty_path);
        mp_integration_path_pub->publish(empty_path);

        sensor_msgs::msg::PointCloud2 empty_cloud;
        empty_cloud.header.stamp = stamp;
        empty_cloud.header.frame_id = "aqua_slam";
        mp_pointcloud_pub->publish(empty_cloud);
    }
}


void RosHandling::PublishLeftImg(const sensor_msgs::msg::Image::SharedPtr &img)
{
	mp_img_l_pub->publish(img);
}
void RosHandling::PublishRightImg(const sensor_msgs::msg::Image::SharedPtr &img)
{
	mp_img_r_pub->publish(img);
}
void RosHandling::PublishImgWithInfo(const sensor_msgs::msg::Image::SharedPtr &img)
{
	img->header.stamp = mp_node->now();
	img->header.frame_id = "aqua_slam";
	mp_img_info_pub->publish(img);
}

/* // original
void RosHandling::PublishOrb(const Eigen::Isometry3d &T_c0_cj_orb,
                             const Eigen::Isometry3d &T_d_c,
// //                              const ros::Time &stamp)  // original  // original
{
	//conversion from ENU to NED
	// Eigen::AngleAxisd r_x(M_PI,Eigen::Vector3d::UnitX());
	// Eigen::AngleAxisd r_z(-0.5*M_PI,Eigen::Vector3d::UnitZ());
	// Eigen::Isometry3d T_enu_ned=Eigen::Isometry3d::Identity();
	// T_enu_ned.rotate(r_x);
	// T_enu_ned.rotate(r_z);
	// Eigen::Isometry3d T_c0_cj_orb_NED=T_c0_cj_orb*T_enu_ned;
    Eigen::Isometry3d T_c_rviz = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd r_z(M_PI / 2, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd r_y(-M_PI / 2, Eigen::Vector3d::UnitY());
    T_c_rviz.rotate(r_z);
    T_c_rviz.rotate(r_y);

	Eigen::Isometry3d T_w_cj = mT_w_c0 * T_c0_cj_orb;
	Eigen::Isometry3d T_d0_dj = T_d_c * T_c0_cj_orb * T_d_c.inverse();

	geometry_msgs::msg::PoseStamped pose_to_pub;
	pose_to_pub.header.frame_id = "aqua_slam";
// // 	//pose_to_pub.header.stamp=ros::Time::now();  // original  // original
	pose_to_pub.header.stamp = stamp;
	pose_to_pub.pose.position.x = T_w_cj.translation().x();
	pose_to_pub.pose.position.y = T_w_cj.translation().y();
	pose_to_pub.pose.position.z = T_w_cj.translation().z();
	// Eigen::Matrix3d rotation_matrix;
	// rotation_matrix<< R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
	// R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
	// R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);
	Eigen::Quaterniond rotation_q(T_w_cj.rotation());
	pose_to_pub.pose.orientation.x = rotation_q.x();
	pose_to_pub.pose.orientation.y = rotation_q.y();
	pose_to_pub.pose.orientation.z = rotation_q.z();
	pose_to_pub.pose.orientation.w = rotation_q.w();

	mp_pose_orb_pub->publish(pose_to_pub);
	// BroadcastTF(T_d0_dj, stamp, "aqua_slam", "camera_dvl2_link");
//	m_path_orb.header = pose_to_pub.header;
//	m_path_orb.poses.push_back(pose_to_pub);
//	mp_path_orb_pub->publish(m_path_orb);


	// pose_to_pub.pose.position.x = T_c0_cj_orb_NED.translation().x();
	// pose_to_pub.pose.position.y = T_c0_cj_orb_NED.translation().y();
	// pose_to_pub.pose.position.z = T_c0_cj_orb_NED.translation().z();
	pose_to_pub.pose.position.x = T_w_cj.translation().x();
	pose_to_pub.pose.position.y = T_w_cj.translation().y();
	pose_to_pub.pose.position.z = T_w_cj.translation().z();
	// Eigen::Matrix3d rotation_matrix;
	// rotation_matrix<< R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
	// R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
	// R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);

	// rotation_q =T_c0_cj_orb_NED.rotation();
	rotation_q = T_w_cj.rotation();
	pose_to_pub.pose.orientation.x = rotation_q.x();
	pose_to_pub.pose.orientation.y = rotation_q.y();
	pose_to_pub.pose.orientation.z = rotation_q.z();
	pose_to_pub.pose.orientation.w = rotation_q.w();
	Eigen::Isometry3d T_w_rviz = T_w_cj * T_c_rviz;
	BroadcastTF(T_w_rviz, stamp, "aqua_slam", "/bluerov/base_link");
	nav_msgs::msg::Odometry odom;
	odom.header = pose_to_pub.header;
	odom.pose.pose = pose_to_pub.pose;
	mp_odom_orb_pub->publish(odom);
}
*/ // original

void RosHandling::PublishOrb(const Eigen::Isometry3d &T_c0_cj_orb,
                             const Eigen::Isometry3d &T_d_c,
                             double timestamp,
                             const cv::Mat &Vwb)
{
    Eigen::Isometry3d T_c_rviz = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd r_z(M_PI / 2, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd r_y(-M_PI / 2, Eigen::Vector3d::UnitY());
    T_c_rviz.rotate(r_z);
    T_c_rviz.rotate(r_y);

    Eigen::Isometry3d T_w_cj = mT_w_c0 * T_c0_cj_orb;

    geometry_msgs::msg::PoseStamped pose_to_pub;
    pose_to_pub.header.frame_id = "aqua_slam";
    pose_to_pub.header.stamp = rclcpp::Time(static_cast<int64_t>(timestamp * 1e9));
    pose_to_pub.pose.position.x = T_w_cj.translation().x();
    pose_to_pub.pose.position.y = T_w_cj.translation().y();
    pose_to_pub.pose.position.z = T_w_cj.translation().z();
    Eigen::Quaterniond rotation_q(T_w_cj.rotation());
    pose_to_pub.pose.orientation.x = rotation_q.x();
    pose_to_pub.pose.orientation.y = rotation_q.y();
    pose_to_pub.pose.orientation.z = rotation_q.z();
    pose_to_pub.pose.orientation.w = rotation_q.w();
    mp_pose_orb_pub->publish(pose_to_pub);
    if (mp_pose_alias_pub) mp_pose_alias_pub->publish(pose_to_pub);

    m_path_orb.header = pose_to_pub.header;
    m_path_orb.poses.push_back(pose_to_pub);
    mp_path_orb_pub->publish(m_path_orb);
    if (mp_path_alias_pub) mp_path_alias_pub->publish(m_path_orb);

    Eigen::Isometry3d T_w_rviz = T_w_cj * T_c_rviz;
    BroadcastTF(T_w_rviz, "aqua_slam", "bluerov/base_link");
    nav_msgs::msg::Odometry odom;
    odom.header = pose_to_pub.header;
    odom.pose.pose = pose_to_pub.pose;
    if (!Vwb.empty() && Vwb.rows == 3) {
        // Vwb: velocity of body expressed in world frame (from IMU propagation / g2o BA)
        cv::Mat Vwb_f;
        Vwb.convertTo(Vwb_f, CV_32F);
        odom.twist.twist.linear.x = Vwb_f.at<float>(0);
        odom.twist.twist.linear.y = Vwb_f.at<float>(1);
        odom.twist.twist.linear.z = Vwb_f.at<float>(2);
    }
    mp_odom_orb_pub->publish(odom);
    if (mp_odom_alias_pub) mp_odom_alias_pub->publish(odom);

    if (mb_calib_initialized) {
        Eigen::Isometry3d T_w_bj = T_w_cj * mT_imu_c.inverse() * mT_body_imu.inverse();
        nav_msgs::msg::Odometry odom_body;
        odom_body.header = pose_to_pub.header;
        odom_body.child_frame_id = "base_link";
        Eigen::Quaterniond q_body(T_w_bj.rotation());
        odom_body.pose.pose.position.x = T_w_bj.translation().x();
        odom_body.pose.pose.position.y = T_w_bj.translation().y();
        odom_body.pose.pose.position.z = T_w_bj.translation().z();
        odom_body.pose.pose.orientation.x = q_body.x();
        odom_body.pose.pose.orientation.y = q_body.y();
        odom_body.pose.pose.orientation.z = q_body.z();
        odom_body.pose.pose.orientation.w = q_body.w();
        if (!Vwb.empty() && Vwb.rows == 3) {
            cv::Mat Vwb_f;
            Vwb.convertTo(Vwb_f, CV_32F);
            Eigen::Vector3d v_w(Vwb_f.at<float>(0), Vwb_f.at<float>(1), Vwb_f.at<float>(2));
            Eigen::Vector3d v_b = T_w_bj.rotation().transpose() * v_w;
            odom_body.twist.twist.linear.x = v_b.x();
            odom_body.twist.twist.linear.y = v_b.y();
            odom_body.twist.twist.linear.z = v_b.z();
        }
        mp_odom_orb_body_pub->publish(odom_body);
    }
}

void RosHandling::UpdateMap(ORB_SLAM3::Atlas *pAtlas)
{
	const std::lock_guard<std::mutex> guard(m_mutex_map);
    if(!pAtlas->isImuInitialized()){
        return;
    }
//	pcl::PointCloud<pcl::PointXYZRGB> cloud;
//	octomap::OcTree tree(0.1);
	mp_cloud_occupied->clear();
	mp_cloud_free->clear();
	mp_octree->clear();
    Map* p_first_map = pAtlas->GetAllMaps().front();
    Eigen::Matrix3d R_b0_w = pAtlas->getRGravity();
    cv::Mat T_imu_c_cv = p_first_map->GetOriginKF()->mImuCalib.mT_imu_c.clone();
    cv::Mat T_d_c = p_first_map->GetOriginKF()->mImuCalib.mT_dvl_c.clone();
    Eigen::Isometry3d T_imu_c = Eigen::Isometry3d::Identity();cv::cv2eigen(T_imu_c_cv,T_imu_c.matrix());
    Eigen::Isometry3d T_body_imu = Eigen::Isometry3d::Identity();
    cv::Mat T_body_imu_cv = p_first_map->GetOriginKF()->mImuCalib.mT_body_imu.clone();
    if (!T_body_imu_cv.empty()) {
        cv::cv2eigen(T_body_imu_cv, T_body_imu.matrix());
    }
    if (!mb_calib_initialized) {
        mT_imu_c = T_imu_c;
        mT_body_imu = T_body_imu;
        mb_calib_initialized = true;
    }

    // Eigen::Isometry3d T_b0_w;
    // T_b0_w.setIdentity();
    // T_b0_w.rotate(R_b0_w);
    Eigen::Isometry3d T_w_c0 = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d R_w_c0 = R_b0_w.inverse() * T_imu_c.rotation();
    T_w_c0.rotate(R_w_c0);
    T_w_c0.pretranslate(R_b0_w.inverse() * T_imu_c.translation());
    // Correct initial body yaw: ORB-SLAM3 minimal-yaw may not align body Forward with world +x.
    // R_world_body0 = R_b0_w.inverse() * T_body_imu.inverse().rotation() (T_imu_c cancels out).
    {
        Eigen::Matrix3d R_world_body0 = R_b0_w.inverse() * T_body_imu.inverse().rotation();
        Eigen::Vector3d fwd_horiz = R_world_body0.col(0);
        fwd_horiz.z() = 0.0;
        if (fwd_horiz.norm() > 1e-6) {
            fwd_horiz.normalize();
            double yaw_err = atan2(fwd_horiz.y(), fwd_horiz.x());
            T_w_c0.prerotate(Eigen::AngleAxisd(-yaw_err, Eigen::Vector3d::UnitZ()));
        }
    }
    mT_w_c0 = T_w_c0;
// //    ROS_INFO_STREAM("pub T_w_c0: \n"<<T_w_c0.matrix());  // original

	vector<Map *> allMaps = pAtlas->GetAllMaps();
	for (vector<Map *>::iterator it = allMaps.begin(); it != allMaps.end(); it++) {
		Map *pMap = *it;
        // Eigen::Matrix3d R_b0_w = pAtlas->getRGravity();
        // cv::Mat T_imu_c_cv = pMap->GetOriginKF()->mImuCalib.mT_imu_c.clone();
        // Eigen::Isometry3d T_imu_c = Eigen::Isometry3d::Identity();
        // cv::cv2eigen(T_imu_c_cv,T_imu_c.matrix());
		const vector<MapPoint *> &vpMPs = pMap->GetAllMapPoints();
		const Eigen::Vector3d &color = pMap->mColor * 255;

		if (vpMPs.empty()) {
			return;
		}
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, free_cloud;
        cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->reserve(50000);
        free_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        free_cloud->reserve(100000);
//		glPointSize(mPointSize);
//		glBegin(GL_POINTS);
//		glColor3f(color[0],color[1],color[2]);


		for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
			if (vpMPs[i]->isBad() || vpMPs[i]->Observations()<5) {
				continue;
			}
			cv::Mat pos_end = vpMPs[i]->GetWorldPos();
//			glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
//			pcl::PointXYZRGB p(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2),(uint8_t)color[0],(uint8_t)color[1],(uint8_t)color[2]);
			pcl::PointXYZRGB p_end;
			p_end.x = pos_end.at<float>(0);
			p_end.y = pos_end.at<float>(1);
			p_end.z = pos_end.at<float>(2);
			p_end.r = (unsigned char)color[0];
			p_end.g = (unsigned char)color[1];
			p_end.b = (unsigned char)color[2];
            cloud->push_back(p_end);


			KeyFrame *ref_kf = vpMPs[i]->GetReferenceKeyFrame();
			cv::Mat pos_start = ref_kf->GetPose();

			Eigen::Isometry3d T_cj_c0 = Eigen::Isometry3d::Identity();
			cv::cv2eigen(pos_start, T_cj_c0.matrix());
			Eigen::Isometry3d T_c0_cj = T_cj_c0.inverse();

			pcl::PointXYZRGB p_start;
			p_start.x = T_c0_cj.translation().x();
			p_start.y = T_c0_cj.translation().y();
			p_start.z = T_c0_cj.translation().z();

			// GenerateFreePointcloud(p_start, p_end, m_octomap_resolution, *free_cloud);
//			cout << "add free pointcloud number: " << free_cloud.size() << endl;


		}
        // Eigen::Isometry3d T_w_c0 = Eigen::Isometry3d::Identity();
        // Eigen::Matrix3d R_w_c0 = R_b0_w.inverse() * T_body_c.rotation();
        // T_w_c0.rotate(R_w_c0);

        Eigen::Isometry3d T_w_c0 = mT_w_c0;

        pcl::transformPointCloud(*cloud, *cloud, T_w_c0.matrix());
        // pcl::transformPointCloud(*free_cloud, *free_cloud, T_w_c0.matrix());

        // *mp_cloud_free += *free_cloud;
        *mp_cloud_occupied += *cloud;

	}
//	cout << "total free pointcloud number: " << mp_cloud_free->size() << endl;

    sensor_msgs::msg::PointCloud2 sparse_map;
    pcl::toROSMsg(*mp_cloud_occupied, sparse_map);
    sparse_map.header.frame_id = "aqua_slam";
    mp_pointcloud_pub->publish(sparse_map);

	// remove noise map point
// 	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
// 	sor.setInputCloud(mp_cloud_occupied);
// 	sor.setMeanK(50);
// 	sor.setStddevMulThresh(1);
// 	sor.filter(*mp_cloud_occupied);
//
// 	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
// 	vg.setInputCloud(mp_cloud_free);
// 	vg.setLeafSize(0.05f, 0.05f, 0.05f);
// 	vg.filter(*mp_cloud_free);
// //	pcl::io::savePCDFileBinary("./data/map.pcd", *mp_cloud_occupied);
// 	vector<KeyFrame *> vpKFs = pAtlas->GetAllKeyFrames();
// 	KeyFrame *kp = vpKFs[0];
// //	Eigen::Isometry3d T_e_c=
// 	Eigen::Isometry3d T_d_c_eigen;
// 	cv::cv2eigen(T_d_c, T_d_c_eigen.matrix());
// 	Eigen::Isometry3d T_rviz_orb = Eigen::Isometry3d::Identity();
// 	Eigen::AngleAxisd r1(M_PI / 2, Eigen::Vector3d::UnitY());
// 	Eigen::AngleAxisd r2(-M_PI / 2, Eigen::Vector3d::UnitZ());
// 	T_rviz_orb.rotate(r1);
// 	T_rviz_orb.rotate(r2);
// 	// convert map point from orb frame to rviz frame
// 	// pcl::transformPointCloud(*mp_cloud_occupied, *mp_cloud_occupied, T_d_c_eigen.matrix());
// 	// pcl::transformPointCloud(*mp_cloud_free, *mp_cloud_free, T_d_c_eigen.matrix());
//
// 	for (auto p: *mp_cloud_occupied) {
// 		octomap::point3d p_oct(p.x, p.y, p.z);
// 		mp_octree->updateNode(p_oct, true);
// 	}
// 	for (auto p: *mp_cloud_free) {
// 		octomap::point3d p_oct(p.x, p.y, p.z);
// 		mp_octree->updateNode(p_oct, false);
// 	}
//
// 	// sensor_msgs::msg::PointCloud2 sparse_map;
// 	// pcl::toROSMsg(*mp_cloud_occupied, sparse_map);
// 	// sparse_map.header.frame_id = "aqua_slam";
// 	// mp_pointcloud_pub->publish(sparse_map);
//
//
// 	octomap_msgs::Octomap map;
// 	map.header = sparse_map.header;
// 	octomap_msgs::fullMapToMsg(*mp_octree, map);
// 	mp_octomap_pub->publish(map);
}
void RosHandling::PublishMap(ORB_SLAM3::Atlas *pAtlas, int state)
{
	int current_map_id = pAtlas->GetCurrentMap()->GetId();
	vector<int> map_ids;
	vector<Map *> maps = pAtlas->GetAllMaps();
	for (auto map: maps) {
		map_ids.push_back(map->GetId());
	}

}
void RosHandling::BroadcastTF(const Eigen::Isometry3d &T_c0_cj_orb,
// //                               const ros::Time &stamp,  // original  // original
                              const string &id,
                              const string &child_id)
{
	Eigen::Quaterniond rotation_q(T_c0_cj_orb.rotation());
	// tf::StampedTransform used in ROS1; replaced with geometry_msgs::msg::TransformStamped in ROS2  // original
	geometry_msgs::msg::TransformStamped tf_stamped;
	tf_stamped.header.stamp = mp_node->now();
	tf_stamped.header.frame_id = id;
	tf_stamped.child_frame_id = child_id;
	tf_stamped.transform.translation.x = T_c0_cj_orb.translation().x();
	tf_stamped.transform.translation.y = T_c0_cj_orb.translation().y();
	tf_stamped.transform.translation.z = T_c0_cj_orb.translation().z();
	tf_stamped.transform.rotation.x = rotation_q.x();
	tf_stamped.transform.rotation.y = rotation_q.y();
	tf_stamped.transform.rotation.z = rotation_q.z();
	tf_stamped.transform.rotation.w = rotation_q.w();
	m_tb->sendTransform(tf_stamped);
}

void RosHandling::GenerateFreePointcloud(const pcl::PointXYZRGB &start,
                                         const pcl::PointXYZRGB &end,
                                         float resolution,
                                         pcl::PointCloud<pcl::PointXYZRGB> &cloud_free)
{
	double start_x = start.x, start_y = start.y, start_z = start.z, end_x = end.x, end_y = end.y, end_z = end.z;
	double dis_x = abs(end_x - start_x);
	double dis_y = abs(end_y - start_y);
	double dis_z = abs(end_z - start_z);
	double max_dis = max(max(dis_x, dis_y), dis_z);
//	cout<<"max_dis: "<<max_dis<<endl;
	if (max_dis >= 3.0) {
//		cout << "distance too large! skip" << endl;
		return;
	}


	if (max_dis == dis_x) {
		if (end_x > start_x) {
			double steps = (end_x - start_x) / resolution;
//			cout << "intepolate along x-axis, steps: " << steps << endl;
			for (int i = 0; i < steps - 1; ++i) {
				pcl::PointXYZRGB p;
				p.x = start_x + resolution * i;
				p.y = LinearInterpolation(start_x, end_x, start_y, end_y, p.x);
				p.z = LinearInterpolation(start_x, end_x, start_z, end_z, p.x);
				cloud_free.push_back(p);
			}
		}
		else {
			double steps = (start_x - end_x) / resolution;
//			cout << "intepolate along x-axis, steps: " << steps << endl;
			for (int i = 0; i < steps - 1; ++i) {
				pcl::PointXYZRGB p;
				p.x = end_x + resolution * i;
				p.y = LinearInterpolation(end_x, start_x, end_y, start_y, p.x);
				p.z = LinearInterpolation(end_x, start_x, end_z, start_z, p.x);
				cloud_free.push_back(p);
			}
		}
	}
	else if (max_dis == dis_y) {
		if (end_y > start_y) {
			double steps = (end_y - start_y) / resolution;
//			cout << "intepolate along y-axis, steps: " << steps << endl;
			for (int i = 0; i < steps - 1; ++i) {
				pcl::PointXYZRGB p;
				p.y = start_y + resolution * i;
				p.x = LinearInterpolation(start_y, end_y, start_x, end_x, p.y);
				p.z = LinearInterpolation(start_y, end_y, start_z, end_z, p.y);
				cloud_free.push_back(p);
			}
		}
		else {
			double steps = (start_y - end_y) / resolution;
//			cout << "intepolate along y-axis, steps: " << steps << endl;
			for (int i = 0; i < steps - 1; ++i) {
				pcl::PointXYZRGB p;
				p.y = end_y + resolution * i;
				p.x = LinearInterpolation(end_y, start_y, end_x, start_x, p.y);
				p.z = LinearInterpolation(end_y, start_y, end_z, start_z, p.y);
				cloud_free.push_back(p);
			}
		}

	}
	else {
		if (end_z > start_z) {
			double steps = (end_z - start_z) / resolution;
//			cout << "intepolate along z-axis, steps: " << steps << endl;
			for (int i = 0; i < steps - 1; ++i) {
				pcl::PointXYZRGB p;
				p.z = start_z + resolution * i;
				p.y = LinearInterpolation(start_z, end_z, start_y, end_y, p.z);
				p.x = LinearInterpolation(start_z, end_z, start_x, end_x, p.z);
				cloud_free.push_back(p);
			}
		}
		else {
			double steps = (start_z - end_z) / resolution;
//			cout << "intepolate along z-axis, steps: " << steps << endl;
			for (int i = 0; i < steps - 1; ++i) {
				pcl::PointXYZRGB p;
				p.z = end_z + resolution * i;
				p.y = LinearInterpolation(end_z, start_z, end_y, start_y, p.z);
				p.x = LinearInterpolation(end_z, start_z, end_x, start_x, p.z);
				cloud_free.push_back(p);
			}
		}

	}

}

double RosHandling::LinearInterpolation(double start_x, double end_x, double start_y, double end_y, double x)
{
	return start_y + (x - start_x) * ((end_y - start_y) / (end_x - start_x));
}
void RosHandling::PublishIntegration(Atlas *pAtlas)
{
    if(!pAtlas->isImuInitialized()){
        return;
    }
    auto maps = pAtlas->GetAllMaps();
    set<KeyFrame*,KFComparator> all_kf;
    visualization_msgs::msg::MarkerArray all_markers;
    m_integration_path.poses.clear();
    m_path_orb.poses.clear();
    m_path_orb_body.poses.clear();
    m_ref_integration_path.poses.clear();
    cv::Mat T_d_c_cv = maps.front()->GetOriginKF()->mImuCalib.mT_dvl_c.clone();
    cv::Mat T_g_d_cv = maps.front()->GetOriginKF()->mImuCalib.mT_imu_dvl.clone();
    Eigen::Isometry3d T_g_d = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_d_c = Eigen::Isometry3d::Identity();
    cv::cv2eigen(T_g_d_cv, T_g_d.matrix());
    cv::cv2eigen(T_d_c_cv, T_d_c.matrix());
    cv::Mat R_g_d_cv = T_g_d_cv.rowRange(0, 3).colRange(0, 3);
    Eigen::Matrix3d R_g_d = T_g_d.rotation();
    // handle gravity dir
    Eigen::Matrix3d R_b0_w = pAtlas->getRGravity();
    Eigen::Isometry3d T_w_c0 = mT_w_c0;

    for(auto pMap:maps){
        auto pKFs = pMap->GetAllKeyFrames();
        if (pKFs.empty()) {
            return;
        }
        else if (pKFs[0]->GetPoseInverse().empty()) {
            return;
        }

        KeyFrame *pKF = pKFs[0];
        while (pKF->mPrevKF) {
            pKF = pKF->mPrevKF;
        }

        vector<Eigen::Isometry3d> poses_integration;
        Eigen::Isometry3d T_c_rviz = Eigen::Isometry3d::Identity();
        Eigen::AngleAxisd r_z(M_PI / 2, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd r_y(-M_PI / 2, Eigen::Vector3d::UnitY());
        T_c_rviz.rotate(r_z);
        T_c_rviz.rotate(r_y);

        Eigen::Isometry3d T_d0_dj = Eigen::Isometry3d::Identity();
        Eigen::Isometry3d T_c0_c1 = Eigen::Isometry3d::Identity();
        //	R_g_d.convertTo(R_g_d,CV_64F);




        // loss integration from loss ref KF to first KF in the new Map
        m_ref_integration_path.poses.clear();


        while (pKF) {
            all_kf.insert(pKF);
            cv::Mat T_c0_c1_cv = pKF->GetPoseInverse();
            cv::cv2eigen(T_c0_c1_cv, T_c0_c1.matrix());

            Eigen::Isometry3d T_d0_d1 = T_d_c * T_c0_c1 * T_d_c.inverse();
            // T_d0_dj = T_d0_dj * T_d0_d1;
            if(pKF->mpDvlPreintegrationLossRefKF){
                Eigen::Matrix3d R_gf_g1 = Eigen::Matrix3d::Identity();
                Eigen::Vector3d t_df_df_d1 = Eigen::Vector3d::Identity();
                cv::cv2eigen(pKF->mpDvlPreintegrationLossRefKF->dR,R_gf_g1);
                cv::cv2eigen(pKF->mpDvlPreintegrationLossRefKF->dP_dvl,t_df_df_d1);
                Eigen::Matrix3d R_df_d1 =  R_g_d.inverse() * R_gf_g1 * R_g_d;
                Eigen::Isometry3d T_df_d1 = Eigen::Isometry3d::Identity();
                T_df_d1.pretranslate(t_df_df_d1);
                T_df_d1.rotate(R_df_d1);
                Eigen::Isometry3d T_cf_c1 = T_d_c.inverse() * T_df_d1 * T_d_c;
                Eigen::Isometry3d T_w_c1 = T_w_c0 * T_c0_c1;
                Eigen::Isometry3d T_w_cf = T_w_c0 * T_c0_c1 * T_cf_c1.inverse();

                cv::Mat T_c0_cf_cv = pKF->mpLossRefKF->GetPoseInverse();
                Eigen::Isometry3d T_c0_cf;
                cv::cv2eigen(T_c0_cf_cv,T_c0_cf.matrix());

                Eigen::Matrix3d R_b0_w_ref = pKF->GetMap()->getRGravity();
                Eigen::Isometry3d T_w_c0_ref = Eigen::Isometry3d::Identity();
                Eigen::Matrix3d R_w_c0_ref = R_b0_w_ref.inverse() * (T_g_d * T_d_c).rotation();
                T_w_c0_ref.rotate(R_w_c0_ref);
                Eigen::Isometry3d T_w_cf_ref = T_w_c0_ref * T_c0_cf;

                geometry_msgs::msg::PoseStamped pose_to_pub;
                pose_to_pub.header.frame_id = "aqua_slam";
                pose_to_pub.header.stamp = rclcpp::Time(static_cast<int64_t>(pKF->mTimeStamp * 1e9));
// //                 //pose_to_pub.header.stamp=ros::Time::now();  // original  // original
// //                 pose_to_pub.header.stamp = ros::Time(pKF->mTimeStamp);  // original  // original
                pose_to_pub.pose.position.x = T_w_c1.translation().x();
                pose_to_pub.pose.position.y = T_w_c1.translation().y();
                pose_to_pub.pose.position.z = T_w_c1.translation().z();
                // Eigen::Matrix3d rotation_matrix;
                // rotation_matrix<< R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
                // R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
                // R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);
                Eigen::Quaterniond rotation_q(T_w_c1.rotation());
                pose_to_pub.pose.orientation.x = rotation_q.x();
                pose_to_pub.pose.orientation.y = rotation_q.y();
                pose_to_pub.pose.orientation.z = rotation_q.z();
                pose_to_pub.pose.orientation.w = rotation_q.w();
                m_ref_integration_path.poses.push_back(pose_to_pub);

                pose_to_pub.pose.position.x = T_w_cf.translation().x();
                pose_to_pub.pose.position.y = T_w_cf.translation().y();
                pose_to_pub.pose.position.z = T_w_cf.translation().z();
                // Eigen::Matrix3d rotation_matrix;
                // rotation_matrix<< R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
                // R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
                // R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);
                rotation_q = Eigen::Quaterniond(T_w_cf.rotation());
                pose_to_pub.pose.orientation.x = rotation_q.x();
                pose_to_pub.pose.orientation.y = rotation_q.y();
                pose_to_pub.pose.orientation.z = rotation_q.z();
                pose_to_pub.pose.orientation.w = rotation_q.w();
                m_ref_integration_path.poses.push_back(pose_to_pub);

                pose_to_pub.pose.position.x = T_w_cf_ref.translation().x();
                pose_to_pub.pose.position.y = T_w_cf_ref.translation().y();
                pose_to_pub.pose.position.z = T_w_cf_ref.translation().z();
                // Eigen::Matrix3d rotation_matrix;
                // rotation_matrix<< R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
                // R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
                // R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);
                rotation_q = Eigen::Quaterniond(T_w_cf_ref.rotation());
                pose_to_pub.pose.orientation.x = rotation_q.x();
                pose_to_pub.pose.orientation.y = rotation_q.y();
                pose_to_pub.pose.orientation.z = rotation_q.z();
                pose_to_pub.pose.orientation.w = rotation_q.w();
                // m_ref_integration_path.poses.push_back(pose_to_pub);

                m_ref_integration_path.header = pose_to_pub.header;
            }

            cv::Mat R_gi_gj_cv = pKF->mpDvlPreintegrationKeyFrame->GetDeltaRotation(pKF->GetImuBias());
            cv::Mat t_di_dj_cv = pKF->mpDvlPreintegrationKeyFrame->GetDVLPosition(pKF->GetImuBias());
            R_gi_gj_cv.convertTo(R_gi_gj_cv, CV_32F);
            cv::Mat R_di_dj_cv = R_g_d_cv.t() * R_gi_gj_cv * R_g_d_cv;
            Eigen::Matrix3d R_di_dj;
            cv::cv2eigen(R_di_dj_cv, R_di_dj);
            Eigen::Vector3d t_di_dj;
            cv::cv2eigen(t_di_dj_cv, t_di_dj);
            Eigen::Isometry3d T_di_dj = Eigen::Isometry3d::Identity();
            T_di_dj.pretranslate(t_di_dj);
            T_di_dj.rotate(R_di_dj);
            T_d0_dj = T_d0_dj * T_di_dj;
//             // ROS_INFO_STREAM("KF ID:"<<pKF->mnId<<" integration: \n"<<T_di_dj.matrix());  // original
            // todo_tightly
            // save inverse
            //		Eigen::Isometry3d T_c0_cj_integration = T_c_enu.inverse() * T_d_c.inverse() * T_d0_dj * T_d_c * T_c_enu;
            Eigen::Isometry3d T_w_cj_integration = T_w_c0 * T_d_c.inverse() * T_d0_dj * T_d_c;
            poses_integration.push_back(T_w_cj_integration);

            geometry_msgs::msg::PoseStamped pose_to_pub;
            pose_to_pub.header.frame_id = "aqua_slam";
            pose_to_pub.header.stamp = rclcpp::Time(static_cast<int64_t>(pKF->mTimeStamp * 1e9));
// //             //pose_to_pub.header.stamp=ros::Time::now();  // original  // original
// //             pose_to_pub.header.stamp = ros::Time(pKF->mTimeStamp);  // original  // original
            pose_to_pub.pose.position.x = T_w_cj_integration.translation().x();
            pose_to_pub.pose.position.y = T_w_cj_integration.translation().y();
            pose_to_pub.pose.position.z = T_w_cj_integration.translation().z();
            // Eigen::Matrix3d rotation_matrix;
            // rotation_matrix<< R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
            // R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
            // R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);
            Eigen::Quaterniond rotation_q(T_w_cj_integration.rotation());
            pose_to_pub.pose.orientation.x = rotation_q.x();
            pose_to_pub.pose.orientation.y = rotation_q.y();
            pose_to_pub.pose.orientation.z = rotation_q.z();
            pose_to_pub.pose.orientation.w = rotation_q.w();

            m_integration_path.header = pose_to_pub.header;
            m_integration_path.poses.push_back(pose_to_pub);


            cv::Mat T_c0_cj_orb_cv = pKF->GetPoseInverse();
            Eigen::Isometry3d T_c0_cj_orb = Eigen::Isometry3d::Identity();
            cv::cv2eigen(T_c0_cj_orb_cv, T_c0_cj_orb.matrix());
            //		T_c0_cj_orb = T_c_enu.inverse() * T_c0_cj_orb * T_c_enu;
            Eigen::Isometry3d T_w_cj_orb = T_w_c0 * T_c0_cj_orb ;

            pose_to_pub.header.frame_id = "aqua_slam";
// //             //pose_to_pub.header.stamp=ros::Time::now();  // original  // original
// //             pose_to_pub.header.stamp = ros::Time(pKF->mTimeStamp);  // original  // original
            pose_to_pub.pose.position.x = T_w_cj_orb.translation().x();
            pose_to_pub.pose.position.y = T_w_cj_orb.translation().y();
            pose_to_pub.pose.position.z = T_w_cj_orb.translation().z();
            // Eigen::Matrix3d rotation_matrix;
            // rotation_matrix<< R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
            // R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
            // R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);
            rotation_q = Eigen::Quaterniond(T_w_cj_orb.rotation());
            pose_to_pub.pose.orientation.x = rotation_q.x();
            pose_to_pub.pose.orientation.y = rotation_q.y();
            pose_to_pub.pose.orientation.z = rotation_q.z();
            pose_to_pub.pose.orientation.w = rotation_q.w();

            // add a marker, set header and pose to header and pose of pose_to_pub, and add marker to all_markers
            visualization_msgs::msg::Marker marker;
            marker.header = pose_to_pub.header;
            marker.pose = pose_to_pub.pose;
            marker.ns = "aqua_slam";
            marker.id = pKF->mnId;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.02;
            marker.scale.y = 0.02;
            marker.scale.z = 0.02;
            if(pKF->mPoorVision){
                marker.color.r = 1.0f;
                marker.color.g = 0.0f;
                marker.color.b = 1.0f;
            }
            else{
                marker.color.r = 0.0f;
                marker.color.g = 1.0f;
                marker.color.b = 0.0f;
            }
            marker.color.a = 1.0f;
// //             marker.lifetime = ros::Duration(0);  // original  // original
            all_markers.markers.push_back(marker);



            pKF = pKF->mNextKF;

        }

    }

	for(auto pKF:all_kf){
		cv::Mat  T_c0_cj_cv = pKF->GetPoseInverse();
		Eigen::Isometry3d T_c0_cj = Eigen::Isometry3d::Identity();
		cv::cv2eigen(T_c0_cj_cv,T_c0_cj.matrix());
		Eigen::Isometry3d T_w_cj = mT_w_c0 * T_c0_cj;
		geometry_msgs::msg::PoseStamped pose_to_pub;
		pose_to_pub.header.frame_id = "aqua_slam";
		pose_to_pub.header.stamp = rclcpp::Time(static_cast<int64_t>(pKF->mTimeStamp * 1e9));
// // 		pose_to_pub.header.stamp = ros::Time(pKF->mTimeStamp);  // original  // original
		pose_to_pub.pose.position.x = T_w_cj.translation().x();
		pose_to_pub.pose.position.y = T_w_cj.translation().y();
		pose_to_pub.pose.position.z = T_w_cj.translation().z();
		Eigen::Quaterniond rotation_q(T_w_cj.rotation());
		pose_to_pub.pose.orientation.x = rotation_q.x();
		pose_to_pub.pose.orientation.y = rotation_q.y();
		pose_to_pub.pose.orientation.z = rotation_q.z();
		pose_to_pub.pose.orientation.w = rotation_q.w();


		m_path_orb.header = pose_to_pub.header;
		m_path_orb.poses.push_back(pose_to_pub);

		if (!mb_calib_initialized) continue;
		Eigen::Isometry3d T_w_bj = T_w_cj * mT_imu_c.inverse() * mT_body_imu.inverse();
		geometry_msgs::msg::PoseStamped body_pose;
		body_pose.header = pose_to_pub.header;
		Eigen::Quaterniond q_bj(T_w_bj.rotation());
		body_pose.pose.position.x = T_w_bj.translation().x();
		body_pose.pose.position.y = T_w_bj.translation().y();
		body_pose.pose.position.z = T_w_bj.translation().z();
		body_pose.pose.orientation.x = q_bj.x();
		body_pose.pose.orientation.y = q_bj.y();
		body_pose.pose.orientation.z = q_bj.z();
		body_pose.pose.orientation.w = q_bj.w();
		m_path_orb_body.header = body_pose.header;
		m_path_orb_body.poses.push_back(body_pose);

	}
	mp_integration_path_pub->publish(m_integration_path);
	mp_path_orb_pub->publish(m_path_orb);
	mp_path_orb_body_pub->publish(m_path_orb_body);
    mp_ref_integration_path_pub->publish(m_ref_integration_path);
    //publish all_marker
    mp_markers_pub->publish(all_markers);
//	file_integration_trajectory.close();
//	file_orb_trajectory.close();

}

void RosHandling::PublishLossKF(set<KeyFrame*, KFComparator> &loss_kfs)
{
    if (loss_kfs.empty()) {
//         ROS_INFO_STREAM("No loss KF");  // original
        return;
    }
    visualization_msgs::msg::MarkerArray all_markers;
    Eigen::Isometry3d T_imu_c = Eigen::Isometry3d::Identity();
    cv::cv2eigen((*loss_kfs.begin())->mImuCalib.mT_imu_c, T_imu_c.matrix());
    Eigen::Isometry3d T_body_imu_lkf = Eigen::Isometry3d::Identity();
    {
        cv::Mat T_body_imu_cv = (*loss_kfs.begin())->mImuCalib.mT_body_imu.clone();
        if (!T_body_imu_cv.empty())
            cv::cv2eigen(T_body_imu_cv, T_body_imu_lkf.matrix());
    }
    Eigen::Isometry3d T_w_c0 = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d R_b0_w = (*loss_kfs.begin())->GetMap()->getRGravity();
    // ROS_DEBUG_STREAM("R_b0_w: " << R_b0_w);  // original
    RCLCPP_DEBUG_STREAM(mp_node->get_logger(), "R_b0_w: " << R_b0_w);
    Eigen::Matrix3d R_w_c0 = R_b0_w.inverse() * T_imu_c.rotation();
    T_w_c0.rotate(R_w_c0);
    {
        Eigen::Matrix3d R_world_body0 = R_b0_w.inverse() * T_body_imu_lkf.inverse().rotation();
        Eigen::Vector3d fwd_horiz = R_world_body0.col(0);
        fwd_horiz.z() = 0.0;
        if (fwd_horiz.norm() > 1e-6) {
            fwd_horiz.normalize();
            double yaw_err = atan2(fwd_horiz.y(), fwd_horiz.x());
            T_w_c0.prerotate(Eigen::AngleAxisd(-yaw_err, Eigen::Vector3d::UnitZ()));
        }
    }

    for (auto pKF: loss_kfs) {
        cv::Mat T_c0_cj_orb_cv = pKF->GetPoseInverse();
        Eigen::Isometry3d T_c0_cj_orb = Eigen::Isometry3d::Identity();
        cv::cv2eigen(T_c0_cj_orb_cv, T_c0_cj_orb.matrix());
        //		T_c0_cj_orb = T_c_enu.inverse() * T_c0_cj_orb * T_c_enu;
        Eigen::Isometry3d T_w_cj_orb = T_w_c0 * T_c0_cj_orb;

        geometry_msgs::msg::PoseStamped pose_to_pub;
        pose_to_pub.header.frame_id = "aqua_slam";
// //         pose_to_pub.header.stamp = ros::Time(pKF->mTimeStamp);  // original  // original
        pose_to_pub.pose.position.x = T_w_cj_orb.translation().x();
        pose_to_pub.pose.position.y = T_w_cj_orb.translation().y();
        pose_to_pub.pose.position.z = T_w_cj_orb.translation().z();
        Eigen::Quaterniond rotation_q(T_w_cj_orb.rotation());
        pose_to_pub.pose.orientation.x = rotation_q.x();
        pose_to_pub.pose.orientation.y = rotation_q.y();
        pose_to_pub.pose.orientation.z = rotation_q.z();
        pose_to_pub.pose.orientation.w = rotation_q.w();

        // add a marker, set header and pose to header and pose of pose_to_pub, and add marker to all_markers
        visualization_msgs::msg::Marker marker;
        marker.header = pose_to_pub.header;
        marker.pose = pose_to_pub.pose;
        marker.ns = "aqua_slam";
        marker.id = pKF->mnId;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.02;
        marker.scale.y = 0.02;
        marker.scale.z = 0.02;
        marker.color.r = 1.0f;
        marker.color.g = 0.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0f;
// //         marker.lifetime = ros::Duration(0);  // original  // original
        all_markers.markers.push_back(marker);
    }
    mp_markers_pub->publish(all_markers);
}

void RosHandling::PublishLossInteration(const Eigen::Isometry3d &T_e0_er, const Eigen::Isometry3d &T_e0_ec)
{
	geometry_msgs::msg::PoseStamped pose_to_pub;
	pose_to_pub.header.frame_id = "aqua_slam";
// // 	pose_to_pub.header.stamp = ros::Time::now();  // original  // original
//	pose_to_pub.header.stamp = stamp;
	pose_to_pub.pose.position.x = T_e0_er.translation().x();
	pose_to_pub.pose.position.y = T_e0_er.translation().y();
	pose_to_pub.pose.position.z = T_e0_er.translation().z();
	// Eigen::Matrix3d rotation_matrix;
	// rotation_matrix<< R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
	// R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
	// R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);
	Eigen::Quaterniond rotation_q(T_e0_er.rotation());
	pose_to_pub.pose.orientation.x = rotation_q.x();
	pose_to_pub.pose.orientation.y = rotation_q.y();
	pose_to_pub.pose.orientation.z = rotation_q.z();
	pose_to_pub.pose.orientation.w = rotation_q.w();

	mp_pose_integration_ref_pub->publish(pose_to_pub);

	pose_to_pub.pose.position.x = T_e0_ec.translation().x();
	pose_to_pub.pose.position.y = T_e0_ec.translation().y();
	pose_to_pub.pose.position.z = T_e0_ec.translation().z();
	// Eigen::Matrix3d rotation_matrix;
	// rotation_matrix<< R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
	// R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
	// R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);
	rotation_q = Eigen::Quaterniond(T_e0_ec.rotation());
	pose_to_pub.pose.orientation.x = rotation_q.x();
	pose_to_pub.pose.orientation.y = rotation_q.y();
	pose_to_pub.pose.orientation.z = rotation_q.z();
	pose_to_pub.pose.orientation.w = rotation_q.w();

	mp_pose_integration_cur_pub->publish(pose_to_pub);
}
// // void RosHandling::PublishCamera(const Eigen::Isometry3d &T_c0_cj_orb, const ros::Time &stamp)  // original  // original
#if 0  // original: body of commented-out PublishCamera
{
	nav_msgs::Odometry pose_to_pub;
	pose_to_pub.header.frame_id = "aqua_slam";
	//pose_to_pub.header.stamp=ros::Time::now();
	pose_to_pub.header.stamp = stamp;

	pose_to_pub.pose.pose.position.x = T_c0_cj_orb.translation().x();
	pose_to_pub.pose.pose.position.y = T_c0_cj_orb.translation().y();
	pose_to_pub.pose.pose.position.z = T_c0_cj_orb.translation().z();
	Eigen::Quaterniond rotation_q(T_c0_cj_orb.rotation());
	pose_to_pub.pose.pose.orientation.x = rotation_q.x();
	pose_to_pub.pose.pose.orientation.y = rotation_q.y();
	pose_to_pub.pose.pose.orientation.z = rotation_q.z();
	pose_to_pub.pose.pose.orientation.w = rotation_q.w();

	mp_pose_orb_camera_pub->publish(pose_to_pub);
}
#endif
// bool RosHandling::SavePose(std::shared_ptr<std_srvs::srv::Empty::Request> &req, std::shared_ptr<std_srvs::srv::Empty::Response> &res)  // original
void RosHandling::SavePose(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res)
{
	string out_path;
// // 	ros::param::get("/aqua_slam/traj_path", out_path);  // original  // original
	// mp_system->SaveKeyFrameTrajectoryTUM(out_path + "KeyFrameTrajectory_TUM_Format");
    mp_system->SaveKeyFrameTrajectory(out_path);
	// mp_system->mpDenseMapper->Save(out_path);
	// mp_system->SaveAtlas(out_path, System::TEXT_FILE);
}
// bool RosHandling::LoadMap(std::shared_ptr<std_srvs::srv::Empty::Request> &req, std::shared_ptr<std_srvs::srv::Empty::Response> &res)  // original
void RosHandling::LoadMap(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res)
{
	string map_file;
// // 	ros::param::get("/aqua_slam/map_file", map_file);  // original  // original
	mp_system->LoadAtlas(map_file,System::TEXT_FILE);
}

void RosHandling::PublishImgMergeCandidate(const cv::Mat &img)
{
	if(img.empty()){
		return;
	}
	cv::Mat img_to_pub;
	img.copyTo(img_to_pub);
	std_msgs::msg::Header header; // empty header
// // 	header.stamp = ros::Time::now(); // time  // original  // original
//	cv::Mat img_with_info=mpFrameDrawer->DrawFrame(true);

	if (img_to_pub.channels() < 3) {
		// cv::cvtColor(img_to_pub, img_to_pub, CV_GRAY2BGR);  // original
        cv::cvtColor(img_to_pub, img_to_pub, cv::COLOR_GRAY2BGR); 
	}
	cv_bridge::CvImage img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, img_to_pub);

	mp_img_merge_cond_pub->publish(img_bridge.toImageMsg());
}
// bool RosHandling::CalibrateDVLGyro(std::shared_ptr<std_srvs::srv::Empty::Request> &req, std::shared_ptr<std_srvs::srv::Empty::Response> &res)  // original
void RosHandling::CalibrateDVLGyro(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res)
{
	mp_LocalMapping->InitializeDvlIMU();
}

// bool RosHandling::FullBA(std::shared_ptr<std_srvs::srv::Empty::Request> &req, std::shared_ptr<std_srvs::srv::Empty::Response> &res)  // original
void RosHandling::FullBA(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res)
{
    mp_LocalMapping->FullBA();
}

void RosHandling::Run(Atlas* pAtlas)
{
    while(1){
        UpdateMap(pAtlas);
        PublishIntegration(pAtlas);
        //sleep for 0.25 second
        usleep(250000);
    }

}
