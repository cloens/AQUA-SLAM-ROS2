//
// Created by da on 03/12/2021.
//
#include "DenseMapper.h"
#include "KeyFrame.h"
#include "Map.h"

// #include <sensor_msgs/PointCloud2.h>  // original
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
// #include <sensor_msgs/Image.h>
// #include <sensor_msgs/CompressedImage.h>  // original
#include <sensor_msgs/msg/compressed_image.hpp>
// #include <cv_bridge/cv_bridge.h>  // original
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <iterator>
#include <map>

#include <thread>
#include <chrono>

namespace ORB_SLAM3
{
    // DenseMapper::DenseMapper(string settingFile)  // original
    DenseMapper::DenseMapper(string settingFile, rclcpp::Node::SharedPtr node)
    {
        mNode = node;
// //         ros::NodeHandle nh_;  // original
        image_transport::ImageTransport it(mNode);
        image_transport::Publisher depth_pub = it.advertise("/aqua_slam/dense_mapper/depth", 10);
        mDepthPub = std::make_shared<image_transport::Publisher>(depth_pub);
        image_transport::Publisher depth_conf_pub = it.advertise(
                "/aqua_slam/dense_mapper/depth_confidence", 10);
        mDepthConfPub = std::make_shared<image_transport::Publisher>(depth_conf_pub);
// //         ros::Publisher pointcloud_pub = nh_.advertise<sensor_msgs::msg::PointCloud2>("/aqua_slam/dense_map", 10);  // original  // original
// //         mMapPub = std::shared_ptr<ros::Publisher>(boost::make_shared<ros::Publisher>(pointcloud_pub));  // original  // original
        mMapPub = mNode->create_publisher<sensor_msgs::msg::PointCloud2>("/aqua_slam/dense_map", 10);
        mMapAliasPub = mNode->create_publisher<sensor_msgs::msg::PointCloud2>("/aqua_slam/map", 10);

        FileStorage fs(settingFile, FileStorage::READ);
        FileNode fsNode = fs["DenseMapper"];
        float P1 = (float) fsNode["P1"];
        float P2 = (float) fsNode["P2"];
        int correlation_window_size = (int) fsNode["correlation_window_size"];
        int disp12MaxDiff = (int) fsNode["disp12MaxDiff"];
        int disparity_range = (int) fsNode["disparity_range"];
        int min_disparity = (int) fsNode["min_disparity"];
        int prefilter_cap = (int) fsNode["prefilter_cap"];
        int prefilter_size = (int) fsNode["prefilter_size"];
        int speckle_range = (int) fsNode["speckle_range"];
        int speckle_size = (int) fsNode["speckle_size"];
        int texture_threshold = (int) fsNode["texture_threshold"];
        float uniqueness_ratio = (float) fsNode["uniqueness_ratio"];
        mLeafSize = (float) fsNode["leaf_size"];
        mMeanK = (int) fsNode["mean_k"];
        mStdThred = (float) fsNode["std_thred"];
        mEnable = !((float) fsNode["enable"] == 0);

        mParam = DepthEstParamters(P1, P2, correlation_window_size, disp12MaxDiff, disparity_range,
                                   min_disparity, prefilter_cap, prefilter_size, speckle_range, speckle_size,
                                   texture_threshold, uniqueness_ratio);

        double image_scale = (double) fs["ImageScale"];
        fx = (double) fs["Camera.fx"] * image_scale;
        fy = (double) fs["Camera.fy"] * image_scale;
        cx = (double) fs["Camera.cx"] * image_scale;
        cy = (double) fs["Camera.cy"] * image_scale;
        bf = (double) fs["Camera.bf"] * image_scale;


    }

    void DenseMapper::InsertNewKF(KeyFrame* pKF)
    {
        if (pKF->GetMap()->GetAllKeyFrames().size() < 5) {
            // ROS_DEBUG_STREAM("DenserMapper: less than 5 KF in current map, skip");  // original
            RCLCPP_DEBUG(mNode->get_logger(), "DenserMapper: less than 5 KF in current map, skip");
            return;
        }
        else if (pKF->GetMap()->GetAllKeyFrames().size() == 5) {
            auto all_KF = pKF->GetMap()->GetAllKeyFrames();
            for (auto pKFi: all_KF) {
                std::lock_guard<std::mutex> lock(mKFQueueMutex);
                mKFQueue.push(pKFi);
            }
        }
        else {
            std::lock_guard<std::mutex> lock(mKFQueueMutex);
            mKFQueue.push(pKF);
        }
    }

    void DenseMapper::ComputeDisp(const Mat &left, const Mat &right, Mat &out, Mat &out_conf,
                                  DepthEstParamters &param)
    {
        using namespace ximgproc;

        String algo = "bm";
        String filter = "wls_conf";
        bool no_downscale = true;

        double lambda = 8000;
        double sigma = 1.5;

        Mat left_for_matcher, right_for_matcher;
        Mat left_disp, right_disp;
        Mat filtered_disp, solved_disp, solved_filtered_disp;
        Mat conf_map = Mat(left.rows, left.cols, CV_8U);
        conf_map = Scalar(255);
        Rect ROI;
        Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
        double matching_time, filtering_time;
        double solving_time = 0;
        //	if (param._disp12MaxDiff <= 0 || param._disp12MaxDiff % 16 != 0) {
        //		cout << "Incorrect max_disparity value: it should be positive and divisible by 16";
        //		return;
        //	}
        if (param._correlation_window_size <= 0 || param._correlation_window_size % 2 != 1) {
//             ROS_ERROR_STREAM("DenseMapper: Incorrect window_size value: it should be positive and odd");  // original
            return;
        }

        if (filter == "wls_conf") // filtering with confidence (significantly better quality than wls_no_conf)
        {
            if (!no_downscale) {
                // downscale the views to speed-up the matching stage, as we will need to compute both left
                // and right disparity maps for confidence map computation
                //! [downscale]
                //			param._disp12MaxDiff /= 2;
                //			if (param._disp12MaxDiff % 16 != 0) {
                //				param._disp12MaxDiff += 16 - (param._disp12MaxDiff % 16);
                //			}
                // resize(left ,left_for_matcher ,Size(),0.5,0.5, INTER_LINEAR_EXACT);
                // resize(right,right_for_matcher,Size(),0.5,0.5, INTER_LINEAR_EXACT);
                resize(left, left_for_matcher, Size(), 0.5, 0.5, INTER_LINEAR);
                resize(right, right_for_matcher, Size(), 0.5, 0.5, INTER_LINEAR);
                //! [downscale]
            }
            else {
                left_for_matcher = left.clone();
                right_for_matcher = right.clone();
            }

            if (algo == "bm") {
                //! [matching]
                Ptr<StereoBM> left_matcher = StereoBM::create();
                left_matcher->setNumDisparities(param._disparity_range);
                left_matcher->setBlockSize(param._correlation_window_size);
                left_matcher->setDisp12MaxDiff(param._disp12MaxDiff);
                left_matcher->setMinDisparity(param._min_disparity);
                left_matcher->setPreFilterCap(param._prefilter_cap);
                left_matcher->setPreFilterSize(param._prefilter_size);
                left_matcher->setSpeckleRange(param._speckle_range);
                left_matcher->setSpeckleWindowSize(param._speckle_size);
                left_matcher->setTextureThreshold(param._texture_threshold);
                left_matcher->setUniquenessRatio(param._uniqueness_ratio);
                wls_filter = createDisparityWLSFilter(left_matcher);
                Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

                if (left_for_matcher.channels() == 3 || left_for_matcher.channels() == 4) {
                    cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
                    cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
                }


                matching_time = (double) getTickCount();
                left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
                right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
                matching_time = ((double) getTickCount() - matching_time) / getTickFrequency();
                //! [matching]
            }
            else if (algo == "sgbm") {
                Ptr<StereoSGBM> left_matcher = StereoSGBM::create(param._min_disparity, param._disparity_range,
                                                                  param._correlation_window_size);
                left_matcher->setP1(param._P1);
                left_matcher->setP2(param._P2);
                left_matcher->setPreFilterCap(param._prefilter_cap);
                left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
                wls_filter = createDisparityWLSFilter(left_matcher);
                Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

                matching_time = (double) getTickCount();
                left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
                right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
                matching_time = ((double) getTickCount() - matching_time) / getTickFrequency();
            }
            else {
                cout << "Unsupported algorithm";
                return;
            }

            //! [filtering]
            wls_filter->setLambda(lambda);
            wls_filter->setSigmaColor(sigma);
            filtering_time = (double) getTickCount();
            wls_filter->filter(left_disp, left, filtered_disp, right_disp);
            filtering_time = ((double) getTickCount() - filtering_time) / getTickFrequency();
            //! [filtering]
            conf_map = wls_filter->getConfidenceMap();

            // Get the ROI that was used in the last filter call:
            ROI = wls_filter->getROI();
            if (!no_downscale) {
                // upscale raw disparity and ROI back for a proper comparison:
                resize(left_disp, left_disp, Size(), 2.0, 2.0, INTER_LINEAR);
                left_disp = left_disp * 2.0;
                ROI = Rect(ROI.x * 2, ROI.y * 2, ROI.width * 2, ROI.height * 2);
            }
        }
        Mat filtered_disp_vis;
        getDisparityVis(left_disp, filtered_disp_vis, 1.0);
        //	out = filtered_disp;
        out = filtered_disp_vis.clone();
        out_conf = conf_map.clone();

    }

    template<typename BasicType>
    std::vector<BasicType> IQR_thres(std::vector<BasicType> &vec, float scale = 0.5)
    {
        std::sort(vec.begin(), vec.end());
        int n = 0;
        if ((vec.size() % 2) == 0) {
            n = static_cast<int>(vec.size() / 2);
        }
        else {
            n = static_cast<int>((vec.size() - 1) / 2);
        }
        BasicType Q1, Q3;
        if ((n % 2) != 0) {
            Q1 = vec[n / 2];
            Q3 = vec[(vec.size() * 2 - n) / 2];
        }
        else {
            Q1 = (vec[n / 2 - 1] + vec[n / 2]) / 2;
            Q3 = (vec[(vec.size() - n + vec.size() - 1) / 2] + vec[(vec.size() - n + vec.size() - 1) / 2 + 1]) /
                 2;
        }
        BasicType IQR = Q3 - Q1;
        std::vector<BasicType> thres;
        thres.push_back(Q1 - IQR * scale);
        thres.push_back(Q3 + IQR * scale);
        return thres; // 1.5 can be changed
    }

    void DenseMapper::GetSubMap(const Mat &img_l, const Mat &img_r, pcl::PointCloud<pcl::PointXYZRGB> &sub_map)
    {
        cv::Mat disp, disp_conf;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sub_map_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sub_map_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

        cv::Mat img_l_rgb, img_r_rgb;
        if (img_l.empty() || img_r.empty()) {
            // ROS_WARN_STREAM("DenserMapper: found empty images!" << "timestamp: " << fixed << setprecision(12) << time);  // original
            RCLCPP_WARN(mNode->get_logger(), "DenserMapper: found empty images!");
            return;
        }
        img_l_rgb = img_l.clone();
        img_r_rgb = img_r.clone();
        ComputeDisp(img_l, img_r, disp, disp_conf, mParam);
        disp_conf.convertTo(disp_conf, CV_8UC1);

        sensor_msgs::msg::Image::_header_type header;
// //         header.stamp = ros::Time::now();  // original  // original

        cv_bridge::CvImage img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, disp);
        mDepthPub->publish(img_bridge.toImageMsg());
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, disp_conf);
        mDepthConfPub->publish(img_bridge.toImageMsg());

        cv::Mat disp_f;
        disp.convertTo(disp_f,CV_32F);
        std::vector<float> vec;
        if (disp_f.isContinuous()) {
            vec.assign((float*) disp_f.datastart, (float*) disp_f.dataend); // Faster
        }
        else {
            for (int i = 0; i < disp_f.rows; ++i) {
                vec.insert(vec.end(), disp_f.ptr<float>(i), disp_f.ptr<float>(i) + disp_f.cols);
            }
        }
        auto thred = IQR_thres(vec,0.5);
        std::cout<<"upper: "<<(int)thred[1]<<" lower: "<<(int)thred[0]<<std::endl;

	for (int x = 0; x < img_l_rgb.cols; x++) {
		for (int y = 0; y < img_l_rgb.rows; y++) {
			float d = static_cast<float >(disp.at<unsigned char>(y, x));
			float d_f = disp_conf.at<float>(y, x);
			unsigned char b = static_cast<int>(img_l_rgb.at<Vec3b>(y, x)[0]);
			unsigned char g = static_cast<int>(img_l_rgb.at<Vec3b>(y, x)[1]);
			unsigned char r = static_cast<int>(img_l_rgb.at<Vec3b>(y, x)[2]);
//				cout << "disp confidence: " << d_f << endl;


            if ((float(d) < thred[0]) || (float(d) > thred[1])) {
			// if (d_f <= 200 || (bf / d) > 10) {
				continue;
			}
			Eigen::Vector3d p_3d(0, 0, 0);
			Project2Dto3D(x, y, d, p_3d);
//				p_3d = T_c0_cj * p_3d;
//				p_3d = T_c0_cj * p_3d;
//				cout << "depth: " << (bf / d) << endl;
			pcl::PointXYZRGB p(r, g, b);
			p.x = p_3d.x();
			p.y = p_3d.y();
			p.z = p_3d.z();

                local_map->push_back(p);
            }
        }

        if(local_map->empty()){
            return;
        }
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sorf;
        sorf.setInputCloud(local_map);
        sorf.setMeanK(mMeanK);
        sorf.setStddevMulThresh(mStdThred);
        sorf.filter(*local_map);

        if(local_map->empty()){
            return;
        }

        // pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
        // // build the filter
        // outrem.setInputCloud(local_map);
        // outrem.setRadiusSearch(0.1);
        // outrem.setMinNeighborsInRadius(10);
        // outrem.setKeepOrganized(true);
        // outrem.filter(*local_map);
        //
        // if(local_map->empty()){
        //     return;
        // }

        pcl::VoxelGrid<pcl::PointXYZRGB> vgf;
        vgf.setInputCloud(local_map);
        vgf.setLeafSize(mLeafSize, mLeafSize, mLeafSize);
        vgf.filter(sub_map);







        //	sub_map = *local_map;
    }

    void DenseMapper::GetSubMapFromExternalDepth(const Mat &img_l, const Mat &depth_m,
                                                 pcl::PointCloud<pcl::PointXYZRGB> &sub_map)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (img_l.empty() || depth_m.empty()) {
            RCLCPP_WARN(mNode->get_logger(), "DenseMapper: empty image/depth for external depth path");
            return;
        }
        cv::Mat img_l_rgb = img_l.clone();
        cv::Mat depth;
        if (depth_m.type() == CV_32FC1)
            depth = depth_m;
        else
            depth_m.convertTo(depth, CV_32FC1);

        if (depth.rows != img_l_rgb.rows || depth.cols != img_l_rgb.cols) {
            cv::resize(depth, depth, img_l_rgb.size(), 0, 0, cv::INTER_NEAREST);
        }

        for (int y = 0; y < img_l_rgb.rows; y += 2) {
            for (int x = 0; x < img_l_rgb.cols; x += 2) {
                float z = depth.at<float>(y, x);
                if (!(z > 0.05f) || z > 20.0f)
                    continue;
                unsigned char b, g, r;
                if (img_l_rgb.channels() == 3) {
                    b = img_l_rgb.at<cv::Vec3b>(y, x)[0];
                    g = img_l_rgb.at<cv::Vec3b>(y, x)[1];
                    r = img_l_rgb.at<cv::Vec3b>(y, x)[2];
                } else {
                    b = g = r = img_l_rgb.at<unsigned char>(y, x);
                }
                Eigen::Vector3d p_3d(0, 0, 0);
                ProjectDepthTo3D(x, y, z, p_3d);
                pcl::PointXYZRGB p(r, g, b);
                p.x = p_3d.x();
                p.y = p_3d.y();
                p.z = p_3d.z();
                local_map->push_back(p);
            }
        }
        if (local_map->empty())
            return;
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sorf;
        sorf.setInputCloud(local_map);
        sorf.setMeanK(mMeanK);
        sorf.setStddevMulThresh(mStdThred);
        sorf.filter(*local_map);
        if (local_map->empty())
            return;
        pcl::VoxelGrid<pcl::PointXYZRGB> vgf;
        vgf.setInputCloud(local_map);
        vgf.setLeafSize(mLeafSize, mLeafSize, mLeafSize);
        vgf.filter(sub_map);
    }

void DenseMapper::MergeSubMap(pcl::PointCloud<pcl::PointXYZRGB> &sub_map, KeyFrame* pKF)
    {
        pcl::PointCloud<pcl::PointXYZRGB> transformed_map;

        cv::Mat T_cj_c0_cv = pKF->GetPose().clone();
        Eigen::Isometry3d T_cj_c0, T_c0_cj, T_b_c, T_b0_cj;
        cv::cv2eigen(T_cj_c0_cv, T_cj_c0.matrix());
        T_c0_cj = T_cj_c0.inverse();
        cv::Mat T_b_c_cv = pKF->mImuCalib.mT_dvl_c.clone();
        cv::cv2eigen(T_b_c_cv, T_b_c.matrix());

        T_b0_cj = T_b_c * T_c0_cj;

        pcl::transformPointCloud(sub_map, transformed_map, T_b0_cj.matrix());

        std::lock_guard<std::mutex> lock(mDenseMapMutex);
        mGlobalMap += transformed_map;

    }

    void DenseMapper::Run()
    {
        while (1) {
            {
                if (!mEnable) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
                    continue;
                }
                std::lock_guard<std::mutex> lock(mKFQueueMutex);
                if (!mKFQueue.empty()) {
                    auto pKF = mKFQueue.front();
                    mKFQueue.pop();
                    if (pKF->isBad()) {
                        continue;
                    }
                    if (pKF->GetPose().empty()) {
                        continue;
                    }
                    if (mKFWithPointCloud.find(pKF) == mKFWithPointCloud.end()) {
                        std::lock_guard<std::mutex> lock(mKFMutex);
                        mKFWithPointCloud[pKF] = pcl::PointCloud<pcl::PointXYZRGB>();
                        if (!pKF->imgDepthScaled.empty()) {
                            GetSubMapFromExternalDepth(pKF->imgLeft, pKF->imgDepthScaled, mKFWithPointCloud[pKF]);
                        } else {
                            GetSubMap(pKF->imgLeft, pKF->imgRight, mKFWithPointCloud[pKF]);
                        }
                        MergeSubMap(mKFWithPointCloud[pKF], pKF);
                    }
                }
            }


            if (mStop) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
    }

    void DenseMapper::Stop()
    {
        mStop = true;
    }

    void DenseMapper::Resume()
    {
        mStop = false;
    }

    void DenseMapper::Update()
    {
        mGlobalMap.clear();
        {
            std::lock_guard<std::mutex> lock(mKFMutex);
            for (auto kf_pointcloud: mKFWithPointCloud) {
                if (kf_pointcloud.first->isBad()) {
                    // ROS_DEBUG_STREAM("DenserMapper: find bad KF, remove and skip");  // original
                    RCLCPP_DEBUG(mNode->get_logger(), "DenserMapper: find bad KF, remove and skip");
                    mKFWithPointCloud.erase(kf_pointcloud.first);
                    return;
                }
                MergeSubMap(kf_pointcloud.second, kf_pointcloud.first);
            }
        }
        PublishMap();
    }

    void DenseMapper::PublishMap()
    {
        sensor_msgs::msg::PointCloud2 dense_map;

        std::lock_guard<std::mutex> lock(mDenseMapMutex);
        pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud;
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(mGlobalMap));
        sor.setLeafSize(0.05f, 0.05f, 0.05f);
        //	sor.filter(filtered_cloud);
        pcl::toROSMsg(mGlobalMap, dense_map);
        dense_map.header.frame_id = "aqua_slam";
        mMapPub->publish(dense_map);
        if (mMapAliasPub) mMapAliasPub->publish(dense_map);
        //	mMapPub->publish(mGlobalMap);
    }

    void DenseMapper::Save(string path)
    {
        if (!mEnable) {
            return;
        }
        std::lock_guard<std::mutex> lock(mDenseMapMutex);
        string file_name = path + "KF_map.pcd";
        pcl::io::savePCDFileBinary(file_name, mGlobalMap);
    }

    void DenseMapper::ReconstructFromBagPose(string bag_path, string pose_path)
    {
        // pcl::PointCloud<pcl::PointXYZRGB> global_map;
        // std::map<double, Eigen::Isometry3d> estimation_traj;
        // std::ifstream estimation_file(pose_path);
        // if (!estimation_file.is_open()) {
        //     std::cout << "Failed to open file." << std::endl;
        //     return;
        // }
        // std::string line;
        // while (std::getline(estimation_file, line)) {
        //     std::istringstream iss(line);
        //     if (line[0] == '#') {
        //         continue;
        //     }
        //     double timestamp, tx, ty, tz, qx, qy, qz, qw;
        //     if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
        //         std::cout << "Failed to parse line: " << line << std::endl;
        //         continue;
        //     }
        //
        //     Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        //     Eigen::Matrix3d R = quaternion.normalized().toRotationMatrix();
        //     Eigen::Vector3d t(tx, ty, tz);
        //     Eigen::Isometry3d T;
        //     T.setIdentity();
        //     T.pretranslate(t);
        //     T.rotate(R);
        //     estimation_traj.insert(std::make_pair(timestamp, T));
        //
        // }
        //
        // estimation_file.close();
        //
        // // Open the bag file
        // rosbag::Bag bag;
        // bag.open(bag_path, rosbag::bagmode::Read);
        //
        // // Specify the topics to read
        // std::vector<std::string> topics;
        // topics.push_back(std::string("/camera/left/image_dehazed/compressed"));  // replace with your topic
        // topics.push_back(std::string("/camera/right/image_dehazed/compressed"));  // replace with your topic
        //
        // // Create a view for the data in the bag file
        // rosbag::View view(bag, rosbag::TopicQuery(topics));
        //
        // // Create multimaps to store the messages for each topic, keyed by their timestamps
// //         // std::multimap<ros::Time, sensor_msgs::CompressedImage::::ConstSharedPtr> left_topic_images;  // original  // original
// //         // std::multimap<ros::Time, sensor_msgs::CompressedImage::::ConstSharedPtr> right_topic_images;  // original  // original
        //
        // // Iterate over the messages in the topics
        // for (const rosbag::MessageInstance &m: view) {
        //     sensor_msgs::CompressedImage::::ConstSharedPtr img_msg = m.instantiate<sensor_msgs::CompressedImage>();
        //     if (img_msg != NULL) {
        //         if (m.getTopic() == "/camera/left/image_dehazed/compressed") {
        //             left_topic_images.insert(std::make_pair(img_msg->header.stamp, img_msg));
        //         }
        //         else if (m.getTopic() == "/camera/right/image_dehazed/compressed") {
        //             right_topic_images.insert(std::make_pair(img_msg->header.stamp, img_msg));
        //         }
        //     }
        // }
        //
        // // Iterate over the left images and find the corresponding right images based on the closest timestamp
        // for (const auto &left_pair: left_topic_images) {
// //         //     ros::Time left_time = left_pair.first;  // original  // original
        //     sensor_msgs::CompressedImage::::ConstSharedPtr left_img_msg = left_pair.second;
        //
        //     auto lower = right_topic_images.lower_bound(left_time);
        //     auto upper = right_topic_images.upper_bound(left_time);
        //
        //     if (lower == right_topic_images.begin() && upper == right_topic_images.end()) {
        //         continue; // No corresponding right image found
        //     }
        //
        //     // Find the right image with the closest timestamp
        //     auto closest_right = std::min_element(lower, upper, [left_time](const auto &a, const auto &b)
        //     {
        //         return std::abs((a.first - left_time).toSec()) < std::abs((b.first - left_time).toSec());
        //     });
        //
        //     cv_bridge::CvImagePtr cv_ptr_l;
        //     try {
        //         cv_ptr_l = cv_bridge::toCvCopy(left_pair.second, sensor_msgs::msg::Image_encodings::BGR8);
        //     } catch (cv_bridge::Exception &e) {
        //         ROS_ERROR("cv_bridge exception: %s", e.what());
        //         return;
        //     }
        //     cv_bridge::CvImagePtr cv_ptr_r;
        //     try {
        //         cv_ptr_r = cv_bridge::toCvCopy(closest_right->second, sensor_msgs::msg::Image_encodings::BGR8);
        //     } catch (cv_bridge::Exception &e) {
        //         ROS_ERROR("cv_bridge exception: %s", e.what());
        //         return;
        //     }
        //
        //     if (estimation_traj.count(left_pair.first.toSec()) > 0) {
        //
        //         pcl::PointCloud<pcl::PointXYZRGB> sub_map;
        //         GetSubMap(cv_ptr_l->image, cv_ptr_r->image, sub_map);
        //         cv::imshow("right", cv_ptr_r->image);
        //         cv::imshow("left", cv_ptr_l->image);
        //         cv::waitKey(1);
        //
        //         Eigen::Isometry3d T_c0_cj = estimation_traj[left_pair.first.toSec()];
        //         pcl::transformPointCloud(sub_map, sub_map, T_c0_cj.matrix().cast<float>());
        //         global_map += sub_map;
        //     }
        //
        //     //save global map to a local file
        //
        //
        //
        //     // Here you can process the corresponding images
        // }
        //
        // bag.close();
        // pcl::io::savePCDFileBinary(
        //         "/home/da/project/ros/orb_dvl2_ws/src/dvl2/dvl2_results/global_map.pcd", global_map);
    }

}
