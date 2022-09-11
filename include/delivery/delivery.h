//
// Created by sumin on 22. 9. 11.
//

#ifndef SRC_DELIVERY_H
#define SRC_DELIVERY_H

#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <darknet_ros_msgs/ObjectCount.h>
#include <darknet_ros_msgs/BoundingBoxes.h>

#include <data_transfer_msg/delivery_state.h>
#include <data_transfer_msg/delivery_mission.h>

#include <sensor_msgs/Image.h>
#include <std_msgs/Int16.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>

#include <iostream>
#include <string>
#include <vector>
#include <string>
#include <algorithm>

struct id_bbox{
    int id;
    int center;
    double area;
};

class Delivery{
private:
    // bbox
    void sign_CB(const darknet_ros_msgs::BoundingBoxes::ConstPtr& msg);
    inline int return_center(int x1, int x2){return int((x1+x2)/2);}
    inline double return_area(cv::Point p1, cv::Point p2);
    static bool compare(id_bbox a , id_bbox b);
    std::string return_ID(int id);
    void delivery(id_bbox id_box);
    int mode(Eigen::MatrixXd data, int size);
    std::string to_go(std::string value);

    //stop line
    void cam_CB(const sensor_msgs::Image::ConstPtr& msg);
    cv::Mat return_BEV(cv::Mat frame);
    void show(std::string win_name, cv::Mat frame, int waitkey);
    double medianMat(cv::Mat frame);
    void calc_thresh(int &low, int &high, double sigma, double mid);
    cv::Mat extract_l_frame(cv::Mat HLS);
    void erode_process(cv::Mat &frame);
    inline double calc_theta(cv::Point pt1, cv::Point pt2);
    void detect_change(bool state);

    ros::NodeHandle nh;
    ros::NodeHandle pnh;

    ros::Subscriber subCAM;
    ros::Subscriber subBBOX;

    ros::Publisher pubState;
    ros::Publisher pubMission;

    //delivery
    data_transfer_msg::delivery_state dl_st;
    data_transfer_msg::delivery_mission dl_ms;

    darknet_ros_msgs::BoundingBoxes BBOX;
    std::vector<id_bbox> bbox_vec;
    Eigen::MatrixXd dl;
    int delivery_cnt;
    std::string delivery_state;
    std::string start;
    std::string destination;


    bool delivery_init;
    bool during_pickup;
    bool pick_up_in_flag;
    bool during_delivery;
    bool delivery_in_flag;



    //stop line
    bool stop_line_process;
    int low, high;
    cv::Mat cameraMatrix, distCoeffs;
    cv::Mat map1, map2;
    cv::Mat frame;
    cv::Mat bev_frame;
    cv::Mat l_frame;
    cv::Mat bin_frame;
    std::vector<cv::Point> points;
    bool stop_line_state;
    bool prev_state;
    int change_cnt;
    bool stop_state;

public:
    Delivery();
};
#endif //SRC_DELIVERY_H
