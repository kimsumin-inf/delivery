//
// Created by sumin on 22. 9. 11.
//

#include "delivery/delivery.h"

using namespace std;
using namespace cv;

static inline void updateSample(Eigen::MatrixXd &sample, Eigen::MatrixXd &data);

Delivery::Delivery()
:nh(""), pnh("")
{
    subCAM = nh.subscribe("/camera_front/image_raw", 1, &Delivery::cam_CB, this);
    subBBOX = nh.subscribe("/darknet_ros/bounding_boxes", 1, &Delivery::sign_CB,this);
    pubState = nh.advertise<data_transfer_msg::delivery_state>("/delivery/state", 1);
    pubMission = nh.advertise<data_transfer_msg::delivery_mission>("/delivery/mission", 1);


    delivery_init = true;
    during_pickup = false;
    pick_up_in_flag =false;
    delivery_in_flag = false;
    during_delivery = false;
    dl = Eigen::MatrixXd::Zero(10,1);
    delivery_cnt = 0;
    dl_st.Pick_up ="";
    dl_st.Now_state ="";
    dl_st.Delivery ="";
    dl_ms.Stop_flag=false;
    dl_ms.In_flag=false;
    dl_ms.Pick_up_clear =false;
    dl_ms.Delivery_clear =false;


    stop_line_process = false;
    cameraMatrix = (Mat1d(3,3)<< 7.480117180553555e+02,0., 6.438774178004406e+02, 0. ,7.474460107834711e+02, 3.663461209630810e+02, 0., 0. ,1);
    distCoeffs= (Mat1d(1,5)<< 0.185901934907552, -0.462158211869201 ,0., 0.,  0.273177418880758);
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), cameraMatrix,Size(1280,720),CV_32FC1, map1, map2);
    stop_state = false;
    stop_line_state = false;
    change_cnt = 0;
    prev_state = false;



}

void Delivery::sign_CB(const darknet_ros_msgs::BoundingBoxes::ConstPtr &msg) {
    printf("\033[2J");
    printf("\033[1;1H");

    BBOX = *msg;
    bbox_vec.clear();
    for (auto i : BBOX.bounding_boxes){
        id_bbox tmp;
        tmp.id = i.id;
        tmp.center = return_center(i.xmin, i.xmax);
        tmp.area = return_area(Point(i.xmin, i.ymin), Point(i.xmax, i.ymax));
        bbox_vec.push_back(tmp);
    }
    sort(bbox_vec.begin(), bbox_vec.end(), compare);
    for (auto i :bbox_vec){
        if (i.id>=5){
            delivery(i);
        }
    }
}
void Delivery::delivery(id_bbox id_box) {
    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(1,1);
    data << id_box.id;
    updateSample(dl, data);
    delivery_cnt +=1;
    if (delivery_cnt >=10){
        delivery_state = return_ID(mode(dl,10));
        ROS_INFO("Delivery Sign Detect: %s", delivery_state.c_str());
        dl_st.Now_state = delivery_state;
        ROS_INFO("Pick UP: %s, Delivery: %s", start.c_str(), destination.c_str());
        if (delivery_init == true && delivery_state !="ERROR" ){
            start = delivery_state;
            destination = to_go(delivery_state);

            dl_st.Pick_up = start;
            dl_st.Delivery = destination;


            during_pickup = true;
            stop_line_process = true;
            delivery_init = false;
        }
        ROS_INFO("BOX Detect, size : { %lf }", id_box.area);

        if (during_pickup ==true && id_box.area>=3000){
            ROS_INFO("================ PICK UP ================");

            pick_up_in_flag = true;
            dl_ms.In_flag = pick_up_in_flag;
            pubMission.publish(dl_ms);
            if (change_cnt ==4){
                ROS_INFO("정지 Flag 충족, 정차 : %d", change_cnt);
                stop_state = true;
                dl_ms.Stop_flag = true;
                pubMission.publish(dl_ms);
                ros::Duration(7).sleep();
                change_cnt =0;
                during_pickup =false;
                pick_up_in_flag = false;
                stop_state = false;
                dl_ms.In_flag = false;
                dl_ms.Stop_flag = false;
                dl_ms.Pick_up_clear = true;
                pubMission.publish(dl_ms);
            }
            if (change_cnt !=0){
                ROS_INFO("정지선 검출 곧 정지 : %d", change_cnt);
            }
        }
        ROS_INFO("BOX Detect, size : { %lf }", id_box.area);
        if (delivery_state== destination && during_pickup == false){
            during_delivery = true;
            if (during_delivery == true && id_box.area>=3000){
                ROS_INFO("================ Delivery ================");
                delivery_in_flag = true;
                dl_ms.In_flag = delivery_in_flag;
                pubMission.publish(dl_ms);
                if (change_cnt == 4){
                    ROS_INFO("정지 Flag 충족, 정차 : %d", change_cnt);
                    stop_state = true;
                    dl_ms.Stop_flag = true;
                    pubMission.publish(dl_ms);
                    ros::Duration(7).sleep();
                    change_cnt =0;
                    stop_state =false;
                    during_delivery = false;
                    stop_line_process = false;

                    dl_ms.Stop_flag = false;
                    dl_ms.In_flag = false;
                    dl_ms.Delivery_clear = true;
                    pubMission.publish(dl_ms);
                }
                if (change_cnt !=0){
                    ROS_INFO("정지선 검출 곧 정지 : %d", change_cnt);
                }
            }
        }
        pubState.publish(dl_st);
        pubMission.publish(dl_ms);

    }


}
string Delivery::to_go(std::string value) {
    if (value =="A1"){
        return "B1";
    }
    else if (value == "A2"){
        return "B2";
    }
    else if (value == "A3"){
        return "B3";
    }
    else return "ERROR";
}
static inline void updateSample(Eigen::MatrixXd &sample, Eigen::MatrixXd &data)
{
    if (data.cols() != sample.cols() || data.rows() != 1)
    {
        ROS_WARN("invalid data for sample matrix");
        return;
    }
    Eigen::MatrixXd temp = sample.block(1,0,sample.rows() - 1, sample.cols());
    sample.block(0,0,sample.rows() - 1, sample.cols()) = temp;
    sample.block(sample.rows()- 1, 0, 1, sample.cols()) = data;
}

int Delivery::mode(Eigen::MatrixXd data, int size) {
    int class_size = 11;
    vector<int> tmp(class_size, 0);
    for (int i = 0; i < size; i++) {
        tmp[data(i, 0)] += 1;
    }
    int max = tmp[0];
    int max_index = 0;
    for (int i = 0; i < class_size; i++) {
        if (max < tmp[i]) {
            max = tmp[i];
            max_index = i;
        }
    }

    return max_index;
}
string Delivery::return_ID(int id) {
    switch(id){
        case 0:
            return "G";
        case 1:
            return "R";
        case 2:
            return "LG";
        case 3:
            return "LR";
        case 4:
            return "Y";
        case 5:
            return "A1";
        case 6:
            return "A2";
        case 7:
            return "A3";
        case 8:
            return "B1";
        case 9:
            return "B2";
        case 10:
            return "B3";
    }
}
inline double Delivery::return_area(cv::Point p1, cv::Point p2) {
    return abs(p2.x - p1.x) * abs(p2.y - p1.y);
}
bool Delivery::compare(id_bbox a, id_bbox b) {
    return a.center> b.center;
}
/*
 * 정지선 판단 알고리즘
 */
void Delivery::cam_CB(const sensor_msgs::Image::ConstPtr &msg) {
    if (stop_line_process == true){
        try{
            frame = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
            remap(frame, frame, map1, map2, INTER_LINEAR);
            bev_frame = return_BEV(frame);
            l_frame = extract_l_frame(bev_frame);
            medianBlur(l_frame,l_frame, 7);
            calc_thresh(low,high,0.33, medianMat(l_frame));

            threshold(l_frame,bin_frame, high, 255, THRESH_BINARY);
            erode_process(bin_frame);
            Sobel(bin_frame, bin_frame,-1, 0,1);
            vector<Vec4i> lines_bin;
            HoughLinesP(bin_frame, lines_bin, 1, CV_PI/180, 10 ,100, 10);
            Mat lines(frame.rows, frame.cols, CV_8UC3, Scalar(0,0,0));
            points.clear();
            for (auto i : lines_bin){
                if (calc_theta(Point(i[0],i[1]), Point(i[2],i[3]))<3  ){
                    points.push_back(Point(i[0],i[1]));
                    points.push_back(Point(i[2],i[3]));
                }
            }
            for (auto i: points){
                circle(bin_frame, i,3, Scalar(255), 3, 8);
            }
            Vec4f line_para;
            if (points.size()>3) {
                stop_line_state = true;
                fitLine(points, line_para, DIST_L2, 0, 1e-2, 1e-2);
                cout << "line_para: " << line_para << endl;


                Point pt0;
                pt0.x = line_para[2];
                pt0.y = line_para[3];
                double k = line_para[1] / line_para[0];


                Point pt1, pt2;
                pt1.x = 0;
                pt1.y = k * (0 - pt0.x) + pt0.y;
                pt2.x = 1280;
                pt2.y = k * (1280 - pt0.x) + pt0.y;

                line(bev_frame, pt1, pt2, Scalar(0, 0, 255), 2, 8, 0);
                show("detect_line", bev_frame, 1);
            }
            else {
                stop_line_state =false;
                show("detect_line", bev_frame, 1);
            }
            ROS_INFO("Stop_line state: %s", stop_line_state ? "True" : "False");
            ROS_INFO("change_cnt: %d", change_cnt );
            detect_change(stop_line_state);
            show("base_frame",frame, 1);

        }catch(cv_bridge::Exception &e){
            ROS_INFO("Error to Convert");
        }
    }
}
void Delivery::detect_change(bool state) {
    if (state != prev_state) {
        change_cnt +=1;
        prev_state = state;
    }
}
Mat Delivery::return_BEV(cv::Mat frame) {
    int width  = frame.cols;
    int height = frame.rows;
    vector<Point2f> ps(4);
    ps[0] = Point2f(width/2 - 200, 20);
    ps[3] = Point2f(width/2 + 200, 20);
    ps[1] = Point2f(-880, height);
    ps[2] = Point2f(2160, height);

    vector<Point2f> pd(4);
    pd[0] = Point2f(0,0);
    pd[3] = Point2f(width,0);
    pd[1] = Point2f(0, height);
    pd[2] = Point2f(width, height);


    Mat perspective = getPerspectiveTransform(ps, pd);
    Mat tmp;
    warpPerspective(frame, tmp, perspective, Size(width, height), INTER_LINEAR);

    ps[0] = Point2f(width/2 , 0);
    ps[1] = Point2f(width/2, height);
    ps[2] = Point2f(width*2/3, height);
    ps[3] = Point2f(width, 0);


    pd[0] = Point2f(0,0);
    pd[1] = Point2f(0,height);
    pd[2] = Point2f(width, height);
    pd[3] = Point2f(width, 0);
    perspective = getPerspectiveTransform(ps, pd);
    warpPerspective(tmp, tmp, perspective, Size(width, height), INTER_LINEAR);
    return tmp;
}

void Delivery::show(std::string win_name, cv::Mat frame, int waitkey) {
    imshow(win_name, frame);
    waitKey(waitkey);
}
double Delivery::medianMat(cv::Mat frame) {
    frame = frame.reshape(0,1);
    vector<double> vecFromMat;
    frame.copyTo(vecFromMat);
    nth_element(vecFromMat.begin(), vecFromMat.begin()+vecFromMat.size()/2, vecFromMat.end());
    return vecFromMat[vecFromMat.size()/2];
}
void Delivery::calc_thresh(int &low, int &high, double sigma, double mid) {
    if ((1.0-sigma)*mid > 0){
        low = (1.0-sigma)*mid;
    }
    else {
        low = 0;
    }
    if((1.0+sigma)*mid<255){
        high =(1.0+sigma)*mid;
    }
    else {
        high = 250;
    }
    ROS_INFO("low: %d, high: %d, mid: %f", low, high, mid);
}
Mat Delivery::extract_l_frame(cv::Mat HLS) {
    Mat tmp ;
    cvtColor(HLS, tmp ,COLOR_BGR2HLS);
    vector<Mat> hls_images(3);
    split(tmp, hls_images);
    Mat l = hls_images[1];
    return l;

}
void Delivery::erode_process(Mat &frame) {
    cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(5,5)
            );
    erode(frame,frame,kernel);
    erode(frame,frame,kernel);
    erode(frame,frame,kernel);
}
inline double Delivery::calc_theta(cv::Point pt1, cv::Point pt2) {
    return abs(atan2(abs(pt1.y-pt2.y),abs(pt1.x-pt2.x)))*180/CV_PI;
}