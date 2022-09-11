//
// Created by sumin on 22. 9. 11.
//
#include "delivery/delivery.h"

int main (int argc, char ** argv){
    ros::init(argc, argv, "delivery");
    Delivery dl;
    ros::spin();
    return 0;
}