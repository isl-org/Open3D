// TestPointCloud.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <Core/PointCloud.h>

int main(int argc, char *argv[])
{
	PointCloud point_cloud;
	std::cout << "Pointcloud has " << point_cloud.points_.size() << " points." << std::endl;

	point_cloud.points_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
	point_cloud.points_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
	point_cloud.points_.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
	point_cloud.points_.push_back(Eigen::Vector3d(0.0, 0.0, 1.0));
	std::cout << "Pointcloud has " << point_cloud.points_.size() << " points." << std::endl;
}