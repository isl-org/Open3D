// TestPointCloud.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <Core/PointCloud.h>
#include <IO/PointCloudIO.h>

void PrintPointCloud(const three::PointCloud &pointcloud) {
	using namespace three;
	
	printf("Pointcloud has %lu points, as follows:\n", pointcloud.points_.size());
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		const Eigen::Vector3d &point = pointcloud.points_[i];
		printf("%.6f %.6f %.6f\n", point(0), point(1), point(2));
	}
	printf("End of the list.\n\n");
}

int main(int argc, char *argv[])
{
	using namespace three;

	// 1. test basic pointcloud functions.
	
	PointCloud pointcloud;
	PrintPointCloud(pointcloud);

	pointcloud.points_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
	pointcloud.points_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
	pointcloud.points_.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
	pointcloud.points_.push_back(Eigen::Vector3d(0.0, 0.0, 1.0));
	PrintPointCloud(pointcloud);

	// 2. test pointcloud IO.

	const std::string filename("test.xyz");
	
	if (WritePointCloudToXYZ(filename, pointcloud)) {
		printf("Successfully wrote %s\n\n", filename.c_str());
	} else {
		printf("Failed to write %s\n\n", filename.c_str());
	}

	PointCloud pointcloud_copy;
	if (ReadPointCloudFromXYZ(filename, pointcloud_copy)) {
		printf("Successfully read %s\n", filename.c_str());
		PrintPointCloud(pointcloud_copy);
	} else {
		printf("Failed to read %s\n\n", filename.c_str());
	}

	// n. test end

	printf("End of the test.\n");
}