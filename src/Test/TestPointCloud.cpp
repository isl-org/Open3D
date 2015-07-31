// TestPointCloud.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <Core/Console.h>
#include <Core/PointCloud.h>
#include <IO/PointCloudIO.h>

void PrintPointCloud(const three::PointCloud &pointcloud) {
	using namespace three;
	
	bool pointcloud_has_normal = pointcloud.HasNormals();
	PrintInfo("Pointcloud has %lu points.\n",
			pointcloud.points_.size());
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		if (pointcloud_has_normal) {
			const Eigen::Vector3d &point = pointcloud.points_[i];
			const Eigen::Vector3d &normal = pointcloud.normals_[i];
			PrintDebug("%.6f %.6f %.6f %.6f %.6f %.6f\n", 
					point(0), point(1), point(2), 
					normal(0), normal(1), normal(2));
		} else {
			const Eigen::Vector3d &point = pointcloud.points_[i];
			PrintDebug("%.6f %.6f %.6f\n", point(0), point(1), point(2));
		}
	}
	PrintDebug("End of the list.\n\n");
}

int main(int argc, char *argv[])
{
	using namespace three;

	//SetVerbosityLevel(VERBOSE_ALWAYS);

	// 1. test basic pointcloud functions.
	
	PointCloud pointcloud;
	PrintPointCloud(pointcloud);

	pointcloud.points_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
	pointcloud.points_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
	pointcloud.points_.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
	pointcloud.points_.push_back(Eigen::Vector3d(0.0, 0.0, 1.0));
	PrintPointCloud(pointcloud);

	// 2. test pointcloud IO.

	const std::string filename_xyz("test.xyz");
	const std::string filename_ply("test.ply");

	if (ReadPointCloudFromPLY(argv[1], pointcloud)) {
		PrintWarning("Successfully read %s\n", argv[1]);
		PrintPointCloud(pointcloud);
		
		PointCloud pointcloud_copy;
		pointcloud_copy.CloneFrom(pointcloud);
		
		if (WritePointCloudToXYZ(filename_xyz, pointcloud)) {
			PrintWarning("Successfully wrote %s\n\n", filename_xyz.c_str());
		} else {
			PrintError("Failed to write %s\n\n", filename_xyz.c_str());
		}

		if (WritePointCloudToPLY(filename_ply, pointcloud_copy)) {
			PrintWarning("Successfully wrote %s\n\n", filename_ply.c_str());
		} else {
			PrintError("Failed to write %s\n\n", filename_ply.c_str());
		}
	} else {
		PrintError("Failed to read %s\n\n", argv[1]);
	}

	// n. test end

	PrintAlways("End of the test.\n");
}