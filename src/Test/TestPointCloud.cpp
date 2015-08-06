// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <iostream>
#include <memory>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

void PrintPointCloud(const three::PointCloud &pointcloud) {
	using namespace three;
	
	bool pointcloud_has_normal = pointcloud.HasNormals();
	PrintInfo("Pointcloud has %lu points.\n",
			pointcloud.points_.size());

	Eigen::Vector3d min_bound = pointcloud.GetMinBound();
	Eigen::Vector3d max_bound = pointcloud.GetMaxBound();
	PrintInfo("Bounding box is: (%.4f, %.4f, %.4f) - (%.4f, %.4f, %.4f)\n",
			min_bound(0), min_bound(1), min_bound(2),
			max_bound(0), max_bound(1), max_bound(2));

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
		//PrintPointCloud(pointcloud);
		
		PointCloud pointcloud_copy;
		pointcloud_copy.CloneFrom(pointcloud);
		
		//if (WritePointCloudToXYZ(filename_xyz, pointcloud)) {
		//	PrintWarning("Successfully wrote %s\n\n", filename_xyz.c_str());
		//} else {
		//	PrintError("Failed to write %s\n\n", filename_xyz.c_str());
		//}

		if (WritePointCloudToPLY(filename_ply, pointcloud_copy)) {
			PrintWarning("Successfully wrote %s\n\n", filename_ply.c_str());
		} else {
			PrintError("Failed to write %s\n\n", filename_ply.c_str());
		}
	} else {
		PrintError("Failed to read %s\n\n", argv[1]);
	}

	// 3. test pointcloud visualization

	Visualizer visualizer;
	std::shared_ptr<PointCloud> pointcloud_ptr(new PointCloud);
	pointcloud_ptr->CloneFrom(pointcloud);
	visualizer.AddPointCloud(pointcloud_ptr);
	visualizer.CreateWindow("Open3DV", 1600, 900);
	visualizer.Run();


	//while (!visualizer.IsWindowTerminated());
	// n. test end

	PrintAlways("End of the test.\n");
}