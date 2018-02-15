// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <cstdio>
#include <vector>
#include <Eigen/Dense>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

int main(int argc, char **argv)
{
	using namespace three;
	using namespace flann;

	SetVerbosityLevel(VerbosityLevel::VerboseAlways);

	if (argc < 2) {
		PrintInfo("Usage:\n");
		PrintInfo("    > TestFlann [filename]\n");
		PrintInfo("    The program will :\n");
		PrintInfo("    1. load the pointcloud in [filename].\n");
		PrintInfo("    2. use KDTreeFlann to compute 50 nearest neighbors of point0.\n");
		PrintInfo("    3. convert the correspondences to LineSet and render it.\n");
		PrintInfo("    4. rotate the point cloud slightly to get another point cloud.\n");
		PrintInfo("    5. find closest point of the original point cloud on the new point cloud, mark as correspondences.\n");
		PrintInfo("    6. convert to LineSet and render it.\n");
		PrintInfo("    6. distance below 0.05 are rendered as red, others as black.\n");
		return 0;
	}

	auto cloud_ptr = CreatePointCloudFromFile(argv[1]);
	std::vector<std::pair<int, int>> correspondences;

	const int nn = 50;
	KDTreeFlann kdtree;
	kdtree.SetGeometry(*cloud_ptr);
	std::vector<int> indices_vec(nn);
	std::vector<double> dists_vec(nn);
	kdtree.SearchKNN(cloud_ptr->points_[0], nn, indices_vec, dists_vec);
	for (int i = 0; i < nn; i++) {
		correspondences.push_back(std::make_pair(0, indices_vec[i]));
	}
	auto lineset_ptr = CreateLineSetFromPointCloudCorrespondences(
			*cloud_ptr, *cloud_ptr, correspondences);
	DrawGeometries({cloud_ptr, lineset_ptr});

	auto new_cloud_ptr = std::make_shared<PointCloud>();
	*new_cloud_ptr = *cloud_ptr;
	BoundingBox bounding_box;
	bounding_box.FitInGeometry(*new_cloud_ptr);
	Eigen::Matrix4d trans_to_origin = Eigen::Matrix4d::Identity();
	trans_to_origin.block<3, 1>(0, 3) = bounding_box.GetCenter() * -1.0;
	Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
	transformation.block<3, 3>(0, 0) = static_cast<Eigen::Matrix3d>(
			Eigen::AngleAxisd(M_PI / 6.0, Eigen::Vector3d::UnitX()));
	new_cloud_ptr->Transform(
			trans_to_origin.inverse() * transformation * trans_to_origin);
	correspondences.clear();
	for (size_t i = 0; i < new_cloud_ptr->points_.size(); i++) {
		kdtree.SearchKNN(new_cloud_ptr->points_[i], 1, indices_vec, dists_vec);
		correspondences.push_back(std::make_pair(indices_vec[0], (int)i));
	}
	auto new_lineset_ptr = CreateLineSetFromPointCloudCorrespondences(
			*cloud_ptr, *new_cloud_ptr, correspondences);
	new_lineset_ptr->colors_.resize(new_lineset_ptr->lines_.size());
	for (size_t i = 0; i < new_lineset_ptr->lines_.size(); i++) {
		auto point_pair = new_lineset_ptr->GetLineCoordinate(i);
		if ((point_pair.first - point_pair.second).norm() < 0.05 * 
				bounding_box.GetSize()) {
			new_lineset_ptr->colors_[i] = Eigen::Vector3d(1.0, 0.0, 0.0);
		} else {
			new_lineset_ptr->colors_[i] = Eigen::Vector3d(0.0, 0.0, 0.0);
		}
	}
	DrawGeometries({cloud_ptr, new_cloud_ptr, new_lineset_ptr});

	return 0;
}
