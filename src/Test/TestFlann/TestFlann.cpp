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

#include <flann/flann.hpp>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

int main(int argc, char **argv)
{
	using namespace three;
	using namespace flann;
	
	if (argc < 2) {
		PrintInfo("Usage:\n");
		PrintInfo("    > TestFlann [filename]\n");
		return 0;
	}
	
	auto cloud_ptr = std::make_shared<PointCloud>();
	if (ReadPointCloud(argv[1], *cloud_ptr)) {
		PrintWarning("Successfully read %s\n", argv[1]);
	} else {
		PrintError("Failed to read %s\n\n", argv[1]);
		return 0;
	}

	if ((int)cloud_ptr->points_.size() < 100) {
		PrintError("Boring point cloud.\n");
		return 0;
	}

	if (cloud_ptr->HasColors() == false) {
		cloud_ptr->colors_.resize(cloud_ptr->points_.size());
		for (size_t i = 0; i < cloud_ptr->points_.size(); i++) {
			cloud_ptr->colors_[i].setZero();
		}
	}

	int nn = std::min(20, (int)cloud_ptr->points_.size() - 1);
	Matrix<double> dataset((double *)cloud_ptr->points_.data(),
			cloud_ptr->points_.size(), 3 );
	Matrix<double> query((double *)cloud_ptr->points_.data(), 1, 3);
	std::vector<int> indices_vec(nn);
	std::vector<double> dists_vec(nn);
	Matrix<int> indices(indices_vec.data(), query.rows, nn);
	Matrix<double> dists(dists_vec.data(), query.rows, nn);
	Index<L2<double>> index(dataset, KDTreeSingleIndexParams(10));
	index.buildIndex();
	index.knnSearch(query, indices, dists, nn, SearchParams(-1, 0.0));
	
	for (size_t i = 0; i < indices_vec.size(); i++) {
		PrintInfo("%d, %f\n", (int)indices_vec[i], sqrt(dists_vec[i]));
		cloud_ptr->colors_[indices_vec[i]] = Eigen::Vector3d(1.0, 0.0, 0.0);
	}

	cloud_ptr->colors_[0] = Eigen::Vector3d(0.0, 1.0, 0.0);
	
	float r = float(sqrt(dists_vec[nn - 1]) * 2.0);
	Matrix<double> query1((double *)cloud_ptr->points_.data() + 3 * 99, 1, 3);
	int k = index.radiusSearch(query1, indices, dists, r * r, 
			SearchParams(-1, 0.0));

	PrintInfo("======== %d, %f ========\n", k, r);
	for (int i = 0; i < k; i++) {
		PrintInfo("%d, %f\n", (int)indices_vec[i], sqrt(dists_vec[i]));
		cloud_ptr->colors_[indices_vec[i]] = Eigen::Vector3d(0.0, 0.0, 1.0);
	}
	cloud_ptr->colors_[99] = Eigen::Vector3d(0.0, 1.0, 1.0);

	DrawGeometries({cloud_ptr}, "TestFlann", 1600, 900);

	auto new_cloud_ptr = std::make_shared<PointCloud>();
	if (ReadPointCloud(argv[1], *new_cloud_ptr)) {
		PrintWarning("Successfully read %s\n", argv[1]);
	} else {
		PrintError("Failed to read %s\n\n", argv[1]);
		return 0;
	}

	if ((int)new_cloud_ptr->points_.size() < 100) {
		PrintError("Boring point cloud.\n");
		return 0;
	}

	if (new_cloud_ptr->HasColors() == false) {
		new_cloud_ptr->colors_.resize(new_cloud_ptr->points_.size());
		for (size_t i = 0; i < new_cloud_ptr->points_.size(); i++) {
			new_cloud_ptr->colors_[i].setZero();
		}
	}

	KDTreeFlann kdtree;
	kdtree.SetGeometry(*new_cloud_ptr);
	std::vector<int> new_indices_vec(nn);
	std::vector<double> new_dists_vec(nn);
	kdtree.SearchKNN(new_cloud_ptr->points_[0], nn, 
			new_indices_vec, new_dists_vec);
	
	for (size_t i = 0; i < new_indices_vec.size(); i++) {
		PrintInfo("%d, %f\n", (int)new_indices_vec[i], sqrt(new_dists_vec[i]));
		new_cloud_ptr->colors_[new_indices_vec[i]] = 
				Eigen::Vector3d(1.0, 0.0, 0.0);
	}

	new_cloud_ptr->colors_[0] = Eigen::Vector3d(0.0, 1.0, 0.0);

	k = kdtree.SearchRadius(new_cloud_ptr->points_[99], r, new_indices_vec,
			new_dists_vec);

	PrintInfo("======== %d, %f ========\n", k, r);
	for (int i = 0; i < k; i++) {
		PrintInfo("%d, %f\n", (int)new_indices_vec[i], sqrt(new_dists_vec[i]));
		new_cloud_ptr->colors_[new_indices_vec[i]] = 
				Eigen::Vector3d(0.0, 0.0, 1.0);
	}
	new_cloud_ptr->colors_[99] = Eigen::Vector3d(0.0, 1.0, 1.0);

	k = kdtree.Search(new_cloud_ptr->points_[199], KDTreeSearchParamRadius(r), 
			new_indices_vec, new_dists_vec);

	PrintInfo("======== %d, %f ========\n", k, r);
	for (int i = 0; i < k; i++) {
		PrintInfo("%d, %f\n", (int)new_indices_vec[i], sqrt(new_dists_vec[i]));
		new_cloud_ptr->colors_[new_indices_vec[i]] = 
				Eigen::Vector3d(0.0, 0.0, 1.0);
	}
	new_cloud_ptr->colors_[199] = Eigen::Vector3d(0.0, 1.0, 1.0);

	DrawGeometries({new_cloud_ptr}, "TestKDTree", 1600, 900);
	return 0;
}
