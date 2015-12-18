// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
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

#include "PointCloud.h"

namespace three{

namespace {

}	// unnamed namespace

bool EstimateNormal(PointCloud &cloud, 
		const KDTreeSearchParam &search_param/* = KDTreeSearchParamKNN()*/,
		const Eigen::Vector3d &orientation_reference
		/* = Eigen::Vector3d(0.0, 0.0, 1.0)*/)
{
	bool has_normal = cloud.HasNormals();
	if (cloud.HasNormals() == false) {
		cloud.normals_.resize(cloud.points_.size());
	}
	if (cloud.HasCurvatures() == false) {
		cloud.curvatures_.resize(cloud.points_.size());
	}
	// TODO
	// https://github.com/PointCloudLibrary/pcl/blob/a654fe4188382416c99322cafbd9319c59a7355c/common/include/pcl/common/impl/centroid.hpp
	// computeMeanAndCovarianceMatrix 
	return true;
}

}	// namespace three
