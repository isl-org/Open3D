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

#include "KDTreeFlann.h"

#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/TriangleMesh.h>

namespace three{

bool KDTreeFlann::SetGeometry(const Geometry &geometry)
{
	switch (geometry.GetGeometryType()) {
	case Geometry::GEOMETRY_POINTCLOUD:
		data_ = ((const PointCloud &)geometry).points_;
		break;
	case Geometry::GEOMETRY_TRIANGLEMESH:
		data_ = ((const TriangleMesh &)geometry).vertices_;
		break;
	case Geometry::GEOMETRY_IMAGE:
	case Geometry::GEOMETRY_UNKNOWN:
	default:
		return false;
		break;
	}
	if (data_.size() == 0) {
		return false;
	}
	flann_dataset_.reset(new flann::Matrix<double>(
		(double *)data_.data(),
		data_.size(), 3
		));
	dimension_ = 3;
	dataset_size_ = data_.size();
	flann_index_.reset(new flann::Index<flann::L2<double>>(*flann_dataset_,
		flann::KDTreeSingleIndexParams(10)));
	flann_index_->buildIndex();
	return true;
}

}	// namespace three
