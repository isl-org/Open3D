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

#include "PointCloud.h"
#include "TriangleMesh.h"

namespace three{

KDTreeFlann::~KDTreeFlann()
{
}
	
bool KDTreeFlann::AddGeometry(std::shared_ptr<const Geometry> geometry_ptr)
{
	switch (geometry_ptr->GetGeometryType()) {
	case Geometry::GEOMETRY_POINTCLOUD:
	case Geometry::GEOMETRY_TRIANGLEMESH:
		geometry_ptr_ = geometry_ptr;
		break;
	case Geometry::GEOMETRY_IMAGE:
	case Geometry::GEOMETRY_UNKNOWN:
	default:
		return false;
		break;
	}
	return UpdateGeometry();
}

bool KDTreeFlann::UpdateGeometry()
{
	if (geometry_ptr_->GetGeometryType() == 
			Geometry::GEOMETRY_POINTCLOUD) {
		const auto &pointcloud = (const PointCloud &)(*geometry_ptr_);
		if (pointcloud.HasPoints() == false) {
			return false;
		}
		flann_dataset_.reset(new flann::Matrix<double>(
				(double *)pointcloud.points_.data(),
				pointcloud.points_.size(), 3
				));
		dimension_ = 3;
	} else if (geometry_ptr_->GetGeometryType() == 
			Geometry::GEOMETRY_TRIANGLEMESH) {
		const auto &mesh = (const TriangleMesh &)(*geometry_ptr_);
		if (mesh.HasVertices() == false) {
			return false;
		}
		flann_dataset_.reset(new flann::Matrix<double>(
				(double *)mesh.vertices_.data(),
				mesh.vertices_.size(), 3
				));
		dimension_ = 3;
	} else {
		return false;
	}
	flann_index_.reset(new flann::Index<flann::L2<double>>(*flann_dataset_,
			flann::KDTreeSingleIndexParams(10)));
	flann_index_->buildIndex();
	return true;
}

bool KDTreeFlann::HasGeometry() const
{
	return bool(geometry_ptr_);
}

}	// namespace three
