// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include "ScalableTSDFVolume.h"

#include <Core/Integration/UniformTSDFVolume.h>

namespace three{

ScalableTSDFVolume::ScalableTSDFVolume(double voxel_length, double sdf_trunc,
		bool with_color, int leaf_tsdf_resolution/* = 32*/) :
		TSDFVolume(voxel_length, sdf_trunc, with_color),
		leaf_tsdf_resolution_(leaf_tsdf_resolution)
{
}

ScalableTSDFVolume::~ScalableTSDFVolume()
{
}

void ScalableTSDFVolume::Reset()
{
}

void ScalableTSDFVolume::Integrate(const RGBDImage &image,
		const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic)
{
}

std::shared_ptr<PointCloud> ScalableTSDFVolume::ExtractPointCloud()
{
	auto pointcloud = std::make_shared<PointCloud>();
	return pointcloud;
}

std::shared_ptr<TriangleMesh> ScalableTSDFVolume::ExtractTriangleMesh()
{
	auto mesh = std::make_shared<TriangleMesh>();
	return mesh;
}

}	// namespace three
