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

#include <Core/Geometry/PointCloud.h>
#include <Core/Integration/UniformTSDFVolume.h>

namespace three{

ScalableTSDFVolume::ScalableTSDFVolume(double voxel_length, double sdf_trunc,
		bool with_color, int volume_unit_resolution/* = 16*/,
		int carving_threshold/* = 8*/) :
		TSDFVolume(voxel_length, sdf_trunc, with_color),
		volume_unit_resolution_(volume_unit_resolution),
		volume_unit_length_(voxel_length * volume_unit_resolution),
		carving_threshold_(carving_threshold)
{
}

ScalableTSDFVolume::~ScalableTSDFVolume()
{
	volume_units_.clear();
}

void ScalableTSDFVolume::Reset()
{
	volume_units_.clear();
}

void ScalableTSDFVolume::Integrate(const RGBDImage &image,
		const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic)
{
	if ((image.depth_.num_of_channels_ != 1) || 
			(image.depth_.bytes_per_channel_ != 4) ||
			(image.depth_.width_ != intrinsic.width_) ||
			(image.depth_.height_ != intrinsic.height_) ||
			(with_color_ && image.color_.num_of_channels_ != 3) ||
			(with_color_ && image.color_.bytes_per_channel_ != 1) ||
			(with_color_ && image.color_.width_ != intrinsic.width_) ||
			(with_color_ && image.color_.height_ != intrinsic.height_)) {
		PrintWarning("[ScalableTSDFVolume::Integrate] Unsupported image format. Please check if you have called CreateRGBDImageFromColorAndDepth() with convert_rgb_to_intensity=false.\n");
		return;
	}
	auto depth2cameradistance = CreateDepthToCameraDistanceMultiplierFloatImage(
			intrinsic);
	auto pointcloud = CreatePointCloudFromRGBDImage(image, intrinsic,
			extrinsic);
	for (const auto &point : pointcloud->points_) {

	}
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
