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

#include "TSDFVolume.h"

namespace three{

TSDFVolume::TSDFVolume(double length, int resolution, double sdf_trunc,
		bool has_color) : length_(length), resolution_(resolution),
		voxel_length_(length / (double)resolution),
		voxel_num_(resolution_ * resolution_ * resolution_),
		sdf_trunc_(sdf_trunc), has_color_(has_color), sdf_(voxel_num_),
		color_(has_color ? voxel_num_ * 3 : 0), weight_(voxel_num_)
{
	Reset();
}

TSDFVolume::~TSDFVolume()
{
}

void TSDFVolume::Reset()
{
	std::fill(sdf_.begin(), sdf_.end(), 0.0f);
	std::fill(weight_.begin(), weight_.end(), 0.0f);
	if (has_color_) {
		std::fill(color_.begin(), color_.end(), 0);
	}
}

void TSDFVolume::Integrate(const Image &depth, const Image &color,
		const Eigen::Matrix4d &extrinsic)
{
}

void TSDFVolume::ExtractPointCloud(PointCloud &pointcloud)
{
}

void TSDFVolume::ExtractTriangleMesh(TriangleMesh &mesh)
{
}

}	// namespace three
