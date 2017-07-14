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

#pragma once

#include <memory>
#include <unordered_map>
#include <Core/Integration/TSDFVolume.h>
#include <Core/Utility/Helper.h>

namespace three {

class UniformTSDFVolume;

class ScalableTSDFVolume : public TSDFVolume {
public:
	struct VolumeUnit {
	public:
		std::shared_ptr<UniformTSDFVolume> volume_;
		float weight_;
	};
public:
	ScalableTSDFVolume(double voxel_length, int leaf_tsdf_resolution,
			double sdf_trunc, bool with_color);
	~ScalableTSDFVolume() override;

public:
	void Reset() override;
	void Integrate(const RGBDImage &image,
			const PinholeCameraIntrinsic &intrinsic,
			const Eigen::Matrix4d &extrinsic) override;
	std::shared_ptr<PointCloud> ExtractPointCloud() override;
	std::shared_ptr<TriangleMesh> ExtractTriangleMesh() override;

public:
	int leaf_tsdf_resolution_;
	std::unordered_map<Eigen::Vector3i, VolumeUnit,
			hash_eigen::hash<Eigen::Vector3i>> volume_units_;
};

}	// namespace three
