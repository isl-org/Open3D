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

#pragma once

#include <Core/Integration/TSDFVolume.h>

namespace three {

class UniformTSDFVolume : public TSDFVolume {
public:
	UniformTSDFVolume(double length, int resolution, double sdf_trunc,
			bool with_color, const Eigen::Vector3d &origin = Eigen::Vector3d::Zero());
	~UniformTSDFVolume() override;

public:
	void Reset() override;
	void Integrate(const RGBDImage &image,
			const PinholeCameraIntrinsic &intrinsic,
			const Eigen::Matrix4d &extrinsic) override;
	std::shared_ptr<PointCloud> ExtractPointCloud() override;
	std::shared_ptr<TriangleMesh> ExtractTriangleMesh() override;

	/// Debug function to extract the voxel data into a point cloud
	std::shared_ptr<PointCloud> ExtractVoxelPointCloud();

	/// Faster Integrate function that uses depth_to_camera_distance_multiplier
	/// precomputed from camera intrinsic
	void IntegrateWithDepthToCameraDistanceMultiplier(const RGBDImage &image,
			const PinholeCameraIntrinsic &intrinsic,
			const Eigen::Matrix4d &extrinsic,
			const Image &depth_to_camera_distance_multiplier);

	inline int IndexOf(int x, int y, int z) const {
		return x * resolution_ * resolution_ + y * resolution_ + z;
	}

	inline int IndexOf(const Eigen::Vector3i &xyz) const {
		return IndexOf(xyz(0), xyz(1), xyz(2));
	}

public:
	Eigen::Vector3d origin_;
	double length_;
	int resolution_;
	int voxel_num_;
	std::vector<float> tsdf_;
	std::vector<Eigen::Vector3f> color_;
	std::vector<float> weight_;

private:
	Eigen::Vector3d GetNormalAt(const Eigen::Vector3d &p);

	double GetTSDFAt(const Eigen::Vector3d &p);
};

}	// namespace three
