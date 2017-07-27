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

/// Class that implements a more memory efficient data structure for volumetric
/// integration
/// This implementation is based on the following repository:
/// https://github.com/qianyizh/ElasticReconstruction/tree/master/Integrate
/// The reference is:
/// Q.-Y. Zhou and V. Koltun
/// Dense Scene Reconstruction with Points of Interest
/// In SIGGRAPH 2013
/// Moreover, this implementation also utilizes observations made by the
/// following paper:
/// J. Chen, D. Bautembach, and S. Izadi
/// Scalable Real-time Volumetric Surface Reconstruction
/// In SIGGRAPH, 2013
///
/// An observed depth pixel gives two types of information: (a) an approximation
/// of the nearby surface, and (b) empty space from the camera to the surface.
/// They induce two core concepts of volumetric integration: weighted average of
/// a truncated signed distance function (TSDF), and carving. The weighted
/// average of TSDF is great in addressing the Gaussian noise along surface
/// normal and producing a smooth surface output. The carving is great in
/// removing outlier structures like floating noise pixels and bumps along
/// structure edges.
/// The scalable volume data structure has two layers of details. The leaf nodes
/// build the weighted average of TSDF. They are allocated only around the
/// surface. The interior nodes are used for carving. The carving algorithm is
/// performed in a conservative way: do not carve when uncertain.

class ScalableTSDFVolume : public TSDFVolume {
public:
	struct VolumeUnit {
	public:
		VolumeUnit() : volume_(NULL), num_of_carving_(0) {}
	public:
		std::shared_ptr<UniformTSDFVolume> volume_;
		int num_of_carving_;
	};
public:
	ScalableTSDFVolume(double voxel_length, double sdf_trunc, bool with_color,
			int volume_unit_resolution = 16, int carving_threshold = 8,
			int depth_sampling_stride = 4);
	~ScalableTSDFVolume() override;

public:
	void Reset() override;
	void Integrate(const RGBDImage &image,
			const PinholeCameraIntrinsic &intrinsic,
			const Eigen::Matrix4d &extrinsic) override;
	std::shared_ptr<PointCloud> ExtractPointCloud() override;
	std::shared_ptr<TriangleMesh> ExtractTriangleMesh() override;
	std::shared_ptr<PointCloud> ExtractVoxelPointCloud();

public:
	int volume_unit_resolution_;
	double volume_unit_length_;
	int carving_threshold_;
	int depth_sampling_stride_;

	/// Assume the index of the volume unit is (x, y, z), then the unit spans
	/// from (x, y, z) * volume_unit_length_
	/// to (x + 1, y + 1, z + 1) * volume_unit_length_
	std::unordered_map<Eigen::Vector3i, VolumeUnit,
			hash_eigen::hash<Eigen::Vector3i>> volume_units_;

private:
	Eigen::Vector3i LocateVolumeUnit(const Eigen::Vector3d &point) {
		return Eigen::Vector3i((int)std::floor(point(0) / volume_unit_length_),
				(int)std::floor(point(1) / volume_unit_length_),
				(int)std::floor(point(2) / volume_unit_length_));
	}
	
	std::shared_ptr<UniformTSDFVolume> OpenVolumeUnit(
			const Eigen::Vector3i &index);
};

}	// namespace three
