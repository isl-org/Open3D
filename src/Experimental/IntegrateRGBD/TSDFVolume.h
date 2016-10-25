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

#pragma once

#include <vector>
#include <Eigen/Core>
#include <memory>
#include <Core/Geometry/Image.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/TriangleMesh.h>

namespace three {

class TSDFVolume {
public:
	TSDFVolume(double length, int resolution, double sdf_trunc, bool has_color);
	~TSDFVolume();

public:
	void Reset();
	void Integrate(const Image &depth, const Image &color,
			const Eigen::Matrix4d &extrinsic);
	void ExtractPointCloud(PointCloud &pointcloud);
	void ExtractTriangleMesh(TriangleMesh &mesh);

protected:
	double length_;
	int resolution_;
	double voxel_length_;
	int voxel_num_;
	double sdf_trunc_;
	bool has_color_;
	std::vector<float> sdf_;
	std::vector<unsigned char> color_;
	std::vector<float> weight_;
	
private:
	int index(int x, int y, int z) {
		return x * resolution_ * resolution_ + y * resolution_ + z;
	}
};

}	// namespace three
