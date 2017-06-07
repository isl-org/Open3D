// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Jaesik Park <syncle@gmail.com>
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

// ?? not sure which modules need to be included.
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <Core/Geometry/Image.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/TriangleMesh.h>
#include <Core/Utility/Console.h>
#include <Core/Camera/PinholeCameraIntrinsic.h>

namespace three {

class Odometry {
public:
	// how to define constructor and descructor
	//Odometry();
	//~Odometry();		
	bool ComputeOdometry(
		Eigen::Matrix4d& Rt, const Eigen::Matrix4d& initRt,
		const Image &color0, const Image &depth0,
		const Image &color1, const Image &depth1,
		Eigen::Matrix3d& cameraMatrix,
		const std::vector<int>& iterCounts);
	//void cvtDepth2Cloud(const Image& depth, Image& cloud,
	//	const Eigen::Matrix4d& cameraMatrix);

	bool computeKsi(const Image& image0, const Image& cloud0,
		const Image& image1, const Image& dI_dx1, const Image& dI_dy1,
		const Image& depth0, const Image& depth1,
		const Image& dD_dx1, const Image& dD_dy1,
		const Eigen::Matrix4d& Rt,
		const Image& corresps, int correspsCount,
		const double& fx, const double& fy, const double& determinantThreshold,
		Eigen::VectorXd& ksi,
		double& res1, double& res2,
		int iter, int level);

	void intensity_normalization(Image& image0, Image& image1, Image& corresps);

	bool Run(
		const Image& color1, const Image& depth1,
		const Image& color2, const Image& depth2,
		Eigen::Matrix4d& init_pose, Eigen::Matrix4d& trans_output, Eigen::MatrixXd& info_output,
		const char* filename,
		const double lambda_dep,
		bool verbose,
		bool fast_reject);

	void LoadCameraFile(const char* filename,
		int& width, int& height, Eigen::Matrix3d& K);

	void PreprocessDepth(const three::Image &depth);

protected:
	bool verbose_;
	double LAMBDA_DEP;
	double LAMBDA_IMG;

private:
		
};

}	// namespace three
