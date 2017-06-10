// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncle@gmail.com>
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
#include <IO/IO.h>

namespace {

	// some parameters - should be in headers?
	const static double LAMBDA_DEP_DEFAULT = 0.5f;
	const static double MINIMUM_CORR = 30000;
	const static int	NUM_PYRAMID = 4;		// 4
	const static int	NUM_ITER = 10;			// 7
	const static double maxDepthDiff = 0.07f;	//in meters	(0.07)
	const static int	ADJ_FRAMES = 1;
	const static double depthedge = 0.3f;
	const static int	depthedgedilation = 1;
	const static double minDepth = 0.f;			//in meters (0.0)
	const static double maxDepth = 4.f; 		//in meters (4.0)	
	const static double sobelScale_i = 1. / 8;
	const static double sobelScale_d = 1. / 8;

} // unnamed namespace

namespace three {

class Odometry {
public:
	
	int ComputeCorrespondence(const Eigen::Matrix3d& K,
		const Eigen::Matrix4d& Rt,
		const Image& depth0, const Image& depth1, Image& corresps);

	bool ComputeKsi(
		const Image& image0, const Image& cloud0, const Image& image1,
		const Image& dI_dx1, const Image& dI_dy1,
		const Image& depth0, const Image& depth1,
		const Image& dD_dx1, const Image& dD_dy1,
		const Eigen::Matrix4d& Rt,
		const Image& corresps, int correspsCount,
		const double& fx, const double& fy,
		const double& determinant_threshold,
		Eigen::VectorXd& ksi,
		int iter, int level);

	bool ComputeOdometry(
			Eigen::Matrix4d& Rt, const Eigen::Matrix4d& initRt,
			const Image &gray0, const Image &depth0,
			const Image &gray1, const Image &depth1,
			Eigen::Matrix3d& cameraMatrix,
			const std::vector<int>& iterCounts);
	
	std::shared_ptr<Image> ConvertDepth2Cloud(
			const Image& depth, const Eigen::Matrix3d& cameraMatrix);

	std::vector<Eigen::Matrix3d> 
			CreateCameraMatrixPyramid(Eigen::Matrix3d& K, int levels);

	Eigen::MatrixXd CreateInfomationMatrix(const Eigen::Matrix4d& Rt,
		const Eigen::Matrix3d& cameraMatrix,
		const Image& depth0, const Image& depth1);

	void LoadCameraFile(const char* filename, Eigen::Matrix3d& K);

	void NormalizeIntensity(Image& image0, Image& image1, Image& corresps);

	/// Function to mask invalid depth (0 depth or larger than maximum distance)
	void PreprocessDepth(const Image &depth);

	bool Run(
			const Image& color0_8bit, const Image& depth0_16bit,
			const Image& color1_8bit, const Image& depth1_16bit,
			Eigen::Matrix4d& init_pose, 
			Eigen::Matrix4d& trans_output, Eigen::MatrixXd& info_output,
			const char* filename,
			const double lambda_dep,
			bool fast_reject,
			bool is_tum);

protected:
	double lambda_dep_;
	double lambda_img_;

private:
		
};

}	// namespace three
