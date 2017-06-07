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


#include "Odometry.h"
#include "Helper.h"


namespace {

	// some parameters
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

	// needs discussion
	void PreprocessDepth(const three::Image &depth)
	{
		float *p = (float *)depth.data_.data();
		for (int i = 0; i < depth.height_ * depth.width_; i++, p++) {
			if ((*p > maxDepth || *p < minDepth || *p <= 0)) {
				*p = std::numeric_limits<float>::quiet_NaN();
			}
		}
	}

} // unnamed namespace

namespace three {

// lets modify open3d built-in
std::shared_ptr<Image> cvtDepth2Cloud(const Image& depth, const Eigen::Matrix3d& cameraMatrix)
{
	auto cloud = std::make_shared<Image>();
	if (depth.num_of_channels_ != 1 || depth.bytes_per_channel_ != 2) {
		PrintDebug("[cvtDepth2Cloud] Unsupported image format.\n");
		return cloud;
	}
	const double inv_fx = 1.f / cameraMatrix(0, 0);
	const double inv_fy = 1.f / cameraMatrix(1, 1);
	const double ox = cameraMatrix(0, 2);
	const double oy = cameraMatrix(1, 2);
	cloud->PrepareImage(depth.width_, depth.height_, 3, 4);

	// is this slow?
	// not sure how the data should be allocated
	int bpl = cloud->BytesPerLine();
	for (int y = 0; y < depth.height_; y++) {
		float *p = (float *)(depth.data_.data() + y * depth.BytesPerLine());
		for (int x = 0; x < cloud->width_; x++, p += 3) {
			float z = depth.data_[y * depth.width_ + x];
			*(p + 0) = (float)((x - ox) * z * inv_fx);
			*(p + 1) = (float)((y - oy) * z * inv_fy);
			*(p + 2) = z;
		}
	}
}

// don't like much. do we really need this?
// unsigned short 16bit
inline void set2shorts(int& dst, int short_v1, int short_v2)
{
	unsigned short* ptr = reinterpret_cast<unsigned short*>(&dst);
	ptr[0] = static_cast<unsigned short>(short_v1);
	ptr[1] = static_cast<unsigned short>(short_v2);
}

inline void get2shorts(int src, int& short_v1, int& short_v2)
{
	typedef union { int vint32; unsigned short vuint16[2]; } s32tou16;
	const unsigned short* ptr = (reinterpret_cast<s32tou16*>(&src))->vuint16;
	short_v1 = ptr[0];
	short_v2 = ptr[1];
}

// template?
// int 32bit
void setconst(const Image& image, const int value) {
	int *p = (int *)image.data_.data();
	for (int i = 0; i < image.height_ * image.width_; i++, p++) {
		*p = value;
	}
}

// 1->0
int computeCorresp(const Eigen::Matrix3d& K,
	const Eigen::Matrix3d& K_inv,
	const Eigen::Matrix4d& Rt,
	const Image& depth0, const Image& depth1, Image& corresps)
{
	corresps.PrepareImage(depth1.width_, depth1.height_, 1, 4);

	Eigen::Matrix3d R = Rt.block<3, 3>(0, 0);
	Eigen::Matrix3d KRK_inv = K * R * K_inv;
	const double* KRK_inv_ptr = reinterpret_cast<const double *>(KRK_inv.data());

	Eigen::Vector3d Kt = Rt.block<3, 1>(0, 3);
	Kt = K * Kt;
	const double * Kt_ptr = reinterpret_cast<const double *>(Kt.data());

	//Rect r(0, 0, depth1.cols, depth1.rows);
	setconst(corresps, -1);
	int correspCount = 0;
	for (int v1 = 0; v1 < depth1.height_; v1++) {
		for (int u1 = 0; u1 < depth1.width_; u1++) {
			float d1 = depth1.data_[v1 * depth1.width_ + u1];
			if (!std::isnan(d1)) {
				float transformed_d1 = (float)(d1 * (KRK_inv_ptr[6] * u1 + KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8]) + Kt_ptr[2]);
				int u0 = (int)((d1 * (KRK_inv_ptr[0] * u1 + KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2]) + Kt_ptr[0]) / transformed_d1 + 0.5f);
				int v0 = (int)((d1 * (KRK_inv_ptr[3] * u1 + KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5]) + Kt_ptr[1]) / transformed_d1 + 0.5f);

				if (u0 >= 0 && u0 < depth1.width_ &&
					v0 >= 0 && v0 < depth1.height_) {
					float d0 = depth0.data_[v0 * depth1.width_ + u0];
					if (!std::isnan(d0) && std::abs(transformed_d1 - d0) <= maxDepthDiff) {
						int c = corresps.data_[v0 * depth1.width_ + u0];
						if (c != -1) {
							int exist_u1, exist_v1;
							get2shorts(c, exist_u1, exist_v1);

							float exist_d1 = (float)(depth1.data_[exist_v1 * depth1.width_ + exist_u1]
								* (KRK_inv_ptr[6] * exist_u1 + KRK_inv_ptr[7] * exist_v1 + KRK_inv_ptr[8]) + Kt_ptr[2]);

							if (transformed_d1 > exist_d1)
								continue;
						}
						else
							correspCount++;

						// can we make something like
						// image.at<type>(u,v)?
						int *p = (int *)(corresps.data_.data()) + (v0 * depth1.width_ + u0);
						set2shorts(*p, u1, v1);
					}
				}
			}
		}
	}

	return correspCount;
}


bool Odometry::computeKsi(const Image& image0, const Image& cloud0,
	const Image& image1, const Image& dI_dx1, const Image& dI_dy1,
	const Image& depth0, const Image& depth1,
	const Image& dD_dx1, const Image& dD_dy1,
	const Eigen::Matrix4d& Rt,
	const Image& corresps, int correspsCount,
	const double& fx, const double& fy, const double& determinantThreshold,
	Eigen::VectorXd& ksi,
	double& res1, double& res2,
	int iter, int level)
{
	Eigen::Matrix3d R = Rt.block<3, 3>(0, 0);
	Eigen::Vector3d t = Rt.block<3, 1>(0, 3);

	int DoF = 6;
	Eigen::MatrixXd J(correspsCount * 2, DoF);
	Eigen::MatrixXd r(correspsCount * 2, 1);
	J.setConstant(0.f);
	r.setConstant(0.f);

	res1 = 0.0;
	res2 = 0.0;

	int pointCount = 0;
	//Eigen::Vector3d p3d_mat;

	double SQRT_LAMBDA_DEP, SQRT_LAMBDA_IMG;
	SQRT_LAMBDA_DEP = sqrt(LAMBDA_DEP_DEFAULT);
	SQRT_LAMBDA_IMG = sqrt(1.0f-LAMBDA_DEP_DEFAULT);

	for (int v0 = 0; v0 < corresps.height_; v0++) {
		for (int u0 = 0; u0 < corresps.width_; u0++) {
			//if (corresps.at<int>(v0, u0) != -1) {
			{
				int u1, v1;
				get2shorts(*PointerAt<int>(corresps, u0, v0), u1, v1);

				double diff = static_cast<double>(*PointerAt<float>(image1, u1, v1)) -
					static_cast<double>(*PointerAt<float>(image0, u0, v0));

				double dIdx = static_cast<double>(*PointerAt<float>(dI_dx1, u1, v1));
				double dIdy = static_cast<double>(*PointerAt<float>(dI_dy1, u1, v1));

				double dDdx = static_cast<double>(*PointerAt<float>(dD_dx1, u1, v1));
				double dDdy = static_cast<double>(*PointerAt<float>(dD_dy1, u1, v1));
				if (std::isnan(dDdx)) dDdx = 0; // isnan used in other open3d function?
				if (std::isnan(dDdy)) dDdy = 0;

				Eigen::Vector3d p3d_mat;
				p3d_mat(0) = *PointerAt<float>(cloud0, u0, v0, 0);
				p3d_mat(1) = *PointerAt<float>(cloud0, u0, v0, 1);
				p3d_mat(2) = *PointerAt<float>(cloud0, u0, v0, 2);

				Eigen::Vector3d p3d_trans = R * p3d_mat + t;

				double diff2 = static_cast<double>(*PointerAt<float>(depth1, u1, v1)) -
					static_cast<double>(p3d_trans(2));

				double invz = 1. / p3d_trans(2),
					c0 = dIdx * fx * invz,
					c1 = dIdy * fy * invz,
					c2 = -(c0 * p3d_trans(0) + c1 * p3d_trans(1)) * invz,
					d0 = dDdx * fx * invz,
					d1 = dDdy * fy * invz,
					d2 = -(d0 * p3d_trans(0) + d1 * p3d_trans(1)) * invz;

				int row1 = pointCount * 2 + 0;
				int row2 = pointCount * 2 + 1;
				J(row1, 0) = SQRT_LAMBDA_IMG * (-p3d_trans(2) * c1 + p3d_trans(1) * c2);
				J(row1, 1) = SQRT_LAMBDA_IMG * (p3d_trans(2) * c0 - p3d_trans(0) * c2);
				J(row1, 2) = SQRT_LAMBDA_IMG * (-p3d_trans(1) * c0 + p3d_trans(0) * c1);
				J(row1, 3) = SQRT_LAMBDA_IMG * (c0);
				J(row1, 4) = SQRT_LAMBDA_IMG * (c1);
				J(row1, 5) = SQRT_LAMBDA_IMG * (c2);
				r(row1, 6) = SQRT_LAMBDA_IMG * diff;
				res1 += abs(diff); // diff^2 is mathmatically correct. but used abs here for easier understand when doing rejection

				J(row2, 0) = SQRT_LAMBDA_DEP * ((-p3d_trans(2) * d1 + p3d_trans(1) * d2) - p3d_trans(1));
				J(row2, 1) = SQRT_LAMBDA_DEP * ((p3d_trans(2) * d0 - p3d_trans(0) * d2) + p3d_trans(0));
				J(row2, 2) = SQRT_LAMBDA_DEP * ((-p3d_trans(1) * d0 + p3d_trans(0) * d1));
				J(row2, 3) = SQRT_LAMBDA_DEP * (d0);
				J(row2, 4) = SQRT_LAMBDA_DEP * (d1);
				J(row2, 5) = SQRT_LAMBDA_DEP * (d2 - 1.0f);
				r(row2, 6) = SQRT_LAMBDA_DEP * diff2;
				res2 += abs(diff2);

				pointCount++;
			}
		}
	}
	res1 /= pointCount;
	res2 /= pointCount;

	if (verbose_) {
		printf("res : %.2e + %.2e (npts : %d)\n", res1, res2, pointCount);
	}

	// solve system
	Eigen::MatrixXd JtJ = J.transpose() * J;
	Eigen::MatrixXd Jtr = J.transpose() * r;

	double det = JtJ.determinant();
	//printf("det : %f\n", det);
	//cout << JtJ << endl;
	//cout << Jtr << endl;

	bool solutionExist = true;
	if (fabs(det) < determinantThreshold || std::isnan(det) || std::isinf(det))
		solutionExist = false;

	if (solutionExist)
	{
		// Robust Cholesky decomposition of a matrix with pivoting.
		ksi = JtJ.ldlt().solve(Jtr); 
	}	

	return solutionExist;
}


// don't like the names. Need to be beautified.
bool Odometry::ComputeOdometry(const Image &color0, const Image &depth0,
	const Image &color1, const Image &depth1, 
	const Eigen::Matrix4d &InitPose,
	Eigen::Matrix4d& Rt) {

	// how can I put assert?
	// assert is allowed in Open3D? guess not.
	assert(((color0.width_ == color1.width_) && (color1.height_ == color1.height_)));
	assert(((depth0.width_ == depth1.width_) && (depth0.height_ == depth1.height_)));
	assert(((color0.width_ == depth0.width_) && (color0.height_ == depth0.height_)));
	assert(((color1.width_ == depth1.width_) && (color1.height_ == depth1.height_)));

	std::vector<int> defaultIterCounts;
	std::vector<int> const* iterCountsPtr;

	PreprocessDepth(depth0);
	PreprocessDepth(depth1);

	// remove them
	auto pyramidImage0 = CreateImagePyramid(color0, NUM_PYRAMID);
	auto pyramidDepth0 = CreateImagePyramid(depth0, NUM_PYRAMID);
	auto pyramidImage1 = CreateImagePyramid(color1, NUM_PYRAMID);
	auto pyramidDepth1 = CreateImagePyramid(depth1, NUM_PYRAMID);

	auto color1_dx = FilterImage(color1, FILTER_SOBEL_3_DX);
	auto color1_dy = FilterImage(color1, FILTER_SOBEL_3_DY);
	auto pyramid_dI_dx1 = CreateImagePyramid(*color1_dx, NUM_PYRAMID);
	auto pyramid_dI_dy1 = CreateImagePyramid(*color1_dy, NUM_PYRAMID);

	auto depth1_dx = FilterImage(depth1, FILTER_SOBEL_3_DX);
	auto depth1_dy = FilterImage(depth1, FILTER_SOBEL_3_DY);
	auto pyramid_dD_dx1 = CreateImagePyramid(*depth1_dx, NUM_PYRAMID);
	auto pyramid_dD_dy1 = CreateImagePyramid(*depth1_dy, NUM_PYRAMID);

	Eigen::Matrix4d resultRt = InitPose.isZero() ? Eigen::Matrix4d::Identity() : InitPose;
	Eigen::Matrix4d currRt;
	Eigen::VectorXd ksi;
	int correspsCount;
	double res1, res2;

	for (int level = (int)iterCountsPtr->size() - 1; level >= 0; level--)
	{
		const Eigen::Matrix3d levelCameraMatrix;// = pyramidCameraMatrix[level];

		auto levelCloud0 = cvtDepth2Cloud(*pyramidDepth0[level], levelCameraMatrix);
		const double fx = levelCameraMatrix(0, 0);
		const double fy = levelCameraMatrix(1, 1);
		const double determinantThreshold = 1e-6;

		// Run transformation search on current level iteratively.
		for (int iter = 0; iter < (*iterCountsPtr)[level]; iter++) {
			if (verbose_) {
				printf("iter : %d, level : %d, ", iter, level);
			}

			Image corresps; // this is not the same way how the new variables are added.
			// it was resultRt.inv(DECOMP_SVD)
			correspsCount = computeCorresp(levelCameraMatrix, levelCameraMatrix.inverse(), resultRt.inverse(),
				*pyramidDepth0[level], *pyramidDepth1[level], corresps);

			if (correspsCount == 0)
			{
				if (verbose_) {
					printf("num of corres is 0!\n");
				}
				break;
			}

			bool solutionExist = computeKsi(
				*pyramidImage0[level], *levelCloud0,
				*pyramidImage1[level], *pyramid_dI_dx1[level], *pyramid_dI_dy1[level],
				*pyramidDepth0[level], *pyramidDepth1[level],
				*pyramid_dD_dx1[level], *pyramid_dD_dy1[level],
				resultRt,
				corresps, correspsCount,
				fx, fy, determinantThreshold,
				ksi,
				res1, res2,
				iter, level);

			if (!solutionExist)
			{
				if (verbose_) {
					printf("no solution!\n");
				}
				break;
			}

			// maybe in the other function
			Eigen::Affine3d aff_mat;
			aff_mat.linear() = (Eigen::Matrix3d) Eigen::AngleAxisd(ksi(2), Eigen::Vector3d::UnitZ())
				* Eigen::AngleAxisd(ksi(1), Eigen::Vector3d::UnitY())
				* Eigen::AngleAxisd(ksi(0), Eigen::Vector3d::UnitX());
			aff_mat.translation() = Eigen::Vector3d(ksi(3), ksi(4), ksi(5));
			currRt = aff_mat.matrix();
			//cout << currRt_eigen;

			resultRt = currRt * resultRt;
				
		}
	}

	Rt = resultRt;

	bool is_success = true;

	//if (Rt.empty()) // is this necessary?
	//	is_success = false;
	if (correspsCount < MINIMUM_CORR)
		is_success = false;
	//if (res1 > MAXIMUM_IMAGE_DIFF) // careful..
	//	is_success = false;

	return is_success;
}

}	// namespace three
