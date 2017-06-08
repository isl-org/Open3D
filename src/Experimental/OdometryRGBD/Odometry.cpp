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


#include <iostream> // this is just for debugging
#include "Odometry.h"

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

} // unnamed namespace

namespace three {

// needs discussion
void Odometry::PreprocessDepth(const three::Image &depth)
{
	//float *p = (float *)depth.data_.data();
	for (int y = 0; y < depth.height_; y++) {
		for (int x = 0; x < depth.width_; x++) {
			float *p = PointerAt<float>(depth, x, y);
			if ((*p > maxDepth || *p < minDepth || *p <= 0)) 
				*p = std::numeric_limits<float>::quiet_NaN();
		}
	}
}

// lets modify open3d built-in
// todo: input does not use & operator?
std::shared_ptr<Image> cvtDepth2Cloud(const Image& depth, const Eigen::Matrix3d& cameraMatrix)
{
	auto cloud = std::make_shared<Image>();
	if (depth.num_of_channels_ != 1 || depth.bytes_per_channel_ != 4) {
		PrintDebug("[cvtDepth2Cloud] Unsupported image format.\n");
		return cloud;
	}
	const double inv_fx = 1.f / cameraMatrix(0, 0);
	const double inv_fy = 1.f / cameraMatrix(1, 1);
	const double ox = cameraMatrix(0, 2);
	const double oy = cameraMatrix(1, 2);
	cloud->PrepareImage(depth.width_, depth.height_, 3, 4); // xyz float type

	// is this slow?
	// not sure how the data should be allocated
	for (int y = 0; y < depth.height_; y++) {		
		for (int x = 0; x < cloud->width_; x++) {
			float *px = PointerAt<float>(*cloud, x, y, 0);
			float *py = PointerAt<float>(*cloud, x, y, 1);
			float *pz = PointerAt<float>(*cloud, x, y, 2);
			float z = *PointerAt<float>(depth, x, y);
			*px = (float)((x - ox) * z * inv_fx);
			*py = (float)((y - oy) * z * inv_fy);
			*pz = z;
		}
	}
	return cloud;
}

// should not in namespace three.
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
	for (int v = 0; v < image.height_; v++) {
		for (int u = 0; u < image.width_; u++) {
			*PointerAt<int>(image, u, v) = value;
		}
	}
}

// 1->0
int computeCorresp(const Eigen::Matrix3d& K,
	const Eigen::Matrix3d& K_inv, /* don't need K_inv */
	const Eigen::Matrix4d& Rt,
	const Image& depth0, const Image& depth1, Image& corresps)
{
	corresps.PrepareImage(depth1.width_, depth1.height_, 1, 4);

	Eigen::Matrix3d R = Rt.block<3, 3>(0, 0);
	Eigen::Matrix3d KRK_inv = K * R * K_inv;
	//std::cout << R << std::endl;
	//std::cout << KRK_inv << std::endl;

	//const double* KRK_inv_ptr = reinterpret_cast<const double *>(KRK_inv.data());
	const double KRK_inv_ptr[9] = 
			{ KRK_inv(0,0), KRK_inv(0,1), KRK_inv(0,2),
			KRK_inv(1,0), KRK_inv(1,1), KRK_inv(1,2),
			KRK_inv(2,0), KRK_inv(2,1), KRK_inv(2,2) };

	Eigen::Vector3d Kt = Rt.block<3, 1>(0, 3);
	Kt = K * Kt;
	const double * Kt_ptr = reinterpret_cast<const double *>(Kt.data());

	//Rect r(0, 0, depth1.cols, depth1.rows);
	setconst(corresps, -1);
	int correspCount = 0;
	for (int v1 = 0; v1 < depth1.height_; v1++) {
		for (int u1 = 0; u1 < depth1.width_; u1++) {
			
			float d1 = *PointerAt<float>(depth1, u1, v1);
			if (!std::isnan(d1)) {
				float transformed_d1 = (float)(d1 * (KRK_inv_ptr[6] * u1 + KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8]) + Kt_ptr[2]);
				int u0 = (int)((d1 * (KRK_inv_ptr[0] * u1 + KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2]) + Kt_ptr[0]) / transformed_d1 + 0.5);
				int v0 = (int)((d1 * (KRK_inv_ptr[3] * u1 + KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5]) + Kt_ptr[1]) / transformed_d1 + 0.5);

				if (u0 >= 0 && u0 < depth1.width_ &&
					v0 >= 0 && v0 < depth1.height_) {

					// todo: this if else loop is not intuitive.
					float d0 = *PointerAt<float>(depth0, u0, v0);
					if (!std::isnan(d0) && std::abs(transformed_d1 - d0) <= maxDepthDiff) {
						int c = *PointerAt<int>(corresps, u0, v0);
						if (c != -1) {
							int exist_u1, exist_v1;
							get2shorts(c, exist_u1, exist_v1);

							//depth1.data_[exist_v1 * depth1.width_ + exist_u1]
							double exist_d1 = double{ *PointerAt<float>(depth1, exist_u1, exist_v1) }
								* (KRK_inv_ptr[6] * exist_u1 + KRK_inv_ptr[7] * exist_v1 + KRK_inv_ptr[8]) + Kt_ptr[2];

							if (transformed_d1 > exist_d1)
								continue;
						}
						else
							correspCount++;

						// can we make something like
						// image.at<type>(u,v)?
						int* p = PointerAt<int>(corresps, u0, v0);
						set2shorts(*p, u1, v1);
						//printf("(%d,%d)->(%d,%d)\n", u0, v0, u1, v1);
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

	const double R_raw[9] =
			{ R(0,0), R(0,1), R(0,2),
			R(1,0), R(1,1), R(1,2),
			R(2,0), R(2,1), R(2,2) };
	const double t_raw[3] =
			{ t(0), t(1), t(1) };

	Eigen::Vector3d temp, p3d_mat, p3d_trans;
	for (int v0 = 0; v0 < corresps.height_; v0++) {
		for (int u0 = 0; u0 < corresps.width_; u0++) {
			int c = *PointerAt<int>(corresps, u0, v0);
			if (c != -1) {

				int u1, v1;
				get2shorts(c, u1, v1);

				double diff = static_cast<double>(*PointerAt<float>(image1, u1, v1)) -
						static_cast<double>(*PointerAt<float>(image0, u0, v0));

				double dIdx = double{ *PointerAt<float>(dI_dx1, u1, v1) };
				double dIdy = double{ *PointerAt<float>(dI_dy1, u1, v1) };

				double dDdx = double{ *PointerAt<float>(dD_dx1, u1, v1) };
				double dDdy = double{ *PointerAt<float>(dD_dy1, u1, v1) };
				if (std::isnan(dDdx)) dDdx = 0; // todoisnan used in other open3d function?
				if (std::isnan(dDdy)) dDdy = 0;

				p3d_mat(0) = double{ *PointerAt<float>(cloud0, u0, v0, 0) };
				p3d_mat(1) = double{ *PointerAt<float>(cloud0, u0, v0, 1) };
				p3d_mat(2) = double{ *PointerAt<float>(cloud0, u0, v0, 2) };

				//temp = R * p3d_mat;
				//p3d_trans = temp + t;
				p3d_trans(0) = R_raw[0] * p3d_mat(0) + R_raw[1] * p3d_mat(1) + 
						R_raw[2] * p3d_mat(2) + t_raw[0];
				p3d_trans(1) = R_raw[3] * p3d_mat(0) + R_raw[4] * p3d_mat(1) +
						R_raw[5] * p3d_mat(2) + t_raw[1];
				p3d_trans(2) = R_raw[6] * p3d_mat(0) + R_raw[7] * p3d_mat(1) +
						R_raw[8] * p3d_mat(2) + t_raw[2];

				double diff2 = double{ (*PointerAt<float>(depth1, u1, v1)) } -
						double{ (p3d_trans(2)) };

				double invz = 1. / p3d_trans(2);
				double c0 = dIdx * fx * invz;
				double c1 = dIdy * fy * invz;
				double c2 = -(c0 * p3d_trans(0) + c1 * p3d_trans(1)) * invz;
				double d0 = dDdx * fx * invz;
				double d1 = dDdy * fy * invz;
				double d2 = -(d0 * p3d_trans(0) + d1 * p3d_trans(1)) * invz;

				int row1 = pointCount * 2 + 0;
				int row2 = pointCount * 2 + 1;
				J(row1, 0) = SQRT_LAMBDA_IMG * (-p3d_trans(2) * c1 + p3d_trans(1) * c2);
				J(row1, 1) = SQRT_LAMBDA_IMG * (p3d_trans(2) * c0 - p3d_trans(0) * c2);
				J(row1, 2) = SQRT_LAMBDA_IMG * (-p3d_trans(1) * c0 + p3d_trans(0) * c1);
				J(row1, 3) = SQRT_LAMBDA_IMG * (c0);
				J(row1, 4) = SQRT_LAMBDA_IMG * (c1);
				J(row1, 5) = SQRT_LAMBDA_IMG * (c2);
				r(row1, 0) = SQRT_LAMBDA_IMG * diff;
				res1 += abs(diff); // diff^2 is mathmatically correct. but used abs here for easier understand when doing rejection

				J(row2, 0) = SQRT_LAMBDA_DEP * ((-p3d_trans(2) * d1 + p3d_trans(1) * d2) - p3d_trans(1));
				J(row2, 1) = SQRT_LAMBDA_DEP * ((p3d_trans(2) * d0 - p3d_trans(0) * d2) + p3d_trans(0));
				J(row2, 2) = SQRT_LAMBDA_DEP * ((-p3d_trans(1) * d0 + p3d_trans(0) * d1));
				J(row2, 3) = SQRT_LAMBDA_DEP * (d0);
				J(row2, 4) = SQRT_LAMBDA_DEP * (d1);
				J(row2, 5) = SQRT_LAMBDA_DEP * (d2 - 1.0f);
				r(row2, 0) = SQRT_LAMBDA_DEP * diff2;
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
	Eigen::MatrixXd Jt = J.transpose();
	Eigen::MatrixXd JtJ = Jt * J;
	Eigen::MatrixXd Jtr = Jt * r;

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

// todo: do we have similar function for this in Open3D?
// todo: fancy c++ style file read.
void Odometry::LoadCameraFile(const char* filename, 
			int& width, int& height, Eigen::Matrix3d& K)
{
	if (strcmp(filename, "") == 0) {
		PrintDebug("Using default camera intrinsic");
		K << 525.0, 0, 319.5,
				0, 525.0, 239.5,
				0, 0, 1;
	}
	else {
		float fx_, fy_, cx_, cy_;
		float ICP_trunc_, integration_trunc_;

		//cout << filename << endl;

		FILE * f = fopen(filename, "r");
		if (f != NULL) {
			char buffer[1024];
			char* temp;
			while (fgets(buffer, 1024, f) != NULL) {
				if (strlen(buffer) > 0 && buffer[0] != '#') {
					sscanf(buffer, "%f", &fx_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &fy_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &cx_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &cy_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &ICP_trunc_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &integration_trunc_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%d", &width);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%d", &height);
				}
			}
			fclose(f);
			//printf("Camera model set to (fx, fy, cx, cy, icp_trunc, int_trunc, w, h):\n\t%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %d, %d\n",
			//	fx_, fy_, cx_, cy_, ICP_trunc_, integration_trunc_, width, height);
		}
		K(0, 0) = fx_;
		K(0, 1) = 0.0f;
		K(0, 2) = cx_;
		K(1, 0) = 0.0f;
		K(1, 1) = fy_;
		K(1, 2) = cy_;
		K(2, 0) = 0.0f;
		K(2, 1) = 0.0f;
		K(2, 2) = 1.0f;
	}
	//cout << K << endl;
}


bool Odometry::Run(
	const Image& color1, const Image& depth1,
	const Image& color2, const Image& depth2,
	Eigen::Matrix4d& init_pose, Eigen::Matrix4d& trans_output, Eigen::MatrixXd& info_output,
	const char* filename,
	const double lambda_dep,
	bool verbose,
	bool fast_reject)
{
	verbose_ = verbose;
	//printf("verbose : %d\n", verbose_);

	if (lambda_dep < 0.0f || lambda_dep > 1.0f)
		LAMBDA_DEP = LAMBDA_DEP_DEFAULT;
	else
		LAMBDA_DEP = lambda_dep;
	LAMBDA_IMG = 1.0f - LAMBDA_DEP;

	printf("LAMBDA_DEP : %f\n", LAMBDA_DEP);
	printf("LAMBDA_IMG : %f\n", LAMBDA_IMG);


	Eigen::Matrix3d cameraMatrix;
	int width, height;
	LoadCameraFile(filename, width, height, cameraMatrix);

	// doto: don't like 0,1 or 1,2
	auto grayImage0_temp = CreateFloatImageFromImage(color1);
	auto grayImage1_temp = CreateFloatImageFromImage(color2);
	// applying blur - filter sizes would be applied to the image the pyramid are [5 9 17]
	auto grayImage0 = FilterImage(*grayImage0_temp, FILTER_GAUSSIAN_3);
	auto grayImage1 = FilterImage(*grayImage1_temp, FILTER_GAUSSIAN_3);

	// depth preprocessing 
	PreprocessDepth(depth1);
	PreprocessDepth(depth2);
	auto depthFlt0_filtered = FilterImage(depth1, FILTER_GAUSSIAN_3);
	auto depthFlt1_filtered = FilterImage(depth2, FILTER_GAUSSIAN_3);

	PrintInfo("grayImage0_temp(100,100) : %f\n", *PointerAt<float>(*grayImage0_temp, 100, 100));
	PrintInfo("grayImage0(100,100) : %f\n", *PointerAt<float>(*grayImage0, 100, 100));
	PrintInfo("depthFlt0(100,100) : %f\n", *PointerAt<float>(depth1, 100, 100));
	PrintInfo("depthFlt0_filtered(100,100) : %f\n", *PointerAt<float>(*depthFlt0_filtered, 100, 100));
	PrintInfo("depthFlt0(200,200) : %f\n", *PointerAt<float>(depth1, 200, 200));
	PrintInfo("depthFlt0_filtered(200,200) : %f\n", *PointerAt<float>(*depthFlt0_filtered, 200, 200));
	
	Eigen::Matrix4d Rt_init = Eigen::Matrix4d::Identity();
	//init_pose.copyTo(Rt_init);

	// todo: how do we print a matrix?
	//if (verbose_) {
	//	std::cout << "Initial camera pose " << init_pose << endl;
	//}

	//std::cout << cameraMatrix << std::endl;

	// if initial camera pose is given, we use it ONLY IF it provides good number of correspondences
	// test input matrix and use identity matrix if it is bad
	Image temp_corresps;
	// it was Rt_init.inv(DECOMP_SVD)
	int correspsCount = computeCorresp(cameraMatrix, cameraMatrix.inverse(), Rt_init.inverse(),
		*depthFlt0_filtered, *depthFlt1_filtered, temp_corresps);

	//if (verbose_) {
	//	cout << "Number of correspondence is " << correspsCount << endl;
	//}

	// if there is not enough correspondences,
	// output zero transformation matrix and information matrix
	if (fast_reject) {
		if (correspsCount < MINIMUM_CORR)
			return false;
	}

	std::vector<int> iterCounts;
	for (int i = 0; i < NUM_PYRAMID; i++)
		iterCounts.push_back(NUM_ITER); 

	//// added by jspark
	//// overlapping region based intensity normalization
	// todo: does not like additional correspondence search
	intensity_normalization(*grayImage0, *grayImage1, temp_corresps); 

	Eigen::Matrix4d Rt;
	bool isFound = ComputeOdometry(
			Rt, Rt_init,
			*grayImage0, *depthFlt0_filtered,
			*grayImage1, *depthFlt1_filtered,
			cameraMatrix,
			iterCounts);

	//if (fast_reject) {
	//	if (!isFound)
	//	{
	//		if (verbose_) {
	//			cout << "Rigid body motion cann't be estimated for given RGBD data." << endl;
	//		}
	//		return false;
	//	}
	//}

	// output result that can be used for FragmentOptimizer.
	//WriteResult(Rt, cameraMatrix, depthFlt0_filtered, depthFlt1_filtered, trans_log, trans_info);
	Eigen::Matrix4d Rt_inv = Rt.inverse();
	trans_output = Rt_inv;
	//GetInfo(Rt, cameraMatrix, depthFlt0_filtered, depthFlt1_filtered, info_output);

	//if (verbose_)
	//{
	//	cout << trans_output << endl;
	//	cout << info_output << endl;
	//}

	return true;
}


void Odometry::intensity_normalization(
		Image& image0, Image& image1, Image& corresps)
{
	if (image0.width_ != image1.width_ ||
		image0.height_ != image1.height_) {
		PrintDebug("[intensity_normalization] Two input images should be same size\n");
		return;
	}

	size_t pointCount = 0;
	double mean0 = 0.0;
	double mean1 = 0.0;
	// todo: uncomment
	// do smart iteration
	for (int v0 = 0; v0 < corresps.height_; v0++) {
		for (int u0 = 0; u0 < corresps.width_; u0++) {
			int c = *PointerAt<int>(corresps, u0, v0);
			if (c != -1) {
				int u1, v1;
				get2shorts(c, u1, v1);
				mean0 += *PointerAt<float>(image0, u0, v0);
				mean1 += *PointerAt<float>(image1, u1, v1);
				pointCount++;
			}			
		}
	}
	mean0 /= (double)pointCount;
	mean1 /= (double)pointCount;
	LinearTransformImage(image0, 0.5 / mean0, 0.0); // todo: why 0.5 applied here?
	LinearTransformImage(image1, 0.5 / mean1, 0.0);
}

std::vector<Eigen::Matrix3d> 
		Odometry::BuildCameraMatrixPyramid(Eigen::Matrix3d& K, int levels)
{
	std::vector<Eigen::Matrix3d> pyramidCameraMatrix;
	pyramidCameraMatrix.reserve(levels);
	for (int i = 0; i < levels; i++) {
		Eigen::Matrix3d levelCameraMatrix = i == 0 ? K : 0.5f * pyramidCameraMatrix[i - 1];
		//Mat levelCameraMatrix = cameraMatrix_dbl;
		levelCameraMatrix(2, 2) = 1.;
		pyramidCameraMatrix.push_back(levelCameraMatrix);
	}
	return pyramidCameraMatrix; // todo: is this good?
}

// don't like the names. Need to be beautified.
bool Odometry::ComputeOdometry(
		Eigen::Matrix4d& Rt, const Eigen::Matrix4d& initRt,
		const Image &color0, const Image &depth0,
		const Image &color1, const Image &depth1, 
		Eigen::Matrix3d& cameraMatrix,
		const std::vector<int>& iterCounts) {

	// how can I put assert?
	// assert is allowed in Open3D? guess not.
	assert(((color0.width_ == color1.width_) && (color1.height_ == color1.height_)));
	assert(((depth0.width_ == depth1.width_) && (depth0.height_ == depth1.height_)));
	assert(((color0.width_ == depth0.width_) && (color0.height_ == depth0.height_)));
	assert(((color1.width_ == depth1.width_) && (color1.height_ == depth1.height_)));

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

	Eigen::Matrix4d resultRt = initRt.isZero() ? Eigen::Matrix4d::Identity() : initRt;
	Eigen::Matrix4d currRt;
	Eigen::VectorXd ksi;
	int correspsCount;
	double res1, res2;

	std::vector<Eigen::Matrix3d> pyramidCameraMatrix =
			BuildCameraMatrixPyramid(cameraMatrix, (int)iterCounts.size());

	for (int level = (int)iterCounts.size() - 1; level >= 0; level--)
	{
		const Eigen::Matrix3d levelCameraMatrix = pyramidCameraMatrix[level];

		auto levelCloud0 = cvtDepth2Cloud(*pyramidDepth0[level], levelCameraMatrix);
		const double fx = levelCameraMatrix(0, 0);
		const double fy = levelCameraMatrix(1, 1);
		const double determinantThreshold = 1e-6;

		// Run transformation search on current level iteratively.
		for (int iter = 0; iter < iterCounts[level]; iter++) {
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
			
			resultRt = currRt * resultRt;

			//std::cout << "currRt" << std::endl;
			//std::cout << currRt << std::endl;
			//std::cout << "resultRt" << std::endl;
			//std::cout << resultRt << std::endl;

				
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
