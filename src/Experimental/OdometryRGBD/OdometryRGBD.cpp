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

#include <iostream>
#include <memory>
// space here?
#include <Core/Core.h>
#include <IO/IO.h>
#include "Odometry.h"

void PrintHelp(char* argv[])
{
	printf("Usage:\n");
	printf("    > %s [color1] [depth1] [color2] [depth2] [options]\n", argv[0]);
	printf("      Given RGBD image pair, estimate 6D odometry.\n");
	printf("      --camera_intrinsic [intrinsic_path]");
	printf("      --TUM : indicate this if depth map is TUM dataset");
	printf("      --verbose : shows more details");
	printf("\n");
}

int main(int argc, char *argv[])
{
	using namespace three;

	if (argc <= 4 || ProgramOptionExists(argc, argv, "--help") ||
			ProgramOptionExists(argc, argv, "-h")) {
		PrintHelp(argv);
		return 1;
	}

	std::string camera_path;
	if(ProgramOptionExists(argc, argv, "--camera_intrinsic")) { 
		camera_path = GetProgramOptionAsString(
				argc, argv, "--camera_intrinsic").c_str();
		PrintInfo("Camera intrinsic path %s\n", camera_path.c_str());
	} else {
		PrintInfo("Camera intrinsic path is not given\n", camera_path);
	}
	bool verbose = ProgramOptionExists(argc, argv, "--verbose");
	if (verbose)
		SetVerbosityLevel(three::VERBOSE_ALWAYS);
	bool is_tum = ProgramOptionExists(argc, argv, "--TUM");	
	
	auto color1_8bit = CreateImageFromFile(argv[1]);
	auto depth1_16bit = CreateImageFromFile(argv[2]);
	auto color2_8bit = CreateImageFromFile(argv[3]);
	auto depth2_16bit = CreateImageFromFile(argv[4]);	
	
	Eigen::Matrix4d trans_initial, trans_output, info_output;
	Eigen::Matrix4d trans_init_odo = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d trans_odo = Eigen::Matrix4d::Identity();
	Eigen::MatrixXd info_odo = Eigen::MatrixXd::Zero(6, 6);
	
	Odometry odo;
	Eigen::Matrix4d trans_init = Eigen::Matrix4d::Identity(4, 4);
	double lambda_dep = 0.95;

	odo.Run(*color1_8bit, *depth1_16bit, *color2_8bit, *depth2_16bit,
			trans_init, trans_odo, info_odo, 
			camera_path.c_str(), lambda_dep, false, is_tum);
	// todo: how we print out eigen matrix using PrintXXX function?
	std::cout << trans_odo << std::endl;
	std::cout << info_odo << std::endl;

	return 0;
}
