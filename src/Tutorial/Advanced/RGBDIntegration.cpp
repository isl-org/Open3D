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

#include <iostream>
#include <memory>
#include <Eigen/Dense>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Core/Utility/Timer.h>

#include <fstream>
#include <sstream>
#include <iomanip>
using namespace three;

typedef int Metadata[3];

class CameraPose
{
public:
	CameraPose(Metadata meta, Eigen::Matrix4d mat)
	{
		for(int i=0; i<3; i++) metadata[i] = meta[i];
		pose = mat;
	};

	friend std::ostream& operator << (std::ostream& os, const CameraPose& p);

	Metadata metadata;
	Eigen::Matrix4d pose;
};

std::ostream& operator << (std::ostream& os, const CameraPose& p)
{
	os << "Metadata : ";
	for(int i=0; i<3; i++){
		os << p.metadata[i] << " ";
	}
	os << std::endl;

	os << "Pose : " << std::endl;
	os << p.pose << std::endl;
}

std::shared_ptr<std::vector<CameraPose>> ReadTrajectory(std::string filename)
{
	std::shared_ptr<std::vector<CameraPose>> traj = std::make_shared<std::vector<CameraPose>>();
	traj->clear();

        std::ifstream fin;
	fin.open(filename, std::ios::in);
	if(!fin.fail()){
                Metadata meta;
		Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

		fin >> meta[0] >> meta[1] >> meta[2];
		while(fin) {
                        for(int i=0; i<4; i++){
                                fin >> pose(i, 0) >> pose(i,1) >> pose(i, 2) >> pose(i, 3);
                        }
                        traj->push_back(CameraPose(meta, pose));
			fin >> meta[0] >> meta[1] >> meta[2];
		}

		fin.close();
	}
	else {
		std::cout << "Can't open " << filename << std::endl;
	}

	return traj;
}

void WriteTrajectory(std::vector<CameraPose>& traj, std::string filename)
{
	std::ofstream fout;
	fout.open(filename, std::ios::out);
	for(unsigned int i=0; i<traj.size(); i++){
		fout << traj[i].metadata[0] << " " << traj[i].metadata[1] << " " << traj[i].metadata[2] << std::endl;
		fout << traj[i].pose << std::endl;
	}
	fout.close();
}

int main(int argc, char *argv[])
{
	SetVerbosityLevel(VerbosityLevel::VerboseAlways);

	bool visualization = true;

#ifdef _OPENMP
	PrintDebug("OpenMP is supported. Using %d threads.", omp_get_num_threads());
#endif

	std::shared_ptr<std::vector<CameraPose>> camera_poses
			 = ReadTrajectory("../../../lib/TestData/RGBD/odometry.log");

	double voxel_length = 4.0 / 512.0;
	double sdf_trunc = 0.04;
	bool with_color = true;
	auto volume = std::shared_ptr<ScalableTSDFVolume>(
			new ScalableTSDFVolume(voxel_length,sdf_trunc, with_color));

	for(unsigned int i=0; i<camera_poses->size(); i++){
		std::cout << "Integrate " << i << "-th image into the volume." << std::endl;
		std::stringstream cpath, dpath;
		cpath << "../../../lib/TestData/RGBD/color/" << std::setfill('0') << std::setw(5) << i << ".jpg";
		dpath << "../../../lib/TestData/RGBD/depth/" << std::setfill('0') << std::setw(5) << i << ".png";
                std::shared_ptr<Image> color, depth;
		color = CreateImageFromFile(cpath.str());
		depth = CreateImageFromFile(dpath.str());
		double depth_scale = 1000;
		double depth_trunc = 4.0;
		bool convert_rgb_to_intensity = false;
                // std::shared_ptr<RGBDImage>
		auto rgbd = CreateRGBDImageFromColorAndDepth(*color, *depth,
				depth_scale, depth_trunc, convert_rgb_to_intensity);
		volume->Integrate(*rgbd, PinholeCameraIntrinsic::PrimeSenseDefault,
				(*camera_poses)[i].pose.inverse());
	}

	std::cout << "Extract a triangle mesh from the volume and visualize it." << std::endl;
        // std::shared_ptr<TriangleMesh>
	auto mesh = volume->ExtractTriangleMesh();
	mesh->ComputeVertexNormals();
	if(visualization){
		DrawGeometries({mesh}, "RGBD Integration");
	}

	return 0;
}
