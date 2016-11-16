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

#include <iostream>
#include <memory>

#include <Eigen/Dense>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

using namespace three;

class CustomVisualizer : public VisualizerWithCustomAnimation
{
protected:
	void KeyPressCallback(GLFWwindow *window,
			int key, int scancode, int action, int mods) override {
		if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) {
			// save to pov
			FILE *f = fopen("render.pov", "w");

			fprintf(f, "#include \"colors.inc\"\n\n");
			fprintf(f, "#declare WhiteSphere = rgb <0.9, 0.9, 0.9>;\n");
			fprintf(f, "#declare PointRadius = 5;\n");
			fprintf(f, "\nbackground\n{\n\tcolor <1, 1, 1>\n}\n");

			auto view = GetViewControl();
			Eigen::Vector3f up = view.GetUp();
			Eigen::Vector3f eye = view.GetEye();
			Eigen::Vector3f gaze = view.GetLookat();
			Eigen::Vector3f front = view.GetFront();
			Eigen::Vector3f right = up.cross(front).normalized();
			fprintf(f, "camera {\n\tangle %8.6f\n\tup <%8.6f, %8.6f, %8.6f>\n\tright <%8.6f, %8.6f, %8.6f>\n\tlocation <%8.6f, %8.6f, %8.6f>\n\tlook_at <%8.6f, %8.6f, %8.6f>\n\tsky <%8.6f, %8.6f, %8.6f>\n}\n\n",
					view.GetFieldOfView(), up(0), up(1), up(2),
					right(0), right(1), right(2),
					eye(0), eye(1), eye(2),
					gaze(0), gaze(1), gaze(2),
					up(0), up(1), up(2));

			fprintf_s(f, "light_source {<%8.6f, %8.6f, %8.6f> rgb <1.0, 1.0, 1.0> * 2.5\n\tarea_light <10, 0, 0>, <0, 10, 0>, 15, 15 adaptive 1 circular\n}\n",
					right(0), right(1), right(2));
			fprintf_s(f, "light_source {<%8.6f, %8.6f, %8.6f> rgb <1.0, 1.0, 1.0> * 0.8 shadowless}\n\n",
					-right(0), -right(1), right(2));

			//const auto &pointcloud = (const PointCloud &)(*geometry_ptrs_[0]);
			//for (auto i = 0; i < pointcloud.points_.size(); i++) {
			//	const auto &pt = pointcloud.points_[i];
			//	const auto &c = pointcloud.colors_[i];
			//	fprintf(f, "sphere {<%8.6f, %8.6f, %8.6f>, PointRadius pigment  { rgb<%.4f, %.4f, %.4f> } finish{ phong 0.1 reflection 0.2 }}\n",
			//		pt(0), pt(1), pt(2), 0.9, 0.9, 0.9);
			//}

			const auto &pointcloud = (const PointCloud &)(*geometry_ptrs_[0]);
			for (auto i = 0; i < pointcloud.points_.size(); i++) {
				const auto &pt = pointcloud.points_[i];
				const auto &c = pointcloud.colors_[i];
				fprintf(f, "sphere {<%8.6f, %8.6f, %8.6f>, PointRadius pigment  { rgb<%.4f, %.4f, %.4f> } finish{ phong 0.1 reflection 0.2 }}\n",
					pt(0), pt(1), pt(2), c(0), c(1), c(2));
			}

			const auto &pointcloud1 = (const PointCloud &)(*geometry_ptrs_[1]);
			for (auto i = 0; i < pointcloud1.points_.size(); i++) {
				const auto &pt = pointcloud1.points_[i];
				const auto &c = pointcloud1.colors_[i];
				fprintf(f, "sphere {<%8.6f, %8.6f, %8.6f>, PointRadius pigment  { rgb<%.4f, %.4f, %.4f> } finish{ phong 0.1 reflection 0.2 }}\n",
					pt(0), pt(1), pt(2), c(0), c(1), c(2));
			}
			fclose(f);
		} else {
			VisualizerWithCustomAnimation::KeyPressCallback(window, key, scancode,
					action, mods);
		}
	}
};

void PrintHelp()
{
	printf("Usage:\n");
	printf("    > ViewPCDMatch [options]\n");
	printf("      View pairwise matching result of point clouds.\n");
	printf("\n");
	printf("Basic options:\n");
	printf("    --help, -h                : Print help information.\n");
	printf("    --log file                : A log file of the pairwise matching results. Must have.\n");
	printf("    --dir directory           : The directory storing all pcd files. By default it is the parent directory of the log file + pcd/.\n");
	printf("    --threshold t             : If specified, the source point cloud is rendered with color coding.\n");
	printf("    --verbose n               : Set verbose level (0-4). Default: 2.\n");
}

int main(int argc, char *argv[])
{
	using namespace three;

	if (argc <= 1 || ProgramOptionExists(argc, argv, "--help") ||
			ProgramOptionExists(argc, argv, "-h")) {
		PrintHelp();
		return 0;
	}
	const int NUM_OF_COLOR_PALETTE = 5;
	Eigen::Vector3d color_palette[NUM_OF_COLOR_PALETTE] = {
		Eigen::Vector3d(255, 180, 0) / 255.0,
		Eigen::Vector3d(0, 166, 237) / 255.0,
		Eigen::Vector3d(246, 81, 29) / 255.0,
		Eigen::Vector3d(127, 184, 0) / 255.0,
		Eigen::Vector3d(13, 44, 84) / 255.0,
	};
	
	int verbose = GetProgramOptionAsInt(argc, argv, "--verbose", 2);
	SetVerbosityLevel((VerbosityLevel)verbose);
	std::string log_filename = GetProgramOptionAsString(argc, argv, "--log");
	std::string pcd_dirname = GetProgramOptionAsString(argc, argv, "--dir");
	if (pcd_dirname.empty()) {
		pcd_dirname = filesystem::GetFileParentDirectory(log_filename) +
				"pcds/";
	}
	double threshold = GetProgramOptionAsDouble(argc, argv, "--threshold");

	FILE * f = fopen(log_filename.c_str(), "r");
	if (f == NULL) {
		PrintWarning("Read LOG failed: unable to open file.\n");
		return false;
	}
	char line_buffer[DEFAULT_IO_BUFFER_SIZE];
	int i, j, k;
	Eigen::Matrix4d trans;
	while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
		if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
			if (sscanf(line_buffer, "%d %d %d", &i, &j, &k) != 3) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(0,0), &trans(0,1),
						&trans(0,2), &trans(0,3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(1,0), &trans(1,1),
						&trans(1,2), &trans(1,3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(2,0), &trans(2,1),
						&trans(2,2), &trans(2,3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(3,0), &trans(3,1),
						&trans(3,2), &trans(3,3));
			}
			PrintInfo("Showing matched point cloud #%d and #%d.\n",
					i, j);
			auto pcd_target = CreatePointCloudFromFile(pcd_dirname +
					"cloud_bin_" + std::to_string(i) + ".pcd");
			pcd_target->colors_.clear();
			pcd_target->colors_.resize(pcd_target->points_.size(),
					color_palette[0]);
			auto pcd_source = CreatePointCloudFromFile(pcd_dirname +
					"cloud_bin_" + std::to_string(j) + ".pcd");
			pcd_source->colors_.clear();
			pcd_source->colors_.resize(pcd_source->points_.size(),
					color_palette[1]);
			PointCloud source = *pcd_source;
			pcd_source->Transform(trans);

			if (threshold > 0.0) {
				ColorMapSummer cm;
				pcd_target->colors_.clear();
				pcd_target->colors_.resize(pcd_target->points_.size(),
						Eigen::Vector3d(0.9, 0.9, 0.9));
				pcd_source->colors_.clear();
				pcd_source->colors_.resize(pcd_source->points_.size(),
						Eigen::Vector3d(0.9, 0.9, 0.9));
				KDTreeFlann tree;
				std::vector<int> indices(1);
				std::vector<double> distance2(1);
				tree.SetGeometry(*pcd_target);
				for (auto l = 0; l < pcd_source->points_.size(); l++) {
					tree.SearchKNN(source.points_[l], 1, indices, distance2);
					if (distance2[0] < 17.22) {
						//double new_dis = (pcd_source->points_[l] - 
						//		pcd_target->points_[indices[0]]).norm();
						//pcd_source->colors_[l] = cm.GetColor(
						//		new_dis / threshold);
						tree.SearchKNN(pcd_source->points_[l], 1, indices,
								distance2);
						pcd_source->colors_[l] = cm.GetColor(
								1.0 - std::sqrt(distance2[0]) / threshold);
					}
				}
			}

			//DrawGeometriesWithCustomAnimation({pcd_target, pcd_source},
			//		"ViewPCDMatch", 1600, 900);
			CustomVisualizer vis;
			vis.CreateWindow("Render", 1600, 1200);
			vis.AddGeometry(pcd_target);
			vis.AddGeometry(pcd_source);
			vis.Run();
			vis.DestroyWindow();
		}
	}
	fclose(f);
	return 1;
}
