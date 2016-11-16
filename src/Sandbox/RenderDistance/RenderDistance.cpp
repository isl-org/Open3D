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

#include <flann/flann.hpp>

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
			fprintf(f, "#declare WhiteDiffuse = rgb <0.9, 0.9, 0.9> * 0.4;\n");
			fprintf(f, "#declare WhiteAmbient = rgb <0.9, 0.9, 0.9> * 1.0;\n");
			fprintf(f, "#declare AD = 0.65;\n");
			fprintf(f, "#declare AA = 0.75;\n");
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
			/*
			const auto &pointcloud = (const PointCloud &)(*geometry_ptrs_[0]);
			for (auto i = 0; i < pointcloud.points_.size(); i++) {
				const auto &pt = pointcloud.points_[i];
				const auto &c = pointcloud.colors_[i];
				fprintf(f, "sphere {<%8.6f, %8.6f, %8.6f>, PointRadius pigment  { rgb<%.4f, %.4f, %.4f> } finish{ phong 0.1 reflection 0.2 }}\n",
					pt(0), pt(1), pt(2), c(0), c(1), c(2));
			}
			*/
			const auto &mesh = (const TriangleMesh &)(*geometry_ptrs_[0]);
			fprintf(f, "mesh2 {\n");

			fprintf(f, 
				"\tvertex_vectors {\n\t\t%d,\n",
				mesh.vertices_.size()
				);
			for (int i = 0; i < (int)mesh.vertices_.size(); i++)
			{
				fprintf(f,
					"\t<%8.6f, %8.6f, %8.6f>,\n",
					mesh.vertices_[i](0), mesh.vertices_[i](1), 
					mesh.vertices_[i](2)
					);
			}
			fprintf(f, "\t}\n");

			fprintf(f,
				"\ttexture_list {\n");
			fprintf(f, "\t%d,\n", mesh.triangles_.size());
			for (int i = 0; i < (int)mesh.triangles_.size(); i++)
			{
				Eigen::Vector3d c = (mesh.vertex_colors_[mesh.triangles_[i](0)]
					+ mesh.vertex_colors_[mesh.triangles_[i](1)] +
					mesh.vertex_colors_[mesh.triangles_[i](2)]) / 3.0;
				fprintf(f, "\ttexture {\n\t\tpigment { color rgb<%.4f, %.4f, %.4f> * AD }\n\t\tfinish{ ambient rgb<%.4f, %.4f, %.4f> * AA diffuse 0.8 }\n\t}\n",
						c(0), c(1), c(2), c(0), c(1), c(2));
			}
			fprintf(f, "\t}\n");

			fprintf(f, 
				"\tface_indices {\n\t\t%d,\n",
				mesh.triangles_.size()
				);
			for (int i = 0; i < (int)mesh.triangles_.size(); i++)
			{
				fprintf(f,
					"\t<%d, %d, %d>,%d,\n",
					mesh.triangles_[i](0),
					mesh.triangles_[i](1),
					mesh.triangles_[i](2),
					i
				);
			}
			fprintf(f, "\t}\n");

			fprintf(f, "}\n");

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
	printf("    > RenderDistance pcd_file distance_file threshold [mesh_file]\n");
}

void ReadBinaryResult(const std::string &filename, std::vector<double> &data)
{
	FILE *f = fopen(filename.c_str(), "rb");
	fread(data.data(), sizeof(double), data.size(), f);
	fclose(f);
}

int main(int argc, char *argv[])
{
	using namespace three;

	if (argc < 4 || ProgramOptionExists(argc, argv, "--help") ||
			ProgramOptionExists(argc, argv, "-h")) {
		PrintHelp();
		return 0;
	}
	SetVerbosityLevel((VerbosityLevel)4);

	auto pcd = CreatePointCloudFromFile(argv[1]);
	std::vector<double> dis(pcd->points_.size());
	ReadBinaryResult(argv[2], dis);
	double max_dis = std::stod(argv[3]);

	Eigen::Vector3d default_color(0.9, 0.9, 0.9);
	pcd->colors_.resize(pcd->points_.size());
	ColorMapHot colormap;
	for (auto i = 0; i < pcd->points_.size(); i++) {
		if (dis[i] < 0.0) {
			pcd->colors_[i] = default_color;
		} else {
			pcd->colors_[i] = colormap.GetColor(dis[i] / max_dis);
		}
	}

	if (argc == 4) {
		CustomVisualizer vis;
		vis.CreateWindow("Render", 1600, 1200);
		vis.AddGeometry(pcd);
		vis.Run();
		vis.DestroyWindow();
	} else {
		auto mesh = CreateMeshFromFile(argv[4]);
		mesh->vertex_colors_.clear();
		mesh->vertex_colors_.resize(mesh->vertices_.size(), 
				Eigen::Vector3d::Ones());
		KDTreeFlann tree;
		tree.SetGeometry(*pcd);
		int k = 4;
		double max_d = 0.05;
		std::vector<int> indices(k);
		std::vector<double> dis2(k);
		for (auto i = 0; i < mesh->vertices_.size(); i++) {
			tree.SearchKNN(mesh->vertices_[i], k, indices, dis2);
			mesh->vertex_colors_[i].setZero();
			double total_weight = 0.0;
			for (int j = 0; j < k; j++) {
				if (std::sqrt(dis2[j]) < max_d) {
					double weight = max_d - std::sqrt(dis2[j]);
					total_weight += weight;
					mesh->vertex_colors_[i] += 
							weight * pcd->colors_[indices[j]];
				}
			}
			if (total_weight == 0.0) {
				mesh->vertex_colors_[i] = default_color;
			} else {
				mesh->vertex_colors_[i] /= total_weight;
			}
		}
		mesh->ComputeVertexNormals();
		CustomVisualizer vis;
		vis.CreateWindow("Render", 1600, 1200);
		vis.AddGeometry(mesh);
		vis.Run();
		vis.DestroyWindow();
		DrawGeometryWithCustomAnimation(mesh);
	}

	return 1;
}
