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
#include <thread>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

int main(int argc, char *argv[])
{
	using namespace three;

	SetVerbosityLevel(VERBOSE_ALWAYS);
	if (argc < 3) {
		PrintInfo("Usage:\n");
		PrintInfo("    > TestVisualizer [mesh|spin|slowspin|pointcloud|rainbow|image] [filename]\n");
		PrintInfo("    > TestVisualizer [animation] [filename] [trajectoryfile]\n");
		return 0;
	}

	std::string option(argv[1]);
	if (option == "mesh") {
		auto mesh_ptr = std::make_shared<TriangleMesh>();
		if (ReadTriangleMesh(argv[2], *mesh_ptr)) {
			PrintWarning("Successfully read %s\n", argv[2]);
		} else {
			PrintError("Failed to read %s\n\n", argv[2]);
			return 0;
		}
		mesh_ptr->ComputeVertexNormals();
		DrawGeometry(mesh_ptr, "Mesh", 1600, 900);
	} else if (option == "spin") {
		auto mesh_ptr = std::make_shared<TriangleMesh>();
		if (ReadTriangleMesh(argv[2], *mesh_ptr)) {
			PrintWarning("Successfully read %s\n", argv[2]);
		} else {
			PrintError("Failed to read %s\n\n", argv[2]);
			return 0;
		}
		mesh_ptr->ComputeVertexNormals();
		DrawGeometryWithAnimationCallback(mesh_ptr, 
				[&](Visualizer &vis) {
					vis.GetViewControl().Rotate(10, 0);
					std::this_thread::sleep_for(std::chrono::milliseconds(30));
					return false;
				}, "Spin", 1600, 900);
	} else if (option == "slowspin") {
		auto mesh_ptr = std::make_shared<TriangleMesh>();
		if (ReadTriangleMesh(argv[2], *mesh_ptr)) {
			PrintWarning("Successfully read %s\n", argv[2]);
		} else {
			PrintError("Failed to read %s\n\n", argv[2]);
			return 0;
		}
		mesh_ptr->ComputeVertexNormals();
		DrawGeometryWithKeyCallback(mesh_ptr, GLFW_KEY_SPACE,
				[&](Visualizer &vis) {
					vis.GetViewControl().Rotate(10, 0);
					std::this_thread::sleep_for(std::chrono::milliseconds(30));
					return false;
				}, "Spin", 1600, 900);
	} else if (option == "pointcloud") {
		auto cloud_ptr = std::make_shared<PointCloud>();
		if (ReadPointCloud(argv[2], *cloud_ptr)) {
			PrintWarning("Successfully read %s\n", argv[2]);
		} else {
			PrintError("Failed to read %s\n\n", argv[2]);
			return 0;
		}
		cloud_ptr->NormalizeNormals();
		DrawGeometry(cloud_ptr, "PointCloud", 1600, 900);
	} else if (option == "rainbow") {
		auto cloud_ptr = std::make_shared<PointCloud>();
		if (ReadPointCloud(argv[2], *cloud_ptr)) {
			PrintWarning("Successfully read %s\n", argv[2]);
		} else {
			PrintError("Failed to read %s\n\n", argv[2]);
			return 0;
		}
		cloud_ptr->NormalizeNormals();
		cloud_ptr->colors_.resize(cloud_ptr->points_.size());
		double color_index = 0.0;
		double color_index_step = 0.05;

		auto update_colors_func = [&cloud_ptr](double index) {
			auto color_map_ptr = GetGlobalColorMap();
			for (auto &c : cloud_ptr->colors_) {
				c = color_map_ptr->GetColor(index);
			}
		};
		update_colors_func(1.0);

		DrawGeometryWithAnimationCallback(cloud_ptr,
				[&](Visualizer &vis) {
					color_index += color_index_step;
					if (color_index > 2.0) color_index -= 2.0;
					update_colors_func(fabs(color_index - 1.0));
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
					return true;
				}, "Rainbow", 1600, 900);
	} else if (option == "image") {
		auto image_ptr = std::make_shared<Image>();
		if (ReadImage(argv[2], *image_ptr)) {
			PrintWarning("Successfully read %s\n", argv[2]);
		} else {
			PrintError("Failed to read %s\n\n", argv[2]);
			return 0;
		}
		DrawGeometry(image_ptr, "Image", image_ptr->width_, image_ptr->height_);
	} else if (option == "animation") {
		auto mesh_ptr = std::make_shared<TriangleMesh>();
		if (ReadTriangleMesh(argv[2], *mesh_ptr)) {
			PrintWarning("Successfully read %s\n", argv[2]);
		} else {
			PrintError("Failed to read %s\n\n", argv[2]);
			return 0;
		}
		mesh_ptr->ComputeVertexNormals();
		if (argc == 3) {
			DrawGeometryWithCustomAnimation(mesh_ptr, "Animation", 1600, 900);
		} else {
			DrawGeometryWithCustomAnimation(mesh_ptr, "Animation", 1600, 900, 50, 50, 
					argv[3]);
		}
	}

	PrintInfo("End of the test.\n");

	return 1;
}