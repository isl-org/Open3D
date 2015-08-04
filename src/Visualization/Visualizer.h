// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
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

#include <string>
#include <memory>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <Core/Core.h>
#include <Visualization/BoundingBox.h>
#include <Visualization/ColorMap.h>

namespace three {

class Visualizer  {
public:
	enum ProjectionOption {
		PROJECTION_PERSPECTIVE = 0,
		PROJECTION_ORTHOGONAL = 1,
	};

	enum ColorMapOption {
		COLORMAP_GRAY = 0,
		COLORMAP_JET = 1,
	};

	enum PointCloudColorOption {
		POINTCLOUDCOLOR_DEFAULT = 0,
		POINTCLOUDCOLOR_COLOR = 1,
		POINTCLOUDCOLOR_X = 2,
		POINTCLOUDCOLOR_Y = 3,
		POINTCLOUDCOLOR_Z = 4,
	};

	struct PointCloudRenderMode {
	public:
		double pointcloud_size;
		PointCloudColorOption pointcloud_color_option;
		bool show_normal;

		PointCloudRenderMode() : 
				pointcloud_size(1.0), 
				pointcloud_color_option(POINTCLOUDCOLOR_DEFAULT),
				show_normal(false)
		{}

		void IncreasePointCloudSize() {
			if (pointcloud_size < 9.5) {
				pointcloud_size += 1.0;
			}
		}
		
		void DecreasePointCloudSize() {
			if (pointcloud_size > 1.5) {
				pointcloud_size -= 1.0;
			}
		}
	};

public:
	Visualizer();
	~Visualizer();

public:
	bool CreateWindow(const std::string window_name = "Open3DV", 
			const int width = 640, const int height = 480,
			const int left = 100, const int top = 100);
	void ResetBoundingBox();
	void ResetViewPoint();
	void Run();
	void AsyncRun();	// this call will not block the main thread.

	// handle geometry
	void AddPointCloud(std::shared_ptr<const PointCloud> pointcloud_ptr);
	bool HasGeometry();

protected:
	// rendering functions
	virtual void InitOpenGL();
	virtual void SetViewPoint();
	virtual void Render();
	virtual void PrintVisualizerHelp();
	void SetDefaultMeshMaterial();
	void SetDefaultLighting(const BoundingBox &bounding_box);
	virtual void DrawPointCloud(const PointCloud &pointcloud);
	virtual void PointCloudColorHandler(const PointCloud &pointcloud,
			size_t i);

	// callback functions
	virtual void WindowRefreshCallback(GLFWwindow *window);
	virtual void WindowResizeCallback(GLFWwindow *window, int w, int h);
	virtual void MouseMoveCallback(GLFWwindow* window, double x, double y);
	virtual void MouseScrollCallback(GLFWwindow* window, double x, double y);
	virtual void MouseButtonCallback(GLFWwindow* window,
			int button, int action, int mods);
	virtual void KeyPressCallback(GLFWwindow *window,
			int key, int scancode, int action, int mods);
	virtual void WindowCloseCallback(GLFWwindow *window);

protected:
	// window
	GLFWwindow* window_;

	// control
	int window_width_;
	int window_height_;

	// view
	bool is_redraw_required_;
	BoundingBox bounding_box_;
	Eigen::Vector3d eye_;
	Eigen::Vector3d lookat_;
	Eigen::Vector3d up_;
	double distance_;
	double field_of_view_;
	double view_ratio_;			// 1.0 / object_size
	double aspect_;

	// rendering properties
	PointCloudRenderMode pointcloud_render_mode_;
	std::shared_ptr<ColorMap> color_map_ptr_;
	ProjectionOption projection_type_;
	Eigen::Vector3d background_color_;

	// geometry to be rendered
	std::vector<std::shared_ptr<const PointCloud>> pointcloud_ptrs_;

	// data to be retrieved
	GLint	m_glViewport[4];
	GLdouble m_glModelview[16];
	GLdouble m_glProjection[16];
};

}	// namespace three
