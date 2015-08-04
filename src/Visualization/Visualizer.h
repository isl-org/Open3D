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

#ifndef GLFW_INCLUDE_GLU
#define GLFW_INCLUDE_GLU
#endif
#include <GLFW/glfw3.h>
#include <Core/Core.h>

#include "ColorMap.h"
#include "BoundingBox.h"
#include "ViewControl.h"

namespace three {

class Visualizer  {
public:
	enum ColorMapOption {
		COLORMAP_GRAY = 0,
		COLORMAP_JET = 1,
	};

	enum PointColorOption {
		POINTCOLOR_DEFAULT = 0,
		POINTCOLOR_COLOR = 1,
		POINTCOLOR_X = 2,
		POINTCOLOR_Y = 3,
		POINTCOLOR_Z = 4,
	};

	struct PointCloudRenderMode {
	public:
		double point_size;
		PointColorOption point_color_option;
		bool show_normal;

		const double POINT_SIZE_MAX = 25.0;
		const double POINT_SIZE_MIN = 1.0;
		const double POINT_SIZE_STEP = 1.0;
		const double POINT_SIZE_DEFAULT = 5.0;

		PointCloudRenderMode() : 
				point_color_option(POINTCOLOR_DEFAULT),
				show_normal(false)
		{
			point_size = POINT_SIZE_DEFAULT;
		}

		void ChangePointSize(double change) {
			double new_point_size = point_size + change * POINT_SIZE_STEP;
			if (new_point_size >= POINT_SIZE_MIN && 
					new_point_size <= POINT_SIZE_MAX)
			{
				point_size = new_point_size;
			}
		}
	};

	struct MouseControl {
	public:
		bool is_mouse_left_button_down;
		bool is_control_key_down;
		double mouse_position_x;
		double mouse_position_y;

		MouseControl() :
				is_mouse_left_button_down(false),
				is_control_key_down(false),
				mouse_position_x(0.0),
				mouse_position_y(0.0)
		{}
	};

public:
	Visualizer();
	~Visualizer();

public:
	bool CreateWindow(const std::string window_name = "Open3DV", 
			const int width = 640, const int height = 480,
			const int left = 100, const int top = 100);
	void ResetViewPoint();
	void Run();
	void AsyncRun();	// this call will not block the main thread.

	// handle geometry
	void AddPointCloud(std::shared_ptr<const PointCloud> pointcloud_ptr);
	bool HasGeometry();

protected:
	// rendering functions

	/// Function to initialize OpenGL
	/// The following things will be set:
	/// PolygonMode
	/// DepthTest
	/// PixelStorei
	/// Mesh material
	/// Note that we use a view point dependent lighting scheme, thus light 
	/// should be set during rendering.
	virtual void InitOpenGL();

	/// Function to do the main rendering
	/// The function first sets view point, then draw geometry (pointclouds and
	/// meshes individually).
	virtual void Render();

	virtual void PrintVisualizerHelp();
	void SetDefaultMeshMaterial();
	void SetDefaultLighting(const BoundingBox &bounding_box);

	/// Function to draw a point cloud
	/// This function use PointCloudColorHandler to assign color for points.
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
	MouseControl mouse_control_;
	ViewControl view_control_;
	bool is_redraw_required_;

	// rendering properties
	PointCloudRenderMode pointcloud_render_mode_;
	std::shared_ptr<ColorMap> color_map_ptr_;
	Eigen::Vector3d background_color_;

	// geometry to be rendered
	std::vector<std::shared_ptr<const PointCloud>> pointcloud_ptrs_;

	// data to be retrieved
	GLint	m_glViewport[4];
	GLdouble m_glModelview[16];
	GLdouble m_glProjection[16];
};

}	// namespace three
