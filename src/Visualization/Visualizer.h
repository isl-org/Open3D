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

		const double MAX_POINT_SIZE = 15.0;
		const double MIN_POINT_SIZE = 1.0;

		PointCloudRenderMode() : 
				pointcloud_size(1.0), 
				pointcloud_color_option(POINTCLOUDCOLOR_DEFAULT),
				show_normal(false)
		{}

		void IncreasePointCloudSize() {
			if (pointcloud_size < MAX_POINT_SIZE - 0.5) {
				pointcloud_size += 1.0;
			}
		}
		
		void DecreasePointCloudSize() {
			if (pointcloud_size > MIN_POINT_SIZE + 0.5) {
				pointcloud_size -= 1.0;
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

	struct ViewControl {
	public:
		const double FIELD_OF_VIEW_MAX = 90.0;
		const double FIELD_OF_VIEW_MIN = 0.0;
		const double FIELD_OF_VIEW_DEFAULT = 60.0;
		const double FIELD_OF_VIEW_STEP = 10.0;

		const double ZOOM_DEFAULT = 0.7;
		const double ZOOM_MIN = 0.1;
		const double ZOOM_MAX = 2.0;
		const double ZOOM_STEP = 0.02;

		enum ProjectionType {
			PROJECTION_PERSPECTIVE = 0,
			PROJECTION_ORTHOGONAL = 1,
		};

		BoundingBox bounding_box_;
		Eigen::Vector3d eye_;
		Eigen::Vector3d lookat_;
		Eigen::Vector3d up_;
		Eigen::Vector3d front_;
		double distance_;
		double field_of_view_;
		double zoom_;
		double view_ratio_;
		double aspect_;
		int window_width_;
		int window_height_;

		ViewControl() :
				window_width_(0), window_height_(0)
		{}

		ProjectionType GetProjectionType() {
			if (field_of_view_ > 
					FIELD_OF_VIEW_MIN + FIELD_OF_VIEW_STEP / 2.0)
			{
				return PROJECTION_PERSPECTIVE;
			} else {
				return PROJECTION_ORTHOGONAL;
			}
		}

		void Reset() {
			field_of_view_ = FIELD_OF_VIEW_DEFAULT;
			zoom_ = ZOOM_DEFAULT;
			lookat_ = bounding_box_.GetCenter();
			up_ = Eigen::Vector3d(0.0, 1.0, 0.0);
			front_ = Eigen::Vector3d(0.0, 0.0, 1.0);
			SetProjectionParameters();
		}

		void SetProjectionParameters() {
			if (GetProjectionType() == PROJECTION_PERSPECTIVE) {
				view_ratio_ = zoom_ * bounding_box_.GetSize();
				distance_ = view_ratio_ / 
						tan(field_of_view_ * 0.5 / 180.0 * M_PI);
				eye_ = lookat_ + front_ * distance_;
			} else {
				view_ratio_ = zoom_ * bounding_box_.GetSize();
				distance_ = bounding_box_.GetSize();
				eye_ = lookat_ + front_ * distance_;
			}
		}

		void ChangeFieldOfView(double step) {
			double field_of_view_new = field_of_view_ + 
					step * FIELD_OF_VIEW_STEP;
			if (field_of_view_new >= FIELD_OF_VIEW_MIN &&
					field_of_view_new <= FIELD_OF_VIEW_MAX)
			{
				field_of_view_ = field_of_view_new;
			}
			SetProjectionParameters();
		}

		void Scale(double scale) {
			double zoom_new = zoom_ + scale * ZOOM_STEP;
			if (zoom_new >= ZOOM_MIN && zoom_new <= ZOOM_MAX) {
				zoom_ = zoom_new;
			}
			SetProjectionParameters();
		}

		void Rotate() {
		}

		void Translate() {
		}
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

	/// Function to set view points
	/// This function obtains OpenGL context and calls OpenGL functions to set
	/// the view point.
	virtual void SetViewPoint();

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
