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

#include "Visualizer.h"

#include <iostream>

namespace three{

namespace {

class GLFWEnvironmentSingleton {
public:
	GLFWEnvironmentSingleton() { PrintDebug("GLFW init.\n");}
	~GLFWEnvironmentSingleton() {
		glfwTerminate();
		PrintDebug("GLFW destruct.\n");
	}

public:
	static int InitGLFW() {
		static GLFWEnvironmentSingleton singleton;
		return glfwInit();
	}

	static void GLFWErrorCallback(int error, const char* description) {
		PrintError("GLFW Error: %s\n", description);
	}
};

}	// unnamed namespace

Visualizer::Visualizer() : 
		window_(NULL),
		mouse_control_(),
		view_control_(),
		is_redraw_required_(true),
		pointcloud_render_mode_(),
		color_map_ptr_(new ColorMapJet),
		background_color_(1.0, 1.0, 1.0)
{
}

Visualizer::~Visualizer()
{
	glfwTerminate();	// to be safe
}

bool Visualizer::CreateWindow(const std::string window_name/* = "Open3DV"*/, 
		const int width/* = 640*/, const int height/* = 480*/,
		const int left/* = 50*/, const int top/* = 50*/)
{
	if (window_) {	// window already created
		glfwSetWindowPos(window_, left, top);
		glfwSetWindowSize(window_, width, height);
		return true;
	}

	glfwSetErrorCallback(GLFWEnvironmentSingleton::GLFWErrorCallback);
	if (!GLFWEnvironmentSingleton::InitGLFW()) {
		PrintError("Failed to initialize GLFW\n");
		return false;
	}

	window_ = glfwCreateWindow(width, height, window_name.c_str(), NULL, NULL);
	if (!window_) {
		PrintError("Failed to create window\n");
		return false;
	}
	glfwSetWindowPos(window_, left, top);
	glfwSetWindowUserPointer(window_, this);

	auto window_refresh_callback = [](GLFWwindow *window) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				WindowRefreshCallback(window);
	};
	glfwSetWindowRefreshCallback(window_, window_refresh_callback);

	auto window_resize_callback = [](GLFWwindow *window, int w, int h) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				WindowResizeCallback(window, w, h);
	};
	glfwSetFramebufferSizeCallback(window_, window_resize_callback);

	auto mouse_move_callback = [](GLFWwindow *window, double x, double y) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				MouseMoveCallback(window, x, y);
	};
	glfwSetCursorPosCallback(window_, mouse_move_callback);

	auto mouse_scroll_callback = [](GLFWwindow *window, double x, double y) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				MouseScrollCallback(window, x, y);
	};
	glfwSetScrollCallback(window_, mouse_scroll_callback);

	auto mouse_button_callback = [](GLFWwindow *window,
			int button, int action, int mods)
	{
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				MouseButtonCallback(window, button, action, mods);
	};
	glfwSetMouseButtonCallback(window_, mouse_button_callback);

	auto key_press_callback = [](GLFWwindow *window,
			int key, int scancode, int action, int mods)
	{
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				KeyPressCallback(window, key, scancode, action, mods);
	};
	glfwSetKeyCallback(window_, key_press_callback);

	auto window_close_callback = [](GLFWwindow *window) {
		static_cast<Visualizer *>(glfwGetWindowUserPointer(window))->
				WindowCloseCallback(window);
	};
	glfwSetWindowCloseCallback(window_, window_close_callback);
	
	glfwMakeContextCurrent(window_);
	glfwSwapInterval(1);

	int window_width, window_height;
	glfwGetFramebufferSize(window_, &window_width, &window_height);
	WindowResizeCallback(window_, window_width, window_height);

	InitOpenGL();
	ResetViewPoint();

	return true;
}

void Visualizer::Run()
{
	glfwMakeContextCurrent(window_);
	while (WaitEvents()) {
	}
}

bool Visualizer::WaitEvents()
{
	glfwMakeContextCurrent(window_);
	if (is_redraw_required_) {
		WindowRefreshCallback(window_);
	}
	glfwWaitEvents();
	return !glfwWindowShouldClose(window_);
}

bool Visualizer::PollEvents()
{
	glfwMakeContextCurrent(window_);
	if (is_redraw_required_) {
		WindowRefreshCallback(window_);
	}
	glfwPollEvents();
	return !glfwWindowShouldClose(window_);
}

void Visualizer::AddPointCloud(
		std::shared_ptr<const PointCloud> pointcloud_ptr)
{
	pointcloud_ptrs_.push_back(pointcloud_ptr);
	view_control_.AddPointCloud(*pointcloud_ptr);
	is_redraw_required_ = true;
}

bool Visualizer::HasGeometry()
{
	return !pointcloud_ptrs_.empty();
}

void Visualizer::InitOpenGL()
{
	// Mesh
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPolygonOffset(1.0, 1.0);

	// depth test
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0f);

	// pixel alignment
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// mesh material
	SetDefaultMeshMaterial();
}

void Visualizer::Render()
{
	glfwMakeContextCurrent(window_);
	view_control_.SetViewPoint();

	// retrieve some OpenGL matrices that can make some operations easy.
	// e.g., gluProject
	glGetDoublev(GL_MODELVIEW_MATRIX, m_glModelview);
	glGetDoublev(GL_PROJECTION_MATRIX, m_glProjection);
	glGetIntegerv(GL_VIEWPORT, m_glViewport);

	glClearColor((GLclampf)background_color_(0),
			(GLclampf)background_color_(1),
			(GLclampf)background_color_(2), 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (size_t i = 0; i < pointcloud_ptrs_.size(); i++) {
		DrawPointCloud(*pointcloud_ptrs_[i]);
	}

	// call this when there is a mesh
	//SetDefaultLighting(bounding_box_);

	glfwSwapBuffers(window_);
}

void Visualizer::PrintVisualizerHelp()
{
	PrintInfo("Mouse options:\n");
	PrintInfo("    Left btn + drag        : Rotate.\n");
	PrintInfo("    Ctrl + left btn + drag : Translate.\n");
	PrintInfo("    Wheel                  : Zoom in/out.\n");
	PrintInfo("\n");
	PrintInfo("Keyboard options:\n");
	PrintInfo("    Q, Esc      : Exit window.\n");
	PrintInfo("    R           : Reset view point.\n");
	PrintInfo("    [/]         : Increase/decrease field of view.\n");
	PrintInfo("    +/-         : Increase/decrease point size.\n");
	PrintInfo("    0..4        : Point color options.\n");
	PrintInfo("                  0 - Default behavior, use z value to render.\n");
	PrintInfo("                  1 - Render point color.\n");
	PrintInfo("                  2 - x coordinate as color.\n");
	PrintInfo("                  3 - y coordinate as color.\n");
	PrintInfo("                  4 - z coordinate as color.\n");
	PrintInfo("    Ctrl + 0..3 : Color map options.\n");
	PrintInfo("                  0 - Gray scale color.\n");
	PrintInfo("                  1 - JET color map.\n");
	PrintInfo("                  2 - SUMMER color map.\n");
	PrintInfo("                  3 - WINTER color map.\n");
	PrintInfo("    N           : Turn on/off normal rendering.\n");
	PrintInfo("\n");
}

void Visualizer::ResetViewPoint()
{
	view_control_.Reset();
	is_redraw_required_ = true;
}

void Visualizer::SetDefaultMeshMaterial()
{
	// default material properties
	// front face
	GLfloat front_specular[] = {0.478814f, 0.457627f, 0.5f};
	GLfloat front_ambient[] =  {0.25f, 0.652647f, 0.254303f};
	GLfloat front_diffuse[] =  {0.25f, 0.652647f, 0.254303f};
	GLfloat front_shininess = 25.f;
	glMaterialfv(GL_FRONT, GL_DIFFUSE, front_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, front_specular);
	glMaterialfv(GL_FRONT, GL_AMBIENT, front_ambient);
	glMaterialf(GL_FRONT, GL_SHININESS, front_shininess);
	//back face
	GLfloat back_specular[] = {0.1596f, 0.1525f, 0.1667f};
	GLfloat back_ambient[] =  {0.175f, 0.3263f, 0.2772f};
	GLfloat back_diffuse[] =  {0.175f, 0.3263f, 0.2772f};
	GLfloat back_shininess = 100.f;
	glMaterialfv(GL_BACK, GL_DIFFUSE, back_diffuse);
	glMaterialfv(GL_BACK, GL_SPECULAR, back_specular);
	glMaterialfv(GL_BACK, GL_AMBIENT, back_ambient);
	glMaterialf(GL_BACK, GL_SHININESS, back_shininess);

	// default light
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);
}

void Visualizer::SetDefaultLighting(const BoundingBox &bounding_box)
{
	//light0
	Eigen::Vector3d light_position_eigen =
			Eigen::Vector3d(-4.0, 3.0, 5.0) * bounding_box.GetSize() * 0.5 +
			bounding_box.GetCenter();
	GLfloat	light_ambient[] = {0.3f, 0.3f, 0.3f, 1.0f};
	GLfloat	light_diffuse[] = {0.6f, 0.6f, 0.6f, 1.0f};
	GLfloat light_specular[] = {0.4f, 0.4f, 0.4f, 1.0f};
	GLfloat light_position[] = {(GLfloat)light_position_eigen(0),
			(GLfloat)light_position_eigen(1),
			(GLfloat)light_position_eigen(2), 0.0f};
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	//light1
//	Eigen::Vector3d light_position_eigen1 =
//			Eigen::Vector3d(-4.0, -4.0, -2.0) * bounding_box.GetSize() * 0.5 +
//			bounding_box.GetCenter();
	GLfloat	light_ambient1[] = {0.3f, 0.3f, 0.3f, 1.0f};
	GLfloat	light_diffuse1[] = {0.4f, 0.4f, 0.4f, 1.0f};
//	GLfloat light_specular1[] = {0.5f, 0.5f, 0.5f, 1.0f};
//	GLfloat light_position1[] = {(GLfloat)light_position_eigen1(0),
//			(GLfloat)light_position_eigen1(1),
//			(GLfloat)light_position_eigen1(2), 0.0f};
	glLightfv( GL_LIGHT1, GL_AMBIENT, light_ambient1);
	glLightfv( GL_LIGHT1, GL_DIFFUSE, light_diffuse1);
//	glLightfv( GL_LIGHT1, GL_SPECULAR, light_specular1);
//	glLightfv( GL_LIGHT1, GL_POSITION, light_position1);
	
	Eigen::Vector3d light_position_eigen2 =
			Eigen::Vector3d(4.0, -5.0, 5.0) * bounding_box.GetSize() * 0.5 +
			bounding_box.GetCenter();
	GLfloat	light_ambient2[] = {0.2f,0.2f,0.2f,1.0f};
	GLfloat	light_diffuse2[] = {0.6f,0.6f,0.6f,1.0f};
	GLfloat light_specular2[]= {0.3f,0.3f,0.3f,1.0f};
	GLfloat light_position2[] = {(GLfloat)light_position_eigen2(0),
			(GLfloat)light_position_eigen2(1),
			(GLfloat)light_position_eigen2(2), 0.0f};
	glLightfv( GL_LIGHT2, GL_AMBIENT, light_ambient2);
	glLightfv( GL_LIGHT2, GL_DIFFUSE, light_diffuse2);
	glLightfv( GL_LIGHT2, GL_SPECULAR, light_specular2);
	glLightfv( GL_LIGHT2, GL_POSITION, light_position2);

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHT2);
	glEnable(GL_LIGHTING);
}

void Visualizer::DrawPointCloud(const PointCloud &pointcloud)
{
	glDisable(GL_LIGHTING);
	glDisable(GL_POLYGON_OFFSET_FILL);
	glPointSize(GLfloat(pointcloud_render_mode_.point_size));
	glBegin(GL_POINTS);
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		PointCloudColorHandler(pointcloud, i);
		const Eigen::Vector3d &point = pointcloud.points_[i];
		glVertex3d(point(0), point(1), point(2));
	}
	glEnd();
	
	DrawPointCloudNormal(pointcloud);
}

void Visualizer::PointCloudColorHandler(const PointCloud &pointcloud, size_t i)
{
	auto point = pointcloud.points_[i];
	Eigen::Vector3d color;
	switch (pointcloud_render_mode_.point_color_option) {
	case POINTCOLOR_X:
		color = color_map_ptr_->GetColor(
				view_control_.GetBoundingBox().GetXPercentage(point(0)));
		break;
	case POINTCOLOR_Y:
		color = color_map_ptr_->GetColor(
				view_control_.GetBoundingBox().GetYPercentage(point(1)));
		break;
	case POINTCOLOR_Z:
		color = color_map_ptr_->GetColor(
				view_control_.GetBoundingBox().GetZPercentage(point(2)));
		break;
	case POINTCOLOR_COLOR:
		if (pointcloud.HasColors()) {
			color = pointcloud.colors_[i];
			break;
		}
	case POINTCOLOR_DEFAULT:
	default:
		color = color_map_ptr_->GetColor(
				view_control_.GetBoundingBox().GetZPercentage(point(2)));
		break;
	}
	glColor3d(color(0), color(1), color(2));
}

void Visualizer::DrawPointCloudNormal(const PointCloud &pointcloud)
{
	if (!pointcloud.HasNormals() || !pointcloud_render_mode_.show_normal) {
		return;
	}
	glDisable(GL_LIGHTING);
	glDisable(GL_POLYGON_OFFSET_FILL);
	glLineWidth(1.0f);
	glColor3d(0.0, 0.0, 0.0);
	glBegin(GL_LINES);
	double line_length = pointcloud_render_mode_.point_size *
			0.01 * view_control_.GetBoundingBox().GetSize();
	for (size_t i = 0; i < pointcloud.normals_.size(); i++) {
		const Eigen::Vector3d &point = pointcloud.points_[i];
		const Eigen::Vector3d &normal = pointcloud.normals_[i];
		Eigen::Vector3d end_point = point + normal * line_length;
		glVertex3d(point(0), point(1), point(2));
		glVertex3d(end_point(0), end_point(1), end_point(2));
	}
	glEnd();
}

void Visualizer::WindowRefreshCallback(GLFWwindow *window)
{
	if (is_redraw_required_) {
		Render();
	}
	is_redraw_required_ = false;
}

void Visualizer::WindowResizeCallback(GLFWwindow *window, int w, int h)
{
	view_control_.ChangeWindowSize(w, h);
	is_redraw_required_ = true;
}

void Visualizer::MouseMoveCallback(GLFWwindow *window, double x, double y)
{
	if (mouse_control_.is_mouse_left_button_down) {
		if (mouse_control_.is_control_key_down) {
			view_control_.Translate(
					mouse_control_.mouse_position_x - x,
					y - mouse_control_.mouse_position_y);
		} else {
			view_control_.Rotate(
					mouse_control_.mouse_position_x - x,
					y - mouse_control_.mouse_position_y);
		}
	}
	mouse_control_.mouse_position_x = x;
	mouse_control_.mouse_position_y = y;
	is_redraw_required_ = true;
}

void Visualizer::MouseScrollCallback(GLFWwindow* window, double x, double y)
{
	view_control_.Scale(y);
	is_redraw_required_ = true;
}

void Visualizer::MouseButtonCallback(GLFWwindow* window,
		int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouse_control_.is_mouse_left_button_down = true;
			if (mods & GLFW_MOD_CONTROL) {
				mouse_control_.is_control_key_down = true;
			} else {
				mouse_control_.is_control_key_down = false;
			}
		} else {
			mouse_control_.is_mouse_left_button_down = false;
			mouse_control_.is_control_key_down = false;
		}
	}
}

void Visualizer::KeyPressCallback(GLFWwindow *window,
		int key, int scancode, int action, int mods)
{
	if (action == GLFW_RELEASE) {
		return;
	}

	switch (key) {
	case GLFW_KEY_H:
		PrintVisualizerHelp();
		break;
	case GLFW_KEY_ESCAPE:
	case GLFW_KEY_Q:
		glfwSetWindowShouldClose(window_, GL_TRUE);
		PrintDebug("[Visualizer] Window closing.\n");
		break;
	case GLFW_KEY_0:
		if (mods & GLFW_MOD_CONTROL) {
			color_map_ptr_.reset(new ColorMapGray);
			PrintDebug("[Visualizer] Color map set to GRAY.\n");
		} else {
			pointcloud_render_mode_.point_color_option = POINTCOLOR_DEFAULT;
			PrintDebug("[Visualizer] Point color set to DEFAULT.\n");
		}
		break;
	case GLFW_KEY_1:
		if (mods & GLFW_MOD_CONTROL) {
			color_map_ptr_.reset(new ColorMapJet);
			PrintDebug("[Visualizer] Color map set to JET.\n");
		} else {
			pointcloud_render_mode_.point_color_option = POINTCOLOR_COLOR;
			PrintDebug("[Visualizer] Point color set to COLOR.\n");
		}
		break;
	case GLFW_KEY_2:
		if (mods & GLFW_MOD_CONTROL) {
			color_map_ptr_.reset(new ColorMapSummer);
			PrintDebug("[Visualizer] Color map set to SUMMER.\n");
		} else {
			pointcloud_render_mode_.point_color_option = POINTCOLOR_X;
			PrintDebug("[Visualizer] Point color set to X.\n");
		}
		break;
	case GLFW_KEY_3:
		if (mods & GLFW_MOD_CONTROL) {
			color_map_ptr_.reset(new ColorMapWinter);
			PrintDebug("[Visualizer] Color map set to WINTER.\n");
		} else {
			pointcloud_render_mode_.point_color_option = POINTCOLOR_Y;
			PrintDebug("[Visualizer] Point color set to Y.\n");
		}
		break;
	case GLFW_KEY_4:
		if (mods & GLFW_MOD_CONTROL) {
		} else {
			pointcloud_render_mode_.point_color_option = POINTCOLOR_Z;
			PrintDebug("[Visualizer] Point color set to Z.\n");
		}
		break;
	case GLFW_KEY_MINUS:
		pointcloud_render_mode_.ChangePointSize(-1.0);
		PrintDebug("[Visualizer] Point size set to %.2f.\n",
				pointcloud_render_mode_.point_size);
		break;
	case GLFW_KEY_EQUAL:
		pointcloud_render_mode_.ChangePointSize(1.0);
		PrintDebug("[Visualizer] Point size set to %.2f.\n",
				pointcloud_render_mode_.point_size);
		break;
	case GLFW_KEY_RIGHT_BRACKET:
		view_control_.ChangeFieldOfView(1.0);
		PrintDebug("[Visualizer] Field of view set to %.2f.\n",
				view_control_.GetFieldOfView());
		break;
	case GLFW_KEY_LEFT_BRACKET:
		view_control_.ChangeFieldOfView(-1.0);
		PrintDebug("[Visualizer] Field of view set to %.2f.\n",
				view_control_.GetFieldOfView());
		break;
	case GLFW_KEY_N:
		pointcloud_render_mode_.show_normal = 
				!pointcloud_render_mode_.show_normal;
		PrintDebug("[Visualizer] Point normal rendering %s.\n",
				pointcloud_render_mode_.show_normal ? "ON" : "OFF");
		break;
	case GLFW_KEY_R:
		ResetViewPoint();
		PrintDebug("[Visualizer] Reset view point.\n");
		break;
	}

	is_redraw_required_ = true;
}

void Visualizer::WindowCloseCallback(GLFWwindow *window)
{
	// happens when user click the close icon to close the window
}

}	// namespace three
