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
	GLFWEnvironmentSingleton() { PrintWarning("init\n");}
	~GLFWEnvironmentSingleton() { glfwTerminate(); PrintWarning("destruct\n");}

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
		is_redraw_required_(true),
		pointcloud_render_mode_(),
		color_map_ptr_(new ColorMapJet),
		projection_type_(PROJECTION_PERSPECTIVE),
		window_width_(0), window_height_(0),
		background_color_(1.0, 1.0, 1.0)
{
}

Visualizer::~Visualizer()
{
	glfwTerminate();	// to be safe
}

bool Visualizer::CreateWindow(const std::string window_name/* = "Open3DV"*/, 
		const int width/* = 640*/, const int height/* = 480*/,
		const int left/* = 0*/, const int top/* = 0*/)
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

	return true;
}

void Visualizer::ResetBoundingBox()
{
	bounding_box_.Reset();
	for (size_t i = 0; i < pointcloud_ptrs_.size(); i++) {
		bounding_box_.AddPointCloud(*pointcloud_ptrs_[i]);
	}
	is_redraw_required_ = true;
}

void Visualizer::ResetViewPoint()
{
	field_of_view_ = 60.0;
	view_ratio_ = bounding_box_.GetSize();
	lookat_ = bounding_box_.GetCenter();
	up_ = Eigen::Vector3d(0.0, 1.0, 0.0);
	distance_ = view_ratio_ / tan(field_of_view_ * 0.5 / 180.0 * M_PI);
	eye_ = lookat_ + Eigen::Vector3d(0.0, 0.0, 1.0) * distance_;
	is_redraw_required_ = true;
}

void Visualizer::Run()
{
	glfwMakeContextCurrent(window_);
	glfwSwapInterval(1);

	int width, height;
	glfwGetFramebufferSize(window_, &width, &height);
	WindowResizeCallback(window_, width, height);

	InitOpenGL();
	ResetViewPoint();

	while (!glfwWindowShouldClose(window_)) {
		if (is_redraw_required_) {
			WindowRefreshCallback(window_);
		}

		// An alternative method is glfwPollEvents().
		// It returns immediately and only process events that are already in the
		// event queue.
		glfwWaitEvents();
	}
}

void Visualizer::AsyncRun()
{
}

void Visualizer::AddPointCloud(
		std::shared_ptr<const PointCloud> pointcloud_ptr)
{
	pointcloud_ptrs_.push_back(pointcloud_ptr);
	bounding_box_.AddPointCloud(*pointcloud_ptr);
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

void Visualizer::SetViewPoint()
{
	if (window_height_ <= 0) {
		return;
	}

	// In this function, we set the view points based on all the viewing
	// variables that have been CORRECTLY set. This function does not check
	// the correctness. All variables must be maintained by other functions.

	glfwMakeContextCurrent(window_);
	glViewport(0, 0, window_width_, window_height_);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (projection_type_ == PROJECTION_PERSPECTIVE) {
		gluPerspective(field_of_view_, aspect_,
				distance_ - 1.0 * bounding_box_.GetSize(),
				distance_ + 3.0 * bounding_box_.GetSize());
	} else {
		glOrtho(-aspect_ * view_ratio_, aspect_ * view_ratio_,
				-view_ratio_, view_ratio_,
				-1.0 * bounding_box_.GetSize(),
				1.0 * bounding_box_.GetSize());
	}
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eye_(0), eye_(1), eye_(2),
			lookat_(0), lookat_(1), lookat_(2),
			up_(0), up_(1), up_(2));
}

void Visualizer::Render()
{
	SetViewPoint();
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
	PrintInfo("Keyboard options:\n");
	PrintInfo("    Q, Esc      : Exit window.\n");
	PrintInfo("    R           : Reset view point.\n");
	PrintInfo("    P           : Switch projection between perspective and orthogonal.\n");
	PrintInfo("    +/-         : Increase/decrease point size.\n");
	PrintInfo("    0..4        : Point color options.\n");
	PrintInfo("                  0 - Default behavior, use z value to render.\n");
	PrintInfo("                  1 - Render point color.\n");
	PrintInfo("                  2 - x coordinate as color.\n");
	PrintInfo("                  3 - y coordinate as color.\n");
	PrintInfo("                  4 - z coordinate as color.\n");
	PrintInfo("    Ctrl + 0..1 : Color map options.\n");
	PrintInfo("                  0 - Gray scale color.\n");
	PrintInfo("                  1 - JET color map.\n");
	PrintInfo("    N           : Turn on/off normal rendering.\n");
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
	Eigen::Vector3d light_position_eigen1 =
			Eigen::Vector3d(-4.0, -4.0, -2.0) * bounding_box.GetSize() * 0.5 +
			bounding_box.GetCenter();
	GLfloat	light_ambient1[] = {0.3f, 0.3f, 0.3f, 1.0f};
	GLfloat	light_diffuse1[] = {0.4f, 0.4f, 0.4f, 1.0f};
	GLfloat light_specular1[] = {0.5f, 0.5f, 0.5f, 1.0f};
	GLfloat light_position1[] = {(GLfloat)light_position_eigen1(0),
			(GLfloat)light_position_eigen1(1),
			(GLfloat)light_position_eigen1(2), 0.0f};
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
	glPointSize(GLfloat(pointcloud_render_mode_.pointcloud_size));
	glBegin(GL_POINTS);
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		PointCloudColorHandler(pointcloud, i);
		auto point = pointcloud.points_[i];
		glVertex3d(point(0), point(1), point(2));
	}
	glEnd();
}

void Visualizer::PointCloudColorHandler(const PointCloud &pointcloud, size_t i)
{
	auto point = pointcloud.points_[i];
	Eigen::Vector3d color;
	switch (pointcloud_render_mode_.pointcloud_color_option) {
	case POINTCLOUDCOLOR_X:
		color = color_map_ptr_->GetColor(
				bounding_box_.GetXPercentage(point(0)));
		break;
	case POINTCLOUDCOLOR_Y:
		color = color_map_ptr_->GetColor(
				bounding_box_.GetYPercentage(point(1)));
		break;
	case POINTCLOUDCOLOR_Z:
		color = color_map_ptr_->GetColor(
				bounding_box_.GetZPercentage(point(2)));
		break;
	case POINTCLOUDCOLOR_COLOR:
		if (pointcloud.HasColors()) {
			color = pointcloud.colors_[i];
			break;
		}
	case POINTCLOUDCOLOR_DEFAULT:
	default:
		color = color_map_ptr_->GetColor(
				bounding_box_.GetZPercentage(point(2)));
		break;
	}
	glColor3d(color(0), color(1), color(2));
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
	window_width_ = w;
	window_height_ = h;
	aspect_ = (double)window_width_ / (double)window_height_;

	SetViewPoint();
	is_redraw_required_ = true;
}

void Visualizer::MouseMoveCallback(GLFWwindow *window, double x, double y)
{
}

void Visualizer::MouseScrollCallback(GLFWwindow* window, double x, double y)
{
}

void Visualizer::MouseButtonCallback(GLFWwindow* window,
		int button, int action, int mods)
{
}

void Visualizer::KeyPressCallback(GLFWwindow *window,
		int key, int scancode, int action, int mods)
{
	if (action != GLFW_PRESS) {
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
			pointcloud_render_mode_.pointcloud_color_option =
					POINTCLOUDCOLOR_DEFAULT;
			PrintDebug("[Visualizer] Point color set to DEFAULT.\n");
		}
		break;
	case GLFW_KEY_1:
		if (mods & GLFW_MOD_CONTROL) {
			color_map_ptr_.reset(new ColorMapJet);
			PrintDebug("[Visualizer] Color map set to JET.\n");
		} else {
			pointcloud_render_mode_.pointcloud_color_option =
					POINTCLOUDCOLOR_COLOR;
			PrintDebug("[Visualizer] Point color set to COLOR.\n");
		}
		break;
	case GLFW_KEY_2:
		pointcloud_render_mode_.pointcloud_color_option = POINTCLOUDCOLOR_X;
		PrintDebug("[Visualizer] Point color set to X.\n");
		break;
	case GLFW_KEY_3:
		pointcloud_render_mode_.pointcloud_color_option = POINTCLOUDCOLOR_Y;
		PrintDebug("[Visualizer] Point color set to Y.\n");
		break;
	case GLFW_KEY_4:
		pointcloud_render_mode_.pointcloud_color_option = POINTCLOUDCOLOR_Z;
		PrintDebug("[Visualizer] Point color set to Z.\n");
		break;
	case GLFW_KEY_MINUS:
		pointcloud_render_mode_.DecreasePointCloudSize();
		PrintDebug("[Visualizer] Point size set to %.2f.\n",
				pointcloud_render_mode_.pointcloud_size);
		break;
	case GLFW_KEY_EQUAL:
		pointcloud_render_mode_.IncreasePointCloudSize();
		PrintDebug("[Visualizer] Point size set to %.2f.\n",
				pointcloud_render_mode_.pointcloud_size);
		break;
	case GLFW_KEY_N:
		pointcloud_render_mode_.show_normal = !pointcloud_render_mode_.show_normal;
		PrintDebug("[Visualizer] Point normal rendering %s.\n",
				pointcloud_render_mode_.show_normal ? "ON" : "OFF");
		break;
	case GLFW_KEY_R:
		ResetViewPoint();
		PrintDebug("[Visualizer] Reset view point.\n");
		break;
	case GLFW_KEY_P:
		if (projection_type_ == PROJECTION_ORTHOGONAL) {
			projection_type_ = PROJECTION_PERSPECTIVE;
			PrintDebug("[Visualizer] Set PERSPECTIVE projection.\n");
		} else {
			projection_type_ = PROJECTION_ORTHOGONAL;
			PrintDebug("[Visualizer] Set ORTHOGONAL projection.\n");
		}
		break;
	}

	is_redraw_required_ = true;
}

void Visualizer::WindowCloseCallback(GLFWwindow *window)
{
	// happens when user click the close icon to close the window
}

}	// namespace three
