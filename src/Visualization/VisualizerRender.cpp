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
#include "Shader.h"

namespace three{

bool Visualizer::InitOpenGL()
{
	if (glewInit() != GLEW_OK) {
		PrintError("Failed to initialize GLEW.\n");
		return false;
	}
	
	if (CompileShaders() == false) {
		return false;
	}
	
	// depth test
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0f);

	// pixel alignment
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// mesh material
	SetDefaultMeshMaterial();
	
	return true;
}
	
bool Visualizer::CompileShaders()
{
	shaders_.vertex_shader_pointcloud_default =
			glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(shaders_.vertex_shader_pointcloud_default, 1,
			&glsl::PointCloudVertexShader, NULL);
	glCompileShader(shaders_.vertex_shader_pointcloud_default);
	if (ValidateShader(shaders_.vertex_shader_pointcloud_default) == false) {
		return false;
	}
	
	shaders_.vertex_shader_pointcloud_x = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(shaders_.vertex_shader_pointcloud_x, 1,
				   &glsl::PointCloudXVertexShader, NULL);
	glCompileShader(shaders_.vertex_shader_pointcloud_x);
	if (ValidateShader(shaders_.vertex_shader_pointcloud_x) == false) {
		return false;
	}

	shaders_.vertex_shader_pointcloud_y = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(shaders_.vertex_shader_pointcloud_y, 1,
				   &glsl::PointCloudYVertexShader, NULL);
	glCompileShader(shaders_.vertex_shader_pointcloud_y);
	if (ValidateShader(shaders_.vertex_shader_pointcloud_y) == false) {
		return false;
	}

	shaders_.vertex_shader_pointcloud_z = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(shaders_.vertex_shader_pointcloud_z, 1,
				   &glsl::PointCloudZVertexShader, NULL);
	glCompileShader(shaders_.vertex_shader_pointcloud_z);
	if (ValidateShader(shaders_.vertex_shader_pointcloud_z) == false) {
		return false;
	}
	
	shaders_.fragment_shader_pointcloud_default =
			glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(shaders_.fragment_shader_pointcloud_default, 1,
				   &glsl::PointCloudFragmentShader, NULL);
	glCompileShader(shaders_.fragment_shader_pointcloud_default);
	if (ValidateShader(shaders_.fragment_shader_pointcloud_default) == false) {
		return false;
	}
	
	return true;
}
	
bool Visualizer::ValidateShader(GLuint shader_index)
{
	GLint result = GL_FALSE;
	int info_log_length;
	glGetShaderiv(shader_index, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) {
		glGetShaderiv(shader_index, GL_INFO_LOG_LENGTH, &info_log_length);
		if (info_log_length > 0) {
			std::vector<char> error_message(info_log_length + 1);
			glGetShaderInfoLog(shader_index, info_log_length, NULL,
					&error_message[0]);
			PrintError("Shader error: %s\n", &error_message[0]);
		}
		return false;
	}
	return true;
}

void Visualizer::Render()
{
	glfwMakeContextCurrent(window_);
	view_control_.SetViewPoint();

	glClearColor((GLclampf)background_color_(0),
			(GLclampf)background_color_(1),
			(GLclampf)background_color_(2), 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (size_t i = 0; i < geometry_ptrs_.size(); i++) {
		// dispatch geometry
		const Geometry &geometry = *geometry_ptrs_[i];
		switch (geometry.GetGeometryType()) {
		case Geometry::GEOMETRY_POINTCLOUD:
			DrawPointCloud(static_cast<const PointCloud &>(geometry));
			break;
		case Geometry::GEOMETRY_TRIANGLEMESH:
			DrawTriangleMesh(static_cast<const TriangleMesh &>(geometry));
			break;
		case Geometry::GEOMETRY_UNKNOWN:
		default:
			break;
		}
	}

	// call this when there is a mesh
	//SetDefaultLighting(bounding_box_);

	glfwSwapBuffers(window_);
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

void Visualizer::DrawTriangleMesh(const TriangleMesh &mesh)
{
	switch (mesh_render_mode_.mesh_render_option) {
	case MESHRENDER_VERTEXCOLOR:
		glDisable(GL_LIGHTING);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		break;
	case MESHRENDER_SMOOTHSHADE:
		SetDefaultLighting(view_control_.GetBoundingBox());
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		break;
	case MESHRENDER_WIREFRAME:
		glDisable(GL_LIGHTING);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		break;
	case MESHRENDER_FLATSHADE:
	default:
		SetDefaultLighting(view_control_.GetBoundingBox());
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		break;
	}
	
	for (size_t i = 0; i < mesh.triangles_.size(); i++) {
		const Eigen::Vector3i & triangle = mesh.triangles_[i];
	}
}

}	// namespace three
