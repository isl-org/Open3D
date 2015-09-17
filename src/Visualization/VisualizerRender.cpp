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

namespace three{

bool Visualizer::InitOpenGL()
{
	if (glewInit() != GLEW_OK) {
		PrintError("Failed to initialize GLEW.\n");
		return false;
	}
		
	// depth test
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0f);

	// pixel alignment
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// mesh material
	SetDefaultMeshMaterial();

	is_initialized_ = true;
	return true;
}

void Visualizer::Render()
{
	glfwMakeContextCurrent(window_);

	if (is_shader_update_required_) {
		UpdateShaders();
		is_shader_update_required_ = false;
	}

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
			glPointSize(GLfloat(pointcloud_render_mode_.GetPointSize()));
			shader_ptrs_[i]->Render(view_control_);
			break;
		case Geometry::GEOMETRY_TRIANGLEMESH:
		case Geometry::GEOMETRY_UNKNOWN:
		default:
			break;
		}
	}

	// call this when there is a mesh
	//SetDefaultLighting(bounding_box_);

	glfwSwapBuffers(window_);
}

void Visualizer::UpdateShaders()
{
	for (size_t i = 0; i < geometry_ptrs_.size(); i++) {
		const Geometry &geometry = *geometry_ptrs_[i];
		switch (geometry.GetGeometryType()) {
		case Geometry::GEOMETRY_POINTCLOUD:
			shader_ptrs_[i]->BindGeometry(
					*geometry_ptrs_[i], 
					pointcloud_render_mode_,
					view_control_);
			break;
		case Geometry::GEOMETRY_TRIANGLEMESH:
		case Geometry::GEOMETRY_UNKNOWN:
		default:
			break;
		}
	}
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
